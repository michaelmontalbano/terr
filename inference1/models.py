"""Model loading utilities."""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import utils as kutils

import config as cfg

try:  # Optional dependency used only for legacy checkpoints
    import h5py
except Exception:  # pragma: no cover - environments without h5py
    h5py = None

logger = logging.getLogger(__name__)


# =============================================================================
# Legacy TF 2.x Lambda alias --------------------------------------------------
# =============================================================================


@kutils.register_keras_serializable(package="Legacy")
class _LegacySlicingOpLambda(tfk.layers.Lambda):
    """Compatibility wrapper for ``SlicingOpLambda`` serialized in TF 2.x."""


# Ensure the alias is globally visible even outside explicit scopes.
kutils.get_custom_objects().setdefault("SlicingOpLambda", _LegacySlicingOpLambda)

# =============================================================================
# Conv2DTranspose legacy "groups" key patch
# =============================================================================
_Conv2DTranspose = tfk.layers.Conv2DTranspose
_CONV2D_TRANSPOSE_PATCHED = False


def _patch_conv2d_transpose() -> None:
    """Ensure legacy HDF5 configs with a ``groups`` key load correctly."""
    global _CONV2D_TRANSPOSE_PATCHED
    if _CONV2D_TRANSPOSE_PATCHED:
        return

    original = getattr(_Conv2DTranspose, "from_config", None)
    original_func = getattr(original, "__func__", None) if original else None
    if not original_func:
        return

    @classmethod
    def _patched_from_config(cls, config):  # type: ignore[override]
        cfg2 = dict(config)
        cfg2.pop("groups", None)
        return original_func(cls, cfg2)

    _Conv2DTranspose.from_config = _patched_from_config  # type: ignore[attr-defined]
    _CONV2D_TRANSPOSE_PATCHED = True


_patch_conv2d_transpose()


class _LegacyConv2DTranspose(tfk.layers.Conv2DTranspose):
    """Kept for completeness; used when explicit custom objects are required."""
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# =============================================================================
# Fallback loss/metrics for legacy ConvGRU checkpoints (kept for completeness)
# =============================================================================
def _fallback_weighted_mse():
    """Return a minimal weighted MSE implementation for ConvGRU checkpoints."""
    def loss(y_true, y_pred):
        mse = tf.square(y_true - y_pred)
        timesteps = tf.shape(y_true)[1]
        timestep_weights = tf.range(1, timesteps + 1, dtype=tf.float32)
        timestep_weights = tf.reshape(timestep_weights, (1, timesteps, 1, 1, 1))
        return mse * timestep_weights
    return loss


def _fallback_csi(threshold: float = 20.0):
    """Simplified CSI metric used by historic ConvGRU training runs."""
    def metric(y_true, y_pred):
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        y_true_binary = tf.cast(y_true > threshold, tf.float32)
        tp = tf.reduce_sum(y_true_binary[:, -1] * y_pred_binary[:, -1])
        fn = tf.reduce_sum(y_true_binary[:, -1] * (1 - y_pred_binary[:, -1]))
        fp = tf.reduce_sum((1 - y_true_binary[:, -1]) * y_pred_binary[:, -1])
        return tp / tf.maximum(tp + fn + fp, 1.0)
    return metric


# =============================================================================
# Types and loader
# =============================================================================
class ModelType(str, Enum):
    CONVGRU = "convgru"
    FLOW = "flow"

    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["ModelType"]:
        if value is None:
            return None
        lowered = value.lower()
        for member in cls:
            if lowered in {member.value, member.name.lower()}:
                return member
        raise ValueError(f"Unknown model type '{value}'")


@dataclass
class LoadedModel:
    model: tfk.Model
    model_type: ModelType


# --- FLOW model rebuild (avoids Lambda deserialization) ----------------------
def _parse_initial_filters_from_name(path: str, default_filters: int = 64) -> int:
    # e.g., "...filters24.h5" -> 24
    m = re.search(r"filters(\d+)", os.path.basename(path))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return default_filters


def _build_flow_architecture(
    initial_filters: int,
    height: int = 256,
    width: int = 256,
    timesteps: int = 12,
    input_channels: int = 8,
) -> tfk.Model:
    # Import your code-based architecture to rebuild the exact topology.
    from flow_matching_model import FlowMatchingUNet  # noqa: WPS433 (local import is intentional)
    unet = FlowMatchingUNet(
        height=height,
        width=width,
        input_channels=input_channels,
        timesteps=timesteps,
        initial_filters=initial_filters,
        num_downsampling=4,
        dropout_rate=0.1,
    )
    return unet.build_model()


def _rebuild_and_load_flow_h5_weights(h5_path: str) -> tfk.Model:
    """Recreate FlowMatchingUNet and load weights from an H5 full-model file."""
    init_filters = _parse_initial_filters_from_name(h5_path, default_filters=64)
    model = _build_flow_architecture(initial_filters=init_filters)

    # In Keras 3, load_weights can read from a full-model H5; skip mismatches just in case.
    try:
        model.load_weights(h5_path, skip_mismatch=True)
        # model.assert_consumed()
        logger.info("Loaded weights from %s into rebuilt FlowMatchingUNet (filters=%d).", h5_path, init_filters)
    except Exception as exc:
        logger.error("Failed to load weights from %s: %s", h5_path, exc)
        raise
    return model


# --- ConvGRU legacy rebuild -----------------------------------------------
def _build_convgru_architecture(
    *,
    timesteps: int = 12,
    height: int = 256,
    width: int = 256,
    channels: int = 8,
    initial_filters: int = 16,
    kernel_size: int = 3,
    num_downsampling: int = 3,
    dropout_rate: float = 0.27,
) -> tfk.Model:
    """Recreate the historic ConvGRU architecture used for legacy checkpoints."""

    from rnn import rnn as build_rnn  # Imported lazily to avoid heavy dependencies

    future_channels = max(0, channels - 1)
    other_fields_shape = (timesteps, height, width, future_channels)
    model = build_rnn(
        timesteps=timesteps,
        height=height,
        width=width,
        channels=channels,
        other_fields_shape=other_fields_shape,
        initial_filters=initial_filters,
        final_activation="linear",
        dropout_rate=dropout_rate,
        l1_reg=0.28,
        l2_reg=0.29,
        x_pad=0,
        y_pad=0,
        kernel_size=kernel_size,
        padding="same",
        num_downsampling=num_downsampling,
        future_channels=future_channels,
    )
    return model


def _rebuild_and_load_convgru_h5_weights(h5_path: str) -> tfk.Model:
    """Recreate ConvGRU graph and load weights from a legacy H5 checkpoint."""

    model = _build_convgru_architecture()
    try:
        model.load_weights(h5_path, skip_mismatch=False)
        logger.info("Loaded weights from %s into rebuilt ConvGRU model.", h5_path)
    except Exception as exc:  # pragma: no cover - exercised only with legacy checkpoints
        logger.error("Failed to load ConvGRU weights from %s: %s", h5_path, exc)
        raise
    return model


# --- Optional: convgru custom objects (kept; not used for FLOW rebuild path) --
def _convgru_custom_objects() -> Dict[str, object]:
    try:
        module = importlib.import_module("models")
        weighted_mse = module.weighted_mse
        csi = module.csi
    except Exception as exc:  # pragma: no cover - logging fallback path
        logger.warning("Falling back to bundled ConvGRU losses: %s", exc)
        weighted_mse = _fallback_weighted_mse
        csi = _fallback_csi()
    from rnn import (  # noqa: WPS347
        reshape_and_stack,
        slice_to_n_steps,
        slice_output_shape,
        ResBlock,
        WarmUpCosineDecayScheduler,
        ConvGRU,
        ConvBlock,
        ZeroLikeLayer,
        ReflectionPadding2D,
        ResGRU,
        GRUResBlock,
    )
    loss_fn = weighted_mse()
    return {
        "loss": loss_fn,
        "weighted_mse": loss_fn,
        "csi": csi,
        "Conv2DTranspose": _LegacyConv2DTranspose,
        # Historic checkpoints saved Lambda layers under the now-removed
        # ``SlicingOpLambda`` name. Mapping the alias back to the standard
        # Lambda layer lets Keras rebuild the graph without needing the
        # original registration from TensorFlow 2.x.
        "SlicingOpLambda": _LegacySlicingOpLambda,
        "reshape_and_stack": reshape_and_stack,
        "slice_to_n_steps": slice_to_n_steps,
        "slice_output_shape": slice_output_shape,
        "ResBlock": ResBlock,
        "WarmUpCosineDecayScheduler": WarmUpCosineDecayScheduler,
        "ConvGRU": ConvGRU,
        "ConvBlock": ConvBlock,
        "ZeroLikeLayer": ZeroLikeLayer,
        "ReflectionPadding2D": ReflectionPadding2D,
        "ResGRU": ResGRU,
        "GRUResBlock": GRUResBlock,
    }


def _custom_objects_for(model_type: ModelType) -> Dict[str, object]:
    # We keep the convgru objects available for legacy convgru loads.
    objects = _convgru_custom_objects()
    return objects


def _guess_model_type(path: str) -> ModelType:
    name = os.path.basename(path).lower()
    if "flow" in name or "diff" in name:
        return ModelType.FLOW
    return ModelType.CONVGRU


class ModelLoader:
    """Load either ConvGRU or flow matching checkpoints."""

    def load(self, config: cfg.EvaluationConfig) -> LoadedModel:
        model_path = config.resolved_model_path()
        if not model_path:
            raise ValueError("A local model path must be supplied via --model_path")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist")

        explicit_type = ModelType.from_string(config.model_type) if config.model_type else None
        model_type = explicit_type or _guess_model_type(model_path)

        # Special FLOW + .h5 handling: rebuild graph from source & load weights
        if model_type is ModelType.FLOW and model_path.endswith(".h5"):
            model = _rebuild_and_load_flow_h5_weights(model_path)
            logger.info("Loaded FLOW (rebuilt) model from %s", model_path)
            return LoadedModel(model=model, model_type=model_type)

        # Otherwise, try normal Keras 3 load (works for .keras zip or SavedModel dir)
        custom_objects = _custom_objects_for(model_type)
        model = self._load_with_fallbacks(model_path, custom_objects, model_type)
        logger.info("Loaded %s model from %s", model_type.value, model_path)
        return LoadedModel(model=model, model_type=model_type)

    @staticmethod
    def _load_with_fallbacks(
        path: str,
        custom_objects: Dict[str, object],
        model_type: ModelType,
    ) -> tfk.Model:
        def _attempt_load(load_path: str) -> tfk.Model:
            try:
                return tfk.models.load_model(
                    load_path,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False,
                )
            except TypeError as exc:
                message = str(exc)
                if (
                    model_type is ModelType.CONVGRU
                    and h5py is not None
                    and h5py.is_hdf5(load_path)
                ):
                    lower = message.lower()
                    if "ellipsis" in lower:
                        logger.info(
                            "Detected ellipsis objects in legacy ConvGRU checkpoint; "
                            "rebuilding architecture and loading weights manually."
                        )
                        return _rebuild_and_load_convgru_h5_weights(load_path)
                    if "__operators__.getitem" in lower or "unsupported callable" in lower:
                        logger.info(
                            "Detected legacy Lambda slicing callable in ConvGRU checkpoint; "
                            "rebuilding architecture and loading weights manually."
                        )
                        return _rebuild_and_load_convgru_h5_weights(load_path)
                raise

        try:
            # Allow legacy graphs & custom objects
            return _attempt_load(path)
        except ValueError as exc:
            message = str(exc)
            # Handle legacy .h5 saved with .keras suffix
            if ("accessible `.keras` zip file" in message) and (h5py is not None) and h5py.is_hdf5(path):
                logger.info("Detected legacy HDF5 checkpoint stored with .keras suffix; reloading via temporary .h5 copy")
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                    temp_path = tmp.name
                try:
                    shutil.copy2(path, temp_path)
                    return _attempt_load(temp_path)
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            raise
