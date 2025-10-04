#!/usr/bin/env python3
"""Standalone MESH verification using on-demand MRMS downloads (cached + ROI tiling),
with optional top-K-by-MESH tile stats printing (raw + normalized)."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from evaluation.config import StandaloneConfig
from evaluation.data import NormalizationBundle, S3ArtifactLoader
from evaluation.models import ModelLoader, ModelType
from evaluation.plotting import plot_ground_truth_prediction_difference
from evaluation.mrms import MRMSConfig, MRMSDataBuilder, NormalizationArrays


THRESHOLDS = np.array([5, 10, 20, 30, 40, 50, 70], dtype=float)
SUGGESTED_DATES = (
    "2025-05-28T22:00Z",
    "2025-05-27T18:00Z",
    "2025-05-26T20:20Z",
    "2025-04-10T23:00Z",
)


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("Datetime string cannot be empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Unable to parse datetime '{value}'") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ensure_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


def _parse_n_tiles(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "all", "everything", "full", "max", "none", "*"}:
        return None
    try:
        parsed = int(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--n_tiles must be a positive integer or 'everything'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--n_tiles must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", choices=["convgru", "flow"])
    parser.add_argument("--datetime", dest="datetimes", action="append", required=True,
                        help="ISO8601; repeat for multiple")
    parser.add_argument("--lead_time", type=_ensure_positive_int, default=60)
    parser.add_argument("--flow_steps", type=_ensure_positive_int, default=64)
    parser.add_argument("--tile_size", type=_ensure_positive_int, default=256)
    parser.add_argument("--stride", type=_ensure_positive_int, default=128)
    parser.add_argument("--n_tiles", type=_parse_n_tiles, default=None,
                        help="[DEPRECATED] use --max_tiles")
    # ROI controls
    parser.add_argument("--roi_channel", type=int, default=7,
                        help="Channel index for ROI mask (default: 7=MESH_dilated)")
    parser.add_argument("--roi_threshold", type=float, default=20.0,
                        help="Threshold to mark ROI (values > threshold are 'active')")
    parser.add_argument("--max_tiles", type=int, default=100,
                        help="Cap the number of tiles to evaluate (after ROI filtering)")
    parser.add_argument("--tile_select", choices=["random", "cover"], default="cover",
                        help="Random sample tiles or pick a coverage-prioritized subset")
    parser.add_argument("--skip_background", action="store_true",
                        help="Only evaluate tiles that overlap the ROI")
    # I/O
    parser.add_argument("--mrms_bucket", default="noaa-mrms-pds")
    parser.add_argument("--model_bucket", default="dev-grib-bucket")
    parser.add_argument("--norm_min_key", default="global_mins.npy")
    parser.add_argument("--norm_max_key", default="global_maxs.npy")
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--use_cache", action="store_true", default=True,
                        help="Cache built input tensors and GT locally (default: True)")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--log_level", default="DEBUG")
    parser.add_argument("--output_dir", default="./outputs")
    # Top-K stats
    parser.add_argument("--print_topk_stats", action="store_true",
                        help="Print stats on top-K tiles ranked by MESH.")
    parser.add_argument("--stats_top_k", type=int, default=32,
                        help="Number of top-MESH tiles to print stats on (default: 32).")
    parser.add_argument("--mesh_channel", type=int, default=2,
                        help="Channel index for MESH used to rank tiles (default: 2).")
    return parser


def _load_local_array(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            logging.getLogger(__name__).warning("Failed to read local array %s", path)
    return None


def _try_load_any_local(paths: List[Path]) -> Optional[np.ndarray]:
    for p in paths:
        arr = _load_local_array(p)
        if arr is not None:
            return arr
    return None


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / Path(key).name


def load_normalization_arrays(
    loader: S3ArtifactLoader,
    bucket: str,
    min_key: str,
    max_key: str,
    cache_dir: Path,
) -> NormalizationArrays:
    log = logging.getLogger(__name__)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def __cache_path(key: str) -> Path:
        return cache_dir / Path(key).name

    def __load_local_array(path: Path) -> Optional[np.ndarray]:
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                log.warning("Failed to read local array %s", path)
        return None

    def __try_load_any_local(paths: List[Path]) -> Optional[np.ndarray]:
        for p in paths:
            arr = __load_local_array(p)
            if arr is not None:
                return arr
        return None

    min_local = __cache_path(min_key)
    max_local = __cache_path(max_key)

    global_min = __try_load_any_local([min_local, Path(min_key).resolve(), Path(Path(min_key).name)])
    if global_min is None:
        log.info("Downloading normalization minima from s3://%s/%s", bucket, min_key)
        arr = loader.load_numpy(bucket, min_key)
        if arr is None:
            raise RuntimeError(f"Unable to load normalization minima from s3://{bucket}/{min_key}")
        np.save(min_local, arr)
        global_min = arr

    global_max = __try_load_any_local([max_local, Path(max_key).resolve(), Path(Path(max_key).name)])
    if global_max is None:
        log.info("Downloading normalization maxima from s3://%s/%s", bucket, max_key)
        arr = loader.load_numpy(bucket, max_key)
        if arr is None:
            raise RuntimeError(f"Unable to load normalization maxima from s3://{bucket}/{max_key}")
        np.save(max_local, arr)
        global_max = arr

    global_min = np.asarray(global_min, dtype=np.float32)
    global_max = np.asarray(global_max, dtype=np.float32)
    if global_min.shape != global_max.shape:
        raise ValueError(f"Min/Max shape mismatch: {global_min.shape} vs {global_max.shape}")
    if not np.all(np.isfinite(global_min)) or not np.all(np.isfinite(global_max)):
        raise ValueError("Non-finite values in normalization arrays")

    log.info("Loaded normalization arrays (shape %s)", global_min.shape)
    return NormalizationArrays(global_min=global_min, global_max=global_max)


# -------------------------- CACHING HELPERS --------------------------

def _dt_key(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%MZ")

def _x_cache_file(cache_dir: Path, dt: datetime) -> Path:
    return cache_dir / f"xdata_{_dt_key(dt)}.npz"

def _gt_cache_file(cache_dir: Path, dt: datetime) -> Path:
    return cache_dir / f"gt_{_dt_key(dt)}.npy"

def _load_or_build_inputs(
    builder: MRMSDataBuilder,
    target_dt: datetime,
    normalization_arrays: NormalizationArrays,
    cache_dir: Path,
    use_cache: bool,
) -> np.ndarray:
    log = logging.getLogger(__name__)
    x_path = _x_cache_file(cache_dir, target_dt)
    if use_cache and x_path.exists():
        log.info("Loading cached inputs from %s", x_path)
        with np.load(x_path) as npz:
            arr = npz["arr"]
        return arr

    log.info("Building inputs for %s", target_dt.isoformat())
    arr = builder.build_input_tensor(target_dt, normalization_arrays, normalize=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if use_cache:
        np.savez_compressed(x_path, arr=arr.astype(np.float32))
        log.info("Cached inputs to %s", x_path)
    return arr

def _load_or_build_gt(
    builder: MRMSDataBuilder,
    gt_dt: datetime,
    cache_dir: Path,
    use_cache: bool,
) -> np.ndarray:
    log = logging.getLogger(__name__)
    gt_path = _gt_cache_file(cache_dir, gt_dt)
    if use_cache and gt_path.exists():
        log.info("Loading cached ground truth from %s", gt_path)
        return np.load(gt_path)

    log.info("Building ground truth for %s", gt_dt.isoformat())
    gt = builder.build_ground_truth(gt_dt)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if use_cache:
        np.save(gt_path, gt.astype(np.float32))
        log.info("Cached ground truth to %s", gt_path)
    return gt


# -------------------------- TILING --------------------------

@dataclass
class ThresholdResults:
    per_threshold: Dict[float, Dict[str, float]]
    best_by_observation: List[Dict[str, float]]


@dataclass
class PredictionSummary:
    prediction: np.ndarray
    ground_truth: np.ndarray
    metrics: ThresholdResults
    plot_path: Optional[str]
    tiles: List["TileDebugInfo"]


@dataclass
class TileDebugInfo:
    index: int
    y: int
    x: int
    prediction: np.ndarray
    normalized_inputs: np.ndarray
    raw_inputs: Optional[np.ndarray]
    plot_path: Optional[str] = None


class TiledPredictor:
    """Run tiled inference across the CONUS domain with optional ROI gating + sampling."""

    def __init__(
        self,
        model,
        model_type: ModelType,
        tile_size: int,
        stride: int,
        flow_steps: int,
        roi_channel: Optional[int] = None,
        roi_threshold: float = 0.0,
        skip_background: bool = False,
        max_tiles: Optional[int] = None,
        tile_select: str = "random",  # or "cover"
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.tile_size = tile_size
        self.stride = stride
        self.flow_steps = flow_steps
        self.roi_channel = roi_channel
        self.roi_threshold = roi_threshold
        self.skip_background = skip_background
        self.max_tiles = max_tiles
        self.tile_select = tile_select

    def _candidate_origins(self, tensor: np.ndarray) -> List[Tuple[int, int]]:
        _, H, W, C = tensor.shape
        origins: List[Tuple[int, int]] = []
        use_roi = self.skip_background and (self.roi_channel is not None) and (0 <= self.roi_channel < C)

        roi_mask: Optional[np.ndarray] = None
        if use_roi:
            roi_mask = (tensor[..., self.roi_channel] > self.roi_threshold).any(axis=0)

        for y in range(0, H - self.tile_size + 1, self.stride):
            y2 = y + self.tile_size
            for x in range(0, W - self.tile_size + 1, self.stride):
                if roi_mask is not None:
                    if not roi_mask[y:y2, x:x + self.tile_size].any():
                        continue
                origins.append((y, x))

        logging.getLogger(__name__).info("Tile candidates: %d", len(origins))

        if self.max_tiles and len(origins) > self.max_tiles:
            if self.tile_select == "random":
                rng = np.random.default_rng(12345)
                idx = rng.choice(len(origins), size=self.max_tiles, replace=False)
                origins = [origins[i] for i in idx]
            else:  # "cover"
                step = max(1, len(origins) // self.max_tiles)
                origins = origins[::step][: self.max_tiles]
            logging.getLogger(__name__).info("Selected %d tiles (strategy=%s)", len(origins), self.tile_select)

        return origins

    def predict(
        self,
        tensor: np.ndarray,
        raw_tensor: Optional[np.ndarray] = None,
        debug_limit: Optional[int] = None,
    ) -> tuple[np.ndarray, List[TileDebugInfo]]:
        T, H, W, _ = tensor.shape
        predictions = np.zeros((T, H, W), dtype=np.float32)
        counts = np.zeros((T, H, W), dtype=np.float32)

        origins = self._candidate_origins(tensor)
        tiles_processed = 0
        debug_tiles: List[TileDebugInfo] = []

        for tile_index, (y, x) in enumerate(origins):
            tile = tensor[:, y : y + self.tile_size, x : x + self.tile_size, :]
            sequence = self._predict_tile(tile)
            predictions[:, y : y + self.tile_size, x : x + self.tile_size] += sequence
            counts[:, y : y + self.tile_size, x : x + self.tile_size] += 1
            tiles_processed += 1

            if debug_limit is not None and len(debug_tiles) < debug_limit:
                raw_slice: Optional[np.ndarray] = None
                if raw_tensor is not None:
                    raw_slice = raw_tensor[:, y : y + self.tile_size, x : x + self.tile_size, :].copy()
                debug_tiles.append(
                    TileDebugInfo(
                        index=tile_index,
                        y=y,
                        x=x,
                        prediction=sequence.copy(),
                        normalized_inputs=tile.copy(),
                        raw_inputs=raw_slice,
                    )
                )

            if tiles_processed % 100 == 0:
                logging.getLogger(__name__).info("Processed %d tiles", tiles_processed)

        mask = counts > 0
        predictions[mask] /= counts[mask]
        uncovered = int(np.count_nonzero(counts[0] == 0))
        if uncovered:
            logging.getLogger(__name__).warning("%d pixels were not covered by any tile", uncovered)
        logging.getLogger(__name__).info("Processed %d total tiles", tiles_processed)
        return predictions, debug_tiles

    def _predict_tile(self, tile: np.ndarray) -> np.ndarray:
        if self.model_type is ModelType.FLOW:
            return self._predict_flow_tile(tile)
        return self._predict_convgru_tile(tile)

    def _predict_convgru_tile(self, tile: np.ndarray) -> np.ndarray:
        batch = np.expand_dims(tile, axis=0)
        outputs = self.model.predict(batch, verbose=0)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        array = np.asarray(outputs, dtype=np.float32)
        if array.ndim == 5:
            array = array[0, ..., 0]
        elif array.ndim == 4:
            array = array[0]
        return array

    def _predict_flow_tile(self, tile: np.ndarray) -> np.ndarray:
        condition = np.expand_dims(tile.astype(np.float32), axis=0)
        frames: List[np.ndarray] = []
        n_steps = max(1, int(self.flow_steps))
        dt = 1.0 / float(n_steps)
        times = np.linspace(0.0, 1.0 - dt, n_steps, dtype=np.float32)

        for timestep_idx in range(tile.shape[0]):
            state = np.zeros((1, tile.shape[1], tile.shape[2], 1), dtype=np.float32)
            timestep_value = np.array([float(timestep_idx)], dtype=np.float32)
            for t_val in times:
                inputs = {"x_t": state, "condition": condition, "t": np.array([t_val], dtype=np.float32),
                          "timestep_idx": timestep_value}
                velocity = self.model.predict(inputs, verbose=0)
                if isinstance(velocity, (list, tuple)):
                    velocity = velocity[0]
                state = state + velocity.astype(np.float32) * dt
            frames.append(state[0, ..., 0])
        return np.stack(frames, axis=0)


# -------------------------- METRICS --------------------------

def compute_threshold_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> ThresholdResults:
    num_thresholds = len(THRESHOLDS)
    tp_surface = np.zeros((num_thresholds, num_thresholds), dtype=np.float64)
    fp_surface = np.zeros_like(tp_surface)
    fn_surface = np.zeros_like(tp_surface)

    pred_masks = [prediction >= thr for thr in THRESHOLDS]
    obs_masks = [ground_truth >= thr for thr in THRESHOLDS]

    for p_idx, pred_mask in enumerate(pred_masks):
        inv_pred_mask = ~pred_mask
        for o_idx, obs_mask in enumerate(obs_masks):
            hits = float(np.sum(pred_mask & obs_mask))
            false_alarms = float(np.sum(pred_mask & ~obs_mask))
            misses = float(np.sum(inv_pred_mask & obs_mask))
            tp_surface[p_idx, o_idx] = hits
            fp_surface[p_idx, o_idx] = false_alarms
            fn_surface[p_idx, o_idx] = misses

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = tp_surface + fp_surface + fn_surface
        csi_surface = np.where(denom > 0.0, tp_surface / denom, 0.0)
        pod_surface = np.where((tp_surface + fn_surface) > 0.0, tp_surface / (tp_surface + fn_surface), 0.0)
        far_surface = np.where((tp_surface + fp_surface) > 0.0, fp_surface / (tp_surface + fp_surface), 0.0)
        bias_surface = np.where((tp_surface + fn_surface) > 0.0, (tp_surface + fp_surface) / (tp_surface + fn_surface), 0.0)

    per_threshold: Dict[float, Dict[str, float]] = {}
    for idx, threshold in enumerate(THRESHOLDS):
        per_threshold[threshold] = {
            "csi": float(csi_surface[idx, idx]),
            "pod": float(pod_surface[idx, idx]),
            "far": float(far_surface[idx, idx]),
            "bias": float(bias_surface[idx, idx]),
            "tp": float(tp_surface[idx, idx]),
            "fp": float(fp_surface[idx, idx]),
            "fn": float(fn_surface[idx, idx]),
        }

    best_by_observation: List[Dict[str, float]] = []
    for o_idx, obs_threshold in enumerate(THRESHOLDS):
        best_pred_idx = int(np.argmax(csi_surface[:, o_idx]))
        best_by_observation.append(
            {
                "obs_threshold": float(obs_threshold),
                "best_pred_threshold": float(THRESHOLDS[best_pred_idx]),
                "csi": float(csi_surface[best_pred_idx, o_idx]),
                "pod": float(pod_surface[best_pred_idx, o_idx]),
                "far": float(far_surface[best_pred_idx, o_idx]),
                "bias": float(bias_surface[best_pred_idx, o_idx]),
                "tp": float(tp_surface[best_pred_idx, o_idx]),
                "fp": float(fp_surface[best_pred_idx, o_idx]),
                "fn": float(fn_surface[best_pred_idx, o_idx]),
            }
        )

    return ThresholdResults(per_threshold=per_threshold, best_by_observation=best_by_observation)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _print_metrics(metrics: ThresholdResults) -> None:
    print(f"\n{'Threshold':>10} {'CSI':>8} {'POD':>8} {'FAR':>8} {'Bias':>8} {'Hits':>10} {'FAs':>10}")
    print("-" * 80)
    for threshold in THRESHOLDS:
        record = metrics.per_threshold[threshold]
        print(
            f"{int(threshold):>10} mm "
            f"{record['csi']:>8.3f} {record['pod']:>8.3f} {record['far']:>8.3f} {record['bias']:>8.3f} "
            f"{int(record['tp']):>10} {int(record['fp']):>10}"
        )

    print(f"\n{'Obs Thr':>10} {'Best Pred':>12} {'CSI':>8} {'POD':>8} {'FAR':>8} {'Bias':>8}")
    print("-" * 80)
    for record in metrics.best_by_observation:
        print(
            f"{int(record['obs_threshold']):>10} mm "
            f"{int(record['best_pred_threshold']):>12} mm "
            f"{record['csi']:>8.3f} {record['pod']:>8.3f} {record['far']:>8.3f} {record['bias']:>8.3f}"
        )


# -------------------------- FIELD/TILE DEBUG PRINTS --------------------------

def _print_topk_stats(raw_inputs: np.ndarray,
                      normalized_inputs: np.ndarray,
                      mesh_channel: int,
                      tile_size: int,
                      stride: int,
                      k: int) -> None:
    """Rank tiles by max MESH (over T,H,W) and print per-channel stats on those tiles only."""
    T, H, W, C = raw_inputs.shape
    # Precompute per-pixel max over time for MESH channel
    mesh_max_over_t = raw_inputs[..., mesh_channel].max(axis=0)  # (H, W)
    # Enumerate tile origins
    origins: List[Tuple[int, int]] = []
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            origins.append((y, x))
    # Score tiles by max mesh inside tile
    scored: List[Tuple[float, int, int]] = []
    for (y, x) in origins:
        s = float(mesh_max_over_t[y:y+tile_size, x:x+tile_size].max())
        scored.append((s, y, x))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:max(1, int(k))]

    print("\n" + "=" * 72)
    print(f"TOP-{len(top)} TILE STATS  (ranked by max MESH, ch={mesh_channel}, tile={tile_size}, stride={stride})")
    for rank, (score, y, x) in enumerate(top, start=1):
        raw_tile = raw_inputs[:, y:y+tile_size, x:x+tile_size, :]
        norm_tile = normalized_inputs[:, y:y+tile_size, x:x+tile_size, :]
        print(f"\nTile rank {rank:02d} origin=(y={y}, x={x})  maxMESH={score:.2f}")
        for tag, arr in (("RAW", raw_tile), ("NORM", norm_tile)):
            flat = arr.reshape(-1, arr.shape[-1])  # (T*H*W, C)
            mins = flat.min(axis=0)
            maxs = flat.max(axis=0)
            means = flat.mean(axis=0)
            print(f"  {tag} per-channel:")
            for c in range(flat.shape[-1]):
                print(f"    ch{c:02d} min={mins[c]:.3f} max={maxs[c]:.3f} mean={means[c]:.3f}")


def _summarize_array(label: str, array: np.ndarray) -> None:
    print(f"    {label}: min={float(array.min()):.3f}, max={float(array.max()):.3f}, mean={float(array.mean()):.3f}")


def _summarize_channels(label: str, array: np.ndarray) -> None:
    _summarize_array(label, array)
    if array.ndim >= 4:
        for channel_idx in range(array.shape[-1]):
            channel_slice = array[..., channel_idx]
            print(f"      Channel {channel_idx}: min={float(channel_slice.min()):.3f}, "
                  f"max={float(channel_slice.max()):.3f}, mean={float(channel_slice.mean()):.3f}")


def _process_tile_debug_information(
    tiles: List[TileDebugInfo],
    ground_truth: np.ndarray,
    lead_index: int,
    lead_time: int,
    output_dir: Path,
    plot: bool,
) -> None:
    if not tiles:
        return
    print("\n" + "=" * 60)
    print("TILE DEBUG STATISTICS")
    print("=" * 60)
    tile_plot_dir = output_dir / "tiles"
    if plot:
        _ensure_directory(tile_plot_dir)
    for display_idx, tile in enumerate(tiles, start=1):
        print(f"\nTile {display_idx} (index {tile.index}) origin=(y={tile.y}, x={tile.x})")
        if tile.raw_inputs is not None:
            _summarize_channels("Raw inputs", tile.raw_inputs)
        _summarize_channels("Normalized inputs", tile.normalized_inputs)
        prediction_frame = tile.prediction[lead_index]
        _summarize_array("Prediction", prediction_frame)
        gt_tile = ground_truth[tile.y : tile.y + prediction_frame.shape[0], tile.x : tile.x + prediction_frame.shape[1]]
        _summarize_array("Ground truth", gt_tile)
        if plot:
            filename = f"tile_{tile.index:05d}_lead{lead_time:03d}.png"
            plot_path = plot_ground_truth_prediction_difference(
                ground_truth=gt_tile,
                prediction=prediction_frame,
                title=f"Tile {tile.index} (+{lead_time}m)",
                output_dir=str(tile_plot_dir),
                filename=filename,
            )
            tile.plot_path = plot_path
            if plot_path:
                logging.getLogger(__name__).info("Wrote tile comparison plot to %s", plot_path)


# -------------------------- DRIVER --------------------------

def run_verification(
    target_dt: datetime,
    lead_time: int,
    builder: MRMSDataBuilder,
    normalization_arrays: NormalizationArrays,
    normalization_bundle: NormalizationBundle,
    predictor: TiledPredictor,
    output_dir: Path,
    plot: bool,
    tile_limit: Optional[int],
    cache_dir: Path,
    use_cache: bool,
    args: argparse.Namespace,
) -> PredictionSummary:
    if lead_time % 5 != 0:
        raise ValueError("Lead time must be a multiple of 5 minutes")

    # Inputs (cached)
    raw_inputs = _load_or_build_inputs(builder, target_dt, normalization_arrays, cache_dir, use_cache)
    raw_inputs = raw_inputs[::-1, :, :, :]  # reverse time to match training

    # Normalize
    normalized_inputs = normalization_bundle.normalize(raw_inputs)

    # ---- PRINT TOP-K TILE STATS (by MESH) IF REQUESTED ----
    if args.print_topk_stats:
        _print_topk_stats(
            raw_inputs,
            normalized_inputs,
            args.mesh_channel,
            args.tile_size,
            args.stride,
            args.stats_top_k,
        )

    logging.getLogger(__name__).info("Input tensor shape: %s", tuple(normalized_inputs.shape))

    # Predict
    prediction_sequence, tile_debug = predictor.predict(
        normalized_inputs,
        raw_tensor=raw_inputs,
        debug_limit=tile_limit,
    )

    lead_index = lead_time // 5 - 1
    if lead_index < 0 or lead_index >= prediction_sequence.shape[0]:
        raise ValueError(f"Lead time {lead_time} minutes is outside available range")
    prediction = prediction_sequence[lead_index]

    # Ground truth (cached)
    ground_truth_time = target_dt + timedelta(minutes=lead_time)
    ground_truth = _load_or_build_gt(builder, ground_truth_time, cache_dir, use_cache)

    # Metrics & plots
    metrics = compute_threshold_metrics(prediction, ground_truth)
    _print_metrics(metrics)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Prediction - Min: {prediction.min():.2f}, Max: {prediction.max():.2f}, Mean: {prediction.mean():.3f}")
    print(f"Ground Truth - Min: {ground_truth.min():.2f}, Max: {ground_truth.max():.2f}, Mean: {ground_truth.mean():.3f}")
    print(f"Non-zero pixels - Prediction: {(prediction > 0).sum()}, Ground Truth: {(ground_truth > 0).sum()}")

    plot_path: Optional[str] = None
    if plot:
        _ensure_directory(output_dir)
        filename = f"verify_{target_dt.strftime('%Y%m%d_%H%M')}_lead{lead_time:03d}.png"
        plot_path = plot_ground_truth_prediction_difference(
            ground_truth=ground_truth,
            prediction=prediction,
            title=f"{target_dt.isoformat()} (+{lead_time}m)",
            output_dir=str(output_dir),
            filename=filename,
        )
        if plot_path:
            logging.getLogger(__name__).info("Wrote comparison plot to %s", plot_path)

    _process_tile_debug_information(
        tile_debug,
        ground_truth,
        lead_index,
        lead_time,
        output_dir,
        plot,
    )

    return PredictionSummary(
        prediction=prediction,
        ground_truth=ground_truth,
        metrics=metrics,
        plot_path=plot_path,
        tiles=tile_debug,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    datetimes = [_parse_datetime(value) for value in args.datetimes]
    loader = S3ArtifactLoader()  # bare ok
    cache_dir = Path(args.cache_dir)

    normalization = load_normalization_arrays(
        loader,
        args.model_bucket,
        args.norm_min_key,
        args.norm_max_key,
        cache_dir,
    )
    normalization_bundle = NormalizationBundle(
        global_min=normalization.global_min.copy(),
        global_max=normalization.global_max.copy(),
    )

    mrms_config = MRMSConfig(bucket=args.mrms_bucket, tile_size=args.tile_size, stride=args.stride)
    builder = MRMSDataBuilder(mrms_config)

    config = StandaloneConfig(model_path=args.model_path, model_type=args.model_type, flow_steps=args.flow_steps)
    loaded = ModelLoader().load(config)

    predictor = TiledPredictor(
        loaded.model,
        loaded.model_type,
        tile_size=args.tile_size,
        stride=args.stride,
        flow_steps=args.flow_steps,
        roi_channel=args.roi_channel,
        roi_threshold=args.roi_threshold,
        skip_background=args.skip_background,
        max_tiles=(args.max_tiles or args.n_tiles),
        tile_select=args.tile_select,
    )

    output_dir = Path(args.output_dir)

    for dt in datetimes:
        logging.info("=" * 80)
        logging.info("MESH FORECAST VERIFICATION FOR %s", dt.isoformat())
        logging.info("Lead time: %d minutes", args.lead_time)
        summary = run_verification(
            dt,
            args.lead_time,
            builder,
            normalization,
            normalization_bundle,
            predictor,
            output_dir,
            plot=not args.no_plots,
            tile_limit=args.n_tiles,   # still used for debug image count
            cache_dir=cache_dir,
            use_cache=args.use_cache,
            args=args,
        )
        if args.save_outputs:
            _ensure_directory(output_dir)
            prefix = output_dir / f"verify_{dt.strftime('%Y%m%d_%H%M')}_lead{args.lead_time:03d}"
            np.save(f"{prefix}_prediction.npy", summary.prediction)
            np.save(f"{prefix}_ground_truth.npy", summary.ground_truth)
            with open(f"{prefix}_metrics.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {"thresholds": summary.metrics.per_threshold,
                     "best_thresholds": summary.metrics.best_by_observation},
                    handle, indent=2,
                )
            logging.getLogger(__name__).info("Saved outputs to %s", prefix)

    print("\n" + "=" * 60)
    print("SUGGESTED TEST DATES:")
    print("-" * 60)
    for suggestion in SUGGESTED_DATES:
        print(f"  python {Path(__file__).name} --datetime {suggestion} --model_path {args.model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
