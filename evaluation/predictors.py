"""Prediction helpers for different model families."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import EvaluationConfig
from .models import LoadedModel, ModelType

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    sequence: np.ndarray
    final_frame: np.ndarray


class BasePredictor:
    def __init__(self, loaded: LoadedModel, config: EvaluationConfig):
        self.model = loaded.model
        self.model_type = loaded.model_type
        self.config = config

    def predict(self, inputs: np.ndarray) -> PredictionResult:
        raise NotImplementedError


class ConvGRUPredictor(BasePredictor):
    def predict(self, inputs: np.ndarray) -> PredictionResult:
        batch = np.expand_dims(inputs, axis=0)
        outputs = self.model.predict(batch, verbose=0)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        sequence = np.asarray(outputs, dtype=np.float32)
        final_frame = sequence[-1]
        return PredictionResult(sequence=sequence, final_frame=final_frame)

class FlowPredictor(BasePredictor):
    def _integrate_single_timestep(self, condition_batch, timestep_idx):
        condition_batch = condition_batch[::-1,:,:,:]
        H, W = condition_batch.shape[2], condition_batch.shape[3]
        n_steps = max(1, int(self.config.flow_steps))
        dt = 1.0 / n_steps
        times = np.linspace(0.0, 1.0 - dt, n_steps, dtype=np.float32)

        timestep_idx_arr = np.array([int(timestep_idx)], dtype=np.int32)

        # Start from pure noise (t=0 end)
        rng = np.random.default_rng(int(getattr(self.config, "flow_seed", 0)))
        state = rng.standard_normal((1, H, W, 1)).astype(np.float32)
        for i in range(condition_batch.shape[-1]):
            print(f"condition_batch[:,:,:,{i}] mean:", np.mean(condition_batch[:, :, :, i]))
            print(f"condition_batch[:,:,:,{i}] max:", np.max(condition_batch[:, :, :, i]))
            print(f"condition_batch[:,:,:,{i}] min:", np.min(condition_batch[:, :, :, i]))
        for t_val in times:
            t_arr = np.array([t_val], dtype=np.float32)
            inputs = {
                "x_t": state,
                "condition": condition_batch,
                "t": t_arr,
                "timestep_idx": timestep_idx_arr,
            }
            v = self.model.predict(inputs, verbose=0)
            if isinstance(v, (list, tuple)):
                v = v[0]
            # NOTE: plus sign (target − noise definition)
            state = state + v.astype(np.float32) * dt
        return state[0]


    def predict(self, inputs: np.ndarray) -> PredictionResult:
        """
        inputs: (T, H, W, C) from MRMS builder.
        We generate one frame per discrete lead (timestep_idx = 0..T-1), then keep the last.
        """
        assert inputs.ndim == 4, f"Expected (T,H,W,C), got {inputs.shape}"
        condition = np.expand_dims(inputs.astype(np.float32, copy=False), axis=0)  # (1,T,H,W,C)

        frames = []
        T = inputs.shape[0]
        for idx in range(T):
            frames.append(self._integrate_single_timestep(condition, idx))
        sequence = np.stack(frames, axis=0)         # (T,H,W,1)
        final_frame = sequence[-1]                  # last lead (e.g., 60-min)
        return PredictionResult(sequence=sequence, final_frame=final_frame)



class PredictorFactory:
    @staticmethod
    def create(loaded: LoadedModel, config: EvaluationConfig) -> BasePredictor:
        if loaded.model_type is ModelType.FLOW:
            return FlowPredictor(loaded, config)
        return ConvGRUPredictor(loaded, config)
