"""Orchestrates standalone verification runs for individual tiles."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .config import StandaloneConfig
from .data import EvaluationDataRepository, NormalizationBundle
from .metrics import CSIMetricsCalculator
from .plotting import plot_ground_truth_prediction_difference
from .predictors import PredictionResult, PredictorFactory

logger = logging.getLogger(__name__)


PREDICTION_THRESHOLDS = np.arange(1.0, 61.0, 1.0)
OBS_THRESHOLDS = np.array([5, 10, 20, 30, 40, 50, 70], dtype=float)
TIMESTEPS = 12
INPUT_CHANNELS = 8


@dataclass
class SampleEvaluation:
    index: int
    metadata: Dict[str, Any]
    prediction: PredictionResult
    target_swath: np.ndarray
    plot_path: Optional[str]


class StandaloneVerifier:
    def __init__(
        self,
        config: StandaloneConfig,
        repository: EvaluationDataRepository,
        normalization: NormalizationBundle,
        loaded_model,
    ):
        self.config = config
        self.repository = repository
        self.normalization = normalization
        self.loaded_model = loaded_model
        self.predictor = PredictorFactory.create(loaded_model, config)

    def _prepare_inputs(self, inputs: np.ndarray) -> np.ndarray:
        arr = np.asarray(inputs, dtype=np.float32)
        if arr.ndim < 4:
            arr = arr[..., np.newaxis]
        arr = arr[::-1, :, :, :]
        if arr.shape[0] < TIMESTEPS:
            raise ValueError(f"Expected at least {TIMESTEPS} timesteps; got {arr.shape[0]}")
        arr = arr[:TIMESTEPS]
        if arr.shape[-1] < INPUT_CHANNELS:
            raise ValueError(
                f"Expected at least {INPUT_CHANNELS} channels; got {arr.shape[-1]}"
            )
        arr = arr[..., :INPUT_CHANNELS]
        return self.normalization.normalize(arr)

    @staticmethod
    def _prepare_target(targets: np.ndarray) -> np.ndarray:
        arr = np.asarray(targets, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :, np.newaxis]
        elif arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[np.newaxis, :, :, :]
            else:
                arr = arr[:, :, :, np.newaxis]
        if arr.ndim != 4:
            raise ValueError(f"Unexpected target array shape {arr.shape}")
        return arr

    def _target_swath(self, target_sequence: np.ndarray) -> np.ndarray:
        if target_sequence.shape[0] >= 1:
            return target_sequence[-1, :, :, 0]
        raise ValueError("Target sequence has no timesteps")

    def evaluate(self) -> Dict[str, Any]:
        rows = self.repository.select_rows(self.config.n_tiles, self.config.datetimes)
        metrics = CSIMetricsCalculator(PREDICTION_THRESHOLDS, OBS_THRESHOLDS)
        evaluations: List[SampleEvaluation] = []
        for idx, row in rows.iterrows():
            try:
                inputs, targets = self.repository.load_sample(row)
                norm_inputs = self._prepare_inputs(inputs)
                target_sequence = self._prepare_target(targets)
                target_swath = self._target_swath(target_sequence)
                prediction = self.predictor.predict(norm_inputs)
            except Exception as exc:
                logger.error("Failed to evaluate row %d: %s", idx, exc)
                continue

            metrics.update(prediction.sequence, target_swath)

            plot_path: Optional[str] = None
            if self.config.plot and self.config.save_plots:
                plot_title = f"Tile {idx} ({self.loaded_model.model_type.value})"
                plot_path = plot_ground_truth_prediction_difference(
                    ground_truth=target_swath,
                    prediction=np.squeeze(prediction.final_frame),
                    title=plot_title,
                    output_dir=self.config.plot_dir,
                    filename=f"tile_{idx:05d}.png",
                )

            meta = {col: row[col] for col in row.index if isinstance(row[col], (str, int, float))}
            evaluations.append(
                SampleEvaluation(
                    index=idx,
                    metadata=meta,
                    prediction=prediction,
                    target_swath=target_swath,
                    plot_path=plot_path,
                )
            )

        summary = metrics.finalize()
        return {
            "model_type": self.loaded_model.model_type.value,
            "num_samples": len(evaluations),
            "evaluations": evaluations,
            "metrics": summary,
        }
