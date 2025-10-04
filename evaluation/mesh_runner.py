"""Evaluator that aggregates CSI metrics across the test dataframe."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .config import MeshEvaluationConfig
from .data import EvaluationDataRepository, NormalizationBundle
from .metrics import CSIMetricsCalculator
from .plotting import plot_ground_truth_prediction_difference
from .predictors import PredictorFactory

logger = logging.getLogger(__name__)


PREDICTION_THRESHOLDS = np.arange(1.0, 61.0, 1.0)
OBS_THRESHOLDS = np.array([5, 10, 20, 30, 40, 50, 70], dtype=float)
TIMESTEPS = 12
INPUT_CHANNELS = 8


def _prepare_inputs(normalization: NormalizationBundle, inputs: np.ndarray) -> np.ndarray:
    arr = np.asarray(inputs, dtype=np.float32)
    if arr.ndim < 4:
        arr = arr[..., np.newaxis]
    arr = arr[::-1, :, :, :]
    if arr.shape[0] < TIMESTEPS:
        raise ValueError(f"Expected at least {TIMESTEPS} timesteps; got {arr.shape[0]}")
    arr = arr[:TIMESTEPS]
    if arr.shape[-1] < INPUT_CHANNELS:
        raise ValueError(f"Expected at least {INPUT_CHANNELS} channels; got {arr.shape[-1]}")
    arr = arr[..., :INPUT_CHANNELS]
    return normalization.normalize(arr)


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
        raise ValueError(f"Unexpected target shape {arr.shape}")
    return arr


def _target_swath(target_sequence: np.ndarray) -> np.ndarray:
    if target_sequence.shape[0] >= 1:
        return target_sequence[-1, :, :, 0]
    raise ValueError("Target sequence has no timesteps")


@dataclass
class MeshSampleSummary:
    index: int
    prediction_stats: Dict[str, float]
    target_stats: Dict[str, float]
    plot_path: Optional[str]


class MeshEvaluator:
    def __init__(
        self,
        config: MeshEvaluationConfig,
        repository: EvaluationDataRepository,
        normalization: NormalizationBundle,
        loaded_model,
    ):
        self.config = config
        self.repository = repository
        self.normalization = normalization
        self.loaded_model = loaded_model
        self.predictor = PredictorFactory.create(loaded_model, config)

    def evaluate(self) -> Dict[str, Any]:
        rows = self.repository.select_rows(self.config.n_tiles, datetimes=[])
        metrics = CSIMetricsCalculator(PREDICTION_THRESHOLDS, OBS_THRESHOLDS)
        summaries: List[MeshSampleSummary] = []
        successes = 0
        failures = 0
        for idx, row in rows.iterrows():
            try:
                inputs, targets = self.repository.load_sample(row)
                norm_inputs = _prepare_inputs(self.normalization, inputs)
                target_sequence = _prepare_target(targets)
                target_swath = _target_swath(target_sequence)
                prediction = self.predictor.predict(norm_inputs)
            except Exception as exc:
                logger.error("Failed to evaluate row %d: %s", idx, exc)
                failures += 1
                continue

            successes += 1
            metrics.update(prediction.sequence, target_swath)
            pred_stats = {
                "min": float(np.min(prediction.sequence)),
                "max": float(np.max(prediction.sequence)),
                "mean": float(np.mean(prediction.sequence)),
                "std": float(np.std(prediction.sequence)),
            }
            target_stats = {
                "min": float(np.min(target_swath)),
                "max": float(np.max(target_swath)),
                "mean": float(np.mean(target_swath)),
                "std": float(np.std(target_swath)),
            }

            plot_path: Optional[str] = None
            if self.config.save_summary:
                plot_path = plot_ground_truth_prediction_difference(
                    ground_truth=target_swath,
                    prediction=np.squeeze(prediction.final_frame[-1]),
                    title=f"Mesh Eval Tile {idx}",
                    output_dir=self.config.summary_dir,
                    filename=f"mesh_tile_{idx:05d}.png",
                )

            summaries.append(
                MeshSampleSummary(
                    index=idx,
                    prediction_stats=pred_stats,
                    target_stats=target_stats,
                    plot_path=plot_path,
                )
            )

        metrics_summary = metrics.finalize()
        result = {
            "model_type": self.loaded_model.model_type.value,
            "num_samples": len(rows),
            "successful": successes,
            "failed": failures,
            "metrics": metrics_summary,
            "samples": summaries,
        }

        if self.config.save_summary:
            os.makedirs(self.config.summary_dir, exist_ok=True)
            summary_path = os.path.join(self.config.summary_dir, "mesh_evaluation.json")
            serializable = {
                **{k: v for k, v in result.items() if k not in {"samples"}},
                "samples": [
                    {
                        "index": summary.index,
                        "prediction_stats": summary.prediction_stats,
                        "target_stats": summary.target_stats,
                        "plot_path": summary.plot_path,
                    }
                    for summary in summaries
                ],
            }
            with open(summary_path, "w", encoding="utf-8") as fp:
                json.dump(serializable, fp, indent=2)
            logger.info("Wrote mesh evaluation summary to %s", summary_path)

        return result
