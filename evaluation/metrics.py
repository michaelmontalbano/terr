"""CSI metric aggregation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class MetricsSummary:
    best_metrics: List[Dict[str, float]]
    csi_surface: np.ndarray
    pod_surface: np.ndarray
    far_surface: np.ndarray
    bias_surface: np.ndarray


class CSIMetricsCalculator:
    """Accumulate CSI statistics per timestep."""

    def __init__(self, prediction_thresholds: Sequence[float], observation_thresholds: Sequence[float]):
        self.pred_thresholds = np.asarray(prediction_thresholds, dtype=float)
        self.obs_thresholds = np.asarray(observation_thresholds, dtype=float)
        P = len(self.pred_thresholds)
        O = len(self.obs_thresholds)
        self.timestep_stats: Dict[int, Dict[str, np.ndarray]] = {}
        for t in range(12):
            self.timestep_stats[t] = {
                "tp": np.zeros((P, O), dtype=np.float64),
                "fp": np.zeros((P, O), dtype=np.float64),
                "fn": np.zeros((P, O), dtype=np.float64),
            }
        self.overall_tp = np.zeros((P, O), dtype=np.float64)
        self.overall_fp = np.zeros((P, O), dtype=np.float64)
        self.overall_fn = np.zeros((P, O), dtype=np.float64)

    def update(self, predictions: np.ndarray, target: np.ndarray) -> None:
        target_flat = target.reshape(-1)
        print(target_flat.shape)
        predictions = predictions[0]
        for t in range(predictions.shape[0]):
            pred_flat = predictions[t].reshape(-1)
            for p_idx, pred_thr in enumerate(self.pred_thresholds):
                pred_mask = pred_flat >= pred_thr
                inv_pred_mask = ~pred_mask
                for o_idx, obs_thr in enumerate(self.obs_thresholds):
                    obs_mask = target_flat >= obs_thr
                    tp = np.count_nonzero(pred_mask & obs_mask)
                    fp = np.count_nonzero(pred_mask & ~obs_mask)
                    fn = np.count_nonzero(inv_pred_mask & obs_mask)
                    self.timestep_stats[t]["tp"][p_idx, o_idx] += tp
                    self.timestep_stats[t]["fp"][p_idx, o_idx] += fp
                    self.timestep_stats[t]["fn"][p_idx, o_idx] += fn
                    self.overall_tp[p_idx, o_idx] += tp
                    self.overall_fp[p_idx, o_idx] += fp
                    self.overall_fn[p_idx, o_idx] += fn

    @staticmethod
    def _compute(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> MetricsSummary:
        denom = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            csi = np.where(denom > 0, tp / denom, 0.0)
            pod = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            far = np.where((tp + fp) > 0, fp / (tp + fp), 0.0)
            bias = np.where((tp + fn) > 0, (tp + fp) / (tp + fn), 0.0)
        best_idx = np.argmax(csi, axis=0)
        records = []
        for o_idx in range(csi.shape[1]):
            records.append(
                {
                    "obs_threshold": float(o_idx),  # placeholder; replaced later
                    "best_pred_threshold": float(best_idx[o_idx]),  # placeholder
                    "CSI": float(csi[best_idx[o_idx], o_idx]),
                    "POD": float(pod[best_idx[o_idx], o_idx]),
                    "FAR": float(far[best_idx[o_idx], o_idx]),
                    "Bias": float(bias[best_idx[o_idx], o_idx]),
                }
            )
        return MetricsSummary(
            best_metrics=records,
            csi_surface=csi,
            pod_surface=pod,
            far_surface=far,
            bias_surface=bias,
        )

    def finalize(self) -> Dict[str, MetricsSummary]:
        results: Dict[str, MetricsSummary] = {}
        for t, stats in self.timestep_stats.items():
            summary = self._compute(stats["tp"], stats["fp"], stats["fn"])
            for idx, record in enumerate(summary.best_metrics):
                record["obs_threshold"] = float(self.obs_thresholds[idx])
                record["best_pred_threshold"] = float(self.pred_thresholds[int(record["best_pred_threshold"])])
            results[f"timestep_{t}"] = summary
        overall_summary = self._compute(self.overall_tp, self.overall_fp, self.overall_fn)
        for idx, record in enumerate(overall_summary.best_metrics):
            record["obs_threshold"] = float(self.obs_thresholds[idx])
            record["best_pred_threshold"] = float(self.pred_thresholds[int(record["best_pred_threshold"])])
        results["overall"] = overall_summary
        return results
