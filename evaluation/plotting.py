"""Plotting utilities for evaluation outputs."""
from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_ground_truth_prediction_difference(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[str]:
    """Create a GT/prediction/difference comparison plot."""
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    difference = prediction - ground_truth

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, subtitle in zip(
        axes,
        [ground_truth, prediction, difference],
        ["Ground Truth", "Prediction", "Prediction - GT"],
    ):
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(subtitle)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()

    saved_path: Optional[str] = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = filename or f"comparison_{np.random.randint(0, 1_000_000):06d}.png"
        saved_path = os.path.join(output_dir, filename)
        fig.savefig(saved_path)
        logger.info("Saved comparison plot to %s", saved_path)
    plt.close(fig)
    return saved_path
