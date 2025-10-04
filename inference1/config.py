"""Configuration dataclasses shared across evaluation entry-points."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_BUCKET = "dev-grib-bucket"
DEFAULT_TEST_DF_KEY = "dataframes/test.csv"
DEFAULT_NORM_MIN_KEY = "global_mins.npy"
DEFAULT_NORM_MAX_KEY = "global_maxs.npy"


@dataclass
class EvaluationConfig:
    """Base configuration for loading evaluation assets."""

    bucket: str = DEFAULT_BUCKET
    test_df_key: str = DEFAULT_TEST_DF_KEY
    norm_min_key: str = DEFAULT_NORM_MIN_KEY
    norm_max_key: str = DEFAULT_NORM_MAX_KEY
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    n_tiles: Optional[int] = None
    flow_steps: int = 64

    def resolved_model_path(self) -> Optional[str]:
        return self.model_path


@dataclass
class StandaloneConfig(EvaluationConfig):
    """Configuration specific to standalone verification runs."""

    datetimes: List[str] = field(default_factory=list)
    plot_dir: Optional[str] = None
    plot: bool = True
    save_plots: bool = True


@dataclass
class MeshEvaluationConfig(EvaluationConfig):
    """Configuration for mesh-wide CSI evaluation."""

    save_summary: bool = True
    summary_dir: str = "./evaluation_results"
