"""Data access helpers for evaluation scripts."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Optional, Sequence, Tuple

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Channels to min/max normalize (mask channel index 7 excluded)
DEFAULT_NORMALIZED_CHANNELS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)


class S3ArtifactLoader:
    """Load numpy arrays and CSV data from S3 with optional local caching."""

    def __init__(self, cache_dir: str = "./cache", session: Optional[boto3.session.Session] = None):
        self._session = session or boto3.session.Session()
        self._s3 = self._session.client("s3")
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    def _cache_path(self, bucket: str, key: str, suffix: str) -> str:
        safe_key = key.replace("/", "_")
        path = os.path.join(self._cache_dir, bucket, f"{safe_key}{suffix}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def load_numpy(self, bucket: str, key: str) -> Optional[np.ndarray]:
        cache_path = self._cache_path(bucket, key, ".npy")
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                logger.warning("Failed to read cached numpy %s; refreshing", cache_path)
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        try:
            obj = self._s3.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            arr = np.load(BytesIO(data))
            with open(cache_path, "wb") as fp:
                fp.write(data)
            return arr
        except Exception as exc:
            logger.error("Unable to load numpy from s3://%s/%s: %s", bucket, key, exc)
            return None

    def load_csv(self, bucket: str, key: str) -> Optional[pd.DataFrame]:
        cache_path = self._cache_path(bucket, key, ".csv")
        if os.path.exists(cache_path):
            try:
                return pd.read_csv(cache_path)
            except Exception:
                logger.warning("Failed to read cached csv %s; refreshing", cache_path)
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        try:
            obj = self._s3.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            df = pd.read_csv(BytesIO(data))
            df.to_csv(cache_path, index=False)
            return df
        except Exception as exc:
            logger.error("Unable to load csv from s3://%s/%s: %s", bucket, key, exc)
            return None


@dataclass
class NormalizationBundle:
    """Holds global normalization arrays and applies them to inputs."""

    global_min: np.ndarray
    global_max: np.ndarray
    normalized_channels: Tuple[int, ...] = DEFAULT_NORMALIZED_CHANNELS

    @classmethod
    def from_loader(
        cls,
        loader: S3ArtifactLoader,
        bucket: str,
        min_key: str,
        max_key: str,
    ) -> "NormalizationBundle":
        gmin = loader.load_numpy(bucket, min_key)
        gmax = loader.load_numpy(bucket, max_key)
        if gmin is None or gmax is None:
            raise ValueError("Failed to load normalization arrays from S3")
        return cls(global_min=gmin.astype(np.float32), global_max=gmax.astype(np.float32))

    def normalize(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim != 4:
            raise ValueError(f"Expected input with shape (T, H, W, C); got {sample.shape}")
        normalized = np.zeros_like(sample, dtype=np.float32)
        print(sample.shape)
        for c in range(sample.shape[-1]):
            channel_slice = sample[..., c]
            if c in self.normalized_channels:
                min_val = float(self.global_min[c])
                max_val = float(self.global_max[c])
                denom = max(max_val - min_val, 1e-5)
                norm = (channel_slice - min_val) / denom
                norm = np.clip(norm, 1e-5, 1.0)
                normalized[..., c] = norm.astype(np.float32)
            else:
                normalized[..., c] = channel_slice
        return normalized


def infer_target_key(input_key: str) -> str:
    """Infer the target key given an input key following project conventions."""
    if "/input/" in input_key:
        return input_key.replace("/input/", "/mesh_swath_intervals/")
    if "input/" in input_key:
        return input_key.replace("input/", "mesh_swath_intervals/")
    if "input" in input_key:
        suffix = input_key.split("input", 1)[1]
        return f"data/int5/mesh_swath_intervals{suffix}"
    base = os.path.basename(input_key)
    return input_key.replace(base, f"mesh_swath_intervals_{base}")


class EvaluationDataRepository:
    """Wraps S3 access plus dataframe filtering utilities."""

    def __init__(self, loader: S3ArtifactLoader, bucket: str, dataframe_key: str):
        self.loader = loader
        self.bucket = bucket
        self.dataframe_key = dataframe_key

    def load_index(self) -> pd.DataFrame:
        df = self.loader.load_csv(self.bucket, self.dataframe_key)
        if df is None:
            raise RuntimeError(
                f"Unable to load evaluation dataframe from s3://{self.bucket}/{self.dataframe_key}"
            )
        return df

    @staticmethod
    def _candidate_datetime_columns(df: pd.DataFrame) -> List[str]:
        cols = []
        for col in df.columns:
            lower = col.lower()
            if "date" in lower or "time" in lower or "datetime" in lower:
                cols.append(col)
        return cols

    @staticmethod
    def _normalize_datetimes(values: Sequence[str]) -> Tuple[List[pd.Timestamp], List[str]]:
        normalized: List[pd.Timestamp] = []
        raw_strings: List[str] = []
        for value in values:
            if not value:
                continue
            raw = value.strip()
            if not raw:
                continue
            raw_strings.append(raw)
            try:
                normalized.append(pd.to_datetime(raw, utc=True))
            except Exception:
                logger.warning("Failed to parse datetime filter '%s'", raw)
        return normalized, raw_strings

    @staticmethod
    def _expanded_datetime_strings(
        normalized: Sequence[pd.Timestamp], raw_strings: Sequence[str]
    ) -> set:
        expanded: set = set()
        for ts in normalized:
            try:
                iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                compact = ts.strftime("%Y%m%d%H%M%S")
                expanded.update({iso, iso.replace("T", " "), compact})
                expanded.add(compact[:8])
                expanded.add(compact[8:])
            except Exception:
                continue
        for raw in raw_strings:
            stripped = raw.strip()
            if not stripped:
                continue
            expanded.add(stripped)
            expanded.add(stripped.replace("Z", ""))
            expanded.add(stripped.replace("-", "").replace(":", "").replace("T", ""))
            expanded.add(stripped.replace(":", "").replace("T", " "))
        return {s for s in expanded if s}

    @staticmethod
    def _normalize_date_component(value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        text = text.replace("-", "").replace("/", "")
        try:
            if text.isdigit():
                return text.zfill(8)
            parsed = pd.to_datetime(text, utc=True, errors="coerce")
            if pd.isna(parsed):
                return ""
            return parsed.strftime("%Y%m%d")
        except Exception:
            return ""

    @staticmethod
    def _normalize_time_component(value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        text = text.replace(":", "")
        try:
            if text.isdigit():
                return text.zfill(6)
            parsed = pd.to_datetime(text, utc=True, errors="coerce")
            if pd.isna(parsed):
                return ""
            return parsed.strftime("%H%M%S")
        except Exception:
            return ""

    def filter_by_datetimes(self, df: pd.DataFrame, datetimes: Sequence[str]) -> pd.DataFrame:
        if not datetimes:
            return df
        normalized, raw_strings = self._normalize_datetimes(datetimes)
        if not normalized and not raw_strings:
            return df
        candidates = self._candidate_datetime_columns(df)
        column_lookup = {col.lower(): col for col in df.columns}
        date_col = column_lookup.get("date")
        time_col = column_lookup.get("timestamp")
        if not candidates and (not date_col or not time_col):
            logger.warning("Datetime filters provided but dataframe lacks datetime-like columns")
            return df.iloc[0:0]
        expanded_strings = self._expanded_datetime_strings(normalized, raw_strings)
        mask = pd.Series(False, index=df.index, dtype=bool)
        for column in candidates:
            series = df[column]
            parsed = pd.to_datetime(series, utc=True, errors="coerce")
            if normalized:
                mask |= parsed.isin(normalized)
            if expanded_strings:
                mask |= series.astype(str).str.strip().isin(expanded_strings)
        if date_col and time_col:
            date_strings = df[date_col].apply(self._normalize_date_component)
            time_strings = df[time_col].apply(self._normalize_time_component)
            valid_components = (date_strings.str.len() == 8) & (time_strings.str.len() == 6)
            combined_compact = pd.Series("", index=df.index, dtype=object)
            combined_iso = pd.Series("", index=df.index, dtype=object)
            combined_compact.loc[valid_components] = (
                date_strings.loc[valid_components] + time_strings.loc[valid_components]
            )
            combined_iso.loc[valid_components] = (
                date_strings.loc[valid_components].str.slice(0, 4)
                + "-"
                + date_strings.loc[valid_components].str.slice(4, 6)
                + "-"
                + date_strings.loc[valid_components].str.slice(6, 8)
                + "T"
                + time_strings.loc[valid_components].str.slice(0, 2)
                + ":"
                + time_strings.loc[valid_components].str.slice(2, 4)
                + ":"
                + time_strings.loc[valid_components].str.slice(4, 6)
                + "Z"
            )
            parsed_combo = pd.to_datetime(
                combined_compact.where(valid_components),
                format="%Y%m%d%H%M%S",
                errors="coerce",
                utc=True,
            )
            if normalized:
                mask |= parsed_combo.isin(normalized)
            if expanded_strings:
                mask |= combined_compact.isin(expanded_strings)
                mask |= combined_iso.isin(expanded_strings)
        filtered = df[mask]
        logger.info("Datetime filter reduced rows from %d to %d", len(df), len(filtered))
        return filtered

    def select_rows(self, n_tiles: Optional[int], datetimes: Sequence[str]) -> pd.DataFrame:
        df = self.load_index()
        if datetimes:
            df = self.filter_by_datetimes(df, datetimes)
        if df.empty:
            if datetimes:
                raise RuntimeError(
                    "No samples match the provided datetime filters: "
                    + ", ".join(datetimes)
                )
            raise RuntimeError("No samples available for evaluation")
        if n_tiles is not None and n_tiles > 0:
            df = df.head(n_tiles)
        return df.reset_index(drop=True)

    def load_sample(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        input_key = row.get("file_path") or row.get("input_path") or row.get("inputs_path")
        target_key = row.get("target_path") or row.get("target") or row.get("y_path")
        if not isinstance(input_key, str) or not input_key.strip():
            raise RuntimeError("Row is missing an input path column")
        input_key = input_key.strip()
        if not isinstance(target_key, str) or not target_key.strip():
            target_key = infer_target_key(input_key)
        target_key = target_key.strip()
        inputs = self.loader.load_numpy(self.bucket, input_key)
        targets = self.loader.load_numpy(self.bucket, target_key)
        if inputs is None or targets is None:
            raise RuntimeError(f"Failed to load arrays for {input_key} / {target_key}")
        if inputs.ndim < 4:
            inputs = np.expand_dims(inputs, axis=-1)
        inputs = inputs.astype(np.float32)
        if targets.ndim == 2:
            targets = targets[np.newaxis, :, :, np.newaxis]
        elif targets.ndim == 3:
            if targets.shape[0] >= inputs.shape[0]:
                targets = targets[:, :, :, np.newaxis]
            else:
                targets = targets[np.newaxis, :, :, np.newaxis]
        elif targets.ndim == 4:
            targets = targets.astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected target shape {targets.shape}")
        return inputs, targets
