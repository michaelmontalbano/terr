"""Utilities for building MRMS-based inputs and ground truth on demand."""
from __future__ import annotations

import gzip
import logging
import os
import shutil
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import boto3
import numpy as np
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from scipy.ndimage import binary_dilation, zoom

logger = logging.getLogger(__name__)

MRMS_TIME_FORMAT = "%Y%m%d-%H%M%S"


def _as_naive_utc(dt: datetime) -> datetime:
    """
    Return a timezone-naive datetime in UTC.
    - If dt is aware, convert to UTC and drop tzinfo.
    - If dt is naive, return as-is (assumed UTC).
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@dataclass
class MRMSConfig:
    """Configuration describing how to assemble MRMS tensors."""

    bucket: str = "noaa-mrms-pds"
    target_height: int = 3500
    target_width: int = 7000
    tile_size: int = 256
    stride: int = 128
    n_channels: int = 8
    n_timesteps: int = 12
    offsets: Tuple[int, ...] = (
        0,
        -5,
        -10,
        -15,
        -20,
        -25,
        -30,
        -35,
        -40,
        -45,
        -50,
        -55,
    )
    ground_truth_minutes: Tuple[int, ...] = tuple(range(0, 61, 2))
    dilation_size: int = 10
    mesh_max_field: str = "MESH_Max_60min_00.50"
    mesh_field: str = "MESH_00.50"
    additional_fields: Tuple[str, ...] = (
        "MESH_00.50",
        "HeightCompositeReflectivity_00.50",
        "EchoTop_50_00.50",
        "PrecipRate_00.00",
        "Reflectivity_0C_00.50",
        "Reflectivity_-20C_00.50",
    )


def _channel_names() -> Tuple[str, ...]:
    return (
        "MESH_Max_60min(-60)",
        "MESH_Max_60min(0)",
        "MESH",
        "HeightCompositeReflectivity",
        "EchoTop_50",
        "PrecipRate",
        "Reflectivity_0C",
        "MESH_dilated",
    )


@dataclass
class NormalizationArrays:
    global_min: np.ndarray
    global_max: np.ndarray

    def min(self, channel: int, default: float = 0.0) -> float:
        if channel < len(self.global_min):
            return float(self.global_min[channel])
        return default

    def max(self, channel: int, default: float = 100.0) -> float:
        if channel < len(self.global_max):
            return float(self.global_max[channel])
        return default


class MRMSDataBuilder:
    """Build normalized model inputs and ground-truth swaths directly from MRMS."""

    def __init__(
        self,
        config: MRMSConfig,
        session: Optional[boto3.session.Session] = None,
    ) -> None:
        self.config = config
        self._session = session or boto3.session.Session()
        self._anon_client = self._session.client("s3", config=Config(signature_version=UNSIGNED))
        self._parsed_cache: Dict[Tuple[str, str], List[Tuple[datetime, str]]] = {}

    # ------------------------------------------------------------------
    # S3 helpers
    def _list_files(self, prefix: str) -> List[str]:
        paginator = self._anon_client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def _gather_and_sort_files(self, target_dt: datetime, field: str) -> List[Tuple[datetime, str]]:
        # Normalize to naive UTC for date math and prefix construction
        target_dt = _as_naive_utc(target_dt)

        date_str = target_dt.strftime("%Y%m%d")
        cache_key = (field, date_str)
        if cache_key in self._parsed_cache:
            return self._parsed_cache[cache_key]

        prefixes = [f"CONUS/{field}/{date_str}/"]
        # Also consider adjacent days near midnight boundaries
        if target_dt.hour == 0:
            prev = (target_dt - timedelta(days=1)).strftime("%Y%m%d")
            prefixes.append(f"CONUS/{field}/{prev}/")
        if target_dt.hour == 23:
            nxt = (target_dt + timedelta(days=1)).strftime("%Y%m%d")
            prefixes.append(f"CONUS/{field}/{nxt}/")

        keys: List[str] = []
        for prefix in prefixes:
            keys.extend(self._list_files(prefix))

        parsed: List[Tuple[datetime, str]] = []
        for key in keys:
            try:
                timestamp = key.split("_")[-1].split(".")[0]
                parsed_dt = datetime.strptime(timestamp, MRMS_TIME_FORMAT)  # naive (UTC)
                parsed.append((parsed_dt, key))
            except Exception:
                continue

        parsed.sort(key=lambda item: item[0])
        self._parsed_cache[cache_key] = parsed
        return parsed

    @staticmethod
    def _find_closest(entries: Sequence[Tuple[datetime, str]], target: datetime) -> Optional[str]:
        if not entries:
            return None
        # Ensure target is timezone-naive UTC to match parsed entry datetimes
        target = _as_naive_utc(target)

        times = [item[0] for item in entries]  # all naive UTC
        position = bisect_left(times, target)
        if position == 0:
            return entries[0][1]
        if position >= len(entries):
            return entries[-1][1]
        before = entries[position - 1]
        after = entries[position]
        if after[0] - target < target - before[0]:
            return after[1]
        return before[1]

    def _load_grib_dataset(self, key: str) -> tuple[Optional[xr.Dataset], Optional[str], Optional[str]]:
        """
        Download and decompress a GRIB2 .gz to /tmp, open with cfgrib, and return:
        (dataset, gz_path, grib_path)

        NOTE: The caller is responsible for closing the dataset and deleting paths.
        """
        tmp_base = os.path.join("/tmp", os.path.basename(key))
        gz_path = tmp_base
        grib_path = tmp_base[:-3] if gz_path.endswith(".gz") else f"{tmp_base}.grib2"
        try:
            # Download .gz
            with open(gz_path, "wb") as handle:
                obj = self._anon_client.get_object(Bucket=self.config.bucket, Key=key)
                handle.write(obj["Body"].read())

            # Decompress to .grib2
            with gzip.open(gz_path, "rb") as zipped, open(grib_path, "wb") as out:
                shutil.copyfileobj(zipped, out)

            # Open with cfgrib; keep index in-memory to avoid .idx files
            ds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},  # no on-disk index
            )
            return ds, gz_path, grib_path

        except Exception as exc:
            logger.error("Error loading MRMS file %s: %s", key, exc)
            # Best-effort cleanup if we failed early
            for path in (gz_path, grib_path):
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
            return None, None, None


    def _extract_array(self, dataset: xr.Dataset) -> Optional[np.ndarray]:
        data_var: Optional[xr.DataArray] = None
        # Prefer explicitly named fields but gracefully fall back to the first
        # available data variable.  Recent MRMS files occasionally expose the
        # field under different names (e.g. ``hail_size``) instead of the
        # historical ``unknown`` key that older archives used.
        preferred_names = ("unknown","MESH","MESHMax60min","MaximumEstimatedHailSize","hail_size",)

        for name in preferred_names:
            if name in dataset:
                data_var = dataset[name]
                break

        if data_var is None:
            try:
                first_name = next(iter(dataset.data_vars))
            except StopIteration:
                return None
            logging.getLogger(__name__).debug(
                "Falling back to first data variable '%s' in MRMS dataset", first_name
            )
            data_var = dataset[first_name]

        values = np.asarray(np.ma.getdata(data_var.values))
        if values.ndim > 2:
            values = np.squeeze(values)
        if values.ndim != 2:
            return None

        array = np.flipud(values)
        if array.shape == (self.config.target_width, self.config.target_height):
            array = zoom(array, zoom=0.5, order=2)
        if array.shape != (self.config.target_height, self.config.target_width):
            return None
        array = array.astype(np.float32)
        array[array < 0] = 0
        return array

    # ------------------------------------------------------------------
    # Public API
    def build_input_tensor(
        self,
        target_dt: datetime,
        normalization: NormalizationArrays,
        *,
        normalize: bool = True,
        return_raw: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # Normalize once at entry
        target_dt = _as_naive_utc(target_dt)

        logger.info("Building input tensor for %s", target_dt.isoformat())
        data = np.zeros(
            (
                self.config.n_timesteps,
                self.config.target_height,
                self.config.target_width,
                self.config.n_channels,
            ),
            dtype=np.float32,
        )

        mesh_entries = self._gather_and_sort_files(target_dt, self.config.mesh_max_field)
        for timestep_idx, offset in enumerate(self.config.offsets):
            current_dt = target_dt + timedelta(minutes=offset)
            minus_sixty_key = self._find_closest(mesh_entries, current_dt + timedelta(minutes=-60))
            now_key = self._find_closest(mesh_entries, current_dt)
            data[timestep_idx, ..., 0] = self._load_mesh_field(minus_sixty_key)
            data[timestep_idx, ..., 1] = self._load_mesh_field(now_key)

        field_channels = {
            "MESH_00.50": 2,
            "HeightCompositeReflectivity_00.50": 3,
            "EchoTop_50_00.50": 4,
            "PrecipRate_00.00": 5,
            "Reflectivity_0C_00.50": 6,
            "Reflectivity_-20C_00.50": 7,
        }

        for field, channel in field_channels.items():
            logger.info("Processing field %s", field)
            entries = self._gather_and_sort_files(target_dt, field)
            for timestep_idx, offset in enumerate(self.config.offsets):
                current_dt = target_dt + timedelta(minutes=offset)
                key = self._find_closest(entries, current_dt)
                if key is None:
                    continue
                array = self._load_mesh_field(key)
                if array is None:
                    continue
                data[timestep_idx, ..., channel] = array
                if field == self.config.mesh_field and channel == 2:
                    data[timestep_idx, ..., 7] = self._enlarge_hail_region(array)

        self._log_channel_stats("before normalization", data)
        if normalize:
            normalized = self._normalize(data, normalization)
            self._log_channel_stats("after normalization", normalized)
        else:
            normalized = data.copy()
        if return_raw:
            return normalized.copy(), data.copy()
        return normalized

    def build_ground_truth(self, target_dt: datetime) -> np.ndarray:
        # Normalize once at entry
        target_dt = _as_naive_utc(target_dt)

        logger.info("Building ground truth for %s", target_dt.isoformat())
        entries = self._gather_and_sort_files(target_dt, self.config.mesh_field)
        arrays: List[np.ndarray] = []
        for minutes_back in self.config.ground_truth_minutes:
            lookup_dt = target_dt - timedelta(minutes=minutes_back)
            key = self._find_closest(entries, lookup_dt)
            if key is None:
                continue
            array = self._load_mesh_field(key)
            if array is None:
                continue
            arrays.append(array)
        if arrays:
            stacked = np.maximum.reduce(arrays)
            logger.info(
                "Built ground truth from %d timesteps (max %.1f, non-zero %d)",
                len(arrays),
                float(stacked.max()),
                int((stacked > 0).sum()),
            )
            return stacked
        logger.error("No MESH data available for %s", target_dt.isoformat())
        return np.zeros((self.config.target_height, self.config.target_width), dtype=np.float32)

    def _load_mesh_field(self, key: Optional[str]) -> np.ndarray:
        if key is None:
            return np.zeros((self.config.target_height, self.config.target_width), dtype=np.float32)

        ds, gz_path, grib_path = self._load_grib_dataset(key)
        if ds is None:
            return np.zeros((self.config.target_height, self.config.target_width), dtype=np.float32)

        try:
            array = self._extract_array(ds)
        finally:
            # Close dataset before deleting the files it still references
            try:
                ds.close()
            except Exception:
                pass
            # Now safe to delete temp files
            for path in (gz_path, grib_path):
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

        if array is None:
            logger.error("Dataset %s did not contain expected data", key)
            return np.zeros((self.config.target_height, self.config.target_width), dtype=np.float32)
        return array


    def _normalize(self, tensor: np.ndarray, normalization: NormalizationArrays) -> np.ndarray:
        normalized = tensor.copy()
        for t in range(tensor.shape[0]):
            for c in range(tensor.shape[-1]):
                if c == 2:  # Raw MESH channel remains unnormalized
                    continue
                channel_slice = tensor[t, :, :, c]
                min_val = normalization.min(c)
                max_val = normalization.max(c)
                denom = max(max_val - min_val, 1e-5)
                norm = (channel_slice - min_val) / denom
                norm = np.clip(norm, 1e-5, 1.0)
                normalized[t, :, :, c] = norm.astype(np.float32)
        return normalized

    def _enlarge_hail_region(self, array: np.ndarray) -> np.ndarray:
        hail_mask = array >= 20
        structure = np.ones(
            (2 * self.config.dilation_size + 1, 2 * self.config.dilation_size + 1),
            dtype=bool,
        )
        dilated = binary_dilation(hail_mask, structure=structure)
        return np.where(dilated, 20.0, 0.0).astype(np.float32)

    def _log_channel_stats(self, label: str, tensor: np.ndarray) -> None:
        names = _channel_names()
        logger.info("Channel statistics %s:", label)
        for idx in range(min(tensor.shape[-1], len(names))):
            channel = tensor[..., idx]
            logger.info(
                "  Channel %d (%s): min=%.3f max=%.3f mean=%.3f",
                idx,
                names[idx],
                float(channel.min()),
                float(channel.max()),
                float(channel.mean()),
            )
