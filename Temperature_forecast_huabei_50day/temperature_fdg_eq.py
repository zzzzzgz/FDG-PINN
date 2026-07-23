"""
Temperature field interface for FDG-PINN style data reconstruction.

The input CSV is expected to contain normalized coordinates:
    system:index, day_index, lat, lon, temp_C

The runner uses inputs in the order:
    x = normalized longitude, y = normalized latitude, t = normalized training-time coordinate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "FDG_PINN_HUABEI_50DAY_Temperature_preprocessed.csv"
RESULTS_DIR = BASE_DIR / "fdg_temperature_results"

NAME = "temperature_fdg_eq"
INPUT_DIM = 3
OUTPUT_DIM = 1
TRAIN_DAY_START = 1
TRAIN_DAY_END = 20
FORECAST_DAY_START = 21
FORECAST_DAY_END = 50


@dataclass(frozen=True)
class TemperatureData:
    system_index: np.ndarray
    day: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    temp_c: np.ndarray
    unique_days: np.ndarray
    unique_lats: np.ndarray
    unique_lons: np.ndarray
    is_boundary: np.ndarray
    valid_spatial_mask: np.ndarray
    boundary_spatial_mask: np.ndarray


def load_temperature_data(csv_path: Path = DATA_CSV) -> TemperatureData:
    system_index: list[str] = []
    day: list[int] = []
    lat: list[float] = []
    lon: list[float] = []
    temp_c: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["system:index", "day_index", "lat", "lon", "temp_C"]
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"Missing columns in {csv_path}: {missing}")

        for row in reader:
            system_index.append(row["system:index"])
            day.append(int(float(row["day_index"])))
            lat.append(float(row["lat"]))
            lon.append(float(row["lon"]))
            temp_c.append(float(row["temp_C"]))

    day_arr = np.asarray(day, dtype=np.int64)
    lat_arr = np.asarray(lat, dtype=np.float32)
    lon_arr = np.asarray(lon, dtype=np.float32)
    temp_arr = np.asarray(temp_c, dtype=np.float32)
    unique_lats = np.unique(lat_arr)
    unique_lons = np.unique(lon_arr)
    unique_days = np.unique(day_arr)

    day_index = np.searchsorted(unique_days, day_arr)
    lat_index = np.searchsorted(unique_lats, lat_arr)
    lon_index = np.searchsorted(unique_lons, lon_arr)
    presence = np.zeros((unique_days.size, unique_lats.size, unique_lons.size), dtype=bool)
    presence[day_index, lat_index, lon_index] = True
    if not np.all(presence == presence[0]):
        raise ValueError("The valid spatial mask is not identical on every day")

    valid_spatial_mask = presence[0]
    padded = np.pad(valid_spatial_mask, 1, mode="constant", constant_values=False)
    four_neighbor_interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary_spatial_mask = valid_spatial_mask & (~four_neighbor_interior)
    is_boundary = boundary_spatial_mask[lat_index, lon_index]

    return TemperatureData(
        system_index=np.asarray(system_index, dtype=object),
        day=day_arr,
        lat=lat_arr,
        lon=lon_arr,
        temp_c=temp_arr,
        unique_days=unique_days,
        unique_lats=unique_lats,
        unique_lons=unique_lons,
        is_boundary=is_boundary,
        valid_spatial_mask=valid_spatial_mask,
        boundary_spatial_mask=boundary_spatial_mask,
    )


def make_inputs(
    data: TemperatureData,
    indices: np.ndarray,
    train_day_start: int = TRAIN_DAY_START,
    train_day_end: int = TRAIN_DAY_END,
) -> np.ndarray:
    """Return model inputs [lon, lat, t] with the training window mapped to [0, 1]."""
    span = float(train_day_end - train_day_start)
    if span <= 0.0:
        raise ValueError("train_day_end must be greater than train_day_start")
    t = (data.day[indices].astype(np.float32) - float(train_day_start)) / span
    return np.stack([data.lon[indices], data.lat[indices], t], axis=1).astype(np.float32)


def make_targets(data: TemperatureData, indices: np.ndarray) -> np.ndarray:
    return data.temp_c[indices, None].astype(np.float32)


def select_training_indices(
    data: TemperatureData,
    rng: np.random.Generator,
    ic_all: bool = True,
    bc_fraction: float = 0.20,
    data_fraction: float = 0.10,
    train_day_start: int = TRAIN_DAY_START,
    train_day_end: int = TRAIN_DAY_END,
) -> dict[str, np.ndarray]:
    train_mask = (data.day >= train_day_start) & (data.day <= train_day_end)
    ic_mask = train_mask & (data.day == train_day_start)
    bc_mask = train_mask & data.is_boundary
    interior_mask = train_mask & (~data.is_boundary)

    ic_indices = np.flatnonzero(ic_mask)
    if not ic_all:
        n_ic = max(1, int(round(0.20 * ic_indices.size)))
        ic_indices = rng.choice(ic_indices, size=n_ic, replace=False)

    bc_parts: list[np.ndarray] = []
    for d in range(train_day_start, train_day_end + 1):
        day_bc = np.flatnonzero(bc_mask & (data.day == d))
        if day_bc.size:
            n_bc = max(1, int(round(float(bc_fraction) * day_bc.size)))
            bc_parts.append(rng.choice(day_bc, size=n_bc, replace=False))
    bc_indices = np.concatenate(bc_parts) if bc_parts else np.empty(0, dtype=np.int64)

    interior_indices = np.flatnonzero(interior_mask)
    n_data = max(1, int(round(float(data_fraction) * interior_indices.size)))
    data_indices = rng.choice(interior_indices, size=n_data, replace=False)

    return {
        "ic": np.sort(ic_indices),
        "bc": np.sort(bc_indices),
        "data": np.sort(data_indices),
        "train_all": np.flatnonzero(train_mask),
    }


def day_indices(data: TemperatureData, day: int) -> np.ndarray:
    return np.flatnonzero(data.day == day)


def values_to_masked_grid(
    data: TemperatureData,
    indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Place valid land samples on the regular grid and leave sea cells as NaN."""
    grid = np.full(data.valid_spatial_mask.shape, np.nan, dtype=np.float32)
    lat_index = np.searchsorted(data.unique_lats, data.lat[indices])
    lon_index = np.searchsorted(data.unique_lons, data.lon[indices])
    grid[lat_index, lon_index] = np.asarray(values, dtype=np.float32).reshape(-1)
    return grid
