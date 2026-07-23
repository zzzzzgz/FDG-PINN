"""Visualize the ERA5-Land mask and four-neighbor boundary used for sampling."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import temperature_fdg_eq as eq


# Replace these paths when diagnosing another preprocessed temperature dataset.
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "FDG_PINN_HUABEI_50DAY_Temperature_preprocessed.csv"
OUTPUT_DIR = BASE_DIR / "land_boundary_diagnostic"

AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["ps.fonttype"] = 42


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = eq.load_temperature_data(DATA_CSV)
    valid = data.valid_spatial_mask
    boundary = data.boundary_spatial_mask
    boundary_ij = np.argwhere(boundary)
    boundary_lat = data.unique_lats[boundary_ij[:, 0]]
    boundary_lon = data.unique_lons[boundary_ij[:, 1]]

    valid_fig, valid_ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    valid_ax.imshow(
        valid,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        interpolation="nearest",
        aspect="equal",
        cmap=ListedColormap(["white", "#3f4954"]),
        vmin=0,
        vmax=1,
    )
    valid_ax.set_xlabel("normalized longitude", fontsize=AXIS_LABEL_FONTSIZE)
    valid_ax.set_ylabel("normalized latitude", fontsize=AXIS_LABEL_FONTSIZE)
    valid_ax.set_box_aspect(1)
    valid_ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    boundary_fig, boundary_ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    boundary_ax.imshow(
        valid,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        interpolation="nearest",
        aspect="equal",
        cmap=ListedColormap(["white", "#d9dde2"]),
        vmin=0,
        vmax=1,
    )
    boundary_ax.scatter(boundary_lon, boundary_lat, s=5.0, c="#c62828", marker="s", linewidths=0)
    boundary_ax.set_xlabel("normalized longitude", fontsize=AXIS_LABEL_FONTSIZE)
    boundary_ax.set_ylabel("normalized latitude", fontsize=AXIS_LABEL_FONTSIZE)
    boundary_ax.set_box_aspect(1)
    boundary_ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    for fig, name in (
        (valid_fig, "huabei_valid_land_cells"),
        (boundary_fig, "huabei_four_neighbor_land_boundary"),
    ):
        stem = OUTPUT_DIR / name
        fig.savefig(stem.with_suffix(".png"), dpi=220)
        fig.savefig(stem.with_suffix(".svg"), dpi=300)
        fig.savefig(stem.with_suffix(".eps"), dpi=300)
        plt.close(fig)

    with (OUTPUT_DIR / "boundary_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lat_normalized", "lon_normalized"])
        writer.writerows(zip(boundary_lat, boundary_lon))

    with (OUTPUT_DIR / "diagnostic_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"grid_shape={valid.shape}\n")
        handle.write(f"valid_spatial_count={int(valid.sum())}\n")
        handle.write(f"sea_spatial_count={int((~valid).sum())}\n")
        handle.write(f"boundary_spatial_count={int(boundary.sum())}\n")
        handle.write(f"interior_spatial_count={int((valid & ~boundary).sum())}\n")
        handle.write("boundary_mode=four_neighbor_land_mask\n")

    print(f"Saved boundary diagnostic to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
