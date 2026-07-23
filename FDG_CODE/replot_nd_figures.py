from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "FDG_CODE" / "Figures"
INDIA_CSV = ROOT / "Temperature_forecast_india" / "temperature_model_comparison_results" / "daily_reconstruction_comparison.csv"
HUABEI_CSV = ROOT / "Temperature_forecast_huabei_50day" / "temperature_model_comparison_results_landmask" / "daily_reconstruction_comparison.csv"
HUABEI_DIR = ROOT / "Temperature_forecast_huabei_50day"
HUABEI_DATA = HUABEI_DIR / "FDG_PINN_HUABEI_50DAY_Temperature_preprocessed.csv"

FONT = {
    "family": "sans-serif",
    "sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "size": 10,
}

plt.rcParams.update(
    {
        "font.family": FONT["family"],
        "font.sans-serif": FONT["sans-serif"],
        "font.size": FONT["size"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
    }
)


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def style_day_axis(ax: plt.Axes, days: np.ndarray) -> None:
    ax.set_xlim(float(days.min()), float(days.max()))
    ax.set_xticks(days.astype(int))
    ax.tick_params(length=2.5, width=0.6, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_combined_curves(rows: list[dict[str, float]], stem: str, region: str) -> None:
    days = np.asarray([row["day"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), constrained_layout=True)

    axes[0].plot(days, [row["true_mean_C"] for row in rows], color="#202124", marker="o", markersize=2.6, label="True")
    axes[0].plot(days, [row["pinn_mean_C"] for row in rows], color="#d55e00", marker="s", markersize=2.4, label="Standard PINN")
    axes[0].plot(days, [row["fdg_mean_C"] for row in rows], color="#0072b2", marker="^", markersize=2.4, label="FDG-PINN")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Spatial mean temperature ($^\\circ$C)")
    axes[0].text(0.01, 0.98, "(a)", transform=axes[0].transAxes, va="top", ha="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="best")
    style_day_axis(axes[0], days)

    axes[1].plot(days, [row["pinn_relative_l2"] for row in rows], color="#d55e00", marker="s", markersize=2.4, label="Standard PINN")
    axes[1].plot(days, [row["fdg_relative_l2"] for row in rows], color="#0072b2", marker="^", markersize=2.4, label="FDG-PINN")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Relative discrete $L_2$ error")
    axes[1].set_ylim(bottom=0.0)
    axes[1].text(0.01, 0.98, "(b)", transform=axes[1].transAxes, va="top", ha="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="best")
    style_day_axis(axes[1], days)

    for ext in (".svg", ".eps", ".png"):
        fig.savefig(FIG_DIR / f"{stem}{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_huabei_boundary() -> None:
    sys.path.insert(0, str(HUABEI_DIR))
    import temperature_fdg_eq as eq

    data = eq.load_temperature_data(HUABEI_DATA)
    valid = data.valid_spatial_mask
    boundary = data.boundary_spatial_mask
    boundary_ij = np.argwhere(boundary)
    boundary_lat = data.unique_lats[boundary_ij[:, 0]]
    boundary_lon = data.unique_lons[boundary_ij[:, 1]]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    axes[0].imshow(
        valid,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        interpolation="nearest",
        aspect="equal",
        cmap=ListedColormap(["white", "#3f4954"]),
        vmin=0,
        vmax=1,
    )
    axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes, va="top", ha="left", fontweight="bold")
    axes[0].set_xlabel("normalized longitude")
    axes[0].set_ylabel("normalized latitude")

    axes[1].imshow(
        valid,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        interpolation="nearest",
        aspect="equal",
        cmap=ListedColormap(["white", "#d9dde2"]),
        vmin=0,
        vmax=1,
    )
    axes[1].scatter(boundary_lon, boundary_lat, s=1.8, c="#c62828", marker="s", linewidths=0)
    axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes, va="top", ha="left", fontweight="bold")
    axes[1].set_xlabel("normalized longitude")
    axes[1].set_ylabel("normalized latitude")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(length=2.5, width=0.6, pad=1.5)
        ax.set_box_aspect(1)

    for ext in (".svg", ".eps", ".png"):
        fig.savefig(FIG_DIR / f"huabei_land_boundary_diagnostic_nd{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_combined_curves(read_rows(INDIA_CSV), "india_mean_l2_combined_nd_v2", "India")
    plot_combined_curves(read_rows(HUABEI_CSV), "huabei_mean_l2_combined_nd_v2", "North China")
    plot_huabei_boundary()


if __name__ == "__main__":
    main()
