"""Plot the observed day-5 India temperature field in the Fig. 9 style."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INDIA = HERE.parent / "Temperature_forecast_india"
OUTPUT = HERE / "Figures_ND"
sys.path.insert(0, str(INDIA))
import temperature_fdg_eq as eq


def main():
    data = eq.load_temperature_data()
    indices = eq.day_indices(data, 5)
    lats, lons = data.unique_lats, data.unique_lons
    grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    lat_index = {round(float(value), 10): i for i, value in enumerate(lats)}
    lon_index = {round(float(value), 10): i for i, value in enumerate(lons)}
    for index in indices:
        i = lat_index[round(float(data.lat[index]), 10)]
        j = lon_index[round(float(data.lon[index]), 10)]
        grid[i, j] = data.temp_c[index]
    if np.isnan(grid).any():
        raise RuntimeError("Day-5 grid is incomplete; do not silently interpolate observations")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(4.5, 3.7), constrained_layout=True)
    image = ax.imshow(grid, extent=(0, 1, 0, 1), origin="lower", aspect="equal",
                      cmap="viridis", vmin=-5, vmax=40, interpolation="nearest")
    ax.set(title="True", xlabel="normalized longitude", ylabel="normalized latitude")
    ax.set_box_aspect(1)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(-5, 41, 5), fraction=0.046, pad=0.04)
    colorbar.set_label(r"Temperature ($^\circ$C)")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = OUTPUT / "india_true_temperature_day05_for_block"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"day=5 points={len(indices)} grid={grid.shape} min={grid.min():.3f} max={grid.max():.3f}")
    print(base.with_suffix(".svg"))
    print(base.with_suffix(".png"))


if __name__ == "__main__":
    main()
