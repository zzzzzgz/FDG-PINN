"""One-axis comparison of single-and double-gate Global L2 errors."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
CASES = (
    ("fdg_transport1d_eq", "average_seed1111_5555", "Transport"),
    ("fdg_burgers_mixed1d_eq", "average_seed1111_5555", "Burgers"),
    ("fdg_helmholtz2d_mixedfreq_eq", "average_seed1111_5555", "Mixed-frequency\nHelmholtz"),
    ("fdg_reaction_diffusion_mixed1d_eq", "average_seed1111_5555", "Reaction-\ndiffusion"),
)
ALL_CASES = (
    CASES[0],
    CASES[1],
    ("fdg_helmholtz2d_k10_eq", "average_seed1111_5555_probe", "Helmholtz\n$k=10$"),
    ("fdg_helmholtz2d_k20_eq", "average_seed1111_5555", "Helmholtz\n$k=20$"),
    CASES[2],
    CASES[3],
)
SEEDS = (1111, 2222, 3333, 4444, 5555)


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def values(equation: str, folder: str):
    single = {
        int(row["seed"]): float(row["global_l2"])
        for row in read_csv(HERE / "raw_runs.csv")
        if row["equation"] == equation
    }
    double = {
        int(row["seed"]): float(row["global_l2"])
        for row in read_csv(BASELINES / equation / folder / "raw_runs.csv")
        if row["method"] == "FDG-PINN"
    }
    if set(single) != set(SEEDS) or set(double) != set(SEEDS):
        raise ValueError(f"Expected five matching seeds for {equation}")
    return np.array([single[s] for s in SEEDS]), np.array([double[s] for s in SEEDS])


def plot_comparison(cases, filename, width):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(width, 4.1), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.4, h_pad=0.4)
    x = np.arange(len(cases))
    colors = ("#c34842", "#2677a7")
    labels = ("Single gate", "Double gate")
    offsets = (-0.11, 0.11)

    for index, (equation, folder, _) in enumerate(cases):
        for series, data in enumerate(values(equation, folder)):
            mean = float(data.mean())
            std = float(data.std(ddof=1))
            # Global L2 is nonnegative; a lower SD endpoint below zero cannot
            # be shown on a logarithmic axis. Stop at the smallest observed run.
            lower = min(std, mean - float(data.min()))
            ax.errorbar(
                x[index] + offsets[series], mean,
                yerr=np.array([[lower], [std]]),
                fmt="o", color=colors[series], ecolor=colors[series],
                markersize=7, elinewidth=1.1, capsize=3, capthick=1.1,
                label=labels[series] if index == 0 else None,
                zorder=3,
            )

    ax.set_xticks(x, [case[2] for case in cases])
    ax.set_xlim(-0.45, len(cases) - 0.55)
    ax.set_yscale("log")
    ax.set_ylim(1.5e-4, 5e-2)
    ax.set_ylabel(r"$L_2$ error")
    ax.grid(False, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.3,
              columnspacing=1.4)

    for suffix in ("png", "svg"):
        fig.savefig(HERE / f"{filename}.{suffix}", dpi=220)
    plt.close(fig)


def main():
    plot_comparison(CASES, "gate_error_comparison_4eq", 8.4)
    plot_comparison(ALL_CASES, "gate_error_comparison_all_eq", 11.2)


if __name__ == "__main__":
    main()
