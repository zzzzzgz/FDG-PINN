"""Compare capacity-matched Ref. [23], single-gate, and FDG-PINN errors."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
SEEDS = (1111, 2222, 3333, 4444, 5555)
FOUR_CASES = (
    ("fdg_transport1d_eq", "average_seed1111_5555", "Transport"),
    ("fdg_burgers_mixed1d_eq", "average_seed1111_5555", "Burgers"),
    ("fdg_helmholtz2d_mixedfreq_eq", "average_seed1111_5555", "Mixed-frequency\nHelmholtz"),
    ("fdg_reaction_diffusion_mixed1d_eq", "average_seed1111_5555", "Reaction-\ndiffusion"),
)
ALL_CASES = (
    FOUR_CASES[0],
    FOUR_CASES[1],
    ("fdg_helmholtz2d_k10_eq", "average_seed1111_5555_probe", "Helm.\n$k=10$"),
    ("fdg_helmholtz2d_k20_eq", "average_seed1111_5555", "Helm.\n$k=20$"),
    ("fdg_helmholtz2d_mixedfreq_eq", "average_seed1111_5555", "Mixed\nHelm."),
    ("fdg_reaction_diffusion_mixed1d_eq", "average_seed1111_5555", "React.-\ndiff."),
)
SERIES = (
    ("ref23_matched_params", "Ref. [23] modified MLP", "#c34842", -0.16),
    ("first_gate_matched_params", "Single gate", "#39854b", 0.0),
    ("fdg", "FDG-PINN", "#2677a7", 0.16),
)


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def values(equation: str, folder: str):
    new_rows = read_csv(HERE / "raw_runs.csv")
    base_rows = read_csv(BASELINES / equation / folder / "raw_runs.csv")
    result = {}
    for variant, _, _, _ in SERIES:
        if variant == "fdg":
            rows = (row for row in base_rows if row["method"] == "FDG-PINN")
        else:
            rows = (row for row in new_rows if row["equation"] == equation and row["variant"] == variant)
        by_seed = {int(row["seed"]): float(row["global_l2"]) for row in rows}
        if set(by_seed) != set(SEEDS):
            raise ValueError(f"Expected five matching seeds for {equation}, {variant}")
        result[variant] = np.array([by_seed[seed] for seed in SEEDS])
    return result


def plot_comparison(cases, filename: str, width: float, compact: bool = False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8 if compact else 10,
        "axes.labelsize": 8 if compact else 10,
        "xtick.labelsize": 6.5 if compact else 9,
        "ytick.labelsize": 7 if compact else 9,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(width, 4.4 if compact else 4.1), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.15 if compact else 0.4, h_pad=0.2 if compact else 0.4)
    x = np.arange(len(cases))

    for index, (equation, folder, _) in enumerate(cases):
        data = values(equation, folder)
        for variant, label, color, offset in SERIES:
            runs = data[variant]
            mean = float(runs.mean())
            std = float(runs.std(ddof=1))
            lower = min(std, mean - float(runs.min()))
            ax.errorbar(
                x[index] + offset, mean,
                yerr=np.array([[lower], [std]]),
                fmt="o", color=color, ecolor=color,
                markersize=4.5 if compact else 7,
                elinewidth=0.8 if compact else 1.1,
                capsize=2 if compact else 3,
                capthick=0.8 if compact else 1.1,
                label=label if index == 0 else None,
                zorder=3,
            )

    ax.set_xticks(x, [case[2] for case in cases])
    ax.set_xlim(-0.5, len(cases) - 0.5)
    ax.set_yscale("log")
    ax.set_ylim(1.5e-4, 1.5e-1)
    ax.set_ylabel(r"$L_2$ error")
    ax.grid(False, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=1 if compact else 3,
              fontsize=7 if compact else 10, handletextpad=0.3,
              labelspacing=0.4, columnspacing=1.2)

    for suffix in ("png", "svg"):
        fig.savefig(HERE / f"{filename}.{suffix}", dpi=220)
    plt.close(fig)


def main():
    plot_comparison(FOUR_CASES, "capacity_matched_error_comparison_4eq", 8.4)
    plot_comparison(ALL_CASES, "capacity_matched_error_comparison_all_eq", 3.6, compact=True)


if __name__ == "__main__":
    main()
