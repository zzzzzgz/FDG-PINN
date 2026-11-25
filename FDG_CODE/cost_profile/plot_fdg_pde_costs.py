"""Summarize and plot one FDG-PINN cost-profile run per paper PDE."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "copied_runner_runs"
EQUATIONS = (
    ("fdg_transport1d_eq", "Transport", "#1b6ca8"),
    ("fdg_burgers_mixed1d_eq", "Burgers", "#d45c28"),
    ("fdg_helmholtz2d_k10_eq", r"Helmholtz $k=10$", "#7855a4"),
    ("fdg_helmholtz2d_k20_eq", r"Helmholtz $k=20$", "#9a7720"),
    ("fdg_helmholtz2d_mixedfreq_eq", "Mixed Helmholtz", "#168f79"),
    ("fdg_reaction_diffusion_mixed1d_eq", "Reaction--diffusion", "#bd4764"),
)


def read_run(root: Path, equation: str, seed: int) -> tuple[dict, list[dict]]:
    record_path = root / equation / f"fdg_pinn_seed{seed}.json"
    with record_path.open(encoding="utf-8") as stream:
        record = json.load(stream)
    with (record_path.parent / record["trace_csv"]).open(
        encoding="utf-8", newline=""
    ) as stream:
        trace = list(csv.DictReader(stream))
    if not trace or trace[-1]["stage"] != "Final selected model":
        raise ValueError(f"Incomplete profile trace: {record_path}")
    return record, trace


def generate(root: Path, seed: int, out_dir: Path) -> None:
    runs = [(equation, label, color, *read_run(root, equation, seed))
            for equation, label, color in EQUATIONS]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"fdg_pde_cost_summary_seed{seed}.csv"
    fields = (
        "equation", "method", "seed", "network_parameters",
        "adaptive_weight_parameters", "training_time_s", "global_l2",
        "paper_seed_global_l2", "peak_allocated_mib", "peak_reserved_mib",
        "lbfgs_iterations", "lbfgs_closure_calls", "lbfgs_optimizer_func_evals",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in fields}
                         for _, _, _, record, _ in runs)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for _, label, color, record, trace in runs:
        for stage, style in (("Adam", "-"), ("L-BFGS trial", "--")):
            points = [row for row in trace if row["stage"] == stage]
            x = np.array([float(row["elapsed_s"]) for row in points])
            y = np.array([float(row["global_l2"]) for row in points])
            if len(x) and (not np.all(np.diff(x) >= 0) or np.any(y <= 0)):
                raise ValueError(f"Invalid time/error trace for {label}: {stage}")
            ax.plot(x, y, style, color=color, linewidth=1.6,
                    alpha=0.95 if stage == "Adam" else 0.8)
        final = trace[-1]
        ax.scatter(float(final["elapsed_s"]), float(final["global_l2"]),
                   s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5)
        if abs(float(final["global_l2"]) - float(record["global_l2"])) > 1e-12:
            raise ValueError(f"Final trace error disagrees with result for {label}")

    ax.set(xlabel="Elapsed training time (s)", ylabel=r"Global $L_2$ error")
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], color=color, lw=2, label=label)
               for _, label, color in EQUATIONS]
    handles.extend((
        Line2D([0], [0], color="black", lw=1.6, ls="-", label="Adam"),
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="L-BFGS evaluations"),
    ))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False, fontsize=9, columnspacing=1.5)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.11, right=0.98)
    stem = out_dir / f"fdg_pde_error_vs_time_seed{seed}"
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    print(summary_path)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".svg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    generate(args.root, args.seed, args.output)


if __name__ == "__main__":
    main()
