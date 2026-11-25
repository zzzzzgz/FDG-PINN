"""Re-run the paper's FDG-PINN transport case with sampled weight diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PUBLISHED = HERE.parent
sys.path.insert(0, str(PUBLISHED))
sys.path.insert(0, str(HERE / "runner"))
import FDG_runner as runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = HERE / "transport_seed1111_every10"
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "weight_gradient_diagnostics.csv"
    if args.plot_only:
        with data_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        for row in rows:
            for field in ("iteration", "q", "weight", "loss", "weighted_loss", "weighted_gradient_l2"):
                row[field] = float(row[field])
    else:
        runner.ROOT_DIR = str(output)
        runner.SAVE_ARTIFACTS = False
        runner.SAVE_PREDICTION_DATA = False
        runner.APPEND_TEX = False
        runner.WRITE_LOG_XLSX = False
        runner.seed = 1111
        metrics = runner.run_combo("fdg_transport1d_eq", "fourier_double_gate", "aw")
        rows = metrics["weight_diagnostics"]
        if not rows:
            raise RuntimeError("No weight diagnostics recorded")
        with data_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paper_l2 = 0.0026339372465383993
        result = {
            "equation": "fdg_transport1d_eq",
            "seed": 1111,
            "method": "FDG-PINN",
            "global_l2": float(metrics["global_l2"]),
            "paper_seed_global_l2": paper_l2,
            "exact_match": float(metrics["global_l2"]) == paper_l2,
            "active_names": metrics["active_names"],
            "diagnostic_rows": len(rows),
            "diagnostic_interval_adam": 10,
            "diagnostic_interval_lbfgs_evaluations": 10,
            "lbfgs_iterations": int(metrics["lbfgs_iterations"]),
            "lbfgs_closure_calls": int(metrics["lbfgs_closure_calls"]),
            "training_time_s": float(metrics["training_time"]),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    palette = {"pde": "#2563a6", "ic": "#cc5a2a", "bc": "#16846c"}
    for term in result["active_names"]:
        color = palette.get(term, "#7856a5")
        adam = [row for row in rows if row["term"] == term and row["stage"] == "Adam"]
        lbfgs = [row for row in rows if row["term"] == term and row["stage"] == "L-BFGS evaluation"]
        end = next(row for row in rows if row["term"] == term and row["stage"] == "L-BFGS end")
        for ax, field in zip(axes, ("weight", "weighted_gradient_l2")):
            x = [row["iteration"] for row in adam]
            y = [row[field] for row in adam]
            ax.plot(x, y, color=color, linewidth=1.7, label=term.upper())
            lbfgs_x = [row["iteration"] for row in lbfgs] + [end["iteration"]]
            lbfgs_y = [row[field] for row in lbfgs] + [end[field]]
            ax.plot(lbfgs_x, lbfgs_y, color=color, linestyle="--", linewidth=1.5)
            ax.scatter([end["iteration"]], [end[field]], color=color, s=20, zorder=4)
    for ax in axes:
        ax.axvline(2000, color="0.45", linestyle=":", linewidth=1)
        ax.set_xlabel("Adam step / L-BFGS evaluation count")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"Adaptive weight $e^{-q_i}$")
    axes[1].set_ylabel(r"$\|\nabla_\theta(e^{-q_i}\mathcal{L}_i)\|_2$")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"weight_gradient_sample.{suffix}", dpi=240)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
