# Testing FF-PINN in ablation experiments
from __future__ import annotations
import csv
import importlib
from pathlib import Path

import FDG_runner as runner


EQUATIONS = [
    "fdg_transport1d_eq",
    "fdg_helmholtz2d_eq",
    "fdg_helmholtz2d_k20_eq",
    "fdg_helmholtz2d_mixedfreq_eq",
    "fdg_reaction_diffusion_mixed1d_eq",
    "fdg_burgers_mixed1d_eq",
    "fdg_nls1d_eq",
]


def fmt(value: object) -> str:
    try:
        return f"{float(value):.12e}"
    except Exception:
        return str(value)


def main() -> None:
    out_path = Path(__file__).resolve().parent / "ff_ablation_results.csv"
    rows: list[dict[str, object]] = []
    for eq_name in EQUATIONS:
        metrics = runner.run_combo(eq_name, "fourier_base", "aw")
        eq = importlib.import_module(eq_name)
        row = {
            "equation": eq_name,
            "display_name": eq.NAME,
            "method": "FF-PINN + AW",
            "training_time": metrics["training_time"],
            "final_pde_loss": metrics["final_pde_loss"],
            "final_ic_loss": metrics["final_ic_loss"],
            "final_bc_loss": metrics["final_bc_loss"],
            "train_l2": metrics["train_l2"],
            "train_mse": metrics["train_mse"],
            "eval_mse": metrics["mse"],
            "global_l2": metrics["global_l2"],
            "discrete_l2": metrics["time_avg_l2"],
        }
        rows.append(row)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            for item in rows:
                writer.writerow({k: fmt(v) for k, v in item.items()})
    print(f"Saved FF-PINN ablation results to: {out_path}")


if __name__ == "__main__":
    main()
