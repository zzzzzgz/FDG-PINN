from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

import numpy as np
import torch

import FDG_runner as runner


DEFAULT_EQUATIONS = [
    "fdg_transport1d_eq",
    "fdg_burgers_mixed1d_eq",
    "fdg_helmholtz2d_k1_eq",
    "fdg_helmholtz2d_k2_eq",
    "fdg_helmholtz2d_eq",
    "fdg_helmholtz2d_k10_eq",
    "fdg_helmholtz2d_k20_eq",
    "fdg_helmholtz2d_mixedfreq_eq",
    "fdg_reaction_diffusion_mixed1d_eq",
    "fdg_nls1d_eq",
]

METHODS = [
    ("PINN", "base", "fixed"),
    ("phi-NN", "improved_nn", "aw"),
    ("DG-PINN", "double_gate", "aw"),
    ("FF-PINN", "fourier_base", "aw"),
    ("FDG-PINN", "fourier_double_gate", "aw"),
]

METRIC_KEYS = [
    "training_time",
    "final_pde_loss",
    "final_ic_loss",
    "final_bc_loss",
    "train_l2",
    "global_l2",
    "discrete_l2",
]


def fmt_sci(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return rf"{mantissa:.3f}\times10^{{{exponent}}}"


def fmt_pm(mean: float, std: float, sci: bool = True) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nan"
    if sci:
        return rf"${fmt_sci(mean)}\pm{fmt_sci(std)}$"
    return rf"${mean:.2f}\pm{std:.2f}$"


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "equation",
        "seed",
        "method",
        "model_kind",
        "weight_mode",
        *METRIC_KEYS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_seed(seed: int) -> None:
    runner.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(eq_name: str, seed: int, method: str, model_kind: str, weight_mode: str) -> dict[str, object]:
    set_seed(seed)
    metrics = runner.run_combo(eq_name, model_kind, weight_mode)
    return {
        "equation": eq_name,
        "seed": seed,
        "method": method,
        "model_kind": model_kind,
        "weight_mode": weight_mode,
        "training_time": metrics["training_time"],
        "final_pde_loss": metrics["final_pde_loss"],
        "final_ic_loss": metrics["final_ic_loss"],
        "final_bc_loss": metrics["final_bc_loss"],
        "train_l2": metrics["train_l2"],
        "global_l2": metrics["global_l2"],
        "discrete_l2": metrics["time_avg_l2"],
    }


def summarize(eq_name: str, rows: list[dict[str, object]], out_dir: Path) -> None:
    summary_rows: list[dict[str, object]] = []
    for method, model_kind, weight_mode in METHODS:
        selected = [r for r in rows if r["equation"] == eq_name and r["method"] == method]
        if not selected:
            continue
        item: dict[str, object] = {
            "equation": eq_name,
            "method": method,
            "model_kind": model_kind,
            "weight_mode": weight_mode,
            "n": len(selected),
        }
        for key in METRIC_KEYS:
            vals = np.array([float(r[key]) for r in selected], dtype=np.float64)
            item[f"{key}_mean"] = float(np.mean(vals))
            item[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(item)

    csv_path = out_dir / "summary_mean_std.csv"
    fieldnames = ["equation", "method", "model_kind", "weight_mode", "n"]
    for key in METRIC_KEYS:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    tex_path = out_dir / "summary_mean_std.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write(
            "Method & Time(s) & Final PDE & Final IC & Final BC & "
            "$L_2$(training) & Global $L_2$ & Discrete $L_2$ \\\\\n"
        )
        f.write("\\midrule\n")
        for row in summary_rows:
            cells = [
                str(row["method"]),
                fmt_pm(float(row["training_time_mean"]), float(row["training_time_std"]), sci=False),
                fmt_pm(float(row["final_pde_loss_mean"]), float(row["final_pde_loss_std"])),
                fmt_pm(float(row["final_ic_loss_mean"]), float(row["final_ic_loss_std"])),
                fmt_pm(float(row["final_bc_loss_mean"]), float(row["final_bc_loss_std"])),
                fmt_pm(float(row["train_l2_mean"]), float(row["train_l2_std"])),
                fmt_pm(float(row["global_l2_mean"]), float(row["global_l2_std"])),
                fmt_pm(float(row["discrete_l2_mean"]), float(row["discrete_l2_std"])),
            ]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation", action="append", choices=DEFAULT_EQUATIONS)
    parser.add_argument("--seeds", default="2022,2023,2024,2025,2026")
    parser.add_argument("--out-tag", default="average")
    parser.add_argument("--method", action="append", choices=[m[0] for m in METHODS])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # This statistics run only needs final table metrics, not field plots/GIFs.
    runner.SAVE_ARTIFACTS = False
    runner.SAVE_PREDICTION_DATA = False
    runner.APPEND_TEX = False
    runner.WRITE_LOG_XLSX = False

    equations = args.equation or DEFAULT_EQUATIONS
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [m for m in METHODS if not args.method or m[0] in set(args.method)]
    root = Path(__file__).resolve().parent

    for eq_name in equations:
        importlib.import_module(eq_name)
        out_dir = root / eq_name / args.out_tag
        raw_path = out_dir / "raw_runs.csv"
        rows: list[dict[str, object]] = read_existing(raw_path) if args.resume else []
        done = {(str(r["equation"]), int(r["seed"]), str(r["method"])) for r in rows}

        for seed in seeds:
            for method, model_kind, weight_mode in methods:
                key = (eq_name, seed, method)
                if key in done:
                    print(f"Skip existing: {eq_name} seed={seed} method={method}", flush=True)
                    continue
                print(f"Run: {eq_name} seed={seed} method={method}", flush=True)
                row = run_one(eq_name, seed, method, model_kind, weight_mode)
                rows.append(row)
                write_rows(raw_path, rows)
                summarize(eq_name, rows, out_dir)

        summarize(eq_name, rows, out_dir)
        print(f"Saved average results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
