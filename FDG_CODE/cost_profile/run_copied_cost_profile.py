"""Profile Burgers using a minimally instrumented copy of the paper runner."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from run_cost_profile import METHODS, PILOT, PUBLISHED, read_paper_baseline


HERE = Path(__file__).resolve().parent
COPIED = HERE / "copied_runner"
OUTPUT = Path(os.environ.get("PINN_COST_OUTPUT", str(HERE / "copied_runner_runs")))
PAPER_METHODS = tuple(METHODS)[:5]


def result_path(equation: str, method: str, seed: int) -> Path:
    slug = method.lower().replace(" ", "_").replace("-", "_")
    return OUTPUT / equation / f"{slug}_seed{seed}.json"


def child(equation: str, method: str, seed: int) -> None:
    sys.path.insert(0, str(PUBLISHED))
    sys.path.insert(0, str(COPIED))
    import FDG_runner as runner

    runner.ROOT_DIR = str(OUTPUT)
    runner.SAVE_ARTIFACTS = False
    runner.SAVE_PREDICTION_DATA = False
    runner.APPEND_TEX = False
    runner.WRITE_LOG_XLSX = False
    runner.seed = seed
    kind, weighting = METHODS[method]
    metrics = runner.run_combo(equation, kind, weighting)
    output = result_path(equation, method, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output.with_name(output.stem + "_trace.csv")
    with trace_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("elapsed_s", "global_l2", "stage", "lbfgs_closure_calls"))
        writer.writerows(metrics["profile_trace"])
    import importlib
    eq = importlib.import_module(equation)
    model = runner.build_model(eq, kind)
    network_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record = {
        "equation": equation,
        "method": method,
        "seed": seed,
        "model_kind": kind,
        "weight_mode": weighting,
        "network_parameters": network_parameters,
        "adaptive_weight_parameters": len(metrics["active_names"]) if weighting == "aw" else 0,
        "training_time_s": float(metrics["training_time"]),
        "global_l2": float(metrics["global_l2"]),
        "paper_seed_global_l2": read_paper_baseline(equation, method, seed),
        "peak_allocated_mib": float(metrics["peak_allocated_mib"]),
        "peak_reserved_mib": float(metrics["peak_reserved_mib"]),
        "lbfgs_closure_calls": int(metrics["lbfgs_closure_calls"]),
        "lbfgs_optimizer_func_evals": int(metrics["lbfgs_optimizer_func_evals"]),
        "lbfgs_iterations": int(metrics["lbfgs_iterations"]),
        "trace_points": len(metrics["profile_trace"]),
        "trace_csv": trace_path.name,
    }
    with output.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
    print("COPIED_RESULT " + json.dumps(record, ensure_ascii=False), flush=True)


def write_summary(equation: str, seed: int, methods: tuple[str, ...]) -> None:
    rows = []
    for method in methods:
        path = result_path(equation, method, seed)
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                rows.append(json.load(stream))
    if not rows:
        return
    path = OUTPUT / equation / f"profile_seed{seed}.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation", default="fdg_burgers_mixed1d_eq")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--method", action="append", choices=PAPER_METHODS)
    parser.add_argument("--all-paper-methods", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.equation, args.method[0], args.seed)
        return
    methods = PAPER_METHODS if args.all_paper_methods else tuple(args.method or PILOT)
    for method in methods:
        output = result_path(args.equation, method, args.seed)
        if output.exists():
            print(f"Skip completed {method} seed={args.seed}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child",
                   "--equation", args.equation, "--seed", str(args.seed),
                   "--method", method]
        with output.with_suffix(".log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           cwd=HERE, env=os.environ.copy(), check=True)
        write_summary(args.equation, args.seed, methods)
        print(f"Saved {method} seed={args.seed} -> {output}", flush=True)
    write_summary(args.equation, args.seed, methods)


if __name__ == "__main__":
    main()
