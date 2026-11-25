"""Run paper-configured baselines on the unforced Burgers front benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PUBLISHED = HERE
OUTPUT = HERE / "fdg_burgers_front1d_eq" / "probe_seed1111"
METHODS = {
    "PINN": ("base", "fixed"),
    "FF-PINN": ("fourier_base", "aw"),
    "FDG-PINN": ("fourier_double_gate", "aw"),
}


def run_child(method: str, seed: int, equation: str) -> None:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(PUBLISHED))
    import FDG_runner as runner

    runner.ROOT_DIR = str(OUTPUT / method.lower().replace("-", "_"))
    runner.SAVE_ARTIFACTS = True
    runner.SAVE_PREDICTION_DATA = True
    runner.APPEND_TEX = False
    runner.WRITE_LOG_XLSX = False
    runner.seed = seed
    kind, weighting = METHODS[method]
    metrics = runner.run_combo(equation, kind, weighting)
    record = {
        "equation": equation,
        "seed": seed,
        "method": method,
        "model_kind": kind,
        "weight_mode": weighting,
        "adam_max_steps": runner.steps_adam,
        "lbfgs_max_steps": runner.steps_lbfgs,
        "training_time_s": metrics["training_time"],
        "final_pde_loss": metrics["final_pde_loss"],
        "final_ic_loss": metrics["final_ic_loss"],
        "final_bc_loss": metrics["final_bc_loss"],
        "training_l2": metrics["train_l2"],
        "global_l2": metrics["global_l2"],
        "discrete_l2": metrics["time_avg_l2"],
    }
    result = OUTPUT / f"{method.lower().replace('-', '_')}_seed{seed}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps(record), flush=True)


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--equation", default="fdg_burgers_front1d_eq",
                        choices=("fdg_burgers_front1d_eq", "fdg_burgers_front1d_focused_eq"))
    parser.add_argument("--method", action="append", choices=tuple(METHODS))
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    OUTPUT = HERE / args.equation / f"probe_seed{args.seed}"
    methods = tuple(args.method or METHODS)
    if args.child:
        run_child(methods[0], args.seed, args.equation)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pinn_result = OUTPUT / f"pinn_seed{args.seed}.json"
    pinn_log = OUTPUT / f"pinn_seed{args.seed}.log"
    if (args.equation == "fdg_burgers_front1d_eq" and args.seed == 1111
            and not pinn_result.exists() and pinn_log.exists()):
        completed = pinn_log.read_text(encoding="utf-8")
        fields = {
            "training_time_s": r"Training time \(s\): ([0-9.eE+-]+)",
            "final_pde_loss": r"\[Final\].*?pde=([0-9.eE+-]+)",
            "final_ic_loss": r"\[Final\].*?ic=([0-9.eE+-]+)",
            "final_bc_loss": r"\[Final\].*?bc=([0-9.eE+-]+)",
            "training_l2": r"Training-sample supervised relative L2-error = ([0-9.eE+-]+)",
            "global_l2": r"Global spatio-temporal relative L2-error = ([0-9.eE+-]+)",
            "discrete_l2": r"Time-averaged relative discrete L2-error = ([0-9.eE+-]+)",
        }
        matches = {key: re.search(pattern, completed) for key, pattern in fields.items()}
        if all(matches.values()) and "Field data saved to:" in completed:
            record = {"equation": "fdg_burgers_front1d_eq", "seed": args.seed,
                      "method": "PINN", "model_kind": "base", "weight_mode": "fixed",
                      "adam_max_steps": 2000, "lbfgs_max_steps": 1000}
            record.update({key: float(match.group(1)) for key, match in matches.items()})
            pinn_result.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"Recovered completed PINN run: {pinn_result}", flush=True)
    for method in methods:
        name = method.lower().replace("-", "_")
        result = OUTPUT / f"{name}_seed{args.seed}.json"
        if result.exists():
            print(f"Skip {method}: {result}", flush=True)
            continue
        log = OUTPUT / f"{name}_seed{args.seed}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child",
                   "--seed", str(args.seed), "--equation", args.equation,
                   "--method", method]
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           cwd=HERE, env=os.environ.copy(), check=True)
        print(f"Saved {method}: {result}", flush=True)
    rows = [json.loads((OUTPUT / f"{method.lower().replace('-', '_')}_seed{args.seed}.json")
                       .read_text(encoding="utf-8")) for method in methods]
    with (OUTPUT / f"comparison_seed{args.seed}.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
