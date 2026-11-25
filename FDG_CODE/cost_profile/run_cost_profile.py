"""Re-run paper PDE methods with non-invasive time, memory, and L-BFGS probes.

The original runner supplies the equation, model, sampling, objective, and
optimizer settings. Each profiled run has its own process so CUDA cache from
another model cannot contaminate its PyTorch peak-memory measurement.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PUBLISHED = HERE.parent
CAPACITY = HERE.parent / "ref23_capacity_ablation"
METHODS = {
    "PINN": ("base", "fixed"),
    "phi-NN": ("improved_nn", "aw"),
    "DG-PINN": ("double_gate", "aw"),
    "FF-PINN": ("fourier_base", "aw"),
    "FDG-PINN": ("fourier_double_gate", "aw"),
    "Single gate": ("first_gate_matched_params", "aw"),
    "Modified MLP": ("ref23_matched_params", "aw"),
}
PILOT = ("PINN", "FF-PINN", "FDG-PINN")


def result_path(equation: str, method: str, seed: int) -> Path:
    slug = method.lower().replace(" ", "_").replace("-", "_")
    return HERE / equation / f"{slug}_seed{seed}.json"


def read_paper_baseline(equation: str, method: str, seed: int) -> float | None:
    folder = "average_seed1111_5555_probe" if equation == "fdg_helmholtz2d_k10_eq" else "average_seed1111_5555"
    path = PUBLISHED / equation / folder / "raw_runs.csv"
    if not path.exists() or method not in PILOT + ("phi-NN", "DG-PINN"):
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] == method and int(row["seed"]) == seed:
                return float(row["global_l2"])
    return None


def run_child(equation: str, method: str, seed: int) -> None:
    sys.path.insert(0, str(PUBLISHED))
    sys.path.insert(0, str(CAPACITY))
    import run_ablation as capacity

    runner = capacity.runner
    torch = runner.torch
    runner.ROOT_DIR = str(HERE)
    runner.SAVE_ARTIFACTS = False
    runner.SAVE_PREDICTION_DATA = False
    runner.APPEND_TEX = False
    runner.WRITE_LOG_XLSX = False
    runner.seed = seed

    state = SimpleNamespace(start=None, end=None, peak_allocated=0,
                            peak_reserved=0, lbfgs=None, trace=[])
    original_time = time.time
    original_evaluate = runner.evaluate_error_metrics
    original_lbfgs = torch.optim.LBFGS

    def clock_time():
        stamp = original_time()
        if state.start is None:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            state.start = stamp
        elif state.end is None:
            state.end = stamp
            if torch.cuda.is_available():
                state.peak_allocated = torch.cuda.max_memory_allocated()
                state.peak_reserved = torch.cuda.max_memory_reserved()
        return stamp

    def evaluate_with_timestamp(eq, model):
        metrics = original_evaluate(eq, model)
        if state.start is not None and state.end is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            state.trace.append({
                "elapsed_s": original_time() - state.start,
                "global_l2": float(metrics["global_l2"]),
                "stage": "Adam" if state.lbfgs is None else "L-BFGS evaluation",
                "lbfgs_closure_calls": 0 if state.lbfgs is None else state.lbfgs.calls,
            })
        return metrics

    class TrackedLBFGS(original_lbfgs):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0
            self.actual_iterations = 0
            self.optimizer_func_evals = 0
            state.lbfgs = self

        def step(self, closure):
            def counted_closure():
                self.calls += 1
                return closure()

            try:
                return super().step(counted_closure)
            finally:
                info = self.state.get(self._params[0], {})
                self.actual_iterations = int(info.get("n_iter", 0))
                self.optimizer_func_evals = int(info.get("func_evals", 0))

    runner.time = SimpleNamespace(time=clock_time)
    runner.evaluate_error_metrics = evaluate_with_timestamp
    torch.optim.LBFGS = TrackedLBFGS

    kind, weighting = METHODS[method]
    eq = importlib.import_module(equation)
    model = runner.build_model(eq, kind)
    network_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    metrics = runner.run_combo(equation, kind, weighting)
    if state.start is None or state.end is None:
        raise RuntimeError("Training timer was not observed")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    final = {
        "elapsed_s": float(metrics["training_time"]),
        "global_l2": float(metrics["global_l2"]),
        "stage": "Final selected model",
        "lbfgs_closure_calls": 0 if state.lbfgs is None else state.lbfgs.calls,
    }
    state.trace.append(final)

    output = result_path(equation, method, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output.with_name(output.stem + "_trace.csv")
    with trace_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(final))
        writer.writeheader()
        writer.writerows(state.trace)

    baseline = read_paper_baseline(equation, method, seed)
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
        "paper_seed_global_l2": baseline,
        "peak_allocated_mib": state.peak_allocated / 2**20,
        "peak_reserved_mib": state.peak_reserved / 2**20,
        "lbfgs_closure_calls": 0 if state.lbfgs is None else state.lbfgs.calls,
        "lbfgs_optimizer_func_evals": 0 if state.lbfgs is None else state.lbfgs.optimizer_func_evals,
        "lbfgs_iterations": 0 if state.lbfgs is None else state.lbfgs.actual_iterations,
        "trace_points": len(state.trace),
        "trace_csv": trace_path.name,
    }
    with output.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
    print("PROFILE_RESULT " + json.dumps(record, ensure_ascii=False), flush=True)


def write_summary(equation: str, seed: int, methods: tuple[str, ...]) -> None:
    rows = []
    for method in methods:
        path = result_path(equation, method, seed)
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                rows.append(json.load(stream))
    if not rows:
        return
    path = HERE / equation / f"profile_seed{seed}.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation", default="fdg_burgers_mixed1d_eq")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--method", action="append", choices=tuple(METHODS))
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        run_child(args.equation, args.method[0], args.seed)
        return
    methods = tuple(METHODS) if args.all_methods else tuple(args.method or PILOT)
    for method in methods:
        output = result_path(args.equation, method, args.seed)
        if output.exists():
            print(f"Skip completed {method} seed={args.seed}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".log")
        command = [sys.executable, str(Path(__file__).resolve()), "--child",
                   "--equation", args.equation, "--seed", str(args.seed),
                   "--method", method]
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           cwd=HERE, env=os.environ.copy(), check=True)
        write_summary(args.equation, args.seed, methods)
        print(f"Saved {method} seed={args.seed} -> {output}", flush=True)
    write_summary(args.equation, args.seed, methods)


if __name__ == "__main__":
    main()
