"""Run the published India FDG temperature model at a rolling forecast origin."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
INDIA = HERE.parent / "Temperature_forecast_india"
OUTPUT = HERE / "review_temperature_rolling_origins"
sys.path.insert(0, str(INDIA))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=int, required=True, choices=range(2, 21))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    import temperature_fdg_eq as eq
    import FDG_runner_temperature_forecast as runner

    eq.TRAIN_DAY_END = args.origin
    eq.FORECAST_DAY_START = args.origin + 1
    eq.FORECAST_DAY_END = args.origin + 5
    eq.make_inputs = partial(eq.make_inputs, train_day_start=1,
                             train_day_end=args.origin)
    eq.select_training_indices = partial(eq.select_training_indices,
                                         train_day_start=1,
                                         train_day_end=args.origin)
    runner.SEED = args.seed
    runner.FORECAST_DAY_START = args.origin + 1
    runner.FORECAST_DAY_END = args.origin + 5
    runner.FORECAST_SNAPSHOT_DAYS = ()
    run_root = OUTPUT / f"origin_{args.origin:02d}" / f"seed_{args.seed}"
    if (run_root / "rolling_result.json").exists():
        raise FileExistsError(f"Preserving completed result in {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    runner.FORECAST_RESULTS_DIR = run_root
    runner.main()

    completed = sorted(run_root.glob("forecast_fdg_aw_*/forecast_summary.txt"),
                       key=lambda path: path.stat().st_mtime)
    summary_path = completed[-1]
    metrics_path = summary_path.parent / "forecast_metrics_by_day.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        daily = list(csv.DictReader(stream))
    if len(daily) != 5:
        raise RuntimeError(f"Expected five forecast days, found {len(daily)}")
    record = {"origin": args.origin, "seed": args.seed,
              "run_dir": str(summary_path.parent), "daily": daily}
    result = run_root / "rolling_result.json"
    result.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"RESULT_FILE={result}", flush=True)


if __name__ == "__main__":
    main()
