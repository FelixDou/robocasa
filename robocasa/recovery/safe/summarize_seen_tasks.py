"""Aggregate the three seeds of the all-five-seen SAFE experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def summarize(root, expected_seeds=(0, 1, 2)):
    root = Path(root).resolve()
    runs = []
    for path in sorted(root.glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        runs.append({"run_dir": str(path.parent), **metrics})
    expected_seeds = set(expected_seeds)
    output = {
        "schema_version": 1,
        "protocol": "all five tasks seen; 7 success + 7 failure train and 3 + 3 test per task",
        "models": {},
    }
    for model in sorted({run["model"] for run in runs}):
        values = [run for run in runs if run["model"] == model]
        seeds = {run["seed"] for run in values}
        if seeds != expected_seeds:
            raise ValueError(f"{model} seeds are {sorted(seeds)}, expected {sorted(expected_seeds)}")
        scalar_keys = sorted({key for run in values for key in run["scalar_metrics"]})
        scalars = {}
        for key in scalar_keys:
            numbers = [run["scalar_metrics"].get(key) for run in values]
            numbers = [float(value) for value in numbers if value is not None]
            scalars[key] = {
                "mean": float(np.mean(numbers)) if numbers else None,
                "std": float(np.std(numbers)) if numbers else None,
            }
        output["models"][model] = {
            "seeds": sorted(seeds),
            "primary_metric": "falert_early_roc_auc/model_test",
            "test_roc_auc_mean": scalars["falert_early_roc_auc/model_test"]["mean"],
            "test_roc_auc_std": scalars["falert_early_roc_auc/model_test"]["std"],
            "test_prc_auc_mean": scalars["falert_early_prc_auc/model_test"]["mean"],
            "test_prc_auc_std": scalars["falert_early_prc_auc/model_test"]["std"],
            "scalar_metrics": scalars,
            "run_dirs": [run["run_dir"] for run in values],
        }
    output["num_completed_runs"] = len(runs)
    (root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = summarize(args.root, args.expected_seeds)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
