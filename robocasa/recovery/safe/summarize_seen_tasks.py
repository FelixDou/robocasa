"""Aggregate final SAFE refits on one fixed same-task held-out split."""

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
    count_records = [run.get("counts") for run in runs if run.get("counts") is not None]
    if count_records and any(
        counts != count_records[0] for counts in count_records[1:]
    ):
        raise ValueError("Final runs do not use identical train/test counts")
    split_counts = count_records[0] if count_records else None
    task_counts = {
        run.get("num_tasks") for run in runs if run.get("num_tasks") is not None
    }
    if len(task_counts) > 1:
        raise ValueError(f"Final runs disagree on task count: {sorted(task_counts)}")
    num_tasks = next(iter(task_counts)) if task_counts else None
    task_type_filters = {run.get("task_type_filter", "all") for run in runs}
    if len(task_type_filters) > 1:
        raise ValueError(
            f"Final runs mix task-type filters: {sorted(task_type_filters)}"
        )
    task_type_maps = {
        json.dumps(run.get("task_types", {}), sort_keys=True) for run in runs
    }
    if len(task_type_maps) > 1:
        raise ValueError("Final runs disagree on selected task identities")
    objectives = {
        json.dumps(
            run.get(
                "training_objective",
                {
                    "loss_mode": "official",
                    "focal_gamma": 2.0,
                    "score_output": "pinned_safe_output",
                },
            ),
            sort_keys=True,
        )
        for run in runs
    }
    if len(objectives) > 1:
        raise ValueError("Final runs mix incompatible training objectives")
    causal_protocols = {
        json.dumps(run.get("causal_subtask_safe"), sort_keys=True) for run in runs
    }
    if len(causal_protocols) > 1:
        raise ValueError("Final runs mix incompatible causal Subtask-SAFE protocols")
    output = {
        "schema_version": 1,
        "protocol": "same tasks in train and test; fixed outcome-stratified outer split",
        "num_tasks": num_tasks,
        "task_type_filter": (
            next(iter(task_type_filters)) if task_type_filters else "all"
        ),
        "task_types": (
            json.loads(next(iter(task_type_maps))) if task_type_maps else {}
        ),
        "split_counts": split_counts,
        "training_objective": (
            json.loads(next(iter(objectives))) if objectives else None
        ),
        "causal_subtask_safe": (
            json.loads(next(iter(causal_protocols))) if causal_protocols else None
        ),
        "models": {},
    }
    for model in sorted({run["model"] for run in runs}):
        values = [run for run in runs if run["model"] == model]
        seeds = {run["seed"] for run in values}
        if seeds != expected_seeds:
            raise ValueError(
                f"{model} seeds are {sorted(seeds)}, expected {sorted(expected_seeds)}"
            )
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
    (root / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
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
