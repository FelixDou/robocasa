"""Select all-seen SAFE hyperparameters from training-only inner CV."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def summarize_seen_cv(root, expected_folds=(0, 1, 2)):
    root = Path(root).resolve()
    records = []
    for path in sorted(root.glob("*/metrics.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "complete":
            records.append(record)
    groups = defaultdict(list)
    for record in records:
        key = (
            record["model"],
            str(record["horizon_selector"]),
            str(record["diffusion_selector"]),
            float(record["learning_rate"]),
            float(record["lambda_reg"]),
        )
        groups[key].append(record)
    expected_folds = set(expected_folds)
    outer_counts = [
        record.get("counts", {}).get("outer_train")
        for record in records
        if record.get("counts", {}).get("outer_train") is not None
    ]
    if outer_counts and any(counts != outer_counts[0] for counts in outer_counts[1:]):
        raise ValueError("CV runs disagree on the outer training split counts")
    test_counts = [
        record.get("counts", {}).get("outer_test_untouched")
        for record in records
        if record.get("counts", {}).get("outer_test_untouched") is not None
    ]
    if test_counts and any(counts != test_counts[0] for counts in test_counts[1:]):
        raise ValueError("CV runs disagree on the untouched outer test counts")
    rows = []
    for key, values in groups.items():
        folds = {int(value["fold"]) for value in values}
        scores = [float(value["selection_value"]) for value in values]
        rows.append(
            {
                "model": key[0],
                "horizon_selector": key[1],
                "diffusion_selector": key[2],
                "learning_rate": key[3],
                "lambda_reg": key[4],
                "folds": sorted(folds),
                "complete_fold_set": folds == expected_folds,
                "num_folds": len(folds),
                "inner_val_mean": float(np.mean(scores)),
                "inner_val_std": float(np.std(scores)),
            }
        )
    rows.sort(key=lambda row: (row["model"], not row["complete_fold_set"], -row["inner_val_mean"]))
    best = {}
    for model in sorted({row["model"] for row in rows}):
        candidates = [
            row
            for row in rows
            if row["model"] == model and row["complete_fold_set"]
        ]
        if candidates:
            best[model] = max(candidates, key=lambda row: row["inner_val_mean"])
    result = {
        "schema_version": 1,
        "protocol": "training-only outcome-stratified CV inside a fixed outer training pool",
        "selection_rule": "maximum mean matched-earliest inner-validation ROC-AUC",
        "outer_test_used_for_selection": False,
        "outer_train_counts": outer_counts[0] if outer_counts else None,
        "outer_test_counts": test_counts[0] if test_counts else None,
        "num_completed_fits": len(records),
        "num_configurations": len(rows),
        "best_by_model": best,
        "configurations": rows,
    }
    (root / "cv_selection_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if rows:
        with (root / "cv_selection_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "folds": "-".join(map(str, row["folds"]))})
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = summarize_seen_cv(args.root, args.expected_folds)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
