"""Summarize matched-size SAFE natural-rate screens without pseudoreplication."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


METRICS = {
    "test_roc_auc": "falert_early_roc_auc/model_test",
    "test_prc_auc": "falert_early_prc_auc/model_test",
}


def mean_std(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(value) for value in values],
    }


def summarize(root, expected_subset_seeds=(0, 1, 2, 3, 4), expected_model_seeds=(0, 1, 2)):
    root = Path(root).resolve()
    records = []
    for run_path in sorted(root.glob("*/screen_run.json")):
        metrics_path = run_path.parent / "metrics.json"
        if not metrics_path.is_file():
            continue
        run = json.loads(run_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        records.append({**run, "metrics": metrics})
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["regime"])].append(record)
    expected_subset_seeds = set(expected_subset_seeds)
    expected_model_seeds = set(expected_model_seeds)
    output = {
        "schema_version": 1,
        "protocol": (
            "fixed held-out test; subset-seed means are the independent "
            "resampling units"
        ),
        "num_completed_runs": len(records),
        "groups": {},
    }
    for (model, regime), values in sorted(grouped.items()):
        by_subset = defaultdict(list)
        for record in values:
            by_subset[int(record["subset_seed"])].append(record)
        if set(by_subset) != expected_subset_seeds:
            raise ValueError(
                f"{model}/{regime} subset seeds are {sorted(by_subset)}, "
                f"expected {sorted(expected_subset_seeds)}"
            )
        subset_rows = []
        for subset_seed, subset_values in sorted(by_subset.items()):
            model_seeds = {int(value["model_seed"]) for value in subset_values}
            if model_seeds != expected_model_seeds:
                raise ValueError(
                    f"{model}/{regime}/subset-{subset_seed} model seeds are "
                    f"{sorted(model_seeds)}, expected {sorted(expected_model_seeds)}"
                )
            row = {
                "subset_seed": subset_seed,
                "model_seeds": sorted(model_seeds),
            }
            for label, key in METRICS.items():
                scores = [
                    float(value["metrics"]["scalar_metrics"][key])
                    for value in subset_values
                ]
                row[label] = float(np.mean(scores))
            subset_rows.append(row)
        result = {
            "model": model,
            "regime": regime,
            "num_runs": len(values),
            "num_subset_seeds": len(subset_rows),
            "subset_seed_results": subset_rows,
            "counts": values[0]["metrics"]["counts"],
            "class_weighting": values[0]["class_weighting"],
        }
        for label in METRICS:
            result[label] = mean_std([row[label] for row in subset_rows])
        output["groups"][f"{model}/{regime}"] = result
    comparisons = {}
    for model in sorted({key[0] for key in grouped}):
        available = {
            regime: output["groups"].get(f"{model}/{regime}")
            for regime in (
                "matched_weighted",
                "natural_weighted",
                "natural_unweighted",
            )
        }
        if not all(available.values()):
            continue
        model_comparisons = {}
        for label in METRICS:
            matched = {
                row["subset_seed"]: row[label]
                for row in available["matched_weighted"]["subset_seed_results"]
            }
            natural = {
                row["subset_seed"]: row[label]
                for row in available["natural_weighted"]["subset_seed_results"]
            }
            unweighted = {
                row["subset_seed"]: row[label]
                for row in available["natural_unweighted"]["subset_seed_results"]
            }
            model_comparisons[label] = {
                "natural_minus_matched": mean_std(
                    [natural[seed] - matched[seed] for seed in sorted(natural)]
                ),
                "weighted_minus_unweighted": mean_std(
                    [natural[seed] - unweighted[seed] for seed in sorted(natural)]
                ),
            }
        comparisons[model] = model_comparisons
    output["paired_comparisons"] = comparisons
    (root / "natural_rate_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--expected-subset-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--expected-model-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = summarize(
        args.root,
        args.expected_subset_seeds,
        args.expected_model_seeds,
    )
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
