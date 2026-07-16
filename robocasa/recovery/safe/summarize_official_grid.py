"""Select official SAFE configurations by mean val-seen early ROC-AUC across seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate_official_safe import SELECTION_METRIC, write_json


def collect_grid_results(output_root):
    output_root = Path(output_root).resolve()
    results = []
    for metrics_path in sorted(output_root.glob("*/evaluation/metrics.json")):
        evaluation_dir = metrics_path.parent
        provenance_path = evaluation_dir / "provenance.json"
        if not provenance_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text())
        provenance = json.loads(provenance_path.read_text())
        model_cfg = provenance["model_config"]
        dataset_cfg = provenance["dataset_config"]
        results.append(
            {
                "run_dir": str(evaluation_dir.parent),
                "model": provenance["model"],
                "horizon_selector": str(dataset_cfg["horizon_idx_rel"]),
                "diffusion_selector": str(dataset_cfg["diff_idx_rel"]),
                "learning_rate": float(model_cfg["lr"]),
                "lambda_reg": float(model_cfg["lambda_reg"]),
                "seed": int(provenance["seed"]),
                "val_seen": metrics["scalar_metrics"].get(SELECTION_METRIC),
                "val_unseen": metrics["scalar_metrics"].get(
                    "falert_early_roc_auc/model_val_unseen"
                ),
                "train": metrics["scalar_metrics"].get(
                    "falert_early_roc_auc/model_train"
                ),
            }
        )
    return results


def summarize_grid(results, expected_seeds=(0, 1, 2)):
    groups = defaultdict(list)
    for result in results:
        key = (
            result["model"],
            result["horizon_selector"],
            result["diffusion_selector"],
            result["learning_rate"],
            result["lambda_reg"],
        )
        groups[key].append(result)
    rows = []
    expected_seeds = set(expected_seeds)
    for key, values in groups.items():
        seeds = {item["seed"] for item in values}
        row = {
            "model": key[0],
            "horizon_selector": key[1],
            "diffusion_selector": key[2],
            "learning_rate": key[3],
            "lambda_reg": key[4],
            "seeds": sorted(seeds),
            "complete_seed_set": seeds == expected_seeds,
            "num_runs": len(values),
        }
        for metric in ("train", "val_seen", "val_unseen"):
            numbers = [item[metric] for item in values if item[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(numbers)) if numbers else None
            row[f"{metric}_std"] = float(np.std(numbers)) if numbers else None
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["model"],
            not row["complete_seed_set"],
            -(row["val_seen_mean"] if row["val_seen_mean"] is not None else -1),
        )
    )
    best = {}
    for model in sorted({row["model"] for row in rows}):
        candidates = [
            row
            for row in rows
            if row["model"] == model
            and row["complete_seed_set"]
            and row["val_seen_mean"] is not None
        ]
        if candidates:
            best[model] = max(candidates, key=lambda row: row["val_seen_mean"])
    return {
        "schema_version": 1,
        "selection_rule": "maximum mean val-seen falert_early ROC-AUC across seeds",
        "selection_metric": SELECTION_METRIC,
        "num_completed_runs": len(results),
        "num_configurations": len(rows),
        "best_by_model": best,
        "configurations": rows,
    }


def save_summary(summary, output_root):
    output_root = Path(output_root)
    write_json(output_root / "selection_summary.json", summary)
    rows = summary["configurations"]
    if rows:
        with (output_root / "selection_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "seeds": "-".join(map(str, row["seeds"]))})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    results = collect_grid_results(args.output_root)
    summary = summarize_grid(results, args.expected_seeds)
    save_summary(summary, args.output_root)
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
