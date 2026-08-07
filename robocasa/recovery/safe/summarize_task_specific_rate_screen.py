"""Summarize one-model-per-task balanced versus natural-rate SAFE runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


METRICS = {
    "roc_auc": "falert_early_roc_auc/model_test",
    "average_precision": "falert_early_prc_auc/model_test",
}
REGIMES = ("matched_weighted", "natural_weighted")


def mean_std(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(value) for value in values],
    }


def summarize(
    root,
    expected_subset_seeds=(0, 1, 2, 3, 4),
    expected_model_seeds=(0, 1, 2),
    formats=(),
):
    root = Path(root).resolve()
    records = []
    for run_path in sorted(root.glob("**/task_screen_run.json")):
        metrics_path = run_path.parent / "metrics.json"
        if not metrics_path.is_file():
            continue
        run = json.loads(run_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        records.append({**run, "metrics": metrics})
    if not records:
        raise ValueError(f"No completed task-specific runs found in {root}")
    expected_subset_seeds = set(map(int, expected_subset_seeds))
    expected_model_seeds = set(map(int, expected_model_seeds))
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task_name"], record["model"], record["regime"])].append(
            record
        )
    task_results = {}
    csv_rows = []
    frozen_test_by_task = {}
    for (task_name, model, regime), values in sorted(grouped.items()):
        by_subset = defaultdict(list)
        for value in values:
            by_subset[int(value["subset_seed"])].append(value)
        if set(by_subset) != expected_subset_seeds:
            raise ValueError(
                f"{task_name}/{model}/{regime} subset seeds are "
                f"{sorted(by_subset)}, expected {sorted(expected_subset_seeds)}"
            )
        subset_rows = []
        frozen_test = None
        for subset_seed, subset_values in sorted(by_subset.items()):
            model_seeds = {int(value["model_seed"]) for value in subset_values}
            if model_seeds != expected_model_seeds:
                raise ValueError(
                    f"{task_name}/{model}/{regime}/subset-{subset_seed} model "
                    f"seeds are {sorted(model_seeds)}, expected "
                    f"{sorted(expected_model_seeds)}"
                )
            split_paths = {
                str(value["selection_manifest"]) for value in subset_values
            }
            if len(split_paths) != 1:
                raise ValueError("Model seeds do not share one selection manifest")
            manifest = json.loads(Path(next(iter(split_paths))).read_text())
            test_ids = tuple(manifest["test"])
            shared_test = frozen_test_by_task.setdefault(task_name, test_ids)
            if test_ids != shared_test:
                raise ValueError(
                    f"Balanced/natural frozen test IDs differ for {task_name}"
                )
            if frozen_test is None:
                frozen_test = test_ids
            elif test_ids != frozen_test:
                raise ValueError(f"Frozen test IDs changed for {task_name}")
            row = {
                "subset_seed": subset_seed,
                "model_seeds": sorted(model_seeds),
                "train_counts": manifest["counts"]["train"],
                "test_counts": manifest["counts"]["test"],
            }
            for label, key in METRICS.items():
                row[label] = float(
                    np.mean(
                        [
                            value["metrics"]["scalar_metrics"][key]
                            for value in subset_values
                        ]
                    )
                )
            subset_rows.append(row)
        result = {
            "task_name": task_name,
            "task_type": values[0]["task_type"],
            "model": model,
            "regime": regime,
            "num_runs": len(values),
            "num_subset_seeds": len(subset_rows),
            "subset_results": subset_rows,
            "frozen_test_rollouts": len(frozen_test),
        }
        for label in METRICS:
            result[label] = mean_std([row[label] for row in subset_rows])
        key = f"{task_name}/{model}/{regime}"
        task_results[key] = result
        csv_rows.append(
            {
                "task_name": task_name,
                "task_type": result["task_type"],
                "model": model,
                "regime": regime,
                "train_successes": subset_rows[0]["train_counts"]["successes"],
                "train_failures": subset_rows[0]["train_counts"]["failures"],
                "test_successes": subset_rows[0]["test_counts"]["successes"],
                "test_failures": subset_rows[0]["test_counts"]["failures"],
                "roc_auc_mean": result["roc_auc"]["mean"],
                "roc_auc_std": result["roc_auc"]["std"],
                "average_precision_mean": result["average_precision"]["mean"],
                "average_precision_std": result["average_precision"]["std"],
            }
        )

    tasks = sorted({record["task_name"] for record in records})
    models = sorted({record["model"] for record in records})
    missing_groups = [
        f"{task}/{model}/{regime}"
        for task in tasks
        for model in models
        for regime in REGIMES
        if f"{task}/{model}/{regime}" not in task_results
    ]
    if missing_groups:
        raise ValueError("Incomplete task/regime comparison: " + ", ".join(missing_groups))
    aggregate = {}
    aggregate_by_task_type = {}
    paired = {}
    for model in models:
        aggregate[model] = {}
        aggregate_by_task_type[model] = {}
        for regime in REGIMES:
            keys = [f"{task}/{model}/{regime}" for task in tasks]
            subset_macro = []
            for subset_seed in sorted(expected_subset_seeds):
                row = {"subset_seed": subset_seed}
                for metric in METRICS:
                    row[metric] = float(
                        np.mean(
                            [
                                next(
                                    item[metric]
                                    for item in task_results[key]["subset_results"]
                                    if item["subset_seed"] == subset_seed
                                )
                                for key in keys
                            ]
                        )
                    )
                subset_macro.append(row)
            aggregate[model][regime] = {
                metric: mean_std([row[metric] for row in subset_macro])
                for metric in METRICS
            }
            aggregate[model][regime]["subset_macro_results"] = subset_macro
            aggregate_by_task_type[model][regime] = {}
            task_types = sorted(
                {task_results[key]["task_type"] for key in keys}
            )
            for task_type in task_types:
                type_keys = [
                    key for key in keys if task_results[key]["task_type"] == task_type
                ]
                type_subset_rows = []
                for subset_seed in sorted(expected_subset_seeds):
                    row = {"subset_seed": subset_seed}
                    for metric in METRICS:
                        row[metric] = float(
                            np.mean(
                                [
                                    next(
                                        item[metric]
                                        for item in task_results[key]["subset_results"]
                                        if item["subset_seed"] == subset_seed
                                    )
                                    for key in type_keys
                                ]
                            )
                        )
                    type_subset_rows.append(row)
                aggregate_by_task_type[model][regime][task_type] = {
                    metric: mean_std(
                        [row[metric] for row in type_subset_rows]
                    )
                    for metric in METRICS
                }
                aggregate_by_task_type[model][regime][task_type][
                    "num_tasks"
                ] = len(type_keys)
        if all(regime in aggregate[model] for regime in REGIMES):
            paired[model] = {}
            for metric in METRICS:
                balanced = {
                    row["subset_seed"]: row[metric]
                    for row in aggregate[model]["matched_weighted"][
                        "subset_macro_results"
                    ]
                }
                natural = {
                    row["subset_seed"]: row[metric]
                    for row in aggregate[model]["natural_weighted"][
                        "subset_macro_results"
                    ]
                }
                paired[model][f"natural_minus_balanced_{metric}"] = mean_std(
                    [natural[seed] - balanced[seed] for seed in sorted(balanced)]
                )
    output = {
        "schema_version": 1,
        "protocol": (
            "task-specific normal SAFE; fixed per-task test; subset seeds are "
            "training-resampling units"
        ),
        "num_completed_runs": len(records),
        "tasks": tasks,
        "models": models,
        "task_results": task_results,
        "macro_across_tasks": aggregate,
        "macro_by_task_type": aggregate_by_task_type,
        "paired_comparisons": paired,
        "interpretation_guardrail": (
            "The same frozen test episodes are reused across training subset and "
            "model seeds; variation across runs measures training sensitivity, not "
            "independent test-set uncertainty."
        ),
    }
    write_path = root / "task_specific_summary.json"
    write_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    with (root / "task_specific_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    if formats:
        plot_results(csv_rows, root, formats)
    return output


def plot_results(rows, output_dir, formats):
    import matplotlib.pyplot as plt

    labels = {"matched_weighted": "Balanced", "natural_weighted": "Natural rate"}
    colors = {"matched_weighted": "#2563A6", "natural_weighted": "#D97706"}
    for model in sorted({row["model"] for row in rows}):
        values = [row for row in rows if row["model"] == model]
        tasks = sorted({row["task_name"] for row in values})
        for metric, title, filename in (
            ("roc_auc", "Held-out ROC-AUC", "task_specific_roc_auc"),
            ("average_precision", "Held-out average precision", "task_specific_ap"),
        ):
            x = np.arange(len(tasks))
            width = 0.38
            fig, axis = plt.subplots(figsize=(14, 6.5))
            for offset, regime in ((-0.5, "matched_weighted"), (0.5, "natural_weighted")):
                selected = {
                    row["task_name"]: row
                    for row in values
                    if row["regime"] == regime
                }
                means = [selected[task][f"{metric}_mean"] for task in tasks]
                stds = [selected[task][f"{metric}_std"] for task in tasks]
                axis.bar(
                    x + offset * width,
                    means,
                    width,
                    yerr=stds,
                    capsize=3,
                    label=labels[regime],
                    color=colors[regime],
                    alpha=0.9,
                )
            axis.axhline(0.5, color="#64748B", linestyle="--", linewidth=1)
            axis.set_ylim(0.35, 1.0)
            axis.set_ylabel(title)
            axis.set_title(f"Task-specific normal SAFE ({model})")
            axis.set_xticks(x)
            axis.set_xticklabels(tasks, rotation=35, ha="right")
            axis.legend()
            axis.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            for format_name in formats:
                fig.savefig(
                    Path(output_dir) / f"{filename}_{model}.{format_name}",
                    dpi=180,
                    bbox_inches="tight",
                )
            plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--expected-subset-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]
    )
    parser.add_argument(
        "--expected-model-seeds", nargs="+", type=int, default=[0, 1, 2]
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = summarize(
            args.root,
            args.expected_subset_seeds,
            args.expected_model_seeds,
            args.formats,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
