"""Analyze natural-rate SAFE screens with training-only task normalization."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .calibrate_seen_tasks import (
        fit_task_normalization,
        load_score_records,
        truncated_scores,
    )
except ImportError:
    from calibrate_seen_tasks import (
        fit_task_normalization,
        load_score_records,
        truncated_scores,
    )


REGIMES = ("matched_weighted", "natural_weighted", "natural_unweighted")
MODELS = ("indep", "lstm")
SCALAR_METRICS = (
    "raw_pooled_roc_auc",
    "task_z_pooled_roc_auc",
    "macro_task_roc_auc",
    "raw_average_precision",
    "task_z_average_precision",
    "macro_task_average_precision",
)
ROC_METRICS = (
    "raw_pooled_roc_auc",
    "task_z_pooled_roc_auc",
    "macro_task_roc_auc",
)


def mean_std(values):
    values = [float(value) for value in values]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": values,
    }


def _metric(function, labels, scores, description):
    if len(set(labels)) != 2:
        raise ValueError(f"{description} does not contain both outcomes")
    return float(function(labels, scores))


def _evaluate_values(records, normalized_maxima):
    test = [record for record in records if record["split"] == "test"]
    if not test:
        raise ValueError("Score records contain no test split")
    labels = [int(record["failed"]) for record in test]
    raw = [float(np.max(truncated_scores(record))) for record in test]
    normalized = [normalized_maxima[record["rollout_id"]] for record in test]
    return {
        "raw_pooled_roc_auc": _metric(
            roc_auc_score, labels, raw, "Pooled test split"
        ),
        "task_z_pooled_roc_auc": _metric(
            roc_auc_score, labels, normalized, "Task-z pooled test split"
        ),
        "raw_average_precision": _metric(
            average_precision_score, labels, raw, "Pooled test split"
        ),
        "task_z_average_precision": _metric(
            average_precision_score,
            labels,
            normalized,
            "Task-z pooled test split",
        ),
        "num_rollouts": len(test),
        "successes": sum(not label for label in labels),
        "failures": sum(labels),
    }


def evaluate_run(run_root):
    run_root = Path(run_root)
    run = json.loads((run_root / "screen_run.json").read_text())
    metrics = json.loads((run_root / "metrics.json").read_text())
    records = load_score_records(run_root / "scores.jsonl")
    normalization = fit_task_normalization(records)
    normalized_maxima = {}
    for record in records:
        raw_maximum = float(np.max(truncated_scores(record)))
        stats = normalization[record["task_name"]]
        normalized_maxima[record["rollout_id"]] = (
            raw_maximum - stats["location"]
        ) / stats["scale"]
    result = {
        "model": run["model"],
        "regime": run["regime"],
        "subset_seed": int(run["subset_seed"]),
        "model_seed": int(run["model_seed"]),
        "class_weighting": run["class_weighting"],
        "run_dir": str(run_root.resolve()),
        "counts": metrics["counts"],
        "task_normalization": normalization,
    }
    result.update(_evaluate_values(records, normalized_maxima))
    test = [record for record in records if record["split"] == "test"]
    per_task = {}
    task_types = {}
    for task_name in sorted({record["task_name"] for record in test}):
        selected = [record for record in test if record["task_name"] == task_name]
        values = _evaluate_values(selected, normalized_maxima)
        per_task[task_name] = values
        types = {record.get("task_type") for record in selected}
        if len(types) != 1 or next(iter(types)) not in {"atomic", "composite"}:
            raise ValueError(f"Task {task_name} has invalid task-type provenance: {types}")
        task_types[task_name] = next(iter(types))
    result["per_task"] = per_task
    result["task_types"] = task_types
    result["macro_task_roc_auc"] = float(
        np.mean([values["raw_pooled_roc_auc"] for values in per_task.values()])
    )
    result["macro_task_average_precision"] = float(
        np.mean([values["raw_average_precision"] for values in per_task.values()])
    )
    by_type = {}
    for task_type in ("atomic", "composite"):
        selected = [
            record
            for record in records
            if task_types[record["task_name"]] == task_type
        ]
        values = _evaluate_values(selected, normalized_maxima)
        selected_tasks = [
            task for task, value in task_types.items() if value == task_type
        ]
        values["macro_task_roc_auc"] = float(
            np.mean(
                [
                    per_task[task]["raw_pooled_roc_auc"]
                    for task in selected_tasks
                ]
            )
        )
        values["macro_task_average_precision"] = float(
            np.mean(
                [
                    per_task[task]["raw_average_precision"]
                    for task in selected_tasks
                ]
            )
        )
        values["tasks"] = selected_tasks
        by_type[task_type] = values
    result["by_task_type"] = by_type
    official_raw = float(
        metrics["scalar_metrics"]["falert_early_roc_auc/model_test"]
    )
    if not np.isclose(result["raw_pooled_roc_auc"], official_raw, atol=1e-8):
        raise ValueError(
            f"Recomputed raw ROC {result['raw_pooled_roc_auc']} disagrees "
            f"with official metric {official_raw} in {run_root}"
        )
    return result


def _validate_identities(run_records):
    test_sets = set()
    train_sets = defaultdict(set)
    for record in run_records:
        scores = load_score_records(Path(record["run_dir"]) / "scores.jsonl")
        test_ids = tuple(
            sorted(item["rollout_id"] for item in scores if item["split"] == "test")
        )
        train_ids = tuple(
            sorted(item["rollout_id"] for item in scores if item["split"] == "train")
        )
        test_sets.add(test_ids)
        train_sets[(record["regime"], record["subset_seed"])].add(train_ids)
    if len(test_sets) != 1:
        raise ValueError("Detailed-analysis runs do not share one fixed test set")
    invalid = {
        key: len(values)
        for key, values in train_sets.items()
        if len(values) != 1
    }
    if invalid:
        raise ValueError(
            "Model/model-seed runs disagree on training IDs within selections: "
            f"{invalid}"
        )
    return len(next(iter(test_sets)))


def _aggregate_group(records, expected_subset_seeds, expected_model_seeds):
    by_subset = defaultdict(list)
    for record in records:
        by_subset[record["subset_seed"]].append(record)
    if set(by_subset) != set(expected_subset_seeds):
        raise ValueError(
            f"Subset seeds are {sorted(by_subset)}, expected {sorted(expected_subset_seeds)}"
        )
    subset_rows = []
    for subset_seed, values in sorted(by_subset.items()):
        seeds = {record["model_seed"] for record in values}
        if seeds != set(expected_model_seeds):
            raise ValueError(
                f"Subset {subset_seed} model seeds are {sorted(seeds)}, "
                f"expected {sorted(expected_model_seeds)}"
            )
        row = {"subset_seed": subset_seed, "model_seeds": sorted(seeds)}
        for metric in SCALAR_METRICS:
            row[metric] = float(np.mean([record[metric] for record in values]))
        row["per_task"] = {}
        for task in sorted(values[0]["per_task"]):
            row["per_task"][task] = {
                metric: float(
                    np.mean([record["per_task"][task][metric] for record in values])
                )
                for metric in (
                    "raw_pooled_roc_auc",
                    "raw_average_precision",
                )
            }
        row["by_task_type"] = {}
        for task_type in ("atomic", "composite"):
            row["by_task_type"][task_type] = {
                metric: float(
                    np.mean(
                        [
                            record["by_task_type"][task_type][metric]
                            for record in values
                        ]
                    )
                )
                for metric in SCALAR_METRICS
            }
        subset_rows.append(row)
    output = {
        "model": records[0]["model"],
        "regime": records[0]["regime"],
        "class_weighting": records[0]["class_weighting"],
        "counts": records[0]["counts"],
        "num_runs": len(records),
        "subset_results": subset_rows,
    }
    for metric in SCALAR_METRICS:
        output[metric] = mean_std([row[metric] for row in subset_rows])
    output["per_task"] = {}
    for task in sorted(subset_rows[0]["per_task"]):
        output["per_task"][task] = {
            "task_type": records[0]["task_types"][task],
            "roc_auc": mean_std(
                [
                    row["per_task"][task]["raw_pooled_roc_auc"]
                    for row in subset_rows
                ]
            ),
            "average_precision": mean_std(
                [
                    row["per_task"][task]["raw_average_precision"]
                    for row in subset_rows
                ]
            ),
        }
    output["by_task_type"] = {}
    for task_type in ("atomic", "composite"):
        output["by_task_type"][task_type] = {
            metric: mean_std(
                [row["by_task_type"][task_type][metric] for row in subset_rows]
            )
            for metric in SCALAR_METRICS
        }
    return output


def _paired_comparisons(groups):
    output = {}
    for model in MODELS:
        model_groups = {
            regime: groups.get(f"{model}/{regime}") for regime in REGIMES
        }
        if not all(model_groups.values()):
            continue
        comparison = {}
        for metric in SCALAR_METRICS:
            values = {
                regime: {
                    row["subset_seed"]: row[metric]
                    for row in group["subset_results"]
                }
                for regime, group in model_groups.items()
            }
            seeds = sorted(values["natural_weighted"])
            comparison[metric] = {
                "natural_minus_matched": mean_std(
                    [
                        values["natural_weighted"][seed]
                        - values["matched_weighted"][seed]
                        for seed in seeds
                    ]
                ),
                "weighted_minus_unweighted": mean_std(
                    [
                        values["natural_weighted"][seed]
                        - values["natural_unweighted"][seed]
                        for seed in seeds
                    ]
                ),
            }
        output[model] = comparison
    return output


def write_tables(summary, output_dir):
    output_dir = Path(output_dir)
    group_rows = []
    task_rows = []
    for key, group in sorted(summary["groups"].items()):
        row = {
            "group": key,
            "model": group["model"],
            "regime": group["regime"],
            "class_weighting": group["class_weighting"],
            "train_rollouts": group["counts"]["train"],
            "test_rollouts": group["counts"]["test"],
        }
        for metric in SCALAR_METRICS:
            row[f"{metric}_mean"] = group[metric]["mean"]
            row[f"{metric}_std"] = group[metric]["std"]
        group_rows.append(row)
        for task, values in group["per_task"].items():
            task_rows.append(
                {
                    "group": key,
                    "model": group["model"],
                    "regime": group["regime"],
                    "task": task,
                    "task_type": values["task_type"],
                    "roc_auc_mean": values["roc_auc"]["mean"],
                    "roc_auc_std": values["roc_auc"]["std"],
                    "average_precision_mean": values["average_precision"]["mean"],
                    "average_precision_std": values["average_precision"]["std"],
                }
            )
    for path, rows in (
        (output_dir / "group_metrics.csv", group_rows),
        (output_dir / "per_task_metrics.csv", task_rows),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#1F2937",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.7,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "text.color": "#111827",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    return plt


def _save(fig, output_dir, stem, formats):
    outputs = []
    for extension in formats:
        path = Path(output_dir) / f"{stem}.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None)
        outputs.append(str(path))
    return outputs


def plot_summary(summary, output_dir, formats):
    plt = configure_matplotlib()
    regime_labels = {
        "matched_weighted": "Matched weighted",
        "natural_weighted": "Natural weighted",
        "natural_unweighted": "Natural unweighted",
    }
    colors = {
        "matched_weighted": "#2563A6",
        "natural_weighted": "#D97706",
        "natural_unweighted": "#7C3AED",
    }
    markers = {
        "matched_weighted": "o",
        "natural_weighted": "s",
        "natural_unweighted": "^",
    }
    metric_labels = ("Raw pooled", "Task-z pooled", "Macro task")
    outputs = []
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    x = np.arange(len(ROC_METRICS))
    offsets = (-0.18, 0.0, 0.18)
    for ax, model in zip(axes, MODELS):
        for offset, regime in zip(offsets, REGIMES):
            group = summary["groups"][f"{model}/{regime}"]
            means = [group[metric]["mean"] for metric in ROC_METRICS]
            stds = [group[metric]["std"] for metric in ROC_METRICS]
            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                color=colors[regime],
                marker=markers[regime],
                linestyle="none",
                capsize=4,
                markersize=7,
                label=regime_labels[regime],
            )
        ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.2)
        ax.set_xticks(x, metric_labels)
        ax.set_title("MLP" if model == "indep" else "LSTM")
        ax.grid(axis="y", alpha=0.75)
    axes[0].set_ylabel("Held-out ROC-AUC")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("SAFE discrimination by training regime and aggregation")
    fig.text(
        0.5,
        0.01,
        "Mean and population SD across five subset seeds after averaging three model seeds",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    outputs.extend(_save(fig, output_dir, "roc_aggregation_comparison", formats))
    plt.close(fig)

    for model in MODELS:
        tasks = sorted(
            summary["groups"][f"{model}/{REGIMES[0]}"]["per_task"],
            key=lambda task: summary["groups"][f"{model}/{REGIMES[0]}"]["per_task"][task]["roc_auc"]["mean"],
        )
        fig, ax = plt.subplots(figsize=(10.2, 6.5))
        y = np.arange(len(tasks))
        for offset, regime in zip(offsets, REGIMES):
            group = summary["groups"][f"{model}/{regime}"]
            means = [group["per_task"][task]["roc_auc"]["mean"] for task in tasks]
            stds = [group["per_task"][task]["roc_auc"]["std"] for task in tasks]
            ax.errorbar(
                means,
                y + offset,
                xerr=stds,
                color=colors[regime],
                marker=markers[regime],
                linestyle="none",
                capsize=3,
                markersize=6,
                label=regime_labels[regime],
            )
        labels = [
            f"{task} ({summary['groups'][f'{model}/{REGIMES[0]}']['per_task'][task]['task_type'][0].upper()})"
            for task in tasks
        ]
        ax.axvline(0.5, color="#6B7280", linestyle="--", linewidth=1.2)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Held-out per-task ROC-AUC")
        ax.set_title(
            f"Per-task SAFE discrimination — {'MLP' if model == 'indep' else 'LSTM'}"
        )
        ax.grid(axis="x", alpha=0.75)
        ax.legend(frameon=False, loc="lower right")
        fig.text(
            0.5,
            0.01,
            "A = atomic, C = composite; mean and population SD across five subset seeds",
            ha="center",
            color="#4B5563",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        outputs.extend(_save(fig, output_dir, f"per_task_roc_{model}", formats))
        plt.close(fig)
    return outputs


def analyze(
    root,
    output_dir,
    *,
    expected_subset_seeds=(0, 1, 2, 3, 4),
    expected_model_seeds=(0, 1, 2),
    formats=("png", "pdf"),
    make_plots=True,
):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_records = []
    for path in sorted(root.glob("*/screen_run.json")):
        run_root = path.parent
        if (run_root / "metrics.json").is_file() and (run_root / "scores.jsonl").is_file():
            run_records.append(evaluate_run(run_root))
    expected_runs = (
        len(MODELS)
        * len(REGIMES)
        * len(expected_subset_seeds)
        * len(expected_model_seeds)
    )
    if len(run_records) != expected_runs:
        raise ValueError(
            f"Found {len(run_records)} completed score runs, expected {expected_runs}"
        )
    fixed_test_count = _validate_identities(run_records)
    groups = {}
    for model in MODELS:
        for regime in REGIMES:
            selected = [
                record
                for record in run_records
                if record["model"] == model and record["regime"] == regime
            ]
            groups[f"{model}/{regime}"] = _aggregate_group(
                selected,
                expected_subset_seeds,
                expected_model_seeds,
            )
    summary = {
        "schema_version": 1,
        "protocol": (
            "fixed held-out test; task normalization fitted from each run's "
            "training records only; model seeds averaged within subset seed"
        ),
        "root": str(root),
        "num_runs": len(run_records),
        "fixed_test_rollouts": fixed_test_count,
        "groups": groups,
        "paired_comparisons": _paired_comparisons(groups),
    }
    with (output_dir / "run_metrics.jsonl").open("w") as stream:
        for record in run_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_tables(summary, output_dir)
    summary["figures"] = (
        plot_summary(summary, output_dir, formats) if make_plots else []
    )
    (output_dir / "detailed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        summary = analyze(
            args.root,
            args.output_dir,
            expected_subset_seeds=args.expected_subset_seeds,
            expected_model_seeds=args.expected_model_seeds,
            formats=args.formats,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
