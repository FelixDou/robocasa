"""Analyze Subtask-SAFE discrimination with parent-aware uncertainty controls."""

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
        validate_seed_alignment,
        write_json,
    )
    from .subtask_safe_evaluation import (
        elapsed_hazard_scores,
        fit_elapsed_hazard,
        parent_id,
        validate_parent_groups,
    )
except ImportError:
    from calibrate_seen_tasks import (
        fit_task_normalization,
        load_score_records,
        truncated_scores,
        validate_seed_alignment,
        write_json,
    )
    from subtask_safe_evaluation import (
        elapsed_hazard_scores,
        fit_elapsed_hazard,
        parent_id,
        validate_parent_groups,
    )


DEFAULT_PREFIX_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128)


def _safe_metric(function, labels, scores):
    return (
        None
        if len(labels) < 2 or len(set(labels)) < 2
        else float(function(labels, scores))
    )


def _metrics(records, scores):
    labels = [int(record["failed"]) for record in records]
    return {
        "segments": len(records),
        "successes": labels.count(0),
        "failures": labels.count(1),
        "roc_auc": _safe_metric(roc_auc_score, labels, scores),
        "average_precision": _safe_metric(average_precision_score, labels, scores),
    }


def _normalized_trajectory(record, normalization):
    raw = truncated_scores(record)
    stats = normalization[record["task_name"]]
    return (raw - stats["location"]) / stats["scale"]


def _score_bundle(records, normalization, elapsed_model):
    bundle = {}
    for record in records:
        raw = truncated_scores(record)
        normalized = _normalized_trajectory(record, normalization)
        elapsed = elapsed_hazard_scores(
            record,
            elapsed_model,
            lambda value: len(truncated_scores(value)),
        )
        bundle[record["rollout_id"]] = {
            "raw": raw,
            "normalized": normalized,
            "elapsed": elapsed,
        }
    return bundle


def _metric_block(records, bundle, source):
    scores = [float(np.max(bundle[record["rollout_id"]][source])) for record in records]
    pooled = _metrics(records, scores)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task_name"]].append(record)
    per_task = {}
    for task, values in sorted(grouped.items()):
        task_scores = [
            float(np.max(bundle[value["rollout_id"]][source])) for value in values
        ]
        per_task[task] = _metrics(values, task_scores)
    estimable = [
        values for values in per_task.values() if values["roc_auc"] is not None
    ]
    pooled["estimable_tasks"] = len(estimable)
    pooled["macro_task_roc_auc"] = (
        float(np.mean([values["roc_auc"] for values in estimable]))
        if estimable
        else None
    )
    pooled["macro_task_average_precision"] = (
        float(np.mean([values["average_precision"] for values in estimable]))
        if estimable
        else None
    )
    return {"overall": pooled, "per_task": per_task}


def _prefix_metrics(records, bundle, horizons):
    rows = []
    for horizon in horizons:
        active = [
            record
            for record in records
            if len(bundle[record["rollout_id"]]["normalized"]) >= horizon
        ]
        labels = [int(record["failed"]) for record in active]
        for source in ("normalized", "elapsed"):
            scores = [
                float(np.max(bundle[record["rollout_id"]][source][:horizon]))
                for record in active
            ]
            rows.append(
                {
                    "method": "safe" if source == "normalized" else "elapsed_time",
                    "horizon_inferences": int(horizon),
                    "active_segments": len(active),
                    "active_parents": len({parent_id(record) for record in active}),
                    "successes": labels.count(0),
                    "failures": labels.count(1),
                    "roc_auc": _safe_metric(roc_auc_score, labels, scores),
                    "average_precision": _safe_metric(
                        average_precision_score, labels, scores
                    ),
                }
            )
    return rows


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        return path
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _aggregate_seed_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, values in sorted(grouped.items()):
        record = {key: value for key, value in zip(keys, identity)}
        for key in (
            "roc_auc",
            "average_precision",
            "macro_task_roc_auc",
            "macro_task_average_precision",
        ):
            numbers = [value.get(key) for value in values if value.get(key) is not None]
            record[f"{key}_mean"] = float(np.mean(numbers)) if numbers else None
            record[f"{key}_std"] = float(np.std(numbers)) if numbers else None
        for key in (
            "segments",
            "successes",
            "failures",
            "active_segments",
            "active_parents",
        ):
            numbers = [value.get(key) for value in values if value.get(key) is not None]
            if numbers:
                record[key] = int(numbers[0])
        output.append(record)
    return output


def _parent_bootstrap(seed_payloads, replicates, seed):
    rng = np.random.RandomState(seed)
    model_seeds = sorted(seed_payloads)
    values = defaultdict(list)
    for _ in range(replicates):
        model_seed = model_seeds[int(rng.randint(len(model_seeds)))]
        payload = seed_payloads[model_seed]
        parents = payload["parents"]
        chosen = rng.choice(parents, size=len(parents), replace=True)
        sampled = []
        for key in chosen:
            sampled.extend(payload["records_by_parent"][str(key)])
        for source in ("normalized", "elapsed"):
            block = _metric_block(sampled, payload["bundle"], source)["overall"]
            method = "safe" if source == "normalized" else "elapsed_time"
            for metric in (
                "roc_auc",
                "average_precision",
                "macro_task_roc_auc",
                "macro_task_average_precision",
            ):
                if block[metric] is not None:
                    values[(method, metric)].append(block[metric])
    summary = {}
    for (method, metric), numbers in sorted(values.items()):
        array = np.asarray(numbers, dtype=np.float64)
        summary.setdefault(method, {})[metric] = {
            "replicates": int(len(array)),
            "mean": float(np.mean(array)),
            "ci_95_low": float(np.percentile(array, 2.5)),
            "ci_95_high": float(np.percentile(array, 97.5)),
            "sampling_unit": "parent_rollout",
            "model_seed_resampled": True,
        }
    return summary


def _create_plots(summary, output_dir, formats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "grid.color": "#D1D5DB",
            "text.color": "#1F2937",
        }
    )
    output_dir = Path(output_dir)
    paths = []

    per_task = [
        row
        for row in summary["per_subtask"]
        if row["model"] == "indep" and row["roc_auc_mean"] is not None
    ]
    per_task.sort(key=lambda row: row["roc_auc_mean"])
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.42 * len(per_task))))
    y = np.arange(len(per_task))
    mean = np.asarray([row["roc_auc_mean"] for row in per_task])
    std = np.asarray([row["roc_auc_std"] for row in per_task])
    ax.errorbar(mean, y, xerr=std, fmt="o", color="#2563A6", capsize=3)
    ax.axvline(0.5, color="#4B5563", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{row['task_name']}  ({row['successes']}S/{row['failures']}F)"
            for row in per_task
        ]
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Held-out ROC-AUC")
    ax.set_title("Subtask-SAFE MLP performance by semantic subtask")
    ax.grid(axis="x", alpha=0.6)
    fig.tight_layout()
    for extension in formats:
        path = output_dir / f"per_subtask_roc.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None)
        paths.append(path)
    plt.close(fig)

    prefix = [row for row in summary["prefix_metrics"] if row["model"] == "indep"]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True)
    for method, label, color, marker in (
        ("safe", "Subtask-SAFE MLP", "#2563A6", "o"),
        ("elapsed_time", "Elapsed-time hazard", "#D97706", "s"),
    ):
        rows = [row for row in prefix if row["method"] == method]
        x = [row["horizon_inferences"] for row in rows]
        y = [
            np.nan if row["roc_auc_mean"] is None else row["roc_auc_mean"]
            for row in rows
        ]
        axes[0].plot(x, y, marker=marker, color=color, label=label)
    axes[0].axhline(0.5, color="#4B5563", linestyle="--", linewidth=1)
    axes[0].set_ylabel("ROC-AUC among active segments")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.55)
    axes[0].legend(frameon=False)
    support_rows = [row for row in prefix if row["method"] == "safe"]
    axes[1].plot(
        [row["horizon_inferences"] for row in support_rows],
        [row["active_segments"] for row in support_rows],
        color="#374151",
        marker="o",
    )
    axes[1].set_ylabel("Active held-out segments")
    axes[1].set_xlabel("Observed policy inferences within current subtask")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(
        [row["horizon_inferences"] for row in support_rows],
        labels=[str(row["horizon_inferences"]) for row in support_rows],
    )
    axes[1].grid(alpha=0.55)
    fig.suptitle("Causal prefix discrimination versus elapsed time", fontsize=13)
    fig.tight_layout()
    for extension in formats:
        path = output_dir / f"causal_prefix_roc.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None)
        paths.append(path)
    plt.close(fig)

    bootstrap_rows = []
    for model in summary["models"]:
        for method in ("safe", "elapsed_time"):
            values = summary["parent_bootstrap"][model][method]["roc_auc"]
            bootstrap_rows.append(
                {
                    "label": f"{model.upper()} | " + (
                        "Subtask-SAFE" if method == "safe" else "elapsed hazard"
                    ),
                    **values,
                }
            )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(bootstrap_rows))
    means = np.asarray([row["mean"] for row in bootstrap_rows])
    low = np.asarray([row["ci_95_low"] for row in bootstrap_rows])
    high = np.asarray([row["ci_95_high"] for row in bootstrap_rows])
    colors = ["#2563A6" if "SAFE" in row["label"] else "#D97706" for row in bootstrap_rows]
    for index, (mean, lower, upper, color) in enumerate(
        zip(means, low, high, colors)
    ):
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color=color,
            capsize=4,
        )
    ax.axvline(0.5, color="#4B5563", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in bootstrap_rows])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Held-out ROC-AUC with 95% parent-bootstrap interval")
    ax.set_title("Subtask-SAFE and elapsed-time baseline uncertainty")
    ax.grid(axis="x", alpha=0.55)
    fig.tight_layout()
    for extension in formats:
        path = output_dir / f"parent_bootstrap_roc.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None)
        paths.append(path)
    plt.close(fig)

    support = [row for row in summary["per_subtask"] if row["model"] == "indep"]
    support.sort(key=lambda row: row["task_name"])
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.42 * len(support))))
    y = np.arange(len(support))
    successes = np.asarray([row["successes"] for row in support])
    failures = np.asarray([row["failures"] for row in support])
    ax.barh(y, successes, color="#2563A6", label="Success")
    ax.barh(y, failures, left=successes, color="#D97706", label="Failure")
    ax.set_yticks(y)
    ax.set_yticklabels([row["task_name"] for row in support])
    ax.set_xlabel("Held-out semantic segments")
    ax.set_title("Held-out Subtask-SAFE label support")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.55)
    fig.tight_layout()
    for extension in formats:
        path = output_dir / f"per_subtask_support.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def analyze_subtask_results(
    final_root,
    output_dir,
    *,
    models=("indep", "lstm"),
    seeds=(0, 1, 2),
    prefix_horizons=DEFAULT_PREFIX_HORIZONS,
    bootstrap_replicates=2000,
    bootstrap_seed=0,
    formats=("png", "pdf"),
):
    final_root = Path(final_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    prefix_rows = []
    per_task_rows = []
    bootstrap = {}
    support_reference = None
    for model in models:
        records_by_seed = {
            int(seed): load_score_records(
                final_root / f"{model}_seed{seed}" / "scores.jsonl"
            )
            for seed in seeds
        }
        validate_seed_alignment(records_by_seed)
        seed_payloads = {}
        for seed, records in records_by_seed.items():
            validate_parent_groups(records)
            normalization = fit_task_normalization(records)
            elapsed_model = fit_elapsed_hazard(
                records,
                lambda record: len(truncated_scores(record)),
            )
            bundle = _score_bundle(records, normalization, elapsed_model)
            test = [record for record in records if record["split"] == "test"]
            records_by_parent = defaultdict(list)
            for record in test:
                records_by_parent[parent_id(record)].append(record)
            seed_payloads[seed] = {
                "bundle": bundle,
                "parents": sorted(records_by_parent),
                "records_by_parent": dict(records_by_parent),
            }
            for source in ("raw", "normalized", "elapsed"):
                block = _metric_block(test, bundle, source)
                method = {
                    "raw": "safe_raw",
                    "normalized": "safe_task_z",
                    "elapsed": "elapsed_time",
                }[source]
                seed_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "method": method,
                        **block["overall"],
                    }
                )
                if source == "normalized":
                    for task, values in block["per_task"].items():
                        per_task_rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "task_name": task,
                                **values,
                            }
                        )
            for row in _prefix_metrics(test, bundle, prefix_horizons):
                prefix_rows.append({"model": model, "seed": seed, **row})
        bootstrap[model] = _parent_bootstrap(
            seed_payloads,
            int(bootstrap_replicates),
            int(bootstrap_seed),
        )

        support = {
            row["task_name"]: (row["successes"], row["failures"])
            for row in per_task_rows
            if row["model"] == model and row["seed"] == int(seeds[0])
        }
        if support_reference is None:
            support_reference = support
        elif support != support_reference:
            raise ValueError("Models do not use identical held-out subtask support")

    aggregate = _aggregate_seed_rows(seed_rows, ("model", "method"))
    prefix = _aggregate_seed_rows(
        prefix_rows,
        ("model", "method", "horizon_inferences"),
    )
    per_subtask = _aggregate_seed_rows(
        per_task_rows,
        ("model", "task_name"),
    )
    summary = {
        "schema_version": 1,
        "protocol": (
            "training-only task normalization; causal active-subtask prefixes; "
            "training-only elapsed hazard; parent-rollout hierarchical bootstrap"
        ),
        "final_root": str(final_root),
        "models": list(models),
        "seeds": list(seeds),
        "prefix_horizons": list(prefix_horizons),
        "bootstrap_replicates": int(bootstrap_replicates),
        "aggregate": aggregate,
        "per_subtask": per_subtask,
        "prefix_metrics": prefix,
        "parent_bootstrap": bootstrap,
        "notes": [
            "Prefix metrics include only semantic segments still active at the stated inference horizon.",
            "Elapsed-time hazard is fitted from training labels and training segment survival only.",
            "Per-subtask ROC/AP are null when held-out support contains one class.",
            "Bootstrap replicates resample parent rollouts and model seed, preserving within-parent segment dependence.",
        ],
    }
    _write_csv(output_dir / "per_seed_metrics.csv", seed_rows)
    _write_csv(output_dir / "causal_prefix_metrics.csv", prefix_rows)
    _write_csv(output_dir / "per_subtask_metrics.csv", per_task_rows)
    summary["plots"] = [
        str(path) for path in _create_plots(summary, output_dir, tuple(formats))
    ]
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["indep", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--prefix-horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_PREFIX_HORIZONS),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = analyze_subtask_results(
        args.final_root,
        args.output_dir,
        models=args.models,
        seeds=args.seeds,
        prefix_horizons=args.prefix_horizons,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        formats=args.formats,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
