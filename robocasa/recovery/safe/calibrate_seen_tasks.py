"""Calibrate and evaluate task-normalized SAFE scores on held-out seen tasks.

This stage consumes the score trajectories written by ``train_seen_tasks.py``.
It never retrains a model. Task normalization is fitted from training records
only, held-out successful test records are deterministically partitioned for
functional conformal calibration, and every remaining held-out record is used
for threshold evaluation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

import numpy as np

try:
    from .conformal import calibrate_functional_threshold, first_detection
    from .evaluate import evaluate_groups
    from .subtask_safe_evaluation import (
        detection_event,
        fit_elapsed_hazard,
        parent_grouped_calibration_split,
        parent_id,
        replace_with_elapsed_scores,
    )
except ImportError:
    from conformal import calibrate_functional_threshold, first_detection
    from evaluate import evaluate_groups
    from subtask_safe_evaluation import (
        detection_event,
        fit_elapsed_hazard,
        parent_grouped_calibration_split,
        parent_id,
        replace_with_elapsed_scores,
    )


NORMALIZATION_EPS = 1e-8
DEFAULT_ALPHAS = (0.05, 0.10, 0.15, 0.20)
TASK_TYPE_FILTERS = ("all", "atomic", "composite")


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, sort_keys=True) + "\n")
    return path


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for value in values:
            stream.write(json.dumps(json_value(value), sort_keys=True) + "\n")
    return path


def load_score_records(path):
    records = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No score records found in {path}")
    rollout_ids = [record["rollout_id"] for record in records]
    if len(rollout_ids) != len(set(rollout_ids)):
        raise ValueError(f"Score file contains duplicate rollout IDs: {path}")
    return records


def truncated_scores(record):
    scores = np.asarray(record["scores"], dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError(
            f"Rollout {record.get('rollout_id')} has an invalid score trajectory"
        )
    cutoff = min(int(record["task_min_step"]), len(scores))
    if cutoff < 1:
        raise ValueError(
            f"Rollout {record.get('rollout_id')} has an invalid task_min_step"
        )
    return scores[:cutoff]


def record_signature(record):
    return (
        record["split"],
        record["task_name"],
        bool(record["failed"]),
        int(record["task_min_step"]),
        record.get("parent_rollout_id"),
        record.get("parent_task_name"),
        record.get("parent_rollout_failed"),
    )


def validate_seed_alignment(records_by_seed):
    seeds = sorted(records_by_seed)
    reference = {
        record["rollout_id"]: record_signature(record)
        for record in records_by_seed[seeds[0]]
    }
    if set(split for split, *_ in reference.values()) != {"train", "test"}:
        raise ValueError("Final score files must contain exactly train and test splits")
    for seed in seeds[1:]:
        current = {
            record["rollout_id"]: record_signature(record)
            for record in records_by_seed[seed]
        }
        if current != reference:
            raise ValueError(
                f"Model seed {seed} does not use the same rollout identities and split"
            )
    return reference


def load_final_task_types(final_root, model, seeds, *, required=False):
    task_type_maps = []
    for seed in seeds:
        path = Path(final_root) / f"{model}_seed{seed}" / "metrics.json"
        if not path.is_file():
            if required:
                raise ValueError(
                    f"Task-type filtering requires final metrics: {path}"
                )
            continue
        task_types = json.loads(path.read_text()).get("task_types", {})
        if not task_types:
            if required:
                raise ValueError(f"Final metrics contain no task_types: {path}")
            continue
        task_type_maps.append(task_types)
    if not task_type_maps:
        return {}
    reference = task_type_maps[0]
    if any(task_types != reference for task_types in task_type_maps[1:]):
        raise ValueError("Final model seeds disagree on task-type provenance")
    invalid = {
        task: task_type
        for task, task_type in reference.items()
        if task_type not in TASK_TYPE_FILTERS[1:]
    }
    if invalid:
        raise ValueError(f"Final metrics have invalid task types: {invalid}")
    return reference


def filter_score_task_type(records_by_seed, task_types, task_type):
    if task_type not in TASK_TYPE_FILTERS:
        raise ValueError(
            f"Unknown task type {task_type!r}; expected one of {TASK_TYPE_FILTERS}"
        )
    if task_type == "all":
        selected_names = sorted(
            {
                record["task_name"]
                for records in records_by_seed.values()
                for record in records
            }
        )
        return records_by_seed, selected_names
    selected_names = sorted(
        task for task, value in task_types.items() if value == task_type
    )
    if not selected_names:
        raise ValueError(f"No {task_type} tasks are present in final metrics")
    selected = set(selected_names)
    filtered = {
        seed: [
            record for record in records if record["task_name"] in selected
        ]
        for seed, records in records_by_seed.items()
    }
    if any(not records for records in filtered.values()):
        raise ValueError(f"Task-type filter {task_type} selected no score records")
    return filtered, selected_names


def deterministic_calibration_split(
    records,
    *,
    successes_per_task,
    split_seed,
    reference_fraction,
    conformal_seed,
):
    test = [record for record in records if record["split"] == "test"]
    train = [record for record in records if record["split"] == "train"]
    grouped_successes = defaultdict(list)
    grouped_failures = defaultdict(list)
    for record in test:
        target = grouped_failures if record["failed"] else grouped_successes
        target[record["task_name"]].append(record["rollout_id"])
    tasks = sorted({record["task_name"] for record in records})
    calibration_ids = []
    per_task = {}
    for task in tasks:
        successes = sorted(grouped_successes[task])
        failures = sorted(grouped_failures[task])
        if len(successes) <= successes_per_task:
            raise ValueError(
                f"Task {task} has {len(successes)} held-out successes; "
                f"need more than {successes_per_task}"
            )
        if not failures:
            raise ValueError(f"Task {task} has no held-out failures")
        rng = random.Random(f"{split_seed}:{task}")
        rng.shuffle(successes)
        selected = successes[:successes_per_task]
        calibration_ids.extend(selected)
        per_task[task] = {
            "train_successes": sum(
                not record["failed"]
                for record in train
                if record["task_name"] == task
            ),
            "train_failures": sum(
                record["failed"]
                for record in train
                if record["task_name"] == task
            ),
            "calibration_successes": len(selected),
            "evaluation_successes": len(successes) - len(selected),
            "evaluation_failures": len(failures),
            "calibration_success_ids": sorted(selected),
        }
    calibration_ids = sorted(calibration_ids)
    order = np.random.RandomState(conformal_seed).permutation(len(calibration_ids))
    ordered = [calibration_ids[index] for index in order]
    reference_size = int(len(ordered) * reference_fraction)
    if reference_size < 1 or reference_size >= len(ordered):
        raise ValueError(
            "Reference fraction must produce non-empty reference and conformal subsets"
        )
    reference_ids = ordered[:reference_size]
    conformal_ids = ordered[reference_size:]
    calibration_set = set(calibration_ids)
    evaluation_ids = sorted(
        record["rollout_id"]
        for record in test
        if record["rollout_id"] not in calibration_set
    )
    evaluation_set = set(evaluation_ids)
    train_ids = sorted(record["rollout_id"] for record in train)
    if set(train_ids) & calibration_set or set(train_ids) & evaluation_set:
        raise AssertionError("Training identities overlap a held-out split")
    if calibration_set & evaluation_set:
        raise AssertionError("Calibration and evaluation identities overlap")
    if calibration_set | evaluation_set != {
        record["rollout_id"] for record in test
    }:
        raise AssertionError("Calibration and evaluation are not exhaustive")
    return {
        "schema_version": 1,
        "protocol": (
            "training-only task early-score normalization; deterministic held-out "
            "success calibration; remaining held-out evaluation"
        ),
        "split_seed": int(split_seed),
        "conformal_seed": int(conformal_seed),
        "calibration_successes_per_task": int(successes_per_task),
        "official_reference_fraction": float(reference_fraction),
        "task_names": tasks,
        "counts": {
            "train": len(train_ids),
            "calibration_successes": len(calibration_ids),
            "calibration_reference_successes": len(reference_ids),
            "calibration_nonconformity_successes": len(conformal_ids),
            "evaluation": len(evaluation_ids),
            "evaluation_successes": sum(
                not record["failed"]
                for record in test
                if record["rollout_id"] in evaluation_set
            ),
            "evaluation_failures": sum(
                record["failed"]
                for record in test
                if record["rollout_id"] in evaluation_set
            ),
        },
        "per_task": per_task,
        "train_ids": train_ids,
        "calibration_success_ids": calibration_ids,
        "calibration_reference_ids": reference_ids,
        "calibration_nonconformity_ids": conformal_ids,
        "evaluation_ids": evaluation_ids,
    }


def fit_task_normalization(records):
    grouped = defaultdict(list)
    for record in records:
        if record["split"] != "train":
            continue
        grouped[record["task_name"]].append(float(np.max(truncated_scores(record))))
    if not grouped:
        raise ValueError("No training records are available for task normalization")
    result = {}
    for task, values in sorted(grouped.items()):
        values = np.asarray(values, dtype=np.float64)
        scale = float(np.std(values))
        degenerate = scale < NORMALIZATION_EPS
        result[task] = {
            "location": float(np.mean(values)),
            "scale": 1.0 if degenerate else scale,
            "raw_scale": scale,
            "degenerate_scale_fallback": degenerate,
            "num_training_rollouts": int(len(values)),
            "source_statistic": (
                "maximum SAFE score before the training-derived task_min_step"
            ),
        }
    return result


def normalize_records(
    records,
    normalization,
    split_manifest,
    task_types=None,
):
    calibration_ids = set(split_manifest["calibration_success_ids"])
    calibration_excluded_ids = set(
        split_manifest.get("calibration_failure_ids_excluded", [])
    )
    evaluation_ids = set(split_manifest["evaluation_ids"])
    normalized = []
    for record in records:
        full_raw = np.asarray(record["scores"], dtype=np.float64)
        if (
            full_raw.ndim != 1
            or not len(full_raw)
            or not np.all(np.isfinite(full_raw))
        ):
            raise ValueError(
                f"Rollout {record.get('rollout_id')} has an invalid full score trajectory"
            )
        raw = truncated_scores(record)
        stats = normalization[record["task_name"]]
        scores = (raw - stats["location"]) / stats["scale"]
        full_scores = (full_raw - stats["location"]) / stats["scale"]
        if record["split"] == "train":
            split = "train"
        elif record["rollout_id"] in calibration_ids:
            split = "calibration"
        elif record["rollout_id"] in calibration_excluded_ids:
            split = "calibration_excluded"
        elif record["rollout_id"] in evaluation_ids:
            split = "evaluation"
        else:
            raise AssertionError(
                f"Rollout {record['rollout_id']} is absent from the calibration manifest"
            )
        inference_steps = record.get("inference_environment_steps")
        full_inference_steps = inference_steps
        if inference_steps is not None:
            inference_steps = inference_steps[: len(scores)]
        normalized.append(
            {
                **record,
                "task_type": (
                    task_types.get(record["task_name"])
                    if task_types
                    else record.get("task_type")
                ),
                "original_split": record["split"],
                "split": split,
                "raw_scores": raw.tolist(),
                "scores": scores.tolist(),
                "full_raw_scores": full_raw.tolist(),
                "full_scores": full_scores.tolist(),
                "num_inferences": int(len(scores)),
                "full_num_inferences": int(len(full_scores)),
                "inference_environment_steps": inference_steps,
                "full_inference_environment_steps": full_inference_steps,
                "normalization": "training_task_early_max_z",
                "normalization_location": stats["location"],
                "normalization_scale": stats["scale"],
                "raw_early_score": float(np.max(raw)),
                "normalized_early_score": float(np.max(scores)),
            }
        )
    return normalized


def _count_records(records):
    return {
        "rollouts": len(records),
        "successes": sum(not record["failed"] for record in records),
        "failures": sum(record["failed"] for record in records),
    }


def _lead_time_summary(records, calibration):
    events = []
    for record in records:
        if not record["failed"]:
            continue
        index = first_detection(record["scores"], calibration)
        events.append(
            {
                "rollout_id": record["rollout_id"],
                "parent_rollout_id": parent_id(record),
                "parent_task_name": record.get("parent_task_name"),
                "task_name": record["task_name"],
                "subtask_id": record.get("subtask_id"),
                "detection_index": index,
                **detection_event(record, index),
            }
        )
    detected = [event for event in events if event["detected"]]
    lead = [
        event["lead_environment_steps"]
        for event in detected
        if event["lead_environment_steps"] is not None
    ]
    normalized = [
        event["normalized_lead_time"]
        for event in detected
        if event["normalized_lead_time"] is not None
    ]
    return {
        "num_failures": len(events),
        "num_detected_failures": len(detected),
        "mean_lead_environment_steps": float(np.mean(lead)) if lead else None,
        "median_lead_environment_steps": float(np.median(lead)) if lead else None,
        "mean_normalized_lead_time": (
            float(np.mean(normalized)) if normalized else None
        ),
        "events": events,
    }


def _flatten_overall(seed, alpha, metrics, *, method, lead_time):
    overall = metrics["overall"]
    return {
        "method": method,
        "seed": seed,
        "alpha": alpha,
        "num_rollouts": overall["num_rollouts"],
        "num_successes": overall["num_successes"],
        "num_failures": overall["num_failures"],
        "roc_auc": overall["roc_auc"],
        "auprc": overall["auprc"],
        "true_positive_rate": overall["true_positive_rate"],
        "false_positive_rate": overall["false_positive_rate"],
        "true_negative_rate": overall["true_negative_rate"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "normalized_detection_time": overall["normalized_detection_time"],
        "mean_lead_environment_steps": lead_time["mean_lead_environment_steps"],
        "median_lead_environment_steps": lead_time["median_lead_environment_steps"],
        "mean_normalized_lead_time": lead_time["mean_normalized_lead_time"],
        **{
            f"confusion_{key}": value
            for key, value in overall["confusion"].items()
        },
    }


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _aggregate_method_metrics(overall_rows, task_rows, selected_alpha):
    task_metrics = (
        "roc_auc",
        "auprc",
        "true_positive_rate",
        "false_positive_rate",
        "true_negative_rate",
        "balanced_accuracy",
        "normalized_detection_time",
    )
    overall_metrics = task_metrics + (
        "mean_lead_environment_steps",
        "median_lead_environment_steps",
        "mean_normalized_lead_time",
    )
    by_alpha = {}
    for alpha in sorted({row["alpha"] for row in overall_rows}):
        selected = [row for row in overall_rows if row["alpha"] == alpha]
        summary = {"seeds": sorted(row["seed"] for row in selected)}
        for metric in overall_metrics:
            values = np.asarray(
                [row[metric] for row in selected if row[metric] is not None],
                dtype=np.float64,
            )
            summary[f"{metric}_mean"] = (
                float(np.mean(values)) if len(values) else None
            )
            summary[f"{metric}_std"] = (
                float(np.std(values)) if len(values) else None
            )
        by_alpha[f"{alpha:g}"] = summary
    per_task = {}
    selected_rows = [
        row for row in task_rows if np.isclose(row["alpha"], selected_alpha)
    ]
    for task in sorted({row["task_name"] for row in selected_rows}):
        values = [row for row in selected_rows if row["task_name"] == task]
        summary = {}
        for metric in task_metrics:
            array = np.asarray(
                [row[metric] for row in values if row[metric] is not None],
                dtype=np.float64,
            )
            summary[f"{metric}_mean"] = (
                float(np.mean(array)) if len(array) else None
            )
            summary[f"{metric}_std"] = (
                float(np.std(array)) if len(array) else None
            )
        per_task[task] = summary
    return {
        "selected_alpha": float(selected_alpha),
        "by_alpha": by_alpha,
        "per_task_at_selected_alpha": per_task,
    }


def aggregate_metrics(overall_rows, task_rows, selected_alpha):
    methods = {}
    for method in sorted({row["method"] for row in overall_rows}):
        methods[method] = _aggregate_method_metrics(
            [row for row in overall_rows if row["method"] == method],
            [row for row in task_rows if row["method"] == method],
            selected_alpha,
        )
    safe = methods["safe"]
    return {
        **safe,
        "methods": methods,
    }


def create_plots(summary, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    by_alpha = summary["aggregate"]["by_alpha"]
    methods = summary["aggregate"].get("methods", {})
    alphas = np.asarray(sorted(float(alpha) for alpha in by_alpha))
    paths = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for metric, label, color in (
        ("true_positive_rate", "TPR", "#D55E00"),
        ("false_positive_rate", "FPR", "#0072B2"),
        ("balanced_accuracy", "Balanced accuracy", "#374151"),
    ):
        mean = np.asarray(
            [by_alpha[f"{alpha:g}"][f"{metric}_mean"] for alpha in alphas]
        )
        std = np.asarray(
            [by_alpha[f"{alpha:g}"][f"{metric}_std"] for alpha in alphas]
        )
        ax.plot(alphas, mean, marker="o", label=label, color=color)
        ax.fill_between(alphas, mean - std, mean + std, color=color, alpha=0.15)
    elapsed = methods.get("elapsed_time", {}).get("by_alpha", {})
    if elapsed:
        mean = np.asarray(
            [elapsed[f"{alpha:g}"]["balanced_accuracy_mean"] for alpha in alphas]
        )
        std = np.asarray(
            [elapsed[f"{alpha:g}"]["balanced_accuracy_std"] for alpha in alphas]
        )
        ax.plot(
            alphas,
            mean,
            marker="s",
            linestyle="--",
            label="Elapsed-time balanced accuracy",
            color="#6B7280",
        )
        ax.fill_between(alphas, mean - std, mean + std, color="#6B7280", alpha=0.10)
    ax.set(
        xlabel="Conformal significance level (alpha)",
        ylabel="Rate",
        ylim=(0.0, 1.0),
        title=(
            "Task-normalized SAFE conformal trade-off "
            f"({summary['task_type_filter']})"
        ),
    )
    ax.grid(alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "conformal_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    per_task = summary["aggregate"]["per_task_at_selected_alpha"]
    elapsed_per_task = methods.get("elapsed_time", {}).get(
        "per_task_at_selected_alpha", {}
    )
    tasks = sorted(per_task)
    means = [
        np.nan
        if per_task[task]["balanced_accuracy_mean"] is None
        else per_task[task]["balanced_accuracy_mean"]
        for task in tasks
    ]
    stds = [
        0.0
        if per_task[task]["balanced_accuracy_std"] is None
        else per_task[task]["balanced_accuracy_std"]
        for task in tasks
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    positions = np.arange(len(tasks))
    width = 0.38
    ax.bar(
        positions - width / 2,
        means,
        width,
        yerr=stds,
        capsize=3,
        color="#2563A6",
        label="Subtask-SAFE",
    )
    if elapsed_per_task:
        elapsed_means = [
            np.nan
            if elapsed_per_task.get(task, {}).get("balanced_accuracy_mean") is None
            else elapsed_per_task[task]["balanced_accuracy_mean"]
            for task in tasks
        ]
        elapsed_stds = [
            0.0
            if elapsed_per_task.get(task, {}).get("balanced_accuracy_std") is None
            else elapsed_per_task[task]["balanced_accuracy_std"]
            for task in tasks
        ]
        ax.bar(
            positions + width / 2,
            elapsed_means,
            width,
            yerr=elapsed_stds,
            capsize=3,
            color="#D97706",
            label="Elapsed-time hazard",
        )
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set(
        ylabel="Balanced accuracy",
        ylim=(0.0, 1.0),
        title=(
            f"{summary['task_type_filter'].title()} per-task performance at "
            f"alpha={summary['selected_alpha']:.2f}"
        ),
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(tasks, rotation=38, ha="right")
    ax.grid(axis="y", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "per_task_balanced_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def run_seen_calibration(
    final_root,
    output_dir,
    *,
    model="indep",
    seeds=(0, 1, 2),
    calibration_successes_per_task=3,
    calibration_parent_fraction=0.4,
    split_unit="auto",
    split_seed=0,
    conformal_seed=0,
    reference_fraction=0.3,
    alphas=DEFAULT_ALPHAS,
    selected_alpha=0.15,
    modulation="tfunc",
    task_type="all",
    make_plots=True,
):
    final_root = Path(final_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in seeds)
    alphas = tuple(float(alpha) for alpha in alphas)
    if not seeds:
        raise ValueError("At least one model seed is required")
    if calibration_successes_per_task < 1:
        raise ValueError("calibration_successes_per_task must be positive")
    if not alphas or any(not 0 < alpha < 1 for alpha in alphas):
        raise ValueError("Every alpha must lie in (0, 1)")
    matching_alphas = [
        alpha for alpha in alphas if np.isclose(selected_alpha, alpha)
    ]
    if not matching_alphas:
        raise ValueError("selected_alpha must be present in alphas")
    selected_alpha = matching_alphas[0]
    records_by_seed = {
        seed: load_score_records(
            final_root / f"{model}_seed{seed}" / "scores.jsonl"
        )
        for seed in seeds
    }
    task_types = load_final_task_types(
        final_root,
        model,
        seeds,
        required=task_type != "all",
    )
    records_by_seed, selected_task_names = filter_score_task_type(
        records_by_seed,
        task_types,
        task_type,
    )
    validate_seed_alignment(records_by_seed)
    reference_records = records_by_seed[seeds[0]]
    has_parent_ids = all(
        record.get("parent_rollout_id") for record in reference_records
    )
    if split_unit not in ("auto", "rollout", "parent_rollout"):
        raise ValueError(
            "split_unit must be one of auto, rollout, parent_rollout"
        )
    resolved_split_unit = (
        "parent_rollout"
        if split_unit == "parent_rollout" or (split_unit == "auto" and has_parent_ids)
        else "rollout"
    )
    if resolved_split_unit == "parent_rollout":
        if not has_parent_ids:
            raise ValueError(
                "Parent-rollout calibration requires parent_rollout_id on every score record"
            )
        split_manifest = parent_grouped_calibration_split(
            reference_records,
            parent_fraction=calibration_parent_fraction,
            split_seed=split_seed,
            reference_fraction=reference_fraction,
            conformal_seed=conformal_seed,
        )
    else:
        split_manifest = deterministic_calibration_split(
            reference_records,
            successes_per_task=calibration_successes_per_task,
            split_seed=split_seed,
            reference_fraction=reference_fraction,
            conformal_seed=conformal_seed,
        )
        split_manifest["split_unit"] = "rollout"
    split_manifest.update(
        {
            "model": model,
            "model_seeds": list(seeds),
            "task_type_filter": task_type,
            "selected_task_names": selected_task_names,
            "selected_task_types": {
                task: task_types[task]
                for task in selected_task_names
                if task in task_types
            },
            "source_final_root": str(final_root),
            "source_score_files": {
                str(seed): str(
                    final_root / f"{model}_seed{seed}" / "scores.jsonl"
                )
                for seed in seeds
            },
        }
    )
    write_json(output_dir / "split_manifest.json", split_manifest)
    reference_ids = split_manifest["calibration_reference_ids"]
    conformal_ids = split_manifest["calibration_nonconformity_ids"]
    overall_rows = []
    task_rows = []
    detection_rows = []
    seed_summaries = {}
    for seed in seeds:
        seed_root = output_dir / f"{model}_seed{seed}"
        normalization = fit_task_normalization(records_by_seed[seed])
        write_json(seed_root / "task_normalization.json", normalization)
        records = normalize_records(
            records_by_seed[seed],
            normalization,
            split_manifest,
            task_types,
        )
        write_jsonl(seed_root / "normalized_scores.jsonl", records)
        by_id = {record["rollout_id"]: record for record in records}
        reference = [by_id[rollout_id] for rollout_id in reference_ids]
        conformal = [by_id[rollout_id] for rollout_id in conformal_ids]
        evaluation = [
            record for record in records if record["split"] == "evaluation"
        ]
        elapsed_model = fit_elapsed_hazard(
            records,
            lambda record: len(truncated_scores(record)),
        )
        write_json(seed_root / "elapsed_hazard_model.json", elapsed_model)
        elapsed_records = replace_with_elapsed_scores(
            records,
            elapsed_model,
            lambda record: len(truncated_scores(record)),
        )
        elapsed_by_id = {
            record["rollout_id"]: record for record in elapsed_records
        }
        elapsed_reference = [elapsed_by_id[rollout_id] for rollout_id in reference_ids]
        elapsed_conformal = [elapsed_by_id[rollout_id] for rollout_id in conformal_ids]
        elapsed_evaluation = [
            record for record in elapsed_records if record["split"] == "evaluation"
        ]
        seed_summary = {
            "normalization": "training_task_early_max_z",
            "counts": {
                "reference": _count_records(reference),
                "conformal": _count_records(conformal),
                "evaluation": _count_records(evaluation),
            },
            "alphas": {},
        }
        for alpha in alphas:
            alpha_slug = str(alpha).replace(".", "p")
            alpha_root = seed_root / f"alpha_{alpha_slug}"
            calibration = calibrate_functional_threshold(
                [record["scores"] for record in reference],
                [record["scores"] for record in conformal],
                alpha=alpha,
                modulation=modulation,
                alignment="extend",
            )
            calibration.update(
                {
                    "protocol": "official_safe_seen_task_normalized",
                    "model": model,
                    "model_seed": seed,
                    "task_type_filter": task_type,
                    "task_normalization": "training_task_early_max_z",
                    "split_manifest": str(
                        (output_dir / "split_manifest.json").resolve()
                    ),
                    "reference_rollout_ids": reference_ids,
                    "calibration_rollout_ids": conformal_ids,
                }
            )
            write_json(alpha_root / "calibration.json", calibration)
            metrics = evaluate_groups(evaluation, calibration)
            lead_time = _lead_time_summary(evaluation, calibration)
            elapsed_calibration = calibrate_functional_threshold(
                [record["scores"] for record in elapsed_reference],
                [record["scores"] for record in elapsed_conformal],
                alpha=alpha,
                modulation=modulation,
                alignment="extend",
            )
            elapsed_calibration.update(
                {
                    "protocol": "training_elapsed_subtask_hazard",
                    "model": model,
                    "model_seed": seed,
                    "task_type_filter": task_type,
                    "split_manifest": str(
                        (output_dir / "split_manifest.json").resolve()
                    ),
                    "reference_rollout_ids": reference_ids,
                    "calibration_rollout_ids": conformal_ids,
                }
            )
            write_json(
                alpha_root / "elapsed_time_calibration.json",
                elapsed_calibration,
            )
            elapsed_metrics = evaluate_groups(
                elapsed_evaluation,
                elapsed_calibration,
            )
            elapsed_lead_time = _lead_time_summary(
                elapsed_evaluation,
                elapsed_calibration,
            )
            result = {
                "schema_version": 1,
                "model": model,
                "model_seed": seed,
                "task_type_filter": task_type,
                "alpha": alpha,
                "normalization": "training_task_early_max_z",
                "split_manifest": str(
                    (output_dir / "split_manifest.json").resolve()
                ),
                "calibration": str(
                    (alpha_root / "calibration.json").resolve()
                ),
                **metrics,
                "lead_time": {
                    key: value
                    for key, value in lead_time.items()
                    if key != "events"
                },
                "elapsed_time_baseline": {
                    "calibration": str(
                        (alpha_root / "elapsed_time_calibration.json").resolve()
                    ),
                    "lead_time": {
                        key: value
                        for key, value in elapsed_lead_time.items()
                        if key != "events"
                    },
                    **elapsed_metrics,
                },
            }
            write_json(alpha_root / "metrics.json", result)
            for method, method_metrics, method_lead in (
                ("safe", metrics, lead_time),
                ("elapsed_time", elapsed_metrics, elapsed_lead_time),
            ):
                overall_rows.append(
                    _flatten_overall(
                        seed,
                        alpha,
                        method_metrics,
                        method=method,
                        lead_time=method_lead,
                    )
                )
                for event in method_lead["events"]:
                    detection_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "alpha": alpha,
                            **event,
                        }
                    )
                for task, values in method_metrics["per_task"].items():
                    task_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "alpha": alpha,
                            "task_name": task,
                            **{
                                key: value
                                for key, value in values.items()
                                if key != "confusion"
                            },
                            **{
                                f"confusion_{key}": value
                                for key, value in values["confusion"].items()
                            },
                        }
                    )
            seed_summary["alphas"][f"{alpha:g}"] = {
                "calibration": str(alpha_root / "calibration.json"),
                "metrics": str(alpha_root / "metrics.json"),
                "overall": metrics["overall"],
                "lead_time": {
                    key: value for key, value in lead_time.items() if key != "events"
                },
                "elapsed_time_baseline": {
                    "calibration": str(alpha_root / "elapsed_time_calibration.json"),
                    "overall": elapsed_metrics["overall"],
                    "lead_time": {
                        key: value
                        for key, value in elapsed_lead_time.items()
                        if key != "events"
                    },
                },
            }
        seed_summaries[str(seed)] = seed_summary
    aggregate = aggregate_metrics(overall_rows, task_rows, selected_alpha)
    summary = {
        "schema_version": 1,
        "protocol": "task-normalized held-out seen-task functional conformal calibration",
        "split_unit": resolved_split_unit,
        "model": model,
        "model_seeds": list(seeds),
        "task_type_filter": task_type,
        "selected_task_names": selected_task_names,
        "selected_task_types": split_manifest["selected_task_types"],
        "normalization": "training_task_early_max_z",
        "score_cutoff": "training-derived task_min_step",
        "modulation": modulation,
        "alphas": list(alphas),
        "selected_alpha": float(selected_alpha),
        "split_manifest": str((output_dir / "split_manifest.json").resolve()),
        "split_counts": split_manifest["counts"],
        "seed_results": seed_summaries,
        "aggregate": aggregate,
        "notes": [
            "Task normalization uses training rollouts only.",
            "Calibration uses held-out successful rollouts only.",
            "Calibration and evaluation rollout IDs are disjoint.",
            "For Subtask-SAFE, complete parent rollouts are disjoint across calibration and evaluation.",
            "The elapsed-time baseline is fitted from training subtask survival only.",
            "This task-conditioned normalization protocol applies to known seen tasks only.",
            "The selected alpha is fixed before evaluation; other alphas are sensitivity analyses.",
            "Standard deviations are population standard deviations across model seeds.",
        ],
    }
    _write_csv(output_dir / "per_seed_alpha_metrics.csv", overall_rows)
    _write_csv(output_dir / "per_task_alpha_metrics.csv", task_rows)
    _write_csv(output_dir / "detection_events.csv", detection_rows)
    plot_paths = create_plots(summary, output_dir) if make_plots else []
    summary["plots"] = [str(path) for path in plot_paths]
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=("indep", "lstm"), default="indep")
    parser.add_argument(
        "--task-type",
        choices=TASK_TYPE_FILTERS,
        default="all",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--calibration-successes-per-task", type=int, default=3)
    parser.add_argument("--calibration-parent-fraction", type=float, default=0.4)
    parser.add_argument(
        "--split-unit",
        choices=("auto", "rollout", "parent_rollout"),
        default="auto",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--conformal-seed", type=int, default=0)
    parser.add_argument("--reference-fraction", type=float, default=0.3)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--selected-alpha", type=float, default=0.15)
    parser.add_argument(
        "--modulation",
        choices=("tfunc", "stdev", "constant"),
        default="tfunc",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = run_seen_calibration(
        args.final_root,
        args.output_dir,
        model=args.model,
        seeds=args.seeds,
        calibration_successes_per_task=args.calibration_successes_per_task,
        calibration_parent_fraction=args.calibration_parent_fraction,
        split_unit=args.split_unit,
        split_seed=args.split_seed,
        conformal_seed=args.conformal_seed,
        reference_fraction=args.reference_fraction,
        alphas=args.alphas,
        selected_alpha=args.selected_alpha,
        modulation=args.modulation,
        task_type=args.task_type,
        make_plots=not args.no_plots,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
