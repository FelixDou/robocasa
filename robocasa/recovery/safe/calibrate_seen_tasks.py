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
    from .conformal import calibrate_functional_threshold
    from .evaluate import evaluate_groups
except ImportError:
    from conformal import calibrate_functional_threshold
    from evaluate import evaluate_groups


NORMALIZATION_EPS = 1e-8
DEFAULT_ALPHAS = (0.05, 0.10, 0.15, 0.20)


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


def normalize_records(records, normalization, split_manifest):
    calibration_ids = set(split_manifest["calibration_success_ids"])
    evaluation_ids = set(split_manifest["evaluation_ids"])
    normalized = []
    for record in records:
        raw = truncated_scores(record)
        stats = normalization[record["task_name"]]
        scores = (raw - stats["location"]) / stats["scale"]
        if record["split"] == "train":
            split = "train"
        elif record["rollout_id"] in calibration_ids:
            split = "calibration"
        elif record["rollout_id"] in evaluation_ids:
            split = "evaluation"
        else:
            raise AssertionError(
                f"Rollout {record['rollout_id']} is absent from the calibration manifest"
            )
        inference_steps = record.get("inference_environment_steps")
        if inference_steps is not None:
            inference_steps = inference_steps[: len(scores)]
        normalized.append(
            {
                **record,
                "original_split": record["split"],
                "split": split,
                "raw_scores": raw.tolist(),
                "scores": scores.tolist(),
                "num_inferences": int(len(scores)),
                "inference_environment_steps": inference_steps,
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


def _flatten_overall(seed, alpha, metrics):
    overall = metrics["overall"]
    return {
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


def aggregate_metrics(overall_rows, task_rows, selected_alpha):
    metrics = (
        "roc_auc",
        "auprc",
        "true_positive_rate",
        "false_positive_rate",
        "true_negative_rate",
        "balanced_accuracy",
        "normalized_detection_time",
    )
    by_alpha = {}
    for alpha in sorted({row["alpha"] for row in overall_rows}):
        selected = [row for row in overall_rows if row["alpha"] == alpha]
        summary = {"seeds": sorted(row["seed"] for row in selected)}
        for metric in metrics:
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
        for metric in metrics:
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


def create_plots(summary, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    by_alpha = summary["aggregate"]["by_alpha"]
    alphas = np.asarray(sorted(float(alpha) for alpha in by_alpha))
    paths = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for metric, label, color in (
        ("true_positive_rate", "TPR", "#D55E00"),
        ("false_positive_rate", "FPR", "#0072B2"),
        ("balanced_accuracy", "Balanced accuracy", "#009E73"),
    ):
        mean = np.asarray(
            [by_alpha[f"{alpha:g}"][f"{metric}_mean"] for alpha in alphas]
        )
        std = np.asarray(
            [by_alpha[f"{alpha:g}"][f"{metric}_std"] for alpha in alphas]
        )
        ax.plot(alphas, mean, marker="o", label=label, color=color)
        ax.fill_between(alphas, mean - std, mean + std, color=color, alpha=0.15)
    ax.set(
        xlabel="Conformal significance level (alpha)",
        ylabel="Rate",
        ylim=(0.0, 1.0),
        title="Task-normalized SAFE conformal trade-off",
    )
    ax.grid(alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "conformal_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    per_task = summary["aggregate"]["per_task_at_selected_alpha"]
    tasks = sorted(per_task)
    means = [per_task[task]["balanced_accuracy_mean"] for task in tasks]
    stds = [per_task[task]["balanced_accuracy_std"] for task in tasks]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    positions = np.arange(len(tasks))
    ax.bar(positions, means, yerr=stds, capsize=3, color="#56B4E9")
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set(
        ylabel="Balanced accuracy",
        ylim=(0.0, 1.0),
        title=(
            "Per-task conformal performance at "
            f"alpha={summary['selected_alpha']:.2f}"
        ),
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(tasks, rotation=38, ha="right")
    ax.grid(axis="y", alpha=0.5)
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
    split_seed=0,
    conformal_seed=0,
    reference_fraction=0.3,
    alphas=DEFAULT_ALPHAS,
    selected_alpha=0.15,
    modulation="tfunc",
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
    validate_seed_alignment(records_by_seed)
    split_manifest = deterministic_calibration_split(
        records_by_seed[seeds[0]],
        successes_per_task=calibration_successes_per_task,
        split_seed=split_seed,
        reference_fraction=reference_fraction,
        conformal_seed=conformal_seed,
    )
    split_manifest.update(
        {
            "model": model,
            "model_seeds": list(seeds),
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
    seed_summaries = {}
    for seed in seeds:
        seed_root = output_dir / f"{model}_seed{seed}"
        normalization = fit_task_normalization(records_by_seed[seed])
        write_json(seed_root / "task_normalization.json", normalization)
        records = normalize_records(
            records_by_seed[seed], normalization, split_manifest
        )
        write_jsonl(seed_root / "normalized_scores.jsonl", records)
        by_id = {record["rollout_id"]: record for record in records}
        reference = [by_id[rollout_id] for rollout_id in reference_ids]
        conformal = [by_id[rollout_id] for rollout_id in conformal_ids]
        evaluation = [
            record for record in records if record["split"] == "evaluation"
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
            result = {
                "schema_version": 1,
                "model": model,
                "model_seed": seed,
                "alpha": alpha,
                "normalization": "training_task_early_max_z",
                "split_manifest": str(
                    (output_dir / "split_manifest.json").resolve()
                ),
                "calibration": str(
                    (alpha_root / "calibration.json").resolve()
                ),
                **metrics,
            }
            write_json(alpha_root / "metrics.json", result)
            overall_rows.append(_flatten_overall(seed, alpha, metrics))
            for task, values in metrics["per_task"].items():
                task_rows.append(
                    {
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
            }
        seed_summaries[str(seed)] = seed_summary
    aggregate = aggregate_metrics(overall_rows, task_rows, selected_alpha)
    summary = {
        "schema_version": 1,
        "protocol": "task-normalized held-out seen-task functional conformal calibration",
        "model": model,
        "model_seeds": list(seeds),
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
            "This task-conditioned normalization protocol applies to known seen tasks only.",
            "The selected alpha is fixed before evaluation; other alphas are sensitivity analyses.",
            "Standard deviations are population standard deviations across model seeds.",
        ],
    }
    _write_csv(output_dir / "per_seed_alpha_metrics.csv", overall_rows)
    _write_csv(output_dir / "per_task_alpha_metrics.csv", task_rows)
    plot_paths = create_plots(summary, output_dir) if make_plots else []
    summary["plots"] = [str(path) for path in plot_paths]
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=("indep", "lstm"), default="indep")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--calibration-successes-per-task", type=int, default=3)
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
        split_seed=args.split_seed,
        conformal_seed=args.conformal_seed,
        reference_fraction=args.reference_fraction,
        alphas=args.alphas,
        selected_alpha=args.selected_alpha,
        modulation=args.modulation,
        make_plots=not args.no_plots,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
