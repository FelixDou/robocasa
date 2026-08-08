"""Compare causal elapsed-time, SAFE, and SAFE+time failure detectors.

This module consumes score trajectories produced by ``train_seen_tasks.py``.
Only the current prefix, task identity, and training-derived task horizon are
available to a detector.  Final rollout duration is never used as an input.
Complete rollouts are split before prefix rows are constructed, and all model
fitting plus threshold selection is confined to the source training split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


DETECTORS = ("time_only", "safe_only", "safe_time_task")
DEFAULT_LANDMARKS = (0.10, 0.25, 0.50, 0.75, 1.00)
DEFAULT_REGULARIZATIONS = (0.01, 0.1, 1.0, 10.0)


def _load_jsonl(path):
    path = Path(path)
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No score records found in {path}")
    ids = [str(record["rollout_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate rollout IDs in {path}")
    for record in records:
        scores = np.asarray(record.get("scores"), dtype=np.float64)
        if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
            raise ValueError(f"Invalid scores for {record.get('rollout_id')}")
        if record.get("split") not in {"train", "test"}:
            raise ValueError(f"Invalid split for {record.get('rollout_id')}")
    return records


def _record_signature(records):
    return {
        str(record["rollout_id"]): (
            str(record["split"]),
            str(record["task_name"]),
            bool(record["failed"]),
            len(record["scores"]),
        )
        for record in records
    }


def load_runs(final_root, models=("indep", "lstm"), seeds=(0, 1, 2)):
    """Load score runs and verify identical rollout identities."""
    final_root = Path(final_root).resolve()
    runs = {}
    reference = None
    for model in models:
        for seed in seeds:
            path = final_root / f"{model}_seed{seed}" / "scores.jsonl"
            records = _load_jsonl(path)
            signature = _record_signature(records)
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(
                    f"Rollout identities differ for {model} seed {seed}"
                )
            runs[(model, int(seed))] = records
    return runs, reference


def stratified_meta_split(records, validation_per_class=5, seed=0):
    """Split source training rollouts before causal prefix expansion."""
    groups = defaultdict(list)
    test_ids = set()
    for record in records:
        rollout_id = str(record["rollout_id"])
        if record["split"] == "test":
            test_ids.add(rollout_id)
            continue
        groups[(str(record["task_name"]), bool(record["failed"]))].append(
            rollout_id
        )
    if not groups:
        raise ValueError("No source training rollouts were found")
    rng = np.random.default_rng(seed)
    fit_ids = set()
    validation_ids = set()
    counts = {}
    for (task, failed), ids in sorted(groups.items()):
        ids = sorted(ids)
        if len(ids) <= validation_per_class:
            raise ValueError(
                f"{task} outcome={int(failed)} has {len(ids)} training rollouts; "
                f"needs more than validation_per_class={validation_per_class}"
            )
        order = rng.permutation(len(ids))
        selected = {ids[int(index)] for index in order[:validation_per_class]}
        validation_ids.update(selected)
        fit_ids.update(set(ids) - selected)
        counts.setdefault(task, {})["failure" if failed else "success"] = {
            "fit": len(ids) - len(selected),
            "validation": len(selected),
        }
    if fit_ids & validation_ids or fit_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Fit, validation, and test rollout IDs overlap")
    return {
        "fit": fit_ids,
        "validation": validation_ids,
        "test": test_ids,
        "counts": counts,
    }


def _selected(records, ids):
    return [record for record in records if str(record["rollout_id"]) in ids]


def training_task_horizons(records):
    """Estimate task timeout in policy inferences from fit failures only."""
    horizons = {}
    for task in sorted({str(record["task_name"]) for record in records}):
        failures = [
            len(record["scores"])
            for record in records
            if str(record["task_name"]) == task and bool(record["failed"])
        ]
        if not failures:
            raise ValueError(f"Task {task} has no fit failures")
        horizons[task] = int(max(failures))
    return horizons


def fit_safe_normalization(records, horizons):
    """Fit robust task-specific SAFE scale from fit rollouts only."""
    result = {}
    for task in sorted(horizons):
        maxima = []
        for record in records:
            if str(record["task_name"]) != task:
                continue
            values = np.asarray(record["scores"], dtype=np.float64)
            maxima.append(float(np.max(values[: horizons[task]])))
        if not maxima:
            raise ValueError(f"Task {task} has no fit SAFE scores")
        array = np.asarray(maxima, dtype=np.float64)
        location = float(np.median(array))
        mad_scale = float(1.4826 * np.median(np.abs(array - location)))
        std_scale = float(np.std(array))
        scale = max(mad_scale, std_scale, 1e-6)
        result[task] = {
            "location": location,
            "scale": scale,
            "fit_rollouts": len(array),
        }
    return result


def fit_time_risk(records, horizons, prior=0.5):
    """Fit monotone P(failure | still active at t, task) curves."""
    from sklearn.isotonic import IsotonicRegression

    curves = {}
    for task, horizon in sorted(horizons.items()):
        selected = [record for record in records if record["task_name"] == task]
        raw = []
        weights = []
        progress = []
        for step in range(1, horizon + 1):
            active = [record for record in selected if len(record["scores"]) >= step]
            failures = sum(bool(record["failed"]) for record in active)
            risk = (failures + prior) / (len(active) + 2.0 * prior)
            raw.append(float(risk))
            weights.append(max(1, len(active)))
            progress.append(step / horizon)
        fitted = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        ).fit_transform(progress, raw, sample_weight=weights)
        curves[task] = {
            "risk": np.asarray(fitted, dtype=np.float64),
            "raw_risk": np.asarray(raw, dtype=np.float64),
            "active_counts": np.asarray(weights, dtype=np.int64),
        }
    return curves


def detector_trajectories(record, horizons, normalization, time_curves):
    """Construct deployable prefix features without reading final duration."""
    task = str(record["task_name"])
    horizon = int(horizons[task])
    raw_scores = np.asarray(record["scores"], dtype=np.float64)[:horizon]
    running_safe = np.maximum.accumulate(raw_scores)
    stats = normalization[task]
    safe_z = (running_safe - stats["location"]) / stats["scale"]
    steps = np.arange(1, len(raw_scores) + 1, dtype=np.int64)
    curve = time_curves[task]["risk"]
    time_risk = curve[np.minimum(steps, len(curve)) - 1]
    progress = np.minimum(steps / horizon, 1.0)
    return {
        "safe_z": safe_z,
        "time_risk": time_risk,
        "progress": progress,
    }


def _feature_matrix(prefix, task, task_names):
    task_index = task_names.index(task)
    one_hot = np.zeros((len(prefix["safe_z"]), len(task_names) - 1))
    if task_index > 0:
        one_hot[:, task_index - 1] = 1.0
    safe = np.asarray(prefix["safe_z"], dtype=np.float64)
    time = np.asarray(prefix["time_risk"], dtype=np.float64)
    progress = np.asarray(prefix["progress"], dtype=np.float64)
    columns = [
        safe,
        time,
        progress,
        progress**2,
        safe * progress,
    ]
    if one_hot.shape[1]:
        columns.extend(
            [one_hot, one_hot * progress[:, None], one_hot * safe[:, None]]
        )
    return np.column_stack(columns)


def fit_hybrid(records, horizons, normalization, time_curves, regularization=1.0):
    """Fit SAFE+time+task logistic risk with equal rollout weighting."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    task_names = sorted(horizons)
    matrices = []
    labels = []
    weights = []
    for record in records:
        prefix = detector_trajectories(
            record, horizons, normalization, time_curves
        )
        matrix = _feature_matrix(prefix, str(record["task_name"]), task_names)
        matrices.append(matrix)
        labels.extend([int(bool(record["failed"]))] * len(matrix))
        weights.extend([1.0 / len(matrix)] * len(matrix))
    labels = np.asarray(labels, dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Hybrid fit requires both rollout outcomes")
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(regularization),
            max_iter=5000,
            solver="lbfgs",
        ),
    )
    estimator.fit(
        np.concatenate(matrices, axis=0),
        labels,
        logisticregression__sample_weight=np.asarray(weights, dtype=np.float64),
    )
    return estimator, task_names


def score_records(records, horizons, normalization, time_curves, hybrid, task_names):
    scored = []
    for record in records:
        prefix = detector_trajectories(
            record, horizons, normalization, time_curves
        )
        matrix = _feature_matrix(prefix, str(record["task_name"]), task_names)
        trajectories = {
            "time_only": np.asarray(prefix["time_risk"], dtype=np.float64),
            "safe_only": np.asarray(prefix["safe_z"], dtype=np.float64),
            "safe_time_task": hybrid.predict_proba(matrix)[:, 1],
        }
        scored.append(
            {
                "record": record,
                "trajectories": trajectories,
            }
        )
    return scored


def select_hybrid(
    fit_records,
    validation_records,
    horizons,
    normalization,
    time_curves,
    regularizations,
):
    """Select hybrid regularization and alert threshold on validation only."""
    candidates = []
    for regularization in regularizations:
        estimator, task_names = fit_hybrid(
            fit_records,
            horizons,
            normalization,
            time_curves,
            regularization=regularization,
        )
        scored = score_records(
            validation_records,
            horizons,
            normalization,
            time_curves,
            estimator,
            task_names,
        )
        threshold = select_threshold(scored, "safe_time_task")
        rates = threshold["validation_rates"]
        candidates.append(
            {
                "regularization": float(regularization),
                "estimator": estimator,
                "task_names": task_names,
                "threshold": threshold,
                "rank": (
                    float(rates["balanced_accuracy"]),
                    -float(rates["false_positive_rate"]),
                    float(rates["true_positive_rate"]),
                    -abs(math.log10(float(regularization))),
                ),
            }
        )
    selected = max(candidates, key=lambda value: value["rank"])
    audit = [
        {
            "regularization": value["regularization"],
            "threshold": value["threshold"]["threshold"],
            "validation_rates": value["threshold"]["validation_rates"],
            "selected": value is selected,
        }
        for value in candidates
    ]
    return (
        selected["estimator"],
        selected["task_names"],
        selected["threshold"],
        selected["regularization"],
        audit,
    )


def _confusion(labels, predicted):
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    fp = int(np.sum(~labels & predicted))
    tn = int(np.sum(~labels & ~predicted))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def _rates(confusion):
    tp, fn = confusion["tp"], confusion["fn"]
    fp, tn = confusion["fp"], confusion["tn"]
    tpr = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    tnr = tn / (tn + fp) if tn + fp else None
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    balanced = None if tpr is None or tnr is None else 0.5 * (tpr + tnr)
    return {
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "true_negative_rate": tnr,
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced,
    }


def select_threshold(scored, detector):
    """Select event threshold by validation balanced accuracy only."""
    labels = np.asarray(
        [int(bool(item["record"]["failed"])) for item in scored], dtype=np.int64
    )
    values = np.asarray(
        [float(np.max(item["trajectories"][detector])) for item in scored],
        dtype=np.float64,
    )
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Threshold validation requires both outcomes")
    unique = np.unique(values)
    candidates = np.concatenate(
        ([np.nextafter(unique[0], -np.inf)], unique, [np.nextafter(unique[-1], np.inf)])
    )
    choices = []
    for threshold in candidates:
        confusion = _confusion(labels, values >= threshold)
        rates = _rates(confusion)
        choices.append(
            (
                float(rates["balanced_accuracy"]),
                -float(rates["false_positive_rate"]),
                float(rates["true_positive_rate"]),
                float(threshold),
                confusion,
                rates,
            )
        )
    best = max(choices, key=lambda item: item[:3])
    return {
        "threshold": best[3],
        "selection_metric": "validation_balanced_accuracy",
        "validation_confusion": best[4],
        "validation_rates": best[5],
    }


def _safe_auc(function, labels, scores):
    return None if len(set(int(value) for value in labels)) < 2 else float(
        function(labels, scores)
    )


def evaluate_event_scores(scored, detector, threshold, horizons):
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = [int(bool(item["record"]["failed"])) for item in scored]
    values = [
        float(np.max(item["trajectories"][detector])) for item in scored
    ]
    predicted = [value >= threshold for value in values]
    confusion = _confusion(labels, predicted)
    detection_fractions = []
    failure_detection_fractions = []
    for item in scored:
        trajectory = item["trajectories"][detector]
        crossings = np.flatnonzero(trajectory >= threshold)
        fraction = None
        if len(crossings):
            task = str(item["record"]["task_name"])
            fraction = min(1.0, float((int(crossings[0]) + 1) / horizons[task]))
            detection_fractions.append(fraction)
            if bool(item["record"]["failed"]):
                failure_detection_fractions.append(fraction)
    return {
        "rollouts": len(scored),
        "successes": sum(not value for value in labels),
        "failures": sum(labels),
        "roc_auc": _safe_auc(roc_auc_score, labels, values),
        "average_precision": _safe_auc(average_precision_score, labels, values),
        "threshold": float(threshold),
        "confusion": confusion,
        **_rates(confusion),
        "mean_detection_fraction": (
            float(np.mean(detection_fractions)) if detection_fractions else None
        ),
        "mean_detected_failure_fraction": (
            float(np.mean(failure_detection_fractions))
            if failure_detection_fractions
            else None
        ),
        "missed_failure_adjusted_detection_fraction": float(
            np.mean(
                [
                    (
                        min(
                            1.0,
                            (int(np.flatnonzero(item["trajectories"][detector] >= threshold)[0]) + 1)
                            / horizons[str(item["record"]["task_name"])],
                        )
                        if len(np.flatnonzero(item["trajectories"][detector] >= threshold))
                        else 1.0
                    )
                    for item in scored
                    if bool(item["record"]["failed"])
                ]
            )
        ),
    }


def evaluate_landmarks(scored, horizons, fractions):
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    tasks = sorted(horizons)
    for fraction in fractions:
        selected = []
        for item in scored:
            task = str(item["record"]["task_name"])
            step = max(1, int(math.ceil(float(fraction) * horizons[task])))
            if len(item["record"]["scores"]) >= step:
                selected.append((item, step))
        for detector in DETECTORS:
            labels = [int(bool(item["record"]["failed"])) for item, _ in selected]
            values = [
                float(item["trajectories"][detector][step - 1])
                for item, step in selected
            ]
            rows.append(
                {
                    "scope": "pooled",
                    "task_name": "all",
                    "landmark_fraction": float(fraction),
                    "detector": detector,
                    "rollouts": len(selected),
                    "successes": sum(not value for value in labels),
                    "failures": sum(labels),
                    "roc_auc": _safe_auc(roc_auc_score, labels, values),
                    "average_precision": _safe_auc(
                        average_precision_score, labels, values
                    ),
                }
            )
            for task in tasks:
                task_selected = [
                    pair for pair in selected if pair[0]["record"]["task_name"] == task
                ]
                task_labels = [
                    int(bool(item["record"]["failed"]))
                    for item, _ in task_selected
                ]
                task_values = [
                    float(item["trajectories"][detector][step - 1])
                    for item, step in task_selected
                ]
                rows.append(
                    {
                        "scope": "task",
                        "task_name": task,
                        "landmark_fraction": float(fraction),
                        "detector": detector,
                        "rollouts": len(task_selected),
                        "successes": sum(not value for value in task_labels),
                        "failures": sum(task_labels),
                        "roc_auc": _safe_auc(
                            roc_auc_score, task_labels, task_values
                        ),
                        "average_precision": _safe_auc(
                            average_precision_score, task_labels, task_values
                        ),
                    }
                )
    return rows


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return {
        "mean": float(np.mean(values)) if len(values) else None,
        "std": float(np.std(values)) if len(values) else None,
        "values": values.tolist(),
    }


def aggregate_runs(run_results):
    output = []
    for model in sorted({result["model"] for result in run_results}):
        selected = [result for result in run_results if result["model"] == model]
        for detector in DETECTORS:
            metrics = [result["test_metrics"][detector] for result in selected]
            output.append(
                {
                    "model": model,
                    "detector": detector,
                    "num_seeds": len(metrics),
                    **{
                        key: _mean_std([metric[key] for metric in metrics])
                        for key in (
                            "roc_auc",
                            "average_precision",
                            "accuracy",
                            "balanced_accuracy",
                            "true_positive_rate",
                            "false_positive_rate",
                            "mean_detected_failure_fraction",
                            "missed_failure_adjusted_detection_fraction",
                        )
                    },
                }
            )
    return output


def create_plots(summary, output_dir, formats):
    if not formats:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    outputs = []
    labels = {
        "time_only": "Time only",
        "safe_only": "SAFE only",
        "safe_time_task": "SAFE + time + task",
    }
    for model in sorted({row["model"] for row in summary["aggregate"]}):
        rows = [row for row in summary["aggregate"] if row["model"] == model]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
        for ax, metric, title in (
            (axes[0], "roc_auc", "Held-out ROC-AUC"),
            (axes[1], "balanced_accuracy", "Held-out balanced accuracy"),
        ):
            values = [next(row for row in rows if row["detector"] == detector)[metric] for detector in DETECTORS]
            ax.bar(
                np.arange(len(DETECTORS)),
                [value["mean"] for value in values],
                yerr=[value["std"] for value in values],
                color=("#6B7280", "#2563A6", "#D97706"),
                capsize=4,
            )
            ax.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
            ax.set_xticks(np.arange(len(DETECTORS)), [labels[value] for value in DETECTORS], rotation=18, ha="right")
            ax.set_ylim(0.4, 1.0)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.35)
        fig.suptitle(f"Causal time/Safe comparison — {model}")
        fig.tight_layout()
        for extension in formats:
            path = output_dir / f"event_comparison_{model}.{extension}"
            fig.savefig(path, dpi=220 if extension == "png" else None)
            outputs.append(path)
        plt.close(fig)
    return outputs


def analyze_time_safe_hybrid(
    final_root,
    output_dir,
    *,
    models=("indep", "lstm"),
    seeds=(0, 1, 2),
    validation_per_class=5,
    split_seed=0,
    regularizations=DEFAULT_REGULARIZATIONS,
    landmark_fractions=DEFAULT_LANDMARKS,
    formats=("png", "pdf"),
):
    final_root = Path(final_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = tuple(models)
    seeds = tuple(int(seed) for seed in seeds)
    landmark_fractions = tuple(float(value) for value in landmark_fractions)
    if not landmark_fractions or any(not 0 < value <= 1 for value in landmark_fractions):
        raise ValueError("Every landmark fraction must lie in (0, 1]")
    formats = tuple(dict.fromkeys(formats))
    if set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("Formats must be png, pdf, or svg")
    regularizations = tuple(float(value) for value in regularizations)
    if not regularizations or any(value <= 0 for value in regularizations):
        raise ValueError("Every regularization candidate must be positive")

    runs, signature = load_runs(final_root, models=models, seeds=seeds)
    reference_records = next(iter(runs.values()))
    split = stratified_meta_split(
        reference_records,
        validation_per_class=validation_per_class,
        seed=split_seed,
    )
    run_results = []
    landmark_rows = []
    event_rows = []
    event_predictions = []
    for (model, seed), records in sorted(runs.items()):
        fit_records = _selected(records, split["fit"])
        validation_records = _selected(records, split["validation"])
        test_records = _selected(records, split["test"])
        horizons = training_task_horizons(fit_records)
        normalization = fit_safe_normalization(fit_records, horizons)
        time_curves = fit_time_risk(fit_records, horizons)
        (
            hybrid,
            task_names,
            hybrid_threshold,
            selected_regularization,
            hybrid_selection,
        ) = select_hybrid(
            fit_records,
            validation_records,
            horizons,
            normalization,
            time_curves,
            regularizations,
        )
        validation_scored = score_records(
            validation_records,
            horizons,
            normalization,
            time_curves,
            hybrid,
            task_names,
        )
        test_scored = score_records(
            test_records,
            horizons,
            normalization,
            time_curves,
            hybrid,
            task_names,
        )
        thresholds = {
            "time_only": select_threshold(validation_scored, "time_only"),
            "safe_only": select_threshold(validation_scored, "safe_only"),
            "safe_time_task": hybrid_threshold,
        }
        test_metrics = {
            detector: evaluate_event_scores(
                test_scored,
                detector,
                thresholds[detector]["threshold"],
                horizons,
            )
            for detector in DETECTORS
        }
        for item in test_scored:
            record = item["record"]
            task = str(record["task_name"])
            for detector in DETECTORS:
                trajectory = item["trajectories"][detector]
                threshold = float(thresholds[detector]["threshold"])
                crossings = np.flatnonzero(trajectory >= threshold)
                detection_index = int(crossings[0]) if len(crossings) else None
                event_predictions.append(
                    {
                        "model": model,
                        "seed": seed,
                        "rollout_id": str(record["rollout_id"]),
                        "task_name": task,
                        "failed": bool(record["failed"]),
                        "detector": detector,
                        "event_score": float(np.max(trajectory)),
                        "threshold": threshold,
                        "detected": detection_index is not None,
                        "detection_index": detection_index,
                        "detection_fraction_of_training_horizon": (
                            None
                            if detection_index is None
                            else min(
                                1.0,
                                float((detection_index + 1) / horizons[task]),
                            )
                        ),
                        "observed_num_inferences": len(record["scores"]),
                        "training_task_horizon": horizons[task],
                    }
                )
        for detector, values in test_metrics.items():
            event_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "detector": detector,
                    **{
                        key: value
                        for key, value in values.items()
                        if not isinstance(value, dict)
                    },
                    **{
                        f"confusion_{key}": value
                        for key, value in values["confusion"].items()
                    },
                }
            )
        rows = evaluate_landmarks(test_scored, horizons, landmark_fractions)
        for row in rows:
            landmark_rows.append({"model": model, "seed": seed, **row})
        runtime_root = output_dir / "runtime" / f"{model}_seed{seed}"
        runtime_root.mkdir(parents=True, exist_ok=True)
        import joblib

        hybrid_path = runtime_root / "safe_time_task.joblib"
        joblib.dump(hybrid, hybrid_path)
        runtime_config = {
            "schema_version": 1,
            "model": model,
            "seed": seed,
            "task_names": task_names,
            "training_task_horizons": horizons,
            "safe_normalization": normalization,
            "time_risk_curves": {
                task: {
                    "risk": values["risk"].tolist(),
                    "raw_risk": values["raw_risk"].tolist(),
                    "active_counts": values["active_counts"].tolist(),
                }
                for task, values in time_curves.items()
            },
            "selected_hybrid_regularization": selected_regularization,
            "thresholds": {
                detector: float(values["threshold"])
                for detector, values in thresholds.items()
            },
            "hybrid_model": str(hybrid_path.resolve()),
            "feature_order": [
                "safe_z",
                "time_risk",
                "progress",
                "progress_squared",
                "safe_z_times_progress",
                "task_one_hot",
                "task_one_hot_times_progress",
                "task_one_hot_times_safe_z",
            ],
            "online_inputs_only": True,
            "subtask_safe": False,
        }
        runtime_config_path = runtime_root / "runtime.json"
        runtime_config_path.write_text(
            json.dumps(runtime_config, indent=2, sort_keys=True) + "\n"
        )
        run_results.append(
            {
                "model": model,
                "seed": seed,
                "runtime_config": str(runtime_config_path.resolve()),
                "hybrid_model": str(hybrid_path.resolve()),
                "selected_hybrid_regularization": selected_regularization,
                "hybrid_selection": hybrid_selection,
                "thresholds": thresholds,
                "test_metrics": test_metrics,
            }
        )
    aggregate = aggregate_runs(run_results)
    summary = {
        "schema_version": 1,
        "protocol": "causal task-conditioned time, SAFE, and SAFE+time comparison",
        "final_root": str(final_root),
        "output_dir": str(output_dir),
        "models": list(models),
        "seeds": list(seeds),
        "detectors": list(DETECTORS),
        "label": "binary final rollout failure only",
        "subtask_safe": False,
        "causality": {
            "available_online": [
                "current SAFE prefix",
                "current inference index",
                "task identity",
                "training-derived task horizon",
                "rollout remains active",
            ],
            "forbidden": [
                "final rollout duration",
                "future SAFE scores",
                "subtask progress",
                "failure onset",
            ],
            "split_before_prefix_expansion": True,
            "prefix_rows_equal_weight_per_rollout": True,
        },
        "threshold_selection": {
            "source": "source-training validation rollouts only",
            "metric": "balanced_accuracy",
            "validation_per_task_outcome": int(validation_per_class),
            "split_seed": int(split_seed),
        },
        "hybrid_regularization_candidates": list(regularizations),
        "landmark_fractions": list(landmark_fractions),
        "counts": {
            "unique_rollouts": len(signature),
            "fit": len(split["fit"]),
            "validation": len(split["validation"]),
            "test": len(split["test"]),
        },
        "split_counts": split["counts"],
        "split_rollout_ids": {
            key: sorted(split[key]) for key in ("fit", "validation", "test")
        },
        "artifacts": {
            "per_seed_event_metrics": str(
                (output_dir / "per_seed_event_metrics.csv").resolve()
            ),
            "landmark_metrics": str(
                (output_dir / "landmark_metrics.csv").resolve()
            ),
            "event_predictions": str(
                (output_dir / "event_predictions.jsonl").resolve()
            ),
            "runtime_root": str((output_dir / "runtime").resolve()),
        },
        "runs": run_results,
        "aggregate": aggregate,
        "notes": [
            "Time-only risk is a monotone training-fitted estimate of eventual failure among rollouts still active at each task-conditioned inference index.",
            "SAFE-only uses a running maximum and task normalization fitted from meta-fit rollouts only.",
            "The hybrid logistic model sees only causal SAFE, elapsed-time, progress, and task features.",
            "Thresholds maximize balanced accuracy on a disjoint subset of source training rollouts; the source test split is evaluation-only for this analysis.",
            "If the source test was opened by earlier experiments, this comparison is post-hoc and needs a fresh independent test before a confirmatory claim.",
        ],
    }
    plot_paths = create_plots(summary, output_dir, formats)
    summary["figures"] = [str(path) for path in plot_paths]
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(output_dir / "per_seed_event_metrics.csv", event_rows)
    _write_csv(output_dir / "landmark_metrics.csv", landmark_rows)
    with (output_dir / "event_predictions.jsonl").open("w") as stream:
        for row in event_predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", choices=("indep", "lstm"), default=["indep", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--regularizations",
        nargs="+",
        type=float,
        default=list(DEFAULT_REGULARIZATIONS),
    )
    parser.add_argument("--landmark-fractions", nargs="+", type=float, default=list(DEFAULT_LANDMARKS))
    parser.add_argument("--formats", nargs="*", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = analyze_time_safe_hybrid(
        args.final_root,
        args.output_dir,
        models=args.models,
        seeds=args.seeds,
        validation_per_class=args.validation_per_class,
        split_seed=args.split_seed,
        regularizations=args.regularizations,
        landmark_fractions=args.landmark_fractions,
        formats=args.formats,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
