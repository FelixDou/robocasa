"""Outcome-only causal-prefix utilities for early Xiaomi SAFE experiments.

The helpers in this module deliberately use only the final binary rollout
outcome.  They never consume Subtask-SAFE annotations, predicate progress,
failure onset, future observations, or a rollout's final length as an online
feature.  Complete source rollouts must be split before prefix construction.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import math

import numpy as np


DEFAULT_LANDMARKS = (0.10, 0.25, 0.50, 0.75)
DEFAULT_PRIMARY_LANDMARKS = (0.25, 0.50)
DETECTORS = ("time_only", "prefix_safe", "residual_safe_time")


def validate_landmarks(values):
    landmarks = tuple(float(value) for value in values)
    if not landmarks or any(not 0.0 < value <= 1.0 for value in landmarks):
        raise ValueError("Causal prefix landmarks must lie in (0, 1]")
    if len(set(landmarks)) != len(landmarks):
        raise ValueError("Causal prefix landmarks must be unique")
    return tuple(sorted(landmarks))


def rollout_id(rollout, identity):
    return str(identity[id(rollout)][1]["rollout_id"])


def stratified_meta_split(
    train_rollouts,
    identity,
    *,
    validation_per_class=5,
    seed=0,
):
    """Split complete source-training parents by task and final outcome."""
    groups = defaultdict(list)
    for rollout in train_rollouts:
        groups[(int(rollout.task_id), int(rollout.episode_success))].append(
            rollout_id(rollout, identity)
        )
    if not groups:
        raise ValueError("No source-training rollouts were provided")
    rng = np.random.default_rng(int(seed))
    fit_ids = set()
    validation_ids = set()
    counts = {}
    for (task_id, success), ids in sorted(groups.items()):
        ids = sorted(ids)
        if len(ids) <= int(validation_per_class):
            raise ValueError(
                f"Task {task_id} success={success} has {len(ids)} source-training "
                f"rollouts; needs more than validation_per_class="
                f"{validation_per_class}"
            )
        order = rng.permutation(len(ids))
        chosen = {ids[int(index)] for index in order[:validation_per_class]}
        validation_ids.update(chosen)
        fit_ids.update(set(ids) - chosen)
        counts.setdefault(task_id, {})["success" if success else "failure"] = {
            "fit": len(ids) - len(chosen),
            "validation": len(chosen),
        }
    if fit_ids & validation_ids:
        raise AssertionError("Causal-prefix fit and validation parents overlap")
    return {
        "fit": fit_ids,
        "validation": validation_ids,
        "counts": counts,
    }


def select_rollouts(rollouts, identity, ids):
    ids = {str(value) for value in ids}
    return [item for item in rollouts if rollout_id(item, identity) in ids]


def training_task_horizons(rollouts):
    """Estimate task timeout using only failures in the meta-fit split."""
    grouped = defaultdict(list)
    tasks = sorted({int(item.task_id) for item in rollouts})
    for rollout in rollouts:
        if not bool(int(rollout.episode_success)):
            grouped[int(rollout.task_id)].append(len(rollout.hidden_states))
    missing = [task for task in tasks if not grouped[task]]
    if missing:
        raise ValueError(f"Meta-fit split has no failures for tasks {missing}")
    return {task: int(max(grouped[task])) for task in tasks}


def fit_time_risk(rollouts, horizons, *, prior=0.5):
    """Fit monotone P(eventual failure | still active, task) curves."""
    from sklearn.isotonic import IsotonicRegression

    curves = {}
    for task_id, horizon in sorted(horizons.items()):
        selected = [item for item in rollouts if int(item.task_id) == task_id]
        raw = []
        active_counts = []
        progress = []
        for step in range(1, int(horizon) + 1):
            active = [item for item in selected if len(item.hidden_states) >= step]
            failures = sum(not bool(int(item.episode_success)) for item in active)
            risk = (failures + float(prior)) / (
                len(active) + 2.0 * float(prior)
            )
            raw.append(float(risk))
            active_counts.append(len(active))
            progress.append(step / int(horizon))
        weights = np.maximum(1, np.asarray(active_counts, dtype=np.int64))
        fitted = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        ).fit_transform(progress, raw, sample_weight=weights)
        curves[int(task_id)] = {
            "risk": np.asarray(fitted, dtype=np.float64),
            "raw_risk": np.asarray(raw, dtype=np.float64),
            "active_counts": np.asarray(active_counts, dtype=np.int64),
        }
    return curves


def time_risk_at(curves, task_id, step):
    risk = curves[int(task_id)]["risk"]
    index = min(max(1, int(step)), len(risk)) - 1
    return float(risk[index])


def summarize_prefix(sequence, step, *, window=4):
    """Return current, delta, recent mean, and recent slope feature blocks."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"Expected a nonempty [time, feature] array, got {values.shape}")
    step = int(step)
    if step < 1 or step > len(values):
        raise ValueError(f"Prefix step {step} is invalid for sequence length {len(values)}")
    end = step - 1
    start = max(0, step - int(window))
    current = values[end]
    previous = values[max(0, end - 1)]
    recent = values[start:step]
    slope = current - values[start]
    return np.concatenate(
        [current, current - previous, recent.mean(axis=0), slope], axis=0
    ).astype(np.float32, copy=False)


def summarize_trajectory(sequence, *, horizon=None, window=4):
    """Vectorize :func:`summarize_prefix` over every available causal step."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"Expected a nonempty [time, feature] array, got {values.shape}")
    length = len(values) if horizon is None else min(len(values), int(horizon))
    values = values[:length]
    current = values
    previous = np.concatenate([values[:1], values[:-1]], axis=0)
    delta = current - previous
    cumulative = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0)],
        axis=0,
    )
    means = []
    starts = []
    for end in range(1, length + 1):
        start = max(0, end - int(window))
        starts.append(start)
        means.append((cumulative[end] - cumulative[start]) / (end - start))
    means = np.asarray(means, dtype=np.float32)
    slopes = current - values[np.asarray(starts, dtype=np.int64)]
    return np.concatenate([current, delta, means, slopes], axis=1).astype(
        np.float32, copy=False
    )


def build_landmark_rows(
    rollouts,
    identity,
    *,
    horizons,
    time_curves,
    landmarks=DEFAULT_LANDMARKS,
    window=4,
    split,
):
    """Construct deployable at-risk prefix examples from disjoint parents."""
    landmarks = validate_landmarks(landmarks)
    rows = []
    excluded = []
    for rollout in rollouts:
        task_id = int(rollout.task_id)
        if task_id not in horizons:
            raise ValueError(f"Task {task_id} is absent from meta-fit horizons")
        source_length = len(rollout.hidden_states)
        parent_id = rollout_id(rollout, identity)
        for landmark in landmarks:
            step = max(1, int(math.ceil(float(landmark) * horizons[task_id])))
            if source_length < step:
                excluded.append(
                    {
                        "rollout_id": parent_id,
                        "task_id": task_id,
                        "landmark_fraction": float(landmark),
                        "source_inferences": source_length,
                        "required_inferences": step,
                        "reason": "naturally_terminated_before_landmark",
                    }
                )
                continue
            rows.append(
                {
                    "rollout_id": parent_id,
                    "task_id": task_id,
                    "failed": not bool(int(rollout.episode_success)),
                    "split": str(split),
                    "landmark_fraction": float(landmark),
                    "step": step,
                    "horizon": int(horizons[task_id]),
                    "time_risk": time_risk_at(time_curves, task_id, step),
                    "features": summarize_prefix(
                        rollout.hidden_states, step, window=window
                    ),
                }
            )
    return rows, excluded


def balance_training_rows(rows, *, seed=0):
    """Balance outcomes within task/landmark, then equalize parent totals.

    Unsupported task/landmark strata are excluded rather than teaching the
    feature residual a deterministic time label.  The time-only offset remains
    responsible for those strata.
    """
    groups = defaultdict(lambda: {False: [], True: []})
    for index, row in enumerate(rows):
        groups[(int(row["task_id"]), float(row["landmark_fraction"]))][
            bool(row["failed"])
        ].append(index)
    rng = np.random.default_rng(int(seed))
    selected = set()
    support = {}
    excluded = []
    for key, outcomes in sorted(groups.items()):
        successes = list(outcomes[False])
        failures = list(outcomes[True])
        target = min(len(successes), len(failures))
        support[f"{key[0]}@{key[1]:g}"] = {
            "available_successes": len(successes),
            "available_failures": len(failures),
            "selected_per_outcome": target,
        }
        if target:
            selected.update(
                int(value) for value in rng.permutation(successes)[:target]
            )
            selected.update(
                int(value) for value in rng.permutation(failures)[:target]
            )
        for index in successes + failures:
            if index not in selected:
                excluded.append(
                    {
                        "rollout_id": rows[index]["rollout_id"],
                        "task_id": key[0],
                        "landmark_fraction": key[1],
                        "failed": bool(rows[index]["failed"]),
                        "reason": (
                            "unsupported_outcome_stratum"
                            if target == 0
                            else "deterministic_outcome_balance"
                        ),
                    }
                )
    balanced = [copy.copy(row) for index, row in enumerate(rows) if index in selected]
    parent_counts = defaultdict(int)
    for row in balanced:
        parent_counts[str(row["rollout_id"])] += 1
    for row in balanced:
        row["sample_weight"] = 1.0 / parent_counts[str(row["rollout_id"])]
    total = sum(float(row["sample_weight"]) for row in balanced)
    if not balanced or total <= 0:
        raise ValueError("No supported balanced causal-prefix training rows remain")
    scale = len(balanced) / total
    for row in balanced:
        row["sample_weight"] *= scale
    return balanced, support, excluded


def fit_feature_scaler(rows):
    matrix = np.stack([row["features"] for row in rows]).astype(np.float64)
    weights = np.asarray(
        [float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float64
    )
    weights /= weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum(((matrix - mean) ** 2) * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def transform_features(matrix, scaler):
    values = np.asarray(matrix, dtype=np.float32)
    return (values - scaler["mean"]) / scaler["scale"]


def logit(values, epsilon=1e-4):
    values = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(values / (1.0 - values))


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    if len(set(labels.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def safe_average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    if len(set(labels.tolist())) < 2:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def landmark_metrics(scored_rows, detectors=DETECTORS):
    output = []
    landmarks = sorted({float(row["landmark_fraction"]) for row in scored_rows})
    tasks = sorted({int(row["task_id"]) for row in scored_rows})
    for landmark in landmarks:
        landmark_rows = [
            row for row in scored_rows if float(row["landmark_fraction"]) == landmark
        ]
        for detector in detectors:
            for scope, task_id, selected in [
                ("pooled", None, landmark_rows),
                *[
                    (
                        "task",
                        task,
                        [row for row in landmark_rows if int(row["task_id"]) == task],
                    )
                    for task in tasks
                ],
            ]:
                if not selected:
                    continue
                labels = [int(bool(row["failed"])) for row in selected]
                scores = [float(row["scores"][detector]) for row in selected]
                output.append(
                    {
                        "scope": scope,
                        "task_id": task_id,
                        "landmark_fraction": landmark,
                        "detector": detector,
                        "rollouts": len(selected),
                        "successes": sum(not value for value in labels),
                        "failures": sum(labels),
                        "roc_auc": safe_auc(labels, scores),
                        "average_precision": safe_average_precision(labels, scores),
                    }
                )
    return output


def primary_landmark_score(metrics, detector, primary_landmarks):
    """Return task-macro AUC across preregistered supported landmarks."""
    primary = set(validate_landmarks(primary_landmarks))
    values = [
        float(row["roc_auc"])
        for row in metrics
        if row["scope"] == "task"
        and row["detector"] == detector
        and float(row["landmark_fraction"]) in primary
        and row["roc_auc"] is not None
    ]
    if not values:
        raise ValueError(
            f"Detector {detector} has no supported task-level primary landmark AUC"
        )
    return float(np.mean(values)), len(values)


def threshold_at_fpr(scored, detector, target_fpr=0.05):
    """Select the most sensitive validation threshold under an empirical FPR cap."""
    values = np.asarray(
        [float(np.max(item["trajectories"][detector])) for item in scored],
        dtype=np.float64,
    )
    labels = np.asarray(
        [int(bool(item["failed"])) for item in scored], dtype=np.int64
    )
    candidates = np.concatenate(
        [[np.inf], np.unique(values)[::-1], [-np.inf]]
    )
    feasible = []
    for threshold in candidates:
        predicted = values >= threshold
        negatives = labels == 0
        positives = labels == 1
        fpr = float(np.mean(predicted[negatives])) if np.any(negatives) else None
        tpr = float(np.mean(predicted[positives])) if np.any(positives) else None
        if fpr is not None and tpr is not None and fpr <= float(target_fpr) + 1e-12:
            feasible.append((tpr, -fpr, -float(threshold), float(threshold)))
    if not feasible:
        raise ValueError("Validation set has no threshold with a defined FPR and TPR")
    _, _, _, threshold = max(feasible)
    return threshold


def event_metrics(scored, detector, threshold, horizons):
    labels = np.asarray([int(bool(item["failed"])) for item in scored], dtype=np.int64)
    maxima = np.asarray(
        [float(np.max(item["trajectories"][detector])) for item in scored],
        dtype=np.float64,
    )
    predicted = maxima >= float(threshold)
    tp = int(np.sum((labels == 1) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    tpr = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    tnr = tn / (tn + fp) if tn + fp else None
    detection = []
    adjusted = []
    for item in scored:
        if not item["failed"]:
            continue
        crossings = np.flatnonzero(
            np.asarray(item["trajectories"][detector]) >= float(threshold)
        )
        task_id = int(item["task_id"])
        if len(crossings):
            fraction = min(1.0, (int(crossings[0]) + 1) / horizons[task_id])
            detection.append(float(fraction))
            adjusted.append(float(fraction))
        else:
            adjusted.append(1.0)
    return {
        "detector": detector,
        "threshold": float(threshold),
        "rollouts": len(scored),
        "successes": int(np.sum(labels == 0)),
        "failures": int(np.sum(labels == 1)),
        "roc_auc": safe_auc(labels, maxima),
        "average_precision": safe_average_precision(labels, maxima),
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": (
            None if tpr is None or tnr is None else float(0.5 * (tpr + tnr))
        ),
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "mean_detected_failure_fraction": (
            float(np.mean(detection)) if detection else None
        ),
        "missed_failure_adjusted_detection_fraction": (
            float(np.mean(adjusted)) if adjusted else None
        ),
    }


def clone_jsonable_time_curves(curves):
    return {
        str(task): {
            key: np.asarray(value).tolist()
            for key, value in payload.items()
        }
        for task, payload in curves.items()
    }
