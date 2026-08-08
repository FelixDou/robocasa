"""Pure utilities for a staged early-SAFE plus elapsed-time fallback detector.

Each scored rollout contains two kinds of causal evidence: SAFE scores at fixed
early checkpoints and a task-conditioned elapsed-time risk trajectory.  The
fallback trajectory is exposed to the cascade only after its declared start.
Thresholds are selected jointly on validation parents under one event-level
false-positive-rate budget.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from .causal_prefix_residual import safe_auc, safe_average_precision
except ImportError:
    from causal_prefix_residual import safe_auc, safe_average_precision


DEFAULT_EARLY_LANDMARKS = (0.10, 0.25)
DEFAULT_TIME_FALLBACK = 0.50
CASCADE_DETECTORS = ("early_safe", "staged_safe_time", "time_only")


def validate_stages(early_landmarks, time_fallback):
    landmarks = tuple(sorted(float(value) for value in early_landmarks))
    fallback = float(time_fallback)
    if not landmarks or len(set(landmarks)) != len(landmarks):
        raise ValueError("Early SAFE landmarks must be nonempty and unique")
    if any(not 0.0 < value < 1.0 for value in landmarks):
        raise ValueError("Early SAFE landmarks must lie in (0, 1)")
    if not 0.0 < fallback <= 1.0:
        raise ValueError("The elapsed-time fallback must lie in (0, 1]")
    if max(landmarks) >= fallback:
        raise ValueError("Every early SAFE landmark must precede the time fallback")
    return landmarks, fallback


def stage_steps(horizon, early_landmarks, time_fallback):
    landmarks, fallback = validate_stages(early_landmarks, time_fallback)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("Task horizon must be positive")
    return {
        "early": {
            float(value): max(1, int(math.ceil(value * horizon))) for value in landmarks
        },
        "time_fallback": max(1, int(math.ceil(fallback * horizon))),
    }


def _maximum(values):
    values = np.asarray(values, dtype=np.float64)
    return None if not len(values) else float(np.max(values))


def detector_maximum(item, detector):
    if detector == "early_safe":
        return _maximum([stage["score"] for stage in item["early_stages"]])
    if detector == "time_only":
        return _maximum(item["full_time_scores"])
    if detector == "late_time":
        return _maximum(item["late_time_scores"])
    raise ValueError(f"Unknown staged detector {detector!r}")


def _finite_threshold_candidates(values):
    values = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    if not len(values):
        return np.asarray([1.0], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Threshold scores must be finite")
    never = np.nextafter(float(np.max(values)), np.inf)
    return np.concatenate([[never], np.unique(values)[::-1]])


def first_alert(item, detector, thresholds):
    """Return the first causal alarm or ``None`` for a scored rollout."""
    candidates = []
    if detector in {"early_safe", "staged_safe_time"}:
        threshold = float(thresholds["early_safe"])
        for stage in item["early_stages"]:
            if float(stage["score"]) >= threshold:
                candidates.append(
                    (int(stage["step"]), "early_safe", float(stage["score"]))
                )
    if detector == "staged_safe_time":
        threshold = float(thresholds["late_time"])
        for step, score in zip(item["late_time_steps"], item["late_time_scores"]):
            if float(score) >= threshold:
                candidates.append((int(step), "late_time", float(score)))
                break
    elif detector == "time_only":
        threshold = float(thresholds["time_only"])
        for step, score in zip(item["full_time_steps"], item["full_time_scores"]):
            if float(score) >= threshold:
                candidates.append((int(step), "time_only", float(score)))
                break
    if not candidates:
        return None
    step, source, score = min(candidates, key=lambda value: (value[0], value[1]))
    return {"step": step, "source": source, "score": score}


def _confusion(labels, predicted):
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=bool)
    return {
        "tp": int(np.sum((labels == 1) & predicted)),
        "fn": int(np.sum((labels == 1) & ~predicted)),
        "fp": int(np.sum((labels == 0) & predicted)),
        "tn": int(np.sum((labels == 0) & ~predicted)),
    }


def cascade_metrics(scored, detector, thresholds, horizons):
    labels = np.asarray([int(bool(item["failed"])) for item in scored], dtype=np.int64)
    alerts = [first_alert(item, detector, thresholds) for item in scored]
    predicted = np.asarray([alert is not None for alert in alerts], dtype=bool)
    confusion = _confusion(labels, predicted)
    tpr = confusion["tp"] / (confusion["tp"] + confusion["fn"])
    fpr = confusion["fp"] / (confusion["fp"] + confusion["tn"])
    tnr = 1.0 - fpr
    detected = []
    adjusted = []
    sources = {"early_safe": 0, "late_time": 0, "time_only": 0}
    for item, alert in zip(scored, alerts):
        if alert is not None:
            sources[alert["source"]] += 1
        if not bool(item["failed"]):
            continue
        if alert is None:
            adjusted.append(1.0)
            continue
        fraction = min(1.0, alert["step"] / int(horizons[int(item["task_id"])]))
        detected.append(float(fraction))
        adjusted.append(float(fraction))

    if detector == "early_safe":
        event_scores = [
            -1.0
            if detector_maximum(item, "early_safe") is None
            else detector_maximum(item, "early_safe")
            for item in scored
        ]
        score_definition = "maximum early checkpoint SAFE probability"
    elif detector == "time_only":
        event_scores = [detector_maximum(item, "time_only") for item in scored]
        score_definition = "maximum full-trajectory task-conditioned time risk"
    else:
        event_scores = []
        for item in scored:
            early = detector_maximum(item, "early_safe")
            late = detector_maximum(item, "late_time")
            margins = []
            if early is not None:
                margins.append(early - float(thresholds["early_safe"]))
            if late is not None:
                margins.append(late - float(thresholds["late_time"]))
            event_scores.append(max(margins) if margins else -1.0)
        score_definition = "maximum validation-threshold margin across cascade stages"

    return {
        "detector": detector,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "event_score_definition": score_definition,
        "rollouts": len(scored),
        "successes": int(np.sum(labels == 0)),
        "failures": int(np.sum(labels == 1)),
        "roc_auc": safe_auc(labels, event_scores),
        "average_precision": safe_average_precision(labels, event_scores),
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "true_positive_rate": float(tpr),
        "false_positive_rate": float(fpr),
        "confusion": confusion,
        "alarm_source_counts": sources,
        "mean_detected_failure_fraction": (
            float(np.mean(detected)) if detected else None
        ),
        "missed_failure_adjusted_detection_fraction": (
            float(np.mean(adjusted)) if adjusted else None
        ),
    }


def select_single_threshold(scored, detector, horizons, target_fpr=0.05):
    maxima = [detector_maximum(item, detector) for item in scored]
    key = "early_safe" if detector == "early_safe" else "time_only"
    candidates = _finite_threshold_candidates(maxima)
    feasible = []
    for threshold in candidates:
        metrics = cascade_metrics(scored, detector, {key: threshold}, horizons)
        if metrics["false_positive_rate"] <= float(target_fpr) + 1e-12:
            feasible.append(
                (
                    metrics["true_positive_rate"],
                    -metrics["missed_failure_adjusted_detection_fraction"],
                    -metrics["false_positive_rate"],
                    float(threshold),
                    metrics,
                )
            )
    if not feasible:
        raise ValueError(
            "No single-detector validation threshold satisfies the FPR cap"
        )
    selected = max(feasible, key=lambda value: value[:4])
    return {"threshold": selected[3], "validation_metrics": selected[4]}


def select_joint_thresholds(scored, horizons, target_fpr=0.05):
    """Jointly select one early-SAFE and one late-time threshold.

    Candidate policies are ranked by validation TPR, then by lower
    missed-failure-adjusted detection fraction, then by lower empirical FPR.
    The outer test is not involved.
    """
    early_candidates = _finite_threshold_candidates(
        [detector_maximum(item, "early_safe") for item in scored]
    )
    late_candidates = _finite_threshold_candidates(
        [detector_maximum(item, "late_time") for item in scored]
    )
    best = None
    feasible_count = 0
    for early_threshold in early_candidates:
        for late_threshold in late_candidates:
            thresholds = {
                "early_safe": float(early_threshold),
                "late_time": float(late_threshold),
            }
            metrics = cascade_metrics(scored, "staged_safe_time", thresholds, horizons)
            if metrics["false_positive_rate"] > float(target_fpr) + 1e-12:
                continue
            feasible_count += 1
            rank = (
                metrics["true_positive_rate"],
                -metrics["missed_failure_adjusted_detection_fraction"],
                -metrics["false_positive_rate"],
                float(early_threshold),
                float(late_threshold),
            )
            if best is None or rank > best[0]:
                best = (rank, thresholds, metrics)
    if best is None:
        raise ValueError("No staged validation threshold pair satisfies the FPR cap")
    return {
        "thresholds": best[1],
        "validation_metrics": best[2],
        "search": {
            "early_candidates": int(len(early_candidates)),
            "late_candidates": int(len(late_candidates)),
            "candidate_pairs": int(len(early_candidates) * len(late_candidates)),
            "feasible_pairs": int(feasible_count),
        },
    }


def prediction_record(item, detector, thresholds, horizons):
    alert = first_alert(item, detector, thresholds)
    task_id = int(item["task_id"])
    return {
        "detector": detector,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "first_detection_inference": None if alert is None else int(alert["step"]),
        "first_detection_fraction": (
            None if alert is None else min(1.0, alert["step"] / int(horizons[task_id]))
        ),
        "alarm_source": None if alert is None else alert["source"],
        "early_stage_scores": [dict(stage) for stage in item["early_stages"]],
        "maximum_full_time_score": detector_maximum(item, "time_only"),
        "maximum_late_time_score": detector_maximum(item, "late_time"),
    }
