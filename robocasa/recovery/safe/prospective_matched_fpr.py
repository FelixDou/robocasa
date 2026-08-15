"""Pure event-level utilities for prospective matched-FPR SAFE evaluation.

The utilities in this module are deliberately independent of RoboCasa and the
official SAFE checkout.  They operate on causal score trajectories that have
already been produced by frozen detector checkpoints.  Thresholds are selected
on calibration rollouts only and can then be serialized for a disjoint shadow
test.
"""

from __future__ import annotations

import math

import numpy as np


DETECTORS = ("safe_only", "staged_safe_time", "time_only")


def validate_stage_windows(safe_end=0.25, time_start=0.50):
    safe_end = float(safe_end)
    time_start = float(time_start)
    if not 0.0 < safe_end < time_start <= 1.0:
        raise ValueError(
            "Stage windows require 0 < safe_end < time_start <= 1"
        )
    return safe_end, time_start


def _window(record, detector, safe_end, time_start):
    horizon = int(record["horizon"])
    if horizon < 1:
        raise ValueError("Every rollout requires a positive task horizon")
    safe = np.asarray(record["safe_scores"], dtype=np.float64)
    time = np.asarray(record["time_scores"], dtype=np.float64)
    if safe.ndim != 1 or time.ndim != 1 or not len(safe) or len(safe) != len(time):
        raise ValueError("SAFE and time trajectories must be aligned 1-D arrays")
    if not np.all(np.isfinite(safe)) or not np.all(np.isfinite(time)):
        raise ValueError("Detector trajectories must be finite")
    steps = np.arange(1, len(safe) + 1, dtype=np.int64)
    if detector == "safe_only":
        return steps, safe
    if detector == "time_only":
        return steps, time
    if detector == "staged_safe_time":
        safe_stop = max(1, int(math.ceil(float(safe_end) * horizon)))
        time_begin = max(1, int(math.ceil(float(time_start) * horizon)))
        return {
            "early_safe": (steps[steps <= safe_stop], safe[steps <= safe_stop]),
            "late_time": (steps[steps >= time_begin], time[steps >= time_begin]),
        }
    raise ValueError(f"Unknown detector {detector!r}")


def _candidates(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Threshold candidates require finite scalar scores")
    return np.concatenate(
        [[np.nextafter(float(np.max(values)), np.inf)], np.unique(values)[::-1]]
    )


def first_alert(record, detector, thresholds, safe_end=0.25, time_start=0.50):
    safe_end, time_start = validate_stage_windows(safe_end, time_start)
    view = _window(record, detector, safe_end, time_start)
    if detector != "staged_safe_time":
        steps, values = view
        threshold = float(thresholds[detector])
        crossings = np.flatnonzero(values >= threshold)
        if not len(crossings):
            return None
        index = int(crossings[0])
        return {
            "step": int(steps[index]),
            "source": detector,
            "score": float(values[index]),
        }
    candidates = []
    for source, threshold_key in (
        ("early_safe", "early_safe"),
        ("late_time", "late_time"),
    ):
        steps, values = view[source]
        crossings = np.flatnonzero(values >= float(thresholds[threshold_key]))
        if len(crossings):
            index = int(crossings[0])
            candidates.append(
                (int(steps[index]), source, float(values[index]))
            )
    if not candidates:
        return None
    step, source, score = min(candidates, key=lambda item: (item[0], item[1]))
    return {"step": step, "source": source, "score": score}


def _maximum(record, detector, safe_end, time_start):
    view = _window(record, detector, safe_end, time_start)
    if detector != "staged_safe_time":
        return float(np.max(view[1]))
    values = [
        float(np.max(stage_values))
        for _, stage_values in view.values()
        if len(stage_values)
    ]
    return max(values) if values else -math.inf


def wilson_interval(successes, total, confidence=0.95):
    if total <= 0:
        return [None, None]
    # 1.959963984540054 is scipy.stats.norm.ppf(0.975), kept local so these
    # deployment utilities do not require scipy.
    z = 1.959963984540054 if math.isclose(confidence, 0.95) else 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def event_metrics(
    records,
    detector,
    thresholds,
    *,
    safe_end=0.25,
    time_start=0.50,
):
    from sklearn.metrics import average_precision_score, roc_auc_score

    safe_end, time_start = validate_stage_windows(safe_end, time_start)
    labels = np.asarray([int(bool(row["failed"])) for row in records], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Event metrics require successes and failures")
    alerts = [
        first_alert(row, detector, thresholds, safe_end, time_start)
        for row in records
    ]
    predicted = np.asarray([alert is not None for alert in alerts], dtype=bool)
    tp = int(np.sum((labels == 1) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    adjusted = []
    detected = []
    recalls = {0.10: 0, 0.25: 0, 0.50: 0}
    sources = {"safe_only": 0, "time_only": 0, "early_safe": 0, "late_time": 0}
    for row, alert in zip(records, alerts):
        if alert is not None:
            sources[alert["source"]] += 1
        if not bool(row["failed"]):
            continue
        if alert is None:
            adjusted.append(1.0)
            continue
        fraction = min(1.0, int(alert["step"]) / int(row["horizon"]))
        detected.append(float(fraction))
        adjusted.append(float(fraction))
        for landmark in recalls:
            recalls[landmark] += int(fraction <= landmark + 1e-12)
    failures = int(np.sum(labels == 1))
    if detector == "staged_safe_time":
        event_scores = []
        for row in records:
            view = _window(row, detector, safe_end, time_start)
            margins = []
            for source in ("early_safe", "late_time"):
                values = view[source][1]
                if len(values):
                    margins.append(
                        float(np.max(values)) - float(thresholds[source])
                    )
            event_scores.append(max(margins) if margins else -1.0)
    else:
        event_scores = [
            _maximum(row, detector, safe_end, time_start) for row in records
        ]
    return {
        "detector": detector,
        "rollouts": len(records),
        "successes": int(np.sum(labels == 0)),
        "failures": failures,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "roc_auc": float(roc_auc_score(labels, event_scores)),
        "average_precision": float(average_precision_score(labels, event_scores)),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "true_positive_rate": float(tpr),
        "false_positive_rate": float(fpr),
        "tpr_wilson_95": wilson_interval(tp, tp + fn),
        "fpr_wilson_95": wilson_interval(fp, fp + tn),
        "balanced_accuracy": float(0.5 * (tpr + 1.0 - fpr)),
        "mean_detected_failure_fraction": (
            float(np.mean(detected)) if detected else None
        ),
        "missed_failure_adjusted_detection_fraction": float(np.mean(adjusted)),
        "failure_recall_by_landmark": {
            f"{landmark:g}": count / failures for landmark, count in recalls.items()
        },
        "alarm_source_counts": sources,
    }


def select_single_threshold(
    records,
    detector,
    *,
    target_fpr=0.05,
    safe_end=0.25,
    time_start=0.50,
):
    if detector not in {"safe_only", "time_only"}:
        raise ValueError("Single threshold selection supports SAFE-only or time-only")
    values = [_maximum(row, detector, safe_end, time_start) for row in records]
    feasible = []
    curve = []
    for threshold in _candidates(values):
        thresholds = {detector: float(threshold)}
        metrics = event_metrics(
            records,
            detector,
            thresholds,
            safe_end=safe_end,
            time_start=time_start,
        )
        curve.append(metrics)
        if metrics["false_positive_rate"] <= float(target_fpr) + 1e-12:
            rank = (
                metrics["true_positive_rate"],
                -metrics["missed_failure_adjusted_detection_fraction"],
                float(threshold),
            )
            feasible.append((rank, threshold, metrics))
    if not feasible:
        raise ValueError("No single-detector threshold satisfies the FPR cap")
    _, threshold, metrics = max(feasible, key=lambda item: item[0])
    return {
        "thresholds": {detector: float(threshold)},
        "validation_metrics": metrics,
        "feasible_thresholds": len(feasible),
        "curve": curve,
    }


def select_staged_thresholds(
    records,
    *,
    target_fpr=0.05,
    safe_end=0.25,
    time_start=0.50,
):
    early_values = []
    late_values = []
    for row in records:
        view = _window(row, "staged_safe_time", safe_end, time_start)
        early_values.append(
            float(np.max(view["early_safe"][1]))
            if len(view["early_safe"][1])
            else -1.0
        )
        late_values.append(
            float(np.max(view["late_time"][1]))
            if len(view["late_time"][1])
            else -1.0
        )
    feasible = []
    curve = []
    early_candidates = _candidates(early_values)
    late_candidates = _candidates(late_values)
    for early_threshold in early_candidates:
        for late_threshold in late_candidates:
            thresholds = {
                "early_safe": float(early_threshold),
                "late_time": float(late_threshold),
            }
            metrics = event_metrics(
                records,
                "staged_safe_time",
                thresholds,
                safe_end=safe_end,
                time_start=time_start,
            )
            if metrics["false_positive_rate"] <= float(target_fpr) + 1e-12:
                curve.append(metrics)
                rank = (
                    metrics["true_positive_rate"],
                    -metrics["missed_failure_adjusted_detection_fraction"],
                    float(early_threshold),
                    float(late_threshold),
                )
                feasible.append((rank, thresholds, metrics))
    if not feasible:
        raise ValueError("No staged threshold pair satisfies the FPR cap")
    _, thresholds, metrics = max(feasible, key=lambda item: item[0])
    return {
        "thresholds": thresholds,
        "validation_metrics": metrics,
        "search": {
            "early_candidates": int(len(early_candidates)),
            "late_candidates": int(len(late_candidates)),
            "candidate_pairs": int(len(early_candidates) * len(late_candidates)),
            "feasible_pairs": len(feasible),
        },
        "curve": curve,
    }


def paired_failure_comparison(
    records,
    detector_a,
    thresholds_a,
    detector_b,
    thresholds_b,
    *,
    safe_end=0.25,
    time_start=0.50,
):
    counts = {"earlier": 0, "same": 0, "later": 0, "a_missed": 0, "b_missed": 0}
    differences = []
    for row in records:
        if not bool(row["failed"]):
            continue
        a = first_alert(row, detector_a, thresholds_a, safe_end, time_start)
        b = first_alert(row, detector_b, thresholds_b, safe_end, time_start)
        a_fraction = 1.0 if a is None else min(1.0, a["step"] / row["horizon"])
        b_fraction = 1.0 if b is None else min(1.0, b["step"] / row["horizon"])
        differences.append(float(a_fraction - b_fraction))
        if a is None:
            counts["a_missed"] += 1
        if b is None:
            counts["b_missed"] += 1
        if a_fraction < b_fraction - 1e-12:
            counts["earlier"] += 1
        elif a_fraction > b_fraction + 1e-12:
            counts["later"] += 1
        else:
            counts["same"] += 1
    return {
        **counts,
        "mean_a_minus_b_detection_fraction": float(np.mean(differences)),
    }
