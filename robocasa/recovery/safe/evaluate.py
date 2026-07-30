"""Metrics and CLI for seen/unseen raw SAFE evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

try:
    from .conformal import (
        calibrate_functional_threshold,
        first_detection,
        load_calibration,
    )
except ImportError:
    from conformal import (
        calibrate_functional_threshold,
        first_detection,
        load_calibration,
    )


def _safe_auc(function, labels, scores):
    return None if len(set(labels)) < 2 else float(function(labels, scores))


def evaluate_rollout_scores(rollouts, calibration):
    """Evaluate `{id, task_name, failed, scores}` rollout dictionaries."""
    labels = [int(item["failed"]) for item in rollouts]
    max_scores = [float(np.max(item["scores"])) for item in rollouts]
    detections = [first_detection(item["scores"], calibration) for item in rollouts]
    predicted = [int(index is not None) for index in detections]
    tp = sum(y and p for y, p in zip(labels, predicted))
    fp = sum((not y) and p for y, p in zip(labels, predicted))
    tn = sum((not y) and (not p) for y, p in zip(labels, predicted))
    fn = sum(y and (not p) for y, p in zip(labels, predicted))
    failures = max(1, sum(labels))
    successes = max(1, len(labels) - sum(labels))
    normalized_detection = [
        index / max(1, len(item["scores"]) - 1)
        for item, index in zip(rollouts, detections)
        if item["failed"] and index is not None
    ]
    return {
        "num_rollouts": len(rollouts),
        "num_successes": int(sum(not x for x in labels)),
        "num_failures": int(sum(labels)),
        "roc_auc": _safe_auc(roc_auc_score, labels, max_scores),
        "auprc": _safe_auc(average_precision_score, labels, max_scores),
        "true_positive_rate": tp / failures,
        "false_positive_rate": fp / successes,
        "true_negative_rate": tn / successes,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predicted))
            if len(set(labels)) == 2
            else None
        ),
        "false_alarms_per_successful_rollout": fp / successes,
        "failed_rollouts_detected_fraction": tp / failures,
        "normalized_detection_time": (
            float(np.mean(normalized_detection)) if normalized_detection else None
        ),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate_groups(rollouts, calibration):
    output = {"overall": evaluate_rollout_scores(rollouts, calibration), "per_task": {}}
    grouped = defaultdict(list)
    for rollout in rollouts:
        grouped[rollout["task_name"]].append(rollout)
    for task, items in sorted(grouped.items()):
        output["per_task"][task] = evaluate_rollout_scores(items, calibration)
    return output


def baseline_rollouts(rollouts, baseline):
    result = []
    for item in rollouts:
        length = len(item["scores"])
        if baseline == "constant":
            scores = [0.5] * length
        elif baseline == "time_only":
            scores = np.linspace(0.0, 1.0, length).tolist()
        else:
            raise ValueError(baseline)
        result.append({**item, "scores": scores})
    return result


def evaluate_experiment(score_records, calibration, model_name="safe"):
    """Evaluate model and controls only on seen/unseen test splits."""
    alpha = calibration["alpha"]
    reference = [x for x in score_records if x["split"] == "train" and not x["failed"]]
    calibration_records = [
        x for x in score_records if x["split"] == "calibration" and not x["failed"]
    ]
    if not reference or not calibration_records:
        raise ValueError("Need successful training and calibration rollouts for baselines")
    baseline_calibrations = {}
    for baseline in ("constant", "time_only"):
        reference_baseline = baseline_rollouts(reference, baseline)
        calibration_baseline = baseline_rollouts(calibration_records, baseline)
        baseline_calibrations[baseline] = calibrate_functional_threshold(
            [x["scores"] for x in reference_baseline],
            [x["scores"] for x in calibration_baseline],
            alpha=alpha,
            normalized_length=calibration["normalized_length"],
            modulation=calibration["modulation"],
        )
    output = {
        "schema_version": 1,
        "alpha": alpha,
        "normalized_time_evaluation": True,
        "models": {},
        "rollout_length_leakage": {},
        "dataset_summary": {},
    }
    eval_records = [
        x for x in score_records if x["split"] in ("seen_test", "unseen_test")
    ]
    for split in ("train", "calibration", "seen_test", "unseen_test"):
        items = [x for x in score_records if x["split"] == split]
        lengths = [x["num_env_steps"] for x in items]
        inferences = [x["num_inferences"] for x in items]
        output["dataset_summary"][split] = {
            "num_rollouts": len(items),
            "num_successes": sum(not x["failed"] for x in items),
            "num_failures": sum(x["failed"] for x in items),
            "tasks": sorted({x["task_name"] for x in items}),
            "seeds": sorted({x.get("environment_seed") for x in items}),
            "env_steps_mean": float(np.mean(lengths)) if lengths else None,
            "env_steps_range": [min(lengths), max(lengths)] if lengths else None,
            "inference_calls_mean": float(np.mean(inferences)) if inferences else None,
            "inference_calls_range": [min(inferences), max(inferences)] if inferences else None,
        }
    output["policy_ids"] = sorted({x.get("policy_id") for x in score_records})
    output["checkpoints"] = sorted({x.get("checkpoint") for x in score_records})
    output["feature_shapes"] = sorted({str(x.get("feature_shape")) for x in score_records})
    labels = [int(x["failed"]) for x in eval_records]
    durations = [int(x["num_env_steps"]) for x in eval_records]
    duration_auc = _safe_auc(roc_auc_score, labels, durations)
    output["rollout_length_leakage"] = {
        "duration_only_roc_auc": duration_auc,
        "warning": (
            "Rollout duration alone is unusually predictive; interpret SAFE with caution."
            if duration_auc is not None and (duration_auc >= 0.75 or duration_auc <= 0.25)
            else None
        ),
    }
    for name in (model_name, "constant", "time_only"):
        source = score_records if name == model_name else baseline_rollouts(score_records, name)
        threshold = calibration if name == model_name else baseline_calibrations[name]
        output["models"][name] = {}
        for split in ("seen_test", "unseen_test"):
            selected = [x for x in source if x["split"] == split]
            output["models"][name][split] = (
                evaluate_groups(selected, threshold) if selected else None
            )
    return output


def save_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="JSON file of rollout score trajectories")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="safe")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rollouts = json.loads(Path(args.scores).read_text())
    calibration = load_calibration(args.calibration)
    save_results(evaluate_experiment(rollouts, calibration, args.model_name), args.output)


if __name__ == "__main__":
    main()
