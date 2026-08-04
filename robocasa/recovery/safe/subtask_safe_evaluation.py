"""Leakage-safe evaluation primitives for semantic Subtask-SAFE segments."""

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


def parent_id(record):
    """Return the independent sampling unit for a score record."""
    return record.get("parent_rollout_id") or record["rollout_id"]


def validate_parent_groups(records):
    """Validate that a parent never crosses the fixed train/test boundary."""
    grouped = defaultdict(list)
    for record in records:
        grouped[parent_id(record)].append(record)
    for key, values in grouped.items():
        splits = {value["split"] for value in values}
        if len(splits) != 1:
            raise ValueError(
                f"Parent rollout {key} crosses score splits: {sorted(splits)}"
            )
        parent_tasks = {
            value.get("parent_task_name") or value["task_name"] for value in values
        }
        if len(parent_tasks) != 1:
            raise ValueError(f"Parent rollout {key} has inconsistent task provenance")
        parent_outcomes = {
            bool(value.get("parent_rollout_failed", value["failed"]))
            for value in values
        }
        if len(parent_outcomes) != 1:
            raise ValueError(f"Parent rollout {key} has inconsistent outcome provenance")
    return grouped


def _parent_identity(values):
    first = values[0]
    return (
        first.get("parent_task_name") or first["task_name"],
        bool(first.get("parent_rollout_failed", first["failed"])),
    )


def _stratified_parent_sample(parents, fraction, seed):
    if not 0.0 < fraction < 1.0:
        raise ValueError("Calibration parent fraction must lie strictly in (0, 1)")
    strata = defaultdict(list)
    for key, values in parents.items():
        if any(not value["failed"] for value in values):
            strata[_parent_identity(values)].append(key)
    selected = []
    per_stratum = {}
    for identity, keys in sorted(strata.items()):
        keys = sorted(keys)
        random.Random(f"{seed}:{identity[0]}:{int(identity[1])}").shuffle(keys)
        if len(keys) < 2:
            count = 0
        else:
            count = min(len(keys) - 1, max(1, int(round(len(keys) * fraction))))
        chosen = keys[:count]
        selected.extend(chosen)
        per_stratum[f"{identity[0]}|failed={int(identity[1])}"] = {
            "candidate_parents": len(keys),
            "calibration_parents": len(chosen),
            "evaluation_parents": len(keys) - len(chosen),
        }
    if len(selected) < 2:
        raise ValueError("Parent-grouped calibration needs at least two parent rollouts")
    return sorted(selected), per_stratum


def parent_grouped_calibration_split(
    records,
    *,
    parent_fraction,
    split_seed,
    reference_fraction,
    conformal_seed,
):
    """Partition complete test parents into calibration or evaluation.

    Only successful segments from calibration parents are used to fit the
    functional threshold. Failure segments from those parents are deliberately
    consumed but excluded, so no segment from a calibration parent can leak
    into threshold evaluation.
    """
    grouped = validate_parent_groups(records)
    train_parents = {
        key: values for key, values in grouped.items() if values[0]["split"] == "train"
    }
    test_parents = {
        key: values for key, values in grouped.items() if values[0]["split"] == "test"
    }
    calibration_parents, per_stratum = _stratified_parent_sample(
        test_parents, parent_fraction, split_seed
    )
    calibration_parent_set = set(calibration_parents)
    evaluation_parents = sorted(set(test_parents) - calibration_parent_set)
    if not evaluation_parents:
        raise ValueError("Parent-grouped calibration left no evaluation parents")

    ordered = list(calibration_parents)
    random.Random(f"reference:{conformal_seed}").shuffle(ordered)
    reference_size = int(round(len(ordered) * reference_fraction))
    reference_size = min(len(ordered) - 1, max(1, reference_size))
    reference_parents = sorted(ordered[:reference_size])
    conformal_parents = sorted(ordered[reference_size:])
    reference_parent_set = set(reference_parents)
    conformal_parent_set = set(conformal_parents)

    train_records = [value for key in train_parents for value in train_parents[key]]
    calibration_records = [
        value
        for key in calibration_parents
        for value in test_parents[key]
    ]
    evaluation_records = [
        value for key in evaluation_parents for value in test_parents[key]
    ]
    reference_successes = [
        value
        for value in calibration_records
        if parent_id(value) in reference_parent_set and not value["failed"]
    ]
    conformal_successes = [
        value
        for value in calibration_records
        if parent_id(value) in conformal_parent_set and not value["failed"]
    ]
    if not reference_successes or not conformal_successes:
        raise ValueError(
            "Parent reference and nonconformity groups must both contain successes"
        )

    tasks = sorted({value["task_name"] for value in records})
    per_task = {}
    for task in tasks:
        task_train = [value for value in train_records if value["task_name"] == task]
        task_cal = [value for value in calibration_records if value["task_name"] == task]
        task_eval = [value for value in evaluation_records if value["task_name"] == task]
        per_task[task] = {
            "train_successes": sum(not value["failed"] for value in task_train),
            "train_failures": sum(value["failed"] for value in task_train),
            "calibration_successes": sum(not value["failed"] for value in task_cal),
            "calibration_failures_excluded": sum(value["failed"] for value in task_cal),
            "evaluation_successes": sum(not value["failed"] for value in task_eval),
            "evaluation_failures": sum(value["failed"] for value in task_eval),
            "calibration_parent_ids": sorted(
                {parent_id(value) for value in task_cal}
            ),
        }

    calibration_successes = [value for value in calibration_records if not value["failed"]]
    calibration_failures = [value for value in calibration_records if value["failed"]]
    train_ids = sorted(value["rollout_id"] for value in train_records)
    calibration_success_ids = sorted(
        value["rollout_id"] for value in calibration_successes
    )
    evaluation_ids = sorted(value["rollout_id"] for value in evaluation_records)
    return {
        "schema_version": 2,
        "protocol": (
            "training-only task early-score normalization; complete parent-rollout "
            "calibration assignment; successful calibration segments only; remaining "
            "parents used for evaluation"
        ),
        "split_unit": "parent_rollout",
        "split_seed": int(split_seed),
        "conformal_seed": int(conformal_seed),
        "calibration_parent_fraction": float(parent_fraction),
        "official_reference_fraction": float(reference_fraction),
        "task_names": tasks,
        "counts": {
            "train": len(train_records),
            "train_parents": len(train_parents),
            "calibration_parents": len(calibration_parents),
            "calibration_successes": len(calibration_successes),
            "calibration_failures_excluded": len(calibration_failures),
            "calibration_reference_parents": len(reference_parents),
            "calibration_reference_successes": len(reference_successes),
            "calibration_nonconformity_parents": len(conformal_parents),
            "calibration_nonconformity_successes": len(conformal_successes),
            "evaluation_parents": len(evaluation_parents),
            "evaluation": len(evaluation_records),
            "evaluation_successes": sum(not value["failed"] for value in evaluation_records),
            "evaluation_failures": sum(value["failed"] for value in evaluation_records),
        },
        "per_parent_stratum": per_stratum,
        "per_task": per_task,
        "train_ids": train_ids,
        "train_parent_ids": sorted(train_parents),
        "calibration_success_ids": calibration_success_ids,
        "calibration_failure_ids_excluded": sorted(
            value["rollout_id"] for value in calibration_failures
        ),
        "calibration_parent_ids": calibration_parents,
        "calibration_reference_ids": sorted(
            value["rollout_id"] for value in reference_successes
        ),
        "calibration_reference_parent_ids": reference_parents,
        "calibration_nonconformity_ids": sorted(
            value["rollout_id"] for value in conformal_successes
        ),
        "calibration_nonconformity_parent_ids": conformal_parents,
        "evaluation_ids": evaluation_ids,
        "evaluation_parent_ids": evaluation_parents,
    }


def fit_elapsed_hazard(records, score_length):
    """Fit P(failure | semantic subtask still active at inference k)."""
    grouped = defaultdict(list)
    for record in records:
        if record["split"] == "train":
            grouped[record["task_name"]].append(record)
    if not grouped:
        raise ValueError("Elapsed-time baseline needs training score records")
    model = {}
    for task, values in sorted(grouped.items()):
        maximum = max(score_length(value) for value in values)
        hazards = []
        support = []
        for step in range(1, maximum + 1):
            active = [value for value in values if score_length(value) >= step]
            failures = sum(bool(value["failed"]) for value in active)
            # Laplace smoothing prevents exact 0/1 thresholds in sparse subtasks.
            hazards.append(float((failures + 1) / (len(active) + 2)))
            support.append(len(active))
        model[task] = {
            "hazard": hazards,
            "active_training_segments": support,
            "num_training_segments": len(values),
            "source": "training segments only",
        }
    return model


def elapsed_hazard_scores(record, model, score_length):
    length = score_length(record)
    hazard = np.asarray(model[record["task_name"]]["hazard"], dtype=np.float64)
    if length <= len(hazard):
        values = hazard[:length]
    else:
        values = np.concatenate((hazard, np.repeat(hazard[-1], length - len(hazard))))
    # Risk accumulated from elapsed evidence should never decrease online.
    return np.maximum.accumulate(values)


def replace_with_elapsed_scores(records, model, score_length):
    return [
        {
            **record,
            "scores": elapsed_hazard_scores(record, model, score_length).tolist(),
            "score_source": "training_elapsed_subtask_hazard",
        }
        for record in records
    ]


def detection_event(record, detection_index):
    """Convert a causal detection index into environment-step lead time."""
    segment = record.get("subtask_safe_segment") or {}
    entry = segment.get("entry_environment_step")
    end = segment.get("end_environment_step")
    steps = record.get("inference_environment_steps") or []
    if detection_index is None or entry is None or end is None or not steps:
        return {
            "detected": detection_index is not None,
            "detection_environment_step": None,
            "lead_environment_steps": None,
            "normalized_lead_time": None,
        }
    index = min(int(detection_index), len(steps) - 1)
    detected_step = int(steps[index])
    duration = max(1, int(end) - int(entry))
    lead = max(0, int(end) - detected_step)
    return {
        "detected": True,
        "detection_environment_step": detected_step,
        "lead_environment_steps": lead,
        "normalized_lead_time": float(lead / duration),
    }
