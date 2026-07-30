"""Oracle-aligned subtask records for Subtask-SAFE collection and training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from robocasa.recovery.subtask_eval import build_subtask_trace


SUBTASK_SAFE_SCHEMA_VERSION = 1
SUBTASK_FAILURE_LABEL_SEMANTICS = (
    "active_subtask_eventually_fails_before_completion"
)


def _required_subtasks(
    subtask_evals: list[dict[str, Any] | None],
) -> list[str]:
    orders = [
        list(payload.get("required_predicates", []))
        for payload in subtask_evals
        if payload is not None
    ]
    if not orders or not orders[0]:
        raise ValueError("Subtask-SAFE requires a non-empty ordered subtask sequence")
    reference = orders[0]
    if any(order != reference for order in orders[1:]):
        raise ValueError("Ordered required subtasks changed within the rollout")
    if len(reference) != len(set(reference)):
        raise ValueError("Ordered required subtasks contain duplicates")
    return reference


def _transition_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions = []
    previous_signature = None
    for entry in trace:
        signature = (
            tuple(entry.get("ordered_completed_subtasks", [])),
            entry.get("ordered_current_subtask"),
            tuple(entry.get("failed_preconditions", [])),
        )
        noteworthy = bool(
            entry.get("ordered_newly_completed_subtasks")
            or entry.get("regressed_predicates")
        )
        if (
            previous_signature is None
            or signature != previous_signature
            or noteworthy
            or entry is trace[-1]
        ):
            transitions.append(
                {
                    "environment_step": int(entry["step"]),
                    "ordered_completed_subtasks": list(
                        entry.get("ordered_completed_subtasks", [])
                    ),
                    "ordered_current_subtask": entry.get(
                        "ordered_current_subtask"
                    ),
                    "ordered_newly_completed_subtasks": list(
                        entry.get("ordered_newly_completed_subtasks", [])
                    ),
                    "ordered_subtask_progress": float(
                        entry.get("ordered_subtask_progress", 0.0)
                    ),
                    "regressed_predicates": list(
                        entry.get("regressed_predicates", [])
                    ),
                    "failed_preconditions": list(
                        entry.get("failed_preconditions", [])
                    ),
                }
            )
        previous_signature = signature
    return transitions


def build_subtask_safe_record(
    subtask_evals: list[dict[str, Any] | None],
    inference_environment_steps: list[int],
    *,
    rollout_failed: bool,
    rollout_id: str | None = None,
) -> dict[str, Any]:
    """Align ordered oracle subtasks to real policy inferences and label segments.

    ``subtask_evals`` contains the reset state followed by the state after every
    environment action, so its index is the environment-step coordinate. A
    policy inference at environment step ``s`` is aligned to trace entry ``s``.
    """
    if not subtask_evals:
        raise ValueError("Subtask-SAFE received an empty subtask trace")
    unavailable = [
        index for index, payload in enumerate(subtask_evals) if payload is None
    ]
    if unavailable:
        preview = unavailable[:5]
        raise ValueError(
            "Subtask evaluation is unavailable at environment steps "
            f"{preview}{'...' if len(unavailable) > len(preview) else ''}"
        )
    inference_environment_steps = [
        int(step) for step in inference_environment_steps
    ]
    if inference_environment_steps != sorted(set(inference_environment_steps)):
        raise ValueError(
            "Subtask-SAFE inference environment steps must be strictly increasing"
        )
    if any(
        step < 0 or step >= len(subtask_evals)
        for step in inference_environment_steps
    ):
        raise ValueError("Subtask-SAFE inference step lies outside the trace")

    required_subtasks = _required_subtasks(subtask_evals)
    trace = build_subtask_trace(subtask_evals)
    valid_trace = [
        entry for entry in trace if entry.get("subtask_eval_available")
    ]
    final = valid_trace[-1]
    final_completed = list(final.get("ordered_completed_subtasks", []))
    terminal_active_subtask = final.get("ordered_current_subtask")
    if not rollout_failed and terminal_active_subtask is not None:
        raise ValueError(
            "Successful rollout ended with an incomplete ordered subtask: "
            f"{terminal_active_subtask}"
        )

    completion_steps: dict[str, int] = {}
    active_steps: dict[str, list[int]] = {
        name: [] for name in required_subtasks
    }
    for entry in valid_trace:
        step = int(entry["step"])
        current = entry.get("ordered_current_subtask")
        if current in active_steps:
            active_steps[current].append(step)
        for name in entry.get("ordered_newly_completed_subtasks", []):
            completion_steps.setdefault(name, step)

    subtask_indices = {
        name: index for index, name in enumerate(required_subtasks)
    }
    inference_records = []
    within_subtask_counts = {name: 0 for name in required_subtasks}
    inference_indices_by_subtask = {
        name: [] for name in required_subtasks
    }
    for inference_index, environment_step in enumerate(
        inference_environment_steps
    ):
        entry = trace[environment_step]
        current = entry.get("ordered_current_subtask")
        segment_index = subtask_indices.get(current)
        within_index = None
        if current in within_subtask_counts:
            within_index = within_subtask_counts[current]
            within_subtask_counts[current] += 1
            inference_indices_by_subtask[current].append(inference_index)
        inference_records.append(
            {
                "inference_index": inference_index,
                "environment_step": environment_step,
                "segment_index": segment_index,
                "ordered_current_subtask": current,
                "within_subtask_inference_index": within_index,
                "ordered_subtask_progress": float(
                    entry.get("ordered_subtask_progress", 0.0)
                ),
            }
        )

    completed_set = set(final_completed)
    segments = []
    previous_completion_step = 0
    for segment_index, name in enumerate(required_subtasks):
        completed = name in completed_set
        is_terminal_failure = bool(
            rollout_failed
            and not completed
            and name == terminal_active_subtask
        )
        entered = bool(active_steps[name] or completed or is_terminal_failure)
        if not entered:
            break
        entry_step = (
            min(active_steps[name])
            if active_steps[name]
            else previous_completion_step
        )
        completion_step = completion_steps.get(name)
        end_step = (
            completion_step
            if completion_step is not None
            else len(subtask_evals) - 1
        )
        indices = inference_indices_by_subtask[name]
        if indices and indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(
                f"Policy inferences for ordered subtask {name!r} are not contiguous"
            )
        failure_label = 1 if is_terminal_failure else 0 if completed else None
        segments.append(
            {
                "segment_index": segment_index,
                "segment_id": (
                    f"{rollout_id}:{segment_index}:{name}"
                    if rollout_id is not None
                    else f"{segment_index}:{name}"
                ),
                "subtask_name": name,
                "entry_environment_step": int(entry_step),
                "end_environment_step": int(end_step),
                "completion_environment_step": (
                    int(completion_step)
                    if completion_step is not None
                    else None
                ),
                "completed": completed,
                "eventually_failed": is_terminal_failure,
                "failure_label": failure_label,
                "inference_start_index": (
                    int(indices[0]) if indices else None
                ),
                "inference_end_index_exclusive": (
                    int(indices[-1] + 1) if indices else None
                ),
                "num_policy_inferences": len(indices),
                "usable_for_safe": bool(
                    failure_label is not None and indices
                ),
            }
        )
        if completion_step is not None:
            previous_completion_step = completion_step

    failure_segments = [
        segment for segment in segments if segment["failure_label"] == 1
    ]
    if len(failure_segments) > 1:
        raise AssertionError("A rollout cannot contain multiple failed active segments")
    labeling_status = "complete"
    if rollout_failed and not failure_segments:
        labeling_status = "failed_rollout_without_active_subtask"

    return {
        "schema_version": SUBTASK_SAFE_SCHEMA_VERSION,
        "rollout_id": rollout_id,
        "label_semantics": SUBTASK_FAILURE_LABEL_SEMANTICS,
        "ordering_semantics": "first_ordered_completion_monotonic",
        "trace_coordinate": (
            "state at reset is environment_step 0; state after action i is "
            "environment_step i+1"
        ),
        "rollout_failed": bool(rollout_failed),
        "num_environment_states": len(subtask_evals),
        "num_policy_inferences": len(inference_environment_steps),
        "required_subtasks": required_subtasks,
        "terminal_active_subtask": terminal_active_subtask,
        "labeling_status": labeling_status,
        "transitions": _transition_trace(trace),
        "inference_records": inference_records,
        "segments": segments,
    }


def validate_subtask_safe_record(
    record: dict[str, Any],
    *,
    rollout_id: str,
    rollout_failed: bool,
    inference_environment_steps: list[int],
) -> dict[str, int]:
    """Validate a serialized Subtask-SAFE record and return label counts."""
    if record.get("schema_version") != SUBTASK_SAFE_SCHEMA_VERSION:
        raise ValueError("Unsupported Subtask-SAFE schema version")
    if record.get("rollout_id") != rollout_id:
        raise ValueError("Subtask-SAFE rollout_id disagrees with manifest")
    if record.get("rollout_failed") != bool(rollout_failed):
        raise ValueError("Subtask-SAFE rollout failure label disagrees with manifest")
    if record.get("label_semantics") != SUBTASK_FAILURE_LABEL_SEMANTICS:
        raise ValueError("Unsupported Subtask-SAFE label semantics")
    inference_records = record.get("inference_records", [])
    if len(inference_records) != len(inference_environment_steps):
        raise ValueError("Subtask-SAFE inference count disagrees with manifest")
    actual_steps = [
        int(item["environment_step"]) for item in inference_records
    ]
    if actual_steps != list(inference_environment_steps):
        raise ValueError("Subtask-SAFE inference alignment disagrees with manifest")

    segments = record.get("segments", [])
    all_labels = [
        segment.get("failure_label")
        for segment in segments
        if segment.get("failure_label") is not None
    ]
    labels = [
        segment.get("failure_label")
        for segment in segments
        if segment.get("usable_for_safe")
    ]
    if any(label not in {0, 1} for label in all_labels):
        raise ValueError("Labeled Subtask-SAFE segments require binary labels")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("Usable Subtask-SAFE segments require binary labels")
    all_failure_count = sum(label == 1 for label in all_labels)
    if all_failure_count > 1:
        raise ValueError("Subtask-SAFE rollout contains multiple failed segments")
    if rollout_failed and all_failure_count != 1:
        raise ValueError(
            "Failed rollout must assign exactly one terminal active subtask"
        )
    if not rollout_failed and all_failure_count:
        raise ValueError("Successful rollout contains a failed subtask segment")
    return {
        "segments": len(segments),
        "usable_segments": len(labels),
        "successful_segments": sum(label == 0 for label in labels),
        "failed_segments": sum(label == 1 for label in labels),
        "labeled_without_inference": len(all_labels) - len(labels),
    }


def atomic_write_subtask_safe_record(
    path: str | Path, record: dict[str, Any]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return path
