"""Oracle-aligned semantic subtask records for Subtask-SAFE."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from robocasa.recovery.create_recovery_failure_dataset import (
    mapped_subtask_sequence,
)
from robocasa.recovery.subtask_eval import summarize_subtask_rollout


SUBTASK_SAFE_SCHEMA_VERSION = 3
SUBTASK_FAILURE_LABEL_SEMANTICS = "active_subtask_eventually_fails_before_completion"
SUBTASK_DEFINITION_SOURCE = (
    "robocasa.recovery.create_recovery_failure_dataset:"
    "mapped_subtask_sequence observation-safe semantic normalization"
)


def _valid_evals(
    subtask_evals: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    unavailable = [
        index for index, payload in enumerate(subtask_evals) if payload is None
    ]
    if unavailable:
        preview = unavailable[:5]
        raise ValueError(
            "Subtask evaluation is unavailable at environment steps "
            f"{preview}{'...' if len(unavailable) > len(preview) else ''}"
        )
    return [payload for payload in subtask_evals if payload is not None]


def _task_name(
    subtask_evals: list[dict[str, Any]],
    explicit_task_name: str | None,
) -> str:
    observed = {
        str(payload["task_name"])
        for payload in subtask_evals
        if payload.get("task_name")
    }
    if len(observed) > 1:
        raise ValueError("Subtask task_name changed within the rollout")
    if explicit_task_name is not None:
        if observed and observed != {explicit_task_name}:
            raise ValueError(
                "Explicit task_name disagrees with subtask-evaluation payload"
            )
        return explicit_task_name
    if observed:
        return next(iter(observed))
    raise ValueError(
        "Semantic Subtask-SAFE requires task_name in the collection call or "
        "subtask-evaluation payload"
    )


def _validate_predicate_contract(
    subtask_evals: list[dict[str, Any]],
) -> None:
    orders = [list(payload.get("required_predicates", [])) for payload in subtask_evals]
    if not orders or not orders[0]:
        raise ValueError("Semantic Subtask-SAFE requires non-empty runtime predicates")
    if any(order != orders[0] for order in orders[1:]):
        raise ValueError("Runtime required predicates changed within the rollout")


def _semantic_subtasks(
    task_name: str,
    subtask_evals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = summarize_subtask_rollout(
        subtask_evals,
        include_trace=False,
    )
    sequence = mapped_subtask_sequence(summary, task_name)
    predicate_contract = subtask_evals[-1].get("predicates", {})
    if not predicate_contract:
        raise ValueError("Semantic Subtask-SAFE requires runtime predicate metadata")
    definitions = []
    for entry in sequence:
        subtask_id = str(entry.get("subtask_id") or "").strip()
        instruction = str(entry.get("instruction") or "").strip()
        predicate_names = [
            str(name) for name in entry.get("predicate_names", []) if name
        ]
        if not subtask_id or not instruction or not predicate_names:
            raise ValueError(
                f"Invalid semantic subtask definition for {task_name}: {entry}"
            )
        definitions.append(
            {
                "subtask_index": len(definitions),
                "subtask_id": subtask_id,
                "instruction": instruction,
                "predicate_names": predicate_names,
                "source_subtask_ids": list(
                    entry.get("source_subtask_ids") or [subtask_id]
                ),
                "required_for_official_success": any(
                    bool(predicate_contract.get(name, {}).get("required", True))
                    for name in predicate_names
                ),
            }
        )
    if not definitions:
        raise ValueError(f"No semantic subtask sequence is available for {task_name}")
    ids = [definition["subtask_id"] for definition in definitions]
    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Semantic subtask identifiers contain duplicates for {task_name}"
        )
    available_predicates = {
        name for payload in subtask_evals for name in payload.get("predicates", {})
    }
    missing = sorted(
        {
            name
            for definition in definitions
            for name in definition["predicate_names"]
            if name not in available_predicates
        }
    )
    if missing:
        raise ValueError(
            f"Semantic subtasks reference unavailable predicates: {missing}"
        )
    return definitions


def _predicate_values(payload: dict[str, Any]) -> dict[str, bool]:
    return {
        name: bool(predicate.get("value", False))
        for name, predicate in payload.get("predicates", {}).items()
    }


def _failed_preconditions(payload: dict[str, Any]) -> list[str]:
    return sorted(
        name
        for name, predicate in payload.get("predicates", {}).items()
        if predicate.get("stage") == "precondition"
        and not predicate.get("value", False)
    )


def _first_terminally_unsatisfied_subtask(
    payload: dict[str, Any],
    definitions: list[dict[str, Any]],
) -> tuple[str | None, list[str]]:
    """Find the earliest completed semantic unit whose postcondition regressed.

    Ordered completion is intentionally monotonic so a later manipulation does
    not erase genuine progress.  An unsuccessful rollout can therefore finish
    after every semantic unit was observed complete, while one of those units'
    required predicates is false at the terminal state.  Optional transient
    units, such as grasp predicates that normally become false after release,
    cannot explain official task failure and are skipped.  The earliest
    terminally-unsatisfied required unit is the ordered subtask whose success
    did not persist until overall task completion.
    """
    values = _predicate_values(payload)
    for definition in definitions:
        if not definition["required_for_official_success"]:
            continue
        unsatisfied = [
            name
            for name in definition["predicate_names"]
            if not values.get(name, False)
        ]
        if unsatisfied:
            return definition["subtask_id"], unsatisfied
    return None, []


def _build_semantic_trace(
    subtask_evals: list[dict[str, Any]],
    definitions: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    dict[str, str],
    dict[str, list[int]],
    dict[str, int],
]:
    trace = []
    completion_steps: dict[str, int] = {}
    completion_evidence: dict[str, str] = {}
    active_steps = {definition["subtask_id"]: [] for definition in definitions}
    completed: list[str] = []
    bypassed: list[str] = []
    bypass_steps: dict[str, int] = {}
    ordered_index = 0
    previous_predicate_values: dict[str, bool] = {}
    for environment_step, payload in enumerate(subtask_evals):
        values = _predicate_values(payload)
        newly_completed = []
        newly_inferred_completed = []
        newly_bypassed = []
        while ordered_index < len(definitions):
            definition = definitions[ordered_index]
            predicates_satisfied = all(
                values.get(name, False) for name in definition["predicate_names"]
            )
            if predicates_satisfied:
                subtask_id = definition["subtask_id"]
                completed.append(subtask_id)
                newly_completed.append(subtask_id)
                completion_steps[subtask_id] = environment_step
                completion_evidence[subtask_id] = "predicate_observed"
                ordered_index += 1
                continue
            next_definition = (
                definitions[ordered_index + 1]
                if ordered_index + 1 < len(definitions)
                else None
            )
            next_satisfied = bool(
                next_definition is not None
                and all(
                    values.get(name, False)
                    for name in next_definition["predicate_names"]
                )
            )
            if definition["required_for_official_success"] and next_satisfied:
                # Some procedural predicates are deliberately stricter than the
                # official task (for example, a cabinet-open threshold).  If the
                # immediately following ordered outcome is observed, that is
                # causal evidence that the current operation was completed well
                # enough to proceed.  Preserve the inference explicitly instead
                # of either blocking the complete trace or fabricating a direct
                # predicate observation.
                subtask_id = definition["subtask_id"]
                completed.append(subtask_id)
                newly_completed.append(subtask_id)
                newly_inferred_completed.append(subtask_id)
                completion_steps[subtask_id] = environment_step
                completion_evidence[subtask_id] = "downstream_subtask_observed"
                ordered_index += 1
                continue
            if not definition["required_for_official_success"] and next_satisfied:
                subtask_id = definition["subtask_id"]
                bypassed.append(subtask_id)
                newly_bypassed.append(subtask_id)
                bypass_steps[subtask_id] = environment_step
                ordered_index += 1
                continue
            if payload.get("task_success", False):
                subtask_id = definition["subtask_id"]
                if definition["required_for_official_success"]:
                    # Official task completion is authoritative evidence for a
                    # remaining required semantic outcome when a stricter
                    # diagnostic proxy (commonly release distance) never fires.
                    completed.append(subtask_id)
                    newly_completed.append(subtask_id)
                    newly_inferred_completed.append(subtask_id)
                    completion_steps[subtask_id] = environment_step
                    completion_evidence[subtask_id] = "official_task_success"
                else:
                    bypassed.append(subtask_id)
                    newly_bypassed.append(subtask_id)
                    bypass_steps[subtask_id] = environment_step
                ordered_index += 1
                continue
            break
        current = (
            definitions[ordered_index] if ordered_index < len(definitions) else None
        )
        if current is not None:
            active_steps[current["subtask_id"]].append(environment_step)
        regressed = sorted(
            name
            for name, value in values.items()
            if previous_predicate_values.get(name, False) and not value
        )
        trace.append(
            {
                "environment_step": environment_step,
                "completed_subtask_ids": list(completed),
                "bypassed_optional_subtask_ids": list(bypassed),
                "current_subtask_id": (
                    current["subtask_id"] if current is not None else None
                ),
                "current_subtask_instruction": (
                    current["instruction"] if current is not None else None
                ),
                "current_subtask_predicate_names": (
                    list(current["predicate_names"]) if current is not None else []
                ),
                "newly_completed_subtask_ids": newly_completed,
                "newly_inferred_completed_subtask_ids": newly_inferred_completed,
                "newly_bypassed_optional_subtask_ids": newly_bypassed,
                "subtask_progress": float(
                    (len(completed) + len(bypassed)) / len(definitions)
                ),
                "regressed_predicates": regressed,
                "failed_preconditions": _failed_preconditions(payload),
                "task_success": bool(payload.get("task_success", False)),
            }
        )
        previous_predicate_values = values
    return (
        trace,
        completion_steps,
        completion_evidence,
        active_steps,
        bypass_steps,
    )


def _transition_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions = []
    previous_signature = None
    for entry in trace:
        signature = (
            tuple(entry["completed_subtask_ids"]),
            entry["current_subtask_id"],
            tuple(entry["failed_preconditions"]),
        )
        noteworthy = bool(
            entry["newly_completed_subtask_ids"]
            or entry["newly_bypassed_optional_subtask_ids"]
            or entry["regressed_predicates"]
        )
        if (
            previous_signature is None
            or signature != previous_signature
            or noteworthy
            or entry is trace[-1]
        ):
            transitions.append(dict(entry))
        previous_signature = signature
    return transitions


def build_subtask_safe_record(
    subtask_evals: list[dict[str, Any] | None],
    inference_environment_steps: list[int],
    *,
    rollout_failed: bool,
    rollout_id: str | None = None,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Align ordered natural-language subtasks to genuine policy inferences."""
    if not subtask_evals:
        raise ValueError("Subtask-SAFE received an empty subtask trace")
    valid_evals = _valid_evals(subtask_evals)
    task_name = _task_name(valid_evals, task_name)
    _validate_predicate_contract(valid_evals)
    definitions = _semantic_subtasks(task_name, valid_evals)

    inference_environment_steps = [int(step) for step in inference_environment_steps]
    if inference_environment_steps != sorted(set(inference_environment_steps)):
        raise ValueError(
            "Subtask-SAFE inference environment steps must be strictly increasing"
        )
    if any(
        step < 0 or step >= len(valid_evals) for step in inference_environment_steps
    ):
        raise ValueError("Subtask-SAFE inference step lies outside the trace")

    (
        trace,
        completion_steps,
        completion_evidence,
        active_steps,
        bypass_steps,
    ) = _build_semantic_trace(valid_evals, definitions)
    final = trace[-1]
    final_completed = set(final["completed_subtask_ids"])
    final_bypassed = set(final["bypassed_optional_subtask_ids"])
    terminal_active_subtask = final["current_subtask_id"]
    definition_by_id = {
        definition["subtask_id"]: definition for definition in definitions
    }
    terminal_failure_reason = None
    terminal_unsatisfied_predicates: list[str] = []
    if not rollout_failed and terminal_active_subtask is not None:
        raise ValueError(
            "Successful rollout ended with an incomplete semantic subtask: "
            f"{terminal_active_subtask}"
        )
    if rollout_failed:
        if terminal_active_subtask is None:
            (
                terminal_active_subtask,
                terminal_unsatisfied_predicates,
            ) = _first_terminally_unsatisfied_subtask(
                valid_evals[-1],
                definitions,
            )
            terminal_failure_reason = (
                "completed_subtask_regressed_before_task_completion"
            )
        else:
            terminal_unsatisfied_predicates = [
                name
                for name in definition_by_id[terminal_active_subtask]["predicate_names"]
                if not _predicate_values(valid_evals[-1]).get(name, False)
            ]
            terminal_failure_reason = "active_subtask_never_completed"
        if terminal_active_subtask is None:
            raise ValueError(
                "Failed rollout has no active or terminally regressed semantic "
                "subtask; the semantic mapping does not explain the official "
                "task failure"
            )

    within_subtask_counts = {definition["subtask_id"]: 0 for definition in definitions}
    inference_indices_by_subtask = {
        definition["subtask_id"]: [] for definition in definitions
    }
    inference_records = []
    for inference_index, environment_step in enumerate(inference_environment_steps):
        entry = trace[environment_step]
        subtask_id = entry["current_subtask_id"]
        definition = definition_by_id[subtask_id] if subtask_id is not None else None
        within_index = None
        if definition is not None:
            within_index = within_subtask_counts[subtask_id]
            within_subtask_counts[subtask_id] += 1
            inference_indices_by_subtask[subtask_id].append(inference_index)
        inference_records.append(
            {
                "inference_index": inference_index,
                "environment_step": environment_step,
                "segment_index": (
                    definition["subtask_index"] if definition is not None else None
                ),
                "subtask_id": subtask_id,
                "subtask_instruction": (
                    definition["instruction"] if definition is not None else None
                ),
                "predicate_names": (
                    list(definition["predicate_names"])
                    if definition is not None
                    else []
                ),
                "source_subtask_ids": (
                    list(definition["source_subtask_ids"])
                    if definition is not None
                    else []
                ),
                "within_subtask_inference_index": within_index,
                "subtask_progress": float(entry["subtask_progress"]),
            }
        )

    segments = []
    excluded_completed_subtasks = []
    excluded_bypassed_subtasks = []
    for definition in definitions:
        subtask_index = definition["subtask_index"]
        subtask_id = definition["subtask_id"]
        observed_completed = subtask_id in final_completed
        is_terminal_failure = bool(
            rollout_failed and subtask_id == terminal_active_subtask
        )
        completed = observed_completed and not is_terminal_failure
        observed_active = bool(active_steps[subtask_id])
        if subtask_id in final_bypassed:
            excluded_bypassed_subtasks.append(
                {
                    **definition,
                    "entry_environment_step": (
                        int(min(active_steps[subtask_id])) if observed_active else None
                    ),
                    "bypass_environment_step": int(bypass_steps[subtask_id]),
                    "num_policy_inferences": len(
                        inference_indices_by_subtask[subtask_id]
                    ),
                    "reason": (
                        "optional_transient_unobserved_before_following_"
                        "subtask_completion"
                    ),
                }
            )
            continue
        if completed and not observed_active:
            excluded_completed_subtasks.append(
                {
                    **definition,
                    "completion_environment_step": int(completion_steps[subtask_id]),
                    "reason": ("completed_without_observed_active_environment_state"),
                }
            )
            continue
        if not observed_active and not is_terminal_failure:
            break

        completion_step = completion_steps.get(subtask_id)
        evidence = completion_evidence.get(subtask_id)
        observed_completion_step = (
            completion_step if evidence == "predicate_observed" else None
        )
        entry_step = (
            min(active_steps[subtask_id])
            if observed_active
            else int(observed_completion_step or 0)
        )
        completion_step = None if is_terminal_failure else completion_step
        end_step = (
            len(valid_evals) - 1
            if is_terminal_failure
            else completion_step
            if completion_step is not None
            else len(valid_evals) - 1
        )
        indices = inference_indices_by_subtask[subtask_id]
        if indices and indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(
                "Policy inferences for semantic subtask "
                f"{subtask_id!r} are not contiguous"
            )
        failure_label = 1 if is_terminal_failure else 0 if completed else None
        segments.append(
            {
                "segment_index": subtask_index,
                "segment_id": (
                    f"{rollout_id}:{subtask_index}:{subtask_id}"
                    if rollout_id is not None
                    else f"{subtask_index}:{subtask_id}"
                ),
                "subtask_id": subtask_id,
                "subtask_name": subtask_id,
                "subtask_instruction": definition["instruction"],
                "predicate_names": list(definition["predicate_names"]),
                "source_subtask_ids": list(definition["source_subtask_ids"]),
                "entry_environment_step": int(entry_step),
                "end_environment_step": int(end_step),
                "completion_environment_step": (
                    int(completion_step) if completion_step is not None else None
                ),
                "first_observed_completion_environment_step": (
                    int(observed_completion_step)
                    if observed_completion_step is not None
                    else None
                ),
                "completion_evidence": (
                    evidence if not is_terminal_failure else None
                ),
                "completed": completed,
                "eventually_failed": is_terminal_failure,
                "terminal_failure_reason": (
                    terminal_failure_reason if is_terminal_failure else None
                ),
                "terminal_unsatisfied_predicate_names": (
                    list(terminal_unsatisfied_predicates) if is_terminal_failure else []
                ),
                "failure_label": failure_label,
                "inference_start_index": (int(indices[0]) if indices else None),
                "inference_end_index_exclusive": (
                    int(indices[-1] + 1) if indices else None
                ),
                "num_policy_inferences": len(indices),
                "usable_for_safe": bool(failure_label is not None and indices),
            }
        )

    failure_segments = [
        segment for segment in segments if segment["failure_label"] == 1
    ]
    if rollout_failed and len(failure_segments) != 1:
        raise ValueError(
            "Failed rollout must create exactly one active semantic failure"
        )
    if not rollout_failed and failure_segments:
        raise ValueError("Successful rollout created a failed semantic subtask")

    terminal_definition = (
        definition_by_id[terminal_active_subtask]
        if terminal_active_subtask is not None
        else None
    )
    return {
        "schema_version": SUBTASK_SAFE_SCHEMA_VERSION,
        "rollout_id": rollout_id,
        "task_name": task_name,
        "semantic_layer": "ordered_natural_language_subtask",
        "subtask_definition_source": SUBTASK_DEFINITION_SOURCE,
        "label_semantics": SUBTASK_FAILURE_LABEL_SEMANTICS,
        "ordering_semantics": (
            "first_ordered_completion_monotonic_with_explicit_downstream_implication"
        ),
        "trace_coordinate": (
            "state at reset is environment_step 0; state after action i is "
            "environment_step i+1"
        ),
        "rollout_failed": bool(rollout_failed),
        "num_environment_states": len(valid_evals),
        "num_policy_inferences": len(inference_environment_steps),
        "semantic_subtasks": definitions,
        "required_subtasks": [definition["subtask_id"] for definition in definitions],
        "terminal_active_subtask": terminal_active_subtask,
        "terminal_active_subtask_instruction": (
            terminal_definition["instruction"]
            if terminal_definition is not None
            else None
        ),
        "terminal_failure_reason": terminal_failure_reason,
        "terminal_unsatisfied_predicate_names": terminal_unsatisfied_predicates,
        "labeling_status": "complete",
        "transitions": _transition_trace(trace),
        "inference_records": inference_records,
        "segments": segments,
        "excluded_completed_subtasks": excluded_completed_subtasks,
        "excluded_bypassed_subtasks": excluded_bypassed_subtasks,
    }


def validate_subtask_safe_record(
    record: dict[str, Any],
    *,
    rollout_id: str,
    rollout_failed: bool,
    inference_environment_steps: list[int],
    task_name: str | None = None,
) -> dict[str, int]:
    """Validate a serialized semantic Subtask-SAFE record."""
    if record.get("schema_version") != SUBTASK_SAFE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Subtask-SAFE schema version; rerun semantic "
            "Subtask-SAFE collection"
        )
    if record.get("rollout_id") != rollout_id:
        raise ValueError("Subtask-SAFE rollout_id disagrees with manifest")
    if task_name is not None and record.get("task_name") != task_name:
        raise ValueError("Subtask-SAFE task_name disagrees with manifest")
    if record.get("rollout_failed") != bool(rollout_failed):
        raise ValueError("Subtask-SAFE rollout failure label disagrees with manifest")
    if record.get("label_semantics") != SUBTASK_FAILURE_LABEL_SEMANTICS:
        raise ValueError("Unsupported Subtask-SAFE label semantics")
    if record.get("semantic_layer") != "ordered_natural_language_subtask":
        raise ValueError("Subtask-SAFE record is not semantic-subtask level")

    definitions = record.get("semantic_subtasks", [])
    if not definitions:
        raise ValueError("Subtask-SAFE semantic subtask definitions are empty")
    definition_by_id = {}
    for expected_index, definition in enumerate(definitions):
        subtask_id = definition.get("subtask_id")
        if (
            not subtask_id
            or not definition.get("instruction")
            or not definition.get("predicate_names")
            or not definition.get("source_subtask_ids")
            or definition.get("subtask_index") != expected_index
        ):
            raise ValueError("Invalid natural-language subtask definition")
        if subtask_id in definition_by_id:
            raise ValueError("Duplicate semantic subtask identifier")
        definition_by_id[subtask_id] = definition

    inference_records = record.get("inference_records", [])
    if len(inference_records) != len(inference_environment_steps):
        raise ValueError("Subtask-SAFE inference count disagrees with manifest")
    actual_steps = [int(item["environment_step"]) for item in inference_records]
    if actual_steps != list(inference_environment_steps):
        raise ValueError("Subtask-SAFE inference alignment disagrees with manifest")
    for item in inference_records:
        subtask_id = item.get("subtask_id")
        if subtask_id is None:
            if item.get("subtask_instruction") is not None:
                raise ValueError("Inference without active subtask has an instruction")
            continue
        definition = definition_by_id.get(subtask_id)
        if definition is None:
            raise ValueError("Inference references unknown semantic subtask")
        if (
            item.get("subtask_instruction") != definition["instruction"]
            or item.get("predicate_names") != definition["predicate_names"]
            or item.get("source_subtask_ids") != definition["source_subtask_ids"]
        ):
            raise ValueError(
                "Inference natural-language subtask metadata is inconsistent"
            )

    segments = record.get("segments", [])
    seen_segment_ids = set()
    all_labels = []
    usable_labels = []
    for segment in segments:
        subtask_id = segment.get("subtask_id")
        definition = definition_by_id.get(subtask_id)
        if definition is None:
            raise ValueError("Segment references unknown semantic subtask")
        if subtask_id in seen_segment_ids:
            raise ValueError("Semantic subtask appears in multiple segments")
        seen_segment_ids.add(subtask_id)
        if (
            segment.get("subtask_instruction") != definition["instruction"]
            or segment.get("predicate_names") != definition["predicate_names"]
            or segment.get("source_subtask_ids") != definition["source_subtask_ids"]
            or segment.get("segment_index") != definition["subtask_index"]
        ):
            raise ValueError(
                "Segment natural-language subtask metadata is inconsistent"
            )
        label = segment.get("failure_label")
        if label is not None:
            if label not in {0, 1}:
                raise ValueError(
                    "Labeled semantic subtask segments require binary labels"
                )
            all_labels.append(label)
        expected_usable = bool(
            label is not None and segment.get("num_policy_inferences", 0)
        )
        if bool(segment.get("usable_for_safe")) != expected_usable:
            raise ValueError("Semantic segment usable_for_safe is inconsistent")
        if expected_usable:
            usable_labels.append(label)

    excluded = record.get("excluded_completed_subtasks", [])
    excluded_ids = set()
    for entry in excluded:
        subtask_id = entry.get("subtask_id")
        definition = definition_by_id.get(subtask_id)
        if definition is None or subtask_id in excluded_ids:
            raise ValueError("Invalid excluded semantic subtask")
        if subtask_id in seen_segment_ids:
            raise ValueError("Semantic subtask cannot be both attempted and excluded")
        if (
            entry.get("instruction") != definition["instruction"]
            or entry.get("predicate_names") != definition["predicate_names"]
            or entry.get("source_subtask_ids") != definition["source_subtask_ids"]
        ):
            raise ValueError(
                "Excluded natural-language subtask metadata is inconsistent"
            )
        excluded_ids.add(subtask_id)

    bypassed = record.get("excluded_bypassed_subtasks", [])
    bypassed_ids = set()
    for entry in bypassed:
        subtask_id = entry.get("subtask_id")
        definition = definition_by_id.get(subtask_id)
        if definition is None or subtask_id in bypassed_ids:
            raise ValueError("Invalid bypassed semantic subtask")
        if subtask_id in seen_segment_ids or subtask_id in excluded_ids:
            raise ValueError("Semantic subtask cannot be both attempted and excluded")
        if definition.get("required_for_official_success", True):
            raise ValueError("Required semantic subtask cannot be bypassed")
        if (
            entry.get("instruction") != definition["instruction"]
            or entry.get("predicate_names") != definition["predicate_names"]
            or entry.get("source_subtask_ids") != definition["source_subtask_ids"]
        ):
            raise ValueError(
                "Bypassed natural-language subtask metadata is inconsistent"
            )
        bypassed_ids.add(subtask_id)

    failure_count = sum(label == 1 for label in all_labels)
    if rollout_failed and failure_count != 1:
        raise ValueError(
            "Failed rollout must assign exactly one terminal semantic subtask"
        )
    if not rollout_failed and failure_count:
        raise ValueError("Successful rollout contains a failed semantic subtask")
    return {
        "segments": len(segments),
        "usable_segments": len(usable_labels),
        "successful_segments": sum(label == 0 for label in usable_labels),
        "failed_segments": sum(label == 1 for label in usable_labels),
        "labeled_without_inference": len(all_labels) - len(usable_labels),
        "excluded_completed_subtasks": len(excluded),
        "excluded_bypassed_subtasks": len(bypassed),
    }


def atomic_write_subtask_safe_record(path: str | Path, record: dict[str, Any]) -> Path:
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
