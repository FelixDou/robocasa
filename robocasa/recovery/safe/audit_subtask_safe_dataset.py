"""Audit raw Subtask-SAFE coverage and rank collection deficits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

from .atomic_tasks import registered_atomic_tasks, registered_composite_tasks
from .collect_atomic_rollouts import SUMMARY_NAME, atomic_write_json
from .dataset import load_manifest
from .schema import compatibility_key
from .subtask_safe import validate_subtask_safe_record
from .validate_atomic_dataset import validate_atomic_dataset


AUDIT_SCHEMA_VERSION = 1
PARTIAL_VALIDATION_ERROR = "Dataset summary marks collection as partial/incomplete"


def _median(values: list[int]) -> float | None:
    return float(median(values)) if values else None


def _task_type(task_name: str, configured: dict[str, str]) -> str:
    if task_name in configured:
        return configured[task_name]
    if task_name in registered_atomic_tasks():
        return "atomic"
    if task_name in registered_composite_tasks():
        return "composite"
    return "unknown"


def _coverage_warning(row: dict[str, Any]) -> list[str]:
    warnings = []
    if row["usable_successes"] == 0:
        warnings.append("no_usable_successes")
    if row["usable_failures"] == 0:
        warnings.append("no_usable_failures")
    if row["labeled_without_inference"]:
        warnings.append("labeled_segments_without_inference")
    if row["excluded_completed_without_activation"]:
        warnings.append("completed_without_observed_activation")
    if row["excluded_bypassed_optional"]:
        warnings.append("optional_transient_bypassed")
    if row["not_reached_rollouts"]:
        warnings.append("not_reached_in_some_rollouts")
    if len(row["segment_indices"]) > 1:
        warnings.append("inconsistent_segment_index")
    return warnings


def summarize_subtask_records(
    rollout_records: list[dict[str, Any]],
    *,
    target_successes: int = 30,
    target_failures: int = 20,
    task_type_filter: str = "all",
) -> dict[str, Any]:
    """Summarize validated serialized records without loading feature tensors.

    Each item requires ``task_name``, ``task_type``, ``rollout_id``, ``failed``,
    and a validated Subtask-SAFE payload under ``subtask_record``.
    """
    if target_successes < 0 or target_failures < 0:
        raise ValueError("Subtask-SAFE targets must be non-negative")
    if task_type_filter not in {"all", "atomic", "composite"}:
        raise ValueError(f"Unsupported task-type filter {task_type_filter!r}")

    filtered = [
        item
        for item in rollout_records
        if task_type_filter == "all" or item["task_type"] == task_type_filter
    ]
    if not filtered:
        raise ValueError(
            f"No {task_type_filter} Subtask-SAFE rollouts remain after filtering"
        )

    rollout_ids = [item["rollout_id"] for item in filtered]
    if len(rollout_ids) != len(set(rollout_ids)):
        raise ValueError("Duplicate rollout IDs are present in audit inputs")

    task_rollouts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rollout_ids": [],
            "rollout_successes": 0,
            "rollout_failures": 0,
        }
    )
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for item in filtered:
        task_name = item["task_name"]
        task = task_rollouts[task_name]
        task["task_type"] = item["task_type"]
        task["rollout_ids"].append(item["rollout_id"])
        task["rollout_failures" if item["failed"] else "rollout_successes"] += 1

        definitions = item["subtask_record"].get("semantic_subtasks", [])
        for definition in definitions:
            subtask_id = definition["subtask_id"]
            key = (task_name, subtask_id)
            expected = {
                "task_name": task_name,
                "task_type": item["task_type"],
                "subtask_id": subtask_id,
                "subtask_name": subtask_id,
                "subtask_instruction": definition["instruction"],
                "predicate_names": list(definition["predicate_names"]),
                "source_subtask_ids": list(definition["source_subtask_ids"]),
                "segment_indices": {int(definition["subtask_index"])},
            }
            group = groups.get(key)
            if group is not None:
                actual = {
                    field: group[field]
                    for field in (
                        "task_name",
                        "task_type",
                        "subtask_id",
                        "subtask_name",
                        "subtask_instruction",
                        "predicate_names",
                        "source_subtask_ids",
                        "segment_indices",
                    )
                }
                if actual != expected:
                    raise ValueError(
                        "Semantic subtask definitions changed across rollouts "
                        f"for {task_name}/{subtask_id}"
                    )
                continue
            groups[key] = {
                **expected,
                "entered_segments": 0,
                "completed_segments": 0,
                "terminal_failed_segments": 0,
                "unlabeled_segments": 0,
                "usable_successes": 0,
                "usable_failures": 0,
                "usable_active_failures": 0,
                "usable_regression_failures": 0,
                "labeled_without_inference": 0,
                "excluded_completed_without_activation": 0,
                "excluded_bypassed_optional": 0,
                "not_reached_rollouts": 0,
                "_inference_counts": [],
                "_environment_durations": [],
            }

        segment_by_id = {
            segment["subtask_id"]: segment
            for segment in item["subtask_record"].get("segments", [])
        }
        excluded_ids = {
            entry["subtask_id"]
            for entry in item["subtask_record"].get("excluded_completed_subtasks", [])
        }
        bypassed_ids = {
            entry["subtask_id"]
            for entry in item["subtask_record"].get("excluded_bypassed_subtasks", [])
        }
        for definition in definitions:
            key = (task_name, definition["subtask_id"])
            group = groups[key]
            segment = segment_by_id.get(definition["subtask_id"])
            if segment is None:
                if definition["subtask_id"] in excluded_ids:
                    group["excluded_completed_without_activation"] += 1
                elif definition["subtask_id"] in bypassed_ids:
                    group["excluded_bypassed_optional"] += 1
                else:
                    group["not_reached_rollouts"] += 1
                continue
            group["segment_indices"].add(int(segment["segment_index"]))
            group["entered_segments"] += 1
            group["completed_segments"] += int(bool(segment["completed"]))
            group["terminal_failed_segments"] += int(bool(segment["eventually_failed"]))
            label = segment.get("failure_label")
            group["unlabeled_segments"] += int(label is None)
            usable = bool(segment.get("usable_for_safe"))
            if usable:
                if label == 1:
                    group["usable_failures"] += 1
                    failure_key = (
                        "usable_regression_failures"
                        if segment.get("terminal_failure_reason")
                        == "completed_subtask_regressed_before_task_completion"
                        else "usable_active_failures"
                    )
                    group[failure_key] += 1
                else:
                    group["usable_successes"] += 1
            elif label is not None:
                group["labeled_without_inference"] += 1
            group["_inference_counts"].append(
                int(segment.get("num_policy_inferences", 0))
            )
            group["_environment_durations"].append(
                int(segment["end_environment_step"])
                - int(segment["entry_environment_step"])
            )

    rows = []
    for group in groups.values():
        row = {key: value for key, value in group.items() if not key.startswith("_")}
        row["segment_indices"] = sorted(row["segment_indices"])
        labeled = (
            row["usable_successes"]
            + row["usable_failures"]
            + row["labeled_without_inference"]
        )
        usable = row["usable_successes"] + row["usable_failures"]
        if (
            row["usable_active_failures"]
            + row["usable_regression_failures"]
            != row["usable_failures"]
        ):
            raise ValueError("Subtask failure-mode counts are inconsistent")
        row.update(
            {
                "labeled_segments": labeled,
                "usable_segments": usable,
                "inference_coverage": (float(usable / labeled) if labeled else None),
                "median_policy_inferences": _median(group["_inference_counts"]),
                "min_policy_inferences": (
                    min(group["_inference_counts"])
                    if group["_inference_counts"]
                    else None
                ),
                "max_policy_inferences": (
                    max(group["_inference_counts"])
                    if group["_inference_counts"]
                    else None
                ),
                "median_environment_steps": _median(group["_environment_durations"]),
                "success_deficit": max(0, target_successes - row["usable_successes"]),
                "failure_deficit": max(0, target_failures - row["usable_failures"]),
            }
        )
        row["target_reached"] = bool(
            row["success_deficit"] == 0 and row["failure_deficit"] == 0
        )
        row["warnings"] = _coverage_warning(row)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["task_name"],
            min(row["segment_indices"]),
            row["subtask_name"],
        )
    )

    task_summaries = []
    for task_name, task in sorted(task_rollouts.items()):
        task_rows = [row for row in rows if row["task_name"] == task_name]
        success_deficit = sum(row["success_deficit"] for row in task_rows)
        failure_deficit = sum(row["failure_deficit"] for row in task_rows)
        task_summaries.append(
            {
                "task_name": task_name,
                "task_type": task["task_type"],
                "rollouts": len(task["rollout_ids"]),
                "rollout_successes": task["rollout_successes"],
                "rollout_failures": task["rollout_failures"],
                "subtasks_observed": len(task_rows),
                "success_deficit": success_deficit,
                "failure_deficit": failure_deficit,
                "total_deficit": success_deficit + failure_deficit,
                "all_subtask_targets_reached": bool(
                    task_rows and all(row["target_reached"] for row in task_rows)
                ),
            }
        )
    task_priority = sorted(
        task_summaries,
        key=lambda row: (
            -row["failure_deficit"],
            -row["success_deficit"],
            row["task_name"],
        ),
    )
    for rank, task in enumerate(task_priority, 1):
        task["collection_priority"] = rank
    priority_by_task = {
        task["task_name"]: task["collection_priority"] for task in task_priority
    }
    for task in task_summaries:
        task["collection_priority"] = priority_by_task[task["task_name"]]

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol": "oracle_guided_subtask_safe_coverage_audit",
        "semantic_layer": "ordered_natural_language_subtask",
        "label_semantics": ("active_subtask_eventually_fails_before_completion"),
        "task_type_filter": task_type_filter,
        "targets": {
            "usable_successes_per_task_subtask": target_successes,
            "usable_failures_per_task_subtask": target_failures,
        },
        "collection_guidance": {
            "sampling": "natural full rollouts without subtask quotas",
            "split_unit": "rollout before segment extraction",
            "weighting": "training-only inverse-frequency subtask loss",
            "priority_rule": (
                "descending summed failure deficit, then success deficit"
            ),
            "deficits_are_exact_rollout_requirements": False,
        },
        "counts": {
            "rollouts": len(filtered),
            "rollout_successes": sum(not item["failed"] for item in filtered),
            "rollout_failures": sum(item["failed"] for item in filtered),
            "tasks": len(task_summaries),
            "task_subtask_pairs": len(rows),
            "usable_success_segments": sum(row["usable_successes"] for row in rows),
            "usable_failure_segments": sum(row["usable_failures"] for row in rows),
            "usable_active_failure_segments": sum(
                row["usable_active_failures"] for row in rows
            ),
            "usable_regression_failure_segments": sum(
                row["usable_regression_failures"] for row in rows
            ),
            "labeled_without_inference": sum(
                row["labeled_without_inference"] for row in rows
            ),
            "excluded_completed_without_activation": sum(
                row["excluded_completed_without_activation"] for row in rows
            ),
            "excluded_bypassed_optional": sum(
                row["excluded_bypassed_optional"] for row in rows
            ),
            "pairs_reaching_target": sum(row["target_reached"] for row in rows),
        },
        "tasks": task_summaries,
        "task_priority": [task["task_name"] for task in task_priority],
        "task_subtask_rows": rows,
    }


def audit_subtask_datasets(
    dataset_dirs: list[str | Path],
    *,
    target_successes: int = 30,
    target_failures: int = 20,
    task_type_filter: str = "all",
    allow_partial: bool = False,
    allow_unregistered_tasks: bool = False,
) -> dict[str, Any]:
    if not dataset_dirs:
        raise ValueError("At least one dataset directory is required")

    rollout_records = []
    source_reports = []
    episode_identities = set()
    compatibility_keys = set()
    for raw_path in dataset_dirs:
        dataset_dir = Path(raw_path).resolve()
        validation = validate_atomic_dataset(
            dataset_dir,
            allow_unregistered=allow_unregistered_tasks,
        )
        remaining_errors = [
            error
            for error in validation["errors"]
            if not (allow_partial and error == PARTIAL_VALIDATION_ERROR)
        ]
        if remaining_errors:
            details = "\n".join(f"- {error}" for error in remaining_errors)
            raise ValueError(f"Dataset validation failed for {dataset_dir}:\n{details}")
        summary = json.loads((dataset_dir / SUMMARY_NAME).read_text())
        configured_types = summary.get("config", {}).get("task_types", {})
        source_records = load_manifest(dataset_dir)
        for record in source_records:
            if not record.subtask_recording_requested:
                raise ValueError(
                    f"Rollout {record.rollout_id} has no Subtask-SAFE trace"
                )
            episode_identity = (
                record.task_name,
                record.environment_seed,
                record.environment_reset_index,
            )
            if episode_identity in episode_identities:
                raise ValueError(
                    "Duplicate task/seed/reset episode across audit inputs: "
                    f"{episode_identity}"
                )
            episode_identities.add(episode_identity)
            compatibility_keys.add(compatibility_key(record))
            subtask_record = json.loads(
                (dataset_dir / record.subtask_trace_path).read_text()
            )
            validate_subtask_safe_record(
                subtask_record,
                rollout_id=record.rollout_id,
                rollout_failed=record.failed,
                inference_environment_steps=record.inference_env_steps,
                task_name=record.task_name,
            )
            rollout_records.append(
                {
                    "task_name": record.task_name,
                    "task_type": _task_type(record.task_name, configured_types),
                    "rollout_id": record.rollout_id,
                    "failed": bool(record.failed),
                    "subtask_record": subtask_record,
                }
            )
        source_reports.append(
            {
                "dataset_dir": str(dataset_dir),
                "partial": bool(summary.get("partial")),
                "rollouts": len(source_records),
                "validation_warnings": validation["warnings"],
            }
        )
    if len(compatibility_keys) > 1:
        raise ValueError("Audit inputs mix incompatible policy or SAFE feature schemas")

    result = summarize_subtask_records(
        rollout_records,
        target_successes=target_successes,
        target_failures=target_failures,
        task_type_filter=task_type_filter,
    )
    result["sources"] = source_reports
    result["compatibility_key"] = (
        next(iter(compatibility_keys)) if compatibility_keys else None
    )
    return result


def format_report(result: dict[str, Any]) -> str:
    counts = result["counts"]
    targets = result["targets"]
    lines = [
        "=== SUBTASK-SAFE COVERAGE AUDIT ===",
        (
            f"Rollouts: {counts['rollouts']} "
            f"({counts['rollout_successes']} successes, "
            f"{counts['rollout_failures']} failures)"
        ),
        (
            f"Task/subtask pairs: {counts['task_subtask_pairs']} "
            f"({counts['pairs_reaching_target']} reached "
            f"{targets['usable_successes_per_task_subtask']}S/"
            f"{targets['usable_failures_per_task_subtask']}F)"
        ),
        (
            "Usable segments: "
            f"{counts['usable_success_segments']} successes, "
            f"{counts['usable_failure_segments']} failures; "
            f"labeled without inference: "
            f"{counts['labeled_without_inference']}"
        ),
        (
            "Usable failure modes: "
            f"active={counts['usable_active_failure_segments']}, "
            "later-regression="
            f"{counts['usable_regression_failure_segments']}"
        ),
        (
            "Completed without observed activation: "
            f"{counts['excluded_completed_without_activation']}"
        ),
        (
            "Bypassed optional transient subtasks: "
            f"{counts['excluded_bypassed_optional']}"
        ),
        "",
        "task                             type idx subtask"
        "                              entered   S   F noinf  cov% need S/F",
    ]
    for row in result["task_subtask_rows"]:
        indices = ",".join(map(str, row["segment_indices"]))
        coverage = (
            f"{100.0 * row['inference_coverage']:5.1f}"
            if row["inference_coverage"] is not None
            else "  n/a"
        )
        lines.append(
            f"{row['task_name']:<32s} "
            f"{row['task_type'][:4]:<4s} "
            f"{indices:>3s} "
            f"{row['subtask_name']:<36s} "
            f"{row['entered_segments']:>7d} "
            f"{row['usable_successes']:>3d} "
            f"{row['usable_failures']:>3d} "
            f"{row['labeled_without_inference']:>5d} "
            f"{coverage} "
            f"{row['success_deficit']:>3d}/{row['failure_deficit']:<3d}"
        )
        lines.append(f"    instruction: {row['subtask_instruction']}")
        lines.append("    predicates: " + ", ".join(row["predicate_names"]))
        if row["source_subtask_ids"] != [row["subtask_id"]]:
            lines.append("    merged from: " + ", ".join(row["source_subtask_ids"]))
        lines.append(
            "    rollout states: "
            f"active-failure={row['usable_active_failures']} "
            f"later-regression={row['usable_regression_failures']} "
            f"excluded-completed={row['excluded_completed_without_activation']} "
            f"bypassed-optional={row['excluded_bypassed_optional']} "
            f"not-reached={row['not_reached_rollouts']}"
        )
    lines.extend(["", "Collection priority (diagnostic, not rollout counts):"])
    task_by_name = {task["task_name"]: task for task in result["tasks"]}
    for task_name in result["task_priority"]:
        task = task_by_name[task_name]
        lines.append(
            f"  {task['collection_priority']:>2d}. "
            f"{task_name:<32s} "
            f"rollouts={task['rollouts']:>3d} "
            f"rollout S/F={task['rollout_successes']}/"
            f"{task['rollout_failures']} "
            f"summed need S/F={task['success_deficit']}/"
            f"{task['failure_deficit']}"
        )
    lines.extend(
        [
            "",
            "Guidance: collect natural full rollouts; split by rollout before "
            "segment extraction.",
            "Deficits count segment examples and are not exact additional "
            "rollout requirements.",
        ]
    )
    return "\n".join(lines)


def _atomic_write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "segment_indices": ",".join(map(str, row["segment_indices"])),
                    "warnings": ",".join(row["warnings"]),
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", nargs="+", required=True)
    parser.add_argument(
        "--task-type",
        choices=("all", "atomic", "composite"),
        default="all",
    )
    parser.add_argument("--target-successes", type=int, default=30)
    parser.add_argument("--target-failures", type=int, default=20)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-unregistered-tasks", action="store_true")
    parser.add_argument("--json-output")
    parser.add_argument("--csv-output")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = audit_subtask_datasets(
            args.dataset_dir,
            target_successes=args.target_successes,
            target_failures=args.target_failures,
            task_type_filter=args.task_type,
            allow_partial=args.allow_partial,
            allow_unregistered_tasks=args.allow_unregistered_tasks,
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if args.json_output:
        atomic_write_json(args.json_output, result)
    if args.csv_output:
        _atomic_write_csv(args.csv_output, result["task_subtask_rows"])
    if not args.quiet:
        print(format_report(result))


if __name__ == "__main__":
    main()
