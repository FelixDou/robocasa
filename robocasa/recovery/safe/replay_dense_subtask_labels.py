"""Replay stored SAFE actions and add dense semantic-subtask outcome labels.

This tool never mutates the source SAFE dataset.  It recreates each RoboCasa
environment from the recorded task and seed, replays the recorded per-step
actions, evaluates the existing ordered semantic predicates, and writes a
sidecar annotation for every rollout whose replayed terminal outcome matches
the immutable source label.

The stored dense label follows the user-facing convention:

* ``subtask_success_label == 1`` for inference samples in a subtask that
  completed successfully;
* ``subtask_success_label == 0`` for inference samples in the active subtask
  that eventually failed.

``failure_label`` is stored as the exact complement for SAFE-compatible risk
training.  Splitting and resampling must still use ``parent_rollout_id``; dense
inference records are not statistically independent rollouts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from robocasa.recovery.subtask_eval import get_subtask_eval

from .collect_atomic_rollouts import atomic_write_json, current_robocasa_commit
from .dataset import load_manifest
from .schema import SafeRolloutMetadata
from .subtask_safe import (
    SUBTASK_FAILURE_LABEL_SEMANTICS,
    build_subtask_safe_record,
    validate_subtask_safe_record,
)


ANNOTATION_SCHEMA_VERSION = 1
ANNOTATION_MANIFEST = "annotation_manifest.jsonl"
SUMMARY_NAME = "annotation_summary.json"
DEFAULT_XR1_DENSE_PILOT_TASKS = (
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CuttingToolSelection",
    "GarnishPancake",
)
DEFAULT_XR1_DENSE_EXPANDED_TASKS = DEFAULT_XR1_DENSE_PILOT_TASKS + (
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "MakeIceLemonade",
    "WashFruitColander",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_action_trajectory(path: str | Path) -> list[Any]:
    """Restore the vector or dictionary action stored by ``--record-actions``."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if "num_steps" not in payload:
            raise ValueError(f"Action artifact has no num_steps: {path}")
        num_steps = int(payload["num_steps"])
        key_map = (
            json.loads(str(payload["action_key_map_json"]))
            if "action_key_map_json" in payload
            else {}
        )
        keys = sorted(
            key
            for key in payload.files
            if key not in {"num_steps", "action_key_map_json"}
        )
        if not keys and num_steps:
            raise ValueError(f"Action artifact has no action arrays: {path}")
        arrays = {}
        for key in keys:
            value = np.asarray(payload[key])
            if value.dtype == object:
                raise ValueError(
                    "Object-valued recorded actions cannot be replayed safely: "
                    f"{path}:{key}"
                )
            if len(value) != num_steps:
                raise ValueError(
                    f"Action array {key!r} has {len(value)} rows, expected {num_steps}"
                )
            arrays[key] = value.copy()
    if keys == ["action"] and key_map.get("action", "action") == "action":
        return [arrays["action"][index] for index in range(num_steps)]
    return [
        {
            key_map.get(key, key): arrays[key][index]
            for key in keys
        }
        for index in range(num_steps)
    ]


def dense_inference_labels(subtask_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Broadcast semantic subtask outcomes to genuine policy inferences."""
    segment_by_subtask = {
        str(segment["subtask_id"]): segment
        for segment in subtask_record.get("segments", [])
        if segment.get("failure_label") in {0, 1}
    }
    dense = []
    for inference in subtask_record.get("inference_records", []):
        subtask_id = inference.get("subtask_id")
        segment = segment_by_subtask.get(str(subtask_id)) if subtask_id else None
        if segment is None:
            continue
        failure_label = int(segment["failure_label"])
        dense.append(
            {
                **inference,
                "parent_rollout_id": subtask_record.get("rollout_id"),
                "parent_task_name": subtask_record.get("task_name"),
                "segment_id": segment.get("segment_id"),
                "subtask_success_label": 1 - failure_label,
                "failure_label": failure_label,
                "label_source": "eventual_active_subtask_outcome",
            }
        )
    if not dense:
        raise ValueError("Semantic annotation contains no labeled inference samples")
    return dense


def replay_rollout(
    record: SafeRolloutMetadata,
    actions: list[Any],
    *,
    make_env_fn: Callable[..., Any],
    step_fn: Callable[[Any, Any], tuple],
    success_fn: Callable[..., bool],
    subtask_eval_fn: Callable[[Any], dict[str, Any] | None] = get_subtask_eval,
    env_interface: str = "gym",
    split: str = "pretrain",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay one immutable source rollout and return annotation plus audit."""
    if not record.action_recording_requested or not record.action_path:
        raise ValueError(f"Rollout {record.rollout_id} has no recorded actions")
    if len(actions) != int(record.num_env_steps):
        raise ValueError(
            f"Rollout {record.rollout_id} has {len(actions)} actions but "
            f"num_env_steps={record.num_env_steps}"
        )
    env = make_env_fn(
        record.task_name,
        env_interface,
        split,
        int(record.environment_seed),
        False,
    )
    subtask_evals = []
    final_info = {}
    final_reward = 0.0
    done_step = None
    try:
        reset_kwargs = (
            {"seed": int(record.environment_seed)}
            if record.seed_protocol == "official_xiaomi"
            else {}
        )
        reset_result = env.reset(**reset_kwargs)
        reset_info = (
            reset_result[1]
            if isinstance(reset_result, tuple)
            and len(reset_result) > 1
            and isinstance(reset_result[1], dict)
            else {}
        )
        subtask_evals.append(
            reset_info.get("subtask_eval")
            if reset_info.get("subtask_eval") is not None
            else subtask_eval_fn(env)
        )
        for index, action in enumerate(actions):
            result = step_fn(env, action)
            if len(result) == 5:
                _, final_reward, terminated, truncated, final_info = result
                done = bool(terminated or truncated)
            else:
                _, final_reward, done, final_info = result
                done = bool(done)
            final_info = final_info if isinstance(final_info, dict) else {}
            subtask_evals.append(
                final_info.get("subtask_eval")
                if final_info.get("subtask_eval") is not None
                else subtask_eval_fn(env)
            )
            if done and done_step is None:
                done_step = index + 1
        replay_success = bool(success_fn(final_info, final_reward, env))
        source_success = not bool(record.failed)
        if replay_success != source_success:
            raise ValueError(
                "Replay terminal outcome mismatch for "
                f"{record.rollout_id}: source_success={source_success}, "
                f"replay_success={replay_success}"
            )
        subtask_record = build_subtask_safe_record(
            subtask_evals,
            list(record.inference_env_steps),
            rollout_failed=bool(record.failed),
            rollout_id=record.rollout_id,
            task_name=record.task_name,
        )
        counts = validate_subtask_safe_record(
            subtask_record,
            rollout_id=record.rollout_id,
            rollout_failed=bool(record.failed),
            inference_environment_steps=list(record.inference_env_steps),
            task_name=record.task_name,
        )
        dense = dense_inference_labels(subtask_record)
        annotation = {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "source_rollout_id": record.rollout_id,
            "source_task_name": record.task_name,
            "source_failed": bool(record.failed),
            "source_environment_seed": int(record.environment_seed),
            "source_environment_reset_index": record.environment_reset_index,
            "source_num_env_steps": int(record.num_env_steps),
            "source_num_policy_inferences": int(record.valid_sequence_length),
            "label_semantics": {
                "subtask_success_label": (
                    "1=active semantic subtask eventually completed; "
                    "0=active semantic subtask eventually failed"
                ),
                "failure_label": SUBTASK_FAILURE_LABEL_SEMANTICS,
                "unattempted_future_subtasks": "not_materialized",
            },
            "subtask_safe_record": subtask_record,
            "dense_inference_labels": dense,
        }
        audit = {
            "replay_terminal_outcome_matches": True,
            "replay_success": replay_success,
            "environment_done_step": done_step,
            "semantic_segment_counts": counts,
            "dense_samples": len(dense),
            "dense_success_samples": sum(
                row["subtask_success_label"] for row in dense
            ),
            "dense_failure_samples": sum(row["failure_label"] for row in dense),
        }
        return annotation, audit
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _load_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid annotation manifest line {line_number}: {error}"
                ) from error
    ids = [row["source_rollout_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Annotation manifest contains duplicate rollout IDs")
    return rows


def select_records(
    records: Iterable[SafeRolloutMetadata],
    tasks: Iterable[str],
    *,
    max_rollouts_per_task: int | None,
    allowed_rollout_ids: set[str] | None = None,
    successes_per_task: int | None = None,
    failures_per_task: int | None = None,
) -> list[SafeRolloutMetadata]:
    if (successes_per_task is None) != (failures_per_task is None):
        raise ValueError(
            "successes_per_task and failures_per_task must be provided together"
        )
    if max_rollouts_per_task is not None and successes_per_task is not None:
        raise ValueError(
            "max_rollouts_per_task cannot be combined with class-specific limits"
        )
    if max_rollouts_per_task is not None and int(max_rollouts_per_task) < 1:
        raise ValueError("max_rollouts_per_task must be positive")
    if successes_per_task is not None and (
        int(successes_per_task) < 1 or int(failures_per_task) < 1
    ):
        raise ValueError("Class-specific rollout limits must be positive")
    requested = list(dict.fromkeys(str(task) for task in tasks))
    grouped = defaultdict(list)
    for record in records:
        if record.task_name in requested and (
            allowed_rollout_ids is None or record.rollout_id in allowed_rollout_ids
        ):
            grouped[record.task_name].append(record)
    missing = [task for task in requested if not grouped[task]]
    if missing:
        raise ValueError(f"Requested tasks are absent from source: {missing}")
    selected = []
    for task in requested:
        values = sorted(
            grouped[task],
            key=lambda record: (
                record.environment_seed,
                record.environment_reset_index
                if record.environment_reset_index is not None
                else -1,
                record.rollout_id,
            ),
        )
        if successes_per_task is not None:
            successes = [record for record in values if not record.failed]
            failures = [record for record in values if record.failed]
            if (
                len(successes) < int(successes_per_task)
                or len(failures) < int(failures_per_task)
            ):
                raise ValueError(
                    f"Selected source for {task} has {len(successes)} successes "
                    f"and {len(failures)} failures; requested "
                    f"{successes_per_task}/{failures_per_task}"
                )
            values = sorted(
                successes[: int(successes_per_task)]
                + failures[: int(failures_per_task)],
                key=lambda record: (
                    record.environment_seed,
                    record.environment_reset_index
                    if record.environment_reset_index is not None
                    else -1,
                    record.rollout_id,
                ),
            )
        elif max_rollouts_per_task is not None:
            values = values[: int(max_rollouts_per_task)]
        if {bool(record.failed) for record in values} != {False, True}:
            raise ValueError(f"Selected source for {task} does not contain both outcomes")
        selected.extend(values)
    return selected


def load_selection_rollout_ids(path: str | Path) -> set[str]:
    """Load exact selected source IDs from an official export report."""
    report = json.loads(Path(path).read_text())
    mapping = report.get("mapping")
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("Selection report has no non-empty mapping")
    values = [str(item.get("rollout_id", "")) for item in mapping]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("Selection report contains missing or duplicate rollout IDs")
    return set(values)


def build_summary(
    dataset_dir: Path,
    output_dir: Path,
    tasks: list[str],
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    support = defaultdict(
        lambda: {
            "rollouts": 0,
            "source_successes": 0,
            "source_failures": 0,
            "dense_success_samples": 0,
            "dense_failure_samples": 0,
            "stages": defaultdict(lambda: Counter(successes=0, failures=0, samples=0)),
        }
    )
    for row in rows:
        task = row["source_task_name"]
        payload = support[task]
        payload["rollouts"] += 1
        payload["source_failures" if row["source_failed"] else "source_successes"] += 1
        payload["dense_success_samples"] += int(row["dense_success_samples"])
        payload["dense_failure_samples"] += int(row["dense_failure_samples"])
        annotation = json.loads((output_dir / row["annotation_path"]).read_text())
        seen = set()
        for dense in annotation["dense_inference_labels"]:
            stage = str(dense["subtask_id"])
            support[task]["stages"][stage]["samples"] += 1
            key = (stage, int(dense["subtask_success_label"]))
            if key in seen:
                continue
            seen.add(key)
            support[task]["stages"][stage][
                "successes" if key[1] else "failures"
            ] += 1
    serializable = {}
    for task, payload in sorted(support.items()):
        serializable[task] = {
            key: value for key, value in payload.items() if key != "stages"
        }
        serializable[task]["stages"] = {
            stage: dict(counts) for stage, counts in sorted(payload["stages"].items())
        }
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "status": "complete" if not errors else "complete_with_errors",
        "source_dataset": str(dataset_dir),
        "output_dir": str(output_dir),
        "source_manifest_sha256": sha256_file(dataset_dir / "manifest.jsonl"),
        "robocasa_commit": current_robocasa_commit(),
        "tasks": tasks,
        "rollouts": len(rows),
        "errors": len(errors),
        "dense_samples": sum(int(row["dense_samples"]) for row in rows),
        "dense_success_samples": sum(
            int(row["dense_success_samples"]) for row in rows
        ),
        "dense_failure_samples": sum(
            int(row["dense_failure_samples"]) for row in rows
        ),
        "support": serializable,
        "error_records": errors,
        "source_dataset_unchanged": True,
    }


def annotate_dataset(args) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    tasks = list(args.tasks or DEFAULT_XR1_DENSE_PILOT_TASKS)
    allowed_rollout_ids = (
        load_selection_rollout_ids(args.selection_report)
        if args.selection_report
        else None
    )
    records = select_records(
        load_manifest(dataset_dir),
        tasks,
        max_rollouts_per_task=args.max_rollouts_per_task,
        allowed_rollout_ids=allowed_rollout_ids,
        successes_per_task=args.successes_per_task,
        failures_per_task=args.failures_per_task,
    )
    if args.dry_run:
        missing_actions = [
            record.rollout_id
            for record in records
            if not record.action_recording_requested
            or not record.action_path
            or not (dataset_dir / str(record.action_path)).is_file()
        ]
        return {
            "status": "dry_run_valid" if not missing_actions else "dry_run_invalid",
            "source_dataset": str(dataset_dir),
            "output_dir": str(output_dir),
            "tasks": tasks,
            "rollouts": len(records),
            "selection_report": args.selection_report,
            "per_task": dict(Counter(record.task_name for record in records)),
            "source_outcomes": {
                task: {
                    "successes": sum(
                        record.task_name == task and not record.failed for record in records
                    ),
                    "failures": sum(
                        record.task_name == task and record.failed for record in records
                    ),
                }
                for task in tasks
            },
            "missing_action_artifacts": missing_actions,
            "replay_required_before_training": True,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / ANNOTATION_MANIFEST
    existing = _load_existing_manifest(manifest_path) if args.resume else []
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"Annotation output already exists; use --resume: {manifest_path}"
        )
    row_by_id = {row["source_rollout_id"]: row for row in existing}
    rows = list(existing)
    errors = []

    from robocasa.recovery.evaluate_recovery_benchmark import make_env
    from robocasa.recovery.recovery_rollout import _is_task_success, _step_env

    for index, record in enumerate(records, 1):
        if record.rollout_id in row_by_id:
            annotation_path = output_dir / row_by_id[record.rollout_id]["annotation_path"]
            if not annotation_path.is_file():
                raise ValueError(
                    f"Completed annotation is missing for {record.rollout_id}"
                )
            continue
        try:
            action_path = dataset_dir / str(record.action_path)
            if not action_path.is_file():
                raise FileNotFoundError(
                    f"Recorded action artifact is missing: {action_path}"
                )
            actions = load_action_trajectory(action_path)
            annotation, audit = replay_rollout(
                record,
                actions,
                make_env_fn=make_env,
                step_fn=_step_env,
                success_fn=_is_task_success,
                env_interface=args.env_interface,
                split=args.split,
            )
            relative = Path("annotations") / record.task_name / f"{record.rollout_id}.json"
            path = output_dir / relative
            atomic_write_json(path, annotation)
            row = {
                "source_rollout_id": record.rollout_id,
                "source_task_name": record.task_name,
                "source_failed": bool(record.failed),
                "source_environment_seed": int(record.environment_seed),
                "source_environment_reset_index": record.environment_reset_index,
                "source_tensor_path": record.tensor_path,
                "source_action_path": record.action_path,
                "source_action_sha256": sha256_file(action_path),
                "annotation_path": str(relative),
                **audit,
            }
            rows.append(row)
            row_by_id[record.rollout_id] = row
            _atomic_write_jsonl(manifest_path, rows)
            print(
                f"[{index}/{len(records)}] {record.task_name} {record.rollout_id} "
                f"dense={row['dense_samples']}"
            )
        except Exception as error:
            event = {
                "source_rollout_id": record.rollout_id,
                "source_task_name": record.task_name,
                "source_environment_seed": int(record.environment_seed),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            errors.append(event)
            print(f"ERROR {record.rollout_id}: {event['error_type']}: {event['error']}")
            if not args.continue_on_error:
                raise
    selected_ids = {record.rollout_id for record in records}
    rows = [row for row in rows if row["source_rollout_id"] in selected_ids]
    rows.sort(
        key=lambda row: (
            row["source_task_name"],
            row["source_environment_seed"],
            row["source_environment_reset_index"]
            if row["source_environment_reset_index"] is not None
            else -1,
            row["source_rollout_id"],
        )
    )
    _atomic_write_jsonl(manifest_path, rows)
    summary = build_summary(dataset_dir, output_dir, tasks, rows, errors)
    summary["selection_report"] = (
        str(Path(args.selection_report).resolve()) if args.selection_report else None
    )
    summary["selection_report_sha256"] = (
        sha256_file(args.selection_report) if args.selection_report else None
    )
    atomic_write_json(output_dir / SUMMARY_NAME, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--max-rollouts-per-task", type=int)
    parser.add_argument("--successes-per-task", type=int)
    parser.add_argument("--failures-per-task", type=int)
    parser.add_argument(
        "--selection-report",
        help=(
            "Optional official conversion_report.json whose mapping freezes the "
            "exact selected source rollout IDs"
        ),
    )
    parser.add_argument("--env-interface", choices=("gym", "robosuite"), default="gym")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> None:
    summary = annotate_dataset(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
