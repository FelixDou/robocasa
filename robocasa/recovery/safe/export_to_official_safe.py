"""Materialize RoboCasa SAFE data in the official SAFE loader layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random

import numpy as np

from .collect_atomic_rollouts import SUMMARY_NAME, atomic_write_json, utc_now
from .dataset import MANIFEST_NAME, assert_compatible, load_manifest
from .validate_atomic_dataset import validate_atomic_dataset


REPORT_NAME = "conversion_report.json"


def _record_sort_key(record):
    return (
        record.task_name,
        record.environment_seed,
        (
            record.environment_reset_index
            if record.environment_reset_index is not None
            else -1
        ),
        record.rollout_id,
    )


def select_balanced_records(
    records,
    *,
    rollouts_per_task=None,
    successes_per_task=None,
    failures_per_task=None,
    seed=0,
):
    """Select an exact deterministic class balance without mutating source data."""
    if rollouts_per_task is not None:
        if successes_per_task is not None or failures_per_task is not None:
            raise ValueError(
                "--rollouts-per-task cannot be combined with "
                "--successes-per-task/--failures-per-task"
            )
        if rollouts_per_task < 1:
            raise ValueError("--rollouts-per-task must be positive")
        tasks = sorted({record.task_name for record in records})
        selected = []
        per_task = {}
        for task in tasks:
            candidates = sorted(
                (record for record in records if record.task_name == task),
                key=_record_sort_key,
            )
            if len(candidates) < rollouts_per_task:
                raise ValueError(
                    f"Task {task} has {len(candidates)} rollouts; "
                    f"requested {rollouts_per_task}"
                )
            successes = [record for record in candidates if not record.failed]
            failures = [record for record in candidates if record.failed]
            source_total = len(candidates)
            selected_successes = (
                rollouts_per_task * len(successes) + source_total // 2
            ) // source_total
            lower_successes = max(0, rollouts_per_task - len(failures))
            upper_successes = min(rollouts_per_task, len(successes))
            if successes and failures and rollouts_per_task >= 2:
                lower_successes = max(lower_successes, 1)
                upper_successes = min(upper_successes, rollouts_per_task - 1)
            selected_successes = min(
                max(selected_successes, lower_successes),
                upper_successes,
            )
            selected_failures = rollouts_per_task - selected_successes
            success_rng = random.Random(f"{seed}:{task}:success")
            failure_rng = random.Random(f"{seed}:{task}:failure")
            success_rng.shuffle(successes)
            failure_rng.shuffle(failures)
            chosen = (
                successes[:selected_successes]
                + failures[:selected_failures]
            )
            selected.extend(chosen)
            per_task[task] = {
                "source_rollouts": len(candidates),
                "source_successes": len(successes),
                "source_failures": len(failures),
                "selected_rollouts": len(chosen),
                "selected_successes": selected_successes,
                "selected_failures": selected_failures,
            }
        selected.sort(key=_record_sort_key)
        return selected, {
            "mode": "per_task_natural_rate_total",
            "seed": int(seed),
            "rollouts_per_task": int(rollouts_per_task),
            "source_num_rollouts": len(records),
            "selected_num_rollouts": len(selected),
            "per_task": per_task,
        }
    if successes_per_task is None and failures_per_task is None:
        selected = sorted(records, key=_record_sort_key)
        return selected, {
            "mode": "all",
            "source_num_rollouts": len(records),
            "selected_num_rollouts": len(selected),
        }
    if successes_per_task is None or failures_per_task is None:
        raise ValueError(
            "--successes-per-task and --failures-per-task must be provided together"
        )
    if successes_per_task < 1 or failures_per_task < 1:
        raise ValueError("Per-task success and failure counts must be positive")

    rng = random.Random(seed)
    tasks = sorted({record.task_name for record in records})
    selected = []
    per_task = {}
    for task in tasks:
        successes = sorted(
            (record for record in records if record.task_name == task and not record.failed),
            key=_record_sort_key,
        )
        failures = sorted(
            (record for record in records if record.task_name == task and record.failed),
            key=_record_sort_key,
        )
        if len(successes) < successes_per_task or len(failures) < failures_per_task:
            raise ValueError(
                f"Task {task} has {len(successes)} successes and {len(failures)} failures; "
                f"requested {successes_per_task} and {failures_per_task}"
            )
        rng.shuffle(successes)
        rng.shuffle(failures)
        chosen_successes = successes[:successes_per_task]
        chosen_failures = failures[:failures_per_task]
        selected.extend(chosen_successes)
        selected.extend(chosen_failures)
        per_task[task] = {
            "source_successes": len(successes),
            "source_failures": len(failures),
            "selected_successes": len(chosen_successes),
            "selected_failures": len(chosen_failures),
        }

    selected.sort(key=_record_sort_key)
    return selected, {
        "mode": "per_task_class_balance",
        "seed": int(seed),
        "successes_per_task": int(successes_per_task),
        "failures_per_task": int(failures_per_task),
        "source_num_rollouts": len(records),
        "selected_num_rollouts": len(selected),
        "per_task": per_task,
    }


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def source_fingerprint(dataset_dir):
    data = (Path(dataset_dir) / MANIFEST_NAME).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _check_existing_pickle(path, rollout_id, resume, *, inference_index=None):
    if not path.exists():
        return False
    if not resume:
        raise FileExistsError(f"Output exists: {path}; pass --resume")
    try:
        with path.open("rb") as stream:
            existing = pickle.load(stream)
        if existing.get("rollout_id") != rollout_id:
            raise ValueError(f"Existing output belongs to another rollout: {path}")
        if inference_index is not None and existing.get("inference_index") != inference_index:
            raise ValueError(
                f"Existing output has the wrong inference index for {path}: "
                f"{existing.get('inference_index')!r} != {inference_index}"
            )
    except Exception as error:
        raise ValueError(f"Cannot safely resume existing output {path}: {error}") from error
    return True


def export_to_official_safe(
    dataset_dir,
    output_dir,
    *,
    resume=False,
    dry_run=False,
    allow_unregistered=False,
    rollouts_per_task=None,
    successes_per_task=None,
    failures_per_task=None,
    selection_seed=0,
):
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    validation = validate_atomic_dataset(
        dataset_dir, allow_unregistered=allow_unregistered
    )
    if not validation["valid"]:
        raise ValueError("Source dataset is invalid: " + "; ".join(validation["errors"]))
    if not validation["official_safe_loader_compatible"]:
        raise ValueError("Source dataset lacks fields required by the official SAFE π0 loader")
    source_records = load_manifest(dataset_dir)
    source_summary = json.loads((dataset_dir / SUMMARY_NAME).read_text())
    task_types = {
        task: source_summary["config"].get("task_types", {}).get(task, "atomic")
        for task in sorted({record.task_name for record in source_records})
    }
    records, selection = select_balanced_records(
        source_records,
        rollouts_per_task=rollouts_per_task,
        successes_per_task=successes_per_task,
        failures_per_task=failures_per_task,
        seed=selection_seed,
    )
    compatibility = assert_compatible(records)
    fingerprint = source_fingerprint(dataset_dir)
    task_ids = {task: index for index, task in enumerate(sorted({r.task_name for r in records}))}
    plan = {
        "source_dataset": str(dataset_dir),
        "output_dir": str(output_dir),
        "source_fingerprint": fingerprint,
        "compatibility_key": compatibility,
        "selection": selection,
        "num_rollouts": len(records),
        "num_policy_records": sum(r.valid_sequence_length for r in records),
        "task_ids": task_ids,
        "task_types": task_types,
        "model_families": sorted({record.model_family for record in records}),
    }
    if dry_run:
        return {"dry_run": True, **plan}
    report_path = output_dir / REPORT_NAME
    if output_dir.exists() and not resume and any(output_dir.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output_dir}; pass --resume")
    if report_path.exists() and resume:
        previous = json.loads(report_path.read_text())
        if previous.get("source_fingerprint") != fingerprint:
            raise ValueError("Existing export report refers to a different source manifest")
        if previous.get("selection") != selection:
            raise ValueError("Existing export report uses a different rollout selection")
        if previous.get("complete"):
            expected_env = len(records)
            expected_policy = sum(r.valid_sequence_length for r in records)
            if (
                len(list((output_dir / "env_records").glob("*.pkl"))) == expected_env
                and len(list((output_dir / "policy_records").glob("*meta.pkl"))) == expected_policy
            ):
                return previous
    env_dir = output_dir / "env_records"
    policy_dir = output_dir / "policy_records"
    env_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    global_policy_index = 0
    materialized_bytes = 0
    episode_by_task = {}
    for rollout_index, record in enumerate(records):
        episode_index = episode_by_task.get(record.task_name, 0)
        episode_by_task[record.task_name] = episode_index + 1
        env_path = env_dir / f"rollout_{rollout_index:08d}--{record.rollout_id}.pkl"
        env_record = {
            "rollout_id": record.rollout_id,
            "task_suite_name": f"robocasa_{task_types[record.task_name]}",
            "task_type": task_types[record.task_name],
            "task_id": task_ids[record.task_name],
            "task_name": record.task_name,
            "task_description": record.task_instruction,
            "episode_idx": episode_index,
            "episode_success": int(not record.failed),
            "environment_seed": record.environment_seed,
            "environment_reset_index": record.environment_reset_index,
            "seed_protocol": record.seed_protocol,
            "video_frame_stride": record.video_frame_stride,
            "model_infer_times": record.valid_sequence_length,
            "replan_steps": record.replan_steps,
            "policy_name": record.policy_id,
            "policy_checkpoint": record.checkpoint,
            "model_family": record.model_family,
            "robocasa_manifest_record": record.to_dict(),
        }
        if not _check_existing_pickle(env_path, record.rollout_id, resume):
            atomic_pickle(env_path, env_record)
        video_link = env_path.with_suffix(".mp4")
        if record.video_path:
            source_video = (dataset_dir / record.video_path).resolve()
            if video_link.exists() or video_link.is_symlink():
                if not resume:
                    raise FileExistsError(f"Video link exists: {video_link}")
            else:
                video_link.symlink_to(os.path.relpath(source_video, video_link.parent))
        with np.load(dataset_dir / record.tensor_path, allow_pickle=False) as payload:
            features = payload["features"]
            chunks = payload["policy_action_chunks"]
            steps = payload["inference_environment_steps"]
        policy_paths = []
        for inference_index in range(record.valid_sequence_length):
            policy_path = policy_dir / (
                f"step_{global_policy_index:012d}--{record.rollout_id}--"
                f"infer_{inference_index:06d}--meta.pkl"
            )
            policy_record = {
                "rollout_id": record.rollout_id,
                "inference_index": inference_index,
                "environment_step": int(steps[inference_index]),
                "pre_velocity": np.asarray(features[inference_index], dtype=np.float32),
                "actions": np.asarray(chunks[inference_index], dtype=np.float32),
                "feature_layer": record.feature_layer,
                "model_family": record.model_family,
                "policy_name": record.policy_id,
                "policy_checkpoint": record.checkpoint,
            }
            if not _check_existing_pickle(
                policy_path,
                record.rollout_id,
                resume,
                inference_index=inference_index,
            ):
                atomic_pickle(policy_path, policy_record)
                materialized_bytes += policy_path.stat().st_size
            policy_paths.append(str(policy_path.relative_to(output_dir)))
            global_policy_index += 1
        mapping.append(
            {
                "rollout_id": record.rollout_id,
                "source_feature_path": record.tensor_path,
                "env_record": str(env_path.relative_to(output_dir)),
                "policy_records": policy_paths,
                "episode_success": int(not record.failed),
            }
        )
    format_by_family = {
        "pi0": "official_safe_pizero_env_records_policy_records",
        "rldx1": "official_safe_rldx1_env_records_policy_records",
        "xiaomi_robotics_1": (
            "official_safe_xiaomi_robotics_1_env_records_policy_records"
        ),
    }
    model_families = plan["model_families"]
    report = {
        "schema_version": 1,
        "format": (
            format_by_family[model_families[0]]
            if len(model_families) == 1
            else "official_safe_mixed_model_families_env_records_policy_records"
        ),
        "complete": True,
        "created_at": utc_now(),
        **plan,
        "materialized_feature_bytes_this_run": materialized_bytes,
        "note": (
            "The official SAFE loader requires one pickle per inference, so raw feature arrays "
            "are materialized. Videos use symlinks and the source rollout ID is preserved."
        ),
        "mapping": mapping,
    }
    atomic_write_json(report_path, report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unregistered-tasks", action="store_true")
    parser.add_argument("--allow-unregistered-atomic-tasks", action="store_true")
    parser.add_argument("--successes-per-task", type=int)
    parser.add_argument("--failures-per-task", type=int)
    parser.add_argument(
        "--rollouts-per-task",
        type=int,
        help="Select exactly this many natural-outcome rollouts per task",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a compact result; the complete mapping remains in conversion_report.json",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = export_to_official_safe(
            args.dataset_dir,
            args.output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            allow_unregistered=(
                args.allow_unregistered_tasks
                or args.allow_unregistered_atomic_tasks
            ),
            rollouts_per_task=args.rollouts_per_task,
            successes_per_task=args.successes_per_task,
            failures_per_task=args.failures_per_task,
            selection_seed=args.selection_seed,
        )
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"error: {error}") from error
    printed = report
    if args.summary_only and not report.get("dry_run"):
        printed = {
            key: report.get(key)
            for key in (
                "complete",
                "format",
                "source_dataset",
                "output_dir",
                "num_rollouts",
                "num_policy_records",
                "selection",
            )
        }
    print(json.dumps(printed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
