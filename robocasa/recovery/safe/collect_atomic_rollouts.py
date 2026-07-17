"""Resumable atomic-task RoboCasa rollout collection with official SAFE π0 features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np

from .atomic_tasks import registered_atomic_task_horizons, validate_atomic_tasks
from .collect_rollouts import collect_single_rollout
from .dataset import MANIFEST_NAME, load_manifest, save_rollout
from .schema import SAFE_SCHEMA_VERSION, SafeRolloutMetadata, compatibility_key


SAFE_COMMIT = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
OFFICIAL_SAFE_OPENPI_COMMIT = "9c99ed53f6a0c9be93a1c63cee5792620777d96b"
ROBOCASA_OPENPI_COMMIT = "5a6beda9ff99da30b4e1b59320f6a32971d7c397"
SUMMARY_NAME = "summary.json"
ERRORS_NAME = "errors.jsonl"
SKIPPED_NAME = "skipped.jsonl"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)
    return path


def append_jsonl(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(value, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_jsonl(path: str | Path):
    path = Path(path)
    if not path.exists():
        return []
    values = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return values


def current_robocasa_commit(repo_root=None):
    repo_root = Path(repo_root or Path(__file__).resolve().parents[3])
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_policy_config(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"--policy-config must be a JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("--policy-config must be a JSON object")
    return parsed


def stable_rollout_id(task_name, seed, identity):
    payload = json.dumps(
        {"task_name": task_name, "environment_seed": int(seed), **identity},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def planned_seeds(args):
    if args.seed_end is not None:
        if args.seed_end < args.seed:
            raise ValueError("--seed-end must be greater than or equal to --seed")
        return list(range(args.seed, args.seed_end + 1))
    if args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be positive")
    return list(range(args.seed, args.seed + args.num_rollouts))


def quota_reached(successes, failures, success_quota, failure_quota):
    if success_quota is None and failure_quota is None:
        return False
    return (success_quota is None or successes >= success_quota) and (
        failure_quota is None or failures >= failure_quota
    )


def class_quota_reached(
    *, failed, successes, failures, success_quota, failure_quota
):
    quota = failure_quota if failed else success_quota
    count = failures if failed else successes
    return quota is not None and count >= quota


def artifact_paths(output_dir, task_name, rollout_id, record_actions, record_videos):
    output_dir = Path(output_dir)
    feature_path = output_dir / "rollouts" / f"{rollout_id}.npz"
    action_path = (
        output_dir / "actions" / task_name / f"{rollout_id}.npz"
        if record_actions
        else None
    )
    return {
        "feature": feature_path,
        "feature_temp": feature_path.with_suffix(".tmp.npz"),
        "action": action_path,
        "action_temp": action_path.with_suffix(".tmp.npz") if action_path else None,
        "video": (
            output_dir / "videos" / task_name / f"{rollout_id}.mp4"
            if record_videos
            else None
        ),
    }


def quarantine_incomplete(output_dir, rollout_id, paths, reason):
    moved = []
    destination = Path(output_dir) / "incomplete" / rollout_id
    for kind, path in paths.items():
        if path is None or not Path(path).exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"{kind}--{Path(path).name}"
        if target.exists():
            target = destination / f"{utc_now().replace(':', '-')}-{kind}--{Path(path).name}"
        shutil.move(str(path), str(target))
        moved.append(str(target))
    return {"reason": reason, "quarantined_paths": moved}


def save_actions_atomic(actions, path):
    from robocasa.recovery.create_recovery_failure_dataset import stack_action_payloads

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **stack_action_payloads(actions))
    os.replace(temp, path)
    return path


def make_summary(config, records, errors, skipped, *, partial):
    per_task = defaultdict(lambda: Counter(successes=0, failures=0))
    for record in records:
        key = "failures" if record.failed else "successes"
        per_task[record.task_name][key] += 1
    per_task_summary = {}
    for task in config.get("tasks", sorted(per_task)):
        counts = per_task[task]
        task_summary = dict(counts)
        if (
            config.get("success_quota") is not None
            or config.get("failure_quota") is not None
        ):
            task_summary["quota_reached"] = quota_reached(
                counts["successes"],
                counts["failures"],
                config.get("success_quota"),
                config.get("failure_quota"),
            )
        per_task_summary[task] = task_summary
    return {
        "schema_version": SAFE_SCHEMA_VERSION,
        "dataset_type": "robocasa_atomic_safe_rollouts",
        "partial": bool(partial),
        "updated_at": utc_now(),
        "config": config,
        "counts": {
            "valid_rollouts": len(records),
            "successes": sum(not record.failed for record in records),
            "failures": sum(record.failed for record in records),
            "errors": len(errors),
            "skipped": len(skipped),
        },
        "per_task": per_task_summary,
        "attempted": [
            {
                key: event.get(key)
                for key in (
                    "task_name",
                    "environment_seed",
                    "environment_reset_index",
                    "rollout_horizon",
                    "rollout_id",
                    "status",
                )
            }
            for event in errors + skipped
        ]
        + [
            {
                "task_name": record.task_name,
                "environment_seed": record.environment_seed,
                "environment_reset_index": record.environment_reset_index,
                "rollout_horizon": record.rollout_horizon,
                "rollout_id": record.rollout_id,
                "status": "valid",
            }
            for record in records
        ],
    }


def _runtime():
    from robocasa.recovery.evaluate_recovery_benchmark import (
        call_factory,
        load_factory,
        make_env,
        open_video_writer,
        parse_policy_args,
    )
    from robocasa.recovery.recovery_rollout import (
        _append_video_frame_from_env,
        _is_task_success,
        _step_env,
    )

    return {
        "call_factory": call_factory,
        "load_factory": load_factory,
        "make_env": make_env,
        "open_video_writer": open_video_writer,
        "parse_policy_args": parse_policy_args,
        "append_frame": _append_video_frame_from_env,
        "success_fn": _is_task_success,
        "step_fn": _step_env,
    }


def prepare_plan(args):
    if args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be positive")
    if args.replan_steps <= 0:
        raise ValueError("--replan-steps must be positive")
    if args.horizon is not None and args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.video_frame_stride <= 0:
        raise ValueError("--video-frame-stride must be positive")
    if args.max_errors is not None and args.max_errors <= 0:
        raise ValueError("--max-errors must be positive")
    if args.seed_protocol == "official_openpi" and args.seed_end is not None:
        raise ValueError(
            "--seed-end is incompatible with --seed-protocol official_openpi"
        )
    for name in ("success_quota", "failure_quota"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if (
        args.retain_only_quota
        and args.success_quota is None
        and args.failure_quota is None
    ):
        raise ValueError(
            "--retain-only-quota requires --success-quota and/or --failure-quota"
        )
    tasks = validate_atomic_tasks(
        args.tasks,
        allow_unregistered=args.allow_unregistered_atomic_tasks,
    )
    if args.horizon is not None:
        task_horizons = {task: args.horizon for task in tasks}
        horizon_source = "command_line_override"
    else:
        if args.allow_unregistered_atomic_tasks:
            raise ValueError(
                "--horizon is required with --allow-unregistered-atomic-tasks"
            )
        registered_horizons = registered_atomic_task_horizons()
        missing_horizons = [task for task in tasks if task not in registered_horizons]
        if missing_horizons:
            raise ValueError(
                "Official horizons are missing for atomic tasks: "
                + ", ".join(missing_horizons)
            )
        task_horizons = {task: registered_horizons[task] for task in tasks}
        horizon_source = "robocasa_dataset_registry"
    if args.seed_protocol == "official_openpi":
        seeds = [args.seed]
        attempt_coordinates = [
            (args.seed, reset_index) for reset_index in range(args.num_rollouts)
        ]
    else:
        seeds = planned_seeds(args)
        attempt_coordinates = [(seed, None) for seed in seeds]
    policy_config = parse_policy_config(args.policy_config)
    robocasa_commit = args.robocasa_commit or current_robocasa_commit()
    identity = {
        "split": args.split,
        "seed_protocol": args.seed_protocol,
        "policy_name": args.policy_name,
        "policy_checkpoint": args.checkpoint,
        "policy_config": policy_config,
        "replan_steps": args.replan_steps,
        "openpi_repository_commit": args.openpi_repository_commit,
    }
    attempts = []
    for task in tasks:
        for seed, reset_index in attempt_coordinates:
            attempt_identity = {
                **identity,
                "rollout_horizon": task_horizons[task],
                "environment_reset_index": reset_index,
            }
            attempts.append(
                {
                    "task_name": task,
                    "environment_seed": seed,
                    "environment_reset_index": reset_index,
                    "rollout_horizon": task_horizons[task],
                    "rollout_id": stable_rollout_id(task, seed, attempt_identity),
                }
            )
    config = {
        "tasks": tasks,
        "seeds": seeds,
        "base_environment_seed": args.seed,
        "seed_protocol": args.seed_protocol,
        "environment_reset_indices": (
            list(range(args.num_rollouts))
            if args.seed_protocol == "official_openpi"
            else None
        ),
        "split": args.split,
        "output_dir": str(args.output_dir),
        "policy_module": args.policy_module,
        "policy_name": args.policy_name,
        "policy_checkpoint": args.checkpoint,
        "policy_config": policy_config,
        "host": args.host,
        "port": args.port,
        "replan_steps": args.replan_steps,
        "task_horizons": task_horizons,
        "horizon_source": horizon_source,
        "record_actions": args.record_actions,
        "record_videos": args.record_videos,
        "video_frame_stride": args.video_frame_stride,
        "record_safe_features": args.record_safe_features,
        "success_quota": args.success_quota,
        "failure_quota": args.failure_quota,
        "retain_only_quota": args.retain_only_quota,
        "max_errors": args.max_errors,
        "safe_repository_commit": args.safe_repository_commit,
        "official_safe_openpi_commit": args.official_safe_openpi_commit,
        "openpi_repository_commit": args.openpi_repository_commit,
        "robocasa_commit": robocasa_commit,
    }
    return {"config": config, "identity": identity, "attempts": attempts}


def _assert_resume_compatible(previous, current):
    keys = (
        "tasks",
        "split",
        "base_environment_seed",
        "seed_protocol",
        "environment_reset_indices",
        "policy_module",
        "policy_name",
        "policy_checkpoint",
        "policy_config",
        "replan_steps",
        "task_horizons",
        "horizon_source",
        "record_actions",
        "record_videos",
        "video_frame_stride",
        "record_safe_features",
        "success_quota",
        "failure_quota",
        "retain_only_quota",
        "max_errors",
        "openpi_repository_commit",
    )
    mismatches = [key for key in keys if previous.get(key) != current.get(key)]
    if mismatches:
        raise ValueError("Resume configuration is incompatible for: " + ", ".join(mismatches))


def run_collection(args, runtime=None):
    plan = prepare_plan(args)
    if not args.record_safe_features and not args.dry_run:
        raise ValueError("A valid SAFE dataset requires --record-safe-features")
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return {"dry_run": True, **plan}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_NAME
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(f"Dataset already exists at {output_dir}; pass --resume")
        previous_summary = json.loads(summary_path.read_text())
        _assert_resume_compatible(previous_summary.get("config", {}), plan["config"])
    elif (output_dir / MANIFEST_NAME).exists() and not args.resume:
        raise FileExistsError(f"Manifest already exists at {output_dir}; pass --resume")

    records = load_manifest(output_dir) if (output_dir / MANIFEST_NAME).exists() else []
    errors = read_jsonl(output_dir / ERRORS_NAME)
    skipped = read_jsonl(output_dir / SKIPPED_NAME)
    record_by_id = {record.rollout_id: record for record in records}
    compatibility_keys = {compatibility_key(record) for record in records}
    if len(compatibility_keys) > 1:
        raise ValueError("Existing dataset contains incompatible SAFE schemas")

    runtime = runtime or _runtime()
    factory = runtime["load_factory"](args.policy_module)
    policy_args = runtime["parse_policy_args"](args.policy_arg)
    policy_args.update(
        {
            "host": args.host,
            "port": args.port,
            "replan_steps": args.replan_steps,
            "collect_safe_features": True,
            "policy_name": args.policy_name,
            "policy_checkpoint": args.checkpoint,
        }
    )
    counts = defaultdict(lambda: Counter(successes=0, failures=0))
    for record in records:
        counts[record.task_name]["failures" if record.failed else "successes"] += 1
    atomic_write_json(
        summary_path,
        make_summary(plan["config"], records, errors, skipped, partial=True),
    )

    attempts_by_task = defaultdict(list)
    for attempt in plan["attempts"]:
        attempts_by_task[attempt["task_name"]].append(attempt)
    for task_name in plan["config"]["tasks"]:
        task_attempts = attempts_by_task[task_name]
        shared_env = None
        shared_policy = None
        for attempt_index, attempt in enumerate(task_attempts):
            rollout_id = attempt["rollout_id"]
            seed = attempt["environment_seed"]
            rollout_horizon = attempt["rollout_horizon"]
            paths = artifact_paths(
                output_dir,
                task_name,
                rollout_id,
                args.record_actions,
                args.record_videos,
            )
            if rollout_id in record_by_id:
                if args.resume:
                    completed = record_by_id[rollout_id]
                    required_completed = [("SAFE features", completed.tensor_path)]
                    if completed.action_recording_requested:
                        required_completed.append(("actions", completed.action_path))
                    if completed.video_recording_requested:
                        required_completed.append(("video", completed.video_path))
                    missing_completed = [
                        f"{kind}: {relative_path!r}"
                        for kind, relative_path in required_completed
                        if relative_path is None
                        or not (output_dir / relative_path).is_file()
                    ]
                    if missing_completed:
                        raise ValueError(
                            f"Completed manifest record {rollout_id} has missing artifacts: "
                            f"{missing_completed}"
                        )
                    later_collection_pending = any(
                        later["rollout_id"] not in record_by_id
                        for later in task_attempts[attempt_index + 1 :]
                    )
                    if (
                        args.seed_protocol == "official_openpi"
                        and later_collection_pending
                    ):
                        if shared_env is None:
                            shared_env = runtime["make_env"](
                                task_name,
                                args.env_interface,
                                args.split,
                                seed,
                                args.record_videos,
                            )
                            try:
                                shared_policy = runtime["call_factory"](
                                    factory, shared_env, policy_args
                                )
                            except Exception:
                                shared_env.close()
                                shared_env = None
                                raise
                        shared_env.reset()
                        reset_policy = getattr(shared_policy, "reset", None)
                        if reset_policy is not None:
                            reset_policy()
                    event = {**attempt, "status": "resume_completed", "created_at": utc_now()}
                    append_jsonl(output_dir / SKIPPED_NAME, event)
                    skipped.append(event)
                    continue
                raise FileExistsError(f"Rollout {rollout_id} is already complete")
            existing = [str(path) for path in paths.values() if path is not None and Path(path).exists()]
            if existing:
                if not args.resume:
                    raise FileExistsError(f"Incomplete artifacts exist for {rollout_id}: {existing}")
                detail = quarantine_incomplete(output_dir, rollout_id, paths, "resume_incomplete")
                event = {
                    **attempt,
                    "status": "incomplete_quarantined",
                    "created_at": utc_now(),
                    **detail,
                }
                append_jsonl(output_dir / ERRORS_NAME, event)
                errors.append(event)
            if quota_reached(
                counts[task_name]["successes"],
                counts[task_name]["failures"],
                args.success_quota,
                args.failure_quota,
            ):
                for remaining in task_attempts[attempt_index:]:
                    event = {
                        **remaining,
                        "status": "skipped_quota_reached",
                        "created_at": utc_now(),
                    }
                    append_jsonl(output_dir / SKIPPED_NAME, event)
                    skipped.append(event)
                break

            env = shared_env
            writer = None
            try:
                if env is None:
                    env = runtime["make_env"](
                        task_name,
                        args.env_interface,
                        args.split,
                        seed,
                        args.record_videos,
                    )
                if args.seed_protocol == "official_openpi":
                    if shared_policy is None:
                        try:
                            shared_policy = runtime["call_factory"](
                                factory, env, policy_args
                            )
                        except Exception:
                            env.close()
                            env = None
                            raise
                    shared_env = env
                    policy = shared_policy
                else:
                    policy = runtime["call_factory"](factory, env, policy_args)
                writer = runtime["open_video_writer"](paths["video"], args.video_fps)

                def frame_fn(environment, video_writer, obs, previous):
                    return runtime["append_frame"](
                        environment,
                        video_writer,
                        camera_name=args.video_camera_name,
                        height=args.video_height,
                        width=args.video_width,
                        obs=obs,
                        previous_frame=previous,
                    )

                rollout = collect_single_rollout(
                    policy,
                    env,
                    horizon=rollout_horizon,
                    step_fn=runtime["step_fn"],
                    success_fn=runtime["success_fn"],
                    video_writer=writer,
                    frame_fn=frame_fn,
                    video_frame_stride=args.video_frame_stride,
                    require_safe_features=True,
                )
                if writer is not None:
                    writer.close()
                    writer = None
                if args.record_videos and not paths["video"].is_file():
                    raise RuntimeError("Video recording was requested but no video file was finalized")
                failed = not rollout["success"]
                if args.retain_only_quota and class_quota_reached(
                    failed=failed,
                    successes=counts[task_name]["successes"],
                    failures=counts[task_name]["failures"],
                    success_quota=args.success_quota,
                    failure_quota=args.failure_quota,
                ):
                    if paths["video"] is not None and paths["video"].exists():
                        paths["video"].unlink()
                    event = {
                        **attempt,
                        "status": "skipped_class_quota_reached",
                        "success": not failed,
                        "failed": failed,
                        "created_at": utc_now(),
                    }
                    append_jsonl(output_dir / SKIPPED_NAME, event)
                    skipped.append(event)
                    continue
                if args.record_actions:
                    save_actions_atomic(rollout["actions"], paths["action"])
                feature_meta = rollout["feature_metadata"]
                if feature_meta.get("policy_name") != args.policy_name:
                    raise RuntimeError(
                        "SAFE policy identity disagrees with --policy-name: "
                        f"{feature_meta.get('policy_name')!r} != {args.policy_name!r}"
                    )
                if feature_meta.get("policy_checkpoint") != args.checkpoint:
                    raise RuntimeError(
                        "SAFE checkpoint identity disagrees with --checkpoint: "
                        f"{feature_meta.get('policy_checkpoint')!r} != {args.checkpoint!r}"
                    )
                metadata = SafeRolloutMetadata(
                    rollout_id=rollout_id,
                    task_name=task_name,
                    task_instruction=rollout["instruction"],
                    environment_seed=seed,
                    environment_split=args.split,
                    seed_protocol=args.seed_protocol,
                    environment_reset_index=attempt["environment_reset_index"],
                    video_frame_stride=args.video_frame_stride,
                    policy_id=args.policy_name,
                    checkpoint=args.checkpoint,
                    failed=failed,
                    num_env_steps=rollout["num_env_steps"],
                    inference_env_steps=rollout["inference_env_steps"],
                    valid_sequence_length=len(rollout["inference_env_steps"]),
                    action_horizon=int(feature_meta["action_horizon"]),
                    replan_steps=args.replan_steps,
                    feature_layer=feature_meta["feature_layer"],
                    feature_aggregation=feature_meta.get(
                        "feature_aggregation", feature_meta.get("aggregation", "raw")
                    ),
                    feature_dtype=feature_meta["feature_dtype"],
                    feature_shape=list(rollout["features"].shape),
                    flow_steps=int(rollout["features"].shape[1]),
                    termination_reason=rollout["termination_reason"],
                    timeout_horizon=rollout_horizon,
                    video_path=(
                        str(paths["video"].relative_to(output_dir))
                        if paths["video"] is not None
                        else None
                    ),
                    policy_config={
                        **plan["config"]["policy_config"],
                        "official_safe_openpi_commit": args.official_safe_openpi_commit,
                    },
                    rollout_horizon=rollout_horizon,
                    action_path=(
                        str(paths["action"].relative_to(output_dir))
                        if paths["action"] is not None
                        else None
                    ),
                    safe_repository_commit=args.safe_repository_commit,
                    openpi_repository_commit=args.openpi_repository_commit,
                    robocasa_commit=plan["config"]["robocasa_commit"],
                    action_recording_requested=args.record_actions,
                    video_recording_requested=args.record_videos,
                )
                new_compatibility_key = compatibility_key(metadata)
                if compatibility_keys and new_compatibility_key not in compatibility_keys:
                    raise RuntimeError(
                        "SAFE feature schema/checkpoint is incompatible with existing dataset"
                    )
                save_rollout(
                    output_dir,
                    metadata,
                    rollout["features"],
                    rollout["policy_action_chunks"],
                )
                records.append(metadata)
                record_by_id[rollout_id] = metadata
                counts[task_name]["failures" if metadata.failed else "successes"] += 1
                compatibility_keys.add(new_compatibility_key)
            except KeyboardInterrupt:
                if shared_env is not None:
                    shared_env.close()
                    shared_env = None
                    env = None
                raise
            except Exception:
                if writer is not None:
                    writer.close()
                    writer = None
                detail = quarantine_incomplete(output_dir, rollout_id, paths, "collection_error")
                event = {
                    **attempt,
                    "status": "error",
                    "created_at": utc_now(),
                    "error": traceback.format_exc(),
                    **detail,
                }
                append_jsonl(output_dir / ERRORS_NAME, event)
                errors.append(event)
                if args.max_errors is not None and len(errors) >= args.max_errors:
                    if shared_env is not None:
                        shared_env.close()
                        shared_env = None
                        env = None
                    raise RuntimeError(
                        f"Collection stopped after reaching --max-errors={args.max_errors}"
                    )
                if not args.continue_on_error:
                    if shared_env is not None:
                        shared_env.close()
                        shared_env = None
                        env = None
                    raise
            finally:
                if writer is not None:
                    writer.close()
                if env is not None and env is not shared_env:
                    env.close()
                atomic_write_json(
                    summary_path,
                    make_summary(plan["config"], records, errors, skipped, partial=True),
                )
        if shared_env is not None:
            shared_env.close()
    quotas_requested = args.success_quota is not None or args.failure_quota is not None
    target_reached = not quotas_requested or all(
        quota_reached(
            counts[task_name]["successes"],
            counts[task_name]["failures"],
            args.success_quota,
            args.failure_quota,
        )
        for task_name in plan["config"]["tasks"]
    )
    summary = make_summary(
        plan["config"], records, errors, skipped, partial=not target_reached
    )
    atomic_write_json(summary_path, summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--allow-unregistered-atomic-tasks", action="store_true")
    parser.add_argument("--num-rollouts", type=int, default=1, help="Maximum attempts per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-end", type=int)
    parser.add_argument(
        "--seed-protocol",
        choices=("rollout_index", "official_openpi"),
        default="rollout_index",
        help=(
            "Use a distinct seed per rollout, or match the official OpenPI evaluator "
            "by creating one environment per task and repeatedly resetting it"
        ),
    )
    parser.add_argument("--policy-module", default="robocasa.recovery.openpi_websocket_policy:make_policy")
    parser.add_argument("--policy-arg", action="append", default=[])
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-config", default="{}", help="JSON object")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument(
        "--horizon",
        type=int,
        help="Override the official registry horizon for every selected task",
    )
    parser.add_argument("--env-interface", choices=["gym", "robosuite"], default="gym")
    parser.add_argument("--split", default="test")
    parser.add_argument("--record-safe-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--success-quota", type=int)
    parser.add_argument("--failure-quota", type=int)
    parser.add_argument(
        "--retain-only-quota",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After one class reaches its quota, keep attempting rollouts but discard "
            "additional examples of that class instead of storing an imbalanced dataset"
        ),
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        help="Stop collection after this many recorded errors",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--video-camera-name", default="robot0_agentview_center")
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--video-width", type=int, default=768)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--video-frame-stride",
        type=int,
        default=1,
        help="Record one frame every N environment steps (official OpenPI uses 2)",
    )
    parser.add_argument("--safe-repository-commit", default=SAFE_COMMIT)
    parser.add_argument("--official-safe-openpi-commit", default=OFFICIAL_SAFE_OPENPI_COMMIT)
    parser.add_argument("--openpi-repository-commit", default=ROBOCASA_OPENPI_COMMIT)
    parser.add_argument("--robocasa-commit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run_collection(args)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
