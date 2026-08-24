"""Run the bounded Phase 2 complete-snapshot replay validation.

The runner collects new nominal parents, commits two predeclared inference
boundaries in one target semantic stage per task, and then executes two
same-seed repeats, four explicit candidate-seed branches, and an environment-
only negative control. Recovery quality is intentionally out of scope: this
phase validates counterfactual branch identification.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import traceback

import numpy as np

from robocasa.recovery.counterfactual_branch import (
    BranchSpec,
    analyze_phase2_replay,
    load_branch_records,
    run_counterfactual_branch,
    save_branch_result,
)
from robocasa.recovery.full_snapshot import (
    capture_full_snapshot,
    load_full_snapshot,
    restore_full_snapshot,
    save_full_snapshot,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
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
                    f"Invalid JSONL line {line_number} in {path}"
                ) from error
    return rows


def current_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def repository_commit(path):
    if path is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def checkpoint_provenance(model_path, *, include_shards):
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Model path is not a directory: {root}")
    names = ("config.json", "modeling_mibot.py", "model.safetensors.index.json")
    files = [root / name for name in names if (root / name).is_file()]
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"Checkpoint has no safetensor shards: {root}")
    if include_shards:
        print(
            f"Hashing {len(shards)} checkpoint shard(s) for Phase 2 provenance...",
            flush=True,
        )
        files.extend(shards)
    return {
        "path": str(root),
        "content_hash_scope": (
            "metadata_and_all_safetensor_shards"
            if include_shards
            else "metadata_only_dry_run"
        ),
        "files": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "safetensor_shards_present": len(shards),
        "safetensor_shards_hashed": len(shards) if include_shards else 0,
    }


def parse_task_stages(values, option="--target-stage"):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"{option} entries must use TASK=TASK::STAGE or TASK=STAGE"
            )
        task, stage = value.split("=", 1)
        task, stage = task.strip(), stage.strip()
        if not task or not stage or task in result:
            raise ValueError(f"Invalid or duplicate {option}: {value}")
        if "::" in stage and stage.split("::", 1)[0] != task:
            raise ValueError(f"Task prefix differs in {option}: {value}")
        result[task] = stage if "::" in stage else f"{task}::{stage}"
    return result


def parse_target_stages(values):
    return parse_task_stages(values, "--target-stage")


def load_horizons(runtime_bundle, target_stages):
    bundle = json.loads(Path(runtime_bundle).read_text())
    if bundle.get("status") != "frozen":
        raise ValueError("Stage-aware runtime bundle is not frozen")
    stage_horizons = bundle.get("horizons", {}).get("stage", {})
    missing = sorted(set(target_stages.values()) - set(stage_horizons))
    if missing:
        raise ValueError(f"Runtime bundle has no training horizon for: {missing}")
    return bundle, {
        task: int(stage_horizons[stage]) for task, stage in target_stages.items()
    }


def stable_parent_id(task_name, environment_seed, reset_index, protocol):
    payload = json.dumps(
        {
            "task_name": task_name,
            "environment_seed": int(environment_seed),
            "environment_reset_index": int(reset_index),
            "protocol": protocol,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _runtime():
    from robocasa.recovery.evaluate_recovery_benchmark import (
        call_factory,
        load_factory,
        make_env,
        parse_policy_args,
    )
    from robocasa.recovery.recovery_rollout import (
        _is_task_success,
        _step_env,
        call_policy,
    )
    from robocasa.recovery.subtask_eval import build_subtask_trace, get_subtask_eval

    return {
        "call_factory": call_factory,
        "load_factory": load_factory,
        "make_env": make_env,
        "parse_policy_args": parse_policy_args,
        "success_fn": _is_task_success,
        "step_fn": _step_env,
        "call_policy": call_policy,
        "build_subtask_trace": build_subtask_trace,
        "get_subtask_eval": get_subtask_eval,
    }


def build_plan(args):
    if args.num_parents_per_task < 1:
        raise ValueError("--num-parents-per-task must be positive")
    if args.max_parent_attempts_per_task < args.num_parents_per_task:
        raise ValueError(
            "--max-parent-attempts-per-task must be at least " "--num-parents-per-task"
        )
    if args.candidate_count < 2:
        raise ValueError("--candidate-count must be at least two")
    if args.suffix_steps < 1 or args.parent_horizon < 1:
        raise ValueError("horizons must be positive")
    if args.canonical_camera_render_repeats < 1:
        raise ValueError("--canonical-camera-render-repeats must be positive")
    if args.canonical_camera_observations and args.env_interface != "gym":
        raise ValueError("Canonical camera observations require --env-interface gym")
    if not 0 < args.landmark_fraction <= 1:
        raise ValueError("--landmark-fraction must be in (0, 1]")
    targets = parse_target_stages(args.target_stage)
    triggers = parse_task_stages(args.trigger_stage, "--trigger-stage")
    if set(targets) != set(args.tasks) or set(triggers) != set(args.tasks):
        raise ValueError(
            "Tasks, frozen target stages, and live trigger stages differ: "
            f"tasks={sorted(args.tasks)} targets={sorted(targets)} "
            f"triggers={sorted(triggers)}"
        )
    bundle, horizons = load_horizons(args.runtime_bundle, targets)
    candidate_parents = []
    for task_index, task_name in enumerate(args.tasks):
        for reset_index in range(args.max_parent_attempts_per_task):
            seed = (
                args.seed + task_index * args.max_parent_attempts_per_task + reset_index
            )
            candidate_parents.append(
                {
                    "task_name": task_name,
                    "environment_seed": seed,
                    "environment_reset_index": reset_index,
                    "parent_id": stable_parent_id(
                        task_name,
                        seed,
                        reset_index,
                        "phase2_complete_snapshot_replay",
                    ),
                }
            )
    source_identities = {
        tuple(value) for value in bundle.get("source_parent_identities", [])
    }
    planned_identities = {
        (
            row["task_name"],
            int(row["environment_seed"]),
            int(row["environment_reset_index"]),
        )
        for row in candidate_parents
    }
    identity_overlap = sorted(source_identities & planned_identities)
    if identity_overlap:
        raise ValueError(
            "Planned Phase 2 identities overlap frozen development identities: "
            f"{identity_overlap[:5]}"
        )
    target_parents = len(args.tasks) * args.num_parents_per_task
    checkpoint = checkpoint_provenance(args.model_path, include_shards=not args.dry_run)
    server_entrypoint = (
        None
        if args.server_repository is None
        else args.server_repository.resolve() / "deploy" / "server.py"
    )
    if server_entrypoint is not None and not server_entrypoint.is_file():
        raise ValueError(f"Xiaomi server entrypoint is missing: {server_entrypoint}")
    plan = {
        "schema_version": 3,
        "protocol": "phase2_complete_snapshot_replay",
        "created_at": utc_now(),
        "robocasa_commit": current_commit(),
        "runtime_bundle": str(Path(args.runtime_bundle).resolve()),
        "runtime_bundle_sha256": sha256_file(args.runtime_bundle),
        "runtime_bundle_primary_detector": bundle.get("primary_detector"),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_provenance": checkpoint,
        "checkpoint_provenance_sha256": sha256_json(checkpoint),
        "policy_module": args.policy_module,
        "server_repository": (
            None
            if args.server_repository is None
            else str(args.server_repository.resolve())
        ),
        "server_repository_commit": repository_commit(args.server_repository),
        "server_entrypoint": (
            None if server_entrypoint is None else str(server_entrypoint)
        ),
        "server_entrypoint_sha256": (
            None if server_entrypoint is None else sha256_file(server_entrypoint)
        ),
        "tasks": list(args.tasks),
        "target_stages": targets,
        "trigger_stages": triggers,
        "training_stage_horizons": horizons,
        "snapshot_prefix": int(args.snapshot_prefix),
        "landmark_fraction": float(args.landmark_fraction),
        "num_parents_per_task": int(args.num_parents_per_task),
        "max_parent_attempts_per_task": int(args.max_parent_attempts_per_task),
        "expected_parents": target_parents,
        "expected_snapshots": target_parents * 2,
        "same_seed_repeats": 2,
        "candidate_count": int(args.candidate_count),
        "environment_only_controls": 1,
        "expected_primary_branches": target_parents * 2 * (2 + args.candidate_count),
        "expected_total_branches": target_parents * 2 * (3 + args.candidate_count),
        "suffix_steps": int(args.suffix_steps),
        "parent_horizon": int(args.parent_horizon),
        "seed": int(args.seed),
        "candidate_seed_base": int(args.candidate_seed_base),
        "ordinary_sampling_seed_base": int(args.ordinary_sampling_seed_base),
        "split": args.split,
        "env_interface": args.env_interface,
        "canonical_camera_observations": bool(
            args.canonical_camera_observations
        ),
        "canonical_camera_render_repeats": int(
            args.canonical_camera_render_repeats
        ),
        "fresh_branch_contexts": bool(args.fresh_branch_contexts),
        "deterministic_environment_construction": True,
        "development_identity_overlap": len(identity_overlap),
        "candidate_parents": candidate_parents,
    }
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def _branch_specs(snapshot, nominal_seed, args, snapshot_index):
    specs = [
        BranchSpec(
            branch_id=f"{snapshot.snapshot_id}--repeat-{repeat_index}",
            kind="same_seed_repeat",
            sampling_seed=nominal_seed,
            suffix_steps=args.suffix_steps,
            repeat_index=repeat_index,
        )
        for repeat_index in range(2)
    ]
    base = args.candidate_seed_base + snapshot_index * 100
    specs.extend(
        BranchSpec(
            branch_id=f"{snapshot.snapshot_id}--candidate-{candidate_index}",
            kind="candidate",
            sampling_seed=base + candidate_index,
            suffix_steps=args.suffix_steps,
        )
        for candidate_index in range(args.candidate_count)
    )
    specs.append(
        BranchSpec(
            branch_id=f"{snapshot.snapshot_id}--environment-only",
            kind="environment_only",
            sampling_seed=nominal_seed,
            suffix_steps=args.suffix_steps,
        )
    )
    return specs


def _latest_trace(runtime, subtask_evals):
    trace = runtime["build_subtask_trace"](subtask_evals)
    return trace[-1] if trace else {}


def _reset_env(env, seed):
    result = env.reset(seed=seed)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _call_with_deterministic_environment_seed(seed, function, *args, **kwargs):
    """Call legacy environment setup under a temporary reproducible RNG seed.

    Some RoboCasa fixture constructors still sample visual properties from the
    module-level Python / NumPy RNGs rather than the environment RNG. Preserve
    the caller's streams while ensuring a nominal parent and every fresh branch
    compile byte-identical model XML for the same environment seed.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        return function(*args, **kwargs)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _configure_phase2_environment(env, args):
    if not args.canonical_camera_observations:
        return
    target = getattr(env, "unwrapped", env)
    setter = getattr(target, "set_canonical_camera_observations", None)
    if not callable(setter):
        raise TypeError(
            "--canonical-camera-observations requires the RoboCasa gym wrapper"
        )
    setter(True, render_repeats=args.canonical_camera_render_repeats)


def _make_fresh_branch_context(snapshot, args, runtime, factory, policy_args):
    """Create an independently initialized renderer/environment per branch.

    Full snapshot restoration makes the causal environment and policy state
    identical.  Constructing a fresh wrapper here additionally prevents
    offscreen renderer and observable-cache history from one branch leaking
    into a later branch's camera observations.
    """
    environment_seed = int(snapshot.metadata["environment_seed"])
    env = _call_with_deterministic_environment_seed(
        environment_seed,
        runtime["make_env"],
        snapshot.task_name,
        args.env_interface,
        args.split,
        environment_seed,
        True,
    )
    policy = None
    try:
        _configure_phase2_environment(env, args)
        local_policy_args = dict(policy_args)
        local_policy_args["sampling_seed_base"] = snapshot.policy_state[
            "sampling_config"
        ]["sampling_seed_base"]
        policy = runtime["call_factory"](factory, env, local_policy_args)
        _call_with_deterministic_environment_seed(
            environment_seed,
            _reset_env,
            env,
            environment_seed,
        )
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        return env, policy
    except Exception:
        close = getattr(policy, "close", None)
        try:
            if callable(close):
                close()
        finally:
            env.close()
        raise


def _close_branch_context(env, policy):
    close = getattr(policy, "close", None)
    try:
        if callable(close):
            close()
    finally:
        env.close()


def _execute_snapshot_branches(
    snapshot,
    nominal_seed,
    args,
    runtime,
    env,
    policy,
    snapshot_index,
    *,
    factory=None,
    policy_args=None,
    completed_branch_ids=None,
):
    completed_branch_ids = set(completed_branch_ids or ())
    summaries = []
    specs = _branch_specs(snapshot, nominal_seed, args, snapshot_index)
    for spec in specs:
        if spec.branch_id in completed_branch_ids:
            continue
        branch_env = env
        branch_policy = policy
        owns_branch_context = False
        if args.fresh_branch_contexts:
            if factory is None or policy_args is None:
                raise ValueError(
                    "Fresh branch contexts require a policy factory and arguments"
                )
            branch_env, branch_policy = _make_fresh_branch_context(
                snapshot,
                args,
                runtime,
                factory,
                policy_args,
            )
            owns_branch_context = True
        try:
            if spec.kind == "environment_only":
                # First advance the policy/cache so environment-only rewind is
                # a meaningful negative control.
                restore_full_snapshot(snapshot, branch_env, branch_policy)
                mutation_action = runtime["call_policy"](
                    branch_policy, deepcopy(snapshot.observation)
                )
                runtime["step_fn"](branch_env, mutation_action)
                pop_requests = getattr(branch_policy, "pop_request_records", None)
                if callable(pop_requests):
                    pop_requests()
                pop_inference = getattr(branch_policy, "pop_inference_record", None)
                if callable(pop_inference):
                    pop_inference()
            result = run_counterfactual_branch(
                snapshot,
                branch_env,
                branch_policy,
                spec,
                step_fn=runtime["step_fn"],
                success_fn=runtime["success_fn"],
                subtask_eval_fn=runtime["get_subtask_eval"],
                restore_atol=args.restore_atol,
                restore_rtol=args.restore_rtol,
            )
            summaries.append(save_branch_result(result, args.output_dir))
        finally:
            if owns_branch_context:
                _close_branch_context(branch_env, branch_policy)
    return summaries


def _validate_resume_plan(args, plan):
    checks = {
        "tasks": list(args.tasks),
        "target_stages": parse_target_stages(args.target_stage),
        "trigger_stages": parse_task_stages(args.trigger_stage, "--trigger-stage"),
        "num_parents_per_task": int(args.num_parents_per_task),
        "max_parent_attempts_per_task": int(args.max_parent_attempts_per_task),
        "snapshot_prefix": int(args.snapshot_prefix),
        "landmark_fraction": float(args.landmark_fraction),
        "suffix_steps": int(args.suffix_steps),
        "candidate_count": int(args.candidate_count),
        "candidate_seed_base": int(args.candidate_seed_base),
        "ordinary_sampling_seed_base": int(args.ordinary_sampling_seed_base),
        "split": args.split,
        "env_interface": args.env_interface,
        "canonical_camera_observations": bool(
            args.canonical_camera_observations
        ),
        "canonical_camera_render_repeats": int(
            args.canonical_camera_render_repeats
        ),
        "fresh_branch_contexts": bool(args.fresh_branch_contexts),
        "deterministic_environment_construction": True,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.checkpoint_revision,
        "policy_module": args.policy_module,
        "server_repository": (
            None
            if args.server_repository is None
            else str(args.server_repository.resolve())
        ),
    }
    mismatches = {
        key: {"frozen": plan.get(key), "requested": value}
        for key, value in checks.items()
        if plan.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume arguments differ from frozen plan: {mismatches}")
    if sha256_file(args.runtime_bundle) != plan.get("runtime_bundle_sha256"):
        raise ValueError("Runtime bundle content changed since the frozen plan")
    if str(args.model_path.resolve()) != plan["checkpoint_provenance"]["path"]:
        raise ValueError("Model path differs from the frozen plan")
    current_checkpoint = checkpoint_provenance(args.model_path, include_shards=True)
    if sha256_json(current_checkpoint) != plan.get("checkpoint_provenance_sha256"):
        raise ValueError("Checkpoint content changed since the frozen plan")
    server_entrypoint = plan.get("server_entrypoint")
    if server_entrypoint is not None:
        if not Path(server_entrypoint).is_file():
            raise ValueError("Frozen Xiaomi server entrypoint is missing")
        if sha256_file(server_entrypoint) != plan.get("server_entrypoint_sha256"):
            raise ValueError("Xiaomi server entrypoint changed since the frozen plan")
    return plan


def _resume_snapshot_parent(
    snapshots,
    args,
    runtime,
    factory,
    policy_args,
    completed_branch_ids,
):
    if len(snapshots) != 2:
        raise ValueError("A resumable parent must have exactly two snapshots")
    parent_ids = {snapshot.parent_id for snapshot in snapshots}
    if len(parent_ids) != 1:
        raise ValueError("Resume snapshot group mixes parent IDs")
    task_names = {snapshot.task_name for snapshot in snapshots}
    if len(task_names) != 1:
        raise ValueError("Resume snapshot group mixes task names")
    if any(
        snapshot.metadata.get("plan_sha256") != snapshots[0].metadata.get("plan_sha256")
        for snapshot in snapshots
    ):
        raise ValueError("Resume snapshot group mixes frozen plans")
    metadata = snapshots[0].metadata
    environment_seed = int(metadata["environment_seed"])
    env = _call_with_deterministic_environment_seed(
        environment_seed,
        runtime["make_env"],
        snapshots[0].task_name,
        args.env_interface,
        args.split,
        environment_seed,
        True,
    )
    _configure_phase2_environment(env, args)
    local_policy_args = dict(policy_args)
    local_policy_args["sampling_seed_base"] = snapshots[0].policy_state[
        "sampling_config"
    ]["sampling_seed_base"]
    policy = None
    summaries = []
    try:
        policy = runtime["call_factory"](factory, env, local_policy_args)
        _call_with_deterministic_environment_seed(
            environment_seed,
            _reset_env,
            env,
            environment_seed,
        )
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        for snapshot in sorted(
            snapshots, key=lambda value: int(value.metadata["snapshot_index"])
        ):
            summaries.extend(
                _execute_snapshot_branches(
                    snapshot,
                    int(snapshot.metadata["nominal_sampling_seed"]),
                    args,
                    runtime,
                    env,
                    policy,
                    int(snapshot.metadata["snapshot_index"]),
                    factory=factory,
                    policy_args=local_policy_args,
                    completed_branch_ids=completed_branch_ids,
                )
            )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
        env.close()
    return {
        "parent_id": snapshots[0].parent_id,
        "task_name": snapshots[0].task_name,
        "captured": {
            (
                "prefix"
                if snapshot.trigger_name.startswith("stage_prefix")
                else "landmark"
            ): snapshot.snapshot_id
            for snapshot in snapshots
        },
        "captured_count": 2,
        "branch_count": args.candidate_count * 2 + 6,
        "target_stage": metadata["target_stage"],
        "trigger_stage": metadata["trigger_stage"],
        "landmark_cutoff": None,
        "valid": True,
        "resumed": True,
    }


def _run_parent(parent, args, plan, runtime, factory, policy_args, snapshot_index):
    task_name = parent["task_name"]
    target_stage = plan["target_stages"][task_name]
    trigger_stage = plan["trigger_stages"][task_name]
    raw_target_stage = trigger_stage.split("::", 1)[1]
    stage_horizon = plan["training_stage_horizons"][task_name]
    landmark_cutoff = max(
        args.snapshot_prefix + 1,
        int(math.ceil(args.landmark_fraction * stage_horizon)),
    )
    environment_seed = int(parent["environment_seed"])
    env = _call_with_deterministic_environment_seed(
        environment_seed,
        runtime["make_env"],
        task_name,
        args.env_interface,
        args.split,
        environment_seed,
        True,
    )
    _configure_phase2_environment(env, args)
    local_policy_args = dict(policy_args)
    local_policy_args["sampling_seed_base"] = (
        args.ordinary_sampling_seed_base + snapshot_index * 10000
    )
    policy = None
    try:
        policy = runtime["call_factory"](factory, env, local_policy_args)
        obs, _ = _call_with_deterministic_environment_seed(
            environment_seed,
            _reset_env,
            env,
            environment_seed,
        )
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        subtask_evals = [runtime["get_subtask_eval"](env)]
        current_stage = None
        local_inferences = 0
        current_stage_inferences = 0
        observed_stage_sequence = []
        stage_diagnostics = {}
        stage_unavailable_steps = 0
        environment_steps = 0
        parent_task_success = False
        parent_termination_reason = "parent_horizon"
        captured = {}
        captured_snapshots = []
        branch_summaries = []

        for environment_step in range(args.parent_horizon):
            trace_entry = _latest_trace(runtime, subtask_evals)
            raw_stage = trace_entry.get("ordered_current_subtask")
            if raw_stage != current_stage:
                current_stage = raw_stage
                local_inferences = 0
                current_stage_inferences = 0
                if raw_stage is not None:
                    observed_stage_sequence.append(raw_stage)
                    stage_diagnostics.setdefault(
                        raw_stage,
                        {
                            "visits": 0,
                            "environment_steps": 0,
                            "policy_inferences": 0,
                            "max_consecutive_policy_inferences": 0,
                        },
                    )["visits"] += 1
            if raw_stage is None:
                stage_unavailable_steps += 1
            else:
                stage_diagnostics[raw_stage]["environment_steps"] += 1
            at_boundary = bool(getattr(policy, "at_inference_boundary", False))
            trigger = None
            if raw_stage == raw_target_stage and at_boundary:
                if (
                    local_inferences == args.snapshot_prefix
                    and "prefix" not in captured
                ):
                    trigger = f"stage_prefix_{args.snapshot_prefix}"
                elif local_inferences >= landmark_cutoff and "landmark" not in captured:
                    trigger = f"stage_landmark_{args.landmark_fraction:g}"
            if trigger is not None:
                nominal_seed = getattr(policy, "next_ordinary_sampling_seed", None)
                if nominal_seed is None:
                    raise RuntimeError(
                        "Nominal continuation lacks an explicit sampling seed"
                    )
                assigned_snapshot_index = snapshot_index + len(captured_snapshots)
                snapshot = capture_full_snapshot(
                    env,
                    policy,
                    parent_id=parent["parent_id"],
                    task_name=task_name,
                    trigger_name=trigger,
                    environment_step=environment_step,
                    observation=obs,
                    subtask_eval=subtask_evals[-1],
                    metadata={
                        "target_stage": target_stage,
                        "trigger_stage": trigger_stage,
                        "local_inferences": local_inferences,
                        "training_stage_horizon": stage_horizon,
                        "robocasa_commit": plan["robocasa_commit"],
                        "runtime_bundle": plan["runtime_bundle"],
                        "runtime_bundle_sha256": plan["runtime_bundle_sha256"],
                        "checkpoint_provenance_sha256": plan[
                            "checkpoint_provenance_sha256"
                        ],
                        "server_repository_commit": plan["server_repository_commit"],
                        "plan_sha256": plan["plan_sha256"],
                        "environment_seed": parent["environment_seed"],
                        "environment_reset_index": parent["environment_reset_index"],
                        "nominal_sampling_seed": int(nominal_seed),
                        "snapshot_index": int(assigned_snapshot_index),
                        "candidate_seed_base": int(
                            args.candidate_seed_base + assigned_snapshot_index * 100
                        ),
                    },
                )
                key = "prefix" if trigger.startswith("stage_prefix") else "landmark"
                captured[key] = snapshot.snapshot_id
                captured_snapshots.append((snapshot, int(nominal_seed)))

            action = runtime["call_policy"](policy, obs)
            requests = policy.pop_request_records()
            inference = policy.pop_inference_record()
            if inference is not None:
                if not requests:
                    raise RuntimeError("Nominal inference lacks a request record")
                if raw_stage == raw_target_stage:
                    local_inferences += 1
                if raw_stage is not None:
                    current_stage_inferences += 1
                    diagnostics = stage_diagnostics[raw_stage]
                    diagnostics["policy_inferences"] += 1
                    diagnostics["max_consecutive_policy_inferences"] = max(
                        diagnostics["max_consecutive_policy_inferences"],
                        current_stage_inferences,
                    )
            obs, reward, done, info = runtime["step_fn"](env, action)
            environment_steps = environment_step + 1
            subtask_evals.append(runtime["get_subtask_eval"](env))
            if runtime["success_fn"](info=info, reward=reward, env=env):
                parent_task_success = True
                parent_termination_reason = "success"
                break
            if done:
                parent_termination_reason = "environment_done"
                break
            if len(captured) == 2:
                parent_termination_reason = "snapshot_pair_captured"
                break

        target_diagnostics = stage_diagnostics.get(
            raw_target_stage,
            {
                "visits": 0,
                "environment_steps": 0,
                "policy_inferences": 0,
                "max_consecutive_policy_inferences": 0,
            },
        )
        reachability = {
            "observed_stage_sequence": observed_stage_sequence,
            "stage_diagnostics": stage_diagnostics,
            "stage_unavailable_steps": stage_unavailable_steps,
            "target_stage_reached": bool(target_diagnostics["visits"]),
            "trigger_stage": trigger_stage,
            "target_stage_environment_steps": target_diagnostics["environment_steps"],
            "target_stage_policy_inferences": target_diagnostics["policy_inferences"],
            "target_stage_max_consecutive_policy_inferences": target_diagnostics[
                "max_consecutive_policy_inferences"
            ],
            "environment_steps": environment_steps,
            "task_success": parent_task_success,
            "termination_reason": parent_termination_reason,
        }

        if len(captured_snapshots) != 2:
            for snapshot, _ in captured_snapshots:
                snapshot_path = (
                    args.output_dir
                    / "incomplete_snapshots"
                    / f"{snapshot.snapshot_id}.pkl.gz"
                )
                save_full_snapshot(snapshot, snapshot_path)
            return {
                "parent_id": parent["parent_id"],
                "task_name": task_name,
                "captured": captured,
                "captured_count": len(captured),
                "branch_count": 0,
                "target_stage": target_stage,
                "trigger_stage": trigger_stage,
                "landmark_cutoff": landmark_cutoff,
                "snapshot_index": snapshot_index,
                "valid": False,
                "reason": "did_not_reach_both_predeclared_boundaries",
                **reachability,
            }

        for snapshot, _ in captured_snapshots:
            snapshot_path = (
                args.output_dir / "snapshots" / f"{snapshot.snapshot_id}.pkl.gz"
            )
            save_full_snapshot(snapshot, snapshot_path)
        for snapshot, nominal_seed in captured_snapshots:
            assigned_snapshot_index = int(snapshot.metadata["snapshot_index"])
            branch_summaries.extend(
                _execute_snapshot_branches(
                    snapshot,
                    nominal_seed,
                    args,
                    runtime,
                    env,
                    policy,
                    assigned_snapshot_index,
                    factory=factory,
                    policy_args=local_policy_args,
                )
            )
            snapshot_index = max(snapshot_index, assigned_snapshot_index + 1)

        return {
            "parent_id": parent["parent_id"],
            "task_name": task_name,
            "captured": captured,
            "captured_count": len(captured),
            "branch_count": len(branch_summaries),
            "target_stage": target_stage,
            "trigger_stage": trigger_stage,
            "landmark_cutoff": landmark_cutoff,
            "snapshot_index": snapshot_index,
            "valid": len(captured) == 2,
            **reachability,
        }
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
        env.close()


def run(args, runtime=None):
    if args.resume:
        plan_path = args.output_dir / "plan.json"
        if not plan_path.is_file():
            raise ValueError(f"Resume plan is missing: {plan_path}")
        plan = _validate_resume_plan(args, json.loads(plan_path.read_text()))
    else:
        plan = build_plan(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return {"dry_run": True, **plan}
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        atomic_write_json(args.output_dir / "plan.json", plan)
    atomic_write_json(
        args.output_dir / "status.json",
        {"status": "running", "updated_at": utc_now()},
    )
    runtime = runtime or _runtime()
    factory = runtime["load_factory"](args.policy_module)
    policy_args = runtime["parse_policy_args"](args.policy_arg)
    policy_args.update(
        {
            "model_path": str(args.model_path),
            "host": args.host,
            "port": args.port,
            "replan_steps": args.replan_steps,
            "collect_safe_features": True,
            "policy_name": args.policy_name,
            "policy_checkpoint": args.checkpoint,
            "safe_best_of_k": 1,
        }
    )
    parents = load_jsonl(args.output_dir / "parent_records.jsonl")
    ineligible_parents = load_jsonl(args.output_dir / "ineligible_parent_records.jsonl")
    errors = load_jsonl(args.output_dir / "errors.jsonl")
    branch_records = load_branch_records(args.output_dir / "branch_records.jsonl")
    completed_branch_ids = {row["branch_id"] for row in branch_records}
    valid_by_task = defaultdict(int)
    attempted_by_task = defaultdict(int)
    for row in parents:
        valid_by_task[row["task_name"]] += 1
        attempted_by_task[row["task_name"]] += 1
    for row in ineligible_parents:
        attempted_by_task[row["task_name"]] += 1
    completed_parent_ids = {row["parent_id"] for row in parents}
    ineligible_parent_ids = {row["parent_id"] for row in ineligible_parents}
    error_parent_ids = {row["parent_id"] for row in errors}

    snapshot_paths = sorted((args.output_dir / "snapshots").glob("*.pkl.gz"))
    saved_snapshots = [load_full_snapshot(path) for path in snapshot_paths]
    snapshot_index = max(
        (int(snapshot.metadata["snapshot_index"]) + 1 for snapshot in saved_snapshots),
        default=0,
    )
    orphan_groups = defaultdict(list)
    for snapshot in saved_snapshots:
        if snapshot.parent_id not in completed_parent_ids:
            orphan_groups[snapshot.parent_id].append(snapshot)
    for parent_id, snapshots in sorted(orphan_groups.items()):
        if len(snapshots) != 2:
            continue
        if any(
            snapshot.metadata.get("plan_sha256") != plan["plan_sha256"]
            for snapshot in snapshots
        ):
            raise ValueError(
                f"Orphan snapshots for {parent_id} do not match the frozen plan"
            )
        result = _resume_snapshot_parent(
            snapshots,
            args,
            runtime,
            factory,
            policy_args,
            completed_branch_ids,
        )
        parents.append(result)
        valid_by_task[result["task_name"]] += 1
        attempted_by_task[result["task_name"]] += 1
        completed_parent_ids.add(parent_id)
        append_jsonl(args.output_dir / "parent_records.jsonl", result)
        branch_records = load_branch_records(args.output_dir / "branch_records.jsonl")
        completed_branch_ids = {row["branch_id"] for row in branch_records}

    for parent in plan["candidate_parents"]:
        task_name = parent["task_name"]
        if valid_by_task[task_name] >= args.num_parents_per_task:
            continue
        if parent["parent_id"] in completed_parent_ids | ineligible_parent_ids:
            continue
        if parent["parent_id"] in error_parent_ids and not args.retry_errors:
            continue
        attempted_by_task[task_name] += 1
        try:
            result = _run_parent(
                parent,
                args,
                plan,
                runtime,
                factory,
                policy_args,
                snapshot_index,
            )
            snapshot_index = result.pop("snapshot_index")
            if result["valid"]:
                parents.append(result)
                valid_by_task[task_name] += 1
                completed_parent_ids.add(result["parent_id"])
                append_jsonl(args.output_dir / "parent_records.jsonl", result)
            else:
                ineligible_parents.append(result)
                append_jsonl(
                    args.output_dir / "ineligible_parent_records.jsonl", result
                )
        except Exception as error:  # evidence is preserved before fail-fast
            record = {
                **parent,
                "created_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            errors.append(record)
            append_jsonl(args.output_dir / "errors.jsonl", record)
            if args.fail_fast:
                break
        atomic_write_json(
            args.output_dir / "status.json",
            {
                "status": "running",
                "updated_at": utc_now(),
                "parents_completed": len(parents),
                "parents_attempted": sum(attempted_by_task.values()),
                "ineligible_parents": len(ineligible_parents),
                "valid_by_task": dict(valid_by_task),
                "errors": len(errors),
                "snapshots": snapshot_index,
            },
        )

    final_snapshot_paths = sorted((args.output_dir / "snapshots").glob("*.pkl.gz"))
    final_snapshots = [load_full_snapshot(path) for path in final_snapshot_paths]
    branch_records = load_branch_records(args.output_dir / "branch_records.jsonl")
    analysis = analyze_phase2_replay(branch_records, errors)
    analysis.update(
        {
            "expected_parents": plan["expected_parents"],
            "completed_parents": len(parents),
            "attempted_parents": sum(attempted_by_task.values()),
            "ineligible_parents": len(ineligible_parents),
            "valid_parents_by_task": dict(valid_by_task),
            "expected_snapshots": plan["expected_snapshots"],
            "completed_snapshots": len(parents) * 2,
            "saved_snapshot_files": len(final_snapshots),
            "orphan_snapshot_files": sum(
                snapshot.parent_id not in completed_parent_ids
                for snapshot in final_snapshots
            ),
            "expected_primary_branches": plan["expected_primary_branches"],
            "expected_total_branches": plan["expected_total_branches"],
            "parent_support_complete": (
                len(parents) == plan["expected_parents"]
                and all(parent["valid"] for parent in parents)
                and all(
                    valid_by_task[task] == args.num_parents_per_task
                    for task in args.tasks
                )
            ),
        }
    )
    analysis["gates"]["parent_and_snapshot_support_complete"] = bool(
        analysis["parent_support_complete"]
        and analysis["completed_snapshots"] == plan["expected_snapshots"]
    )
    analysis["gates"]["branch_support_complete"] = (
        analysis["primary_records"] == plan["expected_primary_branches"]
        and analysis["records"] == plan["expected_total_branches"]
    )
    grouped = defaultdict(lambda: defaultdict(int))
    for record in branch_records:
        grouped[record["snapshot_id"]][record["kind"]] += 1
    analysis["branch_support_by_snapshot"] = {
        snapshot_id: dict(counts) for snapshot_id, counts in sorted(grouped.items())
    }
    analysis["gates"]["branch_group_composition_exact"] = bool(grouped) and all(
        counts.get("same_seed_repeat", 0) == 2
        and counts.get("candidate", 0) == args.candidate_count
        and counts.get("environment_only", 0) == 1
        and sum(counts.values()) == args.candidate_count + 3
        for counts in grouped.values()
    )
    analysis["all_pass"] = all(analysis["gates"].values())
    atomic_write_json(args.output_dir / "analysis.json", analysis)
    atomic_write_json(
        args.output_dir / "status.json",
        {
            "status": "complete" if analysis["all_pass"] else "failed_gate",
            "updated_at": utc_now(),
            "all_pass": analysis["all_pass"],
            "parents": len(parents),
            "parents_attempted": sum(attempted_by_task.values()),
            "ineligible_parents": len(ineligible_parents),
            "snapshots": analysis["completed_snapshots"],
            "branches": len(branch_records),
            "errors": len(errors),
        },
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return analysis


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--target-stage", action="append", required=True)
    parser.add_argument("--trigger-stage", action="append", required=True)
    parser.add_argument("--num-parents-per-task", type=int, default=5)
    parser.add_argument("--max-parent-attempts-per-task", type=int, default=15)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--env-interface", choices=("gym", "robosuite"), default="gym")
    parser.add_argument(
        "--canonical-camera-observations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--canonical-camera-render-repeats", type=int, default=2)
    parser.add_argument(
        "--fresh-branch-contexts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Construct a fresh environment, renderer, and policy wrapper for "
            "every branch (default: enabled)."
        ),
    )
    parser.add_argument("--parent-horizon", type=int, default=3000)
    parser.add_argument("--snapshot-prefix", type=int, default=2)
    parser.add_argument("--landmark-fraction", type=float, default=0.25)
    parser.add_argument("--suffix-steps", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--candidate-seed-base", type=int, default=7000000)
    parser.add_argument("--ordinary-sampling-seed-base", type=int, default=6000000)
    parser.add_argument("--restore-atol", type=float, default=1e-8)
    parser.add_argument("--restore-rtol", type=float, default=1e-8)
    parser.add_argument(
        "--policy-module",
        default="robocasa.recovery.xiaomi_robotics_1_policy:make_policy",
    )
    parser.add_argument("--policy-arg", action="append", default=[])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--policy-name", default="Xiaomi-Robotics-1-RoboCasa365")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-revision",
        default="0d1aa76d0d82debc9b611e4d1e231096434d5be4",
    )
    parser.add_argument("--server-repository", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument(
        "--fail-fast", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
