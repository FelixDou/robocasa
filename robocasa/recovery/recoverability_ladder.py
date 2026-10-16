"""Recoverability ladder: approximate branching to locate recovery value.

Natural rollouts are run once. At every genuine policy inference the collector
keeps a cheap simulator snapshot (flattened MuJoCo state plus qpos/qvel and the
ordered-subtask count); the model XML and episode metadata are written once per
parent. After the rollout ends, branches restart from the failure-detection
state and apply one rung of an escalating intervention ladder:

``resample_continue``  restore the detection state, fresh policy sampling seed
``robot_reset``        detection world, robot joints restored to an anchor
``subtask_reprompt``   detection state, prompt the current ordered subtask,
                       then return to the task instruction once it completes
``world_reset``        restore the full anchor state (oracle)

Anchors are restore targets measured in policy inferences: ``detection``,
``last_good`` (first inference after the last ordered-subtask completion) and
``last_good_minus_2``. Cells whose interventions are identical for a parent
(for example a world reset to the detection state) are run once and aliased.

Determinism is approximate by design: restore residuals are recorded, never
gated on. This module deliberately does not use the bit-exact full-snapshot
replay stack; it reuses ``recovery_rollout.apply_recovery_mode`` (the
``env_to_last_good``/``eef_to_last_good`` modes of
``evaluate_recovery_benchmark.py``) and the env's ordered subtask predicates.

Subcommands: ``plan`` (dry run + GPU-hour estimate), ``collect``, ``analyze``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from robocasa.recovery.recovery_rollout import (
    _get_obs_after_state_change,
    _get_sim,
    _is_task_success,
    _refresh_sim_visuals,
    _resolve_recovery_target,
    _set_env_instruction,
    _step_env,
    _subtask_is_complete,
    apply_recovery_mode,
    call_policy,
    set_observation_instruction,
)
from robocasa.recovery.subtask_eval import get_subtask_eval


SCHEMA_VERSION = 1
RUNGS = ("resample_continue", "robot_reset", "subtask_reprompt", "world_reset")
ANCHORS = ("detection", "last_good", "last_good_minus_2")
DETECTORS = ("stall", "time", "end")
DEFAULT_CELLS = (
    ("resample_continue", "detection"),
    ("robot_reset", "last_good"),
    ("robot_reset", "last_good_minus_2"),
    ("subtask_reprompt", "detection"),
    ("world_reset", "last_good"),
    ("world_reset", "last_good_minus_2"),
)
# Ladder used for per-parent failure attribution: the first rung whose
# estimated success probability reaches the threshold names the failure class.
LADDER = (
    ("resample_continue", "detection", "stochastic"),
    ("robot_reset", "last_good", "robot_state"),
    ("subtask_reprompt", "detection", "instruction"),
    ("world_reset", "last_good", "world_state"),
)
UNRECOVERED_CLASS = "competence"

PARENTS_NAME = "parents.jsonl"
BRANCHES_NAME = "branches.jsonl"
ERRORS_NAME = "errors.jsonl"
PLAN_NAME = "plan.json"
STATUS_NAME = "status.json"

DEFAULTS = {
    "tasks": None,
    "split": "pretrain",
    "env_interface": "gym",
    "seed_protocol": "repeated_reset",
    "base_seed": 300007,
    "parents_per_task": 24,
    "failed_parents_per_task": 8,
    "success_controls_per_task": 2,
    "k_branches": 4,
    "cells": [list(cell) for cell in DEFAULT_CELLS],
    "detector": "stall",
    "stall_fraction": 0.15,
    "time_fraction": 0.5,
    "budget_fraction": 0.5,
    "max_budget_steps": None,
    "anchor_back_inferences": 2,
    "policy_context": "reset",
    "reprompt_level": "atomic",
    "snapshot_stride": 1,
    "save_model_xml": True,
    "record_policy_features": False,
    "horizons": None,
    "policy_module": None,
    "policy_args": {},
    "max_errors": 10,
    # Planning-only fields.
    "expected_success_rates": {},
    "seconds_per_env_step": 0.06,
    "seconds_per_reset": 20.0,
    "success_step_fraction": 0.6,
    "assumed_branch_success_rate": 0.3,
    "gpus_per_worker": 2,
    "num_workers": 1,
    "state_dim_estimate": 400,
    "feature_bytes_per_inference": 5 * 30 * 1024 * 2,
}


# --------------------------------------------------------------------------
# Small utilities


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unserializable {type(value).__name__}")


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default))
    tmp.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=json_default) + "\n")
        handle.flush()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn final line after a hard kill is expected; skip it.
            continue
    return rows


def stable_seed(*parts):
    digest = hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def parent_identity(task_name, env_seed, config):
    return hashlib.sha256(
        json.dumps(
            {
                "task_name": task_name,
                "environment_seed": int(env_seed),
                "split": config["split"],
                "seed_protocol": config["seed_protocol"],
                "policy_module": config.get("policy_module"),
                "schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]


def current_git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def resolve_config(config=None, overrides=None):
    merged = dict(DEFAULTS)
    merged["policy_args"] = {}
    for source in (config or {}, overrides or {}):
        for key, value in source.items():
            if key.startswith("_"):
                continue
            if key not in DEFAULTS:
                raise ValueError(f"Unknown recoverability-ladder option: {key}")
            if value is not None:
                merged[key] = value
    cells = [tuple(cell) for cell in merged["cells"]]
    for rung, anchor in cells:
        if rung not in RUNGS or anchor not in ANCHORS:
            raise ValueError(f"Invalid ladder cell: {(rung, anchor)}")
    if len(set(cells)) != len(cells):
        raise ValueError("Duplicate ladder cells")
    merged["cells"] = [list(cell) for cell in cells]
    if merged["detector"] not in DETECTORS:
        raise ValueError(f"--detector must be one of {DETECTORS}")
    if merged["policy_context"] not in {"reset", "restore"}:
        raise ValueError("--policy-context must be reset or restore")
    if merged["seed_protocol"] not in {"repeated_reset", "fresh_env"}:
        raise ValueError("--seed-protocol must be repeated_reset or fresh_env")
    for key in ("k_branches", "parents_per_task", "snapshot_stride"):
        if int(merged[key]) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("stall_fraction", "time_fraction", "budget_fraction"):
        if not 0 < float(merged[key]) <= 1.5:
            raise ValueError(f"{key} must be in (0, 1.5]")
    return merged


def task_budget_steps(horizon, config):
    budget = max(1, int(math.ceil(float(config["budget_fraction"]) * int(horizon))))
    if config.get("max_budget_steps"):
        budget = min(budget, int(config["max_budget_steps"]))
    return budget


# --------------------------------------------------------------------------
# Ordered subtask progress and snapshots


class OrderedProgress:
    """Incremental version of the ordered prefix in ``build_subtask_trace``."""

    def __init__(self, required=None, count=0):
        self.required = list(required) if required else []
        self.count = int(count)

    @property
    def completed(self):
        return self.required[: self.count]

    @property
    def current(self):
        return self.required[self.count] if self.count < len(self.required) else None

    def update(self, subtask_eval):
        if not subtask_eval:
            return []
        if not self.required:
            self.required = list(subtask_eval.get("required_predicates", []))
        predicates = subtask_eval.get("predicates", {})
        newly = []
        while self.count < len(self.required):
            name = self.required[self.count]
            if not bool(predicates.get(name, {}).get("value", False)):
                break
            newly.append(name)
            self.count += 1
        if subtask_eval.get("task_success", False):
            while self.count < len(self.required):
                newly.append(self.required[self.count])
                self.count += 1
        return newly


def capture_snapshot(env):
    sim = _get_sim(env)
    if sim is None:
        raise RuntimeError("Recoverability ladder requires an env exposing sim state")
    state = sim.get_state()
    return {
        "states": np.asarray(state.flatten(), dtype=np.float64).copy(),
        "qpos": np.asarray(state.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(state.qvel, dtype=np.float64).copy(),
    }


def state_residual(env, flat_state):
    sim = _get_sim(env)
    current = np.asarray(sim.get_state().flatten(), dtype=np.float64)
    if current.shape != np.shape(flat_state):
        return float("inf")
    return float(np.max(np.abs(current - flat_state))) if current.size else 0.0


def restore_world(env, snapshot):
    """Full world restore through the evaluator's ``env_to_last_good`` mode."""
    return apply_recovery_mode(env, "env_to_last_good", {"states": snapshot["states"]})


def restore_robot(env, snapshot):
    """Robot-only restore through the evaluator's ``eef_to_last_good`` mode."""
    return apply_recovery_mode(
        env,
        "eef_to_last_good",
        {
            "states": snapshot["states"],
            "_sim_state_qpos": snapshot["qpos"],
            "_sim_state_qvel": snapshot["qvel"],
        },
    )


def inference_boundary(policy, step, stride):
    flag = getattr(policy, "at_inference_boundary", None)
    if flag is not None and not callable(flag):
        return bool(flag)
    plan = getattr(policy, "action_plan", None)
    if plan is not None:
        return len(plan) == 0
    return step % max(1, int(stride)) == 0


def seed_policy(policy, seed):
    """Give the policy a fresh sampling stream; return how it was controlled."""
    np.random.seed(int(seed) % (2**32))
    if hasattr(policy, "sampling_seed_base"):
        policy.sampling_seed_base = int(seed)
        return "policy_sampling_seed_base"
    setter = getattr(policy, "set_sampling_seed", None)
    if callable(setter):
        setter(int(seed))
        return "policy_set_sampling_seed"
    return "numpy_global_only"


def reset_policy(policy):
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def clear_env_instruction(env):
    clear = getattr(env, "clear_task_description", None)
    if callable(clear):
        clear()


def env_instruction(env, obs):
    if isinstance(obs, dict):
        for key in ("annotation.human.task_description", "language"):
            value = obs.get(key)
            if isinstance(value, str) and value:
                return value
    get_ep_meta = getattr(env, "get_ep_meta", None)
    if callable(get_ep_meta):
        try:
            return get_ep_meta().get("lang") or None
        except Exception:
            return None
    return None


def model_xml_and_meta(env):
    sim = _get_sim(env)
    xml = None
    if sim is not None and hasattr(sim.model, "get_xml"):
        try:
            xml = sim.model.get_xml()
        except Exception:
            xml = None
    meta = None
    get_ep_meta = getattr(env, "get_ep_meta", None)
    if callable(get_ep_meta):
        try:
            meta = get_ep_meta()
        except Exception:
            meta = None
    return xml, meta


# --------------------------------------------------------------------------
# Parent rollout


def run_parent_rollout(env, policy, *, task_name, horizon, env_seed, policy_seed, config):
    started = time.time()
    reset_kwargs = {"seed": int(env_seed)} if config["seed_protocol"] == "repeated_reset" else {}
    reset_result = env.reset(**reset_kwargs)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    reset_info = (
        reset_result[1]
        if isinstance(reset_result, tuple) and len(reset_result) > 1
        and isinstance(reset_result[1], dict)
        else {}
    )
    reset_policy(policy)
    seed_control = seed_policy(policy, policy_seed)
    instruction = env_instruction(env, obs)
    xml, ep_meta = model_xml_and_meta(env) if config["save_model_xml"] else (None, None)

    tracker = OrderedProgress()
    tracker.update(reset_info.get("subtask_eval") or get_subtask_eval(env))
    snapshots = []
    policy_states = []
    features = []
    feature_steps = []
    events = []
    success = False
    num_steps = 0
    keep_policy_state = (
        config["policy_context"] == "restore"
        and callable(getattr(policy, "get_state", None))
    )
    for step in range(int(horizon)):
        if inference_boundary(policy, step, config["snapshot_stride"]):
            snap = capture_snapshot(env)
            snap.update(step=step, ordered_count=tracker.count, is_inference=True)
            snapshots.append(snap)
            policy_states.append(policy.get_state() if keep_policy_state else None)
        action = call_policy(policy, obs)
        pop = getattr(policy, "pop_inference_record", None)
        record = pop() if callable(pop) else None
        if config["record_policy_features"] and isinstance(record, dict):
            if record.get("features") is not None:
                features.append(np.asarray(record["features"], dtype=np.float16))
                feature_steps.append(step)
        obs, reward, done, info = _step_env(env, action)
        num_steps = step + 1
        current_eval = (info or {}).get("subtask_eval") or get_subtask_eval(env)
        for name in tracker.update(current_eval):
            events.append(
                {"step": num_steps, "subtask": name, "ordered_count": tracker.required.index(name) + 1}
            )
        if _is_task_success(info, reward=reward, env=env):
            success = True
            break
        if done:
            break
    terminal = capture_snapshot(env)
    terminal.update(step=num_steps, ordered_count=tracker.count, is_inference=False)
    snapshots.append(terminal)
    policy_states.append(policy.get_state() if keep_policy_state else None)
    return {
        "success": success,
        "num_steps": num_steps,
        "instruction": instruction,
        "required_subtasks": list(tracker.required),
        "progress_events": events,
        "final_ordered_count": tracker.count,
        "snapshots": snapshots,
        "policy_states": policy_states,
        "features": features,
        "feature_steps": feature_steps,
        "model_xml": xml,
        "ep_meta": ep_meta,
        "seed_control": seed_control,
        "wall_seconds": time.time() - started,
    }


# --------------------------------------------------------------------------
# Detection and anchors


def detector_indices(steps, is_inference, events, horizon, config):
    """Return the first firing snapshot index of every detector.

    ``steps`` are snapshot environment steps; the last snapshot is terminal.
    A detector that never fires falls back to the terminal snapshot.
    """
    terminal = len(steps) - 1
    inference = [i for i in range(len(steps)) if is_inference[i]]
    stall_steps = int(math.ceil(float(config["stall_fraction"]) * horizon))
    time_step = float(config["time_fraction"]) * horizon
    progress_steps = sorted(event["step"] for event in events)

    def last_progress(step):
        previous = [value for value in progress_steps if value <= step]
        return previous[-1] if previous else 0

    result = {"end": {"index": terminal, "fired": True}}
    stall = next(
        (i for i in inference if steps[i] - last_progress(steps[i]) >= stall_steps), None
    )
    result["stall"] = {"index": terminal if stall is None else stall, "fired": stall is not None}
    timed = next((i for i in inference if steps[i] >= time_step), None)
    result["time"] = {"index": terminal if timed is None else timed, "fired": timed is not None}
    for name, payload in result.items():
        payload["step"] = int(steps[payload["index"]])
    return result


def compute_anchors(steps, is_inference, events, detection_index, back):
    """Anchors as snapshot indices, all at or before ``detection_index``."""
    detection_step = steps[detection_index]
    completed = [event["step"] for event in events if event["step"] <= detection_step]
    last_good = 0
    if completed:
        target = max(completed)
        last_good = next(
            (
                i
                for i in range(detection_index + 1)
                if steps[i] >= target and (is_inference[i] or i == detection_index)
            ),
            detection_index,
        )
    return {
        "detection": int(detection_index),
        "last_good": int(last_good),
        "last_good_minus_2": int(max(0, last_good - int(back))),
    }


def control_detection_index(steps, is_inference, horizon, fraction):
    """Matched branch time for successful parents: last inference <= fraction*H."""
    target = float(fraction) * horizon
    candidates = [i for i in range(len(steps)) if is_inference[i] and steps[i] <= target]
    if candidates:
        return candidates[-1]
    inference = [i for i in range(len(steps)) if is_inference[i]]
    return inference[0] if inference else len(steps) - 1


def intervention_key(rung, anchor_index, detection_index):
    if rung == "subtask_reprompt":
        return ("reprompt",)
    if rung == "resample_continue" or anchor_index == detection_index:
        return ("none",)
    return (rung, int(anchor_index))


def schedule_cells(cells, anchors):
    """Map each requested cell to the canonical cell that is actually run."""
    detection = anchors["detection"]
    canonical_by_key = {}
    schedule = []
    for rung, anchor in cells:
        index = anchors["detection"] if rung in ("resample_continue", "subtask_reprompt") else anchors[anchor]
        key = intervention_key(rung, index, detection)
        canonical = canonical_by_key.setdefault(key, (rung, anchor))
        schedule.append(
            {
                "rung": rung,
                "anchor": anchor,
                "anchor_index": int(index),
                "run": canonical == (rung, anchor),
                "canonical": list(canonical),
            }
        )
    return schedule


# --------------------------------------------------------------------------
# Branches


def run_branch(env, policy, *, parent, rollout, cell, k, budget, config, task_name):
    started = time.time()
    rung, anchor = cell["rung"], cell["anchor"]
    snapshots = rollout["snapshots"]
    source_index = parent["anchors"]["detection"]
    anchor_index = cell["anchor_index"]
    source = snapshots[source_index]
    anchor_snap = snapshots[anchor_index]
    branch_seed = stable_seed(parent["parent_id"], rung, anchor, int(k))
    record = {
        "schema_version": SCHEMA_VERSION,
        "parent_id": parent["parent_id"],
        "attempt_uid": parent["attempt_uid"],
        "task_name": task_name,
        "role": parent["role"],
        "rung": rung,
        "anchor": anchor,
        "k": int(k),
        "branch_seed": branch_seed,
        "source_index": int(source_index),
        "anchor_index": int(anchor_index),
        "anchor_distance_inferences": int(source_index - anchor_index),
        "anchor_distance_steps": int(source["step"] - anchor_snap["step"]),
        "budget_steps": int(budget),
    }
    restore_world(env, source)
    record["source_restore_residual"] = state_residual(env, source["states"])
    start_count = int(source["ordered_count"])
    state_index = source_index
    if rung == "world_reset":
        restore_world(env, anchor_snap)
        record["anchor_restore_residual"] = state_residual(env, anchor_snap["states"])
        start_count = int(anchor_snap["ordered_count"])
        state_index = anchor_index
    elif rung == "robot_reset":
        restore_robot(env, anchor_snap)
    _refresh_sim_visuals(env, num_forwards=3)
    obs = _get_obs_after_state_change(env, fallback_obs=None)

    task_instruction = rollout["instruction"] or env_instruction(env, obs)
    tracker = OrderedProgress(rollout["required_subtasks"], count=start_count)
    start_eval = get_subtask_eval(env)
    reprompt_target = reprompt_instruction = None
    reprompt_active = False
    if rung == "subtask_reprompt":
        reprompt_target, reprompt_instruction, kind = _resolve_recovery_target(
            start_eval,
            tracker.current,
            high_level_summary=None,
            recovery_level=config["reprompt_level"],
            atomic_instruction=task_instruction,
            evaluated_task_name=task_name,
        )
        record["reprompt_target_kind"] = kind
        reprompt_active = bool(reprompt_instruction) and reprompt_target not in (None, "task_success")
        record["reprompt_equivalent_to_task_prompt"] = (
            not reprompt_active or reprompt_instruction == task_instruction
        )
    record["target_subtask"] = reprompt_target if reprompt_active else tracker.current
    record["reprompt_instruction"] = reprompt_instruction if reprompt_active else None
    record["start_ordered_count"] = start_count

    policy_state = rollout["policy_states"][state_index]
    if config["policy_context"] == "restore" and policy_state is not None:
        policy.set_state(policy_state)
        record["policy_context"] = "restore"
    else:
        reset_policy(policy)
        record["policy_context"] = "reset"
    record["seed_control"] = seed_policy(policy, branch_seed)

    success = False
    steps_used = 0
    reprompt_steps = None
    inferences = 0
    # Detector features are only stored for parents; skip the transfer here.
    collect_features = getattr(policy, "collect_safe_features", None)
    if collect_features is not None:
        policy.collect_safe_features = False
    if reprompt_active:
        _set_env_instruction(env, reprompt_instruction)
    try:
        for step in range(int(budget)):
            instruction = reprompt_instruction if reprompt_active else task_instruction
            if inference_boundary(policy, step, config["snapshot_stride"]):
                inferences += 1
            action = call_policy(policy, set_observation_instruction(obs, instruction), instruction=instruction)
            obs, reward, done, info = _step_env(env, action)
            steps_used = step + 1
            current_eval = (info or {}).get("subtask_eval") or get_subtask_eval(env)
            tracker.update(current_eval)
            if reprompt_active and (
                reprompt_target in tracker.completed
                or _subtask_is_complete(current_eval, reprompt_target)
            ):
                reprompt_active = False
                reprompt_steps = steps_used
                clear_env_instruction(env)
            if _is_task_success(info, reward=reward, env=env):
                success = True
                break
            if done:
                break
    finally:
        clear_env_instruction(env)
        if collect_features is not None:
            policy.collect_safe_features = collect_features
    record.update(
        {
            "success": bool(success),
            "steps_used": int(steps_used),
            "final_ordered_count": int(tracker.count),
            "stages_advanced": int(tracker.count - int(source["ordered_count"])),
            "target_completed": bool(
                record["target_subtask"] is not None
                and record["target_subtask"] in tracker.completed
            ),
            "reprompt_steps": reprompt_steps,
            "approx_policy_inferences": int(inferences),
            "wall_seconds": time.time() - started,
            "created_at": utc_now(),
        }
    )
    return record


# --------------------------------------------------------------------------
# Collection driver


class Runtime:
    """Live simulator/policy factories. Tests pass a stand-in."""

    def __init__(self, config, host=None, port=None):
        from robocasa.recovery.evaluate_recovery_benchmark import (
            call_factory,
            load_factory,
            make_env,
        )
        from robocasa.recovery.safe.atomic_tasks import registered_safe_task_horizons

        if not config.get("policy_module"):
            raise ValueError("collect requires policy_module")
        self._make_env = make_env
        self._call_factory = call_factory
        self._factory = load_factory(config["policy_module"])
        self._policy_args = dict(config.get("policy_args") or {})
        if host is not None:
            self._policy_args["host"] = host
        if port is not None:
            self._policy_args["port"] = int(port)
        self._config = config
        self._horizons = registered_safe_task_horizons()

    def task_horizon(self, task_name):
        overrides = self._config.get("horizons") or {}
        return int(overrides.get(task_name) or self._horizons[task_name])

    def make_env(self, task_name, seed):
        return self._make_env(
            task_name,
            env_interface=self._config["env_interface"],
            split=self._config["split"],
            seed=int(seed),
            enable_render=True,
        )

    def make_policy(self, env):
        return self._call_factory(self._factory, env, self._policy_args)


def shard_tasks(tasks, horizons, num_shards, shard_index):
    if num_shards <= 1:
        return list(tasks)
    loads = [0] * num_shards
    shards = [[] for _ in range(num_shards)]
    for task in sorted(tasks, key=lambda name: (-horizons.get(name, 0), name)):
        target = loads.index(min(loads))
        shards[target].append(task)
        loads[target] += horizons.get(task, 0)
    return [task for task in tasks if task in set(shards[shard_index])]


def _save_parent_artifacts(output_dir, parent_id, rollout, config):
    snap_dir = output_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snaps = rollout["snapshots"]
    snapshot_path = snap_dir / f"{parent_id}.npz"
    np.savez_compressed(
        snapshot_path,
        steps=np.array([s["step"] for s in snaps], dtype=np.int64),
        states=np.stack([s["states"] for s in snaps]),
        qpos=np.stack([s["qpos"] for s in snaps]),
        qvel=np.stack([s["qvel"] for s in snaps]),
        ordered_count=np.array([s["ordered_count"] for s in snaps], dtype=np.int64),
        is_inference=np.array([s["is_inference"] for s in snaps], dtype=bool),
    )
    paths = {"snapshot_path": str(snapshot_path.relative_to(output_dir))}
    if rollout["model_xml"]:
        model_path = output_dir / "models" / f"{parent_id}.xml.gz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(model_path, "wt") as handle:
            handle.write(rollout["model_xml"])
        paths["model_xml_path"] = str(model_path.relative_to(output_dir))
    if rollout["ep_meta"] is not None:
        meta_path = output_dir / "models" / f"{parent_id}.ep_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(rollout["ep_meta"], default=json_default))
        paths["ep_meta_path"] = str(meta_path.relative_to(output_dir))
    if config["record_policy_features"] and rollout["features"]:
        feature_path = output_dir / "features" / f"{parent_id}.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_path,
            features=np.stack(rollout["features"]),
            feature_steps=np.array(rollout["feature_steps"], dtype=np.int64),
        )
        paths["features_path"] = str(feature_path.relative_to(output_dir))
    return paths


def _write_status(output_dir, plan, parents, errors, state):
    by_task = defaultdict(lambda: defaultdict(int))
    parent_steps = parent_seconds = branch_steps = branch_seconds = 0.0
    for parent in parents:
        counts = by_task[parent["task_name"]]
        counts["parents"] += 1
        counts[parent["role"]] += 1
        counts["successes"] += int(parent["success"])
        parent_steps += parent["num_steps"]
        parent_seconds += parent["wall_seconds"]
        branch_steps += parent.get("branch_steps", 0)
        branch_seconds += parent.get("branch_wall_seconds", 0.0)
    total_steps = parent_steps + branch_steps
    status = {
        "status": state,
        "updated_at": utc_now(),
        "tasks": plan["config"]["tasks"],
        "by_task": {task: dict(counts) for task, counts in by_task.items()},
        "num_parents": len(parents),
        "num_errors": len(errors),
        "measured_seconds_per_env_step": (
            (parent_seconds + branch_seconds) / total_steps if total_steps else None
        ),
        "parent_env_steps": int(parent_steps),
        "branch_env_steps": int(branch_steps),
    }
    atomic_write_json(output_dir / STATUS_NAME, status)
    return status


def run_collection(config, output_dir, *, runtime=None, resume=False, shard_index=0, num_shards=1, host=None, port=None, log=print):
    config = resolve_config(config)
    if not config["tasks"]:
        raise ValueError("No tasks configured")
    output_dir = Path(output_dir)
    runtime = runtime or Runtime(config, host=host, port=port)
    horizons = {task: runtime.task_horizon(task) for task in config["tasks"]}
    tasks = shard_tasks(config["tasks"], horizons, int(num_shards), int(shard_index))
    plan = {
        "schema_version": SCHEMA_VERSION,
        "config": {**config, "tasks": tasks},
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "horizons": {task: horizons[task] for task in tasks},
        "budgets": {task: task_budget_steps(horizons[task], config) for task in tasks},
        "git_commit": current_git_commit(),
        "created_at": utc_now(),
    }
    plan_path = output_dir / PLAN_NAME
    if plan_path.exists():
        if not resume:
            raise FileExistsError(f"{plan_path} exists; pass --resume or choose a new output dir")
        previous = json.loads(plan_path.read_text())
        frozen = ("tasks", "cells", "k_branches", "detector", "stall_fraction", "time_fraction", "budget_fraction", "base_seed", "split", "seed_protocol", "policy_module")
        diffs = {key: (previous["config"].get(key), plan["config"].get(key)) for key in frozen if previous["config"].get(key) != plan["config"].get(key)}
        if diffs:
            raise ValueError(f"Resume config differs from the registered plan: {diffs}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(plan_path, plan)

    parents = read_jsonl(output_dir / PARENTS_NAME)
    errors = read_jsonl(output_dir / ERRORS_NAME)
    done_ids = {parent["parent_id"] for parent in parents}
    _write_status(output_dir, plan, parents, errors, "running")
    cells = [tuple(cell) for cell in config["cells"]]

    for task_name in tasks:
        horizon = horizons[task_name]
        budget = task_budget_steps(horizon, config)
        task_parents = [p for p in parents if p["task_name"] == task_name]
        failed = sum(p["role"] == "failure" for p in task_parents)
        controls = sum(p["role"] == "success_control" for p in task_parents)
        failure_fractions = [
            p["anchor_steps"]["detection"] / horizon for p in task_parents if p["role"] == "failure"
        ]
        env = policy = None
        try:
            for attempt in range(int(config["parents_per_task"])):
                if failed >= config["failed_parents_per_task"] and controls >= config["success_controls_per_task"]:
                    break
                env_seed = int(config["base_seed"]) + attempt
                parent_id = parent_identity(task_name, env_seed, config)
                if parent_id in done_ids and config["seed_protocol"] == "fresh_env":
                    continue
                if env is None or config["seed_protocol"] == "fresh_env":
                    _close(env, policy)
                    env = runtime.make_env(task_name, env_seed)
                    policy = runtime.make_policy(env)
                if parent_id in done_ids:
                    # Mirror the official repeated-reset history on resume.
                    env.reset(seed=env_seed)
                    continue
                if len(errors) >= int(config["max_errors"]):
                    raise RuntimeError(f"Stopping after {len(errors)} errors; inspect {ERRORS_NAME}")
                try:
                    parent = _collect_parent(
                        env, policy, task_name=task_name, env_seed=env_seed, parent_id=parent_id,
                        horizon=horizon, budget=budget, cells=cells, config=config, output_dir=output_dir,
                        need_failure=failed < config["failed_parents_per_task"],
                        need_control=controls < config["success_controls_per_task"],
                        failure_fractions=failure_fractions, log=log,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    event = {"task_name": task_name, "environment_seed": env_seed, "parent_id": parent_id, "error": traceback.format_exc(), "created_at": utc_now()}
                    append_jsonl(output_dir / ERRORS_NAME, event)
                    errors.append(event)
                    log(f"[ladder] error on {task_name} seed {env_seed}; see {ERRORS_NAME}")
                    _close(env, policy)
                    env = policy = None
                    continue
                append_jsonl(output_dir / PARENTS_NAME, parent)
                parents.append(parent)
                done_ids.add(parent_id)
                failed += parent["role"] == "failure"
                controls += parent["role"] == "success_control"
                if parent["role"] == "failure":
                    failure_fractions.append(parent["anchor_steps"]["detection"] / horizon)
                _write_status(output_dir, plan, parents, errors, "running")
                log(
                    f"[ladder] {task_name} seed={env_seed} success={parent['success']} role={parent['role']} "
                    f"failures={failed}/{config['failed_parents_per_task']} controls={controls}/{config['success_controls_per_task']}"
                )
        finally:
            _close(env, policy)
    return _write_status(output_dir, plan, parents, errors, "complete")


def _close(env, policy):
    for obj in (policy, env):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _collect_parent(env, policy, *, task_name, env_seed, parent_id, horizon, budget, cells, config, output_dir, need_failure, need_control, failure_fractions, log):
    rollout = run_parent_rollout(
        env, policy, task_name=task_name, horizon=horizon, env_seed=env_seed,
        policy_seed=stable_seed(parent_id, "parent"), config=config,
    )
    snaps = rollout["snapshots"]
    steps = [s["step"] for s in snaps]
    is_inference = [s["is_inference"] for s in snaps]
    detectors = detector_indices(steps, is_inference, rollout["progress_events"], horizon, config)
    if rollout["success"]:
        role = "success_control" if need_control else "unbranched"
        fraction = float(np.median(failure_fractions)) if failure_fractions else float(config["time_fraction"])
        detection_index = control_detection_index(steps, is_inference, horizon, fraction)
        detection_source = "matched_failure_fraction" if failure_fractions else "time_fraction"
        detector_fired_before_success = bool(detectors[config["detector"]]["fired"] and config["detector"] != "end")
    else:
        role = "failure" if need_failure else "unbranched"
        detection_index = detectors[config["detector"]]["index"]
        detection_source = config["detector"]
        detector_fired_before_success = None
    anchors = compute_anchors(steps, is_inference, rollout["progress_events"], detection_index, config["anchor_back_inferences"])
    detection_step = steps[detection_index]
    last_progress = max([e["step"] for e in rollout["progress_events"] if e["step"] <= detection_step] or [0])
    parent = {
        "schema_version": SCHEMA_VERSION,
        "parent_id": parent_id,
        "attempt_uid": uuid.uuid4().hex,
        "task_name": task_name,
        "environment_seed": int(env_seed),
        "role": role,
        "success": bool(rollout["success"]),
        "num_steps": int(rollout["num_steps"]),
        "horizon": int(horizon),
        "budget_steps": int(budget),
        "instruction": rollout["instruction"],
        "required_subtasks": rollout["required_subtasks"],
        "progress_events": rollout["progress_events"],
        "final_ordered_count": int(rollout["final_ordered_count"]),
        "num_snapshots": len(snaps),
        "num_inferences": int(sum(is_inference)),
        "detectors": detectors,
        "detector": config["detector"],
        "detection_source": detection_source,
        "detector_fired_on_success": detector_fired_before_success,
        "anchors": anchors,
        "anchor_steps": {name: int(steps[index]) for name, index in anchors.items()},
        "anchor_ordered_counts": {name: int(snaps[index]["ordered_count"]) for name, index in anchors.items()},
        "scores_at_detection": {
            "time": detection_step / horizon,
            "stall": (detection_step - last_progress) / horizon,
            "ordered_progress": (
                snaps[detection_index]["ordered_count"] / len(rollout["required_subtasks"])
                if rollout["required_subtasks"] else float(rollout["success"])
            ),
        },
        "seed_control": rollout["seed_control"],
        "wall_seconds": rollout["wall_seconds"],
        **_save_parent_artifacts(output_dir, parent_id, rollout, config),
    }
    schedule = schedule_cells(cells, anchors) if role != "unbranched" else []
    parent["cells"] = schedule
    branch_steps = 0
    branch_seconds = 0.0
    for cell in schedule:
        if not cell["run"]:
            continue
        for k in range(int(config["k_branches"])):
            record = run_branch(env, policy, parent=parent, rollout=rollout, cell=cell, k=k, budget=budget, config=config, task_name=task_name)
            append_jsonl(output_dir / BRANCHES_NAME, record)
            branch_steps += record["steps_used"]
            branch_seconds += record["wall_seconds"]
    parent["branch_steps"] = int(branch_steps)
    parent["branch_wall_seconds"] = branch_seconds
    parent["created_at"] = utc_now()
    return parent


# --------------------------------------------------------------------------
# Planning / dry run


def build_plan(config, *, calibration=None, horizons=None):
    """Simulator-free cost estimate. ``calibration`` is a status.json dict."""
    config = resolve_config(config)
    if horizons is None:
        from robocasa.recovery.safe.atomic_tasks import registered_safe_task_horizons

        registered = registered_safe_task_horizons()
        horizons = {task: int((config.get("horizons") or {}).get(task) or registered[task]) for task in config["tasks"]}
    seconds_per_step = float(config["seconds_per_env_step"])
    throughput_source = "assumed"
    if calibration and calibration.get("measured_seconds_per_env_step"):
        seconds_per_step = float(calibration["measured_seconds_per_env_step"])
        throughput_source = "measured"
    k = int(config["k_branches"])
    n_cells = len(config["cells"])
    anchor_independent = sum(rung in ("resample_continue", "subtask_reprompt") for rung, _ in config["cells"])
    rows = []
    warnings = []
    for task in config["tasks"]:
        horizon = int(horizons[task])
        budget = task_budget_steps(horizon, config)
        rate = config["expected_success_rates"].get(task)
        if rate is None:
            warnings.append(f"{task}: no expected success rate; assuming 0.5")
            rate = 0.5
        rate = float(rate)
        if not 0.2 <= rate <= 0.8:
            warnings.append(f"{task}: expected success rate {rate:.2f} is outside 20-80%")
        fail_target = int(config["failed_parents_per_task"])
        control_target = int(config["success_controls_per_task"])
        needed = max(
            math.ceil(fail_target / max(1e-6, 1 - rate)),
            math.ceil(control_target / max(1e-6, rate)),
        )
        parents = min(int(config["parents_per_task"]), needed)
        failures = min(fail_target, parents * (1 - rate))
        successes = parents * rate
        controls = min(control_target, successes)
        parent_steps = parents * (1 - rate) * horizon + successes * float(config["success_step_fraction"]) * horizon
        branched = failures + controls
        branch_steps_upper = branched * n_cells * k * budget
        r = float(config["assumed_branch_success_rate"])
        branch_steps_expected = branch_steps_upper * (1 - 0.5 * r)
        n_inference = horizon / 16.0
        storage = parents * (
            n_inference * float(config["state_dim_estimate"]) * 8 * 3
            + 2e6
            + (n_inference * float(config["feature_bytes_per_inference"]) if config["record_policy_features"] else 0)
        )
        rows.append(
            {
                "task": task,
                "horizon": horizon,
                "budget_steps": budget,
                "expected_success_rate": rate,
                "expected_parents": parents,
                "expected_failed_parents": round(failures, 2),
                "expected_success_controls": round(controls, 2),
                "branches_upper_bound": int(round(branched * n_cells * k)),
                "parent_env_steps": int(parent_steps),
                "branch_env_steps_upper": int(branch_steps_upper),
                "branch_env_steps_expected": int(branch_steps_expected),
                "storage_bytes_estimate": int(storage),
            }
        )
    reset_seconds = sum(row["expected_parents"] for row in rows) * float(config["seconds_per_reset"])
    totals = {}
    for kind in ("upper", "expected"):
        steps = sum(row["parent_env_steps"] + row[f"branch_env_steps_{kind}"] for row in rows)
        worker_hours = (steps * seconds_per_step + reset_seconds) / 3600.0
        totals[kind] = {
            "env_steps": int(steps),
            "worker_hours": round(worker_hours, 2),
            "gpu_hours": round(worker_hours * int(config["gpus_per_worker"]), 2),
            "wall_hours_with_workers": round(worker_hours / max(1, int(config["num_workers"])), 2),
        }
    return {
        "dry_run": True,
        "tasks": config["tasks"],
        "cells": config["cells"],
        "k_branches": k,
        "anchor_independent_cells": anchor_independent,
        "seconds_per_env_step": seconds_per_step,
        "throughput_source": throughput_source,
        "gpus_per_worker": int(config["gpus_per_worker"]),
        "num_workers": int(config["num_workers"]),
        "per_task": rows,
        "totals": totals,
        "storage_gib_estimate": round(sum(row["storage_bytes_estimate"] for row in rows) / 2**30, 2),
        "warnings": warnings,
        "notes": [
            "upper: every scheduled branch runs its full budget and no cell is aliased",
            "expected: branches succeed at assumed_branch_success_rate, stopping halfway on average",
            "calibrate seconds_per_env_step from a smoke run with --calibration-dir",
        ],
    }


# --------------------------------------------------------------------------
# Analysis


def load_run(run_dirs):
    parents, branches = [], []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        completed = {}
        for parent in read_jsonl(run_dir / PARENTS_NAME):
            completed[parent["parent_id"]] = parent
        for branch in read_jsonl(run_dir / BRANCHES_NAME):
            parent = completed.get(branch["parent_id"])
            if parent is not None and parent["attempt_uid"] == branch["attempt_uid"]:
                branches.append(branch)
        parents.extend(completed.values())
    return parents, branches


def parent_cell_estimates(parents, branches):
    """P(success) etc. per (parent, rung, anchor), resolving aliased cells."""
    grouped = defaultdict(list)
    for branch in branches:
        grouped[(branch["parent_id"], branch["rung"], branch["anchor"])].append(branch)
    rows = []
    for parent in parents:
        for cell in parent.get("cells", []):
            key = (parent["parent_id"], *cell["canonical"])
            outcomes = grouped.get(key, [])
            if not outcomes:
                continue
            rows.append(
                {
                    "parent_id": parent["parent_id"],
                    "task_name": parent["task_name"],
                    "role": parent["role"],
                    "rung": cell["rung"],
                    "anchor": cell["anchor"],
                    "aliased_from": None if cell["run"] else "/".join(cell["canonical"]),
                    "anchor_distance_inferences": parent["anchors"]["detection"] - cell["anchor_index"],
                    "n": len(outcomes),
                    "p_success": float(np.mean([b["success"] for b in outcomes])),
                    "p_stage_advance": float(np.mean([b["stages_advanced"] > 0 for b in outcomes])),
                    "mean_steps_success": (
                        float(np.mean([b["steps_used"] for b in outcomes if b["success"]]))
                        if any(b["success"] for b in outcomes) else None
                    ),
                    "max_restore_residual": float(
                        max(max(b.get("source_restore_residual", 0.0), b.get("anchor_restore_residual", 0.0)) for b in outcomes)
                    ),
                }
            )
    return rows


def cluster_bootstrap(values_by_parent, *, strata=None, n_boot=2000, seed=0, statistic=np.mean):
    """Percentile CI resampling parents (within strata when given)."""
    ids = list(values_by_parent)
    if not ids:
        return None, None, None
    point = float(statistic(np.array([values_by_parent[i] for i in ids], dtype=float)))
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for parent_id in ids:
        groups[(strata or {}).get(parent_id)].append(parent_id)
    stats = []
    for _ in range(int(n_boot)):
        sample = []
        for members in groups.values():
            picks = rng.integers(0, len(members), size=len(members))
            sample.extend(values_by_parent[members[i]] for i in picks)
        stats.append(statistic(np.array(sample, dtype=float)))
    low, high = np.percentile(stats, [2.5, 97.5])
    return point, float(low), float(high)


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _distance_bucket(distance):
    if distance <= 0:
        return "0"
    if distance <= 2:
        return "1-2"
    if distance <= 5:
        return "3-5"
    if distance <= 10:
        return "6-10"
    return ">10"


def analyze(run_dirs, *, scores=None, threshold=0.5, n_boot=2000, seed=0):
    parents, branches = load_run(run_dirs)
    rows = parent_cell_estimates(parents, branches)
    strata = {p["parent_id"]: p["task_name"] for p in parents}
    by_parent = {p["parent_id"]: p for p in parents}

    ladder = []
    for role in ("failure", "success_control"):
        role_rows = [r for r in rows if r["role"] == role]
        cells = sorted({(r["rung"], r["anchor"]) for r in role_rows}, key=lambda c: (RUNGS.index(c[0]), ANCHORS.index(c[1])))
        tasks = sorted({r["task_name"] for r in role_rows})
        for task in ["ALL"] + tasks:
            for rung, anchor in cells:
                selected = [r for r in role_rows if (r["rung"], r["anchor"]) == (rung, anchor) and task in ("ALL", r["task_name"])]
                if not selected:
                    continue
                for metric in ("p_success", "p_stage_advance"):
                    values = {r["parent_id"]: r[metric] for r in selected}
                    point, low, high = cluster_bootstrap(values, strata=strata if task == "ALL" else None, n_boot=n_boot, seed=seed)
                    ladder.append({"role": role, "task": task, "rung": rung, "anchor": anchor, "metric": metric, "parents": len(values), "estimate": point, "ci_low": low, "ci_high": high, "aliased_parents": sum(r["aliased_from"] is not None for r in selected)})

    # Paired contrasts against resample_continue at detection (failures).
    contrasts = []
    failure_rows = [r for r in rows if r["role"] == "failure"]
    estimates = {(r["parent_id"], r["rung"], r["anchor"]): r["p_success"] for r in failure_rows}
    base_key = ("resample_continue", "detection")
    for rung, anchor in sorted({(r["rung"], r["anchor"]) for r in failure_rows}):
        if (rung, anchor) == base_key:
            continue
        diffs = {
            pid: estimates[(pid, rung, anchor)] - estimates[(pid, *base_key)]
            for (pid, ru, an) in estimates
            if (ru, an) == (rung, anchor) and (pid, *base_key) in estimates
        }
        point, low, high = cluster_bootstrap(diffs, strata=strata, n_boot=n_boot, seed=seed)
        if point is not None:
            contrasts.append({"rung": rung, "anchor": anchor, "minus": "/".join(base_key), "parents": len(diffs), "estimate": point, "ci_low": low, "ci_high": high})

    # Per-parent attribution along the ladder.
    attribution = {}
    for pid in sorted({r["parent_id"] for r in failure_rows}):
        label = UNRECOVERED_CLASS
        for rung, anchor, name in LADDER:
            value = estimates.get((pid, rung, anchor))
            if value is not None and value >= threshold:
                label = name
                break
        attribution[pid] = label
    classes = [name for _, _, name in LADDER] + [UNRECOVERED_CLASS]
    attribution_rows = []
    for name in classes:
        indicator = {pid: float(label == name) for pid, label in attribution.items()}
        point, low, high = cluster_bootstrap(indicator, strata=strata, n_boot=n_boot, seed=seed)
        attribution_rows.append({"class": name, "parents": int(sum(indicator.values())), "fraction": point, "ci_low": low, "ci_high": high})

    # Recoverability decay with anchor distance (world resets; distance 0 = resample).
    decay = []
    decay_rows = [r for r in failure_rows if r["rung"] in ("world_reset", "resample_continue")]
    for bucket in ("0", "1-2", "3-5", "6-10", ">10"):
        values = {}
        for r in decay_rows:
            if _distance_bucket(r["anchor_distance_inferences"]) == bucket:
                values.setdefault(r["parent_id"], []).append(r["p_success"])
        values = {pid: float(np.mean(v)) for pid, v in values.items()}
        point, low, high = cluster_bootstrap(values, strata=strata, n_boot=n_boot, seed=seed)
        if point is not None:
            decay.append({"distance_inferences": bucket, "parents": len(values), "p_success": point, "ci_low": low, "ci_high": high})

    # Detector score at detection vs branch-estimated recoverability.
    external = defaultdict(dict)
    for row in scores or []:
        key = (row.get("name", "safe"), int(row["snapshot_index"]))
        external[row["parent_id"]][key] = float(row["score"])
    correlations = []
    best_by_parent = defaultdict(list)
    for (pid, _, _), value in estimates.items():
        best_by_parent[pid].append(value)
    targets = {
        "p_resample": lambda pid: estimates.get((pid, *base_key)),
        "p_best_cell": lambda pid: max(best_by_parent[pid]) if best_by_parent[pid] else None,
    }
    score_names = ["time", "stall", "ordered_progress"] + sorted({n for d in external.values() for n, _ in d})
    for score_name in score_names:
        for target_name, target_fn in targets.items():
            pairs = {}
            for pid in attribution:
                parent = by_parent[pid]
                if score_name in parent["scores_at_detection"]:
                    score = parent["scores_at_detection"][score_name]
                else:
                    score = external.get(pid, {}).get((score_name, parent["anchors"]["detection"]))
                    if score is None:
                        continue
                target = target_fn(pid)
                if target is not None:
                    pairs[pid] = (score, target)
            rho = spearman([v[0] for v in pairs.values()], [v[1] for v in pairs.values()])
            low = high = None
            if rho is not None:
                rng = np.random.default_rng(seed)
                ids = list(pairs)
                boots = []
                for _ in range(int(n_boot)):
                    pick = [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
                    value = spearman([pairs[i][0] for i in pick], [pairs[i][1] for i in pick])
                    if value is not None:
                        boots.append(value)
                if boots:
                    low, high = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
            correlations.append({"score": score_name, "target": target_name, "parents": len(pairs), "spearman": rho, "ci_low": low, "ci_high": high})

    residuals = [r["max_restore_residual"] for r in rows]
    summary = {
        "num_parents": len(parents),
        "num_branches": len(branches),
        "parents_by_role": {role: sum(p["role"] == role for p in parents) for role in ("failure", "success_control", "unbranched")},
        "base_success_rate_by_task": {
            task: float(np.mean([p["success"] for p in parents if p["task_name"] == task]))
            for task in sorted({p["task_name"] for p in parents})
        },
        "attribution_threshold": threshold,
        "restore_residual_max": float(max(residuals)) if residuals else None,
        "restore_residual_median": float(np.median(residuals)) if residuals else None,
        "seed_control": sorted({b.get("seed_control") for b in branches}),
        "ladder": ladder,
        "contrasts_vs_resample": contrasts,
        "attribution": attribution_rows,
        "decay": decay,
        "score_correlations": correlations,
    }
    return summary, rows


def _write_csv(path, rows):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


def write_analysis(summary, rows, output_dir):
    output_dir = Path(output_dir)
    atomic_write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "parent_cells.csv", rows)
    _write_csv(output_dir / "ladder.csv", summary["ladder"])
    _write_csv(output_dir / "decay.csv", summary["decay"])
    _write_csv(output_dir / "score_correlations.csv", summary["score_correlations"])
    lines = [
        "# Recoverability ladder",
        "",
        f"Parents: {summary['num_parents']} {summary['parents_by_role']}; branches: {summary['num_branches']}.",
        f"Restore residual max/median: {_fmt(summary['restore_residual_max'])} / {_fmt(summary['restore_residual_median'])} (recorded, not gated).",
        f"Seed control: {', '.join(str(s) for s in summary['seed_control'])}.",
        "",
        "## P(success) by rung (failed parents, all tasks; parent-cluster bootstrap 95% CI)",
        "",
        "| rung | anchor | parents | P(success) | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary["ladder"]:
        if row["role"] == "failure" and row["task"] == "ALL" and row["metric"] == "p_success":
            lines.append(f"| {row['rung']} | {row['anchor']} | {row['parents']} | {_fmt(row['estimate'])} | {_fmt(row['ci_low'])}-{_fmt(row['ci_high'])} |")
    lines += ["", "## Harm control (successful parents)", "", "| rung | anchor | parents | P(success) | 95% CI |", "|---|---|---:|---:|---|"]
    for row in summary["ladder"]:
        if row["role"] == "success_control" and row["task"] == "ALL" and row["metric"] == "p_success":
            lines.append(f"| {row['rung']} | {row['anchor']} | {row['parents']} | {_fmt(row['estimate'])} | {_fmt(row['ci_low'])}-{_fmt(row['ci_high'])} |")
    lines += ["", f"## Failure attribution (first rung with P >= {summary['attribution_threshold']})", "", "| class | parents | fraction | 95% CI |", "|---|---:|---:|---|"]
    for row in summary["attribution"]:
        lines.append(f"| {row['class']} | {row['parents']} | {_fmt(row['fraction'])} | {_fmt(row['ci_low'])}-{_fmt(row['ci_high'])} |")
    lines += ["", "## Decay with anchor distance (world reset; 0 = resample)", "", "| inferences back | parents | P(success) | 95% CI |", "|---|---:|---:|---|"]
    for row in summary["decay"]:
        lines.append(f"| {row['distance_inferences']} | {row['parents']} | {_fmt(row['p_success'])} | {_fmt(row['ci_low'])}-{_fmt(row['ci_high'])} |")
    lines += ["", "## Score at detection vs recoverability (Spearman)", "", "| score | target | parents | rho | 95% CI |", "|---|---|---:|---:|---|"]
    for row in summary["score_correlations"]:
        lines.append(f"| {row['score']} | {row['target']} | {row['parents']} | {_fmt(row['spearman'])} | {_fmt(row['ci_low'])}-{_fmt(row['ci_high'])} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return output_dir / "summary.md"


# --------------------------------------------------------------------------
# CLI


def _parse_policy_args(values):
    parsed = {}
    for value in values or []:
        key, sep, raw = value.partition("=")
        if not sep:
            raise ValueError(f"--policy-arg must be key=value, got {value!r}")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def _load_config(args):
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    overrides = {}
    for key in ("tasks", "k_branches", "parents_per_task", "failed_parents_per_task", "success_controls_per_task", "base_seed", "budget_fraction", "max_budget_steps", "detector", "policy_context", "num_workers"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "policy_arg", None):
        overrides["policy_args"] = {**config.get("policy_args", {}), **_parse_policy_args(args.policy_arg)}
    return resolve_config(config, overrides)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", type=Path)
        p.add_argument("--tasks", nargs="+")
        p.add_argument("--k-branches", type=int)
        p.add_argument("--parents-per-task", type=int)
        p.add_argument("--failed-parents-per-task", type=int)
        p.add_argument("--success-controls-per-task", type=int)
        p.add_argument("--base-seed", type=int)
        p.add_argument("--budget-fraction", type=float)
        p.add_argument("--max-budget-steps", type=int)
        p.add_argument("--detector", choices=DETECTORS)
        p.add_argument("--policy-context", choices=("reset", "restore"))
        p.add_argument("--num-workers", type=int)

    plan = sub.add_parser("plan", help="dry run: validate config and estimate GPU hours")
    common(plan)
    plan.add_argument("--calibration-dir", type=Path, help="run dir whose status.json gives measured s/step")
    plan.add_argument("--output", type=Path)

    collect = sub.add_parser("collect", help="run parents and ladder branches")
    common(collect)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--policy-arg", action="append", default=[])
    collect.add_argument("--host")
    collect.add_argument("--port", type=int)
    collect.add_argument("--shard-index", type=int, default=0)
    collect.add_argument("--num-shards", type=int, default=1)
    collect.add_argument("--resume", action="store_true")

    analysis = sub.add_parser("analyze", help="ladder tables, decay and score correlation")
    analysis.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--scores-jsonl", type=Path, help="rows: parent_id, snapshot_index, score[, name]")
    analysis.add_argument("--threshold", type=float, default=0.5)
    analysis.add_argument("--n-boot", type=int, default=2000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        config = _load_config(args)
        calibration = None
        if args.calibration_dir:
            calibration = json.loads((args.calibration_dir / STATUS_NAME).read_text())
        plan = build_plan(config, calibration=calibration)
        text = json.dumps(plan, indent=2, sort_keys=True)
        print(text)
        if args.output:
            atomic_write_json(args.output, plan)
        return plan
    if args.command == "collect":
        config = _load_config(args)
        return run_collection(
            config, args.output_dir, resume=args.resume, shard_index=args.shard_index,
            num_shards=args.num_shards, host=args.host, port=args.port,
            log=lambda message: print(message, flush=True),
        )
    scores = read_jsonl(args.scores_jsonl) if args.scores_jsonl else None
    summary, rows = analyze(args.run_dirs, scores=scores, threshold=args.threshold, n_boot=args.n_boot)
    path = write_analysis(summary, rows, args.output_dir)
    print(path.read_text())
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
