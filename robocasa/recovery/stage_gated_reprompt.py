"""Prospective, end-to-end trial of stage-gated re-prompting (no branching).

The recoverability ladder (``recoverability_ladder.py``) found that switching
the policy's instruction to the current stage's atomic instruction changes
success with a stage-dependent sign: it helps at ArrangeTea's mug stage and
BreadSelection's jam stage and hurts at GarnishPancake's strawberry stage.
This module tests the deployable consequence on fresh seeds: natural
rollouts, one per arm, from the same environment seed and the same policy
sampling-seed schedule.

``baseline``         task instruction throughout
``gated_reprompt``   when the ordered stage tracker enters a whitelisted stage,
                     switch to that stage's atomic instruction
                     (``_resolve_recovery_target``) until the stage predicate
                     holds or ``reprompt_step_cap`` steps pass, then return to
                     the task instruction
``always_reprompt``  the same at every stage (shows why gating matters)

The stage is the oracle stage from simulator predicates (``OrderedProgress``
over the env's required predicates, with the ladder's ``exclude_predicates``
and ``required_subtask_overrides``). Each stage fires at most once per
rollout. A task without whitelisted stages is a control: its gated arm can
never switch, so it must match its baseline.

Every instruction change resets the policy context, as XR-1's client does on
its own (``xiaomi_robotics_1_policy.py`` ``__call__``) and as the ladder's
re-prompt branches did (``policy_context=reset``), so a switch takes effect at
the step it fires. Because that reset also restarts XR-1's inference index,
the sampling seed is rebased to ``policy_seed + global inference index``: the
n-th inference of every arm uses the same sampling seed.

Primary endpoint: full-task success, paired by seed, ``gated_reprompt`` minus
``baseline``, macro over the whitelisted tasks, with the exact McNemar test
and a paired (seed-stratified) bootstrap CI. Per-task contrasts, harm on
would-be successes, stage completion, time to success and switch statistics
are secondary.

Subcommands: ``plan`` (dry run, GPU-hour estimate and McNemar power),
``collect``, ``analyze``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

import numpy as np

from robocasa.recovery.recoverability_ladder import (
    OrderedProgress,
    PredicateRecorder,
    Runtime,
    _check_required,
    _close,
    _parse_policy_args,
    _record_latches,
    append_jsonl,
    atomic_write_json,
    capture_snapshot,
    clear_env_instruction,
    current_git_commit,
    effective_required,
    env_instruction,
    inference_boundary,
    read_jsonl,
    reset_policy,
    seed_policy,
    set_env_instruction,
    stable_seed,
    task_value,
    utc_now,
)
from robocasa.recovery.recovery_rollout import (
    _is_task_success,
    _resolve_recovery_target,
    _step_env,
    _subtask_is_complete,
    call_policy,
    set_observation_instruction,
)
from robocasa.recovery.subtask_eval import get_subtask_eval


SCHEMA_VERSION = 1
KIND = "stage_gated_reprompt"
ARMS = ("baseline", "gated_reprompt", "always_reprompt")
PRIMARY_CONTRAST = ("gated_reprompt", "baseline")
CONTRASTS = (
    ("gated_reprompt", "baseline"),
    ("always_reprompt", "baseline"),
    ("gated_reprompt", "always_reprompt"),
)

ROLLOUTS_NAME = "rollouts.jsonl"
PAIRS_NAME = "pairs.jsonl"
ERRORS_NAME = "errors.jsonl"
PLAN_NAME = "plan.json"
STATUS_NAME = "status.json"

DEFAULTS = {
    "tasks": None,
    "arms": list(ARMS),
    # {task: [ordered stage names]}; tasks absent or empty are controls.
    "reprompt_whitelist": {},
    "seeds_per_task": 50,
    "base_seed": 400007,
    "split": "pretrain",
    "env_interface": "gym",
    "seed_protocol": "repeated_reset",
    # "counterbalanced": arm order per seed is a stable permutation; "fixed": config order.
    "arm_order": "counterbalanced",
    # Seeds are processed in blocks (all tasks per block) so a stopped run stays balanced.
    "seed_block_size": 10,
    # Steps one switch may last before returning to the task instruction (scalar or {task: steps}).
    "reprompt_step_cap": 1280,
    "reprompt_level": "atomic",
    "required_subtask_overrides": {},
    "exclude_predicates": {},
    "regression_ignore_stages": ["precondition", "release"],
    "regression_ignore_predicates": {},
    # Simulator-state hash every this many steps (arm divergence check).
    "state_hash_stride": 16,
    "snapshot_stride": 1,
    "horizons": None,
    "policy_module": None,
    "policy_args": {},
    "max_errors": 10,
    "alpha": 0.05,
    # Planning-only fields.
    "expected_success_rates": {},
    # Scalar or {task: fraction of H at which successes end}.
    "success_step_fraction": 0.6,
    "seconds_per_env_step": 0.055,
    "seconds_per_reset": 20.0,
    "seconds_per_env_creation": 60.0,
    "gpus_per_worker": 2,
    "num_workers": 2,
    "allocation_hours": 24.0,
    "allocation_overhead_hours": 0.5,
    "power_target_difference": 0.15,
}

# Plan entries a resumed collection must not change.
FROZEN_PLAN_KEYS = (
    "tasks", "arms", "reprompt_whitelist", "seeds_per_task", "base_seed", "split",
    "env_interface", "seed_protocol", "arm_order", "reprompt_step_cap", "reprompt_level",
    "required_subtask_overrides", "exclude_predicates", "state_hash_stride", "policy_module",
)


# --------------------------------------------------------------------------
# Configuration


def _validate_name_map(value, key):
    value = value or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must map task names to predicate-name lists")
    for names in value.values():
        if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"{key} must map task names to predicate-name lists")
    return {task: list(names) for task, names in value.items()}


def resolve_config(config=None, overrides=None):
    merged = dict(DEFAULTS)
    merged["policy_args"] = {}
    for source in (config or {}, overrides or {}):
        for key, value in source.items():
            if key.startswith("_"):
                continue
            if key not in DEFAULTS:
                raise ValueError(f"Unknown stage-gated re-prompt option: {key}")
            if value is not None:
                merged[key] = value
    arms = list(merged["arms"])
    if not arms or arms[0] != "baseline" or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError(f"arms must start with baseline and be distinct members of {ARMS}")
    merged["arms"] = arms
    for key in ("reprompt_whitelist", "required_subtask_overrides", "exclude_predicates", "regression_ignore_predicates"):
        merged[key] = _validate_name_map(merged[key], key)
    unknown = sorted(set(merged["reprompt_whitelist"]) - set(merged["tasks"] or []))
    if unknown:
        raise ValueError(f"reprompt_whitelist names tasks that are not configured: {unknown}")
    if merged["seed_protocol"] not in {"repeated_reset", "fresh_env"}:
        raise ValueError("seed_protocol must be repeated_reset or fresh_env")
    if merged["arm_order"] not in {"counterbalanced", "fixed"}:
        raise ValueError("arm_order must be counterbalanced or fixed")
    for key in ("seeds_per_task", "seed_block_size", "state_hash_stride", "snapshot_stride", "num_workers"):
        if int(merged[key]) < 1:
            raise ValueError(f"{key} must be positive")
    cap = merged["reprompt_step_cap"]
    caps = cap.values() if isinstance(cap, dict) else [cap]
    if any(value is not None and int(value) < 1 for value in caps):
        raise ValueError("reprompt_step_cap must be positive (or null for no cap)")
    if not 0 < float(merged["alpha"]) < 1:
        raise ValueError("alpha must be in (0, 1)")
    merged["regression_ignore_stages"] = list(merged["regression_ignore_stages"] or [])
    return merged


def whitelisted_tasks(config):
    return [task for task in config["tasks"] if config["reprompt_whitelist"].get(task)]


def control_tasks(config):
    return [task for task in config["tasks"] if not config["reprompt_whitelist"].get(task)]


def task_seeds(config):
    base = int(config["base_seed"])
    return [base + index for index in range(int(config["seeds_per_task"]))]


def shard_seeds(seeds, num_shards, shard_index):
    """Seed indices round-robin over shards; every arm of a seed stays on one worker."""
    return [seed for index, seed in enumerate(seeds) if index % int(num_shards) == int(shard_index)]


def seed_blocks(seeds, size):
    size = max(1, int(size))
    return [seeds[start:start + size] for start in range(0, len(seeds), size)]


def arm_order(task_name, env_seed, arms, mode="counterbalanced"):
    if mode == "fixed":
        return list(arms)
    return sorted(arms, key=lambda arm: (stable_seed(KIND, "arm_order", task_name, int(env_seed), arm), arm))


def rollout_policy_seed(task_name, env_seed):
    """Policy sampling-seed base shared by every arm of one (task, seed)."""
    return stable_seed(KIND, "policy", task_name, int(env_seed))


def reset_rng_seed(task_name, env_seed):
    return stable_seed(KIND, "reset", task_name, int(env_seed))


def state_hash(env):
    flat = np.round(np.asarray(capture_snapshot(env)["states"], dtype=np.float64), 8) + 0.0
    return hashlib.sha256(flat.tobytes()).hexdigest()[:16]


def first_divergence(hashes_a, hashes_b):
    """First hash index where two arms differ; None if one trace is a prefix-equal copy."""
    for index, (a, b) in enumerate(zip(hashes_a, hashes_b)):
        if a != b:
            return index
    if len(hashes_a) != len(hashes_b):
        return min(len(hashes_a), len(hashes_b))
    return None


# --------------------------------------------------------------------------
# Gate


def resolve_stage_instruction(subtask_eval, stage, *, task_instruction, task_name, reprompt_level):
    return _resolve_recovery_target(
        subtask_eval,
        stage,
        high_level_summary=None,
        recovery_level=reprompt_level,
        atomic_instruction=task_instruction,
        evaluated_task_name=task_name,
    )


class StageGate:
    """Instruction controller of one rollout arm.

    ``update`` is called after the reset evaluation (step 0) and after every
    env step with the ordered tracker already updated. An active switch ends
    when its target predicate holds (or its stage latches) or after
    ``step_cap`` steps; then, if the tracker's current stage is eligible and
    has not fired yet, a new switch starts at the same step. Each stage fires
    at most once per rollout.
    """

    def __init__(self, *, arm, task_name, task_instruction, whitelist=(), step_cap=None, reprompt_level="atomic"):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}")
        self.arm = arm
        self.task_name = task_name
        self.task_instruction = task_instruction
        if arm == "always_reprompt":
            self.stages = None
        elif arm == "gated_reprompt":
            self.stages = set(whitelist or [])
        else:
            self.stages = set()
        self.step_cap = None if step_cap is None else int(step_cap)
        self.reprompt_level = reprompt_level
        self.active = None
        self.fired = []
        self.events = []
        self.records = []

    def eligible(self, stage):
        if stage is None:
            return False
        return self.stages is None or stage in self.stages

    @property
    def instruction(self):
        return self.active["instruction"] if self.active else self.task_instruction

    def update(self, step, tracker, subtask_eval):
        step = int(step)
        if self.active is not None:
            target = self.active["target"]
            if _subtask_is_complete(subtask_eval, target) or target in tracker.completed:
                self._end(step, "target_complete", target_completed=True)
            elif self.step_cap is not None and step - self.active["switch_step"] >= self.step_cap:
                self._end(step, "step_cap", target_completed=False)
        stage = tracker.current
        if self.active is None and self.eligible(stage) and stage not in self.fired:
            self.fired.append(stage)
            target, instruction, kind = resolve_stage_instruction(
                subtask_eval, stage, task_instruction=self.task_instruction,
                task_name=self.task_name, reprompt_level=self.reprompt_level,
            )
            record = {
                "stage": stage,
                "stage_index": tracker.count,
                "target": target,
                "instruction": instruction,
                "target_kind": kind,
                "entry_step": step,
            }
            if not instruction or target in (None, "task_success") or instruction == self.task_instruction:
                record.update(switched=False, skip_reason="equivalent_to_task_prompt")
                self.records.append(record)
                self.events.append({"step": step, "event": "skip", "stage": stage, "reason": "equivalent_to_task_prompt"})
            else:
                record.update(switched=True, switch_step=step)
                self.records.append(record)
                self.active = record
                self.events.append(
                    {"step": step, "event": "switch", "stage": stage, "target": target, "instruction": instruction}
                )
        return self.instruction

    def finish(self, step, reason):
        """Close an open switch at the end of the rollout."""
        if self.active is not None:
            self._end(int(step), reason, target_completed=False)

    def _end(self, step, reason, *, target_completed):
        record = self.active
        record.update(
            return_step=step,
            return_reason=reason,
            steps_active=step - record["switch_step"],
            target_completed=bool(target_completed),
        )
        self.events.append({"step": step, "event": "return", "stage": record["stage"], "reason": reason})
        self.active = None


# --------------------------------------------------------------------------
# Rollouts


class StepTrace:
    """{predicate: [[first_step, last_step], ...]} runs where it was true."""

    def __init__(self):
        self.runs = {}
        self._open = {}

    def observe(self, subtask_eval, step):
        predicates = (subtask_eval or {}).get("predicates", {}) or {}
        for name, payload in predicates.items():
            self.runs.setdefault(name, [])
            value = bool((payload or {}).get("value", False))
            if value and name not in self._open:
                self._open[name] = [int(step), int(step)]
                self.runs[name].append(self._open[name])
            elif value:
                self._open[name][1] = int(step)
            elif name in self._open:
                del self._open[name]
        for name in list(self._open):
            if name not in predicates:
                del self._open[name]


def run_rollout(env, policy, *, task_name, arm, env_seed, horizon, config):
    """One natural rollout of ``arm``; returns its record (without pair fields)."""
    started = time.time()
    seed_value = reset_rng_seed(task_name, env_seed)
    # Identical global RNG state before every arm's reset.
    np.random.seed(seed_value % (2**32))
    random.seed(seed_value)
    reset_kwargs = {"seed": int(env_seed)} if config["seed_protocol"] == "repeated_reset" else {}
    reset_result = env.reset(**reset_kwargs)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    reset_info = (
        reset_result[1]
        if isinstance(reset_result, tuple) and len(reset_result) > 1 and isinstance(reset_result[1], dict)
        else {}
    )
    clear_env_instruction(env)
    policy_seed = rollout_policy_seed(task_name, env_seed)
    reset_policy(policy)
    seed_control = seed_policy(policy, policy_seed)
    task_instruction = env_instruction(env, obs)
    initial_hash = state_hash(env)

    env_required = []

    def required_filter(names):
        env_required[:] = list(names)
        return effective_required(task_name, names, config)

    tracker = OrderedProgress(required_filter=required_filter)
    recorder = PredicateRecorder(task_name, config)
    trace = StepTrace()
    events = []
    reset_eval = reset_info.get("subtask_eval") or get_subtask_eval(env)
    _record_latches(events, tracker, tracker.update(reset_eval), 0)
    _check_required(task_name, tracker.required, reset_eval)
    whitelist = list(config["reprompt_whitelist"].get(task_name) or [])
    missing = [stage for stage in whitelist if stage not in tracker.required]
    if missing:
        raise ValueError(f"{task_name}: whitelisted stages {missing} are not in the ordered list {tracker.required}")
    recorder.observe(reset_eval, 0, tracker)
    trace.observe(reset_eval, 0)
    gate = StageGate(
        arm=arm, task_name=task_name, task_instruction=task_instruction, whitelist=whitelist,
        step_cap=task_value(config["reprompt_step_cap"], task_name), reprompt_level=config["reprompt_level"],
    )
    gate.update(0, tracker, reset_eval)

    stride = int(config["state_hash_stride"])
    hashes = []
    inferences = 0
    policy_resets = 0
    switch_log = []
    last_instruction = None
    success = False
    done = False
    num_steps = 0
    try:
        for step in range(int(horizon)):
            if step % stride == 0:
                hashes.append(initial_hash if step == 0 else state_hash(env))
            instruction = gate.instruction
            if instruction != last_instruction:
                if last_instruction is not None:
                    # Fresh policy context on every switch (XR-1 does this itself);
                    # keep the n-th inference's sampling seed equal across arms.
                    reset_policy(policy)
                    seed_policy(policy, policy_seed + inferences)
                    policy_resets += 1
                    switch_log.append(
                        {"step": step, "inference_index": inferences, "sampling_seed_base": policy_seed + inferences, "instruction": instruction}
                    )
                if instruction == task_instruction:
                    clear_env_instruction(env)
                else:
                    set_env_instruction(env, instruction)
                last_instruction = instruction
            if inference_boundary(policy, step, config["snapshot_stride"]):
                inferences += 1
            action = call_policy(policy, set_observation_instruction(obs, instruction), instruction=instruction)
            pop = getattr(policy, "pop_inference_record", None)
            if callable(pop):
                pop()
            obs, reward, done, info = _step_env(env, action)
            num_steps = step + 1
            current_eval = (info or {}).get("subtask_eval") or get_subtask_eval(env)
            _record_latches(events, tracker, tracker.update(current_eval), num_steps)
            recorder.observe(current_eval, num_steps, tracker)
            trace.observe(current_eval, num_steps)
            if _is_task_success(info, reward=reward, env=env):
                success = True
                break
            if done:
                break
            gate.update(num_steps, tracker, current_eval)
    finally:
        clear_env_instruction(env)
    end_reason = "task_success" if success else ("env_done" if done else "horizon")
    gate.finish(num_steps, end_reason)
    latch_steps = {}
    for event in events:
        latch_steps.setdefault(event["subtask"], event["step"])
    switched = [record for record in gate.records if record.get("switched")]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "task_name": task_name,
        "environment_seed": int(env_seed),
        "arm": arm,
        "policy_seed": int(policy_seed),
        "seed_control": seed_control,
        "reset_rng_seed": int(seed_value),
        "horizon": int(horizon),
        "success": bool(success),
        "num_steps": int(num_steps),
        "steps_to_success": int(num_steps) if success else None,
        "end_reason": end_reason,
        "instruction": task_instruction,
        "env_required_subtasks": list(env_required or tracker.required),
        "required_subtasks": list(tracker.required),
        "whitelist": sorted(gate.stages) if gate.stages is not None else "all",
        "progress_events": events,
        "stage_latch_steps": latch_steps,
        "final_ordered_count": int(tracker.count),
        "final_durable_count": int(recorder.durable_count(tracker)),
        "predicate_trace": trace.runs,
        "predicate_first_true_step": dict(recorder.first_true),
        "predicate_values_at_end": dict(recorder.values),
        "regression_events": recorder.regressions,
        "gate_events": gate.events,
        "gate_stages": gate.records,
        "instruction_switches": switch_log,
        "fired": bool(switched),
        "n_switches": len(switched),
        "first_switch_step": switched[0]["switch_step"] if switched else None,
        "cap_hits": sum(record.get("return_reason") == "step_cap" for record in switched),
        "policy_resets": int(policy_resets),
        "approx_policy_inferences": int(inferences),
        "initial_state_sha": initial_hash,
        "state_hash_stride": stride,
        "state_hashes": hashes,
        "wall_seconds": time.time() - started,
        "created_at": utc_now(),
    }


def run_seed(get_env, *, task_name, env_seed, horizon, config, output_dir, log=print):
    """All arms of one (task, seed), back to back; writes rollouts then the pair row."""
    pair_uid = uuid.uuid4().hex
    order = arm_order(task_name, env_seed, config["arms"], config["arm_order"])
    rows = {}
    for position, arm in enumerate(order):
        env, policy = get_env()
        row = run_rollout(env, policy, task_name=task_name, arm=arm, env_seed=env_seed, horizon=horizon, config=config)
        row.update(pair_uid=pair_uid, arm_position=position, arm_order=order)
        append_jsonl(Path(output_dir) / ROLLOUTS_NAME, row)
        rows[arm] = row
    base = rows["baseline"]
    pair = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "pair_uid": pair_uid,
        "task_name": task_name,
        "environment_seed": int(env_seed),
        "policy_seed": base["policy_seed"],
        "arm_order": order,
        "success": {arm: row["success"] for arm, row in rows.items()},
        "fired": {arm: row["fired"] for arm, row in rows.items()},
        "num_steps": {arm: row["num_steps"] for arm, row in rows.items()},
        "initial_state_match": len({row["initial_state_sha"] for row in rows.values()}) == 1,
        "first_divergence_step": {
            arm: (
                None
                if (index := first_divergence(base["state_hashes"], row["state_hashes"])) is None
                else index * int(config["state_hash_stride"])
            )
            for arm, row in rows.items()
            if arm != "baseline"
        },
        "first_switch_step": {arm: row["first_switch_step"] for arm, row in rows.items()},
        "wall_seconds": sum(row["wall_seconds"] for row in rows.values()),
        "env_steps": sum(row["num_steps"] for row in rows.values()),
        "created_at": utc_now(),
    }
    append_jsonl(Path(output_dir) / PAIRS_NAME, pair)
    return pair


# --------------------------------------------------------------------------
# Collection driver


def _write_status(output_dir, plan, pairs, errors, state):
    by_task = defaultdict(lambda: defaultdict(int))
    steps = seconds = 0.0
    for pair in pairs:
        counts = by_task[pair["task_name"]]
        counts["seeds"] += 1
        for arm, value in pair["success"].items():
            counts[f"{arm}_successes"] += int(value)
        for arm, value in pair["fired"].items():
            counts[f"{arm}_fired"] += int(value)
        counts["initial_state_mismatches"] += int(not pair["initial_state_match"])
        steps += pair["env_steps"]
        seconds += pair["wall_seconds"]
    status = {
        "status": state,
        "updated_at": utc_now(),
        "tasks": plan["config"]["tasks"],
        "arms": plan["config"]["arms"],
        "seeds_assigned_per_task": len(plan["seeds"]),
        "by_task": {task: dict(counts) for task, counts in by_task.items()},
        "num_pairs": len(pairs),
        "num_errors": len(errors),
        "env_steps": int(steps),
        "measured_seconds_per_env_step": seconds / steps if steps else None,
    }
    atomic_write_json(Path(output_dir) / STATUS_NAME, status)
    return status


def run_collection(config, output_dir, *, runtime=None, resume=False, shard_index=0, num_shards=1, host=None, port=None, log=print):
    config = resolve_config(config)
    if not config["tasks"]:
        raise ValueError("No tasks configured")
    output_dir = Path(output_dir)
    runtime = runtime or Runtime(config, host=host, port=port)
    horizons = {task: runtime.task_horizon(task) for task in config["tasks"]}
    seeds = shard_seeds(task_seeds(config), num_shards, shard_index)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "config": config,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "seeds": seeds,
        "horizons": horizons,
        "whitelisted_tasks": whitelisted_tasks(config),
        "control_tasks": control_tasks(config),
        "git_commit": current_git_commit(),
        "created_at": utc_now(),
    }
    plan_path = output_dir / PLAN_NAME
    if plan_path.exists():
        if not resume:
            raise FileExistsError(f"{plan_path} exists; pass --resume or choose a new output dir")
        previous = json.loads(plan_path.read_text())
        diffs = {
            key: (previous["config"].get(key), config.get(key))
            for key in FROZEN_PLAN_KEYS
            if previous["config"].get(key) != config.get(key)
        }
        for key in ("num_shards", "shard_index"):
            if previous.get(key) != plan[key]:
                diffs[key] = (previous.get(key), plan[key])
        if diffs:
            raise ValueError(f"Resume config differs from the registered plan: {diffs}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(plan_path, plan)

    pairs = read_jsonl(output_dir / PAIRS_NAME)
    errors = read_jsonl(output_dir / ERRORS_NAME)
    done = {(pair["task_name"], pair["environment_seed"]) for pair in pairs}
    _write_status(output_dir, plan, pairs, errors, "running")
    fresh = config["seed_protocol"] == "fresh_env"
    for block in seed_blocks(seeds, config["seed_block_size"]):
        for task_name in config["tasks"]:
            todo = [seed for seed in block if (task_name, seed) not in done]
            if not todo:
                continue
            holder = {"env": None, "policy": None}

            def get_env(seed):
                if holder["env"] is None or fresh:
                    _close(holder["env"], holder["policy"])
                    holder["env"] = runtime.make_env(task_name, seed)
                    holder["policy"] = runtime.make_policy(holder["env"])
                return holder["env"], holder["policy"]

            try:
                for env_seed in todo:
                    if len(errors) >= int(config["max_errors"]):
                        raise RuntimeError(f"Stopping after {len(errors)} errors; inspect {ERRORS_NAME}")
                    try:
                        pair = run_seed(
                            lambda seed=env_seed: get_env(seed), task_name=task_name, env_seed=env_seed,
                            horizon=horizons[task_name], config=config, output_dir=output_dir, log=log,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        event = {"task_name": task_name, "environment_seed": env_seed, "error": traceback.format_exc(), "created_at": utc_now()}
                        append_jsonl(output_dir / ERRORS_NAME, event)
                        errors.append(event)
                        log(f"[gate] error on {task_name} seed {env_seed}; see {ERRORS_NAME}")
                        _close(holder["env"], holder["policy"])
                        holder["env"] = holder["policy"] = None
                        continue
                    pairs.append(pair)
                    done.add((task_name, env_seed))
                    _write_status(output_dir, plan, pairs, errors, "running")
                    outcome = " ".join(
                        f"{arm}={int(pair['success'][arm])}{'*' if pair['fired'][arm] else ''}" for arm in config["arms"]
                    )
                    log(
                        f"[gate] {task_name} seed={env_seed} {outcome} "
                        f"initial_match={pair['initial_state_match']} pairs={len(pairs)}"
                    )
            finally:
                _close(holder["env"], holder["policy"])
    return _write_status(output_dir, plan, pairs, errors, "complete")


# --------------------------------------------------------------------------
# Statistics


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts ``b`` and ``c``."""
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def holm(p_values):
    """Holm step-down adjusted p-values (same order as the input)."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [None] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def mcnemar_power(n, p_help, p_harm, alpha=0.05):
    """Exact power of the two-sided McNemar test with ``n`` pairs.

    ``p_help``: P(arm succeeds, baseline fails); ``p_harm``: the reverse. The
    expected success difference is ``p_help - p_harm``.
    """
    n = int(n)
    p_conc = 1.0 - p_help - p_harm
    if n < 1 or p_conc < -1e-12:
        raise ValueError("invalid power scenario")
    reject = {}
    power = 0.0
    log_n = math.lgamma(n + 1)
    for b in range(n + 1):
        for c in range(n - b + 1):
            key = (min(b, c), b + c)
            if key not in reject:
                reject[key] = mcnemar_exact(b, c) <= alpha
            if not reject[key] or (b and p_help <= 0) or (c and p_harm <= 0):
                continue
            log_p = log_n - math.lgamma(b + 1) - math.lgamma(c + 1) - math.lgamma(n - b - c + 1)
            log_p += (b * math.log(p_help) if b else 0.0) + (c * math.log(p_harm) if c else 0.0)
            log_p += (n - b - c) * math.log(max(p_conc, 1e-300)) if n - b - c else 0.0
            power += math.exp(log_p)
    return min(1.0, power)


def minimum_detectable_difference(n, alpha=0.05, power=0.8, harm_ratio=0.0):
    """Smallest difference (0.01 grid) with the target power; harm = harm_ratio * help."""
    for step in range(1, 100):
        diff = step / 100.0
        p_help = diff / (1.0 - harm_ratio) if harm_ratio < 1 else diff
        p_harm = p_help - diff
        if p_help + p_harm > 1:
            break
        if mcnemar_power(n, p_help, p_harm, alpha) >= power:
            return diff
    return None


def _percentile_ci(samples, level=0.95):
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return None, None
    tail = (1 - level) / 2 * 100
    return float(np.percentile(samples, tail)), float(np.percentile(samples, 100 - tail))


def paired_bootstrap(diffs_by_task, *, n_boot=2000, seed=0):
    """Mean paired difference per task and their macro mean, seeds resampled within task.

    ``diffs_by_task``: {task: [per-seed difference]}. Returns
    {task: (estimate, lo, hi)} and ("macro", ...).
    """
    rng = np.random.default_rng(seed)
    tasks = [task for task, values in diffs_by_task.items() if len(values)]
    arrays = {task: np.asarray(diffs_by_task[task], dtype=float) for task in tasks}
    out = {}
    boots = {}
    for task in tasks:
        values = arrays[task]
        index = rng.integers(0, len(values), size=(int(n_boot), len(values)))
        boots[task] = values[index].mean(axis=1)
        lo, hi = _percentile_ci(boots[task])
        out[task] = (float(values.mean()), lo, hi)
    if tasks:
        macro = np.mean([boots[task] for task in tasks], axis=0)
        lo, hi = _percentile_ci(macro)
        out["macro"] = (float(np.mean([arrays[task].mean() for task in tasks])), lo, hi)
    return out


# --------------------------------------------------------------------------
# Analysis


def load_run(run_dirs):
    """Complete pairs and their rollouts; later run dirs win duplicate (task, seed)."""
    pairs = {}
    duplicates = []
    rollouts_by_uid = defaultdict(dict)
    plans = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if (run_dir / PLAN_NAME).exists():
            plans.append(json.loads((run_dir / PLAN_NAME).read_text()))
        for row in read_jsonl(run_dir / ROLLOUTS_NAME):
            rollouts_by_uid[row["pair_uid"]][row["arm"]] = row
        for pair in read_jsonl(run_dir / PAIRS_NAME):
            key = (pair["task_name"], pair["environment_seed"])
            if key in pairs:
                duplicates.append(list(key))
            pairs[key] = pair
    complete = []
    for key, pair in sorted(pairs.items()):
        rollouts = rollouts_by_uid.get(pair["pair_uid"], {})
        if set(rollouts) >= set(pair["success"]):
            complete.append((pair, rollouts))
    return complete, duplicates, plans


def _restricted_steps(row):
    return row["num_steps"] if row["success"] else row["horizon"]


def _stage_entered(row, stage):
    required = row["required_subtasks"]
    if stage not in required:
        return False
    index = required.index(stage)
    return index == 0 or required[index - 1] in row["stage_latch_steps"]


def analyze(run_dirs, *, n_boot=2000, seed=0, alpha=0.05):
    complete, duplicates, plans = load_run(run_dirs)
    config = plans[0]["config"] if plans else {}
    whitelist = config.get("reprompt_whitelist") or {}
    arms = list(config.get("arms") or ARMS)
    tasks = list(config.get("tasks") or sorted({pair["task_name"] for pair, _ in complete}))
    present_arms = [arm for arm in arms if any(arm in rollouts for _, rollouts in complete)]
    whitelisted = [task for task in tasks if whitelist.get(task)]
    controls = [task for task in tasks if not whitelist.get(task)]
    by_task = defaultdict(list)
    for pair, rollouts in complete:
        by_task[pair["task_name"]].append((pair, rollouts))

    rollout_rows = []
    for pair, rollouts in complete:
        for arm, row in rollouts.items():
            rollout_rows.append(
                {
                    "task": row["task_name"],
                    "environment_seed": row["environment_seed"],
                    "arm": arm,
                    "arm_position": row.get("arm_position"),
                    "success": int(row["success"]),
                    "num_steps": row["num_steps"],
                    "final_ordered_count": row["final_ordered_count"],
                    "n_required": len(row["required_subtasks"]),
                    "fired": int(row["fired"]),
                    "n_switches": row["n_switches"],
                    "cap_hits": row["cap_hits"],
                    "first_switch_step": row["first_switch_step"],
                    "switched_stages": ";".join(r["stage"] for r in row["gate_stages"] if r.get("switched")),
                    "initial_state_match": int(pair["initial_state_match"]),
                    "first_divergence_step": (pair.get("first_divergence_step") or {}).get(arm),
                }
            )

    # Per task x arm descriptive table.
    arm_rows = []
    stage_rows = []
    switch_rows = []
    for task in tasks:
        entries = by_task.get(task, [])
        for arm in present_arms:
            rows = [rollouts[arm] for _, rollouts in entries if arm in rollouts]
            if not rows:
                continue
            successes = [row["steps_to_success"] for row in rows if row["success"]]
            switched = [r for row in rows for r in row["gate_stages"] if r.get("switched")]
            arm_rows.append(
                {
                    "task": task,
                    "arm": arm,
                    "n": len(rows),
                    "success_rate": float(np.mean([row["success"] for row in rows])),
                    "mean_ordered_progress": float(
                        np.mean([row["final_ordered_count"] / max(1, len(row["required_subtasks"])) for row in rows])
                    ),
                    "median_steps_to_success": float(np.median(successes)) if successes else None,
                    "restricted_mean_steps": float(np.mean([_restricted_steps(row) for row in rows])),
                    "fired_rate": float(np.mean([row["fired"] for row in rows])),
                    "mean_switches": float(np.mean([row["n_switches"] for row in rows])),
                    "cap_hit_rate": (
                        float(np.mean([r.get("return_reason") == "step_cap" for r in switched])) if switched else None
                    ),
                }
            )
            required = rows[0]["required_subtasks"]
            for index, stage in enumerate(required):
                stage_rows.append(
                    {
                        "task": task,
                        "arm": arm,
                        "stage_index": index,
                        "stage": stage,
                        "whitelisted": int(stage in (whitelist.get(task) or [])),
                        "entered_rate": float(np.mean([_stage_entered(row, stage) for row in rows])),
                        "completed_rate": float(np.mean([stage in row["stage_latch_steps"] for row in rows])),
                    }
                )
            stages = sorted({r["stage"] for r in switched}, key=lambda s: required.index(s) if s in required else 99)
            for stage in stages:
                records = [r for r in switched if r["stage"] == stage]
                reasons = Counter(r.get("return_reason") for r in records)
                switch_rows.append(
                    {
                        "task": task,
                        "arm": arm,
                        "stage": stage,
                        "instruction": records[0]["instruction"],
                        "n_fired": len(records),
                        "fired_rate": len(records) / len(rows),
                        "target_completed_rate": float(np.mean([r.get("target_completed", False) for r in records])),
                        "cap_hits": reasons.get("step_cap", 0),
                        "ended_by_task_success": reasons.get("task_success", 0),
                        "ended_by_horizon": reasons.get("horizon", 0),
                        "median_steps_active": float(np.median([r["steps_active"] for r in records])),
                        "median_switch_step": float(np.median([r["switch_step"] for r in records])),
                    }
                )

    # Paired contrasts per task and macro.
    contrast_rows = []
    for arm_a, arm_b in CONTRASTS:
        if arm_a not in present_arms or arm_b not in present_arms:
            continue
        diffs = {}
        rmst = {}
        per_task = {}
        for task in tasks:
            entries = [(pair, rollouts) for pair, rollouts in by_task.get(task, []) if arm_a in rollouts and arm_b in rollouts]
            if not entries:
                continue
            a = np.array([rollouts[arm_a]["success"] for _, rollouts in entries], dtype=int)
            b = np.array([rollouts[arm_b]["success"] for _, rollouts in entries], dtype=int)
            # Pairs whose instructions differed: at least one arm switched.
            fired = np.array([rollouts[arm_a]["fired"] or rollouts[arm_b]["fired"] for _, rollouts in entries], dtype=bool)
            diffs[task] = (a - b).tolist()
            rmst[task] = [
                _restricted_steps(rollouts[arm_a]) - _restricted_steps(rollouts[arm_b]) for _, rollouts in entries
            ]
            help_count = int(((a == 1) & (b == 0)).sum())
            harm_count = int(((a == 0) & (b == 1)).sum())
            per_task[task] = {
                "n_pairs": len(entries),
                "success_a": float(a.mean()),
                "success_b": float(b.mean()),
                "b_help": help_count,
                "c_harm": harm_count,
                "mcnemar_p": mcnemar_exact(help_count, harm_count),
                "rescue_rate": help_count / int((b == 0).sum()) if (b == 0).any() else None,
                "harm_rate_on_b_successes": harm_count / int((b == 1).sum()) if (b == 1).any() else None,
                "n_fired": int(fired.sum()),
                "diff_when_fired": float((a - b)[fired].mean()) if fired.any() else None,
                "discordant_when_not_fired": int(((a != b) & ~fired).sum()),
                "initial_state_mismatches": int(sum(not pair["initial_state_match"] for pair, _ in entries)),
            }
        boot = paired_bootstrap(diffs, n_boot=n_boot, seed=seed)
        boot_rmst = paired_bootstrap(rmst, n_boot=n_boot, seed=seed + 1)
        groups = [(task, [task]) for task in per_task]
        groups.append(("macro_whitelisted", [task for task in whitelisted if task in per_task]))
        groups.append(("macro_all", list(per_task)))
        for label, members in groups:
            if not members:
                continue
            if label in per_task:
                stats = dict(per_task[label])
                estimate, lo, hi = boot[label]
                rmst_estimate, rmst_lo, rmst_hi = boot_rmst[label]
            else:
                sub = paired_bootstrap({task: diffs[task] for task in members}, n_boot=n_boot, seed=seed)["macro"]
                sub_rmst = paired_bootstrap({task: rmst[task] for task in members}, n_boot=n_boot, seed=seed + 1)["macro"]
                estimate, lo, hi = sub
                rmst_estimate, rmst_lo, rmst_hi = sub_rmst
                help_count = sum(per_task[task]["b_help"] for task in members)
                harm_count = sum(per_task[task]["c_harm"] for task in members)
                fired_values = [per_task[task]["n_fired"] for task in members]
                stats = {
                    "n_pairs": sum(per_task[task]["n_pairs"] for task in members),
                    "success_a": float(np.mean([per_task[task]["success_a"] for task in members])),
                    "success_b": float(np.mean([per_task[task]["success_b"] for task in members])),
                    "b_help": help_count,
                    "c_harm": harm_count,
                    "mcnemar_p": mcnemar_exact(help_count, harm_count),
                    "rescue_rate": None,
                    "harm_rate_on_b_successes": None,
                    "n_fired": sum(fired_values),
                    "diff_when_fired": None,
                    "discordant_when_not_fired": sum(per_task[task]["discordant_when_not_fired"] for task in members),
                    "initial_state_mismatches": sum(per_task[task]["initial_state_mismatches"] for task in members),
                }
            contrast_rows.append(
                {
                    "contrast": f"{arm_a} - {arm_b}",
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "group": label,
                    "tasks": ";".join(members),
                    "difference": estimate,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "restricted_mean_steps_difference": rmst_estimate,
                    "rmst_ci_lo": rmst_lo,
                    "rmst_ci_hi": rmst_hi,
                    **stats,
                }
            )
    primary_tests = [
        row for row in contrast_rows
        if (row["arm_a"], row["arm_b"]) == PRIMARY_CONTRAST and row["group"] in whitelisted
    ]
    for row, adjusted in zip(primary_tests, holm([row["mcnemar_p"] for row in primary_tests])):
        row["mcnemar_p_holm"] = adjusted
    primary = next(
        (
            row for row in contrast_rows
            if (row["arm_a"], row["arm_b"]) == PRIMARY_CONTRAST and row["group"] == "macro_whitelisted"
        ),
        None,
    )

    # Control tasks: the gated arm cannot switch, so it must match baseline.
    control_rows = []
    for task in controls:
        entries = [(pair, rollouts) for pair, rollouts in by_task.get(task, []) if "gated_reprompt" in rollouts]
        if not entries:
            continue
        gated = [rollouts["gated_reprompt"] for _, rollouts in entries]
        control_rows.append(
            {
                "task": task,
                "n_pairs": len(entries),
                "gated_switches": int(sum(row["n_switches"] for row in gated)),
                "outcome_concordance": float(
                    np.mean([rollouts["gated_reprompt"]["success"] == rollouts["baseline"]["success"] for _, rollouts in entries])
                ),
                "identical_state_traces": float(
                    np.mean([(pair.get("first_divergence_step") or {}).get("gated_reprompt") is None for pair, _ in entries])
                ),
                "initial_state_mismatches": int(sum(not pair["initial_state_match"] for pair, _ in entries)),
            }
        )

    summary = {
        "kind": KIND,
        "run_dirs": [str(path) for path in run_dirs],
        "tasks": tasks,
        "arms": present_arms,
        "whitelist": whitelist,
        "whitelisted_tasks": whitelisted,
        "control_tasks": controls,
        "n_complete_pairs": len(complete),
        "pairs_per_task": {task: len(by_task.get(task, [])) for task in tasks},
        "duplicate_pairs": duplicates,
        "initial_state_mismatches": int(sum(not pair["initial_state_match"] for pair, _ in complete)),
        "alpha": alpha,
        "primary": primary,
        "arms_by_task": arm_rows,
        "contrasts": contrast_rows,
        "control_checks": control_rows,
        "switches": switch_rows,
        "stage_completion": stage_rows,
    }
    tables = {
        "rollouts": rollout_rows,
        "arms_by_task": arm_rows,
        "contrasts": contrast_rows,
        "control_checks": control_rows,
        "switches": switch_rows,
        "stage_completion": stage_rows,
    }
    return summary, tables


def _num(value, spec=".2f"):
    return "-" if value is None else format(value, spec)


def _signed(value):
    return "-" if value is None else f"{value:+.3f}"


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(summary, tables, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "summary.json", summary)
    for name, rows in tables.items():
        _write_csv(output_dir / f"{name}.csv", rows)
    lines = [
        "# Stage-gated re-prompt trial",
        "",
        f"Runs: {', '.join(summary['run_dirs'])}",
        f"Complete pairs: {summary['n_complete_pairs']} "
        f"({', '.join(f'{t} {n}' for t, n in summary['pairs_per_task'].items())}); arms: {', '.join(summary['arms'])}.",
        f"Whitelist: {json.dumps(summary['whitelist'])}; controls: {', '.join(summary['control_tasks']) or '-'}.",
    ]
    if summary["initial_state_mismatches"]:
        lines.append(f"**Initial-state mismatches between arms: {summary['initial_state_mismatches']}** (pairing is suspect).")
    if summary["duplicate_pairs"]:
        lines.append(f"Duplicate (task, seed) pairs across run dirs: {len(summary['duplicate_pairs'])} (last one kept).")
    primary = summary["primary"]
    lines += ["", "## Primary endpoint", ""]
    if primary:
        lines.append(
            f"Full-task success, gated_reprompt minus baseline, macro over {primary['tasks'].replace(';', ', ')}: "
            f"**{_signed(primary['difference'])}** [{_signed(primary['ci_lo'])}, {_signed(primary['ci_hi'])}] "
            f"(paired bootstrap 95% CI, {primary['n_pairs']} pairs); exact McNemar on pooled pairs "
            f"{primary['b_help']} helped / {primary['c_harm']} harmed, p = {primary['mcnemar_p']:.4f} "
            f"(alpha {summary['alpha']})."
        )
    else:
        lines.append("No complete gated/baseline pairs on whitelisted tasks.")
    lines += [
        "",
        "## Paired contrasts",
        "",
        "`help` = arm A succeeded and B failed; `harm` = the reverse; `harm|B ok` = harm among B's successes; "
        "`switched` = pairs in which either arm switched instruction; `Δ|switched` = difference among them; "
        "`disc|¬switched` = discordant pairs in which neither arm switched (nondeterminism); `Δsteps` = difference in "
        "restricted mean steps (failures count as H).",
        "",
        "| contrast | group | n | A | B | Δ [95% CI] | help | harm | McNemar p | Holm p | harm\\|B ok | switched | Δ\\|switched | disc\\|¬switched | Δsteps [95% CI] |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["contrasts"]:
        lines.append(
            f"| {row['contrast']} | {row['group']} | {row['n_pairs']} | {row['success_a']:.2f} | {row['success_b']:.2f} | "
            f"{_signed(row['difference'])} [{_signed(row['ci_lo'])}, {_signed(row['ci_hi'])}] | {row['b_help']} | {row['c_harm']} | "
            f"{row['mcnemar_p']:.4f} | {_num(row.get('mcnemar_p_holm'), '.4f')} | "
            f"{_num(row['harm_rate_on_b_successes'])} | {row['n_fired']} | "
            f"{_signed(row['diff_when_fired'])} | {row['discordant_when_not_fired']} | "
            f"{row['restricted_mean_steps_difference']:+.0f} [{row['rmst_ci_lo']:+.0f}, {row['rmst_ci_hi']:+.0f}] |"
        )
    lines += [
        "",
        "## Arms by task",
        "",
        "| task | arm | n | success | ordered progress | median steps to success | restricted mean steps | fired | switches/rollout | cap hits/switch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["arms_by_task"]:
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['n']} | {row['success_rate']:.2f} | {row['mean_ordered_progress']:.2f} | "
            f"{_num(row['median_steps_to_success'], '.0f')} | "
            f"{row['restricted_mean_steps']:.0f} | {row['fired_rate']:.2f} | {row['mean_switches']:.2f} | "
            f"{_num(row['cap_hit_rate'])} |"
        )
    lines += [
        "",
        "## Switches",
        "",
        "| task | arm | stage | instruction | fired | fired/rollout | target done | cap hits | median switch step | median active steps |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["switches"]:
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['stage']} | {row['instruction']} | {row['n_fired']} | {row['fired_rate']:.2f} | "
            f"{row['target_completed_rate']:.2f} | {row['cap_hits']} | {row['median_switch_step']:.0f} | {row['median_steps_active']:.0f} |"
        )
    lines += [
        "",
        "## Control tasks (no whitelisted stage: gated must equal baseline)",
        "",
        "| task | n | gated switches | outcome concordance | identical state traces | initial-state mismatches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["control_checks"]:
        lines.append(
            f"| {row['task']} | {row['n_pairs']} | {row['gated_switches']} | {row['outcome_concordance']:.2f} | "
            f"{row['identical_state_traces']:.2f} | {row['initial_state_mismatches']} |"
        )
    lines += [
        "",
        "## Stage completion",
        "",
        "| task | arm | stage | whitelisted | entered | completed |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary["stage_completion"]:
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['stage_index']}: {row['stage']} | {row['whitelisted']} | "
            f"{row['entered_rate']:.2f} | {row['completed_rate']:.2f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return output_dir / "summary.md"


# --------------------------------------------------------------------------
# Dry run


def build_plan(config, *, horizons=None, calibration=None):
    """Simulator-free cost estimate and McNemar power."""
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
    workers = int(config["num_workers"])
    seeds = int(config["seeds_per_task"])
    arms = config["arms"]
    blocks_per_shard = math.ceil(math.ceil(seeds / workers) / int(config["seed_block_size"]))
    rows = []
    warnings = []
    for task in config["tasks"]:
        horizon = int(horizons[task])
        rate = config["expected_success_rates"].get(task)
        if rate is None:
            warnings.append(f"{task}: no expected success rate; assuming 0.5")
            rate = 0.5
        fraction = float(task_value(config["success_step_fraction"], task) or DEFAULTS["success_step_fraction"])
        rollouts = seeds * len(arms)
        expected_steps = rollouts * ((1 - rate) * horizon + rate * fraction * horizon)
        upper_steps = rollouts * horizon
        env_creations = blocks_per_shard * workers if config["seed_protocol"] == "repeated_reset" else rollouts
        overhead = rollouts * float(config["seconds_per_reset"]) + env_creations * float(config["seconds_per_env_creation"])
        rows.append(
            {
                "task": task,
                "horizon": horizon,
                "whitelist": config["reprompt_whitelist"].get(task) or [],
                "reprompt_step_cap": task_value(config["reprompt_step_cap"], task),
                "expected_success_rate": rate,
                "success_step_fraction": fraction,
                "rollouts": rollouts,
                "env_steps_expected": int(expected_steps),
                "env_steps_upper": int(upper_steps),
                "worker_hours_expected": round((expected_steps * seconds_per_step + overhead) / 3600.0, 2),
                "worker_hours_upper": round((upper_steps * seconds_per_step + overhead) / 3600.0, 2),
                "overhead_hours": round(overhead / 3600.0, 2),
            }
        )
    usable = float(config["allocation_hours"]) - float(config["allocation_overhead_hours"])
    totals = {}
    for kind in ("expected", "upper"):
        worker_hours = sum(row[f"worker_hours_{kind}"] for row in rows)
        wall = worker_hours / workers
        allocations = max(1, math.ceil(wall / usable))
        totals[kind] = {
            "env_steps": int(sum(row[f"env_steps_{kind}"] for row in rows)),
            "worker_hours": round(worker_hours, 2),
            "gpu_hours": round(worker_hours * int(config["gpus_per_worker"]), 2),
            "wall_hours_with_workers": round(wall, 2),
            "wall_hours_one_worker": round(worker_hours, 2),
            "allocations_needed": allocations,
            "fits_single_allocation": allocations == 1,
        }
    alpha = float(config["alpha"])
    target = float(config["power_target_difference"])
    n_white = len(whitelisted_tasks(config))
    scenarios = [(target, 0.0), (target + 0.05, 0.05), (target + 0.10, 0.10), (0.10, 0.0)]
    power_rows = []
    for label, n in (("per_task", seeds), ("pooled_whitelisted", seeds * n_white)):
        if n < 1:
            continue
        for p_help, p_harm in scenarios:
            power_rows.append(
                {
                    "group": label,
                    "n_pairs": n,
                    "difference": round(p_help - p_harm, 3),
                    "p_help": p_help,
                    "p_harm": p_harm,
                    "power": round(mcnemar_power(n, p_help, p_harm, alpha), 3),
                }
            )
    mdd = {
        label: {
            "no_harm": minimum_detectable_difference(n, alpha),
            "harm_one_third_of_help": minimum_detectable_difference(n, alpha, harm_ratio=1 / 3),
        }
        for label, n in (("per_task", seeds), ("pooled_whitelisted", seeds * n_white))
        if n >= 1
    }
    return {
        "dry_run": True,
        "kind": KIND,
        "tasks": config["tasks"],
        "arms": arms,
        "whitelisted_tasks": whitelisted_tasks(config),
        "control_tasks": control_tasks(config),
        "seeds_per_task": seeds,
        "seed_range": [int(config["base_seed"]), int(config["base_seed"]) + seeds - 1],
        "seconds_per_env_step": seconds_per_step,
        "throughput_source": throughput_source,
        "num_workers": workers,
        "gpus_per_worker": int(config["gpus_per_worker"]),
        "usable_hours_per_allocation": usable,
        "per_task": rows,
        "totals": totals,
        "power": {
            "test": "two-sided exact McNemar",
            "alpha": alpha,
            "rows": power_rows,
            "minimum_detectable_difference_80pct": mdd,
        },
        "warnings": warnings,
        "notes": [
            "upper: every rollout runs the full horizon; expected: successes end at success_step_fraction * H",
            "power depends on discordant pairs: p_help = arm succeeds and baseline fails, p_harm = the reverse",
            "the primary test pools the whitelisted tasks' pairs (gated_reprompt vs baseline)",
        ],
    }


# --------------------------------------------------------------------------
# CLI


def _load_config(args):
    config = json.loads(Path(args.config).read_text()) if getattr(args, "config", None) else {}
    overrides = {}
    for key in ("tasks", "arms", "seeds_per_task", "base_seed", "num_workers"):
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
        p.add_argument("--arms", nargs="+", choices=ARMS)
        p.add_argument("--seeds-per-task", type=int)
        p.add_argument("--base-seed", type=int)
        p.add_argument("--num-workers", type=int)

    plan = sub.add_parser("plan", help="dry run: validate config, estimate GPU hours and power")
    common(plan)
    plan.add_argument("--calibration-dir", type=Path, help="run dir whose status.json gives measured s/step")
    plan.add_argument("--output", type=Path)

    collect = sub.add_parser("collect", help="run all arms for every assigned (task, seed)")
    common(collect)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--policy-arg", action="append", default=[])
    collect.add_argument("--host")
    collect.add_argument("--port", type=int)
    collect.add_argument("--shard-index", type=int, default=0)
    collect.add_argument("--num-shards", type=int, default=1)
    collect.add_argument("--resume", action="store_true")

    analysis = sub.add_parser("analyze", help="paired success contrasts, McNemar, bootstrap CIs, switch statistics")
    analysis.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--n-boot", type=int, default=2000)
    analysis.add_argument("--alpha", type=float, default=0.05)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        config = _load_config(args)
        calibration = json.loads((args.calibration_dir / STATUS_NAME).read_text()) if args.calibration_dir else None
        plan = build_plan(config, calibration=calibration)
        print(json.dumps(plan, indent=2, sort_keys=True))
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
    summary, tables = analyze(args.run_dirs, n_boot=args.n_boot, alpha=args.alpha)
    path = write_analysis(summary, tables, args.output_dir)
    print(path.read_text())
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
