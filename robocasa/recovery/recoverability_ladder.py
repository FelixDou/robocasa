"""Recoverability ladder: approximate branching to locate recovery value.

Natural rollouts are run once. At every genuine policy inference the collector
keeps a cheap simulator snapshot (flattened MuJoCo state plus qpos/qvel), the
latched and durable ordered-subtask counts and the raw value of every subtask
predicate; the model XML and episode metadata are written once per parent.
After the rollout ends, a failure detector picks the detection snapshot and
branches restart from parent snapshots under one rung of an escalating
intervention ladder:

``resample_continue``  restore the anchor state, fresh policy sampling seed
``robot_reset``        detection world, robot joints restored to an anchor
``subtask_reprompt``   restore the anchor state, prompt the current ordered
                       (or regressed) subtask until it completes, then return
                       to the task instruction
``world_reset``        restore the full anchor state at detection time (oracle)

Anchors are snapshot indices counted in policy inferences: ``detection``,
``last_good``/``stage_entry`` (first inference after the last ordered-subtask
completion before detection), ``last_good_minus_2`` and the time-curve anchors
``stage_entry_plus_<n>`` (clipped to detection). Resample and reprompt branches
start at their anchor, so their cells trace recoverability over time; robot
and world resets are applied at detection. Cells whose interventions are
identical for a parent are run once and aliased.

Schema 2 adds the stage-overdue and regression detectors, a durable progress
count, raw predicate traces, ladder-level required-subtask overrides,
time-curve anchors and the ``remaining_horizon`` budget. Schema-1 shards stay
readable by ``analyze``, ``calibrate-stages`` and ``audit-predicates``.

Determinism is approximate by design: restore residuals are recorded, never
gated on. This module deliberately does not use the bit-exact full-snapshot
replay stack; it reuses ``recovery_rollout.apply_recovery_mode`` (the
``env_to_last_good``/``eef_to_last_good`` modes of
``evaluate_recovery_benchmark.py``) and the env's ordered subtask predicates.

Subcommands: ``plan`` (dry run + GPU-hour estimate), ``collect``,
``audit-predicates``, ``calibrate-stages``, ``analyze``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    _step_env,
    _subtask_is_complete,
    apply_recovery_mode,
    call_policy,
    set_observation_instruction,
)
from robocasa.recovery.subtask_eval import get_subtask_eval


SCHEMA_VERSION = 2
RUNGS = ("resample_continue", "robot_reset", "subtask_reprompt", "world_reset")
BASE_ANCHORS = ("detection", "last_good", "last_good_minus_2", "stage_entry")
TIME_CURVE_PREFIX = "stage_entry_plus_"
DETECTORS = ("stall", "time", "end", "stage_overdue", "regression", "overdue_or_regression")
BUDGET_MODES = ("fixed_fraction", "remaining_horizon")
V1_CELLS = (
    ("resample_continue", "detection"),
    ("robot_reset", "last_good"),
    ("robot_reset", "last_good_minus_2"),
    ("subtask_reprompt", "detection"),
    ("world_reset", "last_good"),
    ("world_reset", "last_good_minus_2"),
)
DEFAULT_CONTROL_CELLS = (
    ("resample_continue", "detection"),
    ("robot_reset", "last_good"),
    ("subtask_reprompt", "detection"),
    ("world_reset", "last_good"),
)
TIME_CURVE_ANCHORS = ("stage_entry", "stage_entry_plus_4", "stage_entry_plus_8", "stage_entry_plus_16")
DEFAULT_CELLS = DEFAULT_CONTROL_CELLS + tuple(
    (rung, anchor) for rung in ("resample_continue", "subtask_reprompt") for anchor in TIME_CURVE_ANCHORS
)
# Ladder used for (exploratory) per-parent failure attribution: the first rung
# whose estimated success probability reaches the threshold names the class.
LADDER = (
    ("resample_continue", "detection", "stochastic"),
    ("robot_reset", "last_good", "robot_state"),
    ("subtask_reprompt", "detection", "instruction"),
    ("world_reset", "last_good", "world_state"),
)
UNRECOVERED_CLASS = "competence"
MAX_REGRESSION_EVENTS = 500
RELEASE_STAGES = ("release",)

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
    # Natural parents only (no branching): audit and stage-calibration input.
    "parents_only": False,
    "k_branches": 4,
    "cells": [list(cell) for cell in DEFAULT_CELLS],
    # Cells for success controls; None means the same cells as failures.
    "control_cells": None,
    "detector": "overdue_or_regression",
    "stall_fraction": 0.15,
    "time_fraction": 0.5,
    # Stage-overdue detector: JSON written by ``calibrate-stages``.
    "stage_calibration": None,
    "stage_quantile": 0.9,
    "stage_min_examples": 3,
    "stage_overdue_min_steps": 32,
    # Regression detector: a latched predicate false for this many steps.
    "regression_min_steps": 16,
    "regression_ignore_stages": ["precondition", "release"],
    "regression_ignore_predicates": {},
    # Ladder-level required ordered list: {task: [names]} replaces the env's
    # list, then {task: [names]} are dropped. Shared predicates are untouched.
    "required_subtask_overrides": {},
    "exclude_predicates": {},
    "budget_mode": "fixed_fraction",
    "budget_fraction": 0.5,
    # Scalar or {task: steps}.
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
    "expected_detection_fraction": 0.35,
    "expected_stage_entry_fraction": 0.15,
    "seconds_per_env_step": 0.06,
    "seconds_per_reset": 20.0,
    "success_step_fraction": 0.6,
    "assumed_branch_success_rate": 0.3,
    "assumed_control_success_rate": 0.5,
    "assumed_success_budget_fraction": 0.5,
    "gpus_per_worker": 2,
    "num_workers": 1,
    "allocation_hours": 24.0,
    "allocation_overhead_hours": 0.5,
    "state_dim_estimate": 400,
    "feature_bytes_per_inference": 5 * 30 * 1024 * 2,
}

# Plan entries that a resumed collection must not change.
FROZEN_PLAN_KEYS = (
    "tasks", "cells", "control_cells", "k_branches", "detector", "stall_fraction",
    "time_fraction", "budget_mode", "budget_fraction", "max_budget_steps", "base_seed",
    "split", "seed_protocol", "policy_module", "parents_only", "stage_quantile",
    "stage_min_examples", "stage_overdue_min_steps", "regression_min_steps",
    "regression_ignore_stages", "regression_ignore_predicates",
    "required_subtask_overrides", "exclude_predicates",
)


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


def anchor_offset(anchor):
    """Inferences after stage entry for time-curve anchors, else None."""
    if anchor == "stage_entry":
        return 0
    if isinstance(anchor, str) and anchor.startswith(TIME_CURVE_PREFIX):
        suffix = anchor[len(TIME_CURVE_PREFIX):]
        if suffix.isdigit() and int(suffix) > 0:
            return int(suffix)
    return None


def valid_anchor(anchor):
    return anchor in BASE_ANCHORS or anchor_offset(anchor) is not None


def cell_offsets(cells):
    return sorted({anchor_offset(anchor) for _, anchor in cells if anchor_offset(anchor)})


def anchor_sort_key(anchor):
    offset = anchor_offset(anchor)
    if anchor == "last_good_minus_2":
        return (0, 0)
    if anchor == "last_good":
        return (1, 0)
    if offset is not None:
        return (2, offset)
    if anchor == "detection":
        return (3, 0)
    return (4, 0)


def cell_sort_key(cell):
    return (RUNGS.index(cell[0]), anchor_sort_key(cell[1]))


def task_value(value, task_name):
    """A scalar option, or the per-task entry of a {task: value} option."""
    if isinstance(value, dict):
        return value.get(task_name)
    return value


def _validate_cells(cells, key):
    cells = [tuple(cell) for cell in cells]
    for cell in cells:
        if len(cell) != 2 or cell[0] not in RUNGS or not valid_anchor(cell[1]):
            raise ValueError(f"Invalid ladder cell in {key}: {cell}")
    if len(set(cells)) != len(cells):
        raise ValueError(f"Duplicate ladder cells in {key}")
    return [list(cell) for cell in cells]


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
                raise ValueError(f"Unknown recoverability-ladder option: {key}")
            if value is not None:
                merged[key] = value
    merged["cells"] = _validate_cells(merged["cells"], "cells")
    if merged["control_cells"] is not None:
        merged["control_cells"] = _validate_cells(merged["control_cells"], "control_cells")
    if merged["detector"] not in DETECTORS:
        raise ValueError(f"--detector must be one of {DETECTORS}")
    if merged["budget_mode"] not in BUDGET_MODES:
        raise ValueError(f"--budget-mode must be one of {BUDGET_MODES}")
    if merged["policy_context"] not in {"reset", "restore"}:
        raise ValueError("--policy-context must be reset or restore")
    if merged["seed_protocol"] not in {"repeated_reset", "fresh_env"}:
        raise ValueError("--seed-protocol must be repeated_reset or fresh_env")
    for key in ("k_branches", "parents_per_task", "snapshot_stride", "stage_min_examples"):
        if int(merged[key]) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("stall_fraction", "time_fraction", "budget_fraction"):
        if not 0 < float(merged[key]) <= 1.5:
            raise ValueError(f"{key} must be in (0, 1.5]")
    if not 0 < float(merged["stage_quantile"]) <= 1:
        raise ValueError("stage_quantile must be in (0, 1]")
    for key in ("stage_overdue_min_steps", "regression_min_steps"):
        if int(merged[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    for key in ("required_subtask_overrides", "exclude_predicates", "regression_ignore_predicates"):
        merged[key] = _validate_name_map(merged[key], key)
    merged["regression_ignore_stages"] = list(merged["regression_ignore_stages"] or [])
    cap = merged["max_budget_steps"]
    caps = cap.values() if isinstance(cap, dict) else ([] if cap is None else [cap])
    if any(int(value) < 1 for value in caps):
        raise ValueError("max_budget_steps must be positive")
    merged["parents_only"] = bool(merged["parents_only"])
    return merged


def task_budget_steps(horizon, config, task_name=None):
    """Fixed-fraction budget: ``ceil(budget_fraction * H)`` (schema 1)."""
    budget = max(1, int(math.ceil(float(config["budget_fraction"]) * int(horizon))))
    cap = task_value(config.get("max_budget_steps"), task_name)
    if cap:
        budget = min(budget, int(cap))
    return budget


def branch_budget(horizon, start_step, config, task_name=None):
    """Branch budget in env steps for a branch whose clock starts at ``start_step``.

    ``fixed_fraction`` (schema 1): the same ``ceil(budget_fraction * H)`` for
    every branch. ``remaining_horizon``: the steps the natural rollout had left,
    ``H - start_step``, capped at the task horizon (and ``max_budget_steps``).
    """
    if config["budget_mode"] != "remaining_horizon":
        return task_budget_steps(horizon, config, task_name)
    budget = min(int(horizon), max(1, int(horizon) - int(round(float(start_step)))))
    cap = task_value(config.get("max_budget_steps"), task_name)
    if cap:
        budget = min(budget, int(cap))
    return budget


def effective_required(task_name, env_required, config):
    """The ladder's ordered required list: env list, overridden, minus exclusions."""
    override = (config.get("required_subtask_overrides") or {}).get(task_name)
    base = list(override) if override else list(env_required or [])
    excluded = set((config.get("exclude_predicates") or {}).get(task_name) or [])
    return [name for name in base if name not in excluded]


def regression_eligible(name, stage, task_name, config):
    """Whether a latched predicate must stay true (precondition/release need not)."""
    if stage in set(config.get("regression_ignore_stages") or []):
        return False
    return name not in set((config.get("regression_ignore_predicates") or {}).get(task_name) or [])


def is_release_predicate(name, stage=None):
    if stage is not None:
        return stage in RELEASE_STAGES
    return str(name).endswith("released")


# --------------------------------------------------------------------------
# Ordered subtask progress, predicate traces and snapshots


class OrderedProgress:
    """Incremental version of the ordered prefix in ``build_subtask_trace``.

    ``count`` is the latched ordered count: a predicate counts once it was true
    while every earlier one had latched, even if it is false later.
    """

    def __init__(self, required=None, count=0, required_filter=None):
        self.required = list(required) if required else []
        self.count = int(count)
        self._required_filter = required_filter

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
            required = list(subtask_eval.get("required_predicates", []))
            self.required = self._required_filter(required) if self._required_filter else required
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


class PredicateRecorder:
    """Raw predicate values, first-true steps and post-latch regressions.

    A regression event opens when a latched required predicate is false and
    closes when it is true again. Events of predicates that need not persist
    (``regression_eligible`` false) are recorded with ``eligible=False``.
    """

    def __init__(self, task_name, config):
        self.task_name = task_name
        self.config = config
        self.names = []
        self.stages = {}
        self.first_true = {}
        self.values = {}
        self.at_reset = None
        self.regressions = []
        self.truncated = False
        self._open = {}

    def eligible(self, name):
        return regression_eligible(name, self.stages.get(name), self.task_name, self.config)

    def observe(self, subtask_eval, step, tracker):
        if not subtask_eval:
            return
        predicates = subtask_eval.get("predicates", {}) or {}
        for name, payload in predicates.items():
            if name not in self.stages:
                self.names.append(name)
                self.stages[name] = (payload or {}).get("stage")
                self.first_true[name] = None
        values = {name: bool((predicates.get(name) or {}).get("value", False)) for name in self.names}
        for name, value in values.items():
            if value and self.first_true[name] is None:
                self.first_true[name] = int(step)
        if self.at_reset is None:
            self.at_reset = dict(values)
        self.values = values
        if subtask_eval.get("task_success", False):
            # Success latches the remaining stages; those are not regressions.
            return
        for index, name in enumerate(tracker.completed):
            event = self._open.get(name)
            if not values.get(name, False):
                if event is None:
                    if len(self.regressions) >= MAX_REGRESSION_EVENTS:
                        self.truncated = True
                        continue
                    event = {
                        "subtask": name,
                        "ordered_index": index,
                        "stage": self.stages.get(name),
                        "eligible": bool(self.eligible(name)),
                        "start_step": int(step),
                        "end_step": None,
                    }
                    self.regressions.append(event)
                    self._open[name] = event
            elif event is not None:
                event["end_step"] = int(step)
                del self._open[name]

    def durable_count(self, tracker):
        """Latched predicates still true now (precondition/release are exempt)."""
        return int(
            sum(1 for name in tracker.completed if self.values.get(name, False) or not self.eligible(name))
        )

    def regressed(self, tracker):
        return [name for name in tracker.completed if self.eligible(name) and not self.values.get(name, False)]

    def vector(self):
        return np.array([self.values.get(name, False) for name in self.names], dtype=bool)


def predicate_matrix(names, vectors):
    matrix = np.zeros((len(vectors), len(names)), dtype=bool)
    for row, vector in enumerate(vectors):
        matrix[row, : len(vector)] = vector
    return matrix


def predicate_trace(names, matrix):
    """{name: [[first, last], ...]} runs of snapshot indices where it was true."""
    trace = {}
    for column, name in enumerate(names):
        runs, start = [], None
        for row, value in enumerate(matrix[:, column]):
            if value and start is None:
                start = row
            elif not value and start is not None:
                runs.append([start, row - 1])
                start = None
        if start is not None:
            runs.append([start, len(matrix) - 1])
        trace[name] = runs
    return trace


def trace_value(trace, name, index):
    return any(first <= index <= last for first, last in trace.get(name, []))


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


def _wrapper_chain(env):
    seen = set()
    node = env
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        yield node
        node = getattr(node, "env", None)


def _env_method(env, name):
    """First callable ``name`` on the wrapper chain (gymnasium does not forward)."""
    for node in _wrapper_chain(env):
        method = getattr(node, name, None)
        if callable(method):
            return method
    return None


def set_env_instruction(env, instruction):
    setter = _env_method(env, "set_task_description")
    if setter is not None:
        setter(instruction)


def live_observation(env):
    """Fresh observation after a restore, from the Gym adapter beneath wrappers.

    ``_get_obs_after_state_change`` looks only one level down, so through
    non-forwarding gymnasium wrappers it returns its fallback. Feeding the
    policy a missing or pre-restore observation would bias the first branch
    inference, so this fails loudly instead.
    """
    getter = _env_method(env, "get_current_observation")
    if getter is not None:
        observation = getter()
        if isinstance(observation, dict):
            return observation
    observation = _get_obs_after_state_change(env, fallback_obs=None)
    if not isinstance(observation, dict):
        raise RuntimeError("No live observation getter found beneath the environment wrappers")
    return observation


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
    clear = _env_method(env, "clear_task_description")
    if clear is not None:
        clear()


def env_instruction(env, obs):
    if isinstance(obs, dict):
        for key in ("annotation.human.task_description", "language"):
            value = obs.get(key)
            if isinstance(value, str) and value:
                return value
    get_ep_meta = _env_method(env, "get_ep_meta")
    if get_ep_meta is not None:
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
    get_ep_meta = _env_method(env, "get_ep_meta")
    if get_ep_meta is not None:
        try:
            meta = get_ep_meta()
        except Exception:
            meta = None
    return xml, meta


# --------------------------------------------------------------------------
# Parent rollout


def _record_latches(events, tracker, newly, step):
    for name in newly:
        events.append({"step": int(step), "subtask": name, "ordered_count": tracker.required.index(name) + 1})


def _check_required(task_name, required, subtask_eval):
    predicates = (subtask_eval or {}).get("predicates", {}) or {}
    unknown = [name for name in required if predicates and name not in predicates]
    if unknown:
        raise ValueError(f"{task_name}: required-subtask override names unknown predicates {unknown}")


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

    env_required = []

    def required_filter(names):
        env_required[:] = list(names)
        return effective_required(task_name, names, config)

    tracker = OrderedProgress(required_filter=required_filter)
    recorder = PredicateRecorder(task_name, config)
    events = []
    reset_eval = reset_info.get("subtask_eval") or get_subtask_eval(env)
    # Stages already true at reset latch at step 0 (schema 1 did not record them).
    _record_latches(events, tracker, tracker.update(reset_eval), 0)
    _check_required(task_name, tracker.required, reset_eval)
    recorder.observe(reset_eval, 0, tracker)
    snapshots = []
    vectors = []
    policy_states = []
    features = []
    feature_steps = []
    success = False
    num_steps = 0
    keep_policy_state = (
        config["policy_context"] == "restore"
        and callable(getattr(policy, "get_state", None))
    )

    def snapshot(step, is_inference):
        snap = capture_snapshot(env)
        snap.update(
            step=step,
            ordered_count=tracker.count,
            durable_count=recorder.durable_count(tracker),
            regressed=recorder.regressed(tracker),
            is_inference=is_inference,
        )
        snapshots.append(snap)
        vectors.append(recorder.vector())
        policy_states.append(policy.get_state() if keep_policy_state else None)

    for step in range(int(horizon)):
        if inference_boundary(policy, step, config["snapshot_stride"]):
            snapshot(step, True)
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
        _record_latches(events, tracker, tracker.update(current_eval), num_steps)
        recorder.observe(current_eval, num_steps, tracker)
        if _is_task_success(info, reward=reward, env=env):
            success = True
            break
        if done:
            break
    snapshot(num_steps, False)
    matrix = predicate_matrix(recorder.names, vectors)
    return {
        "success": success,
        "num_steps": num_steps,
        "instruction": instruction,
        "env_required_subtasks": list(env_required or tracker.required),
        "required_subtasks": list(tracker.required),
        "progress_events": events,
        "final_ordered_count": tracker.count,
        "final_durable_count": recorder.durable_count(tracker),
        "predicate_names": list(recorder.names),
        "predicate_stages": dict(recorder.stages),
        "predicate_first_true_step": dict(recorder.first_true),
        "predicate_values_at_reset": dict(recorder.at_reset or {}),
        "predicate_values_at_end": dict(recorder.values),
        "predicate_matrix": matrix,
        "regression_events": recorder.regressions,
        "regression_events_truncated": recorder.truncated,
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
# Stage calibration, detection and anchors


def load_stage_calibration(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("kind") != "recoverability_ladder_stage_calibration":
        raise ValueError(f"{path} is not a stage calibration written by calibrate-stages")
    payload["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["_path"] = str(path)
    return payload


def stage_thresholds(calibration, task_name, required, config):
    """Per-stage overdue thresholds in steps, None where examples are too few.

    The threshold is the ``stage_quantile`` of the calibrated stage durations
    of successful parents, floored at ``stage_overdue_min_steps``. A task whose
    calibrated ordered list differs from ``required`` gets no thresholds.
    """
    if not calibration:
        return {}, "uncalibrated"
    entry = (calibration.get("tasks") or {}).get(task_name)
    if entry is None:
        return {}, "task_missing"
    if list(entry.get("required_subtasks") or []) != list(required):
        return {}, "required_mismatch"
    thresholds = {}
    for name in required:
        durations = ((entry.get("stages") or {}).get(name) or {}).get("durations") or []
        if len(durations) >= int(config["stage_min_examples"]):
            quantile = float(np.quantile(np.asarray(durations, dtype=float), float(config["stage_quantile"])))
            thresholds[name] = max(quantile, float(config["stage_overdue_min_steps"]))
        else:
            thresholds[name] = None
    calibrated = sum(value is not None for value in thresholds.values())
    if calibrated == len(required):
        status = "calibrated"
    else:
        status = "partial" if calibrated else "fallback"
    return thresholds, status


def _stage_entry_steps(events):
    entries = {0: 0}
    for event in events:
        count = event.get("ordered_count")
        if count is not None:
            entries.setdefault(int(count), int(event["step"]))
    return entries


def _unfired(steps, terminal):
    return {"index": terminal, "fired": False, "step": int(steps[terminal]), "rule": None, "stage": None}


def _stage_overdue(steps, inference, events, horizon, config, ordered_counts, thresholds, required):
    terminal = len(steps) - 1
    entries = _stage_entry_steps(events)
    progress = sorted(event["step"] for event in events)
    stall_steps = int(math.ceil(float(config["stall_fraction"]) * horizon))
    for i in inference:
        count = int(ordered_counts[i])
        if required is not None and count >= len(required):
            continue
        stage = required[count] if required else None
        if count in entries:
            entry = entries[count]
        else:
            entry = max([value for value in progress if value <= steps[i]] or [0])
        elapsed = int(steps[i]) - int(entry)
        threshold = (thresholds or {}).get(stage)
        if threshold is not None:
            fired, rule = elapsed > threshold, "stage_quantile"
        else:
            fired, rule = elapsed >= stall_steps, "stall_fallback"
        if fired:
            return {
                "index": i,
                "fired": True,
                "step": int(steps[i]),
                "rule": rule,
                "stage": stage,
                "stage_index": count,
                "stage_entry_step": int(entry),
                "time_in_stage": elapsed,
                "threshold_steps": threshold if threshold is not None else stall_steps,
            }
    return _unfired(steps, terminal)


def _regression_detection(steps, inference, regressions, config):
    terminal = len(steps) - 1
    min_steps = int(config["regression_min_steps"])
    eligible = [event for event in regressions or [] if event.get("eligible", True)]
    for i in inference:
        step = int(steps[i])
        active = [
            event
            for event in eligible
            if event["start_step"] + min_steps <= step
            and (event.get("end_step") is None or event["end_step"] > step)
        ]
        if active:
            event = min(active, key=lambda item: (item["ordered_index"], item["start_step"]))
            return {
                "index": i,
                "fired": True,
                "step": step,
                "rule": "regression",
                "stage": event["subtask"],
                "stage_index": int(event["ordered_index"]),
                "regression_start_step": int(event["start_step"]),
            }
    return _unfired(steps, terminal)


def detector_indices(
    steps, is_inference, events, horizon, config, *, ordered_counts=None,
    regressions=None, stage_thresholds=None, required=None,
):
    """Return the first firing snapshot index of every detector.

    ``steps`` are snapshot environment steps; the last snapshot is terminal.
    A detector that never fires falls back to the terminal snapshot. The
    schema-2 detectors need ``ordered_counts`` (latched count per snapshot).
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
    if ordered_counts is not None:
        overdue = _stage_overdue(steps, inference, events, horizon, config, ordered_counts, stage_thresholds, required)
        regression = _regression_detection(steps, inference, regressions, config)
        result["stage_overdue"] = overdue
        result["regression"] = regression
        # Ties go to the regression: it names the stage that was undone.
        fired = [payload for payload in (regression, overdue) if payload["fired"]]
        result["overdue_or_regression"] = (
            dict(min(fired, key=lambda payload: payload["index"])) if fired else _unfired(steps, terminal)
        )
    return result


def compute_anchors(steps, is_inference, events, detection_index, back, *, offsets=None):
    """Anchors as snapshot indices, all at or before ``detection_index``.

    With ``offsets`` (schema 2) also ``stage_entry`` (equal to ``last_good``)
    and ``stage_entry_plus_<n>`` = ``stage_entry + n`` inferences clipped to
    detection. Snapshots are taken at every inference, so one index is one
    inference.
    """
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
    anchors = {
        "detection": int(detection_index),
        "last_good": int(last_good),
        "last_good_minus_2": int(max(0, last_good - int(back))),
    }
    if offsets is not None:
        anchors["stage_entry"] = int(last_good)
        for offset in offsets:
            if int(offset) > 0:
                anchors[f"{TIME_CURVE_PREFIX}{int(offset)}"] = int(min(last_good + int(offset), detection_index))
    return anchors


def control_detection_index(steps, is_inference, horizon, fraction):
    """Matched branch time for successful parents: last inference <= fraction*H."""
    target = float(fraction) * horizon
    candidates = [i for i in range(len(steps)) if is_inference[i] and steps[i] <= target]
    if candidates:
        return candidates[-1]
    inference = [i for i in range(len(steps)) if is_inference[i]]
    return inference[0] if inference else len(steps) - 1


def cell_plan(rung, anchor, anchors):
    """Where a cell's branch starts, which robot state it imposes, and its clock.

    Resample and reprompt branches start at their anchor with the anchor's
    clock. Robot and world resets are interventions applied at detection, so
    their budget clock is the detection time.
    """
    detection = anchors["detection"]
    index = anchors[anchor]
    if rung == "resample_continue":
        return {"start": index, "robot": None, "clock": index, "prompt": "task"}
    if rung == "subtask_reprompt":
        return {"start": index, "robot": None, "clock": index, "prompt": "reprompt"}
    if rung == "robot_reset":
        return {"start": detection, "robot": None if index == detection else index, "clock": detection, "prompt": "task"}
    if rung == "world_reset":
        return {"start": index, "robot": None, "clock": detection, "prompt": "task"}
    raise ValueError(f"Unknown rung {rung}")


def schedule_cells(cells, anchors, *, budget_fn=None, steps=None):
    """Map each requested cell to the canonical cell that is actually run.

    Two cells alias when they start from the same state, impose the same robot
    state and prompt, and get the same budget. ``budget_fn`` maps the clock
    step (``steps[clock]``, or the clock value itself without ``steps``) to a
    budget.
    """
    canonical_by_key = {}
    schedule = []
    for rung, anchor in cells:
        plan = cell_plan(rung, anchor, anchors)
        clock_step = steps[plan["clock"]] if steps is not None else plan["clock"]
        budget = None if budget_fn is None else int(budget_fn(clock_step))
        key = (plan["prompt"], plan["start"], plan["robot"], budget)
        canonical = canonical_by_key.setdefault(key, (rung, anchor))
        schedule.append(
            {
                "rung": rung,
                "anchor": anchor,
                "anchor_index": anchors[anchor],
                "start_index": plan["start"],
                "robot_index": plan["robot"],
                "clock_index": plan["clock"],
                "budget_steps": budget,
                "run": canonical == (rung, anchor),
                "canonical": list(canonical),
            }
        )
    return schedule


def stage_label(required, count):
    if not required:
        return None
    return required[count] if count < len(required) else "all_latched"


# --------------------------------------------------------------------------
# Branches


def run_branch(env, policy, *, parent, rollout, cell, k, config, task_name, budget=None):
    started = time.time()
    rung, anchor = cell["rung"], cell["anchor"]
    snapshots = rollout["snapshots"]
    detection_index = parent["anchors"]["detection"]
    anchor_index = cell["anchor_index"]
    plan = cell_plan(rung, anchor, parent["anchors"])
    start_index = cell.get("start_index", plan["start"])
    robot_index = cell.get("robot_index", plan["robot"])
    clock_index = cell.get("clock_index", plan["clock"])
    budget = int(cell.get("budget_steps") or budget)
    start = snapshots[start_index]
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
        "source_index": int(detection_index),
        "anchor_index": int(anchor_index),
        "start_index": int(start_index),
        "robot_index": None if robot_index is None else int(robot_index),
        "clock_index": int(clock_index),
        "anchor_step": int(anchor_snap["step"]),
        "anchor_distance_inferences": int(detection_index - anchor_index),
        "anchor_distance_steps": int(snapshots[detection_index]["step"] - anchor_snap["step"]),
        "inferences_since_stage_entry": int(anchor_index - parent["anchors"]["stage_entry"]),
        "inferences_before_detection": int(detection_index - anchor_index),
        "stage_at_detection": parent.get("stage_at_detection"),
        "budget_mode": config["budget_mode"],
        "budget_start_step": int(snapshots[clock_index]["step"]),
        "budget_steps": int(budget),
    }
    restore_world(env, start)
    record["source_restore_residual"] = state_residual(env, start["states"])
    if rung == "world_reset":
        record["anchor_restore_residual"] = record["source_restore_residual"]
    if robot_index is not None:
        restore_robot(env, snapshots[robot_index])
    _refresh_sim_visuals(env, num_forwards=3)
    obs = live_observation(env)

    task_instruction = rollout["instruction"] or env_instruction(env, obs)
    start_count = int(start["ordered_count"])
    tracker = OrderedProgress(rollout["required_subtasks"], count=start_count)
    start_eval = get_subtask_eval(env)
    reprompt_target = reprompt_instruction = None
    reprompt_active = False
    regressed_target = False
    if rung == "subtask_reprompt":
        regressed = list(start.get("regressed") or [])
        candidate = regressed[0] if regressed else tracker.current
        regressed_target = bool(regressed)
        reprompt_target, reprompt_instruction, kind = _resolve_recovery_target(
            start_eval,
            candidate,
            high_level_summary=None,
            recovery_level=config["reprompt_level"],
            atomic_instruction=task_instruction,
            evaluated_task_name=task_name,
        )
        record["reprompt_target_kind"] = kind
        record["reprompt_target_source"] = "regression" if regressed_target else "current_stage"
        reprompt_active = bool(reprompt_instruction) and reprompt_target not in (None, "task_success")
        record["reprompt_equivalent_to_task_prompt"] = (
            not reprompt_active or reprompt_instruction == task_instruction
        )
    record["target_subtask"] = reprompt_target if reprompt_active else tracker.current
    record["reprompt_instruction"] = reprompt_instruction if reprompt_active else None
    record["start_ordered_count"] = start_count

    policy_state = rollout["policy_states"][start_index]
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
    current_eval = start_eval
    # Detector features are only stored for parents; skip the transfer here.
    collect_features = getattr(policy, "collect_safe_features", None)
    if collect_features is not None:
        policy.collect_safe_features = False
    if reprompt_active:
        set_env_instruction(env, reprompt_instruction)
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
            if reprompt_active:
                # A regressed target is already latched: wait for it to be true again.
                done_target = _subtask_is_complete(current_eval, reprompt_target)
                if not regressed_target:
                    done_target = done_target or reprompt_target in tracker.completed
                if done_target:
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
    final_values = {
        name: bool(payload.get("value", False))
        for name, payload in ((current_eval or {}).get("predicates") or {}).items()
    }
    stages = rollout.get("predicate_stages") or {}
    durable = sum(
        1
        for name in tracker.completed
        if final_values.get(name, False) or not regression_eligible(name, stages.get(name), task_name, config)
    )
    record.update(
        {
            "success": bool(success),
            "steps_used": int(steps_used),
            "final_ordered_count": int(tracker.count),
            "final_durable_count": int(durable),
            "stages_advanced": int(tracker.count - start_count),
            "stages_advanced_vs_detection": int(tracker.count - int(snapshots[detection_index]["ordered_count"])),
            "target_completed": bool(
                record["target_subtask"] is not None
                and (
                    record["target_subtask"] in tracker.completed
                    if not regressed_target
                    else final_values.get(record["target_subtask"], False) or success
                )
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
    names = rollout.get("predicate_names") or []
    np.savez_compressed(
        snapshot_path,
        steps=np.array([s["step"] for s in snaps], dtype=np.int64),
        states=np.stack([s["states"] for s in snaps]),
        qpos=np.stack([s["qpos"] for s in snaps]),
        qvel=np.stack([s["qvel"] for s in snaps]),
        ordered_count=np.array([s["ordered_count"] for s in snaps], dtype=np.int64),
        durable_count=np.array([s.get("durable_count", s["ordered_count"]) for s in snaps], dtype=np.int64),
        is_inference=np.array([s["is_inference"] for s in snaps], dtype=bool),
        predicate_names=np.array(names, dtype=str) if names else np.zeros(0, dtype="<U1"),
        predicate_values=np.asarray(rollout.get("predicate_matrix", np.zeros((len(snaps), 0), dtype=bool)), dtype=bool),
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
    calibration = load_stage_calibration(config["stage_calibration"]) if config["stage_calibration"] else None
    runtime = runtime or Runtime(config, host=host, port=port)
    horizons = {task: runtime.task_horizon(task) for task in config["tasks"]}
    tasks = shard_tasks(config["tasks"], horizons, int(num_shards), int(shard_index))
    plan = {
        "schema_version": SCHEMA_VERSION,
        "config": {**config, "tasks": tasks},
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "horizons": {task: horizons[task] for task in tasks},
        "budget_mode": config["budget_mode"],
        "budgets": {
            task: task_budget_steps(horizons[task], config, task) if config["budget_mode"] == "fixed_fraction" else None
            for task in tasks
        },
        "stage_calibration_sha256": calibration["_sha256"] if calibration else None,
        "stage_calibration_tasks": sorted((calibration or {}).get("tasks") or {}),
        "git_commit": current_git_commit(),
        "created_at": utc_now(),
    }
    plan_path = output_dir / PLAN_NAME
    if plan_path.exists():
        if not resume:
            raise FileExistsError(f"{plan_path} exists; pass --resume or choose a new output dir")
        previous = json.loads(plan_path.read_text())
        diffs = {
            key: (previous["config"].get(key, DEFAULTS.get(key)), plan["config"].get(key))
            for key in FROZEN_PLAN_KEYS
            if previous["config"].get(key, DEFAULTS.get(key)) != plan["config"].get(key)
        }
        if previous.get("stage_calibration_sha256") != plan["stage_calibration_sha256"]:
            diffs["stage_calibration_sha256"] = (previous.get("stage_calibration_sha256"), plan["stage_calibration_sha256"])
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
    control_cells = [tuple(cell) for cell in (config["control_cells"] or config["cells"])]

    for task_name in tasks:
        horizon = horizons[task_name]
        task_parents = [p for p in parents if p["task_name"] == task_name]
        failed = sum(p["role"] == "failure" for p in task_parents)
        controls = sum(p["role"] == "success_control" for p in task_parents)
        failure_fractions = [
            p["anchor_steps"]["detection"] / horizon for p in task_parents if p["role"] == "failure"
        ]
        env = policy = None
        try:
            for attempt in range(int(config["parents_per_task"])):
                if (
                    not config["parents_only"]
                    and failed >= config["failed_parents_per_task"]
                    and controls >= config["success_controls_per_task"]
                ):
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
                        horizon=horizon, cells=cells, control_cells=control_cells, config=config,
                        output_dir=output_dir, calibration=calibration,
                        need_failure=not config["parents_only"] and failed < config["failed_parents_per_task"],
                        need_control=not config["parents_only"] and controls < config["success_controls_per_task"],
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
                    f"detection={parent['detection_rule']}@{parent['anchor_steps']['detection']} "
                    f"stage={parent['stage_at_detection']} "
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


def _collect_parent(env, policy, *, task_name, env_seed, parent_id, horizon, cells, control_cells, config, output_dir, calibration, need_failure, need_control, failure_fractions, log):
    rollout = run_parent_rollout(
        env, policy, task_name=task_name, horizon=horizon, env_seed=env_seed,
        policy_seed=stable_seed(parent_id, "parent"), config=config,
    )
    snaps = rollout["snapshots"]
    steps = [s["step"] for s in snaps]
    is_inference = [s["is_inference"] for s in snaps]
    counts = [s["ordered_count"] for s in snaps]
    required = rollout["required_subtasks"]
    events = rollout["progress_events"]
    thresholds, calibration_status = stage_thresholds(calibration, task_name, required, config)
    detectors = detector_indices(
        steps, is_inference, events, horizon, config, ordered_counts=counts,
        regressions=rollout["regression_events"], stage_thresholds=thresholds, required=required,
    )
    configured = detectors[config["detector"]]
    if rollout["success"]:
        role = "success_control" if need_control else "unbranched"
        fraction = float(np.median(failure_fractions)) if failure_fractions else float(config["time_fraction"])
        detection_index = control_detection_index(steps, is_inference, horizon, fraction)
        detection_source = "matched_failure_fraction" if failure_fractions else "time_fraction"
        detection_rule = detection_source
        detector_fired_before_success = bool(configured["fired"] and config["detector"] != "end")
        detection_payload = {}
    else:
        role = "failure" if need_failure else "unbranched"
        detection_index = configured["index"]
        detection_source = config["detector"]
        detection_rule = configured.get("rule") or (config["detector"] if configured["fired"] else "terminal_fallback")
        detector_fired_before_success = None
        detection_payload = configured
    offsets = cell_offsets(list(cells) + list(control_cells))
    anchors = compute_anchors(steps, is_inference, events, detection_index, config["anchor_back_inferences"], offsets=offsets)
    detection_step = steps[detection_index]
    det_count = int(counts[detection_index])
    latched_stage = stage_label(required, det_count)
    stage = detection_payload.get("stage") if detection_payload.get("rule") == "regression" else latched_stage
    last_progress = max([e["step"] for e in events if e["step"] <= detection_step] or [0])
    threshold = thresholds.get(latched_stage) if latched_stage in thresholds else None
    n_required = len(required)
    budget_fn = lambda clock_step: branch_budget(horizon, clock_step, config, task_name)
    matrix = rollout["predicate_matrix"]
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
        "budget_mode": config["budget_mode"],
        "budget_steps": int(budget_fn(detection_step)),
        "instruction": rollout["instruction"],
        "env_required_subtasks": rollout["env_required_subtasks"],
        "required_subtasks": required,
        "excluded_predicates": [name for name in rollout["env_required_subtasks"] if name not in required],
        "progress_events": events,
        "final_ordered_count": int(rollout["final_ordered_count"]),
        "final_durable_count": int(rollout["final_durable_count"]),
        "num_snapshots": len(snaps),
        "num_inferences": int(sum(is_inference)),
        "snapshot_steps": [int(step) for step in steps],
        "ordered_counts": [int(count) for count in counts],
        "durable_counts": [int(s["durable_count"]) for s in snaps],
        "predicate_names": rollout["predicate_names"],
        "predicate_stages": rollout["predicate_stages"],
        "predicate_first_true_step": rollout["predicate_first_true_step"],
        "predicate_values_at_reset": rollout["predicate_values_at_reset"],
        "predicate_values_at_end": rollout["predicate_values_at_end"],
        "predicate_trace": predicate_trace(rollout["predicate_names"], matrix),
        "regression_events": rollout["regression_events"],
        "regression_events_truncated": bool(rollout["regression_events_truncated"]),
        "stage_calibration_status": calibration_status,
        "stage_thresholds": thresholds,
        "detectors": detectors,
        "detector": config["detector"],
        "detection_source": detection_source,
        "detection_rule": detection_rule,
        "detector_fired_on_success": detector_fired_before_success,
        "latched_stage_at_detection": latched_stage,
        "stage_at_detection": stage,
        "regressed_at_detection": list(snaps[detection_index].get("regressed") or []),
        "anchors": anchors,
        "anchor_steps": {name: int(steps[index]) for name, index in anchors.items()},
        "anchor_ordered_counts": {name: int(counts[index]) for name, index in anchors.items()},
        "anchor_durable_counts": {name: int(snaps[index]["durable_count"]) for name, index in anchors.items()},
        "scores_at_detection": {
            "time": detection_step / horizon,
            "stall": (detection_step - last_progress) / horizon,
            "ordered_progress": det_count / n_required if n_required else float(rollout["success"]),
            "durable_progress": (
                snaps[detection_index]["durable_count"] / n_required if n_required else float(rollout["success"])
            ),
            "stage_overdue_ratio": (detection_step - last_progress) / threshold if threshold else None,
        },
        "seed_control": rollout["seed_control"],
        "wall_seconds": rollout["wall_seconds"],
        **_save_parent_artifacts(output_dir, parent_id, rollout, config),
    }
    role_cells = control_cells if role == "success_control" else cells
    schedule = schedule_cells(role_cells, anchors, budget_fn=budget_fn, steps=steps) if role != "unbranched" else []
    for cell in schedule:
        cell["anchor_step"] = int(steps[cell["anchor_index"]])
        cell["inferences_since_stage_entry"] = int(cell["anchor_index"] - anchors["stage_entry"])
        cell["inferences_before_detection"] = int(anchors["detection"] - cell["anchor_index"])
    parent["cells"] = schedule
    branch_steps = 0
    branch_seconds = 0.0
    for cell in schedule:
        if not cell["run"]:
            continue
        for k in range(int(config["k_branches"])):
            record = run_branch(env, policy, parent=parent, rollout=rollout, cell=cell, k=k, config=config, task_name=task_name)
            append_jsonl(output_dir / BRANCHES_NAME, record)
            branch_steps += record["steps_used"]
            branch_seconds += record["wall_seconds"]
    parent["branch_steps"] = int(branch_steps)
    parent["branch_wall_seconds"] = branch_seconds
    parent["created_at"] = utc_now()
    return parent


# --------------------------------------------------------------------------
# Loading runs (schema 1 and 2)


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


def normalize_parent(parent):
    """Fill schema-2 fields that a schema-1 parent row can provide."""
    row = dict(parent)
    row["schema_version"] = int(row.get("schema_version", 1))
    anchors = dict(row.get("anchors") or {})
    anchors.setdefault("stage_entry", anchors.get("last_good", anchors.get("detection", 0)))
    row["anchors"] = anchors
    steps = dict(row.get("anchor_steps") or {})
    steps.setdefault("stage_entry", steps.get("last_good", steps.get("detection", 0)))
    row["anchor_steps"] = steps
    required = list(row.get("required_subtasks") or [])
    row.setdefault("env_required_subtasks", required)
    count = (row.get("anchor_ordered_counts") or {}).get("detection")
    latched = stage_label(required, int(count)) if count is not None else None
    row.setdefault("latched_stage_at_detection", latched)
    row.setdefault("stage_at_detection", latched)
    row.setdefault("detection_rule", row.get("detection_source"))
    row.setdefault("budget_mode", "fixed_fraction")
    row.setdefault("anchor_durable_counts", {})
    return row


# --------------------------------------------------------------------------
# Stage calibration from completed runs


def _latch_steps_from_events(parent, required):
    by_count = {}
    for event in parent.get("progress_events") or []:
        by_count.setdefault(int(event["ordered_count"]), int(event["step"]))
    latches = [by_count.get(count) for count in range(1, len(required) + 1)]
    # Schema 1 did not record stages that were already true at reset.
    leading = 0
    while leading < len(latches) and latches[leading] is None:
        leading += 1
    if any(value is None for value in latches[leading:]):
        return None
    if leading and parent.get("schema_version", 1) >= 2:
        return None
    return [0] * leading + latches[leading:]


def _latch_steps_from_trace(parent, required):
    trace = parent.get("predicate_trace")
    steps = parent.get("snapshot_steps")
    if trace is None or not steps:
        return None
    if any(name not in trace for name in required):
        return None
    latches = [None] * len(required)
    count = 0
    for index, step in enumerate(steps):
        while count < len(required) and trace_value(trace, required[count], index):
            latches[count] = int(step)
            count += 1
    if parent.get("success"):
        for position in range(count, len(required)):
            latches[position] = int(parent["num_steps"])
    return latches if all(value is not None for value in latches) else None


def success_latch_steps(parent, required):
    """Latch step of every stage of ``required`` in a successful parent.

    Uses the recorded progress events when the parent ran with the same
    ordered list; otherwise recomputes the ordered latches at snapshot
    resolution from the parent's predicate trace (schema 2 only).
    """
    if list(parent.get("required_subtasks") or []) == list(required):
        latches = _latch_steps_from_events(parent, required)
        return (latches, "progress_events") if latches is not None else (None, "incomplete_progress_events")
    latches = _latch_steps_from_trace(parent, required)
    if latches is not None:
        return latches, "recomputed_from_predicate_trace"
    return None, "required_mismatch_without_predicate_trace"


def build_stage_calibration(run_dirs, config=None):
    config = resolve_config(config or {})
    parents, _ = load_run(run_dirs)
    tasks = {}
    for parent in sorted(parents, key=lambda p: (p["task_name"], p.get("environment_seed", 0))):
        if not parent.get("success"):
            continue
        task = parent["task_name"]
        env_required = parent.get("env_required_subtasks") or parent.get("required_subtasks") or []
        required = effective_required(task, env_required, config)
        entry = tasks.setdefault(
            task,
            {
                "required_subtasks": required,
                "successful_parents": 0,
                "used_parents": 0,
                "skipped_parents": defaultdict(int),
                "sources": defaultdict(int),
                "stages": {name: {"index": i, "durations": []} for i, name in enumerate(required)},
            },
        )
        if entry["required_subtasks"] != required:
            entry["skipped_parents"]["inconsistent_env_required_subtasks"] += 1
            continue
        entry["successful_parents"] += 1
        latches, source = success_latch_steps(parent, required)
        if latches is None:
            entry["skipped_parents"][source] += 1
            continue
        entry["used_parents"] += 1
        entry["sources"][source] += 1
        previous = 0
        for name, step in zip(required, latches):
            entry["stages"][name]["durations"].append(int(step) - int(previous))
            previous = step
    quantile = float(config["stage_quantile"])
    for entry in tasks.values():
        entry["skipped_parents"] = dict(entry["skipped_parents"])
        entry["sources"] = dict(entry["sources"])
        for stage in entry["stages"].values():
            durations = np.asarray(stage["durations"], dtype=float)
            stage["n"] = int(durations.size)
            stage["median"] = float(np.median(durations)) if durations.size else None
            stage["max"] = float(durations.max()) if durations.size else None
            stage[f"quantile_{quantile:g}"] = float(np.quantile(durations, quantile)) if durations.size else None
            stage["calibrated"] = bool(durations.size >= int(config["stage_min_examples"]))
    return {
        "kind": "recoverability_ladder_stage_calibration",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "git_commit": current_git_commit(),
        "source_runs": [str(Path(run_dir)) for run_dir in run_dirs],
        "required_subtask_overrides": config["required_subtask_overrides"],
        "exclude_predicates": config["exclude_predicates"],
        "quantile_hint": quantile,
        "min_examples_hint": int(config["stage_min_examples"]),
        "tasks": tasks,
    }


# --------------------------------------------------------------------------
# Predicate audit


def _fraction(flags):
    flags = [bool(flag) for flag in flags]
    return (sum(flags) / len(flags)) if flags else None


def _success_blocker(parent):
    """First ordered stage that latched only at the success step."""
    required = parent.get("required_subtasks") or []
    latches = {}
    for event in parent.get("progress_events") or []:
        latches.setdefault(int(event["ordered_count"]), int(event["step"]))
    for count, name in enumerate(required, start=1):
        if latches.get(count) == int(parent["num_steps"]):
            return name
    return None


def _regressed(parent, name, min_steps):
    for event in parent.get("regression_events") or []:
        if event.get("subtask") != name:
            continue
        end = event.get("end_step")
        end = int(parent["num_steps"]) if end is None else int(end)
        if end - int(event["start_step"]) >= int(min_steps):
            return True
    return False


def audit_predicates(run_dirs, config=None, *, unreached_threshold=0.25, reset_threshold=0.5, transient_threshold=0.25):
    """Flag required ordered predicates that cannot work as ladder stages.

    ``true_at_reset``: true in the reset state of most parents (raw traces).
    ``unreached_before_success``: in successful parents, first true only at
    the success step (raw traces), or the first stage to latch at the success
    step (progress events; schema 1 too), for a stage before the task's final
    non-release stage. ``regresses_after_latch``: false again for at least
    ``regression_min_steps`` after latching; ``transient_in_successes`` when it
    happens in successful parents too, so it need not persist for success.
    """
    config = resolve_config(config or {})
    min_steps = int(config["regression_min_steps"])
    parents, _ = load_run(run_dirs)
    by_task = defaultdict(list)
    for parent in parents:
        by_task[parent["task_name"]].append(parent)
    report = {
        "kind": "recoverability_ladder_predicate_audit",
        "created_at": utc_now(),
        "source_runs": [str(Path(run_dir)) for run_dir in run_dirs],
        "thresholds": {
            "unreached_before_success": unreached_threshold,
            "true_at_reset": reset_threshold,
            "transient_in_successes": transient_threshold,
            "regression_min_steps": min_steps,
        },
        "tasks": {},
        "suggested_config": {"exclude_predicates": {}, "regression_ignore_predicates": {}},
    }
    for task, rows in sorted(by_task.items()):
        env_required = next((list(p.get("env_required_subtasks") or p.get("required_subtasks") or []) for p in rows if p.get("env_required_subtasks") or p.get("required_subtasks")), [])
        effective = effective_required(task, env_required, config)
        successes = [p for p in rows if p["success"]]
        failures = [p for p in rows if not p["success"]]
        raw = [p for p in rows if p.get("predicate_first_true_step") is not None]
        raw_successes = [p for p in raw if p["success"]]
        raw_failures = [p for p in raw if not p["success"]]
        stages = {}
        for parent in raw:
            stages.update(parent.get("predicate_stages") or {})
        non_release = [i for i, name in enumerate(env_required) if not is_release_predicate(name, stages.get(name))]
        completion_index = non_release[-1] if non_release else len(env_required) - 1
        blockers = Counter(_success_blocker(p) for p in successes)
        predicates = []
        for index, name in enumerate(env_required):
            stage = stages.get(name)
            with_name = [p for p in successes if name in (p.get("required_subtasks") or [])]
            first_true = lambda p: p["predicate_first_true_step"].get(name)
            row = {
                "predicate": name,
                "index": index,
                "stage": stage,
                "in_ladder_required": name in effective,
                "true_at_reset": _fraction((p.get("predicate_values_at_reset") or {}).get(name, False) for p in raw),
                "unreached_before_success": _fraction(
                    first_true(p) is None or first_true(p) >= p["num_steps"] for p in raw_successes
                ),
                "never_true": _fraction(p["predicate_first_true_step"].get(name) is None for p in raw),
                "success_blocker": (blockers.get(name, 0) / len(with_name)) if with_name else None,
                "regressed_in_successes": _fraction(_regressed(p, name, min_steps) for p in raw_successes),
                "regressed_in_failures": _fraction(_regressed(p, name, min_steps) for p in raw_failures),
                "flags": [],
            }
            if row["true_at_reset"] is not None and row["true_at_reset"] >= reset_threshold:
                row["flags"].append("true_at_reset")
            unreached = max(value for value in (row["unreached_before_success"], row["success_blocker"], 0.0) if value is not None)
            if index < completion_index and unreached >= unreached_threshold:
                row["flags"].append("unreached_before_success")
                if name in effective:
                    report["suggested_config"]["exclude_predicates"].setdefault(task, []).append(name)
            if (row["regressed_in_successes"] or 0) > 0 or (row["regressed_in_failures"] or 0) > 0:
                row["flags"].append("regresses_after_latch")
            if (row["regressed_in_successes"] or 0) >= transient_threshold:
                row["flags"].append("transient_in_successes")
                if regression_eligible(name, stage, task, config):
                    report["suggested_config"]["regression_ignore_predicates"].setdefault(task, []).append(name)
            predicates.append(row)
        report["tasks"][task] = {
            "parents": len(rows),
            "successes": len(successes),
            "failures": len(failures),
            "parents_with_predicate_traces": len(raw),
            "mode": "raw_and_latch" if raw else "latch_only",
            "env_required_subtasks": env_required,
            "ladder_required_subtasks": effective,
            "completion_predicate": env_required[completion_index] if env_required else None,
            "failed_with_every_stage_latched": sum(
                1 for p in failures if p.get("required_subtasks") and p["final_ordered_count"] >= len(p["required_subtasks"])
            ),
            "failed_with_durable_below_latched": sum(
                1 for p in failures if p.get("final_durable_count") is not None and p["final_durable_count"] < p["final_ordered_count"]
            ),
            "predicates": predicates,
        }
    return report


def write_audit(report, output_dir):
    output_dir = Path(output_dir)
    atomic_write_json(output_dir / "predicate_audit.json", report)
    lines = [
        "# Predicate audit",
        "",
        "Fractions are over parents (`unreached`, `blocker`: successful parents).",
        "`unreached`: first raw true only at the success step; `blocker`: first ordered stage latched only at the success step.",
        "",
    ]
    for task, entry in report["tasks"].items():
        lines += [
            f"## {task}",
            "",
            f"Parents {entry['parents']} (successes {entry['successes']}), traces {entry['parents_with_predicate_traces']} ({entry['mode']}). "
            f"Failed with every stage latched: {entry['failed_with_every_stage_latched']}; durable below latched at the end: {entry['failed_with_durable_below_latched']}.",
            f"Ladder ordered list: {', '.join(entry['ladder_required_subtasks']) or '-'}",
            "",
            "| predicate | stage | in ladder | true@reset | unreached | blocker | regress S/F | flags |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
        for row in entry["predicates"]:
            lines.append(
                f"| {row['predicate']} | {row['stage'] or '-'} | {'yes' if row['in_ladder_required'] else 'no'} | "
                f"{_fmt(row['true_at_reset'])} | {_fmt(row['unreached_before_success'])} | {_fmt(row['success_blocker'])} | "
                f"{_fmt(row['regressed_in_successes'])}/{_fmt(row['regressed_in_failures'])} | {', '.join(row['flags']) or '-'} |"
            )
        lines.append("")
    lines += ["## Suggested ladder config", "", "```json", json.dumps(report["suggested_config"], indent=2, sort_keys=True), "```", ""]
    (output_dir / "predicate_audit.md").write_text("\n".join(lines))
    return output_dir / "predicate_audit.md"


# --------------------------------------------------------------------------
# Planning / dry run


def reference_fractions(run_dirs):
    """Median failed-parent detection and stage-entry fractions per task."""
    parents, _ = load_run(run_dirs)
    by_task = defaultdict(list)
    for parent in parents:
        by_task[parent["task_name"]].append(normalize_parent(parent))
    result = {}
    for task, rows in by_task.items():
        failed = [p for p in rows if not p["success"]]
        if not failed:
            continue
        result[task] = {
            "detection_fraction": float(np.median([p["anchor_steps"]["detection"] / p["horizon"] for p in failed])),
            "stage_entry_fraction": float(np.median([p["anchor_steps"]["stage_entry"] / p["horizon"] for p in failed])),
            "failed_parents": len(failed),
            "base_success_rate": float(np.mean([p["success"] for p in rows])),
        }
    return result


def _plan_role_steps(cells, anchors, budget_fn, k, p_success, success_fraction):
    schedule = schedule_cells(cells, anchors, budget_fn=budget_fn)
    per_branch = (1 - p_success) + p_success * success_fraction
    return {
        "upper_steps": k * sum(cell["budget_steps"] for cell in schedule),
        "expected_steps": k * per_branch * sum(cell["budget_steps"] for cell in schedule if cell["run"]),
        "cells": len(schedule),
        "run_cells": sum(cell["run"] for cell in schedule),
    }


def build_plan(config, *, calibration=None, horizons=None, reference=None):
    """Simulator-free cost estimate.

    ``calibration`` is a status.json dict (measured s/step); ``reference`` is
    ``reference_fractions`` of an earlier run (anchor positions).
    """
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
    cells = [tuple(cell) for cell in config["cells"]]
    control_cells = [tuple(cell) for cell in (config["control_cells"] or config["cells"])]
    spacing = int((config.get("policy_args") or {}).get("replan_steps") or 16)
    back = int(config["anchor_back_inferences"])
    offsets = cell_offsets(cells + control_cells)
    success_fraction = float(config["assumed_success_budget_fraction"])
    rows = []
    warnings = []
    for task in config["tasks"]:
        horizon = int(horizons[task])
        rate = config["expected_success_rates"].get(task)
        if rate is None:
            warnings.append(f"{task}: no expected success rate; assuming 0.5")
            rate = 0.5
        rate = float(rate)
        if not 0.2 <= rate <= 0.8:
            warnings.append(f"{task}: expected success rate {rate:.2f} is outside 20-80%")
        source = "config"
        f_det = task_value(config["expected_detection_fraction"], task)
        f_entry = task_value(config["expected_stage_entry_fraction"], task)
        if reference and task in reference:
            f_det = reference[task]["detection_fraction"]
            f_entry = reference[task]["stage_entry_fraction"]
            source = "reference_run"
        f_det = float(DEFAULTS["expected_detection_fraction"] if f_det is None else f_det)
        f_entry = min(float(DEFAULTS["expected_stage_entry_fraction"] if f_entry is None else f_entry), f_det)
        fail_target = int(config["failed_parents_per_task"])
        control_target = int(config["success_controls_per_task"])
        if config["parents_only"]:
            parents = int(config["parents_per_task"])
            failures = controls = 0.0
        else:
            needed = max(
                math.ceil(fail_target / max(1e-6, 1 - rate)),
                math.ceil(control_target / max(1e-6, rate)),
            )
            parents = min(int(config["parents_per_task"]), needed)
            failures = min(fail_target, parents * (1 - rate))
            controls = min(control_target, parents * rate)
        successes = parents * rate
        parent_steps = parents * (1 - rate) * horizon + successes * float(config["success_step_fraction"]) * horizon
        detection = f_det * horizon
        entry = f_entry * horizon
        anchors = {
            "detection": detection,
            "last_good": entry,
            "stage_entry": entry,
            "last_good_minus_2": max(0.0, entry - back * spacing),
        }
        for offset in offsets:
            anchors[f"{TIME_CURVE_PREFIX}{offset}"] = min(entry + offset * spacing, detection)
        budget_fn = lambda start, horizon=horizon, task=task: branch_budget(horizon, start, config, task)
        failure_cost = _plan_role_steps(cells, anchors, budget_fn, k, float(config["assumed_branch_success_rate"]), success_fraction)
        control_cost = _plan_role_steps(control_cells, anchors, budget_fn, k, float(config["assumed_control_success_rate"]), success_fraction)
        branch_steps_upper = failures * failure_cost["upper_steps"] + controls * control_cost["upper_steps"]
        branch_steps_expected = failures * failure_cost["expected_steps"] + controls * control_cost["expected_steps"]
        n_inference = horizon / float(spacing)
        storage = parents * (
            n_inference * float(config["state_dim_estimate"]) * 8 * 3
            + 2e6
            + (n_inference * float(config["feature_bytes_per_inference"]) if config["record_policy_features"] else 0)
        )
        expected_hours = ((parent_steps + branch_steps_expected) * seconds_per_step + parents * float(config["seconds_per_reset"])) / 3600.0
        rows.append(
            {
                "task": task,
                "horizon": horizon,
                "budget_mode": config["budget_mode"],
                "budget_steps": int(budget_fn(detection)),
                "expected_success_rate": rate,
                "expected_detection_fraction": round(f_det, 3),
                "expected_stage_entry_fraction": round(f_entry, 3),
                "anchor_source": source,
                "expected_parents": parents,
                "expected_failed_parents": round(failures, 2),
                "expected_success_controls": round(controls, 2),
                "branches_upper_bound": int(round((failures * failure_cost["cells"] + controls * control_cost["cells"]) * k)),
                "branches_expected": int(round((failures * failure_cost["run_cells"] + controls * control_cost["run_cells"]) * k)),
                "parent_env_steps": int(parent_steps),
                "branch_env_steps_upper": int(branch_steps_upper),
                "branch_env_steps_expected": int(branch_steps_expected),
                "worker_hours_expected": round(expected_hours, 2),
                "storage_bytes_estimate": int(storage),
            }
        )
    reset_seconds = sum(row["expected_parents"] for row in rows) * float(config["seconds_per_reset"])
    workers = max(1, int(config["num_workers"]))
    usable = float(config["allocation_hours"]) - float(config["allocation_overhead_hours"])
    totals = {}
    for kind in ("upper", "expected"):
        steps = sum(row["parent_env_steps"] + row[f"branch_env_steps_{kind}"] for row in rows)
        worker_hours = (steps * seconds_per_step + reset_seconds) / 3600.0
        wall = worker_hours / workers
        allocations = max(1, math.ceil(wall / usable)) if usable > 0 else None
        totals[kind] = {
            "env_steps": int(steps),
            "worker_hours": round(worker_hours, 2),
            "gpu_hours": round(worker_hours * int(config["gpus_per_worker"]), 2),
            "wall_hours_with_workers": round(wall, 2),
            "allocations_needed": allocations,
            "fits_single_allocation": allocations == 1,
            "fits_with_one_resume": allocations is not None and allocations <= 2,
        }
    return {
        "dry_run": True,
        "tasks": config["tasks"],
        "cells": config["cells"],
        "control_cells": config["control_cells"] or config["cells"],
        "k_branches": k,
        "budget_mode": config["budget_mode"],
        "parents_only": config["parents_only"],
        "seconds_per_env_step": seconds_per_step,
        "throughput_source": throughput_source,
        "gpus_per_worker": int(config["gpus_per_worker"]),
        "num_workers": workers,
        "allocation_hours": float(config["allocation_hours"]),
        "usable_hours_per_allocation": usable,
        "per_task": rows,
        "totals": totals,
        "storage_gib_estimate": round(sum(row["storage_bytes_estimate"] for row in rows) / 2**30, 2),
        "warnings": warnings,
        "notes": [
            "upper: every scheduled branch runs its full budget and no cell is aliased",
            "expected: branches succeed at the assumed rates, using assumed_success_budget_fraction of their budget",
            "anchors are placed at expected_detection_fraction/expected_stage_entry_fraction of H unless --reference-run is given",
            "allocations_needed counts 24 h allocations minus allocation_overhead_hours for server start-up",
            "calibrate seconds_per_env_step from a smoke run with --calibration-dir",
        ],
    }


# --------------------------------------------------------------------------
# Analysis


SINCE_ENTRY_BINS = (("0", 0, 0), ("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16), (">16", 17, math.inf))
BEFORE_DETECTION_BINS = (("0", 0, 0), ("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16), ("17-32", 17, 32), (">32", 33, math.inf))
RUNG_LABELS = {"resample_continue": "R", "subtask_reprompt": "P", "robot_reset": "RR", "world_reset": "WR"}


def cell_label(rung, anchor):
    offset = anchor_offset(anchor)
    if offset is not None:
        short = "e" if offset == 0 else f"e+{offset}"
    else:
        short = {"detection": "d", "last_good": "lg", "last_good_minus_2": "lg-2"}.get(anchor, anchor)
    return f"{RUNG_LABELS.get(rung, rung)}@{short}"


def parent_cell_estimates(parents, branches):
    """P(success) etc. per (parent, rung, anchor), resolving aliased cells."""
    grouped = defaultdict(list)
    for branch in branches:
        grouped[(branch["parent_id"], branch["rung"], branch["anchor"])].append(branch)
    rows = []
    for parent in parents:
        parent = normalize_parent(parent)
        anchors = parent["anchors"]
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
                    "anchor_distance_inferences": anchors["detection"] - cell["anchor_index"],
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
                    "schema_version": parent["schema_version"],
                    "stage_at_detection": str(parent.get("stage_at_detection")),
                    "detection_rule": parent.get("detection_rule"),
                    "anchor_index": int(cell["anchor_index"]),
                    "inferences_since_stage_entry": int(cell["anchor_index"] - anchors["stage_entry"]),
                    "inferences_before_detection": int(anchors["detection"] - cell["anchor_index"]),
                    "budget_mode": parent["budget_mode"],
                    "budget_steps": float(np.mean([b["budget_steps"] for b in outcomes])),
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


def _paired_contrasts(estimates, cells, base_key, strata, n_boot, seed):
    contrasts = []
    for rung, anchor in cells:
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
    return contrasts


def _time_curves(rows, strata, n_boot, seed):
    unique = {}
    for row in rows:
        if row["role"] == "failure" and row["rung"] in ("resample_continue", "subtask_reprompt"):
            unique.setdefault((row["parent_id"], row["rung"], row["anchor_index"]), row)
    curves = []
    axes = (
        ("since_stage_entry", "inferences_since_stage_entry", SINCE_ENTRY_BINS),
        ("before_detection", "inferences_before_detection", BEFORE_DETECTION_BINS),
    )
    for axis, field, bins in axes:
        for rung in ("resample_continue", "subtask_reprompt"):
            rung_rows = [row for row in unique.values() if row["rung"] == rung]
            for task in ["ALL"] + sorted({row["task_name"] for row in rung_rows}):
                for label, low_edge, high_edge in bins:
                    per_parent = defaultdict(list)
                    budgets = []
                    for row in rung_rows:
                        if task in ("ALL", row["task_name"]) and low_edge <= row[field] <= high_edge:
                            per_parent[row["parent_id"]].append(row["p_success"])
                            budgets.append(row["budget_steps"])
                    values = {pid: float(np.mean(v)) for pid, v in per_parent.items()}
                    point, low, high = cluster_bootstrap(values, strata=strata if task == "ALL" else None, n_boot=n_boot, seed=seed)
                    if point is not None:
                        curves.append({"axis": axis, "rung": rung, "task": task, "bin": label, "parents": len(values), "p_success": point, "ci_low": low, "ci_high": high, "mean_budget_steps": float(np.mean(budgets))})
    return curves


def _detector_health(parents):
    rows = []
    tasks = sorted({p["task_name"] for p in parents})
    for task in ["ALL"] + tasks:
        selected = [p for p in parents if task in ("ALL", p["task_name"])]
        successes = [p for p in selected if p["success"]]
        failures = [p for p in selected if p["role"] == "failure"]
        row = {"task": task, "successful_parents": len(successes), "failed_parents": len(failures)}
        for name in DETECTORS:
            if name == "end":
                continue
            have = [p for p in successes if name in (p.get("detectors") or {})]
            row[f"{name}_fired_before_success"] = (
                f"{sum(bool(p['detectors'][name]['fired']) for p in have)}/{len(have)}" if have else None
            )
        row["failures_progress_after_detection"] = sum(
            1 for p in failures if any(e["step"] > p["anchor_steps"]["detection"] for e in p.get("progress_events") or [])
        )
        row["failures_every_stage_latched"] = sum(
            1 for p in failures if p.get("required_subtasks") and p["final_ordered_count"] >= len(p["required_subtasks"])
        )
        row["detection_rules"] = dict(sorted(Counter(str(p.get("detection_rule")) for p in failures).items()))
        rows.append(row)
    return rows


def _progress_at_detection(parents):
    rows = []
    failures = [p for p in parents if p["role"] == "failure" and p.get("required_subtasks")]
    for task in ["ALL"] + sorted({p["task_name"] for p in failures}):
        selected = [p for p in failures if task in ("ALL", p["task_name"])]
        latched = [p["anchor_ordered_counts"]["detection"] / len(p["required_subtasks"]) for p in selected]
        durable_rows = [p for p in selected if "detection" in (p.get("anchor_durable_counts") or {})]
        durable = [p["anchor_durable_counts"]["detection"] / len(p["required_subtasks"]) for p in durable_rows]
        rows.append(
            {
                "task": task,
                "failed_parents": len(selected),
                "mean_latched_progress": float(np.mean(latched)) if latched else None,
                "mean_durable_progress": float(np.mean(durable)) if durable else None,
                "parents_with_durable_record": len(durable_rows),
                "durable_below_latched": sum(
                    p["anchor_durable_counts"]["detection"] < p["anchor_ordered_counts"]["detection"] for p in durable_rows
                ),
            }
        )
    return rows


def analyze(run_dirs, *, scores=None, threshold=0.5, n_boot=2000, seed=0):
    raw_parents, branches = load_run(run_dirs)
    parents = [normalize_parent(p) for p in raw_parents]
    rows = parent_cell_estimates(parents, branches)
    strata = {p["parent_id"]: p["task_name"] for p in parents}
    by_parent = {p["parent_id"]: p for p in parents}

    ladder = []
    for role in ("failure", "success_control"):
        role_rows = [r for r in rows if r["role"] == role]
        cells = sorted({(r["rung"], r["anchor"]) for r in role_rows}, key=cell_sort_key)
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
    failure_rows = [r for r in rows if r["role"] == "failure"]
    estimates = {(r["parent_id"], r["rung"], r["anchor"]): r["p_success"] for r in failure_rows}
    base_key = ("resample_continue", "detection")
    failure_cells = sorted({(r["rung"], r["anchor"]) for r in failure_rows}, key=cell_sort_key)
    contrasts = _paired_contrasts(estimates, failure_cells, base_key, strata, n_boot, seed)

    # Ladder and paired contrasts per (task, stage at detection).
    stage_ladder, stage_contrasts = [], []
    for task, stage in sorted({(r["task_name"], r["stage_at_detection"]) for r in failure_rows}):
        group = [r for r in failure_rows if (r["task_name"], r["stage_at_detection"]) == (task, stage)]
        group_cells = sorted({(r["rung"], r["anchor"]) for r in group}, key=cell_sort_key)
        for rung, anchor in group_cells:
            values = {r["parent_id"]: r["p_success"] for r in group if (r["rung"], r["anchor"]) == (rung, anchor)}
            point, low, high = cluster_bootstrap(values, n_boot=n_boot, seed=seed)
            stage_ladder.append({"task": task, "stage": stage, "rung": rung, "anchor": anchor, "parents": len(values), "p_success": point, "ci_low": low, "ci_high": high})
        group_estimates = {key: value for key, value in estimates.items() if by_parent[key[0]]["task_name"] == task and str(by_parent[key[0]]["stage_at_detection"]) == stage}
        for contrast in _paired_contrasts(group_estimates, group_cells, base_key, None, n_boot, seed):
            stage_contrasts.append({"task": task, "stage": stage, **contrast})

    # Per-parent attribution along the ladder (exploratory: K is small).
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

    # Schema-1 view: world resets by anchor distance, resample at detection as 0.
    decay = []
    decay_rows = [r for r in failure_rows if r["rung"] == "world_reset" or (r["rung"], r["anchor"]) == base_key]
    for bucket in ("0", "1-2", "3-5", "6-10", ">10"):
        values = {}
        for r in decay_rows:
            if _distance_bucket(r["anchor_distance_inferences"]) == bucket:
                values.setdefault(r["parent_id"], []).append(r["p_success"])
        values = {pid: float(np.mean(v)) for pid, v in values.items()}
        point, low, high = cluster_bootstrap(values, strata=strata, n_boot=n_boot, seed=seed)
        if point is not None:
            decay.append({"distance_inferences": bucket, "parents": len(values), "p_success": point, "ci_low": low, "ci_high": high})

    time_curve = _time_curves(rows, strata, n_boot, seed)

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
    builtin = ["time", "stall", "ordered_progress"]
    builtin += [name for name in ("durable_progress", "stage_overdue_ratio") if any(name in by_parent[pid]["scores_at_detection"] for pid in attribution)]
    score_names = builtin + sorted({n for d in external.values() for n, _ in d})
    for score_name in score_names:
        for target_name, target_fn in targets.items():
            pairs = {}
            for pid in attribution:
                parent = by_parent[pid]
                if score_name in builtin:
                    score = parent["scores_at_detection"].get(score_name)
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
        "schema_versions": sorted({p["schema_version"] for p in parents}),
        "budget_modes": sorted({p["budget_mode"] for p in parents if p["role"] != "unbranched"}),
        "parents_by_role": {role: sum(p["role"] == role for p in parents) for role in ("failure", "success_control", "unbranched")},
        "base_success_rate_by_task": {
            task: float(np.mean([p["success"] for p in parents if p["task_name"] == task]))
            for task in sorted({p["task_name"] for p in parents})
        },
        "attribution_threshold": threshold,
        "attribution_label": "exploratory",
        "restore_residual_max": float(max(residuals)) if residuals else None,
        "restore_residual_median": float(np.median(residuals)) if residuals else None,
        "seed_control": sorted({str(b.get("seed_control")) for b in branches}),
        "detector_health": _detector_health(parents),
        "progress_at_detection": _progress_at_detection(parents),
        "ladder": ladder,
        "contrasts_vs_resample": contrasts,
        "stage_ladder": stage_ladder,
        "stage_contrasts": stage_contrasts,
        "time_curve": time_curve,
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
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


def _ci(row, key="estimate"):
    return f"{_fmt(row[key])} | {_fmt(row['ci_low'])}-{_fmt(row['ci_high'])}"


def write_analysis(summary, rows, output_dir):
    output_dir = Path(output_dir)
    atomic_write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "parent_cells.csv", rows)
    _write_csv(output_dir / "ladder.csv", summary["ladder"])
    _write_csv(output_dir / "stage_ladder.csv", summary["stage_ladder"])
    _write_csv(output_dir / "stage_contrasts.csv", summary["stage_contrasts"])
    _write_csv(output_dir / "time_curve.csv", summary["time_curve"])
    _write_csv(output_dir / "detector_health.csv", summary["detector_health"])
    _write_csv(output_dir / "decay.csv", summary["decay"])
    _write_csv(output_dir / "score_correlations.csv", summary["score_correlations"])
    lines = [
        "# Recoverability ladder",
        "",
        f"Parents: {summary['num_parents']} {summary['parents_by_role']}; branches: {summary['num_branches']}.",
        f"Schema versions: {summary['schema_versions']}; budget modes: {summary['budget_modes']}.",
        f"Restore residual max/median: {_fmt(summary['restore_residual_max'])} / {_fmt(summary['restore_residual_median'])} (recorded, not gated).",
        f"Seed control: {', '.join(str(s) for s in summary['seed_control'])}.",
    ]
    if len(summary["schema_versions"]) > 1 or len(summary["budget_modes"]) > 1:
        lines.append("**Warning:** runs with different schemas or budget modes are pooled; analyze them separately for claims.")
    lines += [
        "",
        "## Detector health",
        "",
        "Fired before success: successful parents on which each detector fired before the task succeeded (false alarms).",
        "",
        "| task | successes | stall | stage_overdue | regression | overdue_or_regression | failures | progress after detection | every stage latched | detection rules |",
        "|---|---:|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary["detector_health"]:
        lines.append(
            f"| {row['task']} | {row['successful_parents']} | {row['stall_fired_before_success'] or '-'} | "
            f"{row['stage_overdue_fired_before_success'] or '-'} | {row['regression_fired_before_success'] or '-'} | "
            f"{row['overdue_or_regression_fired_before_success'] or '-'} | {row['failed_parents']} | "
            f"{row['failures_progress_after_detection']} | {row['failures_every_stage_latched']} | "
            f"{', '.join(f'{k}: {v}' for k, v in row['detection_rules'].items()) or '-'} |"
        )
    lines += [
        "",
        "## Progress at detection (failed parents)",
        "",
        "Latched: ordered stages ever completed. Durable: latched stages still true at detection (precondition/release exempt).",
        "",
        "| task | parents | latched | durable | durable < latched |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["progress_at_detection"]:
        lines.append(
            f"| {row['task']} | {row['failed_parents']} | {_fmt(row['mean_latched_progress'])} | "
            f"{_fmt(row['mean_durable_progress'])} | {row['durable_below_latched']}/{row['parents_with_durable_record']} |"
        )
    lines += ["", "## P(success) by cell (failed parents, all tasks; parent-cluster bootstrap 95% CI)", "", "| rung | anchor | parents | P(success) | 95% CI |", "|---|---|---:|---:|---|"]
    for row in summary["ladder"]:
        if row["role"] == "failure" and row["task"] == "ALL" and row["metric"] == "p_success":
            lines.append(f"| {row['rung']} | {row['anchor']} | {row['parents']} | {_ci(row)} |")
    lines += ["", "## Harm control (successful parents)", "", "| rung | anchor | parents | P(success) | 95% CI |", "|---|---|---:|---:|---|"]
    for row in summary["ladder"]:
        if row["role"] == "success_control" and row["task"] == "ALL" and row["metric"] == "p_success":
            lines.append(f"| {row['rung']} | {row['anchor']} | {row['parents']} | {_ci(row)} |")
    lines += ["", "## Paired contrasts vs resample at detection (failed parents, all tasks)", "", "| rung | anchor | parents | difference | 95% CI |", "|---|---|---:|---:|---|"]
    for row in summary["contrasts_vs_resample"]:
        lines.append(f"| {row['rung']} | {row['anchor']} | {row['parents']} | {_ci(row)} |")
    lines += ["", "## Ladder by task and stage at detection (failed parents)", "", "| task | stage | cell | parents | P(success) | 95% CI | vs R@d | 95% CI |", "|---|---|---|---:|---:|---|---:|---|"]
    contrast_index = {(c["task"], c["stage"], c["rung"], c["anchor"]): c for c in summary["stage_contrasts"]}
    for row in summary["stage_ladder"]:
        contrast = contrast_index.get((row["task"], row["stage"], row["rung"], row["anchor"]))
        paired = _ci(contrast) if contrast else "- | -"
        lines.append(f"| {row['task']} | {row['stage']} | {cell_label(row['rung'], row['anchor'])} | {row['parents']} | {_ci(row, 'p_success')} | {paired} |")
    for axis, title in (("since_stage_entry", "inferences since stage entry"), ("before_detection", "inferences before detection")):
        lines += ["", f"## Recoverability over time: P(success) vs {title} (failed parents)", "", "| rung | task | bin | parents | P(success) | 95% CI | mean budget |", "|---|---|---|---:|---:|---|---:|"]
        for row in summary["time_curve"]:
            if row["axis"] == axis:
                lines.append(f"| {RUNG_LABELS[row['rung']]} | {row['task']} | {row['bin']} | {row['parents']} | {_ci(row, 'p_success')} | {row['mean_budget_steps']:.0f} |")
    failure_rows = [r for r in rows if r["role"] == "failure"]
    columns = sorted({(r["rung"], r["anchor"]) for r in failure_rows}, key=cell_sort_key)
    lines += [
        "",
        "## Per-parent cells (failed parents; P(success) over K branches)",
        "",
        "R resample, P subtask reprompt, RR robot reset, WR world reset; @d detection, @e stage entry, @e+n n inferences later (clipped to detection), @lg last good.",
        "",
        "| task | parent | stage | rule | e→d | " + " | ".join(cell_label(*c) for c in columns) + " |",
        "|---|---|---|---|---:|" + "---:|" * len(columns),
    ]
    by_parent = defaultdict(dict)
    for row in failure_rows:
        by_parent[row["parent_id"]][(row["rung"], row["anchor"])] = row
    for pid, cells in sorted(by_parent.items(), key=lambda item: (next(iter(item[1].values()))["task_name"], item[0])):
        first = next(iter(cells.values()))
        span = next((r["inferences_before_detection"] + r["inferences_since_stage_entry"] for r in cells.values()), 0)
        values = " | ".join(f"{cells[c]['p_success']:.2f}" if c in cells else "-" for c in columns)
        lines.append(f"| {first['task_name']} | {pid[:8]} | {first['stage_at_detection']} | {first['detection_rule']} | {span} | {values} |")
    lines += ["", "## Decay with anchor distance (world reset; 0 = resample at detection)", "", "| inferences back | parents | P(success) | 95% CI |", "|---|---:|---:|---|"]
    for row in summary["decay"]:
        lines.append(f"| {row['distance_inferences']} | {row['parents']} | {_ci(row, 'p_success')} |")
    lines += ["", "## Score at detection vs recoverability (Spearman)", "", "| score | target | parents | rho | 95% CI |", "|---|---|---:|---:|---|"]
    for row in summary["score_correlations"]:
        lines.append(f"| {row['score']} | {row['target']} | {row['parents']} | {_ci(row, 'spearman')} |")
    lines += [
        "",
        f"## Failure attribution (exploratory; first rung with P >= {summary['attribution_threshold']})",
        "",
        "With K branches per cell this class count is noisy; prefer the per-parent and per-stage tables above.",
        "",
        "| class | parents | fraction | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for row in summary["attribution"]:
        lines.append(f"| {row['class']} | {row['parents']} | {_ci(row, 'fraction')} |")
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
    config = json.loads(Path(args.config).read_text()) if getattr(args, "config", None) else {}
    overrides = {}
    for key in ("tasks", "k_branches", "parents_per_task", "failed_parents_per_task", "success_controls_per_task", "base_seed", "budget_fraction", "budget_mode", "max_budget_steps", "detector", "policy_context", "num_workers", "stage_calibration"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = str(value) if isinstance(value, Path) else value
    if getattr(args, "parents_only", False):
        overrides["parents_only"] = True
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
        p.add_argument("--parents-only", action="store_true", help="natural parents only, no branches")
        p.add_argument("--base-seed", type=int)
        p.add_argument("--budget-mode", choices=BUDGET_MODES)
        p.add_argument("--budget-fraction", type=float)
        p.add_argument("--max-budget-steps", type=int)
        p.add_argument("--detector", choices=DETECTORS)
        p.add_argument("--policy-context", choices=("reset", "restore"))
        p.add_argument("--num-workers", type=int)

    plan = sub.add_parser("plan", help="dry run: validate config and estimate GPU hours")
    common(plan)
    plan.add_argument("--calibration-dir", type=Path, help="run dir whose status.json gives measured s/step")
    plan.add_argument("--reference-run", nargs="+", type=Path, help="run dirs whose failed parents place the anchors")
    plan.add_argument("--output", type=Path)

    collect = sub.add_parser("collect", help="run parents and ladder branches")
    common(collect)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--stage-calibration", type=Path, help="calibrate-stages JSON for the stage-overdue detector")
    collect.add_argument("--policy-arg", action="append", default=[])
    collect.add_argument("--host")
    collect.add_argument("--port", type=int)
    collect.add_argument("--shard-index", type=int, default=0)
    collect.add_argument("--num-shards", type=int, default=1)
    collect.add_argument("--resume", action="store_true")

    audit = sub.add_parser("audit-predicates", help="flag required predicates that cannot work as ladder stages")
    audit.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    audit.add_argument("--config", type=Path, help="ladder config (overrides, exclusions, regression rules)")
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--unreached-threshold", type=float, default=0.25)
    audit.add_argument("--reset-threshold", type=float, default=0.5)
    audit.add_argument("--transient-threshold", type=float, default=0.25)

    calibrate = sub.add_parser("calibrate-stages", help="stage durations of successful parents for the stage-overdue detector")
    calibrate.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    calibrate.add_argument("--config", type=Path, help="ladder config whose overrides/exclusions define the stages")
    calibrate.add_argument("--output", type=Path, required=True)

    analysis = sub.add_parser("analyze", help="ladder tables, time curves, decay and score correlation")
    analysis.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--scores-jsonl", type=Path, help="rows: parent_id, snapshot_index, score[, name]")
    analysis.add_argument("--threshold", type=float, default=0.5)
    analysis.add_argument("--n-boot", type=int, default=2000)
    return parser


def _calibration_summary(calibration):
    lines = []
    for task, entry in sorted(calibration["tasks"].items()):
        stages = ", ".join(
            f"{name}: n={stage['n']} median={_fmt(stage['median'])}{'' if stage['calibrated'] else ' (stall fallback)'}"
            for name, stage in entry["stages"].items()
        )
        skipped = f" skipped {entry['skipped_parents']}" if entry["skipped_parents"] else ""
        lines.append(f"{task}: used {entry['used_parents']}/{entry['successful_parents']} successes{skipped} | {stages}")
    return "\n".join(lines)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        config = _load_config(args)
        calibration = None
        if args.calibration_dir:
            calibration = json.loads((args.calibration_dir / STATUS_NAME).read_text())
        reference = reference_fractions(args.reference_run) if args.reference_run else None
        plan = build_plan(config, calibration=calibration, reference=reference)
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
    if args.command == "audit-predicates":
        config = json.loads(args.config.read_text()) if args.config else {}
        report = audit_predicates(
            args.run_dirs, config, unreached_threshold=args.unreached_threshold,
            reset_threshold=args.reset_threshold, transient_threshold=args.transient_threshold,
        )
        path = write_audit(report, args.output_dir)
        print(path.read_text())
        return report
    if args.command == "calibrate-stages":
        config = json.loads(args.config.read_text()) if args.config else {}
        calibration = build_stage_calibration(args.run_dirs, config)
        atomic_write_json(args.output, calibration)
        print(_calibration_summary(calibration))
        print(f"wrote {args.output}")
        return calibration
    scores = read_jsonl(args.scores_jsonl) if args.scores_jsonl else None
    summary, rows = analyze(args.run_dirs, scores=scores, threshold=args.threshold, n_boot=args.n_boot)
    path = write_analysis(summary, rows, args.output_dir)
    print(path.read_text())
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
