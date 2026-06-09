"""Evaluate RLDX recovery while reusing the official RLDX rollout path.

This script is intentionally separate from ``evaluate_recovery_benchmark.py``.
The generic recovery benchmark calls a policy adapter as ``policy(obs)``; that
is fragile for RLDX because the official evaluator owns the MultiStepWrapper,
temporal video history, action chunk execution, and policy memory options.

Here the high-level rollout follows the official RLDX loop:

    actions, _ = policy.get_action(observations, options=options)
    observations, rewards, terminations, truncations, infos = env.step(actions)

Recovery hooks are inserted around that loop for a single environment
(``n_envs=1``): track ordered subtask progress, restore the requested state
after high-level failure, change the language instruction to the failed subtask,
then continue stepping through the same official RLDX policy/env path.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import random
import sys
import traceback
import uuid

import numpy as np

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None):
        return text

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def parse_policy_args(values):
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Policy args must be key=value, got {value!r}")
        key, raw = value.split("=", 1)
        parsed[key] = raw
    return parsed


def resolve_tasks(task_set, envs):
    from robocasa.utils.dataset_registry import TARGET_TASKS

    if envs:
        return envs
    if task_set == "all_composite":
        return TARGET_TASKS["composite_seen"] + TARGET_TASKS["composite_unseen"]
    if task_set == "all_target":
        return (
            TARGET_TASKS["atomic_seen"]
            + TARGET_TASKS["composite_seen"]
            + TARGET_TASKS["composite_unseen"]
        )
    return list(TARGET_TASKS[task_set])


def summarize_results(results):
    valid = [r for r in results if "error" not in r]
    attempted = [r for r in valid if r.get("recovery_attempted")]
    high_level_successes = [
        r for r in valid if (r.get("high_level") or {}).get("success")
    ]
    recovery_successes = [
        r for r in attempted if (r.get("subtask") or {}).get("success")
    ]
    return {
        "num_rollouts": len(valid),
        "high_level_success_count": len(high_level_successes),
        "high_level_success_rate": len(high_level_successes) / len(valid)
        if valid
        else 0.0,
        "recovery_attempt_count": len(attempted),
        "recovery_subtask_success_count": len(recovery_successes),
        "recovery_subtask_success_rate": len(recovery_successes) / len(attempted)
        if attempted
        else 0.0,
    }


def _load_official_rldx_rollout_module(rldx_repo):
    if rldx_repo:
        repo = Path(rldx_repo).expanduser().resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
    return importlib.import_module("rldx.eval.rollout_policy")


def _make_wrapper_configs(rollout_policy, policy, max_episode_steps, n_action_steps):
    modality_configs = policy.get_modality_config()
    video_delta_indices = np.array(modality_configs["video"].delta_indices)
    state_delta_indices = (
        np.array(modality_configs["state"].delta_indices)
        if "state" in modality_configs
        else None
    )
    print(f"Using video_delta_indices: {video_delta_indices}")
    if state_delta_indices is not None:
        print(f"Using state_delta_indices: {state_delta_indices}")

    return rollout_policy.WrapperConfigs(
        video=rollout_policy.VideoConfig(
            video_dir=None,
            max_episode_steps=max_episode_steps,
            n_action_steps=n_action_steps,
        ),
        multistep=rollout_policy.MultiStepConfig(
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=False,
        ),
    )


def _single_env_from_vector(vec_env):
    envs = getattr(vec_env, "envs", None)
    if not envs or len(envs) != 1:
        raise RuntimeError("RLDX recovery evaluator requires a SyncVectorEnv with one env")
    return envs[0]


def _unbatch(value):
    if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
        return value[0]
    if isinstance(value, dict):
        return {k: _unbatch(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _env_info_for_single_env(infos):
    if not isinstance(infos, dict):
        return infos or {}
    out = {}
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
            out[key] = value[0]
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            out[key] = value[0]
        else:
            out[key] = value
    return out


def _get_info_value(info, key):
    if not isinstance(info, dict) or key not in info:
        return None
    value = info[key]
    if isinstance(value, np.ndarray):
        if value.shape[:1] == (1,):
            return value[0]
        if value.dtype == object and value.size:
            return value.reshape(-1)[0]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _bool_from_vector(value):
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    if isinstance(value, (list, tuple)):
        return bool(any(_bool_from_vector(v) for v in value))
    return bool(value)


def _scalar_from_vector(value):
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def _set_instruction_in_observation(observation, instruction):
    if instruction is None:
        return observation
    obs = copy.deepcopy(observation)
    keys = (
        "annotation.human.task_description",
        "annotation.human.action.task_description",
        "language",
        "task",
    )
    for key in keys:
        if key in obs:
            value = obs[key]
            if isinstance(value, np.ndarray):
                obs[key] = np.asarray([[instruction]], dtype=object)
            elif isinstance(value, list):
                obs[key] = [[instruction]]
            else:
                obs[key] = instruction
    return obs


def _as_batched_language(value, batch_size):
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        text = str(arr.item())
        return [[text] for _ in range(batch_size)]
    if arr.ndim == 1:
        return [[str(v)] for v in arr.tolist()]
    return arr.tolist()


def _normalize_video_array(value, history_len):
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = arr[None, None, ...]
    elif arr.ndim == 4:
        arr = arr[None, ...]
    if arr.ndim == 5 and arr.shape[1] == 1 and history_len > 1:
        arr = np.repeat(arr, history_len, axis=1)
    return arr


def _normalize_state_array(value):
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr[None, None, ...]
    elif arr.ndim == 2:
        arr = arr[:, None, ...]
    return arr


def _normalize_rldx_observation(observation, video_history=4):
    """Convert official flat RoboCasa obs keys to the nested RLDX server shape.

    This does not change the rollout protocol: observations still come from the
    official RLDX MultiStepWrapper and actions still go straight into the
    official env.step(actions). It only mirrors flat keys such as
    ``video.robot0_agentview_left`` into ``observation["video"][...]`` for RLDX
    server builds that require nested modality groups.
    """
    if not isinstance(observation, dict):
        return observation

    obs = dict(observation)
    video = {}
    state = {}
    language = {}

    if isinstance(observation.get("video"), dict):
        for key, value in observation["video"].items():
            video[key] = _normalize_video_array(value, video_history)
    if isinstance(observation.get("state"), dict):
        for key, value in observation["state"].items():
            state[key] = _normalize_state_array(value)
    if isinstance(observation.get("language"), dict):
        language.update(observation["language"])

    for key, value in observation.items():
        if key.startswith("video."):
            short_key = key.split(".", 1)[1]
            video[short_key] = _normalize_video_array(value, video_history)
        elif key.startswith("state."):
            short_key = key.split(".", 1)[1]
            state[short_key] = _normalize_state_array(value)

    video_aliases = {
        "robot0_agentview_left": "res256_image_side_0",
        "robot0_agentview_right": "res256_image_side_1",
        "robot0_eye_in_hand": "res256_image_wrist_0",
    }
    for src, dst in video_aliases.items():
        if src in video and dst not in video:
            video[dst] = video[src]
        if dst in video and src not in video:
            video[src] = video[dst]

    batch_size = 1
    for value in video.values():
        arr = np.asarray(value)
        if arr.ndim:
            batch_size = int(arr.shape[0])
            break

    language_keys = (
        "annotation.human.task_description",
        "annotation.human.action.task_description",
        "language.instruction",
        "language.task_description",
        "language",
        "task",
    )
    for key in language_keys:
        if key in observation:
            short_key = key.split(".", 1)[1] if key.startswith("language.") else key
            language[short_key] = _as_batched_language(observation[key], batch_size)

    if "annotation.human.task_description" in observation:
        task_desc = _as_batched_language(
            observation["annotation.human.task_description"], batch_size
        )
        language.setdefault("instruction", task_desc)
        language.setdefault("task_description", task_desc)
        language.setdefault("annotation.human.task_description", task_desc)
        language.setdefault("annotation.human.action.task_description", task_desc)
        obs.setdefault("annotation.human.action.task_description", task_desc)
        obs["annotation"] = {
            "human": {
                "task_description": task_desc,
                "action": {"task_description": task_desc},
            }
        }

    if video:
        obs["video"] = video
        for key, value in video.items():
            obs[f"video.{key}"] = value
    if state:
        obs["state"] = state
        for key, value in state.items():
            obs[f"state.{key}"] = value
    if language:
        obs["language"] = language
    return obs


def _batched_action_like(value, batch_size, shape, fill_value=0.0):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.shape[:1] != (batch_size,):
        arr = np.broadcast_to(arr, (batch_size,) + tuple(arr.shape)).copy()
    expected_shape = (batch_size,) + tuple(shape)
    if arr.ndim == len(expected_shape) and arr.shape[0] == batch_size:
        slices = [slice(None)]
        can_slice = True
        for actual, expected in zip(arr.shape[1:], expected_shape[1:]):
            if actual < expected:
                can_slice = False
                break
            slices.append(slice(0, expected))
        if can_slice:
            arr = arr[tuple(slices)]
    if arr.shape != expected_shape:
        arr = np.broadcast_to(arr, expected_shape).astype(np.float32).copy()
    if fill_value:
        arr[...] = fill_value
    return arr


def _zero_action_for_space(space, batch_size):
    return np.zeros((batch_size,) + tuple(space.shape), dtype=np.float32)


def _one_action_for_space(space, batch_size):
    return np.ones((batch_size,) + tuple(space.shape), dtype=np.float32)


def _normalize_actions_for_env(actions, vec_env):
    """Map RLDX action aliases to the official RoboCasa Gym action keys."""
    if not isinstance(actions, dict):
        return actions

    action_space = getattr(vec_env, "single_action_space", None)
    if action_space is None:
        action_space = getattr(vec_env, "action_space", None)
    required_spaces = getattr(action_space, "spaces", None)
    if not required_spaces:
        return actions

    normalized = dict(actions)
    batch_size = int(getattr(vec_env, "num_envs", 1))
    aliases = {
        "action.end_effector_position": (
            "action.end_effector_position",
            "end_effector_position",
            "action.eef_pos_delta",
            "eef_pos_delta",
        ),
        "action.end_effector_rotation": (
            "action.end_effector_rotation",
            "end_effector_rotation",
            "action.eef_rot_delta",
            "eef_rot_delta",
        ),
        "action.gripper_close": (
            "action.gripper_close",
            "gripper_close",
        ),
        "action.base_motion": (
            "action.base_motion",
            "base_motion",
        ),
        "action.control_mode": (
            "action.control_mode",
            "control_mode",
        ),
    }

    for required_key, key_aliases in aliases.items():
        if required_key in normalized or required_key not in required_spaces:
            continue
        for alias in key_aliases:
            if alias in normalized:
                normalized[required_key] = normalized[alias]
                break

    if (
        "action.gripper_close" in required_spaces
        and "action.gripper_close" not in normalized
        and "action.gripper" in normalized
    ):
        normalized["action.gripper_close"] = 1.0 - np.asarray(
            normalized["action.gripper"], dtype=np.float32
        )

    for required_key, space in required_spaces.items():
        if required_key in normalized:
            normalized[required_key] = _batched_action_like(
                normalized[required_key], batch_size, space.shape
            )
            continue
        if required_key == "action.base_motion":
            normalized[required_key] = _zero_action_for_space(space, batch_size)
        elif required_key == "action.control_mode":
            normalized[required_key] = _one_action_for_space(space, batch_size)

    return normalized


def _set_env_instruction(env, instruction):
    if instruction is None:
        return
    setter = getattr(env, "set_task_description", None)
    if setter is not None:
        setter(instruction)
        return
    getter = getattr(env, "get_wrapper_attr", None)
    if getter is not None:
        try:
            getter("set_task_description")(instruction)
        except Exception:
            pass


def _refresh_multistep_observation(env, instruction=None):
    """Refresh the official wrapped observation after a simulator state change.

    RLDX's official MultiStepWrapper owns the temporal observation stacks. The
    wrapper API is not guaranteed to expose a public reset-to-current-state
    method, so try the known non-destructive observation paths and fail loudly if
    none are available.
    """
    candidates = (
        "get_current_observation",
        "_get_observation",
        "_get_obs",
        "get_observation",
    )
    for name in candidates:
        method = getattr(env, name, None)
        if method is None and hasattr(env, "get_wrapper_attr"):
            try:
                method = env.get_wrapper_attr(name)
            except Exception:
                method = None
        if method is None:
            continue
        try:
            obs = method()
        except TypeError:
            continue
        if obs is not None:
            return _set_instruction_in_observation(obs, instruction)
    raise RuntimeError(
        "Could not refresh official RLDX MultiStepWrapper observation after "
        "state recovery. Inspect the cluster's MultiStepWrapper for a current "
        "observation method and add it to _refresh_multistep_observation()."
    )


def _step_official(
    vec_env,
    policy,
    observations,
    is_first_step,
    session_id,
    video_history=4,
):
    options = {
        "reset_memory": [bool(is_first_step)],
        "session_ids": [session_id],
    }
    policy_observations = _normalize_rldx_observation(
        observations, video_history=video_history
    )
    actions, _ = policy.get_action(policy_observations, options=options)
    actions = _normalize_actions_for_env(actions, vec_env)
    next_obs, rewards, terminations, truncations, infos = vec_env.step(actions)
    info = _env_info_for_single_env(infos)
    done = _bool_from_vector(terminations) or _bool_from_vector(truncations)
    reward = _scalar_from_vector(rewards)
    success = False
    if "success" in info:
        success = _bool_from_vector(info["success"])
    if "final_info" in infos and infos["final_info"][0] is not None:
        final_info = infos["final_info"][0]
        if "success" in final_info:
            success = success or _bool_from_vector(final_info["success"])
    return next_obs, reward, done, info, success


def _latest_ordered_trace_entry(subtask_evals):
    from robocasa.recovery.subtask_eval import build_subtask_trace

    trace = build_subtask_trace(subtask_evals)
    for entry in reversed(trace):
        if entry.get("subtask_eval_available"):
            return entry
    return None


def _subtask_is_complete(subtask_eval, subtask_name):
    if subtask_name is None:
        return False
    predicate = (subtask_eval or {}).get("predicates", {}).get(subtask_name, {})
    return bool(predicate.get("value", False))


def _subtask_instruction(subtask_eval, subtask_name):
    if subtask_name is None:
        return None
    predicate = (subtask_eval or {}).get("predicates", {}).get(subtask_name, {})
    return predicate.get("description") or f"Complete this subtask: {subtask_name}."


def _ordered_current_subtask_from_eval(subtask_eval):
    entry = _latest_ordered_trace_entry([subtask_eval])
    if not entry:
        return None
    return entry.get("ordered_current_subtask") or entry.get("current_subtask_estimate")


def _safe_get_subtask_eval(env, warnings, context):
    from robocasa.recovery.subtask_eval import get_subtask_eval

    try:
        return get_subtask_eval(env)
    except Exception as exc:
        warnings.append(
            {
                "context": context,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
        return None


def _same_env_state(state):
    if state is None:
        return None
    state = copy.deepcopy(state)
    state.pop("model", None)
    state.pop("ep_meta", None)
    return state


def _direct_restore_same_sim(env, state):
    if state is None:
        return False, "no_state"
    sim = getattr(getattr(env, "unwrapped", env), "sim", None)
    if sim is None and hasattr(env, "get_wrapper_attr"):
        try:
            sim = env.get_wrapper_attr("sim")
        except Exception:
            sim = None
    if sim is None:
        return False, "no_sim"

    qpos = state.get("_sim_state_qpos")
    qvel = state.get("_sim_state_qvel")
    if qpos is not None and qvel is not None:
        current = sim.get_state()
        if len(qpos) != len(current.qpos) or len(qvel) != len(current.qvel):
            return False, (
                "state_dim_mismatch:"
                f" qpos {len(qpos)}->{len(current.qpos)},"
                f" qvel {len(qvel)}->{len(current.qvel)}"
            )
        try:
            current.qpos[:] = qpos
            current.qvel[:] = qvel
            sim.set_state(current)
        except Exception:
            if not hasattr(current, "_replace"):
                raise
            sim.set_state(current._replace(qpos=qpos, qvel=qvel))
        sim.forward()
        return True, None

    flat_state = state.get("states")
    if flat_state is None:
        return False, "no_sim_state"
    try:
        sim.set_state_from_flattened(flat_state)
    except ValueError as exc:
        return False, f"state_dim_mismatch: {exc}"
    sim.forward()
    return True, None


def _apply_recovery_mode_same_env(env, mode, last_good_state):
    from robocasa.recovery.recovery_rollout import (
        RecoveryMode,
        normalize_recovery_mode,
        _reset_to_state,
        _restore_robot_state_only,
    )

    mode = normalize_recovery_mode(mode)
    if mode == RecoveryMode.CONTINUE_FROM_FAILURE:
        return {"mode": mode.value, "state_restored": False}
    if last_good_state is None:
        return {"mode": mode.value, "state_restored": False, "reason": "no_state"}
    if mode == RecoveryMode.ENV_TO_LAST_GOOD:
        restored, reason = _direct_restore_same_sim(env, last_good_state)
        if restored:
            return {
                "mode": mode.value,
                "state_restored": True,
                "same_env_direct_sim_state": True,
            }
        try:
            _reset_to_state(env, _same_env_state(last_good_state))
            return {
                "mode": mode.value,
                "state_restored": True,
                "same_env_state_only": True,
            }
        except ValueError as exc:
            return {
                "mode": mode.value,
                "state_restored": False,
                "reason": reason or str(exc),
            }
    if mode == RecoveryMode.EEF_TO_LAST_GOOD:
        _restore_robot_state_only(env, last_good_state)
        return {"mode": mode.value, "state_restored": True, "robot_only": True}
    raise ValueError(f"Unhandled recovery mode: {mode}")


def run_one_official_rldx_recovery_rollout(
    env_name,
    policy,
    rollout_policy,
    wrapper_configs,
    mode,
    split,
    seed,
    high_level_horizon,
    subtask_horizon,
    match_recovery_horizon_to_no_progress,
    stuck_patience,
    include_trace,
    n_action_steps,
):
    import gymnasium as gym

    from robocasa.recovery.recovery_rollout import (
        _capture_state,
    )
    from robocasa.recovery.subtask_eval import (
        summarize_subtask_rollout,
    )

    env_fn = lambda: rollout_policy.create_eval_env(
        env_name=f"robocasa/{env_name}",
        env_idx=0,
        total_n_envs=1,
        wrapper_configs=wrapper_configs,
        start_episode_id=0,
        seed=seed,
        robocasa_split=split,
    )
    vec_env = gym.vector.SyncVectorEnv([env_fn])
    single_env = None
    video_history = int(len(wrapper_configs.multistep.video_delta_indices))
    try:
        observations, _ = vec_env.reset()
        single_env = _single_env_from_vector(vec_env)
        if hasattr(policy, "reset"):
            policy.reset()
        session_id = f"{env_name}_env0_{uuid.uuid4().hex[:8]}"
        is_first_step = True
        subtask_eval_warnings = []

        subtask_evals = [
            _safe_get_subtask_eval(single_env, subtask_eval_warnings, "initial")
        ]
        initial_trace_entry = _latest_ordered_trace_entry(subtask_evals) or {}
        best_ordered_count = len(
            initial_trace_entry.get("ordered_completed_subtasks", [])
        )
        last_good_subtask = (
            initial_trace_entry.get("ordered_completed_subtasks", [])[-1]
            if best_ordered_count
            else None
        )
        last_good_state = _capture_state(single_env)
        last_good_step = 0
        high_level_success = False
        high_level_steps = 0

        for step_i in range(high_level_horizon):
            high_level_steps = step_i + 1
            observations, reward, done, info, step_success = _step_official(
                vec_env,
                policy,
                observations,
                is_first_step=is_first_step,
                session_id=session_id,
                video_history=video_history,
            )
            is_first_step = False
            if step_success or reward > 0:
                high_level_success = True
                break
            current_eval = _get_info_value(info, "subtask_eval")
            if current_eval is None:
                current_eval = _safe_get_subtask_eval(
                    single_env, subtask_eval_warnings, f"high_level_step_{step_i}"
                )
            subtask_evals.append(current_eval)

            trace_entry = _latest_ordered_trace_entry(subtask_evals) or {}
            ordered_completed = trace_entry.get("ordered_completed_subtasks", [])
            if len(ordered_completed) > best_ordered_count:
                best_ordered_count = len(ordered_completed)
                last_good_subtask = ordered_completed[-1] if ordered_completed else None
                last_good_state = _capture_state(single_env)
                last_good_step = high_level_steps

            if done:
                break

        high_level_summary = summarize_subtask_rollout(
            subtask_evals,
            stuck_patience=stuck_patience,
            include_trace=include_trace,
        )
        high_level_summary["success"] = high_level_success
        high_level_summary["num_policy_steps"] = high_level_steps
        high_level_summary["num_action_steps_estimate"] = high_level_steps * n_action_steps
        high_level_summary["last_good_ordered_subtask"] = last_good_subtask
        high_level_summary["last_good_ordered_subtask_count"] = best_ordered_count
        high_level_summary["last_good_policy_step"] = last_good_step
        high_level_summary["policy_steps_since_last_good_progress"] = (
            high_level_steps - last_good_step
        )
        high_level_summary["subtask_eval_warnings"] = subtask_eval_warnings

        result = {
            "high_level": high_level_summary,
            "recovery_attempted": False,
            "recovery": None,
            "subtask": None,
        }
        if high_level_success:
            return result

        final_eval = high_level_summary.get("final_subtask_eval")
        high_level_subtask_name = (
            high_level_summary.get("ordered_current_subtask")
            or high_level_summary.get("current_subtask_estimate")
            or high_level_summary.get("stuck_subtask")
        )
        high_level_instruction = _subtask_instruction(
            final_eval, high_level_subtask_name
        )

        recovery_meta = _apply_recovery_mode_same_env(
            single_env, mode, last_good_state
        )
        recovery_start_eval = _safe_get_subtask_eval(
            single_env, subtask_eval_warnings, "recovery_start"
        )
        subtask_name = (
            _ordered_current_subtask_from_eval(recovery_start_eval)
            or high_level_subtask_name
        )
        instruction = _subtask_instruction(recovery_start_eval, subtask_name)
        recovery_meta["high_level_target_subtask"] = high_level_subtask_name
        recovery_meta["high_level_target_instruction"] = high_level_instruction
        recovery_meta["recovery_start_target_subtask"] = subtask_name
        recovery_meta["recovery_start_target_instruction"] = instruction
        _set_env_instruction(single_env, instruction)
        observations = _refresh_multistep_observation(single_env, instruction)
        is_first_step = True
        session_id = f"{env_name}_recovery_{uuid.uuid4().hex[:8]}"

        retry_evals = [recovery_start_eval]
        retry_success = _subtask_is_complete(retry_evals[-1], subtask_name)
        retry_steps = 0
        retry_horizon = (
            max(1, high_level_steps - last_good_step)
            if match_recovery_horizon_to_no_progress
            else subtask_horizon
        )
        for step_i in range(retry_horizon):
            if retry_success:
                break
            retry_steps = step_i + 1
            observations, reward, done, info, _ = _step_official(
                vec_env,
                policy,
                observations,
                is_first_step=is_first_step,
                session_id=session_id,
                video_history=video_history,
            )
            is_first_step = False
            current_eval = _get_info_value(info, "subtask_eval")
            if current_eval is None:
                current_eval = _safe_get_subtask_eval(
                    single_env, subtask_eval_warnings, f"recovery_step_{step_i}"
                )
            retry_evals.append(current_eval)
            retry_success = _subtask_is_complete(current_eval, subtask_name)
            if done:
                break

        retry_summary = summarize_subtask_rollout(
            retry_evals,
            stuck_patience=stuck_patience,
            include_trace=include_trace,
        )
        retry_summary["success"] = retry_success
        retry_summary["num_policy_steps"] = retry_steps
        retry_summary["num_action_steps_estimate"] = retry_steps * n_action_steps
        retry_summary["horizon"] = retry_horizon
        retry_summary["horizon_unit"] = "policy_steps"
        retry_summary["horizon_source"] = (
            "policy_steps_since_last_good_progress"
            if match_recovery_horizon_to_no_progress
            else "fixed_subtask_horizon"
        )
        retry_summary["target_subtask"] = subtask_name
        retry_summary["target_instruction"] = instruction
        result.update(
            {
                "recovery_attempted": True,
                "recovery": recovery_meta,
                "subtask": retry_summary,
            }
        )
        return result
    finally:
        try:
            vec_env.close()
        except Exception:
            if single_env is not None:
                single_env.close()


def run_benchmark(args):
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    rollout_policy = _load_official_rldx_rollout_module(args.rldx_repo)
    policy = rollout_policy.create_rldx_sim_policy(
        model_path=args.model_path,
        embodiment_tag=rollout_policy.EmbodimentTag.GENERAL_EMBODIMENT,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
    )
    wrapper_configs = _make_wrapper_configs(
        rollout_policy,
        policy,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
    )
    tasks = resolve_tasks(args.task_set, args.envs)
    output = {
        "config": vars(args),
        "tasks": tasks,
        "high_level_horizon_by_task": {
            task: (
                args.high_level_horizon
                if args.high_level_horizon is not None
                else get_task_horizon(task)
            )
            for task in tasks
        },
        "modes": {},
        "uses_official_rldx_rollout_path": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def update_mode_output(mode, mode_results, completed=False):
        output["modes"][mode] = {
            "summary": summarize_results(mode_results),
            "num_errors": sum(1 for result in mode_results if "error" in result),
            "rollouts": mode_results,
            "completed": bool(completed),
        }

    def write_output(partial=True):
        output["partial"] = bool(partial)
        with args.output.open("w") as f:
            json.dump(output, f, indent=2, default=json_default)

    for mode in args.modes:
        mode_results = []
        update_mode_output(mode, mode_results, completed=False)
        write_output(partial=True)
        print(colored(f"Evaluating official RLDX recovery mode: {mode}", "green"))
        for task_name in tasks:
            print(colored(f"  Task: {task_name}", "cyan"))
            for rollout_i in tqdm(range(args.num_rollouts), leave=False):
                seed = args.seed + rollout_i
                try:
                    random.seed(seed)
                    np.random.seed(seed)
                    high_level_horizon = (
                        args.high_level_horizon
                        if args.high_level_horizon is not None
                        else get_task_horizon(task_name)
                    )
                    # The official MultiStepWrapper counts policy steps, each
                    # executing n_action_steps raw env steps.
                    high_level_policy_horizon = max(
                        1, int(np.ceil(high_level_horizon / args.n_action_steps))
                    )
                    result = run_one_official_rldx_recovery_rollout(
                        env_name=task_name,
                        policy=policy,
                        rollout_policy=rollout_policy,
                        wrapper_configs=wrapper_configs,
                        mode=mode,
                        split=args.split,
                        seed=seed,
                        high_level_horizon=high_level_policy_horizon,
                        subtask_horizon=args.subtask_horizon,
                        match_recovery_horizon_to_no_progress=(
                            args.match_recovery_horizon_to_no_progress
                        ),
                        stuck_patience=args.stuck_patience,
                        include_trace=args.include_trace,
                        n_action_steps=args.n_action_steps,
                    )
                    result["task"] = task_name
                    result["rollout_index"] = rollout_i
                    result["seed"] = seed
                    result["mode"] = mode
                    result["resolved_high_level_horizon"] = high_level_horizon
                    result["resolved_high_level_policy_horizon"] = (
                        high_level_policy_horizon
                    )
                    result["n_action_steps"] = args.n_action_steps
                    mode_results.append(result)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    mode_results.append(
                        {
                            "task": task_name,
                            "rollout_index": rollout_i,
                            "seed": seed,
                            "mode": mode,
                            "error": traceback.format_exc(),
                        }
                    )
                update_mode_output(mode, mode_results, completed=False)
                write_output(partial=True)

        update_mode_output(mode, mode_results, completed=True)
        write_output(partial=True)

    write_output(partial=False)
    print(colored(f"Wrote official RLDX recovery results to {args.output}", "yellow"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rldx-repo", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--policy-client-host", type=str, default="")
    parser.add_argument("--policy-client-port", type=int, default=None)
    parser.add_argument(
        "--task-set",
        choices=[
            "atomic_seen",
            "composite_seen",
            "composite_unseen",
            "all_composite",
            "all_target",
        ],
        default="atomic_seen",
    )
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["env_to_last_good"],
        choices=["env_to_last_good", "eef_to_last_good", "continue_from_failure"],
    )
    parser.add_argument("--split", default="target", choices=["pretrain", "target"])
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--high-level-horizon", type=int, default=None)
    parser.add_argument("--subtask-horizon", type=int, default=120)
    parser.add_argument(
        "--match-recovery-horizon-to-no-progress",
        action="store_true",
    )
    parser.add_argument("--stuck-patience", type=int, default=10)
    parser.add_argument("--include-trace", action="store_true")
    args = parser.parse_args()

    if bool(args.model_path) == bool(args.policy_client_host or args.policy_client_port):
        parser.error(
            "Provide exactly one policy source: either --model-path for in-process "
            "RLDX or --policy-client-host/--policy-client-port for a server."
        )
    if (args.policy_client_host and args.policy_client_port is None) or (
        args.policy_client_port is not None and not args.policy_client_host
    ):
        parser.error("--policy-client-host and --policy-client-port must be passed together")

    run_benchmark(args)


if __name__ == "__main__":
    main()
