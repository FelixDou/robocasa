"""Create a model-agnostic recovery failure dataset from RoboCasa rollouts.

The script runs RoboCasa tasks with any policy that follows the common recovery
benchmark policy interface, records failed rollouts, and writes a JSON manifest
with links to videos, action trajectories, and task/subtask/predicate progress.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback

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

def import_rollout_runtime():
    """Import RoboCasa-dependent helpers lazily so --help stays lightweight."""
    from robocasa.recovery.evaluate_recovery_benchmark import (
        RandomPolicy,
        call_factory,
        json_default,
        load_factory,
        make_env,
        open_deferred_video_writer,
        open_video_writer,
        parse_policy_args,
        resolve_tasks,
    )
    from robocasa.recovery.recovery_rollout import (
        _append_video_frame_from_env,
        _env_task_instruction,
        _is_task_success,
        _step_env,
        call_policy,
    )
    from robocasa.recovery.subtask_eval import (
        get_subtask_eval,
        summarize_subtask_rollout,
    )
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    return {
        "RandomPolicy": RandomPolicy,
        "call_factory": call_factory,
        "json_default": json_default,
        "load_factory": load_factory,
        "make_env": make_env,
        "open_deferred_video_writer": open_deferred_video_writer,
        "open_video_writer": open_video_writer,
        "parse_policy_args": parse_policy_args,
        "resolve_tasks": resolve_tasks,
        "_append_video_frame_from_env": _append_video_frame_from_env,
        "_env_task_instruction": _env_task_instruction,
        "_is_task_success": _is_task_success,
        "_step_env": _step_env,
        "call_policy": call_policy,
        "get_subtask_eval": get_subtask_eval,
        "get_task_horizon": get_task_horizon,
        "summarize_subtask_rollout": summarize_subtask_rollout,
    }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def safe_filename_part(value, max_len=80):
    text = "none" if value is None else str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return (text or "none")[:max_len]


def action_npz_key(key):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(key)).strip("_") or "action"


def action_to_npz_payload(action):
    """Convert a policy action into flat arrays suitable for np.savez."""
    if isinstance(action, dict):
        payload = {}
        for key, value in sorted(action.items()):
            safe_key = action_npz_key(key)
            payload[safe_key] = np.asarray(value)
        return payload
    return {"action": np.asarray(action)}


def stack_action_payloads(actions):
    """Stack a sequence of actions into an npz payload.

    Dict actions may have scalar and vector values. If a key cannot be stacked
    cleanly, it is stored as an object array so the original values are still
    recoverable for later inspection.
    """
    if not actions:
        return {"num_steps": np.asarray(0, dtype=np.int64)}

    per_step = [action_to_npz_payload(action) for action in actions]
    keys = sorted({key for payload in per_step for key in payload})
    stacked = {"num_steps": np.asarray(len(actions), dtype=np.int64)}
    original_key_map = {}
    for action in actions:
        if isinstance(action, dict):
            for key in action:
                original_key_map[action_npz_key(key)] = str(key)
        else:
            original_key_map["action"] = "action"
    stacked["action_key_map_json"] = np.asarray(
        json.dumps(original_key_map, sort_keys=True)
    )
    for key in keys:
        values = [payload.get(key) for payload in per_step]
        if any(value is None for value in values):
            stacked[key] = np.asarray(values, dtype=object)
            continue
        try:
            stacked[key] = np.stack(values, axis=0)
        except ValueError:
            stacked[key] = np.asarray(values, dtype=object)
    return stacked


def save_action_trajectory(actions, action_path):
    action_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(action_path, **stack_action_payloads(actions))
    return action_path


def load_subtask_plans(path):
    if path is None or not Path(path).exists():
        return {}, None
    with Path(path).open() as f:
        data = json.load(f)
    plans = {plan["task"]: plan for plan in data.get("plans", []) if "task" in plan}
    return plans, {
        "path": str(Path(path)),
        "metadata": data.get("metadata", {}),
    }


def load_registered_atomic_task_names():
    path = Path(__file__).with_name("eval_composite_predicates.py")
    spec = importlib.util.spec_from_file_location("eval_composite_predicates", path)
    if spec is None or spec.loader is None:
        return set()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(getattr(module, "ATOMIC_TASK_PREDICATES", {}))


def load_task_subtask_description_overrides():
    path = Path(__file__).with_name("eval_composite_predicates.py")
    spec = importlib.util.spec_from_file_location("eval_composite_predicates", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(getattr(module, "_TASK_PREDICATE_DESCRIPTION_OVERRIDES", {}))


def load_task_subtask_group_overrides():
    path = Path(__file__).with_name("eval_composite_predicates.py")
    spec = importlib.util.spec_from_file_location("eval_composite_predicates", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(getattr(module, "_TASK_SUBTASK_GROUP_OVERRIDES", {}))


def load_composite_atomic_task_overrides():
    path = Path(__file__).with_name("eval_composite_predicates.py")
    spec = importlib.util.spec_from_file_location("eval_composite_predicates", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(getattr(module, "_COMPOSITE_ATOMIC_TASK_OVERRIDES", {}))


def validate_dataset_tasks(tasks, allow_unregistered=False):
    if allow_unregistered:
        return
    registered = load_registered_atomic_task_names()
    composite = set(load_composite_atomic_task_overrides())
    known = registered | composite
    unknown = [task for task in tasks if task not in known]
    if unknown:
        valid = ", ".join(sorted(known))
        bad = ", ".join(unknown)
        raise ValueError(
            "Recovery failure dataset only supports registered atomic tasks or "
            f"composite tasks with curated decompositions by default. Unknown "
            f"task(s): {bad}. Known tasks: {valid}. Pass "
            "--allow-unregistered-atomic-envs only for custom environments that "
            "still expose subtask predicates."
        )


# Backward-compatible name for older tests/imports.
validate_atomic_tasks = validate_dataset_tasks


def is_composite_task(task_name):
    return task_name in load_composite_atomic_task_overrides()


def compact_task_plan(plan):
    if not plan:
        return None
    return {
        "task": plan.get("task"),
        "description": plan.get("description"),
        "confidence": plan.get("confidence"),
        "declared_num_subtasks": plan.get("declared_num_subtasks"),
        "derived_step_count": plan.get("derived_step_count"),
        "derived_plan": plan.get("derived_plan", []),
        "gaps": plan.get("gaps", []),
        "source_file": plan.get("source_file"),
    }


def infer_atomic_skill_from_plan(plan):
    steps = (plan or {}).get("derived_plan") or []
    skill_ids = [step.get("skill_id") for step in steps if step.get("skill_id")]
    if not skill_ids:
        return None
    non_release = [skill for skill in skill_ids if skill != "release_and_retreat"]
    return non_release[0] if non_release else skill_ids[0]


def infer_atomic_skill_from_task(task_name, summary=None):
    name = str(task_name or "")
    if name.startswith("Open"):
        return "open_fixture"
    if name.startswith("Close"):
        return "close_fixture"
    if name.startswith("PickPlace"):
        failed = " ".join((summary or {}).get("failed_required_predicates_final", []))
        if any(token in failed for token in ("grasped", "picked")):
            return "pick_object"
        if "_in_" in failed or "inside" in failed:
            return "place_object_in_target"
        return "place_object_on_target"
    if name in {"TurnOnStove", "TurnOffStove"}:
        return "turn_knob"
    if name in {"TurnOnSinkFaucet"}:
        return "turn_lever"
    if name in {"TurnOnMicrowave", "TurnOnElectricKettle", "CoffeeSetupMug"}:
        return "press_button"
    if "Rack" in name:
        return "slide_rack"
    if "Navigate" in name:
        return "navigate"
    return None


def runtime_subtask_sequence(summary):
    final_eval = summary.get("final_subtask_eval") or {}
    predicates = final_eval.get("predicates") or {}
    sequence = []
    for index, name in enumerate(final_eval.get("required_predicates") or [], start=1):
        predicate = predicates.get(name, {})
        sequence.append(
            {
                "index": index,
                "predicate": name,
                "description": predicate.get("description"),
                "stage": predicate.get("stage"),
                "required": bool(predicate.get("required", True)),
                "value_final": bool(predicate.get("value", False)),
            }
        )
    return sequence


def _split_pick_place_instruction(instruction):
    text = str(instruction or "").strip()
    lowered = text.lower()
    for marker in (" and place ", ", then place ", " then place "):
        index = lowered.find(marker)
        if index >= 0:
            pick = text[:index].strip()
            place = "Place " + text[index + len(marker) :].strip()
            return pick, place
    return text or "Pick the object.", text or "Place the object at the target."


def _expand_composite_atomic_step_subtasks(step, step_index, required_predicates):
    """Expand one composite atomic task into semantic subtasks.

    The monitor may only expose a placement predicate for a composite step even
    when the atomic task semantics are pick, move, release. In that case we
    intentionally reuse the same predicate across multiple semantic subtasks so
    the stored hierarchy stays comparable to the standalone atomic task.
    """
    atomic_task = step.get("atomic_task") or f"atomic_step_{step_index}"
    instruction = step.get("language_instruction") or atomic_task
    predicate_names = [
        name for name in step.get("predicate_names", []) if name in required_predicates
    ]
    if not predicate_names:
        return []

    if not str(atomic_task).startswith("PickPlace"):
        return [
            {
                "subtask_id": f"{atomic_task}_{step_index}",
                "instruction": instruction,
                "predicate_names": predicate_names,
            }
        ]

    grasp = [name for name in predicate_names if name.endswith("_grasped")]
    release = [
        name
        for name in predicate_names
        if name.endswith("_released") or name == "gripper_released"
    ]
    placement = [
        name
        for name in predicate_names
        if name not in set(grasp) and name not in set(release)
    ]
    if not placement:
        placement = [predicate_names[-1]]
    pick_text, place_text = _split_pick_place_instruction(instruction)
    pick_predicates = grasp or [placement[0]]
    release_predicates = placement + release if release else [placement[-1]]
    return [
        {
            "subtask_id": f"{atomic_task}_{step_index}_pick",
            "instruction": pick_text.rstrip(".") + ".",
            "predicate_names": pick_predicates,
        },
        {
            "subtask_id": f"{atomic_task}_{step_index}_place",
            "instruction": place_text.rstrip(".") + ".",
            "predicate_names": placement,
        },
        {
            "subtask_id": f"{atomic_task}_{step_index}_release",
            "instruction": "Release the object at the target location.",
            "predicate_names": release_predicates,
        },
    ]


def mapped_subtask_sequence(summary, task_name):
    """Return the semantic subtask layer mapped from runtime predicates.

    Runtime predicates are still exported separately in ``predicate_sequence``.
    This layer gives each required predicate the recovery-facing subtask text
    that we curated in ``eval_composite_predicates.py``.
    """
    final_eval = summary.get("final_subtask_eval") or {}
    predicates = final_eval.get("predicates") or {}
    descriptions = load_task_subtask_description_overrides().get(task_name, {})
    group_overrides = load_task_subtask_group_overrides().get(task_name)
    composite_atomic_steps = load_composite_atomic_task_overrides().get(task_name)
    completed = set(summary.get("ordered_completed_required_subtasks", []))
    failed = set(summary.get("failed_required_predicates_final", []))
    sequence = []
    required_predicates = list(final_eval.get("required_predicates") or [])
    if group_overrides:
        groups = [
            {
                "subtask_id": subtask_id,
                "instruction": instruction,
                "predicate_names": [
                    name for name in predicate_names if name in required_predicates
                ],
            }
            for subtask_id, instruction, predicate_names in group_overrides
        ]
        grouped = {
            name
            for group in groups
            for name in group["predicate_names"]
        }
        groups.extend(
            {
                "subtask_id": name,
                "instruction": (
                    descriptions.get(name)
                    or predicates.get(name, {}).get("description")
                    or name.replace("_", " ").capitalize() + "."
                ),
                "predicate_names": [name],
            }
            for name in required_predicates
            if name not in grouped
        )
    elif composite_atomic_steps:
        groups = []
        for index, step in enumerate(composite_atomic_steps, start=1):
            groups.extend(
                _expand_composite_atomic_step_subtasks(
                    step, index, required_predicates
                )
            )
        grouped = {
            name
            for group in groups
            for name in group["predicate_names"]
        }
        groups.extend(
            {
                "subtask_id": name,
                "instruction": (
                    descriptions.get(name)
                    or predicates.get(name, {}).get("description")
                    or name.replace("_", " ").capitalize() + "."
                ),
                "predicate_names": [name],
            }
            for name in required_predicates
            if name not in grouped
        )
    else:
        groups = [
            {
                "subtask_id": name,
                "instruction": (
                    descriptions.get(name)
                    or predicates.get(name, {}).get("description")
                    or name.replace("_", " ").capitalize() + "."
                ),
                "predicate_names": [name],
            }
            for name in required_predicates
        ]

    for index, group in enumerate(groups, start=1):
        predicate_names = group["predicate_names"]
        if not predicate_names:
            continue
        predicate_entries = [predicates.get(name, {}) for name in predicate_names]
        success = all(
            bool(predicate.get("value", False)) or name in completed
            for name, predicate in zip(predicate_names, predicate_entries)
        )
        sequence.append(
            {
                "index": index,
                "subtask_id": group["subtask_id"],
                "kind": "semantic_subtask",
                "instruction": group["instruction"],
                "success": success,
                "value_final": success,
                "predicate_names": predicate_names,
                "stage": next(
                    (
                        predicate.get("stage")
                        for predicate in predicate_entries
                        if predicate.get("stage")
                    ),
                    None,
                ),
                "required": any(
                    bool(predicate.get("required", True))
                    for predicate in predicate_entries
                ),
                "failed": any(name in failed for name in predicate_names),
            }
        )
    return sequence


def mapped_composite_atomic_task_sequence(summary, task_name):
    """Return the atomic-task layer for a composite task, when curated."""
    steps = load_composite_atomic_task_overrides().get(task_name, [])
    if not steps:
        return []
    completed = set(summary.get("ordered_completed_required_subtasks", []))
    failed = set(summary.get("failed_required_predicates_final", []))
    final_eval = summary.get("final_subtask_eval") or {}
    predicates = final_eval.get("predicates") or {}
    sequence = []
    for index, step in enumerate(steps, start=1):
        predicate_names = list(step.get("predicate_names") or [])
        success = all(
            bool(predicates.get(name, {}).get("value", False)) or name in completed
            for name in predicate_names
        )
        sequence.append(
            {
                "index": index,
                "step_id": step.get("atomic_task") or f"{task_name}_step_{index}",
                "kind": "atomic_task",
                "atomic_task": step.get("atomic_task"),
                "atomic_task_source": step.get("atomic_task_source", "derived"),
                "is_registered_atomic_task": step.get("atomic_task_source")
                == "registered",
                "language_instruction": step.get("language_instruction"),
                "instruction": step.get("language_instruction"),
                "success": success,
                "value_final": success,
                "predicate_names": predicate_names,
                "subtask_ids": [
                    entry["subtask_id"]
                    for entry in mapped_subtask_sequence(summary, task_name)
                    if any(
                        predicate_name in entry.get("predicate_names", [])
                        for predicate_name in predicate_names
                    )
                ],
                "completed_predicates": [
                    name
                    for name in predicate_names
                    if bool(predicates.get(name, {}).get("value", False))
                    or name in completed
                ],
                "failed_predicates": [
                    name for name in predicate_names if name in failed
                ],
            }
        )
    return sequence


def mapped_atomic_step_sequence(summary, task_plan, task_name, instruction):
    """Return the atomic-task layer for an atomic-task failure sample."""
    final_eval = summary.get("final_subtask_eval") or {}
    required = list(final_eval.get("required_predicates") or [])
    task_success = bool(summary.get("task_success"))
    atomic_skill = infer_atomic_skill_from_plan(task_plan)
    if atomic_skill is None:
        atomic_skill = infer_atomic_skill_from_task(task_name, summary=summary)

    return [
        {
            "index": 1,
            "step_id": task_name,
            "kind": "atomic_task",
            "atomic_task": task_name,
            "skill_id": atomic_skill,
            "instruction": instruction,
            "success": task_success,
            "value_final": task_success,
            "predicate_names": required,
            "subtask_ids": [
                entry["subtask_id"]
                for entry in mapped_subtask_sequence(summary, task_name)
            ],
            "completed_predicates": list(
                summary.get("ordered_completed_required_subtasks", [])
            ),
            "failed_predicates": (
                []
                if task_success
                else list(summary.get("failed_required_predicates_final", []))
            ),
        }
    ]


def completed_atomic_steps(summary, task_name):
    if summary.get("task_success"):
        return [task_name]
    return []


def failed_atomic_step(summary, task_name):
    if summary.get("task_success"):
        return None
    return task_name


def atomic_step_progress(summary):
    return 1.0 if summary.get("task_success") else 0.0


def completed_subtasks(summary, task_name):
    return [
        {
            "subtask_id": entry["subtask_id"],
            "instruction": entry["instruction"],
            "predicate_names": entry["predicate_names"],
        }
        for entry in mapped_subtask_sequence(summary, task_name)
        if entry.get("success")
    ]


def failed_subtask(summary, task_name):
    for entry in mapped_subtask_sequence(summary, task_name):
        if entry.get("failed"):
            return {
                "subtask_id": entry["subtask_id"],
                "instruction": entry["instruction"],
                "predicate_names": entry["predicate_names"],
            }
    for entry in mapped_subtask_sequence(summary, task_name):
        if entry.get("required", True) and not entry.get("success"):
            return {
                "subtask_id": entry["subtask_id"],
                "instruction": entry["instruction"],
                "predicate_names": entry["predicate_names"],
            }
    return None


def infer_failure_stage(summary):
    if summary.get("task_success"):
        return "success"
    trace = summary.get("subtask_trace") or []
    valid = [entry for entry in trace if entry.get("subtask_eval_available")]
    if not valid:
        return "subtask_eval_unavailable"
    max_progress = float(summary.get("max_subtask_progress") or 0.0)
    final_progress = float(valid[-1].get("ordered_subtask_progress") or 0.0)
    any_regression = any(entry.get("regressed_predicates") for entry in valid)
    if any_regression:
        return "regressed_after_progress"
    if max_progress == 0.0:
        return "no_progress"
    if final_progress >= 0.67:
        return "timeout_near_success"
    return "partial_progress_then_stall"


def first_failed_subtask(summary):
    for key in (
        "ordered_current_subtask",
        "stuck_subtask",
        "current_subtask_estimate",
    ):
        value = summary.get(key)
        if value:
            return value
    failed = summary.get("failed_required_predicates_final") or []
    return failed[0] if failed else None


def build_failure_diagnostic(summary, task_plan, task_name):
    final_eval = summary.get("final_subtask_eval") or {}
    failed_subtask = first_failed_subtask(summary)
    failure_modes = list(summary.get("failure_modes") or [])
    atomic_skill = infer_atomic_skill_from_plan(task_plan)
    if atomic_skill is None:
        atomic_skill = infer_atomic_skill_from_task(task_name, summary=summary)
    return {
        "diagnostic_source": "predicate_progress",
        "label_status": "weak_auto",
        "failed_atomic": task_name if not summary.get("task_success") else None,
        "failed_atomic_step": failed_atomic_step(summary, task_name),
        "atomic_skill": atomic_skill,
        "failed_subtask": failed_subtask,
        "failed_subtask_predicate": failed_subtask,
        "failure_stage": infer_failure_stage(summary),
        "failure_type_weak_label": failure_modes[0] if failure_modes else "unknown",
        "failure_modes": failure_modes,
        "failed_required_predicates": summary.get("failed_required_predicates_final", []),
        "failed_preconditions": summary.get("failed_preconditions", []),
        "failed_preconditions_ever": summary.get("failed_preconditions_ever", []),
        "completed_predicates_ever": summary.get("completed_predicates_ever", []),
        "required_predicates": final_eval.get("required_predicates", []),
        "max_subtask_progress": summary.get("max_subtask_progress", 0.0),
    }


def make_artifact_paths(output_dir, task_name, rollout_i, seed):
    safe_task = safe_filename_part(task_name)
    stem = f"{safe_task}__rollout{rollout_i:04d}__seed{seed}"
    return {
        "video": output_dir / "videos" / safe_task / f"{stem}.mp4",
        "actions": output_dir / "actions" / safe_task / f"{stem}.npz",
    }


def make_labeled_video_path(video_path, sample):
    if video_path is None:
        return None
    label = "SUCCESS" if sample.get("success") else "FAILURE"
    diagnostic = sample.get("failure_diagnostic") or {}
    failed = diagnostic.get("failed_subtask") or "none"
    stage = diagnostic.get("failure_stage") or "none"
    return video_path.with_name(
        "__".join(
            [
                label,
                safe_filename_part(sample.get("task_name")),
                f"rollout{int(sample.get('rollout_index', 0)):04d}",
                f"seed{sample.get('seed')}",
                safe_filename_part(stage, 50),
                "failed_" + safe_filename_part(failed, 80),
            ]
        )
        + video_path.suffix
    )


def maybe_relabel_video(video_path, sample):
    if video_path is None or not video_path.exists():
        return video_path
    labeled_path = make_labeled_video_path(video_path, sample)
    if labeled_path is None or labeled_path == video_path:
        return video_path
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    if labeled_path.exists():
        labeled_path.unlink()
    video_path.rename(labeled_path)
    return labeled_path


def maybe_unlink(path):
    if path is not None and Path(path).exists():
        Path(path).unlink()


def run_atomic_rollout(policy, env, horizon, video_writer, args):
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    instruction = _env_task_instruction(env) or observation_instruction(obs)
    subtask_evals = [get_subtask_eval(env)]
    actions = []
    rewards = []
    dones = []
    infos = []
    success = False
    num_steps = 0
    last_video_frame = None

    for step_i in range(horizon):
        num_steps = step_i + 1
        action = call_policy(policy, obs, instruction=instruction)
        actions.append(action)
        obs, reward, done, info = _step_env(env, action)
        rewards.append(float(reward) if np.isscalar(reward) else reward)
        dones.append(bool(done))
        infos.append(info or {})
        video_frame = _append_video_frame_from_env(
            env,
            video_writer,
            camera_name=args.video_camera_name,
            height=args.video_height,
            width=args.video_width,
            obs=obs,
            previous_frame=last_video_frame,
            prefer_env_render=not args.video_direct_sim_render,
            render_source=args.video_render_source,
        )
        if video_frame is not None:
            last_video_frame = video_frame

        current_eval = (info or {}).get("subtask_eval")
        if current_eval is None:
            current_eval = get_subtask_eval(env)
        subtask_evals.append(current_eval)

        if _is_task_success(info, reward=reward, env=env):
            success = True
            break
        if done:
            break

    return {
        "success": success,
        "num_steps": num_steps,
        "instruction": instruction,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "infos": infos,
        "subtask_evals": subtask_evals,
    }


def observation_instruction(obs):
    if not isinstance(obs, dict):
        return None
    for key in ("annotation.human.task_description", "language"):
        value = obs.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
            if isinstance(value, str) and value:
                return value
    return None


def build_sample(
    *,
    args,
    task_name,
    rollout_i,
    seed,
    horizon,
    rollout,
    task_plan,
    action_path,
    video_path,
):
    summary = summarize_subtask_rollout(
        rollout["subtask_evals"],
        stuck_patience=args.stuck_patience,
        include_trace=args.include_trace,
    )
    diagnostic = build_failure_diagnostic(summary, task_plan, task_name)
    final_eval = summary.get("final_subtask_eval") or {}
    instruction = rollout.get("instruction")
    return {
        "sample_id": f"{task_name}::{rollout_i:04d}::seed{seed}",
        "task_granularity": "composite" if is_composite_task(task_name) else "atomic",
        "granularity_levels": ["hl_task", "atomic_task", "subtask", "predicate"],
        "hl_task": {
            "task_name": task_name,
            "instruction": instruction,
            "success": bool(rollout["success"]),
        },
        "task_name": task_name,
        "atomic_task": task_name if not is_composite_task(task_name) else None,
        "atomic_skill": diagnostic.get("atomic_skill"),
        "task_set": args.task_set,
        "split": args.split,
        "rollout_index": rollout_i,
        "seed": seed,
        "policy_name": args.policy_name,
        "policy_module": args.policy_module,
        "policy_args": args.policy_arg,
        "env_interface": args.env_interface,
        "language_instruction": rollout.get("instruction"),
        "horizon": horizon,
        "num_steps": rollout["num_steps"],
        "success": bool(rollout["success"]),
        "failure_step": None if rollout["success"] else rollout["num_steps"],
        "atomic_sequence_source": "atomic_task_mapping",
        "atomic_sequence": mapped_atomic_step_sequence(
            summary, task_plan, task_name, instruction
        ),
        "subtask_sequence_source": "task_predicate_description_mapping",
        "subtask_sequence": mapped_subtask_sequence(summary, task_name),
        "predicate_sequence_source": "runtime_required_predicates",
        "predicate_sequence": runtime_subtask_sequence(summary),
        "derived_subtask_plan": compact_task_plan(task_plan),
        "initial_subtask_progress": (
            (rollout["subtask_evals"][0] or {}).get("subtask_progress")
            if rollout.get("subtask_evals")
            else None
        ),
        "final_subtask_progress": final_eval.get("subtask_progress"),
        "max_subtask_progress": summary.get("max_subtask_progress", 0.0),
        "atomic_step_progress": atomic_step_progress(summary),
        "completed_atomic_steps": completed_atomic_steps(summary, task_name),
        "failed_atomic_step": failed_atomic_step(summary, task_name),
        "completed_subtasks": completed_subtasks(summary, task_name),
        "failed_subtask": failed_subtask(summary, task_name),
        "completed_subtask_predicates": summary.get(
            "ordered_completed_required_subtasks", []
        ),
        "failed_subtask_predicates": summary.get(
            "failed_required_predicates_final", []
        ),
        "subtask_summary": summary,
        "failure_diagnostic": diagnostic,
        "action_trajectory_path": str(action_path) if action_path else None,
        "video_path": str(video_path) if video_path else None,
        "artifact_format": {
            "actions": "npz",
            "video": "mp4",
            "manifest": "json",
        },
    }


def summarize_dataset(
    samples,
    errors,
    attempted_by_task=None,
    successes_by_task=None,
    failures_by_task=None,
):
    attempted_by_task = attempted_by_task or {}
    successes_by_task = successes_by_task or {}
    failures_by_task = failures_by_task or {}
    by_task = {}
    all_tasks = (
        set(attempted_by_task)
        | set(successes_by_task)
        | set(failures_by_task)
        | {sample["task_name"] for sample in samples}
    )
    for task_name in sorted(all_tasks):
        by_task[task_name] = {
            "attempted": int(attempted_by_task.get(task_name, 0)),
            "recorded_samples": 0,
            "recorded_failures": 0,
            "recorded_successes": 0,
            "rollout_successes": int(successes_by_task.get(task_name, 0)),
            "rollout_failures": int(failures_by_task.get(task_name, 0)),
        }
    for sample in samples:
        entry = by_task.setdefault(
            sample["task_name"],
            {
                "attempted": 0,
                "recorded_samples": 0,
                "recorded_failures": 0,
                "recorded_successes": 0,
                "rollout_successes": 0,
                "rollout_failures": 0,
            },
        )
        entry["recorded_samples"] += 1
        if sample.get("success"):
            entry["recorded_successes"] += 1
        else:
            entry["recorded_failures"] += 1
    num_attempted = sum(attempted_by_task.values())
    num_rollout_successes = sum(successes_by_task.values())
    num_rollout_failures = sum(failures_by_task.values())
    num_recorded_successes = sum(1 for sample in samples if sample.get("success"))
    num_recorded_failures = sum(1 for sample in samples if not sample.get("success"))
    return {
        "num_attempted_rollouts": int(num_attempted),
        "num_recorded_samples": len(samples),
        "num_recorded_failures": int(num_recorded_failures),
        "num_recorded_successes": int(num_recorded_successes),
        "num_rollout_failures": int(num_rollout_failures),
        "num_rollout_successes": int(num_rollout_successes),
        "num_discarded_successes": int(num_rollout_successes - num_recorded_successes),
        "num_samples": len(samples),
        "num_failures": int(num_recorded_failures),
        "num_successes": int(num_recorded_successes),
        "num_errors": len(errors),
        "by_task": by_task,
    }


def run_dataset_creation(args):
    globals().update(import_rollout_runtime())

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output
    if manifest_path is None:
        manifest_path = output_dir / "recovery_failure_dataset.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    task_plans, task_plan_source = load_subtask_plans(args.subtask_plans)
    tasks = resolve_tasks(args)
    validate_dataset_tasks(
        tasks,
        allow_unregistered=args.allow_unregistered_atomic_envs,
    )
    policy_factory = None if args.random_policy else load_factory(args.policy_module)
    policy_args = parse_policy_args(args.policy_arg)

    samples = []
    errors = []
    attempted_by_task = {task_name: 0 for task_name in tasks}
    failures_by_task = {task_name: 0 for task_name in tasks}
    successes_by_task = {task_name: 0 for task_name in tasks}

    manifest = {
        "dataset_type": "recovery_failure_dataset",
        "schema_version": 1,
        "config": vars(args),
        "tasks": tasks,
        "subtask_plan_source": task_plan_source,
        "samples": samples,
        "errors": errors,
        "summary": {},
        "partial": True,
    }

    def write_manifest(partial=True):
        manifest["summary"] = summarize_dataset(
            samples,
            errors,
            attempted_by_task=attempted_by_task,
            successes_by_task=successes_by_task,
            failures_by_task=failures_by_task,
        )
        manifest["partial"] = bool(partial)
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=json_default)

    write_manifest(partial=True)

    for task_name in tasks:
        print(colored(f"Task: {task_name}", "cyan"))
        task_plan = task_plans.get(task_name)
        horizon = (
            args.horizon
            if args.horizon is not None
            else get_task_horizon(task_name)
        )
        for rollout_i in tqdm(range(args.num_rollouts), leave=False):
            if (
                args.max_failed_per_task is not None
                and failures_by_task.get(task_name, 0) >= args.max_failed_per_task
            ):
                break
            if (
                args.stop_after_failures is not None
                and sum(failures_by_task.values()) >= args.stop_after_failures
            ):
                break

            seed = args.seed + rollout_i
            env = None
            video_writer = None
            video_path = None
            action_path = None
            try:
                attempted_by_task[task_name] += 1
                paths = make_artifact_paths(output_dir, task_name, rollout_i, seed)
                video_path = paths["video"] if args.record_videos else None
                action_path = paths["actions"] if args.record_actions else None
                env = make_env(
                    task_name,
                    env_interface=args.env_interface,
                    split=args.split,
                    seed=seed,
                    enable_render=args.enable_render or video_path is not None,
                )
                if video_path is not None and args.video_render_source == "deferred":

                    def render_env_factory(task_name=task_name, seed=seed):
                        return make_env(
                            task_name,
                            env_interface=args.env_interface,
                            split=args.split,
                            seed=seed,
                            enable_render=True,
                        )

                    video_writer = open_deferred_video_writer(
                        video_path,
                        fps=args.video_fps,
                        camera_name=args.video_camera_name,
                        height=args.video_height,
                        width=args.video_width,
                        render_env_factory=render_env_factory,
                    )
                else:
                    video_writer = open_video_writer(video_path, args.video_fps)

                policy = (
                    RandomPolicy(env)
                    if args.random_policy
                    else call_factory(policy_factory, env, policy_args)
                )
                rollout = run_atomic_rollout(
                    policy=policy,
                    env=env,
                    horizon=horizon,
                    video_writer=video_writer,
                    args=args,
                )
            except KeyboardInterrupt:
                raise
            except Exception:
                errors.append(
                    {
                        "task_name": task_name,
                        "rollout_index": rollout_i,
                        "seed": seed,
                        "error": traceback.format_exc(),
                    }
                )
                write_manifest(partial=True)
                continue
            finally:
                if video_writer is not None:
                    video_writer.close()
                if env is not None:
                    env.close()

            keep_sample = (not rollout["success"]) or args.include_successes
            if rollout["success"]:
                successes_by_task[task_name] += 1
                if (
                    args.max_success_per_task is not None
                    and successes_by_task[task_name] > args.max_success_per_task
                ):
                    keep_sample = False

            if keep_sample and action_path is not None:
                save_action_trajectory(rollout["actions"], action_path)

            sample = build_sample(
                args=args,
                task_name=task_name,
                rollout_i=rollout_i,
                seed=seed,
                horizon=horizon,
                rollout=rollout,
                task_plan=task_plan,
                action_path=action_path if keep_sample else None,
                video_path=video_path if keep_sample else None,
            )

            if video_path is not None:
                if keep_sample:
                    labeled_path = maybe_relabel_video(video_path, sample)
                    sample["video_path"] = str(labeled_path) if labeled_path else None
                elif not args.keep_discarded_videos:
                    maybe_unlink(video_path)

            if keep_sample:
                samples.append(sample)
                if not sample["success"]:
                    failures_by_task[task_name] += 1
            write_manifest(partial=True)

    write_manifest(partial=False)
    print(colored(f"Wrote recovery failure dataset manifest to {manifest_path}", "yellow"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--policy-module", type=str, default=None)
    parser.add_argument("--policy-name", type=str, default=None)
    parser.add_argument("--policy-arg", action="append", default=[])
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument(
        "--env-interface",
        choices=["gym", "robosuite"],
        default="gym",
        help="Use gym for VLA-style dict actions, robosuite for raw action vectors.",
    )
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
        "--allow-unregistered-atomic-envs",
        action="store_true",
        help=(
            "Allow explicit --envs not listed in the atomic predicate registry. "
            "Use only for custom atomic tasks that expose get_subtask_progress()."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--stuck-patience", type=int, default=10)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--include-successes", action="store_true")
    parser.add_argument("--max-failed-per-task", type=int, default=None)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    parser.add_argument("--stop-after-failures", type=int, default=None)
    parser.add_argument(
        "--subtask-plans",
        type=Path,
        default=None,
        help=(
            "Optional path to derived_subtask_plans.json for extra metadata. "
            "Subtask sequences come from runtime required predicates plus the "
            "curated task decomposition mapping."
        ),
    )
    parser.add_argument("--record-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-discarded-videos", action="store_true")
    parser.add_argument("--enable-render", action="store_true")
    parser.add_argument("--video-camera-name", default="robot0_agentview_center")
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--video-width", type=int, default=768)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-direct-sim-render", action="store_true")
    parser.add_argument(
        "--video-render-source",
        choices=["auto", "obs", "obs_exact", "sim", "deferred"],
        default="auto",
    )
    args = parser.parse_args()

    if not args.random_policy and args.policy_module is None:
        parser.error("Pass --policy-module module:factory or --random-policy.")
    if args.policy_name is None:
        args.policy_name = "random" if args.random_policy else args.policy_module

    run_dataset_creation(args)


if __name__ == "__main__":
    main()
