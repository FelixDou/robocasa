"""Reproduce the official RLDX-1 RoboCasa365 evaluation with subtask traces.

The policy and simulator still use RLDX-1's official server/client,
``MultiStepWrapper``, batched action-chunk execution, episode seeds, task
horizons, and video wrapper. This runner only adds collection of RoboCasa's
``subtask_eval`` info payloads and writes one resumable JSON artifact per task.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD_RLDX_COMMIT = "ef05cd4ae634ff97d672d42275febbc0b92cc192"
TASK_GROUPS = ("atomic_seen", "composite_seen", "composite_unseen")


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=json_default))
    temporary.replace(path)


def load_rldx_task_config(path: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Read the simple task-list and horizon sections from RLDX's YAML file."""

    task_sets = {group: [] for group in TASK_GROUPS}
    horizons = {}
    section = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        top_level = re.fullmatch(r"([A-Za-z0-9_]+):\s*", line)
        if top_level:
            section = top_level.group(1)
            continue
        task_item = re.fullmatch(r"\s+-\s+([A-Za-z0-9_]+)\s*", line)
        if task_item and section in task_sets:
            task_sets[section].append(task_item.group(1))
            continue
        horizon_item = re.fullmatch(r"\s+([A-Za-z0-9_]+):\s*([0-9]+)\s*", line)
        if horizon_item and section == "task_horizons":
            horizons[horizon_item.group(1)] = int(horizon_item.group(2))

    missing_groups = [group for group, tasks in task_sets.items() if not tasks]
    if missing_groups:
        raise ValueError(f"Missing task-set sections in {path}: {missing_groups}")
    missing_horizons = [
        task for tasks in task_sets.values() for task in tasks if task not in horizons
    ]
    if missing_horizons:
        raise ValueError(f"Missing task horizons in {path}: {missing_horizons}")
    return task_sets, horizons


def resolve_tasks(
    task_sets: dict[str, list[str]], task_set: str, envs: list[str]
) -> list[str]:
    if envs:
        known = {task for tasks in task_sets.values() for task in tasks}
        unknown = [task for task in envs if task not in known]
        if unknown:
            raise ValueError(f"Tasks are not in the RLDX target50 config: {unknown}")
        return list(envs)
    if task_set == "target50":
        return [task for group in TASK_GROUPS for task in task_sets[group]]
    return list(task_sets[task_set])


def task_group(task_name: str, task_sets: dict[str, list[str]]) -> str:
    for group in TASK_GROUPS:
        if task_name in task_sets[group]:
            return group
    raise KeyError(f"Unknown target task: {task_name}")


def _git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _local_robocasa_env_fn(
    env_name: str,
    seed: int = 0,
    robocasa_split: str = "pretrain",
    embodiment_tag: str = "general_embodiment",
):
    """RLDX-compatible env factory using this checkout's registered wrappers."""

    del embodiment_tag

    def env_fn():
        import gymnasium as gym
        import robocasa  # noqa: F401
        import robocasa.wrappers.gym_wrapper  # noqa: F401

        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        return gym.make(env_name, split=robocasa_split, seed=seed)

    return env_fn


def load_official_rldx_rollout_module(rldx_repo: Path):
    """Load RLDX's evaluator while keeping this RoboCasa checkout importable."""

    rldx_repo = rldx_repo.expanduser().resolve()
    for path in (str(rldx_repo), str(REPO_ROOT)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(rldx_repo))
    rollout_policy = importlib.import_module("rldx.eval.rollout_policy")

    # RLDX's pinned RoboCasa365 checkout registers ``robocasa/...`` through
    # wrappers/gym_wrapper.py. Some RLDX revisions still import the older
    # robocasa.utils.gym_utils path before gym.make(); replacing only this env
    # factory preserves all official RLDX policy and wrapper behavior while
    # allowing the subtask-enabled local RoboCasa checkout to provide the env.
    rollout_policy.get_robocasa_env_fn = _local_robocasa_env_fn
    return rollout_policy


def _extract_subtask_payloads(value) -> list[dict]:
    """Flatten batched/vector/chunk info and retain every primitive-step payload."""

    if value is None:
        return []
    if isinstance(value, dict):
        if "predicates" in value and "required_predicates" in value:
            return [value]
        payloads = []
        for child in value.values():
            payloads.extend(_extract_subtask_payloads(child))
        return payloads
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _extract_subtask_payloads(value.item())
        payloads = []
        for child in value:
            payloads.extend(_extract_subtask_payloads(child))
        return payloads
    if isinstance(value, (list, tuple)):
        payloads = []
        for child in value:
            payloads.extend(_extract_subtask_payloads(child))
        return payloads
    return []


def _vector_item(value, env_idx: int):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value[env_idx]
    if isinstance(value, (list, tuple)):
        return value[env_idx]
    if isinstance(value, dict):
        return {key: _vector_item(child, env_idx) for key, child in value.items()}
    return value


def _final_info_for_env(infos: dict, env_idx: int):
    final_infos = infos.get("final_info") if isinstance(infos, dict) else None
    if final_infos is None:
        return None
    try:
        return _vector_item(final_infos, env_idx)
    except (IndexError, KeyError, TypeError):
        return None


def _main_info_for_env(infos: dict, env_idx: int) -> dict:
    if not isinstance(infos, dict):
        return {}
    result = {}
    for key, value in infos.items():
        if key == "final_info" or key.startswith("_"):
            continue
        try:
            result[key] = _vector_item(value, env_idx)
        except (IndexError, KeyError, TypeError):
            continue
    return result


def _episode_step_info(infos: dict, env_idx: int, episode_finished: bool) -> dict:
    if episode_finished:
        final_info = _final_info_for_env(infos, env_idx)
        if isinstance(final_info, dict):
            return final_info
    return _main_info_for_env(infos, env_idx)


def _bool_any(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_bool_any(child) for child in value.values())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    if isinstance(value, (list, tuple)):
        return any(_bool_any(child) for child in value)
    return bool(value)


def _primitive_step_count(info: dict, fallback: int) -> int:
    for key in ("rewards", "dones"):
        if key not in info:
            continue
        value = np.asarray(info[key])
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    subtask_steps = len(_extract_subtask_payloads(info.get("subtask_eval")))
    if subtask_steps:
        return subtask_steps
    return int(fallback)


def summarize_rollouts(rollouts: list[dict]) -> dict:
    count = len(rollouts)
    successes = sum(bool(rollout.get("success")) for rollout in rollouts)
    progress = sum(
        float(rollout.get("max_subtask_progress", 0.0)) for rollout in rollouts
    )
    unavailable = sum(
        "subtask_eval_unavailable" in rollout.get("failure_modes", [])
        for rollout in rollouts
    )
    return {
        "num_rollouts": count,
        "num_successes": successes,
        "success_rate": successes / count if count else 0.0,
        "mean_max_subtask_progress": progress / count if count else 0.0,
        "subtask_eval_unavailable_rollouts": unavailable,
    }


def _task_output(
    task_name: str,
    group: str,
    rollouts: list[dict],
    config: dict,
    completed: bool,
) -> dict:
    rollouts = sorted(rollouts, key=lambda row: (row["episode_idx"], row["env_idx"]))
    return {
        "env_name": task_name,
        "task_group": group,
        "policy_name": "RLDX-1-FT-RC365",
        "uses_official_rldx_rollout_path": True,
        "config": config,
        "summary": summarize_rollouts(rollouts),
        "rollouts": rollouts,
        "partial": not completed,
    }


def _existing_rollouts(
    path: Path,
    n_envs: int,
    target_per_env: int,
    expected_config: dict,
) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    existing_config = data.get("config", {})
    mismatches = {
        key: (existing_config.get(key), value)
        for key, value in expected_config.items()
        if existing_config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Refusing to merge incompatible resume data in {path}: {mismatches}"
        )
    rollouts = data.get("rollouts", [])
    keys = set()
    by_env = defaultdict(list)
    for rollout in rollouts:
        key = (int(rollout["env_idx"]), int(rollout["episode_idx"]))
        if key in keys:
            raise ValueError(f"Duplicate episode key in {path}: {key}")
        keys.add(key)
        if 0 <= key[0] < n_envs and 0 <= key[1] < target_per_env:
            by_env[key[0]].append(key[1])
    for env_idx in range(n_envs):
        episode_ids = sorted(by_env[env_idx])
        if episode_ids != list(range(len(episode_ids))):
            raise ValueError(
                f"Non-contiguous resume episodes for env {env_idx} in {path}: {episode_ids}"
            )
    return [
        rollout
        for rollout in rollouts
        if 0 <= int(rollout["env_idx"]) < n_envs
        and 0 <= int(rollout["episode_idx"]) < target_per_env
    ]


def run_task(
    *,
    task_name: str,
    group: str,
    horizon: int,
    policy,
    rollout_policy,
    args,
    rldx_commit: str | None,
) -> dict:
    import gymnasium as gym
    from robocasa.recovery.subtask_eval import summarize_subtask_rollout

    if args.n_episodes % args.n_envs:
        raise ValueError("--n-episodes must be divisible by --n-envs")
    target_per_env = args.n_episodes // args.n_envs
    task_dir = args.output_dir / task_name
    output_path = task_dir / "subtask_rollouts.json"
    video_dir = None if args.no_videos else task_dir / "videos"
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_path": args.model_path,
        "rldx_repo": str(args.rldx_repo),
        "rldx_commit": rldx_commit,
        "leaderboard_rldx_commit": LEADERBOARD_RLDX_COMMIT,
        "split": args.split,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "n_envs": args.n_envs,
        "n_action_steps": args.n_action_steps,
        "max_episode_steps": horizon,
        "videos_enabled": not args.no_videos,
        "include_trace": args.include_trace,
    }
    rollouts = _existing_rollouts(
        output_path,
        args.n_envs,
        target_per_env,
        expected_config=config,
    )
    existing_by_env = defaultdict(list)
    for rollout in rollouts:
        existing_by_env[int(rollout["env_idx"])].append(int(rollout["episode_idx"]))
    episode_indices = [len(existing_by_env[idx]) for idx in range(args.n_envs)]
    completed_episodes = len(rollouts)
    if completed_episodes >= args.n_episodes:
        return _task_output(task_name, group, rollouts, config, completed=True)

    modality_configs = policy.get_modality_config()
    video_delta_indices = np.array(modality_configs["video"].delta_indices)
    state_delta_indices = (
        np.array(modality_configs["state"].delta_indices)
        if "state" in modality_configs
        else None
    )
    wrapper_configs = rollout_policy.WrapperConfigs(
        video=rollout_policy.VideoConfig(
            video_dir=str(video_dir) if video_dir is not None else None,
            max_episode_steps=horizon,
            n_action_steps=args.n_action_steps,
        ),
        multistep=rollout_policy.MultiStepConfig(
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=args.n_action_steps,
            max_episode_steps=horizon,
            terminate_on_success=True,
        ),
    )
    env_fns = [
        (
            lambda env_idx=env_idx: rollout_policy.create_eval_env(
                env_name=f"robocasa/{task_name}",
                env_idx=env_idx,
                total_n_envs=args.n_envs,
                wrapper_configs=wrapper_configs,
                start_episode_id=episode_indices[env_idx],
                seed=args.seed,
                robocasa_split=args.split,
            )
        )
        for env_idx in range(args.n_envs)
    ]
    if args.n_envs == 1 or args.sync_envs:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    episode_evals = [[] for _ in range(args.n_envs)]
    episode_successes = [False] * args.n_envs
    policy_steps = [0] * args.n_envs
    primitive_steps = [0] * args.n_envs
    is_first_step = [True] * args.n_envs
    session_ids = [
        f"{task_name}_env{idx}_{uuid.uuid4().hex[:8]}" for idx in range(args.n_envs)
    ]
    completed_since_write = 0
    max_iterations_without_completion = max(
        100, 3 * math.ceil(horizon / args.n_action_steps)
    )
    iterations_without_completion = 0

    try:
        observations, reset_infos = env.reset()
        for env_idx in range(args.n_envs):
            initial_info = _main_info_for_env(reset_infos, env_idx)
            episode_evals[env_idx].extend(
                _extract_subtask_payloads(initial_info.get("subtask_eval"))
            )
        policy.reset()

        while completed_episodes < args.n_episodes:
            options = {"reset_memory": is_first_step, "session_ids": session_ids}
            actions, _ = policy.get_action(observations, options=options)
            is_first_step = [False] * args.n_envs
            observations, rewards, terminations, truncations, infos = env.step(actions)
            finished_this_iteration = 0

            for env_idx in range(args.n_envs):
                finished = bool(terminations[env_idx] or truncations[env_idx])
                step_info = _episode_step_info(infos, env_idx, finished)
                episode_evals[env_idx].extend(
                    _extract_subtask_payloads(step_info.get("subtask_eval"))
                )
                episode_successes[env_idx] |= _bool_any(step_info.get("success"))
                episode_successes[env_idx] |= bool(
                    np.asarray(rewards[env_idx]).item() > 0
                )
                policy_steps[env_idx] += 1
                primitive_steps[env_idx] += _primitive_step_count(
                    step_info, args.n_action_steps
                )

                if not finished:
                    continue
                local_episode_idx = episode_indices[env_idx]
                if local_episode_idx < target_per_env:
                    summary = summarize_subtask_rollout(
                        episode_evals[env_idx],
                        stuck_patience=args.stuck_patience,
                        include_trace=args.include_trace,
                    )
                    summary.update(
                        {
                            "success": bool(episode_successes[env_idx]),
                            "task_success": bool(episode_successes[env_idx]),
                            "env_idx": env_idx,
                            "episode_idx": local_episode_idx,
                            "episode_seed": (
                                (args.seed + env_idx) * 100000 + local_episode_idx
                                if not args.no_videos
                                else None
                            ),
                            "num_policy_steps": policy_steps[env_idx],
                            "num_action_steps": primitive_steps[env_idx],
                        }
                    )
                    rollouts.append(summary)
                    completed_episodes += 1
                    completed_since_write += 1
                    finished_this_iteration += 1
                    print(
                        f"[{task_name}] {completed_episodes}/{args.n_episodes} "
                        f"env={env_idx} episode={local_episode_idx} "
                        f"success={summary['success']} "
                        f"progress={summary['max_subtask_progress']:.3f}",
                        flush=True,
                    )

                episode_indices[env_idx] += 1
                episode_evals[env_idx] = []
                episode_successes[env_idx] = False
                policy_steps[env_idx] = 0
                primitive_steps[env_idx] = 0
                is_first_step[env_idx] = True

                # With same-step autoreset, main info belongs to the newly reset
                # episode while final_info belongs to the episode just recorded.
                if _final_info_for_env(infos, env_idx) is not None:
                    reset_info = _main_info_for_env(infos, env_idx)
                    episode_evals[env_idx].extend(
                        _extract_subtask_payloads(reset_info.get("subtask_eval"))
                    )

            if completed_since_write:
                completed_since_write = 0
                completed = completed_episodes >= args.n_episodes
                _atomic_write_json(
                    output_path,
                    _task_output(task_name, group, rollouts, config, completed),
                )

            if finished_this_iteration:
                iterations_without_completion = 0
            else:
                iterations_without_completion += 1
            if iterations_without_completion > max_iterations_without_completion:
                raise RuntimeError(
                    f"No episode completed in {iterations_without_completion} policy calls "
                    f"for {task_name}; inspect RLDX/env logs before continuing"
                )
    finally:
        env.close()

    result = _task_output(task_name, group, rollouts, config, completed=True)
    _atomic_write_json(output_path, result)
    return result


def aggregate_task_outputs(
    output_dir: Path, expected_tasks: list[str] | None = None
) -> dict:
    task_rows = []
    errors = []
    for path in sorted(output_dir.glob("*/subtask_rollouts.json")):
        try:
            data = json.loads(path.read_text())
            summary = data.get("summary", summarize_rollouts(data.get("rollouts", [])))
            task_rows.append(
                {
                    "task": data.get("env_name", path.parent.name),
                    "group": data.get("task_group"),
                    "partial": bool(data.get("partial", False)),
                    **summary,
                    "path": str(path),
                }
            )
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    groups = {}
    for group in TASK_GROUPS:
        rows = [row for row in task_rows if row.get("group") == group]
        groups[group] = {
            "num_tasks": len(rows),
            "mean_task_success_rate": (
                sum(float(row["success_rate"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_task_max_subtask_progress": (
                sum(float(row["mean_max_subtask_progress"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "subtask_eval_unavailable_rollouts": sum(
                int(row["subtask_eval_unavailable_rollouts"]) for row in rows
            ),
        }
    expected_tasks = expected_tasks or []
    present = {row["task"] for row in task_rows}
    complete_rows = [row for row in task_rows if not row["partial"]]
    return {
        "num_tasks": len(task_rows),
        "num_complete_tasks": len(complete_rows),
        "overall_mean_task_success_rate": (
            sum(float(row["success_rate"]) for row in task_rows) / len(task_rows)
            if task_rows
            else 0.0
        ),
        "overall_mean_task_max_subtask_progress": (
            sum(float(row["mean_max_subtask_progress"]) for row in task_rows)
            / len(task_rows)
            if task_rows
            else 0.0
        ),
        "groups": groups,
        "missing_tasks": [task for task in expected_tasks if task not in present],
        "errors": errors,
        "tasks": task_rows,
    }


def run_benchmark(args) -> dict:
    task_yaml = args.task_yaml or (
        args.rldx_repo / "run_scripts/eval/robocasa_365/task_sets.yaml"
    )
    task_sets, horizons = load_rldx_task_config(task_yaml)
    all_tasks = resolve_tasks(task_sets, args.task_set, args.envs)
    selected_tasks = [
        task
        for idx, task in enumerate(all_tasks)
        if idx % args.num_shards == args.shard_index
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        summary = aggregate_task_outputs(args.output_dir, expected_tasks=all_tasks)
        _atomic_write_json(args.output_dir / "benchmark_summary.json", summary)
        return summary

    rldx_commit = _git_commit(args.rldx_repo)
    if rldx_commit != LEADERBOARD_RLDX_COMMIT:
        message = (
            f"RLDX checkout is {rldx_commit or 'unknown'}, leaderboard submission used "
            f"{LEADERBOARD_RLDX_COMMIT}"
        )
        if args.require_leaderboard_commit:
            raise RuntimeError(message)
        print(f"WARNING: {message}", flush=True)

    rollout_policy = load_official_rldx_rollout_module(args.rldx_repo)
    policy = rollout_policy.create_rldx_sim_policy(
        model_path=args.model_path,
        embodiment_tag=rollout_policy.EmbodimentTag.GENERAL_EMBODIMENT,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
    )
    task_results = []
    task_errors = []
    for task_name in selected_tasks:
        try:
            task_results.append(
                run_task(
                    task_name=task_name,
                    group=task_group(task_name, task_sets),
                    horizon=horizons[task_name],
                    policy=policy,
                    rollout_policy=rollout_policy,
                    args=args,
                    rldx_commit=rldx_commit,
                )
            )
        except Exception as exc:
            task_errors.append(
                {
                    "task": task_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"ERROR [{task_name}]: {type(exc).__name__}: {exc}", flush=True)
            if len(task_errors) >= args.max_errors:
                break

    summary = aggregate_task_outputs(args.output_dir, expected_tasks=all_tasks)
    summary.update(
        {
            "selected_tasks": selected_tasks,
            "completed_selected_tasks": [result["env_name"] for result in task_results],
            "task_errors": task_errors,
            "rldx_commit": rldx_commit,
            "leaderboard_rldx_commit": LEADERBOARD_RLDX_COMMIT,
            "partial": bool(task_errors)
            or any(
                task not in {result["env_name"] for result in task_results}
                for task in selected_tasks
            ),
        }
    )
    shard_path = args.output_dir / f"shard_{args.shard_index:02d}_summary.json"
    _atomic_write_json(shard_path, summary)
    if args.num_shards == 1:
        _atomic_write_json(args.output_dir / "benchmark_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rldx-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-RC365")
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=20100)
    parser.add_argument(
        "--task-set",
        choices=(*TASK_GROUPS, "target50"),
        default="target50",
    )
    parser.add_argument("--envs", nargs="*", default=[])
    parser.add_argument("--split", choices=("pretrain", "target"), default="target")
    parser.add_argument("--task-yaml", type=Path)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stuck-patience", type=int, default=10)
    parser.add_argument(
        "--include-trace", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--sync-envs", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--require-leaderboard-commit", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_episodes < 1 or args.n_envs < 1 or args.n_action_steps < 1:
        parser.error("episode, environment, and action-step counts must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= --shard-index < --num-shards")
    if not args.aggregate_only and not args.policy_client_host:
        parser.error("--policy-client-host is required for server-based evaluation")
    result = run_benchmark(args)
    print(json.dumps(result, indent=2, default=json_default))


if __name__ == "__main__":
    main()
