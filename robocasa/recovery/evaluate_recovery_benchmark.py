"""Evaluate recovery modes on RoboCasa benchmark tasks.

The script is intentionally policy-pluggable: pass a real VLA policy factory via
``--policy-module module:callable`` or use ``--random-policy`` for smoke tests.
"""

import argparse
import importlib
import inspect
import json
from pathlib import Path
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


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def make_video_path(video_dir, mode, task_name, rollout_i, seed):
    if video_dir is None:
        return None
    safe_mode = mode.replace("/", "_")
    safe_task = task_name.replace("/", "_")
    return video_dir / safe_mode / safe_task / f"rollout_{rollout_i:04d}_seed_{seed}.mp4"


def open_video_writer(video_path, fps):
    if video_path is None:
        return None
    import imageio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(video_path), fps=fps)


def parse_policy_args(values):
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Policy args must be key=value, got {value!r}")
        key, raw = value.split("=", 1)
        parsed[key] = raw
    return parsed


def load_factory(spec):
    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError("--policy-module must be formatted as module:callable")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory


def call_factory(factory, env, policy_args):
    kwargs = {"env": env, **policy_args}
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(env)

    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
        or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
    }
    try:
        return factory(**accepted)
    except TypeError:
        return factory(env)


class RandomPolicy:
    """Policy used only for plumbing smoke tests."""

    def __init__(self, env):
        self.env = env

    def __call__(self, obs, instruction=None):
        if hasattr(self.env, "action_space"):
            return self.env.action_space.sample()
        low, high = self.env.action_spec
        return np.random.uniform(low=low, high=high)


def make_env(task_name, env_interface, split, seed, enable_render):
    import robocasa  # noqa: F401

    if env_interface == "gym":
        import gymnasium as gym

        return gym.make(
            f"robocasa/{task_name}",
            split=split,
            seed=seed,
            enable_render=enable_render,
        )

    from robocasa.utils.env_utils import create_env

    return create_env(
        env_name=task_name,
        seed=seed,
        split=split,
        has_renderer=False,
        has_offscreen_renderer=enable_render,
    )


def resolve_tasks(args):
    from robocasa.utils.dataset_registry import TARGET_TASKS

    if args.envs:
        return args.envs
    if args.task_set == "all_composite":
        return TARGET_TASKS["composite_seen"] + TARGET_TASKS["composite_unseen"]
    if args.task_set == "all_target":
        return (
            TARGET_TASKS["atomic_seen"]
            + TARGET_TASKS["composite_seen"]
            + TARGET_TASKS["composite_unseen"]
        )
    return list(TARGET_TASKS[args.task_set])


def summarize_results(results):
    attempted = [r for r in results if r.get("recovery_attempted")]
    high_level_successes = [
        r for r in results if (r.get("high_level") or {}).get("success")
    ]
    recovery_successes = [
        r for r in attempted if (r.get("subtask") or {}).get("success")
    ]
    return {
        "num_rollouts": len(results),
        "high_level_success_count": len(high_level_successes),
        "high_level_success_rate": len(high_level_successes) / len(results)
        if results
        else 0.0,
        "recovery_attempt_count": len(attempted),
        "recovery_subtask_success_count": len(recovery_successes),
        "recovery_subtask_success_rate": len(recovery_successes) / len(attempted)
        if attempted
        else 0.0,
    }


def run_benchmark(args):
    from robocasa.recovery.recovery_rollout import (
        RecoveryConfig,
        run_recovery_after_failed_rollout,
    )

    modes = args.modes or [
        "eef_to_last_good",
        "env_to_last_good",
        "continue_from_failure",
    ]
    tasks = resolve_tasks(args)
    policy_factory = None if args.random_policy else load_factory(args.policy_module)
    policy_args = parse_policy_args(args.policy_arg)

    output = {
        "config": vars(args),
        "tasks": tasks,
        "modes": {},
    }

    for mode in modes:
        mode_results = []
        print(colored(f"Evaluating recovery mode: {mode}", "green"))
        for task_name in tasks:
            print(colored(f"  Task: {task_name}", "cyan"))
            for rollout_i in tqdm(range(args.num_rollouts), leave=False):
                env = None
                video_path = None
                video_writer = None
                try:
                    seed = args.seed + rollout_i
                    video_path = make_video_path(args.video_dir, mode, task_name, rollout_i, seed)
                    env = make_env(
                        task_name,
                        env_interface=args.env_interface,
                        split=args.split,
                        seed=seed,
                        enable_render=args.enable_render or video_path is not None,
                    )
                    video_writer = open_video_writer(video_path, args.video_fps)
                    policy = (
                        RandomPolicy(env)
                        if args.random_policy
                        else call_factory(policy_factory, env, policy_args)
                    )
                    result = run_recovery_after_failed_rollout(
                        policy,
                        env,
                        RecoveryConfig(
                            mode=mode,
                            high_level_horizon=args.high_level_horizon,
                            subtask_horizon=args.subtask_horizon,
                            stuck_patience=args.stuck_patience,
                            include_trace=args.include_trace,
                        ),
                        video_writer=video_writer,
                        video_camera_name=args.video_camera_name,
                        video_height=args.video_height,
                        video_width=args.video_width,
                    )
                    result["task"] = task_name
                    result["rollout_index"] = rollout_i
                    result["seed"] = seed
                    result["mode"] = mode
                    if video_path is not None:
                        result["video_path"] = str(video_path)
                    mode_results.append(result)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    mode_results.append(
                        {
                            "task": task_name,
                            "rollout_index": rollout_i,
                            "seed": args.seed + rollout_i,
                            "mode": mode,
                            "video_path": str(video_path) if video_path else None,
                            "error": traceback.format_exc(),
                        }
                    )
                finally:
                    if video_writer is not None:
                        video_writer.close()
                    if env is not None:
                        env.close()

        output["modes"][mode] = {
            "summary": summarize_results(
                [result for result in mode_results if "error" not in result]
            ),
            "num_errors": sum(1 for result in mode_results if "error" in result),
            "rollouts": mode_results,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(colored(f"Wrote recovery benchmark results to {args.output}", "yellow"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-module", type=str, default=None)
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
        default="all_composite",
    )
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--high-level-horizon", type=int, default=400)
    parser.add_argument("--subtask-horizon", type=int, default=120)
    parser.add_argument("--stuck-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-render", action="store_true")
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Optional directory for rollout videos, grouped by mode and task.",
    )
    parser.add_argument("--video-camera-name", default="robot0_agentview_center")
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--video-width", type=int, default=768)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    if not args.random_policy and args.policy_module is None:
        parser.error("Pass --policy-module module:callable or --random-policy")

    run_benchmark(args)


if __name__ == "__main__":
    main()
