"""Evaluate recovery modes on RoboCasa benchmark tasks.

The script is intentionally policy-pluggable: pass a real VLA policy factory via
``--policy-module module:callable`` or use ``--random-policy`` for smoke tests.
"""

import argparse
import importlib
import inspect
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


def safe_filename_part(value, max_len=80):
    text = "none" if value is None else str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return (text or "none")[:max_len]


def recovery_video_label(result):
    if "error" in result:
        return "ERROR"
    if (result.get("high_level") or {}).get("success"):
        return "HIGH_LEVEL_SUCCESS"
    if result.get("recovery_attempted"):
        subtask_success = (result.get("subtask") or {}).get("success")
        progress_count = (
            (result.get("high_level") or {}).get("last_good_ordered_subtask_count")
            or 0
        )
        if subtask_success and progress_count > 0:
            return "RECOVERY_SUCCESS_WITH_PROGRESS"
        if subtask_success:
            return "RECOVERY_SUCCESS"
        return "RECOVERY_FAILED"
    return "HIGH_LEVEL_FAILED"


def make_labeled_video_path(video_path, result):
    if video_path is None:
        return None
    label = recovery_video_label(result)
    task = safe_filename_part(result.get("task"))
    rollout_i = result.get("rollout_index")
    seed = result.get("seed")
    parts = [
        label,
        task,
        f"rollout{int(rollout_i):04d}" if rollout_i is not None else "rolloutNA",
        f"seed{seed}",
    ]

    high_level = result.get("high_level") or {}
    completed = high_level.get("ordered_completed_subtasks") or []
    if completed:
        parts.append("completed_" + safe_filename_part("_then_".join(completed), 120))

    target = (result.get("subtask") or {}).get("target_subtask")
    if target:
        parts.append("recover_" + safe_filename_part(target, 80))

    return video_path.with_name("__".join(parts) + video_path.suffix)


def rename_video_with_result(video_path, result):
    if video_path is None or not video_path.exists():
        return video_path
    labeled_path = make_labeled_video_path(video_path, result)
    if labeled_path == video_path:
        return video_path
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    if labeled_path.exists():
        labeled_path.unlink()
    video_path.rename(labeled_path)
    return labeled_path


def open_video_writer(video_path, fps):
    if video_path is None:
        return None
    import imageio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(video_path), fps=fps)


class DeferredVideoWriter:
    """Render stored simulator states after rollout using a fresh env.

    Live offscreen rendering can corrupt after long failing rollouts on some
    EGL setups. This writer keeps the policy loop render-free and performs a
    separate playback-style render pass at close().
    """

    def __init__(
        self,
        video_path,
        fps,
        camera_name,
        height,
        width,
        render_env_factory,
    ):
        self.video_path = video_path
        self.fps = fps
        self.camera_name = camera_name
        self.height = height
        self.width = width
        self.render_env_factory = render_env_factory
        self.entries = []
        self.closed = False

    def placeholder_frame(self, height=None, width=None):
        return np.zeros(
            (int(height or self.height), int(width or self.width), 3),
            dtype=np.uint8,
        )

    def append_env_state(self, state):
        if state is not None:
            self.entries.append(("state", state))

    def append_separator_text(self, text, num_frames=0):
        if num_frames > 0:
            self.entries.append(("separator", str(text or ""), int(num_frames)))

    def append_data(self, frame):
        frame = self._normalize_frame(frame)
        if frame is not None:
            self.entries.append(("frame", frame))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.video_path is None or not self.entries:
            return

        import imageio
        from robocasa.recovery.recovery_rollout import _make_separator_frame

        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(self.video_path), fps=self.fps)
        env = None
        last_frame = None
        loaded_model = None
        render_failure_count = 0
        try:
            env = self.render_env_factory()
            for entry in self.entries:
                kind = entry[0]
                if kind == "state":
                    frame, error, loaded_model = self._render_state(
                        env, entry[1], loaded_model=loaded_model
                    )
                    if frame is None:
                        if env is not None:
                            env.close()
                        env = self.render_env_factory()
                        loaded_model = None
                        frame, error, loaded_model = self._render_state(
                            env, entry[1], loaded_model=loaded_model
                        )
                    if frame is None:
                        render_failure_count += 1
                        if render_failure_count <= 3:
                            base = (
                                last_frame
                                if last_frame is not None
                                else self.placeholder_frame()
                            )
                            failure = _make_separator_frame(
                                base,
                                "Deferred video render failed\n"
                                f"{error or 'unknown error'}",
                            )
                            writer.append_data(failure)
                            last_frame = failure
                        continue
                    writer.append_data(frame)
                    last_frame = frame
                elif kind == "frame":
                    frame = entry[1]
                    writer.append_data(frame)
                    last_frame = frame
                elif kind == "separator":
                    _, text, num_frames = entry
                    base = last_frame if last_frame is not None else self.placeholder_frame()
                    separator = _make_separator_frame(base, text)
                    for _ in range(num_frames):
                        writer.append_data(separator)
                    last_frame = separator
        finally:
            writer.close()
            if env is not None:
                env.close()

    def _render_state(self, env, state, loaded_model=None):
        if state is None:
            return None, "state is None", loaded_model

        clean_state = self._clean_state(state)
        model = clean_state.get("model") if isinstance(clean_state, dict) else None
        try:
            loaded_model = self._restore_state_like_playback(
                env,
                clean_state,
                loaded_model=loaded_model,
            )
        except Exception as exc:
            if model is not None:
                try:
                    loaded_model = self._restore_state_like_playback(
                        env,
                        clean_state,
                        loaded_model=None,
                    )
                except Exception as retry_exc:
                    return None, f"{type(retry_exc).__name__}: {retry_exc}", loaded_model
            else:
                return None, f"{type(exc).__name__}: {exc}", loaded_model

        sim = self._get_sim(env)
        if sim is None:
            return None, "render environment does not expose sim", loaded_model
        try:
            sim.forward()
            frame = sim.render(
                height=self.height,
                width=self.width,
                camera_name=self.camera_name,
            )[::-1]
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", loaded_model
        return self._normalize_frame(frame), None, loaded_model

    @classmethod
    def _restore_state_like_playback(cls, env, state, loaded_model=None):
        inner = getattr(env, "env", env)
        model = state.get("model") if isinstance(state, dict) else None
        if model is not None and model != loaded_model:
            if state.get("ep_meta", None) is not None:
                import json

                ep_meta = json.loads(state["ep_meta"])
            else:
                ep_meta = {}
            if hasattr(inner, "set_attrs_from_ep_meta"):
                inner.set_attrs_from_ep_meta(ep_meta)
            elif hasattr(inner, "set_ep_meta"):
                inner.set_ep_meta(ep_meta)

            inner.reset()
            try:
                import robosuite

                robosuite_version_id = int(robosuite.__version__.split(".")[1])
            except Exception:
                robosuite_version_id = 4
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml

                xml = postprocess_model_xml(state["model"])
            elif hasattr(inner, "edit_model_xml"):
                xml = inner.edit_model_xml(state["model"])
            else:
                xml = state["model"]

            inner.reset_from_xml_string(xml)
            inner.sim.reset()
            loaded_model = model

        if "states" not in state:
            raise RuntimeError("Captured state does not contain flattened simulator state")
        inner.sim.set_state_from_flattened(state["states"])
        inner.sim.forward()
        for attr in ("update_sites", "update_state"):
            fn = getattr(inner, attr, None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
        return loaded_model

    @staticmethod
    def _clean_state(state):
        return {
            key: value for key, value in state.items()
            if not str(key).startswith("_sim_state_")
        }

    @staticmethod
    def _get_sim(env):
        inner = getattr(env, "env", env)
        sim_holder = inner if hasattr(inner, "sim") else getattr(inner, "env", inner)
        return getattr(sim_holder, "sim", None)

    @classmethod
    def _set_flattened_state(cls, env, state):
        sim = cls._get_sim(env)
        if sim is None:
            raise RuntimeError("Render environment does not expose sim")
        sim.set_state_from_flattened(state)
        sim.forward()
        inner = getattr(env, "env", env)
        for attr in ("update_sites", "update_state"):
            fn = getattr(inner, attr, None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass

    def _normalize_frame(self, frame):
        if frame is None:
            return None
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.ndim == 3 and frame.shape[2] > 3:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                max_value = float(np.nanmax(frame)) if frame.size else 1.0
                if max_value <= 1.0:
                    frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)


def open_deferred_video_writer(
    video_path,
    fps,
    camera_name,
    height,
    width,
    render_env_factory,
):
    if video_path is None:
        return None
    video_path.parent.mkdir(parents=True, exist_ok=True)
    return DeferredVideoWriter(
        video_path=video_path,
        fps=fps,
        camera_name=camera_name,
        height=height,
        width=width,
        render_env_factory=render_env_factory,
    )


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
    from robocasa.utils.dataset_registry_utils import get_task_horizon
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
        "high_level_horizon_by_task": {
            task_name: (
                args.high_level_horizon
                if args.high_level_horizon is not None
                else get_task_horizon(task_name)
            )
            for task_name in tasks
        },
        "modes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def update_mode_output(mode, mode_results, completed=False):
        output["modes"][mode] = {
            "summary": summarize_results(
                [result for result in mode_results if "error" not in result]
            ),
            "num_errors": sum(1 for result in mode_results if "error" in result),
            "rollouts": mode_results,
            "completed": bool(completed),
        }

    def write_output(partial=True):
        output["partial"] = bool(partial)
        with args.output.open("w") as f:
            json.dump(output, f, indent=2, default=json_default)

    for mode in modes:
        mode_results = []
        update_mode_output(mode, mode_results, completed=False)
        write_output(partial=True)
        print(colored(f"Evaluating recovery mode: {mode}", "green"))
        for task_name in tasks:
            print(colored(f"  Task: {task_name}", "cyan"))
            for rollout_i in tqdm(range(args.num_rollouts), leave=False):
                env = None
                video_path = None
                video_writer = None
                record = None
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
                    if args.video_render_source == "deferred":
                        def render_env_factory(
                            task_name=task_name,
                            seed=seed,
                        ):
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
                    video_separator_frames = (
                        args.video_separator_frames
                        if args.video_separator_frames is not None
                        else round(args.video_separator_seconds * args.video_fps)
                    )
                    high_level_horizon = (
                        args.high_level_horizon
                        if args.high_level_horizon is not None
                        else get_task_horizon(task_name)
                    )
                    result = run_recovery_after_failed_rollout(
                        policy,
                        env,
                        RecoveryConfig(
                            mode=mode,
                            high_level_horizon=high_level_horizon,
                            subtask_horizon=args.subtask_horizon,
                            match_recovery_horizon_to_no_progress=(
                                args.match_recovery_horizon_to_no_progress
                            ),
                            stuck_patience=args.stuck_patience,
                            include_trace=args.include_trace,
                            video_separator_frames=max(0, int(video_separator_frames)),
                            video_separator_text=args.video_separator_text,
                        ),
                        video_writer=video_writer,
                        video_camera_name=args.video_camera_name,
                        video_height=args.video_height,
                        video_width=args.video_width,
                        video_direct_sim_render=args.video_direct_sim_render,
                        video_render_source=args.video_render_source,
                    )
                    result["task"] = task_name
                    result["rollout_index"] = rollout_i
                    result["seed"] = seed
                    result["mode"] = mode
                    result["resolved_high_level_horizon"] = high_level_horizon
                    if video_path is not None:
                        result["video_path"] = str(video_path)
                    record = result
                    mode_results.append(record)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    record = {
                        "task": task_name,
                        "rollout_index": rollout_i,
                        "seed": args.seed + rollout_i,
                        "mode": mode,
                        "video_path": str(video_path) if video_path else None,
                        "error": traceback.format_exc(),
                    }
                    mode_results.append(record)
                finally:
                    if (
                        env is not None
                        and args.video_render_source == "deferred"
                    ):
                        env.close()
                        env = None
                    if video_writer is not None:
                        video_writer.close()
                    if video_path is not None and record is not None:
                        labeled_path = rename_video_with_result(video_path, record)
                        if labeled_path is not None:
                            record["video_path"] = str(labeled_path)
                    if env is not None:
                        env.close()
                update_mode_output(mode, mode_results, completed=False)
                write_output(partial=True)

        update_mode_output(mode, mode_results, completed=True)
        write_output(partial=True)

    write_output(partial=False)
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
    parser.add_argument(
        "--high-level-horizon",
        type=int,
        default=None,
        help=(
            "High-level rollout horizon. Defaults to the task-specific horizon "
            "from robocasa.utils.dataset_registry_utils.get_task_horizon()."
        ),
    )
    parser.add_argument("--subtask-horizon", type=int, default=120)
    parser.add_argument(
        "--match-recovery-horizon-to-no-progress",
        action="store_true",
        help=(
            "Use high_level_steps - last_good_step as the recovery horizon, "
            "instead of the fixed --subtask-horizon."
        ),
    )
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
    parser.add_argument(
        "--video-direct-sim-render",
        action="store_true",
        help=(
            "Render video frames directly from sim.render(camera_name=...) instead "
            "of env.render(). Use this when wrapper-rendered MP4s show corrupted "
            "or stale frames."
        ),
    )
    parser.add_argument(
        "--video-render-source",
        choices=["auto", "obs", "sim", "deferred"],
        default="auto",
        help=(
            "Frame source for rollout videos. Use 'obs' to record the policy "
            "camera observations without live EGL rendering. Use 'sim' to match "
            "the historical non-recovery video path: sim.render(...)[::-1]. "
            "Use 'deferred' to store simulator states during rollout and render "
            "them after the rollout from a fresh environment."
        ),
    )
    parser.add_argument(
        "--video-separator-frames",
        type=int,
        default=None,
        help=(
            "Number of recovery-message frames to insert at the recovery boundary. "
            "Overrides --video-separator-seconds. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--video-separator-seconds",
        type=float,
        default=4.0,
        help="Duration of the reset-message screen in seconds.",
    )
    parser.add_argument(
        "--video-separator-text",
        default=None,
        help=(
            "Optional custom header shown on the recovery-message screen. "
            "Defaults to text derived from the selected recovery mode."
        ),
    )
    args = parser.parse_args()

    if not args.random_policy and args.policy_module is None:
        parser.error("Pass --policy-module module:callable or --random-policy")

    run_benchmark(args)


if __name__ == "__main__":
    main()
