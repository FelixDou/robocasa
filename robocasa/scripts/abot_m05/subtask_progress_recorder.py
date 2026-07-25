"""Opt-in Gym wrapper that records RoboCasa subtask progress for ABot runs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from robocasa.recovery.subtask_eval import summarize_subtask_rollout


TRACKING_ENV = "ROBOCASA_TRACK_SUBTASK_PROGRESS"
OUTPUT_ENV = "ROBOCASA_SUBTASK_PROGRESS_JSON"
SAVE_JSON_ENV = "SAVE_JSON"
_PATCH_MARKER = "_robocasa_subtask_progress_make_installed"


def _enabled() -> bool:
    return os.environ.get(TRACKING_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _default_output_path() -> Path | None:
    explicit = os.environ.get(OUTPUT_ENV, "").strip()
    if explicit:
        return Path(explicit)
    save_json = os.environ.get(SAVE_JSON_ENV, "").strip()
    if save_json:
        return Path(save_json).with_name("subtask_progress.json")
    return None


def _progress_events(trace: list[dict]) -> list[dict]:
    """Keep semantic transitions plus the first and final trace entries."""
    if not trace:
        return []
    selected = []
    previous_progress = None
    for index, entry in enumerate(trace):
        progress = entry.get("ordered_subtask_progress", 0.0)
        changed = progress != previous_progress
        semantic_event = bool(
            entry.get("ordered_newly_completed_subtasks")
            or entry.get("regressed_predicates")
        )
        if index == 0 or index == len(trace) - 1 or changed or semantic_event:
            selected.append(entry)
        previous_progress = progress
    return selected


class SubtaskProgressRecorder:
    """Record compact, atomic sidecars without changing ABot's pinned source."""

    def __init__(self, env, output_path: Path | None = None):
        self.env = env
        self.output_path = output_path or _default_output_path()
        self.episodes: list[dict] = []
        self._active = False
        self._episode_index = -1
        self._episode_seed: int | None = None
        self._steps = 0
        self._success = False
        self._subtask_evals: list[dict | None] = []

    def _record_info(self, info: Any) -> None:
        if not isinstance(info, dict):
            self._subtask_evals.append(None)
            return
        self._subtask_evals.append(info.get("subtask_eval"))
        self._success = self._success or bool(info.get("success", False))

    def _write(self) -> None:
        if self.output_path is None:
            return
        payload = {
            "schema_version": 1,
            "env_name": os.environ.get(
                "ENV_NAME",
                getattr(getattr(self.env, "unwrapped", self.env), "env_name", None),
            ),
            "split": os.environ.get("SPLIT", "pretrain"),
            "episodes": self.episodes,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_name(
            f".{self.output_path.name}.{os.getpid()}.tmp"
        )
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
            handle.write("\n")
        os.replace(temporary_path, self.output_path)

    def _finalize_episode(self) -> None:
        if not self._active:
            return
        try:
            summary = summarize_subtask_rollout(
                self._subtask_evals,
                include_trace=True,
            )
            trace = summary.pop("subtask_trace", [])
            summary.update(
                {
                    "episode_index": self._episode_index,
                    "seed": self._episode_seed,
                    "steps": self._steps,
                    "success": self._success,
                    "subtask_eval_available": any(self._subtask_evals),
                    "tracked_step_count": len(self._subtask_evals),
                    "progress_events": _progress_events(trace),
                }
            )
            self.episodes.append(summary)
            self._write()
        except Exception as exc:  # Tracking must never destroy a benchmark run.
            print(
                f"[subtask-progress] failed to save episode "
                f"{self._episode_index}: {exc!r}",
                file=sys.stderr,
            )
        finally:
            self._active = False
            self._subtask_evals = []

    def reset(self, *, seed=None, options=None):
        self._finalize_episode()
        result = self.env.reset(seed=seed, options=options)
        self._episode_index += 1
        self._episode_seed = seed
        self._steps = 0
        self._success = False
        self._subtask_evals = []
        self._active = True
        if isinstance(result, tuple) and len(result) == 2:
            self._record_info(result[1])
        return result

    def step(self, action):
        result = self.env.step(action)
        self._steps += 1
        if isinstance(result, tuple) and len(result) >= 5:
            self._record_info(result[4])
        return result

    def close(self):
        self._finalize_episode()
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def install_gym_make_hook() -> bool:
    """Wrap environments created by ``gym.make`` when tracking is enabled."""
    import gymnasium as gym

    if not _enabled() or getattr(gym, _PATCH_MARKER, False):
        return False

    original_make = gym.make

    def tracked_make(*args, **kwargs):
        return SubtaskProgressRecorder(original_make(*args, **kwargs))

    gym.make = tracked_make
    setattr(gym, _PATCH_MARKER, True)
    print("[subtask-progress] enabled", file=sys.stderr)
    return True
