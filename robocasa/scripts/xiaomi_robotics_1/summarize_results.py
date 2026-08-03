#!/usr/bin/env python3
"""Validate and summarize Xiaomi-Robotics-1 RoboCasa365 results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_SPLITS = {
    "atomic_seen": [
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    ],
    "composite_seen": [
        "DeliverStraw",
        "GetToastedBread",
        "KettleBoiling",
        "LoadDishwasher",
        "PackIdenticalLunches",
        "PreSoakPan",
        "PrepareCoffee",
        "RinseSinkBasin",
        "ScrubCuttingBoard",
        "SearingMeat",
        "SetUpCuttingStation",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "StirVegetables",
        "StoreLeftoversInBowl",
        "WashLettuce",
    ],
    "composite_unseen": [
        "ArrangeBreadBasket",
        "ArrangeTea",
        "BreadSelection",
        "CategorizeCondiments",
        "CuttingToolSelection",
        "GarnishPancake",
        "GatherTableware",
        "HeatKebabSandwich",
        "MakeIceLemonade",
        "PanTransfer",
        "PortionHotDogs",
        "RecycleBottlesByType",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    ],
}
TARGET50 = [task for tasks in TASK_SPLITS.values() for task in tasks]
TASK_TO_INDEX = {task: index for index, task in enumerate(TARGET50)}

RELEASED_REFERENCE = {
    "overall": {"successes": 1432, "episodes": 2500, "success_rate": 0.5728},
    "atomic_seen": {"successes": 736, "episodes": 900, "success_rate": 736 / 900},
    "composite_seen": {"successes": 464, "episodes": 800, "success_rate": 0.58},
    "composite_unseen": {"successes": 232, "episodes": 800, "success_rate": 0.29},
}
LEADERBOARD_REFERENCE = {
    "overall": 0.574,
    "atomic_seen": 0.802,
    "composite_seen": 0.571,
    "composite_unseen": 0.321,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def validate_task(
    task_name: str,
    stats: dict[str, Any],
    expected_episodes: int,
    expected_seed: int,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    episodes = stats.get("episodes")
    if not isinstance(episodes, list):
        episodes = []
        issues.append(f"{task_name}: episodes must be a list")

    if stats.get("env_name") != task_name:
        issues.append(f"{task_name}: env_name does not match the task key")
    if stats.get("split") != "pretrain":
        issues.append(f"{task_name}: expected split=pretrain")
    if stats.get("num_episodes") != expected_episodes:
        issues.append(
            f"{task_name}: expected {expected_episodes} episodes, "
            f"found {stats.get('num_episodes')}"
        )
    if len(episodes) != expected_episodes:
        issues.append(
            f"{task_name}: episode record count is {len(episodes)}, "
            f"expected {expected_episodes}"
        )

    task_index = TASK_TO_INDEX[task_name]
    seen_episode_indices: set[int] = set()
    successes = 0
    for episode in episodes:
        episode_index = episode.get("episode")
        if not isinstance(episode_index, int):
            issues.append(f"{task_name}: invalid episode index {episode_index!r}")
            continue
        if episode_index in seen_episode_indices:
            issues.append(f"{task_name}: duplicate episode {episode_index}")
        seen_episode_indices.add(episode_index)
        global_index = task_index * expected_episodes + episode_index
        if episode.get("global_episode_index") != global_index:
            issues.append(
                f"{task_name} episode {episode_index}: unexpected global index"
            )
        if episode.get("seed") != expected_seed + global_index:
            issues.append(f"{task_name} episode {episode_index}: unexpected seed")
        successes += int(bool(episode.get("success", False)))

    expected_indices = set(range(expected_episodes))
    if seen_episode_indices != expected_indices:
        missing = sorted(expected_indices - seen_episode_indices)
        extra = sorted(seen_episode_indices - expected_indices)
        issues.append(f"{task_name}: episode indices missing={missing} extra={extra}")

    reported_successes = stats.get("successes")
    if reported_successes != successes:
        issues.append(
            f"{task_name}: reported successes={reported_successes}, counted={successes}"
        )
    rate = successes / expected_episodes if expected_episodes else 0.0
    reported_rate = stats.get("success_rate")
    if not isinstance(reported_rate, (int, float)) or not close_enough(
        float(reported_rate), rate
    ):
        issues.append(
            f"{task_name}: reported success_rate={reported_rate}, counted={rate}"
        )

    return {
        "task": task_name,
        "split": next(
            name for name, tasks in TASK_SPLITS.items() if task_name in tasks
        ),
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": rate,
    }, issues


def validate_queue(
    queue_dir: Path,
    expected_result_count: int,
    expected_episodes: int | None = None,
    expected_seed: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    counts = {}
    for name in ("pending", "running", "results", "errors"):
        directory = queue_dir / name
        count = len(list(directory.glob("*.json"))) if directory.is_dir() else 0
        counts[name] = count
    manifest_path = queue_dir / "manifest.json"
    manifest = None
    if not manifest_path.is_file():
        issues.append(f"missing scheduler manifest: {manifest_path}")
    else:
        manifest = load_json(manifest_path)
        config = manifest.get("config", {})
        expected_config = {
            "robot_type": "robocasa365",
            "split": "pretrain",
            "task_set": "target50",
            "replan_steps": 16,
            "obs_history": 4,
            "obs_interval": 2,
            "crop_ratio": 0.95,
        }
        if expected_episodes is not None:
            expected_config["num_trials"] = expected_episodes
        if expected_seed is not None:
            expected_config["seed"] = expected_seed
        for key, expected_value in expected_config.items():
            if config.get(key) != expected_value:
                issues.append(
                    f"scheduler config {key}={config.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        selected_tasks = manifest.get("selected_tasks", [])
        jobs = manifest.get("jobs", [])
        if not isinstance(selected_tasks, list) or not selected_tasks:
            issues.append("scheduler selected_tasks is empty or invalid")
        if not isinstance(jobs, list) or len(jobs) != expected_result_count:
            count = len(jobs) if isinstance(jobs, list) else "invalid"
            issues.append(
                f"scheduler manifest has {count} jobs, "
                f"expected {expected_result_count}"
            )
        elif len({job.get("id") for job in jobs}) != len(jobs):
            issues.append("scheduler manifest contains duplicate job ids")
    if counts["pending"]:
        issues.append(f"scheduler still has {counts['pending']} pending jobs")
    if counts["running"]:
        issues.append(f"scheduler still has {counts['running']} running jobs")
    if counts["errors"]:
        issues.append(f"scheduler contains {counts['errors']} error records")
    if counts["results"] != expected_result_count:
        issues.append(
            f"scheduler has {counts['results']} results, "
            f"expected {expected_result_count}"
        )
    return {
        "path": str(queue_dir),
        "manifest": str(manifest_path) if manifest is not None else None,
        **counts,
    }, issues


def summarize(
    summary_path: Path,
    *,
    expected_episodes: int,
    expected_seed: int,
    queue_dir: Path | None = None,
    allow_partial: bool = False,
    source_commit: str | None = None,
    checkpoint_revision: str | None = None,
) -> dict[str, Any]:
    payload = load_json(summary_path)
    issues: list[str] = []

    if payload.get("robot_type") != "robocasa365":
        issues.append("expected robot_type=robocasa365")
    if payload.get("split") != "pretrain":
        issues.append("expected split=pretrain")
    if payload.get("task_set") != "target50":
        issues.append("expected task_set=target50")
    if payload.get("replan_steps") != 16:
        issues.append("expected replan_steps=16")
    if payload.get("obs_history") != 4:
        issues.append("expected obs_history=4")
    if payload.get("obs_interval") != 2:
        issues.append("expected obs_interval=2")

    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        tasks = {}
        issues.append("summary tasks must be an object")

    unknown_tasks = sorted(set(tasks) - set(TARGET50))
    if unknown_tasks:
        issues.append(f"unknown target50 tasks: {unknown_tasks}")

    selected_tasks = [task for task in TARGET50 if task in tasks]
    missing_tasks = [task for task in TARGET50 if task not in tasks]
    if not selected_tasks:
        issues.append("summary contains no target50 task results")
    if not allow_partial and missing_tasks:
        issues.append(f"missing target50 tasks: {missing_tasks}")

    task_rows = []
    for task_name in selected_tasks:
        row, task_issues = validate_task(
            task_name,
            tasks[task_name],
            expected_episodes,
            expected_seed,
        )
        task_rows.append(row)
        issues.extend(task_issues)

    total_episodes = sum(row["episodes"] for row in task_rows)
    total_successes = sum(row["successes"] for row in task_rows)
    overall_rate = total_successes / total_episodes if total_episodes else 0.0

    if payload.get("num_tasks") != len(task_rows):
        issues.append(
            f"reported num_tasks={payload.get('num_tasks')}, counted={len(task_rows)}"
        )
    if payload.get("num_episodes") != total_episodes:
        issues.append(
            f"reported num_episodes={payload.get('num_episodes')}, "
            f"counted={total_episodes}"
        )
    if payload.get("successes") != total_successes:
        issues.append(
            f"reported successes={payload.get('successes')}, counted={total_successes}"
        )
    reported_overall = payload.get("episode_success_rate")
    if not isinstance(reported_overall, (int, float)) or not close_enough(
        float(reported_overall), overall_rate
    ):
        issues.append(
            f"reported episode_success_rate={reported_overall}, "
            f"counted={overall_rate}"
        )
    reported_task_mean = payload.get("mean_task_success_rate")
    mean_task_rate = (
        sum(row["success_rate"] for row in task_rows) / len(task_rows)
        if task_rows
        else 0.0
    )
    if not isinstance(reported_task_mean, (int, float)) or not close_enough(
        float(reported_task_mean), mean_task_rate
    ):
        issues.append(
            f"reported mean_task_success_rate={reported_task_mean}, "
            f"counted={mean_task_rate}"
        )

    split_summaries = {}
    for split_name, split_tasks in TASK_SPLITS.items():
        rows = [row for row in task_rows if row["task"] in split_tasks]
        episodes = sum(row["episodes"] for row in rows)
        successes = sum(row["successes"] for row in rows)
        split_summaries[split_name] = {
            "task_count": len(rows),
            "expected_task_count": len(split_tasks),
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else None,
        }

    complete = (
        not missing_tasks
        and not unknown_tasks
        and len(task_rows) == 50
        and total_episodes == 50 * expected_episodes
        and all(row["episodes"] == expected_episodes for row in task_rows)
    )

    queue_summary = None
    if queue_dir is not None:
        queue_summary, queue_issues = validate_queue(
            queue_dir,
            total_episodes,
            expected_episodes=expected_episodes,
            expected_seed=expected_seed,
        )
        issues.extend(queue_issues)

    released_delta = None
    leaderboard_delta = None
    if complete and expected_episodes == 50:
        released_delta = {
            "overall": overall_rate - RELEASED_REFERENCE["overall"]["success_rate"],
            **{
                split: split_summaries[split]["success_rate"]
                - RELEASED_REFERENCE[split]["success_rate"]
                for split in TASK_SPLITS
            },
        }
        leaderboard_delta = {
            "overall": overall_rate - LEADERBOARD_REFERENCE["overall"],
            **{
                split: split_summaries[split]["success_rate"]
                - LEADERBOARD_REFERENCE[split]
                for split in TASK_SPLITS
            },
        }

    return {
        "schema_version": 1,
        "summary_path": str(summary_path),
        "complete_official_protocol": complete and expected_episodes == 50,
        "allow_partial": allow_partial,
        "source_commit": source_commit,
        "checkpoint_revision": checkpoint_revision,
        "configuration": {
            "robot_type": payload.get("robot_type"),
            "split": payload.get("split"),
            "task_set": payload.get("task_set"),
            "expected_episodes_per_task": expected_episodes,
            "expected_seed": expected_seed,
            "replan_steps": payload.get("replan_steps"),
            "obs_history": payload.get("obs_history"),
            "obs_interval": payload.get("obs_interval"),
        },
        "task_count": len(task_rows),
        "expected_task_count": 50,
        "missing_tasks": missing_tasks,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": overall_rate,
        "splits": split_summaries,
        "released_reference": RELEASED_REFERENCE,
        "leaderboard_reference": LEADERBOARD_REFERENCE,
        "delta_vs_released_reference": released_delta,
        "delta_vs_leaderboard": leaderboard_delta,
        "scheduler": queue_summary,
        "issues": issues,
        "valid": not issues and (allow_partial or complete),
        "tasks": task_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Xiaomi's released RoboCasa365 summary and compute "
            "Atomic-Seen, Composite-Seen, and Composite-Unseen scores."
        )
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--queue-dir", type=Path, default=None)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--expected-seed", type=int, default=7)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")
    if args.expected_seed < 0:
        parser.error("--expected-seed must be non-negative")
    if not args.summary.is_file():
        parser.error(f"summary does not exist: {args.summary}")

    result = summarize(
        args.summary,
        expected_episodes=args.expected_episodes,
        expected_seed=args.expected_seed,
        queue_dir=args.queue_dir,
        allow_partial=args.allow_partial,
        source_commit=args.source_commit,
        checkpoint_revision=args.checkpoint_revision,
    )
    output = args.output or args.summary.with_name("reproduction_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for split_name, split in result["splits"].items():
        rate = split["success_rate"]
        rate_text = "n/a" if rate is None else f"{100 * rate:.2f}%"
        print(
            f"{split_name}: tasks={split['task_count']}/"
            f"{split['expected_task_count']} episodes={split['episodes']} "
            f"success={rate_text}"
        )
    print(
        f"overall: tasks={result['task_count']}/50 "
        f"episodes={result['episodes']} "
        f"success={100 * result['success_rate']:.2f}%"
    )
    print(f"complete official protocol: {result['complete_official_protocol']}")
    print(f"summary: {output}")

    if result["issues"]:
        for issue in result["issues"]:
            print(f"ERROR: {issue}", file=sys.stderr)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
