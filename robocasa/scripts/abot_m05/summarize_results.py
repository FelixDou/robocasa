"""Validate and summarize an ABot-M0.5 RoboCasa365 evaluation run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPECTED_TASK_COUNTS = {
    "atomic_seen": 18,
    "composite_seen": 16,
    "composite_unseen": 16,
}


def load_split_summary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing split summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("per_env"), list):
        raise ValueError(f"invalid per_env list: {path}")
    return data


def validate_split_summary(
    run_root: Path,
    split_name: str,
    expected_episodes: int,
) -> dict:
    """Validate one aggregated split before accepting a launcher retry exit."""
    expected_tasks = EXPECTED_TASK_COUNTS[split_name]
    summary_path = run_root / split_name / "summary.json"
    issues: list[str] = []
    try:
        source = load_split_summary(summary_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {
            "split": split_name,
            "summary_path": str(summary_path),
            "complete": False,
            "task_count": 0,
            "expected_task_count": expected_tasks,
            "issues": [str(exc)],
        }

    task_names: set[str] = set()
    episode_count = 0
    for entry in source["per_env"]:
        task_name = str(entry.get("env_name", "")).strip()
        if not task_name:
            issues.append(f"{split_name}: result with empty env_name")
            continue
        if task_name in task_names:
            issues.append(f"{split_name}: duplicate task {task_name}")
            continue
        task_names.add(task_name)

        num_episodes = int(entry.get("num_episodes", 0))
        success_count = int(entry.get("success_count", 0))
        episode_count += num_episodes
        if num_episodes != expected_episodes:
            issues.append(
                f"{split_name}/{task_name}: expected {expected_episodes} "
                f"episodes, found {num_episodes}"
            )
        if success_count < 0 or success_count > num_episodes:
            issues.append(
                f"{split_name}/{task_name}: invalid success count "
                f"{success_count}/{num_episodes}"
            )
            continue
        expected_rate = success_count / max(1, num_episodes)
        recorded_rate = float(entry.get("success_rate", expected_rate))
        if abs(recorded_rate - expected_rate) > 1e-9:
            issues.append(
                f"{split_name}/{task_name}: recorded success_rate "
                f"{recorded_rate} does not match {expected_rate}"
            )

    if len(task_names) != expected_tasks:
        issues.append(
            f"{split_name}: expected {expected_tasks} tasks, "
            f"found {len(task_names)}"
        )

    return {
        "split": split_name,
        "summary_path": str(summary_path),
        "complete": not issues,
        "task_count": len(task_names),
        "expected_task_count": expected_tasks,
        "episode_count": episode_count,
        "expected_episode_count": expected_tasks * expected_episodes,
        "issues": issues,
    }


def summarize_run(run_root: Path, expected_episodes: int) -> dict:
    issues: list[str] = []
    split_summaries: dict[str, dict] = {}
    all_tasks: set[str] = set()
    task_rates: list[float] = []
    total_successes = 0
    total_episodes = 0

    for split_name, expected_tasks in EXPECTED_TASK_COUNTS.items():
        summary_path = run_root / split_name / "summary.json"
        try:
            source = load_split_summary(summary_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            issues.append(str(exc))
            split_summaries[split_name] = {
                "summary_path": str(summary_path),
                "complete": False,
                "task_count": 0,
                "expected_task_count": expected_tasks,
            }
            continue

        per_env = source["per_env"]
        split_task_names: set[str] = set()
        split_rates: list[float] = []
        split_successes = 0
        split_episodes = 0

        for entry in per_env:
            task_name = str(entry.get("env_name", "")).strip()
            if not task_name:
                issues.append(f"{split_name}: result with empty env_name")
                continue
            if task_name in split_task_names:
                issues.append(f"{split_name}: duplicate task {task_name}")
                continue
            if task_name in all_tasks:
                issues.append(f"task appears in multiple splits: {task_name}")
            split_task_names.add(task_name)
            all_tasks.add(task_name)

            num_episodes = int(entry.get("num_episodes", 0))
            success_count = int(entry.get("success_count", 0))
            if num_episodes != expected_episodes:
                issues.append(
                    f"{split_name}/{task_name}: expected {expected_episodes} "
                    f"episodes, found {num_episodes}"
                )
            if success_count < 0 or success_count > num_episodes:
                issues.append(
                    f"{split_name}/{task_name}: invalid success count "
                    f"{success_count}/{num_episodes}"
                )
                continue

            rate = success_count / max(1, num_episodes)
            recorded_rate = float(entry.get("success_rate", rate))
            if abs(recorded_rate - rate) > 1e-9:
                issues.append(
                    f"{split_name}/{task_name}: recorded success_rate "
                    f"{recorded_rate} does not match {rate}"
                )
            split_rates.append(rate)
            task_rates.append(rate)
            split_successes += success_count
            split_episodes += num_episodes
            total_successes += success_count
            total_episodes += num_episodes

        if len(split_task_names) != expected_tasks:
            issues.append(
                f"{split_name}: expected {expected_tasks} tasks, "
                f"found {len(split_task_names)}"
            )

        split_summaries[split_name] = {
            "summary_path": str(summary_path),
            "complete": (
                len(split_task_names) == expected_tasks
                and all(
                    int(entry.get("num_episodes", 0)) == expected_episodes
                    for entry in per_env
                )
            ),
            "task_count": len(split_task_names),
            "expected_task_count": expected_tasks,
            "success_count": split_successes,
            "episode_count": split_episodes,
            "mean_task_success_rate": (
                sum(split_rates) / len(split_rates) if split_rates else 0.0
            ),
            "episode_success_rate": (
                split_successes / split_episodes if split_episodes else 0.0
            ),
        }

    expected_total_tasks = sum(EXPECTED_TASK_COUNTS.values())
    if len(all_tasks) != expected_total_tasks:
        issues.append(
            f"overall: expected {expected_total_tasks} unique tasks, "
            f"found {len(all_tasks)}"
        )

    return {
        "model": "ABot-M0.5-RoboCasa365",
        "run_root": str(run_root),
        "expected_episodes_per_task": expected_episodes,
        "complete": not issues,
        "issues": issues,
        "splits": split_summaries,
        "unique_task_count": len(all_tasks),
        "expected_task_count": expected_total_tasks,
        "success_count": total_successes,
        "episode_count": total_episodes,
        "overall_task_average": (
            sum(task_rates) / len(task_rates) if task_rates else 0.0
        ),
        "overall_episode_success_rate": (
            total_successes / total_episodes if total_episodes else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the three official ABot-M0.5 split summaries and "
            "compute the 50-task leaderboard score."
        )
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--expected-episodes",
        type=int,
        required=True,
        help="Required episode count for every task.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <run_root>/overall_summary.json.",
    )
    parser.add_argument(
        "--single-split",
        choices=sorted(EXPECTED_TASK_COUNTS),
        default=None,
        help=(
            "Validate only this split's aggregated summary. Used to distinguish "
            "a complete retry-recovered split from a genuinely failed launcher."
        ),
    )
    args = parser.parse_args()

    if args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")

    if args.single_split:
        result = validate_split_summary(
            args.run_root,
            args.single_split,
            args.expected_episodes,
        )
        print(
            f"{args.single_split}: tasks={result['task_count']}/"
            f"{result['expected_task_count']} "
            f"episodes={result.get('episode_count', 0)}/"
            f"{result.get('expected_episode_count', 0)}"
        )
        if result["issues"]:
            for issue in result["issues"]:
                print(f"ERROR: {issue}", file=sys.stderr)
            return 1
        print(f"validated split summary: {result['summary_path']}")
        return 0

    result = summarize_run(args.run_root, args.expected_episodes)
    output_path = args.output or args.run_root / "overall_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for split_name, split_result in result["splits"].items():
        print(
            f"{split_name}: tasks={split_result['task_count']}/"
            f"{split_result['expected_task_count']} "
            f"mean_task_success={100 * split_result.get('mean_task_success_rate', 0):.2f}%"
        )
    print(
        f"overall: tasks={result['unique_task_count']}/"
        f"{result['expected_task_count']} "
        f"mean_task_success={100 * result['overall_task_average']:.2f}%"
    )
    print(f"summary: {output_path}")

    if result["issues"]:
        for issue in result["issues"]:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
