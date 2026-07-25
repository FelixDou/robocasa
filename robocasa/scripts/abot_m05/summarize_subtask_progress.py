"""Aggregate ABot per-batch subtask-progress sidecars."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_TASK_COUNTS = {
    "atomic_seen": 18,
    "composite_seen": 16,
    "composite_unseen": 16,
}


def summarize_subtask_progress(run_root: Path, expected_episodes: int) -> dict:
    split_episodes: dict[str, list[dict]] = defaultdict(list)
    issues = []
    source_files = sorted(run_root.glob("*/envs/*/batches/*/subtask_progress.json"))

    for source_path in source_files:
        relative_parts = source_path.relative_to(run_root).parts
        split_name = relative_parts[0]
        if split_name not in EXPECTED_TASK_COUNTS:
            continue
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"could not read {source_path}: {exc}")
            continue
        env_name = payload.get("env_name")
        for episode in payload.get("episodes", []):
            record = dict(episode)
            record["env_name"] = env_name
            record["source"] = str(source_path)
            split_episodes[split_name].append(record)

    split_summaries = {}
    all_episodes = []
    for split_name, task_count in EXPECTED_TASK_COUNTS.items():
        episodes = split_episodes.get(split_name, [])
        all_episodes.extend(episodes)
        expected_count = task_count * expected_episodes
        if len(episodes) != expected_count:
            issues.append(
                f"{split_name}: expected {expected_count} tracked rollouts, "
                f"found {len(episodes)}"
            )
        available = [ep for ep in episodes if ep.get("subtask_eval_available")]
        if len(available) != len(episodes):
            issues.append(
                f"{split_name}: subtask evaluation unavailable for "
                f"{len(episodes) - len(available)} rollouts"
            )
        progress_values = [
            float(ep.get("max_subtask_progress", 0.0)) for ep in available
        ]
        split_summaries[split_name] = {
            "rollout_count": len(episodes),
            "subtask_eval_available_rollouts": len(available),
            "mean_max_subtask_progress": (
                statistics.fmean(progress_values) if progress_values else 0.0
            ),
            "task_success_count": sum(bool(ep.get("success")) for ep in episodes),
            "stuck_subtasks": dict(
                Counter(
                    ep["stuck_subtask"]
                    for ep in available
                    if ep.get("stuck_subtask")
                ).most_common()
            ),
            "failure_modes": dict(
                Counter(
                    mode
                    for ep in available
                    for mode in ep.get("failure_modes", [])
                ).most_common()
            ),
        }

    available_all = [
        episode for episode in all_episodes if episode.get("subtask_eval_available")
    ]
    summary = {
        "run_root": str(run_root),
        "expected_episodes_per_task": expected_episodes,
        "source_file_count": len(source_files),
        "rollout_count": len(all_episodes),
        "subtask_eval_available_rollouts": len(available_all),
        "splits": split_summaries,
        "issues": issues,
        "complete": not issues,
    }
    output_path = run_root / "subtask_progress_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate ABot subtask_progress.json sidecars."
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    args = parser.parse_args()
    summary = summarize_subtask_progress(args.run_root, args.expected_episodes)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
