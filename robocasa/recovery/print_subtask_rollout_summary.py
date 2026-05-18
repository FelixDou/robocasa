"""Print plain-text subtask summaries from subtask rollout JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robocasa.recovery.eval_composite_predicates import _describe_predicate


def _description(predicates: dict, name: str) -> str:
    predicate = predicates.get(name, {})
    return predicate.get("description") or _describe_predicate(name, predicate)


def _print_items(title: str, names: list[str], predicates: dict) -> None:
    print(f"{title}:")
    if not names:
        print("- none")
        return
    for name in names:
        print(f"- {_description(predicates, name)}")


def _print_rollout(path: Path) -> None:
    data = json.loads(path.read_text())
    print(f"\n=== {data.get('env_name', path.parent.name)} ===")
    print(f"file: {path}")

    for rollout in data.get("rollouts", []):
        final_eval = rollout.get("final_subtask_eval") or {}
        predicates = final_eval.get("predicates", {})
        required = final_eval.get("required_predicates", [])
        optional = [
            name
            for name, predicate in predicates.items()
            if not predicate.get("required", True) and name != "task_success"
        ]
        completed_ever = set(rollout.get("completed_predicates_ever", []))
        ordered_completed = rollout.get(
            "ordered_completed_required_subtasks",
            rollout.get("completed_subtask_estimate", []),
        )
        ordered_failed = rollout.get(
            "failed_required_subtasks_ordered",
            [name for name in required if name not in set(ordered_completed)],
        )

        print(
            f"\nEpisode {rollout.get('episode_idx', '?')} "
            f"success={bool(rollout.get('success'))} "
            f"max_progress={float(rollout.get('max_subtask_progress', 0.0)):.2f}"
        )

        _print_items(
            "Completed required subtasks in order",
            [name for name in required if name in set(ordered_completed)],
            predicates,
        )
        next_subtask = rollout.get("ordered_current_subtask") or rollout.get(
            "current_subtask_estimate"
        )
        if next_subtask:
            print(f"Next required subtask: {_description(predicates, next_subtask)}")
        else:
            print("Next required subtask: none")
        _print_items(
            "Required subtasks not yet completed in order", ordered_failed, predicates
        )
        _print_items(
            "Observed optional/prerequisite subtasks",
            [name for name in optional if name in completed_ever],
            predicates,
        )
        _print_items(
            "Not observed optional/prerequisite subtasks",
            [name for name in optional if name not in completed_ever],
            predicates,
        )

        print("Diagnosis:")
        stuck = rollout.get("stuck_subtask") or rollout.get("current_subtask_estimate")
        if stuck:
            print(f"- Stuck at: {_description(predicates, stuck)}")
        failure_modes = rollout.get("failure_modes", [])
        print(f"- Failure modes: {', '.join(failure_modes) if failure_modes else 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_json", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.rollout_json:
        _print_rollout(path)


if __name__ == "__main__":
    main()
