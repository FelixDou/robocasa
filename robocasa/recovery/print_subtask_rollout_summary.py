"""Print plain-text subtask summaries from subtask rollout JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robocasa.recovery.eval_composite_predicates import _describe_predicate


def _description(predicates: dict, name: str) -> str:
    predicate = predicates.get(name, {})
    return predicate.get("description") or _describe_predicate(name, predicate)


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
        completed = set(final_eval.get("completed_required_predicates", []))
        failed = set(final_eval.get("failed_required_predicates", []))
        completed_ever = set(rollout.get("completed_predicates_ever", []))

        print(
            f"\nEpisode {rollout.get('episode_idx', '?')} "
            f"success={bool(rollout.get('success'))} "
            f"max_progress={float(rollout.get('max_subtask_progress', 0.0)):.2f}"
        )

        print("Successful required subtasks:")
        for name in required:
            if name in completed:
                print(f"- {_description(predicates, name)}")

        print("Not completed required subtasks:")
        for name in required:
            if name in failed:
                print(f"- {_description(predicates, name)}")

        observed_optional = [name for name in optional if name in completed_ever]
        if observed_optional:
            print("Observed optional subtasks:")
            for name in observed_optional:
                print(f"- {_description(predicates, name)}")

        not_observed_optional = [name for name in optional if name not in completed_ever]
        if not_observed_optional:
            print("Not observed optional subtasks:")
            for name in not_observed_optional:
                print(f"- {_description(predicates, name)}")

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
