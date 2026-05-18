"""Validate eval-composite subtask predicate coverage and rollout summaries."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECOMPOSITION_CSV = (
    ROOT
    / "outputs"
    / "manual-20260513-robocasa-recovery"
    / "eval_composite_task_subtask_decomposition.csv"
)
LOCAL_TASK_OVERRIDES = {"MakeIceLemonade"}


def _load_eval_registry() -> dict:
    path = Path(__file__).with_name("eval_composite_predicates.py")
    spec = importlib.util.spec_from_file_location("eval_composite_predicates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EVAL_COMPOSITE_PREDICATES


def _load_eval_tasks(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return sorted({row["task"] for row in csv.DictReader(f)})


def check_coverage(decomposition_csv: Path) -> dict:
    tasks = _load_eval_tasks(decomposition_csv)
    registry = _load_eval_registry()
    missing = [
        task
        for task in tasks
        if task not in registry and task not in LOCAL_TASK_OVERRIDES
    ]
    return {
        "csv_task_count": len(tasks),
        "registry_task_count": len(registry),
        "local_override_tasks": sorted(LOCAL_TASK_OVERRIDES & set(tasks)),
        "missing_runtime_predicate_tasks": missing,
        "extra_registry_tasks": sorted(set(registry) - set(tasks)),
    }


def summarize_rollout_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Rollout JSON not found: {path}. Generate it first with "
            "run_random_subtask_rollouts(...) or pass the actual output path from "
            "your policy evaluation."
        )

    data = json.loads(path.read_text())
    rollouts = data.get("rollouts", [])
    failure_modes = Counter()
    stuck_subtasks = Counter()
    progress_sum = 0.0
    success_progress_mismatches = []
    unavailable = 0

    for i, rollout in enumerate(rollouts):
        final_eval = rollout.get("final_subtask_eval")
        if final_eval is None:
            unavailable += 1
        progress_sum += float(rollout.get("max_subtask_progress", 0.0))
        failure_modes.update(rollout.get("failure_modes", []))
        stuck = rollout.get("stuck_subtask")
        if stuck:
            stuck_subtasks[stuck] += 1

        success = bool(rollout.get("success") or rollout.get("task_success"))
        failed_required = rollout.get("failed_required_predicates_final", [])
        if success and (
            failed_required or rollout.get("max_subtask_progress", 0.0) < 1.0
        ):
            success_progress_mismatches.append(
                {
                    "rollout_index": i,
                    "max_subtask_progress": rollout.get("max_subtask_progress"),
                    "failed_required_predicates_final": failed_required,
                }
            )

    count = len(rollouts)
    return {
        "rollout_json": str(path),
        "rollout_count": count,
        "num_success_rollouts": data.get("num_success_rollouts"),
        "mean_max_subtask_progress": progress_sum / count if count else 0.0,
        "subtask_eval_unavailable_rollouts": unavailable,
        "failure_modes": dict(failure_modes),
        "stuck_subtasks": dict(stuck_subtasks),
        "success_progress_mismatches": success_progress_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decomposition-csv", type=Path, default=DEFAULT_DECOMPOSITION_CSV
    )
    parser.add_argument("--rollout-json", type=Path, action="append", default=[])
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = {"coverage": check_coverage(args.decomposition_csv)}
    if args.rollout_json:
        report["rollouts"] = [
            summarize_rollout_json(path) for path in args.rollout_json
        ]

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
