"""Print per-subtask accuracy from policy subtask rollout JSON files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path


def _load_describe_predicate():
    path = Path(__file__).with_name("eval_composite_predicates.py")
    spec = importlib.util.spec_from_file_location("eval_composite_predicates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._describe_predicate


_describe_predicate = _load_describe_predicate()


def _description(predicates: dict, name: str) -> str:
    predicate = predicates.get(name, {})
    return predicate.get("description") or _describe_predicate(name, predicate)


def _percent(numerator: int | float, denominator: int) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _format_rate(rate: float) -> str:
    return f"{100.0 * rate:.1f}%"


def _task_group(task_name: str) -> str:
    atomic_seen = {
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
    }
    composite_seen = {
        "DeliverStraw",
        "GetToastedBread",
        "KettleBoiling",
        "LoadDishwasher",
        "MakeIceLemonade",
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
    }
    if task_name in atomic_seen:
        return "atomic_seen"
    if task_name in composite_seen:
        return "composite_seen"
    return "composite_unseen"


def summarize_subtask_accuracy(path: Path) -> dict:
    data = json.loads(path.read_text())
    task_name = data.get("env_name", path.parent.name)
    rollouts = data.get("rollouts", [])
    rollout_count = len(rollouts)
    success_count = sum(
        1 for rollout in rollouts if rollout.get("success") or rollout.get("task_success")
    )
    progress_sum = sum(
        float(rollout.get("max_subtask_progress", 0.0)) for rollout in rollouts
    )

    required_order = []
    predicates = {}
    completed_counts = Counter()
    first_blocker_counts = Counter()

    for rollout in rollouts:
        final_eval = rollout.get("final_subtask_eval") or {}
        rollout_predicates = final_eval.get("predicates", {})
        predicates.update(rollout_predicates)
        for name in final_eval.get("required_predicates", []):
            if name not in required_order:
                required_order.append(name)

        completed = set(
            rollout.get(
                "ordered_completed_required_subtasks",
                rollout.get("completed_subtask_estimate", []),
            )
        )
        for name in completed:
            completed_counts[name] += 1

        blocker = rollout.get("ordered_current_subtask") or rollout.get(
            "stuck_subtask"
        )
        if blocker:
            first_blocker_counts[blocker] += 1

    subtasks = []
    for name in required_order:
        completed = completed_counts[name]
        subtasks.append(
            {
                "name": name,
                "description": _description(predicates, name),
                "completed_rollouts": completed,
                "rollout_count": rollout_count,
                "accuracy": completed / rollout_count if rollout_count else 0.0,
                "first_blocker_count": first_blocker_counts[name],
            }
        )

    return {
        "task": task_name,
        "group": _task_group(task_name),
        "rollout_json": str(path),
        "rollout_count": rollout_count,
        "success_count": success_count,
        "success_rate": success_count / rollout_count if rollout_count else 0.0,
        "mean_max_subtask_progress": progress_sum / rollout_count
        if rollout_count
        else 0.0,
        "subtasks": subtasks,
    }


def _print_task(summary: dict) -> None:
    rollout_count = summary["rollout_count"]
    print(f"\n=== {summary['task']} ({summary['group']}) ===")
    print(f"file: {summary['rollout_json']}")
    print(
        "task_success: "
        f"{summary['success_count']}/{rollout_count} "
        f"({_percent(summary['success_count'], rollout_count):.1f}%)"
    )
    print(
        "mean_ordered_subtask_progress: "
        f"{100.0 * summary['mean_max_subtask_progress']:.1f}%"
    )
    print("subtask_accuracy:")
    if not summary["subtasks"]:
        print("- none")
        return

    for subtask in summary["subtasks"]:
        print(
            f"- {subtask['description']}: "
            f"{subtask['completed_rollouts']}/{subtask['rollout_count']} "
            f"({100.0 * subtask['accuracy']:.1f}%)"
            f" | first_blocker={subtask['first_blocker_count']}"
            f" | key={subtask['name']}"
        )


def _print_group_summary(summaries: list[dict]) -> None:
    groups = {}
    for summary in summaries:
        groups.setdefault(summary["group"], []).append(summary)

    print("=== Group subtask accuracy ===")
    for group in sorted(groups):
        group_summaries = groups[group]
        task_success = sum(summary["success_count"] for summary in group_summaries)
        task_rollouts = sum(summary["rollout_count"] for summary in group_summaries)
        completed_subtasks = sum(
            subtask["completed_rollouts"]
            for summary in group_summaries
            for subtask in summary["subtasks"]
        )
        total_subtasks = sum(
            subtask["rollout_count"]
            for summary in group_summaries
            for subtask in summary["subtasks"]
        )
        print(
            f"{group}: task_success={task_success}/{task_rollouts} "
            f"({_percent(task_success, task_rollouts):.1f}%), "
            f"subtask_accuracy={completed_subtasks}/{total_subtasks} "
            f"({_percent(completed_subtasks, total_subtasks):.1f}%)"
        )


def _hardest_subtask(summary: dict) -> dict | None:
    subtasks = summary["subtasks"]
    if not subtasks:
        return None
    return min(
        subtasks,
        key=lambda subtask: (
            subtask["accuracy"],
            -subtask["first_blocker_count"],
            subtask["name"],
        ),
    )


def build_bottleneck_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        hardest = _hardest_subtask(summary)
        first_blocker = max(
            summary["subtasks"],
            key=lambda subtask: (
                subtask["first_blocker_count"],
                -subtask["accuracy"],
                subtask["name"],
            ),
            default=None,
        )
        rows.append(
            {
                "task": summary["task"],
                "group": summary["group"],
                "task_success": summary["success_rate"],
                "mean_ordered_progress": summary["mean_max_subtask_progress"],
                "hardest_subtask": hardest["description"] if hardest else "",
                "hardest_subtask_key": hardest["name"] if hardest else "",
                "hardest_subtask_accuracy": hardest["accuracy"] if hardest else 0.0,
                "hardest_subtask_completed": hardest["completed_rollouts"]
                if hardest
                else 0,
                "hardest_subtask_total": hardest["rollout_count"] if hardest else 0,
                "most_common_first_blocker": first_blocker["description"]
                if first_blocker
                else "",
                "most_common_first_blocker_key": first_blocker["name"]
                if first_blocker
                else "",
                "most_common_first_blocker_count": first_blocker[
                    "first_blocker_count"
                ]
                if first_blocker
                else 0,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["group"],
            row["task_success"],
            row["mean_ordered_progress"],
            row["task"],
        ),
    )


def write_bottleneck_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "group",
        "task_success",
        "mean_ordered_progress",
        "hardest_subtask",
        "hardest_subtask_key",
        "hardest_subtask_accuracy",
        "hardest_subtask_completed",
        "hardest_subtask_total",
        "most_common_first_blocker",
        "most_common_first_blocker_key",
        "most_common_first_blocker_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_bottleneck_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Task | Group | Success | Mean Progress | Hardest Subtask | Hardest Accuracy | First Blocker | Blocked Rollouts |",
        "|---|---|---:|---:|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task"],
                    row["group"],
                    _format_rate(row["task_success"]),
                    _format_rate(row["mean_ordered_progress"]),
                    row["hardest_subtask"],
                    _format_rate(row["hardest_subtask_accuracy"]),
                    row["most_common_first_blocker"],
                    str(row["most_common_first_blocker_count"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_json", type=Path, nargs="+")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-bottleneck-csv", type=Path)
    parser.add_argument("--out-bottleneck-md", type=Path)
    args = parser.parse_args()

    summaries = [summarize_subtask_accuracy(path) for path in args.rollout_json]
    bottleneck_rows = build_bottleneck_rows(summaries)
    _print_group_summary(summaries)
    for summary in summaries:
        _print_task(summary)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({"tasks": summaries, "bottlenecks": bottleneck_rows}, indent=2)
        )
    if args.out_bottleneck_csv:
        write_bottleneck_csv(bottleneck_rows, args.out_bottleneck_csv)
    if args.out_bottleneck_md:
        write_bottleneck_markdown(bottleneck_rows, args.out_bottleneck_md)


if __name__ == "__main__":
    main()
