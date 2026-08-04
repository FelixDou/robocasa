"""Plan targeted Subtask-SAFE collection from observed stage deficits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .causal_subtask_safe import stage_name
    from .train_seen_tasks import load_env_records, load_outer_split_ids, write_json
except ImportError:
    from causal_subtask_safe import stage_name
    from train_seen_tasks import load_env_records, load_outer_split_ids, write_json


def _empty_counts():
    return {"successes": 0, "failures": 0, "segments": 0, "parents": set()}


def _add(counts, env):
    failed = not bool(int(env["episode_success"]))
    counts["failures" if failed else "successes"] += 1
    counts["segments"] += 1
    parent = str(env.get("parent_rollout_id", ""))
    if parent:
        counts["parents"].add(parent)


def build_collection_plan(
    env_records,
    *,
    selection=None,
    min_train_successes=10,
    min_train_failures=5,
    target_train_successes=25,
    target_train_failures=15,
    target_test_successes=10,
    target_test_failures=10,
    requested_stages=None,
):
    requested = None if requested_stages is None else set(requested_stages)
    train_ids = None if selection is None else selection["train"]
    test_ids = set() if selection is None else selection["test"]
    if selection is not None and selection.get("split_unit") != "parent_rollout":
        raise ValueError("Subtask collection planning requires a parent_rollout split")

    grouped = {}
    seen_ids = set()
    for _, env in env_records:
        segment_id = str(env["rollout_id"])
        seen_ids.add(segment_id)
        name = stage_name(env)
        stage = grouped.setdefault(
            name,
            {
                "task_name": str(env.get("parent_task_name", name.split("::", 1)[0])),
                "subtask_id": str(env.get("subtask_id", name.split("::", 1)[1])),
                "subtask_instruction": env.get("subtask_instruction"),
                "overall": _empty_counts(),
                "train": _empty_counts(),
                "test": _empty_counts(),
            },
        )
        _add(stage["overall"], env)
        if train_ids is None or segment_id in train_ids:
            _add(stage["train"], env)
        elif segment_id in test_ids:
            _add(stage["test"], env)

    if selection is not None:
        missing = sorted((selection["train"] | selection["test"]) - seen_ids)
        if missing:
            raise ValueError(
                "Selection manifest references segments absent from the export: "
                + ", ".join(missing[:5])
            )
    if requested is not None:
        unknown = sorted(requested - set(grouped))
        if unknown:
            raise ValueError("Unknown requested stages: " + ", ".join(unknown))

    rows = []
    for name, stage in sorted(grouped.items()):
        train = stage["train"]
        test = stage["test"]
        supported = train["successes"] >= int(min_train_successes) and train[
            "failures"
        ] >= int(min_train_failures)
        selected = (requested is None and supported) or (
            requested is not None and name in requested
        )
        train_success_deficit = max(0, int(target_train_successes) - train["successes"])
        train_failure_deficit = max(0, int(target_train_failures) - train["failures"])
        test_success_deficit = max(0, int(target_test_successes) - test["successes"])
        test_failure_deficit = max(0, int(target_test_failures) - test["failures"])
        total_deficit = sum(
            (
                train_success_deficit,
                train_failure_deficit,
                test_success_deficit,
                test_failure_deficit,
            )
        )
        rows.append(
            {
                "stage": name,
                "task_name": stage["task_name"],
                "subtask_id": stage["subtask_id"],
                "subtask_instruction": stage["subtask_instruction"],
                "supported_by_training": supported,
                "selected_for_v2": selected,
                "train_successes": train["successes"],
                "train_failures": train["failures"],
                "train_parents": len(train["parents"]),
                "test_successes": test["successes"],
                "test_failures": test["failures"],
                "test_parents": len(test["parents"]),
                "train_success_deficit": train_success_deficit,
                "train_failure_deficit": train_failure_deficit,
                "test_success_deficit": test_success_deficit,
                "test_failure_deficit": test_failure_deficit,
                "total_segment_deficit": total_deficit,
                "priority": (
                    "collect_failures"
                    if train_failure_deficit + test_failure_deficit > 0
                    else "collect_successes"
                    if train_success_deficit + test_success_deficit > 0
                    else "ready"
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            not row["selected_for_v2"],
            -row["total_segment_deficit"],
            row["stage"],
        )
    )
    return {
        "schema_version": 1,
        "split_unit": "all_segments" if selection is None else "parent_rollout",
        "selection_manifest": None if selection is None else selection["path"],
        "support_thresholds": {
            "min_train_successes": int(min_train_successes),
            "min_train_failures": int(min_train_failures),
        },
        "targets": {
            "train_successes": int(target_train_successes),
            "train_failures": int(target_train_failures),
            "test_successes": int(target_test_successes),
            "test_failures": int(target_test_failures),
        },
        "selected_stages": [row["stage"] for row in rows if row["selected_for_v2"]],
        "num_stages": len(rows),
        "num_selected_stages": sum(row["selected_for_v2"] for row in rows),
        "total_selected_segment_deficit": sum(
            row["total_segment_deficit"] for row in rows if row["selected_for_v2"]
        ),
        "rows": rows,
        "caveat": (
            "Deficits count labeled stage segments, not exact additional parent "
            "rollouts. Collect natural full rollouts and re-split by parent before "
            "segment extraction."
        ),
    }


def write_plan(plan, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "collection_plan.json"
    csv_path = output / "collection_plan.csv"
    write_json(json_path, plan)
    rows = plan["rows"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["stage"])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--selection-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-train-successes", type=int, default=10)
    parser.add_argument("--min-train-failures", type=int, default=5)
    parser.add_argument("--target-train-successes", type=int, default=25)
    parser.add_argument("--target-train-failures", type=int, default=15)
    parser.add_argument("--target-test-successes", type=int, default=10)
    parser.add_argument("--target-test-failures", type=int, default=10)
    parser.add_argument("--stages", nargs="+")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    selection = load_outer_split_ids(args.selection_manifest)
    plan = build_collection_plan(
        load_env_records(args.export_dir),
        selection=selection,
        min_train_successes=args.min_train_successes,
        min_train_failures=args.min_train_failures,
        target_train_successes=args.target_train_successes,
        target_train_failures=args.target_train_failures,
        target_test_successes=args.target_test_successes,
        target_test_failures=args.target_test_failures,
        requested_stages=args.stages,
    )
    paths = write_plan(plan, args.output_dir)
    print(
        json.dumps(
            {
                "selected_stages": plan["num_selected_stages"],
                "segment_deficit": plan["total_selected_segment_deficit"],
                "json": str(paths[0]),
                "csv": str(paths[1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
