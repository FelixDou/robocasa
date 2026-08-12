"""Build a deterministic task/outcome-stratified SAFE train/test manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import random


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_env_records(export_dir):
    values = []
    for path in sorted((Path(export_dir) / "env_records").glob("*.pkl")):
        with path.open("rb") as stream:
            values.append(pickle.load(stream))
    if not values:
        raise ValueError(f"No official SAFE env records found in {export_dir}")
    return values


def build_seen_task_split(
    export_dir,
    output_path,
    *,
    test_per_class=1,
    num_inner_folds=3,
    seed=0,
):
    export_dir = Path(export_dir).resolve()
    output_path = Path(output_path).resolve()
    if test_per_class < 1:
        raise ValueError("test_per_class must be positive")
    if num_inner_folds < 2:
        raise ValueError("num_inner_folds must be at least two")
    records = load_env_records(export_dir)
    grouped = defaultdict(lambda: {0: [], 1: []})
    task_ids = {}
    seen_ids = set()
    for record in records:
        rollout_id = str(record["rollout_id"])
        if rollout_id in seen_ids:
            raise ValueError(f"Duplicate rollout ID in export: {rollout_id}")
        seen_ids.add(rollout_id)
        task = str(record["task_name"])
        success = int(record["episode_success"])
        grouped[task][success].append(rollout_id)
        task_ids[task] = int(record["task_id"])

    minimum_per_class = test_per_class + num_inner_folds
    train = []
    test = []
    excluded = []
    per_task = {}
    for task in sorted(grouped):
        counts = {success: len(grouped[task][success]) for success in (0, 1)}
        if min(counts.values()) < minimum_per_class:
            excluded.append(task)
            per_task[task] = {
                "task_id": task_ids[task],
                "included": False,
                "source_failures": counts[0],
                "source_successes": counts[1],
                "reason": (
                    f"requires at least {minimum_per_class} of each outcome for "
                    f"{test_per_class} outer-test plus {num_inner_folds}-fold CV"
                ),
            }
            continue
        task_counts = {
            "task_id": task_ids[task],
            "included": True,
            "source_failures": counts[0],
            "source_successes": counts[1],
        }
        for success in (0, 1):
            values = sorted(grouped[task][success])
            rng = random.Random(f"{seed}:{task}:{success}")
            rng.shuffle(values)
            group_test = sorted(values[:test_per_class])
            group_train = sorted(values[test_per_class:])
            test.extend(group_test)
            train.extend(group_train)
            label = "success" if success else "failure"
            task_counts[f"train_{label}"] = len(group_train)
            task_counts[f"test_{label}"] = len(group_test)
        per_task[task] = task_counts

    if not train or not test:
        raise ValueError("No tasks satisfy the requested split support")
    output = {
        "schema_version": 1,
        "protocol": "task_outcome_stratified_natural_rate_seen_task_split",
        "split_unit": "rollout",
        "split_seed": int(seed),
        "official_export": str(export_dir),
        "conversion_report_sha256": file_sha256(
            export_dir / "conversion_report.json"
        ),
        "test_per_class": int(test_per_class),
        "num_inner_folds": int(num_inner_folds),
        "minimum_source_per_class": int(minimum_per_class),
        "included_tasks": [task for task in sorted(grouped) if task not in excluded],
        "excluded_tasks": excluded,
        "per_task": per_task,
        "counts": {
            "source": len(records),
            "train": len(train),
            "test": len(test),
            "excluded": len(records) - len(train) - len(test),
            "included_tasks": len(grouped) - len(excluded),
            "excluded_tasks": len(excluded),
        },
        "train": sorted(train),
        "test": sorted(test),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-per-class", type=int, default=1)
    parser.add_argument("--num-inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = build_seen_task_split(
        args.export_dir,
        args.output,
        test_per_class=args.test_per_class,
        num_inner_folds=args.num_inner_folds,
        seed=args.seed,
    )
    print("SAFE seen-task split: VALID")
    print(f"Included tasks: {result['counts']['included_tasks']}")
    print(f"Excluded tasks: {result['counts']['excluded_tasks']}")
    print(f"Train/test rollouts: {result['counts']['train']}/{result['counts']['test']}")
    for task in result["excluded_tasks"]:
        values = result["per_task"][task]
        print(
            f"Excluded {task}: successes={values['source_successes']} "
            f"failures={values['source_failures']}"
        )
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
