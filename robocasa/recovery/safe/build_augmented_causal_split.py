"""Build an immutable parent split for augmented causal Subtask-SAFE training.

The protocol combines an original parent-training pool with unassigned parents
from a later target-aware allocation.  The original test parents stay unused;
the allocation's calibration and evaluation parents remain frozen together as
the outer test pool.  Inner CV is subsequently run only inside ``parent_train``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

try:
    from .allocate_causal_subtask_data import PROTOCOL as ALLOCATION_PROTOCOL
    from .causal_subtask_safe import stage_name
    from .train_seen_tasks import load_env_records, write_json
except ImportError:
    from allocate_causal_subtask_data import PROTOCOL as ALLOCATION_PROTOCOL
    from causal_subtask_safe import stage_name
    from train_seen_tasks import load_env_records, write_json


PROTOCOL = "causal_subtask_augmented_parent_training_frozen_holdout"


def _sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _string_set(payload, key, *, required=True):
    values = payload.get(key)
    if not isinstance(values, list):
        if required:
            raise ValueError(f"Manifest field {key!r} must be a list")
        return set()
    result = {str(value) for value in values}
    if required and not result:
        raise ValueError(f"Manifest field {key!r} cannot be empty")
    if len(result) != len(values):
        raise ValueError(f"Manifest field {key!r} contains duplicate IDs")
    return result


def _assert_count(label, values, expected):
    if expected is not None and len(values) != int(expected):
        raise ValueError(
            f"Expected {int(expected)} {label}, found {len(values)}"
        )


def _segment_counts(records, selected_stages):
    counts = Counter()
    parents = defaultdict(set)
    for env in records:
        name = stage_name(env)
        if name not in selected_stages:
            continue
        failed = not bool(int(env["episode_success"]))
        counts[(name, failed)] += 1
        parents[name].add(str(env["parent_rollout_id"]))
    return {
        name: {
            "successes": counts[(name, False)],
            "failures": counts[(name, True)],
            "parents": len(parents[name]),
        }
        for name in sorted(selected_stages)
    }


def build_augmented_causal_split(
    env_records,
    *,
    original_split,
    target_allocation,
    source_export=None,
    original_split_path=None,
    target_allocation_path=None,
    expected_original_train_parents=None,
    expected_original_test_parents=None,
    expected_unassigned_parents=None,
    expected_calibration_parents=None,
    expected_evaluation_parents=None,
):
    """Return an exact non-exhaustive train/test selection manifest."""
    if original_split.get("split_unit") != "parent_rollout":
        raise ValueError("Original split must use parent_rollout as its split unit")
    if target_allocation.get("protocol") != ALLOCATION_PROTOCOL:
        raise ValueError("Target allocation uses an unknown protocol")
    if not target_allocation.get("complete"):
        raise ValueError("Target-aware allocation is incomplete")

    original_train = _string_set(original_split, "parent_train")
    original_test = _string_set(original_split, "parent_test")
    unassigned = _string_set(target_allocation, "unassigned_parent_ids")
    calibration = _string_set(target_allocation, "calibration_parent_ids")
    evaluation = _string_set(target_allocation, "evaluation_parent_ids")
    if original_train & original_test:
        raise ValueError("Original training and test parents overlap")
    allocation_groups = {
        "unassigned": unassigned,
        "calibration": calibration,
        "evaluation": evaluation,
    }
    allocation_names = list(allocation_groups)
    for index, left_name in enumerate(allocation_names):
        for right_name in allocation_names[index + 1 :]:
            overlap = allocation_groups[left_name] & allocation_groups[right_name]
            if overlap:
                raise ValueError(
                    f"Target allocation {left_name}/{right_name} parents overlap"
                )
    original_parents = original_train | original_test
    allocated_parents = unassigned | calibration | evaluation
    if original_parents & allocated_parents:
        raise ValueError("Original and new-seed parent pools overlap")

    _assert_count(
        "original training parents",
        original_train,
        expected_original_train_parents,
    )
    _assert_count(
        "original old-test parents",
        original_test,
        expected_original_test_parents,
    )
    _assert_count(
        "unassigned new-seed parents", unassigned, expected_unassigned_parents
    )
    _assert_count(
        "frozen calibration parents", calibration, expected_calibration_parents
    )
    _assert_count(
        "frozen evaluation parents", evaluation, expected_evaluation_parents
    )

    by_parent = defaultdict(list)
    segment_ids = set()
    task_ids = set()
    for _, env in env_records:
        segment_id = str(env["rollout_id"])
        parent = str(env.get("parent_rollout_id") or segment_id)
        if segment_id in segment_ids:
            raise ValueError(f"Combined export contains duplicate segment {segment_id}")
        segment_ids.add(segment_id)
        task_ids.add(int(env["task_id"]))
        by_parent[parent].append(env)
    if not by_parent:
        raise ValueError("Combined export contains no parent rollouts")

    required_parents = original_parents | allocated_parents
    missing = sorted(required_parents - set(by_parent))
    if missing:
        raise ValueError(
            "Required parents are absent from the combined export: "
            + ", ".join(missing[:5])
        )

    parent_train = original_train | unassigned
    parent_test = calibration | evaluation
    if parent_train & parent_test:
        raise AssertionError("Augmented training and frozen holdout parents overlap")
    train = {
        str(env["rollout_id"])
        for parent in parent_train
        for env in by_parent[parent]
    }
    test = {
        str(env["rollout_id"])
        for parent in parent_test
        for env in by_parent[parent]
    }
    if not train or not test or train & test:
        raise AssertionError("Augmented segment train/test assignment is invalid")

    selected_stages = tuple(
        dict.fromkeys(str(value) for value in target_allocation["selected_stages"])
    )
    if not selected_stages:
        raise ValueError("Target allocation contains no selected stages")
    observed_stages = {
        stage_name(env) for records in by_parent.values() for env in records
    }
    missing_stages = sorted(set(selected_stages) - observed_stages)
    if missing_stages:
        raise ValueError(
            "Selected stages are absent from the combined export: "
            + ", ".join(missing_stages)
        )

    train_records = [env for parent in parent_train for env in by_parent[parent]]
    test_records = [env for parent in parent_test for env in by_parent[parent]]
    effective_original_train = {
        parent
        for parent in original_train
        if any(stage_name(env) in selected_stages for env in by_parent[parent])
    }
    effective_unassigned = {
        parent
        for parent in unassigned
        if any(stage_name(env) in selected_stages for env in by_parent[parent])
    }
    effective_calibration = {
        parent
        for parent in calibration
        if any(stage_name(env) in selected_stages for env in by_parent[parent])
    }
    effective_evaluation = {
        parent
        for parent in evaluation
        if any(stage_name(env) in selected_stages for env in by_parent[parent])
    }
    if effective_calibration != calibration or effective_evaluation != evaluation:
        raise ValueError(
            "Frozen calibration/evaluation parents do not all contain a selected stage"
        )

    selected_task_ids = {
        int(env["task_id"]) for env in train_records + test_records
    }
    if selected_task_ids != task_ids:
        raise ValueError(
            "Selected parents do not preserve combined-export task coverage: "
            f"{sorted(selected_task_ids)} != {sorted(task_ids)}"
        )

    parent_unused = set(by_parent) - parent_train - parent_test
    unused = {
        str(env["rollout_id"])
        for parent in parent_unused
        for env in by_parent[parent]
    }
    source_report = None
    if source_export is not None:
        report_path = Path(source_export).resolve() / "conversion_report.json"
        if report_path.is_file():
            source_report = json.loads(report_path.read_text())

    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "analysis_status": (
            "locked-holdout paired follow-up after training-distribution expansion; "
            "the holdout was inspected in an earlier experiment and is not a pristine "
            "confirmatory test set"
        ),
        "split_unit": "parent_rollout",
        "split_seed": original_split.get("split_seed", 0),
        "source_export": (
            None if source_export is None else str(Path(source_export).resolve())
        ),
        "source_fingerprint": (
            None if source_report is None else source_report.get("source_fingerprint")
        ),
        "original_split_manifest": (
            None
            if original_split_path is None
            else str(Path(original_split_path).resolve())
        ),
        "target_allocation_manifest": (
            None
            if target_allocation_path is None
            else str(Path(target_allocation_path).resolve())
        ),
        "target_definition": target_allocation["target_definition"],
        "selected_stages": list(selected_stages),
        "train": sorted(train),
        "test": sorted(test),
        "unused": sorted(unused),
        "parent_train": sorted(parent_train),
        "parent_test": sorted(parent_test),
        "parent_unused": sorted(parent_unused),
        "original_training_parent_ids": sorted(original_train),
        "additional_unassigned_training_parent_ids": sorted(unassigned),
        "frozen_calibration_parent_ids": sorted(calibration),
        "frozen_evaluation_parent_ids": sorted(evaluation),
        "unused_original_test_parent_ids": sorted(original_test),
        "counts": {
            "combined_export_segments": len(segment_ids),
            "combined_export_parents": len(by_parent),
            "train_segments": len(train),
            "test_segments": len(test),
            "unused_segments": len(unused),
            "train_parents": len(parent_train),
            "test_parents": len(parent_test),
            "unused_parents": len(parent_unused),
            "original_training_parents": len(original_train),
            "additional_unassigned_training_parents": len(unassigned),
            "frozen_calibration_parents": len(calibration),
            "frozen_evaluation_parents": len(evaluation),
            "unused_original_test_parents": len(original_test),
            "effective_selected_stage_training_parents": len(
                effective_original_train | effective_unassigned
            ),
            "effective_original_selected_stage_training_parents": len(
                effective_original_train
            ),
            "effective_additional_selected_stage_training_parents": len(
                effective_unassigned
            ),
        },
        "selected_stage_support": {
            "train": _segment_counts(train_records, set(selected_stages)),
            "test": _segment_counts(test_records, set(selected_stages)),
        },
        "audit": {
            "original_train_test_disjoint": not bool(
                original_train & original_test
            ),
            "original_and_new_seed_parents_disjoint": not bool(
                original_parents & allocated_parents
            ),
            "training_holdout_parent_disjoint": not bool(
                parent_train & parent_test
            ),
            "calibration_evaluation_parent_disjoint": not bool(
                calibration & evaluation
            ),
            "segment_train_test_disjoint": not bool(train & test),
            "original_old_test_excluded": original_test <= parent_unused,
            "calibration_frozen_exactly": calibration <= parent_test,
            "evaluation_frozen_exactly": evaluation <= parent_test,
            "outer_test_used_for_selection": False,
        },
    }
    manifest["manifest_fingerprint"] = _sha256_json(manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--original-split-manifest", required=True)
    parser.add_argument("--target-allocation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-original-train-parents", type=int, default=100)
    parser.add_argument("--expected-original-test-parents", type=int, default=50)
    parser.add_argument("--expected-unassigned-parents", type=int, default=180)
    parser.add_argument("--expected-calibration-parents", type=int, default=20)
    parser.add_argument("--expected-evaluation-parents", type=int, default=80)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    original_path = Path(args.original_split_manifest).resolve()
    allocation_path = Path(args.target_allocation_manifest).resolve()
    manifest = build_augmented_causal_split(
        load_env_records(args.export_dir),
        original_split=json.loads(original_path.read_text()),
        target_allocation=json.loads(allocation_path.read_text()),
        source_export=args.export_dir,
        original_split_path=original_path,
        target_allocation_path=allocation_path,
        expected_original_train_parents=args.expected_original_train_parents,
        expected_original_test_parents=args.expected_original_test_parents,
        expected_unassigned_parents=args.expected_unassigned_parents,
        expected_calibration_parents=args.expected_calibration_parents,
        expected_evaluation_parents=args.expected_evaluation_parents,
    )
    write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "manifest_fingerprint": manifest["manifest_fingerprint"],
                "counts": manifest["counts"],
                "audit": manifest["audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
