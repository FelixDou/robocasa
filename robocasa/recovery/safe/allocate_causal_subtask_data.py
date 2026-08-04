"""Allocate parent-disjoint calibration/evaluation data for causal Subtask-SAFE."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

try:
    from .causal_subtask_safe import stage_name
    from .train_seen_tasks import load_env_records, write_json
except ImportError:
    from causal_subtask_safe import stage_name
    from train_seen_tasks import load_env_records, write_json


PROTOCOL = "causal_subtask_finite_horizon_parent_allocation"


def _source_length(env):
    value = env.get("model_infer_times")
    if value is None:
        value = len(env.get("inference_environment_steps") or [])
    value = int(value)
    if value <= 0:
        raise ValueError(f"Segment {env.get('rollout_id')} has no policy inferences")
    return value


def finite_horizon_record(env, *, prefix, failure_horizon):
    """Create the target-prefix identity without loading feature tensors."""
    length = _source_length(env)
    if length < int(prefix):
        return None
    source_failed = not bool(int(env["episode_success"]))
    remaining = length - int(prefix)
    target_failed = source_failed and remaining <= int(failure_horizon)
    source_id = str(env["rollout_id"])
    parent = str(env.get("parent_rollout_id") or source_id)
    return {
        "source_segment_id": source_id,
        "causal_segment_id": f"{source_id}::prefix-{int(prefix):04d}",
        "parent_rollout_id": parent,
        "parent_task_name": str(
            env.get("parent_task_name") or stage_name(env).split("::", 1)[0]
        ),
        "parent_rollout_failed": bool(
            env.get("parent_rollout_failed", source_failed)
        ),
        "stage": stage_name(env),
        "source_segment_failed": source_failed,
        "source_segment_num_inferences": length,
        "causal_prefix_inferences": int(prefix),
        "remaining_inferences_to_terminal": remaining,
        "causal_failure_horizon_inferences": int(failure_horizon),
        "causal_target_failed": target_failed,
    }


def _validate_parent_provenance(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["parent_rollout_id"]].append(record)
    for parent, values in grouped.items():
        outcomes = {value["parent_rollout_failed"] for value in values}
        tasks = {value["parent_task_name"] for value in values}
        if len(outcomes) != 1 or len(tasks) != 1:
            raise ValueError(f"Parent rollout {parent} has inconsistent provenance")
    return grouped


def _parent_contributions(parent_records, selected_stages):
    counts = Counter()
    for record in parent_records:
        if record["stage"] in selected_stages:
            counts[(record["stage"], bool(record["causal_target_failed"]))] += 1
    return counts


def _select_greedily(candidates, contributions, deficits, *, seed, admissible=None):
    remaining = set(candidates)
    order = sorted(remaining)
    random.Random(int(seed)).shuffle(order)
    rank = {parent: index for index, parent in enumerate(order)}
    selected = []
    while any(value > 0 for value in deficits.values()):
        best = None
        best_key = None
        for parent in remaining:
            if admissible is not None and not admissible(parent, remaining):
                continue
            contribution = contributions[parent]
            gain = sum(
                min(deficits.get(key, 0), count)
                for key, count in contribution.items()
            )
            if gain <= 0:
                continue
            overshoot = sum(
                max(0, count - deficits.get(key, 0))
                for key, count in contribution.items()
            )
            key = (gain, -overshoot, -rank[parent])
            if best_key is None or key > best_key:
                best, best_key = parent, key
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        for key, count in contributions[best].items():
            deficits[key] = max(0, deficits.get(key, 0) - count)
    return selected


def _counts(records, stages):
    grouped = {
        stage: {"successes": 0, "failures": 0, "segments": 0, "parents": set()}
        for stage in stages
    }
    for record in records:
        if record["stage"] not in grouped:
            continue
        values = grouped[record["stage"]]
        values["failures" if record["causal_target_failed"] else "successes"] += 1
        values["segments"] += 1
        values["parents"].add(record["parent_rollout_id"])
    return {
        stage: {**values, "parents": len(values["parents"])}
        for stage, values in grouped.items()
    }


def _fingerprint(records, prefix, failure_horizon):
    identity = [
        (
            record["source_segment_id"],
            record["parent_rollout_id"],
            record["stage"],
            record["source_segment_num_inferences"],
            record["source_segment_failed"],
        )
        for record in records
    ]
    payload = {
        "prefix": int(prefix),
        "failure_horizon": int(failure_horizon),
        "identity": sorted(identity),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_target_aware_allocation(
    env_records,
    *,
    stages,
    prefix=32,
    failure_horizon=128,
    calibration_successes_per_stage=5,
    evaluation_successes_per_stage=10,
    evaluation_failures_per_stage=10,
    seed=0,
    source_export=None,
):
    """Allocate whole parents while balancing the actual finite-horizon target."""
    if int(prefix) <= 0 or int(failure_horizon) <= 0:
        raise ValueError("Prefix and failure horizon must be positive")
    quotas = (
        int(calibration_successes_per_stage),
        int(evaluation_successes_per_stage),
        int(evaluation_failures_per_stage),
    )
    if any(value < 0 for value in quotas):
        raise ValueError("Allocation quotas cannot be negative")
    selected_stages = tuple(dict.fromkeys(str(value) for value in stages))
    if not selected_stages:
        raise ValueError("At least one semantic stage must be selected")

    eligible = []
    excluded_short = Counter()
    observed_stages = set()
    for _, env in env_records:
        name = stage_name(env)
        observed_stages.add(name)
        if name not in selected_stages:
            continue
        record = finite_horizon_record(
            env, prefix=int(prefix), failure_horizon=int(failure_horizon)
        )
        if record is None:
            excluded_short[name] += 1
        else:
            eligible.append(record)
    unknown = sorted(set(selected_stages) - observed_stages)
    if unknown:
        raise ValueError("Unknown requested stages: " + ", ".join(unknown))
    if not eligible:
        raise ValueError("No selected segment reaches the requested causal prefix")

    parents = _validate_parent_provenance(eligible)
    selected_set = set(selected_stages)
    contributions = {
        parent: _parent_contributions(values, selected_set)
        for parent, values in parents.items()
    }
    calibration_candidates = [
        parent
        for parent, values in parents.items()
        if not values[0]["parent_rollout_failed"]
        and not any(value["causal_target_failed"] for value in values)
    ]
    calibration_deficits = {
        (stage, False): int(calibration_successes_per_stage)
        for stage in selected_stages
    }
    non_calibration_candidates = set(parents) - set(calibration_candidates)

    # Calibration cannot consume the negative examples reserved for evaluation.
    def preserves_evaluation_successes(parent, remaining):
        after = non_calibration_candidates | (set(remaining) - {parent})
        for stage in selected_stages:
            available = sum(
                contributions[key].get((stage, False), 0)
                for key in after
            )
            if available < int(evaluation_successes_per_stage):
                return False
        return True

    calibration_parents = _select_greedily(
        calibration_candidates,
        contributions,
        calibration_deficits,
        seed=seed,
        admissible=preserves_evaluation_successes,
    )
    calibration_parent_set = set(calibration_parents)
    evaluation_candidates = sorted(set(parents) - calibration_parent_set)
    evaluation_deficits = {
        **{
            (stage, False): int(evaluation_successes_per_stage)
            for stage in selected_stages
        },
        **{
            (stage, True): int(evaluation_failures_per_stage)
            for stage in selected_stages
        },
    }
    evaluation_parents = _select_greedily(
        evaluation_candidates,
        contributions,
        evaluation_deficits,
        seed=int(seed) + 1,
    )
    evaluation_parent_set = set(evaluation_parents)
    calibration_records = [
        record
        for record in eligible
        if record["parent_rollout_id"] in calibration_parent_set
    ]
    evaluation_records = [
        record
        for record in eligible
        if record["parent_rollout_id"] in evaluation_parent_set
    ]
    calibration_counts = _counts(calibration_records, selected_stages)
    evaluation_counts = _counts(evaluation_records, selected_stages)

    rows = []
    for stage in selected_stages:
        cal = calibration_counts[stage]
        evaluation = evaluation_counts[stage]
        cal_success_deficit = max(
            0, int(calibration_successes_per_stage) - cal["successes"]
        )
        eval_success_deficit = max(
            0, int(evaluation_successes_per_stage) - evaluation["successes"]
        )
        eval_failure_deficit = max(
            0, int(evaluation_failures_per_stage) - evaluation["failures"]
        )
        rows.append(
            {
                "stage": stage,
                "parent_task_name": stage.split("::", 1)[0],
                "eligible_successes": sum(
                    not record["causal_target_failed"]
                    for record in eligible
                    if record["stage"] == stage
                ),
                "eligible_failures": sum(
                    record["causal_target_failed"]
                    for record in eligible
                    if record["stage"] == stage
                ),
                "excluded_shorter_than_prefix": excluded_short[stage],
                "calibration_successes": cal["successes"],
                "calibration_failures": cal["failures"],
                "evaluation_successes": evaluation["successes"],
                "evaluation_failures": evaluation["failures"],
                "calibration_success_deficit": cal_success_deficit,
                "evaluation_success_deficit": eval_success_deficit,
                "evaluation_failure_deficit": eval_failure_deficit,
                "total_deficit": (
                    cal_success_deficit
                    + eval_success_deficit
                    + eval_failure_deficit
                ),
                "collection_priority": (
                    "collect_target_failures"
                    if eval_failure_deficit
                    else "collect_target_successes"
                    if cal_success_deficit + eval_success_deficit
                    else "ready"
                ),
            }
        )
    complete = all(row["total_deficit"] == 0 for row in rows)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_export": (
            None if source_export is None else str(Path(source_export).resolve())
        ),
        "source_fingerprint": _fingerprint(eligible, prefix, failure_horizon),
        "split_unit": "parent_rollout",
        "allocation_seed": int(seed),
        "target_definition": {
            "label": "failure_within_H",
            "formula": "source_segment_failed and (L - prefix <= H)",
            "causal_prefix_inferences": int(prefix),
            "failure_horizon_inferences": int(failure_horizon),
        },
        "calibration_eligibility": (
            "overall-successful parent rollouts with no positive selected-stage target"
        ),
        "selected_stages": list(selected_stages),
        "quotas_per_stage": {
            "calibration_successes": int(calibration_successes_per_stage),
            "evaluation_successes": int(evaluation_successes_per_stage),
            "evaluation_failures": int(evaluation_failures_per_stage),
        },
        "complete": complete,
        "calibration_parent_ids": sorted(calibration_parent_set),
        "evaluation_parent_ids": sorted(evaluation_parent_set),
        "unassigned_parent_ids": sorted(
            set(parents) - calibration_parent_set - evaluation_parent_set
        ),
        "calibration_segment_ids": sorted(
            record["causal_segment_id"] for record in calibration_records
        ),
        "evaluation_segment_ids": sorted(
            record["causal_segment_id"] for record in evaluation_records
        ),
        "counts": {
            "eligible_segments": len(eligible),
            "eligible_parents": len(parents),
            "calibration_segments": len(calibration_records),
            "calibration_parents": len(calibration_parent_set),
            "evaluation_segments": len(evaluation_records),
            "evaluation_parents": len(evaluation_parent_set),
            "unassigned_parents": len(
                set(parents) - calibration_parent_set - evaluation_parent_set
            ),
            "total_deficit": sum(row["total_deficit"] for row in rows),
        },
        "per_stage": rows,
        "audit": {
            "calibration_evaluation_parent_disjoint": not bool(
                calibration_parent_set & evaluation_parent_set
            ),
            "calibration_contains_target_failures": any(
                record["causal_target_failed"] for record in calibration_records
            ),
            "calibration_contains_failed_parents": any(
                record["parent_rollout_failed"] for record in calibration_records
            ),
        },
        "collection_guidance": (
            "Collect natural complete parent rollouts for stages with deficits, then "
            "rerun this allocator. Deficits are finite-horizon stage labels, not "
            "rollout-level outcomes or exact additional rollout counts."
        ),
    }


def write_allocation(allocation, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "target_aware_allocation.json"
    csv_path = output / "target_aware_collection_plan.csv"
    write_json(json_path, allocation)
    rows = allocation["per_stage"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["stage"])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--failure-horizon", type=int, default=128)
    parser.add_argument("--calibration-successes-per-stage", type=int, default=5)
    parser.add_argument("--evaluation-successes-per-stage", type=int, default=10)
    parser.add_argument("--evaluation-failures-per-stage", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    allocation = build_target_aware_allocation(
        load_env_records(args.export_dir),
        stages=args.stages,
        prefix=args.prefix,
        failure_horizon=args.failure_horizon,
        calibration_successes_per_stage=args.calibration_successes_per_stage,
        evaluation_successes_per_stage=args.evaluation_successes_per_stage,
        evaluation_failures_per_stage=args.evaluation_failures_per_stage,
        seed=args.seed,
        source_export=args.export_dir,
    )
    json_path, csv_path = write_allocation(allocation, args.output_dir)
    print(
        json.dumps(
            {
                "complete": allocation["complete"],
                "total_deficit": allocation["counts"]["total_deficit"],
                "allocation": str(json_path),
                "collection_plan": str(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
