"""Analyze the registered Phase 3B training-horizon continuation.

The primary unit is a snapshot nested in a parent and task.  The four
candidate treatment seeds define realized random-of-four and oracle-best-of-
four completion; the two identical nominal repeats are collapsed only after
exact outcome agreement.  Sentinel scientific outcomes are diagnostic and can
never gate progression to the full registered screen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from robocasa.recovery.counterfactual_branch import load_branch_records
from robocasa.recovery.phase3b_training_horizon import (
    atomic_write_json,
    completion_event,
    load_branch_payload,
    route_phase3b,
    sha256_file,
    validate_registration_source,
    write_jsonl,
)


def _mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _predicate(registration, task_name):
    return registration["trigger_stages"][task_name].split("::", 1)[-1]


def _outcome(root, registration, record):
    payload = load_branch_payload(root, record)
    event = completion_event(payload, _predicate(registration, record["task_name"]))
    trace = payload.get("subtask_trace") or []
    initial_progress = float(
        (payload.get("subtask_evals") or [{}])[0].get("subtask_progress", 0.0)
    )
    final_progress = float(
        (payload.get("subtask_evals") or [{}])[-1].get("subtask_progress", 0.0)
    )
    return {
        "branch_id": record["branch_id"],
        "source_branch_id": record["source_branch_id"],
        "snapshot_id": record["snapshot_id"],
        "parent_id": record["parent_id"],
        "task_name": record["task_name"],
        "boundary": record["boundary"],
        "kind": record["kind"],
        "sampling_seed": int(record["sampling_seed"]),
        "stage_completed": bool(event["completed"] or record.get("task_success")),
        "task_success": bool(record.get("task_success")),
        "first_completion_environment_step": event[
            "first_completion_environment_step"
        ],
        "first_completion_replan": event["first_completion_replan"],
        "durable_to_branch_end": event["durable_to_branch_end"],
        "regressed_after_completion": event["regressed_after_completion"],
        "regression": bool(record.get("regressed_predicates")),
        "safety_termination": record.get("termination_reason")
        == "safety_termination",
        "termination_reason": record.get("termination_reason"),
        "num_steps": int(record.get("num_steps", 0)),
        "num_policy_requests": int(record.get("num_policy_requests", 0)),
        "initial_progress": initial_progress,
        "final_progress": final_progress,
        "progress_gain": final_progress - initial_progress,
        "final_stage": trace[-1].get("ordered_current_subtask") if trace else None,
    }


def _snapshot_row(snapshot_id, outcomes):
    kinds = defaultdict(list)
    for row in outcomes:
        kinds[row["kind"]].append(row)
    if len(kinds["candidate"]) != 4 or len(kinds["same_seed_repeat"]) != 2:
        raise ValueError(f"Invalid scientific branch support for {snapshot_id}")
    repeats = sorted(kinds["same_seed_repeat"], key=lambda row: row["branch_id"])
    nominal_keys = (
        "stage_completed",
        "task_success",
        "regression",
        "safety_termination",
        "first_completion_environment_step",
        "durable_to_branch_end",
    )
    if any(repeats[0][key] != repeats[1][key] for key in nominal_keys):
        raise ValueError(f"Nominal long-horizon repeats disagree for {snapshot_id}")
    nominal = repeats[0]
    candidates = sorted(kinds["candidate"], key=lambda row: row["sampling_seed"])
    completion = np.asarray(
        [row["stage_completed"] for row in candidates], dtype=np.float64
    )
    task_success = np.asarray(
        [row["task_success"] for row in candidates], dtype=np.float64
    )
    regression = np.asarray(
        [row["regression"] for row in candidates], dtype=np.float64
    )
    safety = np.asarray(
        [row["safety_termination"] for row in candidates], dtype=np.float64
    )
    completed = [row for row in candidates if row["stage_completed"]]
    durable = (
        float(np.mean([row["durable_to_branch_end"] for row in completed]))
        if completed
        else 0.0
    )
    row = {
        "snapshot_id": snapshot_id,
        "parent_id": nominal["parent_id"],
        "task_name": nominal["task_name"],
        "boundary": nominal["boundary"],
        "mixed_outcome": bool(float(np.min(completion)) != float(np.max(completion))),
        "random_of_four_stage_completion": float(np.mean(completion)),
        "oracle_best_of_four_stage_completion": float(np.max(completion)),
        "nominal_stage_completion": float(nominal["stage_completed"]),
        "random_minus_nominal": float(np.mean(completion))
        - float(nominal["stage_completed"]),
        "oracle_minus_random": float(np.max(completion) - np.mean(completion)),
        "oracle_minus_nominal": float(np.max(completion))
        - float(nominal["stage_completed"]),
        "candidate_task_success": float(np.mean(task_success)),
        "oracle_task_success": float(np.max(task_success)),
        "candidate_regression_rate": float(np.mean(regression)),
        "nominal_regression": float(nominal["regression"]),
        "regression_increase_vs_nominal": float(np.mean(regression))
        - float(nominal["regression"]),
        "candidate_safety_termination_rate": float(np.mean(safety)),
        "durable_completion_fraction": durable,
        "candidate_progress_gain": float(
            np.mean([candidate["progress_gain"] for candidate in candidates])
        ),
        "nominal_progress_gain": float(nominal["progress_gain"]),
        "candidate_environment_steps_mean": float(
            np.mean([candidate["num_steps"] for candidate in candidates])
        ),
        "candidate_policy_requests_mean": float(
            np.mean([candidate["num_policy_requests"] for candidate in candidates])
        ),
    }
    return row, candidates, repeats


def _aggregate(rows):
    keys = (
        "mixed_outcome",
        "random_of_four_stage_completion",
        "oracle_best_of_four_stage_completion",
        "nominal_stage_completion",
        "random_minus_nominal",
        "oracle_minus_random",
        "oracle_minus_nominal",
        "candidate_task_success",
        "oracle_task_success",
        "candidate_regression_rate",
        "nominal_regression",
        "regression_increase_vs_nominal",
        "candidate_safety_termination_rate",
        "durable_completion_fraction",
        "candidate_progress_gain",
        "nominal_progress_gain",
        "candidate_environment_steps_mean",
        "candidate_policy_requests_mean",
    )
    return {"snapshots": len(rows), **{key: _mean(rows, key) for key in keys}}


def _task_macro(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_name"]].append(row)
    task_rows = {task: _aggregate(values) for task, values in sorted(grouped.items())}
    metric_keys = [key for key in next(iter(task_rows.values())) if key != "snapshots"]
    macro = {
        "tasks": len(task_rows),
        "snapshots": len(rows),
        **{
            key: float(np.mean([value[key] for value in task_rows.values()]))
            for key in metric_keys
        },
    }
    return task_rows, macro


def _bootstrap(rows, *, replicates, seed):
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["task_name"]][row["parent_id"]].append(row)
    tasks = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(replicates):
        replicate_tasks = []
        for task in tasks:
            parents = sorted(grouped[task])
            selected_parents = rng.choice(parents, size=len(parents), replace=True)
            selected_rows = []
            for parent in selected_parents:
                snapshots = grouped[task][parent]
                indices = rng.integers(0, len(snapshots), size=len(snapshots))
                selected_rows.extend(snapshots[index] for index in indices)
            replicate_tasks.append(_aggregate(selected_rows))
        for key in replicate_tasks[0]:
            if key == "snapshots":
                continue
            samples[key].append(
                float(np.mean([task_row[key] for task_row in replicate_tasks]))
            )
    return {
        key: {
            "mean": float(np.mean(values)),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
        for key, values in sorted(samples.items())
    }


def _completion_curves(candidate_rows, registration):
    curves = {}
    for task, horizon in registration["task_horizons_environment_steps"].items():
        task_candidates = [row for row in candidate_rows if row["task_name"] == task]
        max_replans = int(horizon) // int(registration.get("replan_steps", 16))
        curves[task] = [
            {
                "replan": replan,
                "environment_step": replan * int(registration.get("replan_steps", 16)),
                "completion_fraction": float(
                    np.mean(
                        [
                            row["first_completion_replan"] is not None
                            and row["first_completion_replan"] <= replan
                            for row in task_candidates
                        ]
                    )
                ),
                "risk_set": sum(
                    row["first_completion_replan"] is None
                    or row["first_completion_replan"] >= replan
                    for row in task_candidates
                ),
            }
            for replan in range(1, max_replans + 1)
        ]
    return curves


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir, *, bootstrap_replicates=2000, bootstrap_seed=0):
    root = Path(run_dir).resolve()
    registration = json.loads((root / "registration.json").read_text())
    engineering = json.loads((root / "analysis.json").read_text())
    if engineering.get("engineering_all_pass") is not True:
        raise ValueError("Scientific analysis requires an all-pass engineering audit")
    validate_registration_source(registration)
    before = sha256_file(root / "branch_records.jsonl")
    records = load_branch_records(root / "branch_records.jsonl")
    grouped = defaultdict(list)
    candidate_manifest = []
    repeat_manifest = []
    for record in records:
        grouped[record["snapshot_id"]].append(_outcome(root, registration, record))
    snapshot_rows = []
    for snapshot_id, outcomes in sorted(grouped.items()):
        row, candidates, repeats = _snapshot_row(snapshot_id, outcomes)
        snapshot_rows.append(row)
        candidate_manifest.extend(candidates)
        repeat_manifest.extend(repeats)
    task_rows, macro = _task_macro(snapshot_rows)
    bootstrap = _bootstrap(
        snapshot_rows, replicates=int(bootstrap_replicates), seed=int(bootstrap_seed)
    )
    mixed_rows = [row for row in snapshot_rows if row["mixed_outcome"]]
    mixed_tasks = {row["task_name"] for row in mixed_rows}
    mixed_parents = {row["parent_id"] for row in mixed_rows}
    full_scope = registration["scope"] == "full"
    gates = {
        "mixed_outcomes_at_least_4_snapshots": len(mixed_rows) >= 4,
        "mixed_outcomes_at_least_3_parents": len(mixed_parents) >= 3,
        "mixed_outcomes_in_both_tasks": mixed_tasks
        == set(registration["tasks"]),
        "task_macro_oracle_minus_random_at_least_0p15": macro[
            "oracle_minus_random"
        ]
        >= 0.15,
        "oracle_minus_nominal_positive_both_tasks": all(
            row["oracle_minus_nominal"] > 0.0 for row in task_rows.values()
        ),
        "task_macro_oracle_minus_nominal_at_least_0p10": macro[
            "oracle_minus_nominal"
        ]
        >= 0.10,
        "zero_catastrophic_safety_terminations": macro[
            "candidate_safety_termination_rate"
        ]
        == 0.0,
        "regression_increase_at_most_0p05": macro[
            "regression_increase_vs_nominal"
        ]
        <= 0.05,
        "completed_candidates_durable": all(
            row["durable_to_branch_end"]
            for row in candidate_manifest
            if row["stage_completed"]
        ),
    }
    route = route_phase3b(task_rows, macro, gates)
    after = sha256_file(root / "branch_records.jsonl")
    if before != after:
        raise RuntimeError("Phase 3B artifacts changed during analysis")
    result = {
        "schema_version": 1,
        "protocol": "phase3b_training_horizon_scientific_analysis",
        "status": "complete",
        "scope": registration["scope"],
        "formal_gate_estimable": full_scope,
        "parents": len({row["parent_id"] for row in snapshot_rows}),
        "snapshots": len(snapshot_rows),
        "candidate_branches": len(candidate_manifest),
        "task_summary": task_rows,
        "task_macro": macro,
        "registered_estimands": {
            "M_mixed_outcome_fraction": macro["mixed_outcome"],
            "O_oracle_minus_random": macro["oracle_minus_random"],
            "G_oracle_minus_nominal": macro["oracle_minus_nominal"],
            "B_random_minus_nominal": macro["random_minus_nominal"],
        },
        "bootstrap": bootstrap,
        "mixed_snapshot_count": len(mixed_rows),
        "mixed_parent_count": len(mixed_parents),
        "mixed_tasks": sorted(mixed_tasks),
        "continuation_gates": gates,
        "formal_continuation_gate_passed": full_scope and all(gates.values()),
        "routing_decision": route,
        "sentinel_scientific_outcomes_can_stop_full_run": False,
        "completion_curves": _completion_curves(candidate_manifest, registration),
        "engineering_analysis_sha256": sha256_file(root / "analysis.json"),
    }
    atomic_write_json(root / "scientific_analysis.json", result)
    write_jsonl(root / "phase3b_snapshot_results.jsonl", snapshot_rows)
    write_jsonl(root / "phase3b_candidate_results.jsonl", candidate_manifest)
    _write_csv(root / "phase3b_snapshot_results.csv", snapshot_rows)
    return result


def print_result(result):
    print("PHASE 3B TRAINING-HORIZON CONTINUATION")
    print("status:", result["status"])
    print("scope:", result["scope"])
    print("parents:", result["parents"])
    print("snapshots:", result["snapshots"])
    print("candidate branches:", result["candidate_branches"])
    print("formal gate estimable:", result["formal_gate_estimable"])
    print("\nTASK-MACRO OUTCOMES")
    macro = result["task_macro"]
    for key in (
        "mixed_outcome",
        "random_of_four_stage_completion",
        "oracle_best_of_four_stage_completion",
        "oracle_minus_random",
        "oracle_minus_nominal",
        "candidate_regression_rate",
        "candidate_safety_termination_rate",
    ):
        print(f"{key:44s} {macro[key]:.4f}")
    print("\nCONTINUATION GATES")
    for key, value in result["continuation_gates"].items():
        print(f"{'PASS' if value else 'FAIL':4s}  {key}")
    print("\nrouting decision:", result["routing_decision"])
    if result["scope"] == "sentinel":
        print("Sentinel outcomes are diagnostic and do not stop the full run.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = analyze(
        args.run_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print_result(result)


if __name__ == "__main__":
    main()
