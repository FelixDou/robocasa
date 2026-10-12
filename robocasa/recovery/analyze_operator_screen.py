"""Paired, task-macro analysis of the frozen XR1 operator screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from robocasa.recovery.operator_screen import OPERATORS, ORACLE, unique_records, validate_registration
from robocasa.recovery.phase3b_training_horizon import (
    atomic_write_json, completion_event, load_branch_payload, sha256_file, write_jsonl,
)


METRICS = ("completion", "durable_completion", "regression", "safety", "task_success", "policy_calls", "wall_seconds")


def branch_outcome(root, registration, record):
    payload = load_branch_payload(root, record)
    predicate = registration["phase2_registration"]["trigger_stages"][record["task_name"]].split("::", 1)[-1]
    event = completion_event(payload, predicate)
    safety = record["termination_reason"] == "safety_termination" or any(
        isinstance(info, dict) and (info.get("safety_violation") or info.get("safety_termination"))
        for info in payload.get("infos", [])
    )
    success = bool(record["task_success"])
    initial_predicates = (payload["subtask_evals"][0].get("predicates") or {})
    initially_complete = {name for name, value in initial_predicates.items()
                          if isinstance(value, dict) and value.get("value")}
    prior_regressions = initially_complete.intersection(record["regressed_predicates"])
    return {
        **{key: record[key] for key in ("branch_id", "snapshot_id", "parent_id", "task_name", "boundary", "operator", "schedule")},
        "completion": float(event["completed"] or success),
        "durable_completion": float(event["durable_to_branch_end"] or success),
        "regression": float(bool(prior_regressions)),
        "any_predicate_regression": float(bool(record["regressed_predicates"])),
        "safety": float(bool(safety)), "task_success": float(success),
        "policy_calls": record["num_policy_requests"], "wall_seconds": record["wall_seconds"],
        "first_completion_environment_step": event["first_completion_environment_step"],
        "num_steps": record["num_steps"],
    }


def snapshot_results(outcomes):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in outcomes:
        grouped[row["snapshot_id"]][row["operator"]].append(row)
    result = []
    for snapshot_id, operators in sorted(grouped.items()):
        if not set(OPERATORS).issubset(operators):
            raise ValueError("Snapshot is missing a deployable operator")
        first = operators["continue"][0]
        row = {key: first[key] for key in ("snapshot_id", "parent_id", "task_name", "boundary")}
        row["operators"] = {}
        for operator, records in operators.items():
            expected = {0} if operator == ORACLE else {0, 1}
            if len(records) != len(expected) or {r["schedule"] for r in records} != expected:
                raise ValueError("Missing or duplicate continuation-seed schedule")
            row["operators"][operator] = {
                key: float(np.mean([r[key] for r in records])) for key in METRICS
            }
        result.append(row)
    return result


def macro_summary(rows):
    tasks = sorted({x["task_name"] for x in rows})
    task_summary = {}
    for task in tasks:
        selected = [x for x in rows if x["task_name"] == task]
        task_summary[task] = {
            operator: {key: float(np.mean([r["operators"][operator][key] for r in selected])) for key in METRICS}
            for operator in OPERATORS
        }
    macro = {operator: {key: float(np.mean([task_summary[t][operator][key] for t in tasks])) for key in METRICS} for operator in OPERATORS}
    # Average the two realized seeds before taking a per-state maximum. This
    # avoids granting the routing oracle an extra per-seed selection privilege.
    best = max(OPERATORS, key=lambda op: macro[op]["completion"])
    oracle = float(np.mean([
        np.mean([max(r["operators"][op]["completion"] for op in OPERATORS) for r in rows if r["task_name"] == t])
        for t in tasks
    ]))
    contrasts = {
        op: {key: macro[op][key] - macro["continue"][key] for key in METRICS}
        for op in OPERATORS
    }
    return {"tasks": task_summary, "macro": macro, "contrasts_vs_continue": contrasts,
            "best_fixed_operator": best, "oracle_routing_completion": oracle,
            "oracle_routing_minus_best_fixed": oracle - macro[best]["completion"]}


def route(rows, summary, spec):
    gates = spec["gates"]
    by_operator = {}
    for operator in OPERATORS[1:]:
        contrast = summary["contrasts_vs_continue"][operator]
        parent_gains = defaultdict(list)
        for row in rows:
            parent_gains[row["parent_id"]].append(row["operators"][operator]["completion"] - row["operators"]["continue"]["completion"])
        benefiting_parents = sum(bool(np.mean(x) > 0) for x in parent_gains.values())
        tests = {
            "completion_gain": contrast["completion"] >= gates["operator_gain"] - 1e-12,
            "durable_gain": contrast["durable_completion"] >= gates["operator_gain"] - 1e-12,
            "positive_in_both_tasks": all(
                values[operator]["completion"] > values["continue"]["completion"]
                for values in summary["tasks"].values()
            ),
            "benefit_in_multiple_parents": benefiting_parents >= gates["min_benefiting_parents"],
            "regression_guardrail": contrast["regression"] <= gates["max_regression_increase"] + 1e-12,
            "zero_declared_safety_events": summary["macro"][operator]["safety"] == 0,
        }
        by_operator[operator] = {"gates": tests, "passed": all(tests.values()), "benefiting_parents": benefiting_parents}
    promising = [op for op in OPERATORS[1:] if by_operator[op]["passed"]]
    decision = "mechanism_redesign_or_earlier_snapshot_study"
    if promising:
        # The optimistic routing arm must satisfy the same durability, safety,
        # regression and multi-parent requirements as a fixed operator.
        best_fixed = summary["best_fixed_operator"]
        router_rows = []
        routing_parent_gains = defaultdict(list)
        for row in rows:
            selected = max(OPERATORS, key=lambda op: row["operators"][op]["completion"])
            chosen = row["operators"][selected]
            router_rows.append({**row, "chosen": chosen})
            routing_parent_gains[row["parent_id"]].append(chosen["completion"] - row["operators"][best_fixed]["completion"])
        router = {key: float(np.mean([
            np.mean([r["chosen"][key] for r in router_rows if r["task_name"] == task])
            for task in summary["tasks"]
        ])) for key in METRICS}
        routing_gates = {
            "routing_gain": summary["oracle_routing_minus_best_fixed"] >= gates["routing_gain"] - 1e-12,
            "durable_routing_gain": router["durable_completion"] - summary["macro"][best_fixed]["durable_completion"] >= gates["routing_gain"] - 1e-12,
            "regression_guardrail": router["regression"] - summary["macro"][best_fixed]["regression"] <= gates["max_regression_increase"] + 1e-12,
            "zero_declared_safety_events": router["safety"] == 0,
            "multiple_parents": sum(bool(np.mean(x) > 0) for x in routing_parent_gains.values()) >= gates["min_benefiting_parents"],
            "positive_in_both_tasks": all(
                np.mean([r["chosen"]["completion"] - r["operators"][best_fixed]["completion"] for r in router_rows if r["task_name"] == task]) > 0
                for task in summary["tasks"]
            ),
        }
        decision = "operator_router_development" if all(routing_gates.values()) else "fixed_operator_fresh_parent_validation"
    else:
        routing_gates = None
    return {"decision": decision, "operator_gates": by_operator, "routing_gates": routing_gates,
            "promising_operators": promising,
            "recommended_fixed_operator": max(promising, key=lambda op: summary["macro"][op]["completion"]) if promising else None}


def bootstrap(rows, replicates=2000, seed=0):
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["task_name"]][row["parent_id"]].append(row)
    samples = defaultdict(list)
    for _ in range(replicates):
        selected = []
        for task, parent_rows in sorted(grouped.items()):
            parents = sorted(parent_rows)
            for parent in rng.choice(parents, size=len(parents), replace=True):
                snapshots = parent_rows[parent]
                selected.extend(snapshots[i] for i in rng.integers(0, len(snapshots), size=len(snapshots)))
        summary = macro_summary(selected)
        for op in OPERATORS[1:]:
            samples[f"{op}_minus_continue"].append(summary["contrasts_vs_continue"][op]["completion"])
        samples["oracle_routing_minus_best_fixed"].append(summary["oracle_routing_minus_best_fixed"])
    return {key: {"estimate_mean": float(np.mean(values)), "ci95": np.quantile(values, [0.025, 0.975]).tolist()} for key, values in samples.items()}


def analyze(root, *, bootstrap_replicates=2000, bootstrap_seed=0):
    from robocasa.recovery.run_operator_screen import engineering_analysis
    root = Path(root).resolve()
    registration = json.loads((root / "registration.json").read_text())
    validate_registration(registration)
    engineering = engineering_analysis(root, registration)
    if not engineering["engineering_all_pass"]:
        raise ValueError("Scientific analysis requires a complete passing engineering audit")
    before = sha256_file(root / "branch_records.jsonl")
    outcomes = [branch_outcome(root, registration, row) for row in unique_records(root, registration).values() if row["role"] == "primary"]
    rows = snapshot_results(outcomes)
    summary = macro_summary(rows)
    routing = route(rows, summary, registration["spec"])
    result = {
        "protocol": registration["protocol"], "scope": registration["scope"],
        "evidence_type": "development_operator_capability_screen_fixed_supervisor_subgoals",
        "parents": len({x["parent_id"] for x in rows}), "snapshots": len(rows),
        "scientific_gate_estimable": registration["scope"] == "full",
        **summary, **routing,
        "bootstrap": bootstrap(rows, bootstrap_replicates, bootstrap_seed),
        "bootstrap_design": "fixed_two_tasks_parent_then_snapshot_paired_all_arms_and_both_seeds",
        "uncertainty_scope": "developmental_and_conditional_on_these_two_tasks_not_population_or_confirmation",
        "oracle_routing_is_optimistic_finite_seed_headroom": True,
        "eventual_success_control_harm_estimable": False,
        "oracle_rewind": [r for r in outcomes if r["operator"] == ORACLE],
        "source_ledger_sha256": before,
    }
    if registration["scope"] == "sentinel":
        result["decision"] = "engineering_only_run_full_regardless_of_efficacy"
    if sha256_file(root / "branch_records.jsonl") != before:
        raise ValueError("Source branch ledger changed during analysis")
    validate_registration(registration)
    write_jsonl(root / "operator_branch_results.jsonl", outcomes)
    write_jsonl(root / "operator_snapshot_results.jsonl", rows)
    with (root / "operator_branch_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    atomic_write_json(root / "scientific_analysis.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.run_dir, bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed), indent=2))


if __name__ == "__main__":
    main()
