"""Paired development-only analysis of all nine common-entry timing cells."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from robocasa.recovery.exact_replay.intervention_timing import PROTOCOL, unique_records
from robocasa.recovery.exact_replay.operator_screen import OPERATORS
from robocasa.recovery.exact_replay.phase3b_training_horizon import (
    completion_event, load_branch_payload, atomic_write_json, sha256_file, write_jsonl,
)
from robocasa.recovery.exact_replay.run_intervention_timing import engineering_analysis


CELLS = ("continue",) + tuple(f"{op}--d{d}" for op in OPERATORS[1:] for d in (0, 1, 2))
METRICS = ("completion", "durable_completion", "ordered_prefix_regression", "legacy_regression",
           "any_predicate_regression", "safety", "task_success", "policy_calls", "wall_seconds", "intervention_executed")


def branch_outcome(root, registration, record):
    payload = load_branch_payload(root, record)
    predicate = registration["phase2_registration"]["trigger_stages"][record["task_name"]].split("::", 1)[-1]
    event = completion_event(payload, predicate)
    initial = {p for p, v in payload["subtask_evals"][0]["predicates"].items() if v.get("value")}
    regressed = set(record["regressed_predicates"])
    prior = initial & set(record["ordered_prefix"])
    safety = record["termination_reason"] == "safety_termination" or any(
        isinstance(i, dict) and (i.get("safety_violation") or i.get("safety_termination")) for i in payload["infos"])
    cell = "continue" if record["delay"] is None else f"{record['operator']}--d{record['delay']}"
    return {**{k: record[k] for k in ("branch_id", "parent_id", "task_name", "operator", "delay", "schedule")},
            "cell": cell, "completion": float(event["completed"]),
            "durable_completion": float(event["durable_to_branch_end"]),
            "ordered_prefix_regression": float(bool(prior & regressed)),
            "legacy_regression": float(bool(initial & regressed)),
            "any_predicate_regression": float(bool(regressed)), "safety": float(bool(safety)),
            "task_success": float(record["task_success"]), "policy_calls": record["num_policy_requests"],
            "wall_seconds": record["wall_seconds"], "intervention_executed": float(record["intervention"]["executed"]),
            "intervention_reason": record["intervention"]["reason"],
            "application_environment_step": record["intervention"]["environment_step"],
            "first_completion_environment_step": event["first_completion_environment_step"],
            "num_steps": record["num_steps"]}


def parent_results(outcomes):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in outcomes:
        grouped[row["parent_id"]][row["cell"]].append(row)
    rows = []
    for parent, cells in sorted(grouped.items()):
        if set(cells) != set(CELLS):
            raise ValueError("Missing or unknown parent/timing cell")
        first = cells["continue"][0]
        row = {"parent_id": parent, "task_name": first["task_name"], "cells": {}}
        for cell, values in cells.items():
            if len(values) != 2 or {v["schedule"] for v in values} != {0, 1}:
                raise ValueError("Missing or duplicate seed schedule")
            row["cells"][cell] = {m: float(np.mean([v[m] for v in values])) for m in METRICS}
        rows.append(row)
    return rows


def macro_summary(rows):
    tasks = sorted({r["task_name"] for r in rows})
    task_values = {t: {c: {m: float(np.mean([r["cells"][c][m] for r in rows if r["task_name"] == t]))
                               for m in METRICS} for c in CELLS} for t in tasks}
    macro = {c: {m: float(np.mean([task_values[t][c][m] for t in tasks])) for m in METRICS} for c in CELLS}
    contrasts = {c: {m: macro[c][m]-macro["continue"][m] for m in METRICS} for c in CELLS[1:]}
    timing = {f"{op}--d{a}_minus_d{b}": {m: macro[f"{op}--d{a}"][m]-macro[f"{op}--d{b}"][m] for m in METRICS}
              for op in OPERATORS[1:] for a, b in ((0, 1), (0, 2), (1, 2))}
    return {"tasks": task_values, "macro": macro, "contrasts_vs_continue": contrasts,
            "earlier_minus_later": timing, "best_observed_cell": max(CELLS, key=lambda c: macro[c]["completion"])}


def screen(rows, summary, gates):
    tests = {}
    for cell in CELLS[1:]:
        delta = summary["contrasts_vs_continue"][cell]
        benefiting = sum(r["cells"][cell]["completion"] > r["cells"]["continue"]["completion"] for r in rows)
        criteria = {
            "completion_gain": delta["completion"] >= gates["operator_gain"]-1e-12,
            "durable_gain": delta["durable_completion"] >= gates["operator_gain"]-1e-12,
            "positive_in_both_tasks": all(t[cell]["completion"] > t["continue"]["completion"] for t in summary["tasks"].values()),
            "multiple_benefiting_parents": benefiting >= gates["min_benefiting_parents"],
            "ordered_prefix_regression_guardrail": delta["ordered_prefix_regression"] <= gates["max_regression_increase"]+1e-12,
            "zero_declared_safety_events": summary["macro"][cell]["safety"] == 0,
        }
        tests[cell] = {"gates": criteria, "passed": all(criteria.values()), "benefiting_parents": benefiting,
                       "legacy_regression_guardrail_sensitivity": delta["legacy_regression"] <= gates["max_regression_increase"]+1e-12}
    promising = [c for c in CELLS[1:] if tests[c]["passed"]]
    return {"cell_gates": tests, "promising_cells": promising,
            "recommended_fresh_validation_cell": max(promising, key=lambda c: summary["macro"][c]["completion"]) if promising else None,
            "decision": "fresh_parent_operator_and_timing_validation" if promising else "mechanism_redesign_or_diagnostic_benchmark"}


def bootstrap(rows, replicates=2000, seed=0):
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    # Keep task identities fixed and keep all regimes + both averaged seeds
    # paired when a parent is selected. Nine-cell selection remains exploratory.
    arrays = {t: np.array([[[r["cells"][c][m] for m in METRICS] for c in CELLS]
                          for r in rows if r["task_name"] == t]) for t in sorted({r["task_name"] for r in rows})}
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(replicates):
        macro = np.mean([a[rng.integers(0, len(a), size=len(a))].mean(axis=0) for a in arrays.values()], axis=0)
        for ci, cell in enumerate(CELLS[1:], 1):
            samples[f"{cell}_minus_continue"].append(macro[ci]-macro[0])
        for op in OPERATORS[1:]:
            for a, b in ((0, 1), (0, 2), (1, 2)):
                samples[f"{op}--d{a}_minus_d{b}"].append(macro[CELLS.index(f"{op}--d{a}")]-macro[CELLS.index(f"{op}--d{b}")])
    return {contrast: {m: {"bootstrap_mean": float(np.mean(values, axis=0)[i]),
                          "ci95": np.quantile(np.asarray(values)[:, i], [0.025, 0.975]).tolist()}
                       for i, m in enumerate(METRICS)} for contrast, values in samples.items()}


def analyze(root, *, bootstrap_replicates=2000, bootstrap_seed=0):
    root = Path(root).resolve()
    registration = json.loads((root/"registration.json").read_text())
    engineering = engineering_analysis(root, registration, "full")
    if not engineering["engineering_all_pass"]:
        raise ValueError("Scientific analysis requires all 200 primaries and passing engineering repeats")
    before = sha256_file(root/"branch_records.jsonl")
    outcomes = [branch_outcome(root, registration, r) for r in unique_records(root, registration).values() if r["role"] == "primary"]
    rows = parent_results(outcomes)
    summary = macro_summary(rows)
    reasons = {cell: dict(Counter(r["intervention_reason"] for r in outcomes if r["cell"] == cell)) for cell in CELLS}
    result = {"protocol": PROTOCOL, "parents": len(rows), "primary_branches": len(outcomes),
              "evidence_type": "exploratory_development_common_stage_entry_timing_supervisor_subgoals",
              **summary, **screen(rows, summary, registration["operator_spec"]["gates"]),
              "bootstrap": bootstrap(rows, bootstrap_replicates, bootstrap_seed),
              "bootstrap_design": "fixed_tasks_resample_parents_paired_all_cells_and_both_seeds_averaged_first",
              "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
              "assignment_analysis": "intention_to_treat_including_ineligible_and_completed_fallbacks",
              "intervention_reasons": reasons,
              "ordered_prefix_regression_definition": "captured_upstream_ordered_prefix_predicates_true_at_entry_then_false; predicate_proxy_not_necessarily_stage_loss",
              "legacy_regression_definition": "any_required_predicate_true_at_entry_then_false",
              "uncertainty_scope": "opened_development_parents_two_tasks_no_multiplicity_adjustment_fresh_validation_required",
              "safe_triggered_recovery_tested": False,
              "source_ledger_sha256": before, "registration_sha256": registration["registration_sha256"]}
    if sha256_file(root/"branch_records.jsonl") != before:
        raise ValueError("Ledger changed during analysis")
    write_jsonl(root/"timing_branch_results.jsonl", outcomes)
    write_jsonl(root/"timing_parent_results.jsonl", rows)
    with (root/"timing_branch_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
        writer.writeheader(); writer.writerows(outcomes)
    atomic_write_json(root/"scientific_analysis.json", result)
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
