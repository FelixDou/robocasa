"""Print a running or completed Phase 2 snapshot-replay audit.

This command intentionally exits normally when scientific gates fail.  A
failed gate is an experimental result, not a shell or terminal failure.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from robocasa.recovery.counterfactual_branch import load_branch_records


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def _read_jsonl(path):
    try:
        with Path(path).open() as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        return []


def print_report(run_dir):
    root = Path(run_dir)
    plan = _read_json(root / "plan.json", {})
    status = _read_json(root / "status.json", {"status": "not_started"})
    analysis = _read_json(root / "analysis.json")
    records = load_branch_records(root / "branch_records.jsonl")
    ineligible = _read_jsonl(root / "ineligible_parent_records.jsonl")
    kinds = Counter(row["kind"] for row in records)

    print("PHASE 2 COMPLETE-SNAPSHOT REPLAY")
    print("status:", status.get("status"))
    print("tasks:", " ".join(plan.get("tasks", [])) or "unknown")
    print(
        "parents:",
        status.get("parents", status.get("parents_completed", 0)),
        "/",
        plan.get("expected_parents", "?"),
    )
    print(
        "snapshots:",
        status.get("snapshots", 0),
        "/",
        plan.get("expected_snapshots", "?"),
    )
    print(
        "branches:",
        len(records),
        "/",
        plan.get("expected_total_branches", "?"),
        dict(sorted(kinds.items())),
    )
    print("errors:", status.get("errors", 0))
    if ineligible:
        print("\nREACHABILITY")
        print("ineligible parents:", len(ineligible))
        print(
            "target stage reached:",
            sum(bool(row.get("target_stage_reached")) for row in ineligible),
        )
        print(
            "captured prefix only:",
            sum(row.get("captured_count") == 1 for row in ineligible),
        )
        print(
            "captured nothing:",
            sum(row.get("captured_count") == 0 for row in ineligible),
        )
        latest = ineligible[-1]
        print(
            "latest observed stages:",
            " -> ".join(latest.get("observed_stage_sequence", [])) or "none",
        )
        print(
            "latest target max consecutive inferences:",
            latest.get("target_stage_max_consecutive_policy_inferences", 0),
        )

    if analysis is None:
        print(
            "\nAnalysis is not frozen yet; collection is still running or stopped early."
        )
        return

    print("\nENGINEERING VALIDITY GATES")
    for name, passed in analysis.get("gates", {}).items():
        print(f"{'PASS' if passed else 'FAIL':4s}  {name}")
    print("\nDIAGNOSTICS")
    print(
        "same-seed suffix agreement:",
        f"{analysis.get('same_seed_suffix_outcome_agreement', 0.0):.4f}",
    )
    print(
        "candidate-diverse snapshots:",
        f"{analysis.get('candidate_diversity_rate', 0.0):.4f}",
    )
    print("branch error rate:", f"{analysis.get('error_rate', 0.0):.4f}")
    failed_pairs = [
        row
        for row in analysis.get("repeat_pairs", [])
        if not row.get("valid")
        or not all(
            row.get(key, False)
            for key in (
                "request_exact",
                "action_exact",
                "action_sequence_exact",
                "transition_exact",
                "observation_exact",
                "causal_environment_sequence_exact",
                "suffix_outcome_equal",
            )
        )
    ]
    print("non-reproducing repeat pairs:", len(failed_pairs))
    for row in failed_pairs:
        failed_components = [
            name
            for name, exact in row.get("transition_components_exact", {}).items()
            if not exact
        ]
        print(
            "  transition mismatch:",
            row.get("snapshot_id"),
            "components=" + (",".join(failed_components) or "unknown"),
        )
    diagnostic_only = [
        row
        for row in analysis.get("repeat_pairs", [])
        if row.get("transition_exact")
        and not row.get("diagnostic_transition_exact", True)
    ]
    print("diagnostic-cache-only differences:", len(diagnostic_only))
    for row in diagnostic_only:
        failed_components = [
            name
            for name, exact in row.get(
                "diagnostic_transition_components_exact", {}
            ).items()
            if not exact
        ]
        print(
            "  diagnostic mismatch:",
            row.get("snapshot_id"),
            "components=" + (",".join(failed_components) or "unknown"),
        )
    print("\nPHASE 2 VALIDITY:", "PASS" if analysis.get("all_pass") else "FAIL")
    print("artifacts:", root)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print_report(args.run_dir)


if __name__ == "__main__":
    main()
