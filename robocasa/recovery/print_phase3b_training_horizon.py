"""Print a running or final Phase 3B continuation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robocasa.recovery.counterfactual_branch import load_branch_records
from robocasa.recovery.phase3b_training_horizon import read_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_dir
    registration = json.loads((root / "registration.json").read_text())
    status = (
        json.loads((root / "status.json").read_text())
        if (root / "status.json").is_file()
        else {"status": "registered"}
    )
    records = load_branch_records(root / "branch_records.jsonl")
    errors = read_jsonl(root / "errors.jsonl")
    print("PHASE 3B TRAINING-HORIZON CONTINUATION")
    print("status:", status["status"])
    print("scope:", registration["scope"])
    print("parents:", len({row["parent_id"] for row in records}), "/", registration["expected_parents"])
    print("snapshots:", len({row["snapshot_id"] for row in records}), "/", registration["expected_snapshots"])
    print("branches:", len(records), "/", registration["expected_branches"])
    print("first-64 exact:", sum(bool(row.get("first64_exact")) for row in records), "/", len(records))
    print("errors:", len(errors))
    if status["status"] == "failed":
        print("failure:", status.get("error_type"), status.get("error"))
    if not (root / "analysis.json").is_file():
        print("Engineering analysis is not frozen yet.")
        return
    analysis = json.loads((root / "analysis.json").read_text())
    print("\nENGINEERING GATES")
    for name, passed in analysis["engineering_gates"].items():
        print(f"{'PASS' if passed else 'FAIL':4s}  {name}")
    print(
        "\nPHASE 3B ENGINEERING VALIDITY:",
        "PASS" if analysis["engineering_all_pass"] else "FAIL",
    )
    scientific_path = root / "scientific_analysis.json"
    if scientific_path.is_file():
        scientific = json.loads(scientific_path.read_text())
        print("routing decision:", scientific["routing_decision"])


if __name__ == "__main__":
    main()
