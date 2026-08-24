"""Audit task-specific Phase 2 replay roots as one frozen validation cohort.

The command is read-only with respect to its source runs. It selects completed
parents for the explicitly declared task in each source, verifies every
snapshot and branch payload reference, and reruns the engineering gates over
the combined branch summaries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from robocasa.recovery.counterfactual_branch import (
    analyze_phase2_replay,
    load_branch_records,
)


COMPATIBILITY_KEYS = (
    "runtime_bundle_sha256",
    "checkpoint_provenance_sha256",
    "server_repository_commit",
    "robocasa_commit",
    "policy_module",
    "policy_name",
    "checkpoint",
    "checkpoint_revision",
    "candidate_count",
    "suffix_steps",
    "snapshot_prefix",
    "landmark_fraction",
    "env_interface",
    "canonical_camera_observations",
    "canonical_camera_render_repeats",
    "replan_steps",
    "split",
)


def _read_json(path: Path, default: Any = None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _read_jsonl(path: Path):
    try:
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        return []


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: Path):
    return _sha256(path) if path.is_file() else None


def _atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_source(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be TASK=RUN_DIR")
    task, raw_path = value.split("=", 1)
    task = task.strip()
    if not task or not raw_path.strip():
        raise argparse.ArgumentTypeError("--source must be TASK=RUN_DIR")
    return task, Path(raw_path).expanduser().resolve()


def _compatible_values(plans):
    reference = plans[0]
    mismatches = {}
    for key in COMPATIBILITY_KEYS:
        values = [plan.get(key) for plan in plans]
        if any(value != values[0] for value in values[1:]):
            mismatches[key] = values
    return {
        key: reference.get(key)
        for key in COMPATIBILITY_KEYS
    }, mismatches


def audit_combined_runs(sources, output_dir: str | Path, parents_per_task=5):
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if parents_per_task < 1:
        raise ValueError("parents_per_task must be positive")

    normalized_sources = []
    seen_tasks = set()
    for task, raw_root in sources:
        root = Path(raw_root).expanduser().resolve()
        if task in seen_tasks:
            raise ValueError(f"Duplicate source task: {task}")
        seen_tasks.add(task)
        if not root.is_dir():
            raise FileNotFoundError(f"Source run does not exist: {root}")
        plan = _read_json(root / "plan.json")
        if not isinstance(plan, dict):
            raise ValueError(f"Missing source plan: {root / 'plan.json'}")
        if task not in plan.get("tasks", []):
            raise ValueError(f"Source {root} does not declare task {task}")
        normalized_sources.append((task, root, plan))

    if len(normalized_sources) < 2:
        raise ValueError("At least two task-specific sources are required")

    compatible, compatibility_mismatches = _compatible_values(
        [entry[2] for entry in normalized_sources]
    )
    if compatibility_mismatches:
        raise ValueError(
            "Incompatible source protocols: "
            + json.dumps(compatibility_mismatches, sort_keys=True)
        )
    candidate_count = int(compatible["candidate_count"])

    records = []
    errors = []
    selected_parents = []
    selected_snapshots = []
    source_provenance = []
    ineligible_by_task = {}
    attempted_by_task = {}
    payload_paths = []

    for task, root, plan in normalized_sources:
        parents = [
            row
            for row in _read_jsonl(root / "parent_records.jsonl")
            if row.get("task_name") == task and row.get("valid") is True
        ]
        if len(parents) != parents_per_task:
            raise ValueError(
                f"{task} has {len(parents)} completed parents; "
                f"expected {parents_per_task}"
            )
        parent_ids = {row["parent_id"] for row in parents}
        if len(parent_ids) != len(parents):
            raise ValueError(f"{task} contains duplicate parent IDs")

        snapshot_ids = []
        for parent in parents:
            captured = parent.get("captured", {})
            parent_snapshots = [
                value for value in captured.values() if isinstance(value, str)
            ]
            if len(parent_snapshots) != 2 or len(set(parent_snapshots)) != 2:
                raise ValueError(
                    f"Parent {parent['parent_id']} does not reference two snapshots"
                )
            for snapshot_id in parent_snapshots:
                snapshot_path = root / "snapshots" / f"{snapshot_id}.pkl.gz"
                if not snapshot_path.is_file():
                    raise FileNotFoundError(
                        f"Missing snapshot artifact: {snapshot_path}"
                    )
            snapshot_ids.extend(parent_snapshots)
        snapshot_id_set = set(snapshot_ids)
        if len(snapshot_id_set) != len(snapshot_ids):
            raise ValueError(f"{task} contains duplicate snapshot IDs")

        source_records = [
            row
            for row in load_branch_records(root / "branch_records.jsonl")
            if row.get("task_name") == task
            and row.get("parent_id") in parent_ids
            and row.get("snapshot_id") in snapshot_id_set
        ]
        expected_records = parents_per_task * 2 * (candidate_count + 3)
        if len(source_records) != expected_records:
            raise ValueError(
                f"{task} has {len(source_records)} selected branches; "
                f"expected {expected_records}"
            )
        for row in source_records:
            relative_payload = row.get("payload_path")
            if not relative_payload:
                raise ValueError(f"Branch {row.get('branch_id')} lacks payload_path")
            payload_path = root / relative_payload
            if not payload_path.is_file():
                raise FileNotFoundError(f"Missing branch payload: {payload_path}")
            payload_paths.append(str(payload_path))

        source_errors = [
            row
            for row in _read_jsonl(root / "errors.jsonl")
            if row.get("task_name") == task
        ]
        ineligible = [
            row
            for row in _read_jsonl(root / "ineligible_parent_records.jsonl")
            if row.get("task_name") == task
        ]
        status = _read_json(root / "status.json", {})
        source_analysis = _read_json(root / "analysis.json")
        source_provenance.append(
            {
                "task_name": task,
                "run_dir": str(root),
                "source_status": status.get("status"),
                "source_analysis_present": source_analysis is not None,
                "plan_sha256": _sha256(root / "plan.json"),
                "parent_records_sha256": _sha256(root / "parent_records.jsonl"),
                "branch_records_sha256": _sha256(root / "branch_records.jsonl"),
                "errors_sha256": _optional_sha256(root / "errors.jsonl"),
                "analysis_sha256": _optional_sha256(root / "analysis.json"),
                "selected_parents": len(parents),
                "selected_snapshots": len(snapshot_ids),
                "selected_branches": len(source_records),
                "selected_errors": len(source_errors),
                "ineligible_parents": len(ineligible),
                "target_stage": plan.get("target_stages", {}).get(task),
                "trigger_stage": plan.get("trigger_stages", {}).get(task),
            }
        )
        records.extend(source_records)
        errors.extend(source_errors)
        selected_parents.extend(parents)
        selected_snapshots.extend(snapshot_ids)
        ineligible_by_task[task] = len(ineligible)
        attempted_by_task[task] = len(parents) + len(ineligible) + len(source_errors)

    branch_ids = [row["branch_id"] for row in records]
    parent_ids = [row["parent_id"] for row in selected_parents]
    identifiers_unique = (
        len(set(branch_ids)) == len(branch_ids)
        and len(set(parent_ids)) == len(parent_ids)
        and len(set(selected_snapshots)) == len(selected_snapshots)
        and len(set(payload_paths)) == len(payload_paths)
    )
    if not identifiers_unique:
        raise ValueError("Combined sources contain duplicate identifiers or payloads")

    analysis = analyze_phase2_replay(records, errors)
    expected_parents = len(normalized_sources) * parents_per_task
    expected_snapshots = expected_parents * 2
    expected_primary = expected_snapshots * (candidate_count + 2)
    expected_total = expected_snapshots * (candidate_count + 3)
    grouped = defaultdict(lambda: defaultdict(int))
    for record in records:
        grouped[record["snapshot_id"]][record["kind"]] += 1

    analysis.update(
        {
            "protocol": "phase2_combined_complete_snapshot_replay_audit",
            "tasks": [entry[0] for entry in normalized_sources],
            "parents_per_task": parents_per_task,
            "expected_parents": expected_parents,
            "completed_parents": len(selected_parents),
            "attempted_parents_by_task": attempted_by_task,
            "ineligible_parents_by_task": ineligible_by_task,
            "expected_snapshots": expected_snapshots,
            "completed_snapshots": len(selected_snapshots),
            "expected_primary_branches": expected_primary,
            "expected_total_branches": expected_total,
            "source_provenance": source_provenance,
            "source_artifacts_copied": False,
            "source_protocol": compatible,
            "identifiers_unique": identifiers_unique,
            "branch_support_by_snapshot": {
                snapshot_id: dict(counts)
                for snapshot_id, counts in sorted(grouped.items())
            },
        }
    )
    analysis["gates"].update(
        {
            "source_protocols_compatible": not compatibility_mismatches,
            "source_artifacts_complete": len(payload_paths) == expected_total,
            "identifiers_unique": identifiers_unique,
            "parent_and_snapshot_support_complete": (
                len(selected_parents) == expected_parents
                and len(selected_snapshots) == expected_snapshots
            ),
            "branch_support_complete": (
                analysis["primary_records"] == expected_primary
                and analysis["records"] == expected_total
            ),
            "branch_group_composition_exact": bool(grouped)
            and all(
                counts.get("same_seed_repeat", 0) == 2
                and counts.get("candidate", 0) == candidate_count
                and counts.get("environment_only", 0) == 1
                and sum(counts.values()) == candidate_count + 3
                for counts in grouped.values()
            ),
        }
    )
    analysis["all_pass"] = all(analysis["gates"].values())

    plan = {
        "protocol": analysis["protocol"],
        "tasks": analysis["tasks"],
        "parents_per_task": parents_per_task,
        "expected_parents": expected_parents,
        "expected_snapshots": expected_snapshots,
        "expected_primary_branches": expected_primary,
        "expected_total_branches": expected_total,
        "candidate_count": candidate_count,
        "source_protocol": compatible,
        "sources": source_provenance,
    }
    status = {
        "status": "complete" if analysis["all_pass"] else "failed_gate",
        "all_pass": analysis["all_pass"],
        "parents": analysis["completed_parents"],
        "snapshots": analysis["completed_snapshots"],
        "branches": analysis["records"],
        "errors": analysis["errors"],
    }
    _atomic_write_json(output_dir / "plan.json", plan)
    _atomic_write_json(output_dir / "analysis.json", analysis)
    _atomic_write_json(output_dir / "status.json", status)
    return analysis


def print_report(analysis, output_dir):
    print("PHASE 2 COMBINED COMPLETE-SNAPSHOT REPLAY AUDIT")
    print("status:", "complete" if analysis["all_pass"] else "failed_gate")
    print("tasks:", " ".join(analysis["tasks"]))
    print(
        "parents:",
        analysis["completed_parents"],
        "/",
        analysis["expected_parents"],
    )
    print(
        "snapshots:",
        analysis["completed_snapshots"],
        "/",
        analysis["expected_snapshots"],
    )
    print("branches:", analysis["records"], "/", analysis["expected_total_branches"])
    print("errors:", analysis["errors"])
    print("source artifacts copied:", analysis["source_artifacts_copied"])
    print("\nENGINEERING VALIDITY GATES")
    for name, passed in analysis["gates"].items():
        print(f"{'PASS' if passed else 'FAIL':4s}  {name}")
    print("\nDIAGNOSTICS")
    print(
        "same-seed suffix agreement:",
        f"{analysis['same_seed_suffix_outcome_agreement']:.4f}",
    )
    print(
        "candidate-diverse snapshots:",
        f"{analysis['candidate_diversity_rate']:.4f}",
    )
    print("branch error rate:", f"{analysis['error_rate']:.4f}")
    print("\nPHASE 2 COMBINED VALIDITY:", "PASS" if analysis["all_pass"] else "FAIL")
    print("artifacts:", Path(output_dir).resolve())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        help="Task-specific source in TASK=RUN_DIR form; repeat per task",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parents-per-task", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    analysis = audit_combined_runs(
        args.source,
        args.output_dir,
        parents_per_task=args.parents_per_task,
    )
    print_report(analysis, args.output_dir)


if __name__ == "__main__":
    main()
