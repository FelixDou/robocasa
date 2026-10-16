"""Bounded timing pilot: engineering exclusions precede treatment assignment."""

from copy import deepcopy
import json
from pathlib import Path

from robocasa.recovery.exact_replay import collect_intervention_timing_references as fresh
from robocasa.recovery.full_snapshot import load_full_snapshot
from robocasa.recovery.exact_replay.intervention_timing import allocate, validate_registration as validate_source
from robocasa.recovery.exact_replay.phase3b_training_horizon import read_jsonl, sha256_file, sha256_json, atomic_write_json
from robocasa.recovery.exact_replay.run_phase2_snapshot_replay import append_jsonl, utc_now

PROTOCOL = "xr1_common_entry_timing_audit_and_exclude_pilot_v1"
SUFFIX_GC_SCOPE = "post_restore_sampled_observation_and_evaluation_through_suffix_capture"


def code_hashes(timing):
    root = Path(__file__).resolve().parents[3]
    names = ("robocasa/recovery/exact_replay/intervention_timing_pilot.py",
             "robocasa/recovery/exact_replay/run_intervention_timing_pilot.py",
             "scripts/exact_replay/run_xr1_timing_pilot.sh")
    return {**fresh.code_hashes(timing), **{name: sha256_file(root/name) for name in names}}


def checked(root, name):
    root, path = Path(root).resolve(), (Path(root)/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Pilot artifact escaped its root")
    return path


def artifacts(root, folder):
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def same_parent(a, b):
    a, b = deepcopy(a), deepcopy(b)
    # Network address is transport configuration, not a policy parameter.
    for p in (a, b):
        for key in ("host", "port"):
            p["policy_args"].pop(key, None)
    return a == b


def reuse_manifest(root, parents):
    """Read only committed v4 evidence; a failed enclosing run is not a pass."""
    if root is None:
        return {}, {}
    root = root.resolve()
    registration = fresh.validate_cohort_registration(root)
    expected = {p["parent_id"]: p for p in parents}
    if (set(expected) != {p["parent_id"] for p in registration["parents"]}
            or any(not same_parent(p, expected[p["parent_id"]]) for p in registration["parents"])):
        raise ValueError("Reuse cohort seed/configuration differs from the fixed pilot pool")
    hashes = {str(root/"registration.json"): sha256_file(root/"registration.json")}
    if (root/"reference_records.jsonl").exists():
        hashes[str(root/"reference_records.jsonl")] = sha256_file(root/"reference_records.jsonl")
    manifest = {}
    for row in read_jsonl(root/"reference_records.jsonl"):
        pid = row["parent_id"]
        if (pid not in expected or pid in manifest or not row["engineering_all_pass"]
                or row["registration_sha256"] != registration["registration_sha256"]
                or not same_parent(row["parent"], expected[pid])):
            raise ValueError("Invalid committed reuse parent")
        marker = checked(root, f"parents/{pid}/committed.json")
        if json.loads(marker.read_text()) != row:
            raise ValueError("Reuse marker/ledger mismatch")
        report = json.loads(checked(root, row["report_path"]).read_text())
        if not report["engineering_all_pass"] or not all(report["engineering_gates"].values()):
            raise ValueError("Reuse reference engineering gates failed")
        for name, digest in row["artifact_hashes"].items():
            path = checked(root, name)
            if sha256_file(path) != digest:
                raise ValueError("Reuse artifact changed")
            hashes[str(path)] = digest
        snapshot = load_full_snapshot(checked(root, row["snapshot_path"]))
        if snapshot.snapshot_id != row["snapshot_id"] or snapshot.parent_id != pid:
            raise ValueError("Reuse snapshot identity mismatch")
        hashes[str(marker)] = sha256_file(marker)
        manifest[pid] = {"kind": "reference_reused", "source_root": str(root), "source_record": row}
    # Adopt a documented pre-inference reset rejection, never an arbitrary
    # run exception or any failure after treatment. Its evidence is frozen now.
    status_path = root/"status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        parent = status.get("failed_parent") or {}
        pid = parent.get("parent_id")
        if status.get("status") == "failed" and pid in expected and pid not in manifest:
            for path in sorted((root/"parents"/pid).glob("collection_*/attempt_1/initial_restore_audit.json")):
                audit = json.loads(path.read_text())
                first_path = path.parents[1]/"attempt_0/report.json"
                if not first_path.exists():
                    continue
                first = json.loads(first_path.read_text())
                if (audit.get("observation_verification", {}).get("performed") is True
                        and audit.get("observation_exact") is False and audit.get("observation_mismatches")
                        and first.get("environment_steps") == 0 and first.get("nominal_requests") == 0
                        and not read_jsonl(path.parent/"requests.jsonl")):
                    folder = path.parents[1]
                    for name, digest in artifacts(root, folder).items():
                        hashes[str(root/name)] = digest
                    hashes[str(status_path)] = sha256_file(status_path)
                    manifest[pid] = {"kind": "reference_excluded", "source_root": str(root),
                        "source_folder": str(folder.relative_to(root)),
                        "reason": "recorded_v4_live_reset_observation_mismatch_before_nominal_requests"}
                    break
    return manifest, hashes


def potential_branches(parents, spec):
    # Potential repeats for every parent let a rejected sentinel be replaced
    # by the first qualifying parent of the same task, before interventions.
    branches = allocate([{**p, "sentinel": True} for p in parents], spec)
    for branch in branches:
        branch["sentinel_candidate"] = branch.pop("sentinel")
    return branches


def build_registration(args):
    timing, plan, config = fresh.inputs(args)
    reuse, hashes = reuse_manifest(args.reference_run_dir, config["parents"])
    result = {**config, "protocol": PROTOCOL, "reference_algorithm": fresh.PROTOCOL,
        "created_at": utc_now(), "phase2_run_dir": timing["phase2_run_dir"],
        "phase2_registration": timing["phase2_registration"], "operator_spec": timing["operator_spec"],
        "spec": deepcopy(timing["spec"]), "branches": potential_branches(config["parents"], timing["spec"]),
        "reference_run_dir": str(args.reference_run_dir.resolve()) if args.reference_run_dir else None,
        "suffix_gc_nominal_scope": SUFFIX_GC_SCOPE,
        "reuse_manifest": reuse, "reuse_artifact_hashes": hashes,
        "code_hashes": code_hashes(timing),
        "eligibility": {"attempt_limit": 10, "replacement_seeds": False,
            "rule": "v4_reference_gates_then_two_exact_full_continue_schedule0_repetitions_before_any_treatment",
            "minimum_parents_per_task_for_two_task_analysis": 2,
            "excluded_parent_outcomes_used": False},
        "assignment": "all_ten_cells_both_schedules_for_eligible_parents; first_eligible_per_task_has_full_schedule0_repeats",
        "late_errors": "record_by_branch_and_operator_continue_collection_block_scientific_inference",
        "image_review": "nominal_camera_gallery_review_without_operator_outcomes_required_for_scientific_analysis",
        "historical_timing_results_reused": False,
    }
    result["registration_sha256"] = sha256_json(result)
    return result, plan


def validate_registration(root):
    root = Path(root)
    registration = json.loads((root/"registration.json").read_text())
    unsigned = dict(registration)
    digest = unsigned.pop("registration_sha256", None)
    if registration.get("protocol") != PROTOCOL or sha256_json(unsigned) != digest:
        raise ValueError("Pilot registration checksum/protocol mismatch")
    source = Path(registration["timing_run_dir"])
    if sha256_file(source/"registration.json") != registration["timing_registration_file_sha256"]:
        raise ValueError("Pilot frozen timing source changed")
    timing = json.loads((source/"registration.json").read_text())
    validate_source(timing)
    if code_hashes(timing) != registration["code_hashes"]:
        raise ValueError("Pilot code changed after registration")
    if registration["branches"] != potential_branches(registration["parents"], registration["spec"]):
        raise ValueError("Pilot allocation changed")
    if sha256_file(Path(registration["phase2_run_dir"])/"plan.json") != registration["phase2_plan_sha256"]:
        raise ValueError("Pilot source plan changed")
    for name, digest in registration["reuse_artifact_hashes"].items():
        if sha256_file(name) != digest:
            raise ValueError("Pilot reuse evidence changed")
    return registration


def commit_parent(root, ledger_name, parent, registration, folder, **data):
    row = {"parent_id": parent["parent_id"], "parent": parent,
           "registration_sha256": registration["registration_sha256"],
           "artifact_hashes": artifacts(root, folder), **data}
    marker = root/ledger_name/parent["parent_id"]/"committed.json"
    atomic_write_json(marker, row)
    append_jsonl(root/f"{ledger_name}_records.jsonl", row)
    return row


def parent_records(root, phase, registration, *, recover=False):
    expected = {p["parent_id"]: p for p in registration["parents"]}
    rows = read_jsonl(root/f"{phase}_records.jsonl")
    known = {r["parent_id"] for r in rows}
    if recover:
        for pid in expected.keys()-known:
            marker = root/phase/pid/"committed.json"
            if marker.exists():
                row = json.loads(marker.read_text())
                append_jsonl(root/f"{phase}_records.jsonl", row)
                rows.append(row)
    result = {}
    for row in rows:
        pid = row["parent_id"]
        if (pid not in expected or pid in result or row["parent"] != expected[pid]
                or row["registration_sha256"] != registration["registration_sha256"]):
            raise ValueError("Unknown/duplicate/unbound pilot parent")
        if json.loads((root/phase/pid/"committed.json").read_text()) != row:
            raise ValueError("Pilot parent marker/ledger mismatch")
        for name, digest in row["artifact_hashes"].items():
            if sha256_file(checked(root, name)) != digest:
                raise ValueError("Pilot parent artifact changed")
        result[pid] = row
    return result


def assignment(registration, qualifications):
    if set(qualifications) != {p["parent_id"] for p in registration["parents"]}:
        raise ValueError("Finish the fixed eligibility pool before assignment")
    eligible = [p for p in registration["parents"] if qualifications[p["parent_id"]]["kind"] == "eligible"]
    sentinels = {}
    for parent in eligible:
        sentinels.setdefault(parent["task_name"], parent["parent_id"])
    ids = {p["parent_id"] for p in eligible}
    branches = [b for b in registration["branches"] if b["parent_id"] in ids and
        (b["role"] == "primary" or b["operator"] == "continue" or b["parent_id"] in sentinels.values())]
    result = {"protocol": PROTOCOL, "registration_sha256": registration["registration_sha256"],
        "qualification_records_sha256": sha256_json(qualifications),
        "eligible_parents": [p["parent_id"] for p in eligible], "sentinels_by_task": sentinels,
        "branches": branches, "expected_primary_branches": 20*len(eligible),
        "expected_total_branches": len(branches)}
    result["assignment_sha256"] = sha256_json(result)
    return result
