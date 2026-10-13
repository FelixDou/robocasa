"""Audit both guarded nominal repeats of all ten frozen timing parents."""

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import traceback

from robocasa.recovery import diagnose_intervention_timing_replay as replay
from robocasa.recovery.intervention_timing import validate_registration
from robocasa.recovery.phase3b_training_horizon import (
    atomic_write_json, load_registered_phase2_snapshot, read_jsonl, sha256_file, sha256_json,
)
from robocasa.recovery.run_phase2_snapshot_replay import append_jsonl, utc_now


PROTOCOL = "xr1_gc_guarded_all_parent_replay_audit_v1"


def parent_gates(report, steps):
    attempts = report.get("attempts", [])
    collection = report.get("garbage_collection", {})
    traces = collection.get("attempts", [])
    observations = report.get("observation_window_comparison", {})
    early = report.get("early_rng_runtime_initialization") or {}
    startup = [e for e in report.get("startup_events", []) if e["phase"].endswith(":before_environment_creation")]
    return {
        "two_complete_nominal_replays": len(attempts) == 2 and all(a["environment_steps"] == steps for a in attempts),
        "both_source_prefix_audits_accepted": len(attempts) == 2 and all(
            a["prefix_replay_accepted"] and a["source_endpoint_all_exact"] and a["replay_error"] is None for a in attempts),
        "initial_and_endpoint_repeats_exact": all(report.get(k, {}).get("all_exact") is True
            for k in ("repeated_initial_state_audit", "repeated_endpoint_audit")),
        "complete_progress_repeats_exact": report.get("first_repeat_divergence", "missing") is None,
        "complete_raw_observation_repeats_exact": observations.get("trace_rows") == [steps+1, steps+1]
            and observations.get("raw_observation_counts") == [steps+1, steps+1]
            and observations.get("trace_step_support_exact") is True
            and observations.get("first_observation_difference_step", "missing") is None
            and observations.get("first_difference_by_observation_key") == {},
        "both_seeded_request_probes_exact": len(attempts) == 2 and all(
            all(a.get("request_repeat_probe", {}).get(k) is True for k in ("performed", "actions_exact", "response_exact"))
            for a in attempts),
        "gc_control_exercised_without_nominal_collections": collection.get("mode") == "disabled"
            and len(traces) == 2 and all(t["mode"] == "disabled" and not t["enabled_at_environment_creation"]
                and not t["enabled_at_end_of_cleanup"] and t["nominal_collection_stops"] == 0
                and t["original_enabled_state_restored"] for t in traces),
        "early_rng_runtime_initialized": report.get("rng_runtime_init") == "before_env"
            and early.get("source_rng_restored_early") is False
            and (not early.get("torch_cuda_available") or len(startup) == 2 and all(e["cuda_initialized"] for e in startup)),
    }


def audit_inputs(args):
    if args.gc_mode != "disabled" or args.rng_runtime_init != "before_env" or not args.record_all_observations:
        raise ValueError("All-parent audit requires disabled GC, before_env RNG initialization, and all raw observations")
    if args.observation_window is not None:
        raise ValueError("Unset the fixed observation window for the all-parent audit; source prefixes have different lengths")
    source = args.timing_run_dir.resolve()
    timing = json.loads((source/"registration.json").read_text())
    validate_registration(timing)
    parents = timing["parents"]
    if len(parents) != 10 or len({p["parent_id"] for p in parents}) != 10 or Counter(
            p["task_name"] for p in parents) != {"ArrangeTea": 5, "CuttingToolSelection": 5}:
        raise ValueError("Require ten unique frozen parents, five per task")
    endpoints = {}
    for parent in parents:
        if parent["parent_id"] in (".", "..") or Path(parent["parent_id"]).name != parent["parent_id"]:
            raise ValueError("Parent identity cannot contain directory components")
        sid = parent["snapshot_id"]
        reference, _ = load_registered_phase2_snapshot(
            Path(timing["phase2_run_dir"])/"snapshots"/f"{sid}.pkl.gz",
            expected_file_sha256=timing["phase2_registration"]["phase2_snapshot_file_sha256"][sid],
            expected_snapshot_id=sid, expected_parent_id=parent["parent_id"], expected_task_name=parent["task_name"])
        endpoints[parent["parent_id"]] = reference.environment_step
    plan = json.loads((Path(timing["phase2_run_dir"])/"plan.json").read_text())
    replay._validate_live_provenance(args, timing["phase2_registration"], plan)
    root = Path(__file__).resolve().parents[2]
    code = [Path(__file__), Path(replay.__file__),
            Path(replay.__file__).with_name("diagnose_intervention_timing_observations.py"),
            root/"scripts/diagnose_xr1_timing_replay.sh"]
    return timing, {
        "protocol": PROTOCOL, "timing_run_dir": str(source),
        "timing_registration_sha256": timing["registration_sha256"],
        "timing_registration_file_sha256": sha256_file(source/"registration.json"),
        "parents": parents, "source_environment_steps": endpoints,
        "replays_per_parent": 2, "expected_parents": 10,
        "gc_mode": args.gc_mode, "rng_runtime_init": args.rng_runtime_init,
        "record_all_observations": True, "model_path": str(args.model_path),
        "host": args.host, "port": args.port,
        "restore_atol": args.restore_atol, "restore_rtol": args.restore_rtol,
        "requested_gpu_assignment": {"simulator": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "server": os.environ.get("XR1_REPLAY_SERVER_GPU"), "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID")},
        "code_hashes": {str(p.relative_to(root)): sha256_file(p) for p in code},
        "scientific_trajectories_collected": 0,
    }


def checked_report(root, row, config, *, check_artifacts):
    path = (root/row["report_path"]).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != row["report_file_sha256"]:
        raise ValueError("Completed parent report changed or escaped the audit root")
    report = json.loads(path.read_text())
    if report.get("protocol") != replay.PROTOCOL or report.get("status") != "diagnostic_complete" or report.get("parent_id") != row["parent_id"]:
        raise ValueError("Completed parent report identity differs")
    if check_artifacts:
        for name, expected in report["artifact_hashes"].items():
            artifact = (path.parent/name).resolve()
            if not artifact.is_relative_to(path.parent) or sha256_file(artifact) != expected:
                raise ValueError("Completed parent diagnostic artifact changed")
    gates = parent_gates(report, config["source_environment_steps"][row["parent_id"]])
    if gates != row["engineering_gates"] or not all(gates.values()):
        raise ValueError("Completed parent no longer passes its replay gates")
    return report


def completed_parents(root, registration):
    rows = read_jsonl(root/"parent_records.jsonl")
    known = {r["parent_id"] for r in rows}
    for parent in registration["parents"]:
        marker = root/"parents"/parent["parent_id"]/"committed.json"
        if parent["parent_id"] not in known and marker.exists():
            row = json.loads(marker.read_text())
            append_jsonl(root/"parent_records.jsonl", row)
            rows.append(row)
    expected = {p["parent_id"]: p for p in registration["parents"]}
    result = {}
    for row in rows:
        pid = row["parent_id"]
        if pid not in expected or pid in result or row["parent"] != expected[pid] or row["registration_sha256"] != registration["registration_sha256"]:
            raise ValueError("Unknown, duplicate, or misregistered completed parent")
        if json.loads((root/"parents"/pid/"committed.json").read_text()) != row:
            raise ValueError("Parent commit marker differs from its ledger")
        checked_report(root, row, registration, check_artifacts=True)
        result[pid] = row
    return result


def run(args, runtime=None):
    timing, config = audit_inputs(args)
    root = args.output_dir.resolve()
    for name in (config["timing_run_dir"], timing["phase2_run_dir"], timing["operator_run_dir"]):
        protected = Path(name).resolve()
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError("Parent audit output must be separate from all frozen inputs")
    registration = None
    if args.resume:
        registration = json.loads((root/"registration.json").read_text())
        unsigned = dict(registration)
        digest = unsigned.pop("registration_sha256", None)
        if sha256_json(unsigned) != digest or {k: registration.get(k) for k in config} != config:
            raise ValueError("Parent audit code, source, or configuration changed after registration")
    if args.dry_run:
        print(json.dumps({**config, "total_nominal_environment_steps": 2*sum(config["source_environment_steps"].values())}, indent=2))
        return config
    root.mkdir(parents=True, exist_ok=True)
    with (root/".parent_audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if registration is None:
            if any(p.name != ".parent_audit.lock" for p in root.iterdir()):
                raise FileExistsError("Parent audit root is not empty; use resume")
            registration = {**config, "created_at": utc_now()}
            registration["registration_sha256"] = sha256_json(registration)
            atomic_write_json(root/"registration.json", registration)
        if read_jsonl(root/"errors.jsonl"):
            raise ValueError("Audit errors recorded; preserve the root and diagnose before a new run")
        completed = completed_parents(root, registration)
        current = None
        try:
            for parent in registration["parents"]:
                pid = parent["parent_id"]
                if pid in completed:
                    continue
                current = parent
                atomic_write_json(root/"status.json", {"status": "running", "parents": len(completed),
                    "expected_parents": 10, "current_parent": pid, "updated_at": utc_now()})
                folder = root/"parents"/pid
                folder.mkdir(parents=True, exist_ok=True)
                # An interrupted uncommitted diagnostic is retained. A new
                # attempt gets a separate child directory on resume.
                attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=folder))/"diagnostic"
                child_args = deepcopy(args)
                child_args.parent_id, child_args.output_dir = pid, attempt
                report = replay.run(child_args, runtime)
                gates = parent_gates(report, registration["source_environment_steps"][pid])
                row = {"parent_id": pid, "parent": parent,
                       "registration_sha256": registration["registration_sha256"],
                       "report_path": str((attempt/"report.json").relative_to(root)),
                       "report_file_sha256": sha256_file(attempt/"report.json"), "engineering_gates": gates}
                if not all(gates.values()):
                    atomic_write_json(folder/"failed_parent.json", row)
                    raise ValueError(f"Guarded parent replay audit failed: {pid}; inspect {attempt}")
                atomic_write_json(folder/"committed.json", row)
                append_jsonl(root/"parent_records.jsonl", row)
                completed[pid] = row
                print(f"parent audit [{len(completed)}/10]: {pid} accepted", flush=True)
        except Exception as exc:
            append_jsonl(root/"errors.jsonl", {"parent": current, "error": str(exc), "traceback": traceback.format_exc()})
            atomic_write_json(root/"status.json", {"status": "failed", "parents": len(completed),
                "expected_parents": 10, "failed_parent": current, "error": str(exc), "updated_at": utc_now()})
            raise
        result = {"protocol": PROTOCOL, "status": "all_parents_pass", "parents": len(completed),
                  "updated_at": utc_now(),
                  "expected_parents": 10, "nominal_replays": 2*len(completed),
                  "parents_by_task": dict(Counter(row["parent"]["task_name"] for row in completed.values())),
                  "engineering_all_pass": len(completed) == 10,
                  "scientific_trajectories_collected": 0, "full_timing_study_complete": False,
                  "parent_reports": list(completed.values()),
                  "artifact_hashes": {name: sha256_file(root/name) for name in ("registration.json", "parent_records.jsonl")}}
        atomic_write_json(root/"report.json", result)
        atomic_write_json(root/"status.json", {k: v for k, v in result.items() if k not in ("parent_reports", "artifact_hashes")})
        print("all-parent replay audit: " + str(root/"report.json"), flush=True)
        return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10326)
    parser.add_argument("--gc-mode", choices=("default", "disabled"), default="disabled")
    parser.add_argument("--rng-runtime-init", choices=("after_reset", "before_env"), default="before_env")
    parser.add_argument("--record-all-observations", action="store_true")
    parser.add_argument("--observation-window", nargs=2, type=int)
    parser.add_argument("--restore-atol", type=float, default=1e-8)
    parser.add_argument("--restore-rtol", type=float, default=1e-8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
