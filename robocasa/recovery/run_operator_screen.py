"""Run the frozen XR1 operator screen: engineering sentinel, then full cohort."""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import gzip
import json
from pathlib import Path
import pickle
import time
import traceback

from robocasa.recovery.counterfactual_branch import (
    BranchSpec, branch_payload_digest, run_counterfactual_branch, save_branch_result,
)
from robocasa.recovery.full_snapshot import stable_digest
from robocasa.recovery.operator_screen import (
    DEFAULT_SPEC, ORACLE, OperatorPolicy, build_registration,
    repeat_channels, unique_records, validate_registration, validate_retreat_controller,
)
from robocasa.recovery.phase3b_training_horizon import (
    atomic_write_json, load_registered_phase2_snapshot, read_jsonl,
    sha256_file,
)
from robocasa.recovery.run_phase2_snapshot_replay import (
    _close_branch_environment, _make_fresh_branch_environment, _runtime,
    append_jsonl, utc_now,
)
from robocasa.recovery.run_phase3b_training_horizon_continuation import (
    _runtime_args, _validate_live_provenance,
)


def engineering_analysis(root, registration):
    root = Path(root)
    validate_registration(registration)
    records = unique_records(root, registration)
    expected = {x["branch_id"] for x in registration["branches"]}
    errors = read_jsonl(root / "errors.jsonl")
    audits = read_jsonl(root / "snapshot_integrity_audits.jsonl")
    repeats = []
    for branch in registration["branches"]:
        if branch["role"] != "repeat":
            continue
        original = branch["branch_id"].rsplit("--r", 1)[0] + "--r0"
        a, b = records.get(original), records.get(branch["branch_id"])
        repeats.append({"branch_id": branch["branch_id"], "exact": bool(a and b and repeat_channels(a) == repeat_channels(b))})
    effect = []
    if registration["scope"] == "sentinel":
        for selected in registration["selected_snapshots"]:
            sid = selected["snapshot_id"]
            a = records.get(f"{sid}--continue--s0--r0")
            b = records.get(f"{sid}--history_reset_replan--s0--r0")
            effect.append(bool(a and b and a["first_request_sha256"] != b["first_request_sha256"]))
    valid_horizons = all(
        0 < r["num_steps"] <= r["registered_horizon_environment_steps"]
        and (r["termination_reason"] != "suffix_horizon" or r["num_steps"] == r["registered_horizon_environment_steps"])
        and r["termination_reason"] in {"suffix_horizon", "success", "environment_done", "safety_termination"}
        for r in records.values()
    )
    gates = {
        "complete_unique_branch_support": set(records) == expected,
        "all_payload_hashes_and_embedded_records_exact": True,
        "snapshot_integrity_coverage": {x["target_snapshot_id"] for x in audits} == {x["snapshot_id"] for x in registration["selected_snapshots"]},
        "all_snapshot_integrity_accepted": bool(audits) and all(x.get("accepted") for x in audits),
        "all_restores_valid": bool(records) and all(r.get("restore_valid") and r.get("restore_subtask_exact") and not r.get("restore_induced_regressed_predicates") for r in records.values()),
        "all_request_alignment_valid": bool(records) and all(r.get("record_alignment_valid") for r in records.values()),
        "paired_continuation_schedules_exact": bool(records) and all(
            r["request_sampling_seeds"] == list(range(r["sampling_seed"], r["sampling_seed"] + r["num_policy_requests"]))
            for r in records.values()
        ),
        "common_step_budgets_valid": bool(records) and valid_horizons,
        "sentinel_repeats_exact": bool(repeats) and all(x["exact"] for x in repeats) if registration["scope"] == "sentinel" else True,
        "history_reset_changes_request": bool(effect) and all(effect) if registration["scope"] == "sentinel" else True,
        "zero_errors": not errors,
        "retreat_controller_validated": all(
            r.get("retreat_controller_audit", {}).get("accepted")
            for r in records.values() if r["operator"] == "retreat_reset_replan"
        ),
    }
    return {
        "protocol": registration["protocol"], "scope": registration["scope"],
        "branches": len(records), "expected_branches": len(expected),
        "errors": len(errors), "engineering_gates": gates,
        "engineering_all_pass": all(gates.values()), "repeat_audits": repeats,
        "scientific_outcomes_used_as_engineering_gate": False,
    }


def _load_snapshot(registration, branch, root):
    target_id = branch["snapshot_id"]
    if branch["operator"] == ORACLE:
        entry = registration["oracle_anchors"][target_id]
        path, snapshot_id, digest = entry["path"], entry["snapshot_id"], entry["sha256"]
    else:
        path = Path(registration["phase2_run_dir"]) / "snapshots" / f"{target_id}.pkl.gz"
        snapshot_id = target_id
        digest = registration["phase2_registration"]["phase2_snapshot_file_sha256"][target_id]
    snapshot, audit = load_registered_phase2_snapshot(
        path, expected_file_sha256=digest, expected_snapshot_id=snapshot_id,
        expected_parent_id=branch["parent_id"], expected_task_name=branch["task_name"],
    )
    audit.update(target_snapshot_id=target_id, operator=branch["operator"])
    append_jsonl(Path(root) / "snapshot_integrity_audits.jsonl", audit)
    return snapshot


def _policy_args(snapshot, args):
    structural = deepcopy(snapshot.policy_state["structural_config"])
    structural.pop("queue_length")  # derived from history settings
    structural.update(snapshot.policy_state["sampling_config"])
    structural.update(host=args.host, port=args.port)
    if str(args.model_path) != structural["model_path"]:
        raise ValueError("Model path must exactly match the frozen snapshot; recreate its node-local copy")
    if structural["safe_best_of_k"] != 1:
        raise ValueError("Operator screen requires ordinary K=1 policy execution")
    return structural


def _save_result(result, branch, wrapped, seconds, root, controller_audit=None):
    summary = result["summary"]
    summary.pop("payload_sha256", None)
    restored_snapshot_id = summary["snapshot_id"]
    summary.update({
        "protocol": "xr1_higher_level_operator_branch_v1", "kind": "operator",
        "snapshot_id": branch["snapshot_id"], "restored_snapshot_id": restored_snapshot_id,
        "operator": branch["operator"], "schedule": branch["schedule"],
        "repeat": branch["repeat"], "role": branch["role"], "boundary": branch["boundary"],
        "registered_horizon_environment_steps": branch["horizon_environment_steps"],
        "operator_events": wrapped.events, "wall_seconds": seconds,
        "semantic_sha256": stable_digest(result["payload"]["subtask_evals"]),
        "retreat_controller_audit": controller_audit,
        "common_randomness_contract": {
            "identical_target_state_before_treatment": branch["operator"] != ORACLE,
            "paired_by_policy_request_index": True,
            "seed_schedule_independent_of_history_reset": True,
        },
    })
    result["payload"]["summary"] = summary
    summary["payload_sha256"] = branch_payload_digest(result["payload"])
    return save_branch_result(result, root)


def _recover_orphans(root, registration):
    """Recover an atomic payload saved immediately before an interrupted ledger append."""
    existing = unique_records(root, registration)
    for branch in registration["branches"]:
        if branch["branch_id"] in existing:
            continue
        path = root / "branches" / f"{branch['branch_id']}.pkl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)  # trusted local experiment only
        summary = deepcopy(payload["summary"])
        if summary["branch_id"] != branch["branch_id"] or branch_payload_digest(payload) != summary["payload_sha256"]:
            raise ValueError("Interrupted branch payload failed integrity validation")
        summary.update(payload_path=str(path.relative_to(root)), payload_file_sha256=sha256_file(path))
        append_jsonl(root / "branch_records.jsonl", summary)
    return unique_records(root, registration)


def run(args, runtime=None):
    root = args.output_dir.resolve()
    spec = json.loads(args.operator_spec.read_text()) if args.operator_spec else deepcopy(DEFAULT_SPEC)
    if args.dry_run:
        registration = build_registration(args.phase3b_run_dir, scope=args.scope, spec=spec,
            sentinel_dir=args.sentinel_run_dir, oracle_anchors=args.oracle_anchors)
        source = registration["phase2_registration"]
        plan = json.loads((Path(registration["phase2_run_dir"]) / "plan.json").read_text())
        _validate_live_provenance(args, source, plan)
        print(json.dumps({k: registration[k] for k in ("scope", "expected_branches", "spec", "selected_snapshots")}, indent=2))
        return registration  # no output writes, no simulator or policy connection
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".operator_screen.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another operator collector owns this output directory") from error
        registration_path = root / "registration.json"
        if args.resume:
            registration = json.loads(registration_path.read_text())
            validate_registration(registration)
            anchors = json.loads(args.oracle_anchors.read_text()) if args.oracle_anchors else {}
            requested = (args.scope, str(args.phase3b_run_dir.resolve()), spec, anchors)
            frozen = (registration["scope"], registration["phase3b_run_dir"], registration["spec"], registration["oracle_anchors"])
            if requested != frozen:
                raise ValueError("Resume differs from the frozen scope/source/operator specification")
            if args.scope == "full" and str(args.sentinel_run_dir.resolve()) != registration["sentinel"]["run_dir"]:
                raise ValueError("Resume sentinel differs")
        else:
            if any(p.name != ".operator_screen.lock" for p in root.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {root}")
            registration = build_registration(args.phase3b_run_dir, scope=args.scope, spec=spec,
                sentinel_dir=args.sentinel_run_dir, oracle_anchors=args.oracle_anchors)
            atomic_write_json(registration_path, registration)
        if read_jsonl(root / "errors.jsonl"):
            raise ValueError("This run has technical errors; preserve it and use a new output root after diagnosis")
        source = registration["phase2_registration"]
        plan = json.loads((Path(registration["phase2_run_dir"]) / "plan.json").read_text())
        _validate_live_provenance(args, source, plan)
        completed = _recover_orphans(root, registration)
        runtime = runtime or _runtime()
        runtime_args = _runtime_args(args, plan)
        factory = runtime["load_factory"](plan["policy_module"])
        policy = None
        try:
            for branch in registration["branches"]:
                if branch["branch_id"] in completed:
                    continue
                atomic_write_json(root / "status.json", {
                    "status": "running", "updated_at": utc_now(), "current_branch": branch["branch_id"],
                    "branches": len(completed), "expected_branches": registration["expected_branches"],
                })
                env = None
                try:
                    snapshot = _load_snapshot(registration, branch, root)
                    env = _make_fresh_branch_environment(snapshot, runtime_args, runtime)
                    controller_audit = validate_retreat_controller(env) if branch["operator"] == "retreat_reset_replan" else None
                    if policy is None:
                        policy = runtime["call_factory"](factory, env, _policy_args(snapshot, args))
                    wrapped = OperatorPolicy(policy, branch, spec)
                    started = time.monotonic()
                    result = run_counterfactual_branch(
                        snapshot, env, wrapped,
                        BranchSpec(branch["branch_id"], "candidate", branch["sampling_seed"], branch["horizon_environment_steps"]),
                        step_fn=runtime["step_fn"], success_fn=runtime["success_fn"],
                        subtask_eval_fn=runtime["get_subtask_eval"],
                        restore_atol=args.restore_atol, restore_rtol=args.restore_rtol,
                        snapshot_integrity_prevalidated=True,
                    )
                    completed[branch["branch_id"]] = _save_result(result, branch, wrapped, time.monotonic() - started, root, controller_audit)
                    print(f"[{len(completed)}/{registration['expected_branches']}] {branch['branch_id']}", flush=True)
                except Exception as error:
                    append_jsonl(root / "errors.jsonl", {**branch, "error": str(error), "traceback": traceback.format_exc(), "created_at": utc_now()})
                    atomic_write_json(root / "status.json", {"status": "failed", "error": str(error), "updated_at": utc_now()})
                    raise
                finally:
                    if env is not None:
                        _close_branch_environment(env)
        finally:
            if policy is not None:
                policy.close()
        analysis = engineering_analysis(root, registration)
        atomic_write_json(root / "analysis.json", analysis)
        atomic_write_json(root / "status.json", {**analysis, "status": "complete" if analysis["engineering_all_pass"] else "failed_gate", "updated_at": utc_now()})
        print(json.dumps(analysis, indent=2))
        if not analysis["engineering_all_pass"]:
            raise ValueError("Operator engineering gate failed; preserve artifacts and diagnose")
        return analysis


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3b-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("sentinel", "full"), required=True)
    parser.add_argument("--sentinel-run-dir", type=Path)
    parser.add_argument("--operator-spec", type=Path)
    parser.add_argument("--oracle-anchors", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10316)
    parser.set_defaults(restore_atol=1e-8, restore_rtol=1e-8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.scope == "full" and args.sentinel_run_dir is None:
        raise SystemExit("--scope full requires --sentinel-run-dir")
    try:
        run(args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
