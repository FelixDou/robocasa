"""Capture audited stage-entry states and execute frozen intervention timings."""

import argparse
from copy import deepcopy
import fcntl
import gzip
import json
from pathlib import Path
import pickle
import tempfile
import time
import traceback

from robocasa.recovery.counterfactual_branch import branch_payload_digest, save_branch_result
from robocasa.recovery.full_snapshot import (
    capture_full_snapshot, save_full_snapshot, load_full_snapshot, restore_full_snapshot,
    stable_digest, compare_structures, _compare_environment_states,
    _capture_global_rng_state, _restore_global_rng_state,
    environment_fingerprint, causal_transition_fingerprint,
)
from robocasa.recovery.intervention_timing import (
    PROTOCOL, DEFAULT_SPEC, TimingPolicy, build_registration, validate_registration,
    unique_records, base_records, ordered_trace, target_value, prefix_audit, repeat_channels,
)
from robocasa.recovery.operator_screen import validate_retreat_controller
from robocasa.recovery.phase3b_training_horizon import (
    atomic_write_json, read_jsonl, sha256_file, load_registered_phase2_snapshot,
    load_branch_payload,
)
from robocasa.recovery.run_operator_screen import _policy_args
from robocasa.recovery.run_phase2_snapshot_replay import (
    _runtime, _make_fresh_branch_environment, _close_branch_environment,
    _call_with_deterministic_environment_seed, _configure_phase2_environment, _reset_env,
    append_jsonl, utc_now,
)
from robocasa.recovery.run_phase3b_training_horizon_continuation import (
    _runtime_args, _validate_live_provenance,
)


def reference_audit(reference, replayed):
    """Compare portable complete state, excluding capture identities/timestamps."""
    mismatches, xml = _compare_environment_states(
        reference.environment_state, replayed.environment_state, atol=0, rtol=0)
    channels = {"environment_state": not mismatches}
    details = {"environment_state": mismatches}
    for field in ("simulator_integration_state", "controller_state", "policy_state",
                  "observation", "subtask_eval", "tracker_state", "global_rng_state",
                  "environment_rng_state", "environment_control_state", "environment_step"):
        differences = compare_structures(getattr(reference, field), getattr(replayed, field))
        channels[field] = not differences
        if differences:
            details[field] = differences
    return {"channels": channels, "mismatches": details, "model_xml_audit": xml,
            "all_exact": all(channels.values()),
            "reference_limit": "complete_endpoint_and_latest_policy_records; source_does_not_store_entire_nominal_request_action_history"}


def capture_entry(reference, env, policy, runtime, *, progress=None):
    """Replay reset -> source prefix-2; keep its real target-stage entry."""
    seed = int(reference.metadata["environment_seed"])
    # Restore the original ordinary sampling configuration, then clear mutable
    # policy state, exactly as for a new nominal parent (no continuation seeds).
    policy.set_state(deepcopy(reference.policy_state))
    policy.reset()
    obs, _ = _call_with_deterministic_environment_seed(seed, _reset_env, env, seed)
    # Setup preserves module RNG streams. The source has no reset-time global
    # RNG checkpoint: reuse its endpoint streams only if this entire nominal
    # replay leaves them idle, and verify them against the source at the end.
    _restore_global_rng_state(reference.global_rng_state)
    rng_digest = stable_digest(_capture_global_rng_state())
    evaluations = [deepcopy(runtime["get_subtask_eval"](env))]
    predicate = reference.metadata["trigger_stage"].split("::", 1)[-1]
    current, local, entry = None, 0, None
    steps = []
    for step in range(reference.environment_step + 1):
        trace = ordered_trace(evaluations)
        stage = trace[-1]["ordered_current_subtask"]
        if stage != current:
            current, local, entry = stage, 0, None
        boundary = bool(policy.at_inference_boundary)
        if stage == predicate and boundary and local == 0 and entry is None:
            entry = capture_full_snapshot(env, policy, parent_id=reference.parent_id,
                task_name=reference.task_name, trigger_name="timing_common_stage_entry",
                environment_step=step, observation=obs, subtask_eval=evaluations[-1],
                metadata={**reference.metadata, "local_inferences": 0,
                          "source_prefix_snapshot_id": reference.snapshot_id,
                          "nominal_sampling_seed": int(policy.next_ordinary_sampling_seed),
                          "ordered_prefix": trace[-1]["ordered_completed_subtasks"],
                          "entry_definition": "first_eligible_inference_boundary_of_source_target_visit"})
        if step == reference.environment_step:
            if entry is None or not boundary or stage != predicate or local != 2:
                raise ValueError("Replay did not reach the original target prefix-2 boundary")
            reproduced = capture_full_snapshot(env, policy, parent_id=reference.parent_id,
                task_name=reference.task_name, trigger_name="timing_source_reference_audit",
                environment_step=step, observation=obs, subtask_eval=evaluations[-1])
            audit = reference_audit(reference, reproduced)
            audit["global_rng_replay_contract"] = "source_endpoint_streams_reused_only_with_no_global_advancement_during_entire_nominal_replay"
            return entry, {"steps": steps, "subtask_evals": evaluations}, audit, reproduced
        action = runtime["call_policy"](policy, obs)
        requests, inference = policy.pop_request_records(), policy.pop_inference_record()
        if bool(requests) != (inference is not None) or len(requests) > 1:
            raise ValueError("Nominal K=1 request/inference alignment failed")
        if inference is not None:
            if (requests[0]["request_sha256"] != inference["request_sha256"]
                    or requests[0]["sampling_seed"] != inference["sampling_seed"]):
                raise ValueError("Nominal request/inference content mismatch")
            if stage == predicate:
                local += 1
        before = stable_digest(obs)
        obs, reward, done, info = runtime["step_fn"](env, action)
        evaluations.append(deepcopy(runtime["get_subtask_eval"](env)))
        steps.append({"step": step, "action": deepcopy(action), "observation_sha256": before,
                      "next_observation_sha256": stable_digest(obs), "requests": deepcopy(requests),
                      "inference": deepcopy(inference),
                      "causal_environment_sha256": stable_digest(causal_transition_fingerprint(environment_fingerprint(env)))})
        if progress is not None:
            progress({"step": step, "observation_sha256": before, "next_observation_sha256": stable_digest(obs),
                      "action_sha256": stable_digest(action), "policy_state_sha256": stable_digest(policy.get_state()),
                      "causal_environment_sha256": steps[-1]["causal_environment_sha256"],
                      "request_sha256": [r["request_sha256"] for r in requests],
                      "request_seeds": [r["sampling_seed"] for r in requests],
                      "target_stage": stage, "local_inferences": local})
        if stable_digest(_capture_global_rng_state()) != rng_digest:
            raise ValueError("Nominal replay advances unrecorded reset-time global RNG; cannot certify the old parent")
        if done or runtime["success_fn"](info=info, reward=reward, env=env):
            raise ValueError("Nominal replay terminated before the source reference")


def _recover_bases(root, registration):
    rows = base_records(root, registration)
    for parent in registration["parents"]:
        marker = root/"bases"/parent["parent_id"]/"committed.json"
        if parent["parent_id"] not in rows and marker.exists():
            row = json.loads(marker.read_text())
            append_jsonl(root/"base_records.jsonl", row)
    return base_records(root, registration)


def _capture_base(root, registration, parent, env, policy, runtime, reference, integrity):
    folder = root/"bases"/parent["parent_id"]
    folder.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=folder))
    snapshot, nominal, audit, reproduced = capture_entry(reference, env, policy, runtime,
        progress=lambda row: append_jsonl(attempt/"nominal_progress.jsonl", row))
    audit["source_integrity"] = integrity
    atomic_write_json(attempt/"replay_audit.json", audit)
    with gzip.open(attempt/"nominal_replay.pkl.gz", "wb") as stream:
        pickle.dump(nominal, stream, protocol=5)
    save_full_snapshot(reproduced, attempt/"reproduced_reference.pkl.gz")
    if not audit["all_exact"]:
        raise ValueError(f"Source replay mismatch; preserved audit at {attempt}")
    save_full_snapshot(snapshot, attempt/"entry.pkl.gz")
    paths = [attempt/name for name in ("entry.pkl.gz", "replay_audit.json", "nominal_replay.pkl.gz", "reproduced_reference.pkl.gz", "nominal_progress.jsonl")]
    row = {"parent_id": parent["parent_id"], "task_name": parent["task_name"],
           "snapshot_id": snapshot.snapshot_id, "snapshot_path": str(paths[0].relative_to(root)),
           "source_prefix_snapshot_id": reference.snapshot_id,
           "entry_environment_step": snapshot.environment_step,
           "ordered_prefix": snapshot.metadata["ordered_prefix"], "replay_audit": audit,
           "registration_sha256": registration["registration_sha256"],
           "artifact_hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths}}
    atomic_write_json(folder/"committed.json", row)  # commit marker precedes append
    append_jsonl(root/"base_records.jsonl", row)
    return row


def execute_branch(snapshot, env, policy, branch, operator_spec, runtime, *, atol=1e-8, rtol=1e-8):
    restore = restore_full_snapshot(snapshot, env, policy, atol=atol, rtol=rtol)
    evaluation = deepcopy(runtime["get_subtask_eval"](env))
    if not restore["valid"] or stable_digest(evaluation) != stable_digest(snapshot.subtask_eval):
        raise ValueError("Common-entry complete restore or semantic equality failed")
    if not policy.at_inference_boundary:
        raise ValueError("Common-entry restore is not an inference boundary")
    predicate = snapshot.metadata["trigger_stage"].split("::", 1)[-1]
    if target_value(evaluation, predicate):
        raise ValueError("Common entry target must initially be incomplete")
    required = list(evaluation["required_predicates"])
    wrapped = TimingPolicy(policy, branch, operator_spec, predicate)
    controller = validate_retreat_controller(env) if branch["operator"] == "retreat_reset_replan" else None
    obs, evaluations = deepcopy(snapshot.observation), [evaluation]
    actions, requests, inferences, request_steps, inference_steps = [], [], [], [], []
    observations, causal_states, rewards, infos, decisions = [stable_digest(obs)], [], [], [], []
    policy_states = [stable_digest(policy.get_state())]
    global_rng_states = [stable_digest(_capture_global_rng_state())]
    policy_step_start = policy.get_state()["environment_step"]
    prefix = snapshot.metadata["ordered_prefix"]
    termination = "suffix_horizon"
    started = time.monotonic()
    for step in range(branch["horizon_environment_steps"]):
        trace = ordered_trace(evaluations, prefix)
        decision_before = deepcopy(wrapped.decision)
        wrapped.decide(evaluations[-1], trace[-1])
        decisions.append({"step": step, "at_inference_boundary": bool(policy.at_inference_boundary),
                          "pre_action_requests": wrapped.requests, "target_complete": target_value(evaluations[-1], predicate),
                          "ordered_current_subtask": trace[-1]["ordered_current_subtask"],
                          "new_decision": deepcopy(wrapped.decision) if decision_before != wrapped.decision else None})
        action = wrapped(obs)
        current, inference = policy.pop_request_records(), policy.pop_inference_record()
        requests.extend(deepcopy(current)); request_steps.extend([step]*len(current))
        if inference is not None:
            inferences.append(deepcopy(inference)); inference_steps.append(step)
        actions.append(deepcopy(action))
        obs, reward, done, info = runtime["step_fn"](env, action)
        rewards.append(reward); infos.append(deepcopy(info))
        observations.append(stable_digest(obs))
        causal_states.append(stable_digest(causal_transition_fingerprint(environment_fingerprint(env))))
        next_evaluation = deepcopy(runtime["get_subtask_eval"](env))
        if next_evaluation.get("required_predicates") != required:
            raise ValueError("Required stage predicates changed during a timing branch")
        for name in required:
            target_value(next_evaluation, name)
        evaluations.append(next_evaluation)
        policy_states.append(stable_digest(policy.get_state()))
        global_rng_states.append(stable_digest(_capture_global_rng_state()))
        if runtime["success_fn"](info=info, reward=reward, env=env):
            termination = "success"; break
        if done:
            safety = isinstance(info, dict) and (info.get("safety_violation") or info.get("safety_termination")
                or "safety" in str(info.get("termination_reason", "")).lower())
            termination = "safety_termination" if safety else "environment_done"
            break
    trace = ordered_trace(evaluations, prefix)
    alignment = (len(requests) == len(inferences) == wrapped.requests and request_steps == inference_steps
        and all(r.get("request_sha256") == i.get("request_sha256") and r.get("sampling_seed") == i.get("sampling_seed")
                for r, i in zip(requests, inferences)))
    # A scripted retreat can legitimately terminate before its first request.
    zero_valid = bool(wrapped.treatment and branch["operator"] == "retreat_reset_replan"
                      and len(actions) <= operator_spec["retreat"]["release_steps"] + operator_spec["retreat"]["lift_steps"]
                      and termination != "suffix_horizon")
    if not alignment or (not requests and not zero_valid):
        raise ValueError("Timing request/inference accounting failed")
    summary = {**branch, "protocol": PROTOCOL, "created_at": utc_now(),
        "source_base_snapshot_id": snapshot.snapshot_id, "snapshot_id": snapshot.snapshot_id,
        "entry_environment_step": snapshot.environment_step, "ordered_prefix": prefix,
        "policy_environment_step_start": policy_step_start,
        "policy_environment_step_end": policy.get_state()["environment_step"],
        "restore_valid": restore["valid"], "restore_audit": restore,
        "restore_subtask_exact": True, "record_alignment_valid": alignment,
        "zero_request_scripted_termination_valid": zero_valid if not requests else False,
        "intervention": wrapped.final_decision(termination),
        "operator_events": wrapped.treatment.events if wrapped.treatment else [],
        "retreat_controller_audit": controller,
        "request_sampling_seeds": [r["sampling_seed"] for r in requests],
        "request_environment_step_indices": request_steps, "inference_environment_step_indices": inference_steps,
        "suffix_request_sha256": [stable_digest(r) for r in requests],
        "suffix_action_sha256": [stable_digest(a) for a in actions],
        "suffix_observation_sha256": observations, "suffix_environment_sha256": causal_states,
        "suffix_policy_state_sha256": policy_states, "suffix_global_rng_sha256": global_rng_states,
        "semantic_sha256": stable_digest(evaluations), "num_steps": len(actions),
        "num_policy_requests": len(requests), "num_inference_records": len(inferences),
        "termination_reason": termination, "task_success": termination == "success",
        "regressed_predicates": sorted({p for e in trace for p in e.get("regressed_predicates", [])}),
        "wall_seconds": time.monotonic()-started}
    payload = {"summary": summary, "actions": actions, "request_records": requests,
               "inference_records": inferences, "subtask_evals": evaluations,
               "subtask_trace": trace, "timing_decisions": decisions, "rewards": rewards, "infos": infos}
    summary["payload_sha256"] = branch_payload_digest(payload)
    return {"summary": summary, "payload": payload}


def timing_audit(payload, branch):
    decision = payload["summary"]["intervention"]
    if branch["operator"] == "continue":
        return decision["reason"] == "shared_continuation" and not decision["executed"]
    actual = [s["new_decision"] for s in payload["timing_decisions"] if s["new_decision"]]
    if not actual:
        return not decision["executed"] and decision["reason"].startswith("boundary_not_reached_")
    if len(actual) != 1 or actual[0] != decision:
        return False
    point = payload["timing_decisions"][decision["environment_step"]]
    predicate = payload["subtask_evals"][0]["required_predicates"]
    # The target is the first stage after the captured upstream ordered prefix.
    prefix = payload["summary"]["ordered_prefix"]
    target = predicate[len(prefix)]
    if decision["executed"]:
        return (point["at_inference_boundary"] and not point["target_complete"]
                and point["ordered_current_subtask"] == target
                and point["pre_action_requests"] == branch["delay"] == decision["pre_application_requests"])
    return ((decision["reason"] == "target_completed_before_application" and point["target_complete"])
            or (decision["reason"] == "target_stage_ineligible_before_application" and point["ordered_current_subtask"] != target))


def engineering_analysis(root, registration, scope):
    root = Path(root)
    validate_registration(registration)
    all_records = unique_records(root, registration)
    selected = [b for b in registration["branches"] if scope == "full" or b["sentinel"]]
    records = {b["branch_id"]: all_records[b["branch_id"]] for b in selected if b["branch_id"] in all_records}
    bases = base_records(root, registration)
    payloads = {bid: load_branch_payload(root, row) for bid, row in records.items()}
    prefixes, repeats, changes = [], [], []
    for branch in selected:
        bid = branch["branch_id"]
        if bid not in records:
            continue
        row = records[bid]
        control_id = f"{branch['parent_id']}--continue--s{branch['schedule']}--r0"
        if branch["operator"] != "continue":
            audit = prefix_audit(payloads[bid], payloads[control_id])
            prefixes.append({"branch_id": bid, **audit})
            if branch["operator"] == "history_reset_replan" and row["intervention"]["executed"]:
                n = row["intervention"]["pre_application_requests"]
                control_requests = payloads[control_id]["request_records"]
                if len(control_requests) > n and row["num_policy_requests"] > n:
                    changes.append(payloads[bid]["request_records"][n]["request_sha256"] != control_requests[n]["request_sha256"])
        if branch["role"] == "repeat":
            original = bid.rsplit("--r", 1)[0]+"--r0"
            repeats.append({"branch_id": bid, "exact": original in records and repeat_channels(row) == repeat_channels(records[original])})
    expected_parents = {b["parent_id"] for b in selected}
    errors = read_jsonl(root/"errors.jsonl")
    gates = {
        "complete_unique_branch_support": set(records) == {b["branch_id"] for b in selected},
        "all_payloads_and_embedded_records_exact": True,
        "source_replay_and_base_integrity": expected_parents.issubset(bases),
        "all_complete_restores_valid": bool(records) and all(r["restore_valid"] and r["restore_subtask_exact"] for r in records.values()),
        "common_base_identity": all(r["parent_id"] in bases and r["source_base_snapshot_id"] == bases[r["parent_id"]]["snapshot_id"] for r in records.values()),
        "all_pre_intervention_prefixes_exact": bool(prefixes) and all(p["all_exact"] for p in prefixes),
        "timing_and_eligibility_exact": bool(records) and all(timing_audit(payloads[bid], records[bid]) for bid in records),
        "request_alignment_and_seed_schedules": bool(records) and all(r["record_alignment_valid"] and
            r["request_sampling_seeds"] == list(range(r["sampling_seed"], r["sampling_seed"]+r["num_policy_requests"])) for r in records.values()),
        "common_absolute_endpoints": bool(records) and all(0 < r["num_steps"] <= r["horizon_environment_steps"] and
            r["policy_environment_step_end"]-r["policy_environment_step_start"] == r["num_steps"] and
            r["termination_reason"] in {"suffix_horizon", "success", "environment_done", "safety_termination"} and
            (r["termination_reason"] != "suffix_horizon" or r["num_steps"] == r["horizon_environment_steps"]) for r in records.values()),
        "sentinel_repeats_exact": len(repeats) == 20 and all(r["exact"] for r in repeats),
        "history_reset_applied": all(any(e["event"] == "history_reset" for e in r["operator_events"])
            for r in records.values() if r["operator"] == "history_reset_replan" and r["intervention"]["executed"]),
        "retreat_controller_validated": all(r.get("retreat_controller_audit", {}).get("accepted") for r in records.values() if r["operator"] == "retreat_reset_replan"),
        "zero_errors": not errors,
    }
    return {"protocol": PROTOCOL, "scope": scope, "branches": len(records), "expected_branches": len(selected),
            "primary_branches": sum(r["role"] == "primary" for r in records.values()), "errors": len(errors),
            "engineering_gates": gates, "engineering_all_pass": all(gates.values()),
            "prefix_audits": prefixes, "repeat_audits": repeats, "history_reset_effect_pairs": len(changes),
            "history_reset_changed_request_pairs": sum(changes),
            "scientific_outcomes_used_as_engineering_gate": False}


def run(args, runtime=None):
    spec = json.loads(args.timing_spec.read_text()) if args.timing_spec else deepcopy(DEFAULT_SPEC)
    root = args.output_dir.resolve()
    if args.dry_run:
        registration = build_registration(args.operator_run_dir, spec)
        plan = json.loads((Path(registration["phase2_run_dir"])/"plan.json").read_text())
        _validate_live_provenance(args, registration["phase2_registration"], plan)
        print(json.dumps({k: registration[k] for k in ("spec", "expected_primary_branches", "expected_total_branches", "expected_sentinel_branches", "parents")}, indent=2))
        return registration
    root.mkdir(parents=True, exist_ok=True)
    with (root/".timing.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another collector owns this timing root") from error
        if args.resume:
            registration = json.loads((root/"registration.json").read_text())
            validate_registration(registration)
            if registration["operator_run_dir"] != str(args.operator_run_dir.resolve()) or spec != registration["spec"]:
                raise ValueError("Resume source/specification differs from registration")
        else:
            if args.scope != "sentinel":
                raise ValueError("Start with the engineering sentinel")
            if any(p.name != ".timing.lock" for p in root.iterdir()):
                raise FileExistsError("Timing root is not empty; use --resume")
            registration = build_registration(args.operator_run_dir, spec)
            atomic_write_json(root/"registration.json", registration)
        if read_jsonl(root/"errors.jsonl"):
            raise ValueError("Technical errors recorded; preserve this root and diagnose before using a new root")
        completed = unique_records(root, registration, recover_orphans=True)
        bases = _recover_bases(root, registration)
        if args.scope == "full":
            sentinel = engineering_analysis(root, registration, "sentinel")
            if not sentinel["engineering_all_pass"]:
                raise ValueError("Full collection blocked by engineering sentinel")
        plan = json.loads((Path(registration["phase2_run_dir"])/"plan.json").read_text())
        _validate_live_provenance(args, registration["phase2_registration"], plan)
        runtime = runtime or _runtime()
        config = _runtime_args(args, plan)
        factory, policy = runtime["load_factory"](plan["policy_module"]), None
        selected = [b for b in registration["branches"] if args.scope == "full" or b["sentinel"]]
        parents = {p["parent_id"]: p for p in registration["parents"]}
        current = None
        try:
            for branch in selected:
                if branch["branch_id"] in completed:
                    continue
                current = branch
                atomic_write_json(root/"status.json", {"status": "running", "scope": args.scope,
                    "branches": sum(b["branch_id"] in completed for b in selected), "expected_branches": len(selected),
                    "current_branch": branch["branch_id"], "updated_at": utc_now()})
                parent = parents[branch["parent_id"]]
                if parent["parent_id"] not in bases:
                    sid = parent["snapshot_id"]
                    reference, integrity = load_registered_phase2_snapshot(
                        Path(registration["phase2_run_dir"])/"snapshots"/f"{sid}.pkl.gz",
                        expected_file_sha256=registration["phase2_registration"]["phase2_snapshot_file_sha256"][sid],
                        expected_snapshot_id=sid, expected_parent_id=parent["parent_id"], expected_task_name=parent["task_name"])
                    seed = int(reference.metadata["environment_seed"])
                    env = _call_with_deterministic_environment_seed(seed, runtime["make_env"],
                        reference.task_name, config.env_interface, config.split, seed, True)
                    try:
                        _configure_phase2_environment(env, config)
                        if policy is None:
                            policy = runtime["call_factory"](factory, env, _policy_args(reference, args))
                        bases[parent["parent_id"]] = _capture_base(root, registration, parent, env, policy, runtime, reference, integrity)
                    finally:
                        _close_branch_environment(env)
                snapshot = load_full_snapshot(root/bases[parent["parent_id"]]["snapshot_path"])
                env = _make_fresh_branch_environment(snapshot, config, runtime)
                try:
                    if policy is None:
                        policy = runtime["call_factory"](factory, env, _policy_args(snapshot, args))
                    result = execute_branch(snapshot, env, policy, branch, registration["operator_spec"], runtime,
                                            atol=args.restore_atol, rtol=args.restore_rtol)
                    completed[branch["branch_id"]] = save_branch_result(result, root)
                finally:
                    _close_branch_environment(env)
                print(f"[{sum(b['branch_id'] in completed for b in selected)}/{len(selected)}] {branch['branch_id']}", flush=True)
        except Exception as error:
            append_jsonl(root/"errors.jsonl", {"branch": current, "error": str(error), "traceback": traceback.format_exc(), "created_at": utc_now()})
            atomic_write_json(root/"status.json", {"status": "failed", "error": str(error), "updated_at": utc_now()})
            raise
        finally:
            if policy is not None:
                policy.close()
        analysis = engineering_analysis(root, registration, args.scope)
        atomic_write_json(root/("sentinel_analysis.json" if args.scope == "sentinel" else "analysis.json"), analysis)
        atomic_write_json(root/"status.json", {**analysis, "status": "complete" if analysis["engineering_all_pass"] else "failed_gate", "updated_at": utc_now()})
        print(json.dumps({k: v for k, v in analysis.items() if k not in {"prefix_audits", "repeat_audits"}}, indent=2))
        if not analysis["engineering_all_pass"]:
            raise ValueError("Timing engineering gate failed; preserve artifacts and diagnose")
        return analysis


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("sentinel", "full"), required=True)
    parser.add_argument("--timing-spec", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10318)
    parser.set_defaults(restore_atol=1e-8, restore_rtol=1e-8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        if not args.dry_run and (args.output_dir/"registration.json").exists():
            atomic_write_json(args.output_dir/"status.json", {
                "status": "failed", "scope": args.scope, "error": str(error), "updated_at": utc_now()})
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
