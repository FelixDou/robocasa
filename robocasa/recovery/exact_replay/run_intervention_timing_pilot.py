"""Resume timing research with a fixed pool, audited exclusions and intact pairing."""

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import html
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from types import SimpleNamespace

from robocasa.recovery.exact_replay import collect_intervention_timing_references as fresh
from robocasa.recovery.exact_replay.intervention_timing_pilot import (
    PROTOCOL, SUFFIX_GC_SCOPE, build_registration, validate_registration, checked, artifacts,
    commit_parent, parent_records, assignment,
)
from robocasa.recovery.full_snapshot import (
    stable_digest, load_full_snapshot, _environment_nodes,
    _capture_global_rng_state, _restore_global_rng_state, _restore_environment_rng_state,
    environment_fingerprint, causal_transition_fingerprint,
)
from robocasa.recovery.exact_replay.intervention_timing_observation_state import capture_observation_runtime, restore_observation_runtime
from robocasa.recovery.exact_replay.intervention_timing import TimingPolicy, ordered_trace, target_value, unique_records, prefix_audit, repeat_channels
from robocasa.recovery.exact_replay.counterfactual_branch import branch_payload_digest, save_branch_result
from robocasa.recovery.exact_replay.run_intervention_timing import timing_audit
from robocasa.recovery.exact_replay.operator_screen import validate_retreat_controller
from robocasa.recovery.exact_replay.phase3b_training_horizon import (
    atomic_write_json, read_jsonl, sha256_file, sha256_json, load_branch_payload, write_jsonl,
)
from robocasa.recovery.exact_replay.run_phase2_snapshot_replay import (
    _runtime, _make_fresh_branch_environment, _close_branch_environment, append_jsonl, utc_now,
)
from robocasa.recovery.exact_replay.run_phase3b_training_horizon_continuation import _runtime_args, _validate_live_provenance
from robocasa.recovery.exact_replay.diagnose_intervention_timing_replay import (
    GCTrace, diagnostic_state, initialize_rng_capture_runtime, record_startup_event,
)


class InvalidParent(ValueError):
    """A recorded eligibility failure before any recovery treatment."""


def cached_observation(env):
    """Reconstruct the policy view from restored live Observable values.

    A saved sampled observation need not equal a new instantaneous measurement.
    This reads the actual adapter's non-forcing API, never snapshot.observation.
    The caller verifies that all declared physical/sampler/RNG state is intact.
    """
    for path, node in _environment_nodes(env):
        if inspect.getattr_static(node, "get_observation", None) is None:
            continue
        inner = getattr(node, "env", None)
        getter = getattr(inner, "_get_observations", None)
        if callable(getter):
            if getattr(inner, "viewer_get_obs", False):
                raise InvalidParent("Viewer-owned observation sampling is unsupported")
            observation = node.get_observation(getter(force_update=False))
            fresh.camera_format(observation)
            return observation, path+".get_observation(env._get_observations(force_update=False))"
    raise InvalidParent("No non-forcing live sampled-observation API found")


def restore_entry(snapshot, env, policy, runtime, folder, *, guard=None):
    restore = fresh.restore_owned_snapshot(snapshot, env, policy)
    if not restore["valid"]:
        atomic_write_json(folder/"restore_audit.json", restore)
        raise InvalidParent("Common-entry physical/control restore differs")
    restore_observation_runtime(env, snapshot.metadata["observation_runtime_state"])
    _restore_environment_rng_state(env, snapshot.environment_rng_state)
    _restore_global_rng_state(snapshot.global_rng_state)
    if guard is not None:
        # reset_to() can explicitly collect during environment reconstruction.
        # That remains setup, as in v4; the audited suffix starts afterward.
        guard.phase, guard.nominal = "restored_entry_observation_and_evaluation", True
        guard.step = snapshot.environment_step
    observation, getter = cached_observation(env)
    evaluation = deepcopy(runtime["get_subtask_eval"](env))
    current = diagnostic_state(env, policy, observation, evaluation, snapshot.environment_step)
    current["observation_runtime_state"] = capture_observation_runtime(env)
    audit = fresh.fresh_state_audit(snapshot, SimpleNamespace(**current))
    restore.update(valid=restore["valid"] and audit["all_exact"], complete_state_audit=audit,
        observation_verification={"getter": getter, "snapshot_fallback_used": False,
            "scope": "restored_sampler_values_not_a_new_instantaneous_sensor_measurement"})
    atomic_write_json(folder/"restore_audit.json", restore)
    if not restore["valid"] or not policy.at_inference_boundary:
        raise InvalidParent("Common-entry sampled observation, runtime or semantic restore differs")
    predicate = snapshot.metadata["trigger_stage"].split("::", 1)[-1]
    if target_value(evaluation, predicate):
        raise InvalidParent("Common-entry target is already complete")
    return observation, evaluation, restore


def pilot_repeat(record):
    return {**repeat_channels(record), **{key: record[key] for key in (
        "suffix_observation_runtime_sha256", "encoded_request_sha256", "encoded_response_sha256")}}


def pilot_prefix(payload, control):
    audit = prefix_audit(payload, control)
    n = audit["pre_intervention_steps"]
    a, b = payload["summary"], control["summary"]
    audit["channels"]["observable_runtime"] = a["suffix_observation_runtime_sha256"][:n+1] == b["suffix_observation_runtime_sha256"][:n+1]
    for key in ("encoded_request_sha256", "encoded_response_sha256"):
        vectors = [[value for value, step in zip(row[key], row["request_environment_step_indices"]) if step < n]
                   for row in (a, b)]
        audit["channels"][key] = vectors[0] == vectors[1]
    audit["all_exact"] = all(audit["channels"].values())
    return audit


def fatal(error):
    # Storage, transport and resource failures stop the run, rather than
    # being relabeled as bad seeds and silently excluding the whole pool.
    return isinstance(error, (OSError, MemoryError, ImportError)) or "out of memory" in str(error).lower()


def suffix_gc_pass(trace):
    return (trace["mode"] == "disabled" and trace["nominal_scope"] == SUFFIX_GC_SCOPE
        and not trace["enabled_at_environment_creation"] and not trace["enabled_at_end_of_cleanup"]
        and trace["original_enabled_state_restored"] and trace["nominal_collection_stops"] == 0
        and all(not e["automatic_gc_enabled"] for e in trace["events"] if e["event"] == "stop"))


def run_one(root, registration, parent, snapshot, branch, plan, args, runtime, factory):
    base = root/"branch_attempts"/branch["branch_id"]
    base.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="attempt_", dir=base))
    (folder/"requests").mkdir()
    guard, env, policy = GCTrace("disabled"), None, None
    guard.start()
    try:
        record_startup_event(folder/"startup_events.jsonl", "before_environment_creation")
        env = _make_fresh_branch_environment(snapshot, _runtime_args(args, plan), runtime)
        policy = runtime["call_factory"](factory, env, deepcopy(parent["policy_args"]))
        policy.client = fresh.RequestRecorder(policy.client, folder)
        guard.phase, guard.nominal = "restore_setup", False
        result = execute_branch(snapshot, env, policy, branch, registration["operator_spec"], runtime, folder,
            guard=guard, progress=lambda step: atomic_write_json(folder/"status.json", {"status": "running", "environment_step": step}))
    finally:
        guard.phase, guard.nominal = "cleanup", False
        try:
            if env is not None:
                _close_branch_environment(env)
        finally:
            try:
                if policy is not None:
                    policy.close()
            finally:
                trace = guard.finish()
                trace["nominal_scope"] = SUFFIX_GC_SCOPE
                atomic_write_json(folder/"gc_trace.json", trace)
    if not suffix_gc_pass(trace):
        raise InvalidParent("Cyclic collection occurred during restored suffix")
    startup = read_jsonl(folder/"startup_events.jsonl")[0]
    if registration["early_rng_runtime_initialization"]["torch_cuda_available"] and not startup["cuda_initialized"]:
        raise InvalidParent("CUDA RNG inspection was not initialized before environment creation")
    atomic_write_json(folder/"status.json", {"status": "complete", "environment_step": result["summary"]["num_steps"]})
    result["summary"].update(registration_sha256=registration["registration_sha256"],
        gc_guard=trace, branch_artifact_hashes=artifacts(root, folder))
    result["summary"]["payload_sha256"] = branch_payload_digest(result["payload"])
    return save_branch_result(result, root)


# Treatment loop retained from the frozen executor, with a new audited restore
# and complete sampler / encoded-response traces. Frozen files stay unchanged.
def execute_branch(snapshot, env, policy, branch, operator_spec, runtime, folder, *, progress=None, guard=None):
    obs, evaluation, restore = restore_entry(snapshot, env, policy, runtime, folder, guard=guard)
    predicate = snapshot.metadata["trigger_stage"].split("::", 1)[-1]
    required = list(evaluation["required_predicates"])
    wrapped = TimingPolicy(policy, branch, operator_spec, predicate)
    controller = validate_retreat_controller(env) if branch["operator"] == "retreat_reset_replan" else None
    evaluations = [evaluation]
    actions, requests, inferences, request_steps, inference_steps = [], [], [], [], []
    observations, causal_states, rewards, infos, decisions = [stable_digest(obs)], [], [], [], []
    policy_states = [stable_digest(policy.get_state())]
    global_rng_states = [stable_digest(_capture_global_rng_state())]
    observer_states = [stable_digest(capture_observation_runtime(env))]
    policy_step_start = policy.get_state()["environment_step"]
    prefix = snapshot.metadata["ordered_prefix"]
    termination = "suffix_horizon"
    started = time.monotonic()
    for step in range(branch["horizon_environment_steps"]):
        if guard is not None:
            guard.phase, guard.step = "policy_call", snapshot.environment_step+step
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
        if guard is not None:
            guard.phase = "environment_step"
        obs, reward, done, info = runtime["step_fn"](env, action)
        if guard is not None:
            guard.phase, guard.step = "state_capture", snapshot.environment_step+step+1
        rewards.append(reward); infos.append(deepcopy(info))
        fresh.camera_format(obs)
        observer_states.append(stable_digest(capture_observation_runtime(env)))
        observations.append(stable_digest(obs))
        if progress is not None and (step+1) % 16 == 0:
            progress(step+1)
        causal_states.append(stable_digest(causal_transition_fingerprint(environment_fingerprint(env))))
        if guard is not None:
            guard.phase = "evaluation"
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
    if guard is not None:
        guard.phase = "suffix_capture"
    encoded = read_jsonl(folder/"requests.jsonl")
    if len(encoded) != len(requests) or any(e["request_sha256"] != r["request_sha256"] for e, r in zip(encoded, requests)):
        raise ValueError("Full encoded request trace differs from policy records")
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
        "suffix_observation_runtime_sha256": observer_states,
        "encoded_request_sha256": [e["request_sha256"] for e in encoded],
        "encoded_response_sha256": [e["response_sha256"] for e in encoded],
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


def collect_references(root, registration, plan, args, runtime, factory):
    rows = parent_records(root, "references", registration, recover=True)
    for parent in registration["parents"]:
        pid = parent["parent_id"]
        if pid in rows:
            continue
        atomic_write_json(root/"status.json", {"status": "screening_references", "attempted_parents": len(rows),
            "excluded_parents": sum(r["kind"] == "excluded" for r in rows.values()), "current_parent": pid})
        base = root/"references"/pid
        base.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="reference_", dir=base))
        imported = registration["reuse_manifest"].get(pid)
        data = {}
        if imported:
            source = Path(imported["source_root"])
            if imported["kind"] == "reference_reused":
                old = imported["source_record"]
                old_folder = Path(old["report_path"]).parent
                files = [name for name in old["artifact_hashes"] if Path(name).is_relative_to(old_folder)]
                if len(files) != len(old["artifact_hashes"]):
                    raise ValueError("Reused reference artifacts do not share their collection directory")
                for name in files:
                    destination = folder/Path(name).relative_to(old_folder)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(checked(source, name), destination)
                data = {"kind": "reference_passed", "reused": True,
                    "snapshot_path": str((folder/Path(old["snapshot_path"]).relative_to(old_folder)).relative_to(root)),
                    "camera_review_path": str((folder/"camera_review.html").relative_to(root)),
                    "source_registration_sha256": old["registration_sha256"]}
            else:
                old_folder = checked(source, imported["source_folder"])
                shutil.copytree(old_folder, folder/"evidence")
                data = {"kind": "excluded", "reason": imported["reason"], "reused": True,
                        "phase": "reference_before_intervention"}
        else:
            collection = folder/"collection"
            collection.mkdir()
            try:
                report = fresh.collect_parent(parent, collection, registration, plan, args, runtime, factory)
                if report["engineering_all_pass"]:
                    data = {"kind": "reference_passed", "reused": False,
                        "snapshot_path": str((collection/"attempt_0/entry.pkl.gz").relative_to(root)),
                        "camera_review_path": str((collection/"camera_review.html").relative_to(root))}
                else:
                    data = {"kind": "excluded", "phase": "reference_before_intervention",
                        "reason": "reference_engineering_gates_failed", "failed_gates": [
                            key for key, valid in report["engineering_gates"].items() if not valid]}
            except ValueError as error:
                known = ("New reset-state", "Expected raw uint8 RGB", "Require a live observation dictionary")
                if not str(error).startswith(known):
                    raise
                data = {"kind": "excluded", "phase": "reference_before_intervention", "reason": str(error)}
                atomic_write_json(folder/"exclusion.json", {**data, "traceback": traceback.format_exc()})
        if data["kind"] == "reference_passed":
            snapshot = load_full_snapshot(checked(root, data["snapshot_path"]))
            if snapshot.parent_id != pid or snapshot.task_name != parent["task_name"] or snapshot.metadata["trigger_stage"] != parent["trigger_stage"]:
                raise ValueError("Pilot reference snapshot identity differs")
            data.update(snapshot_id=snapshot.snapshot_id, entry_environment_step=snapshot.environment_step,
                        ordered_prefix=snapshot.metadata["ordered_prefix"])
        rows[pid] = commit_parent(root, "references", parent, registration, folder, **data)
        print(f"reference [{len(rows)}/10] {pid}: {data['kind']}", flush=True)
    return rows


def checked_branches(root, registration, *, recover=False):
    rows = unique_records(root, registration, recover_orphans=recover)
    for row in rows.values():
        if row.get("registration_sha256") != registration["registration_sha256"] or row.get("protocol") != PROTOCOL:
            raise ValueError("Pilot branch belongs to another registration")
        for name, digest in row["branch_artifact_hashes"].items():
            if sha256_file(checked(root, name)) != digest:
                raise ValueError("Pilot encoded request or branch diagnostic artifact changed")
    return rows


def qualify(root, registration, references, plan, args, runtime, factory, records):
    rows = parent_records(root, "qualification", registration, recover=True)
    for parent in registration["parents"]:
        pid = parent["parent_id"]
        if pid in rows:
            continue
        base = root/"qualification"/pid
        base.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="preflight_", dir=base))
        if references[pid]["kind"] == "excluded":
            data = {"kind": "excluded", "phase": "reference_before_intervention", "reason": references[pid]["reason"]}
        else:
            atomic_write_json(root/"status.json", {"status": "qualifying_parents", "current_parent": pid,
                "qualified_parents": sum(r["kind"] == "eligible" for r in rows.values())})
            snapshot = load_full_snapshot(checked(root, references[pid]["snapshot_path"]))
            controls = [b for b in registration["branches"] if b["parent_id"] == pid and b["operator"] == "continue" and b["schedule"] == 0]
            try:
                for branch in controls:
                    if branch["branch_id"] not in records:
                        records[branch["branch_id"]] = run_one(root, registration, parent, snapshot, branch, plan, args, runtime, factory)
                exact = pilot_repeat(records[controls[0]["branch_id"]]) == pilot_repeat(records[controls[1]["branch_id"]])
                atomic_write_json(folder/"control_repeat_audit.json", {"exact": exact, "branches": [b["branch_id"] for b in controls]})
                data = {"kind": "eligible" if exact else "excluded", "phase": "control_preflight_before_intervention",
                    "reason": "exact_continue_repetitions" if exact else "continue_repetition_mismatch",
                    "control_payload_hashes": {b["branch_id"]: records[b["branch_id"]]["payload_file_sha256"] for b in controls}}
            except Exception as error:
                if fatal(error):
                    raise
                data = {"kind": "excluded", "phase": "control_preflight_before_intervention", "reason": str(error)}
                atomic_write_json(folder/"exclusion.json", {**data, "traceback": traceback.format_exc()})
        rows[pid] = commit_parent(root, "qualification", parent, registration, folder, **data)
        print(f"eligibility [{len(rows)}/10] {pid}: {data['kind']} ({data['reason']})", flush=True)
    return rows


def branch_errors(root, registration, *, recover=False):
    rows = read_jsonl(root/"timing_errors.jsonl")
    known = {r["branch_id"] for r in rows}
    if recover:
        for marker in sorted((root/"timing_errors").glob("*.json")):
            row = json.loads(marker.read_text())
            if row["branch_id"] not in known:
                append_jsonl(root/"timing_errors.jsonl", row)
                rows.append(row)
                known.add(row["branch_id"])
    expected = {b["branch_id"]: b for b in registration["branches"]}
    result = {}
    for row in rows:
        bid = row["branch_id"]
        if bid not in expected or bid in result or row["branch"] != expected[bid] or row["registration_sha256"] != registration["registration_sha256"]:
            raise ValueError("Invalid or duplicate terminal timing error")
        if json.loads((root/"timing_errors"/f"{bid}.json").read_text()) != row:
            raise ValueError("Timing error marker/ledger mismatch")
        for name, digest in row["artifact_hashes"].items():
            if sha256_file(checked(root, name)) != digest:
                raise ValueError("Terminal timing error evidence changed")
        result[bid] = row
    return result


def engineer(root, registration, assigned, records, errors, references):
    payloads = {bid: load_branch_payload(root, r) for bid, r in records.items() if bid in {b["branch_id"] for b in assigned["branches"]}}
    parents = {}
    for pid in assigned["eligible_parents"]:
        branches = [b for b in assigned["branches"] if b["parent_id"] == pid]
        available = {b["branch_id"]: records[b["branch_id"]] for b in branches if b["branch_id"] in records}
        prefixes, repeats = [], []
        for branch in branches:
            bid = branch["branch_id"]
            if bid not in available:
                continue
            if branch["operator"] != "continue":
                control = f"{pid}--continue--s{branch['schedule']}--r0"
                prefixes.append({"branch_id": bid, **(pilot_prefix(payloads[bid], payloads[control]) if control in payloads else {"all_exact": False})})
            if branch["role"] == "repeat":
                original = bid.rsplit("--r", 1)[0]+"--r0"
                repeats.append({"branch_id": bid, "exact": original in available and pilot_repeat(available[bid]) == pilot_repeat(available[original])})
        gates = {
            "complete_assigned_support": set(available) == {b["branch_id"] for b in branches},
            "zero_timing_errors": not any(b["branch_id"] in errors for b in branches),
            "all_restores_and_alignment_valid": bool(available) and all(r["restore_valid"] and r["record_alignment_valid"] for r in available.values()),
            "common_base_identity": all(r["source_base_snapshot_id"] == references[pid]["snapshot_id"] and
                r["entry_environment_step"] == references[pid]["entry_environment_step"] and
                r["ordered_prefix"] == references[pid]["ordered_prefix"] for r in available.values()),
            "all_prefixes_exact": bool(prefixes) and all(a["all_exact"] for a in prefixes),
            "all_assigned_repeats_exact": bool(repeats) and all(a["exact"] for a in repeats),
            "timing_and_eligibility_exact": bool(available) and all(timing_audit(payloads[bid], r) for bid, r in available.items()),
            "seed_schedules_exact": all(r["request_sampling_seeds"] == list(range(r["sampling_seed"], r["sampling_seed"]+r["num_policy_requests"])) for r in available.values()),
            "common_absolute_endpoints": all(0 < r["num_steps"] <= r["horizon_environment_steps"] and
                r["policy_environment_step_end"]-r["policy_environment_step_start"] == r["num_steps"] and
                (r["termination_reason"] != "suffix_horizon" or r["num_steps"] == r["horizon_environment_steps"]) for r in available.values()),
            "retreat_controller_validated": all((r.get("retreat_controller_audit") or {}).get("accepted")
                for r in available.values() if r["operator"] == "retreat_reset_replan"),
            "history_reset_applied": all(any(e["event"] == "history_reset" for e in r["operator_events"])
                for r in available.values() if r["operator"] == "history_reset_replan" and r["intervention"]["executed"]),
        }
        parents[pid] = {"engineering_gates": gates, "engineering_all_pass": all(gates.values()),
                        "prefix_audits": prefixes, "repeat_audits": repeats}
    result = {"parents": parents, "engineering_all_pass": bool(parents) and all(p["engineering_all_pass"] for p in parents.values()),
        "timing_errors": len(errors), "timing_errors_by_operator": dict(Counter(r["branch"]["operator"] for r in errors.values())),
        "scientific_outcomes_used_as_engineering_gate": False}
    atomic_write_json(root/"analysis.json", result)
    return result


def make_gallery(root, registration, references, assigned, report):
    template = {"protocol": PROTOCOL, "registration_sha256": registration["registration_sha256"],
        "assignment_sha256": assigned["assignment_sha256"], "branch_ledger_sha256": report["branch_ledger_sha256"],
        "parents": [{"parent_id": pid, "images_acceptable": None, "reviewed_all_frames": False, "notes": ""}
                    for pid in assigned["eligible_parents"]]}
    atomic_write_json(root/"image_review_template.json", template)
    links = "".join(f'<li>{html.escape(references[pid]["parent"]["task_name"])} seed '
        f'{references[pid]["parent"]["environment_seed"]}: <a href="{references[pid]["camera_review_path"]}">{pid}</a></li>'
        for pid in assigned["eligible_parents"])
    (root/"camera_review.html").write_text('<!doctype html><meta charset="utf-8"><title>Timing pilot nominal camera review</title>'
        '<h1>Nominal cameras: review before opening recovery outcomes</h1>'
        '<p>Inspect every frame in both nominal repetitions. Record corruption as unacceptable. '
        'This gallery contains no recovery outcomes; assessment is based on pre-intervention evidence. '
        'Repeatability alone does not certify camera health.</p><ul>'+links+'</ul>')


def finish(root, registration, references, qualifications, assigned, records, errors):
    engineering = engineer(root, registration, assigned, records, errors, references)
    by_task = {}
    for task in registration["spec"]["horizons"]:
        candidates = [p for p in registration["parents"] if p["task_name"] == task]
        accepted = [p for p in candidates if qualifications[p["parent_id"]]["kind"] == "eligible"]
        by_task[task] = {"attempted": len(candidates), "eligible": len(accepted), "excluded": len(candidates)-len(accepted),
            "timing_engineering_passed": sum(engineering["parents"][p["parent_id"]]["engineering_all_pass"] for p in accepted)}
    attempted = {b["branch_id"] for b in assigned["branches"]}
    resolved = set(records) | set(errors)
    status = "timing_complete_with_errors_awaiting_image_review" if not engineering["engineering_all_pass"] else "timing_complete_awaiting_image_review"
    if not assigned["eligible_parents"]:
        status = "no_eligible_parents"
    report = {"protocol": PROTOCOL, "status": status, "registration_sha256": registration["registration_sha256"],
        "assignment_sha256": assigned["assignment_sha256"], "attempted_parents": len(qualifications),
        "eligible_parents": len(assigned["eligible_parents"]), "excluded_parents": 10-len(assigned["eligible_parents"]),
        "coverage_by_task": by_task, "engineering_all_pass": engineering["engineering_all_pass"],
        "timing_collection_complete": attempted.issubset(resolved), "expected_branches": len(attempted),
        "successful_assigned_branches": len(attempted & set(records)), "timing_errors": len(errors),
        "timing_errors_by_operator": engineering["timing_errors_by_operator"],
        "prequalification_control_records": len(set(records)-attempted),
        "image_health": "pending_human_review", "pilot_scientific_analysis_ready": False,
        "full_timing_study_complete": False, "safe_triggered_recovery_tested": False,
        "branch_ledger_sha256": sha256_file(root/"branch_records.jsonl") if (root/"branch_records.jsonl").exists() else None,
        "updated_at": utc_now()}
    write_jsonl(root/"excluded_parents.jsonl", [r for r in qualifications.values() if r["kind"] == "excluded"])
    make_gallery(root, registration, references, assigned, report)
    atomic_write_json(root/"report.json", report)
    atomic_write_json(root/"status.json", report)
    return report


def run(args, runtime=None):
    root = args.output_dir.resolve()
    if root.is_relative_to(Path("/gs/fs")):
        raise ValueError("Pilot outputs must be outside /gs/fs")
    for source in (args.timing_run_dir, args.reference_run_dir):
        if source and (root.is_relative_to(source.resolve()) or source.resolve().is_relative_to(root)):
            raise ValueError("Pilot root must be separate from frozen sources")
    if args.dry_run:
        if args.resume:
            registration = validate_registration(root)
            validate_resume_args(args, registration)
            plan = json.loads((Path(registration["phase2_run_dir"])/"plan.json").read_text())
            _validate_live_provenance(args, registration["phase2_registration"], plan)
        else:
            registration, _ = build_registration(args)
        print(json.dumps({"protocol": PROTOCOL, "attempt_limit": 10, "potential_branches": len(registration["branches"]),
            "reused_references": sum(r["kind"] == "reference_reused" for r in registration["reuse_manifest"].values()),
            "recorded_exclusions": sum(r["kind"] == "reference_excluded" for r in registration["reuse_manifest"].values()),
            "timing_spec": registration["spec"]}, indent=2))
        return registration
    root.mkdir(parents=True, exist_ok=True)
    with (root/".pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            registration = validate_registration(root)
            validate_resume_args(args, registration)
            plan = json.loads((Path(registration["phase2_run_dir"])/"plan.json").read_text())
            _validate_live_provenance(args, registration["phase2_registration"], plan)
            if initialize_rng_capture_runtime() != registration["early_rng_runtime_initialization"]:
                raise ValueError("Pilot RNG runtime changed on resume")
        else:
            if any(p.name != ".pilot.lock" for p in root.iterdir()):
                raise FileExistsError("Pilot root exists; use resume or a new root")
            registration, plan = build_registration(args)
            registration.pop("registration_sha256")
            registration["early_rng_runtime_initialization"] = initialize_rng_capture_runtime()
            registration["registration_sha256"] = sha256_json(registration)
            atomic_write_json(root/"registration.json", registration)
        if read_jsonl(root/"fatal_errors.jsonl"):
            raise ValueError("Fatal infrastructure/configuration error recorded; inspect before a replacement run")
        # Finalized runs resume idempotently without overwriting image review.
        if (root/"report.json").exists() and json.loads((root/"report.json").read_text()).get("timing_collection_complete"):
            checked_branches(root, registration)
            parent_records(root, "references", registration)
            qualifications = parent_records(root, "qualification", registration)
            if json.loads((root/"assignment.json").read_text()) != assignment(registration, qualifications):
                raise ValueError("Pilot assignment changed")
            branch_errors(root, registration)
            saved = json.loads((root/"report.json").read_text())
            atomic_write_json(root/"status.json", saved)
            return saved
        runtime = runtime or _runtime()
        factory = runtime["load_factory"](plan["policy_module"])
        try:
            references = collect_references(root, registration, plan, args, runtime, factory)
            records = checked_branches(root, registration, recover=True)
            qualifications = qualify(root, registration, references, plan, args, runtime, factory, records)
            assigned = assignment(registration, qualifications)
            if (root/"assignment.json").exists() and json.loads((root/"assignment.json").read_text()) != assigned:
                raise ValueError("Pilot assignment changed after eligibility")
            atomic_write_json(root/"assignment.json", assigned)
            errors = branch_errors(root, registration, recover=True)
            if set(errors) & set(records):
                raise ValueError("A timing branch has both success and terminal error")
            parents = {p["parent_id"]: p for p in registration["parents"]}
            sentinel_ids = set(assigned["sentinels_by_task"].values())
            branches = sorted(assigned["branches"], key=lambda b: b["parent_id"] not in sentinel_ids)
            for branch in branches:
                bid, pid = branch["branch_id"], branch["parent_id"]
                if bid in records or bid in errors:
                    continue
                atomic_write_json(root/"status.json", {"status": "running_timing", "current_branch": bid,
                    "branches": len(set(records) & {b["branch_id"] for b in branches}),
                    "expected_branches": len(branches), "eligible_parents": len(assigned["eligible_parents"]),
                    "excluded_parents": 10-len(assigned["eligible_parents"]), "timing_errors": len(errors)})
                snapshot = load_full_snapshot(checked(root, references[pid]["snapshot_path"]))
                try:
                    records[bid] = run_one(root, registration, parents[pid], snapshot, branch, plan, args, runtime, factory)
                    print(f"timing [{len(set(records) & {b['branch_id'] for b in branches})}/{len(branches)}] {bid}", flush=True)
                except Exception as error:
                    if fatal(error):
                        raise
                    row = {"branch_id": bid, "branch": branch, "registration_sha256": registration["registration_sha256"],
                        "error": str(error), "traceback": traceback.format_exc(),
                        "phase": "timing_after_eligibility_not_a_pre_intervention_exclusion",
                        "artifact_hashes": artifacts(root, root/"branch_attempts"/bid), "created_at": utc_now()}
                    atomic_write_json(root/"timing_errors"/f"{bid}.json", row)
                    append_jsonl(root/"timing_errors.jsonl", row)
                    errors[bid] = row
                    print(f"timing ERROR {bid}: {error}; retained separately", flush=True)
            return finish(root, registration, references, qualifications, assigned, records, errors)
        except BaseException as error:
            interrupted = isinstance(error, KeyboardInterrupt)
            if not interrupted:
                append_jsonl(root/"fatal_errors.jsonl", {"error": str(error), "traceback": traceback.format_exc(), "created_at": utc_now()})
            atomic_write_json(root/"status.json", {"status": "interrupted" if interrupted else "failed",
                "error": str(error), "updated_at": utc_now()})
            raise


def validate_resume_args(args, registration):
    if (str(args.timing_run_dir.resolve()) != registration["timing_run_dir"] or str(args.model_path) != registration["model_path"]
            or args.port != registration["port"] or args.host != registration["host"]
            or (str(args.reference_run_dir.resolve()) if args.reference_run_dir else None) != registration["reference_run_dir"]):
        raise ValueError("Pilot resume source/model/address differs")
    gpu = {"simulator": os.environ.get("CUDA_VISIBLE_DEVICES"), "server": os.environ.get("XR1_REFS_SERVER_GPU"),
           "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID")}
    if gpu != registration["gpu_assignment"]:
        raise ValueError("Pilot GPU assignment changed on resume")


def review(args):
    """CPU-only, immutable pre-treatment camera assessment and paired analysis."""
    from robocasa.recovery.exact_replay.analyze_intervention_timing import branch_outcome, parent_results, macro_summary, bootstrap, screen
    root = args.output_dir.resolve()
    with (root/".pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = validate_registration(root)
        references = parent_records(root, "references", registration)
        qualifications = parent_records(root, "qualification", registration)
        assigned = assignment(registration, qualifications)
        if json.loads((root/"assignment.json").read_text()) != assigned:
            raise ValueError("Pilot assignment changed")
        records, errors = checked_branches(root, registration), branch_errors(root, registration)
        report = json.loads((root/"report.json").read_text())
        assessment = json.loads(args.review_file.read_text())
        if (not report["timing_collection_complete"] or assessment.get("protocol") != PROTOCOL
                or assessment.get("registration_sha256") != registration["registration_sha256"]
                or assessment.get("assignment_sha256") != assigned["assignment_sha256"]
                or assessment.get("branch_ledger_sha256") != report["branch_ledger_sha256"]
                or (sha256_file(root/"branch_records.jsonl") if (root/"branch_records.jsonl").exists() else None) != report["branch_ledger_sha256"]):
            raise ValueError("Image assessment must bind the complete unchanged pilot")
        rows = assessment.get("parents", [])
        if (len(rows) != len(assigned["eligible_parents"]) or {r["parent_id"] for r in rows} != set(assigned["eligible_parents"])
                or any(type(r.get("images_acceptable")) is not bool or r.get("reviewed_all_frames") is not True for r in rows)):
            raise ValueError("Explicit image assessment and all-frame review required for each eligible parent")
        if (root/"image_review.json").exists() and json.loads((root/"image_review.json").read_text()) != assessment:
            raise FileExistsError("Different image assessment already recorded")
        atomic_write_json(root/"image_review.json", assessment)
        engineering = engineer(root, registration, assigned, records, errors, references)
        images = {r["parent_id"] for r in rows if r["images_acceptable"]}
        valid = {pid for pid in images if engineering["parents"][pid]["engineering_all_pass"]}
        outcomes = [branch_outcome(root, registration, row) for row in records.values() if row["parent_id"] in valid and row["role"] == "primary"]
        parents = parent_results(outcomes)
        counts = Counter(r["task_name"] for r in parents)
        coverage = all(counts[t] >= registration["eligibility"]["minimum_parents_per_task_for_two_task_analysis"] for t in registration["spec"]["horizons"])
        ready = coverage and engineering["engineering_all_pass"] and not errors
        science = {"protocol": PROTOCOL, "registration_sha256": registration["registration_sha256"],
            "parents": len(parents), "parents_by_task": dict(counts), "primary_branches": len(outcomes),
            "image_excluded_parents": sorted(set(assigned["eligible_parents"])-images),
            "timing_invalid_parents": sorted(set(assigned["eligible_parents"])-{pid for pid, p in engineering["parents"].items() if p["engineering_all_pass"]}),
            "coverage_by_task": report["coverage_by_task"], "timing_errors_by_operator": report["timing_errors_by_operator"],
            "pilot_scientific_analysis_ready": ready, "safe_triggered_recovery_tested": False,
            "evidence_type": "exploratory_selected_seeds_conditioned_on_pre_treatment_engineering_and_nominal_image_eligibility",
            "assignment_analysis": "all_eligible_assignments_including_completed_and_ineligible_noop_fallbacks",
            "decision": "insufficient_valid_task_coverage" if not coverage else "engineering_errors_require_attention",
            "inference_limit": "late_errors_are_reported_not_silently_excluded; no_population_or_formal_benchmark_claim"}
        if parents:
            summary = macro_summary(parents)
            science.update(summary)
            if ready:
                science.update(screen(parents, summary, registration["operator_spec"]["gates"]),
                               bootstrap=bootstrap(parents, replicates=2000, seed=0))
        write_jsonl(root/"timing_branch_results.jsonl", outcomes)
        write_jsonl(root/"timing_parent_results.jsonl", parents)
        atomic_write_json(root/"scientific_analysis.json", science)
        report.update(status="complete_with_exclusions" if ready and (report["excluded_parents"] or len(images) < len(rows)) else
            "complete" if ready else "complete_limited_coverage" if not coverage else "complete_with_timing_errors",
            pilot_scientific_analysis_ready=ready, image_health="reviewed_with_exclusions" if len(images) < len(rows) else "accepted_human_review",
            image_review_sha256=sha256_file(root/"image_review.json"), updated_at=utc_now())
        report.update(analysis_parents_by_task={task: counts[task] for task in registration["spec"]["horizons"]},
            image_exclusions_by_task=dict(Counter(references[pid]["parent"]["task_name"] for pid in set(assigned["eligible_parents"])-images)))
        atomic_write_json(root/"report.json", report)
        atomic_write_json(root/"status.json", report)
        return science


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-run-dir", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10332)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--review-file", type=Path)
    parser.set_defaults(restore_atol=0, restore_rtol=0)
    return parser


def main():
    args = build_parser().parse_args()
    result = review(args) if args.review_file else run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
