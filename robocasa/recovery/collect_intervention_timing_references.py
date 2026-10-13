"""Collect new common-entry references; never reconstruct a historical endpoint."""

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import html
import json
import os
from pathlib import Path
import socket
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np

from robocasa.recovery.diagnose_intervention_timing_replay import (
    GCTrace, diagnostic_state, initialize_rng_capture_runtime, record_startup_event,
    renderer_cleanup_source, save_payload,
)
from robocasa.recovery.diagnose_intervention_timing_observations import load_state
from robocasa.recovery.full_snapshot import (
    capture_full_snapshot, load_full_snapshot, restore_full_snapshot,
    save_full_snapshot, stable_digest, compare_structures, _environment_nodes,
)
from robocasa.recovery.inspect_intervention_timing_source import (
    request_trace_comparison, write_image_panel,
)
from robocasa.recovery.intervention_timing import ordered_trace, target_value, validate_registration
from robocasa.recovery.phase3b_training_horizon import (
    atomic_write_json, load_registered_phase2_snapshot, read_jsonl, sha256_file, sha256_json,
)
from robocasa.recovery.run_intervention_timing import reference_audit
from robocasa.recovery.run_operator_screen import _policy_args
from robocasa.recovery.run_phase2_snapshot_replay import (
    _runtime, _call_with_deterministic_environment_seed, _configure_phase2_environment,
    _close_branch_environment, _reset_env, append_jsonl, utc_now,
)
from robocasa.recovery.run_phase3b_training_horizon_continuation import (
    _runtime_args, _validate_live_provenance,
)
PROTOCOL = "xr1_fresh_common_entry_reference_cohort_v2"
GC_SCOPE = "first_post_setup_subtask_evaluation_through_entry_capture"
CAMERAS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")


def code_hashes(timing):
    root = Path(__file__).resolve().parents[2]
    names = (
        "robocasa/recovery/collect_intervention_timing_references.py",
        "robocasa/recovery/diagnose_intervention_timing_replay.py",
        "robocasa/recovery/diagnose_intervention_timing_observations.py",
        "robocasa/recovery/inspect_intervention_timing_source.py",
        "scripts/collect_xr1_timing_references.sh",
    )
    return {**timing["code_hashes"], **{name: sha256_file(root/name) for name in names}}


def checked_path(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact escaped the reference root")
    return path


def inputs(args):
    source = args.timing_run_dir.resolve()
    timing = json.loads((source/"registration.json").read_text())
    validate_registration(timing)
    original = timing["parents"]
    if len(original) != 10 or len({p["parent_id"] for p in original}) != 10 or Counter(
            p["task_name"] for p in original) != {"ArrangeTea": 5, "CuttingToolSelection": 5}:
        raise ValueError("Require the original ten unique selected seeds, five per task")
    phase2 = Path(timing["phase2_run_dir"])
    plan = json.loads((phase2/"plan.json").read_text())
    budget = plan.get("parent_horizon")
    if type(budget) is not int or budget < 1:
        raise ValueError("Require a positive frozen Phase 2 parent_horizon search budget")
    _validate_live_provenance(args, timing["phase2_registration"], plan)
    parents = []
    for old in original:
        sid = old["snapshot_id"]
        # Registered legacy files supply configuration/identity only. Never
        # restore their observations, image queues, actions, or endpoint RNGs.
        ref, integrity = load_registered_phase2_snapshot(
            phase2/"snapshots"/f"{sid}.pkl.gz",
            expected_file_sha256=timing["phase2_registration"]["phase2_snapshot_file_sha256"][sid],
            expected_snapshot_id=sid, expected_parent_id=old["parent_id"], expected_task_name=old["task_name"])
        seed = int(ref.metadata["environment_seed"])
        if old.get("environment_seed", seed) != seed:
            raise ValueError("Source parent and snapshot environment seeds disagree")
        stage = timing["phase2_registration"]["trigger_stages"][old["task_name"]]
        if ref.metadata["trigger_stage"] != stage:
            raise ValueError("Source target-stage identity differs")
        policy_args = _policy_args(ref, args)
        if type(policy_args.get("sampling_seed_base")) is not int:
            raise ValueError("Fresh nominal requests require a frozen integer sampling seed base")
        identity = {"protocol": PROTOCOL, "task_name": old["task_name"],
                    "environment_seed": seed, "environment_reset_index": old["environment_reset_index"],
                    "legacy_parent_id": old["parent_id"]}
        parents.append({**identity, "parent_id": sha256_json(identity)[:24],
                        "index": old["index"], "sentinel": old["sentinel"],
                        "trigger_stage": stage, "policy_args": policy_args,
                        "legacy_snapshot_id": sid, "legacy_file_integrity": integrity})
    config = {
        "protocol": PROTOCOL, "timing_run_dir": str(source),
        "timing_registration_sha256": timing["registration_sha256"],
        "timing_registration_file_sha256": sha256_file(source/"registration.json"),
        "phase2_plan_sha256": sha256_file(phase2/"plan.json"),
        "parents": parents, "expected_parents": 10, "repeats_per_parent": 2,
        "nominal_search_budget_environment_steps": budget,
        "entry_definition": "first_inference_boundary_in_target_stage_while_target_incomplete_before_termination",
        "repeat_initialization": "restore_new_actual_reset_snapshot_in_fresh_environment_and_policy",
        "restore_observation_verification": "live_getter_resolved_through_environment_wrappers; no_snapshot_fallback",
        "sampling": "original_ordinary_K1_seed_bases; no_historical_mutable_policy_state",
        "gc_mode": "disabled", "gc_nominal_scope": GC_SCOPE, "rng_runtime_init": "before_env",
        "restore_atol": 0, "restore_rtol": 0,
        "model_path": str(args.model_path), "host": args.host, "port": args.port,
        "gpu_assignment": {"simulator": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "server": os.environ.get("XR1_REFS_SERVER_GPU"),
                           "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID")},
        "code_hashes": code_hashes(timing),
        "image_health_gate": "human_review_of_all_raw_RGB_camera_panels_separate_from_repeatability",
        "selection_scope": "same_previously_selected_ten_seeds; new_trajectories_and_parent_identities",
        "intervention_trajectories_collected": 0,
    }
    return timing, plan, config


class RequestRecorder:
    def __init__(self, client, folder):
        self.client, self.folder, self.calls = client, folder, 0

    def infer(self, request):
        frozen = deepcopy(request)
        if type(frozen.get("sampling_seed")) is not int:
            raise ValueError("Nominal request has no explicit integer seed")
        response = self.client.infer(request)
        name = f"requests/request_{self.calls:06d}.pkl.gz"
        save_payload(self.folder/name, {"request": frozen, "response": response}, compresslevel=1)
        append_jsonl(self.folder/"requests.jsonl", {
            "request_index": self.calls, "sampling_seed": frozen["sampling_seed"],
            "request_sha256": stable_digest(frozen), "response_sha256": stable_digest(response),
            "payload_path": name,
        })
        self.calls += 1
        return response

    def close(self):
        self.client.close()


def camera_format(observation):
    if not isinstance(observation, dict):
        raise ValueError("Require a live observation dictionary, not an unavailable or cached fallback observation")
    for key in CAMERAS:
        frame = np.asarray(observation.get(key))
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"Expected raw uint8 RGB camera observation: {key}")


def live_observation(env):
    """Find the real Gym adapter getter beneath non-forwarding wrappers."""
    for path, node in _environment_nodes(env):
        getter = getattr(node, "get_current_observation", None)
        if callable(getter):
            observation = getter()
            camera_format(observation)
            return observation, path+".get_current_observation"
    raise ValueError("No live observation getter found through the environment wrappers; snapshot fallback is forbidden")


def restore_initial(initial, env, policy, folder):
    # The legacy generic observation helper can fall back to the saved image
    # on Gym wrappers. Keep frozen helpers unchanged and independently require
    # a real observation after all simulator/controller/RNG restoration.
    audit = restore_full_snapshot(initial, env, policy, atol=0, rtol=0, verify_observation=False)
    audit["observation_verification"] = {"performed": False, "snapshot_fallback_used": False}
    audit["observation_exact"] = False  # no claim of equality before live verification
    if not audit["valid"]:
        atomic_write_json(folder/"initial_restore_audit.json", audit)
        raise ValueError("New reset-state restore failed; inspect initial_restore_audit.json")
    try:
        observation, getter = live_observation(env)
    except Exception as exc:
        audit.update(valid=False, observation_exact=False)
        audit["observation_verification"]["error"] = str(exc)
        atomic_write_json(folder/"initial_restore_audit.json", audit)
        raise
    differences = compare_structures(initial.observation, observation, atol=0, rtol=0, path="observation")
    audit.update(observation_exact=not differences, observation_mismatches=differences,
                 valid=audit["valid"] and not differences)
    audit["observation_verification"].update(performed=True, getter=getter)
    atomic_write_json(folder/"initial_restore_audit.json", audit)
    if not audit["valid"]:
        raise ValueError("New reset-state live observation differs; inspect initial_restore_audit.json")
    return observation, audit


def metadata(parent, prefix=()):
    return {k: parent[k] for k in (
        "environment_seed", "environment_reset_index", "trigger_stage",
        "legacy_parent_id", "legacy_snapshot_id") } | {
        "reference_protocol": PROTOCOL, "historical_endpoint_reconstructed": False,
        "local_inferences": 0, "ordered_prefix": list(prefix),
        "entry_definition": "first_eligible_target_stage_boundary_in_new_nominal_trajectory",
    }


def collect_attempt(parent, folder, registration, plan, args, runtime, factory, initial=None):
    folder.mkdir()
    for name in ("states", "transitions", "requests"):
        (folder/name).mkdir()
    env = policy = None
    guard = GCTrace("disabled")
    guard.start()
    step, evaluations, entry, restore = 0, [], None, None
    outcome, terminated = None, False
    startup = folder/"startup_events.jsonl"
    atomic_write_json(folder/"status.json", {"status": "running", "environment_step": 0})
    try:
        record_startup_event(startup, "before_environment_creation")
        seed = parent["environment_seed"]
        config = _runtime_args(args, plan)
        env = _call_with_deterministic_environment_seed(seed, runtime["make_env"],
            parent["task_name"], config.env_interface, config.split, seed, True)
        _configure_phase2_environment(env, config)
        guard.phase, guard.nominal = "nominal_setup", False
        policy = runtime["call_factory"](factory, env, deepcopy(parent["policy_args"]))
        obs, _ = _call_with_deterministic_environment_seed(seed, _reset_env, env, seed)
        policy.reset()
        if initial is not None:
            obs, restore = restore_initial(initial, env, policy, folder)
        else:
            state = policy.get_state()
            if (not policy.at_inference_boundary or state["environment_step"] != 0
                    or state["ordinary_inference_index"] != 0 or state["state_queue"]
                    or any(state["image_queues"].values())):
                raise ValueError("Fresh nominal policy contains mutable history")
        recorder = RequestRecorder(policy.client, folder)
        policy.client = recorder
        guard.phase, guard.nominal = "evaluation", True
        evaluations.append(deepcopy(runtime["get_subtask_eval"](env)))
        required = evaluations[0]["required_predicates"]
        predicate = parent["trigger_stage"].split("::", 1)[-1]
        if predicate not in required:
            raise ValueError("Frozen target stage is absent from the live required stages")
        if initial is not None and stable_digest(evaluations[0]) != stable_digest(initial.subtask_eval):
            raise ValueError("New reset-state semantic restore differs")
        reset = capture_full_snapshot(env, policy, parent_id=parent["parent_id"],
            task_name=parent["task_name"], trigger_name="fresh_reference_reset", environment_step=0,
            observation=obs, subtask_eval=evaluations[0], metadata=metadata(parent))
        save_full_snapshot(reset, folder/"initial.pkl.gz")
        initial_audit = None
        if initial is not None:
            initial_audit = reference_audit(initial, reset)
            initial_audit["reference_limit"] = "new_actual_reset_state_after_live_observation_and_semantic_evaluation"
            atomic_write_json(folder/"initial_state_audit.json", initial_audit)
            if not initial_audit["all_exact"]:
                raise ValueError("New reset-state complete audit differs after live observation/evaluation")
        while True:
            guard.phase, guard.step = "state_capture", step
            camera_format(obs)
            current = diagnostic_state(env, policy, obs, evaluations[-1], step)
            save_payload(folder/f"states/state_{step:06d}.pkl.gz", current, compresslevel=1)
            stage = ordered_trace(evaluations)[-1]
            at_entry = (stage["ordered_current_subtask"] == predicate and policy.at_inference_boundary
                        and not target_value(evaluations[-1], predicate) and not terminated)
            append_jsonl(folder/"states.jsonl", {"environment_step": step,
                "state_sha256": stable_digest(current), "observation_sha256": stable_digest(obs),
                "ordered_current_subtask": stage["ordered_current_subtask"],
                "at_inference_boundary": bool(policy.at_inference_boundary),
                "target_complete": target_value(evaluations[-1], predicate)})
            if at_entry:
                entry = capture_full_snapshot(env, policy, parent_id=parent["parent_id"],
                    task_name=parent["task_name"], trigger_name="fresh_timing_common_entry",
                    environment_step=step, observation=obs, subtask_eval=evaluations[-1],
                    metadata=metadata(parent, stage["ordered_completed_subtasks"]))
                save_full_snapshot(entry, folder/"entry.pkl.gz")
                outcome = "common_entry_captured"
                break
            if terminated or step == registration["nominal_search_budget_environment_steps"]:
                outcome = "terminated_before_entry" if terminated else "search_budget_exhausted"
                break
            guard.phase = "policy_call"
            calls_before = recorder.calls
            action = runtime["call_policy"](policy, obs)
            requests, inference = policy.pop_request_records(), policy.pop_inference_record()
            actual = read_jsonl(folder/"requests.jsonl")[calls_before:]
            if (len(actual) != len(requests) or len(requests) > 1
                    or bool(requests) != (inference is not None)):
                raise ValueError("Nominal K=1 request/inference alignment failed")
            if requests:
                expected_seed = parent["policy_args"]["sampling_seed_base"] + calls_before
                if any(r.get("request_sha256") != actual[0]["request_sha256"]
                       or r.get("sampling_seed") != expected_seed for r in (*requests, inference)):
                    raise ValueError("Nominal encoded request or seed schedule differs")
            guard.phase = "environment_step"
            obs, reward, done, info = runtime["step_fn"](env, action)
            save_payload(folder/f"transitions/transition_{step:06d}.pkl.gz", {
                "step": step, "action": action, "requests": requests, "inference": inference,
                "reward": reward, "done": done, "info": info}, compresslevel=1)
            append_jsonl(folder/"transitions.jsonl", {"step": step, "action_sha256": stable_digest(action),
                "requests_sha256": stable_digest(requests), "inference_sha256": stable_digest(inference),
                "reward_sha256": stable_digest(reward), "done": bool(done),
                "info_sha256": stable_digest(info)})
            step += 1
            guard.phase, guard.step = "evaluation", step
            evaluation = deepcopy(runtime["get_subtask_eval"](env))
            if evaluation["required_predicates"] != required:
                raise ValueError("Required stage sequence changed during nominal collection")
            evaluations.append(evaluation)
            terminated = bool(done or runtime["success_fn"](info=info, reward=reward, env=env))
            if step % 16 == 0:
                atomic_write_json(folder/"status.json", {"status": "running", "environment_step": step,
                    "ordered_current_subtask": ordered_trace(evaluations)[-1]["ordered_current_subtask"]})
                print(f"  {parent['parent_id']} {folder.name}: step {step}, requests {recorder.calls}", flush=True)
        result = {"outcome": outcome, "environment_steps": step, "nominal_requests": recorder.calls,
                  "entry_snapshot_id": entry.snapshot_id if entry else None,
                  "initial_snapshot_id": reset.snapshot_id, "initial_restore_valid": restore["valid"] if restore else None,
                  "initial_state_audit_all_exact": initial_audit["all_exact"] if initial_audit else None,
                  "execution_environment": {"hostname": socket.gethostname(), "simulator_pid": os.getpid(),
                      "policy_server_pid": os.environ.get("XR1_REFS_SERVER_PID"),
                      "gpu_assignment": registration["gpu_assignment"]}}
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
                trace["nominal_scope"] = GC_SCOPE
                atomic_write_json(folder/"gc_trace.json", trace)
    result["gc_guard"] = trace
    atomic_write_json(folder/"report.json", result)
    atomic_write_json(folder/"status.json", {"status": outcome, "environment_step": step})
    return result


def compare_attempts(folder):
    traces = [read_jsonl(folder/f"attempt_{i}/states.jsonl") for i in range(2)]
    support = [[r["environment_step"] for r in t] for t in traces]
    exact_support = support[0] == support[1] == list(range(len(traces[0])))
    audits, first = [], None
    for step in sorted(set(support[0]) & set(support[1])):
        states = [load_state(folder/f"attempt_{i}/states/state_{step:06d}.pkl.gz") for i in range(2)]
        if any(stable_digest(s) != traces[i][step]["state_sha256"] for i, s in enumerate(states)):
            raise ValueError("Stored nominal state differs from its trace")
        audit = reference_audit(*[SimpleNamespace(**s) for s in states])
        audit["reference_limit"] = "complete_new_nominal_states_saved_at_every_environment_step"
        audits.append({"environment_step": step, "channels": audit["channels"],
                       "all_exact": audit["all_exact"], "model_xml_audit": audit["model_xml_audit"]})
        if not audit["all_exact"] and first is None:
            first = {"environment_step": step, "audit": audit}
    transitions = [read_jsonl(folder/f"attempt_{i}/transitions.jsonl") for i in range(2)]
    request_comparison = request_trace_comparison(*[
        read_jsonl(folder/f"attempt_{i}/requests.jsonl") for i in range(2)])
    result = {"state_support_exact": exact_support, "state_counts": list(map(len, traces)),
              "all_state_channels_exact": exact_support and first is None,
              "state_audits": audits, "first_state_difference": first,
              "transition_traces_exact": transitions[0] == transitions[1],
              "request_comparison": request_comparison}
    atomic_write_json(folder/"repeat_audit.json", result)
    return result


def export_gallery(folder, parent, counts):
    gallery = folder/"camera_panels"
    gallery.mkdir()
    for step in range(max(counts)):
        rows = []
        for attempt in range(2):
            if step < counts[attempt]:
                observation = load_state(folder/f"attempt_{attempt}/states/state_{step:06d}.pkl.gz")["observation"]
                rows.append([(f"attempt {attempt}, step {step}\n{key}", observation[key]) for key in CAMERAS])
        write_image_panel(gallery/f"step_{step:06d}.png", rows)
    title = html.escape(f"{parent['task_name']} seed {parent['environment_seed']} / {parent['parent_id']}")
    (folder/"camera_review.html").write_text(f'''<!doctype html><meta charset="utf-8">
<title>Reference cameras</title><style>body{{font:16px sans-serif;margin:20px}}img{{max-width:100%}}</style>
<h1>{title}</h1><p>Both fresh attempts, raw RGB pixels. Inspect every frame from reset through entry.
Repeat agreement alone does not establish image health. Left/right arrows advance one frame.</p>
<input id="step" type="range" min="0" max="{max(counts)-1}" value="0" style="width:70%">
<output id="label"></output><br><img id="panel" alt="Both attempts, three cameras">
<script>const slider=document.getElementById('step');
function show(){{document.getElementById('label').textContent='environment step '+slider.value;
document.getElementById('panel').src='camera_panels/step_'+String(slider.value).padStart(6,'0')+'.png';}}
slider.oninput=show;document.onkeydown=e=>{{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){{
e.preventDefault();slider.value=Number(slider.value)+(e.key==='ArrowRight'?1:-1);show();}}}};show();</script>''')


def gc_pass(trace):
    return (trace["mode"] == "disabled" and trace["nominal_scope"] == GC_SCOPE
            and not trace["enabled_at_environment_creation"] and not trace["enabled_at_end_of_cleanup"]
            and trace["original_enabled_state_restored"] and trace["nominal_collection_stops"] == 0
            and all(not e["automatic_gc_enabled"] for e in trace["events"] if e["event"] == "stop"))


def collect_parent(parent, folder, registration, plan, args, runtime, factory):
    first = collect_attempt(parent, folder/"attempt_0", registration, plan, args, runtime, factory)
    initial = load_full_snapshot(folder/"attempt_0/initial.pkl.gz")
    second = collect_attempt(parent, folder/"attempt_1", registration, plan, args, runtime, factory, initial)
    comparison = compare_attempts(folder)
    export_gallery(folder, parent, comparison["state_counts"])
    startup = [read_jsonl(folder/f"attempt_{i}/startup_events.jsonl")[0] for i in range(2)]
    early = registration["early_rng_runtime_initialization"]
    gates = {
        "both_common_entries_captured": all(r["outcome"] == "common_entry_captured" for r in (first, second)),
        "new_initial_restore_exact": second["initial_restore_valid"] is True,
        "new_initial_full_audit_exact": second["initial_state_audit_all_exact"] is True,
        "every_complete_state_repeats_exact": comparison["all_state_channels_exact"],
        "consumed_actions_and_inferences_repeat_exact": comparison["transition_traces_exact"],
        "encoded_requests_and_responses_repeat_exact": comparison["request_comparison"]["first_difference"] is None,
        "gc_guard_passed_through_cleanup": all(gc_pass(r["gc_guard"]) for r in (first, second)),
        "cuda_initialized_before_both_environments": not early["torch_cuda_available"] or all(e["cuda_initialized"] for e in startup),
    }
    result = {"protocol": PROTOCOL, "parent": parent, "engineering_gates": gates,
              "engineering_all_pass": all(gates.values()), "attempts": [first, second],
              "camera_review": "camera_review.html", "image_health": "pending_human_review",
              "intervention_trajectories_collected": 0, "historical_endpoint_reconstructed": False}
    atomic_write_json(folder/"report.json", result)
    return result


def validate_cohort_registration(root):
    reg = json.loads((root/"registration.json").read_text())
    unsigned = dict(reg)
    digest = unsigned.pop("registration_sha256", None)
    if reg.get("protocol") != PROTOCOL or sha256_json(unsigned) != digest:
        raise ValueError("Fresh-reference registration checksum/protocol mismatch")
    timing_root = Path(reg["timing_run_dir"])
    if sha256_file(timing_root/"registration.json") != reg["timing_registration_file_sha256"]:
        raise ValueError("Frozen seed-source registration changed")
    timing = json.loads((timing_root/"registration.json").read_text())
    validate_registration(timing)
    if code_hashes(timing) != reg["code_hashes"]:
        raise ValueError("Fresh-reference collection code changed after registration")
    if sha256_file(Path(timing["phase2_run_dir"])/"plan.json") != reg["phase2_plan_sha256"]:
        raise ValueError("Frozen nominal search plan changed")
    return reg


def completed_parents(root, registration):
    rows = read_jsonl(root/"reference_records.jsonl")
    known = {r["parent_id"] for r in rows}
    for parent in registration["parents"]:
        marker = root/"parents"/parent["parent_id"]/"committed.json"
        if parent["parent_id"] not in known and marker.exists():
            row = json.loads(marker.read_text())
            append_jsonl(root/"reference_records.jsonl", row)
            rows.append(row)
    result = {}
    parents = {p["parent_id"]: p for p in registration["parents"]}
    for row in rows:
        pid = row["parent_id"]
        if (pid not in parents or pid in result or row["parent"] != parents[pid]
                or row["registration_sha256"] != registration["registration_sha256"]
                or not row["engineering_all_pass"]):
            raise ValueError("Unknown, duplicate, or invalid completed reference")
        marker = checked_path(root, f"parents/{pid}/committed.json")
        if json.loads(marker.read_text()) != row:
            raise ValueError("Reference marker and ledger differ")
        for name, expected in row["artifact_hashes"].items():
            if sha256_file(checked_path(root, name)) != expected:
                raise ValueError("Completed reference artifact changed")
        entry = load_full_snapshot(checked_path(root, row["snapshot_path"]))
        if entry.snapshot_id != row["snapshot_id"] or entry.parent_id != pid or entry.task_name != parents[pid]["task_name"]:
            raise ValueError("Completed reference snapshot identity differs")
        result[pid] = row
    return result


def run(args, runtime=None):
    timing, plan, config = inputs(args)
    root = args.output_dir.resolve()
    protected = [Path(name).resolve() for name in (config["timing_run_dir"], timing["phase2_run_dir"], timing["operator_run_dir"])]
    if root.is_relative_to(Path("/gs/fs")) or any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError("Reference output must be separate from all frozen inputs and outside /gs/fs")
    registration = None
    if args.resume:
        registration = validate_cohort_registration(root)
        if {k: registration.get(k) for k in config} != config:
            raise ValueError("Fresh-reference source or configuration changed after registration")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return config
    root.mkdir(parents=True, exist_ok=True)
    with (root/".references.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if registration is None:
            if any(p.name != ".references.lock" for p in root.iterdir()):
                raise FileExistsError("Reference root is not empty; use resume or a new output directory")
            # Warm CUDA RNG inspection before any renderer exists; never use
            # the historical endpoint's RNG streams for a new nominal rollout.
            early = initialize_rng_capture_runtime()
            registration = {**config, "created_at": utc_now(), "early_rng_runtime_initialization": early}
            registration["registration_sha256"] = sha256_json(registration)
            atomic_write_json(root/"registration.json", registration)
        else:
            early = initialize_rng_capture_runtime()
            if early != registration["early_rng_runtime_initialization"]:
                raise ValueError("RNG runtime capabilities changed on resume")
        if read_jsonl(root/"errors.jsonl"):
            raise ValueError("Reference errors recorded; preserve this root and diagnose before a new run")
        completed = completed_parents(root, registration)
        current = None
        try:
            runtime = runtime or _runtime()
            factory = runtime["load_factory"](plan["policy_module"])
            atomic_write_json(root/"renderer_cleanup_source.json", renderer_cleanup_source())
            for parent in registration["parents"]:
                pid = parent["parent_id"]
                if pid in completed:
                    continue
                current = parent
                atomic_write_json(root/"status.json", {"status": "collecting_references", "parents": len(completed),
                    "expected_parents": 10, "current_parent": pid, "legacy_parent_id": parent["legacy_parent_id"],
                    "hostname": socket.gethostname(), "updated_at": utc_now()})
                folder = root/"parents"/pid
                folder.mkdir(parents=True, exist_ok=True)
                attempt = Path(tempfile.mkdtemp(prefix="collection_", dir=folder))
                report = collect_parent(parent, attempt, registration, plan, args, runtime, factory)
                if not report["engineering_all_pass"]:
                    atomic_write_json(folder/"failed_parent.json", {"report_path": str((attempt/"report.json").relative_to(root)),
                        "engineering_gates": report["engineering_gates"]})
                    raise ValueError(f"Fresh-reference repeat gate failed: {pid}; inspect {attempt}")
                snapshot = load_full_snapshot(attempt/"attempt_0/entry.pkl.gz")
                row = {"parent_id": pid, "parent": parent, "engineering_all_pass": True,
                       "registration_sha256": registration["registration_sha256"],
                       "snapshot_path": str((attempt/"attempt_0/entry.pkl.gz").relative_to(root)),
                       "snapshot_id": snapshot.snapshot_id, "entry_environment_step": snapshot.environment_step,
                       "report_path": str((attempt/"report.json").relative_to(root)),
                       "camera_review_path": str((attempt/"camera_review.html").relative_to(root)),
                       "artifact_hashes": {str(p.relative_to(root)): sha256_file(p) for p in sorted(attempt.rglob("*")) if p.is_file()}}
                atomic_write_json(folder/"committed.json", row)
                append_jsonl(root/"reference_records.jsonl", row)
                completed[pid] = row
                print(f"references [{len(completed)}/10]: {pid}, entry step {snapshot.environment_step}, repeats passed", flush=True)
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            if not interrupted:
                append_jsonl(root/"errors.jsonl", {"parent": current, "error": str(exc), "traceback": traceback.format_exc()})
            atomic_write_json(root/"status.json", {"status": "interrupted" if interrupted else "failed",
                "parents": len(completed), "expected_parents": 10, "failed_parent": current,
                "error": str(exc), "updated_at": utc_now()})
            raise
        result = {"protocol": PROTOCOL, "status": "awaiting_image_review", "parents": len(completed),
                  "expected_parents": 10, "nominal_replays": 2*len(completed),
                  "nominal_environment_steps": 2*sum(r["entry_environment_step"] for r in completed.values()),
                  "parents_by_task": dict(Counter(r["parent"]["task_name"] for r in completed.values())),
                  "engineering_all_pass": len(completed) == 10, "image_health": "pending_human_review",
                  "timing_ready": False, "full_timing_study_complete": False,
                  "intervention_trajectories_collected": 0, "registration_sha256": registration["registration_sha256"],
                  "reference_ledger_sha256": sha256_file(root/"reference_records.jsonl"), "updated_at": utc_now()}
        if not (root/"report.json").exists():
            template = {"protocol": PROTOCOL, "registration_sha256": registration["registration_sha256"],
                        "reference_ledger_sha256": result["reference_ledger_sha256"],
                        "parents": [{"parent_id": pid, "images_acceptable": None, "reviewed_all_frames": False,
                                     "notes": ""} for pid in completed]}
            atomic_write_json(root/"image_review_template.json", template)
            links = "".join(f'<li>{html.escape(r["parent"]["task_name"])} seed {r["parent"]["environment_seed"]}: '
                            f'<a href="{r["camera_review_path"]}">{pid}</a> (entry step {r["entry_environment_step"]})</li>'
                            for pid, r in completed.items())
            (root/"camera_review.html").write_text('<!doctype html><meta charset="utf-8"><title>Reference cohort review</title>'
                '<h1>Fresh reference cohort: camera review pending</h1><p>Inspect every slider frame for each parent. '
                'Engineering repeatability does not certify camera quality.</p><ul>'+links+'</ul>')
            # Publish completion only after the template and gallery exist.
            # An interrupted finalization is rebuilt from committed parents.
            atomic_write_json(root/"report.json", result)
        saved = json.loads((root/"report.json").read_text())
        if (saved["registration_sha256"] != registration["registration_sha256"]
                or saved["reference_ledger_sha256"] != sha256_file(root/"reference_records.jsonl")):
            raise ValueError("Saved cohort report differs from its registration or reference ledger")
        atomic_write_json(root/"status.json", saved)
        print(f"Fresh references: {root}/report.json; inspect {root}/camera_review.html", flush=True)
        return saved


def review(args):
    """Record an explicit image-quality assessment, bound to verified artifacts."""
    root = args.output_dir.resolve()
    with (root/".references.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = validate_cohort_registration(root)
        completed = completed_parents(root, registration)
        report = json.loads((root/"report.json").read_text())
        assessment = json.loads(args.review_file.read_text())
        if (len(completed) != 10 or read_jsonl(root/"errors.jsonl") or not report["engineering_all_pass"]
                or assessment.get("protocol") != PROTOCOL
                or assessment.get("registration_sha256") != registration["registration_sha256"]
                or assessment.get("reference_ledger_sha256") != sha256_file(root/"reference_records.jsonl")):
            raise ValueError("Image review must bind to the complete, unchanged, passing reference cohort")
        rows = assessment.get("parents", [])
        if (len(rows) != 10 or {r["parent_id"] for r in rows} != set(completed)
                or any(type(r.get("images_acceptable")) is not bool or r.get("reviewed_all_frames") is not True for r in rows)):
            raise ValueError("Explicit image assessment and all-frame review required for each of the ten parents")
        if (root/"image_review.json").exists():
            if json.loads((root/"image_review.json").read_text()) != assessment:
                raise FileExistsError("Different image assessment already recorded; preserve it")
        else:
            atomic_write_json(root/"image_review.json", assessment)
        accepted = all(r["images_acceptable"] for r in rows)
        digest = sha256_file(root/"image_review.json")
        if report.get("image_review_sha256") == digest and report.get("timing_ready") is accepted:
            atomic_write_json(root/"status.json", report)
            return report
        report.update(status="references_ready" if accepted else "image_review_rejected",
                      image_health="accepted_human_review" if accepted else "rejected_human_review",
                      timing_ready=accepted, image_review_sha256=digest, updated_at=utc_now())
        atomic_write_json(root/"report.json", report)
        atomic_write_json(root/"status.json", report)
        return report


def validated_references(root):
    """Downstream studies must call this before using the new entry snapshots."""
    root = Path(root).resolve()
    registration = validate_cohort_registration(root)
    rows = completed_parents(root, registration)
    report = json.loads((root/"report.json").read_text())
    if (len(rows) != 10 or read_jsonl(root/"errors.jsonl") or report.get("timing_ready") is not True
            or report.get("status") != "references_ready" or not report.get("engineering_all_pass")
            or report.get("registration_sha256") != registration["registration_sha256"]
            or report.get("reference_ledger_sha256") != sha256_file(root/"reference_records.jsonl")
            or sha256_file(root/"image_review.json") != report.get("image_review_sha256")):
        raise ValueError("Fresh references require complete repeat audits and accepted image review")
    assessment = json.loads((root/"image_review.json").read_text())
    reviewed = assessment.get("parents", [])
    if (assessment.get("protocol") != PROTOCOL
            or assessment.get("registration_sha256") != registration["registration_sha256"]
            or assessment.get("reference_ledger_sha256") != report["reference_ledger_sha256"]
            or len(reviewed) != 10 or {r["parent_id"] for r in reviewed} != set(rows)
            or any(r.get("images_acceptable") is not True or r.get("reviewed_all_frames") is not True for r in reviewed)):
        raise ValueError("Fresh-reference image assessment is incomplete or rejected")
    return registration, rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10327)
    parser.set_defaults(restore_atol=0, restore_rtol=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--review-file", type=Path, help="Record a completed image review; no simulator or policy calls")
    return parser


def main():
    args = build_parser().parse_args()
    report = review(args) if args.review_file else run(args)
    if args.review_file:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
