"""Isolated replay diagnosis; never changes a timing registration or its ledger."""

import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import pickle
import sys
import traceback
from types import SimpleNamespace

from robocasa.recovery.full_snapshot import (
    environment_fingerprint, stable_digest, _capture_global_rng_state, _capture_environment_rng_state,
    _model_xml_roundtrip_audit,
)
from robocasa.recovery.intervention_timing import validate_registration, ordered_trace
from robocasa.recovery.phase3b_training_horizon import (
    atomic_write_json, load_registered_phase2_snapshot, read_jsonl, sha256_file,
)
from robocasa.recovery.run_intervention_timing import capture_entry, reference_audit
from robocasa.recovery.run_operator_screen import _policy_args
from robocasa.recovery.run_phase2_snapshot_replay import (
    _runtime, _call_with_deterministic_environment_seed, _configure_phase2_environment,
    _close_branch_environment, append_jsonl, utc_now,
)
from robocasa.recovery.run_phase3b_training_horizon_continuation import (
    _runtime_args, _validate_live_provenance,
)


PROTOCOL = "xr1_isolated_nominal_replay_diagnostic_v1"
TRACE_FIELDS = (
    "observation_sha256", "next_observation_sha256", "action_sha256",
    "policy_state_sha256", "environment_rng_sha256", "causal_environment_sha256",
    "request_sha256", "request_seeds", "target_stage", "local_inferences",
)


def save_payload(path, value, *, compresslevel=9):
    with gzip.open(path, "wb", compresslevel=compresslevel) as stream:
        pickle.dump(value, stream, protocol=5)


def record_startup_event(path, phase):
    """Observe an already loaded Torch runtime without initializing CUDA."""
    if path is None:
        return
    torch = sys.modules.get("torch")
    append_jsonl(path, {"phase": phase, "torch_loaded": torch is not None,
                       "cuda_initialized": bool(torch.cuda.is_initialized()) if torch is not None else False})


def initialize_rng_capture_runtime():
    """Warm RNG inspection before rendering; keep source restoration unchanged."""
    state = _capture_global_rng_state()
    return {"torch_available": state["torch_available"],
            "torch_cuda_available": state.get("torch_cuda_available", False),
            "cuda_rng_streams": len(state.get("torch_cuda", [])),
            "source_rng_restored_early": False}


def diagnostic_state(env, policy, observation, evaluation, step):
    """Plain diagnostic data, including cached-policy states; not a branch snapshot."""
    before = stable_digest((_capture_global_rng_state(), _capture_environment_rng_state(env)))
    fingerprint = environment_fingerprint(env)
    after = stable_digest((_capture_global_rng_state(), _capture_environment_rng_state(env)))
    if before != after:
        raise ValueError("Diagnostic state inspection advances RNG; cannot treat this as unperturbed replay")
    return {
        "environment_state": fingerprint["simulator_state"]["environment_state"],
        "simulator_integration_state": fingerprint["simulator_state"]["integration_state"],
        "controller_state": fingerprint["controller_state"],
        "environment_control_state": fingerprint["control_state"],
        "environment_rng_state": fingerprint["rng_state"],
        "global_rng_state": _capture_global_rng_state(),
        "policy_state": deepcopy(policy.get_state()), "observation": deepcopy(observation),
        "subtask_eval": deepcopy(evaluation), "tracker_state": None,
        "environment_step": step,
    }


def first_divergence(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        fields = [key for key in ("step", *TRACE_FIELDS) if a.get(key) != b.get(key)]
        if fields:
            return {"row": i, "step": a.get("step"), "fields": fields,
                    "left": {k: a.get(k) for k in fields}, "right": {k: b.get(k) for k in fields}}
    if len(left) != len(right):
        return {"row": min(len(left), len(right)), "fields": ["trace_length"],
                "left_length": len(left), "right_length": len(right)}
    return None


def observation_key_record(observation, step):
    return {"environment_step": step, "after_action_index": step-1 if step else None,
            "observation_sha256": stable_digest(observation),
            "observation_key_sha256": {k: stable_digest(v) for k, v in observation.items()}}


def compare_observation_windows(root):
    """Compare saved policy-loop observations after both environments are closed."""
    # Import only during postprocessing: the recorded-action diagnostic imports
    # this module's protocol, and its pixel helpers need no renderer or policy.
    from robocasa.recovery.diagnose_intervention_timing_observations import (
        observation_difference, camera_statistics, load_state,
    )
    folders = [root/f"attempt_{i}"/"observation_window" for i in range(2)]
    traces = [read_jsonl(p/"trace.jsonl") for p in folders]
    steps = [[r["environment_step"] for r in trace] for trace in traces]
    common = sorted(set(steps[0]) & set(steps[1]))
    by_step = [{r["environment_step"]: r for r in trace} for trace in traces]
    raw_observations = []
    for folder, trace in zip(folders, traces):
        path = folder/"all_observations.pkl.gz"
        values = load_state(path) if path.exists() else None
        if values is not None:
            if set(values) != {r["environment_step"] for r in trace}:
                raise ValueError("Raw observation step support differs from observation trace")
            for row in trace:
                if stable_digest(values[row["environment_step"]]) != row["observation_sha256"]:
                    raise ValueError("Raw observation differs from recorded trace hash")
        raw_observations.append(values)
    first, keys = None, {}
    for step in common:
        a, b = (trace[step] for trace in by_step)
        for key in sorted(set(a["observation_key_sha256"]) | set(b["observation_key_sha256"])):
            if a["observation_key_sha256"].get(key) != b["observation_key_sha256"].get(key):
                keys.setdefault(key, step)
        if first is None and a["observation_sha256"] != b["observation_sha256"]:
            first = step
    audits, state_differences, statistics = [], {}, []
    details = None
    captured = [sorted(p.name for p in folder.glob("state_*.pkl.gz")) for folder in folders]
    for name in sorted(set(captured[0]) & set(captured[1])):
        a, b = (load_state(folder/name) for folder in folders)
        step = a["environment_step"]
        if b["environment_step"] != step:
            raise ValueError("Observation-window state steps differ")
        audit = reference_audit(SimpleNamespace(**a), SimpleNamespace(**b))
        audits.append({"environment_step": step, "channels": audit["channels"],
                       "all_exact": audit["all_exact"]})
        for key, exact in audit["channels"].items():
            if not exact:
                state_differences.setdefault(key, step)
        statistics.append({"environment_step": step,
                           "attempt_0": camera_statistics(a["observation"]),
                           "attempt_1": camera_statistics(b["observation"])})
        if step == first:
            details = {"environment_step": step, "after_action_index": step-1 if step else None,
                       "state_audit": audit,
                       "changed_observation_keys": observation_difference(a["observation"], b["observation"])}
    raw_state_captured = details is not None
    if first is not None and all(values is not None and first in values for values in raw_observations):
        a, b = (values[first] for values in raw_observations)
        if details is None:
            details = {"environment_step": first, "after_action_index": first-1 if first else None,
                       "state_audit": None, "changed_observation_keys": observation_difference(a, b)}
        details["camera_statistics"] = {"attempt_0": camera_statistics(a), "attempt_1": camera_statistics(b)}
    if details is not None and first > 0:
        progress = [read_jsonl(folder.parent/"nominal_progress.jsonl") for folder in folders]
        matching = [next((r for r in trace if r["step"] == first-1), None) for trace in progress]
        if all(row is not None for row in matching):
            details["progress_channel_matches"] = {
                key: matching[0].get(key) == matching[1].get(key) for key in TRACE_FIELDS}
            details["progress_comparison_action_index"] = first-1
            details["causal_fingerprint_scope"] = "registered_environment_fingerprint_excluding_declared_downstream_controller_caches; not_a_full_state_audit"
    return {"trace_rows": [len(t) for t in traces], "trace_step_support_exact": steps[0] == steps[1],
            "common_observations_compared": len(common), "first_observation_difference_step": first,
            "first_difference_by_observation_key": keys,
            "raw_state_step_support_exact": captured[0] == captured[1],
            "first_difference_by_captured_state_channel": state_differences,
            "captured_state_audits": audits, "camera_statistics": statistics,
            "first_observation_difference": details,
            "raw_observation_counts": [len(values) if values is not None else 0 for values in raw_observations],
            "first_difference_raw_observation_captured": details is not None if first is not None else None,
            "first_difference_raw_state_captured": raw_state_captured if first is not None else None}


class RequestProbe:
    """Record nominal calls, then repeat the first input after nominal replay."""
    def __init__(self, client, folder):
        self.client, self.folder = client, folder
        self.first_request = self.first_response = None
        self.calls = 0

    def infer(self, request):
        frozen = deepcopy(request)
        response = self.client.infer(request)
        if self.calls == 0:
            self.first_request, self.first_response = frozen, deepcopy(response)
        append_jsonl(self.folder/"requests.jsonl", {
            "request_index": self.calls, "sampling_seed": frozen.get("sampling_seed"),
            "request_sha256": stable_digest(frozen), "response_sha256": stable_digest(response),
        })
        self.calls += 1
        return response

    def repeat_first(self):
        if self.first_request is None:
            return {"performed": False, "reason": "no_nominal_request"}
        if self.first_request.get("sampling_seed") is None:
            return {"performed": False, "reason": "request_has_no_explicit_sampling_seed"}
        repeated = self.client.infer(deepcopy(self.first_request))
        def actions(response):
            return response.get("actions") if isinstance(response, dict) else response
        result = {
            "performed": True, "sampling_seed": self.first_request["sampling_seed"],
            "nominal_calls_before_probe": self.calls,
            "request_sha256": stable_digest(self.first_request),
            "actions_exact": stable_digest(actions(self.first_response)) == stable_digest(actions(repeated)),
            "response_exact": stable_digest(self.first_response) == stable_digest(repeated),
        }
        save_payload(self.folder/"repeated_request.pkl.gz", {
            "request": self.first_request, "nominal_response": self.first_response,
            "repeated_response": repeated,
        })
        return result

    def close(self):
        self.client.close()


def diagnose_attempt(reference, env, policy, runtime, folder, *, startup_log=None, observation_window=None,
                     record_all_observations=False):
    folder.mkdir()
    rows, evaluations, actions, inferences = [], [], [], []
    state = {"step": 0, "observation": None, "initial": None}
    observation_rows, captured_states = [], {}
    raw_observations = {}
    capture_observations = observation_window is not None or record_all_observations
    probe = RequestProbe(policy.client, folder)
    policy.client = probe

    def evaluate(e):
        value = deepcopy(runtime["get_subtask_eval"](e))
        evaluations.append(value)
        return value

    def call(p, obs):
        state["observation"] = deepcopy(obs)
        if state["initial"] is None:
            record_startup_event(startup_log, f"{folder.name}:first_nominal_observation")
            state["initial"] = diagnostic_state(env, p, obs, evaluations[-1], 0)
            save_payload(folder/"initial_diagnostic_state.pkl.gz", state["initial"])
            if capture_observations:
                observation_rows.append(observation_key_record(obs, 0))
                if record_all_observations:
                    raw_observations[0] = state["initial"]["observation"]
                if observation_window is not None and observation_window[0] == 0:
                    captured_states[0] = state["initial"]
        boundary = p.at_inference_boundary
        action = runtime["call_policy"](p, obs)
        actions.append(deepcopy(action))
        if boundary:
            inferences.append(deepcopy(p.latest_inference_record))
        return action

    def step(e, action):
        result = runtime["step_fn"](e, action)
        state["observation"] = deepcopy(result[0])
        state["step"] += 1
        return result

    def progress(row):
        rows.append(row)
        append_jsonl(folder/"nominal_progress.jsonl", row)
        if capture_observations:
            step = state["step"]
            record = observation_key_record(state["observation"], step)
            if record["observation_sha256"] != row["next_observation_sha256"]:
                raise ValueError("Captured observation differs from frozen replay progress")
            observation_rows.append(record)
            if record_all_observations:
                # This is already a private deep copy made by the step wrapper.
                # Keep it alive without another copy or any renderer call.
                raw_observations[step] = state["observation"]
            if observation_window is not None and observation_window[0] <= step <= observation_window[1]:
                captured_states[step] = diagnostic_state(
                    env, policy, state["observation"], evaluations[-1], step)

    error = None
    try:
        capture_entry(reference, env, policy,
                      {**runtime, "get_subtask_eval": evaluate, "call_policy": call, "step_fn": step},
                      progress=progress)
    except Exception as exc:
        error = {"message": str(exc), "traceback": traceback.format_exc()}
    if state["observation"] is None or not evaluations:
        atomic_write_json(folder/"failure.json", error or {"message": "No replay observation"})
        raise ValueError(f"Replay failed before initial-state capture; inspect {folder}")
    # A diagnostic envelope can retain a non-boundary endpoint without clearing
    # its cache or manufacturing an eligible FullSnapshot.
    endpoint = diagnostic_state(env, policy, state["observation"], evaluations[-1], state["step"])
    save_payload(folder/"endpoint_diagnostic_state.pkl.gz", endpoint)
    # Buffer the extra observations in memory. Compression and extra file I/O
    # happen only after nominal replay and endpoint capture, with no rerenders.
    if capture_observations:
        window_folder = folder/"observation_window"
        window_folder.mkdir()
        for row in observation_rows:
            append_jsonl(window_folder/"trace.jsonl", row)
        for step, value in captured_states.items():
            save_payload(window_folder/f"state_{step:06d}.pkl.gz", value)
        if record_all_observations:
            save_payload(window_folder/"all_observations.pkl.gz", raw_observations, compresslevel=1)
    save_payload(folder/"nominal_replay.pkl.gz", {
        "actions": actions, "inferences": inferences, "subtask_evals": evaluations,
    })
    audit = reference_audit(reference, SimpleNamespace(**endpoint))
    atomic_write_json(folder/"source_endpoint_audit.json", audit)
    initial = state["initial"]
    initial_xml = _model_xml_roundtrip_audit(
        reference.environment_state.get("model"), initial["environment_state"].get("model"))
    initial_xml["available"] = (reference.environment_state.get("model") is not None
                                or initial["environment_state"].get("model") is not None)
    # Extra server calls occur only after the endpoint is captured. They are
    # explicitly excluded from the nominal request/action schedule.
    try:
        request_probe = probe.repeat_first()
    except Exception as exc:
        request_probe = {"performed": False, "error": str(exc), "traceback": traceback.format_exc()}
    report = {
        "replay_error": error, "prefix_replay_accepted": error is None and audit["all_exact"],
        "source_endpoint_all_exact": audit["all_exact"], "source_endpoint_channels": audit["channels"],
        "initial_model_xml_audit": initial_xml, "request_repeat_probe": request_probe,
        "environment_steps": state["step"], "nominal_requests": probe.calls,
        "final_ordered_stage": ordered_trace(evaluations)[-1]["ordered_current_subtask"],
        "final_at_inference_boundary": bool(policy.at_inference_boundary),
        "last_progress": rows[-1] if rows else None,
        "observation_window": {"requested": observation_window,
                               "captured_steps": sorted(captured_states),
                               "observation_trace_rows": len(observation_rows),
                               "record_all_observations": record_all_observations,
                               "raw_observation_count": len(raw_observations),
                               "raw_observation_array_bytes": int(sum(
                                   getattr(value, "nbytes", 0) for obs in raw_observations.values()
                                   for value in obs.values()))},
    }
    atomic_write_json(folder/"report.json", report)
    return report, initial, endpoint, rows


def run(args, runtime=None):
    source = args.timing_run_dir.resolve()
    registration = json.loads((source/"registration.json").read_text())
    validate_registration(registration)
    parent = next((p for p in registration["parents"] if p["parent_id"] == args.parent_id), None)
    if parent is None:
        raise ValueError("Parent is not in the frozen timing registration")
    sid = parent["snapshot_id"]
    reference, integrity = load_registered_phase2_snapshot(
        Path(registration["phase2_run_dir"])/"snapshots"/f"{sid}.pkl.gz",
        expected_file_sha256=registration["phase2_registration"]["phase2_snapshot_file_sha256"][sid],
        expected_snapshot_id=sid, expected_parent_id=parent["parent_id"], expected_task_name=parent["task_name"])
    window = args.observation_window
    if window is not None and not (0 <= window[0] <= window[1] <= reference.environment_step):
        raise ValueError("Observation window must satisfy 0 <= start <= end <= source environment step")
    plan = json.loads((Path(registration["phase2_run_dir"])/"plan.json").read_text())
    _validate_live_provenance(args, registration["phase2_registration"], plan)
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "parent": parent, "replays": 2,
                          "rng_runtime_init": args.rng_runtime_init, "observation_window": window,
                          "record_all_observations": args.record_all_observations}, indent=2))
        return
    root = args.output_dir.resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Diagnostic root must be separate from the frozen timing root")
    root.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).resolve().parents[2]/"scripts/diagnose_xr1_timing_replay.sh"]
    if window is not None or args.record_all_observations:
        paths.append(Path(__file__).with_name("diagnose_intervention_timing_observations.py"))
    device_assignment = {
        "simulator_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy_server_cuda_visible_devices": os.environ.get("XR1_REPLAY_SERVER_GPU"),
        "simulator_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "simulator_pid": os.getpid(), "policy_server_pid": os.environ.get("XR1_REPLAY_SERVER_PID"),
    }
    atomic_write_json(root/"registration.json", {
        "protocol": PROTOCOL, "created_at": utc_now(), "timing_run_dir": str(source),
        "timing_registration_sha256": registration["registration_sha256"],
        "parent": parent, "source_integrity": integrity, "replays": 2,
        "code_hashes": {str(p.name): sha256_file(p) for p in paths},
        "policy_lifecycle": "fresh_policy_and_environment_per_attempt_like_original_phase2",
        "rng_runtime_init": args.rng_runtime_init,
        "device_assignment": device_assignment,
        "observation_window": window,
        "record_all_observations": args.record_all_observations,
        "observation_capture": "per_key_hashes_all_steps; raw_window_after_step_evaluation_before_next_policy_append; buffered_until_endpoint; no_extra_renders",
        "source_rng_restoration": "unchanged_in_frozen_nominal_replay_after_environment_reset",
        "request_probe": "repeat_first_explicitly_seeded_input_after_nominal_endpoint_capture",
        "scientific_trajectories_collected": 0,
    })
    startup_log = root/"startup_events.jsonl"
    atomic_write_json(root/"status.json", {"status": "running", "stage": "runtime_initialization",
                      "updated_at": utc_now()})
    try:
        record_startup_event(startup_log, "before_optional_rng_runtime_initialization")
        early_initialization = None
        if args.rng_runtime_init == "before_env":
            early_initialization = initialize_rng_capture_runtime()
            record_startup_event(startup_log, "after_early_rng_runtime_initialization")
        config = _runtime_args(args, plan)
        runtime = runtime or _runtime()
        factory = runtime["load_factory"](plan["policy_module"])
    except Exception as exc:
        atomic_write_json(root/"status.json", {"status": "failed", "error": str(exc),
                          "traceback": traceback.format_exc(), "updated_at": utc_now()})
        raise
    attempts = []
    for index in range(2):
        atomic_write_json(root/"status.json", {"status": "running", "attempt": index,
                          "updated_at": utc_now()})
        env = policy = None
        try:
            seed = int(reference.metadata["environment_seed"])
            record_startup_event(startup_log, f"attempt_{index}:before_environment_creation")
            env = _call_with_deterministic_environment_seed(seed, runtime["make_env"],
                reference.task_name, config.env_interface, config.split, seed, True)
            record_startup_event(startup_log, f"attempt_{index}:after_environment_creation")
            _configure_phase2_environment(env, config)
            policy = runtime["call_factory"](factory, env, _policy_args(reference, args))
            record_startup_event(startup_log, f"attempt_{index}:after_policy_creation")
            attempts.append(diagnose_attempt(reference, env, policy, runtime, root/f"attempt_{index}",
                                             startup_log=startup_log, observation_window=window,
                                             record_all_observations=args.record_all_observations))
            report = attempts[-1][0]
            print(f"replay {index+1}/2: " + json.dumps({k: report[k] for k in (
                "prefix_replay_accepted", "source_endpoint_channels", "final_ordered_stage", "request_repeat_probe")}), flush=True)
        except Exception as exc:
            atomic_write_json(root/"status.json", {"status": "failed", "error": str(exc),
                              "traceback": traceback.format_exc(), "updated_at": utc_now()})
            raise
        finally:
            try:
                if env is not None:
                    _close_branch_environment(env)
            finally:
                if policy is not None:
                    policy.close()
    initial_audit = reference_audit(SimpleNamespace(**attempts[0][1]), SimpleNamespace(**attempts[1][1]))
    endpoint_audit = reference_audit(SimpleNamespace(**attempts[0][2]), SimpleNamespace(**attempts[1][2]))
    divergence = first_divergence(attempts[0][3], attempts[1][3])
    previous = sorted((source/"bases"/parent["parent_id"]).glob("attempt_*/nominal_progress.jsonl"),
                      key=lambda p: p.stat().st_mtime)
    report = {
        "protocol": PROTOCOL, "status": "diagnostic_complete", "parent_id": parent["parent_id"],
        "rng_runtime_init": args.rng_runtime_init, "early_rng_runtime_initialization": early_initialization,
        "device_assignment": device_assignment,
        "startup_events": read_jsonl(startup_log),
        "attempts": [a[0] for a in attempts], "repeated_initial_state_audit": initial_audit,
        "repeated_endpoint_audit": endpoint_audit, "first_repeat_divergence": divergence,
        "previous_progress_path": str(previous[-1]) if previous else None,
        "first_divergence_from_failed_attempt": first_divergence(read_jsonl(previous[-1]), attempts[0][3]) if previous else None,
        "full_study_remains_incomplete": True, "scientific_trajectories_collected": 0,
    }
    if window is not None or args.record_all_observations:
        try:
            comparison = compare_observation_windows(root)
            comparison["requested_window"] = window
            comparison["requested_window_complete"] = None if window is None else all(
                a[0]["observation_window"]["captured_steps"] == list(range(window[0], window[1]+1))
                for a in attempts)
            report["observation_window_comparison"] = comparison
        except Exception as exc:
            atomic_write_json(root/"status.json", {"status": "failed", "stage": "observation_window_comparison",
                              "error": str(exc), "traceback": traceback.format_exc(), "updated_at": utc_now()})
            raise
    probes = [a[0]["request_repeat_probe"] for a in attempts]
    if any(p.get("actions_exact") is False for p in probes):
        report["finding"] = "identical_seeded_request_produced_different_actions"
    elif not all(p.get("performed") and p.get("response_exact") for p in probes):
        report["finding"] = "request_repeat_probe_incomplete_or_full_response_differs"
    elif not initial_audit["all_exact"]:
        report["finding"] = "fresh_reset_initial_states_differ"
    elif any(a[0]["initial_model_xml_audit"]["available"] and not a[0]["initial_model_xml_audit"]["compatible"] for a in attempts):
        report["finding"] = "reset_model_xml_differs_from_historical_source"
    elif divergence is not None:
        report["finding"] = "nominal_replays_diverge_after_matching_initial_states"
    elif all(a[0]["prefix_replay_accepted"] for a in attempts):
        report["finding"] = "isolated_replays_reproduce_source_full_study_still_blocked"
    else:
        report["finding"] = "repeatable_isolated_replay_differs_from_historical_source"
    report["artifact_hashes"] = {str(p.relative_to(root)): sha256_file(p)
                                 for p in sorted(root.rglob("*"))
                                 if p.is_file() and p not in {root/"report.json", root/"status.json"}}
    atomic_write_json(root/"report.json", report)
    atomic_write_json(root/"status.json", {"status": "diagnostic_complete", "updated_at": utc_now()})
    print("diagnostic report: " + str(root/"report.json"), flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-run-dir", type=Path, required=True)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10319)
    parser.add_argument("--rng-runtime-init", choices=("after_reset", "before_env"), default="after_reset",
                        help="Optionally initialize RNG inspection before either environment is created")
    parser.add_argument("--observation-window", nargs=2, type=int, metavar=("START", "END"),
                        help="Save raw diagnostic state at inclusive environment steps START..END and per-key observation hashes at every step")
    parser.add_argument("--record-all-observations", action="store_true",
                        help="Buffer every returned raw observation and write it after the endpoint, covering a moving divergence boundary")
    parser.add_argument("--restore-atol", type=float, default=1e-8)
    parser.add_argument("--restore-rtol", type=float, default=1e-8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
