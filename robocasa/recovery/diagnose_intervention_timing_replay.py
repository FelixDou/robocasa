"""Isolated replay diagnosis; never changes a timing registration or its ledger."""

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
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


def save_payload(path, value):
    with gzip.open(path, "wb") as stream:
        pickle.dump(value, stream, protocol=5)


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


def diagnose_attempt(reference, env, policy, runtime, folder):
    folder.mkdir()
    rows, evaluations, actions, inferences = [], [], [], []
    state = {"step": 0, "observation": None, "initial": None}
    probe = RequestProbe(policy.client, folder)
    policy.client = probe

    def evaluate(e):
        value = deepcopy(runtime["get_subtask_eval"](e))
        evaluations.append(value)
        return value

    def call(p, obs):
        state["observation"] = deepcopy(obs)
        if state["initial"] is None:
            state["initial"] = diagnostic_state(env, p, obs, evaluations[-1], 0)
            save_payload(folder/"initial_diagnostic_state.pkl.gz", state["initial"])
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
    plan = json.loads((Path(registration["phase2_run_dir"])/"plan.json").read_text())
    _validate_live_provenance(args, registration["phase2_registration"], plan)
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "parent": parent, "replays": 2}, indent=2))
        return
    root = args.output_dir.resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Diagnostic root must be separate from the frozen timing root")
    root.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).resolve().parents[2]/"scripts/diagnose_xr1_timing_replay.sh"]
    atomic_write_json(root/"registration.json", {
        "protocol": PROTOCOL, "created_at": utc_now(), "timing_run_dir": str(source),
        "timing_registration_sha256": registration["registration_sha256"],
        "parent": parent, "source_integrity": integrity, "replays": 2,
        "code_hashes": {str(p.name): sha256_file(p) for p in paths},
        "policy_lifecycle": "fresh_policy_and_environment_per_attempt_like_original_phase2",
        "request_probe": "repeat_first_explicitly_seeded_input_after_nominal_endpoint_capture",
        "scientific_trajectories_collected": 0,
    })
    config = _runtime_args(args, plan)
    runtime = runtime or _runtime()
    factory = runtime["load_factory"](plan["policy_module"])
    attempts = []
    for index in range(2):
        atomic_write_json(root/"status.json", {"status": "running", "attempt": index,
                          "updated_at": utc_now()})
        env = policy = None
        try:
            seed = int(reference.metadata["environment_seed"])
            env = _call_with_deterministic_environment_seed(seed, runtime["make_env"],
                reference.task_name, config.env_interface, config.split, seed, True)
            _configure_phase2_environment(env, config)
            policy = runtime["call_factory"](factory, env, _policy_args(reference, args))
            attempts.append(diagnose_attempt(reference, env, policy, runtime, root/f"attempt_{index}"))
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
        "attempts": [a[0] for a in attempts], "repeated_initial_state_audit": initial_audit,
        "repeated_endpoint_audit": endpoint_audit, "first_repeat_divergence": divergence,
        "previous_progress_path": str(previous[-1]) if previous else None,
        "first_divergence_from_failed_attempt": first_divergence(read_jsonl(previous[-1]), attempts[0][3]) if previous else None,
        "full_study_remains_incomplete": True, "scientific_trajectories_collected": 0,
    }
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
    parser.add_argument("--restore-atol", type=float, default=1e-8)
    parser.add_argument("--restore-rtol", type=float, default=1e-8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
