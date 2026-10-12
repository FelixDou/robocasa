"""Replay recorded actions to isolate observation differences without a policy server."""

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import subprocess
import sys
import traceback

import numpy as np

from robocasa.recovery.diagnose_intervention_timing_replay import PROTOCOL as SOURCE_PROTOCOL
from robocasa.recovery.full_snapshot import (
    environment_fingerprint, causal_transition_fingerprint, stable_digest,
    compare_structures, _compare_environment_states, _capture_global_rng_state,
    _restore_global_rng_state, _capture_environment_rng_state,
)
from robocasa.recovery.intervention_timing import validate_registration
from robocasa.recovery.phase3b_training_horizon import (
    atomic_write_json, load_registered_phase2_snapshot, read_jsonl, sha256_file,
)
from robocasa.recovery.run_intervention_timing import (
    reconstruct_idle_environment_rng, assert_reconstructed_rng_idle,
)
from robocasa.recovery.run_phase2_snapshot_replay import (
    _runtime, _reset_env, _call_with_deterministic_environment_seed,
    _configure_phase2_environment, _close_branch_environment, append_jsonl, utc_now,
)
from robocasa.recovery.run_phase3b_training_horizon_continuation import _runtime_args


PROTOCOL = "xr1_recorded_action_observation_diagnostic_v1"
INIT_ORDER_PROTOCOL = "xr1_rng_renderer_initialization_order_diagnostic_v1"
STATE_FIELDS = (
    "environment_state", "simulator_integration_state", "controller_state",
    "environment_control_state", "environment_rng_state", "global_rng_state",
    "subtask_eval", "environment_step",
)


def save_state(path, value):
    with gzip.open(path, "wb", compresslevel=1) as stream:
        pickle.dump(value, stream, protocol=5)


def load_state(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def record_startup_event(path, phase):
    """Inspect an already imported Torch module without initializing CUDA."""
    if path is None:
        return
    torch = sys.modules.get("torch")
    append_jsonl(path, {"phase": phase, "torch_loaded": torch is not None,
                       "cuda_initialized": bool(torch.cuda.is_initialized()) if torch is not None else False})


def initialize_rng_capture_runtime():
    """Initialize the existing RNG inspection path before creating a renderer.

    No source RNG state is installed here. Its original post-reset restoration
    stays in place, so both startup orders must pass the same initial audit.
    """
    state = _capture_global_rng_state()
    return {"torch_available": state["torch_available"],
            "torch_cuda_available": state.get("torch_cuda_available", False),
            "cuda_rng_streams": len(state.get("torch_cuda", [])),
            "source_rng_restored_early": False}


def camera_statistics(observation):
    result = {}
    for key, value in observation.items():
        if not key.startswith("video."):
            continue
        rgb = np.asarray(value)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            continue
        result[key] = {"mean_rgb": [round(float(v), 2) for v in rgb.mean(axis=(0, 1))],
                       "std": round(float(rgb.std()), 2),
                       "dark_pixel_percent": round(100*float((rgb < 12).all(axis=-1).mean()), 2),
                       "bright_pixel_percent": round(100*float((rgb > 243).all(axis=-1).mean()), 2)}
    return result


def inspect_state(env, observation, evaluation, step):
    before = stable_digest((_capture_global_rng_state(), _capture_environment_rng_state(env)))
    fingerprint = environment_fingerprint(env)
    if before != stable_digest((_capture_global_rng_state(), _capture_environment_rng_state(env))):
        raise ValueError("Diagnostic state inspection advances RNG")
    state = {
        "environment_state": fingerprint["simulator_state"]["environment_state"],
        "simulator_integration_state": fingerprint["simulator_state"]["integration_state"],
        "controller_state": fingerprint["controller_state"],
        "environment_control_state": fingerprint["control_state"],
        "environment_rng_state": fingerprint["rng_state"],
        "global_rng_state": _capture_global_rng_state(),
        "subtask_eval": deepcopy(evaluation), "environment_step": step,
        "observation": deepcopy(observation),
    }
    # Model XML is reported separately: its serialization hash alone is not a
    # physical-state difference. The initial audit checks XML compatibility.
    hashes = {key: stable_digest(state[key]) for key in STATE_FIELDS if key != "environment_state"}
    hashes["environment_state_without_model"] = stable_digest(
        {k: v for k, v in state["environment_state"].items() if k != "model"})
    row = {
        "environment_step": step, "state_sha256": hashes,
        "model_xml_sha256": stable_digest(state["environment_state"].get("model")),
        "causal_environment_sha256": stable_digest(causal_transition_fingerprint(fingerprint)),
        "observation_sha256": stable_digest(state["observation"]),
        "observation_key_sha256": {k: stable_digest(v) for k, v in state["observation"].items()},
    }
    return state, row


def audit_states(left, right, *, observations=True):
    differences, xml = _compare_environment_states(
        left["environment_state"], right["environment_state"], atol=0, rtol=0)
    channels = {"environment_state": not differences}
    details = {"environment_state": differences} if differences else {}
    fields = [k for k in STATE_FIELDS if k != "environment_state"]
    if observations:
        fields.append("observation")
    for key in fields:
        differences = compare_structures(left[key], right[key])
        channels[key] = not differences
        if differences:
            details[key] = differences
    return {"all_exact": all(channels.values()), "channels": channels,
            "mismatches": details, "model_xml_audit": xml}


def observation_difference(left, right):
    result = {}
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            result[key] = {"reason": "missing_key", "left_present": key in left,
                           "right_present": key in right}
            continue
        a, b = left[key], right[key]
        if stable_digest(a) == stable_digest(b):
            continue
        detail = {"left_sha256": stable_digest(a), "right_sha256": stable_digest(b)}
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            detail.update(left_shape=list(a.shape), right_shape=list(b.shape),
                          left_dtype=str(a.dtype), right_dtype=str(b.dtype))
            if a.shape == b.shape and a.dtype == b.dtype and a.dtype.kind in "buif":
                changed = a != b
                detail["changed_elements"] = int(np.count_nonzero(changed))
                error = np.abs(a.astype(np.float64) - b.astype(np.float64))
                finite = error[np.isfinite(error)]
                detail["max_abs_error"] = float(finite.max()) if finite.size else None
                detail["mean_abs_error"] = float(finite.mean()) if finite.size else None
                if key.startswith("video.") and changed.ndim == 3:
                    pixels = np.any(changed, axis=-1)
                    y, x = np.nonzero(pixels)
                    detail["changed_pixels"] = int(pixels.sum())
                    detail["changed_pixel_fraction"] = float(pixels.mean())
                    detail["pixel_bbox_xyxy_inclusive"] = (
                        [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if x.size else None)
        else:
            detail["differences"] = compare_structures(a, b)
        result[key] = detail
    return result


def compare_traces(left, right, left_folder, right_folder):
    if len(left) != len(right) or any(a["environment_step"] != b["environment_step"]
                                      for a, b in zip(left, right)):
        raise ValueError("Recorded-action traces have different step support")
    channels, keys = {}, {}
    first_observation = first_state = first_model = None
    for a, b in zip(left, right):
        step = a["environment_step"]
        for key in sorted(set(a["state_sha256"]) | set(b["state_sha256"])):
            if a["state_sha256"].get(key) != b["state_sha256"].get(key):
                channels.setdefault(key, step)
                if first_state is None:
                    first_state = step
        if a["model_xml_sha256"] != b["model_xml_sha256"] and first_model is None:
            first_model = step
        for key in sorted(set(a["observation_key_sha256"]) | set(b["observation_key_sha256"])):
            if a["observation_key_sha256"].get(key) != b["observation_key_sha256"].get(key):
                keys.setdefault(key, step)
        if a["observation_sha256"] != b["observation_sha256"] and first_observation is None:
            first_observation = step
    details = None
    if first_observation is not None:
        name = f"state_{first_observation:06d}.pkl.gz"
        a, b = load_state(left_folder/name), load_state(right_folder/name)
        details = {"environment_step": first_observation,
                   "after_action_index": first_observation-1 if first_observation else None,
                   "state_audit": audit_states(a, b, observations=False),
                   "changed_observation_keys": observation_difference(a["observation"], b["observation"])}
    return {"observations_compared": len(left), "first_observation_difference_step": first_observation,
            "first_non_observation_difference_step": first_state, "first_model_xml_hash_difference_step": first_model,
            "first_difference_by_state_channel": channels, "first_difference_by_observation_key": keys,
            "first_observation_difference": details}


def render_probe(env, observation, evaluation, step, count, folder, evaluate):
    target = getattr(env, "unwrapped", env)
    render = getattr(target, "get_canonical_camera_observation", None)
    if not callable(render) or not getattr(target, "canonical_camera_observations", False):
        return {"performed": False, "reason": "canonical_camera_observations_unavailable"}
    before, _ = inspect_state(env, observation, evaluation, step)
    previous = before
    results = []
    for index in range(count):
        rendered = render(deepcopy(observation))  # no force-update of raw observables
        after, _ = inspect_state(env, rendered, evaluate(env), step)
        save_state(folder/f"render_{index:02d}.pkl.gz", after)
        results.append({"render_index": index,
                        "environment_audit": audit_states(before, after, observations=False),
                        "changed_observation_keys": observation_difference(previous["observation"], after["observation"])})
        previous = after
    return {"performed": True, "environment_step": step, "renders": results,
            "comparison": "returned_step_observation_to_first_rerender_then_consecutive_rerenders",
            "environment_unchanged": all(r["environment_audit"]["all_exact"] for r in results)}


def replay_actions(reference, initial_reference, actions, env, runtime, folder, *, render_count=0, startup_log=None):
    folder.mkdir()
    seed = int(reference.metadata["environment_seed"])
    observation, _ = _call_with_deterministic_environment_seed(seed, _reset_env, env, seed)
    record_startup_event(startup_log, f"{folder.name}:after_reset_before_source_rng_restore")
    _restore_global_rng_state(reference.global_rng_state)
    record_startup_event(startup_log, f"{folder.name}:after_source_rng_restore")
    global_rng = stable_digest(_capture_global_rng_state())
    record_startup_event(startup_log, f"{folder.name}:after_global_rng_capture")
    reconstructed, alignment = reconstruct_idle_environment_rng(reference, env)
    rows = []
    initial_audit = None
    for step in range(len(actions)+1):
        evaluation = runtime["get_subtask_eval"](env)
        state, row = inspect_state(env, observation, evaluation, step)
        save_state(folder/f"state_{step:06d}.pkl.gz", state)
        append_jsonl(folder/"observations.jsonl", row)
        rows.append(row)
        if step == 0:
            initial_audit = audit_states(initial_reference, state)
            atomic_write_json(folder/"initial_audit.json", initial_audit)
            if not initial_audit["all_exact"]:
                raise ValueError(f"Initial environment/observation differs from recorded action source; inspect {folder}")
        if stable_digest(_capture_global_rng_state()) != global_rng:
            raise ValueError("Recorded-action replay advances unrecorded global RNG")
        assert_reconstructed_rng_idle(env, reconstructed)
        if step == len(actions):
            break
        # Some wrappers mutate input dictionaries. Always keep the frozen action
        # list intact, and never use observations to choose a new action.
        observation, reward, done, info = runtime["step_fn"](env, deepcopy(actions[step]))
        if (done or runtime["success_fn"](info=info, reward=reward, env=env)) and step+1 < len(actions):
            raise ValueError(f"Recorded-action replay terminated early at environment step {step+1}")
    probe = render_probe(env, observation, evaluation, len(actions), render_count, folder,
                         runtime["get_subtask_eval"]) if render_count else None
    report = {"environment_steps": len(actions), "policy_calls": 0,
              "initial_audit": initial_audit, "environment_rng_initial_alignment": alignment,
              "actions_sha256": stable_digest(actions), "render_probe": probe}
    atomic_write_json(folder/"report.json", report)
    return report, rows


def load_inputs(args):
    source = args.source_diagnostic_dir.resolve()
    report = json.loads((source/"report.json").read_text())
    if report.get("protocol") != SOURCE_PROTOCOL or report.get("status") != "diagnostic_complete":
        raise ValueError("Require a completed isolated nominal replay diagnostic")
    names = ["registration.json", *[f"attempt_{args.source_attempt}/{name}" for name in (
        "initial_diagnostic_state.pkl.gz", "nominal_replay.pkl.gz", "nominal_progress.jsonl")]]
    hashes = {}
    for name in names:
        actual = sha256_file(source/name)
        if actual != report.get("artifact_hashes", {}).get(name):
            raise ValueError(f"Diagnostic input checksum mismatch: {name}")
        hashes[name] = actual
    diagnostic = json.loads((source/"registration.json").read_text())
    timing = json.loads((Path(diagnostic["timing_run_dir"])/"registration.json").read_text())
    validate_registration(timing)
    if timing["registration_sha256"] != diagnostic["timing_registration_sha256"]:
        raise ValueError("Diagnostic timing registration identity differs")
    parent = diagnostic["parent"]
    if parent not in timing["parents"] or parent["parent_id"] != report["parent_id"]:
        raise ValueError("Diagnostic parent identity differs")
    sid = parent["snapshot_id"]
    phase2 = Path(timing["phase2_run_dir"])
    reference, integrity = load_registered_phase2_snapshot(
        phase2/"snapshots"/f"{sid}.pkl.gz",
        expected_file_sha256=timing["phase2_registration"]["phase2_snapshot_file_sha256"][sid],
        expected_snapshot_id=sid, expected_parent_id=parent["parent_id"], expected_task_name=parent["task_name"])
    initial = load_state(source/names[1])
    nominal = load_state(source/names[2])
    progress = read_jsonl(source/names[3])
    actions = nominal["actions"]
    if len(actions) != reference.environment_step or len(progress) != len(actions):
        raise ValueError("Source actions do not cover the complete nominal replay")
    if any(row["step"] != index or row["action_sha256"] != stable_digest(action)
           for index, (row, action) in enumerate(zip(progress, actions))):
        raise ValueError("Source action sequence disagrees with its progress trace")
    if not 0 <= args.probe_step <= len(actions) or args.render_probes < 1:
        raise ValueError("Probe step must be within recorded actions and render probes positive")
    return reference, initial, actions, progress, json.loads((phase2/"plan.json").read_text()), {
        "source_diagnostic_dir": str(source), "source_attempt": args.source_attempt,
        "source_report_sha256": sha256_file(source/"report.json"), "input_hashes": hashes,
        "timing_registration_sha256": timing["registration_sha256"],
        "protected_input_directories": [str(source), diagnostic["timing_run_dir"],
                                        str(phase2), timing["operator_run_dir"]],
        "parent": parent, "source_integrity": integrity,
    }


def run(args, runtime=None):
    if args.compare_init_order:
        return compare_initialization_orders(args)
    root, source = args.output_dir.resolve(), args.source_diagnostic_dir.resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Observation diagnostic root must be separate from the source diagnostic")
    reference, initial, actions, progress, plan, provenance = load_inputs(args)
    for name in provenance["protected_input_directories"]:
        protected = Path(name).resolve()
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError("Observation diagnostic root must be separate from all frozen input directories")
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "parent": provenance["parent"],
                          "actions": len(actions), "policy_calls": 0, "probe_step": args.probe_step}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).resolve().parents[2]/"scripts/diagnose_xr1_timing_observations.sh"]
    atomic_write_json(root/"registration.json", {
        **provenance, "protocol": PROTOCOL, "created_at": utc_now(),
        "code_hashes": {p.name: sha256_file(p) for p in paths},
        "actions_sha256": stable_digest(actions), "replays": 2,
        "rng_runtime_init": args.rng_runtime_init,
        "separate_render_probe_replay": {"environment_steps": args.probe_step, "renders": args.render_probes},
        "policy_calls": 0, "scientific_trajectories_collected": 0,
    })
    try:
        startup_log = root/"startup_events.jsonl"
        record_startup_event(startup_log, "before_optional_rng_runtime_initialization")
        early_initialization = None
        if args.rng_runtime_init == "before_env":
            early_initialization = initialize_rng_capture_runtime()
            record_startup_event(startup_log, "after_early_rng_runtime_initialization")
        runtime = runtime or _runtime()
        config = _runtime_args(args, plan)
        attempts = []
        for index in range(3):
            name = f"attempt_{index}" if index < 2 else "render_probe"
            atomic_write_json(root/"status.json", {"status": "running", "attempt": name, "updated_at": utc_now()})
            env = None
            try:
                seed = int(reference.metadata["environment_seed"])
                record_startup_event(startup_log, f"{name}:before_environment_creation")
                env = _call_with_deterministic_environment_seed(seed, runtime["make_env"],
                    reference.task_name, config.env_interface, config.split, seed, True)
                record_startup_event(startup_log, f"{name}:after_environment_creation")
                _configure_phase2_environment(env, config)
                selected = actions if index < 2 else actions[:args.probe_step]
                result = replay_actions(reference, initial, selected, env, runtime, root/name,
                                        render_count=args.render_probes if index == 2 else 0,
                                        startup_log=startup_log)
                attempts.append(result)
                print(f"{name}: {len(selected)} fixed actions, initial audit exact, policy calls 0", flush=True)
            finally:
                if env is not None:
                    _close_branch_environment(env)
        comparison = compare_traces(attempts[0][1], attempts[1][1], root/"attempt_0", root/"attempt_1")
        probe_comparison = compare_traces(attempts[0][1][:args.probe_step+1], attempts[2][1],
                                          root/"attempt_0", root/"render_probe")
        probe = attempts[2][0]["render_probe"]
        if comparison["first_non_observation_difference_step"] is not None:
            finding = "fixed_actions_produced_different_audited_environment_states"
        elif comparison["first_observation_difference_step"] is not None:
            changed = comparison["first_observation_difference"]["changed_observation_keys"]
            finding = ("camera_observations_differ_with_matching_audited_environment_states"
                       if changed and all(k.startswith("video.") for k in changed)
                       else "observations_differ_with_matching_audited_environment_states")
        else:
            finding = "observation_divergence_not_reproduced_in_fixed_action_pair"
        original = next(({"environment_step": row["environment_step"], "after_action_index": p["step"]}
                         for row, p in zip(attempts[0][1][1:], progress)
                         if row["observation_sha256"] != p["next_observation_sha256"]), None)
        report = {"protocol": PROTOCOL, "status": "diagnostic_complete", "finding": finding,
                  "rng_runtime_init": args.rng_runtime_init, "early_rng_runtime_initialization": early_initialization,
                  "startup_events": read_jsonl(startup_log),
                  "camera_statistics_at_step_1": {
                      name: camera_statistics(load_state(root/name/"state_000001.pkl.gz")["observation"])
                      for name in ("attempt_0", "attempt_1", "render_probe") if (root/name/"state_000001.pkl.gz").exists()},
                  "parent_id": reference.parent_id, "environment_steps_per_repeat": len(actions),
                  "source_attempt": args.source_attempt, "policy_calls": 0,
                  "repeat_comparison": comparison, "render_probe_prefix_comparison": probe_comparison,
                  "render_probe": probe, "first_observation_difference_from_recorded_source": original,
                  "full_study_remains_incomplete": True, "scientific_trajectories_collected": 0}
        report["artifact_hashes"] = {str(p.relative_to(root)): sha256_file(p)
                                     for p in sorted(root.rglob("*")) if p.is_file() and p != root/"status.json"}
        atomic_write_json(root/"report.json", report)
        atomic_write_json(root/"status.json", {"status": "diagnostic_complete", "updated_at": utc_now()})
        print("diagnostic report: " + str(root/"report.json"), flush=True)
        return report
    except Exception as exc:
        atomic_write_json(root/"status.json", {"status": "failed", "error": str(exc),
                          "traceback": traceback.format_exc(), "updated_at": utc_now()})
        raise


def compare_initialization_orders(args):
    """Run each startup order in a new interpreter; never reuse a warmed process."""
    root = args.output_dir.resolve()
    reference, _, actions, _, _, provenance = load_inputs(args)
    for name in provenance["protected_input_directories"]:
        protected = Path(name).resolve()
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError("Initialization diagnostic root must be separate from all frozen input directories")
    if args.dry_run:
        print(json.dumps({"protocol": INIT_ORDER_PROTOCOL, "parent": provenance["parent"],
                          "fresh_process_cases": ["after_reset", "before_env"],
                          "environment_steps_per_repeat": len(actions), "policy_calls": 0}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).resolve().parents[2]/"scripts/diagnose_xr1_timing_observations.sh"]
    atomic_write_json(root/"registration.json", {
        **provenance, "protocol": INIT_ORDER_PROTOCOL, "created_at": utc_now(),
        "code_hashes": {p.name: sha256_file(p) for p in paths},
        "actions_sha256": stable_digest(actions), "fresh_process_cases": ["after_reset", "before_env"],
        "source_rng_restoration": "unchanged_after_environment_reset_in_both_cases", "policy_calls": 0,
        "scientific_trajectories_collected": 0,
    })
    try:
        cases = {}
        for mode in ("after_reset", "before_env"):
            atomic_write_json(root/"status.json", {"status": "running", "case": mode, "updated_at": utc_now()})
            command = [sys.executable, "-u", "-m", "robocasa.recovery.diagnose_intervention_timing_observations",
                       "--source-diagnostic-dir", str(args.source_diagnostic_dir.resolve()),
                       "--output-dir", str(root/mode), "--source-attempt", str(args.source_attempt),
                       "--probe-step", str(args.probe_step), "--render-probes", str(args.render_probes),
                       "--rng-runtime-init", mode,
                       "--restore-atol", str(args.restore_atol), "--restore-rtol", str(args.restore_rtol)]
            print("fresh process startup order: " + mode, flush=True)
            subprocess.run(command, check=True)
            case = json.loads((root/mode/"report.json").read_text())
            if case["status"] != "diagnostic_complete" or case["rng_runtime_init"] != mode:
                raise ValueError("Initialization case did not complete with the requested startup order")
            cases[mode] = case
        control, early = cases["after_reset"], cases["before_env"]
        def cuda_at(case, phase):
            return next((event["cuda_initialized"] for event in case["startup_events"]
                         if event["phase"] == phase), None)
        order_evidence = {
            "control_cuda_before_environment": cuda_at(control, "attempt_0:before_environment_creation"),
            "control_cuda_after_reset_before_rng_restore": cuda_at(control, "attempt_0:after_reset_before_source_rng_restore"),
            "control_cuda_after_rng_restore": cuda_at(control, "attempt_0:after_source_rng_restore"),
            "control_cuda_after_rng_capture": cuda_at(control, "attempt_0:after_global_rng_capture"),
            "early_cuda_before_environment": cuda_at(early, "attempt_0:before_environment_creation"),
        }
        order_changed = (order_evidence["control_cuda_before_environment"] is False
                         and order_evidence["control_cuda_after_reset_before_rng_restore"] is False
                         and order_evidence["control_cuda_after_rng_capture"] is True
                         and order_evidence["early_cuda_before_environment"] is True)
        def comparison(a, b):
            return compare_traces(read_jsonl(a/"observations.jsonl"), read_jsonl(b/"observations.jsonl"), a, b)
        cross = comparison(root/"after_reset"/"attempt_0", root/"before_env"/"attempt_0")
        warm = comparison(root/"after_reset"/"attempt_1", root/"before_env"/"attempt_0")
        control_repeat, early_repeat = control["repeat_comparison"], early["repeat_comparison"]
        early_probe_prefix = early["render_probe_prefix_comparison"]
        comparisons = (control_repeat, early_repeat, early_probe_prefix, cross, warm)
        if not order_changed:
            finding = "cuda_initialization_order_hypothesis_not_exercised"
        elif control_repeat["first_observation_difference_step"] is None:
            finding = "control_did_not_reproduce_observation_divergence"
        elif any(c["first_non_observation_difference_step"] is not None
                 or c["first_model_xml_hash_difference_step"] is not None for c in comparisons):
            finding = "audited_environment_difference_prevents_renderer_initialization_conclusion"
        elif (early_repeat["first_observation_difference_step"] is not None
              or early_probe_prefix["first_observation_difference_step"] is not None):
            finding = "observation_divergence_persists_with_early_rng_runtime_initialization"
        elif not early["render_probe"].get("performed"):
            finding = "early_initialization_pair_matches_but_render_probe_unavailable"
        elif (not early["render_probe"]["environment_unchanged"]
              or any(r["changed_observation_keys"] for r in early["render_probe"]["renders"])):
            finding = "early_initialization_pair_matches_but_render_probe_diverges"
        elif warm["first_observation_difference_step"] is None:
            finding = "early_rng_runtime_initialization_removes_first_environment_divergence_in_this_pair"
        else:
            finding = "early_initialization_repeat_matches_but_differs_from_warmed_control"
        case_fields = ("finding", "rng_runtime_init", "early_rng_runtime_initialization", "startup_events",
                       "camera_statistics_at_step_1", "repeat_comparison", "render_probe_prefix_comparison",
                       "render_probe", "first_observation_difference_from_recorded_source")
        report = {"protocol": INIT_ORDER_PROTOCOL, "status": "diagnostic_complete", "finding": finding,
                  "parent_id": reference.parent_id, "policy_calls": 0,
                  "cases": {mode: {key: case[key] for key in case_fields} for mode, case in cases.items()},
                  "cuda_initialization_order_changed": order_changed, "startup_order_evidence": order_evidence,
                  "first_environment_cross_case_comparison": cross,
                  "warmed_control_vs_first_early_environment": warm,
                  "full_study_remains_incomplete": True, "scientific_trajectories_collected": 0,
                  "interpretation_limit": "one_fresh_process_pair; does_not_certify_historical_nominal_prefix_replay"}
        report["artifact_hashes"] = {str(p.relative_to(root)): sha256_file(p)
                                     for p in sorted(root.rglob("*")) if p.is_file() and p != root/"status.json"}
        atomic_write_json(root/"report.json", report)
        atomic_write_json(root/"status.json", {"status": "diagnostic_complete", "updated_at": utc_now()})
        print("initialization comparison report: " + str(root/"report.json"), flush=True)
        return report
    except Exception as exc:
        atomic_write_json(root/"status.json", {"status": "failed", "error": str(exc),
                          "traceback": traceback.format_exc(), "updated_at": utc_now()})
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-attempt", type=int, choices=(0, 1), default=0)
    parser.add_argument("--probe-step", type=int, default=233,
                        help="Environment step, so 233 is immediately after action index 232")
    parser.add_argument("--render-probes", type=int, default=4)
    parser.add_argument("--rng-runtime-init", choices=("after_reset", "before_env"), default="after_reset")
    parser.add_argument("--compare-init-order", action="store_true",
                        help="Run both initialization orders in separate fresh Python processes")
    parser.add_argument("--restore-atol", type=float, default=1e-8)
    parser.add_argument("--restore-rtol", type=float, default=1e-8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
