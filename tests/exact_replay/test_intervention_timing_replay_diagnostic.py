"""Replay diagnostics preserve failed endpoints and never modify frozen evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import gc
import gzip
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()

from tests.exact_replay.test_intervention_timing import snapshot, runtime, source_fixture, LazyBookkeepingEnvironment
from tests.exact_replay.test_operator_screen import make_policy, XiaoEnvironment
from tests.test_safe_xiaomi_policy import FakeClient, safe_response
from robocasa.recovery.full_snapshot import stable_digest
from robocasa.recovery.exact_replay.intervention_timing import build_registration
from robocasa.recovery.exact_replay.phase3b_training_horizon import sha256_file
from robocasa.recovery.exact_replay.diagnose_intervention_timing_replay import (
    RequestProbe, diagnose_attempt, diagnostic_state, first_divergence, run, build_parser,
    initialize_rng_capture_runtime, compare_observation_windows, save_payload, GCTrace,
)


class ReplayDiagnosticTest(unittest.TestCase):
    def test_post_reset_initial_evaluation_collection_counts_even_before_first_policy_call(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = snapshot(step=4)
            rt = runtime()
            evaluate, calls = rt["get_subtask_eval"], []
            def collect_in_initial_evaluation(env):
                if not calls:
                    gc.collect()
                calls.append(env.timestep)
                return evaluate(env)
            rt["get_subtask_eval"] = collect_in_initial_evaluation
            trace = GCTrace("disabled")
            trace.start()
            try:
                report, _, _, _ = diagnose_attempt(reference, XiaoEnvironment(), make_policy(), rt,
                    Path(directory)/"attempt_0", gc_trace=trace)
            finally:
                result = trace.finish()
            self.assertTrue(report["prefix_replay_accepted"])
            self.assertEqual(result["nominal_collection_stops"], 1)
            self.assertEqual(result["setup_collection_stops"], 0)
            stop = next(e for e in result["events"] if e["event"] == "stop")
            self.assertTrue(stop["nominal"])
            self.assertEqual(stop["phase"], "evaluation")
            self.assertEqual(stop["environment_step"], 0)
            self.assertFalse(stop["automatic_gc_enabled"])

    def test_gc_control_defers_cycles_restores_prior_mode_and_traces_explicit_collections(self):
        initial = gc.isenabled()
        destroyed = []
        class Cycle:
            def __init__(self):
                self.cycle = self
            def __del__(self):
                destroyed.append(True)
        try:
            for enabled in (True, False):
                gc.enable() if enabled else gc.disable()
                callbacks = list(gc.callbacks)
                trace = GCTrace("disabled")
                trace.start()
                self.assertFalse(gc.isenabled())
                value = Cycle()
                weak = weakref.ref(value)
                del value
                self.assertIsNotNone(weak())
                trace.phase, trace.step, trace.nominal = "environment_step", 236, True
                # Explicit collection remains possible even with automatic GC
                # disabled, and must be visible rather than reported as absent.
                gc.collect()
                self.assertIsNone(weak())
                result = trace.finish()
                self.assertTrue(result["original_enabled_state_restored"])
                self.assertEqual(gc.isenabled(), enabled)
                self.assertEqual(gc.callbacks, callbacks)
                self.assertEqual(result["nominal_collection_stops"], 1)
                self.assertFalse(result["events"][-1]["automatic_gc_enabled"])
                self.assertEqual(result["events"][-1]["environment_step"], 236)
            self.assertEqual(len(destroyed), 2)
        finally:
            gc.enable() if initial else gc.disable()

    def test_launcher_scopes_server_gpu_without_changing_simulator_visibility(self):
        script = Path(__file__).resolve().parents[2]/"scripts/exact_replay/diagnose_xr1_timing_replay.sh"
        for server_gpu, all_parents in ((None, False), ("1", False), ("1", True)):
            with self.subTest(server_gpu=server_gpu, all_parents=all_parents), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                for name in ("bin", "client/bin", "server/bin", "safe/deploy", "phase2", "source", "run", "robocasa_logs/eval"):
                    (base/name).mkdir(parents=True)
                (base/"source/registration.json").write_text(json.dumps({"phase2_run_dir": str(base/"phase2")}))
                (base/"phase2/plan.json").write_text(json.dumps({"checkpoint_provenance": {"path": str(base/"model")}}))
                # Exercise the real launcher, replacing only the heavyweight
                # executables. No socket, model, simulator, or GPU is used.
                wrapper = f"#!{sys.executable}\n" + '''
import json, os, pathlib, shutil, sys, time, types
root = pathlib.Path(os.environ["XR1_REPLAY_ROOT"])
cli = sys.argv[1:]
assignment = {"visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "pid": os.getpid()}
if cli[:2] == ["-u", "deploy/server.py"]:
    (root/"server.json").write_text(json.dumps(assignment))
    print("Model loaded", flush=True)
    (root/"ready").touch()
    time.sleep(10)
elif cli[:2] == ["-u", "-m"]:
    if "--dry-run" not in cli:
        assignment["module"], assignment["arguments"] = cli[2], cli[3:]
        assignment["server_pid"] = os.environ.get("XR1_REPLAY_SERVER_PID")
        (root/"client.json").write_text(json.dumps(assignment))
elif cli[0] == "-" and cli[1].isdigit():
    raise SystemExit(0 if (root/"ready").exists() else 1)
elif cli[0] == "-":
    shutil.disk_usage = lambda _: types.SimpleNamespace(free=20*2**30)
    sys.argv = cli
    exec(compile(sys.stdin.read(), "<launcher heredoc>", "exec"))
else:
    raise SystemExit("unexpected fake executable arguments")
'''
                for name in ("client/bin/python", "server/bin/python"):
                    (base/name).write_text(wrapper)
                    (base/name).chmod(0o755)
                for name in ("ss", "flock"):
                    (base/"bin"/name).write_text("#!/bin/sh\nexit 0\n")
                    (base/"bin"/name).chmod(0o755)
                (base/"bin/sleep").write_text(f"#!{sys.executable}\nimport time\ntime.sleep(0.02)\n")
                (base/"bin/sleep").chmod(0o755)
                env = {**os.environ, "PATH": str(base/"bin")+os.pathsep+os.environ["PATH"],
                       "STORAGE_BS": str(base), "ROBOCASA_REPO": str(script.parents[2]),
                       "XR1_REPLAY_SOURCE": str(base/"source"), "XR1_REPLAY_ROOT": str(base/"run"),
                       "XR1_CLIENT_ENV": str(base/"client"), "XR1_SERVER_ENV": str(base/"server"),
                       "XR1_SAFE_REPO": str(base/"safe"), "XR1_REPLAY_GPU": "0",
                       "XR1_REPLAY_RNG_RUNTIME_INIT": "before_env", "XR1_REPLAY_RECORD_ALL_OBSERVATIONS": "1",
                       "XR1_REPLAY_OBS_WINDOW_START": "232", "XR1_REPLAY_OBS_WINDOW_END": "236"}
                env.pop("XR1_REPLAY_SERVER_GPU", None)
                env["XR1_REPLAY_ALL_PARENTS"] = "1" if all_parents else "0"
                if all_parents:
                    env["XR1_REPLAY_OBS_WINDOW_START"] = env["XR1_REPLAY_OBS_WINDOW_END"] = ""
                if server_gpu is not None:
                    env["XR1_REPLAY_SERVER_GPU"] = server_gpu
                process = subprocess.run(["bash", str(script), "worker"], env=env,
                                         capture_output=True, text=True, timeout=15)
                self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
                client = json.loads((base/"run/client.json").read_text())
                server = json.loads((base/"run/server.json").read_text())
                self.assertEqual(client["visible"], "0")
                self.assertEqual(client["egl"], "0")
                self.assertEqual(server["visible"], server_gpu or "0")
                self.assertEqual(client["server_pid"], str(server["pid"]))
                self.assertEqual(client["module"], "robocasa.recovery.exact_replay."+(
                    "audit_intervention_timing_parents" if all_parents else "diagnose_intervention_timing_replay"))
                self.assertEqual("--parent-id" in client["arguments"], not all_parents)

    def test_runtime_initialization_uses_rng_inspection_without_early_source_restore(self):
        with patch("robocasa.recovery.exact_replay.diagnose_intervention_timing_replay._capture_global_rng_state",
                   return_value={"torch_available": True, "torch_cuda_available": True,
                                 "torch_cuda": [b"state"]}) as capture, \
             patch("robocasa.recovery.full_snapshot._restore_global_rng_state") as restore:
            result = initialize_rng_capture_runtime()
        capture.assert_called_once_with()
        restore.assert_not_called()
        self.assertTrue(result["torch_cuda_available"])
        self.assertEqual(result["cuda_rng_streams"], 1)
        self.assertFalse(result["source_rng_restored_early"])

    def test_probe_records_identical_input_and_never_inserts_calls_into_nominal_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            client = FakeClient([safe_response(), safe_response(action_value=1)])
            probe = RequestProbe(client, folder)
            request = {"sampling_seed": 123, "input": [1, 2]}
            probe.infer(request)
            request["input"].append(3)
            result = probe.repeat_first()
            self.assertFalse(result["actions_exact"])
            self.assertFalse(result["response_exact"])
            self.assertEqual(client.requests[-1], {"sampling_seed": 123, "input": [1, 2]})
            self.assertEqual(probe.calls, 1)
            self.assertEqual(len((folder/"requests.jsonl").read_text().splitlines()), 1)
            self.assertTrue((folder/"repeated_request.pkl.gz").exists())
            probe.close()
            self.assertTrue(client.closed)

    def test_probe_refuses_unseeded_input_and_checks_features_separately_from_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            probe = RequestProbe(FakeClient([safe_response()]), folder)
            self.assertFalse(probe.repeat_first()["performed"])
            probe.infer({"input": 1})
            self.assertFalse(probe.repeat_first()["performed"])
            self.assertEqual(len(probe.client.requests), 1)
            response = safe_response()
            changed = deepcopy(response)
            changed["safe_features"] += 1
            probe = RequestProbe(FakeClient([response, changed]), folder)
            probe.infer({"sampling_seed": 123})
            result = probe.repeat_first()
            self.assertTrue(result["actions_exact"])
            self.assertFalse(result["response_exact"])

    def test_successful_source_replay_has_exact_endpoint_and_retains_nominal_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            ref = snapshot(step=4)
            report, initial, endpoint, rows = diagnose_attempt(
                ref, XiaoEnvironment(), make_policy(), runtime(), Path(directory)/"attempt")
            self.assertTrue(report["prefix_replay_accepted"], report)
            self.assertEqual(report["environment_steps"], 4)
            self.assertEqual(report["nominal_requests"], 2)
            self.assertTrue(report["request_repeat_probe"]["response_exact"])
            self.assertEqual(initial["environment_step"], 0)
            self.assertEqual(endpoint["environment_step"], 4)
            self.assertEqual(len(rows), 4)
            self.assertEqual(stable_digest(endpoint["policy_state"]), stable_digest(ref.policy_state))
            with gzip.open(Path(directory)/"attempt"/"nominal_replay.pkl.gz", "rb") as stream:
                nominal = pickle.load(stream)
            self.assertEqual(len(nominal["actions"]), 4)
            self.assertEqual(len(nominal["inferences"]), 2)

    def test_rng_reconstruction_survives_inspection_and_mutating_getter_is_rejected(self):
        ref = snapshot(step=4, env_type=LazyBookkeepingEnvironment)
        with tempfile.TemporaryDirectory() as directory:
            report, _, _, _ = diagnose_attempt(
                ref, LazyBookkeepingEnvironment(), make_policy(), runtime(), Path(directory)/"attempt")
            self.assertTrue(report["prefix_replay_accepted"], report)
        env = LazyBookkeepingEnvironment()
        obs, _ = env.reset(seed=1)
        original = env.get_state
        def mutating_getter():
            env.rng.random()
            return original()
        env.get_state = mutating_getter
        with self.assertRaisesRegex(ValueError, "inspection advances RNG"):
            diagnostic_state(env, make_policy(), obs, runtime()["get_subtask_eval"](env), 0)

    def test_wrong_stage_failure_still_saves_complete_endpoint_and_source_audit(self):
        rt = runtime()
        normal = rt["get_subtask_eval"]
        def wrong_stage(env):
            value = normal(env)
            value["required_predicates"].insert(0, "kettle_on_tray")
            value["predicates"]["kettle_on_tray"] = {"value": False}
            return value
        rt["get_subtask_eval"] = wrong_stage
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/"attempt"
            report, _, endpoint, rows = diagnose_attempt(snapshot(step=4), XiaoEnvironment(), make_policy(), rt, folder)
            self.assertIn("prefix-2 boundary", report["replay_error"]["message"])
            self.assertFalse(report["prefix_replay_accepted"])
            self.assertEqual(report["final_ordered_stage"], "kettle_on_tray")
            self.assertFalse(report["source_endpoint_channels"]["subtask_eval"])
            self.assertEqual(endpoint["environment_step"], 4)
            self.assertEqual(len(rows), 4)
            for name in ("endpoint_diagnostic_state.pkl.gz", "initial_diagnostic_state.pkl.gz",
                         "source_endpoint_audit.json", "nominal_replay.pkl.gz", "report.json"):
                self.assertTrue((folder/name).exists(), name)
            self.assertFalse((folder/"committed.json").exists())
            self.assertFalse((folder/"entry.pkl.gz").exists())

    def test_cached_endpoint_failure_does_not_clear_cache_or_manufacture_snapshot(self):
        ref = snapshot(step=4)
        ref.environment_step = 3  # diagnostic reaches a cached step, not a legal source boundary
        with tempfile.TemporaryDirectory() as directory:
            report, _, endpoint, _ = diagnose_attempt(ref, XiaoEnvironment(), make_policy(), runtime(), Path(directory)/"attempt",
                                                      observation_window=(2, 3))
            self.assertFalse(report["final_at_inference_boundary"])
            self.assertFalse(endpoint["policy_state"]["at_inference_boundary"])
            self.assertTrue(endpoint["policy_state"]["action_plan"])
            self.assertFalse(report["prefix_replay_accepted"])
            with gzip.open(Path(directory)/"attempt"/"observation_window"/"state_000003.pkl.gz", "rb") as stream:
                raw = pickle.load(stream)
            self.assertTrue(raw["policy_state"]["action_plan"])
            self.assertFalse(raw["policy_state"]["at_inference_boundary"])

    def test_first_divergence_reports_channel_and_length(self):
        a = [{"step": 0, "action_sha256": "a"}, {"step": 1, "action_sha256": "b"}]
        self.assertIsNone(first_divergence(a, deepcopy(a)))
        b = deepcopy(a)
        b[1]["action_sha256"] = "changed"
        result = first_divergence(a, b)
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["fields"], ["action_sha256"])
        self.assertEqual(first_divergence(a, a[:1])["fields"], ["trace_length"])

    def test_policy_loop_window_distinguishes_camera_state_observation_and_simulator(self):
        for kind in ("camera", "state_observation", "simulator"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                ref, rt = snapshot(step=4), runtime()
                diagnose_attempt(ref, XiaoEnvironment(), make_policy(), rt, root/"attempt_0",
                                 observation_window=(2, 4))
                ordinary = rt["step_fn"]
                def changed(env, action):
                    # No extra compressed state writes are permitted mid-prefix.
                    self.assertFalse((root/"attempt_1"/"observation_window").exists())
                    obs, reward, done, info = ordinary(env, action)
                    if env.timestep == 3:
                        if kind == "camera":
                            obs["video.robot0_agentview_left"][0, 0, 0] += 1
                        elif kind == "state_observation":
                            obs["state.end_effector_position_relative"][0] += 1
                        else:
                            env.sim.data.qvel += 0.001
                    return obs, reward, done, info
                policy = make_policy()
                report, _, _, rows = diagnose_attempt(ref, XiaoEnvironment(), policy,
                    {**rt, "step_fn": changed}, root/"attempt_1", observation_window=(2, 4))
                comparison = compare_observation_windows(root)
                self.assertEqual(report["environment_steps"], 4)
                self.assertEqual(report["nominal_requests"], 2)
                self.assertEqual(len(policy.client.client.requests), 3)  # 2 nominal + existing endpoint probe
                self.assertEqual(comparison["trace_rows"], [5, 5])
                self.assertTrue(comparison["trace_step_support_exact"])
                self.assertEqual(report["observation_window"]["captured_steps"], [2, 3, 4])
                if kind == "simulator":
                    self.assertEqual(comparison["first_difference_by_captured_state_channel"]["simulator_integration_state"], 3)
                else:
                    self.assertEqual(comparison["first_observation_difference_step"], 3)
                    self.assertTrue(comparison["first_difference_raw_state_captured"])
                    detail = comparison["first_observation_difference"]
                    self.assertEqual(detail["after_action_index"], 2)
                    self.assertTrue(all(exact for key, exact in detail["state_audit"]["channels"].items()
                                        if key != "observation"), detail)
                    key = ("video.robot0_agentview_left" if kind == "camera"
                           else "state.end_effector_position_relative")
                    self.assertEqual(comparison["first_difference_by_observation_key"], {key: 3})
                    self.assertEqual(set(detail["changed_observation_keys"]), {key})
                    if kind == "camera":
                        self.assertEqual(detail["changed_observation_keys"][key]["changed_pixels"], 1)
                with gzip.open(root/"attempt_1"/"observation_window"/"state_000003.pkl.gz", "rb") as stream:
                    raw = pickle.load(stream)
                self.assertEqual(stable_digest(raw["observation"]), rows[2]["next_observation_sha256"])
                self.assertFalse((root/"attempt_1"/"entry.pkl.gz").exists())

    def test_mismatch_outside_raw_window_remains_visible_without_claiming_state_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, rt = snapshot(step=4), runtime()
            diagnose_attempt(ref, XiaoEnvironment(), make_policy(), rt, root/"attempt_0",
                             observation_window=(0, 1))
            ordinary = rt["step_fn"]
            def changed(env, action):
                obs, reward, done, info = ordinary(env, action)
                if env.timestep == 3:
                    obs["video.robot0_agentview_left"][0, 0, 0] += 1
                return obs, reward, done, info
            diagnose_attempt(ref, XiaoEnvironment(), make_policy(), {**rt, "step_fn": changed},
                             root/"attempt_1", observation_window=(0, 1))
            comparison = compare_observation_windows(root)
            self.assertEqual(comparison["first_observation_difference_step"], 3)
            self.assertFalse(comparison["first_difference_raw_state_captured"])
            self.assertIsNone(comparison["first_observation_difference"])
            self.assertEqual([r["environment_step"] for r in comparison["captured_state_audits"]], [0, 1])

    def test_all_observations_capture_moving_boundary_without_inventing_state_audit(self):
        for window in (None, (0, 1)):
            with self.subTest(window=window), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                ref, rt = snapshot(step=4), runtime()
                diagnose_attempt(ref, XiaoEnvironment(), make_policy(), rt, root/"attempt_0",
                                 observation_window=window, record_all_observations=True)
                ordinary = rt["step_fn"]
                previous_observations = []
                def changed(env, action):
                    self.assertFalse((root/"attempt_1"/"observation_window").exists())
                    # A producer reusing/mutating an earlier array must not
                    # rewrite the private observation already buffered.
                    if previous_observations:
                        previous_observations[-1]["video.robot0_agentview_left"][:] = 12
                    obs, reward, done, info = ordinary(env, action)
                    if env.timestep == 3:
                        obs["video.robot0_agentview_left"][0, 0, 0] += 1
                    previous_observations.append(obs)
                    return obs, reward, done, info
                policy = make_policy()
                report, _, _, _ = diagnose_attempt(ref, XiaoEnvironment(), policy, {**rt, "step_fn": changed},
                    root/"attempt_1", observation_window=window, record_all_observations=True)
                result = compare_observation_windows(root)
                self.assertEqual(result["first_observation_difference_step"], 3)
                self.assertTrue(result["first_difference_raw_observation_captured"])
                self.assertFalse(result["first_difference_raw_state_captured"])
                self.assertEqual(result["raw_observation_counts"], [5, 5])
                details = result["first_observation_difference"]
                self.assertIsNone(details["state_audit"])
                self.assertEqual(details["after_action_index"], 2)
                self.assertTrue(details["progress_channel_matches"]["causal_environment_sha256"])
                self.assertFalse(details["progress_channel_matches"]["next_observation_sha256"])
                self.assertEqual(details["changed_observation_keys"]["video.robot0_agentview_left"]["changed_pixels"], 1)
                self.assertEqual(report["observation_window"]["raw_observation_count"], 5)
                self.assertGreater(report["observation_window"]["raw_observation_array_bytes"], 0)
                self.assertEqual(len(policy.client.client.requests), 3)
                raw_path = root/"attempt_1"/"observation_window"/"all_observations.pkl.gz"
                with gzip.open(raw_path, "rb") as stream:
                    raw = pickle.load(stream)
                raw[3]["video.robot0_agentview_left"][0, 0, 0] += 1
                save_payload(raw_path, raw)
                with self.assertRaisesRegex(ValueError, "Raw observation differs"):
                    compare_observation_windows(root)

    def test_two_fresh_policy_replays_preserve_failed_source_and_dry_run_has_no_writes(self):
        with tempfile.TemporaryDirectory() as directory, patch("robocasa.recovery.exact_replay.intervention_timing.validate_operator_registration"):
            base = Path(directory)
            old_source = source_fixture(base)
            reg = build_registration(old_source)
            source = base/"timing"
            source.mkdir()
            (source/"registration.json").write_text(json.dumps(reg))
            (source/"errors.jsonl").write_text('{"error":"retained failure"}\n')
            (source/"branch_records.jsonl").write_text('{"branch_id":"preserved"}\n')
            before = {p.name: sha256_file(p) for p in source.iterdir()}
            args = build_parser().parse_args([
                "--timing-run-dir", str(source), "--parent-id", reg["parents"][0]["parent_id"],
                "--output-dir", str(base/"diagnostic"), "--model-path", "/checkpoint", "--dry-run"])
            policies = []
            def factory(*_):
                p = make_policy()
                policies.append(p)
                return p
            rt = runtime()
            rt.update(load_factory=lambda _: None, call_factory=factory)
            with patch("robocasa.recovery.exact_replay.diagnose_intervention_timing_replay._validate_live_provenance"), redirect_stdout(io.StringIO()):
                run(args, rt)
                self.assertFalse(args.output_dir.exists())
                self.assertEqual(policies, [])
                args.dry_run = False
                result = run(args, rt)
            self.assertEqual(len(policies), 2)
            self.assertIsNot(policies[0], policies[1])
            self.assertTrue(all(p.client.client.closed for p in policies))
            self.assertTrue(result["repeated_initial_state_audit"]["all_exact"])
            self.assertTrue(result["repeated_endpoint_audit"]["all_exact"])
            self.assertIsNone(result["first_repeat_divergence"])
            self.assertEqual(result["finding"], "isolated_replays_reproduce_source_full_study_still_blocked")
            self.assertEqual(result["scientific_trajectories_collected"], 0)
            self.assertEqual(before, {p.name: sha256_file(p) for p in source.iterdir()})
            # Early initialization must run before factory loading or environment
            # creation, add no policy requests, and retain exact replay gates.
            control_root = args.output_dir
            args.output_dir = base/"early_diagnostic"
            args.rng_runtime_init = "before_env"
            args.observation_window = [1, 4]
            args.record_all_observations = True
            args.gc_mode = "disabled"
            events = []
            make_env = rt["make_env"]
            def early():
                events.append("initialize_rng_runtime")
                return {"torch_available": False, "source_rng_restored_early": False}
            def create(*a):
                events.append("make_env")
                return make_env(*a)
            rt.update(make_env=create, load_factory=lambda _: events.append("load_factory"))
            module = "robocasa.recovery.exact_replay.diagnose_intervention_timing_replay"
            with patch(module+"._validate_live_provenance"), \
                 patch(module+".initialize_rng_capture_runtime", side_effect=early) as warm, \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "XR1_REPLAY_SERVER_GPU": "1",
                                         "MUJOCO_EGL_DEVICE_ID": "0", "XR1_REPLAY_SERVER_PID": "42"}), \
                 redirect_stdout(io.StringIO()):
                args.dry_run = True
                run(args, rt)
                warm.assert_not_called()
                self.assertFalse(args.output_dir.exists())
                args.dry_run = False
                initialized = run(args, rt)
                warm.assert_called_once_with()
            self.assertEqual(events, ["initialize_rng_runtime", "load_factory", "make_env", "make_env"])
            self.assertEqual(initialized["rng_runtime_init"], "before_env")
            self.assertEqual(initialized["garbage_collection"]["mode"], "disabled")
            for trace in initialized["garbage_collection"]["attempts"]:
                self.assertFalse(trace["enabled_at_environment_creation"])
                self.assertTrue(trace["original_enabled_state_restored"])
                self.assertEqual(trace["nominal_collection_stops"], 0)
            assignment = initialized["device_assignment"]
            self.assertEqual(assignment["simulator_cuda_visible_devices"], "0")
            self.assertEqual(assignment["policy_server_cuda_visible_devices"], "1")
            self.assertEqual(assignment["simulator_egl_device_id"], "0")
            self.assertEqual(assignment["policy_server_pid"], "42")
            self.assertEqual(json.loads((args.output_dir/"registration.json").read_text())["device_assignment"], assignment)
            self.assertTrue(all(a["prefix_replay_accepted"] for a in initialized["attempts"]))
            self.assertTrue(initialized["repeated_initial_state_audit"]["all_exact"])
            self.assertTrue(initialized["repeated_endpoint_audit"]["all_exact"])
            self.assertIsNone(initialized["first_repeat_divergence"])
            window = initialized["observation_window_comparison"]
            self.assertTrue(window["requested_window_complete"])
            self.assertEqual(window["trace_rows"], [5, 5])
            self.assertIsNone(window["first_observation_difference_step"])
            self.assertTrue(all(r["all_exact"] for r in window["captured_state_audits"]))
            self.assertIn("attempt_0/observation_window/state_000003.pkl.gz", initialized["artifact_hashes"])
            self.assertIn("attempt_0/observation_window/all_observations.pkl.gz", initialized["artifact_hashes"])
            self.assertEqual(window["raw_observation_counts"], [5, 5])
            for index in range(2):
                for trace in ("requests.jsonl", "nominal_progress.jsonl"):
                    name = f"attempt_{index}/{trace}"
                    self.assertEqual((control_root/name).read_text(), (args.output_dir/name).read_text())
            self.assertEqual(before, {p.name: sha256_file(p) for p in source.iterdir()})
            self.assertEqual(len(policies), 4)
            self.assertTrue(all(p.client.client.closed for p in policies))
            for invalid in ([-1, 1], [2, 1], [1, 5]):
                args.observation_window = invalid
                args.output_dir = base/"invalid_window"
                with self.assertRaisesRegex(ValueError, "Observation window"):
                    run(args, rt)
                self.assertFalse(args.output_dir.exists())
            args.observation_window = [1, 4]
            args.output_dir = base/"early_diagnostic"
            # Both a reused output root and an overlapping source/output root fail closed.
            with patch("robocasa.recovery.exact_replay.diagnose_intervention_timing_replay._validate_live_provenance"):
                with self.assertRaises(FileExistsError):
                    run(args, rt)
                args.output_dir = source/"nested"
                with self.assertRaisesRegex(ValueError, "separate"):
                    run(args, rt)
            args.output_dir = base/"failed_initialization"
            with patch(module+"._validate_live_provenance"), \
                 patch(module+".initialize_rng_capture_runtime", side_effect=RuntimeError("CUDA initialization failed")):
                with self.assertRaisesRegex(RuntimeError, "CUDA initialization failed"):
                    run(args, rt)
            self.assertEqual(json.loads((args.output_dir/"status.json").read_text())["status"], "failed")
            self.assertTrue((args.output_dir/"registration.json").exists())
            self.assertEqual(events.count("make_env"), 2)
            args.output_dir = base/"failed_environment_creation"
            before_gc, before_callbacks = gc.isenabled(), list(gc.callbacks)
            def fail_environment(*_):
                self.assertFalse(gc.isenabled())
                raise RuntimeError("environment creation failed")
            with patch(module+"._validate_live_provenance"), \
                 patch(module+".initialize_rng_capture_runtime", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "environment creation failed"):
                    run(args, {**rt, "make_env": fail_environment})
            self.assertEqual(gc.isenabled(), before_gc)
            self.assertEqual(gc.callbacks, before_callbacks)
            self.assertTrue(json.loads((args.output_dir/"gc_attempt_0.json").read_text())["original_enabled_state_restored"])
            self.assertEqual(json.loads((args.output_dir/"status.json").read_text())["status"], "failed")
            reg["code_hashes"] = {}
            (source/"registration.json").write_text(json.dumps(reg))
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(args, rt)


if __name__ == "__main__":
    unittest.main()
