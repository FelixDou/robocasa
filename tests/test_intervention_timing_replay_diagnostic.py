"""Replay diagnostics preserve failed endpoints and never modify frozen evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()

from tests.test_intervention_timing import snapshot, runtime, source_fixture, LazyBookkeepingEnvironment
from tests.test_operator_screen import make_policy, XiaoEnvironment
from tests.test_safe_xiaomi_policy import FakeClient, safe_response
from robocasa.recovery.full_snapshot import stable_digest
from robocasa.recovery.intervention_timing import build_registration
from robocasa.recovery.phase3b_training_horizon import sha256_file
from robocasa.recovery.diagnose_intervention_timing_replay import (
    RequestProbe, diagnose_attempt, diagnostic_state, first_divergence, run, build_parser,
    initialize_rng_capture_runtime,
)


class ReplayDiagnosticTest(unittest.TestCase):
    def test_runtime_initialization_uses_rng_inspection_without_early_source_restore(self):
        with patch("robocasa.recovery.diagnose_intervention_timing_replay._capture_global_rng_state",
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
            report, _, endpoint, _ = diagnose_attempt(ref, XiaoEnvironment(), make_policy(), runtime(), Path(directory)/"attempt")
            self.assertFalse(report["final_at_inference_boundary"])
            self.assertFalse(endpoint["policy_state"]["at_inference_boundary"])
            self.assertTrue(endpoint["policy_state"]["action_plan"])
            self.assertFalse(report["prefix_replay_accepted"])

    def test_first_divergence_reports_channel_and_length(self):
        a = [{"step": 0, "action_sha256": "a"}, {"step": 1, "action_sha256": "b"}]
        self.assertIsNone(first_divergence(a, deepcopy(a)))
        b = deepcopy(a)
        b[1]["action_sha256"] = "changed"
        result = first_divergence(a, b)
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["fields"], ["action_sha256"])
        self.assertEqual(first_divergence(a, a[:1])["fields"], ["trace_length"])

    def test_two_fresh_policy_replays_preserve_failed_source_and_dry_run_has_no_writes(self):
        with tempfile.TemporaryDirectory() as directory, patch("robocasa.recovery.intervention_timing.validate_operator_registration"):
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
            with patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"), redirect_stdout(io.StringIO()):
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
            events = []
            make_env = rt["make_env"]
            def early():
                events.append("initialize_rng_runtime")
                return {"torch_available": False, "source_rng_restored_early": False}
            def create(*a):
                events.append("make_env")
                return make_env(*a)
            rt.update(make_env=create, load_factory=lambda _: events.append("load_factory"))
            module = "robocasa.recovery.diagnose_intervention_timing_replay"
            with patch(module+"._validate_live_provenance"), \
                 patch(module+".initialize_rng_capture_runtime", side_effect=early) as warm, \
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
            self.assertTrue(all(a["prefix_replay_accepted"] for a in initialized["attempts"]))
            self.assertTrue(initialized["repeated_initial_state_audit"]["all_exact"])
            self.assertTrue(initialized["repeated_endpoint_audit"]["all_exact"])
            self.assertIsNone(initialized["first_repeat_divergence"])
            for index in range(2):
                name = f"attempt_{index}/requests.jsonl"
                self.assertEqual((control_root/name).read_text(), (args.output_dir/name).read_text())
            self.assertEqual(before, {p.name: sha256_file(p) for p in source.iterdir()})
            self.assertEqual(len(policies), 4)
            self.assertTrue(all(p.client.client.closed for p in policies))
            # Both a reused output root and an overlapping source/output root fail closed.
            with patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"):
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
            reg["code_hashes"] = {}
            (source/"registration.json").write_text(json.dumps(reg))
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(args, rt)


if __name__ == "__main__":
    unittest.main()
