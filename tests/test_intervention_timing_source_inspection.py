"""Offline source inspection validates evidence and never constructs a runtime."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()
from tests.test_intervention_timing import source_fixture, runtime
from tests.test_operator_screen import XiaoEnvironment, make_policy
from tests.test_safe_xiaomi_policy import FakeProcessor
from robocasa.recovery.intervention_timing import build_registration
from robocasa.recovery.diagnose_intervention_timing_replay import build_parser as replay_parser, run as replay
from robocasa.recovery.inspect_intervention_timing_source import (
    build_parser, run, observation_trace_comparison, request_trace_comparison,
)
from robocasa.recovery.phase3b_training_horizon import sha256_file


class SourceInspectionTest(unittest.TestCase):
    def fixture(self, base, comparison_parent=0):
        encode = FakeProcessor.apply_chat_template
        def image_aware(processor, messages, **kwargs):
            result = encode(processor, messages, **kwargs)
            result["test_image"] = np.asarray(messages[0]["content"][1]["video"][0])
            return result
        with patch.object(FakeProcessor, "apply_chat_template", image_aware):
            return self.encoded_fixture(base, comparison_parent)

    def encoded_fixture(self, base, comparison_parent):
        original = XiaoEnvironment.get_current_observation
        def gray(env):
            obs = original(env)
            for key in obs:
                if key.startswith("video."):
                    obs[key][:] = 55
            return obs
        with patch.object(XiaoEnvironment, "get_current_observation", gray):
            source = source_fixture(base)
        timing = base/"timing"
        timing.mkdir()
        registration = build_registration(source)
        (timing/"registration.json").write_text(json.dumps(registration))
        class Colored(XiaoEnvironment):
            def get_current_observation(self):
                obs = super().get_current_observation()
                for key in obs:
                    if key.startswith("video."):
                        obs[key][:] = [210, 20, 30]
                return obs
        rt = runtime()
        rt.update(make_env=lambda *_: Colored(), load_factory=lambda _: None, call_factory=lambda *_: make_policy())
        for name, parent_index in (("isolated", 0), ("cohort", comparison_parent)):
            args = replay_parser().parse_args(["--timing-run-dir", str(timing),
                "--parent-id", registration["parents"][parent_index]["parent_id"],
                "--output-dir", str(base/name), "--model-path", "/checkpoint",
                "--rng-runtime-init", "before_env", "--gc-mode", "disabled", "--record-all-observations"])
            replay(args, rt)
        args = build_parser().parse_args(["--diagnostic-dir", str(base/"isolated"),
            "--comparison-diagnostic-dir", str(base/"cohort"), "--output-dir", str(base/"inspection")])
        protected = {str(p): sha256_file(p) for p in base.rglob("*") if p.is_file()}
        return args, protected

    def test_cpu_only_compares_histories_and_exports_native_images_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch("robocasa.recovery.intervention_timing.validate_operator_registration"), \
             patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"):
            args, protected = self.fixture(Path(directory))
            module = "robocasa.recovery.diagnose_intervention_timing_replay"
            with patch(module+"._runtime", side_effect=AssertionError("No simulator")) as simulator, \
                 patch(module+".initialize_rng_capture_runtime", side_effect=AssertionError("No CUDA warmup")) as warm, \
                 patch("robocasa.recovery.full_snapshot._capture_global_rng_state", side_effect=AssertionError("No RNG inspection")), \
                 patch("socket.socket", side_effect=AssertionError("No server connection")):
                result = run(args)
            simulator.assert_not_called()
            warm.assert_not_called()
            self.assertTrue(result["source_integrity"]["accepted"])
            self.assertEqual(result["raw_observation_counts"], [5, 5])
            self.assertEqual(result["scientific_trajectories_collected"], 0)
            self.assertFalse(result["frozen_eligibility_changed"])
            self.assertFalse(result["source_endpoint_audits"][0]["all_exact"])
            key = "video.robot0_agentview_left"
            stats = result["endpoint_camera_statistics"]
            self.assertEqual(stats["historical source"][key]["mean_rgb"], [55, 55, 55])
            self.assertEqual(stats["historical source"][key]["equal_rgb_pixel_percent"], 100)
            self.assertEqual(stats["replay attempt 0"][key]["mean_rgb"], [210, 20, 30])
            self.assertEqual(stats["replay attempt 0"][key]["equal_rgb_pixel_percent"], 0)
            histories = result["saved_history_statistics"]
            self.assertEqual(histories["historical source"][key][0]["queue_index"], 0)
            self.assertNotEqual(histories["historical source"][key][0]["frame_sha256"],
                                histories["replay attempt 0"][key][0]["frame_sha256"])
            records = result["latest_policy_records"]
            self.assertNotEqual(records["historical source"]["latest_requests"][0]["request_sha256"],
                                records["replay attempt 0"]["latest_requests"][0]["request_sha256"])
            for comparison in result["comparison_with_cohort"]["attempts"]:
                self.assertTrue(comparison["initial_audit"]["all_exact"])
                self.assertTrue(comparison["endpoint_audit"]["all_exact"])
                self.assertIsNone(comparison["first_progress_divergence"])
                self.assertIsNone(comparison["observation_comparison"]["first_observation_difference_step"])
                self.assertIsNone(comparison["request_comparison"]["first_difference"])
                self.assertEqual(comparison["request_comparison"]["same_request_different_response_rows"], [])
            self.assertEqual(len(result["image_artifact_hashes"]), 4)
            for name, expected in result["image_artifact_hashes"].items():
                self.assertEqual(sha256_file(args.output_dir/name), expected)
                with Image.open(args.output_dir/name) as image:
                    self.assertEqual(image.format, "PNG")
            with Image.open(args.output_dir/"endpoint_cameras.png") as image:
                self.assertEqual(image.getpixel((4, 32)), (55, 55, 55))
                self.assertEqual(image.getpixel((4, 82)), (210, 20, 30))
            self.assertEqual(protected, {p: sha256_file(p) for p in protected})
            before = {str(p): sha256_file(p) for p in args.output_dir.rglob("*") if p.is_file()}
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual(before, {str(p): sha256_file(p) for p in args.output_dir.rglob("*") if p.is_file()})

    def test_changed_artifact_is_rejected_before_output_or_unpickling_diagnostic_states(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch("robocasa.recovery.intervention_timing.validate_operator_registration"), \
             patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"):
            args, _ = self.fixture(Path(directory))
            path = args.diagnostic_dir/"attempt_0/endpoint_diagnostic_state.pkl.gz"
            path.write_bytes(path.read_bytes()+b"changed")
            with patch("robocasa.recovery.inspect_intervention_timing_source.load_state") as load:
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    run(args)
                load.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_other_parent_comparison_is_rejected_and_outputs_cannot_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch("robocasa.recovery.intervention_timing.validate_operator_registration"), \
             patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"):
            args, protected = self.fixture(Path(directory), comparison_parent=1)
            with self.assertRaisesRegex(ValueError, "same frozen parent"):
                run(args)
            self.assertFalse(args.output_dir.exists())
            args.comparison_diagnostic_dir = None
            args.output_dir = args.diagnostic_dir/"nested"
            with self.assertRaisesRegex(ValueError, "separate from all frozen inputs"):
                run(args)
            self.assertFalse(args.output_dir.exists())
            self.assertEqual(protected, {p: sha256_file(p) for p in protected})

    def test_missing_observation_support_is_not_reported_as_exact(self):
        row = {"environment_step": 0, "observation_sha256": "a", "observation_key_sha256": {"video.left": "a"}}
        result = observation_trace_comparison([row], [])
        self.assertFalse(result["step_support_exact"])
        changed = {**row, "observation_sha256": "b", "observation_key_sha256": {"video.left": "b"}}
        result = observation_trace_comparison([row], [changed])
        self.assertEqual(result["first_observation_difference_step"], 0)
        self.assertEqual(result["first_difference_by_observation_key"], {"video.left": 0})
        request = {"request_index": 0, "sampling_seed": 1, "request_sha256": "q", "response_sha256": "a"}
        result = request_trace_comparison([request], [{**request, "response_sha256": "b"}])
        self.assertEqual(result["same_request_different_response_rows"], [0])
        self.assertEqual(request_trace_comparison([request], [])["first_difference"]["fields"], ["trace_length"])


if __name__ == "__main__":
    unittest.main()
