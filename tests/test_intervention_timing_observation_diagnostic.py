"""Recorded-action diagnostics distinguish observations from physical divergence."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()

from tests.test_intervention_timing import snapshot, runtime, source_fixture
from tests.test_operator_screen import XiaoEnvironment
from robocasa.recovery.full_snapshot import stable_digest
from robocasa.recovery.intervention_timing import build_registration
from robocasa.recovery.phase3b_training_horizon import sha256_file
from robocasa.recovery.diagnose_intervention_timing_replay import run as run_nominal, build_parser as nominal_parser
from robocasa.recovery.diagnose_intervention_timing_observations import (
    inspect_state, replay_actions, observation_difference, compare_traces,
    build_parser, run,
)


class RenderEnvironment(XiaoEnvironment):
    canonical_camera_observations = True

    def get_canonical_camera_observation(self, observation):
        result = deepcopy(observation)
        result["video.robot0_agentview_left"][0, 0, 0] += 1
        return result


class ObservationDiagnosticTest(unittest.TestCase):
    def test_per_key_differences_include_pixel_count_bbox_and_uint8_error(self):
        a = {"video.left": np.zeros((3, 4, 3), dtype=np.uint8), "state.x": np.array([1.0])}
        b = deepcopy(a)
        b["video.left"][2, 1, 0] = 255
        diff = observation_difference(a, b)
        self.assertEqual(set(diff), {"video.left"})
        self.assertEqual(diff["video.left"]["changed_pixels"], 1)
        self.assertEqual(diff["video.left"]["max_abs_error"], 255)
        self.assertEqual(diff["video.left"]["pixel_bbox_xyxy_inclusive"], [1, 2, 1, 2])
        b["state.x"] = np.array([2.0])
        b["added"] = "value"
        self.assertEqual(set(observation_difference(a, b)), {"video.left", "state.x", "added"})

    def test_identical_fixed_actions_expose_camera_difference_without_changing_environment(self):
        ref = snapshot(step=4)
        rt = runtime(done_at=None)
        env = XiaoEnvironment()
        obs, _ = env.reset(seed=1)
        initial, _ = inspect_state(env, obs, rt["get_subtask_eval"](env), 0)
        actions = [{"action.end_effector_position": np.zeros(3)} for _ in range(4)]
        before = stable_digest(actions)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, left = replay_actions(ref, initial, actions, XiaoEnvironment(), rt, root/"a")
            ordinary = rt["step_fn"]
            def noisy(e, action):
                obs, reward, done, info = ordinary(e, action)
                action["action.end_effector_position"] += 999  # wrappers may mutate their input
                if e.timestep == 3:
                    obs["video.robot0_agentview_left"][0, 0, 0] += 1
                return obs, reward, done, info
            _, right = replay_actions(ref, initial, actions, XiaoEnvironment(), {**rt, "step_fn": noisy}, root/"b")
            comparison = compare_traces(left, right, root/"a", root/"b")
            self.assertEqual(comparison["first_observation_difference_step"], 3)
            self.assertIsNone(comparison["first_non_observation_difference_step"])
            detail = comparison["first_observation_difference"]
            self.assertTrue(detail["state_audit"]["all_exact"])
            self.assertEqual(detail["after_action_index"], 2)
            self.assertEqual(set(detail["changed_observation_keys"]), {"video.robot0_agentview_left"})
        self.assertEqual(stable_digest(actions), before)

    def test_initial_mismatch_fails_closed_and_retains_state(self):
        ref = snapshot(step=4)
        rt = runtime(done_at=None)
        env = XiaoEnvironment()
        obs, _ = env.reset(seed=1)
        initial, _ = inspect_state(env, obs, rt["get_subtask_eval"](env), 0)
        initial["observation"]["state.end_effector_position_relative"][0] = 10
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/"a"
            with self.assertRaisesRegex(ValueError, "Initial environment/observation differs"):
                replay_actions(ref, initial, [], XiaoEnvironment(), rt, folder)
            self.assertTrue((folder/"initial_audit.json").exists())
            self.assertTrue((folder/"state_000000.pkl.gz").exists())

    def test_fixed_action_simulator_difference_is_reported_separately(self):
        ref = snapshot(step=4)
        rt = runtime(done_at=None)
        env = XiaoEnvironment()
        obs, _ = env.reset(seed=1)
        initial, _ = inspect_state(env, obs, rt["get_subtask_eval"](env), 0)
        actions = [{"action.end_effector_position": np.zeros(3)} for _ in range(4)]
        ordinary = rt["step_fn"]
        def changed_simulator(e, action):
            result = ordinary(e, action)
            if e.timestep == 3:
                e.sim.data.qvel += 0.001
            return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, left = replay_actions(ref, initial, actions, XiaoEnvironment(), rt, root/"a")
            _, right = replay_actions(ref, initial, actions, XiaoEnvironment(),
                                      {**rt, "step_fn": changed_simulator}, root/"b")
            comparison = compare_traces(left, right, root/"a", root/"b")
            self.assertEqual(comparison["first_non_observation_difference_step"], 3)
            self.assertEqual(comparison["first_difference_by_state_channel"]["simulator_integration_state"], 3)
            self.assertEqual(comparison["first_observation_difference_step"], 4)

    def test_separate_render_probe_records_changes_and_checks_environment_mutation(self):
        ref = snapshot(step=4)
        rt = runtime(done_at=None)
        env = RenderEnvironment()
        obs, _ = env.reset(seed=1)
        initial, _ = inspect_state(env, obs, rt["get_subtask_eval"](env), 0)
        with tempfile.TemporaryDirectory() as directory:
            report, rows = replay_actions(ref, initial, [], env, rt, Path(directory)/"a", render_count=2)
            probe = report["render_probe"]
            self.assertTrue(probe["performed"])
            self.assertTrue(probe["environment_unchanged"])
            self.assertTrue(probe["renders"][0]["changed_observation_keys"])
            self.assertEqual(len(rows), 1)
        original = env.get_canonical_camera_observation
        def mutating_render(obs):
            env.sim.data.qpos += 1
            return original(obs)
        env.get_canonical_camera_observation = mutating_render
        with tempfile.TemporaryDirectory() as directory:
            report, _ = replay_actions(ref, initial, [], env, rt, Path(directory)/"a", render_count=1)
            self.assertFalse(report["render_probe"]["environment_unchanged"])

    def test_early_termination_and_inspection_rng_changes_are_rejected(self):
        ref = snapshot(step=4)
        env = XiaoEnvironment()
        obs, _ = env.reset(seed=1)
        initial, _ = inspect_state(env, obs, runtime()["get_subtask_eval"](env), 0)
        actions = [{"action.end_effector_position": np.zeros(3)} for _ in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "terminated early"):
                replay_actions(ref, initial, actions, XiaoEnvironment(), runtime(done_at=2), Path(directory)/"a")
        original = env.get_state
        def mutating_getter():
            env.np_random.random()
            return original()
        env.get_state = mutating_getter
        with self.assertRaisesRegex(ValueError, "inspection advances RNG"):
            inspect_state(env, obs, runtime()["get_subtask_eval"](env), 0)

    def test_end_to_end_no_policy_calls_after_original_diagnostic_and_inputs_remain_immutable(self):
        from tests.test_operator_screen import make_policy
        with tempfile.TemporaryDirectory() as directory, patch("robocasa.recovery.intervention_timing.validate_operator_registration"):
            root = Path(directory)
            old = source_fixture(root)
            registration = build_registration(old)
            source = root/"timing"
            source.mkdir()
            (source/"registration.json").write_text(json.dumps(registration))
            rt = runtime(done_at=None)
            rt.update(load_factory=lambda _: None, call_factory=lambda *_: make_policy())
            args = nominal_parser().parse_args([
                "--timing-run-dir", str(source), "--parent-id", registration["parents"][0]["parent_id"],
                "--output-dir", str(root/"nominal"), "--model-path", "/checkpoint"])
            with patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"), redirect_stdout(io.StringIO()):
                run_nominal(args, rt)
            before = {str(p.relative_to(root/"nominal")): sha256_file(p)
                      for p in (root/"nominal").rglob("*") if p.is_file()}
            args = build_parser().parse_args([
                "--source-diagnostic-dir", str(root/"nominal"), "--output-dir", str(root/"observations"),
                "--probe-step", "3", "--dry-run"])
            forbidden = lambda *_: self.fail("No policy, model, or server calls are permitted")
            rt.update(load_factory=forbidden, call_factory=forbidden, call_policy=forbidden)
            environments = []
            def make_env(*_):
                env = RenderEnvironment()
                environments.append(env)
                return env
            rt["make_env"] = make_env
            with redirect_stdout(io.StringIO()):
                run(args, rt)
                self.assertFalse(args.output_dir.exists())
                self.assertEqual(environments, [])
                args.dry_run = False
                result = run(args, rt)
            self.assertEqual(len(environments), 3)
            self.assertTrue(all(env.closed for env in environments))
            self.assertEqual(result["policy_calls"], 0)
            self.assertEqual(result["finding"], "observation_divergence_not_reproduced_in_fixed_action_pair")
            self.assertTrue(result["render_probe"]["environment_unchanged"])
            self.assertTrue(result["full_study_remains_incomplete"])
            self.assertEqual(before, {str(p.relative_to(root/"nominal")): sha256_file(p)
                                     for p in (root/"nominal").rglob("*") if p.is_file()})
            with self.assertRaises(FileExistsError):
                run(args, rt)
            args.output_dir = root/"nominal"/"nested"
            with self.assertRaisesRegex(ValueError, "separate"):
                run(args, rt)
            args.output_dir = source/"nested"
            with self.assertRaisesRegex(ValueError, "separate from all frozen"):
                run(args, rt)
            args.output_dir = root/"another"
            (root/"nominal"/"attempt_0"/"nominal_progress.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(args, rt)


if __name__ == "__main__":
    unittest.main()
