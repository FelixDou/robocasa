"""New references use real reset states, complete traces, and separate image review."""

from contextlib import redirect_stdout
from copy import deepcopy
import gc
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()
from tests.test_intervention_timing import source_fixture, runtime
from tests.test_operator_screen import XiaoEnvironment, make_policy
from robocasa.recovery.intervention_timing import build_registration
from robocasa.recovery.full_snapshot import (
    load_full_snapshot, stable_digest, _capture_environment_control_state, _restore_environment_control_state,
)
from robocasa.recovery.phase3b_training_horizon import read_jsonl, sha256_file
from robocasa.recovery.diagnose_intervention_timing_observations import load_state
from robocasa.recovery.collect_intervention_timing_references import (
    PROTOCOL, build_parser, run, review, validate_cohort_registration, validated_references, atomic_write_json,
    fresh_state_audit,
)
from robocasa.recovery.intervention_timing_observation_state import (
    capture_observation_runtime, restore_observation_runtime,
)


MODULE = "robocasa.recovery.collect_intervention_timing_references"


class NonForwardingWrapper:
    """Like a Gym checker: reset/step/close, but no observation API forwarding."""
    def __init__(self):
        self.env = XiaoEnvironment()

    @property
    def unwrapped(self):
        return self.env

    @property
    def closed(self):
        return self.env.closed

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class ClockedEnvironment(XiaoEnvironment):
    done = False  # class-declared state must not be mistaken for forwarding

    @property
    def cur_time(self):
        return self._clock

    @cur_time.setter
    def cur_time(self, value):
        self._clock = value

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._clock = 0.0
        return result

    def step(self, action):
        result = super().step(action)
        self._clock += .05
        return result


class ReadForwardingWrapper(NonForwardingWrapper):
    def __init__(self):
        self.env = ClockedEnvironment()

    def __getattr__(self, name):
        return getattr(self.env, name)


class SampledObservable:
    """Faithful sampler logic from the supplied installed Observable.update."""
    def __init__(self, env):
        self._sensor = lambda cache: np.asarray([env.value], dtype=np.float64)
        self._filter = lambda value: value
        self._corrupter = lambda value: value
        self._delayer = lambda: float(np.random.uniform(0.0, 0.0))
        self._enabled, self._sampling_timestep = True, .05
        self._time_since_last_sample, self._current_delay = 0.0, 0.0
        self._sampled, self._current_observed_value = False, np.zeros(1)

    def update(self, timestep, obs_cache, force=False):
        if self._enabled:
            self._time_since_last_sample += timestep
            if (not self._sampled and self._sampling_timestep - self._current_delay >= self._time_since_last_sample) or force:
                self._current_observed_value = np.array(self._filter(self._corrupter(self._sensor(obs_cache))))
                obs_cache["proprio"] = self._current_observed_value.copy()
                self._sampled = True
                self._current_delay = self._delayer()
            if self._time_since_last_sample >= self._sampling_timestep:
                if not self._sampled:
                    self._current_observed_value = np.array(self._filter(self._corrupter(self._sensor(obs_cache))))
                    obs_cache["proprio"] = self._current_observed_value.copy()
                    self._current_delay = self._delayer()
                self._time_since_last_sample %= self._sampling_timestep
                self._sampled = False


class SampledEnvironment(XiaoEnvironment):
    def __init__(self):
        super().__init__()
        self._observables = {"proprio": SampledObservable(self)}
        self._obs_cache = {}
        self._resetting = self._stepping = False

    def reset(self, **kwargs):
        self._observables = {"proprio": SampledObservable(self)}
        self._obs_cache = {}
        self._resetting = True
        try:
            return super().reset(**kwargs)
        finally:
            self._resetting = False

    def get_current_observation(self):
        observation = super().get_current_observation()
        self._observables["proprio"].update(.05 if self._resetting else .01,
                                          self._obs_cache, force=not self._stepping)
        observation["state.end_effector_position_relative"][0] = self._observables["proprio"]._current_observed_value[0]
        return observation

    def step(self, action):
        self._stepping = True
        try:
            return super().step(action)
        finally:
            self._stepping = False


class FreshReferenceTest(unittest.TestCase):
    def fixture(self, base, *, entry_at=3, done_at=None, env_type=None):
        source = source_fixture(base)
        plan = base/"phase2/plan.json"
        plan.write_text(json.dumps({"policy_module": "fake:factory", "parent_horizon": 8}))
        timing = base/"timing"
        timing.mkdir()
        (timing/"registration.json").write_text(json.dumps(build_registration(source)))
        (timing/"errors.jsonl").write_text('{"error":"frozen old failure"}\n')
        args = build_parser().parse_args(["--timing-run-dir", str(timing), "--output-dir", str(base/"new"),
                                         "--model-path", "/checkpoint"])
        policies, envs, restored = [], [], []
        class Fresh(XiaoEnvironment):
            def get_current_observation(self):
                obs = super().get_current_observation()
                for key in obs:
                    if key.startswith("video."):
                        obs[key][:] = [120, 110, 70]
                return obs
        def make_env(*_):
            env = (env_type or Fresh)()
            owner = env
            while hasattr(owner, "env"):
                owner = owner.env
            if not hasattr(owner, "_observables"):
                owner._observables, owner._obs_cache = {}, {}
            envs.append(env)
            return env
        def make(*_):
            policy = make_policy()
            setter = policy.set_state
            def set_state(state):
                # The only restored policy history may be a new actual reset.
                self.assertEqual(state["environment_step"], 0)
                self.assertEqual(state["state_queue"], [])
                self.assertTrue(all(not q for q in state["image_queues"].values()))
                restored.append(deepcopy(state))
                setter(state)
            policy.set_state = set_state
            policies.append(policy)
            return policy
        rt = runtime(done_at=done_at)
        def evaluate(env):
            return {"task_success": False, "required_predicates": ["upstream", "mug_on_tray"],
                "predicates": {"upstream": {"value": entry_at is not None and env.timestep >= entry_at},
                               "mug_on_tray": {"value": False}}}
        rt.update(make_env=make_env, call_factory=make, load_factory=lambda _: None, get_subtask_eval=evaluate)
        protected = {str(p): sha256_file(p) for name in ("source", "phase2", "timing")
                     for p in (base/name).rglob("*") if p.is_file()}
        return args, rt, policies, envs, restored, protected

    def setUp(self):
        self.patches = [patch("robocasa.recovery.intervention_timing.validate_operator_registration"),
                        patch(MODULE+"._validate_live_provenance")]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_ten_new_references_full_prefix_audits_fresh_history_no_source_mutation_and_image_review(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, envs, restored, protected = self.fixture(Path(directory))
            original_gc = gc.isenabled()
            result = run(args, rt)
            self.assertEqual(result["protocol"], PROTOCOL)
            self.assertEqual(result["parents_by_task"], {"ArrangeTea": 5, "CuttingToolSelection": 5})
            self.assertEqual(result["nominal_replays"], 20)
            self.assertEqual(result["status"], "awaiting_image_review")
            with self.assertRaisesRegex(ValueError, "accepted image review"):
                validated_references(args.output_dir)
            self.assertFalse(result["timing_ready"])
            self.assertFalse(result["full_timing_study_complete"])
            self.assertEqual(result["intervention_trajectories_collected"], 0)
            self.assertFalse((args.output_dir/"branch_records.jsonl").exists())
            self.assertEqual(len(policies), 20)
            self.assertTrue(all(p.client.client.closed for p in policies))
            self.assertTrue(all(e.closed for e in envs))
            self.assertEqual(len(restored), 10)
            self.assertEqual(gc.isenabled(), original_gc)
            reg = validate_cohort_registration(args.output_dir)
            for parent in reg["parents"]:
                self.assertNotEqual(parent["parent_id"], parent["legacy_parent_id"])
            rows = read_jsonl(args.output_dir/"reference_records.jsonl")
            self.assertEqual(len(rows), 10)
            row = rows[0]
            self.assertEqual(row["entry_environment_step"], 4)  # first boundary after stage starts at step 3
            entry = load_full_snapshot(args.output_dir/row["snapshot_path"])
            self.assertEqual(entry.metadata["ordered_prefix"], ["upstream"])
            self.assertFalse(entry.metadata["historical_endpoint_reconstructed"])
            np.testing.assert_array_equal(entry.observation["video.robot0_agentview_left"][0, 0], [120, 110, 70])
            folder = (args.output_dir/row["report_path"]).parent
            audit = json.loads((folder/"repeat_audit.json").read_text())
            self.assertTrue(audit["all_state_channels_exact"])
            self.assertEqual(audit["state_counts"], [5, 5])
            self.assertEqual(len(audit["state_audits"]), 5)
            self.assertEqual(len(list((folder/"camera_panels").glob("*.png"))), 5)
            requests = read_jsonl(folder/"attempt_0/requests.jsonl")
            self.assertEqual([r["sampling_seed"] for r in requests], [100, 101])
            payload = load_state(folder/"attempt_0"/requests[0]["payload_path"])
            self.assertEqual(stable_digest(payload["request"]), requests[0]["request_sha256"])
            self.assertEqual(stable_digest(payload["response"]), requests[0]["response_sha256"])
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})
            # A human image assessment is required even with all engineering gates passing.
            args.review_file = args.output_dir/"image_review_template.json"
            with self.assertRaisesRegex(ValueError, "Explicit image assessment"):
                review(args)
            assessment = json.loads(args.review_file.read_text())
            for parent in assessment["parents"]:
                parent.update(images_acceptable=True, reviewed_all_frames=True)
            args.review_file = Path(directory)/"human_review.json"
            args.review_file.write_text(json.dumps(assessment))
            def interrupted(path, value):
                if path.name == "report.json":
                    raise KeyboardInterrupt("interrupted between assessment and report commit")
                return atomic_write_json(path, value)
            with patch(MODULE+".atomic_write_json", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    review(args)
            self.assertTrue((args.output_dir/"image_review.json").exists())
            accepted = review(args)
            self.assertEqual(accepted["status"], "references_ready")
            self.assertTrue(accepted["timing_ready"])
            self.assertEqual(len(validated_references(args.output_dir)[1]), 10)
            self.assertEqual(review(args), accepted)
            assessment["parents"][0]["images_acceptable"] = False
            args.review_file.write_text(json.dumps(assessment))
            with self.assertRaisesRegex(FileExistsError, "already recorded"):
                review(args)
            self.assertFalse((args.output_dir/"branch_records.jsonl").exists())

    def test_failed_reset_restore_stops_without_a_manufactured_repeat(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, envs, _, _ = self.fixture(Path(directory))
            with patch(MODULE+".restore_full_snapshot", return_value={"valid": False}):
                with self.assertRaisesRegex(ValueError, "reset-state restore failed"):
                    run(args, rt)
            self.assertEqual(len(policies), 2)
            self.assertTrue(all(e.closed for e in envs))
            self.assertTrue(all(p.client.closed if not hasattr(p.client, "client") else p.client.client.closed for p in policies))
            self.assertTrue((args.output_dir/"errors.jsonl").exists())
            self.assertFalse((args.output_dir/"reference_records.jsonl").exists())

    def test_nonforwarding_gym_wrapper_uses_live_inner_observation_and_exact_restore(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, envs, _, protected = self.fixture(Path(directory), env_type=NonForwardingWrapper)
            evaluate = rt["get_subtask_eval"]
            rt["get_subtask_eval"] = lambda env: evaluate(env.unwrapped)
            # Reproduce why the old generic helper silently returns its default.
            from robocasa.recovery.recovery_rollout import _get_obs_after_state_change
            probe = NonForwardingWrapper()
            self.assertIsNone(_get_obs_after_state_change(probe))
            probe.close()
            self.assertTrue(run(args, rt)["engineering_all_pass"])
            for path in args.output_dir.glob("parents/*/collection_*/attempt_1/initial_restore_audit.json"):
                audit = json.loads(path.read_text())
                self.assertTrue(audit["valid"])
                self.assertTrue(audit["observation_exact"])
                self.assertEqual(audit["observation_verification"], {
                    "performed": True, "snapshot_fallback_used": False,
                    "getter": "env.env.get_current_observation"})
            self.assertEqual(len(policies), 20)
            self.assertTrue(all(env.closed for env in envs))
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})

    def test_legacy_control_setter_creates_stale_forwarded_clock_copies(self):
        source, restored = ReadForwardingWrapper(), ReadForwardingWrapper()
        source.reset(seed=1); restored.reset(seed=1)
        self.assertNotIn("timestep", vars(restored))
        _restore_environment_control_state(restored, _capture_environment_control_state(source))
        self.assertIn("timestep", vars(restored))
        source.step([0]); restored.step([0])
        self.assertEqual(source.timestep, 1)
        self.assertEqual(restored.timestep, 0)
        self.assertEqual(source.cur_time, .05)
        self.assertEqual(restored.cur_time, 0)
        self.assertEqual(source.env.timestep, restored.env.timestep)
        self.assertEqual(source.env.cur_time, restored.env.cur_time)
        source.close(); restored.close()

    def test_v3_restore_preserves_wrapper_aliases_descriptors_and_real_state_owners(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, envs, _, protected = self.fixture(Path(directory), env_type=ReadForwardingWrapper)
            self.assertTrue(run(args, rt)["engineering_all_pass"])
            self.assertEqual(len(policies), 20)
            for env in envs:
                self.assertNotIn("timestep", vars(env))
                self.assertNotIn("cur_time", vars(env))
                self.assertEqual(env.timestep, 4)
                self.assertEqual(env.cur_time, .2)
                self.assertIsInstance(type(env.env).cur_time, property)
            audits = list(args.output_dir.glob("parents/*/collection_*/attempt_1/initial_restore_audit.json"))
            self.assertEqual(len(audits), 10)
            for path in audits:
                audit = json.loads(path.read_text())
                self.assertTrue(audit["control_attribute_ownership"]["exact"])
                removed = audit["control_attribute_ownership"]["removed_forwarded_shadows"]
                self.assertEqual({r["attribute"] for r in removed}, {"_elapsed_steps", "timestep", "cur_time", "done"})
                self.assertTrue(all(r["object_path"] == "env" for r in removed))
                self.assertTrue(audit["valid"])
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})

    def test_force_read_changes_next_proprioception_without_changing_initial_pixels_or_physics(self):
        source, legacy, fixed = (SampledEnvironment() for _ in range(3))
        observations = [env.reset(seed=1)[0] for env in (source, legacy, fixed)]
        saved = capture_observation_runtime(source)
        self.assertEqual(stable_digest(saved), stable_digest(capture_observation_runtime(legacy)))
        verified = legacy.get_current_observation()
        self.assertEqual(stable_digest(observations[0]), stable_digest(verified))
        self.assertNotEqual(stable_digest(saved), stable_digest(capture_observation_runtime(legacy)))
        fixed.get_current_observation()
        callbacks = tuple(fixed._observables["proprio"].__dict__[k] for k in ("_sensor", "_filter", "_delayer"))
        restore_observation_runtime(fixed, saved)
        self.assertEqual(callbacks, tuple(fixed._observables["proprio"].__dict__[k] for k in ("_sensor", "_filter", "_delayer")))
        next_observations = [env.step([.2])[0] for env in (source, legacy, fixed)]
        self.assertEqual(source.value, legacy.value)
        self.assertEqual(source.value, fixed.value)
        for key in next_observations[0]:
            if key.startswith("video."):
                np.testing.assert_array_equal(next_observations[0][key], next_observations[1][key])
        self.assertNotEqual(stable_digest(next_observations[0]), stable_digest(next_observations[1]))
        self.assertEqual(stable_digest(next_observations[0]), stable_digest(next_observations[2]))
        # Source dependency caches are copies, not aliases to the restored env.
        fixed._obs_cache["proprio"][:] = 100
        self.assertNotEqual(stable_digest(saved["owners"][0]["obs_cache"]), stable_digest(fixed._obs_cache))
        for env in (source, legacy, fixed):
            env.close()

    def test_v4_preserves_sampling_caches_and_rng_through_forced_live_verification(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _, protected = self.fixture(Path(directory), env_type=SampledEnvironment)
            self.assertTrue(run(args, rt)["engineering_all_pass"])
            reg = validate_cohort_registration(args.output_dir)
            self.assertEqual(reg["observation_runtime_contract"]["sampling_fields"], [
                "_time_since_last_sample", "_sampled", "_current_delay", "_current_observed_value"])
            for path in args.output_dir.glob("parents/*/collection_*/repeat_audit.json"):
                audit = json.loads(path.read_text())
                self.assertTrue(audit["all_state_channels_exact"])
                self.assertTrue(audit["all_causal_state_channels_exact"])
                self.assertTrue(all(row["channels"]["observation_runtime_state"] for row in audit["state_audits"]))
                self.assertTrue(all(row["channels"]["global_rng_state"] for row in audit["state_audits"]))
                state = load_state(path.parent/"attempt_0/states/state_000001.pkl.gz")
                self.assertEqual(state["observation_runtime_state"]["owners"][0]["object_path"], "env")
                entry = load_full_snapshot(path.parent/"attempt_0/entry.pkl.gz")
                self.assertIn("observation_runtime_state", entry.metadata)
                restored = json.loads((path.parent/"attempt_1/initial_restore_audit.json").read_text())
                self.assertTrue(restored["observation_runtime_restored_after_verification"])
                witness = restored["observation_runtime_verification"]
                self.assertNotEqual(witness["saved_sha256"], witness["after_forced_read_sha256"])
                self.assertEqual(witness["saved_sha256"], witness["restored_sha256"])
                self.assertTrue(witness["forced_read_mismatches"])
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})

    def test_v4_runtime_and_causal_controller_gate_rejects_all_other_differences(self):
        from robocasa.recovery.diagnose_intervention_timing_replay import diagnostic_state
        env, policy = SampledEnvironment(), make_policy()
        obs, _ = env.reset(seed=1)
        state = diagnostic_state(env, policy, obs, {}, 0)
        state["observation_runtime_state"] = capture_observation_runtime(env)
        path = "robots[0].recent_ee_forcetorques['right']"
        state["controller_state"].append({"object_path": path, "attributes": {"current": np.zeros(1)}})
        other = deepcopy(state)
        other["controller_state"][-1]["attributes"]["current"][0] = 2.168404344971009e-19
        audit = fresh_state_audit(SimpleNamespace(**state), SimpleNamespace(**other))
        self.assertFalse(audit["all_exact"])
        self.assertTrue(audit["acceptance_exact"])
        for changed in ("observation", "sampler", "cache", "causal_controller"):
            with self.subTest(changed=changed):
                bad = deepcopy(other)
                if changed == "observation":
                    bad["observation"]["state.end_effector_position_relative"][0] += 1e-7
                elif changed == "sampler":
                    bad["observation_runtime_state"]["owners"][0]["observables"][0]["sampling_state"]["_sampled"] = True
                elif changed == "cache":
                    bad["observation_runtime_state"]["owners"][0]["obs_cache"]["proprio"][0] += 1e-9
                else:
                    bad["controller_state"][0]["attributes"]["unexpected_causal_field"] = 1e-19
                self.assertFalse(fresh_state_audit(SimpleNamespace(**state), SimpleNamespace(**bad))["acceptance_exact"])
        env.close(); policy.close()

    def test_missing_or_unsupported_observable_runtime_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "No owned RoboSuite"):
            capture_observation_runtime(XiaoEnvironment())
        env = SampledEnvironment()
        env.reset(seed=1)
        saved = capture_observation_runtime(env)
        env._observables["proprio"]._sampling_timestep = .04
        with self.assertRaisesRegex(ValueError, "topology/configuration"):
            restore_observation_runtime(env, saved)
        env._observables["proprio"]._filter = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "Unsupported observation callback"):
            capture_observation_runtime(env)
        history = []
        env._observables["proprio"]._filter = lambda value: history.append(value) or value
        with self.assertRaisesRegex(ValueError, "stateful observation callback closure"):
            capture_observation_runtime(env)
        env.close()

    def test_observable_source_cache_does_not_keep_closed_simulators_alive(self):
        env = SampledEnvironment()
        env.reset(seed=1)
        capture_observation_runtime(env)
        reference = weakref.ref(env)
        env.close()
        del env
        gc.collect()
        self.assertIsNone(reference())

    def test_v4_cohort_keeps_full_state_failure_visible_when_only_declared_buffer_differs(self):
        from robocasa.recovery.diagnose_intervention_timing_replay import diagnostic_state
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, envs, _, _ = self.fixture(Path(directory))
            def capture(*a):
                state = diagnostic_state(*a)
                delta = 2.168404344971009e-19 if len(envs) % 2 == 0 and a[-1] > 0 else 0.0
                state["controller_state"].append({"object_path": "robots[0].recent_ee_forcetorques['right']",
                    "attributes": {"current": np.asarray([delta])}})
                return state
            with patch(MODULE+".diagnostic_state", side_effect=capture):
                result = run(args, rt)
            self.assertTrue(result["engineering_all_pass"])
            self.assertFalse(result["full_state_channels_exact"])
            self.assertEqual(result["status"], "awaiting_image_review")
            for row in read_jsonl(args.output_dir/"reference_records.jsonl"):
                self.assertFalse(row["full_state_channels_exact"])
                folder = (args.output_dir/row["report_path"]).parent
                report = json.loads((folder/"report.json").read_text())
                self.assertFalse(report["full_state_channels_exact"])
                self.assertTrue(report["engineering_gates"]["every_declared_causal_state_repeats_exact"])
                audit = json.loads((folder/"repeat_audit.json").read_text())
                self.assertFalse(audit["all_state_channels_exact"])
                self.assertTrue(audit["all_causal_state_channels_exact"])
                self.assertEqual(audit["first_state_difference"]["environment_step"], 1)
                self.assertIsNone(audit["first_causal_state_difference"])

    def test_wrapped_live_image_mismatch_is_rejected_without_saved_observation_fallback(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, envs, _, _ = self.fixture(Path(directory), env_type=NonForwardingWrapper)
            evaluate = rt["get_subtask_eval"]
            rt["get_subtask_eval"] = lambda env: evaluate(env.unwrapped)
            original = rt["make_env"]
            def create(*a):
                env = original(*a)
                if len(envs) == 2:
                    reset_to, getter = env.env.reset_to, env.env.get_current_observation
                    changed = []
                    def reset(state):
                        result = reset_to(state)
                        changed.append(True)
                        return result
                    def observation():
                        obs = getter()
                        if changed:
                            obs["video.robot0_agentview_left"][0, 0, 0] = 99
                        return obs
                    env.env.reset_to, env.env.get_current_observation = reset, observation
                return env
            rt["make_env"] = create
            with self.assertRaisesRegex(ValueError, "live observation differs"):
                run(args, rt)
            audit = json.loads(next(args.output_dir.glob("parents/*/collection_*/attempt_1/initial_restore_audit.json")).read_text())
            self.assertFalse(audit["valid"])
            self.assertFalse(audit["observation_exact"])
            self.assertTrue(audit["observation_verification"]["performed"])
            self.assertFalse(audit["observation_verification"]["snapshot_fallback_used"])
            self.assertEqual(len(policies), 2)
            self.assertEqual(policies[1].get_state()["environment_step"], 0)
            self.assertFalse((args.output_dir/"reference_records.jsonl").exists())

    def test_missing_live_observation_getter_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, _, _, _ = self.fixture(Path(directory))
            original = rt["make_env"]
            def create(*a):
                env = original(*a)
                original_reset = env.reset
                def reset(**kw):
                    obs = original_reset(**kw)
                    env.get_current_observation = lambda: None
                    return obs
                # Only the second attempt needs an unavailable live getter.
                if len(policies) == 1:
                    env.reset = reset
                return env
            rt["make_env"] = create
            with self.assertRaisesRegex(ValueError, "live observation dictionary"):
                run(args, rt)
            audit = json.loads(next(args.output_dir.glob("parents/*/collection_*/attempt_1/initial_restore_audit.json")).read_text())
            self.assertFalse(audit["valid"])
            self.assertFalse(audit["observation_verification"]["snapshot_fallback_used"])

    def test_unseeded_or_misaligned_nominal_requests_fail_before_certification(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, _, _, _ = self.fixture(Path(directory))
            factory = rt["call_factory"]
            def wrong_seed(*a):
                policy = factory(*a)
                policy.sampling_seed_base += 1
                return policy
            rt["call_factory"] = wrong_seed
            with self.assertRaisesRegex(ValueError, "seed schedule differs"):
                run(args, rt)
            self.assertEqual(len(policies), 1)
            self.assertFalse((args.output_dir/"reference_records.jsonl").exists())

    def test_dry_run_no_runtime_or_output_and_root_guards(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, _, _, _ = self.fixture(Path(directory))
            args.dry_run = True
            with patch(MODULE+"._runtime", side_effect=AssertionError("no runtime")), \
                 patch(MODULE+".initialize_rng_capture_runtime", side_effect=AssertionError("no CUDA")):
                reg = run(args)
            self.assertEqual(reg["nominal_search_budget_environment_steps"], 8)
            self.assertFalse(args.output_dir.exists())
            self.assertFalse(policies)
            args.output_dir = args.timing_run_dir/"replacement"
            with self.assertRaisesRegex(ValueError, "separate from all frozen"):
                run(args, rt)
            args.output_dir = Path("/gs/fs/tga-shinoda/felid/new-output")
            with self.assertRaisesRegex(ValueError, "outside /gs/fs"):
                run(args, rt)

    def test_zero_step_entry_is_real_and_has_no_invented_request(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, _, _, _ = self.fixture(Path(directory), entry_at=0)
            result = run(args, rt)
            self.assertTrue(result["engineering_all_pass"])
            rows = read_jsonl(args.output_dir/"reference_records.jsonl")
            self.assertTrue(all(r["entry_environment_step"] == 0 for r in rows))
            self.assertTrue(all(p.client.calls == 0 for p in policies))

    def test_unreachable_target_and_termination_keep_seed_and_fail_before_next_parent(self):
        for done_at in (None, 2):
            with self.subTest(done_at=done_at), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                args, rt, policies, envs, _, protected = self.fixture(Path(directory), entry_at=None, done_at=done_at)
                with self.assertRaisesRegex(ValueError, "repeat gate failed"):
                    run(args, rt)
                self.assertEqual(len(policies), 2)
                self.assertTrue(all(e.closed for e in envs))
                failed = json.loads(next(args.output_dir.glob("parents/*/failed_parent.json")).read_text())
                self.assertFalse(failed["engineering_gates"]["both_common_entries_captured"])
                self.assertTrue(failed["engineering_gates"]["every_declared_causal_state_repeats_exact"])
                report = json.loads((args.output_dir/failed["report_path"]).read_text())
                self.assertEqual(report["attempts"][0]["environment_steps"], 8 if done_at is None else 2)
                self.assertEqual(protected, {path: sha256_file(path) for path in protected})
                args.resume = True
                with self.assertRaisesRegex(ValueError, "errors recorded"):
                    run(args, rt)
                self.assertEqual(len(policies), 2)

    def test_pixel_divergence_localized_and_not_hidden_by_matching_physics(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, _, _, _ = self.fixture(Path(directory))
            ordinary, calls = rt["make_env"], []
            def create(*a):
                env = ordinary(*a)
                calls.append(env)
                if len(calls) == 2:
                    step = env.step
                    def different(action):
                        obs, reward, done, info = step(action)
                        if env.timestep >= 2:
                            obs["video.robot0_agentview_left"][0, 0, 0] += 1
                        return obs, reward, done, info
                    env.step = different
                return env
            rt["make_env"] = create
            with self.assertRaisesRegex(ValueError, "repeat gate failed"):
                run(args, rt)
            audit = json.loads(next(args.output_dir.glob("parents/*/collection_*/repeat_audit.json")).read_text())
            first = audit["first_state_difference"]
            self.assertEqual(first["environment_step"], 2)
            self.assertFalse(first["audit"]["channels"]["observation"])
            self.assertTrue(first["audit"]["channels"]["environment_state"])
            self.assertEqual(len(policies), 2)

    def test_per_step_xml_compatibility_allowed_but_physics_change_rejected(self):
        class XML(XiaoEnvironment):
            mime = True
            def get_state(self):
                hint = ' content_type="model/obj"' if self.mime else ''
                return {**super().get_state(), "model": f'<mujoco><asset><mesh name="a" file="a.obj"{hint}/></asset></mujoco>'}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _, _ = self.fixture(Path(directory), env_type=XML)
            ordinary, count = rt["make_env"], []
            def create(*a):
                env = ordinary(*a)
                env.mime = len(count) % 2 == 0
                count.append(env)
                return env
            rt["make_env"] = create
            self.assertTrue(run(args, rt)["engineering_all_pass"])
            audit = json.loads(next(args.output_dir.glob("parents/*/collection_*/repeat_audit.json")).read_text())
            self.assertTrue(all(r["model_xml_audit"]["compatible"] for r in audit["state_audits"]))
            self.assertTrue(any(not r["model_xml_audit"]["strict_exact"] for r in audit["state_audits"]))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _, _ = self.fixture(Path(directory))
            ordinary, count = rt["make_env"], []
            def create(*a):
                env = ordinary(*a)
                count.append(env)
                if len(count) == 2:
                    step = env.step
                    def different(action):
                        obs, reward, done, info = step(action)
                        if env.timestep == 2:
                            env.sim.data.qpos[:] += .01
                        return obs, reward, done, info
                    env.step = different
                return env
            rt["make_env"] = create
            with self.assertRaisesRegex(ValueError, "repeat gate failed"):
                run(args, rt)
            audit = json.loads(next(args.output_dir.glob("parents/*/collection_*/repeat_audit.json")).read_text())
            self.assertFalse(audit["first_state_difference"]["audit"]["channels"]["simulator_integration_state"])

    def test_gc_collection_during_rollout_fails_but_reset_collection_is_visible_and_allowed(self):
        for during_rollout in (False, True):
            with self.subTest(during_rollout=during_rollout), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                args, rt, _, _, _, _ = self.fixture(Path(directory))
                if during_rollout:
                    ordinary = rt["step_fn"]
                    def step(*a):
                        gc.collect()
                        return ordinary(*a)
                    rt["step_fn"] = step
                    with self.assertRaisesRegex(ValueError, "repeat gate failed"):
                        run(args, rt)
                else:
                    ordinary = rt["make_env"]
                    def create(*a):
                        env = ordinary(*a)
                        reset = env.reset
                        def collecting(*a, **kw):
                            gc.collect()
                            return reset(*a, **kw)
                        env.reset = collecting
                        return env
                    rt["make_env"] = create
                    self.assertTrue(run(args, rt)["engineering_all_pass"])
                trace = json.loads(next(args.output_dir.glob("parents/*/collection_*/attempt_0/gc_trace.json")).read_text())
                self.assertEqual(trace["nominal_collection_stops"], 4 if during_rollout else 0)
                if not during_rollout:
                    self.assertEqual(trace["setup_collection_stops"], 1)

    def test_interrupt_preserves_attempt_resume_recovers_marker_and_detects_artifact_or_config_changes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, _, _, protected = self.fixture(Path(directory))
            original, count = rt["make_env"], []
            def create(*a):
                count.append(True)
                if len(count) == 3:
                    raise KeyboardInterrupt("allocation interrupted")
                return original(*a)
            rt["make_env"] = create
            before = gc.isenabled()
            with self.assertRaises(KeyboardInterrupt):
                run(args, rt)
            self.assertEqual(gc.isenabled(), before)
            self.assertEqual(len(policies), 2)
            self.assertFalse((args.output_dir/"errors.jsonl").exists())
            partial = next(args.output_dir.glob("parents/*/collection_*/attempt_0/status.json"))
            content = partial.read_bytes()
            # Recover an append interrupted after a parent marker was committed.
            (args.output_dir/"reference_records.jsonl").write_text("")
            rt["make_env"] = original
            args.resume = True
            self.assertTrue(run(args, rt)["engineering_all_pass"])
            self.assertEqual(len(policies), 20)
            self.assertEqual(partial.read_bytes(), content)
            self.assertEqual(len(list(args.output_dir.glob("parents/*/collection_*"))), 11)
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})
            args.port += 1
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run(args, rt)
            args.port -= 1
            artifact = next(args.output_dir.glob("parents/*/collection_*/attempt_1/states/state_000002.pkl.gz"))
            artifact.write_bytes(artifact.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run(args, rt)
            self.assertEqual(len(policies), 20)

    def test_image_rejection_never_certifies_references_or_launches_branches(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, policies, _, _, _ = self.fixture(Path(directory), entry_at=0)
            run(args, rt)
            assessment = json.loads((args.output_dir/"image_review_template.json").read_text())
            for row in assessment["parents"]:
                row.update(images_acceptable=True, reviewed_all_frames=True)
            assessment["parents"][0]["images_acceptable"] = False
            args.review_file = Path(directory)/"review.json"
            args.review_file.write_text(json.dumps(assessment))
            result = review(args)
            self.assertEqual(result["status"], "image_review_rejected")
            self.assertFalse(result["timing_ready"])
            self.assertEqual(len(policies), 20)
            args.resume = True
            self.assertEqual(run(args, rt)["status"], "image_review_rejected")

    def test_launcher_status_handles_pre_registration_failures_and_pending_review(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            executable = base/"client/bin/python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            script = Path(__file__).resolve().parents[1]/"scripts/collect_xr1_timing_references.sh"
            env = {**os.environ, "XR1_CLIENT_ENV": str(executable.parents[1]), "XR1_REFS_ROOT": str(base/"run")}
            process = subprocess.run(["bash", str(script), "status"], capture_output=True, text=True, env=env, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("starting", process.stdout)
            (base/"run").mkdir()
            (base/"run/driver_failure.json").write_text('{"exit_code": 1}')
            process = subprocess.run(["bash", str(script), "status"], capture_output=True, text=True, env=env, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("driver status: failed", process.stdout)

    def test_launcher_private_server_order_gpu_separation_failure_cleanup_and_occupied_port(self):
        # Exercise the actual shell worker with tiny local stand-ins, not a
        # model or simulator. The Python collector itself is tested above.
        for failure in (None, "provenance", "client", "port"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                for path in (base/"client/bin", base/"server/bin", base/"safe/deploy", base/"run", base/"bin", base/"robocasa_logs/eval"):
                    path.mkdir(parents=True)
                client = base/"client/bin/python"
                client.write_text('#!'+sys.executable+'\n'+'''import os, pathlib, shutil, socket, sys
args = sys.argv[1:]
events = pathlib.Path(os.environ["TEST_EVENTS"])
def record(value):
    with events.open("a") as stream: stream.write(value+"\\n")
if args[:2] == ["-u", "-m"]:
    if "--dry-run" in args:
        record("provenance")
        raise SystemExit(7 if os.environ.get("TEST_FAILURE") == "provenance" else 0)
    os.kill(int(pathlib.Path(os.environ["TEST_PID"]).read_text()), 0)
    record("client_gpu="+os.environ["CUDA_VISIBLE_DEVICES"])
    raise SystemExit(8 if os.environ.get("TEST_FAILURE") == "client" else 0)
if args[0] == "-":
    source = sys.stdin.read()
    if "shutil.disk_usage" in source:
        shutil.disk_usage = lambda _: shutil._ntuple_diskusage(200*2**30, 0, 200*2**30)
    if "connection.connect_ex" in source:
        # Keep this subprocess test usable in sandboxes without loopback.
        class Readiness:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def connect_ex(self, _):
                return 0 if pathlib.Path(os.environ["TEST_PID"]).exists() else 111
        socket.socket = Readiness
    sys.argv = args
    exec(compile(source, "<launcher-test>", "exec"), {"__name__": "__main__"})
else:
    os.execv(sys.executable, [sys.executable, *args])
''')
                client.chmod(0o755)
                (base/"server/bin/python").symlink_to(sys.executable)
                (base/"safe/deploy/server.py").write_text('''import os, pathlib, time
with pathlib.Path(os.environ["TEST_EVENTS"]).open("a") as stream:
    stream.write("server_gpu="+os.environ["CUDA_VISIBLE_DEVICES"]+"\\n")
print("Model loaded", flush=True)
pathlib.Path(os.environ["TEST_PID"]).write_text(str(os.getpid()))
while True: time.sleep(.1)
''')
                (base/"bin/flock").write_text('#!'+sys.executable+'\nimport fcntl\nfcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)\n')
                (base/"bin/ss").write_text('#!'+sys.executable+'\nimport os\nprint("LISTEN 0 1 127.0.0.1:"+os.environ["XR1_REFS_PORT"] if os.environ.get("TEST_FAILURE")=="port" else "State Recv-Q Send-Q Local")\n')
                (base/"bin/sleep").write_text('#!'+sys.executable+'\nimport time\ntime.sleep(.01)\n')
                for path in (base/"bin").iterdir():
                    path.chmod(0o755)
                port = 20327  # readiness is stubbed; no actual network binding
                script = Path(__file__).resolve().parents[1]/"scripts/collect_xr1_timing_references.sh"
                env = {**os.environ, "PATH": str(base/"bin")+os.pathsep+os.environ["PATH"],
                    "STORAGE_BS": str(base), "ROBOCASA_REPO": str(script.parents[1]),
                    "XR1_CLIENT_ENV": str(base/"client"), "XR1_SERVER_ENV": str(base/"server"),
                    "XR1_SAFE_REPO": str(base/"safe"), "XR1_REFS_MODEL": "/test-checkpoint",
                    "XR1_REFS_ROOT": str(base/"run"), "XR1_REFS_SOURCE": str(base/"source"),
                    "XR1_REFS_RESUME": "0", "XR1_REFS_GPU": "0", "XR1_REFS_SERVER_GPU": "1",
                    "XR1_REFS_PORT": str(port), "XR1_REFS_LOG": str(base/"driver.log"),
                    "TEST_EVENTS": str(base/"events"), "TEST_PID": str(base/"server.pid"),
                    "TEST_FAILURE": failure or ""}
                process = subprocess.run(["bash", str(script), "worker"], env=env, capture_output=True, text=True, timeout=30)
                events = (base/"events").read_text().splitlines()
                expected = ["provenance"] if failure in ("provenance", "port") else ["provenance", "server_gpu=1", "client_gpu=0"]
                self.assertEqual(events, expected, process.stdout+process.stderr)
                self.assertEqual(process.returncode == 0, failure is None, process.stdout+process.stderr)
                if (base/"server.pid").exists():
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int((base/"server.pid").read_text()), 0)
                if failure:
                    self.assertTrue((base/"run/driver_failure.json").exists())
                    self.assertEqual(len(list((base/"run/driver_failures").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
