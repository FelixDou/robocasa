"""The guarded all-parent preflight never collects intervention trajectories."""

from contextlib import redirect_stdout
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()

from tests.test_intervention_timing import source_fixture, runtime
from tests.test_operator_screen import make_policy
from robocasa.recovery.intervention_timing import build_registration
from robocasa.recovery.phase3b_training_horizon import sha256_file
from robocasa.recovery.audit_intervention_timing_parents import build_parser, parent_gates, run


class ParentReplayAuditTest(unittest.TestCase):
    def fixture(self, base):
        source = source_fixture(base)
        timing = base/"timing"
        timing.mkdir()
        (timing/"registration.json").write_text(json.dumps(build_registration(source)))
        (timing/"errors.jsonl").write_text('{"error":"preserved old failure"}\n')
        protected = {str(p): sha256_file(p) for name in ("source", "phase2", "timing")
                     for p in (base/name).rglob("*") if p.is_file()}
        args = build_parser().parse_args(["--timing-run-dir", str(timing), "--output-dir", str(base/"audit"),
                                        "--model-path", "/checkpoint", "--record-all-observations"])
        rt, policies, environments = runtime(), [], []
        make_env = rt["make_env"]
        def create(*a):
            env = make_env(*a)
            environments.append(env)
            return env
        def factory(*_):
            policy = make_policy()
            policies.append(policy)
            return policy
        rt.update(make_env=create, load_factory=lambda _: None, call_factory=factory)
        return args, rt, policies, environments, protected

    def test_all_ten_parents_two_fresh_replays_no_source_writes_resume_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("robocasa.recovery.intervention_timing.validate_operator_registration"), \
             patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"), \
             redirect_stdout(io.StringIO()):
            args, rt, policies, environments, protected = self.fixture(Path(directory))
            args.dry_run = True
            with patch("robocasa.recovery.diagnose_intervention_timing_replay._runtime") as load:
                run(args)
                load.assert_not_called()
            self.assertFalse(args.output_dir.exists())
            args.dry_run = False
            before_gc = gc.isenabled()
            result = run(args, rt)
            self.assertTrue(result["engineering_all_pass"])
            self.assertEqual(result["parents"], 10)
            self.assertEqual(result["nominal_replays"], 20)
            self.assertEqual(result["parents_by_task"], {"ArrangeTea": 5, "CuttingToolSelection": 5})
            self.assertEqual(result["scientific_trajectories_collected"], 0)
            self.assertFalse(result["full_timing_study_complete"])
            self.assertFalse((args.output_dir/"branch_records.jsonl").exists())
            self.assertEqual(len(policies), 20)
            self.assertEqual(len({id(p) for p in policies}), 20)
            self.assertTrue(all(p.client.client.closed for p in policies))
            self.assertTrue(all(e.closed for e in environments))
            self.assertEqual(gc.isenabled(), before_gc)
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})
            first = result["parent_reports"][0]
            parent_report = json.loads((args.output_dir/first["report_path"]).read_text())
            for gate in ("first_repeat_divergence",):
                changed = dict(parent_report)
                changed[gate] = {"row": 1}
                self.assertFalse(all(parent_gates(changed, 4).values()))
            # Commit markers recover a ledger append interrupted after success.
            ledger = args.output_dir/"parent_records.jsonl"
            lines = ledger.read_text().splitlines(True)
            ledger.write_text("".join(lines[1:]))
            args.resume = True
            resumed = run(args, rt)
            self.assertEqual(resumed["parents"], 10)
            self.assertEqual(len(policies), 20)
            args.port += 1
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run(args, rt)
            args.port -= 1
            artifact = args.output_dir/Path(first["report_path"]).parent/"attempt_0/initial_diagnostic_state.pkl.gz"
            artifact.write_bytes(artifact.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run(args, rt)
            self.assertEqual(len(policies), 20)

    def test_interruption_keeps_partial_attempt_and_resumes_without_rerunning_completed_parent(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("robocasa.recovery.intervention_timing.validate_operator_registration"), \
             patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"), \
             redirect_stdout(io.StringIO()):
            args, rt, policies, _, protected = self.fixture(Path(directory))
            ordinary, calls = rt["make_env"], []
            def interrupted(*a):
                calls.append(True)
                if len(calls) == 3:
                    raise KeyboardInterrupt("allocation interrupted")
                return ordinary(*a)
            rt["make_env"] = interrupted
            before = gc.isenabled()
            with self.assertRaises(KeyboardInterrupt):
                run(args, rt)
            self.assertEqual(gc.isenabled(), before)
            self.assertEqual(len(policies), 2)
            partial = list(args.output_dir.glob("parents/*/attempt_*/diagnostic/status.json"))
            self.assertEqual(len(partial), 2)
            saved = {str(p): p.read_bytes() for p in partial}
            args.resume = True
            rt["make_env"] = ordinary
            result = run(args, rt)
            self.assertTrue(result["engineering_all_pass"])
            self.assertEqual(len(policies), 20)
            self.assertEqual(saved, {p: Path(p).read_bytes() for p in saved})
            self.assertEqual(len(list(args.output_dir.glob("parents/*/attempt_*"))), 11)
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})

    def test_failure_stops_cohort_preserves_completed_parent_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("robocasa.recovery.intervention_timing.validate_operator_registration"), \
             patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"), \
             redirect_stdout(io.StringIO()):
            args, rt, policies, _, protected = self.fixture(Path(directory))
            ordinary, calls = rt["make_env"], []
            def failed(*a):
                calls.append(True)
                if len(calls) == 3:
                    raise RuntimeError("renderer initialization failed")
                return ordinary(*a)
            rt["make_env"] = failed
            with self.assertRaisesRegex(RuntimeError, "renderer initialization failed"):
                run(args, rt)
            self.assertEqual(len(policies), 2)
            status = json.loads((args.output_dir/"status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["parents"], 1)
            self.assertTrue((args.output_dir/"errors.jsonl").exists())
            args.resume = True
            with self.assertRaisesRegex(ValueError, "Audit errors"):
                run(args, rt)
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})

    def test_repeatable_trajectory_that_misses_source_is_rejected_before_next_parent(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("robocasa.recovery.intervention_timing.validate_operator_registration"), \
             patch("robocasa.recovery.diagnose_intervention_timing_replay._validate_live_provenance"), \
             redirect_stdout(io.StringIO()):
            args, rt, policies, _, protected = self.fixture(Path(directory))
            ordinary = rt["step_fn"]
            def changed(env, action):
                obs, reward, done, info = ordinary(env, action)
                if env.timestep == 1:
                    obs["video.robot0_agentview_left"][0, 0, 0] += 1
                return obs, reward, done, info
            rt["step_fn"] = changed
            with self.assertRaisesRegex(ValueError, "Guarded parent replay audit failed"):
                run(args, rt)
            self.assertEqual(len(policies), 2)
            failed = json.loads(next(args.output_dir.glob("parents/*/failed_parent.json")).read_text())
            self.assertTrue(failed["engineering_gates"]["complete_progress_repeats_exact"])
            self.assertFalse(failed["engineering_gates"]["both_source_prefix_audits_accepted"])
            self.assertFalse((args.output_dir/"parent_records.jsonl").exists())
            self.assertEqual(protected, {path: sha256_file(path) for path in protected})


if __name__ == "__main__":
    unittest.main()
