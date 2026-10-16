"""Bounded exclusions, common-entry restoration, pairing and interruption recovery."""

from contextlib import redirect_stdout
from copy import deepcopy
import errno
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()
from tests.exact_replay.test_intervention_timing import source_fixture, runtime as timing_runtime
from tests.exact_replay.test_operator_screen import make_policy, XiaoEnvironment
from tests.exact_replay.test_intervention_timing_fresh_references import SampledEnvironment
from robocasa.recovery.exact_replay import collect_intervention_timing_references as fresh
from robocasa.recovery.exact_replay.intervention_timing import build_registration as original_registration
from robocasa.recovery.full_snapshot import capture_full_snapshot, stable_digest
from robocasa.recovery.exact_replay.phase3b_training_horizon import read_jsonl, sha256_file
from robocasa.recovery.exact_replay.intervention_timing_pilot import build_registration, assignment, parent_records, validate_registration
from robocasa.recovery.exact_replay import run_intervention_timing_pilot as pilot


MODULE = "robocasa.recovery.exact_replay.run_intervention_timing_pilot"


class CachedEnvironment(SampledEnvironment):
    def __init__(self):
        super().__init__()
        self.env = self
        self.task_name = "ArrangeTea"
        self.bad_live_reset = False
        self.was_restored = False

    def reset(self, **kwargs):
        self.was_restored = False
        return super().reset(**kwargs)

    def reset_to(self, state):
        super().reset_to(state)
        self.was_restored = True

    def get_current_observation(self):
        observation = super().get_current_observation()
        if self.bad_live_reset and self.was_restored:
            observation["video.robot0_agentview_left"][0, 0, 0] = 99
        return observation

    def _get_observations(self, force_update=False):
        if force_update:
            raise AssertionError("Entry reconstruction must not advance the sampler")
        observation = XiaoEnvironment.get_current_observation(self)
        observation["state.end_effector_position_relative"][0] = self._observables["proprio"]._current_observed_value[0]
        return observation

    def get_observation(self, raw):
        return deepcopy(raw)


class TimingPilotTest(unittest.TestCase):
    def setUp(self):
        for target, kwargs in (
            ("robocasa.recovery.exact_replay.intervention_timing.validate_operator_registration", {}),
            ("robocasa.recovery.exact_replay.collect_intervention_timing_references._validate_live_provenance", {}),
            (MODULE+"._validate_live_provenance", {}),
            (MODULE+".validate_retreat_controller", {"return_value": {"accepted": True}}),
        ):
            item = patch(target, **kwargs)
            item.start()
            self.addCleanup(item.stop)

    def fixture(self, base):
        source = source_fixture(base)
        (base/"phase2/plan.json").write_text(json.dumps({"policy_module": "fake:factory", "parent_horizon": 8}))
        timing = base/"timing"
        timing.mkdir()
        (timing/"registration.json").write_text(json.dumps(original_registration(source)))
        args = pilot.build_parser().parse_args(["--timing-run-dir", str(timing), "--output-dir", str(base/"pilot"),
                                               "--model-path", "/checkpoint"])
        rt, envs, policies = timing_runtime(done_at=6), [], []
        flags = {"bad_cutting": False, "cutting_at_reset": False}
        def make_env(task, *_):
            env = CachedEnvironment()
            env.task_name = task
            env.bad_live_reset = flags["bad_cutting"] and task == "CuttingToolSelection"
            envs.append(env)
            return env
        def make(*_):
            policy = make_policy()
            policies.append(policy)
            return policy
        def evaluate(env):
            return {"task_success": False, "required_predicates": ["upstream", "mug_on_tray"],
                "predicates": {"upstream": {"value": (flags["cutting_at_reset"] and env.task_name == "CuttingToolSelection") or env.timestep >= 3},
                               "mug_on_tray": {"value": False}}}
        rt.update(make_env=make_env, call_factory=make, load_factory=lambda _: None, get_subtask_eval=evaluate)
        return args, rt, envs, policies, flags

    def accept_images(self, args):
        assessment = json.loads((args.output_dir/"image_review_template.json").read_text())
        for row in assessment["parents"]:
            row.update(images_acceptable=True, reviewed_all_frames=True)
        args.review_file = args.output_dir.parent/"assessment.json"
        args.review_file.write_text(json.dumps(assessment))
        return pilot.review(args)

    def test_full_grid_common_endpoints_pairing_and_negative_robot_outcomes_are_kept(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, envs, policies, _ = self.fixture(Path(directory))
            rt["step_fn"] = timing_runtime(done_at=12)["step_fn"]
            result = pilot.run(args, rt)
            self.assertTrue(result["engineering_all_pass"], json.loads((args.output_dir/"analysis.json").read_text()))
            self.assertEqual(result["eligible_parents"], 10)
            self.assertEqual(result["expected_branches"], 228)
            self.assertFalse(result["pilot_scientific_analysis_ready"])
            self.assertFalse((args.output_dir/"scientific_analysis.json").exists())
            rows = read_jsonl(args.output_dir/"branch_records.jsonl")
            self.assertEqual(len(rows), 228)
            self.assertEqual(sum(r["role"] == "primary" for r in rows), 200)
            self.assertEqual({r["horizon_environment_steps"] for r in rows}, {480, 256})
            self.assertTrue(all(r["entry_environment_step"] == 4 and r["num_steps"] == 8 for r in rows))
            self.assertTrue(any(r["delay"] == 2 and r["intervention"]["executed"] for r in rows))
            self.assertTrue(all(not r["task_success"] for r in rows))
            self.assertTrue(all(e.closed for e in envs))
            self.assertTrue(all(p.client.client.closed for p in policies))
            science = self.accept_images(args)
            self.assertTrue(science["pilot_scientific_analysis_ready"])
            self.assertEqual(science["primary_branches"], 200)
            self.assertEqual(science["parents_by_task"], {"ArrangeTea": 5, "CuttingToolSelection": 5})
            self.assertEqual(science["macro"]["continue"]["completion"], 0)
            self.assertEqual(pilot.review(args), science)
            args.resume = True
            (args.output_dir/"status.json").write_text(json.dumps({"status": "interrupted"}))
            self.assertEqual(pilot.run(args, rt)["status"], "complete")
            self.assertEqual(json.loads((args.output_dir/"status.json").read_text())["status"], "complete")
            self.assertEqual(len(read_jsonl(args.output_dir/"branch_records.jsonl")), 228)

    def test_reference_exclusions_continue_fixed_pool_promote_sentinel_and_preserve_seed_indices(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            original = fresh.collect_parent
            skipped = []
            def collect(parent, *a):
                if parent["index"] in (1, 5):
                    skipped.append(parent["parent_id"])
                    raise ValueError("New reset-state live observation differs; synthetic rejection")
                return original(parent, *a)
            with patch.object(fresh, "collect_parent", side_effect=collect):
                result = pilot.run(args, rt)
            self.assertTrue(result["engineering_all_pass"])
            self.assertEqual(result["excluded_parents"], 2)
            self.assertEqual(result["expected_branches"], 186)
            reg = validate_registration(args.output_dir)
            assigned = json.loads((args.output_dir/"assignment.json").read_text())
            parents = {p["parent_id"]: p for p in reg["parents"]}
            self.assertEqual(parents[assigned["sentinels_by_task"]["CuttingToolSelection"]]["index"], 6)
            records = read_jsonl(args.output_dir/"branch_records.jsonl")
            self.assertFalse(any(r["parent_id"] in skipped for r in records))
            for row in records:
                expected = reg["spec"]["continuation_seed_bases"][row["schedule"]] + parents[row["parent_id"]]["index"]*1000
                self.assertEqual(row["sampling_seed"], expected)
            self.assertEqual(len(read_jsonl(args.output_dir/"excluded_parents.jsonl")), 2)
            self.assertTrue(self.accept_images(args)["pilot_scientific_analysis_ready"])

    def test_entry_restores_sampled_values_without_forcing_and_rejects_an_impure_getter(self):
        with tempfile.TemporaryDirectory() as directory:
            env, policy = CachedEnvironment(), make_policy()
            obs, _ = env.reset(seed=1)
            evaluation = {"required_predicates": ["mug_on_tray"], "predicates": {"mug_on_tray": {"value": False}}, "task_success": False}
            parent = {"environment_seed": 1, "environment_reset_index": 0, "trigger_stage": "ArrangeTea::mug_on_tray",
                      "legacy_parent_id": "old", "legacy_snapshot_id": "old-snapshot"}
            snapshot = capture_full_snapshot(env, policy, parent_id="new", task_name="ArrangeTea", trigger_name="entry", environment_step=0,
                observation=obs, subtask_eval=evaluation, metadata=fresh.metadata(parent, env))
            rt = {"get_subtask_eval": lambda _: evaluation}
            expected = stable_digest(snapshot.metadata["observation_runtime_state"])
            restored, _, audit = pilot.restore_entry(snapshot, env, policy, rt, Path(directory))
            self.assertTrue(audit["valid"])
            self.assertEqual(stable_digest(restored), stable_digest(obs))
            self.assertEqual(stable_digest(fresh.capture_observation_runtime(env)), expected)
            env._get_observations = lambda force_update=False: env.get_current_observation()
            with self.assertRaisesRegex(pilot.InvalidParent, "sampled observation"):
                pilot.restore_entry(snapshot, env, policy, rt, Path(directory))
            self.assertFalse(json.loads((Path(directory)/"restore_audit.json").read_text())["valid"])
            env.close(); policy.close()

    def test_control_mismatch_excludes_parent_before_any_recovery_treatment(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            original, rejected = pilot.execute_branch, []
            def execute(snapshot, env, policy, branch, *a, **kw):
                result = original(snapshot, env, policy, branch, *a, **kw)
                if branch["operator"] == "continue" and branch["repeat"] == 1 and branch["task_name"] == "CuttingToolSelection" and not rejected:
                    rejected.append(branch["parent_id"])
                    result["summary"]["suffix_observation_runtime_sha256"][-1] = "different-phase"
                if rejected and branch["parent_id"] == rejected[0]:
                    self.assertEqual(branch["operator"], "continue")
                return result
            with patch.object(pilot, "execute_branch", side_effect=execute):
                result = pilot.run(args, rt)
            self.assertEqual(result["eligible_parents"], 9)
            self.assertEqual(result["prequalification_control_records"], 2)
            self.assertTrue(result["engineering_all_pass"])
            exclusion = read_jsonl(args.output_dir/"excluded_parents.jsonl")[0]
            self.assertEqual(exclusion["reason"], "continue_repetition_mismatch")
            self.assertEqual(exclusion["phase"], "control_preflight_before_intervention")

    def test_late_recovery_error_is_reported_by_operator_continues_and_blocks_inference(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            original, failed = pilot.execute_branch, []
            def execute(snapshot, env, policy, branch, *a, **kw):
                result = original(snapshot, env, policy, branch, *a, **kw)
                if branch["operator"] == "history_reset_replan" and not failed:
                    failed.append(branch["branch_id"])
                    raise RuntimeError("synthetic error after recovery steps")
                return result
            with patch.object(pilot, "execute_branch", side_effect=execute):
                result = pilot.run(args, rt)
            self.assertTrue(result["timing_collection_complete"])
            self.assertEqual(result["excluded_parents"], 0)
            self.assertEqual(result["timing_errors_by_operator"], {"history_reset_replan": 1})
            self.assertEqual(result["successful_assigned_branches"], 227)
            self.assertEqual(len(read_jsonl(args.output_dir/"timing_errors.jsonl")), 1)
            science = self.accept_images(args)
            self.assertFalse(science["pilot_scientific_analysis_ready"])
            self.assertEqual(science["decision"], "engineering_errors_require_attention")
            self.assertEqual(len(science["timing_invalid_parents"]), 1)
            self.assertNotIn("bootstrap", science)
            args.resume = True
            before = len(read_jsonl(args.output_dir/"branch_records.jsonl"))
            with patch.object(pilot, "run_one", side_effect=AssertionError("Terminal errors must not be retried")):
                pilot.run(args, rt)
            self.assertEqual(len(read_jsonl(args.output_dir/"branch_records.jsonl")), before)

    def test_interrupted_control_collection_resumes_without_duplicate_successes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            original, count = pilot.run_one, []
            def interrupt(*a, **kw):
                result = original(*a, **kw)
                count.append(result["branch_id"])
                raise KeyboardInterrupt("after first saved control")
            with patch.object(pilot, "run_one", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    pilot.run(args, rt)
            self.assertEqual(json.loads((args.output_dir/"status.json").read_text())["status"], "interrupted")
            self.assertFalse((args.output_dir/"fatal_errors.jsonl").exists())
            args.resume = True
            result = pilot.run(args, rt)
            rows = read_jsonl(args.output_dir/"branch_records.jsonl")
            self.assertTrue(result["engineering_all_pass"])
            self.assertEqual(len(rows), len({r["branch_id"] for r in rows}))
            self.assertEqual(sum(r["branch_id"] == count[0] for r in rows), 1)

    def test_reuse_v4_pass_and_pre_inference_failure_preserves_old_run(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            base = Path(directory)
            args, rt, _, _, flags = self.fixture(base)
            flags.update(bad_cutting=True, cutting_at_reset=True)
            old = deepcopy(args)
            old.output_dir = base/"old-v4"
            old.port = 10331
            with self.assertRaisesRegex(ValueError, "live observation differs"):
                fresh.run(old, rt)
            preserved = {str(p): sha256_file(p) for p in old.output_dir.rglob("*") if p.is_file()}
            args.reference_run_dir = old.output_dir
            reg, plan = build_registration(args)
            self.assertEqual(CounterKinds(reg["reuse_manifest"]), {"reference_reused": 1, "reference_excluded": 1})
            reg.pop("registration_sha256")
            reg["early_rng_runtime_initialization"] = fresh.initialize_rng_capture_runtime()
            reg["registration_sha256"] = fresh.sha256_json(reg)
            args.output_dir.mkdir()
            flags["bad_cutting"] = False
            original, called = fresh.collect_parent, []
            def collect(parent, *a):
                called.append(parent["parent_id"])
                return original(parent, *a)
            with patch.object(fresh, "collect_parent", side_effect=collect):
                rows = pilot.collect_references(args.output_dir, reg, plan, args, rt, None)
            self.assertEqual(len(called), 8)
            self.assertEqual(sum(r["kind"] == "excluded" for r in rows.values()), 1)
            self.assertEqual(sum(r.get("reused", False) for r in rows.values()), 2)
            self.assertEqual(preserved, {path: sha256_file(path) for path in preserved})

    def test_explicit_reset_collection_is_setup_but_suffix_collection_is_invalid(self):
        class GCReset(CachedEnvironment):
            def reset_to(self, state):
                super().reset_to(state)
                gc.collect()
        for active_collection in (False, True):
            with self.subTest(active_collection=active_collection), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                args, rt, _, _, _ = self.fixture(Path(directory))
                reg, plan = build_registration(args)
                reg["early_rng_runtime_initialization"] = fresh.initialize_rng_capture_runtime()
                parent, branch = reg["parents"][0], reg["branches"][0]
                env, policy = GCReset(), make_policy()
                observation, _ = env.reset(seed=1)
                snapshot = capture_full_snapshot(env, policy, parent_id=parent["parent_id"], task_name=parent["task_name"],
                    trigger_name="entry", environment_step=0, observation=observation, subtask_eval=rt["get_subtask_eval"](env),
                    metadata=fresh.metadata(parent, env))
                rt["make_env"] = lambda *_: GCReset()
                if active_collection:
                    step = rt["step_fn"]
                    def collect(*a):
                        result = step(*a)
                        gc.collect()
                        return result
                    rt["step_fn"] = collect
                    with self.assertRaisesRegex(pilot.InvalidParent, "Cyclic collection"):
                        pilot.run_one(args.output_dir, reg, parent, snapshot, branch, plan, args, rt, None)
                else:
                    row = pilot.run_one(args.output_dir, reg, parent, snapshot, branch, plan, args, rt, None)
                    trace = row["gc_guard"]
                    self.assertEqual(trace["nominal_collection_stops"], 0)
                    setup = [e for e in trace["events"] if e["event"] == "stop" and e["phase"] == "restore_setup"]
                    self.assertEqual(len(setup), 1)
                    self.assertFalse(setup[0]["nominal"])
                env.close(); policy.close()

    def test_storage_failures_are_fatal_not_sample_exclusions_and_dry_run_has_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            args.dry_run = True
            with patch.object(pilot, "initialize_rng_capture_runtime", side_effect=AssertionError("no CUDA in dry run")):
                reg = pilot.run(args, rt)
            self.assertEqual(len(reg["parents"]), 10)
            self.assertFalse(args.output_dir.exists())
            args.dry_run = False
            with patch.object(fresh, "collect_parent", side_effect=OSError(errno.EDQUOT, "Disk quota exceeded")):
                with self.assertRaises(OSError):
                    pilot.run(args, rt)
            self.assertFalse((args.output_dir/"excluded_parents.jsonl").exists())
            self.assertEqual(len(read_jsonl(args.output_dir/"fatal_errors.jsonl")), 1)
            args.resume = True
            with self.assertRaisesRegex(ValueError, "Fatal infrastructure"):
                pilot.run(args, rt)

    def test_missing_task_coverage_is_not_presented_as_the_two_task_result(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            original = fresh.collect_parent
            def collect(parent, *a):
                if parent["task_name"] == "CuttingToolSelection":
                    raise ValueError("New reset-state live observation differs")
                return original(parent, *a)
            with patch.object(fresh, "collect_parent", side_effect=collect):
                result = pilot.run(args, rt)
            self.assertTrue(result["engineering_all_pass"])
            self.assertEqual(result["coverage_by_task"]["CuttingToolSelection"]["eligible"], 0)
            assessment = json.loads((args.output_dir/"image_review_template.json").read_text())
            for row in assessment["parents"]:
                row.update(images_acceptable=True, reviewed_all_frames=True)
            assessment["parents"][0]["images_acceptable"] = False
            args.review_file = Path(directory)/"review.json"
            args.review_file.write_text(json.dumps(assessment))
            science = pilot.review(args)
            self.assertEqual(science["parents_by_task"], {"ArrangeTea": 4})
            self.assertEqual(json.loads((args.output_dir/"report.json").read_text())["analysis_parents_by_task"], {"ArrangeTea": 4, "CuttingToolSelection": 0})
            self.assertEqual(len(science["image_excluded_parents"]), 1)
            self.assertEqual(science["decision"], "insufficient_valid_task_coverage")
            self.assertFalse(science["pilot_scientific_analysis_ready"])
            self.assertNotIn("bootstrap", science)
            assessment["parents"][0]["images_acceptable"] = True
            args.review_file.write_text(json.dumps(assessment))
            with self.assertRaisesRegex(FileExistsError, "already recorded"):
                pilot.review(args)

    def test_no_surviving_parents_is_not_an_engineering_or_scientific_pass(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            with patch.object(fresh, "collect_parent", side_effect=ValueError("New reset-state live observation differs")):
                result = pilot.run(args, rt)
            self.assertEqual(result["status"], "no_eligible_parents")
            self.assertTrue(result["timing_collection_complete"])
            self.assertFalse(result["engineering_all_pass"])
            self.assertEqual(result["excluded_parents"], 10)
            self.assertFalse((args.output_dir/"branch_records.jsonl").exists())

    def test_interrupted_error_commit_recovers_marker_and_never_retries_terminal_branch(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            original, append, failed, calls = pilot.execute_branch, pilot.append_jsonl, [], []
            def execute(snapshot, env, policy, branch, *a, **kw):
                calls.append(branch["branch_id"])
                result = original(snapshot, env, policy, branch, *a, **kw)
                if branch["operator"] == "stage_retry" and not failed:
                    failed.append(branch["branch_id"])
                    raise RuntimeError("synthetic recovery failure")
                return result
            def interrupt(path, row):
                if path.name == "timing_errors.jsonl":
                    raise KeyboardInterrupt("after error marker before error ledger")
                return append(path, row)
            with patch.object(pilot, "execute_branch", side_effect=execute):
                with patch.object(pilot, "append_jsonl", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        pilot.run(args, rt)
                self.assertFalse((args.output_dir/"timing_errors.jsonl").exists())
                self.assertEqual(len(list((args.output_dir/"timing_errors").glob("*.json"))), 1)
                args.resume = True
                result = pilot.run(args, rt)
            self.assertEqual(calls.count(failed[0]), 1)
            self.assertEqual(result["timing_errors"], 1)
            self.assertEqual(len(read_jsonl(args.output_dir/"timing_errors.jsonl")), 1)

    def test_encoded_artifacts_and_assignment_remain_bound_on_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args, rt, _, _, _ = self.fixture(Path(directory))
            pilot.run(args, rt)
            reg = validate_registration(args.output_dir)
            records = pilot.checked_branches(args.output_dir, reg)
            first = next(iter(records.values()))
            encoded = next(name for name in first["branch_artifact_hashes"] if "requests/request_" in name)
            path = args.output_dir/encoded
            original = path.read_bytes()
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                pilot.checked_branches(args.output_dir, reg)
            path.write_bytes(original)
            assigned = args.output_dir/"assignment.json"
            value = json.loads(assigned.read_text())
            value["eligible_parents"].pop()
            assigned.write_text(json.dumps(value))
            args.resume = True
            with self.assertRaisesRegex(ValueError, "assignment changed"):
                pilot.run(args, rt)

    def test_launcher_gpu_separation_provenance_occupied_port_and_private_cleanup(self):
        for failure in (None, "provenance", "client", "port"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                for name in ("client/bin", "server/bin", "safe/deploy", "run", "bin", "robocasa_logs/eval"):
                    (base/name).mkdir(parents=True)
                client = base/"client/bin/python"
                client.write_text('#!'+sys.executable+'\n'+'''import os, pathlib, shutil, socket, sys
args = sys.argv[1:]
def record(value):
    with pathlib.Path(os.environ["TEST_EVENTS"]).open("a") as stream: stream.write(value+"\\n")
if args[:2] == ["-u", "-m"]:
    assert args[2] == "robocasa.recovery.exact_replay.run_intervention_timing_pilot"
    assert "--reference-run-dir" in args
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
        class Readiness:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def connect_ex(self, _): return 0 if pathlib.Path(os.environ["TEST_PID"]).exists() else 111
        socket.socket = Readiness
    sys.argv = args
    exec(compile(source, "<pilot-launcher-test>", "exec"), {"__name__": "__main__"})
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
                (base/"bin/ss").write_text('#!'+sys.executable+'\nimport os\nprint("LISTEN 0 1 127.0.0.1:"+os.environ["XR1_PILOT_PORT"] if os.environ.get("TEST_FAILURE")=="port" else "State Local")\n')
                (base/"bin/sleep").write_text('#!'+sys.executable+'\nimport time\ntime.sleep(.01)\n')
                for path in (base/"bin").iterdir():
                    path.chmod(0o755)
                script = Path(__file__).resolve().parents[2]/"scripts/exact_replay/run_xr1_timing_pilot.sh"
                env = {**os.environ, "PATH": str(base/"bin")+os.pathsep+os.environ["PATH"],
                    "STORAGE_BS": str(base), "ROBOCASA_REPO": str(script.parents[2]),
                    "XR1_CLIENT_ENV": str(base/"client"), "XR1_SERVER_ENV": str(base/"server"), "XR1_SAFE_REPO": str(base/"safe"),
                    "XR1_PILOT_ROOT": str(base/"run"), "XR1_PILOT_SOURCE": str(base/"source"),
                    "XR1_PILOT_REFERENCES": str(base/"v4"), "XR1_PILOT_MODEL": "/test-checkpoint", "XR1_PILOT_RESUME": "0",
                    "XR1_PILOT_GPU": "0", "XR1_PILOT_SERVER_GPU": "1", "XR1_PILOT_PORT": "20332",
                    "XR1_PILOT_LOG": str(base/"driver.log"), "TEST_EVENTS": str(base/"events"), "TEST_PID": str(base/"server.pid"),
                    "TEST_FAILURE": failure or ""}
                result = subprocess.run(["bash", str(script), "worker"], env=env, capture_output=True, text=True, timeout=30)
                events = (base/"events").read_text().splitlines()
                self.assertEqual(events, ["provenance"] if failure in ("provenance", "port") else ["provenance", "server_gpu=1", "client_gpu=0"], result.stdout+result.stderr)
                self.assertEqual(result.returncode == 0, failure is None, result.stdout+result.stderr)
                if (base/"server.pid").exists():
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int((base/"server.pid").read_text()), 0)
                if failure:
                    self.assertTrue((base/"run/driver_failure.json").exists())


def CounterKinds(manifest):
    from collections import Counter
    return dict(Counter(r["kind"] for r in manifest.values()))


if __name__ == "__main__":
    unittest.main()
