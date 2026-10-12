"""Simulator-free paired timing, replay, integrity and collection tests."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()

from tests.test_operator_screen import make_policy, XiaoEnvironment, evaluation
from tests.test_safe_xiaomi_policy import safe_response
from robocasa.recovery.full_snapshot import capture_full_snapshot, save_full_snapshot, stable_digest
from robocasa.recovery.operator_screen import DEFAULT_SPEC as OPERATOR_SPEC, OPERATORS
from robocasa.recovery.intervention_timing import (
    DEFAULT_SPEC, allocate, validate_spec, build_registration, validate_registration,
    ordered_trace, prefix_audit, unique_records, base_records, TimingPolicy,
)
from robocasa.recovery.run_intervention_timing import (
    capture_entry, execute_branch, reference_audit, engineering_analysis, timing_audit,
    run, build_parser,
)
from robocasa.recovery.analyze_intervention_timing import (
    CELLS, METRICS, branch_outcome, parent_results, macro_summary, screen, bootstrap, analyze,
)
from robocasa.recovery.phase3b_training_horizon import sha256_file


def runtime(*, complete_at=None, done_at=6):
    def evaluate(env):
        result = evaluation(env)
        if complete_at is not None:
            result["predicates"]["mug_on_tray"]["value"] = env.timestep >= complete_at
        return result
    def step(env, action):
        obs, reward, _, info = env.step(action["action.end_effector_position"][:1])
        return obs, reward, done_at is not None and env.timestep >= done_at, info
    return {"make_env": lambda *args: XiaoEnvironment(), "step_fn": step,
            "call_policy": lambda p, obs: p(obs), "success_fn": lambda **kw: False,
            "get_subtask_eval": evaluate}


def snapshot(task="ArrangeTea", parent="p", step=0, rt=None):
    rt = rt or runtime(done_at=None)
    env, policy = XiaoEnvironment(), make_policy()
    env.reset(seed=1)
    for _ in range(step):
        action = policy(env.get_current_observation())
        policy.pop_request_records(); policy.pop_inference_record()
        rt["step_fn"](env, action)
    return capture_full_snapshot(env, policy, parent_id=parent, task_name=task,
        trigger_name="stage_prefix_2", environment_step=step,
        observation=env.get_current_observation(), subtask_eval=rt["get_subtask_eval"](env),
        metadata={"environment_seed": 1, "trigger_stage": f"{task}::mug_on_tray",
                  "local_inferences": 2 if step else 0, "ordered_prefix": []})


def branch(op="continue", delay=None, **kwargs):
    return {"branch_id": f"p--{op}--d{delay}--s0--r0", "parent_id": "p", "task_name": "ArrangeTea",
            "operator": op, "delay": delay, "schedule": 0, "sampling_seed": 500,
            "repeat": 0, "role": "primary", "sentinel": True, "horizon_environment_steps": 32, **kwargs}


def synthetic_outcomes(pattern):
    result = []
    for task in ("ArrangeTea", "CuttingToolSelection"):
        for parent in range(5):
            for cell in CELLS:
                for seed in (0, 1):
                    value = float(pattern(task, parent, cell, seed))
                    result.append({"parent_id": f"{task}-{parent}", "task_name": task, "cell": cell, "schedule": seed,
                        **{m: 0.0 for m in METRICS}, "completion": value, "durable_completion": value})
    return result


def source_fixture(base):
    source, phase2 = base/"source", base/"phase2"
    source.mkdir(); (phase2/"snapshots").mkdir(parents=True)
    (phase2/"plan.json").write_text(json.dumps({"policy_module": "fake:factory"}))
    selected, hashes = [], {}
    for task in ("ArrangeTea", "CuttingToolSelection"):
        for i in range(5):
            pid = f"{task}-p{i}"
            ref = snapshot(task=task, parent=pid, step=4)
            path = phase2/"snapshots"/f"{ref.snapshot_id}.pkl.gz"
            save_full_snapshot(ref, path)
            hashes[ref.snapshot_id] = sha256_file(path)
            selected.append({"snapshot_id": ref.snapshot_id, "parent_id": pid,
                             "task_name": task, "boundary": "prefix", "environment_reset_index": i})
            selected.append({**selected[-1], "snapshot_id": ref.snapshot_id+"-landmark", "boundary": "landmark"})
    old_branches, records = [], []
    for selected_snapshot in selected:
        for op in OPERATORS:
            for seed in (0, 1):
                bid = f"{selected_snapshot['snapshot_id']}--{op}--s{seed}--r0"
                path = source/f"{bid}.bin"
                path.write_bytes(bid.encode())
                old_branches.append({"branch_id": bid})
                records.append({"branch_id": bid, "payload_path": path.name, "payload_file_sha256": sha256_file(path)})
    old = {"scope": "full", "selected_snapshots": selected, "branches": old_branches,
           "spec": OPERATOR_SPEC, "phase2_run_dir": str(phase2),
           "phase2_registration": {"phase2_snapshot_file_sha256": hashes,
               "trigger_stages": {t: f"{t}::mug_on_tray" for t in ("ArrangeTea", "CuttingToolSelection")}}}
    (source/"registration.json").write_text(json.dumps(old))
    (source/"branch_records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (source/"analysis.json").write_text(json.dumps({"engineering_all_pass": True, "branches": 160}))
    (source/"scientific_analysis.json").write_text(json.dumps({"decision": "mechanism_redesign_or_earlier_snapshot_study",
        "source_ledger_sha256": sha256_file(source/"branch_records.jsonl")}))
    return source


class InterventionTimingTest(unittest.TestCase):
    def test_spec_and_fixed_allocation(self):
        validate_spec(deepcopy(DEFAULT_SPEC))
        for mutate in (lambda s: s.update(delays=[0, 1, 3]), lambda s: s.update(continuation_seed_bases=[1, 1]),
                       lambda s: s["horizons"].update(ArrangeTea=512), lambda s: s.update(endpoint="suffix")):
            spec = deepcopy(DEFAULT_SPEC); mutate(spec)
            with self.assertRaises(ValueError):
                validate_spec(spec)
        parents = [{"parent_id": f"p{i}", "task_name": "ArrangeTea" if i < 5 else "CuttingToolSelection",
                    "index": i, "sentinel": i in (0, 5)} for i in range(10)]
        rows = allocate(parents, DEFAULT_SPEC)
        self.assertEqual(len(rows), 220)
        self.assertEqual(sum(r["role"] == "primary" for r in rows), 200)
        self.assertEqual(sum(r["sentinel"] for r in rows), 60)
        self.assertEqual(sum(r["operator"] == "continue" and r["role"] == "primary" for r in rows), 20)
        self.assertEqual(len({r["branch_id"] for r in rows}), 220)
        for parent in parents:
            for seed in (0, 1):
                self.assertEqual(len({r["sampling_seed"] for r in rows if r["parent_id"] == parent["parent_id"] and r["schedule"] == seed}), 1)

    def test_source_registration_hash_guards_and_global_parent_indices(self):
        with tempfile.TemporaryDirectory() as directory, patch("robocasa.recovery.intervention_timing.validate_operator_registration"):
            source = source_fixture(Path(directory))
            reg = build_registration(source)
            validate_registration(reg)
            self.assertEqual([p["index"] for p in reg["parents"][:2]], [0, 5])
            with patch("robocasa.recovery.intervention_timing.code_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "code changed"):
                    validate_registration(reg)
            changed = deepcopy(reg); changed["spec"]["delays"] = [0, 1, 3]
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_registration(changed)
            (source/"analysis.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                validate_registration(reg)

    def test_genuine_entry_replay_and_strict_endpoint_audit(self):
        ref = snapshot(step=4)
        entry, nominal, audit, reproduced = capture_entry(ref, XiaoEnvironment(), make_policy(), runtime())
        self.assertTrue(audit["all_exact"], audit)
        self.assertEqual(entry.environment_step, 0)
        self.assertEqual(len(nominal["steps"]), 4)
        self.assertEqual(sum(bool(s["requests"]) for s in nominal["steps"]), 2)
        reproduced.observation["state.end_effector_position_relative"][0] += 1
        self.assertFalse(reference_audit(ref, reproduced)["all_exact"])
        early = runtime(done_at=2)
        with self.assertRaisesRegex(ValueError, "terminated"):
            capture_entry(ref, XiaoEnvironment(), make_policy(), early)
        noisy = runtime()
        normal = noisy["step_fn"]
        def advance_rng(e, a):
            import numpy as np
            np.random.random()
            return normal(e, a)
        noisy["step_fn"] = advance_rng
        with self.assertRaisesRegex(ValueError, "global RNG"):
            capture_entry(ref, XiaoEnvironment(), make_policy(), noisy)

    def test_delays_share_control_prefix_count_chunks_and_scripted_steps(self):
        ref = snapshot()
        policy = make_policy()
        rt = runtime(done_at=None)
        control = execute_branch(ref, XiaoEnvironment(), policy, branch(), OPERATOR_SPEC, rt)["payload"]
        with patch("robocasa.recovery.run_intervention_timing.validate_retreat_controller", return_value={"accepted": True}):
            for op in OPERATORS[1:]:
                for delay in (0, 1, 2):
                    row = branch(op, delay)
                    payload = execute_branch(ref, XiaoEnvironment(), make_policy(), row, OPERATOR_SPEC, rt)["payload"]
                    self.assertTrue(prefix_audit(payload, control)["all_exact"])
                    self.assertTrue(timing_audit(payload, row))
                    summary = payload["summary"]
                    self.assertEqual(summary["intervention"]["environment_step"], delay*2)
                    self.assertEqual(summary["num_steps"], 32)
                    self.assertEqual(summary["request_sampling_seeds"], list(range(500, 500+summary["num_policy_requests"])))
                    if op == "retreat_reset_replan":
                        self.assertEqual(summary["request_environment_step_indices"][delay], delay*2+24)
                        self.assertEqual(summary["num_policy_requests"], 4)

    def test_early_completion_permanent_noop_and_all_assignments_retained(self):
        ref = snapshot()
        rt = runtime(complete_at=1, done_at=6)
        control = execute_branch(ref, XiaoEnvironment(), make_policy(), branch(), OPERATOR_SPEC, rt)["payload"]
        for delay in (1, 2):
            row = branch("stage_retry", delay)
            payload = execute_branch(ref, XiaoEnvironment(), make_policy(), row, OPERATOR_SPEC, rt)["payload"]
            self.assertFalse(payload["summary"]["intervention"]["executed"])
            self.assertEqual(payload["summary"]["intervention"]["reason"], "target_completed_before_application")
            self.assertTrue(prefix_audit(payload, control)["all_exact"])
            self.assertTrue(timing_audit(payload, row))

    def test_scripted_terminal_zero_policy_requests_are_valid(self):
        with patch("robocasa.recovery.run_intervention_timing.validate_retreat_controller", return_value={"accepted": True}):
            result = execute_branch(snapshot(), XiaoEnvironment(), make_policy(), branch("retreat_reset_replan", 0), OPERATOR_SPEC, runtime())
        self.assertEqual(result["summary"]["num_policy_requests"], 0)
        self.assertTrue(result["summary"]["zero_request_scripted_termination_valid"])
        self.assertTrue(result["summary"]["record_alignment_valid"])

    def test_hidden_policy_prefix_change_and_missed_boundary_are_rejected(self):
        ref = snapshot()
        rt = runtime(done_at=None)
        control = execute_branch(ref, XiaoEnvironment(), make_policy(), branch(), OPERATOR_SPEC, rt)["payload"]
        payload = execute_branch(ref, XiaoEnvironment(), make_policy(), branch("stage_retry", 2), OPERATOR_SPEC, rt)["payload"]
        self.assertTrue(prefix_audit(payload, control)["all_exact"])
        payload["summary"]["suffix_policy_state_sha256"][1] = "changed_hidden_state"
        self.assertFalse(prefix_audit(payload, control)["all_exact"])
        wrapped = TimingPolicy(make_policy(), branch("stage_retry", 1), OPERATOR_SPEC, "mug_on_tray")
        wrapped.requests = 2
        with self.assertRaisesRegex(ValueError, "Missed"):
            wrapped.decide(evaluation(None), {"ordered_current_subtask": "mug_on_tray"})

    def test_ineligible_fallback_is_permanent(self):
        wrapped = TimingPolicy(make_policy(), branch("stage_retry", 2), OPERATOR_SPEC, "mug_on_tray")
        wrapped.decide(evaluation(None), {"ordered_current_subtask": "another_stage"})
        decision = deepcopy(wrapped.decision)
        self.assertEqual(decision["reason"], "target_stage_ineligible_before_application")
        wrapped.requests = 2
        wrapped.decide(evaluation(None), {"ordered_current_subtask": "mug_on_tray"})
        self.assertEqual(wrapped.decision, decision)
        self.assertIsNone(wrapped.treatment)

    def test_ordered_prefix_carries_history_without_inventing_raw_success(self):
        evaluations = [{"required_predicates": ["prior", "target"],
                        "predicates": {"prior": {"value": False}, "target": {"value": False}}}]
        raw = deepcopy(evaluations)
        trace = ordered_trace(evaluations, ["prior"])
        self.assertEqual(trace[0]["ordered_current_subtask"], "target")
        self.assertEqual(trace[0]["regressed_predicates"], [])
        self.assertEqual(evaluations, raw)
        with self.assertRaisesRegex(ValueError, "prefix"):
            ordered_trace(evaluations, ["target"])

    def test_regression_proxy_separates_upstream_and_later_relations(self):
        record = {**branch("stage_retry", 0), "ordered_prefix": ["prior"], "num_steps": 2,
                  "regressed_predicates": ["later"], "task_success": False, "wall_seconds": 1,
                  "num_policy_requests": 1, "termination_reason": "suffix_horizon",
                  "intervention": {"executed": True, "reason": "assigned_boundary", "environment_step": 0}}
        payload = {"subtask_evals": [
            {"predicates": {"prior": {"value": True}, "mug_on_tray": {"value": False}, "later": {"value": True}}},
            {"predicates": {"prior": {"value": True}, "mug_on_tray": {"value": True}, "later": {"value": False}}}], "infos": []}
        reg = {"phase2_registration": {"trigger_stages": {"ArrangeTea": "ArrangeTea::mug_on_tray"}}}
        with patch("robocasa.recovery.analyze_intervention_timing.load_branch_payload", return_value=payload):
            values = branch_outcome(None, reg, record)
        self.assertEqual(values["ordered_prefix_regression"], 0)
        self.assertEqual(values["legacy_regression"], 1)

    def test_paired_analysis_all_cells_timing_contrasts_and_harm_gates(self):
        data = synthetic_outcomes(lambda task, p, cell, s: cell == "stage_retry--d0")
        rows = parent_results(data)
        summary = macro_summary(rows)
        self.assertEqual(summary["earlier_minus_later"]["stage_retry--d0_minus_d2"]["completion"], 1)
        self.assertEqual(screen(rows, summary, OPERATOR_SPEC["gates"])["promising_cells"], ["stage_retry--d0"])
        ci = bootstrap(rows, 20, 5)
        self.assertEqual(ci, bootstrap(rows, 20, 5))
        self.assertEqual(ci["stage_retry--d0_minus_continue"]["completion"]["ci95"], [1, 1])
        for metric in ("ordered_prefix_regression", "safety", "durable_completion"):
            harmed = deepcopy(data)
            for row in harmed:
                if row["cell"] == "stage_retry--d0":
                    row[metric] = 0 if metric == "durable_completion" else 1
            selected = parent_results(harmed)
            self.assertFalse(screen(selected, macro_summary(selected), OPERATOR_SPEC["gates"])["promising_cells"])
        with self.assertRaisesRegex(ValueError, "schedule"):
            parent_results(data[:-1])
        opposite_seeds = synthetic_outcomes(lambda t, p, c, s: (c == "continue" and s == 0) or (c == "stage_retry--d0" and s == 1))
        self.assertEqual(macro_summary(parent_results(opposite_seeds))["contrasts_vs_continue"]["stage_retry--d0"]["completion"], 0)

    def test_full_pipeline_sentinel_resume_orphans_integrity_and_analysis(self):
        with tempfile.TemporaryDirectory() as directory, patch("robocasa.recovery.intervention_timing.validate_operator_registration"):
            base = Path(directory)
            source = source_fixture(base)
            root = base/"run"
            args = build_parser().parse_args(["--operator-run-dir", str(source), "--output-dir", str(root),
                                             "--scope", "sentinel", "--model-path", "/checkpoint"])
            rt = runtime()
            policy = make_policy()
            policy.client.responses.extend(safe_response() for _ in range(2000))
            rt.update(load_factory=lambda _: None, call_factory=lambda *args: policy)
            with patch("robocasa.recovery.run_intervention_timing._validate_live_provenance"), \
                 patch("robocasa.recovery.run_intervention_timing.validate_retreat_controller", return_value={"accepted": True}), \
                 redirect_stdout(io.StringIO()):
                sentinel = run(args, rt)
                self.assertTrue(sentinel["engineering_all_pass"], sentinel)
                self.assertEqual(sentinel["branches"], 60)
                reg = json.loads((root/"registration.json").read_text())
                self.assertEqual(len(base_records(root, reg)), 2)
                # Recover payloads and committed bases saved before a ledger append.
                ledger = (root/"branch_records.jsonl").read_text().splitlines(True)
                original = json.loads(ledger[-1])
                raw = (root/original["payload_path"]).read_bytes()
                (root/"branch_records.jsonl").write_text("".join(ledger[:-1]))
                (root/"base_records.jsonl").unlink()
                args.resume = True
                rerun = run(args, rt)
                self.assertTrue(rerun["engineering_all_pass"])
                self.assertEqual((root/original["payload_path"]).read_bytes(), raw)
                args.scope = "full"
                full = run(args, rt)
                self.assertEqual(full["branches"], 220)
                self.assertEqual(full["primary_branches"], 200)
                before = (root/"branch_records.jsonl").read_bytes()
                run(args, rt)
                self.assertEqual((root/"branch_records.jsonl").read_bytes(), before)
                result = analyze(root, bootstrap_replicates=20)
                self.assertEqual(result["parents"], 10)
                self.assertEqual(result["decision"], "mechanism_redesign_or_diagnostic_benchmark")
                self.assertEqual(len(result["contrasts_vs_continue"]), 9)
                self.assertEqual(len(result["earlier_minus_later"]), 9)
                self.assertTrue((root/"timing_branch_results.csv").exists())
                args.operator_run_dir = base/"different_source"
                with self.assertRaisesRegex(ValueError, "Resume source"):
                    run(args, rt)
                args.operator_run_dir = source
                base_row = next(iter(base_records(root, reg).values()))
                base_path = root/base_row["snapshot_path"]
                base_bytes = base_path.read_bytes()
                base_path.write_bytes(base_bytes+b"changed")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    base_records(root, reg)
                base_path.write_bytes(base_bytes)
                # Existing evidence is never silently rerun after an error.
                (root/"errors.jsonl").write_text('{"error":"retained"}\n')
                with self.assertRaisesRegex(ValueError, "Technical errors"):
                    run(args, rt)
                (root/"errors.jsonl").unlink()
                with (root/original["payload_path"]).open("ab") as stream:
                    stream.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "file changed"):
                    unique_records(root, reg)

    def test_full_is_blocked_before_sentinel_and_dry_run_never_opens_runtime(self):
        with tempfile.TemporaryDirectory() as directory, patch("robocasa.recovery.intervention_timing.validate_operator_registration"):
            base = Path(directory)
            source = source_fixture(base)
            root = base/"run"
            args = build_parser().parse_args(["--operator-run-dir", str(source), "--output-dir", str(root),
                                             "--scope", "full", "--model-path", "/checkpoint"])
            with self.assertRaisesRegex(ValueError, "sentinel"):
                run(args)
            args.dry_run = True
            with patch("robocasa.recovery.run_intervention_timing._validate_live_provenance"), \
                 patch("robocasa.recovery.run_intervention_timing._runtime") as load, redirect_stdout(io.StringIO()):
                run(args)
            load.assert_not_called()
            self.assertFalse((root/"registration.json").exists())


if __name__ == "__main__":
    unittest.main()
