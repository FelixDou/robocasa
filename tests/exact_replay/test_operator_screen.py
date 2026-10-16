"""Simulator-free protocol, execution, integrity and causal-analysis checks."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages
install_lightweight_robocasa_packages()

from tests.test_safe_xiaomi_policy import FakeClient, FakeProcessor, observation, safe_response
from tests.exact_replay.test_phase2_snapshot_replay import FakeEnvironment
from tests.exact_replay.test_phase3b_training_horizon import _write_phase2_source
from robocasa.recovery.xiaomi_robotics_1_policy import XiaomiRobotics1Policy
from robocasa.recovery.full_snapshot import capture_full_snapshot, stable_digest
from robocasa.recovery.exact_replay.counterfactual_branch import BranchSpec, run_counterfactual_branch
from robocasa.recovery.exact_replay.phase3b_training_horizon import (
    build_registration as build_phase3b_registration, sha256_file, sha256_json,
)
from robocasa.recovery.exact_replay.operator_screen import (
    DEFAULT_SPEC, OPERATORS, OperatorPolicy, build_registration, clear_history,
    validate_spec, validate_registration, unique_records, validate_retreat_controller,
)
from robocasa.recovery.exact_replay.run_operator_screen import (
    _save_result, _recover_orphans, build_parser, run,
)
from robocasa.recovery.exact_replay.analyze_operator_screen import (
    snapshot_results, macro_summary, route, bootstrap, analyze, branch_outcome,
)


def make_policy():
    return XiaomiRobotics1Policy(
        model_path="/checkpoint", processor=FakeProcessor(),
        client=FakeClient([safe_response() for _ in range(80)]),
        replan_steps=2, collect_safe_features=True, sampling_seed_base=100,
    )


def branch(operator="continue", **kwargs):
    return {"branch_id": "s--continue--s0--r0", "snapshot_id": "s", "parent_id": "p",
            "task_name": "ArrangeTea", "boundary": "prefix", "operator": operator,
            "sampling_seed": 500, "schedule": 0, "repeat": 0, "role": "primary",
            "horizon_environment_steps": 32, **kwargs}


def outcomes(pattern):
    result = []
    for task in ("ArrangeTea", "CuttingToolSelection"):
        for parent in range(4):
            for boundary in ("prefix", "landmark"):
                for operator in OPERATORS:
                    for schedule in (0, 1):
                        completion = float(pattern(operator, parent, schedule))
                        result.append({"snapshot_id": f"{task}-{parent}-{boundary}",
                            "task_name": task, "parent_id": f"{task}-{parent}", "boundary": boundary,
                            "operator": operator, "schedule": schedule,
                            "completion": completion, "durable_completion": completion,
                            "regression": 0.0, "safety": 0.0, "task_success": 0.0,
                            "policy_calls": 3, "wall_seconds": 1.0})
    return result


class XiaoEnvironment(FakeEnvironment):
    def get_current_observation(self):
        obs = observation("Arrange the tea.")
        obs["state.end_effector_position_relative"][0] = self.value
        return obs


def evaluation(env):
    return {"task_success": False, "required_predicates": ["mug_on_tray"],
            "predicates": {"mug_on_tray": {"value": False, "stage": "task"}}}


class OperatorScreenTest(unittest.TestCase):
    def test_retreat_rejects_absolute_or_unknown_controller(self):
        from types import SimpleNamespace
        controller = SimpleNamespace(input_type="delta", control_dim=6, input_ref_frame="base")
        env = SimpleNamespace(sim=object(), robots=[SimpleNamespace(
            composite_controller=SimpleNamespace(part_controllers={"right": controller}))])
        self.assertTrue(validate_retreat_controller(env)["accepted"])
        controller.input_type = "absolute"
        with self.assertRaisesRegex(ValueError, "delta-pose"):
            validate_retreat_controller(env)

    def test_spec_rejects_overlapping_seeds_and_unsafe_retreat(self):
        validate_spec(deepcopy(DEFAULT_SPEC))
        for mutate in (
            lambda s: s.update(continuation_seed_bases=[1, 1]),
            lambda s: s["retreat"].update(lift_command=1.0),
            lambda s: s.update(subgoal_prompts={}),
            lambda s: s.update(retry_steps=0),
        ):
            spec = deepcopy(DEFAULT_SPEC)
            mutate(spec)
            with self.assertRaises(ValueError):
                validate_spec(spec)

    def test_history_reset_preserves_seeds_and_counters(self):
        policy = make_policy()
        policy(observation())
        policy(observation())
        policy.set_next_sampling_seed(999)
        before = policy.get_state()
        clear_history(policy, instruction="fixed goal")
        after = policy.get_state()
        for key in ("sampling_config", "next_sampling_seed", "ordinary_inference_index", "inference_index", "environment_step"):
            self.assertEqual(after[key], before[key])
        self.assertEqual(after["state_queue"], [])
        self.assertTrue(all(not q for q in after["image_queues"].values()))
        policy(observation(), instruction="fixed goal")
        self.assertEqual(policy.client.requests[-1]["sampling_seed"], 999)

    def test_prompt_switch_and_retreat_keep_request_index_schedule(self):
        spec = deepcopy(DEFAULT_SPEC)
        spec["retry_steps"] = 3  # intentionally interrupt a cached 2-step chunk
        for operator in OPERATORS:
            policy = make_policy()
            wrapped = OperatorPolicy(policy, branch(operator), spec)
            for _ in range(32):
                wrapped(observation())
            seeds = [r["sampling_seed"] for r in policy.client.requests]
            self.assertEqual(seeds, list(range(500, 500 + len(seeds))))
            self.assertEqual(policy.get_state()["environment_step"], 32)
            if operator == "stage_retry":
                self.assertEqual(wrapped.events[-1]["step"], 3)
            if operator == "retreat_reset_replan":
                self.assertEqual(len(policy.client.requests), 4)

    def test_real_snapshot_branch_records_scripted_actions_and_resume_orphan(self):
        env = XiaoEnvironment()
        env.reset(seed=1)
        policy = make_policy()
        snapshot = capture_full_snapshot(env, policy, parent_id="p", task_name="ArrangeTea",
            trigger_name="prefix", environment_step=0, observation=env.get_current_observation(),
            subtask_eval=evaluation(env), metadata={"environment_seed": 1})
        spec = deepcopy(DEFAULT_SPEC)
        spec["retry_steps"] = 3
        for operator in OPERATORS:
            row = branch(operator, snapshot_id=snapshot.snapshot_id,
                         branch_id=f"{snapshot.snapshot_id}--{operator}--s0--r0")
            wrapped = OperatorPolicy(policy, row, spec)
            result = run_counterfactual_branch(snapshot, env, wrapped,
                BranchSpec(row["branch_id"], "candidate", 500, 32),
                step_fn=lambda e, a: e.step(a["action.end_effector_position"][:1]),
                success_fn=lambda **kw: False, subtask_eval_fn=evaluation)
            self.assertTrue(result["summary"]["restore_valid"])
            self.assertTrue(result["summary"]["record_alignment_valid"])
            self.assertEqual(result["summary"]["num_steps"], 32)
            expected_first = 24 if operator == "retreat_reset_replan" else 0
            self.assertEqual(result["summary"]["request_environment_step_indices"][0], expected_first)
            if operator == "retreat_reset_replan":
                self.assertEqual(float(result["payload"]["actions"][8]["action.end_effector_position"][2]), np.float32(0.1))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _save_result(result, row, wrapped, 1.0, root)
                registration = {"branches": [row]}
                original = unique_records(root, registration)
                original_bytes = next((root / "branches").iterdir()).read_bytes()
                (root / "branch_records.jsonl").unlink()  # interrupted atomic save
                recovered = _recover_orphans(root, registration)
                self.assertEqual(original, recovered)
                self.assertEqual(next((root / "branches").iterdir()).read_bytes(), original_bytes)
                with (root / "branch_records.jsonl").open("a") as stream:
                    stream.write(json.dumps(next(iter(recovered.values()))) + "\n")
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    unique_records(root, registration)

    def test_registration_allocation_uses_global_snapshot_seed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p2 = root / "phase2"
            p2.mkdir()
            _write_phase2_source(p2)
            source = build_phase3b_registration(p2, scope="sentinel")
            p3 = root / "phase3b"
            p3.mkdir()
            for name in ("registration.json", "analysis.json", "scientific_analysis.json", "branch_records.jsonl"):
                (p3 / name).write_text("" if name.endswith("jsonl") else "{}")
            with patch("robocasa.recovery.exact_replay.operator_screen._source", return_value=source):
                sentinel = build_registration(p3, scope="sentinel")
                self.assertEqual(len(sentinel["selected_snapshots"]), 4)
                self.assertEqual(sentinel["expected_branches"], 48)
                self.assertEqual(sum(b["role"] == "primary" for b in sentinel["branches"]), 32)
                sroot = root / "sentinel"
                sroot.mkdir()
                (sroot / "registration.json").write_text(json.dumps(sentinel))
                (sroot / "branch_records.jsonl").write_text("")
                with patch("robocasa.recovery.exact_replay.run_operator_screen.engineering_analysis", return_value={"engineering_all_pass": True}):
                    full = build_registration(p3, scope="full", sentinel_dir=sroot)
                with patch("robocasa.recovery.exact_replay.run_operator_screen.engineering_analysis", return_value={"engineering_all_pass": False}):
                    with self.assertRaisesRegex(ValueError, "blocked"):
                        build_registration(p3, scope="full", sentinel_dir=sroot)
                self.assertEqual(full["expected_branches"], 160)
                selected = {b["branch_id"]: b for b in full["branches"]}
                for row in sentinel["branches"]:
                    if row["role"] == "primary":
                        self.assertEqual(row["sampling_seed"], selected[row["branch_id"]]["sampling_seed"])
                with self.assertRaisesRegex(ValueError, "sentinel"):
                    build_registration(p3, scope="full")
                changed = deepcopy(sentinel)
                changed["spec"]["retry_steps"] += 1
                with self.assertRaisesRegex(ValueError, "checksum"):
                    validate_registration(changed)
                with patch("robocasa.recovery.exact_replay.operator_screen.code_hashes", return_value={}):
                    with self.assertRaisesRegex(ValueError, "code changed"):
                        validate_registration(sentinel)
                (p3 / "analysis.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    validate_registration(sentinel)

    def test_dry_run_does_not_write_or_open_simulator(self):
        args = build_parser().parse_args(["--phase3b-run-dir", "/source", "--output-dir", "/unused",
                                          "--scope", "sentinel", "--model-path", "/checkpoint", "--dry-run"])
        with patch("robocasa.recovery.exact_replay.run_operator_screen.build_registration", return_value={
            "scope": "sentinel", "expected_branches": 48, "spec": DEFAULT_SPEC,
            "selected_snapshots": [], "phase2_registration": {}, "phase2_run_dir": "/phase2",
        }), patch("robocasa.recovery.exact_replay.run_operator_screen._validate_live_provenance"), \
             patch("pathlib.Path.read_text", return_value="{}"), \
             patch("pathlib.Path.mkdir") as mkdir, \
             patch("robocasa.recovery.exact_replay.run_operator_screen._runtime") as runtime, redirect_stdout(io.StringIO()):
            run(args)
        mkdir.assert_not_called()
        runtime.assert_not_called()

    def test_selection_is_over_seed_averages_not_seedwise_oracle(self):
        data = outcomes(lambda op, parent, seed: int((op == "continue" and seed == 0) or (op == "stage_retry" and seed == 1)))
        rows = snapshot_results(data)
        summary = macro_summary(rows)
        self.assertEqual(summary["oracle_routing_completion"], 0.5)
        self.assertEqual(summary["oracle_routing_minus_best_fixed"], 0)
        self.assertEqual(route(rows, summary, DEFAULT_SPEC)["decision"], "mechanism_redesign_or_earlier_snapshot_study")

    def test_fixed_operator_router_and_null_routing(self):
        cases = (
            (lambda op, parent, seed: int(op == "stage_retry"), "fixed_operator_fresh_parent_validation"),
            (lambda op, parent, seed: int((op == "stage_retry" and parent < 2) or (op == "history_reset_replan" and parent >= 2)), "operator_router_development"),
            (lambda op, parent, seed: 0, "mechanism_redesign_or_earlier_snapshot_study"),
        )
        for pattern, decision in cases:
            rows = snapshot_results(outcomes(pattern))
            self.assertEqual(route(rows, macro_summary(rows), DEFAULT_SPEC)["decision"], decision)

    def test_harm_and_transient_completion_cannot_pass_gate(self):
        for key in ("regression", "safety", "durable_completion"):
            data = outcomes(lambda op, parent, seed: int(op == "stage_retry"))
            for row in data:
                if row["operator"] == "stage_retry":
                    row[key] = 0.0 if key == "durable_completion" else 1.0
            rows = snapshot_results(data)
            self.assertEqual(route(rows, macro_summary(rows), DEFAULT_SPEC)["promising_operators"], [])

    def test_prior_stage_harm_is_separate_from_new_target_regression(self):
        payload = {"subtask_evals": [
            {"predicates": {"prior": {"value": True}, "target": {"value": False}}},
            {"predicates": {"prior": {"value": True}, "target": {"value": True}}},
            {"predicates": {"prior": {"value": True}, "target": {"value": False}}},
        ]}
        record = {**branch(), "termination_reason": "suffix_horizon", "task_success": False,
            "regressed_predicates": ["target"], "num_policy_requests": 1,
            "wall_seconds": 1.0, "num_steps": 2}
        reg = {"phase2_registration": {"trigger_stages": {"ArrangeTea": "ArrangeTea::target"}}}
        with patch("robocasa.recovery.exact_replay.analyze_operator_screen.load_branch_payload", return_value=payload):
            result = branch_outcome(Path("/unused"), reg, record)
            self.assertEqual(result["regression"], 0)
            self.assertEqual(result["any_predicate_regression"], 1)
            self.assertEqual(result["completion"], 1)
            self.assertEqual(result["durable_completion"], 0)
            record["regressed_predicates"].append("prior")
            self.assertEqual(branch_outcome(Path("/unused"), reg, record)["regression"], 1)

    def test_missing_schedule_rejected_and_bootstrap_reproducible(self):
        data = outcomes(lambda op, parent, seed: int(op == "stage_retry"))
        with self.assertRaisesRegex(ValueError, "schedule"):
            snapshot_results(data[:-1])
        rows = snapshot_results(data)
        self.assertEqual(bootstrap(rows, 20, 5), bootstrap(rows, 20, 5))
        self.assertEqual(bootstrap(rows, 20, 5)["stage_retry_minus_continue"]["ci95"], [1.0, 1.0])

    def test_collector_resume_and_file_analysis_end_to_end_with_fake_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            phase2 = base / "phase2"
            phase2.mkdir()
            (phase2 / "plan.json").write_text(json.dumps({"policy_module": "fake:factory"}))
            env = XiaoEnvironment()
            env.reset(seed=1)
            policy = make_policy()
            policy.client.responses.extend(safe_response() for _ in range(200))
            snapshot = capture_full_snapshot(env, policy, parent_id="p", task_name="ArrangeTea",
                trigger_name="prefix", environment_step=0, observation=env.get_current_observation(),
                subtask_eval=evaluation(env), metadata={"environment_seed": 1})
            spec = deepcopy(DEFAULT_SPEC)
            spec["retry_steps"] = 3
            spec_path = base / "spec.json"
            spec_path.write_text(json.dumps(spec))
            branches = [branch(op, snapshot_id=snapshot.snapshot_id,
                branch_id=f"{snapshot.snapshot_id}--{op}--s{schedule}--r0", schedule=schedule,
                sampling_seed=500 + 100 * schedule) for op in OPERATORS for schedule in (0, 1)]
            reg = {"protocol": "mock_integration", "scope": "full", "spec": spec,
                "phase3b_run_dir": str((base / "phase3b").resolve()), "phase2_run_dir": str(phase2),
                "phase2_registration": {"trigger_stages": {"ArrangeTea": "ArrangeTea::mug_on_tray"}},
                "branches": branches, "expected_branches": 8, "oracle_anchors": {},
                "selected_snapshots": [{"snapshot_id": snapshot.snapshot_id}],
                "sentinel": {"run_dir": str((base / "sentinel").resolve())}}
            root = base / "run"
            args = build_parser().parse_args(["--phase3b-run-dir", str(base / "phase3b"),
                "--output-dir", str(root), "--scope", "full", "--sentinel-run-dir", str(base / "sentinel"),
                "--model-path", "/checkpoint", "--operator-spec", str(spec_path)])
            def fresh(*args):
                env = XiaoEnvironment()
                env.reset(seed=1)
                return env
            def load(registration, row, destination):
                from robocasa.recovery.exact_replay.run_phase2_snapshot_replay import append_jsonl
                append_jsonl(destination / "snapshot_integrity_audits.jsonl", {
                    "target_snapshot_id": snapshot.snapshot_id, "accepted": True})
                return snapshot
            runtime = {"load_factory": lambda name: None,
                "call_factory": lambda factory, env, kwargs: policy,
                "step_fn": lambda e, a: e.step(a["action.end_effector_position"][:1]),
                "success_fn": lambda **kw: False, "get_subtask_eval": evaluation}
            with patch("robocasa.recovery.exact_replay.run_operator_screen.build_registration", return_value=reg), \
                 patch("robocasa.recovery.exact_replay.run_operator_screen.validate_registration"), \
                 patch("robocasa.recovery.exact_replay.analyze_operator_screen.validate_registration"), \
                 patch("robocasa.recovery.exact_replay.run_operator_screen._validate_live_provenance"), \
                 patch("robocasa.recovery.exact_replay.run_operator_screen._make_fresh_branch_environment", side_effect=fresh) as create_env, \
                 patch("robocasa.recovery.exact_replay.run_operator_screen._load_snapshot", side_effect=load), \
                 patch("robocasa.recovery.exact_replay.run_operator_screen.validate_retreat_controller", return_value={"accepted": True}), \
                 redirect_stdout(io.StringIO()):
                result = run(args, runtime)
                self.assertTrue(result["engineering_all_pass"])
                before = (root / "branch_records.jsonl").read_bytes()
                args.resume = True
                run(args, runtime)
                self.assertEqual(create_env.call_count, 8)  # no re-execution on resume
                self.assertEqual((root / "branch_records.jsonl").read_bytes(), before)
                result = analyze(root, bootstrap_replicates=10)
                self.assertEqual(result["decision"], "mechanism_redesign_or_earlier_snapshot_study")
                self.assertTrue((root / "operator_branch_results.csv").exists())
                spec_path.write_text(json.dumps({**spec, "retry_steps": 4}))
                with self.assertRaisesRegex(ValueError, "Resume differs"):
                    run(args, runtime)
                spec_path.write_text(json.dumps(spec))
                (root / "errors.jsonl").write_text(json.dumps({"error": "retained technical failure"}) + "\n")
                with self.assertRaisesRegex(ValueError, "technical errors"):
                    run(args, runtime)


if __name__ == "__main__":
    unittest.main()
