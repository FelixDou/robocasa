"""Simulator-free tests for the recoverability-ladder collector and analyzer."""

from collections import deque
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()
from robocasa.recovery import recoverability_ladder as ladder
from robocasa.recovery.subtask_eval import build_subtask_trace


REPO = Path(__file__).resolve().parents[1]
STEP = 0.125
TRIGGER_STEP = 12
TASK_INSTRUCTION = "Reach the object, then move it."


class State:
    def __init__(self, time, qpos, qvel):
        self.time = float(time)
        self.qpos = np.array(qpos, dtype=float)
        self.qvel = np.array(qvel, dtype=float)

    def flatten(self):
        return np.concatenate([[self.time], self.qpos, self.qvel])


class Model:
    names = ["robot0_reach", "robot0_wrist", "obj_x", "obj_stuck"]
    njnt = 4
    jnt_qposadr = np.arange(4)
    jnt_dofadr = np.arange(4)
    jnt_type = np.full(4, 3)

    def joint_id2name(self, index):
        return self.names[index]

    def get_xml(self):
        return "<mujoco model='mock'/>"


class Sim:
    def __init__(self):
        self.model = Model()
        self.time = 0.0
        self.qpos = np.zeros(4)
        self.qvel = np.zeros(4)

    def get_state(self):
        return State(self.time, self.qpos.copy(), self.qvel.copy())

    def set_state(self, state):
        self.time = state.time
        self.qpos = np.array(state.qpos, dtype=float)
        self.qvel = np.array(state.qvel, dtype=float)

    def set_state_from_flattened(self, flat):
        self.time = float(flat[0])
        self.qpos = np.array(flat[1:5], dtype=float)
        self.qvel = np.array(flat[5:9], dtype=float)

    def forward(self):
        pass


class MockEnv:
    """Reach then move. Seeds pick a scripted failure at step 12.

    seed % 3 == 0: success; 1: object gets stuck (world-state failure);
    2: wrist jams (robot-state failure). The trigger fires once per episode and
    is not part of the simulator state, so restored states do not re-trigger.
    """

    def __init__(self, seed=0):
        self.sim = Sim()
        self.override = None
        self.fail_mode = 0
        self.steps = 0
        self.triggered = False
        self.closed = False

    def reset(self, seed=None):
        self.sim = Sim() if self.sim is None else self.sim
        self.sim.qpos[:] = 0
        self.sim.qvel[:] = 0
        self.sim.time = 0.0
        self.fail_mode = (seed or 0) % 3
        self.steps = 0
        self.triggered = False
        self.override = None
        return self.get_current_observation(), {"subtask_eval": self.get_subtask_progress()}

    def get_current_observation(self):
        return {
            "annotation.human.task_description": self.override or TASK_INSTRUCTION,
            "state": self.sim.qpos.copy(),
        }

    def set_task_description(self, text):
        self.override = text

    def clear_task_description(self):
        self.override = None

    def get_ep_meta(self):
        return {"lang": TASK_INSTRUCTION, "layout_id": 1}

    def get_subtask_progress(self):
        reach = bool(self.sim.qpos[0] >= 1.0)
        move = bool(self.sim.qpos[2] >= 1.0)
        predicates = {
            "reach": {"value": reach, "required": True, "description": "Reach the object."},
            "move": {"value": move, "required": True, "description": "Move the object."},
            "task_success": {"value": reach and move, "required": False},
        }
        return {
            "task_name": "MockTask",
            "predicates": predicates,
            "required_predicates": ["reach", "move"],
            "task_success": reach and move,
        }

    def step(self, action):
        self.steps += 1
        qpos = self.sim.qpos
        qpos[0] = min(1.0, qpos[0] + action[0])
        jammed = qpos[1] > 2.0
        if qpos[0] >= 1.0 and qpos[3] < 0.5 and not jammed:
            qpos[2] = min(1.0, qpos[2] + action[1])
        if not self.triggered and self.steps == TRIGGER_STEP and self.fail_mode:
            self.triggered = True
            if self.fail_mode == 1:
                qpos[3] = 1.0
            else:
                qpos[1] = 3.0
        self.sim.time += 0.05
        progress = self.get_subtask_progress()
        success = progress["task_success"]
        info = {"success": success, "subtask_eval": progress}
        return self.get_current_observation(), float(success), False, False, info

    def close(self):
        self.closed = True


class MockPolicy:
    """Chunked policy with an XR-1-like sampling_seed_base contract."""

    def __init__(self, replan_steps=4, feature_shape=(2, 3)):
        self.replan_steps = replan_steps
        self.sampling_seed_base = None
        self.collect_safe_features = True
        self.feature_shape = feature_shape
        self.seeds = []
        self.instructions = []
        self.collect_flags = []
        self.reset()

    def reset(self):
        self.action_plan = deque()
        self.inference_index = 0
        self.pending = None

    def __call__(self, obs, instruction=None):
        instruction = instruction or obs["annotation.human.task_description"]
        self.instructions.append(instruction)
        if not self.action_plan:
            seed = None if self.sampling_seed_base is None else self.sampling_seed_base + self.inference_index
            self.seeds.append(seed)
            self.collect_flags.append(self.collect_safe_features)
            self.inference_index += 1
            for _ in range(self.replan_steps):
                self.action_plan.append(np.array([STEP, STEP, 0.0, 0.0]))
            if self.collect_safe_features:
                self.pending = {"features": np.full(self.feature_shape, self.inference_index, dtype=np.float32)}
        return self.action_plan.popleft()

    def pop_inference_record(self):
        record, self.pending = self.pending, None
        return record


class MockRuntime:
    def __init__(self, horizon=40):
        self.horizon = horizon
        self.envs = []
        self.policies = []

    def task_horizon(self, task_name):
        return self.horizon

    def make_env(self, task_name, seed):
        env = MockEnv(seed)
        self.envs.append(env)
        return env

    def make_policy(self, env):
        policy = MockPolicy()
        self.policies.append(policy)
        return policy


def mock_config(**overrides):
    config = {
        "tasks": ["MockTask"],
        "base_seed": 0,
        "parents_per_task": 6,
        "failed_parents_per_task": 2,
        "success_controls_per_task": 1,
        "k_branches": 2,
        "stall_fraction": 0.25,
        "budget_fraction": 0.5,
        "reprompt_level": "subtask",
        "record_policy_features": True,
        "policy_module": "tests.mock:make_policy",
    }
    config.update(overrides)
    return config


def collect(tmp, **overrides):
    runtime = MockRuntime()
    status = ladder.run_collection(mock_config(**overrides), tmp, runtime=runtime, log=lambda *_: None)
    return runtime, status


class OrderedProgressTest(unittest.TestCase):
    def test_matches_build_subtask_trace_ordered_prefix(self):
        rng = np.random.default_rng(3)
        names = ["a", "b", "c", "d"]
        for _ in range(50):
            evals = []
            for _ in range(30):
                values = rng.random(4) < 0.4
                success = bool(rng.random() < 0.03)
                evals.append(
                    {
                        "predicates": {n: {"value": bool(v)} for n, v in zip(names, values)},
                        "required_predicates": names,
                        "task_success": success,
                    }
                )
            tracker = ladder.OrderedProgress()
            counts = []
            for subtask_eval in evals:
                tracker.update(subtask_eval)
                counts.append(tracker.count)
            expected = [len(e["ordered_completed_subtasks"]) for e in build_subtask_trace(evals)]
            self.assertEqual(counts, expected)


class AnchorTest(unittest.TestCase):
    config = ladder.resolve_config({"stall_fraction": 0.25, "time_fraction": 0.5})

    def test_detectors_and_anchors(self):
        steps = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
        is_inference = [True] * 10 + [False]
        events = [{"step": 8, "subtask": "reach"}]
        detectors = ladder.detector_indices(steps, is_inference, events, 40, self.config)
        self.assertEqual(detectors["stall"], {"index": 5, "fired": True, "step": 20})
        self.assertEqual(detectors["time"]["index"], 5)
        self.assertEqual(detectors["end"]["index"], 10)
        anchors = ladder.compute_anchors(steps, is_inference, events, 5, 2)
        self.assertEqual(anchors, {"detection": 5, "last_good": 2, "last_good_minus_2": 0})

    def test_no_progress_anchors_initial_state_and_stall_falls_back_to_end(self):
        steps = [0, 4, 8, 9]
        detectors = ladder.detector_indices(steps, [True, True, True, False], [], 400, self.config)
        self.assertFalse(detectors["stall"]["fired"])
        self.assertEqual(detectors["stall"]["index"], 3)
        anchors = ladder.compute_anchors(steps, [True, True, True, False], [], 3, 2)
        self.assertEqual(anchors["last_good"], 0)
        self.assertEqual(anchors["last_good_minus_2"], 0)

    def test_progress_after_detection_is_ignored(self):
        steps = [0, 4, 8, 12, 16]
        anchors = ladder.compute_anchors(steps, [True] * 5, [{"step": 13}], 2, 2)
        self.assertEqual(anchors["last_good"], 0)

    def test_identical_interventions_are_aliased(self):
        anchors = {"detection": 4, "last_good": 4, "last_good_minus_2": 2}
        schedule = ladder.schedule_cells(ladder.DEFAULT_CELLS, anchors)
        by_cell = {(c["rung"], c["anchor"]): c for c in schedule}
        self.assertTrue(by_cell[("resample_continue", "detection")]["run"])
        for cell in (("robot_reset", "last_good"), ("world_reset", "last_good")):
            self.assertFalse(by_cell[cell]["run"])
            self.assertEqual(by_cell[cell]["canonical"], ["resample_continue", "detection"])
        self.assertTrue(by_cell[("world_reset", "last_good_minus_2")]["run"])
        anchors = {"detection": 4, "last_good": 0, "last_good_minus_2": 0}
        schedule = ladder.schedule_cells(ladder.DEFAULT_CELLS, anchors)
        by_cell = {(c["rung"], c["anchor"]): c for c in schedule}
        self.assertEqual(by_cell[("world_reset", "last_good_minus_2")]["canonical"], ["world_reset", "last_good"])

    def test_control_time_is_last_inference_before_matched_fraction(self):
        steps = [0, 4, 8, 12, 16]
        self.assertEqual(ladder.control_detection_index(steps, [True] * 4 + [False], 40, 0.5), 3)
        self.assertEqual(ladder.control_detection_index(steps, [True] * 4 + [False], 40, 0.05), 0)


class RestoreTest(unittest.TestCase):
    def test_robot_reset_touches_only_robot_joints(self):
        env = MockEnv()
        env.reset(seed=0)
        env.sim.qpos[:] = [0.25, 0.5, 0.75, 0.0]
        snap = ladder.capture_snapshot(env)
        env.sim.qpos[:] = [1.0, 3.0, 0.1, 1.0]
        ladder.restore_robot(env, snap)
        np.testing.assert_allclose(env.sim.qpos, [0.25, 0.5, 0.1, 1.0])
        ladder.restore_world(env, snap)
        self.assertEqual(ladder.state_residual(env, snap["states"]), 0.0)


class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def test_collection_records_parents_branches_and_artifacts(self):
        runtime, status = collect(self.root)
        parents = ladder.read_jsonl(self.root / ladder.PARENTS_NAME)
        branches = ladder.read_jsonl(self.root / ladder.BRANCHES_NAME)
        self.assertEqual([p["role"] for p in parents], ["success_control", "failure", "failure"])
        self.assertEqual(status["status"], "complete")
        self.assertIsNotNone(status["measured_seconds_per_env_step"])
        failure = parents[1]
        self.assertEqual(failure["anchors"], {"detection": 5, "last_good": 2, "last_good_minus_2": 0})
        self.assertEqual(failure["anchor_ordered_counts"], {"detection": 1, "last_good": 1, "last_good_minus_2": 0})
        self.assertAlmostEqual(failure["scores_at_detection"]["time"], 0.5)
        self.assertEqual(parents[0]["anchors"]["detection"], 3)
        self.assertEqual(parents[0]["detection_source"], "time_fraction")
        # Six distinct cells for every branched parent, K=2 each.
        self.assertEqual(len(branches), 3 * 6 * 2)
        self.assertTrue(all(b["source_restore_residual"] == 0.0 for b in branches))
        self.assertEqual(len({b["branch_seed"] for b in branches}), len(branches))
        self.assertTrue(all(b["seed_control"] == "policy_sampling_seed_base" for b in branches))
        for parent in parents:
            snapshot = np.load(self.root / parent["snapshot_path"])
            self.assertEqual(len(snapshot["steps"]), parent["num_snapshots"])
            self.assertFalse(snapshot["is_inference"][-1])
            self.assertTrue((self.root / parent["model_xml_path"]).is_file())
            features = np.load(self.root / parent["features_path"])
            self.assertEqual(features["features"].shape[0], parent["num_inferences"])
        policy = runtime.policies[0]
        # Features are requested for parent inferences only.
        self.assertIn(False, policy.collect_flags)
        self.assertTrue(policy.collect_safe_features)

    def test_rungs_separate_world_and_robot_failures(self):
        collect(self.root)
        parents = ladder.read_jsonl(self.root / ladder.PARENTS_NAME)
        branches = ladder.read_jsonl(self.root / ladder.BRANCHES_NAME)
        by_seed = {p["environment_seed"]: p["parent_id"] for p in parents}

        def p_success(seed, rung, anchor):
            rows = [b for b in branches if b["parent_id"] == by_seed[seed] and (b["rung"], b["anchor"]) == (rung, anchor)]
            return np.mean([b["success"] for b in rows])

        # seed 1: stuck object (world state)
        self.assertEqual(p_success(1, "resample_continue", "detection"), 0.0)
        self.assertEqual(p_success(1, "robot_reset", "last_good"), 0.0)
        self.assertEqual(p_success(1, "subtask_reprompt", "detection"), 0.0)
        self.assertEqual(p_success(1, "world_reset", "last_good"), 1.0)
        self.assertEqual(p_success(1, "world_reset", "last_good_minus_2"), 1.0)
        # seed 2: jammed wrist (robot state)
        self.assertEqual(p_success(2, "resample_continue", "detection"), 0.0)
        self.assertEqual(p_success(2, "robot_reset", "last_good"), 1.0)
        self.assertEqual(p_success(2, "robot_reset", "last_good_minus_2"), 1.0)
        # seed 0: harm control stays successful
        self.assertEqual(p_success(0, "resample_continue", "detection"), 1.0)
        world = [b for b in branches if b["rung"] == "world_reset"]
        self.assertTrue(all(b["anchor_restore_residual"] == 0.0 for b in world))

    def test_reprompt_uses_subtask_instruction_then_task_instruction(self):
        runtime, _ = collect(self.root)
        branches = ladder.read_jsonl(self.root / ladder.BRANCHES_NAME)
        control = [b for b in branches if b["role"] == "success_control" and b["rung"] == "subtask_reprompt"]
        self.assertTrue(control)
        for branch in control:
            self.assertEqual(branch["target_subtask"], "move")
            self.assertEqual(branch["reprompt_instruction"], "Move the object.")
            self.assertFalse(branch["reprompt_equivalent_to_task_prompt"])
            self.assertTrue(branch["success"])
            self.assertIsNotNone(branch["reprompt_steps"])
        self.assertIn("Move the object.", runtime.policies[0].instructions)

    def test_resume_skips_completed_parents_and_rejects_changed_plan(self):
        collect(self.root)
        before = len(ladder.read_jsonl(self.root / ladder.BRANCHES_NAME))
        with self.assertRaises(FileExistsError):
            collect(self.root)
        runtime = MockRuntime()
        ladder.run_collection(mock_config(), self.root, runtime=runtime, resume=True, log=lambda *_: None)
        self.assertEqual(len(ladder.read_jsonl(self.root / ladder.BRANCHES_NAME)), before)
        with self.assertRaises(ValueError):
            ladder.run_collection(mock_config(k_branches=3), self.root, runtime=MockRuntime(), resume=True, log=lambda *_: None)

    def test_partial_parent_branches_are_ignored_by_analysis(self):
        collect(self.root)
        stray = dict(ladder.read_jsonl(self.root / ladder.BRANCHES_NAME)[0], attempt_uid="crashed", success=False)
        ladder.append_jsonl(self.root / ladder.BRANCHES_NAME, stray)
        parents, branches = ladder.load_run([self.root])
        self.assertEqual(len(branches), 3 * 6 * 2)

    def test_errors_are_recorded_and_bounded(self):
        class Broken(MockRuntime):
            def make_policy(self, env):
                policy = super().make_policy(env)
                def boom(*args, **kwargs):
                    raise RuntimeError("policy server down")
                policy.pop_inference_record = boom
                return policy

        with self.assertRaises(RuntimeError):
            ladder.run_collection(mock_config(max_errors=2), self.root, runtime=Broken(), log=lambda *_: None)
        errors = ladder.read_jsonl(self.root / ladder.ERRORS_NAME)
        self.assertEqual(len(errors), 2)
        self.assertIn("policy server down", errors[0]["error"])


class AnalysisTest(unittest.TestCase):
    def test_analysis_of_mock_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            collect(root)
            summary, rows = ladder.analyze([root], n_boot=200)
            path = ladder.write_analysis(summary, rows, Path(tmp) / "analysis")
            text = path.read_text()
        self.assertIn("Failure attribution", text)
        classes = {row["class"]: row["parents"] for row in summary["attribution"]}
        self.assertEqual(classes["world_state"], 1)
        self.assertEqual(classes["robot_state"], 1)
        self.assertEqual(classes["competence"], 0)
        ladder_rows = {
            (r["role"], r["task"], r["rung"], r["anchor"], r["metric"]): r for r in summary["ladder"]
        }
        world = ladder_rows[("failure", "ALL", "world_reset", "last_good", "p_success")]
        self.assertEqual(world["estimate"], 1.0)
        harm = ladder_rows[("success_control", "ALL", "resample_continue", "detection", "p_success")]
        self.assertEqual(harm["estimate"], 1.0)
        contrast = {(c["rung"], c["anchor"]): c for c in summary["contrasts_vs_resample"]}
        self.assertEqual(contrast[("world_reset", "last_good")]["estimate"], 1.0)
        decay = {row["distance_inferences"]: row for row in summary["decay"]}
        self.assertEqual(decay["0"]["p_success"], 0.0)
        self.assertEqual(decay["3-5"]["p_success"], 1.0)
        self.assertEqual(summary["restore_residual_max"], 0.0)

    def test_external_scores_join_on_detection_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            collect(root)
            parents = [p for p in ladder.read_jsonl(root / ladder.PARENTS_NAME) if p["role"] == "failure"]
            scores = [{"parent_id": p["parent_id"], "snapshot_index": p["anchors"]["detection"], "score": float(i), "name": "safe"} for i, p in enumerate(parents)]
            scores.append({"parent_id": parents[0]["parent_id"], "snapshot_index": 0, "score": 9.0, "name": "safe"})
            summary, _ = ladder.analyze([root], scores=scores, n_boot=50)
        rows = {(r["score"], r["target"]): r for r in summary["score_correlations"]}
        self.assertEqual(rows[("safe", "p_best_cell")]["parents"], 2)

    def test_bootstrap_and_spearman(self):
        point, low, high = ladder.cluster_bootstrap({str(i): float(i % 2) for i in range(40)}, n_boot=500)
        self.assertAlmostEqual(point, 0.5)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        strata = {str(i): "a" if i < 20 else "b" for i in range(40)}
        values = {str(i): 1.0 if i < 20 else 0.0 for i in range(40)}
        point, low, high = ladder.cluster_bootstrap(values, strata=strata, n_boot=200)
        self.assertEqual((point, low, high), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(ladder.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(ladder.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        np.testing.assert_allclose(ladder.rankdata([3, 1, 3, 2]), [3.5, 1, 3.5, 2])
        self.assertIsNone(ladder.spearman([1, 1, 1], [1, 2, 3]))


class PlanTest(unittest.TestCase):
    def load_pilot(self):
        return json.loads((REPO / "scripts" / "recoverability_ladder_xr1_pilot.json").read_text())

    def test_xr1_pilot_dry_run(self):
        plan = ladder.build_plan(self.load_pilot())
        self.assertEqual(len(plan["tasks"]), 5)
        self.assertEqual(plan["warnings"], [])
        for row in plan["per_task"]:
            self.assertTrue(0.2 <= row["expected_success_rate"] <= 0.8)
            self.assertEqual(row["budget_steps"], int(np.ceil(row["horizon"] / 2)))
        totals = plan["totals"]
        self.assertGreater(totals["upper"]["gpu_hours"], totals["expected"]["gpu_hours"])
        self.assertAlmostEqual(totals["upper"]["gpu_hours"], 2 * totals["upper"]["worker_hours"], places=1)
        self.assertEqual(plan["throughput_source"], "assumed")

    def test_calibration_and_cli(self):
        plan = ladder.build_plan(self.load_pilot(), calibration={"measured_seconds_per_env_step": 0.12})
        self.assertEqual(plan["throughput_source"], "measured")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli = ladder.main(["plan", "--config", str(REPO / "scripts" / "recoverability_ladder_xr1_pilot.json"), "--k-branches", "2", "--output", str(out)])
            self.assertEqual(cli["k_branches"], 2)
            self.assertTrue(out.is_file())

    def test_rate_outside_band_warns_and_bad_cells_rejected(self):
        config = self.load_pilot()
        config["expected_success_rates"]["ArrangeTea"] = 0.9
        plan = ladder.build_plan(config)
        self.assertTrue(any("ArrangeTea" in warning for warning in plan["warnings"]))
        with self.assertRaises(ValueError):
            ladder.resolve_config({"cells": [["teleport", "detection"]]})
        with self.assertRaises(ValueError):
            ladder.resolve_config({"unknown_option": 1})

    def test_task_sharding_is_a_partition(self):
        horizons = {"A": 10, "B": 30, "C": 20, "D": 5}
        shards = [ladder.shard_tasks(list(horizons), horizons, 2, i) for i in range(2)]
        self.assertEqual(sorted(shards[0] + shards[1]), sorted(horizons))
        self.assertFalse(set(shards[0]) & set(shards[1]))


if __name__ == "__main__":
    unittest.main()
