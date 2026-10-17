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


class StageMockEnv(MockEnv):
    """Reach then move behind an unreachable lid precondition (CTS-like).

    ``lid_open`` is never true (like CuttingToolSelection ``drawer_open``) and
    ``table_clear`` is true from reset. seed % 4 == 0: success at step 15;
    1: object stuck; 2: wrist jammed; 3: knocked back, so ``reach`` regresses
    after latching and the jammed wrist keeps it false.
    """

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.fail_mode = (seed or 0) % 4
        return self.get_current_observation(), {"subtask_eval": self.get_subtask_progress()}

    def get_subtask_progress(self):
        reach = bool(self.sim.qpos[0] >= 1.0)
        move = bool(self.sim.qpos[2] >= 1.0)
        predicates = {
            "lid_open": {"value": False, "required": True, "stage": "precondition", "description": "Open the lid."},
            "table_clear": {"value": True, "required": True, "stage": "placement", "description": "Clear the table."},
            "reach": {"value": reach, "required": True, "stage": "placement", "description": "Reach the object."},
            "move": {"value": move, "required": True, "stage": "placement", "description": "Move the object."},
            "task_success": {"value": reach and move, "required": False},
        }
        return {
            "task_name": "MockTask",
            "predicates": predicates,
            "required_predicates": ["lid_open", "table_clear", "reach", "move"],
            "task_success": reach and move,
        }

    def step(self, action):
        self.steps += 1
        qpos = self.sim.qpos
        jammed = qpos[1] > 2.0
        if not jammed:
            qpos[0] = min(1.0, qpos[0] + action[0])
        if qpos[0] >= 1.0 and qpos[3] < 0.5 and not jammed:
            qpos[2] = min(1.0, qpos[2] + action[1])
        if not self.triggered and self.steps == TRIGGER_STEP and self.fail_mode:
            self.triggered = True
            if self.fail_mode == 1:
                qpos[3] = 1.0
            elif self.fail_mode == 2:
                qpos[1] = 3.0
            else:
                qpos[0] = 0.0
                qpos[1] = 3.0
        self.sim.time += 0.05
        progress = self.get_subtask_progress()
        success = progress["task_success"]
        info = {"success": success, "subtask_eval": progress}
        return self.get_current_observation(), float(success), False, False, info


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
    def __init__(self, horizon=40, env_class=MockEnv):
        self.horizon = horizon
        self.env_class = env_class
        self.envs = []
        self.policies = []

    def task_horizon(self, task_name):
        return self.horizon

    def make_env(self, task_name, seed):
        env = self.env_class(seed)
        self.envs.append(env)
        return env

    def make_policy(self, env):
        policy = MockPolicy()
        self.policies.append(policy)
        return policy


class GymnasiumLikeWrapper:
    """Forwards only reset/step/close, like gymnasium>=1.0 wrappers."""

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()


class WrappedRuntime(MockRuntime):
    def __init__(self, horizon=40):
        super().__init__(horizon)
        self.env_instructions = []

    def make_env(self, task_name, seed):
        env = super().make_env(task_name, seed)
        setter = env.set_task_description

        def record(text):
            self.env_instructions.append(text)
            setter(text)

        env.set_task_description = record
        return GymnasiumLikeWrapper(GymnasiumLikeWrapper(env))


def mock_config(**overrides):
    """Schema-1 protocol (v1 cells, stall detector) on the schema-2 collector."""
    config = {
        "tasks": ["MockTask"],
        "base_seed": 0,
        "parents_per_task": 6,
        "failed_parents_per_task": 2,
        "success_controls_per_task": 1,
        "k_branches": 2,
        "cells": [list(cell) for cell in ladder.V1_CELLS],
        "detector": "stall",
        "stall_fraction": 0.25,
        "budget_fraction": 0.5,
        "reprompt_level": "subtask",
        "record_policy_features": True,
        "policy_module": "tests.mock:make_policy",
    }
    config.update(overrides)
    return config


V2_CELLS = [
    ["resample_continue", "detection"],
    ["robot_reset", "last_good"],
    ["subtask_reprompt", "detection"],
    ["world_reset", "last_good"],
    ["resample_continue", "stage_entry"],
    ["resample_continue", "stage_entry_plus_1"],
    ["resample_continue", "stage_entry_plus_16"],
    ["subtask_reprompt", "stage_entry"],
]


def stage_config(**overrides):
    """Schema-2 protocol on StageMockEnv (H=40, one inference every 4 steps)."""
    config = {
        "tasks": ["MockTask"],
        "base_seed": 0,
        "parents_per_task": 8,
        "failed_parents_per_task": 3,
        "success_controls_per_task": 1,
        "k_branches": 2,
        "cells": V2_CELLS,
        "control_cells": [list(cell) for cell in ladder.DEFAULT_CONTROL_CELLS],
        "detector": "overdue_or_regression",
        "stall_fraction": 0.25,
        "stage_overdue_min_steps": 4,
        "regression_min_steps": 4,
        "exclude_predicates": {"MockTask": ["lid_open"]},
        "budget_mode": "remaining_horizon",
        # MockTask is not a registered composite task, so atomic reprompts fall back to the task prompt.
        "reprompt_level": "subtask",
        "record_policy_features": False,
        "policy_module": "tests.mock:make_policy",
    }
    config.update(overrides)
    return config


def collect(tmp, **overrides):
    runtime = MockRuntime()
    status = ladder.run_collection(mock_config(**overrides), tmp, runtime=runtime, log=lambda *_: None)
    return runtime, status


def collect_stage(tmp, config=None, **overrides):
    runtime = MockRuntime(env_class=StageMockEnv)
    status = ladder.run_collection(config or stage_config(**overrides), tmp, runtime=runtime, log=lambda *_: None)
    return runtime, status


def by_seed(root):
    return {p["environment_seed"]: p for p in ladder.read_jsonl(Path(root) / ladder.PARENTS_NAME)}


def cell_success(branches, parent_id, rung, anchor):
    rows = [b for b in branches if b["parent_id"] == parent_id and (b["rung"], b["anchor"]) == (rung, anchor)]
    return float(np.mean([b["success"] for b in rows])) if rows else None


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
        schedule = ladder.schedule_cells(ladder.V1_CELLS, anchors)
        by_cell = {(c["rung"], c["anchor"]): c for c in schedule}
        self.assertTrue(by_cell[("resample_continue", "detection")]["run"])
        for cell in (("robot_reset", "last_good"), ("world_reset", "last_good")):
            self.assertFalse(by_cell[cell]["run"])
            self.assertEqual(by_cell[cell]["canonical"], ["resample_continue", "detection"])
        self.assertTrue(by_cell[("world_reset", "last_good_minus_2")]["run"])
        anchors = {"detection": 4, "last_good": 0, "last_good_minus_2": 0}
        schedule = ladder.schedule_cells(ladder.V1_CELLS, anchors)
        by_cell = {(c["rung"], c["anchor"]): c for c in schedule}
        self.assertEqual(by_cell[("world_reset", "last_good_minus_2")]["canonical"], ["world_reset", "last_good"])

    def test_control_time_is_last_inference_before_matched_fraction(self):
        steps = [0, 4, 8, 12, 16]
        self.assertEqual(ladder.control_detection_index(steps, [True] * 4 + [False], 40, 0.5), 3)
        self.assertEqual(ladder.control_detection_index(steps, [True] * 4 + [False], 40, 0.05), 0)


class TimeCurveAnchorTest(unittest.TestCase):
    """Anchors at stage entry + {0, 4, 8, 16} inferences, clipped to detection."""

    steps = [16 * i for i in range(60)] + [1000]
    is_inference = [True] * 60 + [False]
    events = [{"step": 80, "subtask": "a", "ordered_count": 1}]

    def test_offsets_are_inferences_after_stage_entry_clipped_to_detection(self):
        anchors = ladder.compute_anchors(self.steps, self.is_inference, self.events, 18, 2, offsets=[4, 8, 16])
        self.assertEqual(anchors["stage_entry"], anchors["last_good"])
        self.assertEqual(anchors["stage_entry"], 5)
        self.assertEqual(anchors["stage_entry_plus_4"], 9)
        self.assertEqual(anchors["stage_entry_plus_8"], 13)
        self.assertEqual(anchors["stage_entry_plus_16"], 18)
        self.assertEqual(ladder.cell_offsets(ladder.DEFAULT_CELLS), [4, 8, 16])

    def test_clipped_anchors_alias_to_detection_and_resets_keep_last_good(self):
        anchors = ladder.compute_anchors(self.steps, self.is_inference, self.events, 10, 2, offsets=[4, 8, 16])
        budget = lambda start: ladder.branch_budget(1000, start, ladder.resolve_config({"budget_mode": "remaining_horizon"}))
        schedule = ladder.schedule_cells(ladder.DEFAULT_CELLS, anchors, budget_fn=budget, steps=self.steps)
        by_cell = {(c["rung"], c["anchor"]): c for c in schedule}
        for rung in ("resample_continue", "subtask_reprompt"):
            self.assertTrue(by_cell[(rung, "stage_entry_plus_4")]["run"])
            for anchor in ("stage_entry_plus_8", "stage_entry_plus_16"):
                self.assertFalse(by_cell[(rung, anchor)]["run"])
                self.assertEqual(by_cell[(rung, anchor)]["canonical"], [rung, "detection"])
        world = by_cell[("world_reset", "last_good")]
        robot = by_cell[("robot_reset", "last_good")]
        # Resets are applied at detection: detection clock, last-good state.
        self.assertEqual((world["start_index"], world["clock_index"]), (5, 10))
        self.assertEqual((robot["start_index"], robot["robot_index"], robot["clock_index"]), (10, 5, 10))
        # With the remaining-horizon clock, world reset at last good is not resampling from stage entry.
        self.assertTrue(world["run"])
        self.assertEqual(world["budget_steps"], 1000 - 160)
        self.assertEqual(by_cell[("resample_continue", "stage_entry")]["budget_steps"], 1000 - 80)


class BudgetTest(unittest.TestCase):
    def test_budget_modes(self):
        fixed = ladder.resolve_config({"budget_fraction": 0.5})
        self.assertEqual(ladder.branch_budget(1950, 1500, fixed), 975)
        remaining = ladder.resolve_config({"budget_mode": "remaining_horizon"})
        self.assertEqual(ladder.branch_budget(1950, 400, remaining), 1550)
        self.assertEqual(ladder.branch_budget(1950, 0, remaining), 1950)
        self.assertEqual(ladder.branch_budget(1950, 1950, remaining), 1)
        capped = ladder.resolve_config({"budget_mode": "remaining_horizon", "max_budget_steps": {"A": 1000}})
        self.assertEqual(ladder.branch_budget(4350, 0, capped, "A"), 1000)
        self.assertEqual(ladder.branch_budget(4350, 0, capped, "B"), 4350)
        with self.assertRaises(ValueError):
            ladder.resolve_config({"budget_mode": "unbounded"})

    def test_fixed_budget_aliases_world_reset_with_resample_from_the_same_state(self):
        anchors = {"detection": 6, "last_good": 2, "last_good_minus_2": 0, "stage_entry": 2}
        cells = [("resample_continue", "stage_entry"), ("world_reset", "last_good")]
        steps = [4 * i for i in range(8)]
        fixed = ladder.schedule_cells(cells, anchors, budget_fn=lambda s: 20, steps=steps)
        self.assertEqual(fixed[1]["canonical"], ["resample_continue", "stage_entry"])
        config = ladder.resolve_config({"budget_mode": "remaining_horizon"})
        remaining = ladder.schedule_cells(cells, anchors, budget_fn=lambda s: ladder.branch_budget(40, s, config), steps=steps)
        self.assertTrue(remaining[1]["run"])
        self.assertEqual([c["budget_steps"] for c in remaining], [32, 16])


class StageOverdueDetectorTest(unittest.TestCase):
    config = ladder.resolve_config({"stall_fraction": 0.25, "stage_overdue_min_steps": 4, "stage_min_examples": 3})
    steps = [4 * i for i in range(10)] + [40]
    is_inference = [True] * 10 + [False]
    events = [{"step": 0, "subtask": "a", "ordered_count": 1}, {"step": 8, "subtask": "b", "ordered_count": 2}]
    counts = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    required = ["a", "b", "c"]

    def calibration(self, durations):
        return {"tasks": {"T": {"required_subtasks": self.required, "stages": {"c": {"durations": durations}}}}}

    def detect(self, thresholds):
        return ladder.detector_indices(
            self.steps, self.is_inference, self.events, 40, self.config,
            ordered_counts=self.counts, regressions=[], stage_thresholds=thresholds, required=self.required,
        )

    def test_quantile_of_successful_stage_durations(self):
        thresholds, status = ladder.stage_thresholds(self.calibration([4, 5, 6, 7, 20]), "T", self.required, self.config)
        self.assertEqual(status, "partial")
        self.assertIsNone(thresholds["a"])
        self.assertAlmostEqual(thresholds["c"], float(np.quantile([4, 5, 6, 7, 20], 0.9)))
        overdue = self.detect({"c": 6.0})["stage_overdue"]
        self.assertEqual((overdue["index"], overdue["rule"], overdue["stage"]), (4, "stage_quantile", "c"))
        self.assertEqual(overdue["time_in_stage"], 8)
        self.assertEqual(overdue["stage_entry_step"], 8)
        # The floor stops a zero-duration stage from firing at its first inference.
        thresholds, _ = ladder.stage_thresholds(self.calibration([0, 0, 0]), "T", self.required, self.config)
        self.assertEqual(thresholds["c"], 4.0)

    def test_falls_back_to_the_stall_rule_below_min_examples(self):
        thresholds, status = ladder.stage_thresholds(self.calibration([4, 5]), "T", self.required, self.config)
        self.assertEqual((thresholds["c"], status), (None, "fallback"))
        overdue = self.detect(thresholds)["stage_overdue"]
        # stall_fraction 0.25 * H 40 = 10 steps in the stage.
        self.assertEqual((overdue["index"], overdue["rule"], overdue["threshold_steps"]), (5, "stall_fallback", 10))

    def test_mismatched_or_missing_calibration_is_not_used(self):
        self.assertEqual(ladder.stage_thresholds(self.calibration([5] * 5), "T", ["a", "c"], self.config)[1], "required_mismatch")
        self.assertEqual(ladder.stage_thresholds(self.calibration([5] * 5), "U", self.required, self.config)[1], "task_missing")
        self.assertEqual(ladder.stage_thresholds(None, "T", self.required, self.config), ({}, "uncalibrated"))

    def test_completed_tasks_and_unfired_detectors(self):
        detectors = ladder.detector_indices(
            self.steps, self.is_inference, self.events, 40, self.config,
            ordered_counts=[3] * 11, regressions=[], stage_thresholds={}, required=self.required,
        )
        self.assertFalse(detectors["stage_overdue"]["fired"])
        self.assertEqual(detectors["overdue_or_regression"]["index"], 10)


class RegressionDetectorTest(unittest.TestCase):
    config = ladder.resolve_config({"regression_min_steps": 8, "stall_fraction": 0.5})
    steps = [4 * i for i in range(10)] + [40]
    is_inference = [True] * 10 + [False]

    def detect(self, regressions, counts=None):
        return ladder.detector_indices(
            self.steps, self.is_inference, [], 40, self.config, ordered_counts=counts or [0] * 11,
            regressions=regressions, stage_thresholds={}, required=["a", "b"],
        )

    def test_persistent_regression_fires_after_min_steps(self):
        event = {"subtask": "a", "ordered_index": 0, "eligible": True, "start_step": 9, "end_step": None}
        detectors = self.detect([event])
        self.assertEqual(detectors["regression"]["index"], 5)
        self.assertEqual(detectors["regression"]["stage"], "a")
        self.assertEqual(detectors["regression"]["regression_start_step"], 9)
        # Ties and earlier regressions win over the stage-overdue detector.
        self.assertEqual(detectors["overdue_or_regression"]["rule"], "regression")

    def test_short_and_ineligible_regressions_do_not_fire(self):
        brief = {"subtask": "a", "ordered_index": 0, "eligible": True, "start_step": 9, "end_step": 15}
        ignored = {"subtask": "a", "ordered_index": 0, "eligible": False, "start_step": 2, "end_step": None}
        self.assertFalse(self.detect([brief, ignored])["regression"]["fired"])

    def test_combined_detector_takes_the_earlier_rule(self):
        late = {"subtask": "a", "ordered_index": 0, "eligible": True, "start_step": 24, "end_step": None}
        detectors = self.detect([late])
        # stall fallback: 0.5 * 40 = 20 steps without progress.
        self.assertEqual(detectors["stage_overdue"]["index"], 5)
        self.assertEqual(detectors["regression"]["index"], 8)
        self.assertEqual(detectors["overdue_or_regression"]["rule"], "stall_fallback")


class PredicateRecorderTest(unittest.TestCase):
    def evaluation(self, **values):
        stages = {"door": "precondition", "a": "placement", "b": "placement"}
        return {
            "predicates": {name: {"value": values.get(name, False), "stage": stage} for name, stage in stages.items()},
            "required_predicates": ["door", "a", "b"],
            "task_success": False,
        }

    def test_durable_count_and_regression_events(self):
        config = ladder.resolve_config({})
        tracker = ladder.OrderedProgress()
        recorder = ladder.PredicateRecorder("T", config)
        sequence = [
            dict(door=True),
            dict(door=True, a=True),
            dict(a=True),  # the door closes: a precondition need not persist
            dict(),  # a regresses
            dict(a=True),
        ]
        durable = []
        for step, values in enumerate(sequence):
            evaluation = self.evaluation(**values)
            tracker.update(evaluation)
            recorder.observe(evaluation, step, tracker)
            durable.append(recorder.durable_count(tracker))
        self.assertEqual(tracker.count, 2)
        self.assertEqual(durable, [1, 2, 2, 1, 2])
        events = {(e["subtask"], e["eligible"]): e for e in recorder.regressions}
        self.assertEqual(events[("door", False)]["start_step"], 2)
        self.assertEqual((events[("a", True)]["start_step"], events[("a", True)]["end_step"]), (3, 4))
        self.assertEqual(recorder.first_true, {"door": 0, "a": 1, "b": None})
        self.assertEqual(recorder.at_reset, {"door": True, "a": False, "b": False})


class OverrideTest(unittest.TestCase):
    def test_effective_required_and_unknown_override_names(self):
        config = ladder.resolve_config(
            {"required_subtask_overrides": {"T": ["b", "a", "c"]}, "exclude_predicates": {"T": ["c"], "U": ["x"]}}
        )
        self.assertEqual(ladder.effective_required("T", ["a", "b", "c"], config), ["b", "a"])
        self.assertEqual(ladder.effective_required("U", ["x", "y"], config), ["y"])
        self.assertEqual(ladder.effective_required("V", ["x", "y"], config), ["x", "y"])
        with self.assertRaises(ValueError):
            ladder._check_required("T", ["a", "zzz"], {"predicates": {"a": {}}})
        with self.assertRaises(ValueError):
            ladder.resolve_config({"exclude_predicates": {"T": "drawer_open"}})


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
        self.assertEqual(failure["schema_version"], 2)
        self.assertEqual(failure["anchors"], {"detection": 5, "last_good": 2, "last_good_minus_2": 0, "stage_entry": 2})
        self.assertEqual(failure["anchor_ordered_counts"], {"detection": 1, "last_good": 1, "last_good_minus_2": 0, "stage_entry": 1})
        self.assertAlmostEqual(failure["scores_at_detection"]["time"], 0.5)
        self.assertEqual(parents[0]["anchors"]["detection"], 3)
        self.assertEqual(parents[0]["detection_source"], "time_fraction")
        # Six distinct cells for every branched parent, K=2 each.
        self.assertEqual(len(branches), 3 * 6 * 2)
        self.assertTrue(all(b["source_restore_residual"] == 0.0 for b in branches))
        self.assertEqual(len({b["branch_seed"] for b in branches}), len(branches))
        self.assertTrue(all(b["seed_control"] == "policy_sampling_seed_base" for b in branches))
        self.assertTrue(all(b["budget_steps"] == 20 and b["budget_mode"] == "fixed_fraction" for b in branches))
        for parent in parents:
            snapshot = np.load(self.root / parent["snapshot_path"])
            self.assertEqual(len(snapshot["steps"]), parent["num_snapshots"])
            self.assertFalse(snapshot["is_inference"][-1])
            self.assertEqual(snapshot["predicate_values"].shape, (parent["num_snapshots"], 3))
            self.assertEqual(list(snapshot["predicate_names"]), ["reach", "move", "task_success"])
            self.assertTrue((self.root / parent["model_xml_path"]).is_file())
            features = np.load(self.root / parent["features_path"])
            self.assertEqual(features["features"].shape[0], parent["num_inferences"])
        policy = runtime.policies[0]
        # Features are requested for parent inferences only.
        self.assertIn(False, policy.collect_flags)
        self.assertTrue(policy.collect_safe_features)

    def test_rungs_separate_world_and_robot_failures(self):
        collect(self.root)
        parents = by_seed(self.root)
        branches = ladder.read_jsonl(self.root / ladder.BRANCHES_NAME)

        def p_success(seed, rung, anchor):
            return cell_success(branches, parents[seed]["parent_id"], rung, anchor)

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

    def test_branches_get_live_observations_through_non_forwarding_wrappers(self):
        collect(self.root / "plain")
        runtime = WrappedRuntime()
        status = ladder.run_collection(mock_config(), self.root / "wrapped", runtime=runtime, log=lambda *_: None)
        self.assertEqual(status["num_errors"], 0)
        outcome = lambda rows: sorted((b["parent_id"], b["rung"], b["anchor"], b["k"], b["success"], b["steps_used"]) for b in rows)
        plain = ladder.read_jsonl(self.root / "plain" / ladder.BRANCHES_NAME)
        wrapped = ladder.read_jsonl(self.root / "wrapped" / ladder.BRANCHES_NAME)
        self.assertEqual(len(wrapped), 3 * 6 * 2)
        self.assertEqual(outcome(wrapped), outcome(plain))
        # The reprompt reaches the adapter's language override beneath the wrappers.
        reprompts = {b["reprompt_instruction"] for b in wrapped if b["reprompt_instruction"]}
        self.assertEqual(reprompts, {"Move the object."})
        self.assertEqual(set(runtime.env_instructions), reprompts)
        self.assertTrue(all(env.override is None for env in runtime.envs))

    def test_live_observation_refuses_missing_getter(self):
        class Opaque:
            env = None

        with self.assertRaises(RuntimeError):
            ladder.live_observation(Opaque())
        inner = MockEnv()
        inner.reset(seed=0)
        self.assertEqual(ladder.live_observation(GymnasiumLikeWrapper(inner))["state"].tolist(), [0.0] * 4)

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
        for changed in ({"k_branches": 3}, {"budget_mode": "remaining_horizon"}, {"exclude_predicates": {"MockTask": ["move"]}}):
            with self.assertRaises(ValueError):
                ladder.run_collection(mock_config(**changed), self.root, runtime=MockRuntime(), resume=True, log=lambda *_: None)

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

    def test_unknown_override_names_stop_the_parent(self):
        config = mock_config(max_errors=1, required_subtask_overrides={"MockTask": ["reach", "lift"]})
        with self.assertRaises(RuntimeError):
            ladder.run_collection(config, self.root, runtime=MockRuntime(), log=lambda *_: None)
        self.assertIn("unknown predicates ['lift']", ladder.read_jsonl(self.root / ladder.ERRORS_NAME)[0]["error"])


class StageCollectionTest(unittest.TestCase):
    """Schema-2 protocol: overrides, durable progress, detectors, time curve, budgets."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def naturals_and_calibration(self):
        naturals = self.root / "naturals"
        collect_stage(naturals, parents_only=True, parents_per_task=12, base_seed=0)
        calibration = ladder.build_stage_calibration([naturals], stage_config())
        path = self.root / "stage_calibration.json"
        ladder.atomic_write_json(path, calibration)
        return naturals, calibration, path

    def test_unreachable_precondition_blocks_progress_until_excluded(self):
        collect_stage(self.root / "raw", exclude_predicates={}, parents_only=True, parents_per_task=4)
        raw = by_seed(self.root / "raw")
        self.assertEqual(raw[1]["required_subtasks"], ["lid_open", "table_clear", "reach", "move"])
        self.assertEqual(raw[1]["progress_events"], [])
        self.assertEqual(raw[1]["stage_at_detection"], "lid_open")
        # The success latches every stage at once, which the audit reports as a blocker.
        self.assertEqual({e["step"] for e in raw[0]["progress_events"]}, {15})
        collect_stage(self.root / "fixed", parents_only=True, parents_per_task=4)
        fixed = by_seed(self.root / "fixed")
        failure = fixed[1]
        self.assertEqual(failure["env_required_subtasks"], ["lid_open", "table_clear", "reach", "move"])
        self.assertEqual(failure["required_subtasks"], ["table_clear", "reach", "move"])
        self.assertEqual(failure["excluded_predicates"], ["lid_open"])
        # table_clear is already true at reset and latches at step 0.
        self.assertEqual([(e["subtask"], e["step"]) for e in failure["progress_events"]], [("table_clear", 0), ("reach", 8)])
        self.assertEqual(failure["stage_at_detection"], "move")
        self.assertEqual(failure["predicate_first_true_step"]["lid_open"], None)
        self.assertTrue(failure["predicate_values_at_reset"]["table_clear"])
        self.assertTrue(all(p["role"] == "unbranched" for p in fixed.values()))
        self.assertFalse((self.root / "fixed" / ladder.BRANCHES_NAME).exists())

    def test_regression_detection_durable_count_and_regressed_reprompt(self):
        collect_stage(self.root / "run")
        parents = by_seed(self.root / "run")
        branches = ladder.read_jsonl(self.root / "run" / ladder.BRANCHES_NAME)
        knocked = parents[3]
        self.assertEqual(knocked["role"], "failure")
        self.assertEqual(knocked["detection_rule"], "regression")
        self.assertEqual(knocked["stage_at_detection"], "reach")
        self.assertEqual(knocked["latched_stage_at_detection"], "move")
        self.assertEqual(knocked["anchors"]["detection"], 4)
        self.assertEqual(knocked["regressed_at_detection"], ["reach"])
        self.assertEqual(knocked["regression_events"][0]["start_step"], 12)
        # Latched progress keeps the knocked-back stage; durable progress drops it.
        self.assertEqual(knocked["ordered_counts"][2:5], [2, 2, 2])
        self.assertEqual(knocked["durable_counts"][2:5], [2, 1, 1])
        self.assertEqual((knocked["anchor_ordered_counts"]["detection"], knocked["anchor_durable_counts"]["detection"]), (2, 1))
        self.assertAlmostEqual(knocked["scores_at_detection"]["durable_progress"], 1 / 3)
        reprompts = [b for b in branches if b["parent_id"] == knocked["parent_id"] and (b["rung"], b["anchor"]) == ("subtask_reprompt", "detection")]
        self.assertTrue(all(b["target_subtask"] == "reach" and b["reprompt_target_source"] == "regression" for b in reprompts))
        self.assertEqual(cell_success(branches, knocked["parent_id"], "robot_reset", "last_good"), 1.0)
        self.assertEqual(cell_success(branches, knocked["parent_id"], "resample_continue", "detection"), 0.0)
        stuck = parents[1]
        self.assertEqual((stuck["detection_rule"], stuck["anchors"]["detection"]), ("stall_fallback", 5))
        self.assertEqual(stuck["stage_calibration_status"], "uncalibrated")
        self.assertEqual(stuck["durable_counts"], stuck["ordered_counts"])

    def test_calibrated_stage_overdue_detection(self):
        naturals, calibration, path = self.naturals_and_calibration()
        entry = calibration["tasks"]["MockTask"]
        self.assertEqual(entry["required_subtasks"], ["table_clear", "reach", "move"])
        self.assertEqual(entry["used_parents"], 3)
        self.assertEqual(entry["stages"]["move"]["durations"], [7, 7, 7])
        self.assertEqual(entry["stages"]["reach"]["durations"], [8, 8, 8])
        self.assertEqual(entry["stages"]["table_clear"]["durations"], [0, 0, 0])
        collect_stage(self.root / "run", stage_calibration=str(path))
        parents = by_seed(self.root / "run")
        stuck = parents[1]
        self.assertEqual(stuck["stage_calibration_status"], "calibrated")
        self.assertEqual(stuck["detection_rule"], "stage_quantile")
        self.assertEqual(stuck["anchors"]["detection"], 4)
        self.assertEqual(stuck["stage_thresholds"]["move"], 7.0)
        self.assertAlmostEqual(stuck["scores_at_detection"]["stage_overdue_ratio"], 8 / 7)
        # No false alarm on the success: the stage-overdue detector stays quiet.
        self.assertFalse(parents[0]["detectors"]["stage_overdue"]["fired"])
        self.assertFalse(parents[0]["detector_fired_on_success"])
        plan = json.loads((self.root / "run" / ladder.PLAN_NAME).read_text())
        self.assertEqual(plan["stage_calibration_sha256"], ladder.load_stage_calibration(path)["_sha256"])
        # Resume refuses a different calibration file.
        other = self.root / "other.json"
        ladder.atomic_write_json(other, {**calibration, "quantile_hint": 0.5})
        with self.assertRaises(ValueError):
            ladder.run_collection(
                stage_config(stage_calibration=str(other)), self.root / "run",
                runtime=MockRuntime(env_class=StageMockEnv), resume=True, log=lambda *_: None,
            )

    def test_time_curve_cells_and_remaining_horizon_budgets(self):
        collect_stage(self.root / "run")
        parents = by_seed(self.root / "run")
        branches = ladder.read_jsonl(self.root / "run" / ladder.BRANCHES_NAME)
        stuck = parents[1]
        self.assertEqual(stuck["anchors"]["stage_entry"], 2)
        self.assertEqual(stuck["anchors"]["stage_entry_plus_1"], 3)
        self.assertEqual(stuck["anchors"]["stage_entry_plus_16"], 5)
        cells = {(c["rung"], c["anchor"]): c for c in stuck["cells"]}
        self.assertEqual(cells[("resample_continue", "stage_entry_plus_16")]["canonical"], ["resample_continue", "detection"])
        pid = stuck["parent_id"]
        # Before the scripted failure the state is recoverable; after it, not.
        self.assertEqual(cell_success(branches, pid, "resample_continue", "stage_entry"), 1.0)
        self.assertEqual(cell_success(branches, pid, "resample_continue", "stage_entry_plus_1"), 0.0)
        self.assertEqual(cell_success(branches, pid, "resample_continue", "detection"), 0.0)
        self.assertEqual(cell_success(branches, pid, "world_reset", "last_good"), 1.0)
        rows = {(b["rung"], b["anchor"]): b for b in branches if b["parent_id"] == pid}
        self.assertEqual(rows[("resample_continue", "stage_entry")]["budget_steps"], 40 - 8)
        self.assertEqual(rows[("resample_continue", "detection")]["budget_steps"], 40 - 20)
        self.assertEqual(rows[("world_reset", "last_good")]["budget_steps"], 40 - 20)
        self.assertEqual(rows[("world_reset", "last_good")]["budget_start_step"], 20)
        self.assertTrue(all(b["budget_mode"] == "remaining_horizon" for b in branches))
        self.assertEqual(rows[("resample_continue", "stage_entry")]["inferences_since_stage_entry"], 0)
        self.assertEqual(rows[("resample_continue", "stage_entry")]["inferences_before_detection"], 3)
        control = parents[0]
        self.assertEqual(control["role"], "success_control")
        self.assertEqual(sorted({(c["rung"], c["anchor"]) for c in control["cells"]}), sorted(ladder.DEFAULT_CONTROL_CELLS))

    def test_fixed_fraction_budget_on_the_same_protocol(self):
        collect_stage(self.root / "run", budget_mode="fixed_fraction")
        branches = ladder.read_jsonl(self.root / "run" / ladder.BRANCHES_NAME)
        self.assertTrue(all(b["budget_steps"] == 20 and b["budget_mode"] == "fixed_fraction" for b in branches))
        stuck = by_seed(self.root / "run")[1]
        cells = {(c["rung"], c["anchor"]): c for c in stuck["cells"]}
        # Same start state and budget: resampling from stage entry is the world reset to last good.
        self.assertEqual(cells[("resample_continue", "stage_entry")]["canonical"], ["world_reset", "last_good"])


def v1_shard(root):
    """A minimal schema-1 shard written exactly as the v1 collector did."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parents, branches = [], []

    def parent(pid, task, seed, success, role, required, events, num_steps, anchors, cells, detection_source="stall"):
        steps = {name: 4 * index for name, index in anchors.items()}
        counts = {name: sum(1 for e in events if e["step"] <= step) for name, step in steps.items()}
        row = {
            "anchor_ordered_counts": counts, "anchor_steps": steps, "anchors": anchors,
            "attempt_uid": f"uid-{pid}", "branch_steps": 0, "branch_wall_seconds": 0.0, "budget_steps": 20,
            "cells": cells, "created_at": "2026-10-16T00:00:00Z", "detection_source": detection_source,
            "detector": "stall", "detector_fired_on_success": True if success else None,
            "detectors": {"end": {"fired": True, "index": num_steps // 4, "step": num_steps},
                          "stall": {"fired": True, "index": 3, "step": 12},
                          "time": {"fired": False, "index": num_steps // 4, "step": num_steps}},
            "environment_seed": seed, "final_ordered_count": len(events), "horizon": 40,
            "instruction": "Do it.", "num_inferences": num_steps // 4, "num_snapshots": num_steps // 4 + 1,
            "num_steps": num_steps, "parent_id": pid, "progress_events": events,
            "required_subtasks": required, "role": role, "schema_version": 1,
            "scores_at_detection": {"ordered_progress": 0.0, "stall": 0.3, "time": 0.3},
            "seed_control": "policy_sampling_seed_base", "snapshot_path": f"snapshots/{pid}.npz",
            "success": success, "task_name": task, "wall_seconds": 1.0,
        }
        parents.append(row)
        return row

    def v1_cells(anchors):
        cells = []
        canonical = {}
        for rung, anchor in ladder.V1_CELLS:
            index = anchors["detection"] if rung in ("resample_continue", "subtask_reprompt") else anchors[anchor]
            key = ("reprompt",) if rung == "subtask_reprompt" else (("none",) if rung == "resample_continue" or index == anchors["detection"] else (rung, index))
            chosen = canonical.setdefault(key, (rung, anchor))
            cells.append({"anchor": anchor, "anchor_index": index, "canonical": list(chosen), "run": chosen == (rung, anchor), "rung": rung})
        return cells

    def add_branches(row, outcomes):
        for cell in row["cells"]:
            if not cell["run"]:
                continue
            for k in range(2):
                branches.append({
                    "anchor": cell["anchor"], "anchor_distance_inferences": row["anchors"]["detection"] - cell["anchor_index"],
                    "anchor_distance_steps": 0, "anchor_index": cell["anchor_index"], "approx_policy_inferences": 1,
                    "attempt_uid": row["attempt_uid"], "branch_seed": len(branches), "budget_steps": 20,
                    "created_at": "2026-10-16T00:00:00Z", "final_ordered_count": 0, "k": k, "parent_id": row["parent_id"],
                    "policy_context": "reset", "reprompt_instruction": None, "reprompt_steps": None, "role": row["role"],
                    "rung": cell["rung"], "schema_version": 1, "seed_control": "policy_sampling_seed_base",
                    "source_index": row["anchors"]["detection"], "source_restore_residual": 0.0, "stages_advanced": 0,
                    "start_ordered_count": 0, "steps_used": 20, "success": bool(outcomes.get((cell["rung"], cell["anchor"]), 0) > k),
                    "target_completed": False, "target_subtask": None, "task_name": row["task_name"], "wall_seconds": 0.1,
                })

    cut = ["lid_open", "reach", "move"]
    blocked = [{"step": 15, "subtask": name, "ordered_count": i + 1} for i, name in enumerate(cut)]
    for seed in range(3):
        parent(f"cut-s{seed}", "MockCut", seed, True, "unbranched", cut, blocked, 15, {"detection": 3, "last_good": 0, "last_good_minus_2": 0}, [], "time_fraction")
    anchors = {"detection": 3, "last_good": 0, "last_good_minus_2": 0}
    for seed in (10, 11):
        row = parent(f"cut-f{seed}", "MockCut", seed, False, "failure", cut, [], 40, anchors, v1_cells(anchors))
        add_branches(row, {("world_reset", "last_good"): 2, ("resample_continue", "detection"): 1})
    tea = ["a", "b", "c"]
    for seed in range(3):
        events = [{"step": 4, "subtask": "a", "ordered_count": 1}, {"step": 9, "subtask": "b", "ordered_count": 2}, {"step": 16 + seed, "subtask": "c", "ordered_count": 3}]
        parent(f"tea-s{seed}", "MockTea", seed, True, "unbranched", tea, events, 16 + seed, {"detection": 3, "last_good": 3, "last_good_minus_2": 1}, [], "time_fraction")
    anchors = {"detection": 5, "last_good": 1, "last_good_minus_2": 0}
    row = parent("tea-f0", "MockTea", 20, False, "failure", tea, [{"step": 4, "subtask": "a", "ordered_count": 1}], 40, anchors, v1_cells(anchors))
    add_branches(row, {("subtask_reprompt", "detection"): 2})
    for row in parents:
        ladder.append_jsonl(root / ladder.PARENTS_NAME, row)
    for row in branches:
        ladder.append_jsonl(root / ladder.BRANCHES_NAME, row)
    return root


class V1CompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.v1 = v1_shard(Path(self.tmp.name) / "v1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyzer_reads_schema_1_shards(self):
        summary, rows = ladder.analyze([self.v1], n_boot=100)
        self.assertEqual(summary["schema_versions"], [1])
        self.assertEqual(summary["budget_modes"], ["fixed_fraction"])
        cells = {(r["role"], r["task"], r["rung"], r["anchor"], r["metric"]): r for r in summary["ladder"]}
        self.assertEqual(cells[("failure", "MockCut", "world_reset", "last_good", "p_success")]["estimate"], 1.0)
        self.assertEqual(cells[("failure", "MockCut", "resample_continue", "detection", "p_success")]["estimate"], 0.5)
        # Aliased schema-1 cells still get an estimate for every parent.
        self.assertEqual(cells[("failure", "MockCut", "world_reset", "last_good_minus_2", "p_success")]["parents"], 2)
        stages = {(r["task"], r["stage"], r["rung"], r["anchor"]): r for r in summary["stage_ladder"]}
        self.assertEqual(stages[("MockTea", "b", "subtask_reprompt", "detection")]["p_success"], 1.0)
        self.assertEqual(stages[("MockCut", "lid_open", "world_reset", "last_good")]["p_success"], 1.0)
        # Schema-1 resample at detection lands at (detection - last good) inferences since stage entry.
        curve = {(r["axis"], r["rung"], r["task"], r["bin"]): r for r in summary["time_curve"]}
        self.assertEqual(curve[("since_stage_entry", "resample_continue", "MockCut", "1-4")]["parents"], 2)
        self.assertEqual(curve[("before_detection", "resample_continue", "ALL", "0")]["parents"], 3)
        health = {row["task"]: row for row in summary["detector_health"]}
        self.assertEqual(health["ALL"]["stall_fired_before_success"], "6/6")
        self.assertIsNone(health["ALL"]["stage_overdue_fired_before_success"])
        path = ladder.write_analysis(summary, rows, Path(self.tmp.name) / "analysis")
        text = path.read_text()
        self.assertIn("Per-parent cells", text)
        self.assertIn("Failure attribution (exploratory", text)

    def test_audit_and_calibration_from_schema_1_progress_events(self):
        report = ladder.audit_predicates([self.v1])
        cut = {row["predicate"]: row for row in report["tasks"]["MockCut"]["predicates"]}
        self.assertEqual(report["tasks"]["MockCut"]["mode"], "latch_only")
        self.assertEqual(cut["lid_open"]["success_blocker"], 1.0)
        self.assertIn("unreached_before_success", cut["lid_open"]["flags"])
        self.assertEqual(report["suggested_config"]["exclude_predicates"], {"MockCut": ["lid_open"]})
        tea = {row["predicate"]: row for row in report["tasks"]["MockTea"]["predicates"]}
        # The final stage latching at success is normal.
        self.assertEqual(tea["c"]["success_blocker"], 1.0)
        self.assertEqual(tea["c"]["flags"], [])
        calibration = ladder.build_stage_calibration([self.v1], {"exclude_predicates": {"MockCut": ["lid_open"]}})
        self.assertEqual(calibration["tasks"]["MockTea"]["stages"]["c"]["durations"], [7, 8, 9])
        self.assertEqual(calibration["tasks"]["MockCut"]["used_parents"], 0)
        self.assertEqual(calibration["tasks"]["MockCut"]["skipped_parents"], {"required_mismatch_without_predicate_trace": 3})

    def test_schema_1_and_2_runs_analyze_together(self):
        v2 = Path(self.tmp.name) / "v2"
        collect(v2)
        summary, rows = ladder.analyze([self.v1, v2], n_boot=50)
        self.assertEqual(summary["schema_versions"], [1, 2])
        self.assertEqual(summary["num_parents"], 9 + 3)
        text = ladder.write_analysis(summary, rows, Path(self.tmp.name) / "mixed").read_text()
        self.assertIn("Warning:", text)


class AuditTest(unittest.TestCase):
    def test_raw_traces_flag_reset_true_unreached_and_regressing_predicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "naturals"
            collect_stage(root, parents_only=True, parents_per_task=8)
            report = ladder.audit_predicates([root], stage_config())
            path = ladder.write_audit(report, Path(tmp) / "audit")
            self.assertIn("lid_open", path.read_text())
        task = report["tasks"]["MockTask"]
        self.assertEqual(task["mode"], "raw_and_latch")
        self.assertEqual(task["ladder_required_subtasks"], ["table_clear", "reach", "move"])
        rows = {row["predicate"]: row for row in task["predicates"]}
        self.assertEqual(rows["lid_open"]["never_true"], 1.0)
        self.assertEqual(rows["lid_open"]["unreached_before_success"], 1.0)
        self.assertIn("unreached_before_success", rows["lid_open"]["flags"])
        self.assertFalse(rows["lid_open"]["in_ladder_required"])
        # Already excluded, so not suggested again.
        self.assertEqual(report["suggested_config"]["exclude_predicates"], {})
        self.assertIn("true_at_reset", rows["table_clear"]["flags"])
        self.assertGreater(rows["reach"]["regressed_in_failures"], 0)
        self.assertIn("regresses_after_latch", rows["reach"]["flags"])
        self.assertEqual(rows["move"]["flags"], [])
        self.assertEqual(task["failed_with_durable_below_latched"], 2)

    def test_regressions_in_successful_parents_suggest_regression_ignore(self):
        def parent(pid, success, events):
            return {
                "parent_id": pid, "attempt_uid": pid, "task_name": "T", "success": success, "num_steps": 100,
                "required_subtasks": ["hold", "place"], "env_required_subtasks": ["hold", "place"],
                "progress_events": [{"step": 10, "subtask": "hold", "ordered_count": 1}, {"step": 100, "subtask": "place", "ordered_count": 2}],
                "final_ordered_count": 2, "final_durable_count": 2,
                "predicate_first_true_step": {"hold": 10, "place": 90}, "predicate_values_at_reset": {"hold": False, "place": False},
                "predicate_stages": {"hold": "control", "place": "placement"}, "regression_events": events,
            }

        transient = [{"subtask": "hold", "ordered_index": 0, "eligible": True, "start_step": 40, "end_step": 80}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for row in (parent("a", True, transient), parent("b", True, []), parent("c", True, transient)):
                ladder.append_jsonl(root / ladder.PARENTS_NAME, row)
            report = ladder.audit_predicates([root])
        rows = {row["predicate"]: row for row in report["tasks"]["T"]["predicates"]}
        self.assertAlmostEqual(rows["hold"]["regressed_in_successes"], 2 / 3)
        self.assertIn("transient_in_successes", rows["hold"]["flags"])
        self.assertEqual(report["suggested_config"]["regression_ignore_predicates"], {"T": ["hold"]})


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

    def test_schema_2_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            collect_stage(root)
            summary, rows = ladder.analyze([root], n_boot=100)
            text = ladder.write_analysis(summary, rows, Path(tmp) / "analysis").read_text()
            self.assertTrue((Path(tmp) / "analysis" / "time_curve.csv").is_file())
        curve = {(r["axis"], r["rung"], r["task"], r["bin"]): r for r in summary["time_curve"]}
        # Stage entry (before the scripted failures) recovers; later anchors do not.
        self.assertEqual(curve[("since_stage_entry", "resample_continue", "ALL", "0")]["p_success"], 1.0)
        self.assertEqual(curve[("since_stage_entry", "resample_continue", "ALL", "1-4")]["p_success"], 0.0)
        self.assertEqual(curve[("before_detection", "resample_continue", "MockTask", "0")]["p_success"], 0.0)
        stages = {(r["task"], r["stage"], r["rung"], r["anchor"]): r for r in summary["stage_ladder"]}
        self.assertEqual(stages[("MockTask", "reach", "robot_reset", "last_good")]["p_success"], 1.0)
        contrasts = {(r["task"], r["stage"], r["rung"], r["anchor"]): r for r in summary["stage_contrasts"]}
        self.assertEqual(contrasts[("MockTask", "move", "resample_continue", "stage_entry")]["estimate"], 1.0)
        progress = {row["task"]: row for row in summary["progress_at_detection"]}
        self.assertEqual(progress["ALL"]["durable_below_latched"], 1)
        health = {row["task"]: row for row in summary["detector_health"]}
        self.assertEqual(health["ALL"]["detection_rules"], {"regression": 1, "stall_fallback": 2})
        self.assertIn("Recoverability over time", text)
        self.assertIn("| MockTask | reach | RR@lg |", text)
        self.assertIn("R@e+1", text)

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
        self.assertIn(("durable_progress", "p_resample"), rows)

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
    def load(self, name="recoverability_ladder_xr1_pilot.json"):
        return json.loads((REPO / "scripts" / name).read_text())

    def test_xr1_pilot_dry_run(self):
        config = self.load()
        plan = ladder.build_plan(config)
        self.assertEqual(plan["tasks"], ["ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CuttingToolSelection", "GarnishPancake"])
        self.assertEqual(plan["warnings"], [])
        self.assertEqual(plan["budget_mode"], "remaining_horizon")
        self.assertEqual(plan["k_branches"], 4)
        self.assertEqual(len(plan["cells"]), 12)
        self.assertEqual(len(plan["control_cells"]), 4)
        resolved = ladder.resolve_config(config)
        self.assertEqual((resolved["failed_parents_per_task"], resolved["success_controls_per_task"]), (8, 4))
        self.assertEqual(resolved["exclude_predicates"]["CuttingToolSelection"], ["drawer_open"])
        for row in plan["per_task"]:
            self.assertTrue(0.2 <= row["expected_success_rate"] <= 0.8)
            self.assertLessEqual(row["budget_steps"], row["horizon"])
        totals = plan["totals"]
        self.assertGreater(totals["upper"]["gpu_hours"], totals["expected"]["gpu_hours"])
        self.assertAlmostEqual(totals["upper"]["gpu_hours"], 2 * totals["upper"]["worker_hours"], places=1)
        self.assertEqual(plan["throughput_source"], "assumed")
        self.assertEqual(plan["seconds_per_env_step"], 0.053)
        self.assertEqual(plan["num_workers"], 1)
        self.assertEqual(totals["expected"]["fits_with_one_resume"], totals["expected"]["allocations_needed"] <= 2)

    def test_budget_mode_changes_the_estimate(self):
        remaining = ladder.build_plan(self.load())
        fixed = ladder.build_plan({**self.load(), "budget_mode": "fixed_fraction"})
        self.assertGreater(remaining["totals"]["expected"]["env_steps"], fixed["totals"]["expected"]["env_steps"])
        naturals = ladder.build_plan({**self.load(), "parents_only": True, "parents_per_task": 12})
        self.assertTrue(all(row["branch_env_steps_upper"] == 0 for row in naturals["per_task"]))
        self.assertTrue(naturals["totals"]["expected"]["fits_single_allocation"])

    def test_expansion_config(self):
        plan = ladder.build_plan(self.load("recoverability_ladder_xr1_expansion.json"))
        self.assertEqual(len(plan["tasks"]), 5)
        self.assertEqual(plan["warnings"], [])
        self.assertFalse(set(plan["tasks"]) & set(self.load()["tasks"]))
        self.assertNotIn("GetToastedBread", plan["tasks"])

    def test_calibration_reference_and_cli(self):
        plan = ladder.build_plan(self.load(), calibration={"measured_seconds_per_env_step": 0.12})
        self.assertEqual(plan["throughput_source"], "measured")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            collect_stage(root)
            reference = ladder.reference_fractions([root])
            self.assertEqual(reference["MockTask"]["failed_parents"], 3)
            self.assertAlmostEqual(reference["MockTask"]["stage_entry_fraction"], 0.2)
            out = Path(tmp) / "plan.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli = ladder.main(["plan", "--config", str(REPO / "scripts" / "recoverability_ladder_xr1_pilot.json"), "--k-branches", "2", "--budget-mode", "fixed_fraction", "--output", str(out)])
                audit = ladder.main(["audit-predicates", "--run-dirs", str(root), "--output-dir", str(Path(tmp) / "audit")])
                calibration = ladder.main(["calibrate-stages", "--run-dirs", str(root), "--output", str(Path(tmp) / "cal.json")])
            self.assertEqual(cli["k_branches"], 2)
            self.assertEqual(cli["budget_mode"], "fixed_fraction")
            self.assertTrue(out.is_file())
            self.assertIn("MockTask", audit["tasks"])
            self.assertEqual(calibration["kind"], "recoverability_ladder_stage_calibration")
            self.assertTrue((Path(tmp) / "cal.json").is_file())

    def test_rate_outside_band_warns_and_bad_cells_rejected(self):
        config = self.load()
        config["expected_success_rates"]["ArrangeTea"] = 0.9
        plan = ladder.build_plan(config)
        self.assertTrue(any("ArrangeTea" in warning for warning in plan["warnings"]))
        with self.assertRaises(ValueError):
            ladder.resolve_config({"cells": [["teleport", "detection"]]})
        with self.assertRaises(ValueError):
            ladder.resolve_config({"cells": [["resample_continue", "stage_entry_plus_x"]]})
        with self.assertRaises(ValueError):
            ladder.resolve_config({"unknown_option": 1})

    def test_task_sharding_is_a_partition(self):
        horizons = {"A": 10, "B": 30, "C": 20, "D": 5}
        shards = [ladder.shard_tasks(list(horizons), horizons, 2, i) for i in range(2)]
        self.assertEqual(sorted(shards[0] + shards[1]), sorted(horizons))
        self.assertFalse(set(shards[0]) & set(shards[1]))


if __name__ == "__main__":
    unittest.main()
