"""Unit tests validating the 18-task Group-A/B benchmark split, predicate
contracts across EVAL_COMPOSITE_PREDICATES / RECOVERY_PREDICATES / TASK_SPECS /
PREDICATE_PROMPTS, and end-to-end EpisodeData extraction on all 18 tasks.
"""

import types
import unittest

import numpy as np

from robocasa.recovery import eval_composite_predicates as ecp
from robocasa.recovery.safe import subtask_safe
from scripts.xr1_lead_time import analyze_recovery_pilot6 as pilot6

try:
    import cv2
except ImportError:
    cv2 = None


class _DummyEnv:
    """Minimal stub environment allowing all 18 composite predicate evaluators to run."""

    def __init__(self, task_name: str):
        self.__class__ = type(task_name, (_DummyEnv,), {})
        self.cab = object()
        self.cabinet = object()
        self.drawer = object()
        self.fridge = object()
        self.microwave = object()
        self.dishwasher = object()
        self.sink = types.SimpleNamespace(
            check_obj_under_water=lambda env, name: False,
            get_handle_state=lambda env: {"water_on": False},
        )
        self.stove = types.SimpleNamespace(
            get_knobs_state=lambda env: {"front_left": 0.0},
        )
        self.knob = "front_left"
        self.dining_table = object()
        self.dining_counter = object()
        self.food = "avocado"
        self._CUTTING_MAP = {"avocado": "knife"}
        self.num_fruit = 2
        self.robots = [types.SimpleNamespace(gripper=None)]
        self.objects = {}


class Benchmark18ContractTests(unittest.TestCase):
    def test_preregistered_10_train_8_heldout_split_is_exact(self):
        train = set(pilot6.BENCHMARK18_TRAIN_TASKS)
        held = set(pilot6.BENCHMARK18_HELDOUT_TASKS)
        pilot = set(pilot6.PILOT6_TASKS)
        new12 = set(pilot6.BENCHMARK18_NEW12_TASKS)

        self.assertEqual(len(train), 10)
        self.assertEqual(len(held), 8)
        self.assertEqual(train & held, set())
        self.assertEqual(train | held, set(pilot6.BENCHMARK18_ALL_TASKS))
        self.assertTrue(pilot.issubset(train))
        self.assertEqual(pilot | new12, train | held)
        self.assertEqual(pilot & new12, set())

    def test_all_18_tasks_populate_recovery_predicates_without_breaking_required_contract(self):
        for task in pilot6.BENCHMARK18_ALL_TASKS:
            self.assertIn(task, ecp.EVAL_COMPOSITE_PREDICATES)
            self.assertIn(task, subtask_safe.RECOVERY_PREDICATES)
            self.assertIn(task, pilot6.TASK_SPECS)

            env = _DummyEnv(task)
            eval_preds = ecp.EVAL_COMPOSITE_PREDICATES[task](env)
            rec_preds = set(subtask_safe.RECOVERY_PREDICATES[task])
            self.assertTrue(
                rec_preds.issubset(eval_preds.keys()),
                f"{task}: missing in eval_composite_predicates: {sorted(rec_preds - set(eval_preds))}",
            )

            # Every newly added grasp predicate must be transient (required=False)
            for pname, meta in eval_preds.items():
                if pname.endswith("_grasped"):
                    self.assertFalse(meta["required"], f"{task}.{pname} must be required=False")
                    self.assertEqual(meta["stage"], "transient")

            # Every predicate referenced in TASK_SPECS must exist in RECOVERY_PREDICATES
            spec = pilot6.TASK_SPECS[task]
            spec_preds = (
                {gp for _, gps, _ in spec["placements"] for gp in gps}
                | {tp for _, _, tp in spec["placements"]}
                | set(spec["hold_conditions"])
                | set(spec["wrong_grasps"])
            )
            self.assertTrue(
                spec_preds.issubset(rec_preds),
                f"{task}: TASK_SPECS references unrecorded predicates: {sorted(spec_preds - rec_preds)}",
            )

    def test_pilot6_extract_episode_runs_on_all_18_tasks(self):
        for task in pilot6.BENCHMARK18_ALL_TASKS:
            rec_preds = subtask_safe.RECOVERY_PREDICATES[task]
            spec = pilot6.TASK_SPECS[task]
            first_grasp = spec["placements"][0][1][0]
            first_target = spec["placements"][0][2]

            base_vals = {p: False for p in rec_preds}
            g_vals = dict(base_vals)
            g_vals[first_grasp] = True
            t_vals = dict(base_vals)
            t_vals[first_target] = True

            hist = [
                {"environment_step": 0, "values": base_vals},
                {"environment_step": 16, "values": g_vals},
                {"environment_step": 48, "values": base_vals},  # missed placement release at 48
                {"environment_step": 128, "values": t_vals},
                {"environment_step": 160, "values": t_vals},
            ]
            row = {
                "rollout_id": f"{task}_syn",
                "task_name": task,
                "environment_seed": 1,
                "failed": False,
                "rollout_horizon": 320,
                "num_environment_steps": 160,
            }
            trace = {
                "task_name": task,
                "inference_records": [
                    {"environment_step": s, "inference_index": i, "subtask_id": "s0"}
                    for i, s in enumerate(range(0, 160, 16))
                ],
                "recovery_observations": hist,
                "segments": [],
            }
            ed = pilot6.extract_episode(row, trace)
            self.assertEqual(ed.task, task)
            self.assertGreaterEqual(len(ed.attempts), 1)
            self.assertEqual(ed.attempts[0]["release_step"], 48)

    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_vlm_prompts_cover_all_18_tasks(self):
        from scripts.xr1_lead_time.extract_vlm_destinations import (
            PREDICATE_PROMPTS,
            format_predicate_question,
            task_destination_predicates,
        )

        for task in pilot6.BENCHMARK18_ALL_TASKS:
            self.assertIn(task, PREDICATE_PROMPTS)
            preds = task_destination_predicates(task)
            self.assertTrue(preds)
            for p in preds:
                self.assertIn(p, PREDICATE_PROMPTS[task], f"{task}.{p} missing VLM prompt")
                q, _ = format_predicate_question(task, p, "Do the task on the front left burner.")
                self.assertTrue(len(q) > 20)


if __name__ == "__main__":
    unittest.main()
