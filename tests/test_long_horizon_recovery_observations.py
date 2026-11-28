"""Exercise the recovery observation history through the SAFE collector."""

import json
import unittest
from unittest.mock import patch

from tests.test_safe_collect_rollouts import FakePolicy
from robocasa.recovery import eval_composite_predicates as predicates
from robocasa.recovery.safe.collect_rollouts import collect_single_rollout
from robocasa.recovery.safe.subtask_safe import validate_subtask_safe_record


class ArrangeBreadBasket:
    cab = object()
    dining_table = object()

    def reset(self):
        self.steps = 0
        return {}, {}

    def step(self, action):
        self.steps += 1
        done = self.steps == 7
        return {}, 0., done, False, {"success": done}

    def get_subtask_progress(self):
        values = predicates.get_runtime_subtask_predicates(self)
        return dict(task_name="ArrangeBreadBasket", predicates=values, task_success=self.steps == 7,
                    required_predicates=[k for k, v in values.items() if v["required"]])


def bread_rollout():
    env = ArrangeBreadBasket()
    with (patch.object(predicates, "_in", side_effect=lambda *args: env.steps in (1, 4, 5, 6, 7)),
          patch.object(predicates, "_is_open", return_value=True),
          patch.object(predicates, "_far", side_effect=lambda *args: env.steps >= 6),
          patch.object(predicates, "_on_fixture", side_effect=lambda *args: env.steps >= 6),
          patch.object(predicates, "_grasped",
                       side_effect=lambda env, name: (name, env.steps) in {("bread", 3), ("basket", 5)})):
        return collect_single_rollout(FakePolicy(), env, horizon=8, record_subtask_trace=True)


class RecoveryObservationTests(unittest.TestCase):
    def test_between_decision_regrasp_survives_collection_and_serialization(self):
        result = bread_rollout()
        record = json.loads(json.dumps(result["subtask_safe_record"]))
        obj = {r["environment_step"]: r["values"] for r in record["recovery_observations"]}
        self.assertEqual(result["inference_env_steps"], [0, 2, 4, 6])
        self.assertTrue(obj[1]["bread_in_basket"])
        self.assertFalse(obj[2]["bread_in_basket"])
        self.assertTrue(obj[3]["bread_grasped"])
        self.assertFalse(obj[4]["bread_grasped"])
        self.assertTrue(obj[4]["bread_in_basket"])
        self.assertTrue(obj[5]["basket_grasped"])
        self.assertEqual(obj[7], obj[6])  # Explicit terminal coverage even when truth is unchanged.
        self.assertEqual(set(obj[0]), {"bread_grasped", "basket_grasped", "basket_on_dining_counter", "bread_in_basket"})
        self.assertFalse(any(s["subtask_id"] in ("bread_grasped", "basket_grasped") for s in record["segments"]))
        validate_subtask_safe_record(record, rollout_id=None, rollout_failed=False,
                                     inference_environment_steps=result["inference_env_steps"])

    def test_optional_trace_is_validated_when_present_and_legacy_records_still_read(self):
        record = bread_rollout()["subtask_safe_record"]
        args = dict(rollout_id=None, rollout_failed=False, inference_environment_steps=[0, 2, 4, 6])
        history = record.pop("recovery_observations")
        validate_subtask_safe_record(record, **args)
        for invalid in (history[1:], history[:-1], [history[0], history[0], history[-1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_subtask_safe_record(dict(record, recovery_observations=invalid), **args)

    def test_unavailable_cutting_tool_signal_is_unknown(self):
        from robocasa.recovery.safe.subtask_safe import RECOVERY_PREDICATES, _recovery_observations
        names = RECOVERY_PREDICATES["CuttingToolSelection"]
        source = [{"predicates": {"wrong_tool_grasped": {"value": v}}} for v in (False, False, True, False)]
        data = _recovery_observations(source, names)
        self.assertEqual([r["environment_step"] for r in data], [0, 2, 3])
        self.assertEqual([r["values"]["wrong_tool_grasped"] for r in data], [False, True, False])
        self.assertTrue(all(r["values"]["correct_tool_present_on_board"] is None for r in data))

    def test_grasped_prefers_opposing_pad_check_over_finger_width_heuristic(self):
        class FakeObj:
            def __init__(self, geoms):
                self.contact_geoms = geoms

        class FakeRobot:
            def __init__(self, gripper):
                self.gripper = {"right": gripper}

        class FakeEnv:
            def __init__(self, grasped_geoms):
                self._grasped_geoms = grasped_geoms
                self.right_gripper = object()
                self.robots = [FakeRobot(self.right_gripper)]
                self.objects = {
                    "bread": FakeObj(["bread_g0"]),
                    "peeler": FakeObj(["peeler_g0"]),
                }
                self.calls = []

            def _check_grasp(self, *, gripper, object_geoms):
                self.calls.append((gripper, object_geoms))
                return any(g in self._grasped_geoms for g in object_geoms)

        env = FakeEnv({"bread_g0"})
        # Wide bread passes opposing-pad grasp; adjacent peeler grazed by closed finger does not.
        with patch.object(predicates, "_OU", side_effect=AssertionError("should not call OU.check_obj_grasped")):
            self.assertTrue(predicates._grasped(env, "bread"))
            self.assertFalse(predicates._grasped(env, "peeler"))
        self.assertEqual(env.calls, [(env.right_gripper, ["bread_g0"]), (env.right_gripper, ["peeler_g0"])])

        # Fallback path for minimal stub envs without _check_grasp
        class LegacyStubEnv:
            pass

        class FakeOU:
            @staticmethod
            def check_obj_grasped(e, name):
                return name == "bread"

        with patch.object(predicates, "_OU", return_value=FakeOU):
            self.assertTrue(predicates._grasped(LegacyStubEnv(), "bread"))
            self.assertFalse(predicates._grasped(LegacyStubEnv(), "peeler"))

    def test_all_six_pilot_tasks_populate_and_validate_recovery_observations(self):
        from robocasa.recovery.safe.subtask_safe import RECOVERY_PREDICATES

        expected_tasks = {
            "ArrangeBreadBasket",
            "CuttingToolSelection",
            "DeliverStraw",
            "GarnishPancake",
            "MakeIceLemonade",
            "SetUpCuttingStation",
        }
        self.assertTrue(expected_tasks.issubset(RECOVERY_PREDICATES))

        def make_env(task_name):
            class _Base:
                cab = drawer = fridge = dining_table = dining_counter = object()
                _CUTTING_MAP = {"avocado": "knife"}
                food = "avocado"

                def reset(self):
                    self.steps = 0
                    return {}, {}

                def step(self, action):
                    self.steps += 1
                    done = self.steps == 4
                    return {}, 0.0, done, False, {"success": done}

                def get_subtask_progress(self):
                    values = predicates.get_runtime_subtask_predicates(self)
                    return dict(
                        task_name=task_name,
                        predicates=values,
                        task_success=self.steps == 4,
                        required_predicates=[k for k, v in values.items() if v["required"]],
                    )

            cls = type(task_name, (_Base,), {})
            return cls()

        for task in sorted(expected_tasks):
            with self.subTest(task=task):
                env = make_env(task)
                with (
                    patch.object(predicates, "_in", side_effect=lambda *a, **kw: env.steps >= 3),
                    patch.object(predicates, "_on_fixture", side_effect=lambda *a, **kw: env.steps >= 3),
                    patch.object(predicates, "_is_open", side_effect=lambda *a, **kw: env.steps >= 1),
                    patch.object(predicates, "_far", side_effect=lambda *a, **kw: env.steps >= 4),
                    patch.object(predicates, "_grasped", side_effect=lambda *a, **kw: env.steps == 2),
                ):
                    result = collect_single_rollout(FakePolicy(), env, horizon=6, record_subtask_trace=True)
                record = json.loads(json.dumps(result["subtask_safe_record"]))
                self.assertIn("recovery_observations", record)
                for obs in record["recovery_observations"]:
                    self.assertEqual(set(obs["values"]), set(RECOVERY_PREDICATES[task]))
                    self.assertTrue(all(isinstance(v, bool) for v in obs["values"].values()))
                validate_subtask_safe_record(
                    record,
                    rollout_id=None,
                    rollout_failed=False,
                    inference_environment_steps=result["inference_env_steps"],
                )

    def test_verify_expected_frame_steps_for_multiple_strides(self):
        from scripts.xr1_lead_time.verify_recovery_recording import _expected_frame_steps

        self.assertEqual(_expected_frame_steps(8, 1), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(_expected_frame_steps(8, 2), [1, 3, 5, 7, 8])
        self.assertEqual(_expected_frame_steps(8, 3), [1, 4, 7, 8])
        self.assertEqual(_expected_frame_steps(9, 2), [1, 3, 5, 7, 9])
