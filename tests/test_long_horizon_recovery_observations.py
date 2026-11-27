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
