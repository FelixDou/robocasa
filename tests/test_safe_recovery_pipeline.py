import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.recovery_rollout import (
    RecoveryConfig,
    run_recovery_after_failed_rollout,
)
from robocasa.recovery.safe import models
from robocasa.recovery.safe.runtime_monitor import CheckpointSafeMonitor


class SequenceMonitor:
    def __init__(self, detections):
        self.detections = list(detections)

    def reset(self):
        self.index = 0

    def observe_inference(self, record):
        detected = self.detections[self.index]
        decision = {
            "inference_index": self.index,
            "score": 0.9 if detected else 0.1,
            "threshold": 0.5,
            "failure_detected": detected,
        }
        self.index += 1
        return decision


class FakePolicy:
    def __init__(self):
        self.pending = None
        self.instructions = []
        self.inference_index = 0

    def __call__(self, obs, instruction=None):
        self.instructions.append(instruction)
        self.pending = {
            "features": np.ones((1, 1, 2), dtype=np.float32),
            "inference_index": self.inference_index,
            "environment_step": len(self.instructions) - 1,
        }
        self.inference_index += 1
        return np.zeros(1, dtype=np.float32)

    def pop_inference_record(self):
        record = self.pending
        self.pending = None
        return record


class FakeEnv:
    def __init__(self):
        self.completed = 0
        self.steps = 0
        self.instruction = "Do the whole task."
        self.reset_to_calls = []

    def _obs(self):
        return {"annotation.human.task_description": self.instruction}

    def reset(self):
        return self._obs()

    def get_state(self):
        return {"states": np.asarray([self.completed, self.steps], dtype=float)}

    def reset_to(self, state):
        values = np.asarray(state["states"])
        self.completed = int(values[0])
        self.steps = int(values[1])
        self.reset_to_calls.append(values.copy())
        return self._obs()

    def get_current_observation(self):
        return self._obs()

    def set_task_description(self, instruction):
        self.instruction = instruction

    def get_subtask_progress(self):
        return {
            "task_name": "FakeComposite",
            "task_success": self.completed >= 2,
            "required_predicates": ["stage_one", "stage_two"],
            "predicates": {
                "stage_one": {
                    "value": self.completed >= 1,
                    "description": "Complete stage one.",
                    "stage": "placement",
                },
                "stage_two": {
                    "value": self.completed >= 2,
                    "description": "Complete stage two.",
                    "stage": "placement",
                },
            },
        }

    def step(self, action):
        self.steps += 1
        if self.instruction == "Complete stage two.":
            self.completed = 2
        elif self.completed == 0:
            self.completed = 1
        info = {
            "success": self.completed >= 2,
            "subtask_eval": self.get_subtask_progress(),
        }
        return self._obs(), float(info["success"]), False, info


class TestSafeRecoveryPipeline(unittest.TestCase):
    def test_safe_stops_flagged_action_and_retries_current_subtask(self):
        env = FakeEnv()
        env.completed = 1
        policy = FakePolicy()
        result = run_recovery_after_failed_rollout(
            policy,
            env,
            RecoveryConfig(
                mode="continue_from_failure",
                recovery_level="subtask",
                high_level_horizon=5,
                subtask_horizon=2,
                require_subtask_target=True,
            ),
            failure_monitor=SequenceMonitor([True]),
        )

        self.assertEqual(result["high_level"]["termination_reason"], "safe_failure_detected")
        self.assertEqual(result["high_level"]["num_steps"], 0)
        self.assertEqual(result["high_level"]["num_policy_calls"], 1)
        self.assertEqual(
            result["high_level"]["failure_trigger"]["ordered_current_subtask"],
            "stage_two",
        )
        self.assertTrue(result["recovery_attempted"])
        self.assertEqual(result["subtask"]["target_subtask"], "stage_two")
        self.assertEqual(result["subtask"]["target_instruction"], "Complete stage two.")
        self.assertTrue(result["subtask"]["success"])
        self.assertEqual(policy.instructions, [None, "Complete stage two."])

    def test_last_good_state_updates_only_after_ordered_progress(self):
        env = FakeEnv()
        policy = FakePolicy()
        result = run_recovery_after_failed_rollout(
            policy,
            env,
            RecoveryConfig(
                mode="env_to_last_good",
                recovery_level="subtask",
                high_level_horizon=5,
                subtask_horizon=2,
                require_subtask_target=True,
            ),
            failure_monitor=SequenceMonitor([False, True]),
        )

        self.assertEqual(result["high_level"]["num_steps"], 1)
        self.assertEqual(result["high_level"]["last_good_ordered_subtask"], "stage_one")
        self.assertEqual(len(env.reset_to_calls), 1)
        self.assertEqual(env.reset_to_calls[0].tolist(), [1.0, 1.0])
        self.assertTrue(result["recovery"]["state_restored"])
        self.assertTrue(result["subtask"]["success"])

    def test_time_limit_uses_same_recovery_branch(self):
        env = FakeEnv()
        policy = FakePolicy()
        result = run_recovery_after_failed_rollout(
            policy,
            env,
            RecoveryConfig(
                mode="continue_from_failure",
                recovery_level="subtask",
                high_level_horizon=1,
                subtask_horizon=2,
                require_subtask_target=True,
            ),
        )
        self.assertEqual(result["high_level"]["termination_reason"], "time_limit")
        self.assertEqual(result["high_level"]["failure_trigger"]["type"], "time_limit")
        self.assertEqual(result["subtask"]["target_subtask"], "stage_two")
        self.assertTrue(result["subtask"]["success"])


@unittest.skipIf(models.torch is None, "PyTorch is not installed")
class TestCheckpointSafeMonitor(unittest.TestCase):
    def test_checkpoint_score_crosses_calibrated_threshold(self):
        torch = models.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = models.ModelConfig("mlp", 2, num_layers=1)
            model = models.make_model(config)
            with torch.no_grad():
                linear = model.projector[0]
                linear.weight.zero_()
                linear.bias.zero_()
            checkpoint = models.save_checkpoint(
                root / "checkpoint.pt",
                model,
                config,
                {"aggregation": "mean"},
            )
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "threshold": [0.4],
                        "alignment": "extend",
                        "crossing_rule": "score >= threshold",
                    }
                )
            )
            monitor = CheckpointSafeMonitor(checkpoint, calibration)
            decision = monitor.observe_inference(
                {
                    "features": np.ones((1, 1, 2), dtype=np.float32),
                    "inference_index": 0,
                    "environment_step": 0,
                }
            )
            self.assertAlmostEqual(decision["score"], 0.5)
            self.assertAlmostEqual(decision["threshold"], 0.4)
            self.assertTrue(decision["failure_detected"])
            monitor.reset()
            self.assertEqual(monitor.decisions, [])


if __name__ == "__main__":
    unittest.main()
