import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.evaluate_rldx_official_recovery import (  # noqa: E402
    RLDX_SAFE_10X10_TASKS,
    _step_official,
    resolve_tasks,
)
from robocasa.recovery.safe.runtime_monitor import (  # noqa: E402
    OfficialSafeCheckpointMonitor,
    RLDX_SAFE_FEATURE_LAYER,
)


class FakePolicy:
    def __init__(self):
        self.options = None

    def get_action(self, observations, options=None):
        self.options = options
        return np.zeros((1, 2, 7), dtype=np.float32), {
            "safe_features": np.ones((1, 3, 2, 4), dtype=np.float32),
            "safe_feature_metadata": {
                "model_family": "rldx1",
                "feature_layer": RLDX_SAFE_FEATURE_LAYER,
                "aggregation": "raw",
            },
        }


class FakeVectorEnv:
    def __init__(self):
        self.step_calls = 0

    def step(self, actions):
        self.step_calls += 1
        return {"obs": 2}, np.asarray([0.0]), np.asarray([False]), np.asarray(
            [False]
        ), {}


class FakeMonitor:
    def __init__(self, detected):
        self.detected = detected
        self.records = []
        self.decisions = []

    def observe_inference(self, record):
        self.records.append(record)
        decision = {
            "failure_detected": self.detected,
            "score": 0.9 if self.detected else 0.1,
            "threshold": 0.5,
        }
        self.decisions.append(decision)
        return decision


class TestOfficialRldxSafeRecovery(unittest.TestCase):
    def test_ten_task_preset_matches_original_rldx_safe_dataset(self):
        self.assertEqual(
            resolve_tasks("rldx_safe_10x10", None),
            RLDX_SAFE_10X10_TASKS,
        )
        self.assertEqual(len(RLDX_SAFE_10X10_TASKS), 10)

    def test_safe_detection_blocks_flagged_rldx_action_chunk(self):
        env = FakeVectorEnv()
        policy = FakePolicy()
        monitor = FakeMonitor(True)

        result = _step_official(
            env,
            policy,
            {"obs": 1},
            is_first_step=True,
            session_id="test",
            failure_monitor=monitor,
            environment_step=4,
        )

        self.assertEqual(env.step_calls, 0)
        self.assertTrue(policy.options["request_safe_features"])
        self.assertEqual(monitor.records[0]["features"].shape, (3, 2, 4))
        self.assertEqual(monitor.records[0]["environment_step"], 4)
        self.assertTrue(result[-1]["failure_detected"])

    def test_non_detection_executes_rldx_action_chunk(self):
        env = FakeVectorEnv()
        result = _step_official(
            env,
            FakePolicy(),
            {"obs": 1},
            is_first_step=False,
            session_id="test",
            failure_monitor=FakeMonitor(False),
        )

        self.assertEqual(env.step_calls, 1)
        self.assertFalse(result[-1]["failure_detected"])

    def test_official_monitor_requires_one_threshold_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            OfficialSafeCheckpointMonitor(
                "/missing/safe",
                "/missing/checkpoint",
                "/missing/config",
            )


if __name__ == "__main__":
    unittest.main()
