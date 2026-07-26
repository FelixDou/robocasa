import sys
import types
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages


install_lightweight_robocasa_packages()
env_utils = types.ModuleType("robocasa.utils.env_utils")
env_utils.convert_action = lambda action: {"converted": np.asarray(action)}
sys.modules.setdefault("robocasa.utils", types.ModuleType("robocasa.utils"))
sys.modules["robocasa.utils.env_utils"] = env_utils

from robocasa.recovery.rldx_zmq_policy import (  # noqa: E402
    RLDX_SAFE_FEATURE_LAYER,
    RLDXZeroMQPolicy,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.reset_calls = []

    def get_action(self, observation, options=None):
        self.requests.append((observation, options))
        return self.responses.pop(0)

    def reset(self, options=None):
        self.reset_calls.append(options)
        return {}

    def close(self):
        pass


def observation():
    return {
        "annotation.human.task_description": "turn on the sink",
        "video.robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((4, 4, 3), dtype=np.uint8),
        "state.end_effector_position_relative": np.zeros(3),
        "state.end_effector_rotation_relative": np.zeros(3),
        "state.base_position": np.zeros(3),
        "state.base_rotation": np.zeros(3),
        "state.gripper_qpos": np.zeros(2),
    }


def response(with_features=True, shape=(3, 4, 8)):
    actions = np.zeros((1, shape[1], 12), dtype=np.float32)
    info = {}
    if with_features:
        features = np.arange(np.prod(shape), dtype=np.float32).reshape(
            (1, *shape)
        )
        info = {
            "safe_features": features,
            "safe_feature_metadata": {
                "schema_version": 1,
                "model_family": "rldx1",
                "model_id": "RLDX",
                "checkpoint": "RLWRLD/RLDX-1-FT-RC365",
                "feature_layer": RLDX_SAFE_FEATURE_LAYER,
                "feature_shape": list(shape),
                "feature_dtype": "float32",
                "action_horizon": shape[1],
                "flow_steps": shape[0],
                "aggregation": "raw",
            },
        }
    return actions, info


class TestSafeRLDXPolicy(unittest.TestCase):
    def make_policy(self, responses, **kwargs):
        return RLDXZeroMQPolicy(
            client=FakeClient(responses),
            execution_horizon=2,
            video_history=1,
            **kwargs,
        )

    def test_action_only_response_remains_backward_compatible(self):
        policy = self.make_policy([response(False)])
        policy(observation())
        self.assertIsNone(policy.pop_inference_record())

    def test_feature_record_only_for_real_inference(self):
        policy = self.make_policy(
            [response(), response()],
            collect_safe_features=True,
            policy_name="rldx1-robocasa",
            policy_checkpoint="RLWRLD/RLDX-1-FT-RC365",
        )
        policy(observation())
        first = policy.pop_inference_record()
        self.assertEqual(first["environment_step"], 0)
        self.assertEqual(first["inference_index"], 0)
        self.assertEqual(first["features"].shape, (3, 4, 8))
        self.assertEqual(first["actions"].shape, (4, 12))
        self.assertEqual(first["metadata"]["model_family"], "rldx1")
        self.assertTrue(
            policy.client.requests[0][1]["request_safe_features"]
        )
        request_observation = policy.client.requests[0][0]
        self.assertEqual(
            request_observation["annotation.human.task_description"],
            ["turn on the sink"],
        )
        self.assertEqual(
            request_observation["language"][
                "annotation.human.task_description"
            ],
            [["turn on the sink"]],
        )

        policy(observation())
        self.assertIsNone(policy.pop_inference_record())
        self.assertEqual(len(policy.client.requests), 1)

        policy(observation())
        second = policy.pop_inference_record()
        self.assertEqual(second["environment_step"], 2)
        self.assertEqual(second["inference_index"], 1)
        self.assertEqual(len(policy.client.requests), 2)

    def test_missing_or_invalid_features_raise(self):
        policy = self.make_policy(
            [response(False)],
            collect_safe_features=True,
        )
        with self.assertRaisesRegex(RuntimeError, "did not return"):
            policy(observation())

        actions, info = response()
        info["safe_feature_metadata"]["feature_shape"] = [9, 4, 8]
        policy = self.make_policy(
            [(actions, info)],
            collect_safe_features=True,
        )
        with self.assertRaisesRegex(RuntimeError, "declares shape"):
            policy(observation())

        actions, info = response()
        info["safe_features"][0, 0, 0, 0] = np.nan
        policy = self.make_policy(
            [(actions, info)],
            collect_safe_features=True,
        )
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            policy(observation())

    def test_action_horizon_mismatch_and_reset(self):
        actions, info = response()
        actions = actions[:, :-1]
        policy = self.make_policy(
            [(actions, info)],
            collect_safe_features=True,
        )
        with self.assertRaisesRegex(RuntimeError, "action-horizon mismatch"):
            policy(observation())

        policy = self.make_policy([response()], collect_safe_features=True)
        policy(observation())
        policy.reset()
        self.assertEqual(policy._env_step, 0)
        self.assertEqual(policy._inference_index, 0)
        self.assertFalse(policy.action_plan)
        self.assertIsNone(policy.pop_inference_record())


if __name__ == "__main__":
    unittest.main()
