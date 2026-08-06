import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages


install_lightweight_robocasa_packages()

from robocasa.recovery.xiaomi_robotics_1_policy import (  # noqa: E402
    XIAOMI_SAFE_FEATURE_LAYER,
    XiaomiRobotics1Policy,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = False

    def infer(self, request):
        self.requests.append(request)
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def list_robot_types(self):
        return ["robocasa365"]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "state": np.asarray(kwargs["state"], dtype=np.float32),
            "action_mask": np.zeros((1, 30, 60), dtype=np.float32),
        }

    def decode_action(self, actions, robot_type):
        assert robot_type == "robocasa365"
        return actions


def observation(instruction="close the blender lid"):
    return {
        "annotation.human.task_description": instruction,
        "video.robot0_agentview_left": np.zeros((8, 8, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((8, 8, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((8, 8, 3), dtype=np.uint8),
        "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
        "state.end_effector_rotation_relative": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        "state.gripper_qpos": np.zeros(2, dtype=np.float32),
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }


def safe_response(shape=(5, 30, 8)):
    features = np.arange(np.prod(shape), dtype=np.float32).reshape((1, *shape))
    return {
        "actions": np.zeros((1, shape[1], 60), dtype=np.float32),
        "safe_features": features,
        "safe_feature_metadata": {
            "schema_version": 1,
            "model_family": "xiaomi_robotics_1",
            "model_id": "MiBoTForActionGeneration",
            "checkpoint": "/checkpoint",
            "feature_layer": XIAOMI_SAFE_FEATURE_LAYER,
            "feature_shape": list(shape),
            "feature_dtype": "float32",
            "action_horizon": shape[1],
            "flow_steps": shape[0],
            "aggregation": "raw",
        },
    }


class TestSafeXiaomiPolicy(unittest.TestCase):
    def make_policy(self, responses, **kwargs):
        return XiaomiRobotics1Policy(
            model_path="/checkpoint",
            client=FakeClient(responses),
            processor=FakeProcessor(),
            replan_steps=2,
            **kwargs,
        )

    def test_action_only_server_remains_backward_compatible(self):
        actions = np.zeros((1, 30, 60), dtype=np.float32)
        policy = self.make_policy([actions])
        action = policy(observation())
        self.assertEqual(
            set(action),
            {
                "action.end_effector_position",
                "action.end_effector_rotation",
                "action.gripper_close",
                "action.base_motion",
                "action.control_mode",
            },
        )
        self.assertIsNone(policy.pop_inference_record())
        self.assertNotIn("request_safe_features", policy.client.requests[0])

    def test_features_are_recorded_once_per_true_inference(self):
        policy = self.make_policy(
            [safe_response(), safe_response()],
            collect_safe_features=True,
            policy_name="Xiaomi-Robotics-1-RoboCasa365",
            policy_checkpoint="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
        )
        policy(observation())
        first = policy.pop_inference_record()
        self.assertEqual(first["environment_step"], 0)
        self.assertEqual(first["inference_index"], 0)
        self.assertEqual(first["features"].shape, (5, 30, 8))
        self.assertEqual(first["actions"].shape, (30, 12))
        self.assertEqual(first["metadata"]["model_family"], "xiaomi_robotics_1")
        self.assertTrue(policy.client.requests[0]["request_safe_features"])
        self.assertEqual(policy.client.requests[0]["state"].shape, (1, 4, 60))

        policy(observation())
        self.assertIsNone(policy.pop_inference_record())
        self.assertEqual(len(policy.client.requests), 1)

        policy(observation())
        second = policy.pop_inference_record()
        self.assertEqual(second["environment_step"], 2)
        self.assertEqual(second["inference_index"], 1)
        self.assertEqual(len(policy.client.requests), 2)

    def test_invalid_or_missing_features_raise(self):
        actions = np.zeros((1, 30, 60), dtype=np.float32)
        policy = self.make_policy([actions], collect_safe_features=True)
        with self.assertRaisesRegex(RuntimeError, "did not return"):
            policy(observation())

        bad = safe_response()
        bad["safe_feature_metadata"]["feature_layer"] = "wrong"
        policy = self.make_policy([bad], collect_safe_features=True)
        with self.assertRaisesRegex(RuntimeError, "feature_layer"):
            policy(observation())

        bad = safe_response()
        bad["safe_features"][0, 0, 0, 0] = np.nan
        policy = self.make_policy([bad], collect_safe_features=True)
        with self.assertRaisesRegex(RuntimeError, "finite"):
            policy(observation())

    def test_reset_clears_rollout_state_without_closing_server(self):
        policy = self.make_policy([safe_response()], collect_safe_features=True)
        policy(observation())
        policy.reset()
        self.assertEqual(policy._env_step, 0)
        self.assertEqual(policy._inference_index, 0)
        self.assertFalse(policy.action_plan)
        self.assertFalse(policy.state_queue)
        self.assertIsNone(policy.pop_inference_record())
        self.assertFalse(policy.client.closed)


if __name__ == "__main__":
    unittest.main()
