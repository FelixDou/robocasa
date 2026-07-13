import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.openpi_websocket_policy import OpenPIWebsocketPolicy


class FakeImageTools:
    @staticmethod
    def resize_with_pad(image, height, width):
        return image

    @staticmethod
    def convert_to_uint8(image):
        return np.asarray(image, dtype=np.uint8)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def infer(self, observation):
        self.requests.append(observation)
        return self.responses.pop(0)


def observation():
    return {
        "annotation.human.task_description": "do task",
        "video.robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((4, 4, 3), dtype=np.uint8),
        "state.end_effector_position_relative": np.zeros(3),
        "state.end_effector_rotation_relative": np.zeros(3),
        "state.base_position": np.zeros(3),
        "state.base_rotation": np.zeros(3),
        "state.gripper_qpos": np.zeros(2),
    }


def response(with_features=True, shape=(2, 4, 8)):
    result = {"actions": np.zeros((4, 12), dtype=np.float32)}
    if with_features:
        result["safe_features"] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        result["safe_feature_metadata"] = {
            "schema_version": 1,
            "feature_layer": "action_expert_suffix_pre_action_out_proj",
            "feature_shape": list(shape),
            "feature_dtype": "float32",
            "action_horizon": shape[1],
            "flow_steps": shape[0],
            "aggregation": "raw",
            "model_id": "Pi0",
            "checkpoint": "fake-checkpoint",
        }
    return result


class TestSafeOpenPIPolicy(unittest.TestCase):
    def make_policy(self, responses, **kwargs):
        return OpenPIWebsocketPolicy(
            replan_steps=2,
            client=FakeClient(responses),
            image_tools=FakeImageTools,
            **kwargs,
        )

    def test_backward_compatible_response_without_features(self):
        policy = self.make_policy([response(False)])
        policy(observation())
        self.assertIsNone(policy.pop_inference_record())

    def test_feature_record_only_for_real_inference(self):
        policy = self.make_policy(
            [response(), response()],
            collect_safe_features=True,
            policy_name="pi0-robocasa",
            policy_checkpoint="checkpoint-74999",
        )
        policy(observation())
        first = policy.pop_inference_record()
        self.assertEqual(first["env_step"], 0)
        self.assertEqual(first["environment_step"], 0)
        self.assertEqual(first["inference_index"], 0)
        self.assertEqual(first["actions"].shape, (4, 12))
        self.assertEqual(first["metadata"]["feature_aggregation"], "raw")
        self.assertEqual(first["metadata"]["policy_name"], "pi0-robocasa")
        self.assertEqual(
            first["metadata"]["policy_checkpoint"], "checkpoint-74999"
        )
        policy(observation())
        self.assertIsNone(policy.pop_inference_record())
        self.assertEqual(len(policy.client.requests), 1)
        policy(observation())
        second = policy.pop_inference_record()
        self.assertEqual(second["env_step"], 2)
        self.assertEqual(second["inference_index"], 1)
        self.assertEqual(len(policy.client.requests), 2)
        self.assertTrue(policy.client.requests[0]["request_safe_features"])

    def test_missing_features_raises(self):
        policy = self.make_policy([response(False)], collect_safe_features=True)
        with self.assertRaisesRegex(RuntimeError, "did not return"):
            policy(observation())

    def test_invalid_feature_shape_and_nonfinite_raise(self):
        bad = response(shape=(2, 4, 8))
        bad["safe_feature_metadata"]["feature_shape"] = [3, 4, 8]
        policy = self.make_policy([bad], collect_safe_features=True)
        with self.assertRaisesRegex(RuntimeError, "declares shape"):
            policy(observation())
        bad = response()
        bad["safe_features"][0, 0, 0] = np.nan
        policy = self.make_policy([bad], collect_safe_features=True)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            policy(observation())

    def test_zero_features_and_action_horizon_mismatch_raise(self):
        bad = response()
        bad["safe_features"].fill(0)
        policy = self.make_policy([bad], collect_safe_features=True)
        with self.assertRaisesRegex(RuntimeError, "entirely zero"):
            policy(observation())
        bad = response()
        bad["actions"] = np.zeros((3, 12), dtype=np.float32)
        policy = self.make_policy([bad], collect_safe_features=True)
        with self.assertRaisesRegex(RuntimeError, "action-horizon mismatch"):
            policy(observation())

    def test_instruction_change_resets_feature_sequence(self):
        policy = self.make_policy([response(), response()], collect_safe_features=True)
        policy(observation(), instruction="first")
        self.assertEqual(policy.pop_inference_record()["inference_index"], 0)
        policy(observation(), instruction="second")
        record = policy.pop_inference_record()
        self.assertEqual(record["inference_index"], 0)
        self.assertEqual(record["environment_step"], 1)
        self.assertEqual(len(policy.client.requests), 2)

    def test_reset_clears_all_state(self):
        policy = self.make_policy([response()], collect_safe_features=True)
        policy(observation())
        policy.reset()
        self.assertEqual(policy._env_step, 0)
        self.assertEqual(policy._inference_index, 0)
        self.assertFalse(policy.action_plan)
        self.assertIsNone(policy.pop_inference_record())


if __name__ == "__main__":
    unittest.main()
