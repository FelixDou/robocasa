import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.abot_websocket_policy import (
    ABOT_M05_SAFE_FEATURE_LAYER,
    ABotM05WebsocketPolicy,
)


def observation(instruction="open the drawer"):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "annotation.human.task_description": instruction,
        "video.robot0_agentview_left": image,
        "video.robot0_agentview_right": image,
        "video.robot0_eye_in_hand": image,
    }


def inference_response(*, feature_shape=(1, 5, 32, 8)):
    features = np.ones(feature_shape, dtype=np.float32)
    return {
        "action": np.arange(12 * 2 * 16, dtype=np.float32).reshape(12, 2, 16),
        "safe_features": features,
        "safe_feature_metadata": {
            "schema_version": 1,
            "model_family": "abot_m05",
            "feature_layer": ABOT_M05_SAFE_FEATURE_LAYER,
            "feature_shape": list(feature_shape[1:]),
            "feature_dtype": "float32",
            "feature_aggregation": "raw",
            "policy_name": "mock-abot",
            "policy_checkpoint": "mock-checkpoint",
            "action_horizon": feature_shape[2],
            "flow_steps": feature_shape[1],
        },
    }


class FakeClient:
    def __init__(self, responses=None):
        self.requests = []
        self.responses = list(responses or [])
        self.closed = False

    def infer(self, request):
        self.requests.append(request)
        if request.get("reset") or request.get("compute_kv_cache"):
            return {}
        if not self.responses:
            raise AssertionError("Unexpected ABot inference request")
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class TestABotM05WebsocketPolicy(unittest.TestCase):
    def make_policy(self, client):
        return ABotM05WebsocketPolicy(
            client=client,
            collect_safe_features=True,
            replan_steps=32,
            policy_name="mock-abot",
            policy_checkpoint="mock-checkpoint",
        )

    def test_preserves_official_first_chunk_and_kv_cache_protocol(self):
        client = FakeClient([inference_response(), inference_response()])
        policy = self.make_policy(client)
        obs = observation()

        first_action = policy(obs)
        first_record = policy.pop_inference_record()
        self.assertEqual(first_record["environment_step"], 0)
        self.assertEqual(first_record["features"].shape, (5, 32, 8))
        self.assertEqual(first_record["actions"].shape, (32, 12))
        self.assertEqual(first_record["metadata"]["model_family"], "abot_m05")
        self.assertTrue(client.requests[1]["request_safe_features"])
        expected_first = inference_response()["action"][:, 1, 0]
        self.assertTrue(
            np.array_equal(
                np.concatenate(tuple(first_action.values())),
                np.concatenate(
                    (
                        expected_first[5:8],
                        expected_first[8:11],
                        expected_first[11:12],
                        expected_first[0:4],
                        expected_first[4:5],
                    )
                ),
            )
        )

        for _ in range(15):
            policy(obs)
            self.assertIsNone(policy.pop_inference_record())
        policy(obs)
        second_record = policy.pop_inference_record()

        self.assertEqual(second_record["environment_step"], 16)
        self.assertEqual(len(client.requests), 4)
        kv_request = client.requests[2]
        self.assertTrue(kv_request["compute_kv_cache"])
        self.assertEqual(len(kv_request["obs"]), 4)
        self.assertEqual(kv_request["state"].shape, (12, 2, 16))
        self.assertTrue(client.requests[3]["request_safe_features"])

    def test_reset_uses_unique_episode_tags(self):
        client = FakeClient([inference_response(), inference_response()])
        policy = self.make_policy(client)
        policy(observation())
        first_tag = client.requests[0]["episode_tag"]
        policy.reset()
        policy(observation())
        second_tag = client.requests[2]["episode_tag"]
        self.assertNotEqual(first_tag, second_tag)

    def test_rejects_missing_or_invalid_safe_features(self):
        missing = inference_response()
        missing.pop("safe_features")
        with self.assertRaisesRegex(RuntimeError, "did not return 'safe_features'"):
            self.make_policy(FakeClient([missing]))(observation())

        wrong_family = inference_response()
        wrong_family["safe_feature_metadata"]["model_family"] = "pi0"
        with self.assertRaisesRegex(RuntimeError, "model_family"):
            self.make_policy(FakeClient([wrong_family]))(observation())

        wrong_horizon = inference_response(feature_shape=(1, 5, 31, 8))
        with self.assertRaisesRegex(RuntimeError, "action-horizon mismatch"):
            self.make_policy(FakeClient([wrong_horizon]))(observation())

    def test_rejects_wrong_action_chunk_shape_and_replan_interval(self):
        response = inference_response()
        response["action"] = np.zeros((12, 32), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "action chunk must have shape"):
            self.make_policy(FakeClient([response]))(observation())

        with self.assertRaisesRegex(ValueError, "requires replan_steps=32"):
            ABotM05WebsocketPolicy(client=FakeClient(), replan_steps=16)

    def test_close_closes_client(self):
        client = FakeClient()
        policy = self.make_policy(client)
        policy.close()
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
