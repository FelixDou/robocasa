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


def safe_response(shape=(5, 30, 8), action_value=0.0):
    features = np.arange(np.prod(shape), dtype=np.float32).reshape((1, *shape))
    return {
        "actions": np.full((1, shape[1], 60), action_value, dtype=np.float32),
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


class FakeScorer:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def score_candidates(self, features, *, task_name):
        self.calls.append((np.asarray(features).copy(), task_name))
        return {
            "protocol": "test_single_inference",
            "scores": self.scores,
            "seeds": [0, 1, 2],
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
        self.assertEqual(
            first["auxiliary_features"]["observation_state_history"].shape,
            (4 * 60,),
        )
        self.assertTrue(
            np.all(
                np.isfinite(
                    first["auxiliary_features"]["observation_state_history"]
                )
            )
        )
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

    def test_optional_sampling_seed_is_deterministic_per_inference(self):
        actions = np.zeros((1, 30, 60), dtype=np.float32)
        policy = self.make_policy(
            [actions, actions],
            sampling_seed_base=100,
        )
        policy(observation())
        policy(observation())
        policy(observation())
        self.assertEqual(
            [request["sampling_seed"] for request in policy.client.requests],
            [100, 101],
        )

    def test_disabled_best_of_k_recovery_hook_preserves_default_plan(self):
        actions = np.zeros((1, 30, 60), dtype=np.float32)
        policy = self.make_policy([actions])
        policy(observation())
        self.assertEqual(len(policy.action_plan), 1)
        activation = policy.begin_recovery(task_name="CloseBlenderLid")
        self.assertFalse(activation["enabled"])
        self.assertEqual(len(policy.action_plan), 1)

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

    def test_best_of_k_is_recovery_only_and_selects_lowest_safe(self):
        scorer = FakeScorer([0.7, -0.4, 0.2])
        action_only = np.zeros((1, 30, 60), dtype=np.float32)
        policy = self.make_policy(
            [
                action_only,
                safe_response(action_value=1.0),
                safe_response(action_value=2.0),
                safe_response(action_value=3.0),
            ],
            safe_best_of_k=3,
            safe_candidate_seed=40,
            candidate_scorer=scorer,
        )

        policy(observation())
        self.assertEqual(len(policy.client.requests), 1)
        self.assertNotIn("request_safe_features", policy.client.requests[0])
        self.assertIsNone(policy.pop_candidate_selection_record())

        activation = policy.begin_recovery(
            task_name="CloseBlenderLid",
            target_subtask="close_lid",
            instruction="close the blender lid",
        )
        action = policy(observation())
        self.assertTrue(activation["enabled"])
        self.assertEqual(len(policy.client.requests), 4)
        recovery_requests = policy.client.requests[1:]
        self.assertEqual(
            [request["sampling_seed"] for request in recovery_requests], [40, 41, 42]
        )
        self.assertTrue(
            all(request["request_safe_features"] for request in recovery_requests)
        )
        self.assertEqual(float(action["action.end_effector_position"][0]), 2.0)
        self.assertEqual(scorer.calls[0][0].shape, (3, 5, 30, 8))
        self.assertEqual(scorer.calls[0][1], "CloseBlenderLid")

        selection = policy.pop_candidate_selection_record()
        self.assertEqual(selection["selected_index"], 1)
        self.assertEqual(selection["scores"], [0.7, -0.4, 0.2])
        self.assertEqual(selection["candidate_count"], 3)
        self.assertFalse(selection["action_diversity"]["all_identical"])
        chosen = policy.pop_inference_record()
        self.assertEqual(float(chosen["actions"][0, 0]), 2.0)

    def test_highest_safe_control_selects_largest_score(self):
        scorer = FakeScorer([0.1, 0.9])
        policy = self.make_policy(
            [safe_response(action_value=1.0), safe_response(action_value=4.0)],
            safe_best_of_k=2,
            safe_candidate_strategy="highest_safe",
            candidate_scorer=scorer,
        )
        policy.begin_recovery(task_name="CloseBlenderLid")
        action = policy(observation())
        self.assertEqual(float(action["action.end_effector_position"][0]), 4.0)
        self.assertEqual(policy.pop_candidate_selection_record()["selected_index"], 1)

    def test_random_control_is_deterministic(self):
        scorer = FakeScorer([0.1, 0.2, 0.3])
        policy = self.make_policy(
            [
                safe_response(action_value=1.0),
                safe_response(action_value=2.0),
                safe_response(action_value=3.0),
            ],
            safe_best_of_k=3,
            safe_candidate_strategy="random",
            safe_candidate_seed=17,
            candidate_scorer=scorer,
        )
        policy.begin_recovery(task_name="CloseBlenderLid")
        policy(observation())
        expected = int(np.random.default_rng(17).integers(3))
        self.assertEqual(
            policy.pop_candidate_selection_record()["selected_index"], expected
        )


if __name__ == "__main__":
    unittest.main()
