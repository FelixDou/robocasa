import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.collect_rollouts import collect_single_rollout


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return {"annotation.human.task_description": "mock task"}, {}

    def step(self, action):
        self.steps += 1
        return {}, 0.0, self.steps >= 3, False, {"success": self.steps >= 3}


class FakePolicy:
    def __init__(self):
        self.reset()

    def reset(self):
        self._env_step = 0
        self.inference_index = 0
        self.pending = None

    def __call__(self, obs, instruction=None):
        if self._env_step % 2 == 0:
            value = self.inference_index + 1
            self.pending = {
                "env_step": self._env_step,
                "environment_step": self._env_step,
                "inference_index": self.inference_index,
                "features": np.full((2, 4, 8), value, dtype=np.float32),
                "actions": np.full((4, 12), value, dtype=np.float32),
                "metadata": {
                    "schema_version": 1,
                    "action_horizon": 4,
                    "flow_steps": 2,
                    "feature_layer": "action_expert_suffix_pre_action_out_proj",
                    "feature_dtype": "float32",
                    "feature_aggregation": "raw",
                    "policy_name": "mock-pi0",
                    "policy_checkpoint": "mock-checkpoint",
                },
            }
            self.inference_index += 1
        self._env_step += 1
        return np.zeros(12, dtype=np.float32)

    def pop_inference_record(self):
        result, self.pending = self.pending, None
        return result


class TestSafeCollection(unittest.TestCase):
    def test_mocked_end_to_end_collection_no_duplicate_cached_features(self):
        result = collect_single_rollout(FakePolicy(), FakeEnv(), horizon=5)
        self.assertTrue(result["success"])
        self.assertEqual(result["inference_env_steps"], [0, 2])
        self.assertEqual(result["features"].shape, (2, 2, 4, 8))
        self.assertEqual(result["policy_action_chunks"].shape, (2, 4, 12))
        self.assertFalse(np.array_equal(result["features"][0], result["features"][1]))

    def test_video_frame_stride_matches_official_subsampling(self):
        env = FakeEnv()

        def never_done_step(environment, action):
            environment.steps += 1
            return {}, 0.0, False, False, {"success": False}

        frames = []
        result = collect_single_rollout(
            FakePolicy(),
            env,
            horizon=6,
            step_fn=never_done_step,
            video_writer=object(),
            frame_fn=lambda environment, writer, obs, previous: frames.append(
                environment.steps
            ),
            video_frame_stride=2,
        )
        self.assertEqual(frames, [1, 3, 5, 6])
        self.assertEqual(result["num_video_frames"], 4)


if __name__ == "__main__":
    unittest.main()
