import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "recovery"
    / "lingbot_vla_websocket_policy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "lingbot_vla_websocket_policy", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LingBotVLAWebsocketPolicy = MODULE.LingBotVLAWebsocketPolicy
_pack_array = MODULE._pack_array
_unpack_array = MODULE._unpack_array


class FakeClient:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.requests = []

    def infer(self, observation):
        self.requests.append(observation)
        if observation.get("reset"):
            return {"action": None}
        return {"action": self.chunks.pop(0)}


def observation(instruction="open the drawer"):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    return {
        "state.base_position": np.arange(3, dtype=np.float32),
        "state.base_rotation": np.arange(4, dtype=np.float32),
        "state.end_effector_position_relative": np.arange(3, dtype=np.float32),
        "state.end_effector_rotation_relative": np.arange(4, dtype=np.float32),
        "state.gripper_qpos": np.arange(2, dtype=np.float32),
        "video.robot0_agentview_left": image,
        "video.robot0_agentview_right": image,
        "video.robot0_eye_in_hand": image,
        "annotation.human.task_description": instruction,
    }


class LingBotVLAWebsocketPolicyTest(unittest.TestCase):
    def test_numpy_messagepack_codec_round_trip_parts(self):
        value = np.arange(6, dtype=np.float32).reshape(2, 3)
        encoded = _pack_array(value)
        decoded = _unpack_array(encoded)
        np.testing.assert_array_equal(decoded, value)

    def test_reset_payload_and_observation_mapping(self):
        client = FakeClient([np.zeros((2, 11), dtype=np.float32)])
        policy = LingBotVLAWebsocketPolicy(
            client=client, replan_steps=2, robo_name="robocasa"
        )
        policy(observation())

        self.assertEqual(client.requests[0], {"reset": True, "robo_name": "robocasa"})
        request = client.requests[1]
        self.assertEqual(request["task"], "open the drawer")
        self.assertEqual(request["observation.state"].shape, (16,))
        self.assertEqual(
            request["observation.images.robot0_agentview_left"].dtype, np.uint8
        )

    def test_action_chunk_is_cached_and_converted(self):
        chunk = np.stack(
            (np.arange(11, dtype=np.float32) / 10, np.arange(11, dtype=np.float32) / 20)
        )
        client = FakeClient([chunk])
        policy = LingBotVLAWebsocketPolicy(client=client, replan_steps=2)

        first = policy(observation())
        second = policy(observation())

        self.assertEqual(len(client.requests), 2)  # reset plus one inference
        np.testing.assert_allclose(first["action.end_effector_position"], chunk[0, :3])
        np.testing.assert_allclose(
            second["action.end_effector_rotation"], chunk[1, 3:6]
        )
        np.testing.assert_allclose(first["action.base_motion"], chunk[0, 7:11])
        np.testing.assert_allclose(first["action.control_mode"], [0.0])

    def test_instruction_change_discards_cached_actions(self):
        client = FakeClient(
            [
                np.zeros((2, 11), dtype=np.float32),
                np.ones((2, 11), dtype=np.float32),
            ]
        )
        policy = LingBotVLAWebsocketPolicy(client=client, replan_steps=2)
        policy(observation("first task"))
        action = policy(observation("second task"))
        np.testing.assert_allclose(action["action.end_effector_position"], 1.0)
        self.assertEqual(len(client.requests), 3)

    def test_bad_action_shape_is_rejected(self):
        client = FakeClient([np.zeros((2, 12), dtype=np.float32)])
        policy = LingBotVLAWebsocketPolicy(client=client, replan_steps=2)
        with self.assertRaisesRegex(RuntimeError, r"shape \(chunk, 11\)"):
            policy(observation())


if __name__ == "__main__":
    unittest.main()
