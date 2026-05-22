"""OpenPI websocket policy adapter for recovery benchmark evaluation."""

from __future__ import annotations

import collections

import numpy as np

from robocasa.utils.env_utils import convert_action


class OpenPIWebsocketPolicy:
    """Callable policy wrapper compatible with ``recovery_rollout.call_policy``."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8120,
        resize_size: int = 224,
        replan_steps: int = 5,
    ):
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, int(port))
        self.image_tools = image_tools
        self.resize_size = int(resize_size)
        self.replan_steps = int(replan_steps)
        self.action_plan = collections.deque()
        self.last_instruction = None

    def __call__(self, obs, instruction=None):
        instruction = instruction or obs["annotation.human.task_description"]
        if instruction != self.last_instruction:
            self.action_plan.clear()
            self.last_instruction = instruction

        if not self.action_plan:
            element = self._make_openpi_observation(obs, instruction)
            action_chunk = self.client.infer(element)["actions"]
            if len(action_chunk) < self.replan_steps:
                raise RuntimeError(
                    "OpenPI policy returned fewer actions than requested "
                    f"replan_steps={self.replan_steps}: {len(action_chunk)}"
                )
            self.action_plan.extend(action_chunk[: self.replan_steps])

        return convert_action(self.action_plan.popleft())

    def _image(self, obs, key):
        image = np.ascontiguousarray(obs[key])
        image = self.image_tools.resize_with_pad(
            image,
            self.resize_size,
            self.resize_size,
        )
        return self.image_tools.convert_to_uint8(image)

    def _make_openpi_observation(self, obs, instruction):
        state = np.concatenate(
            (
                obs["state.end_effector_position_relative"],
                obs["state.end_effector_rotation_relative"],
                obs["state.base_position"],
                obs["state.base_rotation"],
                obs["state.gripper_qpos"],
            ),
            axis=0,
        )
        return {
            "observation/image": self._image(obs, "video.robot0_agentview_left"),
            "observation/wrist_image": self._image(obs, "video.robot0_eye_in_hand"),
            "observation/right_image": self._image(
                obs, "video.robot0_agentview_right"
            ),
            "observation/state": state,
            "prompt": instruction,
        }


def make_policy(env=None, host="127.0.0.1", port=8120, resize_size=224, replan_steps=5):
    return OpenPIWebsocketPolicy(
        host=host,
        port=port,
        resize_size=resize_size,
        replan_steps=replan_steps,
    )
