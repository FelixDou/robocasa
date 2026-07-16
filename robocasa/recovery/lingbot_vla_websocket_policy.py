"""LingBot-VLA-v2 websocket adapter for RoboCasa rollouts.

This module intentionally depends only on small client-side packages.  The
LingBot model runs in its own Python 3.12 / PyTorch 2.8 environment and the
RoboCasa simulator calls it over the MessagePack websocket protocol shipped by
LingBot-VLA-v2.
"""

from __future__ import annotations

import collections
import functools
import time

import numpy as np


def _as_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _pack_array(obj):
    """MessagePack NumPy encoding used by LingBot's ``deploy.msgpack_numpy``."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class LingBotWebsocketClient:
    """Minimal client compatible with LingBot's WebsocketPolicyServer."""

    def __init__(self, host="127.0.0.1", port=9330, connect_timeout=300.0):
        import msgpack
        import websockets.sync.client

        self._msgpack = msgpack
        self._uri = f"ws://{host}:{int(port)}"
        self._packer = functools.partial(msgpack.Packer, default=_pack_array)()
        deadline = time.monotonic() + float(connect_timeout)
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=min(10.0, max(0.1, deadline - time.monotonic())),
                    ping_interval=None,
                    ping_timeout=None,
                )
                self.metadata = self._unpack(self._ws.recv())
                break
            except (ConnectionRefusedError, TimeoutError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for LingBot server at {self._uri}"
                    )
                time.sleep(2.0)

    def _unpack(self, payload):
        return self._msgpack.unpackb(payload, object_hook=_unpack_array)

    def infer(self, observation):
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"LingBot inference server error:\n{response}")
        return self._unpack(response)

    def close(self):
        self._ws.close()


class LingBotVLAWebsocketPolicy:
    """Callable RoboCasa policy backed by a zero-shot LingBot-VLA-v2 server.

    The companion robot config produces an 11-D action in RoboCasa controller
    order: 6-D delta EEF command, gripper, 3-D mobile base, and torso.  RoboCasa's
    controller-mode bit is not represented by LingBot's canonical action space,
    so it is injected as a fixed value to form the required 12-D action.
    """

    REQUIRED_OBSERVATIONS = (
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    )

    def __init__(
        self,
        host="127.0.0.1",
        port=9330,
        robo_name="robocasa",
        replan_steps=8,
        fixed_control_mode=0.0,
        clip_actions=True,
        wrist_right_source="agentview_right",
        zero_base_motion=False,
        reset_on_connect=True,
        client=None,
    ):
        self.client = client or LingBotWebsocketClient(host=host, port=int(port))
        self.robo_name = str(robo_name)
        self.replan_steps = int(replan_steps)
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        self.fixed_control_mode = float(fixed_control_mode)
        self.clip_actions = _as_bool(clip_actions)
        self.wrist_right_source = str(wrist_right_source).strip().lower()
        if self.wrist_right_source not in {"agentview_right", "eye_in_hand"}:
            raise ValueError(
                "wrist_right_source must be 'agentview_right' or 'eye_in_hand', "
                f"got {wrist_right_source!r}"
            )
        self.zero_base_motion = _as_bool(zero_base_motion)
        self.action_plan = collections.deque()
        self.last_instruction = None
        if reset_on_connect:
            self._reset_remote()

    def _reset_remote(self):
        response = self.client.infer({"reset": True, "robo_name": self.robo_name})
        if isinstance(response, dict) and response.get("action", None) is not None:
            raise RuntimeError("LingBot reset unexpectedly returned an action")

    def reset(self):
        self.action_plan.clear()
        self.last_instruction = None
        self._reset_remote()

    def close(self):
        close = getattr(self.client, "close", None)
        if close is not None:
            close()

    def __call__(self, obs, instruction=None):
        instruction = instruction or obs["annotation.human.task_description"]
        if instruction != self.last_instruction:
            self.action_plan.clear()
            self.last_instruction = instruction

        if not self.action_plan:
            response = self.client.infer(
                self._make_lingbot_observation(obs, instruction)
            )
            action_chunk = self._extract_action_chunk(response)
            if action_chunk.shape[0] < self.replan_steps:
                raise RuntimeError(
                    "LingBot returned fewer actions than requested: "
                    f"{action_chunk.shape[0]} < replan_steps={self.replan_steps}"
                )
            self.action_plan.extend(action_chunk[: self.replan_steps])

        return self._convert_action(self.action_plan.popleft())

    def _make_lingbot_observation(self, obs, instruction):
        missing = [key for key in self.REQUIRED_OBSERVATIONS if key not in obs]
        if missing:
            raise KeyError(f"RoboCasa observation is missing LingBot inputs: {missing}")

        state = np.concatenate(
            (
                np.asarray(obs["state.base_position"], dtype=np.float32),
                np.asarray(obs["state.base_rotation"], dtype=np.float32),
                np.asarray(
                    obs["state.end_effector_position_relative"], dtype=np.float32
                ),
                np.asarray(
                    obs["state.end_effector_rotation_relative"], dtype=np.float32
                ),
                np.asarray(obs["state.gripper_qpos"], dtype=np.float32),
            )
        )
        if state.shape != (16,):
            raise ValueError(f"Expected 16-D PandaOmron state, got {state.shape}")
        wrist_right_key = (
            "video.robot0_eye_in_hand"
            if self.wrist_right_source == "eye_in_hand"
            else "video.robot0_agentview_right"
        )
        return {
            "observation.state": state,
            "observation.images.robot0_agentview_left": self._image(
                obs["video.robot0_agentview_left"]
            ),
            "observation.images.robot0_agentview_right": self._image(
                obs[wrist_right_key]
            ),
            "observation.images.robot0_eye_in_hand": self._image(
                obs["video.robot0_eye_in_hand"]
            ),
            "task": str(instruction),
        }

    @staticmethod
    def _image(value):
        image = np.ascontiguousarray(value)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")
        if image.dtype != np.uint8:
            if (
                np.issubdtype(image.dtype, np.floating)
                and image.size
                and image.max() <= 1.0
            ):
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _extract_action_chunk(response):
        if not isinstance(response, dict) or "action" not in response:
            keys = (
                sorted(response)
                if isinstance(response, dict)
                else type(response).__name__
            )
            raise RuntimeError(f"LingBot response is missing 'action'; received {keys}")
        actions = np.asarray(response["action"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 11:
            raise RuntimeError(
                "Expected LingBot RoboCasa actions with shape (chunk, 11), got "
                f"{actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("LingBot action chunk contains NaN or infinite values")
        return actions

    def _convert_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        # RoboCasa LeRobot trajectories store ``gripper_close`` in the signed
        # controller convention (-1=open, +1=closed), which is therefore the
        # convention used for LingBot normalization. RoboCasaGymEnv's dict API
        # instead thresholds this field at 0.5 as a [0, 1] close command.
        gripper_close = (
            np.clip(action[6:7], -1.0, 1.0) + np.float32(1.0)
        ) / np.float32(2.0)
        base_motion = np.concatenate((action[7:10], action[10:11]))
        if self.zero_base_motion:
            base_motion = np.zeros(4, dtype=np.float32)
        return {
            "action.end_effector_position": action[0:3],
            "action.end_effector_rotation": action[3:6],
            "action.gripper_close": gripper_close,
            "action.base_motion": base_motion,
            "action.control_mode": np.asarray(
                [self.fixed_control_mode], dtype=np.float32
            ),
        }


def make_policy(
    env=None,
    host="127.0.0.1",
    port=9330,
    robo_name="robocasa",
    replan_steps=8,
    fixed_control_mode=0.0,
    clip_actions=True,
    wrist_right_source="agentview_right",
    zero_base_motion=False,
):
    """Factory consumed by ``--policy-module module:make_policy``."""
    return LingBotVLAWebsocketPolicy(
        host=host,
        port=port,
        robo_name=robo_name,
        replan_steps=replan_steps,
        fixed_control_mode=fixed_control_mode,
        clip_actions=clip_actions,
        wrist_right_source=wrist_right_source,
        zero_base_motion=zero_base_motion,
    )
