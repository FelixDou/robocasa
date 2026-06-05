"""RLDX-1 ZeroMQ policy adapter for recovery benchmark evaluation.

RLDX-1 serves policies from its own Python environment via
``rldx/eval/run_rldx_server.py``. This adapter keeps the RoboCasa rollout
process lightweight: it speaks the same msgpack / ZeroMQ protocol as
``rldx.policy.server_client.PolicyClient`` and converts our Gym observation and
action dictionaries to the flat simulator format expected by RLDX.
"""

from __future__ import annotations

import collections
import io
import uuid

import numpy as np

from robocasa.utils.env_utils import convert_action


class MsgSerializer:
    """Small local copy of the RLDX msgpack ndarray serializer."""

    @staticmethod
    def to_bytes(data):
        import msgpack

        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data):
        import msgpack

        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


class RLDXPolicyClient:
    """Minimal client for RLDX's ``PolicyServer`` protocol."""

    def __init__(self, host="127.0.0.1", port=5555, timeout_ms=15000, api_token=None):
        import zmq

        self.zmq = zmq
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.api_token = api_token
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token:
            request["api_token"] = self.api_token
        self.socket.send(MsgSerializer.to_bytes(request))
        try:
            message = self.socket.recv()
        except self.zmq.error.Again as exc:
            raise TimeoutError(
                f"Timed out waiting for RLDX server at {self.host}:{self.port}"
            ) from exc
        if message == b"ERROR":
            raise RuntimeError("RLDX server returned ERROR")
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"RLDX server error: {response['error']}")
        return response

    def ping(self):
        return self.call_endpoint("ping", requires_input=False)

    def get_action(self, observation, options=None):
        response = self.call_endpoint(
            "get_action",
            {"observation": observation, "options": options},
        )
        return tuple(response)

    def reset(self, options=None):
        return self.call_endpoint("reset", {"options": options})

    def close(self):
        self.socket.close()
        self.context.term()


class RLDXZeroMQPolicy:
    """Callable policy wrapper compatible with ``recovery_rollout.call_policy``."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 15000,
        execution_horizon: int = 8,
        video_history: int = 4,
        api_token: str | None = None,
        session_id: str | None = None,
        reset_memory_on_instruction_change: bool = True,
    ):
        self.client = RLDXPolicyClient(
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            api_token=api_token,
        )
        self.execution_horizon = int(execution_horizon)
        self.video_history = int(video_history)
        self.session_id = session_id or f"robocasa-recovery-{uuid.uuid4().hex[:8]}"
        self.reset_memory_on_instruction_change = bool(
            reset_memory_on_instruction_change
        )
        self.action_plan = collections.deque()
        self.last_instruction = None
        self.needs_memory_reset = True

    def __call__(self, obs, instruction=None):
        instruction = instruction or obs["annotation.human.task_description"]
        if instruction != self.last_instruction:
            self.action_plan.clear()
            self.last_instruction = instruction
            if self.reset_memory_on_instruction_change:
                self.needs_memory_reset = True

        if not self.action_plan:
            element = self._make_rldx_observation(obs, instruction)
            options = {
                "session_ids": [self.session_id],
                "reset_memory": [self.needs_memory_reset],
            }
            action_chunk, _ = self.client.get_action(element, options=options)
            self.needs_memory_reset = False
            self.action_plan.extend(self._split_action_chunk(action_chunk))
            if not self.action_plan:
                raise RuntimeError("RLDX policy returned an empty action chunk")

        return self._to_robocasa_action(self.action_plan.popleft())

    def reset(self):
        self.action_plan.clear()
        self.needs_memory_reset = True
        try:
            self.client.reset(options={"session_ids": [self.session_id]})
        except Exception:
            # The next get_action call still carries reset_memory=True.
            pass

    def close(self):
        self.client.close()

    @staticmethod
    def _batch_time(value, dtype):
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 3 and dtype == np.uint8:
            return arr[None, None, ...]
        if arr.ndim == 1:
            return arr[None, None, ...]
        if arr.ndim == 2:
            return arr[None, ...]
        return arr

    def _batch_video_history(self, value):
        arr = self._batch_time(value, np.uint8)
        if arr.ndim >= 5 and arr.shape[1] == 1 and self.video_history > 1:
            arr = np.repeat(arr, self.video_history, axis=1)
        return arr

    def _make_rldx_observation(self, obs, instruction):
        """Map RoboCasa Gym obs to RLDX sim-wrapper flat obs."""

        left_image = self._batch_video_history(obs["video.robot0_agentview_left"])
        right_image = self._batch_video_history(obs["video.robot0_agentview_right"])
        wrist_image = self._batch_video_history(obs["video.robot0_eye_in_hand"])
        video = {
            "robot0_agentview_left": left_image,
            "robot0_agentview_right": right_image,
            "robot0_eye_in_hand": wrist_image,
            "res256_image_side_0": left_image,
            "res256_image_side_1": right_image,
            "res256_image_wrist_0": wrist_image,
        }
        state = {
            "end_effector_position_relative": self._batch_time(
                obs["state.end_effector_position_relative"], np.float32
            ),
            "end_effector_rotation_relative": self._batch_time(
                obs["state.end_effector_rotation_relative"], np.float32
            ),
            "base_position": self._batch_time(obs["state.base_position"], np.float32),
            "base_rotation": self._batch_time(obs["state.base_rotation"], np.float32),
            "gripper_qpos": self._batch_time(obs["state.gripper_qpos"], np.float32),
        }
        language = {
            "instruction": [instruction],
            "task_description": [instruction],
            "annotation.human.action.task_description": [instruction],
            "annotation.human.task_description": [instruction],
        }
        element = {
            "video": video,
            "state": state,
            "language": language,
            "annotation": {
                "human": {
                    "action": {"task_description": [instruction]},
                    "task_description": [instruction],
                }
            },
            "annotation.human.action.task_description": [instruction],
            "annotation.human.task_description": [instruction],
            "language.instruction": [instruction],
            "language.task_description": [instruction],
            "task": [instruction],
        }
        for key, value in video.items():
            element[f"video.{key}"] = value
        for key, value in state.items():
            element[f"state.{key}"] = value
        return element

    def _split_action_chunk(self, action_chunk):
        if isinstance(action_chunk, dict):
            return self._split_action_dict(action_chunk)

        arr = np.asarray(action_chunk, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]
        return list(arr[: self.execution_horizon])

    def _split_action_dict(self, action_chunk):
        horizon = None
        for value in action_chunk.values():
            arr = np.asarray(value)
            if arr.ndim >= 3:
                horizon = arr.shape[1]
                break
            if arr.ndim >= 2:
                horizon = arr.shape[0]
                break
        if horizon is None:
            return [action_chunk]

        steps = []
        for step_i in range(min(horizon, self.execution_horizon)):
            step = {}
            for key, value in action_chunk.items():
                arr = np.asarray(value)
                if arr.ndim >= 3:
                    step[key] = arr[0, step_i]
                elif arr.ndim >= 2:
                    step[key] = arr[step_i]
                else:
                    step[key] = arr
            steps.append(step)
        return steps

    def _to_robocasa_action(self, action):
        if isinstance(action, np.ndarray):
            return convert_action(np.asarray(action, dtype=np.float32))
        if not isinstance(action, dict):
            return convert_action(np.asarray(action, dtype=np.float32))

        if "actions" in action:
            return convert_action(np.asarray(action["actions"], dtype=np.float32))
        if "action" in action and not any(
            key.startswith("action.") for key in action
        ):
            return convert_action(np.asarray(action["action"], dtype=np.float32))

        key_aliases = {
            "action.end_effector_position": (
                "action.end_effector_position",
                "end_effector_position",
                "action.eef_pos_delta",
                "eef_pos_delta",
            ),
            "action.end_effector_rotation": (
                "action.end_effector_rotation",
                "end_effector_rotation",
                "action.eef_rot_delta",
                "eef_rot_delta",
            ),
            "action.gripper_close": (
                "action.gripper_close",
                "gripper_close",
            ),
            "action.base_motion": (
                "action.base_motion",
                "base_motion",
            ),
            "action.control_mode": (
                "action.control_mode",
                "control_mode",
            ),
        }

        output = {}
        for out_key, aliases in key_aliases.items():
            for alias in aliases:
                if alias in action:
                    output[out_key] = np.asarray(action[alias], dtype=np.float32)
                    break

        if "action.gripper" in action and "action.gripper_close" not in output:
            output["action.gripper_close"] = 1.0 - np.asarray(
                action["action.gripper"], dtype=np.float32
            )

        missing = [
            key
            for key in (
                "action.end_effector_position",
                "action.end_effector_rotation",
                "action.gripper_close",
            )
            if key not in output
        ]
        if missing:
            raise KeyError(
                "RLDX action did not contain required RoboCasa action keys: "
                f"{missing}. Available keys: {sorted(action)}"
            )

        output.setdefault("action.base_motion", np.zeros(4, dtype=np.float32))
        output.setdefault("action.control_mode", np.ones(1, dtype=np.float32))
        return output


def make_policy(
    env=None,
    host="127.0.0.1",
    port=5555,
    timeout_ms=15000,
    execution_horizon=8,
    video_history=4,
    api_token=None,
    session_id=None,
    reset_memory_on_instruction_change=True,
):
    return RLDXZeroMQPolicy(
        host=host,
        port=port,
        timeout_ms=timeout_ms,
        execution_horizon=execution_horizon,
        video_history=video_history,
        api_token=api_token,
        session_id=session_id,
        reset_memory_on_instruction_change=reset_memory_on_instruction_change,
    )
