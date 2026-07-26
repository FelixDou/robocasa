"""RLDX-1 ZeroMQ policy adapter for recovery benchmark evaluation.

RLDX-1 serves policies from its own Python environment via
``rldx/eval/run_rldx_server.py``. This adapter keeps the RoboCasa rollout
process lightweight: it speaks the same msgpack / ZeroMQ protocol as
``rldx.policy.server_client.PolicyClient`` and converts our Gym observation and
action dictionaries to the flat simulator format expected by RLDX.
"""

from __future__ import annotations

import collections
import copy
import io
import uuid

import numpy as np

from robocasa.utils.env_utils import convert_action


RLDX_SAFE_FEATURE_LAYER = "action_model_msat_action_suffix_pre_action_decoder"


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
        collect_safe_features: bool = False,
        safe_feature_shape: tuple[int, ...] | None = None,
        policy_name: str | None = None,
        policy_checkpoint: str | None = None,
        client=None,
    ):
        self.client = client or RLDXPolicyClient(
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            api_token=api_token,
        )
        self.execution_horizon = int(execution_horizon)
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        self.replan_steps = self.execution_horizon
        self.video_history = int(video_history)
        self.session_id = session_id or f"robocasa-recovery-{uuid.uuid4().hex[:8]}"
        self.reset_memory_on_instruction_change = bool(
            reset_memory_on_instruction_change
        )
        self.collect_safe_features = bool(collect_safe_features)
        self.safe_feature_shape = (
            tuple(int(value) for value in safe_feature_shape)
            if safe_feature_shape is not None
            else None
        )
        self.policy_name = policy_name
        self.policy_checkpoint = policy_checkpoint
        self.action_plan = collections.deque()
        self.last_instruction = None
        self.needs_memory_reset = True
        self._env_step = 0
        self._inference_index = 0
        self._pending_inference_record = None
        self._latest_inference_record = None

    def __call__(self, obs, instruction=None):
        instruction = instruction or obs["annotation.human.task_description"]
        if instruction != self.last_instruction:
            self.action_plan.clear()
            self._pending_inference_record = None
            self._latest_inference_record = None
            self._inference_index = 0
            self.last_instruction = instruction
            if self.reset_memory_on_instruction_change:
                self.needs_memory_reset = True

        if not self.action_plan:
            element = self._make_rldx_observation(obs, instruction)
            options = {
                "session_ids": [self.session_id],
                "reset_memory": [self.needs_memory_reset],
            }
            if self.collect_safe_features:
                options["request_safe_features"] = True
            action_chunk, info = self.client.get_action(element, options=options)
            self.needs_memory_reset = False
            if self.collect_safe_features:
                self._record_safe_features(info, action_chunk)
            self.action_plan.extend(self._split_action_chunk(action_chunk))
            if not self.action_plan:
                raise RuntimeError("RLDX policy returned an empty action chunk")

        action = self._to_robocasa_action(self.action_plan.popleft())
        self._env_step += 1
        return action

    @staticmethod
    def _action_chunk_matrix(action_chunk):
        """Return the server action chunk as one `(horizon, action_dim)` array."""
        if not isinstance(action_chunk, dict):
            array = np.asarray(action_chunk, dtype=np.float32)
            if array.ndim == 3:
                if array.shape[0] != 1:
                    raise RuntimeError(
                        "SAFE collection requires one RLDX environment per client"
                    )
                array = array[0]
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2:
                raise RuntimeError(
                    "RLDX action chunk must be (horizon, action_dim), got "
                    f"{array.shape}"
                )
            return array

        parts = []
        horizon = None
        for key in sorted(action_chunk):
            array = np.asarray(action_chunk[key], dtype=np.float32)
            if array.ndim == 3:
                if array.shape[0] != 1:
                    raise RuntimeError(
                        "SAFE collection requires one RLDX environment per client"
                    )
                array = array[0]
            if array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2:
                raise RuntimeError(
                    f"RLDX action field {key!r} must be (horizon, dim), got "
                    f"{array.shape}"
                )
            if horizon is None:
                horizon = array.shape[0]
            elif array.shape[0] != horizon:
                raise RuntimeError("RLDX action fields disagree on action horizon")
            parts.append(array)
        if not parts:
            raise RuntimeError("RLDX returned an empty action dictionary")
        return np.concatenate(parts, axis=-1)

    def _record_safe_features(self, info, action_chunk):
        if not isinstance(info, dict) or "safe_features" not in info:
            raise RuntimeError(
                "SAFE feature collection was requested, but the RLDX server "
                "did not return info['safe_features']. Apply the companion "
                "RLDX patch and request SAFE features."
            )
        features = np.asarray(info["safe_features"])
        if features.ndim == 4:
            if features.shape[0] != 1:
                raise RuntimeError(
                    "SAFE collection requires one RLDX environment per client"
                )
            features = features[0]
        metadata = info.get("safe_feature_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "RLDX SAFE response is missing dict 'safe_feature_metadata'"
            )
        metadata = copy.deepcopy(metadata)
        if features.ndim != 3:
            raise RuntimeError(
                "Expected raw RLDX SAFE features with shape "
                "(denoising_steps, action_horizon, feature_dim), got "
                f"{features.shape}"
            )
        if self.safe_feature_shape is not None and features.shape != self.safe_feature_shape:
            raise RuntimeError(
                f"SAFE feature shape {features.shape} does not match configured "
                f"shape {self.safe_feature_shape}"
            )
        declared_shape = metadata.get("feature_shape")
        if declared_shape is not None and tuple(declared_shape) != features.shape:
            raise RuntimeError(
                f"SAFE metadata declares shape {declared_shape}, but payload has "
                f"shape {features.shape}"
            )
        if features.dtype != np.float32:
            raise RuntimeError(f"RLDX SAFE features must be float32, got {features.dtype}")
        if not np.all(np.isfinite(features)):
            raise RuntimeError("RLDX SAFE features contain NaN or infinite values")
        if not np.any(features):
            raise RuntimeError("RLDX SAFE features are entirely zero")

        action_matrix = self._action_chunk_matrix(action_chunk)
        if not np.all(np.isfinite(action_matrix)):
            raise RuntimeError("RLDX action chunk contains NaN or infinite values")
        if action_matrix.shape[0] != features.shape[1]:
            raise RuntimeError(
                "RLDX SAFE action-horizon mismatch: actions have "
                f"{action_matrix.shape[0]} positions but features have "
                f"{features.shape[1]}"
            )

        metadata.setdefault("schema_version", 1)
        metadata.setdefault("model_family", "rldx1")
        metadata.setdefault("feature_aggregation", metadata.get("aggregation", "raw"))
        metadata.setdefault("aggregation", metadata["feature_aggregation"])
        metadata.setdefault("policy_name", self.policy_name or metadata.get("model_id"))
        metadata.setdefault(
            "policy_checkpoint", self.policy_checkpoint or metadata.get("checkpoint")
        )
        metadata.setdefault("feature_dtype", str(features.dtype))
        metadata.setdefault("feature_shape", list(features.shape))
        metadata.setdefault("action_horizon", int(features.shape[1]))
        metadata.setdefault("flow_steps", int(features.shape[0]))
        required = (
            "schema_version",
            "model_family",
            "feature_layer",
            "feature_shape",
            "feature_dtype",
            "feature_aggregation",
            "policy_name",
            "policy_checkpoint",
            "action_horizon",
            "flow_steps",
        )
        missing = [key for key in required if metadata.get(key) is None]
        if missing:
            raise RuntimeError(f"RLDX SAFE metadata is missing required fields: {missing}")
        if int(metadata["schema_version"]) != 1:
            raise RuntimeError("RLDX SAFE metadata schema_version is unsupported")
        if metadata["model_family"] != "rldx1":
            raise RuntimeError("RLDX SAFE metadata model_family must be 'rldx1'")
        if metadata["feature_layer"] != RLDX_SAFE_FEATURE_LAYER:
            raise RuntimeError(
                "RLDX SAFE feature_layer is not the action-token pre-decoder "
                f"layer: {metadata['feature_layer']!r}"
            )
        if metadata["feature_aggregation"] != "raw":
            raise RuntimeError("RLDX SAFE inference capture must preserve raw features")
        if metadata["feature_dtype"] != str(features.dtype):
            raise RuntimeError("RLDX SAFE metadata feature_dtype disagrees with payload")
        if int(metadata["action_horizon"]) != features.shape[1]:
            raise RuntimeError("RLDX SAFE metadata action_horizon disagrees with payload")
        if int(metadata["flow_steps"]) != features.shape[0]:
            raise RuntimeError("RLDX SAFE metadata flow_steps disagrees with payload")

        record = {
            "env_step": self._env_step,
            "environment_step": self._env_step,
            "inference_index": self._inference_index,
            "features": np.asarray(features, dtype=np.float32),
            "actions": np.asarray(action_matrix, dtype=np.float32),
            "metadata": metadata,
        }
        self._inference_index += 1
        self._pending_inference_record = record
        self._latest_inference_record = record

    def pop_inference_record(self):
        """Return one record for the latest real inference, never cached actions."""
        record = self._pending_inference_record
        self._pending_inference_record = None
        return record

    @property
    def latest_inference_record(self):
        return self._latest_inference_record

    def reset(self):
        self.action_plan.clear()
        self.last_instruction = None
        self.needs_memory_reset = True
        self._env_step = 0
        self._inference_index = 0
        self._pending_inference_record = None
        self._latest_inference_record = None
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
            "instruction": [[instruction]],
            "task_description": [[instruction]],
            "annotation.human.action.task_description": [[instruction]],
            "annotation.human.task_description": [[instruction]],
        }
        element = {
            "video": video,
            "state": state,
            "language": language,
            "annotation": {
                "human": {
                    "action": {"task_description": [[instruction]]},
                    "task_description": [[instruction]],
                }
            },
            # The RLDX simulation policy wrapper consumes flat language fields
            # as one string per batch item, then adds the temporal dimension
            # before forwarding to the underlying policy. Keep the nested
            # ``language`` representation above for direct-policy compatibility.
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
    collect_safe_features=False,
    safe_feature_shape=None,
    policy_name=None,
    policy_checkpoint=None,
    replan_steps=None,
):
    if replan_steps is not None:
        execution_horizon = int(replan_steps)
    return RLDXZeroMQPolicy(
        host=host,
        port=port,
        timeout_ms=timeout_ms,
        execution_horizon=execution_horizon,
        video_history=video_history,
        api_token=api_token,
        session_id=session_id,
        reset_memory_on_instruction_change=reset_memory_on_instruction_change,
        collect_safe_features=collect_safe_features,
        safe_feature_shape=safe_feature_shape,
        policy_name=policy_name,
        policy_checkpoint=policy_checkpoint,
    )
