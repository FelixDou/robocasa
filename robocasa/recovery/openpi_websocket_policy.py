"""OpenPI websocket policy adapter for recovery benchmark evaluation."""

from __future__ import annotations

import collections
import copy

import numpy as np


def _convert_action(action):
    """Convert an OpenPI vector without importing the simulator-heavy env utils."""
    action = np.asarray(action).copy()
    return {
        "action.end_effector_position": action[0:3],
        "action.end_effector_rotation": action[3:6],
        "action.gripper_close": action[6:7],
        "action.base_motion": action[7:11],
        "action.control_mode": action[11:12],
    }


class OpenPIWebsocketPolicy:
    """Callable policy wrapper compatible with ``recovery_rollout.call_policy``."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8120,
        resize_size: int = 224,
        replan_steps: int = 5,
        collect_safe_features: bool = False,
        safe_feature_shape: tuple[int, ...] | None = None,
        policy_name: str | None = None,
        policy_checkpoint: str | None = None,
        client=None,
        image_tools=None,
    ):
        if client is None or image_tools is None:
            from openpi_client import image_tools as openpi_image_tools
            from openpi_client import websocket_client_policy

        self.client = client or websocket_client_policy.WebsocketClientPolicy(
            host, int(port)
        )
        self.image_tools = image_tools or openpi_image_tools
        self.resize_size = int(resize_size)
        self.replan_steps = int(replan_steps)
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        self.collect_safe_features = bool(collect_safe_features)
        self.policy_name = policy_name
        self.policy_checkpoint = policy_checkpoint
        self.safe_feature_shape = (
            tuple(int(x) for x in safe_feature_shape)
            if safe_feature_shape is not None
            else None
        )
        self.action_plan = collections.deque()
        self.last_instruction = None
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

        if not self.action_plan:
            element = self._make_openpi_observation(obs, instruction)
            response = self.client.infer(element)
            if "actions" not in response:
                raise RuntimeError("OpenPI response is missing required 'actions' field")
            action_chunk = response["actions"]
            if len(action_chunk) < self.replan_steps:
                raise RuntimeError(
                    "OpenPI policy returned fewer actions than requested "
                    f"replan_steps={self.replan_steps}: {len(action_chunk)}"
                )
            self.action_plan.extend(action_chunk[: self.replan_steps])
            if self.collect_safe_features:
                self._record_safe_features(response, action_chunk)

        action = _convert_action(self.action_plan.popleft())
        self._env_step += 1
        return action

    def _record_safe_features(self, response, action_chunk):
        if "safe_features" not in response:
            raise RuntimeError(
                "SAFE feature collection was requested, but the OpenPI server "
                "did not return 'safe_features'. Apply the companion OpenPI patch "
                "and start the server with SAFE features enabled."
            )
        features = np.asarray(response["safe_features"])
        metadata = response.get("safe_feature_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "OpenPI SAFE response is missing dict 'safe_feature_metadata'"
            )
        metadata = copy.deepcopy(metadata)
        if features.ndim != 3:
            raise RuntimeError(
                "Expected raw SAFE features with shape "
                "(flow_steps, action_horizon, feature_dim), got "
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
            raise RuntimeError(f"Official SAFE features must be float32, got {features.dtype}")
        if not np.all(np.isfinite(features)):
            raise RuntimeError("SAFE features contain NaN or infinite values")
        if not np.any(features):
            raise RuntimeError("SAFE features are entirely zero; refusing a dummy payload")
        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim != 2 or not np.issubdtype(action_chunk.dtype, np.number):
            raise RuntimeError(
                f"OpenPI actions must have shape (action_horizon, action_dim), got {action_chunk.shape}"
            )
        if not np.all(np.isfinite(action_chunk)):
            raise RuntimeError("OpenPI action chunk contains NaN or infinite values")
        if action_chunk.shape[0] != features.shape[1]:
            raise RuntimeError(
                "SAFE action-horizon mismatch: actions have "
                f"{action_chunk.shape[0]} positions but features have {features.shape[1]}"
            )
        metadata.setdefault("schema_version", 1)
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
        required_metadata = (
            "schema_version",
            "feature_layer",
            "feature_shape",
            "feature_dtype",
            "feature_aggregation",
            "policy_name",
            "policy_checkpoint",
            "action_horizon",
            "flow_steps",
        )
        missing = [key for key in required_metadata if metadata.get(key) is None]
        if missing:
            raise RuntimeError(f"SAFE metadata is missing required fields: {missing}")
        if int(metadata["schema_version"]) != 1:
            raise RuntimeError("SAFE metadata schema_version is unsupported")
        if metadata["feature_layer"] != "action_expert_suffix_pre_action_out_proj":
            raise RuntimeError(
                f"SAFE feature_layer is not the official π0 pre-velocity layer: "
                f"{metadata['feature_layer']!r}"
            )
        if metadata["feature_aggregation"] != "raw":
            raise RuntimeError("SAFE inference capture must preserve raw features")
        if metadata["feature_dtype"] != str(features.dtype):
            raise RuntimeError("SAFE metadata feature_dtype disagrees with feature payload")
        if int(metadata["action_horizon"]) != features.shape[1]:
            raise RuntimeError("SAFE metadata action_horizon disagrees with feature payload")
        if int(metadata["flow_steps"]) != features.shape[0]:
            raise RuntimeError("SAFE metadata flow_steps disagrees with feature payload")
        record = {
            "env_step": self._env_step,
            "environment_step": self._env_step,
            "inference_index": self._inference_index,
            "features": np.asarray(features, dtype=np.float32),
            "actions": np.asarray(action_chunk, dtype=np.float32),
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
        """Clear action and feature state before starting a new rollout."""
        self.action_plan.clear()
        self.last_instruction = None
        self._env_step = 0
        self._inference_index = 0
        self._pending_inference_record = None
        self._latest_inference_record = None

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
        element = {
            "observation/image": self._image(obs, "video.robot0_agentview_left"),
            "observation/wrist_image": self._image(obs, "video.robot0_eye_in_hand"),
            "observation/right_image": self._image(
                obs, "video.robot0_agentview_right"
            ),
            "observation/state": state,
            "prompt": instruction,
        }
        if self.collect_safe_features:
            element["request_safe_features"] = True
        return element


def make_policy(
    env=None,
    host="127.0.0.1",
    port=8120,
    resize_size=224,
    replan_steps=5,
    collect_safe_features=False,
    safe_feature_shape=None,
    policy_name=None,
    policy_checkpoint=None,
):
    return OpenPIWebsocketPolicy(
        host=host,
        port=port,
        resize_size=resize_size,
        replan_steps=replan_steps,
        collect_safe_features=collect_safe_features,
        safe_feature_shape=safe_feature_shape,
        policy_name=policy_name,
        policy_checkpoint=policy_checkpoint,
    )
