"""ABot-M0.5 websocket policy adapter with raw SAFE feature capture.

The adapter preserves the released ABot RoboCasa chunk protocol: the first
prediction executes only its second frame, later predictions execute the full
chunk, and completed chunks update the server KV cache from observation
keyframes before the next prediction.
"""

from __future__ import annotations

import collections
import copy

import numpy as np


ABOT_M05_SAFE_FEATURE_LAYER = "action_stream_post_norm_pre_action_proj_out"
ABOT_M05_CAMERA_KEYS = {
    "observation.images.robot0_agentview_left": "video.robot0_agentview_left",
    "observation.images.robot0_agentview_right": "video.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand": "video.robot0_eye_in_hand",
}


def _to_uint8_rgb(value):
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC camera image, got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB camera image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _to_robocasa_action(action):
    """Map ABot's released PandaOmron channel order to RoboCasa."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (12,):
        raise RuntimeError(f"ABot action step must have 12 channels, got {action.shape}")
    return {
        "action.end_effector_position": action[5:8],
        "action.end_effector_rotation": action[8:11],
        "action.gripper_close": action[11:12],
        "action.base_motion": action[0:4],
        "action.control_mode": action[4:5],
    }


class ABotM05WebsocketPolicy:
    """Callable policy that reproduces ABot's official RoboCasa client loop."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 29536,
        frame_chunk_size: int = 2,
        action_per_frame: int = 16,
        keyframe_divisor: int = 4,
        replan_steps: int | None = None,
        collect_safe_features: bool = False,
        safe_feature_shape: tuple[int, ...] | None = None,
        policy_name: str | None = None,
        policy_checkpoint: str | None = None,
        client=None,
    ):
        if client is None:
            from wam_client import WebsocketClientPolicy

            client = WebsocketClientPolicy(host=host, port=int(port))
        self.client = client
        self.frame_chunk_size = int(frame_chunk_size)
        self.action_per_frame = int(action_per_frame)
        self.keyframe_divisor = int(keyframe_divisor)
        if self.frame_chunk_size < 1 or self.action_per_frame < 1:
            raise ValueError("ABot frame_chunk_size and action_per_frame must be positive")
        if self.keyframe_divisor < 1:
            raise ValueError("ABot keyframe_divisor must be positive")
        if self.action_per_frame % self.keyframe_divisor:
            raise ValueError("ABot action_per_frame must be divisible by keyframe_divisor")
        self.actions_per_keyframe = self.action_per_frame // self.keyframe_divisor
        self.action_horizon = self.frame_chunk_size * self.action_per_frame
        if replan_steps is not None and int(replan_steps) != self.action_horizon:
            raise ValueError(
                "ABot's released chunk protocol requires replan_steps="
                f"{self.action_horizon}, got {replan_steps}"
            )
        self.replan_steps = self.action_horizon
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
        self._env_step = 0
        self._episode_index = 0
        self._inference_index = 0
        self._pending_inference_record = None
        self._latest_inference_record = None
        self._first_chunk = True
        self._needs_server_reset = True
        self._awaiting_post_step_observation = False
        self._executed_in_chunk = 0
        self._key_frames = []
        self._raw_action_chunk = None

    def __call__(self, obs, instruction=None):
        instruction = instruction or obs["annotation.human.task_description"]
        if instruction != self.last_instruction:
            if self.last_instruction is not None:
                self.reset()
            self.last_instruction = instruction
        if self._needs_server_reset:
            self.client.infer(
                {
                    "reset": True,
                    "prompt": instruction,
                    "episode_tag": f"safe-{self._episode_index:06d}",
                    "episode_name": instruction,
                }
            )
            self._needs_server_reset = False
            self._episode_index += 1

        self._record_post_step_observation(obs)
        if not self.action_plan:
            self._commit_completed_chunk()
            request = {
                "obs": self._format_observation(obs),
                "prompt": instruction,
            }
            if self.collect_safe_features:
                request["request_safe_features"] = True
            response = self.client.infer(request)
            if not isinstance(response, dict) or "action" not in response:
                raise RuntimeError("ABot response is missing required 'action' field")
            raw_action_chunk = self._validate_raw_action_chunk(response["action"])
            action_matrix = self._flatten_action_chunk(raw_action_chunk)
            if self.collect_safe_features:
                self._record_safe_features(response, action_matrix)

            start = self.action_per_frame if self._first_chunk else 0
            self.action_plan.extend(action_matrix[start:])
            if not self.action_plan:
                raise RuntimeError("ABot policy returned an empty executable action chunk")
            self._first_chunk = False
            self._raw_action_chunk = raw_action_chunk
            self._executed_in_chunk = 0
            self._key_frames = []

        action = _to_robocasa_action(self.action_plan.popleft())
        self._awaiting_post_step_observation = True
        self._env_step += 1
        return action

    def _format_observation(self, obs):
        missing = [source for source in ABOT_M05_CAMERA_KEYS.values() if source not in obs]
        if missing:
            raise KeyError(f"ABot observation is missing camera fields: {missing}")
        return {
            target: _to_uint8_rgb(obs[source])
            for target, source in ABOT_M05_CAMERA_KEYS.items()
        }

    def _record_post_step_observation(self, obs):
        if not self._awaiting_post_step_observation:
            return
        self._awaiting_post_step_observation = False
        self._executed_in_chunk += 1
        if self._executed_in_chunk % self.actions_per_keyframe == 0:
            self._key_frames.append(self._format_observation(obs))

    def _commit_completed_chunk(self):
        if self._raw_action_chunk is None:
            return
        if not self._key_frames:
            raise RuntimeError("ABot completed a chunk without observation keyframes")
        self.client.infer(
            {
                "obs": self._key_frames,
                "compute_kv_cache": True,
                "state": self._raw_action_chunk,
            }
        )
        self._raw_action_chunk = None
        self._key_frames = []
        self._executed_in_chunk = 0

    def _validate_raw_action_chunk(self, value):
        chunk = np.asarray(value)
        expected = (12, self.frame_chunk_size, self.action_per_frame)
        if chunk.shape != expected:
            raise RuntimeError(f"ABot action chunk must have shape {expected}, got {chunk.shape}")
        if not np.issubdtype(chunk.dtype, np.number):
            raise RuntimeError(f"ABot action chunk must be numeric, got {chunk.dtype}")
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError("ABot action chunk contains NaN or infinite values")
        return np.asarray(chunk, dtype=np.float32)

    @staticmethod
    def _flatten_action_chunk(chunk):
        """Convert `(channels, frames, substeps)` to `(horizon, channels)`."""
        return np.asarray(chunk, dtype=np.float32).transpose(1, 2, 0).reshape(
            -1, chunk.shape[0]
        )

    def _record_safe_features(self, response, action_matrix):
        if "safe_features" not in response:
            raise RuntimeError(
                "SAFE feature collection was requested, but the ABot server did "
                "not return 'safe_features'. Apply the companion ABot patch."
            )
        features = np.asarray(response["safe_features"])
        if features.ndim == 4:
            if features.shape[0] != 1:
                raise RuntimeError("SAFE collection requires one ABot environment per client")
            features = features[0]
        metadata = response.get("safe_feature_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("ABot SAFE response is missing safe_feature_metadata")
        metadata = copy.deepcopy(metadata)
        if features.ndim != 3:
            raise RuntimeError(
                "Expected ABot SAFE features with shape "
                "(flow_steps, action_horizon, feature_dim), got "
                f"{features.shape}"
            )
        if self.safe_feature_shape is not None and features.shape != self.safe_feature_shape:
            raise RuntimeError(
                f"ABot SAFE feature shape {features.shape} does not match configured "
                f"shape {self.safe_feature_shape}"
            )
        declared_shape = metadata.get("feature_shape")
        if declared_shape is not None and tuple(declared_shape) != features.shape:
            raise RuntimeError(
                f"ABot SAFE metadata declares {declared_shape}, payload has {features.shape}"
            )
        if features.dtype != np.float32:
            raise RuntimeError(f"ABot SAFE features must be float32, got {features.dtype}")
        if not np.all(np.isfinite(features)):
            raise RuntimeError("ABot SAFE features contain NaN or infinite values")
        if not np.any(features):
            raise RuntimeError("ABot SAFE features are entirely zero")
        if action_matrix.shape[0] != features.shape[1]:
            raise RuntimeError(
                "ABot SAFE action-horizon mismatch: actions have "
                f"{action_matrix.shape[0]} positions, features have {features.shape[1]}"
            )

        metadata.setdefault("schema_version", 1)
        metadata.setdefault("model_family", "abot_m05")
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
            raise RuntimeError(f"ABot SAFE metadata is missing required fields: {missing}")
        if int(metadata["schema_version"]) != 1:
            raise RuntimeError("ABot SAFE metadata schema_version is unsupported")
        if metadata["model_family"] != "abot_m05":
            raise RuntimeError("ABot SAFE metadata model_family must be 'abot_m05'")
        if metadata["feature_layer"] != ABOT_M05_SAFE_FEATURE_LAYER:
            raise RuntimeError(
                "ABot SAFE feature_layer is not the action-stream pre-projection "
                f"layer: {metadata['feature_layer']!r}"
            )
        if metadata["feature_aggregation"] != "raw":
            raise RuntimeError("ABot SAFE capture must preserve raw features")
        if metadata["feature_dtype"] != str(features.dtype):
            raise RuntimeError("ABot SAFE metadata feature_dtype disagrees with payload")
        if int(metadata["action_horizon"]) != features.shape[1]:
            raise RuntimeError("ABot SAFE metadata action_horizon disagrees with payload")
        if int(metadata["flow_steps"]) != features.shape[0]:
            raise RuntimeError("ABot SAFE metadata flow_steps disagrees with payload")

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
        record = self._pending_inference_record
        self._pending_inference_record = None
        return record

    @property
    def latest_inference_record(self):
        return self._latest_inference_record

    def reset(self):
        self.action_plan.clear()
        self.last_instruction = None
        self._env_step = 0
        self._inference_index = 0
        self._pending_inference_record = None
        self._latest_inference_record = None
        self._first_chunk = True
        self._needs_server_reset = True
        self._awaiting_post_step_observation = False
        self._executed_in_chunk = 0
        self._key_frames = []
        self._raw_action_chunk = None

    def close(self):
        close = getattr(self.client, "close", None)
        if close is not None:
            close()
            return
        websocket = getattr(self.client, "_ws", None)
        if websocket is not None:
            websocket.close()


def make_policy(
    env=None,
    host="127.0.0.1",
    port=29536,
    frame_chunk_size=2,
    action_per_frame=16,
    keyframe_divisor=4,
    replan_steps=None,
    collect_safe_features=False,
    safe_feature_shape=None,
    policy_name=None,
    policy_checkpoint=None,
):
    return ABotM05WebsocketPolicy(
        host=host,
        port=port,
        frame_chunk_size=frame_chunk_size,
        action_per_frame=action_per_frame,
        keyframe_divisor=keyframe_divisor,
        replan_steps=replan_steps,
        collect_safe_features=collect_safe_features,
        safe_feature_shape=safe_feature_shape,
        policy_name=policy_name,
        policy_checkpoint=policy_checkpoint,
    )
