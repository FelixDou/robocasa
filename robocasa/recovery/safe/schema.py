"""Versioned schema and compatibility checks for raw SAFE rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

import numpy as np


SAFE_SCHEMA_VERSION = 1
SAFE_FEATURE_SCHEMA_VERSION = 1
OFFICIAL_PI0_FEATURE_LAYER = "action_expert_suffix_pre_action_out_proj"
RLDX1_FEATURE_LAYER = "action_model_msat_action_suffix_pre_action_decoder"
RLDX1_OBSERVATION_FEATURE_LAYER = (
    "memory_aware_backbone_mean_plus_state_encoder_mean"
)
RLDX1_FUSED_EXPORT_LAYER = (
    "action_model_msat_final_step_horizon_mean_plus_memory_aware_observation_context"
)


@dataclass
class SafeRolloutMetadata:
    rollout_id: str
    task_name: str
    task_instruction: str
    environment_seed: int
    policy_id: str
    checkpoint: str
    failed: bool
    num_env_steps: int
    inference_env_steps: list[int]
    valid_sequence_length: int
    action_horizon: int
    replan_steps: int
    feature_layer: str = OFFICIAL_PI0_FEATURE_LAYER
    model_family: str = "pi0"
    feature_aggregation: str = "raw"
    feature_shape: list[int] = field(default_factory=list)
    feature_dtype: str = "float32"
    feature_mode: str = "action"
    observation_feature_layer: str | None = None
    observation_context_shape: list[int] = field(default_factory=list)
    observation_components: dict[str, list[int]] = field(default_factory=dict)
    observation_context_pooling: dict[str, str] = field(default_factory=dict)
    flow_steps: int = 0
    termination_reason: str = "unknown"
    timeout_horizon: int = 0
    video_path: str | None = None
    environment_split: str | None = None
    seed_protocol: str = "rollout_index"
    environment_reset_index: int | None = None
    video_frame_stride: int = 1
    unused_subtask_metadata: dict[str, Any] | None = None
    subtask_trace_path: str | None = None
    subtask_recording_requested: bool = False
    subtask_recording_available: bool = False
    subtask_label_semantics: str | None = None
    schema_version: int = SAFE_SCHEMA_VERSION
    feature_schema_version: int = SAFE_FEATURE_SCHEMA_VERSION
    tensor_path: str | None = None
    policy_config: dict[str, Any] = field(default_factory=dict)
    rollout_horizon: int | None = None
    action_path: str | None = None
    safe_repository_commit: str | None = None
    openpi_repository_commit: str | None = None
    rldx_repository_commit: str | None = None
    robocasa_commit: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    action_recording_requested: bool = False
    video_recording_requested: bool = False
    safe_features_recorded: bool = True
    collection_complete: bool = True

    def validate(self) -> None:
        if self.schema_version != SAFE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported SAFE schema version {self.schema_version}")
        if self.feature_schema_version not in {1, 2}:
            raise ValueError(
                f"Unsupported SAFE feature schema version {self.feature_schema_version}"
            )
        if not self.rollout_id or not self.task_name or not self.policy_id or not self.checkpoint:
            raise ValueError("rollout_id, task_name, policy_id, and checkpoint are required")
        if self.num_env_steps < 0 or self.valid_sequence_length < 1:
            raise ValueError("rollout lengths must be positive")
        if len(self.inference_env_steps) != self.valid_sequence_length:
            raise ValueError("inference_env_steps length must equal valid_sequence_length")
        if self.inference_env_steps != sorted(set(self.inference_env_steps)):
            raise ValueError("inference_env_steps must be strictly increasing")
        if any(step < 0 or step >= self.num_env_steps for step in self.inference_env_steps):
            raise ValueError("inference environment step is outside the rollout")
        if self.action_horizon < 1 or self.replan_steps < 1 or self.flow_steps < 1:
            raise ValueError("action_horizon, replan_steps, and flow_steps must be positive")
        if len(self.feature_shape) != 4:
            raise ValueError("feature_shape must be [inferences, flow_steps, horizon, dim]")
        if self.feature_shape[0] != self.valid_sequence_length:
            raise ValueError("feature_shape inference axis disagrees with valid_sequence_length")
        if self.feature_shape[1] != self.flow_steps:
            raise ValueError("feature_shape flow axis disagrees with flow_steps")
        if self.feature_shape[2] != self.action_horizon:
            raise ValueError("feature_shape horizon axis disagrees with action_horizon")
        if self.feature_dtype != "float32":
            raise ValueError("raw SAFE storage dtype must be float32")
        if self.model_family not in {"pi0", "rldx1"}:
            raise ValueError(f"Unsupported SAFE model_family {self.model_family!r}")
        expected_layer = {
            "pi0": OFFICIAL_PI0_FEATURE_LAYER,
            "rldx1": RLDX1_FEATURE_LAYER,
        }[self.model_family]
        if self.feature_layer != expected_layer:
            raise ValueError(
                f"Feature layer {self.feature_layer!r} is incompatible with "
                f"model_family {self.model_family!r}"
            )
        if self.feature_mode not in {"action", "action_observation_context"}:
            raise ValueError(f"Unsupported SAFE feature_mode {self.feature_mode!r}")
        if (
            self.feature_mode == "action_observation_context"
            and self.model_family != "rldx1"
        ):
            raise ValueError("observation context capture is available only for rldx1")
        expected_feature_schema = (
            2 if self.feature_mode == "action_observation_context" else 1
        )
        if self.feature_schema_version != expected_feature_schema:
            raise ValueError(
                "feature_schema_version disagrees with the captured feature mode"
            )
        if self.feature_mode == "action":
            if (
                self.observation_feature_layer is not None
                or self.observation_context_shape
                or self.observation_components
                or self.observation_context_pooling
            ):
                raise ValueError(
                    "action-only SAFE records cannot declare observation context"
                )
        else:
            if self.observation_feature_layer != RLDX1_OBSERVATION_FEATURE_LAYER:
                raise ValueError("unexpected RLDX observation feature layer")
            if (
                len(self.observation_context_shape) != 2
                or self.observation_context_shape[0] != self.valid_sequence_length
                or self.observation_context_shape[1] < 1
            ):
                raise ValueError(
                    "observation_context_shape must be [inferences, feature_dim]"
                )
            validate_feature_components(
                self.observation_components,
                feature_dim=self.observation_context_shape[-1],
                required_names={"backbone_context", "state_context"},
                field_name="observation_components",
            )
            if set(self.observation_context_pooling) != {"backbone", "state"}:
                raise ValueError(
                    "observation_context_pooling must describe backbone and state"
                )
        if self.seed_protocol not in {
            "rollout_index",
            "official_openpi",
            "official_rldx",
        }:
            raise ValueError(f"Unsupported seed_protocol {self.seed_protocol!r}")
        if self.seed_protocol in {"official_openpi", "official_rldx"}:
            if self.environment_reset_index is None or self.environment_reset_index < 0:
                raise ValueError(
                    "official policy seed protocols require a non-negative "
                    "environment_reset_index"
                )
        elif self.environment_reset_index is not None:
            raise ValueError(
                "environment_reset_index is only valid for official policy records"
            )
        if self.video_frame_stride < 1:
            raise ValueError("video_frame_stride must be positive")
        if self.subtask_recording_requested:
            if not self.subtask_recording_available or not self.subtask_trace_path:
                raise ValueError(
                    "requested Subtask-SAFE recording is missing its trace artifact"
                )
            if not self.subtask_label_semantics:
                raise ValueError(
                    "Subtask-SAFE recording requires explicit label semantics"
                )
        elif (
            self.subtask_recording_available
            or self.subtask_trace_path is not None
            or self.subtask_label_semantics is not None
        ):
            raise ValueError(
                "Subtask-SAFE fields are present when recording was not requested"
            )
        if self.rollout_horizon is None:
            self.rollout_horizon = self.timeout_horizon
        if self.rollout_horizon < self.num_env_steps:
            raise ValueError("rollout_horizon cannot be shorter than num_env_steps")
        if not self.collection_complete:
            raise ValueError("incomplete rollouts cannot enter the valid manifest")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data.update(
            {
                "success": not self.failed,
                "failure_label": int(self.failed),
                "num_environment_steps": self.num_env_steps,
                "num_policy_inferences": self.valid_sequence_length,
                "inference_environment_steps": list(self.inference_env_steps),
                "safe_feature_path": self.tensor_path,
                "safe_feature_shape": list(self.feature_shape),
                "safe_feature_dtype": self.feature_dtype,
                "safe_feature_layer": self.feature_layer,
                "safe_feature_aggregation": self.feature_aggregation,
                "policy_name": self.policy_id,
                "policy_checkpoint": self.checkpoint,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SafeRolloutMetadata":
        data = dict(data)
        aliases = {
            "num_environment_steps",
            "num_policy_inferences",
            "inference_environment_steps",
            "safe_feature_path",
            "safe_feature_shape",
            "safe_feature_dtype",
            "safe_feature_layer",
            "safe_feature_aggregation",
            "policy_name",
            "policy_checkpoint",
        }
        success = data.pop("success", None)
        failure_label = data.pop("failure_label", None)
        for key in aliases:
            data.pop(key, None)
        result = cls(**data)
        if success is not None and bool(success) != (not result.failed):
            raise ValueError("success disagrees with failed/failure_label")
        if failure_label is not None and int(failure_label) != int(result.failed):
            raise ValueError("failure_label must be 0 for success and 1 for failure")
        result.validate()
        return result


def validate_feature_tensor(features: np.ndarray, metadata: SafeRolloutMetadata) -> None:
    features = np.asarray(features)
    if features.dtype != np.float32:
        raise ValueError(f"SAFE features must be float32, got {features.dtype}")
    if features.ndim != 4:
        raise ValueError(
            "Rollout SAFE tensor must have shape "
            "(inferences, flow_steps, action_horizon, feature_dim)"
        )
    if list(features.shape) != metadata.feature_shape:
        raise ValueError(
            f"Tensor shape {list(features.shape)} disagrees with metadata "
            f"{metadata.feature_shape}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("SAFE feature tensor contains NaN or infinite values")


def validate_feature_components(
    components,
    *,
    feature_dim,
    required_names=None,
    field_name="feature_components",
) -> None:
    """Validate named, non-overlapping slices over the final feature axis."""
    if components is None:
        components = {}
    if not isinstance(components, dict):
        raise ValueError(f"{field_name} must be a dictionary")
    if required_names:
        missing = set(required_names) - set(components)
        if missing:
            raise ValueError(
                f"{field_name} is missing components: "
                + ", ".join(sorted(missing))
            )
    previous_end = 0
    for name, bounds in sorted(
        components.items(),
        key=lambda item: item[1][0] if isinstance(item[1], (list, tuple)) and item[1] else -1,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("feature component names must be nonempty strings")
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(not isinstance(value, int) for value in bounds)
        ):
            raise ValueError(
                f"feature component {name!r} must be an integer [start, end] slice"
            )
        start, end = bounds
        if start != previous_end or end <= start or end > int(feature_dim):
            raise ValueError(
                f"feature component {name!r} has invalid or non-contiguous "
                f"slice {list(bounds)} for feature_dim={feature_dim}"
            )
        previous_end = end
    if components and previous_end != int(feature_dim):
        raise ValueError("feature component slices do not cover the feature axis")


def compatibility_key(metadata: SafeRolloutMetadata) -> str:
    """Hash all fields that must match before rollouts can be mixed."""
    identity = {
        "policy_id": metadata.policy_id,
        "checkpoint": metadata.checkpoint,
        "model_family": metadata.model_family,
        "feature_schema_version": metadata.feature_schema_version,
        "feature_layer": metadata.feature_layer,
        "feature_aggregation": metadata.feature_aggregation,
        "feature_tail_shape": metadata.feature_shape[1:],
        "feature_dtype": metadata.feature_dtype,
        "feature_mode": metadata.feature_mode,
        "observation_feature_layer": metadata.observation_feature_layer,
        "observation_context_tail_shape": metadata.observation_context_shape[1:],
        "observation_components": metadata.observation_components,
        "observation_context_pooling": metadata.observation_context_pooling,
        "action_horizon": metadata.action_horizon,
        "flow_steps": metadata.flow_steps,
        "environment_split": metadata.environment_split,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
