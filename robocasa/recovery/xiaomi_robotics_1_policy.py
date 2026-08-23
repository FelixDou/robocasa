"""Xiaomi-Robotics-1 adapter with SAFE capture and recovery reranking."""

from __future__ import annotations

import collections
import copy
import pickle
import socket
import struct

import numpy as np

from robocasa.recovery.openpi_websocket_policy import _convert_action


XIAOMI_SAFE_FEATURE_LAYER = "dit_action_tokens_pre_action_output_layer"
DEFAULT_ROBOT_TYPE = "robocasa365"
DEFAULT_STATE_DIM = 60
DEFAULT_ACTION_DIM = 12
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


def _to_numpy(value, *, dtype=None):
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    if dtype == np.float32:
        to_float = getattr(value, "float", None)
        if to_float is not None:
            value = to_float()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def quat_xyzw_to_axis_angle(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape == (3,):
        return quaternion.astype(np.float32)
    if quaternion.shape != (4,):
        raise ValueError(
            "Expected rotation as a 3D axis angle or 4D xyzw quaternion, got "
            f"{quaternion.shape}"
        )
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def observation_to_state(observation):
    state = np.concatenate(
        [
            np.asarray(
                observation["state.end_effector_position_relative"],
                dtype=np.float32,
            ).reshape(-1),
            quat_xyzw_to_axis_angle(
                observation["state.end_effector_rotation_relative"]
            ),
            np.asarray(observation["state.gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(observation["state.base_position"], dtype=np.float32).reshape(
                -1
            ),
            quat_xyzw_to_axis_angle(observation["state.base_rotation"]),
        ]
    ).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected a 14D RoboCasa365 state, got {state.shape}")
    return state


def sample_history(history, length, interval):
    items = list(history)
    if not items:
        raise ValueError("Observation history cannot be empty")
    indices = [
        max(0, len(items) - 1 - (length - 1 - index) * interval)
        for index in range(length)
    ]
    return np.ascontiguousarray(np.stack([items[index] for index in indices], axis=0))


class XiaomiSocketClient:
    """Length-prefixed pickle transport used by the released XR-1 server."""

    def __init__(self, host="127.0.0.1", port=10086, timeout_seconds=600):
        self.host = host
        self.port = int(port)
        self.socket = socket.create_connection(
            (self.host, self.port), timeout=float(timeout_seconds)
        )
        self.socket.settimeout(float(timeout_seconds))

    @staticmethod
    def _recv_all(connection, length):
        chunks = []
        remaining = length
        while remaining:
            packet = connection.recv(remaining)
            if not packet:
                raise ConnectionError("XR-1 server closed the connection")
            chunks.append(packet)
            remaining -= len(packet)
        return b"".join(chunks)

    def infer(self, request):
        serialized = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.socket.sendall(struct.pack(">I", len(serialized)) + serialized)
        length = struct.unpack(">I", self._recv_all(self.socket, 4))[0]
        return pickle.loads(self._recv_all(self.socket, length))

    def close(self):
        self.socket.close()


class XiaomiRobotics1Policy:
    """Callable XR-1 policy preserving the official RoboCasa365 preprocessing."""

    def __init__(
        self,
        *,
        model_path,
        host="127.0.0.1",
        port=10086,
        timeout_seconds=600,
        robot_type=DEFAULT_ROBOT_TYPE,
        crop_ratio=0.95,
        observation_history=4,
        observation_interval=2,
        replan_steps=16,
        state_dim=DEFAULT_STATE_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        collect_safe_features=False,
        safe_feature_shape=None,
        safe_best_of_k=1,
        safe_candidate_strategy="lowest_safe",
        safe_candidate_seed=0,
        sampling_seed_base=None,
        safe_runtime_bundle=None,
        safe_repo=None,
        safe_device="cpu",
        policy_name=None,
        policy_checkpoint=None,
        client=None,
        processor=None,
        candidate_scorer=None,
    ):
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=True,
            )
        self.processor = processor
        self.client = client or XiaomiSocketClient(
            host=host,
            port=port,
            timeout_seconds=timeout_seconds,
        )
        self.model_path = str(model_path)
        self.robot_type = str(robot_type)
        self.crop_ratio = float(crop_ratio)
        self.observation_history = int(observation_history)
        self.observation_interval = int(observation_interval)
        self.replan_steps = int(replan_steps)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.collect_safe_features = bool(collect_safe_features)
        self.safe_feature_shape = (
            tuple(int(value) for value in safe_feature_shape)
            if safe_feature_shape is not None
            else None
        )
        self.safe_best_of_k = int(safe_best_of_k)
        self.safe_candidate_strategy = str(safe_candidate_strategy)
        self.safe_candidate_seed = int(safe_candidate_seed)
        self.sampling_seed_base = (
            None if sampling_seed_base is None else int(sampling_seed_base)
        )
        if self.safe_best_of_k < 1:
            raise ValueError("safe_best_of_k must be positive")
        if self.safe_candidate_strategy not in {
            "lowest_safe",
            "highest_safe",
            "random",
        }:
            raise ValueError(
                "safe_candidate_strategy must be lowest_safe, highest_safe, or random"
            )
        if self.safe_best_of_k > 1 and candidate_scorer is None:
            if safe_runtime_bundle is None or safe_repo is None:
                raise ValueError(
                    "safe_best_of_k > 1 requires safe_runtime_bundle and safe_repo"
                )
            from robocasa.recovery.safe.xr1_best_of_k import FrozenXr1SafeEnsemble

            candidate_scorer = FrozenXr1SafeEnsemble(
                runtime_bundle=safe_runtime_bundle,
                safe_repo=safe_repo,
                device=safe_device,
            )
        self.candidate_scorer = candidate_scorer
        self.policy_name = policy_name
        self.policy_checkpoint = policy_checkpoint
        if not 0 < self.crop_ratio <= 1:
            raise ValueError("crop_ratio must be in (0, 1]")
        if self.observation_history < 1 or self.observation_interval < 1:
            raise ValueError("observation history and interval must be positive")
        if self.replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        list_robot_types = getattr(self.processor, "list_robot_types", None)
        if list_robot_types is not None:
            robot_types = list_robot_types()
            if self.robot_type not in robot_types:
                raise ValueError(
                    f"Robot type {self.robot_type!r} is unavailable: {robot_types}"
                )
        self.queue_length = (
            self.observation_history - 1
        ) * self.observation_interval + 1
        self.reset()

    @staticmethod
    def _center_crop(image, crop_ratio):
        from PIL import Image

        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        if crop_ratio >= 1.0:
            return pil_image
        width, height = pil_image.size
        crop_width = max(1, int(width * crop_ratio))
        crop_height = max(1, int(height * crop_ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return cropped.resize((width, height), resampling)

    def _build_messages(self, image_history, instruction):
        videos = {
            key: [self._center_crop(frame, self.crop_ratio) for frame in frames]
            for key, frames in image_history.items()
        }
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                    {"type": "text", "text": "\nRight camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                    {"type": "text", "text": "\nWrist camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                    {
                        "type": "text",
                        "text": (
                            "\n\nGenerate robot actions for the task:\n"
                            f"{instruction} /no_cot"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

    def _append_observation(self, observation):
        for key in CAMERA_KEYS:
            self.image_queues[key].append(
                np.ascontiguousarray(observation[key], dtype=np.uint8)
            )
        self.state_queue.append(observation_to_state(observation))

    def _make_request(
        self, instruction, *, request_safe_features=None, sampling_seed=None
    ):
        state_history = sample_history(
            self.state_queue,
            self.observation_history,
            self.observation_interval,
        )
        state = np.zeros((1, state_history.shape[0], self.state_dim), dtype=np.float32)
        state[0, :, : state_history.shape[-1]] = state_history
        # This small causal state-history channel is stored only when SAFE
        # features are requested.  It is a deployable observation/state
        # ablation; it does not depend on future observations or simulator
        # privileged state.
        self._latest_observation_state_history = np.ascontiguousarray(
            state[0].reshape(-1), dtype=np.float32
        )
        image_history = {
            key: sample_history(
                queue,
                self.observation_history,
                self.observation_interval,
            )
            for key, queue in self.image_queues.items()
        }
        inputs = self.processor.apply_chat_template(
            self._build_messages(image_history, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=state,
            robot_type=self.robot_type,
        )
        request = dict(inputs)
        request["task_id"] = self.robot_type
        if request_safe_features is None:
            request_safe_features = self.collect_safe_features
        if request_safe_features:
            request["request_safe_features"] = True
        if sampling_seed is not None:
            request["sampling_seed"] = int(sampling_seed)
        return request

    def _decode_actions(self, raw_actions):
        actions = self.processor.decode_action(raw_actions, robot_type=self.robot_type)
        actions = _to_numpy(actions, dtype=np.float32)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise RuntimeError(
                    "SAFE XR-1 collection requires one environment per client"
                )
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] < self.action_dim:
            raise RuntimeError(
                "Decoded XR-1 actions must have shape (horizon, action_dim), got "
                f"{actions.shape}"
            )
        actions = np.ascontiguousarray(actions[:, : self.action_dim])
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("XR-1 actions contain NaN or infinite values")
        return actions

    def _parse_safe_features(self, response, actions):
        if not isinstance(response, dict) or "safe_features" not in response:
            raise RuntimeError(
                "SAFE feature collection was requested, but the XR-1 server did "
                "not return safe_features. Apply both companion Xiaomi patches."
            )
        features = _to_numpy(response["safe_features"])
        if features.ndim == 4:
            if features.shape[0] != 1:
                raise RuntimeError(
                    "SAFE XR-1 collection requires one environment per client"
                )
            features = features[0]
        features = np.asarray(features)
        metadata = response.get("safe_feature_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("XR-1 SAFE response is missing feature metadata")
        metadata = copy.deepcopy(metadata)
        if features.ndim != 3:
            raise RuntimeError(
                "XR-1 SAFE features must have shape "
                "(flow_steps, action_horizon, feature_dim), got "
                f"{features.shape}"
            )
        if features.dtype != np.float32:
            raise RuntimeError(
                f"XR-1 SAFE features must be float32, got {features.dtype}"
            )
        if (
            self.safe_feature_shape is not None
            and features.shape != self.safe_feature_shape
        ):
            raise RuntimeError(
                f"XR-1 SAFE feature shape {features.shape} does not match "
                f"configured shape {self.safe_feature_shape}"
            )
        if not np.all(np.isfinite(features)) or not np.any(features):
            raise RuntimeError("XR-1 SAFE features must be finite and nonzero")
        if actions.shape[0] != features.shape[1]:
            raise RuntimeError(
                "XR-1 SAFE action-horizon mismatch: "
                f"{actions.shape[0]} != {features.shape[1]}"
            )
        metadata.setdefault("schema_version", 1)
        metadata.setdefault("model_family", "xiaomi_robotics_1")
        metadata.setdefault("feature_aggregation", metadata.get("aggregation", "raw"))
        metadata.setdefault("aggregation", metadata["feature_aggregation"])
        metadata.setdefault("policy_name", self.policy_name or metadata.get("model_id"))
        metadata.setdefault(
            "policy_checkpoint",
            self.policy_checkpoint or metadata.get("checkpoint"),
        )
        metadata.setdefault("feature_shape", list(features.shape))
        metadata.setdefault("feature_dtype", str(features.dtype))
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
            raise RuntimeError(f"XR-1 SAFE metadata is missing fields: {missing}")
        checks = {
            "schema_version": int(metadata["schema_version"]) == 1,
            "model_family": metadata["model_family"] == "xiaomi_robotics_1",
            "feature_layer": metadata["feature_layer"] == XIAOMI_SAFE_FEATURE_LAYER,
            "feature_shape": tuple(metadata["feature_shape"]) == features.shape,
            "feature_dtype": metadata["feature_dtype"] == str(features.dtype),
            "feature_aggregation": metadata["feature_aggregation"] == "raw",
            "action_horizon": int(metadata["action_horizon"]) == features.shape[1],
            "flow_steps": int(metadata["flow_steps"]) == features.shape[0],
        }
        failed_checks = [name for name, valid in checks.items() if not valid]
        if failed_checks:
            raise RuntimeError(
                "XR-1 SAFE metadata disagrees with payload for: "
                + ", ".join(failed_checks)
            )
        return np.ascontiguousarray(features), metadata

    def _record_safe_features(self, response, actions):
        features, metadata = self._parse_safe_features(response, actions)
        record = {
            "env_step": self._env_step,
            "environment_step": self._env_step,
            "inference_index": self._inference_index,
            "features": np.ascontiguousarray(features),
            "actions": np.ascontiguousarray(actions),
            "metadata": metadata,
            "auxiliary_features": {
                "observation_state_history": np.ascontiguousarray(
                    self._latest_observation_state_history, dtype=np.float32
                )
            },
        }
        self._inference_index += 1
        self._pending_inference_record = record
        self._latest_inference_record = record

    @staticmethod
    def _action_diversity(candidate_actions, replan_steps):
        flattened = [
            np.asarray(actions[:replan_steps], dtype=np.float64).reshape(-1)
            for actions in candidate_actions
        ]
        distances = [
            float(np.linalg.norm(flattened[left] - flattened[right]))
            for left in range(len(flattened))
            for right in range(left + 1, len(flattened))
        ]
        return {
            "pairwise_l2_min": min(distances) if distances else 0.0,
            "pairwise_l2_mean": float(np.mean(distances)) if distances else 0.0,
            "all_identical": bool(not distances or max(distances) == 0.0),
        }

    def _select_candidate(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (self.safe_best_of_k,) or not np.all(np.isfinite(scores)):
            raise RuntimeError(
                "SAFE scorer must return one finite score per candidate, got "
                f"{scores.shape}"
            )
        if self.safe_candidate_strategy == "lowest_safe":
            return int(np.argmin(scores))
        if self.safe_candidate_strategy == "highest_safe":
            return int(np.argmax(scores))
        rng = np.random.default_rng(
            self.safe_candidate_seed + self._candidate_selection_index
        )
        return int(rng.integers(self.safe_best_of_k))

    def _infer_best_of_k(self, instruction):
        candidate_actions = []
        candidate_features = []
        candidate_metadata = []
        sampling_seeds = []
        for candidate_index in range(self.safe_best_of_k):
            sampling_seed = (
                self.safe_candidate_seed
                + self._candidate_selection_index * self.safe_best_of_k
                + candidate_index
            )
            response = self.client.infer(
                self._make_request(
                    instruction,
                    request_safe_features=True,
                    sampling_seed=sampling_seed,
                )
            )
            raw_actions = (
                response.get("actions") if isinstance(response, dict) else response
            )
            if raw_actions is None:
                raise RuntimeError("XR-1 response is missing actions")
            actions = self._decode_actions(raw_actions)
            if len(actions) < self.replan_steps:
                raise RuntimeError(
                    f"XR-1 returned {len(actions)} actions, but replan_steps is "
                    f"{self.replan_steps}"
                )
            features, metadata = self._parse_safe_features(response, actions)
            candidate_actions.append(actions)
            candidate_features.append(features)
            candidate_metadata.append(metadata)
            sampling_seeds.append(sampling_seed)

        score_result = self.candidate_scorer.score_candidates(
            np.stack(candidate_features), task_name=self._recovery_task_name
        )
        scores = score_result.get("scores")
        selected_index = self._select_candidate(scores)
        selected_actions = candidate_actions[selected_index]
        selected_response = {
            "safe_features": candidate_features[selected_index],
            "safe_feature_metadata": candidate_metadata[selected_index],
        }
        self._record_safe_features(selected_response, selected_actions)
        score_array = np.asarray(scores, dtype=np.float64)
        record = {
            "schema_version": 1,
            "environment_step": self._env_step,
            "inference_index": self._inference_index - 1,
            "selection_index": self._candidate_selection_index,
            "task_name": self._recovery_task_name,
            "target_subtask": self._recovery_target_subtask,
            "strategy": self.safe_candidate_strategy,
            "candidate_count": self.safe_best_of_k,
            "sampling_seeds": sampling_seeds,
            "scores": score_array.tolist(),
            "selected_index": selected_index,
            "selected_score": float(score_array[selected_index]),
            "score_spread": float(np.ptp(score_array)),
            "scoring_protocol": score_result.get("protocol"),
            "ensemble_seeds": score_result.get("seeds"),
            "raw_scores_by_seed": score_result.get("raw_scores_by_seed"),
            "pre_sigmoid_logits_by_seed": score_result.get(
                "pre_sigmoid_logits_by_seed"
            ),
            "normalized_scores_by_seed": score_result.get("normalized_scores_by_seed"),
            "probability_normalized_scores_by_seed": score_result.get(
                "probability_normalized_scores_by_seed"
            ),
            "scorer_provenance": score_result.get("provenance"),
            "action_diversity": self._action_diversity(
                candidate_actions, self.replan_steps
            ),
        }
        self._candidate_selection_index += 1
        self._pending_candidate_selection_record = record
        self._latest_candidate_selection_record = record
        return selected_actions

    def _infer_one(self, instruction):
        sampling_seed = None
        if self.sampling_seed_base is not None:
            sampling_seed = self.sampling_seed_base + self._ordinary_inference_index
        response = self.client.infer(
            self._make_request(instruction, sampling_seed=sampling_seed)
        )
        self._ordinary_inference_index += 1
        raw_actions = (
            response.get("actions") if isinstance(response, dict) else response
        )
        if raw_actions is None:
            raise RuntimeError("XR-1 response is missing actions")
        actions = self._decode_actions(raw_actions)
        if len(actions) < self.replan_steps:
            raise RuntimeError(
                f"XR-1 returned {len(actions)} actions, but replan_steps is "
                f"{self.replan_steps}"
            )
        if self.collect_safe_features:
            self._record_safe_features(response, actions)
        return actions

    def __call__(self, observation, instruction=None):
        instruction = instruction or observation["annotation.human.task_description"]
        if self.last_instruction is not None and instruction != self.last_instruction:
            self.reset()
        self.last_instruction = instruction
        self._append_observation(observation)
        if not self.action_plan:
            if self._recovery_reranking_active and self.safe_best_of_k > 1:
                actions = self._infer_best_of_k(instruction)
            else:
                actions = self._infer_one(instruction)
            self.action_plan.extend(actions[: self.replan_steps])
        action = _convert_action(self.action_plan.popleft())
        self._env_step += 1
        return action

    def pop_inference_record(self):
        record = self._pending_inference_record
        self._pending_inference_record = None
        return record

    def pop_candidate_selection_record(self):
        record = self._pending_candidate_selection_record
        self._pending_candidate_selection_record = None
        return record

    def begin_recovery(self, *, task_name, target_subtask=None, instruction=None):
        """Enable best-of-K only for the bounded recovery attempt."""
        if self.safe_best_of_k <= 1:
            return {
                "enabled": False,
                "candidate_count": 1,
                "reason": "safe_best_of_k_is_one",
            }
        if not task_name:
            raise ValueError("SAFE recovery reranking requires task_name")
        self.reset()
        self._recovery_reranking_active = True
        self._recovery_task_name = str(task_name)
        self._recovery_target_subtask = (
            None if target_subtask is None else str(target_subtask)
        )
        self.last_instruction = None
        return {
            "enabled": True,
            "candidate_count": self.safe_best_of_k,
            "strategy": self.safe_candidate_strategy,
            "task_name": self._recovery_task_name,
            "target_subtask": self._recovery_target_subtask,
            "instruction": instruction,
        }

    def end_recovery(self):
        self._recovery_reranking_active = False
        self.action_plan.clear()

    @property
    def latest_inference_record(self):
        return self._latest_inference_record

    @property
    def latest_candidate_selection_record(self):
        return self._latest_candidate_selection_record

    def reset(self):
        self.action_plan = collections.deque()
        self.image_queues = {
            key: collections.deque(maxlen=self.queue_length) for key in CAMERA_KEYS
        }
        self.state_queue = collections.deque(maxlen=self.queue_length)
        self.last_instruction = None
        self._env_step = 0
        self._inference_index = 0
        self._pending_inference_record = None
        self._latest_inference_record = None
        self._recovery_reranking_active = False
        self._recovery_task_name = None
        self._recovery_target_subtask = None
        self._candidate_selection_index = 0
        self._ordinary_inference_index = 0
        self._pending_candidate_selection_record = None
        self._latest_candidate_selection_record = None
        self._latest_observation_state_history = None

    def close(self):
        self.client.close()


def make_policy(
    env=None,
    model_path=None,
    host="127.0.0.1",
    port=10086,
    timeout_seconds=600,
    robot_type=DEFAULT_ROBOT_TYPE,
    crop_ratio=0.95,
    observation_history=4,
    observation_interval=2,
    replan_steps=16,
    collect_safe_features=False,
    safe_feature_shape=None,
    safe_best_of_k=1,
    safe_candidate_strategy="lowest_safe",
    safe_candidate_seed=0,
    sampling_seed_base=None,
    safe_runtime_bundle=None,
    safe_repo=None,
    safe_device="cpu",
    policy_name=None,
    policy_checkpoint=None,
):
    if model_path is None:
        model_path = policy_checkpoint
    if model_path is None:
        raise ValueError("XR-1 policy requires model_path or policy_checkpoint")
    return XiaomiRobotics1Policy(
        model_path=model_path,
        host=host,
        port=port,
        timeout_seconds=timeout_seconds,
        robot_type=robot_type,
        crop_ratio=crop_ratio,
        observation_history=observation_history,
        observation_interval=observation_interval,
        replan_steps=replan_steps,
        collect_safe_features=collect_safe_features,
        safe_feature_shape=safe_feature_shape,
        safe_best_of_k=safe_best_of_k,
        safe_candidate_strategy=safe_candidate_strategy,
        safe_candidate_seed=safe_candidate_seed,
        sampling_seed_base=sampling_seed_base,
        safe_runtime_bundle=safe_runtime_bundle,
        safe_repo=safe_repo,
        safe_device=safe_device,
        policy_name=policy_name,
        policy_checkpoint=policy_checkpoint,
    )
