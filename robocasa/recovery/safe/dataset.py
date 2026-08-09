"""Compressed tensor storage, manifests, aggregation, and split generation."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from .schema import (
    RLDX1_FUSED_EXPORT_LAYER,
    SafeRolloutMetadata,
    compatibility_key,
    validate_feature_tensor,
)


MANIFEST_NAME = "manifest.jsonl"
SPLIT_MANIFEST_VERSION = 1
FEATURE_VIEWS = ("all", "action", "observation_context")


def select_feature_view(
    features,
    metadata,
    feature_view="all",
    *,
    observation_context=None,
):
    """Build a paired export view from action latents and optional context."""
    features = np.asarray(features, dtype=np.float32)
    feature_view = str(feature_view)
    if feature_view not in FEATURE_VIEWS:
        raise ValueError(
            f"Unknown feature view {feature_view!r}; choose from {FEATURE_VIEWS}"
        )
    if feature_view == "action":
        return features, metadata.feature_layer

    if metadata.feature_mode == "action":
        if feature_view == "all":
            return features, metadata.feature_layer
        raise ValueError(
            "Observation-context view is unavailable in an action-only dataset"
        )

    if observation_context is None:
        raise ValueError(f"Feature view {feature_view!r} requires observation_context")
    observation_context = np.asarray(observation_context, dtype=np.float32)
    expected_shape = (features.shape[0], metadata.observation_context_shape[-1])
    if observation_context.shape != expected_shape:
        raise ValueError(
            "Observation context shape disagrees with the action sequence: "
            f"{observation_context.shape} != {expected_shape}"
        )
    if not np.all(np.isfinite(observation_context)):
        raise ValueError("Observation context contains NaN or infinite values")
    observation_view = observation_context[:, None, None, :]
    if feature_view == "observation_context":
        return observation_view, metadata.observation_feature_layer

    action_mean = features[:, -1].mean(axis=1, dtype=np.float32)
    fused = np.concatenate((action_mean, observation_context), axis=-1)
    return fused[:, None, None, :].astype(np.float32, copy=False), (
        RLDX1_FUSED_EXPORT_LAYER
    )


def save_rollout(
    output_dir: str | Path,
    metadata: SafeRolloutMetadata,
    features: np.ndarray,
    policy_action_chunks: np.ndarray | None = None,
    observation_context: np.ndarray | None = None,
) -> Path:
    output_dir = Path(output_dir)
    tensor_dir = output_dir / "rollouts"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    metadata.feature_shape = list(features.shape)
    metadata.valid_sequence_length = int(features.shape[0])
    metadata.flow_steps = int(features.shape[1])
    metadata.action_horizon = int(features.shape[2])
    if observation_context is not None:
        observation_context = np.asarray(observation_context, dtype=np.float32)
        metadata.observation_context_shape = list(observation_context.shape)
    tensor_path = tensor_dir / f"{metadata.rollout_id}.npz"
    if tensor_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing SAFE rollout {metadata.rollout_id}"
        )
    metadata.tensor_path = str(tensor_path.relative_to(output_dir))
    validate_feature_tensor(features, metadata)
    if metadata.feature_mode == "action_observation_context":
        expected_context_shape = (
            features.shape[0],
            metadata.observation_context_shape[-1],
        )
        if (
            observation_context is None
            or observation_context.shape != expected_context_shape
        ):
            raise ValueError(
                "observation_context must be (inferences, feature_dim) for dual-stream SAFE"
            )
        if not np.all(np.isfinite(observation_context)):
            raise ValueError("observation_context contains NaN or infinite values")
    elif observation_context is not None:
        raise ValueError("action-only SAFE records cannot store observation_context")
    if policy_action_chunks is not None:
        policy_action_chunks = np.asarray(policy_action_chunks, dtype=np.float32)
        if policy_action_chunks.ndim != 3:
            raise ValueError("policy_action_chunks must be (inferences, horizon, action_dim)")
        if policy_action_chunks.shape[:2] != (features.shape[0], features.shape[2]):
            raise ValueError("policy action chunks disagree with SAFE inference/horizon axes")
    feature_metadata = {
        "schema_version": metadata.feature_schema_version,
        "model_family": metadata.model_family,
        "feature_layer": metadata.feature_layer,
        "feature_shape": metadata.feature_shape,
        "feature_dtype": metadata.feature_dtype,
        "feature_aggregation": metadata.feature_aggregation,
        "feature_mode": metadata.feature_mode,
        "observation_feature_layer": metadata.observation_feature_layer,
        "observation_context_shape": metadata.observation_context_shape,
        "observation_components": metadata.observation_components,
        "observation_context_pooling": metadata.observation_context_pooling,
        "policy_name": metadata.policy_id,
        "policy_checkpoint": metadata.checkpoint,
        "action_horizon": metadata.action_horizon,
        "flow_steps": metadata.flow_steps,
    }
    temp_path = tensor_path.with_suffix(".tmp.npz")
    payload = {
        "features": features,
        "safe_features": features,
        "inference_environment_steps": np.asarray(metadata.inference_env_steps, dtype=np.int64),
        "inference_env_steps": np.asarray(metadata.inference_env_steps, dtype=np.int64),
        "valid_length": np.asarray(metadata.valid_sequence_length, dtype=np.int64),
        "rollout_id": np.asarray(metadata.rollout_id),
        "feature_metadata_json": np.asarray(json.dumps(feature_metadata, sort_keys=True)),
        "failed": np.asarray(int(metadata.failed), dtype=np.int8),
        "schema_version": np.asarray(metadata.schema_version, dtype=np.int64),
    }
    if policy_action_chunks is not None:
        payload["policy_action_chunks"] = policy_action_chunks
    if observation_context is not None:
        payload["observation_context"] = observation_context
    np.savez_compressed(temp_path, **payload)
    os.replace(temp_path, tensor_path)
    manifest_path = output_dir / MANIFEST_NAME
    line = (json.dumps(metadata.to_dict(), sort_keys=True) + "\n").encode()
    descriptor = os.open(manifest_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return tensor_path


def load_manifest(path: str | Path) -> list[SafeRolloutMetadata]:
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(SafeRolloutMetadata.from_dict(json.loads(line)))
                except Exception as error:
                    raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
    ids = [record.rollout_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate rollout IDs in manifest")
    return records


def load_rollout(dataset_dir: str | Path, metadata: SafeRolloutMetadata):
    dataset_dir = Path(dataset_dir)
    if metadata.tensor_path is None:
        raise ValueError("Rollout metadata has no tensor_path")
    with np.load(dataset_dir / metadata.tensor_path, allow_pickle=False) as payload:
        features = payload["features"] if "features" in payload else payload["safe_features"]
        steps = (
            payload["inference_environment_steps"]
            if "inference_environment_steps" in payload
            else payload["inference_env_steps"]
        )
        failed = bool(payload["failed"])
        if "valid_length" in payload and int(payload["valid_length"]) != metadata.valid_sequence_length:
            raise ValueError("Feature valid_length disagrees with manifest")
        if "rollout_id" in payload and str(payload["rollout_id"]) != metadata.rollout_id:
            raise ValueError("Feature rollout_id disagrees with manifest")
    validate_feature_tensor(features, metadata)
    if steps.tolist() != metadata.inference_env_steps or failed != metadata.failed:
        raise ValueError("Tensor payload disagrees with manifest metadata")
    return features, metadata


def aggregate_features(features: np.ndarray, aggregation: str = "mean") -> np.ndarray:
    """Reduce official raw `(T, flow, horizon, D)` features to `(T, D*)`."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 4:
        raise ValueError("Expected (T, flow_steps, action_horizon, feature_dim)")
    if aggregation == "mean":
        return features.mean(axis=(1, 2), dtype=np.float32)
    if aggregation == "first":
        return features[:, 0, 0]
    if aggregation == "last":
        return features[:, -1, -1]
    if aggregation == "flow_mean_horizon_last":
        return features[:, :, -1].mean(axis=1, dtype=np.float32)
    if aggregation == "flow_last_horizon_mean":
        return features[:, -1].mean(axis=1, dtype=np.float32)
    if aggregation.startswith("concat-"):
        count = int(aggregation.split("-", 1)[1])
        if count < 1:
            raise ValueError("concat count must be positive")
        flow_idx = np.linspace(0, features.shape[1] - 1, count).round().astype(int)
        horizon_idx = np.linspace(0, features.shape[2] - 1, count).round().astype(int)
        selected = features[:, flow_idx][:, :, horizon_idx]
        return selected.reshape(features.shape[0], -1)
    raise ValueError(f"Unknown SAFE feature aggregation {aggregation!r}")


def assert_compatible(records: Iterable[SafeRolloutMetadata]) -> str:
    keys = {compatibility_key(record) for record in records}
    if len(keys) != 1:
        raise ValueError(f"Incompatible policy/feature configurations: {sorted(keys)}")
    return next(iter(keys))


def generate_splits(
    records: list[SafeRolloutMetadata],
    unseen_tasks: Iterable[str],
    *,
    seed: int = 0,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict:
    """Split by episode identity, including reset index for official OpenPI runs."""
    if not records:
        raise ValueError("Cannot split an empty dataset")
    if train_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("train and calibration fractions must be positive")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train plus calibration fraction must be below one")
    assert_compatible(records)
    unseen_tasks = set(unseen_tasks)
    known_tasks = {record.task_name for record in records}
    missing = unseen_tasks - known_tasks
    if missing:
        raise ValueError(f"Unseen tasks are absent from dataset: {sorted(missing)}")
    grouped = defaultdict(list)
    for record in records:
        episode_identity = (
            record.environment_reset_index
            if record.seed_protocol in {"official_openpi", "official_rldx"}
            else None
        )
        grouped[
            (record.task_name, record.environment_seed, episode_identity)
        ].append(record)
    rng = random.Random(seed)
    result = {name: [] for name in ("train", "calibration", "seen_test", "unseen_test")}
    seen_groups = defaultdict(list)
    for key, group in grouped.items():
        if key[0] in unseen_tasks:
            result["unseen_test"].extend(record.rollout_id for record in group)
        else:
            seen_groups[key[0]].append((key, group))
    for task, task_groups in sorted(seen_groups.items()):
        rng.shuffle(task_groups)
        n = len(task_groups)
        n_train = max(1, int(n * train_fraction))
        n_cal = max(1, int(n * calibration_fraction)) if n >= 3 else 0
        if n_train + n_cal >= n:
            n_train = max(1, n - n_cal - 1)
        for index, (_, group) in enumerate(task_groups):
            split = "train" if index < n_train else "calibration" if index < n_train + n_cal else "seen_test"
            result[split].extend(record.rollout_id for record in group)
    for values in result.values():
        values.sort()
    all_ids = [rollout_id for values in result.values() for rollout_id in values]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != {r.rollout_id for r in records}:
        raise AssertionError("Split generation lost or duplicated rollouts")
    # Calibration is success-only by construction of its usable conformal subset;
    # failed calibration rollouts remain listed for auditing but are never thresholds.
    success_ids = {r.rollout_id for r in records if not r.failed}
    result["calibration_successes"] = sorted(set(result["calibration"]) & success_ids)
    train_labels = {record.failed for record in records if record.rollout_id in result["train"]}
    if train_labels != {False, True}:
        raise ValueError("Training split must contain both successful and failed rollouts")
    if not result["calibration_successes"]:
        raise ValueError("Calibration split contains no held-out successful rollouts")
    result["schema_version"] = SPLIT_MANIFEST_VERSION
    result["seed"] = seed
    result["unseen_tasks"] = sorted(unseen_tasks)
    result["compatibility_key"] = assert_compatible(records)
    return result


def save_splits(splits: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n")
    return path


def pad_sequences(sequences: list[np.ndarray]):
    if not sequences:
        raise ValueError("No sequences to pad")
    dim = sequences[0].shape[-1]
    if any(seq.ndim != 2 or seq.shape[-1] != dim for seq in sequences):
        raise ValueError("All sequences must be (T, D) with a shared D")
    max_len = max(len(seq) for seq in sequences)
    batch = np.zeros((len(sequences), max_len, dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=bool)
    for index, seq in enumerate(sequences):
        batch[index, : len(seq)] = seq
        mask[index, : len(seq)] = True
    return batch, mask


def build_parser():
    parser = argparse.ArgumentParser(description="Create leakage-safe raw SAFE splits")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--unseen-task", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    records = load_manifest(args.dataset_dir)
    splits = generate_splits(
        records,
        args.unseen_task,
        seed=args.seed,
        train_fraction=args.train_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    save_splits(splits, args.output)


if __name__ == "__main__":
    main()
