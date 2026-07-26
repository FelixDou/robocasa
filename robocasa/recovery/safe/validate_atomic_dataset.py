"""Validate a RoboCasa atomic SAFE rollout dataset and official-loader readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .atomic_tasks import registered_atomic_tasks
from .collect_atomic_rollouts import SUMMARY_NAME, atomic_write_json
from .dataset import MANIFEST_NAME, load_manifest
from .schema import SAFE_SCHEMA_VERSION, validate_feature_tensor


def validate_atomic_dataset(dataset_dir, *, allow_unregistered=False):
    dataset_dir = Path(dataset_dir)
    errors = []
    warnings = []
    records = []
    manifest_path = dataset_dir / MANIFEST_NAME
    summary_path = dataset_dir / SUMMARY_NAME
    summary = None
    if not manifest_path.is_file():
        errors.append(f"Missing manifest: {manifest_path}")
    else:
        try:
            records = load_manifest(dataset_dir)
        except Exception as error:
            errors.append(f"Manifest invalid: {error}")
    if not summary_path.is_file():
        errors.append(f"Missing summary: {summary_path}")
    else:
        try:
            summary = json.loads(summary_path.read_text())
            if summary.get("schema_version") != SAFE_SCHEMA_VERSION:
                errors.append("Summary schema_version is unsupported")
            if summary.get("partial"):
                errors.append("Dataset summary marks collection as partial/incomplete")
        except Exception as error:
            errors.append(f"Summary invalid: {error}")
    if not records:
        errors.append("Dataset contains no valid rollouts")

    registered = registered_atomic_tasks() if not allow_unregistered else set()
    seen_episode_identity = set()
    identities = set()
    feature_files = set()
    official_compatible = True
    for record in records:
        prefix = f"rollout {record.rollout_id}"
        if record.schema_version != SAFE_SCHEMA_VERSION:
            errors.append(f"{prefix}: unsupported schema_version")
        episode_identity = (
            record.task_name,
            record.environment_seed,
            record.environment_reset_index,
        )
        if episode_identity in seen_episode_identity:
            errors.append(
                f"{prefix}: duplicate task/seed/reset record {episode_identity}"
            )
        seen_episode_identity.add(episode_identity)
        if not allow_unregistered and record.task_name not in registered:
            errors.append(f"{prefix}: task is not a registered atomic task: {record.task_name}")
        if not record.collection_complete:
            errors.append(f"{prefix}: rollout is marked incomplete")
        if record.tensor_path is None:
            errors.append(f"{prefix}: safe_feature_path is missing")
            official_compatible = False
            continue
        feature_path = dataset_dir / record.tensor_path
        feature_files.add(feature_path.resolve())
        if not feature_path.is_file():
            errors.append(f"{prefix}: feature file does not exist: {record.tensor_path}")
            official_compatible = False
            continue
        try:
            with np.load(feature_path, allow_pickle=False) as payload:
                required = {
                    "features",
                    "inference_environment_steps",
                    "valid_length",
                    "rollout_id",
                    "feature_metadata_json",
                    "policy_action_chunks",
                    "failed",
                    "schema_version",
                }
                missing = sorted(required - set(payload.files))
                if missing:
                    errors.append(f"{prefix}: feature file missing keys {missing}")
                    official_compatible = False
                    continue
                features = payload["features"]
                steps = payload["inference_environment_steps"]
                valid_length = int(payload["valid_length"])
                rollout_id = str(payload["rollout_id"])
                chunks = payload["policy_action_chunks"]
                failed = int(payload["failed"])
                tensor_schema_version = int(payload["schema_version"])
                metadata = json.loads(str(payload["feature_metadata_json"]))
            validate_feature_tensor(features, record)
            if features.shape[0] == 0:
                errors.append(f"{prefix}: empty feature sequence")
            if valid_length != record.valid_sequence_length or valid_length != features.shape[0]:
                errors.append(f"{prefix}: valid length/inference count mismatch")
            if rollout_id != record.rollout_id:
                errors.append(f"{prefix}: feature rollout_id mismatch")
            if failed != int(record.failed):
                errors.append(f"{prefix}: feature failure label disagrees with manifest")
            if tensor_schema_version != record.schema_version:
                errors.append(f"{prefix}: feature schema_version disagrees with manifest")
            if steps.tolist() != record.inference_env_steps:
                errors.append(f"{prefix}: inference environment steps mismatch")
            if len(steps) and (np.diff(steps) <= 0).any():
                errors.append(f"{prefix}: inference steps are not strictly increasing")
            spacings = np.diff(steps)
            if (
                len(steps) > 1
                and record.model_family == "abot_m05"
                and spacings[0] == record.action_horizon // 2
            ):
                spacings = spacings[1:]
            if len(spacings) and (spacings < record.replan_steps).any():
                errors.append(
                    f"{prefix}: inference spacing is below replan_steps; cached actions may have duplicated features"
                )
            if len(steps) and (steps[0] < 0 or steps[-1] >= record.num_env_steps):
                errors.append(f"{prefix}: inference step lies outside rollout")
            if chunks.ndim != 3 or chunks.shape[:2] != (
                record.valid_sequence_length,
                record.action_horizon,
            ):
                errors.append(f"{prefix}: predicted action chunks are incompatible with official SAFE")
                official_compatible = False
            if not np.isfinite(chunks).all():
                errors.append(f"{prefix}: predicted action chunks contain non-finite values")
            expected_metadata = {
                "model_family": record.model_family,
                "feature_layer": record.feature_layer,
                "feature_dtype": record.feature_dtype,
                "feature_aggregation": record.feature_aggregation,
                "feature_shape": record.feature_shape,
                "policy_name": record.policy_id,
                "policy_checkpoint": record.checkpoint,
                "action_horizon": record.action_horizon,
                "flow_steps": record.flow_steps,
            }
            for key, value in expected_metadata.items():
                actual = metadata.get(
                    key,
                    "pi0" if key == "model_family" else None,
                )
                if actual != value:
                    errors.append(f"{prefix}: feature metadata mismatch for {key}")
        except Exception as error:
            errors.append(f"{prefix}: feature validation failed: {error}")
            official_compatible = False
        identities.add(
            (
                record.policy_id,
                record.checkpoint,
                record.feature_layer,
                tuple(record.feature_shape[1:]),
                record.feature_dtype,
                record.feature_aggregation,
            )
        )
        if record.action_recording_requested:
            if not record.action_path or not (dataset_dir / record.action_path).is_file():
                errors.append(f"{prefix}: requested action artifact is missing")
        if record.video_recording_requested:
            if not record.video_path or not (dataset_dir / record.video_path).is_file():
                errors.append(f"{prefix}: requested video artifact is missing")
    if len(identities) > 1:
        errors.append("Dataset mixes incompatible checkpoints or SAFE feature schemas")
        official_compatible = False

    rollout_dir = dataset_dir / "rollouts"
    if rollout_dir.exists():
        for path in rollout_dir.glob("*.npz"):
            if path.resolve() not in feature_files:
                errors.append(f"Orphan feature file not present in manifest: {path.relative_to(dataset_dir)}")
    incomplete_dir = dataset_dir / "incomplete"
    if incomplete_dir.exists() and any(path.is_file() for path in incomplete_dir.rglob("*")):
        warnings.append("Quarantined incomplete artifacts are present under incomplete/")
    temporary = [path for path in dataset_dir.rglob("*.tmp*") if path.is_file()]
    if temporary:
        errors.extend(f"Incomplete temporary file: {path.relative_to(dataset_dir)}" for path in temporary)
    if summary is not None and records:
        counts = summary.get("counts", {})
        expected = {
            "valid_rollouts": len(records),
            "successes": sum(not record.failed for record in records),
            "failures": sum(record.failed for record in records),
        }
        for key, value in expected.items():
            if counts.get(key) != value:
                errors.append(f"Summary count {key}={counts.get(key)!r}, expected {value}")
    result = {
        "valid": not errors,
        "dataset_dir": str(dataset_dir),
        "num_rollouts": len(records),
        "num_successes": sum(not record.failed for record in records),
        "num_failures": sum(record.failed for record in records),
        "official_safe_loader_compatible": bool(official_compatible and not errors),
        "errors": errors,
        "warnings": warnings,
    }
    return result


def format_report(result):
    status = "VALID" if result["valid"] else "INVALID"
    lines = [
        f"Atomic SAFE dataset: {status}",
        f"Rollouts: {result['num_rollouts']} "
        f"({result['num_successes']} successes, {result['num_failures']} failures)",
        f"Official SAFE loader compatible: {result['official_safe_loader_compatible']}",
    ]
    lines.extend(f"ERROR: {error}" for error in result["errors"])
    lines.extend(f"WARNING: {warning}" for warning in result["warnings"])
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--allow-unregistered-atomic-tasks", action="store_true")
    parser.add_argument("--json-output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = validate_atomic_dataset(
        args.dataset_dir,
        allow_unregistered=args.allow_unregistered_atomic_tasks,
    )
    print(format_report(result))
    if args.json_output:
        atomic_write_json(args.json_output, result)
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
