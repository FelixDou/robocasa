"""Export usable semantic Subtask-SAFE segments for the official SAFE loader."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re

import numpy as np

from .collect_atomic_rollouts import SUMMARY_NAME, atomic_write_json, utc_now
from .dataset import MANIFEST_NAME, assert_compatible, load_manifest
from .export_to_official_safe import atomic_pickle, _check_existing_pickle
from .subtask_safe import validate_subtask_safe_record
from .validate_atomic_dataset import validate_atomic_dataset


REPORT_NAME = "conversion_report.json"
SPLIT_NAME = "parent_rollout_split.json"
FORMAT_NAME = "official_safe_subtask_env_records_policy_records"


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value or "segment"


def subtask_source_fingerprint(dataset_dir: str | Path, records) -> str:
    """Fingerprint both the manifest and semantic label artifacts."""
    dataset_dir = Path(dataset_dir)
    digest = hashlib.sha256()
    manifest = dataset_dir / MANIFEST_NAME
    digest.update(MANIFEST_NAME.encode())
    digest.update(manifest.read_bytes())
    for record in sorted(records, key=lambda item: item.rollout_id):
        if not record.subtask_trace_path:
            raise ValueError(f"Rollout {record.rollout_id} has no subtask trace path")
        path = dataset_dir / record.subtask_trace_path
        digest.update(record.subtask_trace_path.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_usable_segments(dataset_dir: str | Path, *, allow_unregistered=False):
    """Load validated usable segments without materializing feature tensors."""
    dataset_dir = Path(dataset_dir).resolve()
    validation = validate_atomic_dataset(
        dataset_dir,
        allow_unregistered=allow_unregistered,
    )
    if not validation["valid"]:
        raise ValueError("Source dataset is invalid: " + "; ".join(validation["errors"]))
    if not validation["official_safe_loader_compatible"]:
        raise ValueError("Source dataset lacks fields required by the official SAFE loader")

    summary = json.loads((dataset_dir / SUMMARY_NAME).read_text())
    configured_types = summary.get("config", {}).get("task_types", {})
    records = load_manifest(dataset_dir)
    assert_compatible(records)
    segments = []
    excluded = defaultdict(int)
    definitions = {}
    for record in records:
        if not record.subtask_recording_requested or not record.subtask_trace_path:
            raise ValueError(
                f"Rollout {record.rollout_id} was not collected with Subtask-SAFE"
            )
        subtask_record = json.loads(
            (dataset_dir / record.subtask_trace_path).read_text()
        )
        validate_subtask_safe_record(
            subtask_record,
            rollout_id=record.rollout_id,
            rollout_failed=record.failed,
            inference_environment_steps=record.inference_env_steps,
            task_name=record.task_name,
        )
        task_type = configured_types.get(record.task_name, "atomic")
        for definition in subtask_record["semantic_subtasks"]:
            key = (record.task_name, definition["subtask_id"])
            contract = {
                "parent_task_name": record.task_name,
                "parent_task_type": task_type,
                "subtask_id": definition["subtask_id"],
                "subtask_instruction": definition["instruction"],
                "predicate_names": list(definition["predicate_names"]),
                "source_subtask_ids": list(definition["source_subtask_ids"]),
            }
            if key in definitions and definitions[key] != contract:
                raise ValueError(
                    "Semantic subtask definition changed across rollouts for "
                    f"{record.task_name}/{definition['subtask_id']}"
                )
            definitions[key] = contract
        excluded["completed_without_activation"] += len(
            subtask_record.get("excluded_completed_subtasks", [])
        )
        excluded["bypassed_optional"] += len(
            subtask_record.get("excluded_bypassed_subtasks", [])
        )
        for segment in subtask_record.get("segments", []):
            if not segment.get("usable_for_safe"):
                if segment.get("failure_label") is not None:
                    excluded["labeled_without_inference"] += 1
                else:
                    excluded["unlabeled"] += 1
                continue
            start = segment.get("inference_start_index")
            end = segment.get("inference_end_index_exclusive")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > record.valid_sequence_length
                or end - start != int(segment["num_policy_inferences"])
            ):
                raise ValueError(
                    f"Invalid inference slice for segment {segment.get('segment_id')}"
                )
            label = int(segment["failure_label"])
            if label not in (0, 1):
                raise ValueError("Subtask-SAFE failure labels must be binary")
            key = (record.task_name, segment["subtask_id"])
            definition = definitions[key]
            segment_id = str(segment["segment_id"])
            segments.append(
                {
                    **definition,
                    "segment_id": segment_id,
                    "segment_index": int(segment["segment_index"]),
                    "failure_label": label,
                    "parent_rollout_id": record.rollout_id,
                    "parent_rollout_failed": bool(record.failed),
                    "parent_environment_seed": int(record.environment_seed),
                    "parent_environment_reset_index": record.environment_reset_index,
                    "inference_start_index": start,
                    "inference_end_index_exclusive": end,
                    "num_policy_inferences": end - start,
                    "entry_environment_step": int(segment["entry_environment_step"]),
                    "end_environment_step": int(segment["end_environment_step"]),
                    "completion_environment_step": segment.get(
                        "completion_environment_step"
                    ),
                    "terminal_failure_reason": segment.get(
                        "terminal_failure_reason"
                    ),
                    "terminal_unsatisfied_predicate_names": list(
                        segment.get("terminal_unsatisfied_predicate_names", [])
                    ),
                    "source_record": record,
                }
            )
    ids = [segment["segment_id"] for segment in segments]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Subtask-SAFE segment IDs")
    if not segments:
        raise ValueError("Source dataset contains no usable Subtask-SAFE segments")
    if {segment["failure_label"] for segment in segments} != {0, 1}:
        raise ValueError("Usable Subtask-SAFE segments must contain both labels")
    segments.sort(
        key=lambda item: (
            item["parent_task_name"],
            item["parent_environment_seed"],
            item["parent_environment_reset_index"]
            if item["parent_environment_reset_index"] is not None
            else -1,
            item["parent_rollout_id"],
            item["segment_index"],
        )
    )
    return records, segments, dict(excluded)


def _split_counts(segments, ids):
    selected = [segment for segment in segments if segment["segment_id"] in ids]
    failures = sum(segment["failure_label"] for segment in selected)
    return {
        "segments": len(selected),
        "successes": len(selected) - failures,
        "failures": failures,
        "parent_rollouts": len({item["parent_rollout_id"] for item in selected}),
    }


def build_parent_rollout_split(
    segments,
    *,
    train_fraction=0.7,
    seed=0,
    max_attempts=1000,
):
    """Split parent rollouts first, stratified by task and rollout outcome."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")
    parents = {}
    for segment in segments:
        parent_id = segment["parent_rollout_id"]
        identity = (
            segment["parent_task_name"],
            bool(segment["parent_rollout_failed"]),
        )
        current = parents.setdefault(
            parent_id,
            {"identity": identity, "segment_ids": []},
        )
        if current["identity"] != identity:
            raise ValueError(f"Parent rollout metadata changed for {parent_id}")
        current["segment_ids"].append(segment["segment_id"])
    strata = defaultdict(list)
    for parent_id, parent in parents.items():
        strata[parent["identity"]].append(parent_id)
    too_small = {stratum: len(values) for stratum, values in strata.items() if len(values) < 2}
    if too_small:
        raise ValueError(f"Parent task/outcome strata are too small to split: {too_small}")

    task_subtasks = {
        (segment["parent_task_name"], segment["subtask_id"])
        for segment in segments
    }
    for attempt in range(max_attempts):
        parent_train = set()
        parent_test = set()
        for stratum, values in sorted(strata.items()):
            values = sorted(values)
            rng = random.Random(f"{seed + attempt}:{stratum[0]}:{int(stratum[1])}")
            rng.shuffle(values)
            count = max(1, min(len(values) - 1, int(len(values) * train_fraction)))
            parent_train.update(values[:count])
            parent_test.update(values[count:])
        train = {
            segment["segment_id"]
            for segment in segments
            if segment["parent_rollout_id"] in parent_train
        }
        test = {
            segment["segment_id"]
            for segment in segments
            if segment["parent_rollout_id"] in parent_test
        }
        train_pairs = {
            (segment["parent_task_name"], segment["subtask_id"])
            for segment in segments
            if segment["segment_id"] in train
        }
        test_pairs = {
            (segment["parent_task_name"], segment["subtask_id"])
            for segment in segments
            if segment["segment_id"] in test
        }
        train_labels = {
            segment["failure_label"]
            for segment in segments
            if segment["segment_id"] in train
        }
        test_labels = {
            segment["failure_label"]
            for segment in segments
            if segment["segment_id"] in test
        }
        if (
            train_pairs == task_subtasks
            and test_pairs == task_subtasks
            and train_labels == {0, 1}
            and test_labels == {0, 1}
        ):
            break
    else:
        raise ValueError(
            "Could not find a parent-rollout split with complete subtask coverage "
            f"after {max_attempts} attempts"
        )

    all_ids = {segment["segment_id"] for segment in segments}
    if train & test or train | test != all_ids or parent_train & parent_test:
        raise AssertionError("Parent-rollout split leaked, lost, or duplicated segments")
    per_parent_stratum = {}
    for (task_name, failed), values in sorted(strata.items()):
        per_parent_stratum[f"{task_name}::{int(failed)}"] = {
            "parent_task_name": task_name,
            "parent_rollout_failed": failed,
            "source": len(values),
            "train": sum(value in parent_train for value in values),
            "test": sum(value in parent_test for value in values),
        }
    return {
        "schema_version": 1,
        "protocol": "subtask_safe_parent_rollout_stratified",
        "split_unit": "parent_rollout",
        "stratified_by": ["parent_task_name", "parent_rollout_failed"],
        "split_seed": int(seed),
        "effective_split_seed": int(seed + attempt),
        "train_fraction": float(train_fraction),
        "train": sorted(train),
        "test": sorted(test),
        "parent_train": sorted(parent_train),
        "parent_test": sorted(parent_test),
        "counts": {
            "train": _split_counts(segments, train),
            "test": _split_counts(segments, test),
        },
        "per_parent_stratum": per_parent_stratum,
        "subtask_coverage_required_in_each_split": True,
    }


def export_subtask_safe(
    dataset_dir,
    output_dir,
    *,
    train_fraction=0.7,
    split_seed=0,
    resume=False,
    dry_run=False,
    allow_unregistered=False,
):
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    records, segments, excluded = load_usable_segments(
        dataset_dir,
        allow_unregistered=allow_unregistered,
    )
    compatibility = assert_compatible(records)
    fingerprint = subtask_source_fingerprint(dataset_dir, records)
    split = build_parent_rollout_split(
        segments,
        train_fraction=train_fraction,
        seed=split_seed,
    )
    split_fingerprint = hashlib.sha256(
        json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    task_keys = sorted(
        {
            f"{segment['parent_task_name']}::{segment['subtask_id']}"
            for segment in segments
        }
    )
    task_ids = {key: index for index, key in enumerate(task_keys)}
    task_types = {
        f"{segment['parent_task_name']}::{segment['subtask_id']}": segment[
            "parent_task_type"
        ]
        for segment in segments
    }
    failures = sum(segment["failure_label"] for segment in segments)
    plan = {
        "source_dataset": str(dataset_dir),
        "output_dir": str(output_dir),
        "source_fingerprint": fingerprint,
        "compatibility_key": compatibility,
        "format": FORMAT_NAME,
        "num_parent_rollouts": len({item["parent_rollout_id"] for item in segments}),
        "num_rollouts": len(segments),
        "num_segments": len(segments),
        "num_success_segments": len(segments) - failures,
        "num_failure_segments": failures,
        "num_policy_records": sum(item["num_policy_inferences"] for item in segments),
        "task_ids": task_ids,
        "task_types": task_types,
        "parent_task_names": sorted({item["parent_task_name"] for item in segments}),
        "model_families": sorted({record.model_family for record in records}),
        "excluded": excluded,
        "split_fingerprint": split_fingerprint,
        "split": {
            key: value
            for key, value in split.items()
            if key not in {"train", "test", "parent_train", "parent_test"}
        },
    }
    if dry_run:
        return {"dry_run": True, **plan}
    report_path = output_dir / REPORT_NAME
    if output_dir.exists() and not resume and any(output_dir.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output_dir}; pass --resume")
    if report_path.exists() and resume:
        previous = json.loads(report_path.read_text())
        if previous.get("source_fingerprint") != fingerprint:
            raise ValueError("Existing export refers to different Subtask-SAFE labels")
        if previous.get("split") != plan["split"]:
            raise ValueError("Existing export uses a different parent-rollout split")
        if previous.get("split_fingerprint") != split_fingerprint:
            raise ValueError("Existing export has a different parent-rollout assignment")
        if previous.get("complete"):
            expected_env = len(segments)
            expected_policy = plan["num_policy_records"]
            split_path = output_dir / SPLIT_NAME
            existing_split = (
                json.loads(split_path.read_text()) if split_path.is_file() else None
            )
            expected_split = {
                **split,
                "source_fingerprint": fingerprint,
                "export_dir": str(output_dir),
            }
            if (
                len(list((output_dir / "env_records").glob("*.pkl"))) == expected_env
                and len(list((output_dir / "policy_records").glob("*meta.pkl")))
                == expected_policy
                and existing_split == expected_split
            ):
                return previous

    env_dir = output_dir / "env_records"
    policy_dir = output_dir / "policy_records"
    env_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        **split,
        "source_fingerprint": fingerprint,
        "export_dir": str(output_dir),
    }
    atomic_write_json(output_dir / SPLIT_NAME, split_payload)
    mapping = []
    materialized_bytes = 0
    global_policy_index = 0
    episode_by_task = defaultdict(int)
    for segment_index, segment in enumerate(segments):
        source = segment["source_record"]
        task_key = f"{segment['parent_task_name']}::{segment['subtask_id']}"
        episode_index = episode_by_task[task_key]
        episode_by_task[task_key] += 1
        segment_id = segment["segment_id"]
        file_stem = (
            f"segment_{segment_index:08d}--{source.rollout_id}--"
            f"{segment['segment_index']:02d}-{_safe_name(segment['subtask_id'])}"
        )
        env_path = env_dir / f"{file_stem}.pkl"
        serializable_segment = {
            key: value
            for key, value in segment.items()
            if key != "source_record"
        }
        env_record = {
            "rollout_id": segment_id,
            "parent_rollout_id": source.rollout_id,
            "parent_task_name": source.task_name,
            "parent_task_type": segment["parent_task_type"],
            "parent_rollout_failed": bool(source.failed),
            "task_suite_name": "robocasa_subtask",
            "task_type": segment["parent_task_type"],
            "task_id": task_ids[task_key],
            "task_name": task_key,
            "task_description": segment["subtask_instruction"],
            "subtask_id": segment["subtask_id"],
            "subtask_index": segment["segment_index"],
            "subtask_instruction": segment["subtask_instruction"],
            "episode_idx": episode_index,
            "episode_success": 1 - segment["failure_label"],
            "environment_seed": source.environment_seed,
            "environment_reset_index": source.environment_reset_index,
            "seed_protocol": source.seed_protocol,
            "video_frame_stride": source.video_frame_stride,
            "model_infer_times": segment["num_policy_inferences"],
            "inference_environment_steps": list(
                source.inference_env_steps[
                    segment["inference_start_index"] :
                    segment["inference_end_index_exclusive"]
                ]
            ),
            "replan_steps": source.replan_steps,
            "policy_name": source.policy_id,
            "policy_checkpoint": source.checkpoint,
            "model_family": source.model_family,
            "subtask_safe_segment": serializable_segment,
            "robocasa_manifest_record": source.to_dict(),
        }
        if not _check_existing_pickle(env_path, segment_id, resume):
            atomic_pickle(env_path, env_record)
        video_link = env_path.with_suffix(".mp4")
        if source.video_path:
            source_video = (dataset_dir / source.video_path).resolve()
            if video_link.exists() or video_link.is_symlink():
                if not resume:
                    raise FileExistsError(f"Video link exists: {video_link}")
            else:
                video_link.symlink_to(os.path.relpath(source_video, video_link.parent))

        with np.load(dataset_dir / source.tensor_path, allow_pickle=False) as payload:
            features = payload["features"]
            chunks = payload["policy_action_chunks"]
            steps = payload["inference_environment_steps"]
            start = segment["inference_start_index"]
            end = segment["inference_end_index_exclusive"]
            features = features[start:end]
            chunks = chunks[start:end]
            steps = steps[start:end]
        policy_paths = []
        for local_index in range(segment["num_policy_inferences"]):
            source_index = segment["inference_start_index"] + local_index
            policy_path = policy_dir / (
                f"step_{global_policy_index:012d}--{_safe_name(segment_id)}--"
                f"infer_{local_index:06d}--meta.pkl"
            )
            policy_record = {
                "rollout_id": segment_id,
                "parent_rollout_id": source.rollout_id,
                "subtask_id": segment["subtask_id"],
                "inference_index": local_index,
                "source_inference_index": source_index,
                "environment_step": int(steps[local_index]),
                "pre_velocity": np.asarray(features[local_index], dtype=np.float32),
                "actions": np.asarray(chunks[local_index], dtype=np.float32),
                "feature_layer": source.feature_layer,
                "model_family": source.model_family,
                "policy_name": source.policy_id,
                "policy_checkpoint": source.checkpoint,
            }
            if not _check_existing_pickle(
                policy_path,
                segment_id,
                resume,
                inference_index=local_index,
            ):
                atomic_pickle(policy_path, policy_record)
                materialized_bytes += policy_path.stat().st_size
            policy_paths.append(str(policy_path.relative_to(output_dir)))
            global_policy_index += 1
        mapping.append(
            {
                "segment_id": segment_id,
                "parent_rollout_id": source.rollout_id,
                "task_name": source.task_name,
                "subtask_id": segment["subtask_id"],
                "failure_label": segment["failure_label"],
                "source_inference_slice": [
                    segment["inference_start_index"],
                    segment["inference_end_index_exclusive"],
                ],
                "source_feature_path": source.tensor_path,
                "env_record": str(env_path.relative_to(output_dir)),
                "policy_records": policy_paths,
            }
        )
    report = {
        "schema_version": 1,
        "complete": True,
        "created_at": utc_now(),
        **plan,
        "split_manifest": SPLIT_NAME,
        "materialized_feature_bytes_this_run": materialized_bytes,
        "note": (
            "Each official SAFE pseudo-rollout is one usable semantic subtask "
            "segment. Parent rollouts are split before segment extraction; excluded, "
            "bypassed, unlabeled, and labeled-without-inference segments are not exported."
        ),
        "mapping": mapping,
    }
    atomic_write_json(report_path, report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unregistered-tasks", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = export_subtask_safe(
            args.dataset_dir,
            args.output_dir,
            train_fraction=args.train_fraction,
            split_seed=args.split_seed,
            resume=args.resume,
            dry_run=args.dry_run,
            allow_unregistered=args.allow_unregistered_tasks,
        )
    except (FileExistsError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
