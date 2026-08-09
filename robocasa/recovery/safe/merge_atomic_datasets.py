"""Merge validated RoboCasa SAFE rollout shards without duplicating artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from .collect_atomic_rollouts import (
    ERRORS_NAME,
    SKIPPED_NAME,
    SUMMARY_NAME,
    atomic_write_json,
    make_summary,
    read_jsonl,
)
from .dataset import MANIFEST_NAME, assert_compatible, load_manifest
from .validate_atomic_dataset import validate_atomic_dataset


COMPATIBLE_CONFIG_KEYS = (
    "split",
    "seed_protocol",
    "policy_module",
    "policy_name",
    "policy_checkpoint",
    "policy_config",
    "replan_steps",
    "horizon_source",
    "record_actions",
    "video_frame_stride",
    "record_safe_features",
    "record_subtask_trace",
    "model_family",
    "safe_repository_commit",
    "official_safe_openpi_commit",
    "openpi_repository_commit",
    "rldx_repository_commit",
)
POLICY_PROVENANCE_KEY = "collection_provenance"


def _compatible_config_value(config, key):
    if key == "policy_config":
        policy_config = dict(config.get(key) or {})
        policy_config.pop(POLICY_PROVENANCE_KEY, None)
        return policy_config
    return config.get(
        key,
        (
            "pi0"
            if key == "model_family"
            else False
            if key == "record_subtask_trace"
            else None
        ),
    )


def _assert_configs_compatible(configs):
    reference = configs[0]
    for index, config in enumerate(configs[1:], 1):
        mismatches = [
            key
            for key in COMPATIBLE_CONFIG_KEYS
            if _compatible_config_value(config, key)
            != _compatible_config_value(reference, key)
        ]
        if mismatches:
            raise ValueError(
                f"Source dataset {index} has incompatible configuration: "
                + ", ".join(mismatches)
            )


def _materialize(source, destination, *, copy):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Merge destination already exists: {destination}")
    if copy:
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def _write_jsonl(path, values):
    path = Path(path)
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def merge_atomic_datasets(source_dirs, output_dir, *, copy=False):
    sources = [Path(source).resolve() for source in source_dirs]
    output_dir = Path(output_dir).resolve()
    if len(sources) < 2:
        raise ValueError("At least two source datasets are required")
    if len(sources) != len(set(sources)):
        raise ValueError("Source dataset list contains duplicates")
    if output_dir in sources:
        raise ValueError("Output directory cannot also be a source dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Merge output is not empty: {output_dir}")

    source_summaries = []
    source_records = []
    for source in sources:
        validation = validate_atomic_dataset(source)
        if not validation["valid"]:
            raise ValueError(
                f"Source dataset is invalid: {source}: "
                + "; ".join(validation["errors"])
            )
        source_summaries.append(json.loads((source / SUMMARY_NAME).read_text()))
        source_records.append(load_manifest(source))

    configs = [summary["config"] for summary in source_summaries]
    _assert_configs_compatible(configs)
    records = [record for shard in source_records for record in shard]
    assert_compatible(records)

    rollout_ids = [record.rollout_id for record in records]
    if len(rollout_ids) != len(set(rollout_ids)):
        raise ValueError("Source datasets contain duplicate rollout IDs")
    episode_ids = [
        (
            record.task_name,
            record.environment_seed,
            record.environment_reset_index,
        )
        for record in records
    ]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Source datasets contain duplicate task/seed/reset episodes")

    output_dir.mkdir(parents=True, exist_ok=True)
    materialization_counts = {"hardlink": 0, "copy": 0, "copy_fallback": 0}
    for source, shard in zip(sources, source_records):
        for record in shard:
            relative_paths = [record.tensor_path]
            if record.action_path:
                relative_paths.append(record.action_path)
            if record.video_path:
                relative_paths.append(record.video_path)
            if record.subtask_trace_path:
                relative_paths.append(record.subtask_trace_path)
            for relative_path in relative_paths:
                source_path = source / relative_path
                destination = output_dir / relative_path
                mode = _materialize(source_path, destination, copy=copy)
                materialization_counts[mode] += 1

    records.sort(
        key=lambda record: (
            record.task_name,
            record.environment_seed,
            record.environment_reset_index
            if record.environment_reset_index is not None
            else -1,
            record.rollout_id,
        )
    )
    _write_jsonl(output_dir / MANIFEST_NAME, [record.to_dict() for record in records])

    errors = []
    skipped = []
    for source in sources:
        for destination, name in ((errors, ERRORS_NAME), (skipped, SKIPPED_NAME)):
            destination.extend(
                {**event, "source_dataset": str(source)}
                for event in read_jsonl(source / name)
            )
    _write_jsonl(output_dir / ERRORS_NAME, errors)
    _write_jsonl(output_dir / SKIPPED_NAME, skipped)

    tasks = []
    task_horizons = {}
    task_types = {}
    for config in configs:
        for task in config["tasks"]:
            if task in task_horizons:
                if task_horizons[task] != config["task_horizons"][task]:
                    raise ValueError(
                        f"Task {task} has conflicting rollout horizons across shards"
                    )
                continue
            tasks.append(task)
            task_horizons[task] = config["task_horizons"][task]
            task_types[task] = config.get("task_types", {}).get(task, "atomic")
    unique_task_types = set(task_types.values())
    task_scope = (
        next(iter(unique_task_types))
        if len(unique_task_types) == 1
        else "mixed"
    )
    base_environment_seeds = sorted(
        {
            config.get("base_environment_seed")
            for config in configs
            if config.get("base_environment_seed") is not None
        }
    )
    source_collection_quotas = [
        {
            "source_dataset": str(source),
            "success_quota": config.get("success_quota"),
            "failure_quota": config.get("failure_quota"),
            "retain_only_quota": config.get("retain_only_quota", False),
        }
        for source, config in zip(sources, configs)
    ]
    source_artifact_recording = [
        {
            "source_dataset": str(source),
            "record_actions": config.get("record_actions"),
            "record_videos": config.get("record_videos"),
            "record_subtask_trace": config.get("record_subtask_trace", False),
        }
        for source, config in zip(sources, configs)
    ]
    source_policy_provenance = [
        {
            "source_dataset": str(source),
            **config.get("policy_config", {}).get(POLICY_PROVENANCE_KEY, {}),
        }
        for source, config in zip(sources, configs)
        if config.get("policy_config", {}).get(POLICY_PROVENANCE_KEY) is not None
    ]
    record_video_values = {config.get("record_videos") for config in configs}
    robocasa_commits = sorted(
        {
            config.get("robocasa_commit")
            for config in configs
            if config.get("robocasa_commit") is not None
        }
    )
    config = dict(configs[0])
    config["policy_config"] = _compatible_config_value(config, "policy_config")
    config.update(
        {
            "tasks": tasks,
            "task_types": task_types,
            "task_scope": task_scope,
            "dataset_type": f"robocasa_{task_scope}_safe_rollouts",
            "task_horizons": task_horizons,
            "base_environment_seed": None,
            "base_environment_seeds": base_environment_seeds,
            "seeds": base_environment_seeds,
            "environment_reset_indices": None,
            "success_quota": None,
            "failure_quota": None,
            "retain_only_quota": False,
            "record_videos": (
                next(iter(record_video_values))
                if len(record_video_values) == 1
                else None
            ),
            "robocasa_commit": (
                robocasa_commits[0] if len(robocasa_commits) == 1 else None
            ),
            "robocasa_commits": robocasa_commits,
            "output_dir": str(output_dir),
            "source_datasets": [str(source) for source in sources],
            "source_collection_quotas": source_collection_quotas,
            "source_artifact_recording": source_artifact_recording,
            "source_policy_provenance": source_policy_provenance,
            "source_ports": [source_config.get("port") for source_config in configs],
            "artifact_materialization": materialization_counts,
        }
    )
    summary = make_summary(config, records, errors, skipped, partial=False)
    atomic_write_json(output_dir / SUMMARY_NAME, summary)

    validation = validate_atomic_dataset(output_dir)
    if not validation["valid"]:
        raise RuntimeError(
            "Merged dataset failed validation: " + "; ".join(validation["errors"])
        )
    return {"summary": summary, "validation": validation}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy artifacts instead of hard-linking them when source and output share storage",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = merge_atomic_datasets(
        args.source_dirs,
        args.output_dir,
        copy=args.copy,
    )
    validation = result["validation"]
    print(f"Merged dataset: {validation['dataset_dir']}")
    print(
        f"Rollouts: {validation['num_rollouts']} "
        f"({validation['num_successes']} successes, "
        f"{validation['num_failures']} failures)"
    )
    print(f"Official SAFE loader compatible: {validation['official_safe_loader_compatible']}")


if __name__ == "__main__":
    main()
