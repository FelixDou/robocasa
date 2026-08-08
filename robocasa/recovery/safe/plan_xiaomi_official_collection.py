"""Plan raw SAFE collection from Xiaomi's pinned official task results.

This module is intentionally simulator-free.  It validates the released
RoboCasa365 summary against the local target50 registry, selects tasks by an
inclusive official success-count interval, and balances them across collection
shards by official environment horizon.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil

from .atomic_tasks import (
    registered_atomic_tasks,
    registered_safe_task_horizons,
    registered_target50_tasks,
)


XIAOMI_ROBOTICS_1_COMMIT = "4da1db0a4deefa6de7ebb4ef0b8754017290f5f7"
XIAOMI_CHECKPOINT_REVISION = "0d1aa76d0d82debc9b611e4d1e231096434d5be4"
OFFICIAL_NUM_TASKS = 50
OFFICIAL_EPISODES_PER_TASK = 50
OFFICIAL_NUM_EPISODES = 2500
OFFICIAL_SUCCESSES = 1432
PLAN_SCHEMA_VERSION = "xr1_official_safe_collection_plan_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_summary(path: str | Path) -> dict:
    path = Path(path)
    try:
        summary = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Xiaomi summary JSON at {path}: {error}") from error
    if not isinstance(summary, dict):
        raise ValueError("Official Xiaomi summary must be a JSON object")
    return summary


def validate_official_summary(summary: dict, *, registry_path=None) -> list[dict]:
    """Validate the pinned 50-task, 2,500-rollout release summary."""
    expected_top_level = {
        "num_tasks": OFFICIAL_NUM_TASKS,
        "num_episodes": OFFICIAL_NUM_EPISODES,
        "successes": OFFICIAL_SUCCESSES,
        "split": "pretrain",
        "task_set": "target50",
    }
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in expected_top_level.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Official Xiaomi summary metadata mismatch: {mismatches}")

    tasks = summary.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("Official Xiaomi summary has no task mapping")
    target50 = registered_target50_tasks(registry_path)
    if set(tasks) != set(target50):
        missing = sorted(set(target50) - set(tasks))
        extra = sorted(set(tasks) - set(target50))
        raise ValueError(
            f"Official Xiaomi summary task mismatch: missing={missing}, extra={extra}"
        )

    registry_horizons = registered_safe_task_horizons(registry_path)
    atomic_tasks = registered_atomic_tasks(registry_path)
    records = []
    observed_global_indices = []
    for task_index, task_name in enumerate(target50):
        payload = tasks[task_name]
        if not isinstance(payload, dict):
            raise ValueError(f"Task result for {task_name} must be an object")
        num_episodes = payload.get("num_episodes")
        successes = payload.get("successes")
        horizon = payload.get("horizon")
        if num_episodes != OFFICIAL_EPISODES_PER_TASK:
            raise ValueError(
                f"{task_name} has {num_episodes} episodes, expected "
                f"{OFFICIAL_EPISODES_PER_TASK}"
            )
        if not isinstance(successes, int) or isinstance(successes, bool):
            raise ValueError(f"{task_name} has invalid success count {successes!r}")
        if not 0 <= successes <= num_episodes:
            raise ValueError(f"{task_name} has impossible success count {successes}")
        expected_horizon = registry_horizons.get(task_name)
        if horizon != expected_horizon:
            raise ValueError(
                f"{task_name} horizon mismatch: official={horizon}, "
                f"registry={expected_horizon}"
            )

        episodes = payload.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != num_episodes:
            raise ValueError(
                f"{task_name} must contain {num_episodes} official episode records"
            )
        episode_successes = sum(episode.get("success") is True for episode in episodes)
        if episode_successes != successes:
            raise ValueError(
                f"{task_name} episode labels total {episode_successes}, "
                f"but task summary reports {successes}"
            )
        observed_global_indices.extend(
            episode.get("global_episode_index") for episode in episodes
        )
        records.append(
            {
                "task": task_name,
                "target50_index": task_index,
                "task_type": "atomic" if task_name in atomic_tasks else "composite",
                "official_num_episodes": num_episodes,
                "official_successes": successes,
                "official_failures": num_episodes - successes,
                "official_success_rate": successes / num_episodes,
                "horizon": horizon,
            }
        )

    if sum(record["official_successes"] for record in records) != OFFICIAL_SUCCESSES:
        raise ValueError("Per-task Xiaomi successes do not sum to the release total")
    if sorted(observed_global_indices) != list(range(OFFICIAL_NUM_EPISODES)):
        raise ValueError("Official Xiaomi global episode indices are not exactly 0..2499")
    return records


def select_tasks(records, *, min_successes: int, max_successes: int):
    if not 0 <= min_successes <= max_successes <= OFFICIAL_EPISODES_PER_TASK:
        raise ValueError(
            "Success interval must satisfy 0 <= min <= max <= "
            f"{OFFICIAL_EPISODES_PER_TASK}"
        )
    eligible = []
    excluded = []
    for record in records:
        successes = record["official_successes"]
        if min_successes <= successes <= max_successes:
            eligible.append(dict(record))
        else:
            excluded.append(
                {
                    **record,
                    "exclusion_reason": (
                        "below_min_successes"
                        if successes < min_successes
                        else "above_max_successes"
                    ),
                }
            )
    if not eligible:
        raise ValueError("Official success interval selected no tasks")
    return eligible, excluded


def balance_shards(records, *, num_shards: int, rollouts_per_task: int) -> list[dict]:
    if num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if rollouts_per_task < 1:
        raise ValueError("--rollouts-per-task must be positive")
    shards = [
        {"shard_index": index, "tasks": [], "horizon_units": 0}
        for index in range(num_shards)
    ]
    for record in sorted(records, key=lambda item: (-item["horizon"], item["task"])):
        shard = min(
            shards,
            key=lambda item: (
                item["horizon_units"],
                len(item["tasks"]),
                item["shard_index"],
            ),
        )
        shard["tasks"].append(record["task"])
        shard["horizon_units"] += record["horizon"]
    for shard in shards:
        shard["tasks"].sort()
        shard["task_count"] = len(shard["tasks"])
        shard["expected_rollouts"] = len(shard["tasks"]) * rollouts_per_task
        shard["rollout_horizon_units"] = (
            shard["horizon_units"] * rollouts_per_task
        )
    return shards


def build_plan(
    summary: dict,
    *,
    summary_path: str | Path,
    min_successes: int,
    max_successes: int,
    rollouts_per_task: int,
    num_shards: int,
    estimated_gib_per_rollout: float,
    storage_reserve_factor: float,
    registry_path=None,
) -> dict:
    if estimated_gib_per_rollout <= 0:
        raise ValueError("--estimated-gib-per-rollout must be positive")
    if storage_reserve_factor < 1:
        raise ValueError("--storage-reserve-factor must be at least 1")
    records = validate_official_summary(summary, registry_path=registry_path)
    eligible, excluded = select_tasks(
        records,
        min_successes=min_successes,
        max_successes=max_successes,
    )
    shards = balance_shards(
        eligible,
        num_shards=num_shards,
        rollouts_per_task=rollouts_per_task,
    )
    expected_rollouts = len(eligible) * rollouts_per_task
    estimated_storage_gib = expected_rollouts * estimated_gib_per_rollout
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "source": {
            "path": str(Path(summary_path).resolve()),
            "sha256": _sha256(Path(summary_path)),
            "xiaomi_repository_commit": XIAOMI_ROBOTICS_1_COMMIT,
            "xiaomi_checkpoint_revision": XIAOMI_CHECKPOINT_REVISION,
            "num_tasks": OFFICIAL_NUM_TASKS,
            "num_episodes": OFFICIAL_NUM_EPISODES,
            "successes": OFFICIAL_SUCCESSES,
        },
        "selection": {
            "minimum_official_successes_inclusive": min_successes,
            "maximum_official_successes_inclusive": max_successes,
            "official_episodes_per_task": OFFICIAL_EPISODES_PER_TASK,
        },
        "collection": {
            "rollouts_per_task": rollouts_per_task,
            "keep_all_rollouts": True,
            "subtask_safe_enabled": False,
            "eligible_task_count": len(eligible),
            "excluded_task_count": len(excluded),
            "expected_rollouts": expected_rollouts,
            "estimated_gib_per_rollout": estimated_gib_per_rollout,
            "estimated_storage_gib": estimated_storage_gib,
            "storage_reserve_factor": storage_reserve_factor,
            "recommended_free_storage_gib": estimated_storage_gib
            * storage_reserve_factor,
        },
        "eligible_tasks": sorted(
            eligible, key=lambda item: (item["official_successes"], item["task"])
        ),
        "excluded_tasks": sorted(
            excluded, key=lambda item: (item["official_successes"], item["task"])
        ),
        "shards": shards,
    }


def _write_plan_artifacts(plan: dict, output_dir: Path, latest_env: Path | None):
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "collection_plan.json"
    _atomic_write_text(plan_path, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    task_files = []
    for shard in plan["shards"]:
        task_file = output_dir / f"shard{shard['shard_index']}_tasks.txt"
        _atomic_write_text(task_file, "\n".join(shard["tasks"]) + "\n")
        task_files.append(task_file)

    env_lines = [
        f"export XR1_COLLECTION_ROOT={shlex.quote(str(output_dir.resolve()))}",
        f"export XR1_COLLECTION_PLAN={shlex.quote(str(plan_path.resolve()))}",
        f"export XR1_COLLECTION_ROLLOUTS_PER_TASK={plan['collection']['rollouts_per_task']}",
    ]
    for shard, task_file in zip(plan["shards"], task_files):
        index = shard["shard_index"]
        env_lines.extend(
            [
                f"export XR1_SHARD{index}_DIR={shlex.quote(str((output_dir / f'shard{index}').resolve()))}",
                f"export XR1_SHARD{index}_TASK_FILE={shlex.quote(str(task_file.resolve()))}",
            ]
        )
    env_text = "\n".join(env_lines) + "\n"
    env_path = _atomic_write_text(output_dir / "collection.env", env_text)
    if latest_env is not None:
        _atomic_write_text(latest_env, env_text)
    return plan_path, env_path, task_files


def _print_plan(plan: dict, output_dir: Path, available_gib: float):
    collection = plan["collection"]
    recommended = collection["recommended_free_storage_gib"]
    print("COLLECTION PLAN: VALID")
    print(f"Eligible tasks: {collection['eligible_task_count']}")
    print(f"Excluded tasks: {collection['excluded_task_count']}")
    print(f"Expected new rollouts: {collection['expected_rollouts']}")
    print(f"Estimated storage: {collection['estimated_storage_gib']:.1f} GiB")
    print(f"Recommended free storage: {recommended:.1f} GiB")
    print(f"Currently available: {available_gib:.1f} GiB")
    print("\nELIGIBLE TASKS")
    print(f"{'TASK':32s} {'TYPE':10s} {'SUCCESS':>8s} {'HORIZON':>8s}")
    for task in plan["eligible_tasks"]:
        success = f"{task['official_successes']}/{task['official_num_episodes']}"
        print(
            f"{task['task']:32s} {task['task_type']:10s} "
            f"{success:>8s} {task['horizon']:8d}"
        )
    print("\nSHARDS")
    for shard in plan["shards"]:
        print(
            f"shard{shard['shard_index']}: tasks={shard['task_count']} "
            f"horizon_units={shard['horizon_units']}"
        )
        print("  " + " ".join(shard["tasks"]))
    print("\nEXCLUDED")
    for task in plan["excluded_tasks"]:
        success = f"{task['official_successes']}/{task['official_num_episodes']}"
        print(f"{task['task']:32s} {success:>8s} {task['exclusion_reason']}")
    status = "PASS" if available_gib >= recommended else "FAIL"
    print(f"\nSTORAGE PREFLIGHT: {status}")
    print(f"\nSaved plan: {output_dir / 'collection_plan.json'}")
    print(f"Saved environment: {output_dir / 'collection.env'}")
    if status == "FAIL":
        raise RuntimeError(
            f"Only {available_gib:.1f} GiB available; {recommended:.1f} GiB recommended"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-env", type=Path)
    parser.add_argument("--min-successes", type=int, default=5)
    parser.add_argument("--max-successes", type=int, default=45)
    parser.add_argument("--rollouts-per-task", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--estimated-gib-per-rollout", type=float, default=0.04)
    parser.add_argument("--storage-reserve-factor", type=float, default=1.25)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = load_official_summary(args.official_summary)
    plan = build_plan(
        summary,
        summary_path=args.official_summary,
        min_successes=args.min_successes,
        max_successes=args.max_successes,
        rollouts_per_task=args.rollouts_per_task,
        num_shards=args.num_shards,
        estimated_gib_per_rollout=args.estimated_gib_per_rollout,
        storage_reserve_factor=args.storage_reserve_factor,
    )
    plan_path, _, _ = _write_plan_artifacts(plan, args.output_dir, args.latest_env)
    available_gib = shutil.disk_usage(plan_path.parent).free / (1024**3)
    _print_plan(plan, args.output_dir, available_gib)


if __name__ == "__main__":
    main()
