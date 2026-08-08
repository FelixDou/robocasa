import json
from pathlib import Path

import pytest

from robocasa.recovery.safe.plan_xiaomi_official_collection import (
    OFFICIAL_EPISODES_PER_TASK,
    OFFICIAL_NUM_EPISODES,
    OFFICIAL_NUM_TASKS,
    OFFICIAL_SUCCESSES,
    balance_shards,
    build_plan,
    select_tasks,
    validate_official_summary,
)


def _registry_text(tasks):
    atomic = tasks[:18]
    composite = tasks[18:]

    def calls(names, offset):
        return ",\n".join(
            f"    {name}=dict(horizon={450 + 150 * (offset + index)})"
            for index, name in enumerate(names)
        )

    return f"""
ATOMIC_TASK_DATASETS = dict(
{calls(atomic, 0)}
)
COMPOSITE_TASK_DATASETS = dict(
{calls(composite, len(atomic))}
)
TARGET_TASKS = dict(
    atomic_seen={atomic!r},
    composite_seen={composite[:16]!r},
    composite_unseen={composite[16:]!r},
)
"""


def _summary_and_registry(tmp_path):
    tasks = [f"Task{index:02d}" for index in range(OFFICIAL_NUM_TASKS)]
    registry = tmp_path / "dataset_registry.py"
    registry.write_text(_registry_text(tasks))
    horizons = {task: 450 + 150 * index for index, task in enumerate(tasks)}
    successes = [5, 45, 4, 46] + [29] * 45 + [27]
    assert sum(successes) == OFFICIAL_SUCCESSES
    global_index = 0
    payload = {}
    for task, task_successes in zip(tasks, successes):
        episodes = []
        for episode in range(OFFICIAL_EPISODES_PER_TASK):
            episodes.append(
                {
                    "episode": episode,
                    "global_episode_index": global_index,
                    "success": episode < task_successes,
                }
            )
            global_index += 1
        payload[task] = {
            "num_episodes": OFFICIAL_EPISODES_PER_TASK,
            "successes": task_successes,
            "horizon": horizons[task],
            "episodes": episodes,
        }
    summary = {
        "num_tasks": OFFICIAL_NUM_TASKS,
        "num_episodes": OFFICIAL_NUM_EPISODES,
        "successes": OFFICIAL_SUCCESSES,
        "split": "pretrain",
        "task_set": "target50",
        "tasks": payload,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    return summary, path, registry


def test_official_interval_is_inclusive_and_produces_balanced_plan(tmp_path):
    summary, summary_path, registry = _summary_and_registry(tmp_path)
    plan = build_plan(
        summary,
        summary_path=summary_path,
        min_successes=5,
        max_successes=45,
        rollouts_per_task=50,
        num_shards=2,
        estimated_gib_per_rollout=0.04,
        storage_reserve_factor=1.25,
        registry_path=registry,
    )

    assert plan["collection"]["eligible_task_count"] == 48
    assert plan["collection"]["excluded_task_count"] == 2
    assert plan["collection"]["expected_rollouts"] == 2400
    assert plan["collection"]["estimated_storage_gib"] == pytest.approx(96.0)
    assert {item["task"] for item in plan["eligible_tasks"]} >= {"Task00", "Task01"}
    assert {item["task"] for item in plan["excluded_tasks"]} == {"Task02", "Task03"}
    assert sum(shard["task_count"] for shard in plan["shards"]) == 48
    assert sum(shard["expected_rollouts"] for shard in plan["shards"]) == 2400
    horizon_difference = abs(
        plan["shards"][0]["horizon_units"]
        - plan["shards"][1]["horizon_units"]
    )
    assert horizon_difference <= max(item["horizon"] for item in plan["eligible_tasks"])


def test_summary_validation_rejects_wrong_release_total(tmp_path):
    summary, _, registry = _summary_and_registry(tmp_path)
    summary["successes"] -= 1
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_official_summary(summary, registry_path=registry)


def test_select_tasks_rejects_invalid_interval():
    with pytest.raises(ValueError, match="0 <= min <= max"):
        select_tasks([], min_successes=46, max_successes=45)


def test_balance_shards_rejects_nonpositive_rollout_count():
    with pytest.raises(ValueError, match="rollouts-per-task"):
        balance_shards([], num_shards=2, rollouts_per_task=0)
