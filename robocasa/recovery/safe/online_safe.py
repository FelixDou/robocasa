"""Causal rollout-prefix views for online, original SAFE failure detection.

This module deliberately operates on the original binary rollout outcome.  It
does not read or create Subtask-SAFE labels.  Callers must split source
rollouts before invoking :func:`prepare_online_splits`.
"""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict, dataclass
import math
import random

import numpy as np


ONLINE_SAFE_MODES = ("none", "matched_success_length", "fixed_landmark")


@dataclass(frozen=True)
class OnlineSafeConfig:
    mode: str = "none"
    seed: int = 0
    landmark_fraction: float = 0.5


def validate_online_config(config):
    if config.mode not in ONLINE_SAFE_MODES:
        raise ValueError(f"Unknown online SAFE mode {config.mode!r}")
    if not 0.0 < float(config.landmark_fraction) <= 1.0:
        raise ValueError("online SAFE landmark fraction must lie in (0, 1]")
    return config


def online_config_dict(config):
    validate_online_config(config)
    return asdict(config)


def _slice_sequence(value, length):
    sliced = value[:length]
    clone = getattr(sliced, "clone", None)
    return clone() if callable(clone) else np.array(sliced, copy=True)


def _clone_prefix(
    rollout,
    env_record,
    length,
    *,
    mode,
    partner_rollout_id=None,
    task_timeout=None,
    landmark_fraction=None,
):
    source_length = len(rollout.hidden_states)
    if int(length) < 1 or int(length) > source_length:
        raise ValueError(
            f"Online SAFE prefix {length} is invalid for source length {source_length}"
        )
    clone = copy.copy(rollout)
    clone.hidden_states = _slice_sequence(rollout.hidden_states, int(length))
    action_vectors = getattr(rollout, "action_vectors", None)
    if action_vectors is not None:
        if len(action_vectors) != source_length:
            raise ValueError(
                "SAFE action vectors are not aligned with hidden states: "
                f"{len(action_vectors)} != {source_length}"
            )
        clone.action_vectors = _slice_sequence(action_vectors, int(length))
    clone.task_min_step = int(length)

    env = copy.deepcopy(env_record)
    source_id = str(env["rollout_id"])
    env["online_safe_mode"] = str(mode)
    env["online_source_rollout_id"] = source_id
    env["online_source_num_inferences"] = int(source_length)
    env["online_prefix_inferences"] = int(length)
    env["online_partner_rollout_id"] = partner_rollout_id
    env["online_task_timeout_inferences"] = (
        None if task_timeout is None else int(task_timeout)
    )
    env["online_landmark_fraction"] = (
        None if landmark_fraction is None else float(landmark_fraction)
    )
    env["model_infer_times"] = int(length)
    steps = list(env.get("inference_environment_steps") or [])
    if steps:
        env["inference_environment_steps"] = steps[: int(length)]
    return clone, env


def _task_outcomes(rollouts):
    grouped = defaultdict(lambda: {"success": [], "failure": []})
    for rollout in rollouts:
        outcome = "success" if bool(int(rollout.episode_success)) else "failure"
        grouped[int(rollout.task_id)][outcome].append(rollout)
    return grouped


def _ordered_for_matching(values, identity, rng):
    randomized = list(values)
    rng.shuffle(randomized)
    return sorted(randomized, key=lambda item: len(item.hidden_states), reverse=True)


def match_success_lengths(rollouts, identity, *, seed=0, split_name="split"):
    """Pair outcomes within task and truncate every failure to its success length.

    The two outcome classes must have equal support in every task.  Sorting by
    descending source length makes the assignment feasible whenever the
    ordered failure lengths dominate the ordered success lengths.  The seed is
    used only to break equal-length ties reproducibly.
    """
    grouped = _task_outcomes(rollouts)
    transformed = []
    transformed_identity = {}
    pairs = []
    for task_id in sorted(grouped):
        successes = grouped[task_id]["success"]
        failures = grouped[task_id]["failure"]
        if len(successes) != len(failures) or not successes:
            raise ValueError(
                f"Task {task_id} {split_name} needs equal nonzero success/failure "
                f"support for length matching, got {len(successes)}/{len(failures)}"
            )
        rng = random.Random(f"{int(seed)}:{split_name}:{task_id}")
        successes = _ordered_for_matching(successes, identity, rng)
        failures = _ordered_for_matching(failures, identity, rng)
        for success, failure in zip(successes, failures):
            success_path, success_env = identity[id(success)]
            failure_path, failure_env = identity[id(failure)]
            success_id = str(success_env["rollout_id"])
            failure_id = str(failure_env["rollout_id"])
            length = len(success.hidden_states)
            if len(failure.hidden_states) < length:
                raise ValueError(
                    f"Task {task_id} cannot match success {success_id} length {length} "
                    f"to failure {failure_id} length {len(failure.hidden_states)}"
                )
            success_clone, success_clone_env = _clone_prefix(
                success,
                success_env,
                length,
                mode="matched_success_length",
                partner_rollout_id=failure_id,
            )
            failure_clone, failure_clone_env = _clone_prefix(
                failure,
                failure_env,
                length,
                mode="matched_success_length",
                partner_rollout_id=success_id,
            )
            for clone, path, env in (
                (success_clone, success_path, success_clone_env),
                (failure_clone, failure_path, failure_clone_env),
            ):
                transformed.append(clone)
                transformed_identity[id(clone)] = (path, env)
            pairs.append(
                {
                    "task_id": int(task_id),
                    "success_rollout_id": success_id,
                    "failure_rollout_id": failure_id,
                    "matched_inferences": int(length),
                }
            )
    return transformed, transformed_identity, pairs


def training_task_timeouts(train_rollouts):
    """Estimate task timeout lengths from failures in the training split only."""
    grouped = defaultdict(list)
    for rollout in train_rollouts:
        if not bool(int(rollout.episode_success)):
            grouped[int(rollout.task_id)].append(len(rollout.hidden_states))
    tasks = sorted({int(item.task_id) for item in train_rollouts})
    missing = [task_id for task_id in tasks if not grouped[task_id]]
    if missing:
        raise ValueError(f"Training split has no failures for tasks {missing}")
    return {task_id: max(grouped[task_id]) for task_id in tasks}


def landmark_prefixes(
    rollouts,
    identity,
    *,
    task_timeouts,
    landmark_fraction,
    split_name="split",
):
    """Create one fixed, causal at-risk prefix per eligible source rollout."""
    transformed = []
    transformed_identity = {}
    excluded = []
    cutoffs = {
        int(task_id): max(1, int(math.ceil(float(timeout) * landmark_fraction)))
        for task_id, timeout in task_timeouts.items()
    }
    for rollout in rollouts:
        task_id = int(rollout.task_id)
        if task_id not in cutoffs:
            raise ValueError(f"Task {task_id} is absent from training-derived timeouts")
        length = cutoffs[task_id]
        path, env = identity[id(rollout)]
        if len(rollout.hidden_states) < length:
            excluded.append(
                {
                    "rollout_id": str(env["rollout_id"]),
                    "task_id": task_id,
                    "episode_success": int(rollout.episode_success),
                    "source_inferences": len(rollout.hidden_states),
                    "required_inferences": int(length),
                    "reason": "terminal_success_before_landmark",
                }
            )
            continue
        clone, clone_env = _clone_prefix(
            rollout,
            env,
            length,
            mode="fixed_landmark",
            task_timeout=task_timeouts[task_id],
            landmark_fraction=landmark_fraction,
        )
        transformed.append(clone)
        transformed_identity[id(clone)] = (path, clone_env)

    grouped = _task_outcomes(transformed)
    invalid = {
        task_id: {
            "successes": len(values["success"]),
            "failures": len(values["failure"]),
        }
        for task_id, values in grouped.items()
        if not values["success"] or not values["failure"]
    }
    missing_tasks = sorted(set(cutoffs) - set(grouped))
    if invalid or missing_tasks:
        raise ValueError(
            f"Fixed landmark {landmark_fraction:g} leaves {split_name} without both "
            f"outcomes: invalid={invalid}, missing_tasks={missing_tasks}"
        )
    return transformed, transformed_identity, excluded, cutoffs


def balance_task_outcomes(rollouts, identity, *, seed=0, split_name="train"):
    """Downsample each task to equal outcome counts after at-risk filtering."""
    grouped = _task_outcomes(rollouts)
    selected = []
    excluded = []
    for task_id in sorted(grouped):
        successes = list(grouped[task_id]["success"])
        failures = list(grouped[task_id]["failure"])
        target = min(len(successes), len(failures))
        if target < 1:
            raise ValueError(
                f"Task {task_id} {split_name} cannot be balanced after landmark filtering"
            )
        rng = random.Random(f"{int(seed)}:{split_name}:{task_id}:balance")
        rng.shuffle(successes)
        rng.shuffle(failures)
        keep_ids = {id(item) for item in successes[:target] + failures[:target]}
        selected.extend(item for item in successes + failures if id(item) in keep_ids)
        for item in successes + failures:
            if id(item) not in keep_ids:
                excluded.append(
                    {
                        "rollout_id": str(identity[id(item)][1]["rollout_id"]),
                        "task_id": int(task_id),
                        "episode_success": int(item.episode_success),
                        "reason": "training_task_outcome_balance",
                    }
                )
    selected.sort(
        key=lambda item: (
            int(item.task_id),
            str(identity[id(item)][1]["rollout_id"]),
        )
    )
    return selected, excluded


def _rank_auc(labels, values):
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    positive = values[labels == 1]
    negative = values[labels == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size
    )


def duration_audit(rollouts):
    groups = {"overall": list(rollouts)}
    for task_id in sorted({int(item.task_id) for item in rollouts}):
        groups[str(task_id)] = [
            item for item in rollouts if int(item.task_id) == task_id
        ]
    result = {}
    for name, values in groups.items():
        labels = [not bool(int(item.episode_success)) for item in values]
        lengths = [len(item.hidden_states) for item in values]
        result[name] = {
            "rollouts": len(values),
            "successes": sum(not value for value in labels),
            "failures": sum(labels),
            "min_inferences": min(lengths),
            "max_inferences": max(lengths),
            "duration_roc_auc": _rank_auc(labels, lengths),
        }
    return result


def prepare_online_splits(train_rollouts, test_rollouts, identity, *, config):
    """Transform an already disjoint source split into an online SAFE view."""
    validate_online_config(config)
    source_train_ids = {
        str(identity[id(item)][1]["rollout_id"]) for item in train_rollouts
    }
    source_test_ids = {
        str(identity[id(item)][1]["rollout_id"]) for item in test_rollouts
    }
    if source_train_ids & source_test_ids:
        raise ValueError("Online SAFE source train/test rollout IDs overlap")
    if config.mode == "none":
        return {
            "train": train_rollouts,
            "test": test_rollouts,
            "identity": identity,
            "protocol": online_config_dict(config),
            "source_counts": {"train": len(train_rollouts), "test": len(test_rollouts)},
            "counts": {"train": len(train_rollouts), "test": len(test_rollouts)},
            "duration_audit": {
                "train": duration_audit(train_rollouts),
                "test": duration_audit(test_rollouts),
            },
        }
    if config.mode == "matched_success_length":
        train, train_identity, train_pairs = match_success_lengths(
            train_rollouts,
            identity,
            seed=config.seed,
            split_name="train",
        )
        test, test_identity, test_pairs = match_success_lengths(
            test_rollouts,
            identity,
            seed=config.seed,
            split_name="test",
        )
        details = {"pairs": {"train": train_pairs, "test": test_pairs}}
    else:
        timeouts = training_task_timeouts(train_rollouts)
        train, train_identity, train_excluded, cutoffs = landmark_prefixes(
            train_rollouts,
            identity,
            task_timeouts=timeouts,
            landmark_fraction=config.landmark_fraction,
            split_name="train",
        )
        train, train_balance_excluded = balance_task_outcomes(
            train,
            train_identity,
            seed=config.seed,
            split_name="train",
        )
        test, test_identity, test_excluded, _ = landmark_prefixes(
            test_rollouts,
            identity,
            task_timeouts=timeouts,
            landmark_fraction=config.landmark_fraction,
            split_name="test",
        )
        details = {
            "training_task_timeouts": timeouts,
            "task_landmark_cutoffs": cutoffs,
            "excluded_before_landmark": {
                "train": train_excluded,
                "test": test_excluded,
            },
            "excluded_for_training_balance": train_balance_excluded,
            "training_balance": "equal success/failure counts within each task",
        }
    combined_identity = {**train_identity, **test_identity}
    output_train_ids = {
        str(combined_identity[id(item)][1]["online_source_rollout_id"])
        for item in train
    }
    output_test_ids = {
        str(combined_identity[id(item)][1]["online_source_rollout_id"]) for item in test
    }
    if output_train_ids & output_test_ids:
        raise AssertionError("Online SAFE transformation leaked source parents")
    protocol = {
        **online_config_dict(config),
        "binary_final_rollout_labels_only": True,
        "subtask_safe": False,
        "split_before_transformation": True,
        "source_parents_disjoint": True,
        **details,
    }
    return {
        "train": train,
        "test": test,
        "identity": combined_identity,
        "protocol": protocol,
        "source_counts": {"train": len(train_rollouts), "test": len(test_rollouts)},
        "counts": {"train": len(train), "test": len(test)},
        "duration_audit": {
            "train": duration_audit(train),
            "test": duration_audit(test),
        },
    }


def online_payload_summary(payload):
    """Return the strict-JSON protocol/audit subset of a prepared payload."""
    if payload is None:
        return None
    return {
        key: payload[key]
        for key in ("protocol", "source_counts", "counts", "duration_audit")
    }
