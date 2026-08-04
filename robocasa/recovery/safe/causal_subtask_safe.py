"""Causal-prefix and stage-conditioning utilities for Subtask-SAFE v2."""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass
import math
import random

import numpy as np


PREFIX_TRAINING_MODES = ("none", "fixed", "random")
CONDITIONING_MODES = ("none", "subtask_one_hot", "subtask_one_hot_elapsed")
DEFAULT_PREFIX_HORIZONS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class CausalPrefixConfig:
    training_mode: str = "none"
    horizons: tuple[int, ...] = DEFAULT_PREFIX_HORIZONS
    random_prefixes_per_segment: int = 3
    conditioning: str = "none"
    min_stage_successes: int = 0
    min_stage_failures: int = 0

    @property
    def enabled(self):
        return self.training_mode != "none"


def validate_prefix_horizons(values):
    horizons = tuple(sorted({int(value) for value in values}))
    if not horizons or horizons[0] <= 0:
        raise ValueError("Causal prefix horizons must be positive integers")
    return horizons


def stage_name(env):
    """Return the official export identity for one semantic task/subtask."""
    value = str(env.get("task_name", ""))
    if not value:
        parent = str(env.get("parent_task_name", ""))
        subtask = str(env.get("subtask_id", ""))
        value = f"{parent}::{subtask}" if parent and subtask else ""
    if "::" not in value:
        raise ValueError(
            "Causal Subtask-SAFE requires semantic task names of the form "
            "ParentTask::subtask_id"
        )
    return value


def count_stage_support(rollouts, identity):
    grouped = defaultdict(lambda: {"successes": 0, "failures": 0, "parents": set()})
    for rollout in rollouts:
        env = identity[id(rollout)][1]
        name = stage_name(env)
        failed = not bool(int(rollout.episode_success))
        grouped[name]["failures" if failed else "successes"] += 1
        parent = str(env.get("parent_rollout_id", ""))
        if parent:
            grouped[name]["parents"].add(parent)
    return {
        name: {
            "successes": values["successes"],
            "failures": values["failures"],
            "segments": values["successes"] + values["failures"],
            "parents": len(values["parents"]),
        }
        for name, values in sorted(grouped.items())
    }


def select_supported_stages(
    train_rollouts,
    identity,
    *,
    min_successes=0,
    min_failures=0,
    requested_stages=None,
):
    """Select stages from training support only, never from the held-out split."""
    support = count_stage_support(train_rollouts, identity)
    requested = (
        None if requested_stages is None else {str(value) for value in requested_stages}
    )
    unknown = [] if requested is None else sorted(requested - set(support))
    if unknown:
        raise ValueError(
            "Requested stages are absent from training: " + ", ".join(unknown)
        )
    selected = []
    excluded = {}
    for name, counts in support.items():
        reasons = []
        if requested is not None and name not in requested:
            reasons.append("not_requested")
        if counts["successes"] < int(min_successes):
            reasons.append("insufficient_successes")
        if counts["failures"] < int(min_failures):
            reasons.append("insufficient_failures")
        if reasons:
            excluded[name] = reasons
        else:
            selected.append(name)
    if not selected:
        raise ValueError(
            "No semantic stage satisfies the training-only support thresholds "
            f"({min_successes} successes, {min_failures} failures)"
        )
    return {
        "source": "outer training split only",
        "min_successes": int(min_successes),
        "min_failures": int(min_failures),
        "requested_stages": None if requested is None else sorted(requested),
        "selected_stages": selected,
        "excluded_stages": excluded,
        "training_support": support,
    }


def filter_supported_stages(rollouts, identity, selected_stages):
    selected = set(selected_stages)
    return [
        rollout
        for rollout in rollouts
        if stage_name(identity[id(rollout)][1]) in selected
    ]


def _slice_sequence(value, length):
    sliced = value[:length]
    clone = getattr(sliced, "clone", None)
    return clone() if callable(clone) else np.array(sliced, copy=True)


def clone_prefix_rollout(rollout, env_record, length):
    """Create a shallow rollout clone with a causally truncated feature sequence."""
    if length <= 0 or length > len(rollout.hidden_states):
        raise ValueError(
            f"Prefix {length} is invalid for a {len(rollout.hidden_states)}-step segment"
        )
    clone = copy.copy(rollout)
    clone.hidden_states = _slice_sequence(rollout.hidden_states, length)
    action_vectors = getattr(rollout, "action_vectors", None)
    if action_vectors is not None:
        if len(action_vectors) != len(rollout.hidden_states):
            raise ValueError(
                "SAFE rollout action vectors are not aligned with hidden states: "
                f"{len(action_vectors)} != {len(rollout.hidden_states)}"
            )
        clone.action_vectors = _slice_sequence(action_vectors, length)
    clone.task_min_step = int(length)
    env = copy.deepcopy(env_record)
    source_id = str(env["rollout_id"])
    env["source_segment_id"] = source_id
    env["rollout_id"] = f"{source_id}::prefix-{int(length):04d}"
    env["causal_prefix_inferences"] = int(length)
    steps = list(env.get("inference_environment_steps") or [])
    if steps:
        env["inference_environment_steps"] = steps[:length]
    return clone, env


def prefix_lengths_for_rollout(
    length,
    *,
    mode,
    horizons=DEFAULT_PREFIX_HORIZONS,
    random_prefixes_per_segment=3,
    random_seed=0,
    identity_key="",
):
    if mode not in PREFIX_TRAINING_MODES:
        raise ValueError(f"Unknown causal prefix training mode {mode!r}")
    if mode == "none":
        return (int(length),)
    horizons = validate_prefix_horizons(horizons)
    if mode == "fixed":
        return tuple(value for value in horizons if value <= int(length))
    maximum = min(max(horizons), int(length))
    count = min(int(random_prefixes_per_segment), maximum)
    if count <= 0:
        raise ValueError("random_prefixes_per_segment must be positive")
    rng = random.Random(f"{int(random_seed)}:{identity_key}")
    return tuple(sorted(rng.sample(range(1, maximum + 1), count)))


def expand_causal_prefixes(
    rollouts,
    identity,
    *,
    mode,
    horizons=DEFAULT_PREFIX_HORIZONS,
    random_prefixes_per_segment=3,
    random_seed=0,
):
    """Expand already-split semantic segments into causal prefix examples."""
    expanded = []
    expanded_identity = {}
    per_horizon = defaultdict(int)
    for rollout in rollouts:
        path, env = identity[id(rollout)]
        source_id = str(env["rollout_id"])
        lengths = prefix_lengths_for_rollout(
            len(rollout.hidden_states),
            mode=mode,
            horizons=horizons,
            random_prefixes_per_segment=random_prefixes_per_segment,
            random_seed=random_seed,
            identity_key=source_id,
        )
        if not lengths:
            continue
        for length in lengths:
            clone, prefix_env = clone_prefix_rollout(rollout, env, length)
            expanded.append(clone)
            expanded_identity[id(clone)] = (path, prefix_env)
            per_horizon[int(length)] += 1
    if not expanded:
        raise ValueError("Causal prefix expansion produced no examples")
    return expanded, expanded_identity, dict(sorted(per_horizon.items()))


def conditioning_catalog(train_rollouts, identity):
    names = sorted({stage_name(identity[id(rollout)][1]) for rollout in train_rollouts})
    return {name: index for index, name in enumerate(names)}


def _append_numpy_features(hidden_states, extra):
    return np.concatenate((np.asarray(hidden_states), extra), axis=-1)


def _append_torch_features(hidden_states, extra):
    import torch

    values = torch.as_tensor(
        extra, dtype=hidden_states.dtype, device=hidden_states.device
    )
    return torch.cat((hidden_states, values), dim=-1)


def append_causal_conditioning(
    rollouts,
    identity,
    *,
    mode,
    catalog,
    elapsed_scale,
):
    """Append causal stage one-hot and optional elapsed-progress features in place."""
    if mode not in CONDITIONING_MODES:
        raise ValueError(f"Unknown causal conditioning mode {mode!r}")
    if mode == "none":
        return {"mode": mode, "added_dimensions": 0, "elapsed_scale": None}
    if not catalog:
        raise ValueError("Stage conditioning requires a non-empty training catalog")
    if elapsed_scale <= 0:
        raise ValueError("Elapsed conditioning scale must be positive")
    include_elapsed = mode == "subtask_one_hot_elapsed"
    added = len(catalog) + int(include_elapsed)
    for rollout in rollouts:
        env = identity[id(rollout)][1]
        name = stage_name(env)
        if name not in catalog:
            raise ValueError(f"Stage {name!r} is absent from the training catalog")
        length = len(rollout.hidden_states)
        extra = np.zeros((length, added), dtype=np.float32)
        extra[:, catalog[name]] = 1.0
        if include_elapsed:
            steps = np.arange(1, length + 1, dtype=np.float32)
            denominator = math.log1p(float(elapsed_scale))
            extra[:, -1] = np.clip(np.log1p(steps) / denominator, 0.0, 1.0)
        module = type(rollout.hidden_states).__module__
        if module.startswith("torch"):
            rollout.hidden_states = _append_torch_features(rollout.hidden_states, extra)
        else:
            rollout.hidden_states = _append_numpy_features(rollout.hidden_states, extra)
    return {
        "mode": mode,
        "added_dimensions": added,
        "num_stage_dimensions": len(catalog),
        "elapsed_dimension": include_elapsed,
        "elapsed_scale": int(elapsed_scale) if include_elapsed else None,
        "catalog": dict(sorted(catalog.items())),
    }


def prepare_causal_splits(
    train_rollouts,
    test_rollouts,
    identity,
    *,
    config,
    random_seed=0,
    requested_stages=None,
):
    """Filter, prefix-expand, and condition an already parent-disjoint split."""
    support = select_supported_stages(
        train_rollouts,
        identity,
        min_successes=config.min_stage_successes,
        min_failures=config.min_stage_failures,
        requested_stages=requested_stages,
    )
    selected = support["selected_stages"]
    train_source = filter_supported_stages(train_rollouts, identity, selected)
    test_source = filter_supported_stages(test_rollouts, identity, selected)
    if {int(item.episode_success) for item in train_source} != {0, 1}:
        raise ValueError("Selected causal training stages do not contain both labels")
    if {int(item.episode_success) for item in test_source} != {0, 1}:
        raise ValueError("Selected causal test stages do not contain both labels")
    train, train_identity, train_counts = expand_causal_prefixes(
        train_source,
        identity,
        mode=config.training_mode,
        horizons=config.horizons,
        random_prefixes_per_segment=config.random_prefixes_per_segment,
        random_seed=random_seed,
    )
    # Test is always expanded at declared fixed horizons, regardless of how
    # training prefixes were sampled.
    test, test_identity, test_counts = expand_causal_prefixes(
        test_source,
        identity,
        mode="fixed" if config.enabled else "none",
        horizons=config.horizons,
        random_prefixes_per_segment=config.random_prefixes_per_segment,
        random_seed=random_seed,
    )
    combined_identity = {**train_identity, **test_identity}
    catalog = conditioning_catalog(train, combined_identity)
    elapsed_scale = (
        max(config.horizons)
        if config.enabled
        else max(len(item.hidden_states) for item in train)
    )
    conditioning = append_causal_conditioning(
        train + test,
        combined_identity,
        mode=config.conditioning,
        catalog=catalog,
        elapsed_scale=elapsed_scale,
    )
    return {
        "train": train,
        "test": test,
        "identity": combined_identity,
        "source_train": train_source,
        "source_test": test_source,
        "support": support,
        "conditioning": conditioning,
        "prefix_counts": {"train": train_counts, "test": test_counts},
        "protocol": {
            "training_mode": config.training_mode,
            "test_mode": "fixed" if config.enabled else "none",
            "horizons": list(config.horizons),
            "random_prefixes_per_segment": config.random_prefixes_per_segment,
            "split_before_prefix_expansion": True,
            "support_selected_from_training_only": True,
        },
    }
