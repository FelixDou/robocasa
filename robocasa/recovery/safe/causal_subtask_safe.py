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
CAUSAL_LABEL_MODES = ("eventual", "within_horizon")
TEMPORAL_REPRESENTATIONS = (
    "raw",
    "delta",
    "anchor_delta",
    "raw_delta",
    "raw_delta_mean_slope",
)
DEFAULT_PREFIX_HORIZONS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class CausalPrefixConfig:
    training_mode: str = "none"
    horizons: tuple[int, ...] = DEFAULT_PREFIX_HORIZONS
    random_prefixes_per_segment: int = 3
    conditioning: str = "none"
    label_mode: str = "eventual"
    failure_horizon: int | None = None
    temporal_representation: str = "raw"
    temporal_window: int = 4
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


def validate_causal_config(config):
    if config.training_mode not in PREFIX_TRAINING_MODES:
        raise ValueError(
            f"Unknown causal prefix training mode {config.training_mode!r}"
        )
    if config.label_mode not in CAUSAL_LABEL_MODES:
        raise ValueError(f"Unknown causal label mode {config.label_mode!r}")
    if config.temporal_representation not in TEMPORAL_REPRESENTATIONS:
        raise ValueError(
            f"Unknown temporal representation {config.temporal_representation!r}"
        )
    if int(config.temporal_window) <= 0:
        raise ValueError("Temporal representation window must be positive")
    validate_prefix_horizons(config.horizons)
    if config.label_mode == "within_horizon":
        if not config.enabled:
            raise ValueError(
                "within_horizon labels require causal prefix training, not mode='none'"
            )
        if config.failure_horizon is None or int(config.failure_horizon) <= 0:
            raise ValueError("within_horizon labels require a positive failure_horizon")
    elif config.failure_horizon is not None:
        raise ValueError(
            "failure_horizon is only meaningful with label_mode='within_horizon'"
        )
    return config


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
    env["source_segment_num_inferences"] = int(len(rollout.hidden_states))
    env["source_segment_failed"] = not bool(int(rollout.episode_success))
    env["rollout_id"] = f"{source_id}::prefix-{int(length):04d}"
    env["causal_prefix_inferences"] = int(length)
    steps = list(env.get("inference_environment_steps") or [])
    if steps:
        env["inference_environment_steps"] = steps[:length]
    return clone, env


def apply_causal_target(rollout, env, *, label_mode, failure_horizon):
    """Assign the causal prefix target while retaining the source outcome.

    SAFE represents success as one and failure as zero. Under ``within_horizon``,
    an early prefix of an eventually failed segment is a valid negative when the
    failure is observed more than ``failure_horizon`` inference calls later.
    """
    if label_mode not in CAUSAL_LABEL_MODES:
        raise ValueError(f"Unknown causal label mode {label_mode!r}")
    source_failed = bool(env["source_segment_failed"])
    prefix = int(env["causal_prefix_inferences"])
    source_length = int(env["source_segment_num_inferences"])
    remaining = source_length - prefix
    if remaining < 0:
        raise ValueError("Causal prefix extends beyond its source segment")
    if label_mode == "eventual":
        target_failed = source_failed
        horizon = None
    else:
        if failure_horizon is None or int(failure_horizon) <= 0:
            raise ValueError("within_horizon labels require a positive failure_horizon")
        horizon = int(failure_horizon)
        target_failed = source_failed and remaining <= horizon
    rollout.episode_success = int(not target_failed)
    env["causal_label_mode"] = label_mode
    env["causal_failure_horizon_inferences"] = horizon
    env["causal_target_failed"] = bool(target_failed)
    env["remaining_inferences_to_terminal"] = int(remaining)
    return bool(target_failed)


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
    label_mode="eventual",
    failure_horizon=None,
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
            apply_causal_target(
                clone,
                prefix_env,
                label_mode=label_mode,
                failure_horizon=failure_horizon,
            )
            expanded.append(clone)
            expanded_identity[id(clone)] = (path, prefix_env)
            per_horizon[int(length)] += 1
    if not expanded:
        raise ValueError("Causal prefix expansion produced no examples")
    return expanded, expanded_identity, dict(sorted(per_horizon.items()))


def _numpy_temporal_components(values, window):
    values = np.asarray(values)
    delta = np.zeros_like(values)
    delta[1:] = values[1:] - values[:-1]
    anchor_delta = values - values[:1]
    means = np.empty_like(values)
    slopes = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        means[index] = chunk.mean(axis=0)
        span = index - start
        slopes[index] = 0 if span == 0 else (values[index] - values[start]) / span
    return delta, anchor_delta, means, slopes


def _torch_temporal_components(values, window):
    import torch

    delta = torch.zeros_like(values)
    delta[1:] = values[1:] - values[:-1]
    anchor_delta = values - values[:1]
    means = []
    slopes = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        means.append(values[start : index + 1].mean(dim=0))
        span = index - start
        slopes.append(
            torch.zeros_like(values[index])
            if span == 0
            else (values[index] - values[start]) / span
        )
    return delta, anchor_delta, torch.stack(means), torch.stack(slopes)


def transform_temporal_representation(rollouts, *, mode="raw", window=4):
    """Apply a causal, sequence-length-preserving feature transformation."""
    if mode not in TEMPORAL_REPRESENTATIONS:
        raise ValueError(f"Unknown temporal representation {mode!r}")
    if int(window) <= 0:
        raise ValueError("Temporal representation window must be positive")
    source_dimension = None
    output_dimension = None
    for rollout in rollouts:
        values = rollout.hidden_states
        if len(values) == 0:
            raise ValueError("Temporal representation received an empty sequence")
        dimension = int(values.shape[-1])
        if source_dimension is None:
            source_dimension = dimension
        elif dimension != source_dimension:
            raise ValueError("Temporal representation mixes feature dimensions")
        if mode == "raw":
            transformed = values
        else:
            module = type(values).__module__
            components = (
                _torch_temporal_components(values, int(window))
                if module.startswith("torch")
                else _numpy_temporal_components(values, int(window))
            )
            delta, anchor_delta, mean, slope = components
            concatenate = (
                __import__("torch").cat
                if module.startswith("torch")
                else np.concatenate
            )
            if mode == "delta":
                transformed = delta
            elif mode == "anchor_delta":
                transformed = anchor_delta
            elif mode == "raw_delta":
                transformed = (
                    concatenate((values, delta), dim=-1)
                    if module.startswith("torch")
                    else concatenate((values, delta), axis=-1)
                )
            else:
                transformed = (
                    concatenate((values, delta, mean, slope), dim=-1)
                    if module.startswith("torch")
                    else concatenate((values, delta, mean, slope), axis=-1)
                )
        rollout.hidden_states = transformed
        output_dimension = int(transformed.shape[-1])
    return {
        "mode": mode,
        "window": int(window),
        "source_dimension": source_dimension,
        "output_dimension": output_dimension,
        "causal": True,
        "sequence_length_preserved": True,
    }


def causal_target_counts(rollouts, identity):
    grouped = defaultdict(lambda: {"successes": 0, "failures": 0})
    for rollout in rollouts:
        env = identity[id(rollout)][1]
        key = int(env["causal_prefix_inferences"])
        outcome = "successes" if int(rollout.episode_success) else "failures"
        grouped[key][outcome] += 1
    return {str(key): value for key, value in sorted(grouped.items())}


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
    validate_causal_config(config)
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
        label_mode=config.label_mode,
        failure_horizon=config.failure_horizon,
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
        label_mode=config.label_mode,
        failure_horizon=config.failure_horizon,
    )
    for split_name, values in (("training", train), ("test", test)):
        labels = {int(item.episode_success) for item in values}
        if labels != {0, 1}:
            raise ValueError(
                f"Causal {split_name} prefixes lack both target labels under "
                f"label_mode={config.label_mode!r}, "
                f"failure_horizon={config.failure_horizon!r}; expand later prefixes "
                "or increase the prediction horizon"
            )
    combined_identity = {**train_identity, **test_identity}
    representation = transform_temporal_representation(
        train + test,
        mode=config.temporal_representation,
        window=config.temporal_window,
    )
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
        "temporal_representation": representation,
        "prefix_counts": {"train": train_counts, "test": test_counts},
        "target_counts_by_prefix": {
            "train": causal_target_counts(train, combined_identity),
            "test": causal_target_counts(test, combined_identity),
        },
        "protocol": {
            "training_mode": config.training_mode,
            "test_mode": "fixed" if config.enabled else "none",
            "horizons": list(config.horizons),
            "random_prefixes_per_segment": config.random_prefixes_per_segment,
            "label_mode": config.label_mode,
            "failure_horizon": config.failure_horizon,
            "temporal_representation": config.temporal_representation,
            "temporal_window": config.temporal_window,
            "split_before_prefix_expansion": True,
            "support_selected_from_training_only": True,
        },
    }
