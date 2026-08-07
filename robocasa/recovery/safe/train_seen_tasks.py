"""Train official SAFE on a fixed per-task split and evaluate the remainder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import glob
import json
import math
from pathlib import Path
import pickle
import random
import re
import subprocess
import sys

import numpy as np

try:
    from .online_safe import (
        ONLINE_SAFE_MODES,
        OnlineSafeConfig,
        online_config_dict,
        online_payload_summary,
        prepare_online_splits,
    )
except ImportError:
    from online_safe import (
        ONLINE_SAFE_MODES,
        OnlineSafeConfig,
        online_config_dict,
        online_payload_summary,
        prepare_online_splits,
    )

try:
    from .causal_subtask_safe import (
        CAUSAL_LABEL_MODES,
        CONDITIONING_MODES,
        DEFAULT_PREFIX_HORIZONS,
        PREFIX_TRAINING_MODES,
        TEMPORAL_REPRESENTATIONS,
        CausalPrefixConfig,
        prepare_causal_splits,
        validate_prefix_horizons,
    )
except ImportError:
    from causal_subtask_safe import (
        CAUSAL_LABEL_MODES,
        CONDITIONING_MODES,
        DEFAULT_PREFIX_HORIZONS,
        PREFIX_TRAINING_MODES,
        TEMPORAL_REPRESENTATIONS,
        CausalPrefixConfig,
        prepare_causal_splits,
        validate_prefix_horizons,
    )


OFFICIAL_SAFE_COMMIT = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
TASK_TYPE_FILTERS = ("all", "atomic", "composite")
CLASS_WEIGHTING_MODES = ("official_inverse_frequency", "none")
LOSS_MODES = ("official", "bce", "focal")
MODEL_DEFAULTS = {
    "indep": {
        "horizon_selector": 1.0,
        "diffusion_selector": 0.0,
        "learning_rate": 3e-4,
        "lambda_reg": 1e-3,
    },
    "lstm": {
        "horizon_selector": 1.0,
        "diffusion_selector": "concat-2",
        "learning_rate": 1e-3,
        "lambda_reg": 1e-2,
    },
}


def training_objective_metadata(loss_mode, focal_gamma=2.0):
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"Unknown training loss mode {loss_mode!r}")
    if float(focal_gamma) < 0:
        raise ValueError("focal_gamma must be non-negative")
    return {
        "loss_mode": loss_mode,
        "focal_gamma": float(focal_gamma),
        "score_output": (
            "pinned_safe_output"
            if loss_mode == "official"
            else "instantaneous_failure_probability"
        ),
    }


def configure_training_objective(cfg, loss_mode):
    """Keep BCE/focal training and evaluation on the same probabilities."""
    if loss_mode != "official":
        cfg.model.cumsum = False
        cfg.model.rmean = False
    return cfg


def set_task_min_step_from_training(train_rollouts, *other_splits):
    """Freeze matched-horizon cutoffs from training data only."""
    task_cutoffs = {}
    for task_id in sorted({int(rollout.task_id) for rollout in train_rollouts}):
        lengths = [
            len(rollout.hidden_states)
            for rollout in train_rollouts
            if int(rollout.task_id) == task_id
        ]
        task_cutoffs[task_id] = min(lengths)
    for rollout in train_rollouts:
        rollout.task_min_step = task_cutoffs[int(rollout.task_id)]
    for split in other_splits:
        for rollout in split:
            task_id = int(rollout.task_id)
            if task_id not in task_cutoffs:
                raise ValueError(f"Task {task_id} is absent from training cutoffs")
            rollout.task_min_step = min(
                task_cutoffs[task_id], len(rollout.hidden_states)
            )
    return task_cutoffs


def resolve_hyperparameters(
    model_name,
    selection_summary=None,
    expected_task_type=None,
):
    if selection_summary is None:
        return dict(MODEL_DEFAULTS[model_name])
    summary = json.loads(Path(selection_summary).read_text())
    if expected_task_type is not None:
        selected_task_type = summary.get("task_type_filter", "all")
        if selected_task_type != expected_task_type:
            raise ValueError(
                "Selection summary task type "
                f"{selected_task_type!r} does not match {expected_task_type!r}"
            )
    selected = summary["best_by_model"][model_name]

    def selector(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    return {
        "horizon_selector": selector(selected["horizon_selector"]),
        "diffusion_selector": selector(selected["diffusion_selector"]),
        "learning_rate": float(selected["learning_rate"]),
        "lambda_reg": float(selected["lambda_reg"]),
    }


def validate_causal_selection_summary(selection_summary, config, requested_stages):
    if selection_summary is None:
        return
    summary = json.loads(Path(selection_summary).read_text())
    selected = summary.get("causal_subtask_safe")
    requested = None if requested_stages is None else sorted(requested_stages)
    current = None
    if config is not None:
        current = {
            "training_mode": config.training_mode,
            "horizons": list(config.horizons),
            "random_prefixes_per_segment": config.random_prefixes_per_segment,
            "conditioning": config.conditioning,
            "label_mode": config.label_mode,
            "failure_horizon": config.failure_horizon,
            "temporal_representation": config.temporal_representation,
            "temporal_window": config.temporal_window,
            "min_stage_successes": config.min_stage_successes,
            "min_stage_failures": config.min_stage_failures,
            "requested_stages": requested,
        }
    if (selected is None) != (current is None):
        raise ValueError(
            "Selection summary and final refit disagree on whether causal "
            "Subtask-SAFE is enabled"
        )
    if selected is None:
        return
    backward_compatible_defaults = {
        "label_mode": "eventual",
        "failure_horizon": None,
        "temporal_representation": "raw",
        "temporal_window": 4,
    }
    for key, value in current.items():
        selected_value = selected.get(key, backward_compatible_defaults.get(key))
        if selected_value != value:
            raise ValueError(
                f"Selection summary causal setting {key}={selected_value!r} "
                f"does not match final refit value {value!r}"
            )


def causal_args_signature(args):
    enabled = bool(
        args.causal_prefix_mode != "none"
        or args.causal_conditioning != "none"
        or args.causal_label_mode != "eventual"
        or args.causal_failure_horizon is not None
        or args.temporal_representation != "raw"
        or args.min_stage_successes
        or args.min_stage_failures
        or args.stages
    )
    if not enabled:
        return None
    return {
        "training_mode": args.causal_prefix_mode,
        "horizons": list(validate_prefix_horizons(args.causal_prefix_horizons)),
        "random_prefixes_per_segment": int(args.random_prefixes_per_segment),
        "conditioning": args.causal_conditioning,
        "label_mode": args.causal_label_mode,
        "failure_horizon": args.causal_failure_horizon,
        "temporal_representation": args.temporal_representation,
        "temporal_window": int(args.temporal_window),
        "min_stage_successes": int(args.min_stage_successes),
        "min_stage_failures": int(args.min_stage_failures),
        "requested_stages": None if args.stages is None else sorted(args.stages),
    }


def causal_metrics_signature(metrics):
    payload = metrics.get("causal_subtask_safe")
    if payload is None:
        return None
    return {
        "training_mode": payload["protocol"]["training_mode"],
        "horizons": payload["protocol"]["horizons"],
        "random_prefixes_per_segment": payload["protocol"][
            "random_prefixes_per_segment"
        ],
        "conditioning": payload["conditioning"]["mode"],
        "label_mode": payload["protocol"].get("label_mode", "eventual"),
        "failure_horizon": payload["protocol"].get("failure_horizon"),
        "temporal_representation": payload.get(
            "temporal_representation", {"mode": "raw"}
        )["mode"],
        "temporal_window": payload.get("temporal_representation", {"window": 4})[
            "window"
        ],
        "min_stage_successes": payload["stage_support"]["min_successes"],
        "min_stage_failures": payload["stage_support"]["min_failures"],
        "requested_stages": payload["stage_support"]["requested_stages"],
    }


def online_args_signature(args):
    config = OnlineSafeConfig(
        mode=args.online_safe_mode,
        seed=args.online_safe_seed,
        landmark_fraction=args.online_landmark_fraction,
    )
    return None if config.mode == "none" else online_config_dict(config)


def online_metrics_signature(metrics):
    payload = metrics.get("online_safe")
    if payload is None:
        return None
    protocol = payload["protocol"]
    return {
        "mode": protocol["mode"],
        "seed": int(protocol["seed"]),
        "landmark_fraction": float(protocol["landmark_fraction"]),
    }


def validate_online_selection_summary(selection_summary, config):
    if selection_summary is None:
        return
    summary = json.loads(Path(selection_summary).read_text())
    selected = summary.get("online_safe")
    requested = None if config.mode == "none" else online_config_dict(config)
    if selected != requested:
        raise ValueError(
            "Selection summary and final refit use different online SAFE protocols: "
            f"selected={selected!r}, requested={requested!r}"
        )


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def verify_safe_repo(safe_repo):
    safe_repo = Path(safe_repo).resolve()
    commit = subprocess.run(
        ["git", "-C", str(safe_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_SAFE_COMMIT:
        raise ValueError(
            f"SAFE checkout is at {commit}, expected {OFFICIAL_SAFE_COMMIT}"
        )
    sys.path.insert(0, str(safe_repo))
    return safe_repo


def _natural_sort_key(value):
    """Return a deterministic numeric-aware key without requiring natsort."""

    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))
    )


def load_env_records(export_dir):
    paths = sorted(
        glob.glob(str(Path(export_dir) / "env_records" / "*.pkl")),
        key=_natural_sort_key,
    )
    records = []
    for path in paths:
        with open(path, "rb") as stream:
            records.append((Path(path), pickle.load(stream)))
    if not records:
        raise ValueError(f"No env records found in {export_dir}")
    return records


def validate_alignment(rollouts, env_records):
    if len(rollouts) != len(env_records):
        raise ValueError("Official loader/env record count mismatch")
    identity = {}
    for rollout, (path, env) in zip(rollouts, env_records):
        if int(rollout.task_id) != int(env["task_id"]):
            raise ValueError(f"Task mismatch at {path}")
        if int(rollout.episode_success) != int(env["episode_success"]):
            raise ValueError(f"Outcome mismatch at {path}")
        identity[id(rollout)] = (path, env)
    return identity


def task_catalog(export_dir):
    report = json.loads((Path(export_dir) / "conversion_report.json").read_text())
    task_ids = {name: int(task_id) for name, task_id in report["task_ids"].items()}
    task_types = report.get("task_types", {})
    missing_types = sorted(set(task_ids) - set(task_types))
    if missing_types:
        raise ValueError(
            "Official export is missing task-type provenance for: "
            + ", ".join(missing_types)
        )
    invalid_types = {
        name: task_types[name]
        for name in task_ids
        if task_types[name] not in TASK_TYPE_FILTERS[1:]
    }
    if invalid_types:
        raise ValueError(f"Official export has invalid task types: {invalid_types}")
    return {
        "task_ids": task_ids,
        "task_names": {task_id: name for name, task_id in task_ids.items()},
        "task_types": {name: task_types[name] for name in task_ids},
    }


def resolve_task_type_selection(export_dir, task_type="all"):
    if task_type not in TASK_TYPE_FILTERS:
        raise ValueError(
            f"Unknown task type {task_type!r}; expected one of {TASK_TYPE_FILTERS}"
        )
    catalog = task_catalog(export_dir)
    selected_names = sorted(
        name
        for name, value in catalog["task_types"].items()
        if task_type == "all" or value == task_type
    )
    if not selected_names:
        raise ValueError(f"No {task_type} tasks are present in the official export")
    selected_ids = sorted(catalog["task_ids"][name] for name in selected_names)
    return {
        "task_type_filter": task_type,
        "source_num_tasks": len(catalog["task_ids"]),
        "selected_task_ids": selected_ids,
        "selected_task_names": selected_names,
        "selected_task_types": {
            name: catalog["task_types"][name] for name in selected_names
        },
    }


def filter_aligned_task_type(
    rollouts,
    env_records,
    selection,
):
    validate_alignment(rollouts, env_records)
    selected_ids = set(selection["selected_task_ids"])
    pairs = [
        (rollout, env_record)
        for rollout, env_record in zip(rollouts, env_records)
        if int(env_record[1]["task_id"]) in selected_ids
    ]
    if not pairs:
        raise ValueError("Task-type filter selected no aligned rollouts")
    selected_rollouts = [rollout for rollout, _ in pairs]
    selected_env_records = [env_record for _, env_record in pairs]
    identity = validate_alignment(selected_rollouts, selected_env_records)
    return selected_rollouts, selected_env_records, identity


def load_outer_split_ids(path):
    if path is None:
        return None
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    train = manifest.get("train")
    test = manifest.get("test")
    if not isinstance(train, list) or not isinstance(test, list):
        raise ValueError(
            "Outer split manifest must contain rollout-ID lists named train and test"
        )
    train = {str(rollout_id) for rollout_id in train}
    test = {str(rollout_id) for rollout_id in test}
    if not train or not test or train & test:
        raise ValueError("Outer split manifest train/test IDs are empty or overlap")
    return {
        "path": str(path),
        "train": train,
        "test": test,
        "split_seed": manifest.get("split_seed"),
        "split_unit": manifest.get("split_unit", "rollout"),
        "manifest": manifest,
    }


def make_manifest_split(rollouts, identity, selection):
    """Apply an exact, possibly imbalanced and non-exhaustive train/test manifest."""
    if selection is None:
        raise ValueError("A selection manifest is required")
    available = {
        str(identity[id(rollout)][1]["rollout_id"]): rollout for rollout in rollouts
    }
    requested = selection["train"] | selection["test"]
    missing = sorted(requested - set(available))
    if missing:
        raise ValueError(
            "Selection manifest contains rollout IDs absent from the export: "
            + ", ".join(missing[:5])
        )
    train = [available[rollout_id] for rollout_id in sorted(selection["train"])]
    test = [available[rollout_id] for rollout_id in sorted(selection["test"])]
    allow_single_class_tasks = selection.get("split_unit") == "parent_rollout"
    per_task = {}
    task_ids = sorted({int(rollout.task_id) for rollout in train + test})
    for task_id in task_ids:
        per_task[task_id] = {}
        for success in (0, 1):
            group_train = [
                rollout
                for rollout in train
                if int(rollout.task_id) == task_id
                and int(rollout.episode_success) == success
            ]
            group_test = [
                rollout
                for rollout in test
                if int(rollout.task_id) == task_id
                and int(rollout.episode_success) == success
            ]
            if not allow_single_class_tasks and (not group_train or not group_test):
                raise ValueError(
                    f"Task {task_id}, success={success} selection has "
                    f"{len(group_train)} train and {len(group_test)} test rollouts; "
                    "both splits require both outcomes"
                )
            per_task[task_id]["success" if success else "failure"] = {
                "train": len(group_train),
                "test": len(group_test),
            }
    if allow_single_class_tasks:
        train_labels = {int(rollout.episode_success) for rollout in train}
        test_labels = {int(rollout.episode_success) for rollout in test}
        if train_labels != {0, 1} or test_labels != {0, 1}:
            raise ValueError(
                "Parent-rollout Subtask-SAFE splits require both segment labels "
                "globally in train and test"
            )
        parent_train = set(selection["manifest"].get("parent_train", []))
        parent_test = set(selection["manifest"].get("parent_test", []))
        if not parent_train or not parent_test or parent_train & parent_test:
            raise ValueError(
                "Parent-rollout split manifest has empty or overlapping parent IDs"
            )
        actual_train = {
            str(identity[id(rollout)][1].get("parent_rollout_id", ""))
            for rollout in train
        }
        actual_test = {
            str(identity[id(rollout)][1].get("parent_rollout_id", ""))
            for rollout in test
        }
        if (
            actual_train != parent_train
            or actual_test != parent_test
            or actual_train & actual_test
        ):
            raise ValueError(
                "Segment assignments disagree with the parent-rollout split manifest"
            )
    selected_task_ids = {int(rollout.task_id) for rollout in train + test}
    source_task_ids = {int(rollout.task_id) for rollout in rollouts}
    if selected_task_ids != source_task_ids:
        raise ValueError(
            "Selection manifest task coverage differs from the selected export: "
            f"{sorted(selected_task_ids)} != {sorted(source_task_ids)}"
        )
    return train, test, per_task


def make_seen_split(
    rollouts,
    identity,
    *,
    train_per_class=7,
    split_seed=0,
    fixed_split_ids=None,
):
    """Return a fixed outcome-stratified split shared by every model seed."""
    grouped = defaultdict(list)
    for rollout in rollouts:
        grouped[(int(rollout.task_id), int(rollout.episode_success))].append(rollout)
    task_ids = sorted({key[0] for key in grouped})
    rng = random.Random(split_seed)
    train, test = [], []
    per_task = {}
    for task_id in task_ids:
        per_task[task_id] = {}
        for success in (0, 1):
            values = sorted(
                grouped[(task_id, success)],
                key=lambda rollout: str(identity[id(rollout)][1]["rollout_id"]),
            )
            if fixed_split_ids is None:
                rng.shuffle(values)
                group_train = values[:train_per_class]
                group_test = values[train_per_class:]
            else:
                group_train = [
                    rollout
                    for rollout in values
                    if identity[id(rollout)][1]["rollout_id"]
                    in fixed_split_ids["train"]
                ]
                group_test = [
                    rollout
                    for rollout in values
                    if identity[id(rollout)][1]["rollout_id"] in fixed_split_ids["test"]
                ]
                assigned_ids = {
                    identity[id(rollout)][1]["rollout_id"]
                    for rollout in group_train + group_test
                }
                if len(assigned_ids) != len(values):
                    missing = [
                        identity[id(rollout)][1]["rollout_id"]
                        for rollout in values
                        if identity[id(rollout)][1]["rollout_id"] not in assigned_ids
                    ]
                    raise ValueError(
                        "Outer split manifest does not assign selected rollout IDs: "
                        + ", ".join(missing[:5])
                    )
            if len(group_train) != train_per_class or not group_test:
                raise ValueError(
                    f"Task {task_id}, success={success} split has "
                    f"{len(group_train)} train and {len(group_test)} test rollouts; "
                    f"expected {train_per_class} train and at least one test"
                )
            train.extend(group_train)
            test.extend(group_test)
            per_task[task_id]["success" if success else "failure"] = {
                "train": len(group_train),
                "test": len(group_test),
            }
    train_ids = {identity[id(rollout)][1]["rollout_id"] for rollout in train}
    test_ids = {identity[id(rollout)][1]["rollout_id"] for rollout in test}
    if train_ids & test_ids or len(train_ids | test_ids) != len(rollouts):
        raise AssertionError("Seen-task split is not disjoint and exhaustive")
    return train, test, per_task


def make_config(export_dir, model_name, seed, epochs, device, hyperparameters=None):
    from failure_prob.conf import (
        Config,
        IndepModelConfig,
        LstmModelConfig,
        PizeroDatasetConfig,
        TrainConfig,
    )

    defaults = hyperparameters or MODEL_DEFAULTS[model_name]
    dataset = PizeroDatasetConfig(
        data_path=str(Path(export_dir).resolve()),
        data_path_prefix="",
        load_to_cuda=str(device).startswith("cuda"),
        horizon_idx_rel=defaults["horizon_selector"],
        diff_idx_rel=defaults["diffusion_selector"],
    )
    model_type = IndepModelConfig if model_name == "indep" else LstmModelConfig
    model = model_type(
        n_epochs=epochs,
        lr=defaults["learning_rate"],
        lambda_reg=defaults["lambda_reg"],
    )
    train = TrainConfig(seed=seed, log_precomputed=False, log_precomputed_only=False)
    return Config(dataset=dataset, model=model, train=train)


def resolve_class_weights(dataset, class_weighting):
    if class_weighting == "official_inverse_frequency":
        return dataset.get_class_weights()
    if class_weighting == "none":
        # Official aggregate_monitor_loss indexes the vector unconditionally.
        # Equal-frequency base multipliers remove inverse-frequency scaling.
        # Preserve the empirical mean weight of the official vector so this
        # ablation does not also change the monitor/regularization loss scale.
        rollouts = dataset.get_rollouts()
        if not rollouts:
            raise ValueError(
                "Cannot derive scale-matched class weights from no rollouts"
            )
        failures = sum(not int(rollout.episode_success) for rollout in rollouts)
        successes = len(rollouts) - failures
        frequencies = [failures / len(rollouts), successes / len(rollouts)]
        official = dataset.get_class_weights()
        base = [
            float(dataset.cfg.model.lambda_fail),
            float(dataset.cfg.model.lambda_success),
        ]
        official_mean = sum(
            frequency * float(weight)
            for frequency, weight in zip(frequencies, official)
        )
        base_mean = sum(
            frequency * weight for frequency, weight in zip(frequencies, base)
        )
        if base_mean <= 0:
            raise ValueError("Configured class-loss multipliers have non-positive mean")
        scale = official_mean / base_mean
        return [
            scale * base[0],
            scale * base[1],
        ]
    raise ValueError(
        f"Unknown class weighting {class_weighting!r}; "
        f"expected one of {CLASS_WEIGHTING_MODES}"
    )


def train_epoch_without_wandb(
    model,
    optimizer,
    loader,
    device,
    *,
    class_weighting="official_inverse_frequency",
    loss_mode="official",
    focal_gamma=2.0,
):
    import torch
    from failure_prob.utils.torch import move_to_device

    model.train()
    weights = resolve_class_weights(loader.dataset, class_weighting)
    losses = []
    for batch in loader:
        batch = move_to_device(batch, device)
        if loss_mode == "official":
            monitor_loss, _ = model.forward_compute_loss(batch, weights)
        else:
            monitor_loss = causal_binary_monitor_loss(
                model,
                batch,
                weights,
                loss_mode=loss_mode,
                focal_gamma=focal_gamma,
            )
        regularization, _ = model.compute_regularization_loss(
            model.cfg.model.lambda_reg
        )
        total = monitor_loss + regularization
        if not torch.isfinite(total):
            raise RuntimeError("Official SAFE training loss became non-finite")
        optimizer.zero_grad()
        total.backward()
        if model.cfg.model.grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), model.cfg.model.grad_max_norm
            )
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    return float(np.mean(losses))


def causal_binary_monitor_loss(
    model,
    batch,
    weights,
    *,
    loss_mode="bce",
    focal_gamma=2.0,
):
    """BCE/focal alternative on each causal prefix endpoint.

    The official independent monitor cumulatively sums sigmoid outputs. BCE is
    therefore applied to its pre-accumulation projector probability; LSTM
    outputs are already instantaneous probabilities in the pinned SAFE code.
    Only the last valid state is supervised because the finite-horizon target
    describes the prefix endpoint, not every earlier state in that prefix.
    """
    import torch

    if loss_mode not in ("bce", "focal"):
        raise ValueError(f"Unknown binary causal loss mode {loss_mode!r}")
    if float(focal_gamma) < 0:
        raise ValueError("focal_gamma must be non-negative")
    if getattr(model.cfg.model, "name", None) == "indep":
        probabilities = model.projector(batch["features"]).squeeze(-1)
    else:
        probabilities = model(batch).squeeze(-1)
        if bool(getattr(model.cfg.model, "cumsum", False)):
            increments = torch.zeros_like(probabilities)
            increments[:, 0] = probabilities[:, 0]
            increments[:, 1:] = probabilities[:, 1:] - probabilities[:, :-1]
            probabilities = increments
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    valid = batch["valid_masks"].bool()
    valid_lengths = valid.sum(dim=1)
    if torch.any(valid_lengths <= 0):
        raise ValueError("Binary causal loss received an empty sequence")
    row_indices = torch.arange(len(probabilities), device=probabilities.device)
    endpoint_probabilities = probabilities[row_indices, valid_lengths - 1]
    success_labels = batch["success_labels"]
    targets = (1 - success_labels).float()
    losses = torch.nn.functional.binary_cross_entropy(
        endpoint_probabilities, targets, reduction="none"
    )
    if loss_mode == "focal":
        target_probability = torch.where(
            targets > 0.5, endpoint_probabilities, 1.0 - endpoint_probabilities
        )
        losses = losses * (1.0 - target_probability).pow(float(focal_gamma))
    failure_mask = success_labels == 0
    success_mask = success_labels == 1
    return (
        float(weights[0]) * losses[failure_mask].sum()
        + float(weights[1]) * losses[success_mask].sum()
    ) / len(losses)


def score_splits(model, loaders):
    import torch
    from torch.utils.data import DataLoader
    from failure_prob.utils.routines import model_forward_dataloader

    model.eval()
    output = {}
    for split, loader in loaders.items():
        loader = DataLoader(
            loader.dataset,
            batch_size=loader.dataset.cfg.model.batch_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            scores, masks, _ = model_forward_dataloader(model, loader)
        scores = scores.detach().cpu().numpy()
        lengths = masks.sum(dim=-1).detach().cpu().numpy().astype(int)
        output[split] = [scores[index, :length] for index, length in enumerate(lengths)]
        if any(not np.all(np.isfinite(score)) for score in output[split]):
            raise RuntimeError(f"Non-finite model scores in {split}")
    return output


def task_names(export_dir):
    report = json.loads((Path(export_dir) / "conversion_report.json").read_text())
    return {int(value): key for key, value in report["task_ids"].items()}


def export_provenance(export_dir):
    report = json.loads((Path(export_dir) / "conversion_report.json").read_text())
    return {
        "source_fingerprint": report.get("source_fingerprint"),
        "model_families": report.get("model_families", []),
        "task_types": report.get("task_types", {}),
    }


def save_scores(
    path,
    rollouts_by_split,
    scores_by_split,
    identity,
    names,
    task_types,
    model,
    seed,
):
    with Path(path).open("w") as stream:
        for split, rollouts in rollouts_by_split.items():
            for rollout, score in zip(rollouts, scores_by_split[split]):
                env_path, env = identity[id(rollout)]
                metadata = env.get("robocasa_manifest_record", {})
                record = {
                    "rollout_id": env["rollout_id"],
                    "parent_rollout_id": env.get("parent_rollout_id"),
                    "parent_task_name": env.get("parent_task_name"),
                    "parent_rollout_failed": env.get("parent_rollout_failed"),
                    "subtask_id": env.get("subtask_id"),
                    "subtask_index": env.get("subtask_index"),
                    "subtask_instruction": env.get("subtask_instruction"),
                    "subtask_safe_segment": env.get("subtask_safe_segment"),
                    "source_segment_id": env.get("source_segment_id"),
                    "source_segment_num_inferences": env.get(
                        "source_segment_num_inferences"
                    ),
                    "source_segment_failed": env.get("source_segment_failed"),
                    "causal_prefix_inferences": env.get("causal_prefix_inferences"),
                    "causal_label_mode": env.get("causal_label_mode"),
                    "causal_failure_horizon_inferences": env.get(
                        "causal_failure_horizon_inferences"
                    ),
                    "causal_target_failed": env.get("causal_target_failed"),
                    "remaining_inferences_to_terminal": env.get(
                        "remaining_inferences_to_terminal"
                    ),
                    "online_safe_mode": env.get("online_safe_mode"),
                    "online_source_rollout_id": env.get("online_source_rollout_id"),
                    "online_source_num_inferences": env.get(
                        "online_source_num_inferences"
                    ),
                    "online_prefix_inferences": env.get("online_prefix_inferences"),
                    "online_partner_rollout_id": env.get("online_partner_rollout_id"),
                    "online_task_timeout_inferences": env.get(
                        "online_task_timeout_inferences"
                    ),
                    "online_landmark_fraction": env.get("online_landmark_fraction"),
                    "split": split,
                    "task_id": int(rollout.task_id),
                    "task_name": names[int(rollout.task_id)],
                    "task_type": task_types[names[int(rollout.task_id)]],
                    "failed": not bool(rollout.episode_success),
                    "model": model,
                    "seed": seed,
                    "scores": np.asarray(score).tolist(),
                    "num_inferences": len(score),
                    "task_min_step": int(rollout.task_min_step),
                    "video_path": str(env_path.with_suffix(".mp4")),
                    "video_frame_stride": int(metadata.get("video_frame_stride", 1)),
                    "inference_environment_steps": env.get(
                        "inference_environment_steps",
                        metadata.get(
                            "inference_environment_steps",
                            list(
                                range(
                                    0,
                                    len(score) * int(env["replan_steps"]),
                                    int(env["replan_steps"]),
                                )
                            ),
                        ),
                    ),
                }
                stream.write(json.dumps(json_value(record), allow_nan=False) + "\n")


def train_seen_model(args):
    output = Path(args.output_dir).resolve()
    metrics_path = output / "metrics.json"
    if args.resume and metrics_path.is_file():
        previous_metrics = json.loads(metrics_path.read_text())
        if causal_metrics_signature(previous_metrics) != causal_args_signature(args):
            raise ValueError(
                f"Existing final refit uses an incompatible causal protocol: {output}"
            )
        if online_metrics_signature(previous_metrics) != online_args_signature(args):
            raise ValueError(
                f"Existing final refit uses an incompatible online SAFE protocol: {output}"
            )
        previous_objective = previous_metrics.get(
            "training_objective",
            training_objective_metadata("official"),
        )
        current_objective = training_objective_metadata(
            args.loss_mode, args.focal_gamma
        )
        if previous_objective != current_objective:
            raise ValueError(
                f"Existing final refit uses an incompatible objective: {output}"
            )
        return previous_metrics
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output}; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    safe_repo = verify_safe_repo(args.safe_repo)
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from failure_prob.data.pizero import load_rollouts_from_root
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.model import get_model
    from failure_prob.utils.metrics import eval_scores_roc_prc
    from failure_prob.utils.random import seed_everything

    hyperparameters = resolve_hyperparameters(
        args.model,
        args.selection_summary,
        expected_task_type=args.task_type,
    )
    if args.selection_summary is not None:
        selection = json.loads(Path(args.selection_summary).read_text())
        selected_objective = selection.get(
            "training_objective",
            training_objective_metadata("official"),
        )
        current_objective = training_objective_metadata(
            args.loss_mode, args.focal_gamma
        )
        if selected_objective != current_objective:
            raise ValueError(
                "Selection summary training objective does not match final refit"
            )
    cfg = make_config(
        args.export_dir,
        args.model,
        args.seed,
        args.epochs,
        args.device,
        hyperparameters,
    )
    configure_training_objective(cfg, args.loss_mode)
    seed_everything(0)
    source_rollouts = load_rollouts_from_root(Path(args.export_dir), cfg)
    source_env_records = load_env_records(args.export_dir)
    task_selection = resolve_task_type_selection(
        args.export_dir,
        args.task_type,
    )
    all_rollouts, env_records, identity = filter_aligned_task_type(
        source_rollouts,
        source_env_records,
        task_selection,
    )
    if args.outer_split_manifest is not None and args.selection_manifest is not None:
        raise ValueError(
            "--outer-split-manifest and --selection-manifest are mutually exclusive"
        )
    fixed_split_ids = load_outer_split_ids(
        args.selection_manifest or args.outer_split_manifest
    )
    fixed_outer_split_path = (
        fixed_split_ids["manifest"].get("fixed_outer_split_manifest")
        if args.selection_manifest is not None
        else (fixed_split_ids["path"] if fixed_split_ids is not None else None)
    )
    parent_grouped_split = bool(
        fixed_split_ids is not None
        and fixed_split_ids.get("split_unit") == "parent_rollout"
    )
    if (
        fixed_split_ids is not None
        and fixed_split_ids["split_seed"] is not None
        and int(fixed_split_ids["split_seed"]) != args.split_seed
    ):
        raise ValueError("Outer split manifest split_seed does not match --split-seed")
    if args.selection_manifest is not None:
        train_rollouts, test_rollouts, per_task = make_manifest_split(
            all_rollouts,
            identity,
            fixed_split_ids,
        )
    else:
        train_rollouts, test_rollouts, per_task = make_seen_split(
            all_rollouts,
            identity,
            train_per_class=args.train_per_class,
            split_seed=args.split_seed,
            fixed_split_ids=fixed_split_ids,
        )
    causal_requested = bool(
        args.causal_prefix_mode != "none"
        or args.causal_conditioning != "none"
        or args.causal_label_mode != "eventual"
        or args.causal_failure_horizon is not None
        or args.temporal_representation != "raw"
        or args.min_stage_successes
        or args.min_stage_failures
        or args.stages
    )
    causal_payload = None
    causal_config = None
    online_config = OnlineSafeConfig(
        mode=args.online_safe_mode,
        seed=args.online_safe_seed,
        landmark_fraction=args.online_landmark_fraction,
    )
    online_requested = online_config.mode != "none"
    if causal_requested and online_requested:
        raise ValueError(
            "Original online SAFE and causal Subtask-SAFE modes are mutually exclusive"
        )
    validate_online_selection_summary(args.selection_summary, online_config)
    online_payload = None
    if online_requested:
        validate_causal_selection_summary(args.selection_summary, None, None)
        online_payload = prepare_online_splits(
            train_rollouts,
            test_rollouts,
            identity,
            config=online_config,
        )
        train_rollouts = online_payload["train"]
        test_rollouts = online_payload["test"]
        identity = online_payload["identity"]
        all_rollouts = train_rollouts + test_rollouts
        task_cutoffs = {
            int(task_id): "per_online_prefix_length"
            for task_id in sorted({int(item.task_id) for item in all_rollouts})
        }
    elif causal_requested:
        if not parent_grouped_split:
            raise ValueError(
                "Causal Subtask-SAFE requires a parent_rollout selection manifest"
            )
        causal_config = CausalPrefixConfig(
            training_mode=args.causal_prefix_mode,
            horizons=validate_prefix_horizons(args.causal_prefix_horizons),
            random_prefixes_per_segment=args.random_prefixes_per_segment,
            conditioning=args.causal_conditioning,
            label_mode=args.causal_label_mode,
            failure_horizon=args.causal_failure_horizon,
            temporal_representation=args.temporal_representation,
            temporal_window=args.temporal_window,
            min_stage_successes=args.min_stage_successes,
            min_stage_failures=args.min_stage_failures,
        )
        validate_causal_selection_summary(
            args.selection_summary, causal_config, args.stages
        )
        causal_payload = prepare_causal_splits(
            train_rollouts,
            test_rollouts,
            identity,
            config=causal_config,
            random_seed=args.seed,
            requested_stages=args.stages,
        )
        train_rollouts = causal_payload["train"]
        test_rollouts = causal_payload["test"]
        identity = causal_payload["identity"]
        all_rollouts = train_rollouts + test_rollouts
        # Each synthetic example is already causally truncated. Do not replace
        # its own prefix length with a task-wide minimum of one inference.
        for rollout in all_rollouts:
            rollout.task_min_step = len(rollout.hidden_states)
        task_cutoffs = {
            int(task_id): "per_causal_prefix_length"
            for task_id in sorted({int(item.task_id) for item in all_rollouts})
        }
    else:
        validate_causal_selection_summary(args.selection_summary, None, None)
        task_cutoffs = set_task_min_step_from_training(train_rollouts, test_rollouts)
    if cfg.dataset.load_to_cuda:
        all_rollouts = [rollout.to(args.device) for rollout in all_rollouts]
    rollouts_by_split = {"train": train_rollouts, "test": test_rollouts}
    seed_everything(args.seed)
    datasets = {
        split: RolloutDataset(cfg, rollouts)
        for split, rollouts in rollouts_by_split.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=cfg.model.batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    training_class_weights = [
        float(value)
        for value in resolve_class_weights(
            datasets["train"],
            args.class_weighting,
        )
    ]
    model = get_model(cfg, int(train_rollouts[0].hidden_states.shape[-1]))
    model.to(args.device)
    optimizer, scheduler = model.get_optimizer()
    history = []
    for epoch in range(args.epochs):
        history.append(
            train_epoch_without_wandb(
                model,
                optimizer,
                loaders["train"],
                args.device,
                class_weighting=args.class_weighting,
                loss_mode=args.loss_mode,
                focal_gamma=args.focal_gamma,
            )
        )
        if scheduler is not None:
            scheduler.step()
    scores_by_split = score_splits(model, loaders)
    metrics = eval_scores_roc_prc(
        rollouts_by_split,
        copy.deepcopy(scores_by_split),
        "model",
        [0.25, 0.5, 0.75, 1.0],
        plot_auc_curves=False,
        plot_score_curves=False,
    )
    torch.save(model.state_dict(), output / "model_final.ckpt")
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    write_json(output / "train_history.json", {"loss": history})
    all_names = task_names(args.export_dir)
    selected_task_ids = sorted({int(rollout.task_id) for rollout in all_rollouts})
    names = {task_id: all_names[task_id] for task_id in selected_task_ids}
    provenance = export_provenance(args.export_dir)
    selected_task_types = {
        name: provenance["task_types"][name] for name in names.values()
    }
    train_successes = sum(int(rollout.episode_success) for rollout in train_rollouts)
    train_failures = len(train_rollouts) - train_successes
    test_successes = sum(int(rollout.episode_success) for rollout in test_rollouts)
    test_failures = len(test_rollouts) - test_successes
    if causal_payload is not None or online_payload is not None:
        per_task = {}
        for task_id in selected_task_ids:
            per_task[task_id] = {}
            for success, outcome in ((1, "success"), (0, "failure")):
                per_task[task_id][outcome] = {
                    "train": sum(
                        int(item.task_id) == task_id
                        and int(item.episode_success) == success
                        for item in train_rollouts
                    ),
                    "test": sum(
                        int(item.task_id) == task_id
                        and int(item.episode_success) == success
                        for item in test_rollouts
                    ),
                }
    test_per_class_values = {
        counts[outcome]["test"]
        for counts in per_task.values()
        for outcome in ("success", "failure")
    }
    test_per_task_class = (
        next(iter(test_per_class_values)) if len(test_per_class_values) == 1 else None
    )
    split_manifest = {
        "schema_version": 1,
        "protocol": (
            "online_safe_source_rollout_stratified"
            if online_payload is not None
            else (
                "subtask_safe_parent_rollout_stratified"
                if parent_grouped_split
                else "same_task_outcome_stratified"
            )
        ),
        "split_unit": ("parent_rollout" if parent_grouped_split else "rollout"),
        "split_seed": args.split_seed,
        "task_type_filter": args.task_type,
        "source_num_tasks": task_selection["source_num_tasks"],
        "num_tasks": len(names),
        "task_names": [names[key] for key in sorted(names)],
        "task_types": selected_task_types,
        "outer_split_manifest": (fixed_outer_split_path),
        "selection_manifest": (
            fixed_split_ids["path"] if args.selection_manifest is not None else None
        ),
        "train_per_task_class": (
            None if args.selection_manifest is not None else args.train_per_class
        ),
        "class_weighting": args.class_weighting,
        "training_objective": training_objective_metadata(
            args.loss_mode, args.focal_gamma
        ),
        "training_class_weights": training_class_weights,
        "causal_subtask_safe": (
            {
                "protocol": causal_payload["protocol"],
                "conditioning": causal_payload["conditioning"],
                "temporal_representation": causal_payload["temporal_representation"],
                "stage_support": causal_payload["support"],
                "prefix_counts": causal_payload["prefix_counts"],
                "target_counts_by_prefix": causal_payload["target_counts_by_prefix"],
                "source_counts": {
                    "train_segments": len(causal_payload["source_train"]),
                    "test_segments": len(causal_payload["source_test"]),
                },
            }
            if causal_payload is not None
            else None
        ),
        "online_safe": online_payload_summary(online_payload),
        "test_per_task_class": test_per_task_class,
        "counts": {
            "train": len(train_rollouts),
            "test": len(test_rollouts),
            "train_successes": train_successes,
            "train_failures": train_failures,
            "test_successes": test_successes,
            "test_failures": test_failures,
        },
        "per_task": {names[key]: value for key, value in per_task.items()},
        "train": [identity[id(rollout)][1]["rollout_id"] for rollout in train_rollouts],
        "test": [identity[id(rollout)][1]["rollout_id"] for rollout in test_rollouts],
        "parent_train": (
            sorted(
                {
                    identity[id(rollout)][1]["parent_rollout_id"]
                    for rollout in train_rollouts
                }
            )
            if parent_grouped_split
            else None
        ),
        "parent_test": (
            sorted(
                {
                    identity[id(rollout)][1]["parent_rollout_id"]
                    for rollout in test_rollouts
                }
            )
            if parent_grouped_split
            else None
        ),
    }
    write_json(output / "split_manifest.json", split_manifest)
    save_scores(
        output / "scores.jsonl",
        rollouts_by_split,
        scores_by_split,
        identity,
        names,
        selected_task_types,
        args.model,
        args.seed,
    )
    duration_labels = [1 - int(rollout.episode_success) for rollout in test_rollouts]
    duration = [len(rollout.hidden_states) for rollout in test_rollouts]
    from sklearn.metrics import roc_auc_score

    result = {
        "schema_version": 1,
        "model": args.model,
        "seed": args.seed,
        "official_safe_commit": OFFICIAL_SAFE_COMMIT,
        "export_dir": str(Path(args.export_dir).resolve()),
        "source_fingerprint": provenance["source_fingerprint"],
        "model_families": provenance["model_families"],
        "task_type_filter": args.task_type,
        "source_num_tasks": task_selection["source_num_tasks"],
        "task_types": selected_task_types,
        "num_tasks": len(names),
        "selected_hyperparameters": hyperparameters,
        "selection_summary": (
            str(Path(args.selection_summary).resolve())
            if args.selection_summary is not None
            else None
        ),
        "outer_split_manifest": (fixed_outer_split_path),
        "selection_manifest": (
            fixed_split_ids["path"] if args.selection_manifest is not None else None
        ),
        "class_weighting": args.class_weighting,
        "training_objective": training_objective_metadata(
            args.loss_mode, args.focal_gamma
        ),
        "split_unit": ("parent_rollout" if parent_grouped_split else "rollout"),
        "training_class_weights": training_class_weights,
        "causal_subtask_safe": (
            {
                "protocol": causal_payload["protocol"],
                "conditioning": causal_payload["conditioning"],
                "temporal_representation": causal_payload["temporal_representation"],
                "stage_support": causal_payload["support"],
                "prefix_counts": causal_payload["prefix_counts"],
                "target_counts_by_prefix": causal_payload["target_counts_by_prefix"],
                "source_counts": {
                    "train_segments": len(causal_payload["source_train"]),
                    "test_segments": len(causal_payload["source_test"]),
                },
            }
            if causal_payload is not None
            else None
        ),
        "online_safe": online_payload_summary(online_payload),
        "task_min_step_source": (
            "online SAFE transformed prefix length"
            if online_payload is not None
            else (
                f"minimum inference length per task in the {len(train_rollouts)}-rollout "
                "training split only"
            )
        ),
        "task_min_steps": {names[key]: value for key, value in task_cutoffs.items()},
        "counts": {
            "train": len(train_rollouts),
            "test": len(test_rollouts),
            "train_successes": train_successes,
            "train_failures": train_failures,
            "test_successes": test_successes,
            "test_failures": test_failures,
        },
        "scalar_metrics": metrics,
        "primary_metric": "falert_early_roc_auc/model_test",
        "primary_value": metrics["falert_early_roc_auc/model_test"],
        "duration_only_test_roc_auc": float(roc_auc_score(duration_labels, duration)),
        "thresholding": (
            f"No conformal threshold is fitted: all {len(test_rollouts)} held-out "
            "rollouts remain evaluation-only."
        ),
    }
    write_json(output / "metrics.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-per-class", type=int, default=7)
    parser.add_argument(
        "--task-type",
        choices=TASK_TYPE_FILTERS,
        default="all",
    )
    parser.add_argument(
        "--outer-split-manifest",
        help="Reuse train/test rollout IDs from a completed final refit",
    )
    parser.add_argument(
        "--selection-manifest",
        help=(
            "Use exact train/test rollout IDs, allowing an imbalanced and "
            "non-exhaustive training subset"
        ),
    )
    parser.add_argument(
        "--class-weighting",
        choices=CLASS_WEIGHTING_MODES,
        default="official_inverse_frequency",
        help="Official inverse-frequency SAFE weighting or an unweighted ablation",
    )
    parser.add_argument(
        "--loss-mode",
        choices=LOSS_MODES,
        default="official",
        help="Pinned SAFE loss, per-inference BCE, or focal BCE",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-summary")
    parser.add_argument(
        "--online-safe-mode",
        choices=ONLINE_SAFE_MODES,
        default="none",
        help=(
            "Original binary SAFE view: unmodified rollouts, per-task paired "
            "success-length matching, or a training-derived fixed at-risk landmark"
        ),
    )
    parser.add_argument(
        "--online-safe-seed",
        type=int,
        default=0,
        help="Pairing seed shared across detector model seeds",
    )
    parser.add_argument(
        "--online-landmark-fraction",
        type=float,
        default=0.5,
        help="Fraction of each training-derived task timeout for fixed_landmark mode",
    )
    parser.add_argument(
        "--causal-prefix-mode",
        choices=PREFIX_TRAINING_MODES,
        default="none",
        help=(
            "Train on complete segments (none), every declared causal prefix "
            "(fixed), or deterministic random causal prefixes (random)"
        ),
    )
    parser.add_argument(
        "--causal-prefix-horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_PREFIX_HORIZONS),
    )
    parser.add_argument("--random-prefixes-per-segment", type=int, default=3)
    parser.add_argument(
        "--causal-conditioning",
        choices=CONDITIONING_MODES,
        default="none",
        help="Append training-catalog stage one-hot and optional causal elapsed feature",
    )
    parser.add_argument(
        "--causal-label-mode",
        choices=CAUSAL_LABEL_MODES,
        default="eventual",
        help="Eventual segment failure or failure within the configured future horizon",
    )
    parser.add_argument("--causal-failure-horizon", type=int)
    parser.add_argument(
        "--temporal-representation",
        choices=TEMPORAL_REPRESENTATIONS,
        default="raw",
    )
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--min-stage-successes", type=int, default=0)
    parser.add_argument("--min-stage-failures", type=int, default=0)
    parser.add_argument(
        "--stages",
        nargs="+",
        help="Optional explicit ParentTask::subtask_id allowlist",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = train_seen_model(args)
    except (FileExistsError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(json_value(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
