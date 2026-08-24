"""Develop and prospectively evaluate full-parent stage-aware Xiaomi SAFE.

``develop`` never scores the opened outer parents from the five-task pilot.  It
uses only the original development allocation, creates fit/selection/calibration
partitions before constructing inference rows, selects one regularizer per arm,
refits fixed ensembles, and freezes score normalization and FPR thresholds.

``evaluate`` loads that immutable runtime bundle and applies it to a new raw
collection.  Development rollout IDs and task/seed/reset identities must be
disjoint.  No model, scaler, time curve, prototype, or threshold is updated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import traceback

import numpy as np

from .dataset import load_manifest
from .stage_aware_parent_safe import (
    ARMS,
    DEFAULT_FAILURE_HORIZONS,
    DEFAULT_PREFIXES,
    DEFAULT_PRIMARY_PREFIXES,
    MODEL_ARMS,
    allocate_development_parents,
    apply_score_normalizer,
    build_inference_rows,
    conformal_success_threshold,
    fit_scaler,
    fit_score_normalizer,
    fit_stage_time_curves,
    fit_success_prototypes,
    fit_success_scaler,
    fixed_prefix_metrics,
    group_score_trajectories,
    json_value,
    load_parent_sequences,
    load_parent_split,
    make_parent_holdout_split,
    model_rows,
    paired_fixed_prefix_bootstrap,
    paired_parent_bootstrap,
    parent_event_metrics,
    parent_identity,
    parent_stage_weights,
    primary_prefix_score,
    prototype_scores,
    row_target,
    select_parents,
    select_supported_stages,
    stage_event_metrics,
    stage_time_scores,
    task_stage_indices,
    training_horizons,
    transform_row_matrix,
    write_json,
)


SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Stage-aware SAFE training requires PyTorch; activate the vla_safe "
            "cluster environment"
        ) from error
    return torch


def arm_heads(arm, failure_horizons):
    if arm == "terminal":
        return ("terminal",)
    if arm == "stage":
        return ("stage", "terminal")
    if arm in ("multihorizon", "conditioned", "context"):
        return (
            "stage",
            *(f"horizon_{int(value)}" for value in failure_horizons),
            "terminal",
        )
    raise ValueError(f"Arm {arm!r} is not a neural model")


def arm_score_from_logits(torch, arm, logits, failure_horizons):
    probabilities = {name: torch.sigmoid(value) for name, value in logits.items()}
    if arm == "terminal":
        return probabilities["terminal"]
    if arm == "stage":
        return probabilities["stage"]
    near = torch.stack(
        [probabilities[f"horizon_{int(value)}"] for value in failure_horizons],
        dim=0,
    ).amax(dim=0)
    # Fixed probabilistic OR.  Calibration absorbs its scale; it is never tuned
    # on prospective data.
    return 1.0 - (1.0 - probabilities["stage"]) * (1.0 - near)


def make_model(
    *,
    input_dim,
    num_tasks,
    num_stages,
    conditioned,
    heads,
    hidden_dim,
    embedding_dim,
    dropout,
):
    torch = _torch()

    class StageAwareMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conditioned = bool(conditioned)
            added = 0
            if self.conditioned:
                self.task_embedding = torch.nn.Embedding(
                    int(num_tasks), int(embedding_dim)
                )
                self.stage_embedding = torch.nn.Embedding(
                    int(num_stages), int(embedding_dim)
                )
                added = 2 * int(embedding_dim)
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(int(input_dim) + added, int(hidden_dim)),
                torch.nn.LayerNorm(int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
                torch.nn.GELU(),
            )
            self.heads = torch.nn.ModuleDict(
                {name: torch.nn.Linear(int(hidden_dim), 1) for name in heads}
            )

        def forward(self, features, task_indices, stage_indices):
            if self.conditioned:
                features = torch.cat(
                    (
                        features,
                        self.task_embedding(task_indices),
                        self.stage_embedding(stage_indices),
                    ),
                    dim=-1,
                )
            hidden = self.trunk(features)
            return {
                name: head(hidden).squeeze(-1) for name, head in self.heads.items()
            }

    return StageAwareMlp()


def model_config_for_arm(
    arm,
    rows,
    task_catalog,
    stage_catalog,
    scaler,
    *,
    failure_horizons,
    hidden_dim,
    embedding_dim,
    dropout,
):
    return {
        "arm": arm,
        "input_dim": int(len(scaler["mean"])),
        "num_tasks": len(task_catalog),
        "num_stages": max(1, len(stage_catalog)),
        "conditioned": arm in ("conditioned", "context"),
        "heads": list(arm_heads(arm, failure_horizons)),
        "hidden_dim": int(hidden_dim),
        "embedding_dim": int(embedding_dim),
        "dropout": float(dropout),
        "task_catalog": dict(task_catalog),
        "stage_catalog": dict(stage_catalog),
        "failure_horizons": list(failure_horizons),
        "feature_dimension": int(len(scaler["mean"])),
        "training_rows": len(rows),
    }


def instantiate_from_config(config):
    return make_model(
        input_dim=config["input_dim"],
        num_tasks=config["num_tasks"],
        num_stages=config["num_stages"],
        conditioned=config["conditioned"],
        heads=config["heads"],
        hidden_dim=config["hidden_dim"],
        embedding_dim=config["embedding_dim"],
        dropout=config["dropout"],
    )


def _head_loss_weights(arm, failure_horizons, terminal_aux_weight, horizon_weight):
    if arm == "terminal":
        return {"terminal": 1.0}
    weights = {"stage": 1.0, "terminal": float(terminal_aux_weight)}
    if arm in ("multihorizon", "conditioned", "context"):
        weights.update(
            {
                f"horizon_{int(value)}": float(horizon_weight)
                for value in failure_horizons
            }
        )
    return weights


def _training_arrays(rows, arm, scaler, task_catalog, stage_catalog, heads):
    matrix = transform_row_matrix(rows, arm, scaler)
    tasks, stages = task_stage_indices(rows, task_catalog, stage_catalog)
    labels, weights = {}, {}
    for head in heads:
        labels[head] = np.asarray(
            [row_target(row, head) for row in rows], dtype=np.float32
        )
        weights[head] = parent_stage_weights(rows, target=head)
    return matrix, tasks, stages, labels, weights


def predict_model(model, rows, arm, scaler, config, *, device, batch_size=2048):
    torch = _torch()
    model.eval()
    matrix = transform_row_matrix(rows, arm, scaler)
    tasks, stages = task_stage_indices(
        rows, config["task_catalog"], config["stage_catalog"]
    )
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            end = start + int(batch_size)
            features = torch.as_tensor(matrix[start:end], device=device)
            task = torch.as_tensor(tasks[start:end], device=device)
            stage = torch.as_tensor(stages[start:end], device=device)
            logits = model(features, task, stage)
            score = arm_score_from_logits(
                torch, arm, logits, config["failure_horizons"]
            )
            outputs.append(score.detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_model(
    train_rows,
    selection_rows,
    *,
    arm,
    scaler,
    task_catalog,
    stage_catalog,
    failure_horizons,
    seed,
    hidden_dim,
    embedding_dim,
    dropout,
    learning_rate,
    weight_decay,
    epochs,
    patience,
    batch_size,
    device,
    prefixes,
    primary_prefixes,
    terminal_aux_weight,
    horizon_weight,
):
    torch = _torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    heads = arm_heads(arm, failure_horizons)
    config = model_config_for_arm(
        arm,
        train_rows,
        task_catalog,
        stage_catalog,
        scaler,
        failure_horizons=failure_horizons,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    model = instantiate_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    matrix, tasks, stages, labels, weights = _training_arrays(
        train_rows, arm, scaler, task_catalog, stage_catalog, heads
    )
    loss_weights = _head_loss_weights(
        arm, failure_horizons, terminal_aux_weight, horizon_weight
    )
    best = None
    stale = 0
    history = []
    rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        epoch_losses = []
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            features = torch.as_tensor(matrix[indices], device=device)
            task = torch.as_tensor(tasks[indices], device=device)
            stage = torch.as_tensor(stages[indices], device=device)
            logits = model(features, task, stage)
            loss = features.new_tensor(0.0)
            for head in heads:
                target = torch.as_tensor(labels[head][indices], device=device)
                sample_weight = torch.as_tensor(weights[head][indices], device=device)
                valid = sample_weight > 0
                if not torch.any(valid):
                    continue
                values = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[head][valid], target[valid], reduction="none"
                )
                loss = loss + float(loss_weights[head]) * (
                    values * sample_weight[valid]
                ).sum() / sample_weight[valid].sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Stage-aware SAFE loss became non-finite")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "training_loss": float(np.mean(epoch_losses))}
        if selection_rows is not None:
            score = predict_model(
                model, selection_rows, arm, scaler, config, device=device
            )
            metrics = fixed_prefix_metrics(selection_rows, score, prefixes=prefixes)
            selection_value = primary_prefix_score(metrics, primary_prefixes)
            record["selection_value"] = selection_value
            if best is None or selection_value > best["selection_value"] + 1e-7:
                best = {
                    "selection_value": selection_value,
                    "epoch": epoch,
                    "metrics": metrics,
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
                stale = 0
            else:
                stale += 1
            if stale >= int(patience):
                history.append(record)
                break
        history.append(record)
    if selection_rows is None:
        best = {
            "selection_value": None,
            "epoch": int(epochs),
            "metrics": None,
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
        }
    model.load_state_dict(best["state_dict"])
    model.to(device)
    return {"model": model, "config": config, "best": best, "history": history}


def _available_arms(requested, context_available):
    available, unavailable = [], {}
    for arm in requested:
        if arm == "context" and not context_available:
            unavailable[arm] = (
                "raw dataset has no aligned auxiliary observation_state_history"
            )
        else:
            available.append(arm)
    return available, unavailable


def split_raw_parents(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "initializing", "updated_at": utc_now()})
    loaded = load_parent_sequences(
        args.dataset_dir,
        tasks=args.tasks,
        aggregation=args.feature_aggregation,
        context_key=args.context_key,
        require_context=args.require_context,
    )
    parents = loaded["parents"]
    split = make_parent_holdout_split(
        parents, train_fraction=args.train_fraction, seed=args.split_seed
    )
    split.update(
        {
            "source_dataset": str(Path(args.dataset_dir).resolve()),
            "tasks": sorted({parent["task_name"] for parent in parents}),
            "context_key": args.context_key,
            "context_available_for_all": loaded["context_available_for_all"],
        }
    )
    path = output / "parent_rollout_split.json"
    write_json(path, split)
    status = {
        "status": "complete",
        "parents": len(parents),
        "tasks": len(split["tasks"]),
        "development": len(split["parent_train"]),
        "locked_opened_outer": len(split["parent_test"]),
        "context_available_for_all": loaded["context_available_for_all"],
        "split_manifest": str(path),
        "updated_at": utc_now(),
    }
    write_json(output / "status.json", status)
    return status


def _model_selection_rows(rows, arm):
    return model_rows(rows, arm)


def _ensemble_scores(models, rows, arm, scaler, config, device):
    values = [
        predict_model(model, rows, arm, scaler, config, device=device)
        for model in models
    ]
    return np.mean(np.stack(values), axis=0)


def _event_detector_payload(events, detector, normalizer, threshold, horizons):
    normalized = apply_score_normalizer(events, detector, normalizer)
    return stage_event_metrics(
        normalized, detector, threshold["threshold"], horizons
    )


def develop(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "initializing", "updated_at": utc_now()})
    split = load_parent_split(args.selection_manifest)
    loaded = load_parent_sequences(
        args.dataset_dir,
        parent_ids=split["development"],
        tasks=args.tasks,
        aggregation=args.feature_aggregation,
        context_key=args.context_key,
        require_context=False,
    )
    parents = loaded["parents"]
    loaded_ids = {parent["rollout_id"] for parent in parents}
    locked_outer_ids = set(split["locked_opened_outer"])
    if loaded_ids & locked_outer_ids:
        raise AssertionError("Opened outer parents entered development loading")
    source_metadata = {
        record.rollout_id: record for record in load_manifest(args.dataset_dir)
    }
    all_source_ids = loaded_ids | locked_outer_ids
    missing_source_metadata = sorted(all_source_ids - set(source_metadata))
    if missing_source_metadata:
        raise ValueError(
            "Development/opened outer IDs are absent from the raw manifest: "
            + ", ".join(missing_source_metadata[:5])
        )
    all_source_identities = sorted(
        {
            (
                source_metadata[rollout_id].task_name,
                int(source_metadata[rollout_id].environment_seed),
                source_metadata[rollout_id].environment_reset_index,
            )
            for rollout_id in all_source_ids
        }
    )
    allocation = allocate_development_parents(
        parents,
        num_folds=args.meta_folds,
        selection_fold=args.selection_fold,
        calibration_fold=args.calibration_fold,
        seed=args.split_seed,
    )
    fit_parents = select_parents(parents, allocation["fit"])
    selection_parents = select_parents(parents, allocation["selection"])
    calibration_parents = select_parents(parents, allocation["calibration"])
    support = select_supported_stages(
        fit_parents,
        min_successes=args.min_stage_successes,
        min_failures=args.min_stage_failures,
        requested_stages=args.stages,
    )
    stage_catalog = support["stage_catalog"]
    selected_stages = support["selected_stages"]
    task_catalog = {
        name: index for index, name in enumerate(sorted({p["task_name"] for p in parents}))
    }
    horizons = training_horizons(fit_parents, selected_stages)
    build_kwargs = {
        "selected_stages": selected_stages,
        "horizons": horizons,
        "failure_horizons": args.failure_horizons,
        "temporal_window": args.temporal_window,
    }
    fit_rows = build_inference_rows(fit_parents, **build_kwargs)
    selection_rows = build_inference_rows(selection_parents, **build_kwargs)
    calibration_rows = build_inference_rows(calibration_parents, **build_kwargs)
    available_arms, unavailable_arms = _available_arms(
        args.arms, loaded["context_available_for_all"]
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "full_parent_stage_aware_safe_development",
        "created_at": utc_now(),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "selection_manifest": split["path"],
        "feature_aggregation": args.feature_aggregation,
        "context_key": args.context_key,
        "context_available_for_all_development_parents": loaded[
            "context_available_for_all"
        ],
        "requested_arms": args.arms,
        "available_arms": available_arms,
        "unavailable_arms": unavailable_arms,
        "selected_stages": selected_stages,
        "stage_support_fit_only": support,
        "horizons_fit_only": horizons,
        "task_catalog": task_catalog,
        "stage_catalog": stage_catalog,
        "failure_horizons": args.failure_horizons,
        "prefixes": args.prefixes,
        "primary_prefixes": args.primary_prefixes,
        "development_allocation": allocation,
        "locked_opened_outer_parents": len(locked_outer_ids),
        "opened_outer_parent_ids": sorted(locked_outer_ids),
        "opened_outer_scored": False,
        "rows": {
            "fit": len(fit_rows),
            "selection": len(selection_rows),
            "calibration": len(calibration_rows),
        },
        "label_contract": {
            "completed_stage": 0,
            "terminal_failed_stage": 1,
            "future_unattempted_stage": "censored_not_materialized",
            "terminal_parent_auxiliary": True,
        },
        "arguments": vars(args),
    }
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "development_split.json",
        {
            "fit": sorted(allocation["fit"]),
            "selection": sorted(allocation["selection"]),
            "calibration": sorted(allocation["calibration"]),
            "locked_opened_outer": sorted(locked_outer_ids),
        },
    )
    if args.dry_run:
        status = {
            "status": "dry_run_valid",
            "parents": len(parents),
            "selected_stages": len(selected_stages),
            "available_arms": available_arms,
            "opened_outer_scored": False,
        }
        write_json(output / "status.json", status)
        return status

    torch = _torch()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    write_json(output / "status.json", {"status": "screening", "updated_at": utc_now()})
    candidates = []
    candidate_epochs = defaultdict(list)
    model_available = [arm for arm in available_arms if arm in MODEL_ARMS]
    for arm in model_available:
        train_rows = _model_selection_rows(fit_rows, arm)
        validation_rows = [
            row
            for row in _model_selection_rows(selection_rows, arm)
            if row.get("stage_failed") is not None
        ]
        scaler = fit_scaler(train_rows, arm)
        for regularization in args.regularizations:
            for seed in args.seeds:
                trained = train_model(
                    train_rows,
                    validation_rows,
                    arm=arm,
                    scaler=scaler,
                    task_catalog=task_catalog,
                    stage_catalog=stage_catalog,
                    failure_horizons=args.failure_horizons,
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    embedding_dim=args.embedding_dim,
                    dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    weight_decay=regularization,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    device=args.device,
                    prefixes=args.prefixes,
                    primary_prefixes=args.primary_prefixes,
                    terminal_aux_weight=args.terminal_aux_weight,
                    horizon_weight=args.horizon_weight,
                )
                row = {
                    "arm": arm,
                    "regularization": float(regularization),
                    "seed": int(seed),
                    "selection_value": float(trained["best"]["selection_value"]),
                    "best_epoch": int(trained["best"]["epoch"]),
                    "fixed_prefix_metrics": trained["best"]["metrics"],
                }
                candidates.append(row)
                candidate_epochs[(arm, float(regularization))].append(row["best_epoch"])
                print(
                    f"SCREEN arm={arm} reg={regularization:g} seed={seed} "
                    f"value={row['selection_value']:.6f} epoch={row['best_epoch']}",
                    flush=True,
                )

    # Prototype and time-only selection baselines use fit-only preprocessing.
    fit_stage_rows = [row for row in fit_rows if row.get("stage_failed") is not None]
    selection_stage_rows = [
        row for row in selection_rows if row.get("stage_failed") is not None
    ]
    prototype_selection = None
    if "prototype" in available_arms:
        prototype_scaler = fit_success_scaler(fit_stage_rows, "stage")
        prototype_runtime = fit_success_prototypes(
            fit_stage_rows, prototype_scaler, horizons, bins=args.prototype_bins
        )
        score = prototype_scores(
            selection_stage_rows, prototype_scaler, horizons, prototype_runtime
        )
        metrics = fixed_prefix_metrics(
            selection_stage_rows, score, prefixes=args.prefixes
        )
        prototype_selection = {
            "arm": "prototype",
            "selection_value": primary_prefix_score(metrics, args.primary_prefixes),
            "fixed_prefix_metrics": metrics,
        }
    time_curves_fit = fit_stage_time_curves(fit_stage_rows, selected_stages)
    time_selection_scores = stage_time_scores(selection_stage_rows, time_curves_fit)
    time_selection_metrics = fixed_prefix_metrics(
        selection_stage_rows, time_selection_scores, prefixes=args.prefixes
    )
    time_selection_value = primary_prefix_score(
        time_selection_metrics, args.primary_prefixes
    )

    aggregates = []
    selected_regularization = {}
    for arm in model_available:
        choices = []
        for regularization in args.regularizations:
            values = [
                row["selection_value"]
                for row in candidates
                if row["arm"] == arm
                and float(row["regularization"]) == float(regularization)
            ]
            choices.append(
                {
                    "arm": arm,
                    "regularization": float(regularization),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "seeds": len(values),
                }
            )
        best = max(choices, key=lambda row: (row["mean"], -row["regularization"]))
        selected_regularization[arm] = best["regularization"]
        aggregates.extend(choices)
    write_json(
        output / "selection.json",
        {
            "candidates": candidates,
            "aggregate": aggregates,
            "selected_regularization": selected_regularization,
            "prototype": prototype_selection,
            "time_only": {
                "selection_value": time_selection_value,
                "fixed_prefix_metrics": time_selection_metrics,
            },
            "opened_outer_used": False,
        },
    )

    write_json(output / "status.json", {"status": "refitting", "updated_at": utc_now()})
    refit_parents = fit_parents + selection_parents
    refit_horizons = training_horizons(refit_parents, selected_stages)
    refit_build_kwargs = {
        **build_kwargs,
        "horizons": refit_horizons,
    }
    refit_rows = build_inference_rows(refit_parents, **refit_build_kwargs)
    calibration_rows = build_inference_rows(
        calibration_parents, **refit_build_kwargs
    )
    refit_stage_rows = [row for row in refit_rows if row.get("stage_failed") is not None]
    calibration_stage_rows = [
        row for row in calibration_rows if row.get("stage_failed") is not None
    ]
    runtime_models = {}
    train_detector_scores = {}
    calibration_detector_scores = {}
    for arm in model_available:
        arm_train = _model_selection_rows(refit_rows, arm)
        arm_train_eval = [
            row
            for row in _model_selection_rows(refit_stage_rows, arm)
            if row.get("stage_failed") is not None
        ]
        arm_calibration = _model_selection_rows(calibration_stage_rows, arm)
        scaler = fit_scaler(arm_train, arm)
        regularization = selected_regularization[arm]
        refit_epoch = max(
            1,
            int(round(statistics.median(candidate_epochs[(arm, regularization)]))),
        )
        models, states = [], []
        config = None
        for seed in args.seeds:
            trained = train_model(
                arm_train,
                None,
                arm=arm,
                scaler=scaler,
                task_catalog=task_catalog,
                stage_catalog=stage_catalog,
                failure_horizons=args.failure_horizons,
                seed=seed,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=regularization,
                epochs=refit_epoch,
                patience=args.patience,
                batch_size=args.batch_size,
                device=args.device,
                prefixes=args.prefixes,
                primary_prefixes=args.primary_prefixes,
                terminal_aux_weight=args.terminal_aux_weight,
                horizon_weight=args.horizon_weight,
            )
            models.append(trained["model"])
            states.append(trained["best"]["state_dict"])
            config = trained["config"]
        train_detector_scores[arm] = _ensemble_scores(
            models, arm_train_eval, arm, scaler, config, args.device
        )
        calibration_detector_scores[arm] = _ensemble_scores(
            models, arm_calibration, arm, scaler, config, args.device
        )
        runtime_path = output / f"runtime_{arm}.pt"
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "arm": arm,
                "model_config": config,
                "state_dicts": states,
                "scaler": scaler,
                "regularization": regularization,
                "refit_epochs": refit_epoch,
                "seeds": list(args.seeds),
            },
            runtime_path,
        )
        runtime_models[arm] = runtime_path.name
        print(
            f"REFIT arm={arm} models={len(states)} epochs={refit_epoch}", flush=True
        )

    time_curves = fit_stage_time_curves(refit_stage_rows, selected_stages)
    train_detector_scores["time_only"] = stage_time_scores(
        refit_stage_rows, time_curves
    )
    calibration_detector_scores["time_only"] = stage_time_scores(
        calibration_stage_rows, time_curves
    )
    prototype_payload = None
    if "prototype" in available_arms:
        prototype_scaler = fit_success_scaler(refit_stage_rows, "stage")
        prototype_runtime = fit_success_prototypes(
            refit_stage_rows, prototype_scaler, refit_horizons, bins=args.prototype_bins
        )
        train_detector_scores["prototype"] = prototype_scores(
            refit_stage_rows, prototype_scaler, refit_horizons, prototype_runtime
        )
        calibration_detector_scores["prototype"] = prototype_scores(
            calibration_stage_rows,
            prototype_scaler,
            refit_horizons,
            prototype_runtime,
        )
        prototype_payload = {
            "scaler": prototype_scaler,
            "runtime": prototype_runtime,
        }

    train_events = group_score_trajectories(refit_stage_rows, train_detector_scores)
    calibration_events = group_score_trajectories(
        calibration_stage_rows, calibration_detector_scores
    )
    normalizers, thresholds, calibration_metrics = {}, {}, {}
    calibration_parent_outcomes = {
        parent["rollout_id"]: {
            "failed": parent["failed"],
            "task_name": parent["task_name"],
        }
        for parent in calibration_parents
    }
    detectors = sorted(train_detector_scores)
    for detector in detectors:
        normalizer = fit_score_normalizer(train_events, detector)
        normalized_calibration = apply_score_normalizer(
            calibration_events, detector, normalizer
        )
        threshold = conformal_success_threshold(
            normalized_calibration,
            detector,
            target_fpr=args.target_fpr,
            unit="parent",
        )
        result = stage_event_metrics(
            normalized_calibration,
            detector,
            threshold["threshold"],
            refit_horizons,
        )
        stage_predictions = result.pop("predictions")
        parent_result = parent_event_metrics(
            stage_predictions, calibration_parent_outcomes
        )
        parent_result.pop("predictions")
        result["parent_level"] = parent_result
        normalizers[detector] = normalizer
        thresholds[detector] = threshold
        calibration_metrics[detector] = {
            key: value for key, value in result.items()
        }

    selection_values = {
        arm: max(
            row["mean"]
            for row in aggregates
            if row["arm"] == arm
            and float(row["regularization"]) == selected_regularization[arm]
        )
        for arm in model_available
    }
    if prototype_selection is not None:
        selection_values["prototype"] = prototype_selection["selection_value"]
    stage_aware_candidates = {
        key: value for key, value in selection_values.items() if key != "terminal"
    }
    if not stage_aware_candidates:
        raise ValueError("No stage-aware primary detector is available")
    primary_detector = max(stage_aware_candidates, key=stage_aware_candidates.get)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "protocol": "full_parent_stage_aware_safe_runtime",
        "created_at": utc_now(),
        "source_dataset": str(Path(args.dataset_dir).resolve()),
        "source_selection_manifest": split["path"],
        "source_parent_ids": sorted(all_source_ids),
        "source_parent_identities": [list(value) for value in all_source_identities],
        "locked_opened_outer_parent_ids": sorted(locked_outer_ids),
        "opened_outer_scored": False,
        "tasks": sorted(task_catalog),
        "selected_stages": selected_stages,
        "task_catalog": task_catalog,
        "stage_catalog": stage_catalog,
        "feature_aggregation": args.feature_aggregation,
        "context_key": args.context_key,
        "failure_horizons": list(args.failure_horizons),
        "prefixes": list(args.prefixes),
        "primary_prefixes": list(args.primary_prefixes),
        "temporal_window": int(args.temporal_window),
        "horizons": refit_horizons,
        "model_runtime_files": runtime_models,
        "prototype": prototype_payload,
        "time_curves": time_curves,
        "normalizers": normalizers,
        "thresholds": thresholds,
        "target_fpr": float(args.target_fpr),
        "selection_values": selection_values,
        "primary_detector": primary_detector,
        "detectors": detectors,
        "calibration_parent_ids": sorted(allocation["calibration"]),
        "calibration_metrics": calibration_metrics,
        "thresholds_updated_on_prospective_test": False,
    }
    write_json(output / "runtime_bundle.json", bundle)
    write_json(
        output / "analysis.json",
        {
            "status": "complete",
            "primary_detector": primary_detector,
            "selection_values": selection_values,
            "calibration_metrics": calibration_metrics,
            "opened_outer_scored": False,
            "runtime_bundle": str(output / "runtime_bundle.json"),
            "prospective_claim": False,
            "next_step": "apply frozen bundle to a new identity-disjoint collection",
        },
    )
    status = {
        "status": "complete",
        "candidate_fits": len(candidates),
        "refits": len(model_available) * len(args.seeds),
        "primary_detector": primary_detector,
        "opened_outer_scored": False,
        "runtime_bundle": str(output / "runtime_bundle.json"),
        "updated_at": utc_now(),
    }
    write_json(output / "status.json", status)
    return status


def _load_model_runtime(bundle_root, relative_path, device):
    torch = _torch()
    runtime_path = bundle_root / relative_path
    # These runtime files are produced locally by ``develop`` and contain
    # NumPy-backed scaler metadata in addition to tensor state dictionaries.
    # PyTorch >= 2.6 defaults to ``weights_only=True``, which rejects that
    # trusted metadata. Never use this loader for an untrusted checkpoint.
    try:
        payload = torch.load(
            runtime_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:  # PyTorch releases predating the weights_only argument.
        payload = torch.load(runtime_path, map_location=device)
    models = []
    for state in payload["state_dicts"]:
        model = instantiate_from_config(payload["model_config"]).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, payload


def _read_external_score_file(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "scores.jsonl"
    if not path.is_file():
        raise ValueError(f"External score file is missing: {path}")
    records = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rollout_id = str(row["rollout_id"])
            if rollout_id in records:
                raise ValueError(
                    f"Duplicate rollout ID {rollout_id} in {path}:{line_number}"
                )
            values = np.asarray(row["scores"], dtype=np.float64)
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid external score trajectory for {rollout_id}")
            records[rollout_id] = values
    return path, records


def load_external_fixed_prefix_scores(specifications, stage_rows, parent_ids):
    """Load one or more frozen external detector ensembles.

    Each specification is ``NAME=PATH[,PATH...]``.  Multiple paths are
    averaged inference-by-inference and therefore represent a frozen seed
    ensemble.  Every file must score exactly the prospective parent set.
    """
    outputs, provenance = {}, {}
    expected_ids = {str(value) for value in parent_ids}
    for specification in specifications or ():
        if "=" not in specification:
            raise ValueError(
                "External score specification must be NAME=PATH[,PATH...]"
            )
        name, raw_paths = specification.split("=", 1)
        name = name.strip()
        paths = [value.strip() for value in raw_paths.split(",") if value.strip()]
        if not name or not paths:
            raise ValueError(f"Invalid external score specification: {specification}")
        if name in outputs:
            raise ValueError(f"Duplicate external detector name: {name}")
        members = []
        resolved_paths = []
        for raw_path in paths:
            path, records = _read_external_score_file(raw_path)
            actual_ids = set(records)
            if actual_ids != expected_ids:
                raise ValueError(
                    f"External scores for {name} do not match prospective parents: "
                    f"missing={sorted(expected_ids - actual_ids)[:5]}, "
                    f"unexpected={sorted(actual_ids - expected_ids)[:5]}"
                )
            members.append(records)
            resolved_paths.append(str(path))
        aligned = []
        for row in stage_rows:
            rollout_id = str(row["parent_rollout_id"])
            inference_index = int(row["inference_index"])
            values = []
            for records in members:
                trajectory = records[rollout_id]
                if inference_index >= len(trajectory):
                    raise ValueError(
                        f"External detector {name} has only {len(trajectory)} scores "
                        f"for {rollout_id}, but inference {inference_index} is required"
                    )
                values.append(float(trajectory[inference_index]))
            aligned.append(float(np.mean(values)))
        outputs[name] = np.asarray(aligned, dtype=np.float64)
        provenance[name] = {
            "members": len(members),
            "score_files": resolved_paths,
            "aggregation": "inference-wise arithmetic mean",
            "threshold_fitted": False,
            "use": "fixed-prefix ranking only",
        }
    return outputs, provenance


def primary_per_task_values(metrics, primary_prefixes):
    values = defaultdict(list)
    for prefix in primary_prefixes:
        row = metrics.get(str(int(prefix)), {})
        for task, value in row.get("per_task", {}).items():
            values[task].append(float(value))
    return {
        task: float(np.mean(task_values))
        for task, task_values in sorted(values.items())
        if task_values
    }


def evaluate(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "initializing", "updated_at": utc_now()})
    bundle_path = Path(args.runtime_bundle).resolve()
    bundle_root = bundle_path.parent
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("status") != "frozen":
        raise ValueError("Runtime bundle is not frozen")
    locked_outer_ids = set(bundle["locked_opened_outer_parent_ids"])
    opened_outer_mode = bool(args.opened_outer)
    loaded = load_parent_sequences(
        args.dataset_dir,
        parent_ids=locked_outer_ids if opened_outer_mode else None,
        tasks=bundle["tasks"],
        aggregation=bundle["feature_aggregation"],
        context_key=bundle.get("context_key"),
        require_context="context" in bundle["model_runtime_files"],
    )
    parents = loaded["parents"]
    test_ids = {parent["rollout_id"] for parent in parents}
    if opened_outer_mode:
        if test_ids != locked_outer_ids:
            raise ValueError(
                "Opened-outer evaluation must load exactly the frozen locked IDs: "
                f"missing={sorted(locked_outer_ids - test_ids)[:5]}, "
                f"unexpected={sorted(test_ids - locked_outer_ids)[:5]}"
            )
        development_ids_disjoint = False
        identity_disjoint = False
        evaluation_scope = "retrospective_locked_opened_outer"
        prospective_claim = False
    else:
        source_ids = set(bundle["source_parent_ids"]) | locked_outer_ids
        overlap = sorted(source_ids & test_ids)
        if overlap:
            raise ValueError(
                "Prospective rollout IDs overlap development/opened outer IDs: "
                + ", ".join(overlap[:5])
            )
        source_identities = {
            tuple(value) for value in bundle["source_parent_identities"]
        }
        test_identities = {parent_identity(parent) for parent in parents}
        identity_overlap = source_identities & test_identities
        if identity_overlap:
            raise ValueError(
                "Prospective task/seed/reset identities overlap development: "
                + repr(sorted(identity_overlap)[:3])
            )
        development_ids_disjoint = True
        identity_disjoint = True
        evaluation_scope = "prospective_identity_disjoint"
        prospective_claim = True
    rows = build_inference_rows(
        parents,
        selected_stages=bundle["selected_stages"],
        horizons=bundle["horizons"],
        failure_horizons=bundle["failure_horizons"],
        temporal_window=bundle["temporal_window"],
    )
    stage_rows = [row for row in rows if row.get("stage_failed") is not None]
    detector_scores = {}
    for arm, relative_path in bundle["model_runtime_files"].items():
        models, runtime = _load_model_runtime(bundle_root, relative_path, args.device)
        arm_rows = model_rows(stage_rows, arm)
        if len(arm_rows) != len(stage_rows):
            raise ValueError(f"Prospective rows are incomplete for arm {arm}")
        detector_scores[arm] = _ensemble_scores(
            models,
            stage_rows,
            arm,
            runtime["scaler"],
            runtime["model_config"],
            args.device,
        )
    curves = {
        stage: np.asarray(values, dtype=np.float64)
        for stage, values in bundle["time_curves"].items()
    }
    detector_scores["time_only"] = stage_time_scores(stage_rows, curves)
    if bundle.get("prototype") is not None:
        prototype = bundle["prototype"]
        prototype_runtime = {
            "bins": prototype["runtime"]["bins"],
            "prototypes": {
                key: np.asarray(value, dtype=np.float32)
                for key, value in prototype["runtime"]["prototypes"].items()
            },
            "stage_fallback": {
                key: np.asarray(value, dtype=np.float32)
                for key, value in prototype["runtime"]["stage_fallback"].items()
            },
        }
        scaler = {
            "mean": np.asarray(prototype["scaler"]["mean"], dtype=np.float32),
            "scale": np.asarray(prototype["scaler"]["scale"], dtype=np.float32),
        }
        detector_scores["prototype"] = prototype_scores(
            stage_rows, scaler, bundle["horizons"], prototype_runtime
        )
    external_scores, external_provenance = load_external_fixed_prefix_scores(
        args.external_scores,
        stage_rows,
        {parent["rollout_id"] for parent in parents},
    )
    detector_scores.update(external_scores)
    prefixes = bundle.get("prefixes", list(DEFAULT_PREFIXES))
    primary_prefixes = bundle.get(
        "primary_prefixes", list(DEFAULT_PRIMARY_PREFIXES)
    )
    fixed_prefix = {
        detector: fixed_prefix_metrics(stage_rows, scores, prefixes=prefixes)
        for detector, scores in sorted(detector_scores.items())
    }
    primary_prefix_values = {
        detector: primary_prefix_score(metrics, primary_prefixes)
        for detector, metrics in fixed_prefix.items()
    }
    primary_per_task = {
        detector: primary_per_task_values(metrics, primary_prefixes)
        for detector, metrics in fixed_prefix.items()
    }
    events = group_score_trajectories(stage_rows, detector_scores)
    analyses = {}
    predictions_by_detector = {}
    parent_predictions_by_detector = {}
    prospective_parent_outcomes = {
        parent["rollout_id"]: {
            "failed": parent["failed"],
            "task_name": parent["task_name"],
        }
        for parent in parents
    }
    for detector in bundle["detectors"]:
        normalizer = bundle["normalizers"][detector]
        normalized = apply_score_normalizer(events, detector, normalizer)
        threshold = bundle["thresholds"][detector]["threshold"]
        result = stage_event_metrics(
            normalized, detector, threshold, bundle["horizons"]
        )
        predictions_by_detector[detector] = result.pop("predictions")
        parent_result = parent_event_metrics(
            predictions_by_detector[detector], prospective_parent_outcomes
        )
        parent_predictions_by_detector[detector] = parent_result.pop("predictions")
        result["parent_level"] = parent_result
        analyses[detector] = result
    bootstrap = {}
    for detector in bundle["detectors"]:
        if detector == "time_only":
            continue
        bootstrap[detector] = paired_parent_bootstrap(
            parent_predictions_by_detector,
            detector,
            baseline="time_only",
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
    primary = bundle["primary_detector"]
    primary_metrics = analyses[primary]["parent_level"]
    time_metrics = analyses["time_only"]["parent_level"]
    primary_bootstrap = bootstrap[primary]
    criteria = {
        "adjusted_detection_at_least_0p05_earlier": (
            primary_bootstrap["point"] <= -0.05
        ),
        "bootstrap_adjusted_detection_favors_stage_aware": (
            primary_bootstrap["ci95"][1] < 0.0
        ),
        "tpr_not_more_than_0p05_below_time": (
            primary_metrics["failed_stage_tpr"]
            >= time_metrics["failed_stage_tpr"] - 0.05
        ),
        "successful_parent_fpr_at_most_target_plus_0p02": (
            primary_metrics["successful_parent_fpr"] <= bundle["target_fpr"] + 0.02
        ),
    }
    criteria["all_pass"] = all(criteria.values())
    terminal_value = primary_prefix_values.get("terminal")
    time_value = primary_prefix_values.get("time_only")
    primary_value = primary_prefix_values[primary]
    task_primary = primary_per_task[primary]
    task_terminal = primary_per_task.get("terminal", {})
    task_time = primary_per_task.get("time_only", {})
    common_terminal = sorted(set(task_primary) & set(task_terminal))
    common_time = sorted(set(task_primary) & set(task_time))
    primary_over_terminal = [
        task for task in common_terminal if task_primary[task] > task_terminal[task]
    ]
    primary_over_time = [
        task for task in common_time if task_primary[task] > task_time[task]
    ]
    ranking_bootstrap = {}
    ranking_baselines = ["terminal", "time_only", *sorted(external_scores)]
    for baseline in ranking_baselines:
        if baseline == primary or baseline not in detector_scores:
            continue
        ranking_bootstrap[baseline] = paired_fixed_prefix_bootstrap(
            stage_rows,
            detector_scores,
            primary,
            baseline=baseline,
            prefixes=prefixes,
            primary_prefixes=primary_prefixes,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
    ranking_criteria = {
        "primary_prefix_task_stage_macro_roc_auc_at_least_0p60": (
            primary_value >= 0.60
        ),
        "primary_minus_terminal_at_least_0p05": (
            terminal_value is not None and primary_value - terminal_value >= 0.05
        ),
        "primary_beats_terminal_on_at_least_4_of_5_tasks": (
            len(common_terminal) >= 5 and len(primary_over_terminal) >= 4
        ),
        "primary_beats_time_on_at_least_4_of_5_tasks": (
            len(common_time) >= 5 and len(primary_over_time) >= 4
        ),
    }
    ranking_criteria["all_pass"] = all(ranking_criteria.values())
    ranking_confirmation = {
        "primary_detector": primary,
        "prefixes": list(prefixes),
        "primary_prefixes": list(primary_prefixes),
        "primary_values": primary_prefix_values,
        "primary_per_task": primary_per_task,
        "paired_task_parent_bootstrap": ranking_bootstrap,
        "primary_minus_terminal": (
            None if terminal_value is None else primary_value - terminal_value
        ),
        "primary_minus_time_only": (
            None if time_value is None else primary_value - time_value
        ),
        "tasks_primary_beats_terminal": primary_over_terminal,
        "tasks_primary_beats_time_only": primary_over_time,
        "criteria": ranking_criteria,
    }
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": (
            "retrospective_locked_outer_full_parent_stage_aware_safe"
            if opened_outer_mode
            else "prospective_full_parent_stage_aware_safe"
        ),
        "evaluation_scope": evaluation_scope,
        "prospective_claim": prospective_claim,
        "runtime_bundle": str(bundle_path),
        "prospective_dataset": str(Path(args.dataset_dir).resolve()),
        "parents": len(parents),
        "tasks": sorted({parent["task_name"] for parent in parents}),
        "stage_events": len(events),
        "unobserved_failed_stage_parents": len(
            loaded["unobserved_failed_stage_parent_ids"]
        ),
        "unobserved_failed_stage_parent_ids": loaded[
            "unobserved_failed_stage_parent_ids"
        ],
        "development_ids_disjoint": development_ids_disjoint,
        "development_seed_reset_identities_disjoint": identity_disjoint,
        "locked_opened_outer_exact": opened_outer_mode,
        "thresholds_updated_on_test": False,
        "primary_detector": primary,
        "detectors": analyses,
        "fixed_prefix_metrics": fixed_prefix,
        "ranking_confirmation": ranking_confirmation,
        "external_fixed_prefix_detectors": external_provenance,
        "paired_bootstrap": bootstrap,
        "preregistered_success_criteria": criteria,
    }
    write_json(output / "analysis.json", analysis)
    with (output / "stage_event_predictions.jsonl").open("w") as stream:
        for detector, rows_for_detector in sorted(predictions_by_detector.items()):
            for row in rows_for_detector:
                stream.write(json.dumps(json_value({**row, "detector": detector}), sort_keys=True) + "\n")
    with (output / "parent_predictions.jsonl").open("w") as stream:
        for detector, rows_for_detector in sorted(parent_predictions_by_detector.items()):
            for row in rows_for_detector:
                stream.write(json.dumps(json_value({**row, "detector": detector}), sort_keys=True) + "\n")
    status = {
        "status": "complete",
        "parents": len(parents),
        "stage_events": len(events),
        "unobserved_failed_stage_parents": len(
            loaded["unobserved_failed_stage_parent_ids"]
        ),
        "primary_detector": primary,
        "all_success_criteria_passed": criteria["all_pass"],
        "all_ranking_confirmation_criteria_passed": ranking_criteria["all_pass"],
        "thresholds_updated_on_test": False,
        "updated_at": utc_now(),
    }
    write_json(output / "status.json", status)
    return status


def common_parser(parser):
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    split_parser = common_parser(subparsers.add_parser("split"))
    split_parser.add_argument("--tasks", nargs="+")
    split_parser.add_argument("--feature-aggregation", default="first")
    split_parser.add_argument("--context-key", default="observation_state_history")
    split_parser.add_argument("--require-context", action="store_true")
    split_parser.add_argument("--train-fraction", type=float, default=0.68)
    split_parser.add_argument("--split-seed", type=int, default=0)
    develop_parser = common_parser(subparsers.add_parser("develop"))
    develop_parser.add_argument("--selection-manifest", required=True)
    develop_parser.add_argument("--tasks", nargs="+")
    develop_parser.add_argument("--stages", nargs="+")
    develop_parser.add_argument("--feature-aggregation", default="first")
    develop_parser.add_argument("--context-key", default="observation_state_history")
    develop_parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    develop_parser.add_argument("--failure-horizons", nargs="+", type=int, default=list(DEFAULT_FAILURE_HORIZONS))
    develop_parser.add_argument("--prefixes", nargs="+", type=int, default=list(DEFAULT_PREFIXES))
    develop_parser.add_argument("--primary-prefixes", nargs="+", type=int, default=list(DEFAULT_PRIMARY_PREFIXES))
    develop_parser.add_argument("--meta-folds", type=int, default=5)
    develop_parser.add_argument("--selection-fold", type=int, default=1)
    develop_parser.add_argument("--calibration-fold", type=int, default=0)
    develop_parser.add_argument("--split-seed", type=int, default=0)
    develop_parser.add_argument("--min-stage-successes", type=int, default=3)
    develop_parser.add_argument("--min-stage-failures", type=int, default=2)
    develop_parser.add_argument("--temporal-window", type=int, default=4)
    develop_parser.add_argument("--hidden-dim", type=int, default=128)
    develop_parser.add_argument("--embedding-dim", type=int, default=16)
    develop_parser.add_argument("--dropout", type=float, default=0.1)
    develop_parser.add_argument("--learning-rate", type=float, default=3e-4)
    develop_parser.add_argument("--regularizations", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3, 1e-2])
    develop_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    develop_parser.add_argument("--epochs", type=int, default=500)
    develop_parser.add_argument("--patience", type=int, default=50)
    develop_parser.add_argument("--batch-size", type=int, default=256)
    develop_parser.add_argument("--terminal-aux-weight", type=float, default=0.2)
    develop_parser.add_argument("--horizon-weight", type=float, default=0.25)
    develop_parser.add_argument("--prototype-bins", type=int, default=4)
    develop_parser.add_argument("--target-fpr", type=float, default=0.05)
    develop_parser.add_argument("--dry-run", action="store_true")

    evaluate_parser = common_parser(subparsers.add_parser("evaluate"))
    evaluate_parser.add_argument("--runtime-bundle", required=True)
    evaluate_parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    evaluate_parser.add_argument("--bootstrap-seed", type=int, default=0)
    evaluate_parser.add_argument(
        "--opened-outer",
        action="store_true",
        help=(
            "Score exactly the bundle's previously opened locked outer IDs. "
            "This is explicitly retrospective and never a prospective claim."
        ),
    )
    evaluate_parser.add_argument(
        "--external-scores",
        nargs="*",
        default=[],
        metavar="NAME=PATH[,PATH...]",
        help=(
            "Frozen external score trajectories used only for fixed-prefix "
            "ranking. Multiple paths are averaged as a seed ensemble."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    try:
        if args.phase == "split":
            result = split_raw_parents(args)
        elif args.phase == "develop":
            result = develop(args)
        else:
            result = evaluate(args)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "status.json",
            {
                "status": "failed",
                "phase": args.phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "updated_at": utc_now(),
            },
        )
        raise
    print(json.dumps(json_value(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
