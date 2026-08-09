"""Run the locked-test-free stage-adapter Subtask-SAFE diagnostic.

The command uses only parents assigned to the outer training split.  Those
parents are divided into meta-fit, selection, and untouched diagnostic sets
before causal prefixes are constructed.  The outer test is recorded in the
audit but is never scored by this command.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from .subtask_stage_adapter import (
        ARCHITECTURES,
        DEFAULT_LANDMARKS,
        DEFAULT_PRIMARY_LANDMARKS,
        DETECTORS,
        OBJECTIVES,
        balance_bce_rows,
        build_stage_landmark_rows,
        continuation_gate,
        filter_stages,
        fit_feature_scaler,
        fit_time_risk,
        landmark_metrics,
        make_stage_model,
        parent_bootstrap_delta,
        parent_id,
        primary_macro_auc,
        rows_to_arrays,
        score_full_stage_trajectories,
        score_landmark_rows,
        select_parent_rollouts,
        select_stage_thresholds,
        select_supported_stages,
        stage_event_metrics,
        temporal_contrastive_loss,
        three_way_parent_split,
        training_stage_horizons,
        validate_landmarks,
        validate_stage_identity,
    )
    from .train_seen_tasks import (
        OFFICIAL_SAFE_COMMIT,
        filter_aligned_task_type,
        load_env_records,
        load_outer_split_ids,
        make_config,
        make_manifest_split,
        resolve_task_type_selection,
        verify_safe_repo,
    )
except ImportError:
    from subtask_stage_adapter import (
        ARCHITECTURES,
        DEFAULT_LANDMARKS,
        DEFAULT_PRIMARY_LANDMARKS,
        DETECTORS,
        OBJECTIVES,
        balance_bce_rows,
        build_stage_landmark_rows,
        continuation_gate,
        filter_stages,
        fit_feature_scaler,
        fit_time_risk,
        landmark_metrics,
        make_stage_model,
        parent_bootstrap_delta,
        parent_id,
        primary_macro_auc,
        rows_to_arrays,
        score_full_stage_trajectories,
        score_landmark_rows,
        select_parent_rollouts,
        select_stage_thresholds,
        select_supported_stages,
        stage_event_metrics,
        temporal_contrastive_loss,
        three_way_parent_split,
        training_stage_horizons,
        validate_landmarks,
        validate_stage_identity,
    )
    from train_seen_tasks import (
        OFFICIAL_SAFE_COMMIT,
        filter_aligned_task_type,
        load_env_records,
        load_outer_split_ids,
        make_config,
        make_manifest_split,
        resolve_task_type_selection,
        verify_safe_repo,
    )


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return [json_value(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(json_value(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple, set))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def model_state_cpu(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def pipeline_key(architecture, objective, detector, regularization):
    return (
        str(architecture),
        str(objective),
        str(detector),
        float(regularization),
    )


def pipeline_dict(key):
    return {
        "architecture": key[0],
        "objective": key[1],
        "detector": key[2],
        "regularization": float(key[3]),
    }


def objective_loss(
    torch,
    logits,
    targets,
    weights,
    rows,
    *,
    objective,
    rank_margin,
    onset_margin,
    onset_weight,
):
    if objective == "bce":
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        value = (losses * weights).sum() / weights.sum()
        return value, {"bce": value}
    if objective == "temporal_contrastive":
        return temporal_contrastive_loss(
            torch,
            logits,
            rows,
            rank_margin=rank_margin,
            onset_margin=onset_margin,
            onset_weight=onset_weight,
        )
    raise ValueError(f"Unknown objective {objective!r}")


def train_candidate(
    fit_rows,
    selection_rows,
    scaler,
    *,
    architecture,
    objective,
    residual_time,
    num_stages,
    seed,
    hidden_dim,
    adapter_dim,
    dropout,
    learning_rate,
    weight_decay,
    epochs,
    patience,
    rank_margin,
    onset_margin,
    onset_weight,
    device,
):
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    fit_arrays = rows_to_arrays(fit_rows, scaler, residual_time=residual_time)
    selection_arrays = rows_to_arrays(
        selection_rows,
        scaler,
        residual_time=residual_time,
    )
    model = make_stage_model(
        torch,
        architecture=architecture,
        input_dim=fit_arrays[0].shape[1],
        num_stages=num_stages,
        hidden_dim=hidden_dim,
        adapter_dim=adapter_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    def tensors(arrays):
        features, targets, weights, stage_indices, offsets = arrays
        return (
            torch.as_tensor(features, device=device),
            torch.as_tensor(targets, device=device),
            torch.as_tensor(weights, device=device),
            torch.as_tensor(stage_indices, device=device),
            torch.as_tensor(offsets, device=device),
        )

    fit_tensors = tensors(fit_arrays)
    selection_tensors = tensors(selection_arrays)
    history = []
    best = None
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        features, targets, weights, stages, offsets = fit_tensors
        logits = model(features, stages) + offsets
        loss, parts = objective_loss(
            torch,
            logits,
            targets,
            weights,
            fit_rows,
            objective=objective,
            rank_margin=rank_margin,
            onset_margin=onset_margin,
            onset_weight=onset_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Stage-adapter training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            features, targets, weights, stages, offsets = selection_tensors
            logits = model(features, stages) + offsets
            selection_loss, selection_parts = objective_loss(
                torch,
                logits,
                targets,
                weights,
                selection_rows,
                objective=objective,
                rank_margin=rank_margin,
                onset_margin=onset_margin,
                onset_weight=onset_weight,
            )
        record = {
            "epoch": epoch + 1,
            "fit_loss": float(loss.detach().cpu()),
            "selection_loss": float(selection_loss.detach().cpu()),
        }
        for prefix, values in (("fit", parts), ("selection", selection_parts)):
            for name, value in values.items():
                if hasattr(value, "detach"):
                    value = float(value.detach().cpu())
                record[f"{prefix}_{name}"] = value
        history.append(record)
        selection_value = float(selection_loss.detach().cpu())
        if best is None or selection_value < best["selection_loss"] - 1e-7:
            best = {
                "epoch": epoch + 1,
                "selection_loss": selection_value,
                "state_dict": copy.deepcopy(model_state_cpu(model)),
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best is None:
        raise RuntimeError("Stage-adapter training produced no checkpoint")
    model.load_state_dict(best["state_dict"])
    return (
        model,
        history,
        {
            "epoch": best["epoch"],
            "selection_loss": best["selection_loss"],
        },
    )


def predict_rows(model, rows, scaler, *, residual_time, device, batch_size=1024):
    import torch

    arrays = rows_to_arrays(rows, scaler, residual_time=residual_time)
    features, _, _, stages, offsets = arrays
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            stop = start + int(batch_size)
            values = torch.as_tensor(features[start:stop], device=device)
            stage_values = torch.as_tensor(stages[start:stop], device=device)
            offset_values = torch.as_tensor(offsets[start:stop], device=device)
            probabilities = torch.sigmoid(model(values, stage_values) + offset_values)
            output.extend(probabilities.detach().cpu().numpy().tolist())
    return np.asarray(output, dtype=np.float64)


def make_predictor(model, *, device):
    import torch

    def predict(features, stages, offsets):
        output = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(features), 1024):
                stop = start + 1024
                values = torch.as_tensor(features[start:stop], device=device)
                stage_values = torch.as_tensor(stages[start:stop], device=device)
                offset_values = torch.as_tensor(offsets[start:stop], device=device)
                probabilities = torch.sigmoid(
                    model(values, stage_values) + offset_values
                )
                output.extend(probabilities.detach().cpu().numpy().tolist())
        return np.asarray(output, dtype=np.float64)

    return predict


def selection_score(rows, predictions, primary_landmarks):
    scored = score_landmark_rows(rows, predictions)
    metrics = landmark_metrics(scored)
    candidate, support = primary_macro_auc(metrics, "candidate", primary_landmarks)
    time_only, time_support = primary_macro_auc(
        metrics,
        "time_only",
        primary_landmarks,
    )
    if support != time_support:
        raise AssertionError("Selection candidate/time supports differ")
    return {
        "primary_auc": candidate,
        "time_only_primary_auc": time_only,
        "delta_over_time": candidate - time_only,
        "supported_stage_landmarks": support,
    }


def ensemble_landmark_predictions(predictions_by_seed):
    values = np.stack([np.asarray(value) for value in predictions_by_seed])
    return np.mean(values, axis=0)


def ensemble_trajectories(scored_by_seed):
    reference = scored_by_seed[0]
    output = []
    for index, item in enumerate(reference):
        key = (item["parent_rollout_id"], item["segment_id"], item["stage_name"])
        candidates = []
        for scored in scored_by_seed:
            current = scored[index]
            current_key = (
                current["parent_rollout_id"],
                current["segment_id"],
                current["stage_name"],
            )
            if current_key != key:
                raise AssertionError(
                    "Model seeds scored stage trajectories differently"
                )
            candidates.append(np.asarray(current["trajectories"]["candidate"]))
        clone = copy.deepcopy(item)
        clone["trajectories"]["candidate"] = np.mean(
            np.stack(candidates),
            axis=0,
        )
        output.append(clone)
    return output


def aggregate_candidates(audit):
    grouped = {}
    for row in audit:
        key = pipeline_key(
            row["architecture"],
            row["objective"],
            row["detector"],
            row["regularization"],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(grouped.items()):
        output.append(
            {
                **pipeline_dict(key),
                "seeds": len(values),
                "selection_primary_auc_mean": float(
                    np.mean([row["selection_primary_auc"] for row in values])
                ),
                "selection_primary_auc_std": float(
                    np.std([row["selection_primary_auc"] for row in values])
                ),
                "selection_delta_over_time_mean": float(
                    np.mean([row["selection_delta_over_time"] for row in values])
                ),
                "selection_loss_mean": float(
                    np.mean([row["selection_loss"] for row in values])
                ),
            }
        )
    return output


def run_experiment(args):
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"status": "initializing"})

    verify_safe_repo(args.safe_repo)
    import torch
    from failure_prob.data.pizero import load_rollouts_from_root
    from failure_prob.utils.random import seed_everything

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested {device}, but CUDA is unavailable")
    landmarks = tuple(float(value) for value in args.landmarks)
    primary = tuple(float(value) for value in args.primary_landmarks)
    landmarks = validate_landmarks(landmarks)
    primary = validate_landmarks(primary)
    if not set(primary).issubset(landmarks):
        raise ValueError("Primary landmarks must be included in --landmarks")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if not args.architectures or any(
        value not in ARCHITECTURES for value in args.architectures
    ):
        raise ValueError(f"Architectures must be drawn from {ARCHITECTURES}")
    if not args.objectives or any(value not in OBJECTIVES for value in args.objectives):
        raise ValueError(f"Objectives must be drawn from {OBJECTIVES}")
    if not args.detectors or any(value not in DETECTORS for value in args.detectors):
        raise ValueError(f"Detectors must be drawn from {DETECTORS}")
    regularizations = tuple(float(value) for value in args.regularizations)
    if not regularizations or any(value < 0 for value in regularizations):
        raise ValueError("Regularizations must be non-negative")

    cfg = make_config(
        args.export_dir,
        "indep",
        0,
        args.epochs,
        "cpu",
        {
            "horizon_selector": args.horizon_selector,
            "diffusion_selector": args.diffusion_selector,
            "learning_rate": args.learning_rate,
            "lambda_reg": 0.0,
        },
    )
    seed_everything(0)
    source_rollouts = load_rollouts_from_root(Path(args.export_dir), cfg)
    source_env_records = load_env_records(args.export_dir)
    task_selection = resolve_task_type_selection(args.export_dir, "composite")
    all_rollouts, _, identity = filter_aligned_task_type(
        source_rollouts,
        source_env_records,
        task_selection,
    )
    outer_manifest = load_outer_split_ids(args.selection_manifest)
    if outer_manifest is None or outer_manifest["split_unit"] != "parent_rollout":
        raise ValueError(
            "Stage-adapter Subtask-SAFE requires a parent_rollout selection manifest"
        )
    outer_train, outer_test, outer_counts = make_manifest_split(
        all_rollouts,
        identity,
        outer_manifest,
    )
    outer_train_parents = {parent_id(item, identity) for item in outer_train}
    outer_test_parents = {parent_id(item, identity) for item in outer_test}
    if outer_train_parents & outer_test_parents:
        raise AssertionError("Outer train and test parents overlap")

    experiment_outer_train = outer_train
    if args.stages is not None:
        stage_identity = validate_stage_identity(outer_train, identity)
        requested = {str(value) for value in args.stages}
        unknown = sorted(requested - set(stage_identity["name_to_id"]))
        if unknown:
            raise ValueError(
                "Requested stages are absent from outer training: " + ", ".join(unknown)
            )
        experiment_outer_train = filter_stages(outer_train, identity, requested)

    meta = three_way_parent_split(
        experiment_outer_train,
        identity,
        num_folds=args.meta_folds,
        selection_fold=args.selection_fold,
        diagnostic_fold=args.diagnostic_fold,
        seed=args.meta_split_seed,
    )
    fit_rollouts = select_parent_rollouts(
        experiment_outer_train,
        identity,
        meta["fit"],
    )
    selection_rollouts = select_parent_rollouts(
        experiment_outer_train,
        identity,
        meta["selection"],
    )
    diagnostic_rollouts = select_parent_rollouts(
        experiment_outer_train,
        identity,
        meta["diagnostic"],
    )
    stage_support = select_supported_stages(
        fit_rollouts,
        identity,
        requested_stages=args.stages,
        min_successes=args.min_stage_successes,
        min_failures=args.min_stage_failures,
    )
    selected_stages = stage_support["selected_stages"]
    catalog = stage_support["catalog"]
    fit_rollouts = filter_stages(fit_rollouts, identity, selected_stages)
    selection_rollouts = filter_stages(
        selection_rollouts,
        identity,
        selected_stages,
    )
    diagnostic_rollouts = filter_stages(
        diagnostic_rollouts,
        identity,
        selected_stages,
    )
    horizons = training_stage_horizons(
        fit_rollouts,
        identity,
        quantile=args.stage_horizon_quantile,
    )
    time_curves = fit_time_risk(fit_rollouts, horizons, prior=args.time_prior)
    fit_rows, fit_excluded = build_stage_landmark_rows(
        fit_rollouts,
        identity,
        stage_catalog=catalog,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=landmarks,
        window=args.temporal_window,
        split="fit",
    )
    selection_rows, selection_excluded = build_stage_landmark_rows(
        selection_rollouts,
        identity,
        stage_catalog=catalog,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=landmarks,
        window=args.temporal_window,
        split="selection",
    )
    diagnostic_rows, diagnostic_excluded = build_stage_landmark_rows(
        diagnostic_rollouts,
        identity,
        stage_catalog=catalog,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=landmarks,
        window=args.temporal_window,
        split="diagnostic",
    )
    bce_fit_rows, bce_support, bce_excluded = balance_bce_rows(
        fit_rows,
        seed=args.balance_seed,
    )
    scaler = fit_feature_scaler(fit_rows)

    protocol = {
        "schema_version": 1,
        "protocol": "subtask_stage_adapter_three_way_parent_diagnostic",
        "official_safe_commit": OFFICIAL_SAFE_COMMIT,
        "arguments": vars(args),
        "subtask_safe": True,
        "semantic_stage_labels": True,
        "split_before_prefix_expansion": True,
        "outer_test_scored": False,
        "outer_test_used_for_selection": False,
        "selection_manifest": outer_manifest["path"],
        "outer_counts": outer_counts,
        "outer_train_parents": len(outer_train_parents),
        "locked_outer_test_parents": len(outer_test_parents),
        "locked_outer_test_parent_ids": sorted(outer_test_parents),
        "meta_split": meta,
        "stage_support": stage_support,
        "stage_horizons_fit_success_quantile": {
            "quantile": float(args.stage_horizon_quantile),
            "horizons": horizons,
        },
        "landmarks": list(landmarks),
        "primary_landmarks": list(primary),
        "prefix_counts": {
            "fit": len(fit_rows),
            "fit_bce_balanced": len(bce_fit_rows),
            "selection": len(selection_rows),
            "diagnostic": len(diagnostic_rows),
        },
        "excluded_counts": {
            "fit_before_landmark": len(fit_excluded),
            "fit_bce_balance": len(bce_excluded),
            "selection_before_landmark": len(selection_excluded),
            "diagnostic_before_landmark": len(diagnostic_excluded),
        },
        "bce_training_support": bce_support,
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "split_manifest.json", protocol)
    write_json(
        output / "status.json",
        {
            "status": "running",
            "completed_candidates": 0,
        },
    )

    audit = []
    states = {}
    histories = {}
    candidates = list(
        itertools.product(
            args.architectures,
            args.objectives,
            args.detectors,
            regularizations,
        )
    )
    total = len(candidates) * len(args.seeds)
    completed = 0
    for seed in args.seeds:
        for architecture, objective, detector, regularization in candidates:
            key = pipeline_key(architecture, objective, detector, regularization)
            residual_time = detector == "stage_safe_time"
            training_rows = bce_fit_rows if objective == "bce" else fit_rows
            print(
                "TRAIN "
                f"seed={seed} architecture={architecture} objective={objective} "
                f"detector={detector} weight_decay={regularization:g}",
                flush=True,
            )
            model, history, checkpoint = train_candidate(
                training_rows,
                selection_rows,
                scaler,
                architecture=architecture,
                objective=objective,
                residual_time=residual_time,
                num_stages=len(catalog),
                seed=seed,
                hidden_dim=args.hidden_dim,
                adapter_dim=args.adapter_dim,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=regularization,
                epochs=args.epochs,
                patience=args.patience,
                rank_margin=args.rank_margin,
                onset_margin=args.onset_margin,
                onset_weight=args.onset_weight,
                device=device,
            )
            predictions = predict_rows(
                model,
                selection_rows,
                scaler,
                residual_time=residual_time,
                device=device,
            )
            score = selection_score(selection_rows, predictions, primary)
            row = {
                "seed": int(seed),
                **pipeline_dict(key),
                "selection_primary_auc": score["primary_auc"],
                "selection_time_only_primary_auc": score["time_only_primary_auc"],
                "selection_delta_over_time": score["delta_over_time"],
                "supported_stage_landmarks": score["supported_stage_landmarks"],
                "best_epoch": checkpoint["epoch"],
                "selection_loss": checkpoint["selection_loss"],
            }
            audit.append(row)
            write_csv(output / "selection_audit.csv", audit)
            states[(int(seed), key)] = model_state_cpu(model)
            histories[(int(seed), key)] = history
            completed += 1
            write_json(
                output / "status.json",
                {
                    "status": "running",
                    "completed_candidates": completed,
                    "total_candidates": total,
                    "last_candidate": row,
                },
            )
            print(
                "COMPLETE "
                f"seed={seed} architecture={architecture} objective={objective} "
                f"detector={detector} auc={score['primary_auc']:.6f} "
                f"delta_time={score['delta_over_time']:.6f}",
                flush=True,
            )

    aggregate = aggregate_candidates(audit)
    selected_aggregate = max(
        aggregate,
        key=lambda row: (
            row["selection_primary_auc_mean"],
            row["selection_delta_over_time_mean"],
            -row["regularization"],
        ),
    )
    selected_key = pipeline_key(
        selected_aggregate["architecture"],
        selected_aggregate["objective"],
        selected_aggregate["detector"],
        selected_aggregate["regularization"],
    )
    for row in audit:
        row["selected_pipeline"] = (
            pipeline_key(
                row["architecture"],
                row["objective"],
                row["detector"],
                row["regularization"],
            )
            == selected_key
        )
    write_csv(output / "selection_audit.csv", audit)
    write_csv(output / "selection_aggregate.csv", aggregate)

    diagnostic_predictions = []
    selection_trajectories = []
    diagnostic_trajectories = []
    runtime_root = output / "runtime"
    runtime_root.mkdir()
    for seed in args.seeds:
        model = make_stage_model(
            torch,
            architecture=selected_key[0],
            input_dim=len(scaler["mean"]),
            num_stages=len(catalog),
            hidden_dim=args.hidden_dim,
            adapter_dim=args.adapter_dim,
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(states[(int(seed), selected_key)])
        residual_time = selected_key[2] == "stage_safe_time"
        diagnostic_predictions.append(
            predict_rows(
                model,
                diagnostic_rows,
                scaler,
                residual_time=residual_time,
                device=device,
            )
        )
        predictor = make_predictor(model, device=device)
        selection_trajectories.append(
            score_full_stage_trajectories(
                selection_rollouts,
                identity,
                stage_catalog=catalog,
                horizons=horizons,
                time_curves=time_curves,
                scaler=scaler,
                predict=predictor,
                residual_time=residual_time,
                window=args.temporal_window,
            )
        )
        diagnostic_trajectories.append(
            score_full_stage_trajectories(
                diagnostic_rollouts,
                identity,
                stage_catalog=catalog,
                horizons=horizons,
                time_curves=time_curves,
                scaler=scaler,
                predict=predictor,
                residual_time=residual_time,
                window=args.temporal_window,
            )
        )
        seed_root = runtime_root / f"seed{seed}"
        seed_root.mkdir()
        torch.save(
            {
                "state_dict": states[(int(seed), selected_key)],
                "input_dim": len(scaler["mean"]),
                "num_stages": len(catalog),
                "hidden_dim": int(args.hidden_dim),
                "adapter_dim": int(args.adapter_dim),
                "dropout": float(args.dropout),
            },
            seed_root / "model.pt",
        )
        write_json(
            seed_root / "runtime.json",
            {
                "schema_version": 1,
                "selected_pipeline": pipeline_dict(selected_key),
                "seed": int(seed),
                "stage_catalog": catalog,
                "stage_horizons": horizons,
                "stage_time_risk_curves": time_curves,
                "feature_blocks": ["current", "delta", "recent_mean", "recent_slope"],
                "temporal_window": int(args.temporal_window),
                "feature_mean": scaler["mean"],
                "feature_scale": scaler["scale"],
                "history": histories[(int(seed), selected_key)],
            },
        )

    diagnostic_ensemble = ensemble_landmark_predictions(diagnostic_predictions)
    diagnostic_scored_rows = score_landmark_rows(
        diagnostic_rows,
        diagnostic_ensemble,
    )
    diagnostic_landmark_metrics = landmark_metrics(diagnostic_scored_rows)
    write_csv(output / "diagnostic_landmark_metrics.csv", diagnostic_landmark_metrics)
    bootstrap = parent_bootstrap_delta(
        diagnostic_scored_rows,
        primary_landmarks=primary,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    selection_event = ensemble_trajectories(selection_trajectories)
    diagnostic_event = ensemble_trajectories(diagnostic_trajectories)
    candidate_thresholds, candidate_threshold_audit = select_stage_thresholds(
        selection_event,
        "candidate",
        target_fpr=args.target_fpr,
        min_successes=args.min_threshold_successes,
    )
    time_thresholds, time_threshold_audit = select_stage_thresholds(
        selection_event,
        "time_only",
        target_fpr=args.target_fpr,
        min_successes=args.min_threshold_successes,
    )
    candidate_events = stage_event_metrics(
        diagnostic_event,
        "candidate",
        candidate_thresholds,
    )
    time_events = stage_event_metrics(
        diagnostic_event,
        "time_only",
        time_thresholds,
    )
    gate = continuation_gate(
        bootstrap,
        candidate_events,
        min_roc=args.min_roc,
        min_delta=args.min_delta,
        min_tpr=args.min_tpr,
        max_fpr=args.target_fpr,
        min_lead=args.min_lead,
    )
    write_csv(
        output / "diagnostic_event_metrics.csv",
        candidate_events["per_stage"] + time_events["per_stage"],
    )
    with (output / "diagnostic_event_predictions.jsonl").open("w") as stream:
        for row in candidate_events["predictions"]:
            stream.write(json.dumps(json_value(row), allow_nan=False) + "\n")
    write_json(
        output / "thresholds.json",
        {
            "candidate": candidate_thresholds,
            "candidate_audit": candidate_threshold_audit,
            "time_only": time_thresholds,
            "time_only_audit": time_threshold_audit,
        },
    )

    result = {
        "schema_version": 1,
        "status": "complete",
        "official_safe_commit": OFFICIAL_SAFE_COMMIT,
        "export_dir": str(Path(args.export_dir).resolve()),
        "selection_manifest": outer_manifest["path"],
        "output_dir": str(output),
        "selected_pipeline": selected_aggregate,
        "selection_aggregate": aggregate,
        "diagnostic_parent_bootstrap": bootstrap,
        "diagnostic_event_metrics": {
            "candidate": candidate_events["macro"],
            "time_only": time_events["macro"],
        },
        "continuation_gate": gate,
        "protocol": protocol,
        "claims": {
            "developmental_diagnostic": True,
            "outer_test_scored": False,
            "outer_test_used_for_training_or_selection": False,
            "thresholds_fit_on_selection_parents_only": True,
            "diagnostic_parents_used_once_after_pipeline_selection": True,
            "confirmatory_claim_requires_fresh_parent_pool": True,
            "privileged_predicates_online_feature": False,
            "semantic_stage_identity_required_online": True,
        },
        "artifacts": {
            "split_manifest": str((output / "split_manifest.json").resolve()),
            "selection_audit": str((output / "selection_audit.csv").resolve()),
            "selection_aggregate": str((output / "selection_aggregate.csv").resolve()),
            "diagnostic_landmarks": str(
                (output / "diagnostic_landmark_metrics.csv").resolve()
            ),
            "diagnostic_events": str(
                (output / "diagnostic_event_metrics.csv").resolve()
            ),
            "thresholds": str((output / "thresholds.json").resolve()),
            "runtime_root": str(runtime_root.resolve()),
        },
    }
    write_json(output / "analysis.json", result)
    write_json(
        output / "status.json",
        {
            "status": "complete",
            "selected_pipeline": selected_aggregate,
            "continuation_gate_pass": gate["pass"],
        },
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--stages", nargs="+")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["shared_one_hot", "stage_adapter", "stage_specific"],
        choices=ARCHITECTURES,
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(OBJECTIVES),
        choices=OBJECTIVES,
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=list(DETECTORS),
        choices=DETECTORS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--regularizations", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2]
    )
    parser.add_argument(
        "--landmarks", type=float, nargs="+", default=list(DEFAULT_LANDMARKS)
    )
    parser.add_argument(
        "--primary-landmarks",
        type=float,
        nargs="+",
        default=list(DEFAULT_PRIMARY_LANDMARKS),
    )
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument("--selection-fold", type=int, default=1)
    parser.add_argument("--diagnostic-fold", type=int, default=0)
    parser.add_argument("--meta-split-seed", type=int, default=0)
    parser.add_argument("--balance-seed", type=int, default=0)
    parser.add_argument("--stage-horizon-quantile", type=float, default=0.5)
    parser.add_argument("--time-prior", type=float, default=0.5)
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--min-stage-successes", type=int, default=9)
    parser.add_argument("--min-stage-failures", type=int, default=5)
    parser.add_argument("--min-threshold-successes", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--adapter-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--rank-margin", type=float, default=0.20)
    parser.add_argument("--onset-margin", type=float, default=0.10)
    parser.add_argument("--onset-weight", type=float, default=1.0)
    parser.add_argument("--horizon-selector", default="concat-2")
    parser.add_argument("--diffusion-selector", default=1.0)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--min-roc", type=float, default=0.65)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--min-tpr", type=float, default=0.40)
    parser.add_argument("--min-lead", type=float, default=0.25)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run_experiment(args)
    except Exception as error:
        status_path = Path(args.output_dir).resolve() / "status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                status = {}
            if status.get("status") in {"initializing", "running"}:
                write_json(
                    status_path,
                    {
                        **status,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
        raise
    print(json.dumps(json_value(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
