"""Train outcome-only causal-prefix SAFE and a residual SAFE+time head.

This is a developmental early-failure experiment.  It preserves a fixed outer
test split, splits source parents before prefix expansion, and never uses
Subtask-SAFE supervision or final duration as an online feature.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .causal_prefix_residual import (
        DEFAULT_LANDMARKS,
        DEFAULT_PRIMARY_LANDMARKS,
        DETECTORS,
        balance_training_rows,
        build_landmark_rows,
        clone_jsonable_time_curves,
        event_metrics,
        fit_feature_scaler,
        fit_time_risk,
        landmark_metrics,
        logit,
        primary_landmark_score,
        rollout_id,
        select_rollouts,
        stratified_meta_split,
        summarize_trajectory,
        threshold_at_fpr,
        training_task_horizons,
        transform_features,
        validate_landmarks,
    )
    from .train_seen_tasks import (
        OFFICIAL_SAFE_COMMIT,
        filter_aligned_task_type,
        load_env_records,
        load_outer_split_ids,
        make_config,
        make_manifest_split,
        make_seen_split,
        resolve_task_type_selection,
        task_names,
        verify_safe_repo,
    )
except ImportError:
    from causal_prefix_residual import (
        DEFAULT_LANDMARKS,
        DEFAULT_PRIMARY_LANDMARKS,
        DETECTORS,
        balance_training_rows,
        build_landmark_rows,
        clone_jsonable_time_curves,
        event_metrics,
        fit_feature_scaler,
        fit_time_risk,
        landmark_metrics,
        logit,
        primary_landmark_score,
        rollout_id,
        select_rollouts,
        stratified_meta_split,
        summarize_trajectory,
        threshold_at_fpr,
        training_task_horizons,
        transform_features,
        validate_landmarks,
    )
    from train_seen_tasks import (
        OFFICIAL_SAFE_COMMIT,
        filter_aligned_task_type,
        load_env_records,
        load_outer_split_ids,
        make_config,
        make_manifest_split,
        make_seen_split,
        resolve_task_type_selection,
        task_names,
        verify_safe_repo,
    )


MODEL_NAMES = ("prefix_safe", "residual_safe_time")


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
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
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def make_risk_mlp(torch, input_dim, hidden_dim, dropout):
    class RiskMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(int(input_dim), int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(hidden_dim), 1),
            )

        def forward(self, features):
            return self.network(features).squeeze(-1)

    return RiskMLP()


def rows_to_arrays(rows, scaler, *, residual):
    features = transform_features(
        np.stack([row["features"] for row in rows]), scaler
    ).astype(np.float32)
    targets = np.asarray([int(bool(row["failed"])) for row in rows], dtype=np.float32)
    weights = np.asarray(
        [float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float32
    )
    offsets = (
        logit([row["time_risk"] for row in rows]).astype(np.float32)
        if residual
        else np.zeros(len(rows), dtype=np.float32)
    )
    return features, targets, weights, offsets


def train_model(
    train_rows,
    validation_rows,
    scaler,
    *,
    residual,
    seed,
    hidden_dim,
    dropout,
    learning_rate,
    weight_decay,
    epochs,
    patience,
    device,
):
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    train_arrays = rows_to_arrays(train_rows, scaler, residual=residual)
    validation_arrays = rows_to_arrays(
        validation_rows, scaler, residual=residual
    )
    model = make_risk_mlp(
        torch, train_arrays[0].shape[1], hidden_dim, dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    tensors = [torch.as_tensor(value, device=device) for value in train_arrays]
    validation_tensors = [
        torch.as_tensor(value, device=device) for value in validation_arrays
    ]
    best = None
    history = []
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        features, targets, weights, offsets = tensors
        logits = model(features) + offsets
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        loss = (losses * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Causal-prefix training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_features, val_targets, val_weights, val_offsets = validation_tensors
            val_logits = model(val_features) + val_offsets
            val_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                val_logits, val_targets, reduction="none"
            )
            val_loss = float((val_losses * val_weights).sum() / val_weights.sum())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(loss.detach().cpu()),
                "validation_loss": val_loss,
            }
        )
        if best is None or val_loss < best["validation_loss"] - 1e-7:
            best = {
                "epoch": epoch + 1,
                "validation_loss": val_loss,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best is None:
        raise RuntimeError("Causal-prefix training did not produce a checkpoint")
    model.load_state_dict(best["state_dict"])
    return model, history, {key: value for key, value in best.items() if key != "state_dict"}


def predict_matrix(model, matrix, scaler, *, residual_offsets, device, batch_size=1024):
    import torch

    features = transform_features(matrix, scaler).astype(np.float32)
    offsets = np.asarray(residual_offsets, dtype=np.float32)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            stop = start + int(batch_size)
            values = torch.as_tensor(features[start:stop], device=device)
            batch_offsets = torch.as_tensor(offsets[start:stop], device=device)
            probabilities = torch.sigmoid(model(values) + batch_offsets)
            output.extend(probabilities.detach().cpu().numpy().tolist())
    return np.asarray(output, dtype=np.float64)


def score_landmark_rows(rows, prefix_model, residual_model, scaler, device):
    matrix = np.stack([row["features"] for row in rows])
    zeros = np.zeros(len(rows), dtype=np.float32)
    time_offsets = logit([row["time_risk"] for row in rows]).astype(np.float32)
    prefix_scores = predict_matrix(
        prefix_model,
        matrix,
        scaler,
        residual_offsets=zeros,
        device=device,
    )
    residual_scores = predict_matrix(
        residual_model,
        matrix,
        scaler,
        residual_offsets=time_offsets,
        device=device,
    )
    scored = []
    for row, prefix_score, residual_score in zip(
        rows, prefix_scores, residual_scores
    ):
        item = {key: value for key, value in row.items() if key != "features"}
        item["scores"] = {
            "time_only": float(row["time_risk"]),
            "prefix_safe": float(prefix_score),
            "residual_safe_time": float(residual_score),
        }
        scored.append(item)
    return scored


def candidate_score(rows, model, scaler, *, detector, residual, primary, device):
    matrix = np.stack([row["features"] for row in rows])
    offsets = (
        logit([row["time_risk"] for row in rows]).astype(np.float32)
        if residual
        else np.zeros(len(rows), dtype=np.float32)
    )
    predictions = predict_matrix(
        model,
        matrix,
        scaler,
        residual_offsets=offsets,
        device=device,
    )
    scored = []
    for row, prediction in zip(rows, predictions):
        item = {key: value for key, value in row.items() if key != "features"}
        item["scores"] = {
            "time_only": float(row["time_risk"]),
            detector: float(prediction),
        }
        scored.append(item)
    metrics = landmark_metrics(scored, detectors=("time_only", detector))
    value, support = primary_landmark_score(metrics, detector, primary)
    time_value, time_support = primary_landmark_score(metrics, "time_only", primary)
    if support != time_support:
        raise AssertionError("Detector and time-only primary supports differ")
    return {
        "primary_auc": value,
        "time_only_primary_auc": time_value,
        "delta_over_time": value - time_value,
        "supported_task_landmarks": support,
        "metrics": metrics,
    }


def select_model(
    train_rows,
    validation_rows,
    scaler,
    *,
    detector,
    residual,
    seed,
    regularizations,
    primary,
    hidden_dim,
    dropout,
    learning_rate,
    epochs,
    patience,
    device,
):
    candidates = []
    for regularization in regularizations:
        print(
            f"TRAIN detector={detector} seed={seed} "
            f"weight_decay={regularization:g}",
            flush=True,
        )
        model, history, checkpoint = train_model(
            train_rows,
            validation_rows,
            scaler,
            residual=residual,
            seed=seed,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=regularization,
            epochs=epochs,
            patience=patience,
            device=device,
        )
        score = candidate_score(
            validation_rows,
            model,
            scaler,
            detector=detector,
            residual=residual,
            primary=primary,
            device=device,
        )
        candidates.append(
            {
                "regularization": float(regularization),
                "model": model,
                "history": history,
                "checkpoint": checkpoint,
                **score,
            }
        )
        print(
            f"COMPLETE detector={detector} seed={seed} "
            f"weight_decay={regularization:g} "
            f"primary_auc={score['primary_auc']:.6f} "
            f"delta_time={score['delta_over_time']:.6f} "
            f"best_epoch={checkpoint['epoch']}",
            flush=True,
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["primary_auc"],
            item["delta_over_time"],
            -item["regularization"],
        ),
    )
    audit = [
        {
            key: value
            for key, value in item.items()
            if key not in {"model", "history", "metrics"}
        }
        | {"selected": item is selected}
        for item in candidates
    ]
    return selected, audit


def score_rollouts(
    rollouts,
    identity,
    *,
    horizons,
    time_curves,
    prefix_model,
    residual_model,
    scaler,
    window,
    device,
):
    scored = []
    for rollout in rollouts:
        task_id = int(rollout.task_id)
        features = summarize_trajectory(
            rollout.hidden_states,
            horizon=horizons[task_id],
            window=window,
        )
        steps = np.arange(1, len(features) + 1, dtype=np.int64)
        curve = np.asarray(time_curves[task_id]["risk"], dtype=np.float64)
        time_scores = curve[np.minimum(steps, len(curve)) - 1]
        prefix_scores = predict_matrix(
            prefix_model,
            features,
            scaler,
            residual_offsets=np.zeros(len(features), dtype=np.float32),
            device=device,
        )
        residual_scores = predict_matrix(
            residual_model,
            features,
            scaler,
            residual_offsets=logit(time_scores).astype(np.float32),
            device=device,
        )
        scored.append(
            {
                "rollout_id": rollout_id(rollout, identity),
                "task_id": task_id,
                "failed": not bool(int(rollout.episode_success)),
                "trajectories": {
                    "time_only": time_scores,
                    "prefix_safe": prefix_scores,
                    "residual_safe_time": residual_scores,
                },
            }
        )
    return scored


def event_prediction_rows(scored, thresholds, horizons, seed):
    rows = []
    for item in scored:
        record = {
            "seed": int(seed),
            "rollout_id": item["rollout_id"],
            "task_id": int(item["task_id"]),
            "failed": bool(item["failed"]),
            "detectors": {},
        }
        for detector in DETECTORS:
            trajectory = np.asarray(item["trajectories"][detector])
            crossings = np.flatnonzero(trajectory >= thresholds[detector])
            first = None if not len(crossings) else int(crossings[0]) + 1
            record["detectors"][detector] = {
                "maximum": float(np.max(trajectory)),
                "threshold": float(thresholds[detector]),
                "first_detection_inference": first,
                "first_detection_fraction": (
                    None
                    if first is None
                    else min(1.0, first / horizons[int(item["task_id"])])
                ),
            }
        rows.append(record)
    return rows


def state_dict_cpu(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def aggregate(rows, group_keys, value_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(grouped.items()):
        record = dict(zip(group_keys, key))
        for name in value_keys:
            numbers = [float(row[name]) for row in values if row.get(name) is not None]
            record[f"{name}_mean"] = float(np.mean(numbers)) if numbers else None
            record[f"{name}_std"] = float(np.std(numbers)) if numbers else None
        output.append(record)
    return output


def run_experiment(args):
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    verify_safe_repo(args.safe_repo)
    import torch
    from failure_prob.data.pizero import load_rollouts_from_root
    from failure_prob.utils.random import seed_everything

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested {device}, but CUDA is unavailable")
    landmarks = validate_landmarks(args.landmarks)
    primary = validate_landmarks(args.primary_landmarks)
    if not set(primary).issubset(landmarks):
        raise ValueError("Primary landmarks must be included in --landmarks")
    if int(args.temporal_window) < 1:
        raise ValueError("Temporal window must be positive")
    if int(args.hidden_dim) < 1:
        raise ValueError("Hidden dimension must be positive")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError("Dropout must lie in [0, 1)")
    if not 0.0 <= float(args.target_fpr) < 1.0:
        raise ValueError("Target FPR must lie in [0, 1)")
    if not args.seeds or len(set(int(seed) for seed in args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    regularizations = tuple(float(value) for value in args.regularizations)
    if not regularizations or any(value < 0 for value in regularizations):
        raise ValueError("Regularizations must be non-negative")

    hyperparameters = {
        "horizon_selector": args.horizon_selector,
        "diffusion_selector": args.diffusion_selector,
        "learning_rate": args.learning_rate,
        "lambda_reg": 0.0,
    }
    cfg = make_config(
        args.export_dir,
        "indep",
        0,
        args.epochs,
        "cpu",
        hyperparameters,
    )
    seed_everything(0)
    source_rollouts = load_rollouts_from_root(Path(args.export_dir), cfg)
    source_env_records = load_env_records(args.export_dir)
    task_selection = resolve_task_type_selection(args.export_dir, args.task_type)
    all_rollouts, _, identity = filter_aligned_task_type(
        source_rollouts,
        source_env_records,
        task_selection,
    )
    fixed_split = load_outer_split_ids(args.outer_split_manifest)
    if fixed_split is None:
        outer_train, outer_test, outer_counts = make_seen_split(
            all_rollouts,
            identity,
            train_per_class=args.train_per_class,
            split_seed=args.split_seed,
        )
    else:
        outer_train, outer_test, outer_counts = make_manifest_split(
            all_rollouts, identity, fixed_split
        )
    outer_train_ids = {rollout_id(item, identity) for item in outer_train}
    outer_test_ids = {rollout_id(item, identity) for item in outer_test}
    if outer_train_ids & outer_test_ids:
        raise AssertionError("Outer train and test parents overlap")

    meta = stratified_meta_split(
        outer_train,
        identity,
        validation_per_class=args.validation_per_class,
        seed=args.meta_split_seed,
    )
    fit_rollouts = select_rollouts(outer_train, identity, meta["fit"])
    validation_rollouts = select_rollouts(
        outer_train, identity, meta["validation"]
    )
    horizons = training_task_horizons(fit_rollouts)
    time_curves = fit_time_risk(fit_rollouts, horizons, prior=args.time_prior)
    train_rows, train_excluded_before = build_landmark_rows(
        fit_rollouts,
        identity,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=landmarks,
        window=args.temporal_window,
        split="fit",
    )
    validation_rows, validation_excluded = build_landmark_rows(
        validation_rollouts,
        identity,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=landmarks,
        window=args.temporal_window,
        split="validation",
    )
    test_rows, test_excluded = build_landmark_rows(
        outer_test,
        identity,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=landmarks,
        window=args.temporal_window,
        split="test",
    )
    train_rows, training_support, balance_excluded = balance_training_rows(
        train_rows, seed=args.balance_seed
    )
    scaler = fit_feature_scaler(train_rows)

    names = task_names(args.export_dir)
    split_manifest = {
        "schema_version": 1,
        "protocol": "outcome_only_multi_landmark_causal_prefix",
        "binary_final_rollout_labels_only": True,
        "subtask_safe": False,
        "split_before_prefix_expansion": True,
        "outer_split_manifest": (
            None if fixed_split is None else fixed_split["path"]
        ),
        "split_seed": int(args.split_seed),
        "meta_split_seed": int(args.meta_split_seed),
        "balance_seed": int(args.balance_seed),
        "landmarks": list(landmarks),
        "primary_landmarks": list(primary),
        "horizons_fit_failures_only": horizons,
        "task_names": {str(key): names[key] for key in sorted(horizons)},
        "source_counts": {
            "outer_train": len(outer_train),
            "outer_test": len(outer_test),
            "meta_fit": len(fit_rollouts),
            "meta_validation": len(validation_rollouts),
        },
        "prefix_counts": {
            "fit_balanced": len(train_rows),
            "validation_at_risk": len(validation_rows),
            "test_at_risk": len(test_rows),
        },
        "training_support": training_support,
        "excluded_counts": {
            "fit_before_landmark": len(train_excluded_before),
            "fit_balance_or_unsupported": len(balance_excluded),
            "validation_before_landmark": len(validation_excluded),
            "test_before_landmark": len(test_excluded),
        },
        "meta_counts": meta["counts"],
        "outer_counts": outer_counts,
        "fit_parent_ids": sorted(meta["fit"]),
        "validation_parent_ids": sorted(meta["validation"]),
        "test_parent_ids": sorted(outer_test_ids),
    }
    write_json(output / "split_manifest.json", split_manifest)

    event_rows = []
    landmark_rows = []
    selection_rows = []
    prediction_rows = []
    runtime_root = output / "runtime"
    runtime_root.mkdir()
    write_json(
        output / "status.json",
        {
            "status": "running",
            "completed_seeds": 0,
            "total_seeds": len(args.seeds),
        },
    )
    for seed in args.seeds:
        print(f"SEED_START seed={seed}", flush=True)
        selected_models = {}
        seed_selection = {}
        for detector, residual in (
            ("prefix_safe", False),
            ("residual_safe_time", True),
        ):
            selected, audit = select_model(
                train_rows,
                validation_rows,
                scaler,
                detector=detector,
                residual=residual,
                seed=seed,
                regularizations=regularizations,
                primary=primary,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
            )
            selected_models[detector] = selected["model"]
            seed_selection[detector] = {
                key: value
                for key, value in selected.items()
                if key not in {"model", "history", "metrics"}
            }
            for item in audit:
                selection_rows.append(
                    {"seed": int(seed), "detector": detector, **item}
                )
            write_json(
                output / f"{detector}_seed{seed}_history.json",
                selected["history"],
            )

        validation_scored = score_rollouts(
            validation_rollouts,
            identity,
            horizons=horizons,
            time_curves=time_curves,
            prefix_model=selected_models["prefix_safe"],
            residual_model=selected_models["residual_safe_time"],
            scaler=scaler,
            window=args.temporal_window,
            device=device,
        )
        test_scored = score_rollouts(
            outer_test,
            identity,
            horizons=horizons,
            time_curves=time_curves,
            prefix_model=selected_models["prefix_safe"],
            residual_model=selected_models["residual_safe_time"],
            scaler=scaler,
            window=args.temporal_window,
            device=device,
        )
        thresholds = {
            detector: threshold_at_fpr(
                validation_scored, detector, target_fpr=args.target_fpr
            )
            for detector in DETECTORS
        }
        for detector in DETECTORS:
            event_rows.append(
                {
                    "seed": int(seed),
                    **event_metrics(
                        test_scored, detector, thresholds[detector], horizons
                    ),
                }
            )

        test_landmark_scored = score_landmark_rows(
            test_rows,
            selected_models["prefix_safe"],
            selected_models["residual_safe_time"],
            scaler,
            device,
        )
        for row in landmark_metrics(test_landmark_scored):
            landmark_rows.append({"seed": int(seed), **row})
        prediction_rows.extend(
            event_prediction_rows(test_scored, thresholds, horizons, seed)
        )

        for detector in MODEL_NAMES:
            runtime = runtime_root / f"{detector}_seed{seed}"
            runtime.mkdir()
            torch.save(
                {
                    "state_dict": state_dict_cpu(selected_models[detector]),
                    "input_dim": int(len(scaler["mean"])),
                    "hidden_dim": int(args.hidden_dim),
                    "dropout": float(args.dropout),
                    "residual_time_offset": detector == "residual_safe_time",
                },
                runtime / "model.pt",
            )
            write_json(
                runtime / "runtime.json",
                {
                    "schema_version": 1,
                    "detector": detector,
                    "seed": int(seed),
                    "binary_final_rollout_labels_only": True,
                    "subtask_safe": False,
                    "feature_blocks": ["current", "delta", "recent_mean", "recent_slope"],
                    "temporal_window": int(args.temporal_window),
                    "horizon_selector": args.horizon_selector,
                    "diffusion_selector": args.diffusion_selector,
                    "task_horizons": horizons,
                    "time_curves": clone_jsonable_time_curves(time_curves),
                    "feature_mean": scaler["mean"],
                    "feature_scale": scaler["scale"],
                    "threshold": thresholds[detector],
                    "target_validation_fpr": float(args.target_fpr),
                    "selection": seed_selection[detector],
                },
            )
        completed_seeds = list(args.seeds).index(seed) + 1
        write_json(
            output / "status.json",
            {
                "status": "running",
                "completed_seeds": completed_seeds,
                "total_seeds": len(args.seeds),
                "last_completed_seed": int(seed),
            },
        )
        print(f"SEED_COMPLETE seed={seed}", flush=True)

    write_csv(output / "selection_audit.csv", selection_rows)
    write_csv(output / "per_seed_event_metrics.csv", event_rows)
    write_csv(output / "landmark_metrics.csv", landmark_rows)
    with (output / "event_predictions.jsonl").open("w") as stream:
        for row in prediction_rows:
            stream.write(json.dumps(json_value(row), allow_nan=False) + "\n")

    event_aggregate = aggregate(
        event_rows,
        ["detector"],
        [
            "roc_auc",
            "average_precision",
            "accuracy",
            "balanced_accuracy",
            "true_positive_rate",
            "false_positive_rate",
            "mean_detected_failure_fraction",
            "missed_failure_adjusted_detection_fraction",
        ],
    )
    pooled_landmarks = [row for row in landmark_rows if row["scope"] == "pooled"]
    landmark_aggregate = aggregate(
        pooled_landmarks,
        ["detector", "landmark_fraction"],
        ["roc_auc", "average_precision"],
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "official_safe_commit": OFFICIAL_SAFE_COMMIT,
        "export_dir": str(Path(args.export_dir).resolve()),
        "outer_split_manifest": args.outer_split_manifest,
        "output_dir": str(output),
        "seeds": [int(seed) for seed in args.seeds],
        "landmarks": list(landmarks),
        "primary_landmarks": list(primary),
        "target_validation_fpr": float(args.target_fpr),
        "event_aggregate": event_aggregate,
        "landmark_aggregate": landmark_aggregate,
        "selection": selection_rows,
        "protocol": split_manifest,
        "claims": {
            "outcome_only": True,
            "subtask_safe": False,
            "final_duration_online_feature": False,
            "outer_test_used_for_training_or_selection": False,
            "existing_outer_test_previously_opened": True,
            "confirmatory_claim_requires_fresh_test": True,
        },
        "artifacts": {
            "split_manifest": str((output / "split_manifest.json").resolve()),
            "selection_audit": str((output / "selection_audit.csv").resolve()),
            "event_metrics": str((output / "per_seed_event_metrics.csv").resolve()),
            "landmark_metrics": str((output / "landmark_metrics.csv").resolve()),
            "event_predictions": str((output / "event_predictions.jsonl").resolve()),
            "runtime_root": str(runtime_root.resolve()),
        },
    }
    write_json(output / "analysis.json", result)
    write_json(
        output / "status.json",
        {
            "status": "complete",
            "completed_seeds": len(args.seeds),
            "total_seeds": len(args.seeds),
            "analysis": str((output / "analysis.json").resolve()),
        },
    )
    print(
        f"COMPLETE seeds={len(args.seeds)} output={output}",
        flush=True,
    )
    return result


def selector(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-split-manifest")
    parser.add_argument("--task-type", choices=("all", "atomic", "composite"), default="all")
    parser.add_argument("--train-per-class", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--meta-split-seed", type=int, default=0)
    parser.add_argument("--balance-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--landmarks", nargs="+", type=float, default=list(DEFAULT_LANDMARKS))
    parser.add_argument(
        "--primary-landmarks",
        nargs="+",
        type=float,
        default=list(DEFAULT_PRIMARY_LANDMARKS),
    )
    parser.add_argument("--horizon-selector", type=selector, default=1.0)
    parser.add_argument("--diffusion-selector", type=selector, default=1.0)
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--regularizations",
        nargs="+",
        type=float,
        default=[1e-5, 1e-4, 1e-3, 1e-2],
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--time-prior", type=float, default=0.5)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run_experiment(args)
    except (FileExistsError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(json_value(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
