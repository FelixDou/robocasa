"""Train checkpoint-specific early SAFE heads and a late time fallback.

This developmental Xiaomi experiment trains independent outcome-only SAFE heads
at 10 and 25 percent of a task horizon.  They may alarm only at their declared
causal checkpoints.  A task-conditioned elapsed-time detector becomes eligible
at 50 percent.  One SAFE threshold and one time threshold are jointly selected
on validation parents under a shared event-level FPR cap.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .causal_prefix_residual import (
        balance_training_rows,
        build_landmark_rows,
        clone_jsonable_time_curves,
        fit_feature_scaler,
        fit_time_risk,
        landmark_metrics,
        rollout_id,
        select_rollouts,
        stratified_meta_split,
        summarize_prefix,
        training_task_horizons,
    )
    from .early_safe_time_cascade import (
        CASCADE_DETECTORS,
        DEFAULT_EARLY_LANDMARKS,
        DEFAULT_TIME_FALLBACK,
        cascade_metrics,
        prediction_record,
        select_joint_thresholds,
        select_single_threshold,
        stage_steps,
        validate_stages,
    )
    from .run_causal_prefix_residual import (
        aggregate,
        json_value,
        predict_matrix,
        select_model,
        state_dict_cpu,
        write_csv,
        write_json,
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
        balance_training_rows,
        build_landmark_rows,
        clone_jsonable_time_curves,
        fit_feature_scaler,
        fit_time_risk,
        landmark_metrics,
        rollout_id,
        select_rollouts,
        stratified_meta_split,
        summarize_prefix,
        training_task_horizons,
    )
    from early_safe_time_cascade import (
        CASCADE_DETECTORS,
        DEFAULT_EARLY_LANDMARKS,
        DEFAULT_TIME_FALLBACK,
        cascade_metrics,
        prediction_record,
        select_joint_thresholds,
        select_single_threshold,
        stage_steps,
        validate_stages,
    )
    from run_causal_prefix_residual import (
        aggregate,
        json_value,
        predict_matrix,
        select_model,
        state_dict_cpu,
        write_csv,
        write_json,
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


def stage_name(landmark):
    return f"early_safe_{float(landmark):g}".replace(".", "p")


def rows_at_landmark(rows, landmark):
    return [
        row
        for row in rows
        if math.isclose(float(row["landmark_fraction"]), float(landmark), abs_tol=1e-12)
    ]


def score_landmark_stage(rows, model, scaler, detector, device):
    if not rows:
        return []
    predictions = predict_matrix(
        model,
        np.stack([row["features"] for row in rows]),
        scaler,
        residual_offsets=np.zeros(len(rows), dtype=np.float32),
        device=device,
    )
    output = []
    for row, score in zip(rows, predictions):
        item = {key: value for key, value in row.items() if key != "features"}
        item["scores"] = {
            "time_only": float(row["time_risk"]),
            detector: float(score),
        }
        output.append(item)
    return output


def score_staged_rollouts(
    rollouts,
    identity,
    *,
    horizons,
    time_curves,
    early_landmarks,
    time_fallback,
    models,
    scalers,
    window,
    device,
):
    scored = []
    for rollout in rollouts:
        task_id = int(rollout.task_id)
        horizon = int(horizons[task_id])
        schedule = stage_steps(horizon, early_landmarks, time_fallback)
        source_length = min(len(rollout.hidden_states), horizon)
        early_stages = []
        for landmark in early_landmarks:
            step = schedule["early"][landmark]
            if source_length < step:
                continue
            detector = stage_name(landmark)
            features = summarize_prefix(rollout.hidden_states, step, window=window)[
                None, :
            ]
            score = predict_matrix(
                models[landmark],
                features,
                scalers[landmark],
                residual_offsets=np.zeros(1, dtype=np.float32),
                device=device,
            )[0]
            early_stages.append(
                {
                    "landmark_fraction": float(landmark),
                    "detector": detector,
                    "step": int(step),
                    "score": float(score),
                }
            )
        full_steps = np.arange(1, source_length + 1, dtype=np.int64)
        curve = np.asarray(time_curves[task_id]["risk"], dtype=np.float64)
        full_scores = curve[np.minimum(full_steps, len(curve)) - 1]
        fallback_step = int(schedule["time_fallback"])
        keep = full_steps >= fallback_step
        scored.append(
            {
                "rollout_id": rollout_id(rollout, identity),
                "task_id": task_id,
                "failed": not bool(int(rollout.episode_success)),
                "source_inferences": int(source_length),
                "horizon": horizon,
                "early_stages": early_stages,
                "full_time_steps": full_steps,
                "full_time_scores": full_scores,
                "late_time_steps": full_steps[keep],
                "late_time_scores": full_scores[keep],
            }
        )
    return scored


def run_experiment(args):
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    verify_safe_repo(args.safe_repo)

    import torch
    from failure_prob.data.pizero import load_rollouts_from_root
    from failure_prob.utils.random import seed_everything

    early_landmarks, time_fallback = validate_stages(
        args.early_landmarks, args.time_fallback
    )
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested {device}, but CUDA is unavailable")
    if not 0.0 <= float(args.target_fpr) < 1.0:
        raise ValueError("Target FPR must lie in [0, 1)")
    if int(args.temporal_window) < 1 or int(args.hidden_dim) < 1:
        raise ValueError("Temporal window and hidden dimension must be positive")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError("Dropout must lie in [0, 1)")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and unique")
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
    source = load_rollouts_from_root(Path(args.export_dir), cfg)
    env_records = load_env_records(args.export_dir)
    task_selection = resolve_task_type_selection(args.export_dir, args.task_type)
    all_rollouts, _, identity = filter_aligned_task_type(
        source, env_records, task_selection
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
    validation_rollouts = select_rollouts(outer_train, identity, meta["validation"])
    horizons = training_task_horizons(fit_rollouts)
    time_curves = fit_time_risk(fit_rollouts, horizons, prior=args.time_prior)
    raw_train_rows, train_excluded = build_landmark_rows(
        fit_rollouts,
        identity,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=early_landmarks,
        window=args.temporal_window,
        split="fit",
    )
    validation_rows, validation_excluded = build_landmark_rows(
        validation_rollouts,
        identity,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=early_landmarks,
        window=args.temporal_window,
        split="validation",
    )
    test_rows, test_excluded = build_landmark_rows(
        outer_test,
        identity,
        horizons=horizons,
        time_curves=time_curves,
        landmarks=early_landmarks,
        window=args.temporal_window,
        split="test",
    )

    balanced = {}
    scalers = {}
    training_support = {}
    balance_excluded = {}
    for offset, landmark in enumerate(early_landmarks):
        rows, support, excluded = balance_training_rows(
            rows_at_landmark(raw_train_rows, landmark),
            seed=int(args.balance_seed) + offset,
        )
        balanced[landmark] = rows
        scalers[landmark] = fit_feature_scaler(rows)
        training_support[str(landmark)] = support
        balance_excluded[str(landmark)] = excluded

    names = task_names(args.export_dir)
    split_manifest = {
        "schema_version": 1,
        "protocol": "outcome_only_staged_early_safe_time_fallback",
        "binary_final_rollout_labels_only": True,
        "subtask_safe": False,
        "split_before_prefix_expansion": True,
        "outer_split_manifest": None if fixed_split is None else fixed_split["path"],
        "split_seed": int(args.split_seed),
        "meta_split_seed": int(args.meta_split_seed),
        "balance_seed": int(args.balance_seed),
        "early_safe_landmarks": list(early_landmarks),
        "time_fallback_fraction": float(time_fallback),
        "shared_event_level_target_fpr": float(args.target_fpr),
        "horizons_fit_failures_only": horizons,
        "task_names": {str(key): names[key] for key in sorted(horizons)},
        "source_counts": {
            "outer_train": len(outer_train),
            "outer_test": len(outer_test),
            "meta_fit": len(fit_rollouts),
            "meta_validation": len(validation_rollouts),
        },
        "prefix_counts": {
            str(landmark): {
                "fit_balanced": len(balanced[landmark]),
                "validation_at_risk": len(rows_at_landmark(validation_rows, landmark)),
                "test_at_risk": len(rows_at_landmark(test_rows, landmark)),
            }
            for landmark in early_landmarks
        },
        "training_support": training_support,
        "excluded_counts": {
            "fit_before_landmark": len(train_excluded),
            "fit_balance_or_unsupported": {
                key: len(value) for key, value in balance_excluded.items()
            },
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
    threshold_rows = []
    prediction_rows = []
    runtime_root = output / "runtime"
    runtime_root.mkdir()
    write_json(
        output / "status.json",
        {"status": "running", "completed_seeds": 0, "total_seeds": len(seeds)},
    )

    for seed_index, seed in enumerate(seeds):
        print(f"SEED_START seed={seed}", flush=True)
        models = {}
        selections = {}
        for landmark in early_landmarks:
            detector = stage_name(landmark)
            selected, audit = select_model(
                balanced[landmark],
                rows_at_landmark(validation_rows, landmark),
                scalers[landmark],
                detector=detector,
                residual=False,
                seed=seed,
                regularizations=regularizations,
                primary=(landmark,),
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
            )
            models[landmark] = selected["model"]
            selections[str(landmark)] = {
                key: value
                for key, value in selected.items()
                if key not in {"model", "history", "metrics"}
            }
            selection_rows.extend(
                {
                    "seed": seed,
                    "landmark_fraction": landmark,
                    "detector": detector,
                    **item,
                }
                for item in audit
            )
            write_json(
                output / f"{detector}_seed{seed}_history.json",
                selected["history"],
            )

        validation_scored = score_staged_rollouts(
            validation_rollouts,
            identity,
            horizons=horizons,
            time_curves=time_curves,
            early_landmarks=early_landmarks,
            time_fallback=time_fallback,
            models=models,
            scalers=scalers,
            window=args.temporal_window,
            device=device,
        )
        test_scored = score_staged_rollouts(
            outer_test,
            identity,
            horizons=horizons,
            time_curves=time_curves,
            early_landmarks=early_landmarks,
            time_fallback=time_fallback,
            models=models,
            scalers=scalers,
            window=args.temporal_window,
            device=device,
        )
        early_selection = select_single_threshold(
            validation_scored,
            "early_safe",
            horizons,
            target_fpr=args.target_fpr,
        )
        time_selection = select_single_threshold(
            validation_scored,
            "time_only",
            horizons,
            target_fpr=args.target_fpr,
        )
        joint_selection = select_joint_thresholds(
            validation_scored, horizons, target_fpr=args.target_fpr
        )
        detector_thresholds = {
            "early_safe": {"early_safe": early_selection["threshold"]},
            "staged_safe_time": joint_selection["thresholds"],
            "time_only": {"time_only": time_selection["threshold"]},
        }
        threshold_rows.append(
            {
                "seed": seed,
                "early_safe_threshold": early_selection["threshold"],
                "cascade_early_threshold": joint_selection["thresholds"]["early_safe"],
                "cascade_late_time_threshold": joint_selection["thresholds"][
                    "late_time"
                ],
                "time_only_threshold": time_selection["threshold"],
                "cascade_search": joint_selection["search"],
                "early_validation_metrics": early_selection["validation_metrics"],
                "cascade_validation_metrics": joint_selection["validation_metrics"],
                "time_validation_metrics": time_selection["validation_metrics"],
            }
        )
        for detector in CASCADE_DETECTORS:
            event_rows.append(
                {
                    "seed": seed,
                    **cascade_metrics(
                        test_scored,
                        detector,
                        detector_thresholds[detector],
                        horizons,
                    ),
                }
            )

        for landmark in early_landmarks:
            detector = stage_name(landmark)
            scored_rows = score_landmark_stage(
                rows_at_landmark(test_rows, landmark),
                models[landmark],
                scalers[landmark],
                detector,
                device,
            )
            landmark_rows.extend(
                {"seed": seed, **row}
                for row in landmark_metrics(
                    scored_rows, detectors=("time_only", detector)
                )
            )

        for item in test_scored:
            prediction_rows.append(
                {
                    "seed": seed,
                    "rollout_id": item["rollout_id"],
                    "task_id": int(item["task_id"]),
                    "failed": bool(item["failed"]),
                    "source_inferences": int(item["source_inferences"]),
                    "detectors": {
                        detector: prediction_record(
                            item,
                            detector,
                            detector_thresholds[detector],
                            horizons,
                        )
                        for detector in CASCADE_DETECTORS
                    },
                }
            )

        runtime = runtime_root / f"seed{seed}"
        runtime.mkdir()
        for landmark in early_landmarks:
            torch.save(
                {
                    "state_dict": state_dict_cpu(models[landmark]),
                    "input_dim": int(len(scalers[landmark]["mean"])),
                    "hidden_dim": int(args.hidden_dim),
                    "dropout": float(args.dropout),
                    "landmark_fraction": float(landmark),
                },
                runtime / f"{stage_name(landmark)}.pt",
            )
        write_json(
            runtime / "runtime.json",
            {
                "schema_version": 1,
                "detector": "staged_early_safe_time_fallback",
                "seed": seed,
                "binary_final_rollout_labels_only": True,
                "subtask_safe": False,
                "feature_blocks": ["current", "delta", "recent_mean", "recent_slope"],
                "temporal_window": int(args.temporal_window),
                "horizon_selector": args.horizon_selector,
                "diffusion_selector": args.diffusion_selector,
                "early_safe_landmarks": list(early_landmarks),
                "time_fallback_fraction": float(time_fallback),
                "task_horizons": horizons,
                "time_curves": clone_jsonable_time_curves(time_curves),
                "feature_scalers": {
                    str(landmark): scalers[landmark] for landmark in early_landmarks
                },
                "thresholds": detector_thresholds,
                "target_validation_fpr": float(args.target_fpr),
                "selection": selections,
            },
        )
        write_json(
            output / "status.json",
            {
                "status": "running",
                "completed_seeds": seed_index + 1,
                "total_seeds": len(seeds),
                "last_completed_seed": seed,
            },
        )
        print(f"SEED_COMPLETE seed={seed}", flush=True)

    write_csv(output / "selection_audit.csv", selection_rows)
    write_csv(output / "threshold_selection.csv", threshold_rows)
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
        "seeds": list(seeds),
        "early_safe_landmarks": list(early_landmarks),
        "time_fallback_fraction": float(time_fallback),
        "target_validation_fpr": float(args.target_fpr),
        "event_aggregate": event_aggregate,
        "landmark_aggregate": landmark_aggregate,
        "selection": selection_rows,
        "threshold_selection": threshold_rows,
        "protocol": split_manifest,
        "claims": {
            "outcome_only": True,
            "subtask_safe": False,
            "final_duration_online_feature": False,
            "elapsed_time_is_an_explicit_late_fallback": True,
            "outer_test_used_for_training_or_selection": False,
            "existing_outer_test_previously_opened": True,
            "confirmatory_claim_requires_fresh_test": True,
        },
        "artifacts": {
            "split_manifest": str((output / "split_manifest.json").resolve()),
            "selection_audit": str((output / "selection_audit.csv").resolve()),
            "threshold_selection": str((output / "threshold_selection.csv").resolve()),
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
            "completed_seeds": len(seeds),
            "total_seeds": len(seeds),
            "analysis": str((output / "analysis.json").resolve()),
        },
    )
    print(f"COMPLETE seeds={len(seeds)} output={output}", flush=True)
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
    parser.add_argument(
        "--task-type", choices=("all", "atomic", "composite"), default="all"
    )
    parser.add_argument("--train-per-class", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--meta-split-seed", type=int, default=0)
    parser.add_argument("--balance-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--early-landmarks",
        nargs="+",
        type=float,
        default=list(DEFAULT_EARLY_LANDMARKS),
    )
    parser.add_argument("--time-fallback", type=float, default=DEFAULT_TIME_FALLBACK)
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
