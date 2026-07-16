"""Persist evaluation artifacts for a checkpoint trained by pinned official SAFE."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import numpy as np


OFFICIAL_SAFE_COMMIT = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
OFFICIAL_ALPHAS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9)
OFFICIAL_TIME_QUANTILES = (0.25, 0.5, 0.75, 1.0)
SELECTION_METRIC = "falert_early_roc_auc/model_val_seen"


def _json_value(value: Any):
    """Convert numpy/torch-adjacent values into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, value: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


def verify_safe_checkout(safe_repo: str | Path, expected_commit=OFFICIAL_SAFE_COMMIT):
    safe_repo = Path(safe_repo).resolve()
    if not (safe_repo / "failure_prob" / "train.py").is_file():
        raise ValueError(f"Not an official SAFE checkout: {safe_repo}")
    commit = subprocess.run(
        ["git", "-C", str(safe_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if expected_commit and commit != expected_commit:
        raise ValueError(f"SAFE checkout is at {commit}, expected {expected_commit}")
    return safe_repo, commit


def sha256_file(path: str | Path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _load_export_metadata(export_dir: Path):
    try:
        from natsort import natsorted
    except ImportError as error:
        raise RuntimeError("Official SAFE evaluation requires natsort") from error

    paths = natsorted(glob.glob(str(export_dir / "env_records" / "*.pkl")))
    if not paths:
        raise ValueError(f"No env_records found in {export_dir}")
    records = []
    for path in paths:
        with open(path, "rb") as stream:
            record = pickle.load(stream)
        records.append(record)
    return records


def _task_names(export_dir: Path):
    report_path = export_dir / "conversion_report.json"
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text())
    return {int(value): key for key, value in report.get("task_ids", {}).items()}


def _validate_identity_alignment(rollouts, env_records):
    if len(rollouts) != len(env_records):
        raise ValueError(
            f"Official loader returned {len(rollouts)} rollouts for "
            f"{len(env_records)} env records"
        )
    identities = {}
    rollout_ids = set()
    for index, (rollout, env) in enumerate(zip(rollouts, env_records)):
        rollout_id = str(env.get("rollout_id", ""))
        if not rollout_id:
            raise ValueError(f"env record {index} has no rollout_id")
        if rollout_id in rollout_ids:
            raise ValueError(f"duplicate rollout_id in export: {rollout_id}")
        rollout_ids.add(rollout_id)
        checks = {
            "task_id": (int(rollout.task_id), int(env["task_id"])),
            "episode_idx": (int(rollout.episode_idx), int(env["episode_idx"])),
            "episode_success": (
                int(rollout.episode_success),
                int(env["episode_success"]),
            ),
        }
        mismatches = [name for name, values in checks.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError(
                f"official loader/env record mismatch at index {index}: {mismatches}"
            )
        identities[id(rollout)] = env
    return identities


def _counts(rollouts, task_names):
    by_task = defaultdict(lambda: {"successes": 0, "failures": 0})
    for rollout in rollouts:
        task = task_names.get(int(rollout.task_id), str(rollout.task_description))
        key = "successes" if rollout.episode_success else "failures"
        by_task[task][key] += 1
    return {
        "rollouts": len(rollouts),
        "successes": sum(int(x.episode_success) for x in rollouts),
        "failures": sum(not int(x.episode_success) for x in rollouts),
        "tasks": dict(sorted(by_task.items())),
    }


def _safe_roc_auc(labels, values):
    if len(set(int(x) for x in labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, values))


def duration_diagnostics(rollouts_by_split, task_names):
    diagnostics = {}
    warnings = []
    for split, rollouts in rollouts_by_split.items():
        groups = {"overall": list(rollouts)}
        for task_id in sorted({int(x.task_id) for x in rollouts}):
            name = task_names.get(task_id, str(task_id))
            groups[name] = [x for x in rollouts if int(x.task_id) == task_id]
        diagnostics[split] = {}
        for name, group in groups.items():
            labels = [1 - int(x.episode_success) for x in group]
            lengths = [len(x.hidden_states) for x in group]
            auc = _safe_roc_auc(labels, lengths)
            diagnostics[split][name] = {
                "num_rollouts": len(group),
                "inference_length_min": min(lengths) if lengths else None,
                "inference_length_max": max(lengths) if lengths else None,
                "inference_length_mean": float(np.mean(lengths)) if lengths else None,
                "duration_only_roc_auc": auc,
            }
            if name == "overall" and auc is not None and (auc >= 0.75 or auc <= 0.25):
                warnings.append(
                    f"{split}: rollout duration alone is strongly predictive "
                    f"(ROC-AUC={auc:.3f})"
                )
    return diagnostics, warnings


def baseline_scores(scores_by_split, baseline):
    output = {}
    for split, trajectories in scores_by_split.items():
        if baseline == "constant":
            output[split] = [np.full(len(x), 0.5, dtype=np.float64) for x in trajectories]
        elif baseline == "time_only":
            # Absolute inference time is causal and deliberately exposes termination-length leakage.
            output[split] = [np.arange(len(x), dtype=np.float64) for x in trajectories]
        else:
            raise ValueError(f"Unknown baseline {baseline}")
    return output


def evaluate_functional_bands_by_task(
    rollouts,
    scores,
    bands_by_alpha,
    *,
    method,
    task_names,
):
    """Apply official global bands overall and to each task without recalibration."""
    groups = [("all", list(range(len(rollouts))))]
    for task_id in sorted({int(rollout.task_id) for rollout in rollouts}):
        groups.append(
            (
                task_names.get(task_id, str(task_id)),
                [index for index, rollout in enumerate(rollouts) if int(rollout.task_id) == task_id],
            )
        )
    rows = []
    for alpha, raw_band in sorted(bands_by_alpha.items(), key=lambda item: float(item[0])):
        band = np.asarray(raw_band, dtype=np.float64).reshape(-1)
        aligned = [
            np.pad(np.asarray(score), (0, len(band) - len(score)), mode="edge")
            for score in scores
        ]
        for eval_time in ("by final end", "by earliest stop"):
            for task_name, indices in groups:
                labels = np.asarray(
                    [1 - int(rollouts[index].episode_success) for index in indices],
                    dtype=bool,
                )
                detections = []
                relative_times = []
                for index in indices:
                    if eval_time == "by final end":
                        length = len(band)
                    else:
                        length = int(rollouts[index].task_min_step)
                    mask = aligned[index][:length] >= band[:length]
                    found = bool(np.any(mask))
                    detection = int(np.argmax(mask)) if found else length
                    detections.append(found)
                    relative_times.append(detection / length)
                predicted = np.asarray(detections, dtype=bool)
                relative_times = np.asarray(relative_times, dtype=np.float64)
                tp = int(np.sum(predicted & labels))
                fn = int(np.sum(~predicted & labels))
                fp = int(np.sum(predicted & ~labels))
                tn = int(np.sum(~predicted & ~labels))
                tpr = tp / (tp + fn) if tp + fn else 0.0
                tnr = tn / (tn + fp) if tn + fp else 0.0
                fpr = fp / (fp + tn) if fp + tn else 0.0
                fnr = fn / (fn + tp) if fn + tp else 0.0
                accuracy = (tp + tn) / len(indices)
                f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                rows.append(
                    {
                        "source": "official band with persisted taskwise counts",
                        "detect_method": method,
                        "method": method,
                        "cal split": "val_seen",
                        "test split": "val_unseen",
                        "calib on": "neg",
                        "task": task_name,
                        "thresh_method": "functional CP",
                        "alpha": float(alpha),
                        "time": eval_time,
                        "avg_det_time": (
                            float(np.mean(relative_times[labels]))
                            if np.any(labels)
                            else None
                        ),
                        "tpr": tpr,
                        "tnr": tnr,
                        "fpr": fpr,
                        "fnr": fnr,
                        "acc": accuracy,
                        "bal_acc": (tpr + tnr) / 2,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "num_rollouts": len(indices),
                        "num_successes": int(np.sum(~labels)),
                        "num_failures": int(np.sum(labels)),
                        "false_alarms_per_successful_rollout": fpr,
                        "failed_rollouts_detected_fraction": tpr,
                    }
                )
    return rows


def validate_persisted_rows_against_official(rows, official_rows, tolerance=1e-10):
    """Fail if the persisted overall computation drifts from upstream metrics."""
    keys = ("avg_det_time", "tpr", "tnr", "fpr", "fnr", "acc", "bal_acc", "f1")
    lookup = {
        (str(row["time"]), float(row["alpha"])): row
        for row in rows
        if row["task"] == "all"
    }
    for official in official_rows:
        persisted = lookup[(str(official["time"]), float(official["alpha"]))]
        for key in keys:
            left = float(persisted[key])
            right = float(official[key])
            if not np.isclose(left, right, rtol=tolerance, atol=tolerance):
                raise ValueError(
                    f"Persisted functional metric drift for {key}: {left} != {right}"
                )


def _split_manifest(rollouts_by_split, identities, task_names):
    result = {"schema_version": 1, "splits": {}, "counts": {}}
    all_ids = []
    for split, rollouts in rollouts_by_split.items():
        ids = [str(identities[id(rollout)]["rollout_id"]) for rollout in rollouts]
        result["splits"][split] = ids
        result["counts"][split] = _counts(rollouts, task_names)
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Official split contains duplicate rollout IDs")
    result["num_unique_rollouts"] = len(all_ids)
    return result


def _save_scores(path, rollouts_by_split, scores_by_split, identities, task_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for split, rollouts in rollouts_by_split.items():
            for rollout, scores in zip(rollouts, scores_by_split[split]):
                env = identities[id(rollout)]
                record = {
                    "rollout_id": str(env["rollout_id"]),
                    "split": split,
                    "task_id": int(rollout.task_id),
                    "task_name": task_names.get(
                        int(rollout.task_id), str(rollout.task_description)
                    ),
                    "episode_idx": int(rollout.episode_idx),
                    "environment_seed": env.get("environment_seed"),
                    "environment_reset_index": env.get("environment_reset_index"),
                    "failed": not bool(rollout.episode_success),
                    "num_inferences": len(scores),
                    "task_min_step": int(rollout.task_min_step),
                    "scores": np.asarray(scores).tolist(),
                }
                stream.write(json.dumps(_json_value(record), allow_nan=False) + "\n")


def evaluate_official_checkpoint(
    *,
    export_dir,
    safe_repo,
    checkpoint,
    config,
    output_dir,
    device="cuda",
    expected_safe_commit=OFFICIAL_SAFE_COMMIT,
):
    """Evaluate via pinned official loader/model/metrics/conformal implementations."""
    export_dir = Path(export_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    config = Path(config).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_repo, safe_commit = verify_safe_checkout(safe_repo, expected_safe_commit)
    if not checkpoint.is_file() or not config.is_file():
        raise ValueError("Checkpoint and saved official config must both exist")
    sys.path.insert(0, str(safe_repo))

    try:
        import torch
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader

        from failure_prob.data import load_rollouts, split_rollouts
        from failure_prob.data.utils import RolloutDataset
        from failure_prob.model import get_model
        from failure_prob.utils.metrics import (
            eval_functional_conformal,
            eval_scores_roc_prc,
        )
        from failure_prob.utils.random import seed_everything
        from failure_prob.utils.routines import model_forward_dataloader
    except ImportError as error:
        raise RuntimeError(
            "Activate the SAFE environment and evaluate with the pinned SAFE checkout"
        ) from error

    cfg = OmegaConf.load(config)
    cfg.dataset.data_path_prefix = ""
    cfg.dataset.data_path = str(export_dir)
    cfg.dataset.data_path_unseen = None
    cfg.dataset.load_to_cuda = str(device).startswith("cuda")
    cfg.train.log_precomputed = False
    cfg.train.log_precomputed_only = False
    seed = int(cfg.train.seed)

    seed_everything(0)
    all_rollouts = load_rollouts(cfg)
    env_records = _load_export_metadata(export_dir)
    identities = _validate_identity_alignment(all_rollouts, env_records)
    task_names = _task_names(export_dir)
    if cfg.dataset.load_to_cuda:
        all_rollouts = [rollout.to(device) for rollout in all_rollouts]

    seed_everything(seed)
    rollouts_by_split = split_rollouts(cfg, all_rollouts)
    split_manifest = _split_manifest(
        rollouts_by_split, identities=identities, task_names=task_names
    )
    if split_manifest["num_unique_rollouts"] != len(all_rollouts):
        raise ValueError("Official split is not exhaustive over the loaded export")

    datasets = {
        name: RolloutDataset(cfg, rollouts)
        for name, rollouts in rollouts_by_split.items()
    }
    dataloaders = {
        name: DataLoader(
            dataset,
            batch_size=cfg.model.batch_size,
            shuffle=False,
            num_workers=0,
        )
        for name, dataset in datasets.items()
    }
    train_rollouts = rollouts_by_split["train"]
    model = get_model(cfg, int(train_rollouts[0].hidden_states.shape[-1]))
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    scores_by_split = {}
    for split, loader in dataloaders.items():
        with torch.no_grad():
            scores, masks, _ = model_forward_dataloader(model, loader)
        scores = scores.detach().cpu().numpy()
        lengths = masks.sum(dim=-1).detach().cpu().numpy().astype(int)
        scores_by_split[split] = [scores[i, :lengths[i]] for i in range(len(lengths))]
        if any(not np.all(np.isfinite(x)) for x in scores_by_split[split]):
            raise ValueError(f"Model produced non-finite scores in split {split}")

    scalar_metrics = {}
    cp_rows = []
    bands = {}
    methods = {
        "model": scores_by_split,
        "constant": baseline_scores(scores_by_split, "constant"),
        "time_only": baseline_scores(scores_by_split, "time_only"),
    }
    for method, method_scores in methods.items():
        metrics = eval_scores_roc_prc(
            rollouts_by_split,
            copy.deepcopy(method_scores),
            method,
            list(OFFICIAL_TIME_QUANTILES),
            plot_auc_curves=False,
            plot_score_curves=False,
        )
        scalar_metrics.update(metrics)
        np.random.seed(seed)
        dataframe, method_bands = eval_functional_conformal(
            rollouts_by_split,
            copy.deepcopy(method_scores),
            method,
            alphas=list(OFFICIAL_ALPHAS),
            calib_split_names=["val_seen"],
            test_split_names=["val_unseen"],
            align_method="extend",
        )
        official_rows = dataframe.to_dict(orient="records")
        rows = evaluate_functional_bands_by_task(
            rollouts_by_split["val_unseen"],
            method_scores["val_unseen"],
            method_bands,
            method=method,
            task_names=task_names,
        )
        validate_persisted_rows_against_official(rows, official_rows)
        cp_rows.extend(rows)
        for alpha, band in method_bands.items():
            bands[f"{method}_alpha_{float(alpha):.2f}"] = np.asarray(band)

    duration, leakage_warnings = duration_diagnostics(rollouts_by_split, task_names)
    selection_value = scalar_metrics.get(SELECTION_METRIC)
    metrics_output = {
        "schema_version": 1,
        "official_safe_protocol": True,
        "selection_metric_name": SELECTION_METRIC,
        "selection_metric_value": selection_value,
        "scalar_metrics": scalar_metrics,
        "duration_diagnostics": duration,
        "rollout_length_warnings": leakage_warnings,
        "functional_conformal": {
            "calibration_split": "val_seen successes only",
            "test_split": "val_unseen",
            "alignment": "extend final score to global maximum length",
            "calibration_partition": "seeded 30 percent regression, 70 percent modulation",
            "crossing_rule": "score greater than or equal to upper band",
            "alphas": list(OFFICIAL_ALPHAS),
        },
    }

    write_json(output_dir / "metrics.json", metrics_output)
    write_json(output_dir / "split_manifest.json", split_manifest)
    _save_scores(
        output_dir / "scores.jsonl",
        rollouts_by_split,
        scores_by_split,
        identities,
        task_names,
    )
    with (output_dir / "functional_conformal.csv").open("w", newline="") as stream:
        fieldnames = sorted({key for row in cp_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([_json_value(row) for row in cp_rows])
    np.savez_compressed(output_dir / "functional_bands.npz", **bands)
    conversion = export_dir / "conversion_report.json"
    provenance = {
        "schema_version": 1,
        "safe_repository": str(safe_repo),
        "safe_repository_commit": safe_commit,
        "export_dir": str(export_dir),
        "conversion_report_sha256": (
            sha256_file(conversion) if conversion.is_file() else None
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "model": str(cfg.model.name),
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "dataset_config": OmegaConf.to_container(cfg.dataset, resolve=True),
        "seed": seed,
        "device": device,
        "feature": {
            "name": str(cfg.dataset.feat_name),
            "horizon_idx_rel": str(cfg.dataset.horizon_idx_rel),
            "diff_idx_rel": str(cfg.dataset.diff_idx_rel),
            "dimension": int(cfg.dataset.dim_features),
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    return metrics_output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-safe-commit", default=OFFICIAL_SAFE_COMMIT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    metrics = evaluate_official_checkpoint(
        export_dir=args.export_dir,
        safe_repo=args.safe_repo,
        checkpoint=args.checkpoint,
        config=args.config,
        output_dir=args.output_dir,
        device=args.device,
        expected_safe_commit=args.expected_safe_commit,
    )
    print(json.dumps(_json_value(metrics), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
