"""Train official SAFE on 7+7 per task and evaluate on the remaining 3+3."""

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
import subprocess
import sys

import numpy as np


OFFICIAL_SAFE_COMMIT = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
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


def resolve_hyperparameters(model_name, selection_summary=None):
    if selection_summary is None:
        return dict(MODEL_DEFAULTS[model_name])
    summary = json.loads(Path(selection_summary).read_text())
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
    path.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_safe_repo(safe_repo):
    safe_repo = Path(safe_repo).resolve()
    commit = subprocess.run(
        ["git", "-C", str(safe_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_SAFE_COMMIT:
        raise ValueError(f"SAFE checkout is at {commit}, expected {OFFICIAL_SAFE_COMMIT}")
    sys.path.insert(0, str(safe_repo))
    return safe_repo


def load_env_records(export_dir):
    from natsort import natsorted

    paths = natsorted(glob.glob(str(Path(export_dir) / "env_records" / "*.pkl")))
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


def make_seen_split(rollouts, identity, *, train_per_class=7, split_seed=0):
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
            rng.shuffle(values)
            if len(values) <= train_per_class:
                raise ValueError(
                    f"Task {task_id}, success={success} has {len(values)} rollouts; "
                    f"need more than {train_per_class}"
                )
            train.extend(values[:train_per_class])
            test.extend(values[train_per_class:])
            per_task[task_id]["success" if success else "failure"] = {
                "train": len(values[:train_per_class]),
                "test": len(values[train_per_class:]),
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


def train_epoch_without_wandb(model, optimizer, loader, device):
    import torch
    from failure_prob.utils.torch import move_to_device

    model.train()
    weights = loader.dataset.get_class_weights()
    losses = []
    for batch in loader:
        batch = move_to_device(batch, device)
        monitor_loss, _ = model.forward_compute_loss(batch, weights)
        regularization, _ = model.compute_regularization_loss(model.cfg.model.lambda_reg)
        total = monitor_loss + regularization
        if not torch.isfinite(total):
            raise RuntimeError("Official SAFE training loss became non-finite")
        optimizer.zero_grad()
        total.backward()
        if model.cfg.model.grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg.model.grad_max_norm)
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    return float(np.mean(losses))


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


def save_scores(path, rollouts_by_split, scores_by_split, identity, names, model, seed):
    with Path(path).open("w") as stream:
        for split, rollouts in rollouts_by_split.items():
            for rollout, score in zip(rollouts, scores_by_split[split]):
                env_path, env = identity[id(rollout)]
                metadata = env.get("robocasa_manifest_record", {})
                record = {
                    "rollout_id": env["rollout_id"],
                    "split": split,
                    "task_id": int(rollout.task_id),
                    "task_name": names[int(rollout.task_id)],
                    "failed": not bool(rollout.episode_success),
                    "model": model,
                    "seed": seed,
                    "scores": np.asarray(score).tolist(),
                    "num_inferences": len(score),
                    "task_min_step": int(rollout.task_min_step),
                    "video_path": str(env_path.with_suffix(".mp4")),
                    "video_frame_stride": int(metadata.get("video_frame_stride", 1)),
                    "inference_environment_steps": metadata.get(
                        "inference_environment_steps",
                        list(range(0, len(score) * int(env["replan_steps"]), int(env["replan_steps"]))),
                    ),
                }
                stream.write(json.dumps(json_value(record), allow_nan=False) + "\n")


def train_seen_model(args):
    output = Path(args.output_dir).resolve()
    metrics_path = output / "metrics.json"
    if args.resume and metrics_path.is_file():
        return json.loads(metrics_path.read_text())
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

    hyperparameters = resolve_hyperparameters(args.model, args.selection_summary)
    cfg = make_config(
        args.export_dir,
        args.model,
        args.seed,
        args.epochs,
        args.device,
        hyperparameters,
    )
    seed_everything(0)
    all_rollouts = load_rollouts_from_root(Path(args.export_dir), cfg)
    env_records = load_env_records(args.export_dir)
    identity = validate_alignment(all_rollouts, env_records)
    train_rollouts, test_rollouts, per_task = make_seen_split(
        all_rollouts,
        identity,
        train_per_class=args.train_per_class,
        split_seed=args.split_seed,
    )
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
    model = get_model(cfg, int(train_rollouts[0].hidden_states.shape[-1]))
    model.to(args.device)
    optimizer, scheduler = model.get_optimizer()
    history = []
    for epoch in range(args.epochs):
        history.append(train_epoch_without_wandb(model, optimizer, loaders["train"], args.device))
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
    names = task_names(args.export_dir)
    split_manifest = {
        "schema_version": 1,
        "protocol": "all_five_seen_outcome_stratified",
        "split_seed": args.split_seed,
        "train_per_task_class": args.train_per_class,
        "test_per_task_class": len(test_rollouts) // (2 * len(names)),
        "per_task": {names[key]: value for key, value in per_task.items()},
        "train": [identity[id(rollout)][1]["rollout_id"] for rollout in train_rollouts],
        "test": [identity[id(rollout)][1]["rollout_id"] for rollout in test_rollouts],
    }
    write_json(output / "split_manifest.json", split_manifest)
    save_scores(
        output / "scores.jsonl",
        rollouts_by_split,
        scores_by_split,
        identity,
        names,
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
        "selected_hyperparameters": hyperparameters,
        "selection_summary": (
            str(Path(args.selection_summary).resolve())
            if args.selection_summary is not None
            else None
        ),
        "task_min_step_source": "minimum inference length per task in the 70-rollout training split only",
        "task_min_steps": {names[key]: value for key, value in task_cutoffs.items()},
        "counts": {
            "train": len(train_rollouts),
            "test": len(test_rollouts),
            "train_successes": sum(int(rollout.episode_success) for rollout in train_rollouts),
            "train_failures": sum(not int(rollout.episode_success) for rollout in train_rollouts),
            "test_successes": sum(int(rollout.episode_success) for rollout in test_rollouts),
            "test_failures": sum(not int(rollout.episode_success) for rollout in test_rollouts),
        },
        "scalar_metrics": metrics,
        "primary_metric": "falert_early_roc_auc/model_test",
        "primary_value": metrics["falert_early_roc_auc/model_test"],
        "duration_only_test_roc_auc": float(roc_auc_score(duration_labels, duration)),
        "thresholding": (
            "No conformal threshold is fitted: all 30 held-out rollouts remain evaluation-only."
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-summary")
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
