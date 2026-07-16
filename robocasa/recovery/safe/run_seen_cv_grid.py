"""Run the official SAFE grid with three-fold CV inside the 70-rollout train pool."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import sys

import numpy as np

try:
    from .train_seen_tasks import (
        OFFICIAL_SAFE_COMMIT,
        load_env_records,
        make_config,
        make_seen_split,
        score_splits,
        set_task_min_step_from_training,
        train_epoch_without_wandb,
        validate_alignment,
        verify_safe_repo,
        write_json,
    )
except ImportError:
    from train_seen_tasks import (
        OFFICIAL_SAFE_COMMIT,
        load_env_records,
        make_config,
        make_seen_split,
        score_splits,
        set_task_min_step_from_training,
        train_epoch_without_wandb,
        validate_alignment,
        verify_safe_repo,
        write_json,
    )


SELECTORS = ("0.0", "1.0", "concat-2")
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
REGULARIZATION = (1e-3, 1e-2, 1e-1)


def parse_selector(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def slug_value(value):
    return str(value).replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class CvRun:
    model: str
    horizon_selector: str
    diffusion_selector: str
    learning_rate: float
    lambda_reg: float
    fold: int

    @property
    def slug(self):
        return "__".join(
            (
                self.model,
                f"h-{slug_value(self.horizon_selector)}",
                f"d-{slug_value(self.diffusion_selector)}",
                f"lr-{slug_value(f'{self.learning_rate:g}')}",
                f"reg-{slug_value(f'{self.lambda_reg:g}')}",
                f"fold-{self.fold}",
            )
        )


def generate_cv_runs(
    model,
    num_folds=3,
    horizon_selectors=SELECTORS,
    diffusion_selectors=SELECTORS,
    learning_rates=LEARNING_RATES,
    regularization=REGULARIZATION,
):
    return [
        CvRun(model, horizon, diffusion, lr, regularization, fold)
        for horizon, diffusion, lr, regularization, fold in itertools.product(
            horizon_selectors,
            diffusion_selectors,
            learning_rates,
            regularization,
            range(num_folds),
        )
    ]


def make_inner_folds(outer_train, identity, *, num_folds=3, seed=0):
    """Outcome-stratified folds within every task; outer test is never accepted."""
    grouped = defaultdict(list)
    for rollout in outer_train:
        grouped[(int(rollout.task_id), int(rollout.episode_success))].append(rollout)
    validation_by_fold = [[] for _ in range(num_folds)]
    for group_key in sorted(grouped):
        values = sorted(
            grouped[group_key],
            key=lambda rollout: str(identity[id(rollout)][1]["rollout_id"]),
        )
        rng = random.Random(f"{seed}:{group_key[0]}:{group_key[1]}")
        rng.shuffle(values)
        if len(values) < num_folds:
            raise ValueError(f"Group {group_key} is too small for {num_folds} folds")
        for index, rollout in enumerate(values):
            validation_by_fold[index % num_folds].append(rollout)
    outer_ids = {identity[id(rollout)][1]["rollout_id"] for rollout in outer_train}
    folds = []
    for fold, validation in enumerate(validation_by_fold):
        validation_ids = {identity[id(rollout)][1]["rollout_id"] for rollout in validation}
        training = [
            rollout
            for rollout in outer_train
            if identity[id(rollout)][1]["rollout_id"] not in validation_ids
        ]
        training_ids = {identity[id(rollout)][1]["rollout_id"] for rollout in training}
        if training_ids & validation_ids or training_ids | validation_ids != outer_ids:
            raise AssertionError(f"Fold {fold} is not disjoint and exhaustive")
        for split in (training, validation):
            for task_id in sorted({int(rollout.task_id) for rollout in split}):
                labels = {
                    int(rollout.episode_success)
                    for rollout in split
                    if int(rollout.task_id) == task_id
                }
                if labels != {0, 1}:
                    raise ValueError(f"Fold {fold}, task {task_id} is not outcome-stratified")
        folds.append((training, validation))
    return folds


def count_split(rollouts):
    return {
        "rollouts": len(rollouts),
        "successes": sum(int(rollout.episode_success) for rollout in rollouts),
        "failures": sum(not int(rollout.episode_success) for rollout in rollouts),
    }


def run_cv_grid(args):
    safe_repo = verify_safe_repo(args.safe_repo)
    import torch
    from torch.utils.data import DataLoader
    from failure_prob.data.pizero import load_rollouts_from_root
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.model import get_model
    from failure_prob.utils.metrics import eval_scores_roc_prc
    from failure_prob.utils.random import seed_everything

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_runs = generate_cv_runs(
        args.model,
        args.num_folds,
        args.horizon_selectors,
        args.diffusion_selectors,
        args.learning_rates,
        args.regularization,
    )
    plan = {
        "schema_version": 1,
        "official_safe_commit": OFFICIAL_SAFE_COMMIT,
        "protocol": "three-fold outcome-stratified CV within fixed 70-rollout training pool",
        "outer_test_used": False,
        "model": args.model,
        "num_runs": len(all_runs),
        "num_configurations": len(all_runs) // args.num_folds,
        "num_folds": args.num_folds,
        "epochs": args.epochs,
        "split_seed": args.split_seed,
        "inner_seed": args.inner_seed,
        "selection_metric": "mean falert_early_roc_auc/model_inner_val across folds",
    }
    write_json(output_root / f"cv_plan_{args.model}.json", plan)
    env_records = load_env_records(args.export_dir)
    events_path = output_root / f"cv_events_{args.model}.jsonl"
    completed = 0
    for horizon in args.horizon_selectors:
        for diffusion in args.diffusion_selectors:
            pair_runs = [
                run
                for run in all_runs
                if run.horizon_selector == horizon
                and run.diffusion_selector == diffusion
            ]
            if args.resume and all(
                (output_root / run.slug / "metrics.json").is_file()
                for run in pair_runs
            ):
                completed += len(pair_runs)
                print(
                    f"selector pair h={horizon} d={diffusion}: "
                    f"all {len(pair_runs)} runs complete",
                    flush=True,
                )
                continue
            print(f"loading selector pair h={horizon} d={diffusion}", flush=True)
            base_hyperparameters = {
                "horizon_selector": parse_selector(horizon),
                "diffusion_selector": parse_selector(diffusion),
                "learning_rate": LEARNING_RATES[0],
                "lambda_reg": REGULARIZATION[0],
            }
            cfg = make_config(
                args.export_dir,
                args.model,
                0,
                args.epochs,
                args.device,
                base_hyperparameters,
            )
            seed_everything(0)
            all_rollouts = load_rollouts_from_root(Path(args.export_dir), cfg)
            identity = validate_alignment(all_rollouts, env_records)
            outer_train, outer_test, _ = make_seen_split(
                all_rollouts,
                identity,
                train_per_class=args.train_per_class,
                split_seed=args.split_seed,
            )
            task_cutoffs = set_task_min_step_from_training(outer_train, outer_test)
            folds = make_inner_folds(
                outer_train,
                identity,
                num_folds=args.num_folds,
                seed=args.inner_seed,
            )
            if cfg.dataset.load_to_cuda:
                all_rollouts = [rollout.to(args.device) for rollout in all_rollouts]
            for run in pair_runs:
                run_root = output_root / run.slug
                metrics_path = run_root / "metrics.json"
                failure_path = run_root / "failure.json"
                if args.resume and metrics_path.is_file():
                    completed += 1
                    continue
                if args.resume and failure_path.is_file() and not args.retry_errors:
                    print(f"known failed run, skipping: {run.slug}", flush=True)
                    continue
                run_root.mkdir(parents=True, exist_ok=True)
                inner_train, inner_val = folds[run.fold]
                cfg.model.lr = run.learning_rate
                cfg.model.lambda_reg = run.lambda_reg
                cfg.train.seed = run.fold
                cfg.model.n_epochs = args.epochs
                seed_everything(run.fold)
                started = datetime.now(timezone.utc).isoformat()
                datasets = loaders = model = optimizer = scheduler = None
                try:
                    datasets = {
                        "inner_train": RolloutDataset(cfg, inner_train),
                        "inner_val": RolloutDataset(cfg, inner_val),
                    }
                    loaders = {
                        split: DataLoader(
                            dataset,
                            batch_size=cfg.model.batch_size,
                            shuffle=split == "inner_train",
                            num_workers=0,
                        )
                        for split, dataset in datasets.items()
                    }
                    model = get_model(cfg, int(inner_train[0].hidden_states.shape[-1]))
                    model.to(args.device)
                    optimizer, scheduler = model.get_optimizer()
                    history = []
                    for _ in range(args.epochs):
                        history.append(
                            train_epoch_without_wandb(
                                model,
                                optimizer,
                                loaders["inner_train"],
                                args.device,
                            )
                        )
                        if scheduler is not None:
                            scheduler.step()
                    scores = score_splits(model, loaders)
                    metrics = eval_scores_roc_prc(
                        {"inner_train": inner_train, "inner_val": inner_val},
                        copy.deepcopy(scores),
                        "model",
                        [0.25, 0.5, 0.75, 1.0],
                        plot_auc_curves=False,
                        plot_score_curves=False,
                    )
                except Exception as error:
                    failure = {
                        "schema_version": 1,
                        "status": "error",
                        "slug": run.slug,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                    write_json(failure_path, failure)
                    with events_path.open("a") as stream:
                        stream.write(json.dumps(failure) + "\n")
                    print(f"ERROR {run.slug}: {type(error).__name__}: {error}", flush=True)
                    if args.fail_fast:
                        raise
                    del model, optimizer, scheduler, datasets, loaders
                    if str(args.device).startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue
                value = metrics["falert_early_roc_auc/model_inner_val"]
                record = {
                    "schema_version": 1,
                    "status": "complete",
                    "slug": run.slug,
                    "model": run.model,
                    "horizon_selector": run.horizon_selector,
                    "diffusion_selector": run.diffusion_selector,
                    "learning_rate": run.learning_rate,
                    "lambda_reg": run.lambda_reg,
                    "fold": run.fold,
                    "selection_metric": "falert_early_roc_auc/model_inner_val",
                    "selection_value": value,
                    "scalar_metrics": metrics,
                    "counts": {
                        "outer_train": count_split(outer_train),
                        "outer_test_untouched": count_split(outer_test),
                        "inner_train": count_split(inner_train),
                        "inner_val": count_split(inner_val),
                    },
                    "outer_test_scored": False,
                    "task_min_steps_from_outer_train": task_cutoffs,
                    "inner_train_ids": [identity[id(item)][1]["rollout_id"] for item in inner_train],
                    "inner_val_ids": [identity[id(item)][1]["rollout_id"] for item in inner_val],
                    "loss_first": history[0],
                    "loss_last": history[-1],
                    "started_at": started,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                write_json(metrics_path, record)
                failure_path.unlink(missing_ok=True)
                with events_path.open("a") as stream:
                    stream.write(json.dumps({
                        "status": "complete",
                        "slug": run.slug,
                        "selection_value": value,
                        "at": record["completed_at"],
                    }) + "\n")
                completed += 1
                print(
                    f"[{completed}/{len(all_runs)}] {run.slug} val={value:.4f}",
                    flush=True,
                )
                del model, optimizer, scheduler, datasets, loaders
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            del all_rollouts, outer_train, outer_test, folds
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    return plan


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("indep", "lstm"), required=True)
    parser.add_argument("--train-per-class", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--inner-seed", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--horizon-selectors", nargs="+", default=list(SELECTORS))
    parser.add_argument("--diffusion-selectors", nargs="+", default=list(SELECTORS))
    parser.add_argument("--learning-rates", nargs="+", type=float, default=list(LEARNING_RATES))
    parser.add_argument("--regularization", nargs="+", type=float, default=list(REGULARIZATION))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_cv_grid(args)


if __name__ == "__main__":
    main()
