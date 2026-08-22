"""Run the official SAFE grid with training-only, outcome-stratified inner CV."""

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
    from .subtask_safe_evaluation import (
        parent_aggregated_early_metrics,
        parent_failed,
        parent_group_id,
        parent_task_name,
        semantic_subtask_score_records,
        subtask_fixed_prefix_selection,
        validate_subtask_catalog,
    )
except ImportError:
    from subtask_safe_evaluation import (
        parent_aggregated_early_metrics,
        parent_failed,
        parent_group_id,
        parent_task_name,
        semantic_subtask_score_records,
        subtask_fixed_prefix_selection,
        validate_subtask_catalog,
    )

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
        select_supported_stages,
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
        select_supported_stages,
        validate_prefix_horizons,
    )

try:
    from .train_seen_tasks import (
        CLASS_WEIGHTING_MODES,
        LOSS_MODES,
        OFFICIAL_SAFE_COMMIT,
        TASK_TYPE_FILTERS,
        configure_training_objective,
        filter_aligned_task_type,
        load_env_records,
        load_outer_split_ids,
        make_config,
        make_manifest_split,
        make_seen_split,
        resolve_task_type_selection,
        resolve_class_weights,
        score_splits,
        set_task_min_step_from_training,
        train_epoch_without_wandb,
        training_objective_metadata,
        validate_alignment,
        verify_safe_repo,
        write_json,
    )
except ImportError:
    from train_seen_tasks import (
        CLASS_WEIGHTING_MODES,
        LOSS_MODES,
        OFFICIAL_SAFE_COMMIT,
        TASK_TYPE_FILTERS,
        configure_training_objective,
        filter_aligned_task_type,
        load_env_records,
        load_outer_split_ids,
        make_config,
        make_manifest_split,
        make_seen_split,
        resolve_task_type_selection,
        resolve_class_weights,
        score_splits,
        set_task_min_step_from_training,
        train_epoch_without_wandb,
        training_objective_metadata,
        validate_alignment,
        verify_safe_repo,
        write_json,
    )


SELECTORS = ("0.0", "1.0", "concat-2")
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
REGULARIZATION = (1e-3, 1e-2, 1e-1)
SELECTION_UNITS = ("rollout", "parent_early", "subtask_fixed_prefix")


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


def grouped_record_id(record, group_field):
    return (
        parent_group_id(record)
        if group_field == "parent_or_rollout_id"
        else str(record.get(group_field, ""))
    )


def make_inner_folds(
    outer_train,
    identity,
    *,
    num_folds=3,
    seed=0,
    group_field=None,
):
    """Build inner folds, optionally keeping all segments of a parent together."""
    if group_field is not None:
        parent_groups = {}
        for rollout in outer_train:
            env = identity[id(rollout)][1]
            group_id = grouped_record_id(env, group_field)
            if not group_id:
                raise ValueError(f"Environment record is missing {group_field!r}")
            stratum = (parent_task_name(env), int(parent_failed(env)))
            group = parent_groups.setdefault(
                group_id,
                {"stratum": stratum, "rollouts": []},
            )
            if group["stratum"] != stratum:
                raise ValueError(f"Grouped split metadata changed for {group_id}")
            group["rollouts"].append(rollout)
        grouped_parents = defaultdict(list)
        for group_id, group in parent_groups.items():
            grouped_parents[group["stratum"]].append(group_id)
        validation_group_ids = [set() for _ in range(num_folds)]
        for stratum, values in sorted(grouped_parents.items()):
            values = sorted(values)
            if len(values) < num_folds:
                raise ValueError(
                    f"Parent stratum {stratum} has {len(values)} groups, fewer "
                    f"than {num_folds} folds"
                )
            rng = random.Random(f"{seed}:{stratum[0]}:{stratum[1]}")
            rng.shuffle(values)
            for index, group_id in enumerate(values):
                validation_group_ids[index % num_folds].add(group_id)
        validation_by_fold = [
            [
                rollout
                for group_id in sorted(group_ids)
                for rollout in parent_groups[group_id]["rollouts"]
            ]
            for group_ids in validation_group_ids
        ]
    else:
        grouped = defaultdict(list)
        for rollout in outer_train:
            grouped[(int(rollout.task_id), int(rollout.episode_success))].append(
                rollout
            )
        validation_by_fold = [[] for _ in range(num_folds)]
        for group_key in sorted(grouped):
            values = sorted(
                grouped[group_key],
                key=lambda rollout: str(identity[id(rollout)][1]["rollout_id"]),
            )
            rng = random.Random(f"{seed}:{group_key[0]}:{group_key[1]}")
            rng.shuffle(values)
            if len(values) < num_folds:
                raise ValueError(
                    f"Group {group_key} is too small for {num_folds} folds"
                )
            for index, rollout in enumerate(values):
                validation_by_fold[index % num_folds].append(rollout)
    outer_ids = {identity[id(rollout)][1]["rollout_id"] for rollout in outer_train}
    folds = []
    for fold, validation in enumerate(validation_by_fold):
        validation_ids = {
            identity[id(rollout)][1]["rollout_id"] for rollout in validation
        }
        training = [
            rollout
            for rollout in outer_train
            if identity[id(rollout)][1]["rollout_id"] not in validation_ids
        ]
        training_ids = {identity[id(rollout)][1]["rollout_id"] for rollout in training}
        if training_ids & validation_ids or training_ids | validation_ids != outer_ids:
            raise AssertionError(f"Fold {fold} is not disjoint and exhaustive")
        if group_field is None:
            for split in (training, validation):
                for task_id in sorted({int(rollout.task_id) for rollout in split}):
                    labels = {
                        int(rollout.episode_success)
                        for rollout in split
                        if int(rollout.task_id) == task_id
                    }
                    if labels != {0, 1}:
                        raise ValueError(
                            f"Fold {fold}, task {task_id} is not outcome-stratified"
                        )
        else:
            training_groups = {
                grouped_record_id(identity[id(rollout)][1], group_field)
                for rollout in training
            }
            validation_groups = {
                grouped_record_id(identity[id(rollout)][1], group_field)
                for rollout in validation
            }
            if training_groups & validation_groups:
                raise AssertionError(f"Fold {fold} leaks {group_field} groups")
            for split_name, split in (
                ("training", training),
                ("validation", validation),
            ):
                labels = {int(rollout.episode_success) for rollout in split}
                if labels != {0, 1}:
                    raise ValueError(
                        f"Fold {fold} grouped {split_name} split lacks both labels"
                    )
        folds.append((training, validation))
    return folds


def count_split(rollouts):
    return {
        "rollouts": len(rollouts),
        "successes": sum(int(rollout.episode_success) for rollout in rollouts),
        "failures": sum(not int(rollout.episode_success) for rollout in rollouts),
    }


def causal_prefix_roc_auc(rollouts, scores, identity, prefix):
    """Score one preregistered causal horizon on inner validation segments."""
    from sklearn.metrics import roc_auc_score

    pairs = [
        (rollout, score)
        for rollout, score in zip(rollouts, scores)
        if int(identity[id(rollout)][1].get("causal_prefix_inferences", -1))
        == int(prefix)
    ]
    labels = [not bool(int(rollout.episode_success)) for rollout, _ in pairs]
    if len(pairs) < 2 or len(set(labels)) < 2:
        raise ValueError(f"Inner validation prefix {prefix} lacks both failure labels")
    risks = [float(np.max(np.asarray(score, dtype=np.float64))) for _, score in pairs]
    return float(roc_auc_score(labels, risks))


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
    source_env_records = load_env_records(args.export_dir)
    if args.outer_split_manifest is not None and args.selection_manifest is not None:
        raise ValueError(
            "--outer-split-manifest and --selection-manifest are mutually exclusive"
        )
    fixed_split_ids = load_outer_split_ids(
        args.selection_manifest or args.outer_split_manifest
    )
    selected_task_names = None
    if args.selection_manifest is not None:
        selected_task_names = (
            set(fixed_split_ids["manifest"].get("included_tasks", [])) or None
        )
    task_selection = resolve_task_type_selection(
        args.export_dir,
        args.task_type,
        selected_task_names,
    )
    selected_task_ids = set(task_selection["selected_task_ids"])
    env_records = [
        env_record
        for env_record in source_env_records
        if int(env_record[1]["task_id"]) in selected_task_ids
    ]
    fixed_outer_split_path = (
        fixed_split_ids["manifest"].get("fixed_outer_split_manifest")
        if args.selection_manifest is not None
        else (fixed_split_ids["path"] if fixed_split_ids is not None else None)
    )
    inner_group_field = (
        "parent_or_rollout_id"
        if args.selection_unit in {"parent_early", "subtask_fixed_prefix"}
        else (
            "parent_rollout_id"
            if fixed_split_ids is not None
            and fixed_split_ids.get("split_unit") == "parent_rollout"
            else None
        )
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
    if args.selection_unit == "parent_early" and (causal_requested or online_requested):
        raise ValueError(
            "Parent-early selection cannot be combined with prefix-transformed SAFE modes"
        )
    if args.selection_unit == "subtask_fixed_prefix" and (
        causal_requested or online_requested
    ):
        raise ValueError(
            "Subtask-label fixed-prefix selection cannot be combined with "
            "prefix-transformed SAFE modes"
        )
    if (
        args.selection_unit == "subtask_fixed_prefix"
        and args.subtask_evaluation_export_dir is None
    ):
        raise ValueError(
            "--subtask-evaluation-export-dir is required for subtask-label selection"
        )
    if args.selection_unit == "subtask_fixed_prefix":
        prefixes = [int(value) for value in args.subtask_selection_prefixes]
        if not prefixes or any(value <= 0 for value in prefixes):
            raise ValueError(
                "--subtask-selection-prefixes must contain positive integers"
            )
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("--subtask-selection-prefixes must be unique")
        args.subtask_selection_prefixes = prefixes
    if causal_requested and inner_group_field != "parent_rollout_id":
        raise ValueError(
            "Causal Subtask-SAFE CV requires a parent_rollout selection manifest"
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
    if causal_requested and args.causal_selection_prefix not in causal_config.horizons:
        raise ValueError(
            "--causal-selection-prefix must be included in " "--causal-prefix-horizons"
        )
    subtask_catalog = None
    if args.selection_unit == "subtask_fixed_prefix":
        subtask_catalog = validate_subtask_catalog(
            load_env_records(args.subtask_evaluation_export_dir)
        )
    if (
        fixed_split_ids is not None
        and fixed_split_ids["split_seed"] is not None
        and int(fixed_split_ids["split_seed"]) != args.split_seed
    ):
        raise ValueError("Outer split manifest split_seed does not match --split-seed")
    source_groups = defaultdict(int)
    selected_env_by_id = {}
    for _, env in env_records:
        source_groups[(int(env["task_id"]), int(env["episode_success"]))] += 1
        selected_env_by_id[str(env["rollout_id"])] = env
    task_ids = sorted({task_id for task_id, _ in source_groups})
    if args.selection_manifest is None:
        missing_groups = [
            (task_id, success)
            for task_id in task_ids
            for success in (0, 1)
            if source_groups[(task_id, success)] <= args.train_per_class
        ]
        if missing_groups:
            raise ValueError(
                "Each task/outcome group must contain more than train_per_class; "
                f"invalid groups: {missing_groups}"
            )
        outer_train_count = len(task_ids) * 2 * args.train_per_class
        outer_test_count = len(env_records) - outer_train_count
    else:
        requested = fixed_split_ids["train"] | fixed_split_ids["test"]
        missing = sorted(requested - set(selected_env_by_id))
        if missing:
            raise ValueError(
                "Selection manifest contains rollout IDs absent from the selected export: "
                + ", ".join(missing[:5])
            )
        outer_train_count = len(fixed_split_ids["train"])
        outer_test_count = len(fixed_split_ids["test"])
    plan = {
        "schema_version": 1,
        "official_safe_commit": OFFICIAL_SAFE_COMMIT,
        "protocol": "training-only outcome-stratified inner CV within fixed outer training pool",
        "outer_test_used": False,
        "task_type_filter": args.task_type,
        "source_num_tasks": task_selection["source_num_tasks"],
        "selected_task_names": task_selection["selected_task_names"],
        "selected_task_types": task_selection["selected_task_types"],
        "num_tasks": len(task_ids),
        "num_source_rollouts": len(source_env_records),
        "num_selected_rollouts": len(env_records),
        "outer_train_rollouts": outer_train_count,
        "outer_test_rollouts": outer_test_count,
        "train_per_task_class": (
            None if args.selection_manifest is not None else args.train_per_class
        ),
        "selection_manifest": (
            fixed_split_ids["path"] if args.selection_manifest is not None else None
        ),
        "class_weighting": args.class_weighting,
        "training_objective": training_objective_metadata(
            args.loss_mode, args.focal_gamma
        ),
        "model": args.model,
        "num_runs": len(all_runs),
        "num_configurations": len(all_runs) // args.num_folds,
        "num_folds": args.num_folds,
        "epochs": args.epochs,
        "split_seed": args.split_seed,
        "inner_seed": args.inner_seed,
        "inner_group_field": inner_group_field,
        "outer_split_manifest": (fixed_outer_split_path),
        "selection_metric": (
            "mean task-macro parent ROC-AUC at landmarks "
            + ", ".join(str(value) for value in args.parent_selection_landmarks)
            if args.selection_unit == "parent_early"
            else (
                "mean hierarchical task/stage-macro subtask ROC-AUC at fixed "
                "inference prefixes "
                + ", ".join(str(value) for value in args.subtask_selection_prefixes)
                if args.selection_unit == "subtask_fixed_prefix"
                else (
                    f"mean causal prefix-{args.causal_selection_prefix} ROC-AUC across folds"
                    if causal_requested
                    else "mean falert_early_roc_auc/model_inner_val across folds"
                    if not online_requested
                    else "mean online-prefix falert_early_roc_auc/model_inner_val across folds"
                )
            )
        ),
        "selection_unit": args.selection_unit,
        "parent_selection_landmarks": (
            list(args.parent_selection_landmarks)
            if args.selection_unit == "parent_early"
            else None
        ),
        "subtask_evaluation_export_dir": (
            str(Path(args.subtask_evaluation_export_dir).resolve())
            if args.selection_unit == "subtask_fixed_prefix"
            else None
        ),
        "subtask_selection_prefixes": (
            list(args.subtask_selection_prefixes)
            if args.selection_unit == "subtask_fixed_prefix"
            else None
        ),
        "subtask_evaluation_catalog_segments": (
            len(subtask_catalog) if subtask_catalog is not None else None
        ),
        "online_safe": (
            online_config_dict(online_config) if online_requested else None
        ),
        "causal_subtask_safe": (
            {
                "training_mode": causal_config.training_mode,
                "test_mode": "fixed" if causal_config.enabled else "none",
                "horizons": list(causal_config.horizons),
                "random_prefixes_per_segment": (
                    causal_config.random_prefixes_per_segment
                ),
                "conditioning": causal_config.conditioning,
                "label_mode": causal_config.label_mode,
                "failure_horizon": causal_config.failure_horizon,
                "temporal_representation": causal_config.temporal_representation,
                "temporal_window": causal_config.temporal_window,
                "min_stage_successes": causal_config.min_stage_successes,
                "min_stage_failures": causal_config.min_stage_failures,
                "requested_stages": args.stages,
                "selection_prefix": int(args.causal_selection_prefix),
                "split_before_prefix_expansion": True,
            }
            if causal_requested
            else None
        ),
    }
    plan_path = output_root / f"cv_plan_{args.model}.json"
    if plan_path.is_file():
        previous_plan = json.loads(plan_path.read_text())
        if previous_plan != plan:
            raise ValueError(
                f"Existing CV plan is incompatible with this command: {plan_path}"
            )
    else:
        write_json(plan_path, plan)
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
                (output_root / run.slug / "metrics.json").is_file() for run in pair_runs
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
            configure_training_objective(cfg, args.loss_mode)
            seed_everything(0)
            source_rollouts = load_rollouts_from_root(Path(args.export_dir), cfg)
            all_rollouts, selected_env_records, identity = filter_aligned_task_type(
                source_rollouts,
                source_env_records,
                task_selection,
            )
            if [path for path, _ in selected_env_records] != [
                path for path, _ in env_records
            ]:
                raise AssertionError("Task-type filtering changed environment order")
            if args.selection_manifest is not None:
                outer_train, outer_test, _ = make_manifest_split(
                    all_rollouts,
                    identity,
                    fixed_split_ids,
                )
            else:
                outer_train, outer_test, _ = make_seen_split(
                    all_rollouts,
                    identity,
                    train_per_class=args.train_per_class,
                    split_seed=args.split_seed,
                    fixed_split_ids=fixed_split_ids,
                )
            stage_selection = None
            if causal_requested:
                stage_selection = select_supported_stages(
                    outer_train,
                    identity,
                    min_successes=causal_config.min_stage_successes,
                    min_failures=causal_config.min_stage_failures,
                    requested_stages=args.stages,
                )
                task_cutoffs = {
                    "protocol": "per_causal_prefix_length",
                    "selected_stages": stage_selection["selected_stages"],
                }
            elif online_requested:
                task_cutoffs = {
                    "protocol": "per_online_prefix_length",
                    "mode": online_config.mode,
                    "landmark_fraction": online_config.landmark_fraction,
                    "derived_inside_each_inner_training_fold": True,
                }
            else:
                task_cutoffs = set_task_min_step_from_training(outer_train, outer_test)
            folds = make_inner_folds(
                outer_train,
                identity,
                num_folds=args.num_folds,
                seed=args.inner_seed,
                group_field=inner_group_field,
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
                inner_train_source, inner_val_source = folds[run.fold]
                run_identity = identity
                causal_fold = None
                online_fold = None
                if online_requested:
                    online_fold = prepare_online_splits(
                        inner_train_source,
                        inner_val_source,
                        identity,
                        config=online_config,
                    )
                    inner_train = online_fold["train"]
                    inner_val = online_fold["test"]
                    run_identity = online_fold["identity"]
                elif causal_requested:
                    fold_config = CausalPrefixConfig(
                        training_mode=causal_config.training_mode,
                        horizons=causal_config.horizons,
                        random_prefixes_per_segment=(
                            causal_config.random_prefixes_per_segment
                        ),
                        conditioning=causal_config.conditioning,
                        label_mode=causal_config.label_mode,
                        failure_horizon=causal_config.failure_horizon,
                        temporal_representation=causal_config.temporal_representation,
                        temporal_window=causal_config.temporal_window,
                        # Stage support is selected once from the outer training
                        # pool. Inner validation never influences the catalog.
                        min_stage_successes=0,
                        min_stage_failures=0,
                    )
                    causal_fold = prepare_causal_splits(
                        inner_train_source,
                        inner_val_source,
                        identity,
                        config=fold_config,
                        random_seed=args.inner_seed + run.fold,
                        requested_stages=stage_selection["selected_stages"],
                    )
                    inner_train = causal_fold["train"]
                    inner_val = causal_fold["test"]
                    run_identity = causal_fold["identity"]
                    for rollout in inner_train + inner_val:
                        rollout.task_min_step = len(rollout.hidden_states)
                else:
                    inner_train = inner_train_source
                    inner_val = inner_val_source
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
                    training_class_weights = [
                        float(value)
                        for value in resolve_class_weights(
                            datasets["inner_train"],
                            args.class_weighting,
                        )
                    ]
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
                                class_weighting=args.class_weighting,
                                loss_mode=args.loss_mode,
                                focal_gamma=args.focal_gamma,
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
                    parent_selection = None
                    subtask_selection = None
                    if args.selection_unit == "parent_early":
                        parent_selection = parent_aggregated_early_metrics(
                            inner_val,
                            scores["inner_val"],
                            run_identity,
                            landmarks=args.parent_selection_landmarks,
                        )
                        value = parent_selection["selection_value"]
                        selection_metric = parent_selection["selection_metric"]
                    elif args.selection_unit == "subtask_fixed_prefix":
                        inner_train_parents = {
                            parent_group_id(run_identity[id(item)][1])
                            for item in inner_train_source
                        }
                        inner_val_parents = {
                            parent_group_id(run_identity[id(item)][1])
                            for item in inner_val_source
                        }
                        if inner_train_parents & inner_val_parents:
                            raise AssertionError(
                                "Subtask-label selection leaked inner-fold parents"
                            )
                        training_catalog = [
                            record
                            for record in subtask_catalog
                            if parent_group_id(record) in inner_train_parents
                        ]
                        validation_catalog = [
                            record
                            for record in subtask_catalog
                            if parent_group_id(record) in inner_val_parents
                        ]
                        validation_score_records = semantic_subtask_score_records(
                            inner_val,
                            scores["inner_val"],
                            run_identity,
                            validation_catalog,
                        )
                        subtask_selection = subtask_fixed_prefix_selection(
                            training_catalog,
                            validation_score_records,
                            prefixes=args.subtask_selection_prefixes,
                        )
                        value = subtask_selection["selection_value"]
                        selection_metric = subtask_selection["selection_metric"]
                    elif causal_requested:
                        value = causal_prefix_roc_auc(
                            inner_val,
                            scores["inner_val"],
                            run_identity,
                            args.causal_selection_prefix,
                        )
                        selection_metric = (
                            f"causal_prefix_roc_auc/inner_val_prefix_"
                            f"{args.causal_selection_prefix}"
                        )
                    else:
                        value = metrics["falert_early_roc_auc/model_inner_val"]
                        selection_metric = "falert_early_roc_auc/model_inner_val"
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
                    print(
                        f"ERROR {run.slug}: {type(error).__name__}: {error}", flush=True
                    )
                    if args.fail_fast:
                        raise
                    del model, optimizer, scheduler, datasets, loaders
                    if str(args.device).startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue
                record = {
                    "schema_version": 1,
                    "status": "complete",
                    "slug": run.slug,
                    "model": run.model,
                    "class_weighting": args.class_weighting,
                    "training_objective": training_objective_metadata(
                        args.loss_mode, args.focal_gamma
                    ),
                    "training_class_weights": training_class_weights,
                    "selection_manifest": (
                        fixed_split_ids["path"]
                        if args.selection_manifest is not None
                        else None
                    ),
                    "task_type_filter": args.task_type,
                    "selected_task_names": task_selection["selected_task_names"],
                    "horizon_selector": run.horizon_selector,
                    "diffusion_selector": run.diffusion_selector,
                    "learning_rate": run.learning_rate,
                    "lambda_reg": run.lambda_reg,
                    "fold": run.fold,
                    "selection_metric": selection_metric,
                    "selection_value": value,
                    "scalar_metrics": metrics,
                    "counts": {
                        "outer_train": count_split(outer_train),
                        "outer_test_untouched": count_split(outer_test),
                        "inner_train": count_split(inner_train),
                        "inner_val": count_split(inner_val),
                        "inner_train_source_segments": len(inner_train_source),
                        "inner_val_source_segments": len(inner_val_source),
                    },
                    "outer_test_scored": False,
                    "inner_group_field": inner_group_field,
                    "parent_aggregated_selection": parent_selection,
                    "subtask_fixed_prefix_selection": subtask_selection,
                    "task_min_steps_from_outer_train": task_cutoffs,
                    "causal_subtask_safe": (
                        {
                            "protocol": causal_fold["protocol"],
                            "conditioning": causal_fold["conditioning"],
                            "temporal_representation": causal_fold[
                                "temporal_representation"
                            ],
                            "outer_training_stage_support": stage_selection,
                            "inner_prefix_counts": causal_fold["prefix_counts"],
                            "inner_target_counts_by_prefix": causal_fold[
                                "target_counts_by_prefix"
                            ],
                            "selection_prefix": int(args.causal_selection_prefix),
                        }
                        if causal_fold is not None
                        else None
                    ),
                    "online_safe": online_payload_summary(online_fold),
                    "inner_train_ids": [
                        run_identity[id(item)][1]["rollout_id"] for item in inner_train
                    ],
                    "inner_val_ids": [
                        run_identity[id(item)][1]["rollout_id"] for item in inner_val
                    ],
                    "inner_train_parent_ids": (
                        sorted(
                            {
                                grouped_record_id(
                                    run_identity[id(item)][1], inner_group_field
                                )
                                for item in inner_train
                            }
                        )
                        if inner_group_field is not None
                        else None
                    ),
                    "inner_val_parent_ids": (
                        sorted(
                            {
                                grouped_record_id(
                                    run_identity[id(item)][1], inner_group_field
                                )
                                for item in inner_val
                            }
                        )
                        if inner_group_field is not None
                        else None
                    ),
                    "loss_first": history[0],
                    "loss_last": history[-1],
                    "started_at": started,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                write_json(metrics_path, record)
                failure_path.unlink(missing_ok=True)
                with events_path.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "status": "complete",
                                "slug": run.slug,
                                "selection_value": value,
                                "at": record["completed_at"],
                            }
                        )
                        + "\n"
                    )
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
            "non-exhaustive outer training subset"
        ),
    )
    parser.add_argument(
        "--class-weighting",
        choices=CLASS_WEIGHTING_MODES,
        default="official_inverse_frequency",
    )
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default="official")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--inner-seed", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--horizon-selectors", nargs="+", default=list(SELECTORS))
    parser.add_argument("--diffusion-selectors", nargs="+", default=list(SELECTORS))
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=list(LEARNING_RATES)
    )
    parser.add_argument(
        "--regularization", nargs="+", type=float, default=list(REGULARIZATION)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--selection-unit",
        choices=SELECTION_UNITS,
        default="rollout",
        help=(
            "Select by official pseudo-rollout ROC-AUC, by scores stitched at "
            "parent-trajectory landmarks, or by active-subtask labels at fixed "
            "causal inference counts"
        ),
    )
    parser.add_argument(
        "--parent-selection-landmarks",
        nargs="+",
        type=float,
        default=[0.25, 0.5],
    )
    parser.add_argument(
        "--subtask-evaluation-export-dir",
        help=(
            "Official Subtask-SAFE export supplying semantic-stage boundaries and "
            "active-subtask labels for fixed-prefix selection"
        ),
    )
    parser.add_argument(
        "--subtask-selection-prefixes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Causal inference counts since semantic-stage entry",
    )
    parser.add_argument(
        "--online-safe-mode",
        choices=ONLINE_SAFE_MODES,
        default="none",
    )
    parser.add_argument("--online-safe-seed", type=int, default=0)
    parser.add_argument("--online-landmark-fraction", type=float, default=0.5)
    parser.add_argument(
        "--causal-prefix-mode",
        choices=PREFIX_TRAINING_MODES,
        default="none",
    )
    parser.add_argument(
        "--causal-prefix-horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_PREFIX_HORIZONS),
    )
    parser.add_argument("--random-prefixes-per-segment", type=int, default=3)
    parser.add_argument(
        "--causal-selection-prefix",
        type=int,
        default=8,
        help="Preregistered inner-validation prefix used for hyperparameter selection",
    )
    parser.add_argument(
        "--causal-conditioning",
        choices=CONDITIONING_MODES,
        default="none",
    )
    parser.add_argument(
        "--causal-label-mode",
        choices=CAUSAL_LABEL_MODES,
        default="eventual",
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
        help="Optional ParentTask::subtask_id allowlist",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_cv_grid(args)


if __name__ == "__main__":
    main()
