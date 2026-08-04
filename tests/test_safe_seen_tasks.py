import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

from robocasa.recovery.safe.render_score_videos import render_score_videos
from robocasa.recovery.safe.summarize_seen_tasks import summarize
from robocasa.recovery.safe.summarize_seen_cv import summarize_seen_cv
from robocasa.recovery.safe.run_seen_cv_grid import generate_cv_runs, make_inner_folds
from robocasa.recovery.safe.summarize_natural_rate_screen import (
    summarize as summarize_natural_rate_screen,
)
from robocasa.recovery.safe.build_natural_rate_experiment import (
    audit_natural_outcomes,
    build_experiment,
)
from robocasa.recovery.safe.analyze_natural_rate_screen import (
    analyze as analyze_natural_rate_screen,
)
from robocasa.recovery.safe.train_seen_tasks import (
    MODEL_DEFAULTS,
    causal_binary_monitor_loss,
    configure_training_objective,
    filter_aligned_task_type,
    load_outer_split_ids,
    make_manifest_split,
    make_seen_split,
    resolve_task_type_selection,
    resolve_class_weights,
    resolve_hyperparameters,
    set_task_min_step_from_training,
    training_objective_metadata,
)


class TestSeenTaskProtocol(unittest.TestCase):
    def test_binary_objective_disables_accumulated_score_output(self):
        official = SimpleNamespace(model=SimpleNamespace(cumsum=True, rmean=False))
        configure_training_objective(official, "official")
        self.assertTrue(official.model.cumsum)

        binary = SimpleNamespace(model=SimpleNamespace(cumsum=True, rmean=True))
        configure_training_objective(binary, "focal")
        self.assertFalse(binary.model.cumsum)
        self.assertFalse(binary.model.rmean)
        self.assertEqual(
            training_objective_metadata("focal", 1.5),
            {
                "loss_mode": "focal",
                "focal_gamma": 1.5,
                "score_output": "instantaneous_failure_probability",
            },
        )

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_causal_bce_and_focal_losses_use_failure_as_positive_class(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(model=SimpleNamespace(name="indep")),
            projector=torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid()),
        )
        with torch.no_grad():
            model.projector[0].weight.fill_(1.0)
            model.projector[0].bias.zero_()
        batch = {
            "features": torch.tensor([[[2.0]], [[-2.0]]]),
            "valid_masks": torch.ones(2, 1),
            # SAFE convention: one is success, zero is failure.
            "success_labels": torch.tensor([0, 1]),
        }
        bce = causal_binary_monitor_loss(model, batch, [1.0, 1.0], loss_mode="bce")
        focal = causal_binary_monitor_loss(
            model, batch, [1.0, 1.0], loss_mode="focal", focal_gamma=2.0
        )
        self.assertTrue(torch.isfinite(bce))
        self.assertTrue(torch.isfinite(focal))
        self.assertLess(float(focal), float(bce))

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_causal_binary_loss_supervises_only_prefix_endpoint(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(model=SimpleNamespace(name="indep")),
            projector=torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid()),
        )
        with torch.no_grad():
            model.projector[0].weight.fill_(1.0)
            model.projector[0].bias.zero_()
        common = {
            "valid_masks": torch.ones(1, 2),
            "success_labels": torch.tensor([0]),
        }
        low_endpoint = causal_binary_monitor_loss(
            model,
            {"features": torch.tensor([[[10.0], [-10.0]]]), **common},
            [1.0, 1.0],
            loss_mode="bce",
        )
        high_endpoint = causal_binary_monitor_loss(
            model,
            {"features": torch.tensor([[[-10.0], [10.0]]]), **common},
            [1.0, 1.0],
            loss_mode="bce",
        )
        self.assertGreater(float(low_endpoint), float(high_endpoint))

    def test_parent_rollout_manifest_allows_natural_single_class_subtasks(self):
        rollouts = []
        identity = {}
        train_ids = set()
        test_ids = set()
        parent_train = {"train-success", "train-failure"}
        parent_test = {"test-success", "test-failure"}
        for task_id in range(2):
            for split, parents in (("train", parent_train), ("test", parent_test)):
                for parent in sorted(parents):
                    success = int("failure" not in parent)
                    if task_id == 0:
                        success = 1
                    rollout = SimpleNamespace(
                        task_id=task_id,
                        episode_success=success,
                    )
                    rollout_id = f"{parent}-subtask-{task_id}"
                    rollouts.append(rollout)
                    identity[id(rollout)] = (
                        Path(f"{rollout_id}.pkl"),
                        {
                            "rollout_id": rollout_id,
                            "parent_rollout_id": parent,
                        },
                    )
                    (train_ids if split == "train" else test_ids).add(rollout_id)

        train, test, counts = make_manifest_split(
            rollouts,
            identity,
            {
                "train": train_ids,
                "test": test_ids,
                "split_unit": "parent_rollout",
                "manifest": {
                    "parent_train": sorted(parent_train),
                    "parent_test": sorted(parent_test),
                },
            },
        )
        self.assertEqual((len(train), len(test)), (4, 4))
        self.assertEqual(counts[0]["failure"], {"train": 0, "test": 0})
        self.assertEqual(counts[1]["failure"], {"train": 1, "test": 1})

    def test_inner_folds_keep_parent_segments_together(self):
        rollouts = []
        identity = {}
        for task_name in ("TaskA", "TaskB"):
            for failed in (False, True):
                for parent_index in range(3):
                    parent = f"{task_name}-{int(failed)}-{parent_index}"
                    for subtask_index in range(2):
                        rollout = SimpleNamespace(
                            task_id=subtask_index,
                            episode_success=(0 if failed and subtask_index == 1 else 1),
                        )
                        rollout_id = f"{parent}-segment-{subtask_index}"
                        rollouts.append(rollout)
                        identity[id(rollout)] = (
                            Path(f"{rollout_id}.pkl"),
                            {
                                "rollout_id": rollout_id,
                                "parent_rollout_id": parent,
                                "parent_task_name": task_name,
                                "parent_rollout_failed": failed,
                            },
                        )
        folds = make_inner_folds(
            rollouts,
            identity,
            num_folds=3,
            seed=4,
            group_field="parent_rollout_id",
        )
        self.assertEqual(len(folds), 3)
        for train, validation in folds:
            train_parents = {
                identity[id(item)][1]["parent_rollout_id"] for item in train
            }
            validation_parents = {
                identity[id(item)][1]["parent_rollout_id"] for item in validation
            }
            self.assertFalse(train_parents & validation_parents)
            self.assertEqual(len(validation_parents), 4)
            self.assertEqual(
                {int(item.episode_success) for item in validation},
                {0, 1},
            )

    def test_manifest_split_allows_unused_outer_training_pool(self):
        rollouts = []
        identity = {}
        train_ids = set()
        test_ids = set()
        for task_id in range(2):
            for success in (0, 1):
                for index in range(4):
                    rollout = SimpleNamespace(
                        task_id=task_id,
                        episode_success=success,
                    )
                    rollout_id = f"task-{task_id}-success-{success}-{index}"
                    rollouts.append(rollout)
                    identity[id(rollout)] = (
                        Path(f"{rollout_id}.pkl"),
                        {"rollout_id": rollout_id},
                    )
                    if index < 2:
                        train_ids.add(rollout_id)
                    elif index == 3:
                        test_ids.add(rollout_id)
        train, test, counts = make_manifest_split(
            rollouts,
            identity,
            {
                "train": train_ids,
                "test": test_ids,
            },
        )
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 4)
        self.assertEqual(counts[0]["success"], {"train": 2, "test": 1})
        selected = {identity[id(rollout)][1]["rollout_id"] for rollout in train + test}
        self.assertEqual(
            len(
                set(
                    identity_value[1]["rollout_id"]
                    for identity_value in identity.values()
                )
                - selected
            ),
            4,
        )

    def test_class_weighting_switch_uses_official_weights_or_none(self):
        expected = np.array([1.0, 1.5])

        class Dataset:
            def __init__(self):
                self.calls = 0
                self.cfg = SimpleNamespace(
                    model=SimpleNamespace(
                        lambda_fail=1.0,
                        lambda_success=1.0,
                    )
                )
                self.rollouts = [
                    SimpleNamespace(episode_success=0),
                    SimpleNamespace(episode_success=0),
                    SimpleNamespace(episode_success=1),
                ]

            def get_class_weights(self):
                self.calls += 1
                return expected

            def get_rollouts(self):
                return self.rollouts

        dataset = Dataset()
        self.assertIs(
            resolve_class_weights(dataset, "official_inverse_frequency"),
            expected,
        )
        self.assertEqual(dataset.calls, 1)
        unweighted = resolve_class_weights(dataset, "none")
        self.assertAlmostEqual(unweighted[0], 7 / 6)
        self.assertAlmostEqual(unweighted[1], 7 / 6)
        self.assertEqual(dataset.calls, 2)
        with self.assertRaisesRegex(ValueError, "Unknown class weighting"):
            resolve_class_weights(dataset, "other")

    def test_natural_rate_builder_uses_only_observed_quota_discards_and_fixed_test(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard"
            shard.mkdir()
            retained = [
                {
                    "rollout_id": "observed-a-s0",
                    "task_name": "TaskA",
                    "failed": False,
                },
                {
                    "rollout_id": "observed-a-s1",
                    "task_name": "TaskA",
                    "failed": False,
                },
                {
                    "rollout_id": "observed-a-f0",
                    "task_name": "TaskA",
                    "failed": True,
                },
                {
                    "rollout_id": "observed-b-s0",
                    "task_name": "TaskB",
                    "failed": False,
                },
                {
                    "rollout_id": "observed-b-f0",
                    "task_name": "TaskB",
                    "failed": True,
                },
                {
                    "rollout_id": "observed-b-f1",
                    "task_name": "TaskB",
                    "failed": True,
                },
            ]
            (shard / "manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in retained)
            )
            skipped = [
                {
                    "rollout_id": "observed-a-s2",
                    "task_name": "TaskA",
                    "status": "skipped_class_quota_reached",
                    "success": True,
                },
                {
                    "rollout_id": "observed-b-f2",
                    "task_name": "TaskB",
                    "status": "skipped_class_quota_reached",
                    "failed": True,
                },
                {
                    "rollout_id": "not-executed",
                    "task_name": "TaskA",
                    "status": "skipped_quota_reached",
                },
            ]
            (shard / "skipped.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in skipped)
            )
            audit = audit_natural_outcomes([shard])
            self.assertEqual(audit["num_observed_outcomes"], 8)
            self.assertEqual(
                audit["per_task"]["TaskA"]["natural_success_rate"],
                0.75,
            )
            self.assertEqual(
                audit["per_task"]["TaskB"]["natural_success_rate"],
                0.25,
            )
            export = root / "official"
            env_dir = export / "env_records"
            env_dir.mkdir(parents=True)
            outer_train = []
            outer_test = []
            index = 0
            for task_id, task_name in enumerate(("TaskA", "TaskB")):
                for success in (0, 1):
                    for item in range(5):
                        rollout_id = f"{task_name}-{success}-{item}"
                        record = {
                            "rollout_id": rollout_id,
                            "task_id": task_id,
                            "task_name": task_name,
                            "episode_success": success,
                        }
                        with (env_dir / f"{index:03d}.pkl").open("wb") as stream:
                            pickle.dump(record, stream)
                        index += 1
                        (outer_train if item < 4 else outer_test).append(rollout_id)
            outer_path = root / "outer.json"
            outer_path.write_text(
                json.dumps(
                    {
                        "split_seed": 0,
                        "train": outer_train,
                        "test": outer_test,
                    }
                )
            )
            output = root / "experiment"
            plan = build_experiment(
                collection_datasets=[shard],
                official_export=export,
                outer_split_manifest=outer_path,
                output_dir=output,
                subset_seeds=[7],
                min_per_class=1,
            )
            self.assertEqual(plan["samples_per_task"], 4)
            self.assertEqual(plan["fixed_outer_test_ids"], sorted(outer_test))
            natural_path = next(
                Path(record["path"])
                for record in plan["manifests"]
                if record["regime"] == "natural_rate"
            )
            natural = json.loads(natural_path.read_text())
            self.assertEqual(natural["test"], sorted(outer_test))
            self.assertEqual(
                natural["per_task"]["TaskA"]["train_successes"],
                3,
            )
            self.assertEqual(
                natural["per_task"]["TaskB"]["train_successes"],
                1,
            )
            self.assertFalse(set(natural["train"]) & set(natural["test"]))

    def test_natural_rate_summary_pairs_effects_by_subset_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regime_offsets = {
                "matched_weighted": 0.0,
                "natural_weighted": 0.1,
                "natural_unweighted": 0.04,
            }
            for regime, offset in regime_offsets.items():
                for subset_seed in (0, 1):
                    for model_seed in (0, 1):
                        run = (
                            root
                            / f"indep__{regime}__subset-{subset_seed}__seed-{model_seed}"
                        )
                        run.mkdir()
                        (run / "screen_run.json").write_text(
                            json.dumps(
                                {
                                    "model": "indep",
                                    "regime": regime,
                                    "subset_seed": subset_seed,
                                    "model_seed": model_seed,
                                    "class_weighting": (
                                        "none"
                                        if regime == "natural_unweighted"
                                        else "official_inverse_frequency"
                                    ),
                                }
                            )
                        )
                        value = 0.5 + 0.01 * subset_seed + offset
                        (run / "metrics.json").write_text(
                            json.dumps(
                                {
                                    "counts": {"train": 220, "test": 160},
                                    "scalar_metrics": {
                                        "falert_early_roc_auc/model_test": value,
                                        "falert_early_prc_auc/model_test": value - 0.02,
                                    },
                                }
                            )
                        )
            summary = summarize_natural_rate_screen(
                root,
                expected_subset_seeds=(0, 1),
                expected_model_seeds=(0, 1),
            )
            self.assertEqual(summary["num_completed_runs"], 12)
            effect = summary["paired_comparisons"]["indep"]["test_roc_auc"]
            self.assertAlmostEqual(effect["natural_minus_matched"]["mean"], 0.1)
            self.assertAlmostEqual(
                effect["weighted_minus_unweighted"]["mean"],
                0.06,
            )

    def test_detailed_natural_rate_analysis_uses_training_only_task_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "screen"
            output = Path(tmp) / "analysis"
            task_specs = (
                ("AtomicTask", "atomic", 0.1),
                ("CompositeTask", "composite", 0.7),
            )
            for model in ("indep", "lstm"):
                for regime in (
                    "matched_weighted",
                    "natural_weighted",
                    "natural_unweighted",
                ):
                    for subset_seed in (0, 1):
                        for model_seed in (0, 1):
                            run = (
                                root / f"{model}__{regime}__subset-{subset_seed}"
                                f"__seed-{model_seed}"
                            )
                            run.mkdir(parents=True)
                            (run / "screen_run.json").write_text(
                                json.dumps(
                                    {
                                        "model": model,
                                        "regime": regime,
                                        "subset_seed": subset_seed,
                                        "model_seed": model_seed,
                                        "class_weighting": (
                                            "none"
                                            if regime == "natural_unweighted"
                                            else "official_inverse_frequency"
                                        ),
                                    }
                                )
                            )
                            score_records = []
                            test_labels = []
                            test_scores = []
                            for task_name, task_type, offset in task_specs:
                                for failed in (False, True):
                                    for index in range(2):
                                        value = offset + (0.2 if failed else 0.0)
                                        score_records.append(
                                            {
                                                "rollout_id": (
                                                    f"train-{regime}-{subset_seed}-"
                                                    f"{task_name}-{int(failed)}-{index}"
                                                ),
                                                "split": "train",
                                                "task_name": task_name,
                                                "task_type": task_type,
                                                "failed": failed,
                                                "task_min_step": 2,
                                                "scores": [value, value],
                                            }
                                        )
                                    value = offset + (0.2 if failed else 0.0)
                                    score_records.append(
                                        {
                                            "rollout_id": (
                                                f"test-{task_name}-{int(failed)}"
                                            ),
                                            "split": "test",
                                            "task_name": task_name,
                                            "task_type": task_type,
                                            "failed": failed,
                                            "task_min_step": 2,
                                            "scores": [value, value],
                                        }
                                    )
                                    test_labels.append(int(failed))
                                    test_scores.append(value)
                            (run / "scores.jsonl").write_text(
                                "".join(
                                    json.dumps(record) + "\n"
                                    for record in score_records
                                )
                            )
                            from sklearn.metrics import roc_auc_score

                            raw_roc = float(roc_auc_score(test_labels, test_scores))
                            (run / "metrics.json").write_text(
                                json.dumps(
                                    {
                                        "counts": {
                                            "train": 8,
                                            "test": 4,
                                            "train_successes": 4,
                                            "train_failures": 4,
                                            "test_successes": 2,
                                            "test_failures": 2,
                                        },
                                        "scalar_metrics": {
                                            "falert_early_roc_auc/model_test": raw_roc,
                                        },
                                    }
                                )
                            )
            summary = analyze_natural_rate_screen(
                root,
                output,
                expected_subset_seeds=(0, 1),
                expected_model_seeds=(0, 1),
                formats=("png",),
                make_plots=True,
            )
            self.assertEqual(summary["num_runs"], 24)
            self.assertEqual(summary["fixed_test_rollouts"], 4)
            group = summary["groups"]["indep/matched_weighted"]
            self.assertAlmostEqual(group["raw_pooled_roc_auc"]["mean"], 0.75)
            self.assertAlmostEqual(group["task_z_pooled_roc_auc"]["mean"], 1.0)
            self.assertAlmostEqual(group["macro_task_roc_auc"]["mean"], 1.0)
            self.assertTrue((output / "group_metrics.csv").is_file())
            self.assertTrue((output / "per_task_metrics.csv").is_file())
            self.assertEqual(len(summary["figures"]), 3)
            self.assertTrue(all(Path(path).is_file() for path in summary["figures"]))

    def test_fixed_split_scales_to_ten_tasks_without_changing_per_task_balance(self):
        rollouts = []
        identity = {}
        for task_id in range(10):
            for success in (0, 1):
                for index in range(10):
                    rollout = SimpleNamespace(
                        task_id=task_id,
                        episode_success=success,
                    )
                    rollout_id = f"task-{task_id}-success-{success}-{index}"
                    rollouts.append(rollout)
                    identity[id(rollout)] = (
                        Path(f"{rollout_id}.pkl"),
                        {"rollout_id": rollout_id},
                    )
        train, test, counts = make_seen_split(
            rollouts,
            identity,
            train_per_class=7,
            split_seed=13,
        )
        again = make_seen_split(rollouts, identity, train_per_class=7, split_seed=13)
        self.assertEqual(len(train), 140)
        self.assertEqual(len(test), 60)
        self.assertEqual(
            [identity[id(rollout)][1]["rollout_id"] for rollout in train],
            [identity[id(rollout)][1]["rollout_id"] for rollout in again[0]],
        )
        for task_id in range(10):
            self.assertEqual(counts[task_id]["success"], {"train": 7, "test": 3})
            self.assertEqual(counts[task_id]["failure"], {"train": 7, "test": 3})

    def test_frozen_hyperparameters_match_completed_grid_selection(self):
        self.assertEqual(
            MODEL_DEFAULTS["indep"],
            {
                "horizon_selector": 1.0,
                "diffusion_selector": 0.0,
                "learning_rate": 3e-4,
                "lambda_reg": 1e-3,
            },
        )
        self.assertEqual(MODEL_DEFAULTS["lstm"]["diffusion_selector"], "concat-2")
        self.assertEqual(MODEL_DEFAULTS["lstm"]["learning_rate"], 1e-3)

    def test_task_cutoff_uses_training_lengths_only(self):
        train = [
            SimpleNamespace(task_id=0, hidden_states=np.zeros((8, 1))),
            SimpleNamespace(task_id=0, hidden_states=np.zeros((10, 1))),
        ]
        test = [SimpleNamespace(task_id=0, hidden_states=np.zeros((3, 1)))]
        cutoffs = set_task_min_step_from_training(train, test)
        self.assertEqual(cutoffs, {0: 8})
        self.assertEqual([item.task_min_step for item in train], [8, 8])
        self.assertEqual(test[0].task_min_step, 3)

    def test_three_fold_cv_is_stratified_and_never_contains_outer_test(self):
        rollouts = []
        identity = {}
        for task_id in range(5):
            for success in (0, 1):
                for index in range(10):
                    rollout = SimpleNamespace(task_id=task_id, episode_success=success)
                    rollout_id = f"task-{task_id}-success-{success}-{index}"
                    rollouts.append(rollout)
                    identity[id(rollout)] = (
                        Path(f"{rollout_id}.pkl"),
                        {"rollout_id": rollout_id},
                    )
        outer_train, outer_test, _ = make_seen_split(
            rollouts, identity, train_per_class=7, split_seed=0
        )
        folds = make_inner_folds(outer_train, identity, num_folds=3, seed=0)
        outer_test_ids = {identity[id(item)][1]["rollout_id"] for item in outer_test}
        validation_union = set()
        self.assertEqual([len(validation) for _, validation in folds], [30, 20, 20])
        for training, validation in folds:
            ids = {
                identity[id(item)][1]["rollout_id"] for item in training + validation
            }
            self.assertFalse(ids & outer_test_ids)
            validation_union.update(
                identity[id(item)][1]["rollout_id"] for item in validation
            )
        self.assertEqual(len(validation_union), 70)

    def test_task_type_filter_reuses_fixed_outer_split_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_ids = {
                "AtomicA": 0,
                "CompositeA": 1,
                "AtomicB": 2,
                "CompositeB": 3,
            }
            task_types = {
                "AtomicA": "atomic",
                "CompositeA": "composite",
                "AtomicB": "atomic",
                "CompositeB": "composite",
            }
            (root / "conversion_report.json").write_text(
                json.dumps({"task_ids": task_ids, "task_types": task_types})
            )
            rollouts = []
            env_records = []
            train_ids = []
            test_ids = []
            for task_name, task_id in task_ids.items():
                for success in (0, 1):
                    for index in range(2):
                        rollout_id = f"{task_name}-success-{success}-index-{index}"
                        rollouts.append(
                            SimpleNamespace(
                                task_id=task_id,
                                episode_success=success,
                            )
                        )
                        env_records.append(
                            (
                                root / f"{rollout_id}.pkl",
                                {
                                    "rollout_id": rollout_id,
                                    "task_id": task_id,
                                    "episode_success": success,
                                },
                            )
                        )
                        (train_ids if index == 0 else test_ids).append(rollout_id)
            split_path = root / "split_manifest.json"
            split_path.write_text(
                json.dumps(
                    {
                        "split_seed": 0,
                        "train": train_ids,
                        "test": test_ids,
                    }
                )
            )
            selection = resolve_task_type_selection(root, "atomic")
            selected, selected_env, identity = filter_aligned_task_type(
                rollouts,
                env_records,
                selection,
            )
            train, test, per_task = make_seen_split(
                selected,
                identity,
                train_per_class=1,
                split_seed=0,
                fixed_split_ids=load_outer_split_ids(split_path),
            )

            self.assertEqual(selection["selected_task_names"], ["AtomicA", "AtomicB"])
            self.assertEqual(len(selected_env), 8)
            self.assertEqual(
                {identity[id(item)][1]["rollout_id"] for item in train},
                set(train_ids) & {env["rollout_id"] for _, env in selected_env},
            )
            self.assertEqual(
                {identity[id(item)][1]["rollout_id"] for item in test},
                set(test_ids) & {env["rollout_id"] for _, env in selected_env},
            )
            self.assertEqual(set(per_task), {0, 2})

    def test_cv_grid_has_405_fits_per_architecture(self):
        runs = generate_cv_runs("lstm")
        self.assertEqual(len(runs), 405)
        self.assertEqual(len({run.slug for run in runs}), 405)

    def test_summary_aggregates_three_fixed_split_model_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in ("indep", "lstm"):
                for seed, value in ((0, 0.7), (1, 0.8), (2, 0.9)):
                    run = root / f"{model}-seed-{seed}"
                    run.mkdir()
                    metrics = {
                        "model": model,
                        "seed": seed,
                        "task_type_filter": "atomic",
                        "task_types": {f"Task{index}": "atomic" for index in range(10)},
                        "num_tasks": 10,
                        "counts": {
                            "train": 140,
                            "test": 60,
                            "train_successes": 70,
                            "train_failures": 70,
                            "test_successes": 30,
                            "test_failures": 30,
                        },
                        "scalar_metrics": {
                            "falert_early_roc_auc/model_test": value,
                            "falert_early_prc_auc/model_test": value - 0.1,
                        },
                    }
                    (run / "metrics.json").write_text(json.dumps(metrics))
            result = summarize(root)
            self.assertEqual(result["num_completed_runs"], 6)
            self.assertEqual(result["num_tasks"], 10)
            self.assertEqual(result["task_type_filter"], "atomic")
            self.assertEqual(result["split_counts"]["test"], 60)
            self.assertAlmostEqual(result["models"]["indep"]["test_roc_auc_mean"], 0.8)
            self.assertTrue((root / "summary.json").is_file())

    def test_cv_selection_and_final_refit_selector_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in ("indep", "lstm"):
                for config, score in (("low", 0.6), ("high", 0.8)):
                    for fold in (0, 1, 2):
                        run = root / f"{model}-{config}-{fold}"
                        run.mkdir()
                        record = {
                            "status": "complete",
                            "model": model,
                            "task_type_filter": "atomic",
                            "selected_task_names": [
                                f"Task{index}" for index in range(5)
                            ],
                            "horizon_selector": "1.0",
                            "diffusion_selector": "concat-2"
                            if config == "high"
                            else "0.0",
                            "learning_rate": 1e-3 if config == "high" else 1e-4,
                            "lambda_reg": 1e-2,
                            "fold": fold,
                            "selection_value": score + 0.01 * fold,
                            "counts": {
                                "outer_train": {
                                    "rollouts": 140,
                                    "successes": 70,
                                    "failures": 70,
                                },
                                "outer_test_untouched": {
                                    "rollouts": 60,
                                    "successes": 30,
                                    "failures": 30,
                                },
                            },
                        }
                        (run / "metrics.json").write_text(json.dumps(record))
            summary = summarize_seen_cv(root)
            self.assertEqual(summary["num_completed_fits"], 12)
            self.assertEqual(summary["task_type_filter"], "atomic")
            self.assertEqual(summary["outer_train_counts"]["rollouts"], 140)
            self.assertEqual(summary["outer_test_counts"]["rollouts"], 60)
            self.assertEqual(
                summary["best_by_model"]["lstm"]["diffusion_selector"], "concat-2"
            )
            resolved = resolve_hyperparameters(
                "lstm", root / "cv_selection_summary.json"
            )
            self.assertEqual(resolved["horizon_selector"], 1.0)
            self.assertEqual(resolved["diffusion_selector"], "concat-2")

            summary = json.loads((root / "cv_selection_summary.json").read_text())
            summary["task_type_filter"] = "atomic"
            (root / "cv_selection_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_hyperparameters(
                    "lstm",
                    root / "cv_selection_summary.json",
                    expected_task_type="composite",
                )


class TestScoreVideo(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "OpenCV is not installed locally")
    def test_score_overlay_video_is_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for index in range(8):
                frame = np.full((120, 160, 3), 30 + index * 10, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            score_file = root / "scores.jsonl"
            record = {
                "rollout_id": "abc",
                "split": "test",
                "task_name": "OpenDrawer",
                "failed": True,
                "model": "lstm",
                "seed": 0,
                "scores": [0.1, 0.3, 0.8],
                "video_path": str(source),
                "video_frame_stride": 1,
                "inference_environment_steps": [0, 3, 6],
            }
            score_file.write_text(json.dumps(record) + "\n")
            calibration_file = root / "calibration.json"
            calibration_file.write_text(
                json.dumps(
                    {
                        "alpha": 0.15,
                        "alignment": "extend",
                        "threshold": [0.4, 0.4, 0.4],
                    }
                )
                + "\n"
            )
            outputs = render_score_videos(
                score_file,
                root / "rendered",
                calibration_path=calibration_file,
            )
            self.assertEqual(len(outputs), 1)
            self.assertGreater(outputs[0].stat().st_size, 0)
            capture = cv2.VideoCapture(str(outputs[0]))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 8)
            capture.release()


if __name__ == "__main__":
    unittest.main()
