import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.causal_subtask_safe import (  # noqa: E402
    CausalPrefixConfig,
    apply_causal_target,
    clone_prefix_rollout,
    prefix_lengths_for_rollout,
    prepare_causal_splits,
    select_supported_stages,
    transform_temporal_representation,
    validate_causal_config,
)
from robocasa.recovery.safe.allocate_causal_subtask_data import (  # noqa: E402
    build_target_aware_allocation,
)
from robocasa.recovery.safe.build_augmented_causal_split import (  # noqa: E402
    build_augmented_causal_split,
)
from robocasa.recovery.safe.evaluate_causal_subtask_gates import (  # noqa: E402
    evaluate_root,
)
from robocasa.recovery.safe.plan_subtask_safe_collection import (  # noqa: E402
    build_collection_plan,
)
from robocasa.recovery.safe.run_seen_cv_grid import causal_prefix_roc_auc  # noqa: E402


class FakeRollout:
    def __init__(self, task_id, success, length=20, dimensions=4):
        self.task_id = task_id
        self.episode_success = success
        self.hidden_states = np.arange(length * dimensions, dtype=np.float32).reshape(
            length, dimensions
        )
        self.action_vectors = np.arange(length * 192, dtype=np.float32).reshape(
            length, 192
        )
        self.task_min_step = length


def fake_item(index, success, split, stage="Task::stage", parent=None):
    rollout = FakeRollout(0, success)
    parent = parent or f"{split}-parent-{index}"
    env = {
        "rollout_id": f"{split}-segment-{index}",
        "task_id": 0,
        "task_name": stage,
        "parent_task_name": "Task",
        "subtask_id": stage.split("::", 1)[1],
        "parent_rollout_id": parent,
        "episode_success": success,
        "inference_environment_steps": list(range(0, 160, 8)),
        "subtask_safe_segment": {
            "entry_environment_step": 0,
            "end_environment_step": 160,
        },
    }
    return rollout, env


class TestCausalSubtaskSafe(unittest.TestCase):
    @staticmethod
    def _augmented_split_fixture():
        original_train = {"old-train-a", "old-train-b"}
        original_test = {"old-test"}
        unassigned = {"new-train-a", "new-train-b", "new-train-c"}
        calibration = {"new-cal"}
        evaluation = {"new-eval-a", "new-eval-b"}
        parents = (
            original_train
            | original_test
            | unassigned
            | calibration
            | evaluation
            | {"new-ineligible"}
        )
        env_records = []
        for index, parent in enumerate(sorted(parents)):
            task_id = index % 2
            task_name = "TaskA::stage" if task_id == 0 else "TaskB::stage"
            env_records.append(
                (
                    Path(f"/{parent}.pkl"),
                    {
                        "rollout_id": f"segment-{parent}",
                        "parent_rollout_id": parent,
                        "task_id": task_id,
                        "task_name": task_name,
                        "episode_success": index % 2,
                    },
                )
            )
        original = {
            "split_unit": "parent_rollout",
            "split_seed": 0,
            "parent_train": sorted(original_train),
            "parent_test": sorted(original_test),
        }
        allocation = {
            "protocol": "causal_subtask_finite_horizon_parent_allocation",
            "complete": True,
            "target_definition": {
                "label": "failure_within_H",
                "causal_prefix_inferences": 32,
                "failure_horizon_inferences": 128,
            },
            "selected_stages": ["TaskA::stage", "TaskB::stage"],
            "unassigned_parent_ids": sorted(unassigned),
            "calibration_parent_ids": sorted(calibration),
            "evaluation_parent_ids": sorted(evaluation),
        }
        return env_records, original, allocation

    def test_augmented_split_freezes_holdout_and_excludes_old_test(self):
        env_records, original, allocation = self._augmented_split_fixture()
        manifest = build_augmented_causal_split(
            env_records,
            original_split=original,
            target_allocation=allocation,
            expected_original_train_parents=2,
            expected_original_test_parents=1,
            expected_unassigned_parents=3,
            expected_calibration_parents=1,
            expected_evaluation_parents=2,
        )

        self.assertEqual(manifest["counts"]["train_parents"], 5)
        self.assertEqual(manifest["counts"]["test_parents"], 3)
        self.assertEqual(manifest["counts"]["unused_original_test_parents"], 1)
        self.assertEqual(
            set(manifest["frozen_calibration_parent_ids"]), {"new-cal"}
        )
        self.assertEqual(
            set(manifest["frozen_evaluation_parent_ids"]),
            {"new-eval-a", "new-eval-b"},
        )
        self.assertIn("old-test", manifest["parent_unused"])
        self.assertNotIn("old-test", manifest["parent_train"])
        self.assertNotIn("old-test", manifest["parent_test"])
        self.assertFalse(set(manifest["train"]) & set(manifest["test"]))
        self.assertTrue(manifest["audit"]["training_holdout_parent_disjoint"])
        self.assertTrue(manifest["audit"]["original_old_test_excluded"])
        self.assertFalse(manifest["audit"]["outer_test_used_for_selection"])
        self.assertEqual(len(manifest["manifest_fingerprint"]), 64)

    def test_augmented_split_rejects_original_and_new_parent_overlap(self):
        env_records, original, allocation = self._augmented_split_fixture()
        allocation["unassigned_parent_ids"].append("old-train-a")
        with self.assertRaisesRegex(ValueError, "Original and new-seed"):
            build_augmented_causal_split(
                env_records,
                original_split=original,
                target_allocation=allocation,
            )

    def test_augmented_split_rejects_missing_frozen_parent(self):
        env_records, original, allocation = self._augmented_split_fixture()
        env_records = [
            item
            for item in env_records
            if item[1]["parent_rollout_id"] != "new-eval-b"
        ]
        with self.assertRaisesRegex(ValueError, "Required parents are absent"):
            build_augmented_causal_split(
                env_records,
                original_split=original,
                target_allocation=allocation,
            )

    def test_within_horizon_requires_prefixes_and_positive_horizon(self):
        with self.assertRaisesRegex(ValueError, "causal prefix training"):
            validate_causal_config(
                CausalPrefixConfig(
                    training_mode="none",
                    label_mode="within_horizon",
                    failure_horizon=8,
                )
            )
        with self.assertRaisesRegex(ValueError, "positive failure_horizon"):
            validate_causal_config(
                CausalPrefixConfig(
                    training_mode="fixed",
                    label_mode="within_horizon",
                )
            )
        with self.assertRaisesRegex(ValueError, "only meaningful"):
            validate_causal_config(
                CausalPrefixConfig(
                    training_mode="fixed",
                    label_mode="eventual",
                    failure_horizon=8,
                )
            )

    def test_prefix_clone_truncates_all_official_time_aligned_tensors(self):
        rollout, env = fake_item(0, 1, "train")
        rollout.hidden_states = np.zeros((43, 4), dtype=np.float32)
        rollout.action_vectors = np.zeros((43, 192), dtype=np.float32)
        clone, _ = clone_prefix_rollout(rollout, env, 16)
        self.assertEqual(clone.hidden_states.shape, (16, 4))
        self.assertEqual(clone.action_vectors.shape, (16, 192))

    def test_prefix_expansion_and_conditioning_preserve_parent_split(self):
        train_pairs = [fake_item(i, i % 2, "train") for i in range(8)]
        test_pairs = [fake_item(i, i % 2, "test") for i in range(6)]
        train = [item[0] for item in train_pairs]
        test = [item[0] for item in test_pairs]
        identity = {
            id(rollout): (Path(f"/{env['rollout_id']}.pkl"), env)
            for rollout, env in train_pairs + test_pairs
        }
        payload = prepare_causal_splits(
            train,
            test,
            identity,
            config=CausalPrefixConfig(
                training_mode="fixed",
                horizons=(1, 2, 4, 8),
                conditioning="subtask_one_hot_elapsed",
                min_stage_successes=2,
                min_stage_failures=2,
            ),
        )
        self.assertEqual(len(payload["train"]), len(train) * 4)
        self.assertEqual(len(payload["test"]), len(test) * 4)
        self.assertEqual(payload["conditioning"]["added_dimensions"], 2)
        self.assertEqual(payload["train"][0].hidden_states.shape[-1], 6)
        train_parents = {
            payload["identity"][id(item)][1]["parent_rollout_id"]
            for item in payload["train"]
        }
        test_parents = {
            payload["identity"][id(item)][1]["parent_rollout_id"]
            for item in payload["test"]
        }
        self.assertFalse(train_parents & test_parents)
        self.assertTrue(
            all(
                len(item.hidden_states)
                == payload["identity"][id(item)][1]["causal_prefix_inferences"]
                for item in payload["train"] + payload["test"]
            )
        )
        self.assertTrue(
            all(
                len(item.action_vectors) == len(item.hidden_states)
                for item in payload["train"] + payload["test"]
            )
        )

    def test_random_prefixes_are_deterministic(self):
        first = prefix_lengths_for_rollout(
            20,
            mode="random",
            horizons=(1, 2, 4, 8, 16),
            random_prefixes_per_segment=4,
            random_seed=7,
            identity_key="segment",
        )
        second = prefix_lengths_for_rollout(
            20,
            mode="random",
            horizons=(1, 2, 4, 8, 16),
            random_prefixes_per_segment=4,
            random_seed=7,
            identity_key="segment",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(1 <= value <= 16 for value in first))

    def test_within_horizon_target_is_negative_until_failure_is_near(self):
        rollout, env = fake_item(0, 0, "train")
        early, early_env = clone_prefix_rollout(rollout, env, 4)
        late, late_env = clone_prefix_rollout(rollout, env, 16)
        self.assertFalse(
            apply_causal_target(
                early, early_env, label_mode="within_horizon", failure_horizon=8
            )
        )
        self.assertTrue(
            apply_causal_target(
                late, late_env, label_mode="within_horizon", failure_horizon=8
            )
        )
        self.assertEqual((early.episode_success, late.episode_success), (1, 0))
        self.assertEqual(early_env["remaining_inferences_to_terminal"], 16)
        self.assertEqual(late_env["remaining_inferences_to_terminal"], 4)

    def test_temporal_representations_are_causal_and_length_preserving(self):
        rollout = FakeRollout(0, 1, length=5, dimensions=2)
        original = rollout.hidden_states.copy()
        metadata = transform_temporal_representation(
            [rollout], mode="raw_delta_mean_slope", window=3
        )
        self.assertEqual(rollout.hidden_states.shape, (5, 8))
        self.assertEqual(metadata["source_dimension"], 2)
        np.testing.assert_array_equal(rollout.hidden_states[:, :2], original)
        np.testing.assert_array_equal(rollout.hidden_states[0, 2:4], 0)
        np.testing.assert_array_equal(
            rollout.hidden_states[2, 2:4], original[2] - original[1]
        )

    def test_stage_support_uses_training_only(self):
        train_pairs = [fake_item(i, int(i == 0), "train") for i in range(3)]
        identity = {
            id(rollout): (Path("/tmp/item"), env) for rollout, env in train_pairs
        }
        with self.assertRaisesRegex(ValueError, "No semantic stage"):
            select_supported_stages(
                [item[0] for item in train_pairs],
                identity,
                min_successes=2,
                min_failures=2,
            )

    def test_cv_selection_scores_only_preregistered_prefix(self):
        pairs = [fake_item(i, i % 2, "validation") for i in range(4)]
        rollouts = []
        scores = []
        identity = {}
        for rollout, env in pairs:
            for prefix in (4, 8):
                clone = FakeRollout(0, rollout.episode_success, length=prefix)
                prefix_env = {
                    **env,
                    "rollout_id": f"{env['rollout_id']}::prefix-{prefix}",
                    "causal_prefix_inferences": prefix,
                }
                rollouts.append(clone)
                identity[id(clone)] = (Path("/tmp/item"), prefix_env)
                failed = not bool(clone.episode_success)
                # Prefix 4 is reversed, but the preregistered prefix 8 is perfect.
                value = (
                    (0.1 if failed else 0.9)
                    if prefix == 4
                    else (0.9 if failed else 0.1)
                )
                scores.append(np.repeat(value, prefix))
        self.assertEqual(causal_prefix_roc_auc(rollouts, scores, identity, 8), 1.0)

    def test_collection_plan_prioritizes_failure_deficits(self):
        records = []
        for index, success in enumerate((1, 1, 1, 0)):
            _, env = fake_item(index, success, "train")
            records.append((Path(f"/{index}.pkl"), env))
        plan = build_collection_plan(
            records,
            min_train_successes=2,
            min_train_failures=1,
            target_train_successes=3,
            target_train_failures=3,
            target_test_successes=0,
            target_test_failures=0,
        )
        row = plan["rows"][0]
        self.assertTrue(row["selected_for_v2"])
        self.assertEqual(row["train_failure_deficit"], 2)
        self.assertEqual(row["priority"], "collect_failures")

    def test_target_aware_allocation_is_parent_disjoint_and_uses_horizon_label(self):
        records = []
        for index in range(4):
            _, env = fake_item(index, 1, "test")
            env["model_infer_times"] = 200
            env["parent_rollout_failed"] = False
            records.append((Path(f"/success-{index}.pkl"), env))
        for index, length in enumerate((100, 200)):
            _, env = fake_item(index + 10, 0, "test")
            env["model_infer_times"] = length
            env["parent_rollout_failed"] = True
            records.append((Path(f"/failure-{index}.pkl"), env))

        allocation = build_target_aware_allocation(
            records,
            stages=["Task::stage"],
            prefix=32,
            failure_horizon=128,
            calibration_successes_per_stage=2,
            evaluation_successes_per_stage=1,
            evaluation_failures_per_stage=1,
            seed=7,
        )
        row = allocation["per_stage"][0]
        self.assertTrue(allocation["complete"])
        self.assertFalse(
            set(allocation["calibration_parent_ids"])
            & set(allocation["evaluation_parent_ids"])
        )
        self.assertEqual(row["eligible_failures"], 1)
        self.assertEqual(row["eligible_successes"], 5)
        self.assertEqual(row["calibration_successes"], 2)
        self.assertEqual(row["evaluation_successes"], 1)
        self.assertEqual(row["evaluation_failures"], 1)
        self.assertFalse(allocation["audit"]["calibration_contains_failed_parents"])

    def test_target_aware_allocation_reports_finite_horizon_deficits(self):
        records = []
        for index, success in enumerate((1, 1, 1, 0)):
            _, env = fake_item(index, success, "test")
            env["model_infer_times"] = 100 if not success else 200
            env["parent_rollout_failed"] = not bool(success)
            records.append((Path(f"/{index}.pkl"), env))
        allocation = build_target_aware_allocation(
            records,
            stages=["Task::stage"],
            prefix=32,
            failure_horizon=128,
            calibration_successes_per_stage=1,
            evaluation_successes_per_stage=1,
            evaluation_failures_per_stage=2,
        )
        row = allocation["per_stage"][0]
        self.assertFalse(allocation["complete"])
        self.assertEqual(row["evaluation_failure_deficit"], 1)
        self.assertEqual(row["collection_priority"], "collect_target_failures")

    def test_infeasible_stage_does_not_block_other_stage_calibration(self):
        records = []
        _, sparse = fake_item(0, 0, "test", stage="Task::sparse")
        sparse["model_infer_times"] = 100
        sparse["parent_rollout_failed"] = True
        records.append((Path("/sparse.pkl"), sparse))
        for index in range(3):
            _, ready = fake_item(index + 1, 1, "test", stage="Task::ready")
            ready["model_infer_times"] = 200
            ready["parent_rollout_failed"] = False
            records.append((Path(f"/ready-{index}.pkl"), ready))

        allocation = build_target_aware_allocation(
            records,
            stages=["Task::sparse", "Task::ready"],
            prefix=32,
            failure_horizon=128,
            calibration_successes_per_stage=1,
            evaluation_successes_per_stage=1,
            evaluation_failures_per_stage=0,
        )
        rows = {row["stage"]: row for row in allocation["per_stage"]}

        self.assertFalse(allocation["complete"])
        self.assertEqual(rows["Task::sparse"]["calibration_successes"], 0)
        self.assertEqual(rows["Task::ready"]["calibration_successes"], 1)
        self.assertEqual(rows["Task::ready"]["evaluation_successes"], 1)

    def _write_causal_scores(self, final_root):
        for seed in (0, 1, 2):
            records = []
            for split, count in (("train", 16), ("test", 20)):
                for index in range(count):
                    failed = index % 2 == 1
                    parent = f"{split}-parent-{index}"
                    for prefix in (1, 2, 4, 8, 16):
                        # Strong representation signal. Elapsed stage risk is
                        # constant because this fixture contains one stage.
                        success_value = 0.10 if split == "train" else 0.05
                        value = (0.90 if failed else success_value) + seed * 0.001
                        records.append(
                            {
                                "rollout_id": f"{split}-seg-{index}::prefix-{prefix:04d}",
                                "source_segment_id": f"{split}-seg-{index}",
                                "causal_prefix_inferences": prefix,
                                "causal_failure_horizon_inferences": 128,
                                "parent_rollout_id": parent,
                                "parent_task_name": "Task",
                                "parent_rollout_failed": failed,
                                "subtask_safe_segment": {
                                    "entry_environment_step": 0,
                                    "end_environment_step": 160,
                                },
                                "inference_environment_steps": [
                                    step * 8 for step in range(prefix)
                                ],
                                "split": split,
                                "task_name": "Task::stage",
                                "task_type": "composite",
                                "failed": failed,
                                "model": "indep",
                                "seed": seed,
                                "scores": [value] * prefix,
                                "task_min_step": prefix,
                                "num_inferences": prefix,
                            }
                        )
            run = final_root / f"indep_seed{seed}"
            run.mkdir(parents=True)
            (run / "scores.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )

    def test_causal_gate_evaluator_uses_paired_parent_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_root = Path(tmp) / "final"
            self._write_causal_scores(final_root)
            summary = evaluate_root(
                final_root,
                models=("indep",),
                bootstrap_replicates=50,
            )
            result = summary["models"]["indep"]
            self.assertTrue(result["continue_to_recovery_integration"])
            self.assertGreater(result["observed"]["causal_prefix_roc_auc"], 0.99)
            self.assertAlmostEqual(result["observed"]["stage_prior_roc_auc"], 0.5)
            self.assertAlmostEqual(result["observed"]["duration_progress_roc_auc"], 0.5)
            self.assertGreater(
                result["bootstrap"]["safe_minus_elapsed_roc_auc"]["ci_95_low"],
                0,
            )
            self.assertFalse(
                set(summary["calibration"]["calibration_parent_ids"])
                & set(summary["calibration"]["evaluation_parent_ids"])
            )
            self.assertTrue(
                all(
                    row["threshold_source"]
                    == "successful parent-disjoint held-out calibration prefixes"
                    for row in summary["operating_points"]
                )
            )

    def test_causal_gate_evaluator_accepts_preregistered_parent_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_root = Path(tmp) / "final"
            self._write_causal_scores(final_root)
            manifest = {
                "schema_version": 1,
                "protocol": "causal_subtask_finite_horizon_parent_allocation",
                "complete": True,
                "target_definition": {
                    "causal_prefix_inferences": 8,
                    "failure_horizon_inferences": 128,
                },
                "selected_stages": ["Task::stage"],
                "quotas_per_stage": {
                    "calibration_successes": 2,
                    "evaluation_successes": 8,
                    "evaluation_failures": 10,
                },
                "calibration_parent_ids": ["test-parent-0", "test-parent-2"],
                "evaluation_parent_ids": [
                    f"test-parent-{index}" for index in range(4, 20)
                ]
                + ["test-parent-1", "test-parent-3"],
            }
            manifest_path = Path(tmp) / "allocation.json"
            manifest_path.write_text(json.dumps(manifest))
            summary = evaluate_root(
                final_root,
                models=("indep",),
                target_prefix=8,
                prefixes=(1, 2, 4, 8, 16),
                allocation_manifest=manifest_path,
                bootstrap_replicates=50,
            )
            self.assertEqual(
                summary["calibration"]["allocation_manifest"],
                str(manifest_path.resolve()),
            )
            self.assertEqual(
                summary["calibration"]["calibration_parent_ids"],
                ["test-parent-0", "test-parent-2"],
            )


if __name__ == "__main__":
    unittest.main()
