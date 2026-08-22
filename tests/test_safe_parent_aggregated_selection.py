from pathlib import Path
from types import SimpleNamespace
import unittest

from robocasa.recovery.safe.run_seen_cv_grid import make_inner_folds
from robocasa.recovery.safe.subtask_safe_evaluation import (
    fixed_prefix_subtask_metrics,
    freeze_common_subtask_stage_catalog,
    parent_aggregated_early_metrics,
    parent_group_id,
    semantic_subtask_score_records,
    semantic_subtask_scores_from_saved_records,
    subtask_fixed_prefix_selection,
)


def rollout_identity(record, *, task_id=0, success=1):
    rollout = SimpleNamespace(task_id=task_id, episode_success=success)
    return rollout, (Path(f"{record['rollout_id']}.pkl"), record)


class TestParentAggregatedSelection(unittest.TestCase):
    def test_terminal_scores_are_evaluated_by_parent_and_task_macro(self):
        rollouts = []
        scores = []
        identity = {}
        for task_id, task in enumerate(("TaskA", "TaskB")):
            for failed in (False, True):
                rollout_id = f"{task}-{int(failed)}"
                record = {
                    "rollout_id": rollout_id,
                    "task_name": task,
                    "episode_success": int(not failed),
                    "robocasa_manifest_record": {"valid_sequence_length": 4},
                }
                rollout, value = rollout_identity(
                    record, task_id=task_id, success=int(not failed)
                )
                rollouts.append(rollout)
                identity[id(rollout)] = value
                scores.append(
                    [0.8, 0.9, 0.95, 0.99] if failed else [0.05, 0.1, 0.15, 0.2]
                )

        result = parent_aggregated_early_metrics(
            rollouts, scores, identity, landmarks=(0.25, 0.5)
        )

        self.assertEqual(result["parents"], 4)
        self.assertEqual(result["selection_value"], 1.0)
        self.assertEqual(result["per_landmark"]["0.25"]["task_macro_roc_auc"], 1.0)
        self.assertEqual(result["per_landmark"]["0.5"]["parents_without_score"], 0)

    def test_segment_scores_are_stitched_at_source_inference_indices(self):
        rollouts = []
        scores = []
        identity = {}
        for task_id, task in enumerate(("TaskA", "TaskB")):
            for failed in (False, True):
                parent = f"{task}-{int(failed)}"
                for segment_index, (start, end) in enumerate(((0, 2), (2, 4))):
                    rollout_id = f"{parent}-segment-{segment_index}"
                    record = {
                        "rollout_id": rollout_id,
                        "parent_rollout_id": parent,
                        "parent_task_name": task,
                        "parent_rollout_failed": failed,
                        # Stage labels may be one-sided and need not equal the
                        # parent outcome used for parent-level selection.
                        "episode_success": int(segment_index == 0 or not failed),
                        "robocasa_manifest_record": {"valid_sequence_length": 4},
                        "subtask_safe_segment": {
                            "inference_start_index": start,
                            "inference_end_index_exclusive": end,
                        },
                    }
                    rollout, value = rollout_identity(
                        record,
                        task_id=segment_index,
                        success=record["episode_success"],
                    )
                    rollouts.append(rollout)
                    identity[id(rollout)] = value
                    scores.append([0.8, 0.9] if failed else [0.05, 0.1])

        result = parent_aggregated_early_metrics(
            rollouts, scores, identity, landmarks=(0.25, 0.5)
        )

        self.assertEqual(result["parents"], 4)
        self.assertEqual(result["selection_value"], 1.0)
        self.assertTrue(
            all(len(row["segment_ids"]) == 2 for row in result["parent_rows"])
        )

    def test_missing_early_segment_is_counted_as_no_alarm(self):
        rollouts = []
        scores = []
        identity = {}
        for task_id, task in enumerate(("TaskA", "TaskB")):
            for failed in (False, True):
                parent = f"{task}-{int(failed)}"
                record = {
                    "rollout_id": f"{parent}-late-segment",
                    "parent_rollout_id": parent,
                    "parent_task_name": task,
                    "parent_rollout_failed": failed,
                    "episode_success": int(not failed),
                    "robocasa_manifest_record": {"valid_sequence_length": 4},
                    "subtask_safe_segment": {
                        "inference_start_index": 2,
                        "inference_end_index_exclusive": 4,
                    },
                }
                rollout, value = rollout_identity(
                    record, task_id=task_id, success=int(not failed)
                )
                rollouts.append(rollout)
                identity[id(rollout)] = value
                scores.append([0.8, 0.9] if failed else [0.05, 0.1])

        result = parent_aggregated_early_metrics(
            rollouts, scores, identity, landmarks=(0.25, 1.0)
        )

        self.assertEqual(result["per_landmark"]["0.25"]["parents_without_score"], 4)
        self.assertEqual(result["per_landmark"]["0.25"]["task_macro_roc_auc"], 0.5)
        self.assertEqual(result["per_landmark"]["1.0"]["task_macro_roc_auc"], 1.0)

    def test_terminal_and_segmented_data_receive_identical_parent_folds(self):
        terminal = []
        terminal_identity = {}
        segmented = []
        segmented_identity = {}
        for task_id, task in enumerate(("TaskA", "TaskB")):
            for failed in (False, True):
                for parent_index in range(3):
                    parent = f"{task}-{int(failed)}-{parent_index}"
                    terminal_record = {
                        "rollout_id": parent,
                        "task_name": task,
                        "episode_success": int(not failed),
                    }
                    rollout, value = rollout_identity(
                        terminal_record,
                        task_id=task_id,
                        success=int(not failed),
                    )
                    terminal.append(rollout)
                    terminal_identity[id(rollout)] = value

                    for segment_index in range(2):
                        segment_record = {
                            "rollout_id": f"{parent}-segment-{segment_index}",
                            "parent_rollout_id": parent,
                            "parent_task_name": task,
                            "parent_rollout_failed": failed,
                            "episode_success": int(segment_index == 0 or not failed),
                        }
                        segment, segment_value = rollout_identity(
                            segment_record,
                            task_id=segment_index,
                            success=segment_record["episode_success"],
                        )
                        segmented.append(segment)
                        segmented_identity[id(segment)] = segment_value

        terminal_folds = make_inner_folds(
            terminal,
            terminal_identity,
            num_folds=3,
            seed=7,
            group_field="parent_or_rollout_id",
        )
        segmented_folds = make_inner_folds(
            segmented,
            segmented_identity,
            num_folds=3,
            seed=7,
            group_field="parent_or_rollout_id",
        )

        for terminal_fold, segmented_fold in zip(terminal_folds, segmented_folds):
            terminal_validation = {
                parent_group_id(terminal_identity[id(item)][1])
                for item in terminal_fold[1]
            }
            segmented_validation = {
                parent_group_id(segmented_identity[id(item)][1])
                for item in segmented_fold[1]
            }
            self.assertEqual(terminal_validation, segmented_validation)


class TestSubtaskLabelFixedPrefixSelection(unittest.TestCase):
    @staticmethod
    def catalog_record(parent, task, stage, failed, start, end):
        return {
            "rollout_id": f"{parent}:{stage}",
            "parent_rollout_id": parent,
            "parent_task_name": task,
            "parent_rollout_failed": parent.endswith("F"),
            "task_name": f"{task}::{stage}",
            "subtask_id": stage,
            "episode_success": int(not failed),
            "model_infer_times": end - start,
            "inference_environment_steps": list(range(start, end)),
            "subtask_safe_segment": {
                "inference_start_index": start,
                "inference_end_index_exclusive": end,
                "num_policy_inferences": end - start,
                "entry_environment_step": start,
                "end_environment_step": end,
            },
        }

    def test_terminal_scores_are_sliced_and_use_subtask_not_parent_labels(self):
        rollouts = []
        scores = []
        identity = {}
        catalog = []
        for parent, parent_failed in (("parent-S", False), ("parent-F", True)):
            record = {
                "rollout_id": parent,
                "task_name": "Composite",
                "episode_success": int(not parent_failed),
            }
            rollout, value = rollout_identity(record, success=int(not parent_failed))
            rollouts.append(rollout)
            identity[id(rollout)] = value
            scores.append(
                [0.05, 0.10, 0.80, 0.90] if parent_failed else [0.04, 0.08, 0.15, 0.20]
            )
            # The first subtask succeeds even in the failed parent. Only the
            # second subtask inherits the failed active-stage label.
            catalog.extend(
                (
                    self.catalog_record(parent, "Composite", "StageA", False, 0, 2),
                    self.catalog_record(
                        parent,
                        "Composite",
                        "StageB",
                        parent_failed,
                        2,
                        4,
                    ),
                )
            )

        rows = semantic_subtask_score_records(rollouts, scores, identity, catalog)

        self.assertEqual(len(rows), 4)
        self.assertFalse(
            next(
                row
                for row in rows
                if row["parent_rollout_id"] == "parent-F"
                and row["subtask_id"] == "StageA"
            )["failed"]
        )
        stage_b_failure = next(
            row
            for row in rows
            if row["parent_rollout_id"] == "parent-F" and row["subtask_id"] == "StageB"
        )
        self.assertEqual(stage_b_failure["scores"].tolist(), [0.8, 0.9])
        self.assertEqual(stage_b_failure["score_source"], "terminal_sliced")

    def test_segmented_scores_align_by_segment_id(self):
        catalog = [
            self.catalog_record("parent-S", "Composite", "Stage", False, 0, 2),
            self.catalog_record("parent-F", "Composite", "Stage", True, 0, 2),
        ]
        rollouts = []
        scores = []
        identity = {}
        for record, score in zip(catalog, ([0.1, 0.2], [0.7, 0.8])):
            rollout, value = rollout_identity(record, success=record["episode_success"])
            rollouts.append(rollout)
            scores.append(score)
            identity[id(rollout)] = value

        rows = semantic_subtask_score_records(
            rollouts, scores, identity, reversed(catalog)
        )

        self.assertEqual({row["score_source"] for row in rows}, {"segmented"})
        self.assertEqual(
            {row["rollout_id"]: row["scores"].tolist() for row in rows},
            {
                "parent-S:Stage": [0.1, 0.2],
                "parent-F:Stage": [0.7, 0.8],
            },
        )

    def test_saved_terminal_scores_are_sliced_to_catalog_stages(self):
        catalog = [
            self.catalog_record("parent-S", "Composite", "Stage", False, 1, 3),
            self.catalog_record("parent-F", "Composite", "Stage", True, 1, 3),
        ]
        saved = [
            {
                "rollout_id": "parent-S",
                "scores": [0.0, 0.1, 0.2, 0.3],
            },
            {
                "rollout_id": "parent-F",
                "scores": [0.0, 0.8, 0.9, 1.0],
            },
        ]

        rows = semantic_subtask_scores_from_saved_records(saved, catalog)

        self.assertEqual(
            {row["rollout_id"]: row["scores"].tolist() for row in rows},
            {
                "parent-S:Stage": [0.1, 0.2],
                "parent-F:Stage": [0.8, 0.9],
            },
        )
        self.assertEqual({row["score_source"] for row in rows}, {"terminal_sliced"})

    def test_fixed_prefix_keeps_short_completed_segments_and_sparse_stages(self):
        rows = [
            {
                "rollout_id": "success",
                "parent_rollout_id": "parent-success",
                "parent_task_name": "Composite",
                "task_name": "Composite::Estimable",
                "failed": False,
                "scores": [0.1],
            },
            {
                "rollout_id": "failure",
                "parent_rollout_id": "parent-failure",
                "parent_task_name": "Composite",
                "task_name": "Composite::Estimable",
                "failed": True,
                "scores": [0.8, 0.9, 0.95],
            },
            {
                "rollout_id": "sparse-success",
                "parent_rollout_id": "parent-sparse",
                "parent_task_name": "Composite",
                "task_name": "Composite::OneSided",
                "failed": False,
                "scores": [0.2, 0.3],
            },
        ]

        result = fixed_prefix_subtask_metrics(rows, prefixes=(1, 4))

        self.assertEqual(result["selection_value"], 1.0)
        later = result["per_prefix"]["4"]
        self.assertEqual(later["segments"], 3)
        self.assertEqual(later["completed_before_prefix_retained"], 3)
        self.assertEqual(later["estimable_stages"], 1)
        self.assertIsNone(later["per_stage"]["Composite::OneSided"]["roc_auc"])

    def test_elapsed_baseline_is_fitted_from_training_catalog_only(self):
        training = [
            self.catalog_record("train-S", "Composite", "Stage", False, 0, 1),
            self.catalog_record("train-F", "Composite", "Stage", True, 0, 3),
        ]
        validation = [
            {
                "rollout_id": "val-S",
                "parent_rollout_id": "val-S-parent",
                "parent_task_name": "Composite",
                "task_name": "Composite::Stage",
                "failed": False,
                "scores": [0.1],
            },
            {
                "rollout_id": "val-F",
                "parent_rollout_id": "val-F-parent",
                "parent_task_name": "Composite",
                "task_name": "Composite::Stage",
                "failed": True,
                "scores": [0.9, 0.95, 0.99],
            },
        ]

        result = subtask_fixed_prefix_selection(training, validation, prefixes=(1, 2))

        self.assertEqual(result["training_catalog_segments"], 2)
        self.assertEqual(result["safe"]["selection_value"], 1.0)
        self.assertIn("1", result["safe_minus_time_by_prefix"])
        self.assertEqual(
            result["time_model"]["Composite::Stage"]["source"],
            "training segments only",
        )

    def test_common_stage_catalog_uses_identical_supported_stages_in_every_fold(self):
        catalog = []
        parents = []
        for failed in (False, True):
            for index in range(3):
                parent = f"parent-{index}-{'F' if failed else 'S'}"
                parents.append(parent)
                catalog.append(
                    self.catalog_record(
                        parent,
                        "Composite",
                        "Common",
                        failed,
                        0,
                        2,
                    )
                )
                if not failed:
                    catalog.append(
                        self.catalog_record(
                            parent,
                            "Composite",
                            "OneSided",
                            False,
                            2,
                            3,
                        )
                    )

        result = freeze_common_subtask_stage_catalog(
            catalog,
            parents,
            num_folds=3,
            seed=7,
        )

        self.assertEqual(result["selected_stages"], ["Composite::Common"])
        self.assertIn("Composite::OneSided", result["excluded_stages"])
        self.assertEqual(
            {
                tuple(
                    sorted(
                        (
                            values["Composite::Common"]["successes"],
                            values["Composite::Common"]["failures"],
                        )
                    )
                )
                for values in result["support_by_fold"].values()
            },
            {(1, 1)},
        )

    def test_fixed_prefix_metrics_use_only_frozen_evaluation_stages(self):
        rows = [
            {
                "rollout_id": "common-S",
                "parent_rollout_id": "parent-S",
                "parent_task_name": "Composite",
                "task_name": "Composite::Common",
                "failed": False,
                "scores": [0.1],
            },
            {
                "rollout_id": "common-F",
                "parent_rollout_id": "parent-F",
                "parent_task_name": "Composite",
                "task_name": "Composite::Common",
                "failed": True,
                "scores": [0.9],
            },
            {
                "rollout_id": "excluded-S",
                "parent_rollout_id": "parent-extra",
                "parent_task_name": "Composite",
                "task_name": "Composite::OneSided",
                "failed": False,
                "scores": [0.99],
            },
        ]

        result = fixed_prefix_subtask_metrics(
            rows,
            prefixes=(1,),
            evaluation_stages=("Composite::Common",),
        )

        self.assertEqual(result["selection_value"], 1.0)
        self.assertEqual(result["input_segments"], 3)
        self.assertEqual(result["selected_segments"], 2)
        self.assertEqual(result["excluded_segments"], 1)


if __name__ == "__main__":
    unittest.main()
