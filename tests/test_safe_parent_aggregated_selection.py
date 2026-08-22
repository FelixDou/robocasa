from pathlib import Path
from types import SimpleNamespace
import unittest

from robocasa.recovery.safe.run_seen_cv_grid import make_inner_folds
from robocasa.recovery.safe.subtask_safe_evaluation import (
    parent_aggregated_early_metrics,
    parent_group_id,
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
                    [0.8, 0.9, 0.95, 0.99]
                    if failed
                    else [0.05, 0.1, 0.15, 0.2]
                )

        result = parent_aggregated_early_metrics(
            rollouts, scores, identity, landmarks=(0.25, 0.5)
        )

        self.assertEqual(result["parents"], 4)
        self.assertEqual(result["selection_value"], 1.0)
        self.assertEqual(
            result["per_landmark"]["0.25"]["task_macro_roc_auc"], 1.0
        )
        self.assertEqual(
            result["per_landmark"]["0.5"]["parents_without_score"], 0
        )

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
                    scores.append(
                        [0.8, 0.9] if failed else [0.05, 0.1]
                    )

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

        self.assertEqual(
            result["per_landmark"]["0.25"]["parents_without_score"], 4
        )
        self.assertEqual(
            result["per_landmark"]["0.25"]["task_macro_roc_auc"], 0.5
        )
        self.assertEqual(
            result["per_landmark"]["1.0"]["task_macro_roc_auc"], 1.0
        )

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
                            "episode_success": int(
                                segment_index == 0 or not failed
                            ),
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


if __name__ == "__main__":
    unittest.main()
