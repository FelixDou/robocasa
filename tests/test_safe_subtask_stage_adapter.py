from pathlib import Path
from types import SimpleNamespace
import json
import unittest

import numpy as np

from robocasa.recovery.safe.subtask_stage_adapter import (
    assign_parent_weights,
    balance_bce_rows,
    build_stage_landmark_rows,
    continuation_gate,
    fit_time_risk,
    parent_bootstrap_delta,
    select_stage_thresholds,
    select_supported_stages,
    stage_event_metrics,
    three_way_parent_split,
    training_stage_horizons,
)


def make_rollout(
    parent,
    parent_task,
    parent_failed,
    stage,
    task_id,
    success,
    length=8,
    offset=0.0,
):
    rollout = SimpleNamespace(
        task_id=int(task_id),
        episode_success=int(success),
        hidden_states=(
            np.arange(length * 3, dtype=np.float32).reshape(length, 3) + offset
        ),
    )
    segment = f"{parent}::{stage}"
    env = {
        "rollout_id": segment,
        "parent_rollout_id": parent,
        "parent_task_name": parent_task,
        "parent_rollout_failed": bool(parent_failed),
        "task_name": f"{parent_task}::{stage}",
        "subtask_id": stage,
        "task_id": int(task_id),
        "episode_success": int(success),
    }
    return rollout, (Path(f"{segment}.pkl"), env)


def make_dataset(specs):
    rollouts = []
    identity = {}
    for spec in specs:
        rollout, aligned = make_rollout(*spec)
        rollouts.append(rollout)
        identity[id(rollout)] = aligned
    return rollouts, identity


class TestSubtaskStageAdapter(unittest.TestCase):
    def test_three_way_split_keeps_complete_parents_disjoint(self):
        specs = []
        for parent_task, task_offset in (("TaskA", 0), ("TaskB", 2)):
            for parent_failed in (False, True):
                for index in range(5):
                    parent = f"{parent_task}-{int(parent_failed)}-{index}"
                    specs.append(
                        (
                            parent,
                            parent_task,
                            parent_failed,
                            "stage0",
                            task_offset,
                            True,
                        )
                    )
                    specs.append(
                        (
                            parent,
                            parent_task,
                            parent_failed,
                            "stage1",
                            task_offset + 1,
                            not parent_failed,
                        )
                    )
        rollouts, identity = make_dataset(specs)
        split = three_way_parent_split(
            rollouts,
            identity,
            num_folds=5,
            selection_fold=1,
            diagnostic_fold=0,
            seed=7,
        )
        self.assertFalse(split["fit"] & split["selection"])
        self.assertFalse(split["fit"] & split["diagnostic"])
        self.assertFalse(split["selection"] & split["diagnostic"])
        self.assertEqual(split["counts"], {"fit": 12, "selection": 4, "diagnostic": 4})
        for parent in split["selection"]:
            selected_segments = [
                rollout
                for rollout in rollouts
                if identity[id(rollout)][1]["parent_rollout_id"] == parent
            ]
            self.assertEqual(len(selected_segments), 2)

    def test_three_way_split_supports_four_parent_strata_with_five_folds(self):
        specs = []
        for parent_task, task_offset in (("TaskA", 0), ("TaskB", 2)):
            for parent_failed in (False, True):
                for index in range(4):
                    parent = f"{parent_task}-{int(parent_failed)}-{index}"
                    specs.append(
                        (
                            parent,
                            parent_task,
                            parent_failed,
                            "stage0",
                            task_offset,
                            not parent_failed,
                        )
                    )
        rollouts, identity = make_dataset(specs)
        split = three_way_parent_split(
            rollouts,
            identity,
            num_folds=5,
            selection_fold=1,
            diagnostic_fold=0,
            seed=7,
        )
        self.assertEqual(split["counts"], {"fit": 8, "selection": 4, "diagnostic": 4})
        for counts in split["strata"].values():
            self.assertEqual(counts["per_fold"], [1, 1, 1, 1, 0])

    def test_fit_only_stage_support_horizons_and_landmarks(self):
        rollouts, identity = make_dataset(
            [
                ("s0", "Task", False, "stage", 0, True, 8, 0),
                ("s1", "Task", False, "stage", 0, True, 10, 1),
                ("f0", "Task", True, "stage", 0, False, 12, 2),
                ("f1", "Task", True, "stage", 0, False, 12, 3),
            ]
        )
        support = select_supported_stages(
            rollouts,
            identity,
            min_successes=2,
            min_failures=2,
        )
        self.assertEqual(support["catalog"], {"Task::stage": 0})
        horizons = training_stage_horizons(rollouts, identity, quantile=0.5)
        self.assertEqual(horizons, {0: 9})
        curves = fit_time_risk(rollouts, horizons)
        rows, excluded = build_stage_landmark_rows(
            rollouts,
            identity,
            stage_catalog=support["catalog"],
            horizons=horizons,
            time_curves=curves,
            landmarks=(0.25, 0.5),
            window=2,
            split="fit",
        )
        self.assertEqual(len(rows), 8)
        self.assertFalse(excluded)
        self.assertTrue(all(row["features"].shape == (12,) for row in rows))
        parent_totals = {}
        for row in rows:
            parent_totals.setdefault(row["parent_rollout_id"], 0.0)
            parent_totals[row["parent_rollout_id"]] += row["sample_weight"]
        self.assertAlmostEqual(max(parent_totals.values()), min(parent_totals.values()))

    def test_bce_balance_is_per_stage_landmark_and_parent_weighted(self):
        rows = []
        for stage_index, stage in enumerate(("A::x", "B::y")):
            for landmark in (0.25, 0.5):
                for failed, count in ((False, 2), (True, 4)):
                    for index in range(count):
                        rows.append(
                            {
                                "parent_rollout_id": f"{stage}-{int(failed)}-{index}",
                                "segment_id": f"{stage}-{int(failed)}-{index}",
                                "stage_name": stage,
                                "stage_index": stage_index,
                                "task_id": stage_index,
                                "failed": failed,
                                "landmark_fraction": landmark,
                                "features": np.asarray(
                                    [index, landmark], dtype=np.float32
                                ),
                            }
                        )
        rows = assign_parent_weights(rows)
        selected, support, excluded = balance_bce_rows(rows, seed=0)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len(excluded), 8)
        for key, counts in support.items():
            self.assertEqual(counts["selected_per_outcome"], 2, key)
        parent_totals = {}
        for row in selected:
            parent_totals.setdefault(row["parent_rollout_id"], 0.0)
            parent_totals[row["parent_rollout_id"]] += row["sample_weight"]
        self.assertAlmostEqual(max(parent_totals.values()), min(parent_totals.values()))

    def test_stage_thresholds_and_event_metrics_use_first_alarm(self):
        def scored(segment, failed, candidate, time):
            return {
                "parent_rollout_id": segment,
                "segment_id": segment,
                "stage_name": "Task::stage",
                "failed": failed,
                "source_inferences": 4,
                "trajectories": {
                    "candidate": np.asarray(candidate, dtype=np.float64),
                    "time_only": np.asarray(time, dtype=np.float64),
                },
            }

        selection = [
            scored("s0", False, [0.1, 0.2], [0.1, 0.2]),
            scored("s1", False, [0.1, 0.3], [0.1, 0.2]),
            scored("f0", True, [0.8, 0.9], [0.2, 0.8]),
            scored("f1", True, [0.7, 0.9], [0.2, 0.8]),
        ]
        thresholds, audit = select_stage_thresholds(
            selection,
            "candidate",
            target_fpr=0.0,
            min_successes=2,
        )
        self.assertTrue(audit["Task::stage"]["supported"])
        metrics = stage_event_metrics(selection, "candidate", thresholds)
        self.assertEqual(metrics["macro"]["false_positive_rate"], 0.0)
        self.assertEqual(metrics["macro"]["true_positive_rate"], 1.0)
        self.assertEqual(metrics["macro"]["mean_detected_lead"], 0.75)

    def test_strict_fpr_uses_finite_abstention_threshold(self):
        def scored(segment, failed):
            return {
                "parent_rollout_id": segment,
                "segment_id": segment,
                "stage_name": "Task::stage",
                "failed": failed,
                "source_inferences": 2,
                "trajectories": {
                    "time_only": np.asarray([0.5, 0.5], dtype=np.float64),
                },
            }

        selection = [
            scored("s0", False),
            scored("s1", False),
            scored("f0", True),
            scored("f1", True),
        ]
        thresholds, audit = select_stage_thresholds(
            selection,
            "time_only",
            target_fpr=0.05,
            min_successes=2,
        )
        threshold = thresholds["Task::stage"]
        self.assertTrue(np.isfinite(threshold))
        self.assertGreater(threshold, 1.0)
        self.assertTrue(audit["Task::stage"]["abstains_on_selection"])
        json.dumps(thresholds, allow_nan=False)
        metrics = stage_event_metrics(selection, "time_only", thresholds)
        self.assertEqual(metrics["macro"]["false_positive_rate"], 0.0)
        self.assertEqual(metrics["macro"]["true_positive_rate"], 0.0)

    def test_parent_bootstrap_and_gate_reward_incremental_signal(self):
        rows = []
        for stage in ("TaskA::stage", "TaskB::stage"):
            for index in range(6):
                for failed in (False, True):
                    for landmark in (0.25, 0.5):
                        rows.append(
                            {
                                "parent_rollout_id": f"{stage}-{int(failed)}-{index}",
                                "segment_id": f"{stage}-{int(failed)}-{index}",
                                "stage_name": stage,
                                "landmark_fraction": landmark,
                                "failed": failed,
                                "scores": {
                                    "candidate": 0.9 if failed else 0.1,
                                    "time_only": 0.5,
                                },
                            }
                        )
        bootstrap = parent_bootstrap_delta(
            rows,
            primary_landmarks=(0.25, 0.5),
            replicates=200,
            seed=1,
        )
        self.assertEqual(bootstrap["candidate_macro_auc"], 1.0)
        self.assertGreater(bootstrap["delta"], 0.0)
        events = {
            "macro": {
                "true_positive_rate": 0.8,
                "false_positive_rate": 0.0,
                "max_stage_false_positive_rate": 0.0,
                "mean_detected_lead": 0.5,
            }
        }
        gate = continuation_gate(
            bootstrap,
            events,
            min_roc=0.65,
            min_delta=0.01,
            min_tpr=0.4,
            max_fpr=0.05,
            min_lead=0.25,
        )
        self.assertTrue(gate["pass"])

    def test_cli_defaults_keep_outer_test_locked(self):
        from robocasa.recovery.safe.run_subtask_stage_adapter import build_parser

        args = build_parser().parse_args(
            [
                "--export-dir",
                "/data/subtask",
                "--safe-repo",
                "/src/SAFE",
                "--output-dir",
                "/results/stage-adapter",
                "--selection-manifest",
                "/data/subtask/parent_rollout_split.json",
            ]
        )
        self.assertEqual(args.meta_folds, 5)
        self.assertEqual(args.primary_landmarks, [0.25, 0.5])
        self.assertIn("stage_adapter", args.architectures)
        self.assertIn("temporal_contrastive", args.objectives)
        self.assertEqual(args.target_fpr, 0.05)
        self.assertFalse(hasattr(args, "score_outer_test"))


if __name__ == "__main__":
    unittest.main()
