import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages


install_lightweight_robocasa_packages()

from robocasa.recovery.safe.run_stage_aware_parent_safe import (  # noqa: E402
    _load_model_runtime,
    build_parser,
    load_external_fixed_prefix_scores,
    primary_per_task_values,
)
from robocasa.recovery.safe.score_seen_checkpoint_external import (  # noqa: E402
    build_parser as build_external_score_parser,
)
from robocasa.recovery.safe.stage_aware_parent_safe import (  # noqa: E402
    _failure_stage_observation,
    allocate_development_parents,
    apply_score_normalizer,
    build_inference_rows,
    conformal_success_threshold,
    fixed_prefix_metrics,
    fit_score_normalizer,
    fit_success_scaler,
    group_score_trajectories,
    make_parent_holdout_split,
    paired_fixed_prefix_bootstrap,
    paired_parent_bootstrap,
    parent_event_metrics,
    parent_stage_weights,
    select_supported_stages,
    stage_event_metrics,
    training_horizons,
)


def parent(
    rollout_id,
    *,
    task="Task",
    failed=False,
    first_value=0.0,
    context=True,
):
    features = np.asarray(
        [
            [first_value, 0.0],
            [first_value + 1.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
        ],
        dtype=np.float32,
    )
    spans = [
        {
            "stage_name": f"{task}::stage_0",
            "subtask_id": "stage_0",
            "stage_index_in_task": 0,
            "start": 0,
            "end": 2,
            "length": 2,
            "failed": False,
            "completed": True,
            "segment_id": f"{rollout_id}:0",
        },
        {
            "stage_name": f"{task}::stage_1",
            "subtask_id": "stage_1",
            "stage_index_in_task": 1,
            "start": 2,
            "end": 5,
            "length": 3,
            "failed": bool(failed),
            "completed": not bool(failed),
            "segment_id": f"{rollout_id}:1",
        },
    ]
    return {
        "rollout_id": rollout_id,
        "task_name": task,
        "failed": bool(failed),
        "environment_seed": int(rollout_id.rsplit("-", 1)[-1])
        if rollout_id.rsplit("-", 1)[-1].isdigit()
        else 0,
        "environment_reset_index": 0,
        "inference_environment_steps": [0, 2, 4, 6, 8],
        "features": features,
        "context": features * 0.1 if context else None,
        "spans": spans,
        "stage_assignment": [0, 0, 1, 1, 1],
    }


class TestStageAwareParentSafe(unittest.TestCase):
    def test_failed_stage_without_policy_inference_is_censored(self):
        failure_segment = {
            "subtask_id": "stage_1",
            "segment_index": 1,
            "failure_label": 1,
            "num_policy_inferences": 0,
            "usable_for_safe": False,
        }
        result = _failure_stage_observation(
            {"segments": [failure_segment]},
            [{"failed": False}],
            rollout_failed=True,
            rollout_id="parent",
        )
        self.assertFalse(result["observed"])
        self.assertIs(result["segment"], failure_segment)

    def test_failed_stage_with_policy_inference_remains_observed(self):
        failure_segment = {
            "subtask_id": "stage_1",
            "segment_index": 1,
            "failure_label": 1,
            "num_policy_inferences": 2,
            "usable_for_safe": True,
        }
        result = _failure_stage_observation(
            {"segments": [failure_segment]},
            [{"failed": True}],
            rollout_failed=True,
            rollout_id="parent",
        )
        self.assertTrue(result["observed"])

    def test_runtime_loader_explicitly_loads_trusted_numpy_metadata(self):
        fake_torch = mock.Mock()
        fake_torch.load.return_value = {
            "model_config": {},
            "state_dicts": [],
            "scaler": {"mean": np.asarray([0.0])},
        }
        with mock.patch.object(
            sys.modules[_load_model_runtime.__module__],
            "_torch",
            return_value=fake_torch,
        ):
            models, payload = _load_model_runtime(
                Path("/trusted/runtime"), "runtime_stage.pt", "cpu"
            )
        self.assertEqual(models, [])
        self.assertIn("scaler", payload)
        fake_torch.load.assert_called_once_with(
            Path("/trusted/runtime/runtime_stage.pt"),
            map_location="cpu",
            weights_only=False,
        )

    def test_development_allocation_is_parent_disjoint_and_stratified(self):
        parents = []
        for task in ("TaskA", "TaskB"):
            for failed in (False, True):
                for index in range(5):
                    parents.append(
                        parent(
                            f"{task}-{int(failed)}-{index}",
                            task=task,
                            failed=failed,
                        )
                    )
        split = allocate_development_parents(
            parents, num_folds=5, selection_fold=1, calibration_fold=0, seed=7
        )
        self.assertEqual(split["counts"], {"fit": 12, "selection": 4, "calibration": 4})
        self.assertFalse(split["fit"] & split["selection"])
        self.assertFalse(split["fit"] & split["calibration"])
        self.assertFalse(split["selection"] & split["calibration"])
        self.assertTrue(
            all(value["per_fold"] == [1, 1, 1, 1, 1] for value in split["strata"].values())
        )

    def test_raw_parent_holdout_split_does_not_materialize_segments(self):
        parents = []
        for task in ("TaskA", "TaskB"):
            for failed in (False, True):
                for index in range(5):
                    parents.append(parent(f"{task}-{int(failed)}-{index}", task=task, failed=failed))
        split = make_parent_holdout_split(parents, train_fraction=0.6, seed=3)
        self.assertEqual(split["counts"], {"parent_train": 12, "parent_test": 8, "total": 20})
        self.assertFalse(set(split["parent_train"]) & set(split["parent_test"]))
        self.assertEqual(
            set(split["parent_train"]) | set(split["parent_test"]),
            {item["rollout_id"] for item in parents},
        )

    def test_full_parent_rows_preserve_history_and_censor_future_stages(self):
        failed = parent("failed", failed=True, first_value=10.0)
        successful = parent("success", failed=False, first_value=-10.0)
        stages = ["Task::stage_0", "Task::stage_1"]
        horizons = training_horizons([successful], stages)
        rows = build_inference_rows(
            [failed],
            selected_stages=stages,
            horizons=horizons,
            failure_horizons=(2, 4),
            temporal_window=2,
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual([row["stage_failed"] for row in rows[:2]], [False, False])
        self.assertEqual([row["stage_failed"] for row in rows[2:]], [True, True, True])
        self.assertEqual(rows[2]["local_index"], 1)
        self.assertFalse(rows[2]["within_horizon"]["2"])
        self.assertTrue(rows[3]["within_horizon"]["2"])
        np.testing.assert_array_equal(rows[2]["stage_anchor_delta"], np.zeros(2))

        alternative = parent("alternative", failed=True, first_value=-10.0)
        alternative_rows = build_inference_rows(
            [alternative],
            selected_stages=stages,
            horizons=horizons,
            failure_horizons=(2, 4),
            temporal_window=2,
        )
        # Both parents have the same current feature at the beginning of stage 1,
        # but their causal summaries differ because the earlier stage is retained.
        np.testing.assert_array_equal(failed["features"][2], alternative["features"][2])
        self.assertFalse(
            np.array_equal(rows[2]["common_features"], alternative_rows[2]["common_features"])
        )

    def test_stage_support_is_selected_from_fit_only(self):
        parents = [
            parent(f"success-{index}", failed=False) for index in range(3)
        ] + [parent(f"failure-{index}", failed=True) for index in range(2)]
        selection = select_supported_stages(
            parents, min_successes=3, min_failures=2
        )
        self.assertEqual(selection["selected_stages"], ["Task::stage_1"])
        self.assertIn("Task::stage_0", selection["excluded"])

    def test_balanced_weights_equalize_stage_outcome_parent_mass(self):
        rows = []
        for failed, count in ((False, 2), (True, 5)):
            for index in range(count):
                rows.append(
                    {
                        "task_name": "Task",
                        "stage_name": "Task::stage",
                        "parent_rollout_id": f"{int(failed)}-{index // 2}",
                        "stage_failed": failed,
                    }
                )
        weights = parent_stage_weights(rows, target="stage")
        labels = np.asarray([row["stage_failed"] for row in rows])
        self.assertAlmostEqual(float(weights[~labels].sum()), float(weights[labels].sum()))
        self.assertTrue(np.all(weights > 0))

    def test_prototype_scaler_is_success_only(self):
        rows = []
        for failed, offset in ((False, 0.0), (True, 1000.0)):
            for index in range(2):
                rows.append(
                    {
                        "task_name": "Task",
                        "stage_name": "Task::stage",
                        "parent_rollout_id": f"{int(failed)}-{index}",
                        "stage_failed": failed,
                        "common_features": np.asarray([offset + index], dtype=np.float32),
                    }
                )
        scaler = fit_success_scaler(rows)
        self.assertAlmostEqual(float(scaler["mean"][0]), 0.5)

    def test_fixed_prefix_metrics_use_subtask_labels(self):
        rows, scores = [], []
        for local_index in (1, 2):
            for failed, score in ((False, 0.1), (False, 0.2), (True, 0.8), (True, 0.9)):
                rows.append(
                    {
                        "task_name": "Task",
                        "stage_name": "Task::stage",
                        "stage_failed": failed,
                        "local_index": local_index,
                    }
                )
                scores.append(score)
        metrics = fixed_prefix_metrics(rows, scores, prefixes=(1, 2))
        self.assertEqual(metrics["1"]["task_stage_macro_roc_auc"], 1.0)
        self.assertEqual(metrics["2"]["task_stage_macro_roc_auc"], 1.0)
        self.assertEqual(metrics["1"]["at_risk_definition"], "stage remains active at the exact prefix")

    def test_external_score_ensemble_aligns_by_parent_and_inference(self):
        stage_rows = [
            {
                "parent_rollout_id": parent_id,
                "inference_index": inference_index,
            }
            for parent_id in ("parent-a", "parent-b")
            for inference_index in (0, 1)
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for member, offset in enumerate((0.0, 2.0)):
                path = Path(directory) / f"member-{member}.jsonl"
                with path.open("w") as stream:
                    for parent_id in ("parent-a", "parent-b"):
                        stream.write(
                            json.dumps(
                                {
                                    "rollout_id": parent_id,
                                    "scores": [offset + 1.0, offset + 3.0],
                                }
                            )
                            + "\n"
                        )
                paths.append(str(path))
            scores, provenance = load_external_fixed_prefix_scores(
                [f"safe38={','.join(paths)}"],
                stage_rows,
                {"parent-a", "parent-b"},
            )
        np.testing.assert_allclose(scores["safe38"], [2.0, 4.0, 2.0, 4.0])
        self.assertEqual(provenance["safe38"]["members"], 2)
        self.assertFalse(provenance["safe38"]["threshold_fitted"])

    def test_primary_per_task_values_average_requested_prefixes(self):
        metrics = {
            "1": {"per_task": {"A": 0.6, "B": 0.7}},
            "2": {"per_task": {"A": 0.8, "B": 0.5}},
            "4": {"per_task": {"A": 0.1, "B": 0.1}},
        }
        self.assertEqual(
            primary_per_task_values(metrics, (1, 2)),
            {"A": 0.7, "B": 0.6},
        )

    def test_calibration_event_and_parent_metrics(self):
        rows = []
        scores = []
        for parent_id, failed, values in (
            ("success", False, (0.0, 0.1)),
            ("failure", True, (0.2, 2.0)),
        ):
            for index, score in enumerate(values, start=1):
                rows.append(
                    {
                        "parent_rollout_id": parent_id,
                        "task_name": "Task",
                        "stage_name": "Task::stage",
                        "segment_id": f"{parent_id}:stage",
                        "stage_failed": failed,
                        "local_index": index,
                    }
                )
                scores.append(score)
        events = group_score_trajectories(rows, {"safe": np.asarray(scores)})
        normalizer = fit_score_normalizer(
            events
            + [
                {
                    **events[0],
                    "segment_id": "success-2:stage",
                    "parent_rollout_id": "success-2",
                }
            ],
            "safe",
        )
        normalized = apply_score_normalizer(events, "safe", normalizer)
        threshold = conformal_success_threshold(
            [normalized[0]], "safe", target_fpr=0.5
        )
        self.assertTrue(np.isfinite(threshold["threshold"]))
        self.assertEqual(threshold["calibration_unit"], "parent")
        stage = stage_event_metrics(
            normalized,
            "safe",
            threshold["threshold"],
            {"stage": {"Task::stage": 2}},
        )
        parent_metrics = parent_event_metrics(stage["predictions"])
        self.assertEqual(parent_metrics["parents"], 2)
        self.assertEqual(parent_metrics["failures"], 1)

    def test_unsupported_failed_stage_is_excluded_not_relabelled_success(self):
        completed_only = [
            {
                "parent_rollout_id": "failed-parent",
                "task_name": "Task",
                "stage_name": "Task::completed",
                "segment_id": "failed-parent:completed",
                "failed": False,
                "detected": False,
                "adjusted_detection_fraction": None,
            }
        ]
        result = parent_event_metrics(
            completed_only,
            {
                "failed-parent": {"failed": True, "task_name": "Task"},
                "success-parent": {"failed": False, "task_name": "Task"},
            },
        )
        self.assertEqual(result["parents"], 0)
        self.assertEqual(result["excluded_parents"], 2)
        self.assertEqual(result["excluded_failed_stage_not_fit_supported"], 1)

    def test_paired_bootstrap_keeps_task_parent_pairing(self):
        candidate = []
        baseline = []
        for task in ("A", "B"):
            for index in range(3):
                base = {
                    "parent_rollout_id": f"{task}-{index}",
                    "task_name": task,
                    "failed": True,
                }
                candidate.append({**base, "adjusted_detection_fraction": 0.2})
                baseline.append({**base, "adjusted_detection_fraction": 0.7})
        result = paired_parent_bootstrap(
            {"candidate": candidate, "time_only": baseline},
            "candidate",
            replicates=100,
            seed=0,
        )
        self.assertAlmostEqual(result["point"], -0.5)
        self.assertAlmostEqual(result["ci95"][0], -0.5)
        self.assertAlmostEqual(result["ci95"][1], -0.5)

    def test_fixed_prefix_bootstrap_keeps_task_parent_pairing(self):
        rows, candidate, baseline = [], [], []
        for task in ("A", "B"):
            for failed in (False, True):
                parent_id = f"{task}-{int(failed)}"
                for prefix in (1, 2):
                    rows.append(
                        {
                            "parent_rollout_id": parent_id,
                            "task_name": task,
                            "stage_name": f"{task}::stage",
                            "stage_failed": failed,
                            "terminal_failed": failed,
                            "local_index": prefix,
                        }
                    )
                    candidate.append(float(failed))
                    baseline.append(0.5)
        result = paired_fixed_prefix_bootstrap(
            rows,
            {
                "candidate": np.asarray(candidate),
                "terminal": np.asarray(baseline),
            },
            "candidate",
            baseline="terminal",
            prefixes=(1, 2),
            primary_prefixes=(1, 2),
            replicates=50,
            seed=0,
        )
        self.assertAlmostEqual(result["point"], 0.5)
        self.assertAlmostEqual(result["ci95"][0], 0.5)
        self.assertAlmostEqual(result["ci95"][1], 0.5)
        self.assertEqual(result["replicates_estimable"], 50)

    def test_cli_separates_development_and_frozen_evaluation(self):
        parser = build_parser()
        develop = parser.parse_args(
            [
                "develop",
                "--dataset-dir",
                "/raw",
                "--selection-manifest",
                "/split.json",
                "--output-dir",
                "/out",
            ]
        )
        self.assertEqual(develop.phase, "develop")
        self.assertIn("context", develop.arms)
        split = parser.parse_args(
            [
                "split",
                "--dataset-dir",
                "/raw",
                "--output-dir",
                "/split",
                "--require-context",
            ]
        )
        self.assertTrue(split.require_context)
        evaluate = parser.parse_args(
            [
                "evaluate",
                "--dataset-dir",
                "/prospective",
                "--runtime-bundle",
                "/runtime.json",
                "--output-dir",
                "/eval",
                "--external-scores",
                "safe38=/scores/seed0,/scores/seed1",
                "--opened-outer",
            ]
        )
        self.assertEqual(evaluate.phase, "evaluate")
        self.assertTrue(evaluate.opened_outer)
        self.assertEqual(
            evaluate.external_scores,
            ["safe38=/scores/seed0,/scores/seed1"],
        )

    def test_external_scorer_accepts_strict_training_task_subset(self):
        args = build_external_score_parser().parse_args(
            [
                "--export-dir",
                "/export",
                "--safe-repo",
                "/safe",
                "--training-run-dir",
                "/training",
                "--output-dir",
                "/scores",
                "--group",
                "prospective_test",
                "--allow-task-subset",
            ]
        )
        self.assertTrue(args.allow_task_subset)


if __name__ == "__main__":
    unittest.main()
