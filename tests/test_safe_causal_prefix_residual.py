from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from robocasa.recovery.safe.causal_prefix_residual import (
    balance_training_rows,
    build_landmark_rows,
    event_metrics,
    fit_feature_scaler,
    fit_time_risk,
    landmark_metrics,
    primary_landmark_score,
    stratified_meta_split,
    summarize_prefix,
    summarize_trajectory,
    threshold_at_fpr,
    training_task_horizons,
    transform_features,
)


def make_rollout(task, success, length, rollout_id, offset=0.0):
    values = np.arange(length * 3, dtype=np.float32).reshape(length, 3) + offset
    rollout = SimpleNamespace(
        task_id=task,
        episode_success=int(success),
        hidden_states=values,
    )
    return rollout, (
        Path(f"{rollout_id}.pkl"),
        {"rollout_id": rollout_id, "task_id": task},
    )


def make_dataset(specs):
    rollouts = []
    identity = {}
    for spec in specs:
        rollout, aligned = make_rollout(*spec)
        rollouts.append(rollout)
        identity[id(rollout)] = aligned
    return rollouts, identity


class TestCausalPrefixResidual(unittest.TestCase):
    def test_prefix_summary_vectorization_matches_single_prefix(self):
        sequence = np.arange(18, dtype=np.float32).reshape(6, 3)
        trajectory = summarize_trajectory(sequence, window=3)
        self.assertEqual(trajectory.shape, (6, 12))
        for step in range(1, 7):
            np.testing.assert_allclose(
                trajectory[step - 1],
                summarize_prefix(sequence, step, window=3),
                rtol=0,
                atol=1e-6,
            )

    def test_meta_split_is_parent_disjoint_and_outcome_stratified(self):
        specs = []
        for task in (0, 1):
            for success in (False, True):
                for index in range(4):
                    specs.append(
                        (task, success, 8, f"t{task}-s{int(success)}-{index}", index)
                    )
        rollouts, identity = make_dataset(specs)
        split = stratified_meta_split(
            rollouts, identity, validation_per_class=1, seed=3
        )
        self.assertFalse(split["fit"] & split["validation"])
        self.assertEqual(len(split["fit"]), 12)
        self.assertEqual(len(split["validation"]), 4)
        for counts in split["counts"].values():
            for outcome in ("success", "failure"):
                self.assertEqual(counts[outcome], {"fit": 3, "validation": 1})

    def test_landmarks_use_fit_failure_horizon_and_natural_at_risk_support(self):
        rollouts, identity = make_dataset(
            [
                (0, True, 2, "success-early", 0),
                (0, True, 8, "success-late", 10),
                (0, False, 10, "failure-0", 20),
                (0, False, 10, "failure-1", 30),
            ]
        )
        horizons = training_task_horizons(rollouts)
        self.assertEqual(horizons, {0: 10})
        curves = fit_time_risk(rollouts, horizons)
        rows, excluded = build_landmark_rows(
            rollouts,
            identity,
            horizons=horizons,
            time_curves=curves,
            landmarks=(0.25, 0.5),
            window=2,
            split="fit",
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(excluded), 2)
        self.assertEqual(excluded[0]["rollout_id"], "success-early")
        self.assertEqual(excluded[0]["reason"], "naturally_terminated_before_landmark")
        self.assertTrue(all(row["features"].shape == (12,) for row in rows))
        self.assertTrue(all(0.0 <= row["time_risk"] <= 1.0 for row in rows))

    def test_balancing_equalizes_supported_outcomes_and_parent_total_weight(self):
        rows = []
        for fraction in (0.25, 0.5):
            for index in range(2):
                rows.append(
                    {
                        "rollout_id": f"s{index}",
                        "task_id": 0,
                        "failed": False,
                        "landmark_fraction": fraction,
                        "features": np.asarray([index, fraction], dtype=np.float32),
                    }
                )
            for index in range(4):
                rows.append(
                    {
                        "rollout_id": f"f{index}",
                        "task_id": 0,
                        "failed": True,
                        "landmark_fraction": fraction,
                        "features": np.asarray([index, fraction], dtype=np.float32),
                    }
                )
        selected, support, excluded = balance_training_rows(rows, seed=0)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(excluded), 4)
        for fraction in (0.25, 0.5):
            subset = [row for row in selected if row["landmark_fraction"] == fraction]
            self.assertEqual(sum(not row["failed"] for row in subset), 2)
            self.assertEqual(sum(row["failed"] for row in subset), 2)
        parent_totals = {}
        for row in selected:
            parent_totals.setdefault(row["rollout_id"], 0.0)
            parent_totals[row["rollout_id"]] += row["sample_weight"]
        self.assertAlmostEqual(max(parent_totals.values()), min(parent_totals.values()))
        self.assertEqual(support["0@0.25"]["selected_per_outcome"], 2)

        scaler = fit_feature_scaler(selected)
        transformed = transform_features(
            np.stack([row["features"] for row in selected]), scaler
        )
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_validation_threshold_and_metrics_are_causal(self):
        def scored(rollout_id, failed, values):
            trajectories = {
                detector: np.asarray(values, dtype=np.float64)
                for detector in ("time_only", "prefix_safe", "residual_safe_time")
            }
            return {
                "rollout_id": rollout_id,
                "task_id": 0,
                "failed": failed,
                "trajectories": trajectories,
            }

        validation = [
            scored("s0", False, [0.1, 0.2]),
            scored("s1", False, [0.1, 0.3]),
            scored("f0", True, [0.2, 0.9]),
            scored("f1", True, [0.3, 0.8]),
        ]
        threshold = threshold_at_fpr(validation, "prefix_safe", target_fpr=0.0)
        metrics = event_metrics(
            validation, "prefix_safe", threshold, horizons={0: 2}
        )
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertEqual(metrics["true_positive_rate"], 1.0)
        self.assertEqual(metrics["mean_detected_failure_fraction"], 1.0)

        landmark_rows = []
        for task in (0, 1):
            for failed, score in ((False, 0.1), (True, 0.9)):
                landmark_rows.append(
                    {
                        "task_id": task,
                        "failed": failed,
                        "landmark_fraction": 0.25,
                        "scores": {
                            "time_only": 0.5,
                            "prefix_safe": score,
                            "residual_safe_time": score,
                        },
                    }
                )
        rows = landmark_metrics(landmark_rows)
        value, support = primary_landmark_score(rows, "prefix_safe", (0.25,))
        self.assertEqual(value, 1.0)
        self.assertEqual(support, 2)

    def test_cli_exposes_preregistered_early_protocol(self):
        from robocasa.recovery.safe.run_causal_prefix_residual import build_parser

        args = build_parser().parse_args(
            [
                "--export-dir",
                "/data/xiaomi",
                "--safe-repo",
                "/src/SAFE",
                "--output-dir",
                "/results/prefix",
                "--landmarks",
                "0.1",
                "0.25",
                "0.5",
                "--primary-landmarks",
                "0.25",
                "0.5",
            ]
        )
        self.assertEqual(args.landmarks, [0.1, 0.25, 0.5])
        self.assertEqual(args.primary_landmarks, [0.25, 0.5])
        self.assertEqual(args.target_fpr, 0.05)


if __name__ == "__main__":
    unittest.main()
