import unittest

import numpy as np

from robocasa.recovery.safe.early_safe_time_cascade import (
    cascade_metrics,
    first_alert,
    select_joint_thresholds,
    select_single_threshold,
    stage_steps,
    validate_stages,
)


def scored(rollout_id, failed, early, late, full=None):
    early_stages = [
        {
            "landmark_fraction": fraction,
            "detector": f"early_{fraction}",
            "step": step,
            "score": score,
        }
        for fraction, step, score in early
    ]
    late_steps = np.arange(5, 5 + len(late), dtype=np.int64)
    full_values = list(late if full is None else full)
    return {
        "rollout_id": rollout_id,
        "task_id": 0,
        "failed": failed,
        "early_stages": early_stages,
        "late_time_steps": late_steps,
        "late_time_scores": np.asarray(late, dtype=np.float64),
        "full_time_steps": np.arange(1, 1 + len(full_values), dtype=np.int64),
        "full_time_scores": np.asarray(full_values, dtype=np.float64),
    }


class TestEarlySafeTimeCascade(unittest.TestCase):
    def test_stage_schedule_requires_safe_before_time_fallback(self):
        landmarks, fallback = validate_stages((0.25, 0.10), 0.50)
        self.assertEqual(landmarks, (0.10, 0.25))
        self.assertEqual(fallback, 0.50)
        self.assertEqual(
            stage_steps(11, landmarks, fallback),
            {"early": {0.10: 2, 0.25: 3}, "time_fallback": 6},
        )
        with self.assertRaises(ValueError):
            validate_stages((0.10, 0.50), 0.50)

    def test_cascade_uses_only_declared_checkpoints_and_late_window(self):
        item = scored(
            "f0",
            True,
            [(0.10, 1, 0.7), (0.25, 3, 0.9)],
            [0.8, 0.9],
            full=[0.99, 0.99, 0.99, 0.99, 0.8, 0.9],
        )
        alert = first_alert(
            item,
            "staged_safe_time",
            {"early_safe": 0.8, "late_time": 0.75},
        )
        self.assertEqual(alert["step"], 3)
        self.assertEqual(alert["source"], "early_safe")

        alert = first_alert(
            item,
            "staged_safe_time",
            {"early_safe": 0.95, "late_time": 0.75},
        )
        self.assertEqual(alert["step"], 5)
        self.assertEqual(alert["source"], "late_time")

    def test_joint_selection_preserves_shared_fpr_and_rewards_early_alarm(self):
        validation = [
            scored("s0", False, [(0.10, 1, 0.1)], [0.2], full=[0.1, 0.2]),
            scored("s1", False, [(0.10, 1, 0.2)], [0.2], full=[0.1, 0.2]),
            scored("f0", True, [(0.10, 1, 0.9)], [0.8], full=[0.1, 0.8]),
            scored("f1", True, [(0.10, 1, 0.1)], [0.8], full=[0.1, 0.8]),
        ]
        selected = select_joint_thresholds(validation, horizons={0: 10}, target_fpr=0.0)
        metrics = selected["validation_metrics"]
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertEqual(metrics["true_positive_rate"], 1.0)
        self.assertAlmostEqual(
            metrics["missed_failure_adjusted_detection_fraction"], 0.3
        )
        self.assertEqual(metrics["alarm_source_counts"]["early_safe"], 1)
        self.assertEqual(metrics["alarm_source_counts"]["late_time"], 1)
        self.assertGreater(selected["search"]["candidate_pairs"], 0)

    def test_baselines_receive_independent_validation_thresholds(self):
        validation = [
            scored("s", False, [(0.10, 1, 0.2)], [0.2], full=[0.1, 0.2]),
            scored("f", True, [(0.10, 1, 0.8)], [0.8], full=[0.1, 0.8]),
        ]
        early = select_single_threshold(
            validation, "early_safe", {0: 10}, target_fpr=0.0
        )
        time = select_single_threshold(validation, "time_only", {0: 10}, target_fpr=0.0)
        self.assertEqual(early["validation_metrics"]["true_positive_rate"], 1.0)
        self.assertEqual(time["validation_metrics"]["true_positive_rate"], 1.0)
        evaluated = cascade_metrics(
            validation,
            "early_safe",
            {"early_safe": early["threshold"]},
            {0: 10},
        )
        self.assertEqual(evaluated["false_positive_rate"], 0.0)

    def test_cli_defaults_encode_the_preregistered_cascade(self):
        from robocasa.recovery.safe.run_early_safe_time_cascade import build_parser

        args = build_parser().parse_args(
            [
                "--export-dir",
                "/data/xiaomi",
                "--safe-repo",
                "/src/SAFE",
                "--output-dir",
                "/results/cascade",
            ]
        )
        self.assertEqual(args.early_landmarks, [0.10, 0.25])
        self.assertEqual(args.time_fallback, 0.50)
        self.assertEqual(args.target_fpr, 0.05)


if __name__ == "__main__":
    unittest.main()
