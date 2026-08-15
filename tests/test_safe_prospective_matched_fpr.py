import unittest
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from robocasa.recovery.safe.prospective_matched_fpr import (
    event_metrics,
    first_alert,
    paired_failure_comparison,
    select_single_threshold,
    select_staged_thresholds,
    validate_stage_windows,
    wilson_interval,
)


def record(rollout_id, failed, safe, time, task="TaskA", horizon=10):
    return {
        "rollout_id": rollout_id,
        "task_name": task,
        "failed": failed,
        "horizon": horizon,
        "source_inferences": len(safe),
        "safe_scores": safe,
        "time_scores": time,
    }


class TestProspectiveMatchedFpr(unittest.TestCase):
    def test_stage_windows_are_causal_and_disjoint(self):
        self.assertEqual(validate_stage_windows(0.25, 0.50), (0.25, 0.50))
        with self.assertRaises(ValueError):
            validate_stage_windows(0.50, 0.50)
        row = record(
            "f",
            True,
            [0.1, 0.8, 0.9, 0.9, 0.9, 0.9],
            [0.9, 0.9, 0.9, 0.9, 0.7, 0.8],
        )
        alert = first_alert(
            row,
            "staged_safe_time",
            {"early_safe": 0.75, "late_time": 0.65},
        )
        self.assertEqual(alert["step"], 2)
        self.assertEqual(alert["source"], "early_safe")
        alert = first_alert(
            row,
            "staged_safe_time",
            {"early_safe": 0.95, "late_time": 0.65},
        )
        self.assertEqual(alert["step"], 5)
        self.assertEqual(alert["source"], "late_time")

    def test_thresholds_share_the_five_percent_event_fpr_budget(self):
        rows = [
            record("s0", False, [0.1] * 10, [0.1] * 10),
            record("s1", False, [0.2] * 10, [0.2] * 10),
            record("f0", True, [0.9] * 10, [0.8] * 10),
            record("f1", True, [0.1, 0.9] + [0.9] * 8, [0.1] * 5 + [0.8] * 5),
        ]
        safe = select_single_threshold(rows, "safe_only", target_fpr=0.0)
        time = select_single_threshold(rows, "time_only", target_fpr=0.0)
        staged = select_staged_thresholds(rows, target_fpr=0.0)
        self.assertEqual(safe["validation_metrics"]["false_positive_rate"], 0.0)
        self.assertEqual(time["validation_metrics"]["false_positive_rate"], 0.0)
        self.assertEqual(staged["validation_metrics"]["false_positive_rate"], 0.0)
        self.assertEqual(staged["validation_metrics"]["true_positive_rate"], 1.0)

    def test_metrics_include_intervals_recall_and_paired_timing(self):
        rows = [
            record("s0", False, [0.1] * 10, [0.1] * 10),
            record("s1", False, [0.1] * 10, [0.1] * 10),
            record("f0", True, [0.9] + [0.9] * 9, [0.1] * 7 + [0.9] * 3),
            record("f1", True, [0.1] * 10, [0.1] * 7 + [0.9] * 3),
        ]
        safe_threshold = {"safe_only": 0.8}
        time_threshold = {"time_only": 0.8}
        metrics = event_metrics(rows, "safe_only", safe_threshold)
        self.assertEqual(metrics["confusion"], {"tp": 1, "fn": 1, "fp": 0, "tn": 2})
        self.assertEqual(metrics["failure_recall_by_landmark"]["0.1"], 0.5)
        self.assertEqual(len(metrics["fpr_wilson_95"]), 2)
        paired = paired_failure_comparison(
            rows,
            "safe_only",
            safe_threshold,
            "time_only",
            time_threshold,
        )
        self.assertEqual(paired["earlier"], 1)
        self.assertEqual(paired["later"], 1)

    def test_wilson_interval_is_bounded(self):
        low, high = wilson_interval(5, 100)
        self.assertLess(low, 0.05)
        self.assertGreater(high, 0.05)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_two_phase_cli_defaults_are_frozen(self):
        from robocasa.recovery.safe.run_prospective_matched_fpr import build_parser

        args = build_parser().parse_args(
            [
                "calibrate",
                "--final-root",
                "/final",
                "--calibration-score-root",
                "/scores",
                "--output-dir",
                "/out",
            ]
        )
        self.assertEqual(args.target_fpr, 0.05)
        self.assertEqual(args.safe_end, 0.25)
        self.assertEqual(args.time_start, 0.50)
        self.assertEqual(args.min_per_class, 3)

    def test_two_phase_runner_freezes_then_reuses_thresholds(self):
        from robocasa.recovery.safe.run_prospective_matched_fpr import (
            calibrate,
            evaluate,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "final"
            calibration_scores = root / "calibration_scores"
            evaluation_scores = root / "evaluation_scores"
            for seed in (0, 1, 2):
                training_run = final / f"indep_seed{seed}"
                training_run.mkdir(parents=True)
                training_rows = []
                for task_index, task in enumerate(("TaskA", "TaskB")):
                    for index, failed in enumerate((False, False, True, True)):
                        training_rows.append(
                            {
                                "rollout_id": f"train-{task}-{index}",
                                "split": "train",
                                "task_name": task,
                                "failed": failed,
                                "scores": [
                                    0.1 + 0.01 * seed + 0.02 * task_index,
                                    (0.8 if failed else 0.2) + 0.01 * seed,
                                ],
                            }
                        )
                (training_run / "scores.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in training_rows)
                )
                for name in (
                    "model_final.ckpt",
                    "config.yaml",
                    "metrics.json",
                    "split_manifest.json",
                ):
                    (training_run / name).write_text(f"seed={seed} name={name}\n")
                for score_root, group, base_seed in (
                    (calibration_scores, "calibration", 1000),
                    (evaluation_scores, "prospective_test", 2000),
                ):
                    seed_root = score_root / f"seed{seed}"
                    seed_root.mkdir(parents=True)
                    rows = []
                    for task_index, task in enumerate(("TaskA", "TaskB")):
                        for index, failed in enumerate((False, True)):
                            rows.append(
                                {
                                    "rollout_id": f"{group}-{task}-{index}",
                                    "task_name": task,
                                    "task_type": "atomic" if task == "TaskA" else "composite",
                                    "failed": failed,
                                    "scores": [0.1 + 0.01 * seed, 0.9 if failed else 0.2],
                                    "environment_seed": base_seed + task_index,
                                    "environment_reset_index": index,
                                }
                            )
                    (seed_root / "scores.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
                    checkpoint = training_run / "model_final.ckpt"
                    (seed_root / "provenance.json").write_text(
                        json.dumps(
                            {
                                "group": group,
                                "checkpoint_sha256": hashlib.sha256(
                                    checkpoint.read_bytes()
                                ).hexdigest(),
                                "checkpoint_updated": False,
                            }
                        )
                        + "\n"
                    )

            calibration_output = root / "calibration"
            status = calibrate(
                argparse.Namespace(
                    final_root=str(final),
                    calibration_score_root=str(calibration_scores),
                    output_dir=str(calibration_output),
                    seeds=[0, 1, 2],
                    target_fpr=0.05,
                    safe_end=0.25,
                    time_start=0.50,
                    time_prior=0.5,
                    min_per_class=1,
                )
            )
            self.assertEqual(status["status"], "calibration_complete")
            runtime_bundle = Path(status["runtime_bundle"])
            frozen = json.loads(runtime_bundle.read_text())
            self.assertEqual(frozen["target_fpr"], 0.05)

            evaluation_output = root / "evaluation"
            result = evaluate(
                argparse.Namespace(
                    runtime_bundle=str(runtime_bundle),
                    evaluation_score_root=str(evaluation_scores),
                    output_dir=str(evaluation_output),
                    bootstrap_samples=10,
                    bootstrap_seed=0,
                    test_per_class=1,
                )
            )
            self.assertTrue(result["calibration_ids_disjoint"])
            self.assertFalse(result["thresholds_updated_on_test"])
            self.assertEqual(result["counts"]["primary_balanced"]["rollouts"], 4)


if __name__ == "__main__":
    unittest.main()
