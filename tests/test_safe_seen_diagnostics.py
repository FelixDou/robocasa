import json
from pathlib import Path
import tempfile
import unittest


class TestSeenDiagnostics(unittest.TestCase):
    def make_inputs(self, root):
        final_root = root / "final"
        calibration_root = root / "calibration"
        tasks = ("TaskA", "TaskB")

        for model in ("indep", "lstm"):
            model_calibration = calibration_root / model
            model_calibration.mkdir(parents=True)
            (model_calibration / "summary.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "normalization": "training_task_early_max_z",
                    }
                )
            )
            for seed in (0, 1, 2):
                run = final_root / f"{model}_seed{seed}"
                normalized_run = model_calibration / f"{model}_seed{seed}"
                run.mkdir(parents=True)
                normalized_run.mkdir(parents=True)
                raw_records = []
                normalized_records = []
                for task_index, task in enumerate(tasks):
                    for split in ("train", "test"):
                        for failed in (False, True):
                            for example in range(2):
                                rollout_id = (
                                    f"{task}-{split}-{int(failed)}-{example}"
                                )
                                length = 4 if failed else 3
                                offset = 0.02 * seed + 0.03 * task_index
                                terminal = 0.75 if failed else 0.30
                                if model == "lstm":
                                    terminal += 0.03
                                scores = [
                                    0.10 + offset,
                                    0.18 + offset,
                                    terminal + offset,
                                ]
                                if length == 4:
                                    scores.append(terminal + 0.05 + offset)
                                raw = {
                                    "rollout_id": rollout_id,
                                    "split": split,
                                    "task_name": task,
                                    "failed": failed,
                                    "scores": scores,
                                    "num_inferences": length,
                                    "task_min_step": 3,
                                }
                                normalized_scores = [
                                    (value - 0.20) / 0.10 for value in scores[:3]
                                ]
                                normalized = {
                                    **raw,
                                    "original_split": split,
                                    "split": "train" if split == "train" else "evaluation",
                                    "scores": normalized_scores,
                                    "full_scores": [
                                        (value - 0.20) / 0.10 for value in scores
                                    ],
                                    "num_inferences": 3,
                                    "full_num_inferences": length,
                                }
                                raw_records.append(raw)
                                normalized_records.append(normalized)
                (run / "scores.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in raw_records)
                )
                (normalized_run / "normalized_scores.jsonl").write_text(
                    "".join(
                        json.dumps(record) + "\n"
                        for record in normalized_records
                    )
                )
        return final_root, calibration_root

    def test_raw_normalized_and_duration_diagnostics(self):
        try:
            import matplotlib  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("diagnostic plotting dependencies are unavailable")

        from robocasa.recovery.safe.analyze_seen_diagnostics import (
            analyze_seen_diagnostics,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root, calibration_root = self.make_inputs(root)
            output = root / "diagnostics"
            report = analyze_seen_diagnostics(
                final_root,
                calibration_root,
                output,
                formats=("png",),
            )

            self.assertEqual(
                report["counts"],
                {
                    "unique_rollouts": 16,
                    "outer_test_rollouts": 8,
                    "outer_test_successes": 4,
                    "outer_test_failures": 4,
                    "tasks": 2,
                },
            )
            self.assertEqual(len(report["per_task_summary"]), 8)
            self.assertEqual(len(report["duration_conditional_summary"]), 4)
            self.assertEqual(len(report["figures"]), 5)
            self.assertTrue(
                all(Path(path).is_file() for path in report["figures"])
            )
            for name in (
                "analysis.json",
                "per_rollout_scores.csv",
                "per_task_metrics.csv",
                "per_task_summary.csv",
                "duration_conditional_metrics.csv",
                "duration_conditional_summary.csv",
            ):
                self.assertTrue((output / name).is_file(), name)

            raw_task = next(
                row
                for row in report["per_task_summary"]
                if row["model"] == "indep"
                and row["score_variant"] == "raw"
                and row["task_name"] == "TaskA"
            )
            self.assertEqual(raw_task["score_roc_auc_mean"], 1.0)
            self.assertEqual(raw_task["duration_roc_auc_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
