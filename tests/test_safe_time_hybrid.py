import json
from pathlib import Path
import tempfile
import unittest


class TestSafeTimeHybrid(unittest.TestCase):
    def make_final_root(self, root, *, mismatch=False):
        final_root = root / "final"
        tasks = ("TaskA", "TaskB")
        for model in ("indep", "lstm"):
            for seed in (0, 1, 2):
                run = final_root / f"{model}_seed{seed}"
                run.mkdir(parents=True)
                records = []
                for task_index, task in enumerate(tasks):
                    for split, count in (("train", 8), ("test", 4)):
                        for failed in (False, True):
                            for example in range(count):
                                rollout_id = (
                                    f"{task}-{split}-{int(failed)}-{example}"
                                )
                                length = (
                                    8
                                    if failed
                                    else 3 + ((example + task_index) % 4)
                                )
                                offset = 0.01 * seed + 0.02 * task_index
                                if model == "lstm":
                                    offset += 0.01
                                scores = []
                                for step in range(length):
                                    value = 0.10 + offset + 0.01 * step
                                    if failed:
                                        value += 0.035 * step
                                    scores.append(value)
                                record = {
                                    "rollout_id": rollout_id,
                                    "split": split,
                                    "task_name": task,
                                    "task_id": task_index,
                                    "task_type": "atomic",
                                    "failed": failed,
                                    "scores": scores,
                                    "num_inferences": len(scores),
                                    "task_min_step": min(length, 3),
                                }
                                records.append(record)
                if mismatch and model == "lstm" and seed == 2:
                    records[0]["scores"].append(0.9)
                (run / "scores.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
        return final_root

    def test_causal_time_safe_hybrid_analysis(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is unavailable")

        from robocasa.recovery.safe.analyze_time_safe_hybrid import (
            analyze_time_safe_hybrid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root = self.make_final_root(root)
            output = root / "analysis"
            report = analyze_time_safe_hybrid(
                final_root,
                output,
                validation_per_class=2,
                landmark_fractions=(0.25, 0.5, 1.0),
                formats=("png",),
            )

            self.assertEqual(report["counts"]["unique_rollouts"], 48)
            self.assertEqual(report["counts"]["fit"], 24)
            self.assertEqual(report["counts"]["validation"], 8)
            self.assertEqual(report["counts"]["test"], 16)
            self.assertFalse(report["subtask_safe"])
            self.assertIn("final rollout duration", report["causality"]["forbidden"])
            self.assertTrue(report["causality"]["split_before_prefix_expansion"])

            split = report["split_rollout_ids"]
            self.assertFalse(set(split["fit"]) & set(split["validation"]))
            self.assertFalse(set(split["fit"]) & set(split["test"]))
            self.assertFalse(set(split["validation"]) & set(split["test"]))

            self.assertEqual(len(report["runs"]), 6)
            self.assertEqual(len(report["aggregate"]), 6)
            self.assertEqual(len(report["figures"]), 2)
            self.assertTrue(all(Path(path).is_file() for path in report["figures"]))
            for run in report["runs"]:
                self.assertEqual(set(run["test_metrics"]), {
                    "time_only",
                    "safe_only",
                    "safe_time_task",
                })
                self.assertEqual(
                    run["test_metrics"]["time_only"]["rollouts"], 16
                )
                self.assertGreaterEqual(
                    run["test_metrics"]["time_only"]["roc_auc"], 0.5
                )

            self.assertTrue((output / "analysis.json").is_file())
            self.assertTrue((output / "per_seed_event_metrics.csv").is_file())
            self.assertTrue((output / "landmark_metrics.csv").is_file())
            self.assertTrue((output / "event_predictions.jsonl").is_file())
            self.assertTrue(
                (output / "runtime" / "indep_seed0" / "runtime.json").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "runtime"
                    / "indep_seed0"
                    / "safe_time_task.joblib"
                ).is_file()
            )

    def test_rejects_mismatched_rollout_identity(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is unavailable")

        from robocasa.recovery.safe.analyze_time_safe_hybrid import (
            analyze_time_safe_hybrid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root = self.make_final_root(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, "Rollout identities differ"):
                analyze_time_safe_hybrid(
                    final_root,
                    root / "analysis",
                    validation_per_class=2,
                    formats=(),
                )


if __name__ == "__main__":
    unittest.main()
