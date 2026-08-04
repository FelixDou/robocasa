import json
from pathlib import Path
import tempfile
import unittest

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.analyze_subtask_safe_results import (  # noqa: E402
    analyze_subtask_results,
)
from robocasa.recovery.safe.calibrate_seen_tasks import run_seen_calibration  # noqa: E402


def segment_record(parent, task, failed, split, seed, index):
    base = 0.75 if failed else 0.10
    length = 4 + (index % 2)
    scores = [base + seed * 0.002 + step * 0.03 for step in range(length)]
    entry = 10 + index * 20
    return {
        "rollout_id": f"{parent}:{task}",
        "parent_rollout_id": parent,
        "parent_task_name": "CompositeTask",
        "parent_rollout_failed": parent.endswith("F"),
        "subtask_id": task,
        "subtask_index": 0 if task == "SubtaskA" else 1,
        "subtask_instruction": f"Perform {task}",
        "subtask_safe_segment": {
            "entry_environment_step": entry,
            "end_environment_step": entry + length * 8,
        },
        "split": split,
        "task_name": f"CompositeTask::{task}",
        "task_type": "composite",
        "failed": failed,
        "model": "indep",
        "seed": seed,
        "scores": scores,
        "task_min_step": length,
        "num_inferences": length,
        "inference_environment_steps": [entry + step * 8 for step in range(length)],
        "video_path": f"/videos/{parent}.mp4",
        "video_frame_stride": 2,
    }


class TestSubtaskSafeEvaluation(unittest.TestCase):
    def make_final_root(self, root):
        final_root = root / "final"
        for seed in (0, 1, 2):
            records = []
            for split, count in (("train", 8), ("test", 10)):
                for index in range(count):
                    parent_failed = index % 2 == 1
                    parent = f"{split}-parent-{index}-{'F' if parent_failed else 'S'}"
                    records.append(
                        segment_record(
                            parent,
                            "SubtaskA",
                            False,
                            split,
                            seed,
                            index,
                        )
                    )
                    records.append(
                        segment_record(
                            parent,
                            "SubtaskB",
                            parent_failed,
                            split,
                            seed,
                            index,
                        )
                    )
            run = final_root / f"indep_seed{seed}"
            run.mkdir(parents=True)
            (run / "scores.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            (run / "metrics.json").write_text(
                json.dumps(
                    {
                        "task_types": {
                            "CompositeTask::SubtaskA": "composite",
                            "CompositeTask::SubtaskB": "composite",
                        }
                    }
                )
            )
        return final_root

    def test_parent_grouped_calibration_and_elapsed_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root = self.make_final_root(root)
            output = root / "calibration"
            summary = run_seen_calibration(
                final_root,
                output,
                split_unit="parent_rollout",
                calibration_parent_fraction=0.4,
                reference_fraction=0.5,
                alphas=(0.15,),
                selected_alpha=0.15,
                make_plots=False,
            )
            manifest = json.loads((output / "split_manifest.json").read_text())
            self.assertEqual(manifest["split_unit"], "parent_rollout")
            self.assertFalse(
                set(manifest["calibration_parent_ids"])
                & set(manifest["evaluation_parent_ids"])
            )
            self.assertEqual(
                set(manifest["calibration_reference_parent_ids"])
                | set(manifest["calibration_nonconformity_parent_ids"]),
                set(manifest["calibration_parent_ids"]),
            )
            self.assertGreater(
                manifest["counts"]["calibration_failures_excluded"], 0
            )
            self.assertIn("elapsed_time", summary["aggregate"]["methods"])
            self.assertTrue(
                (
                    output
                    / "indep_seed0"
                    / "alpha_0p15"
                    / "elapsed_time_calibration.json"
                ).is_file()
            )
            self.assertTrue((output / "detection_events.csv").is_file())

    def test_parent_bootstrap_and_prefix_analysis(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root = self.make_final_root(root)
            output = root / "analysis"
            summary = analyze_subtask_results(
                final_root,
                output,
                models=("indep",),
                bootstrap_replicates=50,
                prefix_horizons=(1, 2, 4),
                formats=("png",),
            )
            self.assertEqual(summary["bootstrap_replicates"], 50)
            self.assertIn("indep", summary["parent_bootstrap"])
            self.assertEqual(len(summary["per_subtask"]), 2)
            one_class = next(
                row
                for row in summary["per_subtask"]
                if row["task_name"].endswith("SubtaskA")
            )
            self.assertIsNone(one_class["roc_auc_mean"])
            self.assertEqual(one_class["failures"], 0)
            self.assertTrue((output / "causal_prefix_metrics.csv").is_file())
            self.assertTrue((output / "per_subtask_roc.png").is_file())


if __name__ == "__main__":
    unittest.main()
