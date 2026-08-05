import json
from pathlib import Path
import tempfile
import unittest

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.calibrate_seen_tasks import (  # noqa: E402
    normalize_records,
    run_seen_calibration,
    validate_seed_alignment,
)


def score_record(
    rollout_id,
    *,
    split,
    task_name,
    failed,
    offset,
    index,
    seed,
):
    base = 0.60 if failed else 0.10
    jitter = seed * 0.002
    scores = [
        offset + base + 0.01 * index + jitter,
        offset + base + 0.05 + 0.01 * index + jitter,
        offset + base + 0.10 + 0.01 * index + jitter,
    ]
    return {
        "rollout_id": rollout_id,
        "split": split,
        "task_name": task_name,
        "failed": failed,
        "model": "indep",
        "seed": seed,
        "scores": scores,
        "task_min_step": 3,
        "num_inferences": 3,
        "inference_environment_steps": [0, 8, 16],
        "video_path": f"/videos/{rollout_id}.mp4",
        "video_frame_stride": 2,
    }


class TestSeenTaskCalibration(unittest.TestCase):
    def test_normalization_preserves_full_trajectory_for_visualization(self):
        record = {
            "rollout_id": "long",
            "split": "test",
            "task_name": "TaskA",
            "failed": True,
            "scores": [1.0, 2.0, 3.0, 4.0],
            "task_min_step": 2,
            "inference_environment_steps": [0, 8, 16, 24],
        }
        normalized = normalize_records(
            [record],
            {
                "TaskA": {
                    "location": 1.0,
                    "scale": 2.0,
                }
            },
            {
                "calibration_success_ids": [],
                "calibration_failure_ids_excluded": [],
                "evaluation_ids": ["long"],
            },
        )[0]
        self.assertEqual(normalized["scores"], [0.0, 0.5])
        self.assertEqual(normalized["inference_environment_steps"], [0, 8])
        self.assertEqual(normalized["full_scores"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(
            normalized["full_inference_environment_steps"],
            [0, 8, 16, 24],
        )
        self.assertEqual(normalized["full_num_inferences"], 4)

    def make_final_root(self, root):
        final_root = root / "final"
        for seed in (0, 1, 2):
            records = []
            for task_name, offset in (("TaskA", 0.0), ("TaskB", 10.0)):
                for failed in (False, True):
                    for index in range(4):
                        rollout_id = (
                            f"{task_name}-train-failed-{int(failed)}-{index}"
                        )
                        records.append(
                            score_record(
                                rollout_id,
                                split="train",
                                task_name=task_name,
                                failed=failed,
                                offset=offset,
                                index=index,
                                seed=seed,
                            )
                        )
                for index in range(5):
                    rollout_id = f"{task_name}-test-success-{index}"
                    records.append(
                        score_record(
                            rollout_id,
                            split="test",
                            task_name=task_name,
                            failed=False,
                            offset=offset,
                            index=index,
                            seed=seed,
                        )
                    )
                for index in range(4):
                    rollout_id = f"{task_name}-test-failure-{index}"
                    records.append(
                        score_record(
                            rollout_id,
                            split="test",
                            task_name=task_name,
                            failed=True,
                            offset=offset,
                            index=index,
                            seed=seed,
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
                            "TaskA": "atomic",
                            "TaskB": "composite",
                        }
                    }
                )
            )
        return final_root

    def test_calibration_is_disjoint_task_normalized_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root = self.make_final_root(root)
            output = root / "calibration"
            summary = run_seen_calibration(
                final_root,
                output,
                calibration_successes_per_task=2,
                alphas=(0.10, 0.15),
                selected_alpha=0.15,
                make_plots=False,
            )

            manifest = json.loads((output / "split_manifest.json").read_text())
            self.assertEqual(
                manifest["counts"],
                {
                    "train": 16,
                    "calibration_successes": 4,
                    "calibration_reference_successes": 1,
                    "calibration_nonconformity_successes": 3,
                    "evaluation": 14,
                    "evaluation_successes": 6,
                    "evaluation_failures": 8,
                },
            )
            self.assertFalse(
                set(manifest["train_ids"])
                & set(manifest["calibration_success_ids"])
            )
            self.assertFalse(
                set(manifest["calibration_success_ids"])
                & set(manifest["evaluation_ids"])
            )
            self.assertEqual(
                set(manifest["calibration_reference_ids"])
                | set(manifest["calibration_nonconformity_ids"]),
                set(manifest["calibration_success_ids"]),
            )
            for task in ("TaskA", "TaskB"):
                self.assertEqual(
                    manifest["per_task"][task]["calibration_successes"], 2
                )
                self.assertEqual(
                    manifest["per_task"][task]["evaluation_successes"], 3
                )
                self.assertEqual(
                    manifest["per_task"][task]["evaluation_failures"], 4
                )

            normalization = json.loads(
                (output / "indep_seed0" / "task_normalization.json").read_text()
            )
            self.assertAlmostEqual(
                normalization["TaskA"]["location"], 0.465, places=6
            )
            self.assertAlmostEqual(
                normalization["TaskB"]["location"], 10.465, places=6
            )
            self.assertEqual(
                normalization["TaskA"]["num_training_rollouts"], 8
            )

            self.assertEqual(summary["split_counts"]["evaluation"], 14)
            self.assertEqual(summary["selected_alpha"], 0.15)
            self.assertGreater(
                summary["aggregate"]["by_alpha"]["0.15"]["roc_auc_mean"],
                0.9,
            )
            self.assertTrue(
                (
                    output
                    / "indep_seed0"
                    / "alpha_0p15"
                    / "calibration.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    output
                    / "indep_seed0"
                    / "alpha_0p15"
                    / "metrics.json"
                ).is_file()
            )
            calibration = json.loads(
                (
                    output
                    / "indep_seed0"
                    / "alpha_0p15"
                    / "calibration.json"
                ).read_text()
            )
            self.assertEqual(calibration["alignment"], "extend")
            self.assertEqual(calibration["reference_size"], 1)
            self.assertEqual(calibration["calibration_size"], 3)

            atomic_output = root / "atomic_calibration"
            atomic = run_seen_calibration(
                final_root,
                atomic_output,
                task_type="atomic",
                calibration_successes_per_task=2,
                reference_fraction=0.5,
                alphas=(0.15,),
                selected_alpha=0.15,
                make_plots=False,
            )
            atomic_manifest = json.loads(
                (atomic_output / "split_manifest.json").read_text()
            )
            self.assertEqual(atomic["task_type_filter"], "atomic")
            self.assertEqual(atomic["selected_task_names"], ["TaskA"])
            self.assertEqual(
                atomic_manifest["counts"],
                {
                    "train": 8,
                    "calibration_successes": 2,
                    "calibration_reference_successes": 1,
                    "calibration_nonconformity_successes": 1,
                    "evaluation": 7,
                    "evaluation_successes": 3,
                    "evaluation_failures": 4,
                },
            )
            self.assertEqual(
                atomic_manifest["calibration_success_ids"],
                manifest["per_task"]["TaskA"]["calibration_success_ids"],
            )

    def test_seed_alignment_rejects_changed_rollout_identity(self):
        records = [
            {
                "rollout_id": "one",
                "split": "train",
                "task_name": "TaskA",
                "failed": False,
                "task_min_step": 2,
            },
            {
                "rollout_id": "two",
                "split": "test",
                "task_name": "TaskA",
                "failed": True,
                "task_min_step": 2,
            },
        ]
        changed = [dict(record) for record in records]
        changed[1]["task_name"] = "TaskB"
        with self.assertRaisesRegex(ValueError, "same rollout identities"):
            validate_seed_alignment({0: records, 1: changed})

    def test_calibration_plots_are_written(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root = self.make_final_root(root)
            output = root / "calibration_plots"
            summary = run_seen_calibration(
                final_root,
                output,
                seeds=(0,),
                calibration_successes_per_task=2,
                alphas=(0.15,),
                selected_alpha=0.15,
                make_plots=True,
            )

            self.assertEqual(
                {Path(path).name for path in summary["plots"]},
                {
                    "conformal_tradeoff.png",
                    "per_task_balanced_accuracy.png",
                },
            )
            for path in summary["plots"]:
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
