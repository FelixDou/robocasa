import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from robocasa.recovery.safe.render_score_videos import render_score_videos
from robocasa.recovery.safe.summarize_seen_tasks import summarize
from robocasa.recovery.safe.summarize_seen_cv import summarize_seen_cv
from robocasa.recovery.safe.run_seen_cv_grid import generate_cv_runs, make_inner_folds
from robocasa.recovery.safe.train_seen_tasks import (
    MODEL_DEFAULTS,
    filter_aligned_task_type,
    load_outer_split_ids,
    make_seen_split,
    resolve_task_type_selection,
    resolve_hyperparameters,
    set_task_min_step_from_training,
)


class TestSeenTaskProtocol(unittest.TestCase):
    def test_fixed_split_scales_to_ten_tasks_without_changing_per_task_balance(self):
        rollouts = []
        identity = {}
        for task_id in range(10):
            for success in (0, 1):
                for index in range(10):
                    rollout = SimpleNamespace(
                        task_id=task_id,
                        episode_success=success,
                    )
                    rollout_id = f"task-{task_id}-success-{success}-{index}"
                    rollouts.append(rollout)
                    identity[id(rollout)] = (Path(f"{rollout_id}.pkl"), {"rollout_id": rollout_id})
        train, test, counts = make_seen_split(
            rollouts,
            identity,
            train_per_class=7,
            split_seed=13,
        )
        again = make_seen_split(rollouts, identity, train_per_class=7, split_seed=13)
        self.assertEqual(len(train), 140)
        self.assertEqual(len(test), 60)
        self.assertEqual(
            [identity[id(rollout)][1]["rollout_id"] for rollout in train],
            [identity[id(rollout)][1]["rollout_id"] for rollout in again[0]],
        )
        for task_id in range(10):
            self.assertEqual(counts[task_id]["success"], {"train": 7, "test": 3})
            self.assertEqual(counts[task_id]["failure"], {"train": 7, "test": 3})

    def test_frozen_hyperparameters_match_completed_grid_selection(self):
        self.assertEqual(
            MODEL_DEFAULTS["indep"],
            {
                "horizon_selector": 1.0,
                "diffusion_selector": 0.0,
                "learning_rate": 3e-4,
                "lambda_reg": 1e-3,
            },
        )
        self.assertEqual(MODEL_DEFAULTS["lstm"]["diffusion_selector"], "concat-2")
        self.assertEqual(MODEL_DEFAULTS["lstm"]["learning_rate"], 1e-3)

    def test_task_cutoff_uses_training_lengths_only(self):
        train = [
            SimpleNamespace(task_id=0, hidden_states=np.zeros((8, 1))),
            SimpleNamespace(task_id=0, hidden_states=np.zeros((10, 1))),
        ]
        test = [SimpleNamespace(task_id=0, hidden_states=np.zeros((3, 1)))]
        cutoffs = set_task_min_step_from_training(train, test)
        self.assertEqual(cutoffs, {0: 8})
        self.assertEqual([item.task_min_step for item in train], [8, 8])
        self.assertEqual(test[0].task_min_step, 3)

    def test_three_fold_cv_is_stratified_and_never_contains_outer_test(self):
        rollouts = []
        identity = {}
        for task_id in range(5):
            for success in (0, 1):
                for index in range(10):
                    rollout = SimpleNamespace(task_id=task_id, episode_success=success)
                    rollout_id = f"task-{task_id}-success-{success}-{index}"
                    rollouts.append(rollout)
                    identity[id(rollout)] = (Path(f"{rollout_id}.pkl"), {"rollout_id": rollout_id})
        outer_train, outer_test, _ = make_seen_split(rollouts, identity, train_per_class=7, split_seed=0)
        folds = make_inner_folds(outer_train, identity, num_folds=3, seed=0)
        outer_test_ids = {identity[id(item)][1]["rollout_id"] for item in outer_test}
        validation_union = set()
        self.assertEqual([len(validation) for _, validation in folds], [30, 20, 20])
        for training, validation in folds:
            ids = {identity[id(item)][1]["rollout_id"] for item in training + validation}
            self.assertFalse(ids & outer_test_ids)
            validation_union.update(identity[id(item)][1]["rollout_id"] for item in validation)
        self.assertEqual(len(validation_union), 70)

    def test_task_type_filter_reuses_fixed_outer_split_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_ids = {
                "AtomicA": 0,
                "CompositeA": 1,
                "AtomicB": 2,
                "CompositeB": 3,
            }
            task_types = {
                "AtomicA": "atomic",
                "CompositeA": "composite",
                "AtomicB": "atomic",
                "CompositeB": "composite",
            }
            (root / "conversion_report.json").write_text(
                json.dumps({"task_ids": task_ids, "task_types": task_types})
            )
            rollouts = []
            env_records = []
            train_ids = []
            test_ids = []
            for task_name, task_id in task_ids.items():
                for success in (0, 1):
                    for index in range(2):
                        rollout_id = (
                            f"{task_name}-success-{success}-index-{index}"
                        )
                        rollouts.append(
                            SimpleNamespace(
                                task_id=task_id,
                                episode_success=success,
                            )
                        )
                        env_records.append(
                            (
                                root / f"{rollout_id}.pkl",
                                {
                                    "rollout_id": rollout_id,
                                    "task_id": task_id,
                                    "episode_success": success,
                                },
                            )
                        )
                        (train_ids if index == 0 else test_ids).append(
                            rollout_id
                        )
            split_path = root / "split_manifest.json"
            split_path.write_text(
                json.dumps(
                    {
                        "split_seed": 0,
                        "train": train_ids,
                        "test": test_ids,
                    }
                )
            )
            selection = resolve_task_type_selection(root, "atomic")
            selected, selected_env, identity = filter_aligned_task_type(
                rollouts,
                env_records,
                selection,
            )
            train, test, per_task = make_seen_split(
                selected,
                identity,
                train_per_class=1,
                split_seed=0,
                fixed_split_ids=load_outer_split_ids(split_path),
            )

            self.assertEqual(selection["selected_task_names"], ["AtomicA", "AtomicB"])
            self.assertEqual(len(selected_env), 8)
            self.assertEqual(
                {identity[id(item)][1]["rollout_id"] for item in train},
                set(train_ids) & {
                    env["rollout_id"] for _, env in selected_env
                },
            )
            self.assertEqual(
                {identity[id(item)][1]["rollout_id"] for item in test},
                set(test_ids) & {
                    env["rollout_id"] for _, env in selected_env
                },
            )
            self.assertEqual(set(per_task), {0, 2})

    def test_cv_grid_has_405_fits_per_architecture(self):
        runs = generate_cv_runs("lstm")
        self.assertEqual(len(runs), 405)
        self.assertEqual(len({run.slug for run in runs}), 405)

    def test_summary_aggregates_three_fixed_split_model_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in ("indep", "lstm"):
                for seed, value in ((0, 0.7), (1, 0.8), (2, 0.9)):
                    run = root / f"{model}-seed-{seed}"
                    run.mkdir()
                    metrics = {
                        "model": model,
                        "seed": seed,
                        "task_type_filter": "atomic",
                        "task_types": {
                            f"Task{index}": "atomic" for index in range(10)
                        },
                        "num_tasks": 10,
                        "counts": {
                            "train": 140,
                            "test": 60,
                            "train_successes": 70,
                            "train_failures": 70,
                            "test_successes": 30,
                            "test_failures": 30,
                        },
                        "scalar_metrics": {
                            "falert_early_roc_auc/model_test": value,
                            "falert_early_prc_auc/model_test": value - 0.1,
                        },
                    }
                    (run / "metrics.json").write_text(json.dumps(metrics))
            result = summarize(root)
            self.assertEqual(result["num_completed_runs"], 6)
            self.assertEqual(result["num_tasks"], 10)
            self.assertEqual(result["task_type_filter"], "atomic")
            self.assertEqual(result["split_counts"]["test"], 60)
            self.assertAlmostEqual(result["models"]["indep"]["test_roc_auc_mean"], 0.8)
            self.assertTrue((root / "summary.json").is_file())

    def test_cv_selection_and_final_refit_selector_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in ("indep", "lstm"):
                for config, score in (("low", 0.6), ("high", 0.8)):
                    for fold in (0, 1, 2):
                        run = root / f"{model}-{config}-{fold}"
                        run.mkdir()
                        record = {
                            "status": "complete",
                            "model": model,
                            "task_type_filter": "atomic",
                            "selected_task_names": [
                                f"Task{index}" for index in range(5)
                            ],
                            "horizon_selector": "1.0",
                            "diffusion_selector": "concat-2" if config == "high" else "0.0",
                            "learning_rate": 1e-3 if config == "high" else 1e-4,
                            "lambda_reg": 1e-2,
                            "fold": fold,
                            "selection_value": score + 0.01 * fold,
                            "counts": {
                                "outer_train": {
                                    "rollouts": 140,
                                    "successes": 70,
                                    "failures": 70,
                                },
                                "outer_test_untouched": {
                                    "rollouts": 60,
                                    "successes": 30,
                                    "failures": 30,
                                },
                            },
                        }
                        (run / "metrics.json").write_text(json.dumps(record))
            summary = summarize_seen_cv(root)
            self.assertEqual(summary["num_completed_fits"], 12)
            self.assertEqual(summary["task_type_filter"], "atomic")
            self.assertEqual(summary["outer_train_counts"]["rollouts"], 140)
            self.assertEqual(summary["outer_test_counts"]["rollouts"], 60)
            self.assertEqual(summary["best_by_model"]["lstm"]["diffusion_selector"], "concat-2")
            resolved = resolve_hyperparameters("lstm", root / "cv_selection_summary.json")
            self.assertEqual(resolved["horizon_selector"], 1.0)
            self.assertEqual(resolved["diffusion_selector"], "concat-2")

            summary = json.loads(
                (root / "cv_selection_summary.json").read_text()
            )
            summary["task_type_filter"] = "atomic"
            (root / "cv_selection_summary.json").write_text(
                json.dumps(summary)
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_hyperparameters(
                    "lstm",
                    root / "cv_selection_summary.json",
                    expected_task_type="composite",
                )


class TestScoreVideo(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "OpenCV is not installed locally")
    def test_score_overlay_video_is_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for index in range(8):
                frame = np.full((120, 160, 3), 30 + index * 10, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            score_file = root / "scores.jsonl"
            record = {
                "rollout_id": "abc",
                "split": "test",
                "task_name": "OpenDrawer",
                "failed": True,
                "model": "lstm",
                "seed": 0,
                "scores": [0.1, 0.3, 0.8],
                "video_path": str(source),
                "video_frame_stride": 1,
                "inference_environment_steps": [0, 3, 6],
            }
            score_file.write_text(json.dumps(record) + "\n")
            calibration_file = root / "calibration.json"
            calibration_file.write_text(
                json.dumps(
                    {
                        "alpha": 0.15,
                        "alignment": "extend",
                        "threshold": [0.4, 0.4, 0.4],
                    }
                )
                + "\n"
            )
            outputs = render_score_videos(
                score_file,
                root / "rendered",
                calibration_path=calibration_file,
            )
            self.assertEqual(len(outputs), 1)
            self.assertGreater(outputs[0].stat().st_size, 0)
            capture = cv2.VideoCapture(str(outputs[0]))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 8)
            capture.release()


if __name__ == "__main__":
    unittest.main()
