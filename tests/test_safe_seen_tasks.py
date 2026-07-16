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
from robocasa.recovery.safe.train_seen_tasks import MODEL_DEFAULTS, make_seen_split


class TestSeenTaskProtocol(unittest.TestCase):
    def test_fixed_split_is_seven_plus_seven_train_and_three_plus_three_test(self):
        rollouts = []
        identity = {}
        for task_id in range(5):
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
        self.assertEqual(len(train), 70)
        self.assertEqual(len(test), 30)
        self.assertEqual(
            [identity[id(rollout)][1]["rollout_id"] for rollout in train],
            [identity[id(rollout)][1]["rollout_id"] for rollout in again[0]],
        )
        for task_id in range(5):
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
                        "scalar_metrics": {
                            "falert_early_roc_auc/model_test": value,
                            "falert_early_prc_auc/model_test": value - 0.1,
                        },
                    }
                    (run / "metrics.json").write_text(json.dumps(metrics))
            result = summarize(root)
            self.assertEqual(result["num_completed_runs"], 6)
            self.assertAlmostEqual(result["models"]["indep"]["test_roc_auc_mean"], 0.8)
            self.assertTrue((root / "summary.json").is_file())


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
            outputs = render_score_videos(score_file, root / "rendered")
            self.assertEqual(len(outputs), 1)
            self.assertGreater(outputs[0].stat().st_size, 0)
            capture = cv2.VideoCapture(str(outputs[0]))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 8)
            capture.release()


if __name__ == "__main__":
    unittest.main()
