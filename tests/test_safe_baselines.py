import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.baselines.detectors import DETECTOR_REGISTRY, build_detector
from robocasa.recovery.safe.baselines.run_baselines import load_rollout_ids, main
from robocasa.recovery.safe.baselines.signals import (
    action_signal_matrix,
    exponential_moving_average,
    temporal_consistency_error,
)
from robocasa.recovery.safe.dataset import save_rollout
from robocasa.recovery.safe.schema import (
    XIAOMI_ROBOTICS_1_FEATURE_LAYER,
    SafeRolloutMetadata,
)

HORIZON = 6
REPLAN = 2
FLOW = 2
DIM = 8
ACTION_DIM = 3


def synthetic_rollout(rng, failed, length):
    """Failures drift in feature space and produce inconsistent chunks."""
    drift = np.linspace(0.0, 3.0 if failed else 0.0, length)[:, None]
    # Real VLA features have a nonzero mean; keep one so cosine distance is meaningful.
    features = 1.0 + rng.normal(size=(length, FLOW, HORIZON, DIM)) * 0.3
    features[..., 0] += drift[:, :, None]
    base = np.cumsum(rng.normal(scale=0.05, size=(length * REPLAN + HORIZON, ACTION_DIM)), axis=0)
    chunks = np.stack([base[t * REPLAN : t * REPLAN + HORIZON] for t in range(length)])
    if failed:
        chunks = chunks + rng.normal(scale=0.5, size=chunks.shape) * np.linspace(0, 1, length)[:, None, None]
    return features.astype(np.float32), chunks.astype(np.float32)


def make_dataset(root, per_task=8, tasks=("TaskA", "TaskB"), seed=0):
    rng = np.random.default_rng(seed)
    ids = {}
    for task in tasks:
        for index in range(per_task * 2):
            failed = index % 2 == 1
            length = 10 if failed else int(rng.integers(5, 9))
            features, chunks = synthetic_rollout(rng, failed, length)
            rollout_id = f"{task}_{index:03d}"
            metadata = SafeRolloutMetadata(
                rollout_id=rollout_id,
                task_name=task,
                task_instruction="do it",
                environment_seed=index,
                policy_id="xr1",
                checkpoint="ckpt",
                failed=failed,
                num_env_steps=length * REPLAN,
                inference_env_steps=[t * REPLAN for t in range(length)],
                valid_sequence_length=length,
                action_horizon=HORIZON,
                replan_steps=REPLAN,
                feature_layer=XIAOMI_ROBOTICS_1_FEATURE_LAYER,
                model_family="xiaomi_robotics_1",
                flow_steps=FLOW,
                timeout_horizon=10 * REPLAN,
            )
            save_rollout(root, metadata, features, policy_action_chunks=chunks)
            ids[rollout_id] = (task, failed, index)
    return ids


class SignalTests(unittest.TestCase):
    def test_consistent_chunks_score_zero(self):
        base = np.arange(20, dtype=float)[:, None] * np.ones((1, 2))
        chunks = np.stack([base[t * 2 : t * 2 + 6] for t in range(5)])
        error = temporal_consistency_error(chunks, 2)
        self.assertEqual(error[0], 0.0)
        np.testing.assert_allclose(error, 0.0, atol=1e-12)

    def test_disagreement_is_detected_and_causal(self):
        rng = np.random.default_rng(0)
        chunks = rng.normal(size=(5, 6, 2))
        full = temporal_consistency_error(chunks, 2)
        prefix = temporal_consistency_error(chunks[:3], 2)
        np.testing.assert_allclose(full[:3], prefix)
        self.assertTrue(np.all(full[1:] > 0))

    def test_no_overlap_rejected(self):
        with self.assertRaises(ValueError):
            temporal_consistency_error(np.zeros((3, 4, 2)), 4)

    def test_ema_and_matrix(self):
        np.testing.assert_allclose(exponential_moving_average([1, 0, 0], 0.5), [1, 0.5, 0.25])
        self.assertEqual(action_signal_matrix(np.ones((4, 6, 3)), 2).shape, (4, 3))


class DetectorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.rollouts = []
        for index in range(24):
            failed = index % 2 == 1
            features, chunks = synthetic_rollout(rng, failed, 10 if failed else 7)
            self.rollouts.append({
                "rollout_id": str(index),
                "task_name": "TaskA" if index < 12 else "TaskB",
                "failed": failed,
                "features": features.mean(axis=(1, 2)),
                "actions": chunks,
                "replan_steps": REPLAN,
            })

    def test_every_detector_ranks_failures_above_chance(self):
        train, test = self.rollouts[:16], self.rollouts[16:]
        for name in sorted(DETECTOR_REGISTRY):
            kwargs = {"epochs": 30} if name in {"actprobe", "hide_and_seek", "rnd"} else {}
            if name in {"mahalanobis", "pca_kmeans", "hide_and_seek", "rnd"}:
                kwargs["pca_components"] = 4
            if name == "pca_kmeans":
                kwargs["clusters"] = 4
            detector = build_detector(name, seed=0, **kwargs).fit(train)
            peaks = [(float(np.max(detector.score(r))), r["failed"]) for r in test]
            failures = [p for p, f in peaks if f]
            successes = [p for p, f in peaks if not f]
            auc = np.mean([[f > s for s in successes] for f in failures])
            self.assertEqual(detector.score(test[0]).shape, (len(test[0]["features"]),), name)
            self.assertGreater(auc, 0.5, name)

    def test_action_detectors_require_actions(self):
        rollout = dict(self.rollouts[0], actions=None)
        with self.assertRaises(ValueError):
            build_detector("action_consistency").score(rollout)


class RunnerTests(unittest.TestCase):
    def test_frozen_protocol_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = make_dataset(root / "data", per_task=12)
            by_split = {"train": [], "calibration": [], "evaluation": []}
            for rollout_id, (task, failed, index) in sorted(ids.items()):
                split = ("train", "train", "calibration", "evaluation")[(index // 2) % 4]
                by_split[split].append(rollout_id)
            bundle = {
                "task_horizons": {"TaskA": 10, "TaskB": 10},
                "time_risk_curves": {
                    task: {"risk": np.linspace(0.1, 0.9, 10).tolist()} for task in ("TaskA", "TaskB")
                },
                "target_fpr": 0.2,
                "safe_eligibility_end_fraction": 0.5,
                "time_eligibility_start_fraction": 0.6,
                "development_rollout_ids": by_split["train"],
                "primary_calibration_rollout_ids": by_split["calibration"],
            }
            (root / "bundle.json").write_text(json.dumps(bundle))
            with (root / "scores.jsonl").open("w") as stream:
                for rollout_id in by_split["train"] + by_split["evaluation"]:
                    split = "train" if rollout_id in by_split["train"] else "test"
                    stream.write(json.dumps({"rollout_id": rollout_id, "split": split}) + "\n")
            bundle["development_rollout_ids"] = by_split["train"] + by_split["evaluation"]
            (root / "bundle.json").write_text(json.dumps(bundle))
            self.assertEqual(load_rollout_ids(root / "scores.jsonl", split="train"), by_split["train"])
            with (root / "eval.jsonl").open("w") as stream:
                for rollout_id in by_split["evaluation"]:
                    stream.write(json.dumps({"rollout_id": rollout_id}) + "\n")
            self.assertEqual(load_rollout_ids(root / "eval.jsonl"), by_split["evaluation"])
            results = main([
                "--dataset-dir", str(root / "data"),
                "--runtime-bundle", str(root / "bundle.json"),
                "--train-ids", str(root / "scores.jsonl"),
                "--evaluation-ids", str(root / "eval.jsonl"),
                "--output-dir", str(root / "out"),
                "--detectors", "action_consistency", "mahalanobis", "actprobe",
                "--seeds", "0", "1",
                "--epochs", "5",
                "--test-per-class", "3",
            ])
            for name in ("action_consistency", "mahalanobis", "actprobe"):
                evaluation = results["detectors"][name]["evaluation"]
                self.assertEqual(set(evaluation), {"baseline_only", "staged_baseline_time", "time_only"})
                self.assertEqual(results["detectors"][name]["evaluation_rollouts"], 12)
                for metrics in evaluation.values():
                    self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)
            self.assertTrue((root / "out" / "summary.md").exists())
            self.assertTrue((root / "out" / "evaluation_predictions.jsonl").exists())

    def test_overlapping_splits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = sorted(make_dataset(root / "data", per_task=2))
            bundle = {
                "task_horizons": {"TaskA": 10, "TaskB": 10},
                "time_risk_curves": {},
                "target_fpr": 0.05,
                "safe_eligibility_end_fraction": 0.25,
                "time_eligibility_start_fraction": 0.5,
                "development_rollout_ids": ids[:4],
                "primary_calibration_rollout_ids": ids[3:6],
            }
            (root / "bundle.json").write_text(json.dumps(bundle))
            (root / "eval.json").write_text(json.dumps(ids[6:]))
            (root / "train.json").write_text(json.dumps(ids[:4]))
            with self.assertRaises(ValueError):
                main([
                    "--dataset-dir", str(root / "data"),
                    "--runtime-bundle", str(root / "bundle.json"),
                    "--train-ids", str(root / "train.json"),
                    "--evaluation-ids", str(root / "eval.json"),
                    "--output-dir", str(root / "out"),
                ])


if __name__ == "__main__":
    unittest.main()
