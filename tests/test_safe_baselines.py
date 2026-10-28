import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.baselines.detectors import DETECTOR_REGISTRY, build_detector
from robocasa.recovery.safe.baselines.run_baselines import (
    index_datasets,
    load_baseline_rollout,
    load_rollout_ids,
    main,
)
from robocasa.recovery.safe.baselines.signals import (
    action_signal_matrix,
    exponential_moving_average,
    temporal_consistency_error,
)
from robocasa.recovery.safe.dataset import aggregate_features, load_rollout, save_rollout
from robocasa.recovery.safe.schema import (
    XIAOMI_ROBOTICS_1_FEATURE_LAYER,
    SafeRolloutMetadata,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
TORCH_DETECTORS = {"actprobe", "rnd"} | {name for name in DETECTOR_REGISTRY if name.startswith("hide_and_seek")}
HORIZON = 6
REPLAN = 2
FLOW = 2
DIM = 8
ACTION_DIM = 3


def synthetic_rollout(rng, failed, length, replan=REPLAN):
    """Failures drift in feature space and produce inconsistent chunks."""
    drift = np.linspace(0.0, 3.0 if failed else 0.0, length)[:, None]
    # Real VLA features have a nonzero mean; keep one so cosine distance is meaningful.
    features = 1.0 + rng.normal(size=(length, FLOW, HORIZON, DIM)) * 0.3
    features[..., 0] += drift[:, :, None]
    base = np.cumsum(rng.normal(scale=0.05, size=(length * replan + HORIZON, ACTION_DIM)), axis=0)
    chunks = np.stack([base[t * replan : t * replan + HORIZON] for t in range(length)])
    if failed:
        chunks = chunks + rng.normal(scale=0.5, size=chunks.shape) * np.linspace(0, 1, length)[:, None, None]
    return features.astype(np.float32), chunks.astype(np.float32)


def make_dataset(root, per_task=8, tasks=("TaskA", "TaskB"), seed=0, replan=REPLAN):
    rng = np.random.default_rng(seed)
    ids = {}
    for task in tasks:
        for index in range(per_task * 2):
            failed = index % 2 == 1
            length = 10 if failed else int(rng.integers(5, 9))
            features, chunks = synthetic_rollout(rng, failed, length, replan)
            rollout_id = f"{task}_{index:03d}"
            metadata = SafeRolloutMetadata(
                rollout_id=rollout_id,
                task_name=task,
                task_instruction="do it",
                environment_seed=index,
                policy_id="xr1",
                checkpoint="ckpt",
                failed=failed,
                num_env_steps=length * replan,
                inference_env_steps=[t * replan for t in range(length)],
                valid_sequence_length=length,
                action_horizon=HORIZON,
                replan_steps=replan,
                feature_layer=XIAOMI_ROBOTICS_1_FEATURE_LAYER,
                model_family="xiaomi_robotics_1",
                flow_steps=FLOW,
                timeout_horizon=10 * replan,
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

    def test_matrix_without_overlap_keeps_chunk_magnitude_only(self):
        # XR-1 layout: every chunk is executed in full, so chunks never overlap.
        chunks = np.random.default_rng(0).normal(size=(4, 6, 3))
        signals = action_signal_matrix(chunks, 6)
        self.assertEqual(signals.shape, (4, 1))
        np.testing.assert_allclose(signals[:, 0], np.linalg.norm(chunks, axis=-1).mean(axis=-1))


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
            torch_model = name in {"actprobe", "rnd"} or name.startswith("hide_and_seek")
            kwargs = {"epochs": 30} if torch_model else {}
            if name in {"mahalanobis", "pca_kmeans", "rnd"} or name.startswith("hide_and_seek"):
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

    def test_actprobe_runs_without_chunk_overlap(self):
        rng = np.random.default_rng(2)
        rollouts = []
        for index in range(24):
            failed = index % 2 == 1
            features, chunks = synthetic_rollout(rng, failed, 10 if failed else 7, replan=HORIZON)
            rollouts.append({
                "rollout_id": str(index), "task_name": "TaskA", "failed": failed,
                "features": features.mean(axis=(1, 2)), "actions": chunks,
                "replan_steps": HORIZON,
            })
        detector = build_detector("actprobe", seed=0, epochs=5).fit(rollouts)
        self.assertEqual(detector.config()["action_signals"], ["chunk_magnitude"])
        self.assertEqual(detector.score(rollouts[0]).shape, (7,))
        reason = DETECTOR_REGISTRY["action_consistency"].unsupported_reason(rollouts)
        self.assertIn("share no step", reason)
        self.assertIsNone(DETECTOR_REGISTRY["action_consistency"].unsupported_reason(self.rollouts))

    def test_zero_control_sees_only_elapsed_time(self):
        train, test = self.rollouts[:16], self.rollouts[16:]
        detector = build_detector("hide_and_seek_zero", seed=0, epochs=5).fit(train)
        a, b = test[0], dict(test[0], rollout_id="other", features=test[0]["features"] + 5.0)
        np.testing.assert_array_equal(detector.score(a), detector.score(b))
        self.assertEqual(detector.config()["control"], "zero_inputs")

    def test_task_shuffle_control_ignores_the_scored_rollout(self):
        train, test = self.rollouts[:16], self.rollouts[16:]
        detector = build_detector("hide_and_seek_task_shuffle", seed=0, epochs=5,
                                  pca_components=4).fit(train)
        rollout = test[0]
        changed = dict(rollout, features=rollout["features"] + 100.0)
        np.testing.assert_array_equal(detector.score(rollout), detector.score(changed))
        np.testing.assert_array_equal(detector._inputs(rollout), detector._inputs(rollout))
        # Each input row is a training row of the same task at the same step,
        # never one of the scored rollout's own rows.
        for own in (train[0], train[1], train[12], train[13]):
            ids, rows, lengths = detector.pool_[own["task_name"]]
            inputs = detector._inputs(own)
            mine = ids.index(own["rollout_id"])
            for step, row in enumerate(inputs):
                index = min(step, int(lengths.max()) - 1)
                matches = [d for d in range(len(ids)) if lengths[d] > index
                           and np.array_equal(rows[d][index], row)]
                self.assertTrue(matches)
                self.assertNotIn(mine, matches)
        with self.assertRaises(ValueError):
            detector.score(dict(rollout, task_name="UnseenTask"))

    def test_first_inference_control_ignores_later_features(self):
        train, test = self.rollouts[:16], self.rollouts[16:]
        detector = build_detector("hide_and_seek_first", seed=0, epochs=5, pca_components=4).fit(train)
        rollout = test[0]
        later = rollout["features"].copy()
        later[1:] += 50.0
        np.testing.assert_array_equal(detector.score(rollout), detector.score(dict(rollout, features=later)))
        first = rollout["features"].copy()
        first[0] += 50.0
        self.assertFalse(np.array_equal(detector.score(rollout), detector.score(dict(rollout, features=first))))
        self.assertEqual(detector.config()["control"], "first_inference_repeated")

    def test_action_detectors_require_actions(self):
        rollout = dict(self.rollouts[0], actions=None)
        with self.assertRaises(ValueError):
            build_detector("action_consistency").score(rollout)


class LoaderTests(unittest.TestCase):
    def test_loaded_features_own_their_memory(self):
        # "first" and "last" slice the raw (T, flow, horizon, D) tensor; a view would keep
        # the whole tensor alive for every loaded rollout.
        with tempfile.TemporaryDirectory() as directory:
            make_dataset(Path(directory), per_task=1)
            for directory_, metadata in index_datasets([directory]).values():
                raw = load_rollout(directory_, metadata)[0]
                for aggregation in ("first", "last", "mean", "flow_last_horizon_mean"):
                    features = load_baseline_rollout(directory_, metadata, aggregation)["features"]
                    self.assertTrue(features.flags.owndata and features.flags.c_contiguous, aggregation)
                    self.assertEqual(features.dtype, np.float32)
                    np.testing.assert_array_equal(features, aggregate_features(raw, aggregation))

    def check_scores_match_view_features(self, names):
        # Copying the slice must not change any score: fit and score each detector on the
        # copied features and on the old view-of-the-raw-tensor features.
        with tempfile.TemporaryDirectory() as directory:
            make_dataset(Path(directory), per_task=6)
            index = index_datasets([directory])
            copied = [load_baseline_rollout(*index[rid], "last") for rid in sorted(index)]
            viewed = [dict(r, features=aggregate_features(load_rollout(*index[r["rollout_id"]])[0], "last"))
                      for r in copied]
            self.assertIsNotNone(viewed[0]["features"].base)
            self.assertFalse(viewed[0]["features"].flags.c_contiguous)
            # Rollouts alternate success/failure, so split in pairs: both splits get both classes.
            train = [i for i in range(len(copied)) if i % 4 < 2]
            test = [i for i in range(len(copied)) if i % 4 >= 2]
            for split in (train, test):
                self.assertEqual({copied[i]["failed"] for i in split}, {False, True})
            for name in names:
                kwargs = {"epochs": 5} if name in TORCH_DETECTORS else {}
                if name in {"mahalanobis", "pca_kmeans", "rnd"} or name.startswith("hide_and_seek"):
                    kwargs["pca_components"] = 4
                if name == "pca_kmeans":
                    kwargs["clusters"] = 4
                a = build_detector(name, seed=0, **kwargs).fit([copied[i] for i in train])
                b = build_detector(name, seed=0, **kwargs).fit([viewed[i] for i in train])
                for i in test:
                    np.testing.assert_array_equal(a.score(copied[i]), b.score(viewed[i]), err_msg=name)

    def test_copied_features_give_identical_scores(self):
        self.check_scores_match_view_features(sorted(set(DETECTOR_REGISTRY) - TORCH_DETECTORS))

    @unittest.skipUnless(HAS_TORCH, "Torch detectors need PyTorch")
    def test_copied_features_give_identical_scores_torch(self):
        self.check_scores_match_view_features(sorted(TORCH_DETECTORS))


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
            calibration_rows = [
                json.loads(line)
                for line in (root / "out" / "calibration_predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {(row["detector"], row["rollout_id"]) for row in calibration_rows},
                {(name, rid) for name in ("action_consistency", "mahalanobis", "actprobe")
                 for rid in by_split["calibration"]},
            )

    def test_runner_skips_action_consistency_without_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = make_dataset(root / "data", per_task=12, replan=HORIZON)
            by_split = {"train": [], "calibration": [], "evaluation": []}
            for rollout_id, (task, failed, index) in sorted(ids.items()):
                split = ("train", "train", "calibration", "evaluation")[(index // 2) % 4]
                by_split[split].append(rollout_id)
            horizon = 10  # inferences; failures run 10 inferences
            bundle = {
                "task_horizons": {"TaskA": horizon, "TaskB": horizon},
                "time_risk_curves": {
                    task: {"risk": np.linspace(0.1, 0.9, horizon).tolist()}
                    for task in ("TaskA", "TaskB")
                },
                "target_fpr": 0.2,
                "safe_eligibility_end_fraction": 0.5,
                "time_eligibility_start_fraction": 0.6,
                "development_rollout_ids": by_split["train"],
                "primary_calibration_rollout_ids": by_split["calibration"],
            }
            (root / "bundle.json").write_text(json.dumps(bundle))
            (root / "train.json").write_text(json.dumps(by_split["train"]))
            (root / "eval.json").write_text(json.dumps(by_split["evaluation"]))
            results = main([
                "--dataset-dir", str(root / "data"),
                "--runtime-bundle", str(root / "bundle.json"),
                "--train-ids", str(root / "train.json"),
                "--evaluation-ids", str(root / "eval.json"),
                "--output-dir", str(root / "out"),
                "--detectors", "action_consistency", "actprobe",
                "--seeds", "0",
                "--epochs", "5",
                "--test-per-class", "3",
            ])
            self.assertEqual(set(results["skipped"]), {"action_consistency"})
            self.assertEqual(set(results["detectors"]), {"actprobe"})
            self.assertEqual(
                results["detectors"]["actprobe"]["configs"]["0"]["action_signals"],
                ["chunk_magnitude"],
            )
            self.assertIn("Skipped `action_consistency`", (root / "out" / "summary.md").read_text())

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
