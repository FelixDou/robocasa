import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.dataset import (
    aggregate_features,
    generate_splits,
    load_manifest,
    load_rollout,
    pad_sequences,
    save_rollout,
)
from robocasa.recovery.safe.schema import SafeRolloutMetadata


def metadata(rollout_id, task="SeenTask", seed=0, failed=False):
    return SafeRolloutMetadata(
        rollout_id=rollout_id,
        task_name=task,
        task_instruction="instruction",
        environment_seed=seed,
        policy_id="pi0-robocasa",
        checkpoint="checkpoint-74999",
        failed=failed,
        num_env_steps=6,
        inference_env_steps=[0, 2, 4],
        valid_sequence_length=3,
        action_horizon=4,
        replan_steps=2,
        feature_shape=[3, 2, 4, 8],
        flow_steps=2,
        termination_reason="timeout" if failed else "success",
        timeout_horizon=6,
    )


class TestSafeDataset(unittest.TestCase):
    def test_npz_round_trip_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            features = np.arange(3 * 2 * 4 * 8, dtype=np.float32).reshape(3, 2, 4, 8)
            save_rollout(tmp, metadata("r0"), features)
            record = load_manifest(tmp)[0]
            loaded, _ = load_rollout(tmp, record)
            np.testing.assert_array_equal(loaded, features)
            with np.load(Path(tmp) / record.tensor_path, allow_pickle=False) as payload:
                self.assertEqual(int(payload["schema_version"]), 1)

    def test_schema_rejects_inference_mismatch(self):
        record = metadata("r0")
        record.inference_env_steps = [0, 2]
        with self.assertRaisesRegex(ValueError, "valid_sequence_length"):
            record.validate()

    def test_aggregation_and_padding(self):
        raw = np.arange(3 * 2 * 4 * 8, dtype=np.float32).reshape(3, 2, 4, 8)
        self.assertEqual(aggregate_features(raw, "mean").shape, (3, 8))
        self.assertEqual(aggregate_features(raw, "last").shape, (3, 8))
        batch, mask = pad_sequences([np.ones((2, 8)), np.ones((3, 8))])
        self.assertEqual(batch.shape, (2, 3, 8))
        self.assertEqual(mask.tolist(), [[True, True, False], [True, True, True]])

    def test_splits_deterministic_and_leakage_free(self):
        records = []
        for task in ("SeenA", "SeenB", "Unseen"):
            for seed in range(5):
                records.append(metadata(f"{task}-{seed}", task, seed, failed=bool(seed % 2)))
        one = generate_splits(records, ["Unseen"], seed=7)
        two = generate_splits(records, ["Unseen"], seed=7)
        self.assertEqual(one, two)
        self.assertTrue(all(x.startswith("Unseen-") for x in one["unseen_test"]))
        self.assertFalse(any(x.startswith("Unseen-") for x in one["train"]))
        split_by_id = {
            rollout_id: split
            for split in ("train", "calibration", "seen_test", "unseen_test")
            for rollout_id in one[split]
        }
        self.assertEqual(len(split_by_id), len(records))


if __name__ == "__main__":
    unittest.main()
