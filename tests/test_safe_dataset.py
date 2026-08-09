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
    select_feature_view,
)
from robocasa.recovery.safe.schema import (
    RLDX1_FEATURE_LAYER,
    RLDX1_FUSED_EXPORT_LAYER,
    RLDX1_OBSERVATION_FEATURE_LAYER,
    SafeRolloutMetadata,
)


def metadata(
    rollout_id,
    task="SeenTask",
    seed=0,
    failed=False,
    *,
    seed_protocol="rollout_index",
    reset_index=None,
):
    return SafeRolloutMetadata(
        rollout_id=rollout_id,
        task_name=task,
        task_instruction="instruction",
        environment_seed=seed,
        seed_protocol=seed_protocol,
        environment_reset_index=reset_index,
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

    def test_action_view_is_identity_for_legacy_dataset(self):
        record = metadata("legacy-action")
        raw = np.ones((3, 2, 4, 8), dtype=np.float32)
        selected, layer = select_feature_view(raw, record, "action")
        np.testing.assert_array_equal(selected, raw)
        self.assertEqual(layer, record.feature_layer)

    def test_dual_stream_rldx_feature_views_are_paired(self):
        record = metadata("rldx-dual")
        record.model_family = "rldx1"
        record.feature_layer = RLDX1_FEATURE_LAYER
        record.feature_mode = "action_observation_context"
        record.feature_schema_version = 2
        record.observation_feature_layer = RLDX1_OBSERVATION_FEATURE_LAYER
        record.observation_context_shape = [3, 6]
        record.observation_components = {
            "backbone_context": [0, 3],
            "state_context": [3, 6],
        }
        record.observation_context_pooling = {
            "backbone": "attention_masked_mean_after_memory",
            "state": "token_mean_after_state_encoder",
        }
        record.validate()
        raw = np.arange(3 * 2 * 4 * 8, dtype=np.float32).reshape(3, 2, 4, 8)
        context = np.arange(3 * 6, dtype=np.float32).reshape(3, 6)
        action, action_layer = select_feature_view(
            raw, record, "action", observation_context=context
        )
        observation, observation_layer = select_feature_view(
            raw, record, "observation_context", observation_context=context
        )
        fused, fused_layer = select_feature_view(
            raw, record, "all", observation_context=context
        )
        np.testing.assert_array_equal(action, raw)
        np.testing.assert_array_equal(observation[:, 0, 0], context)
        np.testing.assert_allclose(fused[:, 0, 0, :8], raw[:, -1].mean(axis=1))
        np.testing.assert_array_equal(fused[:, 0, 0, 8:], context)
        self.assertEqual(action_layer, RLDX1_FEATURE_LAYER)
        self.assertEqual(observation_layer, RLDX1_OBSERVATION_FEATURE_LAYER)
        self.assertEqual(fused_layer, RLDX1_FUSED_EXPORT_LAYER)
        self.assertEqual(observation.shape, (3, 1, 1, 6))
        self.assertEqual(fused.shape, (3, 1, 1, 14))

        record.observation_components["state_context"] = [4, 6]
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            record.validate()

    def test_dual_stream_round_trip_stores_context_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = metadata("rldx-dual")
            record.model_family = "rldx1"
            record.feature_layer = RLDX1_FEATURE_LAYER
            record.feature_mode = "action_observation_context"
            record.feature_schema_version = 2
            record.observation_feature_layer = RLDX1_OBSERVATION_FEATURE_LAYER
            record.observation_components = {
                "backbone_context": [0, 3],
                "state_context": [3, 6],
            }
            record.observation_context_pooling = {
                "backbone": "attention_masked_mean_after_memory",
                "state": "token_mean_after_state_encoder",
            }
            raw = np.ones((3, 2, 4, 8), dtype=np.float32)
            context = np.ones((3, 6), dtype=np.float32)
            save_rollout(tmp, record, raw, observation_context=context)
            loaded_record = load_manifest(tmp)[0]
            with np.load(
                Path(tmp) / loaded_record.tensor_path, allow_pickle=False
            ) as payload:
                self.assertEqual(payload["features"].shape, (3, 2, 4, 8))
                self.assertEqual(payload["observation_context"].shape, (3, 6))

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

    def test_official_reset_indices_are_distinct_episode_groups(self):
        records = [
            metadata(
                f"official-{index}",
                seed=7,
                failed=bool(index % 2),
                seed_protocol="official_openpi",
                reset_index=index,
            )
            for index in range(20)
        ]
        splits = generate_splits(records, [], seed=7)
        assigned = {
            rollout_id
            for split in ("train", "calibration", "seen_test")
            for rollout_id in splits[split]
        }
        self.assertEqual(assigned, {record.rollout_id for record in records})


if __name__ == "__main__":
    unittest.main()
