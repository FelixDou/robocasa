import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages


install_lightweight_robocasa_packages()

from robocasa.recovery.safe.xr1_best_of_k import (  # noqa: E402
    FrozenXr1SafeEnsemble,
    MODEL_LOADER_PROTOCOL,
    _official_indep_layer_spec,
    normalize_seed_scores,
    select_candidate_features,
    stable_centered_sigmoid_scores,
)


class TestXr1BestOfKScoring(unittest.TestCase):
    @staticmethod
    def _cfg(**overrides):
        values = {
            "n_history_steps": 1,
            "hidden_dim": 16,
            "n_layers": 3,
            "final_act_layer": "none",
            "cumsum": False,
            "rmean": False,
        }
        values.update(overrides)
        return SimpleNamespace(model=SimpleNamespace(**values))

    def test_official_indep_layer_spec_matches_pinned_projector_layout(self):
        self.assertEqual(
            _official_indep_layer_spec(self._cfg(), 1024),
            (
                ("linear", 1024, 16),
                ("relu",),
                ("linear", 16, 16),
                ("relu",),
                ("linear", 16, 1),
            ),
        )
        self.assertEqual(
            _official_indep_layer_spec(
                self._cfg(n_layers=1, final_act_layer="sigmoid"), 8
            ),
            (("linear", 8, 1), ("sigmoid",)),
        )

    def test_official_indep_compat_loader_rejects_history_or_activation_drift(self):
        with self.assertRaisesRegex(ValueError, "n_history_steps=1"):
            _official_indep_layer_spec(self._cfg(n_history_steps=2), 1024)
        with self.assertRaisesRegex(ValueError, "final activation"):
            _official_indep_layer_spec(
                self._cfg(final_act_layer="unsupported"), 1024
            )

    def test_feature_selection_uses_last_horizon_and_diffusion_tokens(self):
        features = np.arange(3 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5)
        selected = select_candidate_features(features)
        np.testing.assert_array_equal(selected, features[:, -1, -1, :])
        self.assertEqual(selected.dtype, np.float32)

    def test_per_task_seed_normalization_is_frozen(self):
        normalized = normalize_seed_scores(
            np.array([8.0, 12.0]),
            task_name="CloseBlenderLid",
            seed=0,
            task_normalizations={
                "0": {"CloseBlenderLid": {"location": 10.0, "scale": 2.0}}
            },
        )
        np.testing.assert_allclose(normalized, [-1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "lacks normalization"):
            normalize_seed_scores(
                np.array([1.0]),
                task_name="UnknownTask",
                seed=0,
                task_normalizations={},
            )

    def test_stable_centered_sigmoid_scores_avoid_probability_saturation(self):
        logits = np.array([20.0, 21.0, 1000.0])
        centered = stable_centered_sigmoid_scores(
            logits,
            task_name="CloseBlenderLid",
            seed=0,
            task_normalizations={
                "0": {"CloseBlenderLid": {"location": 0.25, "scale": 2.0}}
            },
        )
        self.assertTrue(np.all(np.isfinite(centered)))
        self.assertLess(centered[0], centered[1])
        self.assertLess(centered[1], centered[2])
        self.assertLess(centered[0], 0.0)
        self.assertEqual(centered[2], 0.0)

    def test_stable_centering_preserves_normalized_probability_differences(self):
        logits = np.array([-3.0, 0.0, 3.0])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        normalizations = {
            "0": {"CloseBlenderLid": {"location": 0.25, "scale": 2.0}}
        }
        probability_scores = normalize_seed_scores(
            probabilities,
            task_name="CloseBlenderLid",
            seed=0,
            task_normalizations=normalizations,
        )
        centered_scores = stable_centered_sigmoid_scores(
            logits,
            task_name="CloseBlenderLid",
            seed=0,
            task_normalizations=normalizations,
        )
        np.testing.assert_allclose(
            np.diff(centered_scores), np.diff(probability_scores), atol=1e-15
        )

    def test_artifacts_use_local_runtime_fallback_and_verify_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_bundle = {}
            normalizations = {}
            for seed in (0, 1):
                seed_dir = root / "runtime" / f"seed{seed}"
                seed_dir.mkdir(parents=True)
                checkpoint_bundle[str(seed)] = {}
                for name in ("config.yaml", "model_final.ckpt"):
                    payload = f"seed={seed},name={name}".encode()
                    path = seed_dir / name
                    path.write_bytes(payload)
                    checkpoint_bundle[str(seed)][name] = {
                        "path": f"/missing/cluster/seed{seed}/{name}",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                normalizations[str(seed)] = {
                    "CloseBlenderLid": {
                        "location": 0.0,
                        "scale": float(seed + 1),
                    }
                }
            bundle_path = root / "runtime_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": "indep",
                        "model_seeds": [0, 1],
                        "binary_final_rollout_labels_only": True,
                        "subtask_safe": False,
                        "checkpoint_bundle": checkpoint_bundle,
                        "task_normalizations": normalizations,
                    }
                )
            )
            scorer = FrozenXr1SafeEnsemble(
                runtime_bundle=bundle_path,
                safe_repo=root,
            )
            self.assertEqual(
                scorer._resolve_artifact(0, "model_final.ckpt"),
                (root / "runtime" / "seed0" / "model_final.ckpt").resolve(),
            )

            class FakeTensor:
                def __init__(self, value):
                    self.value = np.asarray(value)

                def __getitem__(self, item):
                    return FakeTensor(self.value[item])

                def __mul__(self, multiplier):
                    return FakeTensor(self.value * multiplier)

                def unsqueeze(self, axis):
                    return FakeTensor(np.expand_dims(self.value, axis))

                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return self.value

            class NoGrad:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    return False

            class FakeTorch:
                @staticmethod
                def as_tensor(value, device=None):
                    return FakeTensor(value)

                @staticmethod
                def no_grad():
                    return NoGrad()

            class FakeModel:
                def __init__(self, multiplier):
                    self.multiplier = multiplier

                def forward_pre_activation(self, batch):
                    return batch["features"][..., :1] * self.multiplier

                def __call__(self, batch):
                    logits = self.forward_pre_activation(batch).value
                    return FakeTensor(1.0 / (1.0 + np.exp(-logits)))

            scorer._torch = FakeTorch()
            scorer._models = (FakeModel(1.0), FakeModel(2.0))
            scorer.safe_commit = "test-commit"
            result = scorer.score_candidates(
                np.array([[[[2.0, 0.0]]], [[[5.0, 0.0]]]], dtype=np.float32),
                task_name="CloseBlenderLid",
            )
            expected_seed0 = -1.0 / (1.0 + np.exp(np.array([2.0, 5.0])))
            expected_seed1 = (
                -1.0 / (1.0 + np.exp(np.array([4.0, 10.0]))) / 2.0
            )
            np.testing.assert_allclose(
                result["scores"], (expected_seed0 + expected_seed1) / 2.0
            )
            np.testing.assert_allclose(
                result["pre_sigmoid_logits_by_seed"],
                [[2.0, 5.0], [4.0, 10.0]],
            )
            self.assertLess(result["scores"][0], result["scores"][1])
            self.assertEqual(
                result["provenance"]["safe_repository_commit"], "test-commit"
            )
            self.assertEqual(
                result["provenance"]["model_loader_protocol"],
                MODEL_LOADER_PROTOCOL,
            )

            (root / "runtime" / "seed0" / "model_final.ckpt").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                scorer._resolve_artifact(0, "model_final.ckpt")


if __name__ == "__main__":
    unittest.main()
