import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.conformal import (
    calibrate_functional_threshold,
    first_detection,
    threshold_for_length,
)
from robocasa.recovery.safe.evaluate import evaluate_groups, save_results
from robocasa.recovery.safe import models


class TestSafeConformal(unittest.TestCase):
    def test_functional_calibration_and_crossing(self):
        reference = [np.linspace(0.05, 0.1, 8), np.linspace(0.04, 0.11, 10)]
        calibration = [np.linspace(0.03, 0.12, 7), np.linspace(0.06, 0.1, 9)]
        result = calibrate_functional_threshold(
            reference, calibration, alpha=0.2, normalized_length=20
        )
        self.assertEqual(len(result["threshold"]), 20)
        self.assertTrue(np.all(np.isfinite(result["threshold"])))
        threshold = threshold_for_length(result, 8)
        self.assertIsNone(first_detection(threshold - 0.01, result))
        self.assertEqual(first_detection(threshold + 0.01, result), 0)

    def test_result_serialization(self):
        calibration = {
            "threshold": [0.5] * 10,
            "normalized_length": 10,
        }
        rollouts = [
            {"rollout_id": "s", "task_name": "A", "failed": False, "scores": [0.1, 0.2]},
            {"rollout_id": "f", "task_name": "A", "failed": True, "scores": [0.2, 0.8]},
        ]
        results = evaluate_groups(rollouts, calibration)
        self.assertEqual(results["overall"]["confusion"]["tp"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_results(results, Path(tmp) / "results.json")
            self.assertEqual(json.loads(path.read_text())["overall"]["num_rollouts"], 2)


@unittest.skipIf(models.torch is None, "PyTorch is not installed")
class TestSafeModels(unittest.TestCase):
    def test_mlp_and_lstm_forward_with_variable_lengths(self):
        torch = models.torch
        features = torch.randn(2, 4, 8)
        lengths = torch.tensor([4, 2])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        labels = torch.tensor([0.0, 1.0])
        for model_type in ("mlp", "lstm"):
            model = models.make_model(models.ModelConfig(model_type, 8, hidden_dim=6, num_layers=1))
            scores = model(features, lengths)
            self.assertEqual(tuple(scores.shape), (2, 4))
            self.assertTrue(torch.isfinite(scores).all())
            loss = models.masked_binary_cross_entropy(scores, labels, mask)
            self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
