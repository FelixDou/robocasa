import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from robocasa.recovery.safe.validate_official_export import (
    parse_selector,
    validate_loaded_rollouts,
    validate_official_export,
)


def rollout(task, success, *, task_id=0, finite=True):
    features = np.ones((3, 8), dtype=np.float32)
    if not finite:
        features[0, 0] = np.nan
    return SimpleNamespace(
        task_id=task_id,
        task_description=task,
        episode_success=int(success),
        hidden_states=features,
        action_vectors=np.ones((3, 48), dtype=np.float32),
    )


class TestOfficialSafeExportValidation(unittest.TestCase):
    def test_selector_parsing_matches_official_loader_types(self):
        self.assertEqual(parse_selector("0.0"), 0.0)
        self.assertEqual(parse_selector("1"), 1.0)
        self.assertEqual(parse_selector("mean"), "mean")
        self.assertEqual(parse_selector("concat-2"), "concat-2")
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            parse_selector("2.0")

    def test_loaded_rollout_balance_and_shapes(self):
        report = validate_loaded_rollouts(
            [
                rollout("TaskA", True),
                rollout("TaskA", False),
                rollout("TaskB", True, task_id=1),
                rollout("TaskB", False, task_id=1),
            ],
            expected_rollouts=4,
            expected_successes=2,
            expected_failures=2,
            expected_tasks=["TaskA", "TaskB"],
            task_id_to_name={0: "TaskA", 1: "TaskB"},
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["loaded_feature_dimensions"], [[8]])
        self.assertEqual(report["loaded_action_dimensions"], [[48]])
        self.assertEqual(
            report["task_counts"]["TaskA"], {"successes": 1, "failures": 1}
        )

    def test_loaded_rollout_validation_rejects_bad_data(self):
        report = validate_loaded_rollouts(
            [rollout("TaskA", True, finite=False)],
            expected_failures=1,
            expected_tasks=["TaskB"],
        )
        self.assertFalse(report["valid"])
        self.assertEqual(len(report["errors"]), 3)

    def test_invokes_official_loader_from_explicit_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            safe_repo = Path(tmp) / "SAFE"
            export_dir = Path(tmp) / "export"
            (safe_repo / "failure_prob" / "data").mkdir(parents=True)
            (safe_repo / "failure_prob" / "data" / "pizero.py").write_text("# marker\n")
            export_dir.mkdir()
            (export_dir / "conversion_report.json").write_text(
                '{"task_ids": {"TaskA": 0}}\n'
            )
            fake_loader = mock.Mock(return_value=[rollout("TaskA", True)])
            fake_module = SimpleNamespace(load_rollouts_from_root=fake_loader)
            with mock.patch.dict(
                "sys.modules", {"failure_prob.data.pizero": fake_module}
            ):
                report = validate_official_export(
                    export_dir,
                    safe_repo=safe_repo,
                    expected_rollouts=1,
                    expected_successes=1,
                    expected_failures=0,
                    expected_tasks=["TaskA"],
                    expected_safe_commit=None,
                )
            self.assertTrue(report["valid"])
            fake_loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
