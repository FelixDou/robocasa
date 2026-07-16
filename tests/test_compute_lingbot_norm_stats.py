import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "recovery"
    / "compute_lingbot_norm_stats.py"
)
SPEC = importlib.util.spec_from_file_location("compute_lingbot_norm_stats", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComputeLingBotNormStatsTest(unittest.TestCase):
    def test_maps_lerobot_action_order_to_lingbot_features(self):
        state = np.arange(32, dtype=np.float32).reshape(2, 16)
        action = np.arange(24, dtype=np.float32).reshape(2, 12)

        mapped = MODULE.mapped_features(state, action)

        np.testing.assert_array_equal(
            mapped["observation.state.end.position"], state[:, 7:14]
        )
        np.testing.assert_array_equal(mapped["action.end.position"], action[:, 5:11])
        np.testing.assert_array_equal(
            mapped["action.effector.position"], action[:, 11:12]
        )
        np.testing.assert_array_equal(mapped["action.base.position"], action[:, 0:3])
        np.testing.assert_array_equal(mapped["action.waist.position"], action[:, 3:4])

    def test_statistics_match_population_meanstd_and_quantiles(self):
        value = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)

        stats = MODULE.feature_statistics(value)

        np.testing.assert_allclose(stats["mean"], [1.0, 3.0])
        np.testing.assert_allclose(stats["std"], [1.0, 1.0])
        np.testing.assert_allclose(stats["min"], [0.0, 2.0])
        np.testing.assert_allclose(stats["max"], [2.0, 4.0])
        np.testing.assert_allclose(stats["q01"], [0.02, 2.02])
        np.testing.assert_allclose(stats["q99"], [1.98, 3.98])

    def test_rejects_wrong_action_schema(self):
        with self.assertRaisesRegex(ValueError, r"\(N, 12\)"):
            MODULE.mapped_features(
                np.zeros((2, 16), dtype=np.float32),
                np.zeros((2, 11), dtype=np.float32),
            )

    def test_reads_lingbot_multi_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "datasets.txt"
            path.write_text(
                "# comment\n"
                "robocasa_lerobot /datasets/one/lerobot\n"
                "robocasa_lerobot /datasets/two/lerobot\n"
            )

            self.assertEqual(
                MODULE.read_dataset_manifest(path),
                [Path("/datasets/one/lerobot"), Path("/datasets/two/lerobot")],
            )


if __name__ == "__main__":
    unittest.main()
