import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "recovery"
    / "run_lingbot_v21_training.py"
)
SPEC = importlib.util.spec_from_file_location("run_lingbot_v21_training", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RunLingBotV21TrainingTest(unittest.TestCase):
    def setUp(self):
        self.base_dataset = types.ModuleType("base_dataset")
        self.base_dataset.LEROBOT_DATASET_API = "v3"
        self.base_dataset.decode_video_frames = mock.Mock(name="old_decoder")
        self.video_utils = types.ModuleType("video_utils")
        self.video_utils.decode_video_frames = mock.Mock(return_value="frames")
        self.lerobot_constants = types.ModuleType("lerobot.constants")
        self.lerobot_constants.HF_LEROBOT_HOME = Path("/tmp/lerobot")
        self.lerobot_dataset = types.ModuleType("lerobot.datasets.lerobot_dataset")
        self.lerobot_dataset.CODEBASE_VERSION = "v2.1"
        self.vla_data = types.ModuleType("lingbotvla.data.vla_data")
        self.vla_data.base_dataset = self.base_dataset
        self.vla_data.video_utils = self.video_utils
        self.modules = {
            "lerobot": types.ModuleType("lerobot"),
            "lerobot.constants": self.lerobot_constants,
            "lerobot.datasets": types.ModuleType("lerobot.datasets"),
            "lerobot.datasets.lerobot_dataset": self.lerobot_dataset,
            "lingbotvla": types.ModuleType("lingbotvla"),
            "lingbotvla.data": types.ModuleType("lingbotvla.data"),
            "lingbotvla.data.vla_data": self.vla_data,
        }

    def test_enables_v2_layout_for_lerobot_033(self):
        with mock.patch.dict(sys.modules, self.modules), mock.patch.object(
            MODULE, "version", return_value="0.3.3"
        ):
            result = MODULE.enable_lerobot_v21_layout()
            self.assertIs(
                sys.modules["lerobot.utils.constants"], self.lerobot_constants
            )
            result_frames = self.base_dataset.decode_video_frames(
                "episode.mp4", [0.0], 0.001, backend="torchcodec"
            )

        self.assertEqual(result, ("0.3.3", "v2.1"))
        self.assertEqual(self.base_dataset.LEROBOT_DATASET_API, "v2")
        self.assertEqual(result_frames, "frames")
        self.video_utils.decode_video_frames.assert_called_once_with(
            "episode.mp4", [0.0], 0.001, backend="pyav"
        )

    def test_rejects_unexpected_lerobot_version(self):
        with mock.patch.dict(sys.modules, self.modules), mock.patch.object(
            MODULE, "version", return_value="0.4.2"
        ):
            with self.assertRaisesRegex(RuntimeError, "expected lerobot=0.3.3"):
                MODULE.enable_lerobot_v21_layout()

    @unittest.skipIf(torch is None, "PyTorch is not installed in the local test env")
    def test_combines_joint_and_temporal_action_masks(self):
        joint_mask = torch.tensor(
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [True, True, False],
                ]
            ]
        )
        action_is_pad = torch.tensor([[False, True, False]])

        result = MODULE.combine_action_padding_mask(joint_mask, action_is_pad)

        expected = torch.tensor(
            [
                [
                    [True, True, False],
                    [False, False, False],
                    [True, True, False],
                ]
            ]
        )
        torch.testing.assert_close(result, expected)

    @unittest.skipIf(torch is None, "PyTorch is not installed in the local test env")
    def test_rejects_mismatched_temporal_mask_shape(self):
        with self.assertRaisesRegex(ValueError, "mask shapes disagree"):
            MODULE.combine_action_padding_mask(
                torch.ones((2, 50, 32), dtype=torch.bool),
                torch.zeros((2, 49), dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
