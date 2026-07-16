import argparse
import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "recovery"
    / "prepare_lingbot_expert_config.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_lingbot_expert_config", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareLingBotExpertConfigTest(unittest.TestCase):
    def test_preserves_architecture_and_enables_expert_only(self):
        base = {
            "model": {"model_path": "old", "tokenizer_path": "old", "post_training": False},
            "data": {},
            "train": {
                "token_num_experts": 32,
                "align_params": {
                    "depth": {"moge_path": "old", "morgbd_path": "old"},
                    "video": {"ckpt_path": "old", "config_path": "old"},
                },
            },
        }
        args = argparse.Namespace(
            model_path=Path("/models/lingbot"),
            tokenizer_path=Path("/models/qwen"),
            train_manifest=Path("/data/human300.txt"),
            robot_config_root=Path("/configs/robots"),
            norm_stats_file=Path("/stats/human300.json"),
            output_dir=Path("/outputs/run"),
            moge_path=Path("/teachers/moge.pt"),
            depth_path=Path("/teachers/depth.pt"),
            dino_checkpoint=Path("/teachers/dino.pth"),
            dino_config=Path("/teachers/dino.yaml"),
            num_workers=4,
            global_batch_size=16,
            gradient_accumulation_steps=4,
            max_steps=30000,
            save_steps=5000,
            learning_rate=1.0e-4,
        )

        config = MODULE.configure(base, args)

        self.assertTrue(config["model"]["post_training"])
        self.assertTrue(config["train"]["train_expert_only"])
        self.assertTrue(config["train"]["freeze_vision_encoder"])
        self.assertEqual(config["train"]["token_num_experts"], 32)
        self.assertEqual(config["train"]["global_batch_size"], 16)
        self.assertEqual(config["train"]["gradient_accumulation_steps"], 4)
        self.assertEqual(config["data"]["prompt_type"], "global")
        self.assertEqual(
            config["train"]["align_params"]["depth"]["moge_path"],
            "/teachers/moge.pt",
        )


if __name__ == "__main__":
    unittest.main()
