"""Tests for the XR-1 fine-tuning module (Step 0). PyTorch tests run only where torch is installed."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery import xr1_finetune as xf  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ROBOSUITE = importlib.util.find_spec("robosuite") is not None


class ScheduleTest(unittest.TestCase):
    def test_warmup_then_cosine(self):
        self.assertAlmostEqual(xf.learning_rate(0, 1000), xf.LR_START)
        self.assertAlmostEqual(xf.learning_rate(250, 1000), (xf.LR_START + xf.LR_MAX) / 2)
        self.assertAlmostEqual(xf.learning_rate(500, 1000), xf.LR_MAX)
        self.assertAlmostEqual(xf.learning_rate(1000, 1000), xf.LR_MIN)
        self.assertLess(xf.learning_rate(800, 1000), xf.learning_rate(600, 1000))

    def test_negligible_update_schedule_is_flat(self):
        values = {xf.learning_rate(s, 5, start=1e-12, peak=1e-12, end=1e-12, warmup=0) for s in range(1, 6)}
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values.pop(), 1e-12)


class DerivedFolderTest(unittest.TestCase):
    def _template(self, root, keys):
        template = root / "template"
        template.mkdir()
        (template / "modeling_mibot.py").write_text("# SAFE patched\n")
        (template / "config.json").write_text(json.dumps({"from": "template"}))
        (template / "model-00001-of-00001.safetensors").write_text("old weights")
        (template / xf.WEIGHT_INDEX).write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors"
                                                                           for k in keys}}))
        return template

    class FakeModel:
        def __init__(self, keys):
            self.keys = keys

        def save_pretrained(self, folder, **kwargs):
            folder = Path(folder)
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(json.dumps({"from": "save_pretrained"}))
            (folder / "model-00001-of-00001.safetensors").write_text("new weights")
            (folder / xf.WEIGHT_INDEX).write_text(json.dumps({"weight_map": {k: "x" for k in self.keys}}))

    def test_weights_from_the_model_other_files_from_the_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = self._template(root, ["dit.a", "vlm.b"])
            out = xf.save_derived_folder(self.FakeModel(["dit.a", "vlm.b"]), template, root / "step_1", {"step": 1})
            self.assertEqual((out / "model-00001-of-00001.safetensors").read_text(), "new weights")
            self.assertEqual((out / "modeling_mibot.py").read_text(), "# SAFE patched\n")
            self.assertEqual(json.loads((out / "config.json").read_text()), {"from": "template"})
            self.assertEqual(json.loads((out / xf.MANIFEST).read_text())["step"], 1)
            self.assertFalse((root / "step_1.partial").exists())
            with self.assertRaises(FileExistsError):
                xf.save_derived_folder(self.FakeModel(["dit.a", "vlm.b"]), template, out, {})

    def test_a_changed_key_set_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = self._template(root, ["dit.a", "vlm.b"])
            with self.assertRaises(RuntimeError):
                xf.save_derived_folder(self.FakeModel(["dit.a"]), template, root / "step_1", {})
            self.assertFalse((root / "step_1").exists())


class LeRobotFolderTest(unittest.TestCase):
    def test_falls_back_to_the_one_dated_folder_without_robosuite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "v1.0/target/composite/ArrangeTea/20250812/lerobot"
            folder.mkdir(parents=True)
            if HAS_ROBOSUITE:
                self.skipTest("the registry is importable here, so no fallback is needed")
            self.assertEqual(xf.find_lerobot_dir("ArrangeTea", base), folder)
            (base / "v1.0/target/composite/ArrangeTea/20250901/lerobot").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                xf.find_lerobot_dir("ArrangeTea", base)
            with self.assertRaises(FileNotFoundError):
                xf.find_lerobot_dir("GarnishPancake", base)

    def test_an_override_wins(self):
        self.assertEqual(xf.find_lerobot_dir("ArrangeTea", "/nowhere", {"ArrangeTea": "/data/tea"}),
                         Path("/data/tea"))

    def test_override_parsing(self):
        args = xf.build_parser().parse_args(["input-match", "--model-dir", "m", "--dataset-base", "d",
                                             "--output-dir", "o", "--lerobot-dir", "ArrangeTea=/data/tea"])
        self.assertEqual(xf._overrides(args), {"ArrangeTea": "/data/tea"})
        args.lerobot_dir = ["ArrangeTea"]
        with self.assertRaises(ValueError):
            xf._overrides(args)


class CliTest(unittest.TestCase):
    def test_parser(self):
        args = xf.build_parser().parse_args(["train", "--model-dir", "m", "--dataset-base", "d",
                                             "--output-dir", "o", "--max-steps", "5"])
        self.assertEqual(args.tasks, list(xf.PILOT_TASKS))
        self.assertEqual(args.lr_max, xf.LR_MAX)
        args = xf.build_parser().parse_args(["input-match", "--model-dir", "m", "--dataset-base", "d",
                                             "--output-dir", "o"])
        self.assertEqual(args.episodes_per_task * args.frames_per_episode, args.min_samples_per_task)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class FlowLossTest(unittest.TestCase):
    def _model(self):
        import torch
        from torch import nn

        class Output:
            def __init__(self, cache, position_ids):
                self.past_key_values, self.position_ids = cache, position_ids

        class FakeVLMInner(nn.Module):
            def forward(self, input_ids, attention_mask, pixel_values_videos, video_grid_thw, use_cache):
                batch, length = input_ids.shape
                cache = [(torch.zeros(batch, 1, length, 2), torch.zeros(batch, 1, length, 2))]
                return Output(cache, torch.arange(length).view(1, 1, -1).repeat(3, batch, 1))

        class FakeVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeVLMInner()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm = FakeVLM()
                self.state_projector = nn.Linear(60, 4)
                self.head = nn.Linear(60, 60)

            def rotary_emb(self, x, positions):
                return positions.float().cos(), positions.float().sin()

            def dit_forward(self, noisy_action, t, action_mask, state_embed, position_embeds, past_key_values,
                            attn_mask):
                return self.head((noisy_action * action_mask).float()) + t.float() + state_embed.mean()

        torch.manual_seed(0)
        return FakeModel()

    def _batch(self, size=3):
        import torch

        mask = torch.zeros(size, 16, 60)
        mask[:, :, :12] = 1
        return {"input_ids": torch.ones(size, 5, dtype=torch.long), "attention_mask": torch.ones(size, 5),
                "pixel_values_videos": torch.zeros(1), "video_grid_thw": torch.zeros(1),
                "state": torch.randn(size, 4, 60), "action_mask": mask, "action": torch.randn(size, 16, 60)}

    def test_timesteps_follow_the_released_schedule(self):
        import torch

        t = xf.flow_timesteps(torch.tensor([0.0, 1.0, 0.125]))
        self.assertAlmostEqual(float(t[0]), 0.999)
        self.assertAlmostEqual(float(t[1]), 0.0)
        self.assertAlmostEqual(float(t[2]), 0.999 * (1 - 0.125 ** (1 / 1.5)), places=6)

    def test_per_sample_losses_average_to_the_batch_loss_and_ignore_inactive_dims(self):
        import torch

        model, batch = self._model(), self._batch()
        u, noise = torch.rand(3), torch.randn(3, 16, 60)
        loss, _ = xf.flow_loss(model, batch, u=u, noise=noise)
        per, _ = xf.flow_loss(model, batch, u=u, noise=noise, per_sample=True)
        self.assertEqual(per.shape, (3,))
        self.assertAlmostEqual(float(per.mean()), float(loss), places=5)
        changed = dict(batch, action=batch["action"].clone())
        changed["action"][:, :, 12:] += 100.0            # inactive dims must not matter
        again, _ = xf.flow_loss(model, changed, u=u, noise=noise)
        self.assertAlmostEqual(float(again), float(loss), places=5)

    def test_gradients_reach_the_trainable_modules_only(self):
        model, batch = self._model(), self._batch()
        params = xf.prepare_trainable(model, modules=("head",))
        loss, _ = xf.flow_loss(model, batch)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in params))
        self.assertIsNone(model.state_projector.weight.grad)
        before = [p.detach().clone() for p in params]
        optimizer = xf.MasterAdamW(params)
        norm = optimizer.step(1e-2)
        self.assertGreater(norm, 0)
        self.assertTrue(any(bool((p != b).any()) for p, b in zip(params, before)))
        self.assertTrue(all(p.grad is None for p in params))


if __name__ == "__main__":
    unittest.main()
