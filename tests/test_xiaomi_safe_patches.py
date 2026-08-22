from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATCH = ROOT / "patches/xiaomi_robotics_1_safe_model_0d1aa76.patch"
SERVER_PATCH = ROOT / "patches/xiaomi_robotics_1_safe_server_4da1db0.patch"
BEST_OF_K_SERVER_PATCH = (
    ROOT / "patches/xiaomi_robotics_1_safe_best_of_k_server_4da1db0.patch"
)


class TestXiaomiSafePatches(unittest.TestCase):
    @unittest.skipUnless(shutil.which("patch"), "patch executable is required")
    def test_model_patch_is_well_formed_and_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "modeling_mibot.py"
            lines = ["# synthetic pinned-source filler\n"] * 1900
            lines[1797] = (
                "    def dit_forward(self, noisy_action, t, action_mask, "
                "state_embed, position_embeds, past_key_values, attn_mask):\n"
            )
            lines[1815] = "        output = self.action_output_layer(hidden_states)\n"
            lines[1870] = "            v = dit_forward_fn(x, t)\n"
            source.write_text("".join(lines))

            result = subprocess.run(
                ["patch", "--batch", "--forward", "-p1"],
                cwd=tmp,
                input=MODEL_PATCH.read_bytes(),
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                (result.stdout + result.stderr).decode(errors="replace"),
            )
            patched = source.read_text()
            self.assertIn("safe_features: torch.FloatTensor = None", patched)
            self.assertIn("safe_feature_steps = []", patched)
            self.assertIn("return_safe_features=return_safe_features", patched)
            self.assertLess(
                patched.index("output = self.action_output_layer(hidden_states)"),
                patched.index("return output, hidden_states.detach().float()"),
            )

    def test_model_patch_captures_pre_output_action_tokens_without_casting_actions(
        self,
    ):
        patch = MODEL_PATCH.read_text()
        self.assertIn(
            "+            return output, hidden_states.detach().float()\n", patch
        )
        self.assertIn(
            "+                safe_features=torch.stack(safe_feature_steps, dim=1),\n",
            patch,
        )
        self.assertNotIn("self.action_output_layer(hidden_states.float())", patch)

    def test_server_patch_is_opt_in_and_preserves_action_only_response(self):
        patch = SERVER_PATCH.read_text()
        self.assertIn(
            '+                                input_data.pop("request_safe_features", False)\n',
            patch,
        )
        best_of_k_patch = BEST_OF_K_SERVER_PATCH.read_text()
        self.assertIn(
            '+                            sampling_seed = input_data.pop("sampling_seed", None)\n',
            best_of_k_patch,
        )
        self.assertIn(
            "+                            with torch.random.fork_rng(\n",
            best_of_k_patch,
        )
        self.assertIn(
            "+                                    torch.manual_seed(sampling_seed)\n",
            best_of_k_patch,
        )
        self.assertIn(
            "+                            response_data = outputs.actions.cpu()\n",
            patch,
        )
        self.assertIn("+                            if request_safe_features:\n", patch)
        self.assertIn(
            '+                                        "model_family": "xiaomi_robotics_1",\n',
            patch,
        )
        self.assertIn(
            '+                                        "sampling_seed": sampling_seed,\n',
            best_of_k_patch,
        )

    @unittest.skipUnless(shutil.which("patch"), "patch executable is required")
    def test_best_of_k_server_patch_applies_after_safe_server_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "deploy" / "server.py"
            source.parent.mkdir()
            source.write_text(
                """                            request_safe_features = bool(
                                input_data.pop("request_safe_features", False)
                            )
                            robot_type = input_data["task_id"]
                            data = {key: (value.to(device=self.model.device, dtype=self.model.dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value.to(device=self.model.device) if isinstance(value, torch.Tensor) else value) for key, value in input_data.items()}

                            outputs = self.model(
                                **data,
                                return_safe_features=request_safe_features,
                            )

                            response_data = outputs.actions.cpu()
                            if request_safe_features:
                                features = outputs.safe_features.cpu()
                                if features.ndim != 4:
                                    raise RuntimeError("shape")
                                response_data = {
                                    "safe_feature_metadata": {
                                        "action_horizon": int(features.shape[2]),
                                        "flow_steps": int(features.shape[1]),
                                        "aggregation": "raw",
                                    },
                                }
                            response = pickle.dumps(response_data)
"""
            )
            result = subprocess.run(
                ["patch", "--batch", "--forward", "-p1"],
                cwd=tmp,
                input=BEST_OF_K_SERVER_PATCH.read_bytes(),
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                (result.stdout + result.stderr).decode(errors="replace"),
            )
            patched = source.read_text()
            self.assertIn('input_data.pop("sampling_seed", None)', patched)
            self.assertIn("torch.random.fork_rng", patched)
            self.assertIn('"sampling_seed": sampling_seed', patched)


if __name__ == "__main__":
    unittest.main()
