from pathlib import Path
import unittest


PATCH = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "abot_m05_safe_features_7642747.patch"
)


class TestABotSafePatch(unittest.TestCase):
    def test_patch_captures_action_stream_before_projection(self):
        text = PATCH.read_text()
        capture = text.index("+            safe_features = latent_hidden_states.detach()")
        projection = text.index("             latent_hidden_states = self.action_proj_out")
        self.assertLess(capture, projection)
        self.assertIn("return_safe_features=False", text)
        self.assertIn("action_stream_post_norm_pre_action_proj_out", text)

    def test_patch_excludes_final_cache_only_transformer_call(self):
        text = PATCH.read_text()
        self.assertIn("+                if should_step:", text)
        self.assertIn("+                    safe_feature_steps.append", text)
        self.assertIn("+            torch.stack(safe_feature_steps, dim=1).float()", text)

    def test_server_capture_is_opt_in_and_validated(self):
        text = PATCH.read_text()
        self.assertIn('obs.pop("request_safe_features", False)', text)
        self.assertIn("features.ndim != 4", text)
        self.assertIn("torch.isfinite(features).all()", text)
        self.assertIn('"model_family": "abot_m05"', text)


if __name__ == "__main__":
    unittest.main()
