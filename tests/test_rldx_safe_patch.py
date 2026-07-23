from pathlib import Path
import unittest


PATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "rldx1_safe_features_ef05cd4.patch"
)


class TestRLDXSafePatch(unittest.TestCase):
    def test_patch_preserves_action_path_and_returns_raw_predecoder_features(self):
        patch = PATCH_PATH.read_text()
        self.assertIn(
            '+                safe_feature_steps.append(ao[:, -horizon:].detach())\n',
            patch,
        )
        self.assertIn(
            " pred_velocity = self.action_decoder(ao, embodiment_id)[:, -horizon:]",
            patch,
        )
        self.assertIn(
            '+            output["safe_features"] = '
            "torch.stack(safe_feature_steps, dim=1).float()\n",
            patch,
        )
        self.assertIn(
            '+        return_safe_features = bool('
            'inputs.pop("return_safe_features", False))\n',
            patch,
        )
        self.assertIn(
            '+        request_safe_features=options.get('
            '"request_safe_features", False),\n',
            patch,
        )
        self.assertNotIn(
            "+                    pred_velocity = "
            "self.action_decoder(ao.float(), embodiment_id)",
            patch,
        )


if __name__ == "__main__":
    unittest.main()
