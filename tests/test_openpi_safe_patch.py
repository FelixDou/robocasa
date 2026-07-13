from pathlib import Path
import unittest


PATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "openpi_safe_features_5a6beda.patch"
)


class TestOpenPiSafePatch(unittest.TestCase):
    def test_feature_capture_does_not_cast_policy_action_path(self):
        patch = PATCH_PATH.read_text()
        self.assertIn(
            "+            pre_velocity = suffix_out[:, -self.action_horizon :]\n",
            patch,
        )
        self.assertIn(
            "+                features = features.at[:, step_index].set("
            "pre_velocity.astype(jnp.float32))\n",
            patch,
        )
        self.assertIn("+            v_t = self.action_out_proj(pre_velocity)\n", patch)
        self.assertNotIn(
            "+            pre_velocity = suffix_out[:, -self.action_horizon :]"
            ".astype(jnp.float32)\n",
            patch,
        )
        self.assertIn("+def test_pi0_safe_feature_capture_preserves_actions():\n", patch)
        self.assertIn(
            "+    np.testing.assert_array_equal(np.asarray(safe_actions), np.asarray(actions))\n",
            patch,
        )


if __name__ == "__main__":
    unittest.main()
