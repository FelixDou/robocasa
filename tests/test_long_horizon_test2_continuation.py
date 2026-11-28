"""Unit tests for strictly causal Test-2 pre-recovery continuation feature extraction
and out-of-fold L2 logistic evaluation.
"""

import unittest

import numpy as np

from scripts.xr1_lead_time.eval_test2_continuation_pilot import (
    _auroc,
    _fit_ridge_logistic_oof,
    extract_window_continuation_features,
)


class Test2ContinuationTests(unittest.TestCase):
    def test_window_extraction_is_strictly_causal_before_window_end(self):
        d = 10
        ep_meta = {
            "rollout_id": "ep_test",
            "task": "DeliverStraw",
            "horizon": 320,
            "start_row": 0,
            "end_row": d,
            "decision_steps": [i * 16 for i in range(d)],
        }
        rng = np.random.default_rng(0)
        base_arrays = {
            "executed_actions": rng.normal(size=(d, 18)).astype(np.float32),
            "action_chunks": rng.normal(size=(d, 36)).astype(np.float32),
            "state_history": rng.normal(size=(d, 16)).astype(np.float32),
            "dit_last": rng.normal(size=(d, 32)).astype(np.float32),
            "siglip_3cam": rng.normal(size=(d, 3, 16)).astype(np.float32),
        }
        vlm_base = {"ep_test__straw_in_glass__zoh_logit_diff": np.linspace(-3.0, 2.0, d).astype(np.float32)}

        # Window [32, 96) covers decision indices 2, 3, 4, 5 (steps 32, 48, 64, 80).
        # Mutate ONLY index >= 6 (steps >= 96) — MUST NOT affect any extracted feature!
        mut_arrays = {k: v.copy() for k, v in base_arrays.items()}
        for k in mut_arrays:
            mut_arrays[k][6:] = 999.0
        vlm_mut = {"ep_test__straw_in_glass__zoh_logit_diff": vlm_base["ep_test__straw_in_glass__zoh_logit_diff"].copy()}
        vlm_mut["ep_test__straw_in_glass__zoh_logit_diff"][6:] = 999.0

        f0 = extract_window_continuation_features(
            ep_meta, base_arrays, vlm_base,
            relation="straw_in_glass", onset_step=16, confirmed_step=32, window_end_step=96,
        )
        f1 = extract_window_continuation_features(
            ep_meta, mut_arrays, vlm_mut,
            relation="straw_in_glass", onset_step=16, confirmed_step=32, window_end_step=96,
        )
        self.assertIsNotNone(f0)
        self.assertEqual(f0, f1)
        self.assertEqual(f0["ctrl_win_decisions"], 4.0)

    def test_ridge_logistic_oof_separates_synthetic_signal_without_leakage(self):
        rng = np.random.default_rng(42)
        y = np.array([0] * 6 + [1] * 18, dtype=int)
        X = rng.normal(0, 0.5, size=(24, 4))
        X[y == 1, 0] += 2.5
        groups = [f"g{i}" for i in range(24)]
        oof = _fit_ridge_logistic_oof(X, y, groups, l2=1.0)
        self.assertGreater(_auroc(y, oof), 0.90)


if __name__ == "__main__":
    unittest.main()
