"""Unit tests for strictly causal VLM baseline subtraction, Platt calibration,
placement-only / verified-hold gating, and continuous soft missed-effect evidence.
"""

import unittest

import numpy as np

from scripts.xr1_lead_time.eval_vlm_scoring_upgrades import (
    _fit_1d_platt,
    build_soft_monitor_episode,
    causal_delta_pregrasp,
    causal_delta_t0,
)


class VLMScoringUpgradeTests(unittest.TestCase):
    def test_causal_delta_pregrasp_never_leaks_future_logits(self):
        base = np.array([2.0, 2.2, 1.8, 2.1, 6.0, 6.5, 6.2], dtype=np.float32)
        mut = base.copy()
        mut[4:] = 99.0  # Huge future spike after first grasp at index 3
        grasp = np.array([False, False, False, True, True, False, False])

        d0 = causal_delta_pregrasp(base, grasp)
        d1 = causal_delta_pregrasp(mut, grasp)
        # Prefix up through index 3 must be bit-identical regardless of future logits!
        np.testing.assert_allclose(d0[:4], d1[:4])
        # Baseline is frozen at median(base[:4]) = 2.05 -> post-placement jump is ~+4.0
        self.assertGreater(float(d0[5]), 3.5)
        self.assertLess(abs(float(d0[3])), 0.2)

    def test_causal_delta_t0_cancels_constant_scene_bias(self):
        biased = np.array([3.5, 3.5, 3.6, 7.5, 7.8], dtype=np.float32)
        d = causal_delta_t0(biased)
        self.assertAlmostEqual(float(d[0]), 0.0, places=5)
        self.assertGreater(float(d[3]), 3.9)

    def test_platt_calibration_recovers_positive_slope_and_bias_shift(self):
        x = np.array([1.0, 1.2, 1.5, 1.8, 5.0, 5.2, 5.5, 5.8])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        a, b = _fit_1d_platt(x, y)
        self.assertGreater(a, 0.5)
        # Decision boundary -b/a should lie cleanly between 1.8 and 5.0
        boundary = -b / a
        self.assertGreater(boundary, 2.0)
        self.assertLess(boundary, 4.8)

    def test_soft_evidence_separates_uncertain_success_from_confident_miss_and_freezes_on_arrival(self):
        dec_steps = list(range(0, 240, 16))  # 15 decisions
        d = len(dec_steps)
        grasp = np.zeros(d, dtype=bool)
        grasp[1:3] = True  # steps 16..48, release at step 48 (dec idx 3)

        ep = {
            "rollout_id": "ep_soft",
            "task": "DeliverStraw",
            "seed": 1,
            "success": True,
            "horizon": 320,
            "num_steps": 240,
            "decision_steps": dec_steps,
            "decision_stages": ["s0"] * d,
            "segments": [],
            "recovery_observations": [],
        }

        # Case 1: Clean placement seen right at release (idx 3-4), then camera turns away (idx 5..14 -> prob 0.05)
        p_clean = np.full(d, 0.05, dtype=np.float32)
        p_clean[3:5] = 0.90
        ed_clean = build_soft_monitor_episode(
            ep,
            grasp_bool_map={"straw_grasped": grasp},
            any_grasp_dec=grasp,
            target_prob_map={"straw_in_glass": p_clean},
            target_bool_map={"straw_in_glass": p_clean >= 0.5},
            mode="soft_evidence",
            hold_policy="placement_only",
            use_inferred_stage=True,
            use_universal_grasp=True,
            clear_threshold=0.70,
        )

        # Case 2: Uncertain success (p = 0.45 throughout post-release)
        p_unc = np.full(d, 0.05, dtype=np.float32)
        p_unc[3:] = 0.45
        ed_unc = build_soft_monitor_episode(
            ep,
            grasp_bool_map={"straw_grasped": grasp},
            any_grasp_dec=grasp,
            target_prob_map={"straw_in_glass": p_unc},
            target_bool_map={"straw_in_glass": p_unc >= 0.5},
            mode="soft_evidence",
            hold_policy="placement_only",
            use_inferred_stage=True,
            use_universal_grasp=True,
            clear_threshold=0.70,
        )

        # Case 3: True confident miss (p = 0.05 throughout post-release)
        p_miss = np.full(d, 0.05, dtype=np.float32)
        ed_miss = build_soft_monitor_episode(
            ep,
            grasp_bool_map={"straw_grasped": grasp},
            any_grasp_dec=grasp,
            target_prob_map={"straw_in_glass": p_miss},
            target_bool_map={"straw_in_glass": p_miss >= 0.5},
            mode="soft_evidence",
            hold_policy="placement_only",
            use_inferred_stage=True,
            use_universal_grasp=True,
            clear_threshold=0.70,
        )

        # Clean placement must freeze evidence (stays small even after robot turns away!)
        self.assertLess(float(np.max(ed_clean.missed_effect_age)), 15.0)
        # Uncertain success (p=0.45) must accumulate MUCH less evidence than confident miss (p=0.05)!
        self.assertLess(float(ed_unc.missed_effect_age[-1]), 0.5 * float(ed_miss.missed_effect_age[-1]))
        self.assertGreater(float(ed_miss.missed_effect_age[-1]), 140.0)
        # Placement-only must have no loss_age
        self.assertTrue(np.all(np.isneginf(ed_miss.loss_age)))


if __name__ == "__main__":
    unittest.main()
