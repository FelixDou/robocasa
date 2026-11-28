"""Unit tests for strictly causal grasp-anchored inferred stage clocks."""

import unittest

import numpy as np

from scripts.xr1_lead_time.eval_grasp_anchored_monitor import (
    build_grasp_anchored_stages,
)


class GraspAnchoredStageTests(unittest.TestCase):
    def test_grasp_only_stage_clock_transitions_strictly_causally_at_first_grasp(self):
        dec_steps = list(range(0, 160, 16))
        d = len(dec_steps)
        ep = {
            "rollout_id": "ds_1",
            "task": "DeliverStraw",
            "seed": 1,
            "success": True,
            "horizon": 320,
            "num_steps": 160,
            "decision_steps": dec_steps,
        }
        g = np.array([False, False, False, True, True, False, False, False, False, False])
        stgs, entries, durs = build_grasp_anchored_stages(ep, g, None, include_vlm_stages=False)
        self.assertEqual(stgs[:3], ["g_stage_0"] * 3)
        self.assertEqual(stgs[3:], ["g_stage_1"] * 7)
        self.assertEqual(entries, {"g_stage_0": 0, "g_stage_1": 48})
        self.assertEqual(durs, {"g_stage_0": 48, "g_stage_1": 112})

    def test_grasp_and_vlm_stage_reconstructs_multi_object_pick_place_stages(self):
        dec_steps = list(range(0, 192, 16))
        d = len(dec_steps)
        ep = {
            "rollout_id": "abb_1",
            "task": "ArrangeBreadBasket",
            "seed": 1,
            "success": True,
            "horizon": 320,
            "num_steps": 192,
            "decision_steps": dec_steps,
        }
        # Grasp 1 at idx 2, bread placed at idx 4..5 (2 consecutive latches p1 at idx 5),
        # Grasp 2 (basket) at idx 7, basket placed at idx 9..10 (latches p2 at idx 10)
        g = np.array([k in (2, 3, 7, 8) for k in range(d)], dtype=bool)
        t_map = {
            "bread_in_basket": np.array([k >= 4 for k in range(d)], dtype=bool),
            "basket_on_dining_counter": np.array([k >= 9 for k in range(d)], dtype=bool),
        }
        stgs, entries, durs = build_grasp_anchored_stages(ep, g, t_map, include_vlm_stages=True)
        self.assertEqual(stgs[0], "gv_p0_g0")
        self.assertEqual(stgs[2], "gv_p0_g1")
        # At idx 5, bread_in_basket latches (2 consec) -> advances to p1_g0!
        self.assertEqual(stgs[5], "gv_p1_g0")
        # At idx 7, second grasp advances to p1_g1!
        self.assertEqual(stgs[7], "gv_p1_g1")
        # At idx 10, basket latches -> advances to p2_g0!
        self.assertEqual(stgs[10], "gv_p2_g0")
        self.assertEqual(entries["gv_p0_g1"], 32)
        self.assertEqual(entries["gv_p1_g0"], 80)
        self.assertEqual(entries["gv_p1_g1"], 112)


if __name__ == "__main__":
    unittest.main()
