"""R26 (scripts/xr1_lead_time/r26_alarms_at_limit.py): Table I with an alarm at a failure's final inference as a miss."""
import math
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load
from tests.test_r21_native_bands import TASKS, rows

r26, hk, rx, cl, ctl, r21 = (load(n) for n in ("r26_alarms_at_limit", "hk_free", "r19_extras", "confirm_lead",
                                                 "cross_task_lead", "r21_native_bands"))


class R26Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(26)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        self.rk = {t: 0.5 for t in TASKS}

    def test_recount_by_hand(self):
        """Four failures: two warned early, one only at the final inference, one missed; 2 of 10 successes alarmed."""
        m = r26.recount([True, True, False, True], [0.5, 0.0, 0.0, 0.2], fp=2, n=10)
        self.assertEqual(m["alarms_at_limit"], 1)
        self.assertAlmostEqual(m["tpr"], 0.75)
        self.assertAlmostEqual(m["tpr_before_limit"], 0.5)
        self.assertAlmostEqual(m["fpr"], 0.2)
        self.assertAlmostEqual(m["bacc"], (0.75 + 0.8) / 2)
        self.assertAlmostEqual(m["bacc_before_limit"], (0.5 + 0.8) / 2)
        self.assertAlmostEqual(m["mean_lead"], 0.7 / 4)
        with self.assertRaises(ValueError):
            r26.recount([True, False], [0.3, 0.1], fp=0, n=5)        # a miss with a lead

    def test_final_inference_is_a_hit_with_lead_zero(self):
        """On the real evaluators (hk_free for E, G and P; cross_task_lead for Table I's P), the hits with lead 0
        are exactly the alarms at a failure's final inference ('final'). E must reach the final inference here."""
        light = lambda rs: cl.Cohort([r21.light(r) for r in rs], r21.LIGHT)
        finals = {}
        for timer in ("G_abs", "G", "P"):
            cu, rc = hk.prepared(light(self.cal), timer, "all", None, self.rk)
            eu, re_ = hk.prepared(light(self.te), timer, "all", None, self.rk)
            p = hk.raw_eval(eu, re_, hk.fit(cu, rc)[0])
            m = r26.recount(p["hit"], p["lead"], p["fp"], p["n"])
            self.assertEqual(m["alarms_at_limit"], p["final"], timer)
            finals[timer] = p["final"]
        self.assertGreater(finals["G_abs"], 0)
        calC, teC = cl.Cohort(self.cal, "timer"), cl.Cohort(self.te, "timer")
        tm, _ = cl.fit_fast(calC, self.rk)
        p = ctl.raw_eval(teC, self.rk, tm)
        self.assertEqual(r26.recount(p["hit"], p["lead"], p["fp"], p["n"])["alarms_at_limit"], p["final"])

    def test_saved_counts_give_the_same_recount(self):
        """from_counts on r19_extras.outcome (what R21 saved) equals recount on the per-failure arrays."""
        a = np.array([3.0, 10.0, math.inf, 10.0, 5.0, math.inf, 8.0])
        L = np.array([10.0, 10.0, 10.0, 10.0, 8.0, 6.0, 9.0])
        H = np.array([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 9.0])
        failed = np.array([True, True, True, True, False, False, True])
        o, lead = rx.outcome(a, L, H, failed)
        hit = (a <= L)[failed]
        mine = r26.recount(hit, lead, o["fp"], o["n_success"])
        saved = r26.from_counts(o)
        for k in r26.KEYS:
            self.assertAlmostEqual(mine[k], saved[k], places=12, msg=k)
        self.assertEqual(mine["alarms_at_limit"], 2)

    def test_bands_from_r21_and_gates(self):
        o = lambda at: dict(tpr=0.5, fpr=0.1, bacc=0.7, mean_lead=0.2, fp=4, n_success=40, n_failure=40, alarms_at_limit=at)
        block = {lab: dict(standalone=o(i)) for i, lab in enumerate(("seed_average", "seed_0", "seed_1", "seed_2"))}
        fake = dict(heads=dict(h=dict(seen=dict(P=block), unseen=dict(G_abs=block))))
        rows_ = r26.bands_from_r21(fake, ("h",))
        self.assertAlmostEqual(rows_["h"]["seen"]["seed_1"]["tpr_before_limit"], 0.5 - 2 / 40)
        self.assertAlmostEqual(rows_["h"]["unseen"]["mean_std_over_seeds"]["alarms_at_limit"]["mean"], 2.0)
        ref = {"h": {"seen": {lab: o(0) for lab in block}}}
        g = r26.gate("R", r26.compare(rows_, ref))
        self.assertTrue(g["ok"])
        self.assertEqual(g["compared"], 4)
        self.assertFalse(r26.gate("R", {})["ok"])                       # nothing compared fails


if __name__ == "__main__":
    unittest.main()
