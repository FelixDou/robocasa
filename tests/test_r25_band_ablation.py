"""R25 (scripts/xr1_lead_time/r25_band_ablation.py) on R21's synthetic folds: the full band against ablations."""
import math
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load
from tests.test_r21_native_bands import HELD, TASKS, rows

r21, r25 = (load(n) for n in ("r21_native_bands", "r25_band_ablation"))


class R25Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        self.rk = {t: 0.5 for t in TASKS}
        every = set(TASKS)
        self.d = r21.Design(self.cal, self.te, {F: every - HELD[F] for F in HELD}, dict(HELD))
        # The full band reads h; the "first step" a time-only score (z); the "zero input" is h itself.
        self.keys = {r25.FULL: lambda F: f"h~@{F}", "hide_and_seek_first": lambda F: f"z~@{F}",
                     "hide_and_seek_zero": lambda F: f"h~@{F}"}

    def reference(self, B):
        point = r21.evaluate(self.d, self.keys[r25.FULL], (r25.TIMER,), self.rk)[r25.TIMER]
        ref21 = {p: dict(fp=point[p]["fp"], matched_gain=r21.matched_gain(point[p])) for p in r21.PARTS}
        summary, reps, _ = r21.diagnostic(self.d, {r25.FULL: self.keys[r25.FULL]}, (r25.TIMER, "G"), self.rk, B=B,
                                          variants=("test",))
        return ref21, {s: reps["test"][r25.TIMER][r25.FULL][s] for s in r25.R21B_STATS}

    def test_alarms_and_paired_difference(self):
        """The precomputed alarms give evaluate()'s results; an ablation equal to the full band differs by 0
        at the same alpha, and matched to the full band's false alarms it can only lead more (Delta <= 0)."""
        cols = r25.columns(self.d)
        full = r25.alarm_matrix(self.d, self.keys[r25.FULL], (r25.ALPHA,))[0]
        ev = r21.evaluate(self.d, self.keys[r25.FULL], (r25.TIMER,), self.rk)[r25.TIMER]["standalone"]
        fa, lead = r25.rates(full[None, :], np.arange(len(self.d.test)), cols)
        self.assertEqual(int(fa[0]), ev["fp"])
        self.assertAlmostEqual(float(lead[0].mean()), ev["mean_lead"], places=12)
        same = r25.alarm_matrix(self.d, self.keys["hide_and_seek_zero"], r25.GRID)
        ident = np.arange(len(self.d.test))
        p = r25.paired(full, same, ident, cols)
        self.assertEqual(p["same_alpha"]["delta"], 0.0)
        self.assertEqual(p["same_alpha"]["ablation_fp"], p["budget"])
        self.assertLessEqual(p["matched_fp"], p["budget"])
        self.assertLessEqual(p["delta"], 0.0)
        # A bootstrap pick with repeats: the budget is the picked full band's false alarms.
        pick = self.d.draw_test(np.random.default_rng(3))[2]
        fa_p, _ = r25.rates(full[None, :], pick, cols)
        self.assertEqual(r25.paired(full, same, pick, cols)["budget"], int(fa_p[0]))
        # No grid band within the budget: no match.
        never = np.zeros_like(same)                          # every band alarms at the first inference
        self.assertIsNone(r25.paired(full, never, ident, cols)["matched_fp"])
        self.assertTrue(math.isnan(r25.paired(full, never, ident, cols)["delta"]))

    def test_run_gates(self):
        """G1 and G2 against R21's point and R21b's `test` replicates; G3; a changed reference fails."""
        B = 3
        ref21, reps = self.reference(B)
        res, problems = r25.run(self.d, self.keys, self.rk, ref21=ref21, r21b=reps, r0={"x": 0.0}, B=B)
        self.assertEqual(problems, [], res["checks"])
        self.assertEqual(set(res["heads"]), set(self.keys))
        self.assertEqual(set(res["ablations"]), {"hide_and_seek_first", "hide_and_seek_zero"})
        z = res["ablations"]["hide_and_seek_zero"]
        self.assertEqual(z["paired"]["same_alpha"]["delta"], 0.0)
        self.assertEqual(z["same_alpha_delta_ci"]["ci95"], [0.0, 0.0])
        self.assertEqual(res["samples"], B)
        bad = {p: dict(v) for p, v in ref21.items()}
        bad["standalone"]["matched_gain"] += 1e-9
        self.assertIn("G1", r25.run(self.d, self.keys, self.rk, ref21=bad, r21b=reps, r0={"x": 0.0}, B=B)[1])
        other = {s: v + 1e-9 for s, v in reps.items()}
        self.assertIn("G2", r25.run(self.d, self.keys, self.rk, ref21=ref21, r21b=other, r0={"x": 0.0}, B=B)[1])
        self.assertIn("G0", r25.run(self.d, self.keys, self.rk, ref21=ref21, r21b=reps, r0={"x": 1e-6}, B=B)[1])

    def test_nan_equal_diff(self):
        self.assertEqual(r25.nan_equal_diff([1.0, math.nan], [1.0, math.nan]), 0.0)
        self.assertEqual(r25.nan_equal_diff([1.0, 2.0], [1.0, math.nan]), math.inf)
        self.assertEqual(r25.nan_equal_diff([1.0], [1.0, 2.0]), math.inf)
