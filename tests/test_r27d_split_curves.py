"""R27d (scripts/xr1_lead_time/r27d_split_curves.py): Fig. 2's own-band curves on R27's 30/70 assignments."""
import copy
import math
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load
from tests.test_r21_native_bands import HELD, SEEDS, TASKS, rows

r27d = load("r27d_split_curves")
r27, r21 = r27d.r27, r27d.r21
SAFE = "safe_mlp"


class R27dTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(427)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        for r in self.cal + self.te:                 # a second head: a time ramp per seed
            for s in SEEDS:
                r[f"z~#{s}@seen"] = r["z~@seen"] + 0.05 * int(s)
                for F in HELD:
                    r[f"z~#{s}@{F}"] = r[f"z~@{F}"] + 0.05 * int(s)
        self.rk = {t: 0.5 for t in TASKS}
        every = set(TASKS)
        self.cal_tasks, self.test_tasks = {F: every - HELD[F] for F in HELD}, dict(HELD)
        at = dict(seen=lambda F: "seen", unseen=lambda F: F)
        self.keys = {h: {side: {f"seed_{s}": (lambda F, p=p, s=s, at=at[side]: f"{p}~#{s}@{at(F)}") for s in SEEDS}
                         for side in at} for h, p in ((r27.FULL, "h"), (SAFE, "z"))}
        line = lambda k: [dict(fpr=float(x), mean_lead=k * float(x)) for x in np.linspace(0.0, 0.3, 20)]
        self.r18 = dict(timers=dict(seen=dict(P=line(0.5)), unseen=dict(E=line(0.4), G=line(0.8))))

    def references(self, splits=2):
        """R21's saved single-model curves at the fixed split, and R27b's saved Hide-and-Seek curves on each 30/70
        assignment (the mean of the three models at each target)."""
        ds = r27d.r27c.designs(self.cal, self.te, self.cal_tasks, self.test_tasks)
        curves = {h: {side: {lab: r21.curve(ds[side], k) for lab, k in by_seed.items()} for side, by_seed in by.items()}
                  for h, by in self.keys.items()}
        published = []
        for j in range(splits):
            d = r21.Design(self.cal, self.te, self.cal_tasks, self.test_tasks,
                           split=lambda rs, j=j: r27.split_30_70(rs, r27.SPLIT_SEED + j))
            cv = {f"seed_{s}": r21.curve(d, lambda F, s=s: f"h~#{s}@{F}") for s in SEEDS}
            first = cv["seed_0"]
            published.append(dict(mean_curve=[dict(alpha=first[i]["alpha"], fpr=x, mean_lead=y)
                                              for i, (x, y) in enumerate(r27.mean_curve(cv))]))
        return dict(curves=curves), dict(published=published)

    def test_mean_curve_is_r27s(self):
        d = r21.Design(self.cal, self.te, self.cal_tasks, self.test_tasks)
        single, mean = r27d.head_curves(d, self.keys[r27.FULL]["unseen"])
        self.assertEqual([(q["fpr"], q["mean_lead"]) for q in mean], r27.mean_curve(single))
        self.assertEqual([q["alpha"] for q in mean], list(r21.CURVE_ALPHAS))

    def test_median_curve_and_target(self):
        mk = lambda k, lo: [dict(alpha=a, fpr=lo + a, mean_lead=k * (lo + a)) for a in (0.01, 0.05, 0.1)]
        curves = [mk(1.0, 0.0), mk(2.0, 0.0), mk(3.0, 0.02)]
        med = r27d.median_curve(curves, grid=np.array([0.01, 0.03, 0.06, 0.12]), min_n=3)
        self.assertEqual(med["fpr"], [0.03, 0.06])              # 0.01: the third curve starts at 0.03; 0.12: beyond two
        self.assertAlmostEqual(med["median"][0], 2.0 * 0.03)
        self.assertAlmostEqual(med["min"][1], 0.06)
        self.assertAlmostEqual(med["max"][1], 0.18)
        tp = r27d.target_point(curves)
        self.assertAlmostEqual(tp["fpr"], 0.05)
        self.assertAlmostEqual(tp["mean_lead"], 0.1)
        self.assertEqual(r27d.median_curve(curves, grid=np.array([0.5]))["fpr"], [])

    def test_run_gates_and_summary(self):
        ref21, ref27b = self.references()
        res, problems = r27d.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.keys, self.r18, ref21,
                                 ref27b, splits=2)
        self.assertEqual(problems, [])
        n = len(r21.CURVE_ALPHAS)
        self.assertEqual(res["checks"]["D0_fixed_split_curves_equal_r21"]["compared"], 2 * 2 * 3 * n)
        self.assertEqual(res["checks"]["D1_hide_and_seek_curves_equal_r27b"]["compared"], 2 * n)
        self.assertEqual(len(res["splits"]), 2)
        s = res["summary"][SAFE]
        self.assertEqual(set(s["seen"]["versus"]), {"P"})
        self.assertEqual(set(s["unseen"]["versus"]), {"E", "G"})
        self.assertEqual(set(res["fixed_50_50"]["summary"][SAFE]["unseen"]["versus"]), {"E", "G"})
        # Gates fail on perturbed or missing references.
        bad21 = copy.deepcopy(ref21)
        bad21["curves"][SAFE]["seen"]["seed_1"][3]["mean_lead"] += 1e-9
        self.assertEqual(r27d.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.keys, self.r18,
                                  bad21, ref27b, splits=2)[1], ["D0"])
        del bad21["curves"][SAFE]["unseen"]
        self.assertEqual(r27d.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.keys, self.r18,
                                  bad21, ref27b, splits=2)[1], ["D0"])
        bad27b = copy.deepcopy(ref27b)
        bad27b["published"][1]["mean_curve"][2]["fpr"] += 1e-9
        self.assertEqual(r27d.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.keys, self.r18,
                                  ref21, bad27b, splits=2)[1], ["D1"])

    def test_versus_reports_the_gap(self):
        timer = [(0.0, 0.0), (0.2, 0.1)]
        above = r27d.versus([(0.0, 0.05), (0.2, 0.15)], timer)
        self.assertAlmostEqual(above["gap_min"], 0.05)
        self.assertIsNone(above["last_not_above"])
        crossing = r27d.versus([(0.0, -0.05), (0.15, 0.1)], timer)
        self.assertGreater(crossing["above_from"], 0.0)
        self.assertTrue(math.isfinite(crossing["gap_max"]))
        self.assertIsNone(r27d.versus([(0.1, 0.1)], timer)["overlap"])


if __name__ == "__main__":
    unittest.main()
