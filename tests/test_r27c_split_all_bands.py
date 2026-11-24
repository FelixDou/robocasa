"""R27c (scripts/xr1_lead_time/r27c_split_all_bands.py): every own band on R27's 30/70 assignments vs E, G and P."""
import copy
import math
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load
from tests.test_r21_native_bands import HELD, TASKS, rows

r27c = load("r27c_split_all_bands")
r27, r21 = r27c.r27, r27c.r21
SAFE = "safe_mlp"


class R27cTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(327)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        self.rk = {t: 0.5 for t in TASKS}
        every = set(TASKS)
        self.cal_tasks, self.test_tasks = {F: every - HELD[F] for F in HELD}, dict(HELD)
        # Hide-and-Seek reads the drifting score, the stand-in SAFE head a time ramp.
        self.keys = {r27.FULL: dict(unseen=lambda F: f"h~@{F}", seen=lambda F: "h~@seen"),
                     SAFE: dict(unseen=lambda F: f"z~@{F}", seen=lambda F: "z~@seen")}

    def references(self, splits=2):
        """R21's saved bootstrap points at the fixed split, and R27's saved rows (r25.vs_e on each assignment)."""
        base = r27c.evaluate_all(r27c.designs(self.cal, self.te, self.cal_tasks, self.test_tasks), self.keys, self.rk)
        timers = {}
        for h, by_side in base.items():
            for side, by_timer in by_side.items():
                for t, v in by_timer.items():
                    timers.setdefault(side, {}).setdefault(t, {})[h] = dict(
                        standalone=dict(fp=v["fp"], matched_gain=v["matched_gain"]))
        ref21 = dict(bootstrap={side: dict(timers=by) for side, by in timers.items()})
        saved = []
        for j in range(splits):
            d = r21.Design(self.cal, self.te, self.cal_tasks, self.test_tasks,
                           split=lambda rs, j=j: r27.split_30_70(rs, r27.SPLIT_SEED + j))
            v = r27.r25.vs_e(d, lambda F: f"h~@{F}", self.rk)["standalone"]
            saved.append(dict(matched=dict(fp=v["fp"], gain=v["matched_gain"])))
        return ref21, dict(splits=saved)

    def test_both_sides_share_the_assignment(self):
        split = lambda rs: r27.split_30_70(rs, r27.SPLIT_SEED)
        ds = r27c.designs(self.cal, self.te, self.cal_tasks, self.test_tasks, split=split)
        ids = lambda rs: [r["rollout_id"] for r in rs]
        self.assertEqual(ids(ds["unseen"].ref), ids(ds["seen"].ref))
        for d in ds.values():
            self.assertTrue(r27.check_split(d.cal_rows, d.ref, d.conf))
        self.assertEqual(ds["seen"].folds, [r27c.ALL])
        self.assertEqual(ds["unseen"].folds, sorted(HELD))

    def test_point_matches_r21_evaluate(self):
        d = r21.Design(self.cal, self.te, self.cal_tasks, self.test_tasks)
        p = r27c.point(d, lambda F: f"h~@{F}", ("G_abs", "G"), self.rk)
        ev = r21.evaluate(d, lambda F: f"h~@{F}", ("G_abs", "G"), self.rk)
        for t in ("G_abs", "G"):
            self.assertEqual(p[t]["fp"], ev[t]["standalone"]["fp"])
            self.assertEqual(p[t]["matched_gain"], r21.matched_gain(ev[t]["standalone"]))
            if ev[t]["standalone"]["matched"] is not None:
                self.assertLessEqual(p[t]["matched_timer_fp"], p[t]["fp"])

    def test_run_gates_and_summary(self):
        ref21, ref27 = self.references()
        res, problems = r27c.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.keys, ref21, ref27,
                                 splits=2)
        self.assertEqual(problems, [])
        for k, v in res["checks"].items():
            self.assertTrue(v["ok"], (k, v))
        self.assertEqual(res["checks"]["Q0_fixed_split_equals_r21"]["compared"], 2 * 3)    # 2 heads x (E, G, P)
        self.assertEqual(len(res["splits"]), 2)
        s = res["summary"][SAFE]
        self.assertEqual(set(s), {"unseen", "seen"})
        self.assertEqual(set(s["unseen"]), {"G_abs", "G"})
        self.assertEqual(set(s["seen"]), {"P"})
        self.assertEqual(s["seen"]["P"]["splits"], 2)
        # Gates fail on perturbed references.
        bad21 = copy.deepcopy(ref21)
        bad21["bootstrap"]["seen"]["timers"]["P"][SAFE]["standalone"]["matched_gain"] += 1e-9
        _, problems = r27c.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.keys, bad21, ref27,
                               splits=1)
        self.assertEqual(problems, ["Q0"])
        del bad21["bootstrap"]["unseen"]["timers"]["G"][SAFE]
        _, problems = r27c.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.keys, bad21, ref27,
                               splits=1)
        self.assertEqual(problems, ["Q0"])
        bad27 = copy.deepcopy(ref27)
        bad27["splits"][1]["matched"]["gain"] += 1e-9
        _, problems = r27c.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.keys, ref21, bad27,
                               splits=2)
        self.assertEqual(problems, ["Q1"])

    def test_summary_counts_and_reading(self):
        def row(g):
            v = dict(fp=3, fpr=0.05, mean_lead=0.2, matched_gain=g, matched_timer_fp=3, timer_fp=4)
            return dict(heads={"x": dict(seen=dict(P=v))})
        gains = [0.1] * 18 + [-0.1, math.nan]
        s = r27c.summarize([row(g) for g in gains])["x"]["seen"]["P"]
        self.assertEqual((s["positive"], s["negative"], s["splits"]), (18, 1, 20))
        self.assertTrue(s["clearly"]["earlier"])
        self.assertFalse(s["clearly"]["later"])
        self.assertEqual(s["matched_gain"]["n"], 19)


if __name__ == "__main__":
    unittest.main()
