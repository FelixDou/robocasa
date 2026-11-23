"""R27 (scripts/xr1_lead_time/r27_band_split.py): Hide-and-Seek's own band under the published 30/70 split."""
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load
from tests.test_r21_native_bands import HELD, SEEDS, TASKS, rows

r27 = load("r27_band_split")
r21, r25 = r27.r21, r27.r25


class R27Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(27)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        for r in self.cal + self.te:
            for F in HELD:
                for s in SEEDS:
                    r[f"{r27.FULL}~#{s}@{F}"] = r[f"h~#{s}@{F}"]
                r[f"{r27.FULL}~@{F}"] = r[f"h~@{F}"]
                r[f"{r27.FIRST}~@{F}"] = r[f"z~@{F}"]
        self.rk = {t: 0.5 for t in TASKS}
        every = set(TASKS)
        self.cal_tasks, self.test_tasks = {F: every - HELD[F] for F in HELD}, dict(HELD)
        self.timer = [(float(x), 0.5 * float(x)) for x in np.linspace(0.0, 0.3, 20)]

    def design(self, split=None):
        return r21.Design(self.cal, self.te, self.cal_tasks, self.test_tasks, split=split)

    def references(self):
        d = self.design()
        v = r25.vs_e(d, lambda F: f"{r27.FULL}~@{F}", self.rk)["standalone"]
        curves = {f"seed_{s}": r21.curve(d, lambda F, s=s: f"{r27.FULL}~#{s}@{F}") for s in SEEDS}
        full = r25.alarm_matrix(d, lambda F: f"{r27.FULL}~@{F}", (r27.ALPHA,))[0]
        abl = r25.alarm_matrix(d, lambda F: f"{r27.FIRST}~@{F}", r25.GRID)
        p = r25.paired(full, abl, np.arange(len(d.test)), r25.columns(d))
        ref21 = dict(bootstrap=dict(unseen=dict(timers={r27.TIMER: {r27.FULL: dict(standalone=dict(
            fp=v["fp"], matched_gain=v["matched_gain"]))}})), curves={r27.FULL: dict(unseen=curves)})
        ref25 = dict(ablations={r27.FIRST: dict(paired=p)})
        r18 = dict(timers=dict(unseen=dict(E=[dict(fpr=x, mean_lead=y) for x, y in self.timer])))
        return r18, ref21, ref25

    def test_split_is_a_stratified_30_70_partition(self):
        ref, conf = r27.split_30_70(self.cal, 27)
        self.assertTrue(r27.check_split(self.cal, ref, conf))
        for t in TASKS:                                     # 10 successes per task: 3 + 7
            self.assertEqual(sum(r["task_name"] == t for r in ref), 3)
            self.assertEqual(sum(r["task_name"] == t for r in conf), 7)
        same, _ = r27.split_30_70(self.cal, 27)
        other, _ = r27.split_30_70(self.cal, 28)
        ids = lambda rs: [r["rollout_id"] for r in rs]
        self.assertEqual(ids(ref), ids(same))
        self.assertNotEqual(ids(ref), ids(other))
        self.assertFalse(r27.check_split(self.cal, ref[1:], conf))
        self.assertEqual([r27.reference_count(n) for n in (2, 10, 20)], [1, 3, 6])

    def test_default_design_is_unchanged(self):
        d = self.design()
        ids = lambda rs: [r["rollout_id"] for r in rs]
        b = r21.rab.split_successes(self.cal)
        self.assertEqual(ids(d.ref), ids(b[0]))
        self.assertEqual(ids(d.conf), ids(b[1]))

    def test_versus_timer(self):
        timer = [(0.0, 0.0), (0.2, 0.2)]
        crossing = r27.versus_timer([(0.02, 0.0), (0.04, 0.04), (0.10, 0.20)], timer)     # meets the timer at 0.04
        self.assertAlmostEqual(crossing["last_not_above"], 0.04, places=3)
        self.assertAlmostEqual(crossing["above_from"], 0.04, places=3)
        above = r27.versus_timer([(0.05, 0.2), (0.10, 0.3)], timer)
        self.assertIsNone(above["last_not_above"])
        self.assertAlmostEqual(above["above_from"], 0.05)
        self.assertEqual(above["share_above"], 1.0)
        self.assertIsNone(r27.versus_timer([(0.3, 0.1), (0.4, 0.2)], timer)["overlap"])

    def test_missing_values_compare_equal(self):
        nan = float("nan")
        self.assertEqual(r27.value_diff(None, nan), 0.0)
        self.assertEqual(r27.value_diff(nan, nan), 0.0)
        self.assertEqual(r27.value_diff(None, 0.1), float("inf"))
        self.assertAlmostEqual(r27.value_diff(0.3, 0.1), 0.2)

    def test_one_shift_and_scale_leaves_a_single_models_alarms_unchanged(self):
        """Unseen tasks use one pooled location and scale per fold: a single model's band alarms do not move."""
        ref, conf = r27.split_30_70(self.cal, 27)
        raw = lambda rs: [np.asarray(r["h~#0@0"], float) for r in rs]
        moved = lambda rs: [2.5 * x + 0.7 for x in raw(rs)]
        te = self.te
        a = r21.band_alarms(raw(te), r21.band_fit(raw(ref), raw(conf), 0.1))
        b = r21.band_alarms(moved(te), r21.band_fit(moved(ref), moved(conf), 0.1))
        self.assertTrue(np.array_equal(a, b))

    def test_run_gates_and_summary(self):
        r18, ref21, ref25 = self.references()
        res, problems = r27.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, r18, ref21, ref25, splits=2)
        self.assertEqual(problems, [])
        for k, v in res["checks"].items():
            self.assertTrue(v["ok"], (k, v))
        self.assertEqual(res["summary"]["splits"], 2)
        self.assertEqual(len(res["splits"]), 2)
        self.assertEqual(res["summary"]["matched_gain"]["n"], 2)
        self.assertEqual(len(res["failure_alarms"]), sum(r["failed"] for r in self.te))
        for row in res["failure_alarms"]:
            self.assertTrue(row["band_alarm"] is None or 1 <= row["band_alarm"] <= row["L"])
            self.assertTrue(row["timer_alarm"] is None or 1 <= row["timer_alarm"] <= row["L"])
        ref25["ablations"][r27.FIRST]["paired"] = dict(ref25["ablations"][r27.FIRST]["paired"])
        ref25["ablations"][r27.FIRST]["paired"]["delta"] += 1e-9
        _, problems = r27.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, r18, ref21, ref25, splits=1)
        self.assertEqual(problems, ["K2"])


if __name__ == "__main__":
    unittest.main()
