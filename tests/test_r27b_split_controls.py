"""R27b (scripts/xr1_lead_time/r27b_split_controls.py): random 50/50 controls, curves and the example's alarms."""
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load
from tests.test_r21_native_bands import HELD, SEEDS, TASKS, rows

r27b = load("r27b_split_controls")
r27, r21 = r27b.r27, r27b.r21


class R27bTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(127)
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
        timer = [(float(x), 0.5 * float(x)) for x in np.linspace(0.0, 0.3, 20)]
        self.r18 = dict(timers=dict(unseen=dict(E=[dict(fpr=x, mean_lead=y) for x, y in timer])))

    def r27_reference(self, splits=2):
        d = r21.Design(self.cal, self.te, self.cal_tasks, self.test_tasks)
        v = r27.r25.vs_e(d, lambda F: f"{r27.FULL}~@{F}", self.rk)["standalone"]
        curves = {f"seed_{s}": r21.curve(d, lambda F, s=s: f"{r27.FULL}~#{s}@{F}") for s in SEEDS}
        full = r27.r25.alarm_matrix(d, lambda F: f"{r27.FULL}~@{F}", (r27.ALPHA,))[0]
        abl = r27.r25.alarm_matrix(d, lambda F: f"{r27.FIRST}~@{F}", r27.r25.GRID)
        p = r27.r25.paired(full, abl, np.arange(len(d.test)), r27.r25.columns(d))
        ref21 = dict(bootstrap=dict(unseen=dict(timers={r27.TIMER: {r27.FULL: dict(standalone=dict(
            fp=v["fp"], matched_gain=v["matched_gain"]))}})), curves={r27.FULL: dict(unseen=curves)})
        res, problems = r27.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.r18, ref21,
                                dict(ablations={r27.FIRST: dict(paired=p)}), splits=splits)
        self.assertEqual(problems, [])
        return res

    def test_half_split_is_a_stratified_partition(self):
        ref, conf = r27b.split_half(self.cal, 127)
        self.assertTrue(r27b.check_half(self.cal, ref, conf))
        for t in TASKS:
            self.assertEqual(sum(r["task_name"] == t for r in ref), 5)
            self.assertEqual(sum(r["task_name"] == t for r in conf), 5)
        ids = lambda rs: [r["rollout_id"] for r in rs]
        self.assertEqual(ids(ref), ids(r27b.split_half(self.cal, 127)[0]))
        self.assertNotEqual(ids(ref), ids(r27b.split_half(self.cal, 128)[0]))
        self.assertFalse(r27b.check_half(self.cal, ref[1:], conf))

    def test_representative_and_standing(self):
        make = lambda g: dict(matched=dict(gain=g))
        self.assertEqual(r27b.representative([make(x) for x in (0.1, 0.5, 0.2, 0.9, 0.3)]), 4)
        self.assertEqual(r27b.representative([make(x) for x in (0.1, 0.3)]), 0)     # tie: the first
        self.assertEqual(r27b.standing(0.25, [make(x) for x in (0.1, 0.2, 0.3)], lambda r: r["matched"]["gain"]), 2)

    def test_run_gates_curves_and_example(self):
        ref = self.r27_reference()
        res, problems = r27b.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.r18, ref, splits=2)
        self.assertEqual(problems, [])
        for k, v in res["checks"].items():
            self.assertTrue(v["ok"], (k, v))
        self.assertEqual((len(res["published"]), len(res["half"])), (2, 2))
        self.assertEqual(len(res["published"][0]["mean_curve"]), len(r21.CURVE_ALPHAS))
        self.assertEqual(len(res["example"]["failure_alarms"]), sum(r["failed"] for r in self.te))
        self.assertIn("fixed_assignment_favorable", res["comparison"])
        ref["splits"][0]["matched"]["gain"] += 1e-9
        _, problems = r27b.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.rk, self.r18, ref, splits=1)
        self.assertEqual(problems, ["M1"])


if __name__ == "__main__":
    unittest.main()
