"""R19 (scripts/xr1_lead_time/r19_extras.py) on synthetic folds: bands, bootstrap, leads, saturation."""
import math
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load

rx, hk, cl = load("r19_extras"), load("hk_free"), load("confirm_lead")

TASKS = ("TaskA", "TaskB", "TaskC", "TaskD")
HELD = {"0": {"TaskA"}, "1": {"TaskB"}, "2": {"TaskC"}, "3": {"TaskD"}}
H = {"TaskA": 30, "TaskB": 40, "TaskC": 50, "TaskD": 60}
TYPES = {"TaskA": "atomic", "TaskB": "atomic", "TaskC": "composite", "TaskD": "composite"}


def rows(rng, name, n_s=8, n_f=8):
    out = []
    for t in TASKS:
        for i in range(n_s + n_f):
            failed = i >= n_s
            L = H[t] if failed else int(rng.integers(8, H[t] - 3))
            x = np.cumsum(rng.normal(0.05 if failed else 0.0, 0.3, L))
            r = dict(rollout_id=f"{name}-{t}-{i}", task_name=t, failed=failed, L=L, horizon=H[t])
            for F in HELD:
                r[f"s~@{F}"] = x + 0.01 * int(F)
                r[f"s@{F}"] = np.maximum.accumulate(r[f"s~@{F}"])
                r[f"z@{F}"] = np.maximum.accumulate(np.arange(L, dtype=float) / 10.0)
            out.append(r)
    return out


class R19Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        self.rk = {t: 0.5 for t in TASKS}
        self.design = rx.UnseenDesign(self.cal, self.te, HELD, sorted(HELD))

    def test_point_estimate_is_hk_free_block(self):
        """Gate B0's premise: the un-resampled run equals hk_free.block's seed average."""
        key_of = lambda F: f"s@{F}"
        pairs = lambda lab: {F: (cl.Cohort([r for r in self.cal if r["task_name"] not in HELD[F]], key_of(F)),
                                 cl.Cohort([r for r in self.te if r["task_name"] in HELD[F]], key_of(F))) for F in HELD}
        ref = hk.block(pairs, ["seed_average"], ("G_abs", "G"), None, self.rk)
        boot, point = rx.boot_unseen(self.design, {"s": key_of, "z": lambda F: f"z@{F}"}, ("G_abs", "G"), self.rk, B=4,
                                     contrasts=(("s", "z"),))
        check = rx.check_point({t: {"s": point[t]["s"]} for t in point},
                               {t: {"s": ref[t]["seed_average"]} for t in ("G_abs", "G")})
        self.assertTrue(check["ok"], check)
        for t in ("G_abs", "G"):
            v = boot["timers"][t]["s"]
            self.assertEqual(len(v["gain_ci95"]), 2)
            self.assertIn("s_minus_z", boot["timers"][t])
        # A changed reference fails the gate.
        bad = {t: {"s": dict(ref[t]["seed_average"], gain=ref[t]["seed_average"]["gain"] + 1e-6)} for t in ("G_abs", "G")}
        self.assertFalse(rx.check_point({t: {"s": point[t]["s"]} for t in point}, bad)["ok"])

    def test_draw_keeps_task_and_class_sizes(self):
        rng = np.random.default_rng(0)
        cal_pick, te_pick = self.design.draw(rng)
        self.assertEqual(len(cal_pick), len(self.cal))
        self.assertEqual(len(te_pick), len(self.te))
        pairs = self.design.pairs(lambda F: f"s@{F}", cal_pick, te_pick)
        for F, (c, e) in pairs.items():
            self.assertTrue(set(e.task) <= HELD[F])
            self.assertFalse(set(c.task) & HELD[F])

    def test_native_band_and_timers(self):
        parts = {F: ([r for r in self.cal if r["task_name"] not in HELD[F]], [r for r in self.te if r["task_name"] in HELD[F]])
                 for F in HELD}
        out = rx.native_band(parts, lambda F: f"s~@{F}", lambda F: f"s@{F}", "G_abs", self.rk)
        self.assertTrue(out["checks"]["ok"])
        f = out["fused"]
        self.assertEqual(f["n_success"], 8 * len(TASKS))
        self.assertEqual(f["n_failure"], 8 * len(TASKS))
        # The fused alarm is never later than the timer at alpha/2, which is never earlier than the timer at alpha.
        self.assertGreaterEqual(f["mean_lead"] + 1e-12, 0.0)
        self.assertIn("gain", f)
        for F, i in out["folds"].items():
            self.assertLessEqual(i["timer_lam"], i["timer_lam_half"] + 1e-12)
            self.assertGreater(i["reference"], 0)

    def test_leads_seconds_per_task_and_limit(self):
        pairs = self.design.pairs(lambda F: f"s@{F}")
        d = rx.leads_detail(pairs, "G_abs", self.rk)
        ref = hk.run_folds(pairs, "G_abs", None, self.rk)
        self.assertAlmostEqual(float(np.mean(d["fused"] - d["timer"])), ref["gain"], places=12)
        self.assertAlmostEqual(float(np.mean(d["fused"] - d["matched"])), ref["matched"]["gain"], places=12)
        sec = rx.seconds(d, TYPES, 0.5)
        self.assertAlmostEqual(sec["all"]["gain_seconds"], float(np.mean((d["fused"] - d["timer"]) * d["H"] * 0.5)), places=12)
        self.assertEqual(sec["atomic"]["failures"] + sec["composite"]["failures"], sec["all"]["failures"])
        pt = rx.per_task(d)["same_target"]
        self.assertEqual(pt["tasks"], len(TASKS))
        self.assertEqual(pt["above_zero"] + pt["below_zero"] + pt["at_zero"], len(TASKS))
        lim = rx.timer_at_limit(d)
        self.assertEqual(lim["at_limit"], d["timer_pooled"]["final"])
        self.assertLessEqual(lim["bacc_limit_as_miss"], lim["bacc"] + 1e-12)

    def test_saturation(self):
        flat = [dict(r, k=np.zeros(r["L"])) for r in self.cal]
        s = rx.saturation(flat, "k")
        self.assertEqual(s["all"]["distinct_maxima"], 1)
        self.assertEqual(s["all"]["largest_tie_share"], 1.0)
        varied = rx.saturation(self.cal, "s@0")
        self.assertGreater(varied["all"]["distinct_maxima"], 10)


if __name__ == "__main__":
    unittest.main()
