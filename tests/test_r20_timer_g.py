"""R20 (scripts/xr1_lead_time/r20_timer_g.py) on synthetic folds: timer G against P and E."""
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load

r20, hk, cl = load("r20_timer_g"), load("hk_free"), load("confirm_lead")

TASKS = ("TaskA", "TaskB", "TaskC", "TaskD")
HELD = {"0": {"TaskA"}, "1": {"TaskB"}, "2": {"TaskC"}, "3": {"TaskD"}, "all": set()}
FOLDS = ("0", "1", "2", "3")
H = {"TaskA": 30, "TaskB": 40, "TaskC": 50, "TaskD": 60}


def rows(rng, name, n_s=10, n_f=10):
    out = []
    for t in TASKS:
        for i in range(n_s + n_f):
            failed = i >= n_s
            L = H[t] if failed else int(rng.integers(6, H[t] - 2))
            x = np.maximum.accumulate(np.cumsum(rng.normal(0.05 if failed else 0.0, 0.3, L)))
            r = dict(rollout_id=f"{name}-{t}-{i}", task_name=t, failed=failed, L=L, horizon=H[t], s=x)
            for F in FOLDS:
                r[f"s@{F}"] = x
            out.append(r)
    return out


class R20Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        self.rk = {"TaskA": 0.45, "TaskB": 0.5, "TaskC": 0.55, "TaskD": 0.6}

    def hk_timer(self, parts, timer):
        """hk_free.run_folds's timer alone, read through a real score key (the score is ignored)."""
        out = hk.run_folds(parts, timer, None, self.rk)["timer"]
        return dict(fp=out["fp"], fpr=out["fpr"], mean_lead=out["mean_lead"])

    def test_timer_alone_equals_hk_free(self):
        seen = {"all": (cl.Cohort(self.cal, "s"), cl.Cohort(self.te, "s"))}
        unseen = {F: (cl.Cohort([r for r in self.cal if r["task_name"] not in HELD[F]], f"s@{F}"),
                      cl.Cohort([r for r in self.te if r["task_name"] in HELD[F]], f"s@{F}")) for F in FOLDS}
        res, problems = r20.run(self.cal, self.te, HELD, FOLDS, self.rk, {})
        got = r20.alone_values(res)
        for (side, t), parts in ((("seen", "G"), seen), (("seen", "P"), seen),
                                 (("unseen", "G"), unseen), (("unseen", "G_abs"), unseen)):
            ref = self.hk_timer(parts, t)
            self.assertEqual(got[(side, t)]["fp"], ref["fp"], (side, t))
            self.assertAlmostEqual(got[(side, t)]["fpr"], ref["fpr"], places=14)
            self.assertAlmostEqual(got[(side, t)]["mean_lead"], ref["mean_lead"], places=14)
        # An empty reference compares nothing, so T0 cannot pass.
        self.assertEqual(problems, ["T0"])

    def test_gain_is_paired_and_matched_respects_budget(self):
        res, _ = r20.run(self.cal, self.te, HELD, FOLDS, self.rk, {})
        for side in ("seen", "unseen"):
            v = res[side]
            self.assertAlmostEqual(v["gain"], v["mean_lead"] - v["timer"]["mean_lead"], places=14)
            if v["matched"] is not None:
                self.assertLessEqual(v["matched"]["timer_fp"], v["fp"])
        self.assertEqual((res["seen"]["timer_name"], res["seen"]["comparator"]), ("G", "P"))
        self.assertEqual((res["unseen"]["timer_name"], res["unseen"]["comparator"]), ("G", "G_abs"))

    def test_against_itself(self):
        parts = r20.seen_parts(self.cal, self.te)
        v = r20.versus(parts, "G", "G", self.rk)
        self.assertEqual(v["gain"], 0.0)
        # The matched comparator is the test-tuned smallest margin within the same budget: never later.
        self.assertLessEqual(v["matched"]["gain"], 1e-15)

    def test_gate_t0(self):
        res, _ = r20.run(self.cal, self.te, HELD, FOLDS, self.rk, {})
        ref = {}
        for (side, t), v in r20.alone_values(res).items():
            ref.setdefault(side, {})[t] = dict(v)
        res2, problems = r20.run(self.cal, self.te, HELD, FOLDS, self.rk, ref)
        self.assertEqual(problems, [])
        self.assertEqual(res2["checks"]["T0_timers_equal_reference"]["compared"], 4)
        bad = {s: {t: dict(v) for t, v in by.items()} for s, by in ref.items()}
        bad["unseen"]["G_abs"]["mean_lead"] += 1e-9
        self.assertEqual(r20.run(self.cal, self.te, HELD, FOLDS, self.rk, bad)[1], ["T0"])
        bad = {s: {t: dict(v) for t, v in by.items()} for s, by in ref.items()}
        bad["seen"]["P"]["fp"] += 1
        self.assertEqual(r20.run(self.cal, self.te, HELD, FOLDS, self.rk, bad)[1], ["T0"])
        # A timer without a reference is reported, not compared (RLDX-1's seen G).
        partial = {s: {t: dict(v) for t, v in by.items() if (s, t) != ("seen", "G")} for s, by in ref.items()}
        res3, problems = r20.run(self.cal, self.te, HELD, FOLDS, self.rk, partial)
        self.assertEqual(problems, [])
        self.assertEqual(res3["checks"]["T0_timers_equal_reference"]["compared"], 3)

    def test_rounded_information(self):
        res, _ = r20.run(self.cal, self.te, HELD, FOLDS, self.rk, {})
        s = res["seen"]
        rounded = dict(fpr=round(s["fpr"], 4), mean_lead=round(s["mean_lead"], 4), gain_over_P=round(s["gain"], 4))
        res2, _ = r20.run(self.cal, self.te, HELD, FOLDS, self.rk, {}, rounded)
        self.assertTrue(res2["checks"]["seen_G_vs_R16_rounded"]["matches"])
        rounded["mean_lead"] += 0.01
        res3, _ = r20.run(self.cal, self.te, HELD, FOLDS, self.rk, {}, rounded)
        self.assertFalse(res3["checks"]["seen_G_vs_R16_rounded"]["matches"])

    def test_rows_are_not_mutated(self):
        before = [sorted(r) for r in self.cal + self.te]
        r20.run(self.cal, self.te, HELD, FOLDS, self.rk, {})
        self.assertEqual(before, [sorted(r) for r in self.cal + self.te])


if __name__ == "__main__":
    unittest.main()
