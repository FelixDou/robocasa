"""R21 (scripts/xr1_lead_time/r21_native_bands.py) on synthetic folds: the band against the timers."""
import math
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load

r21, rx, rab, hk, cl, lfc = (load(n) for n in ("r21_native_bands", "r19_extras", "review_analyses_b", "hk_free",
                                                 "confirm_lead", "lead_fpr_curves"))

TASKS = ("TaskA", "TaskB", "TaskC", "TaskD")
HELD = {"0": {"TaskA"}, "1": {"TaskB"}, "2": {"TaskC"}, "3": {"TaskD"}}
H = {"TaskA": 30, "TaskB": 40, "TaskC": 50, "TaskD": 60}
SEEDS = ("0", "1", "2")
NORMS = {s: {t: dict(location=0.1 * i + 0.05 * int(s), scale=1.0 + 0.1 * i) for i, t in enumerate(TASKS)} for s in SEEDS}


def rows(rng, name, n_s=10, n_f=10):
    """Rollouts whose per-seed SAFE-like raw scores drift up on failures; keys as the R21 hooks attach them."""
    out = []
    for t in TASKS:
        for i in range(n_s + n_f):
            failed = i >= n_s
            L = H[t] if failed else int(rng.integers(8, H[t] - 3))
            base = np.cumsum(rng.normal(0.06 if failed else 0.0, 0.3, L))
            per_raw = {s: (base + rng.normal(0, 0.2, L)).tolist() for s in SEEDS}
            r = dict(rollout_id=f"{name}-{t}-{i:02d}", task_name=t, failed=failed, L=L, horizon=H[t],
                     per_seed_raw_safe=per_raw, timer=np.zeros(L))
            per = {s: (np.asarray(per_raw[s]) - NORMS[s][t]["location"]) / NORMS[s][t]["scale"] for s in SEEDS}
            for s in SEEDS:
                r[f"h~#{s}@seen"] = per[s]
                r[f"h#{s}@seen"] = np.maximum.accumulate(per[s])
            r["h~@seen"] = np.mean([per[s] for s in SEEDS], axis=0)
            for F in HELD:                       # pooled normalization: one shift per fold
                for s in SEEDS:
                    r[f"h~#{s}@{F}"] = per[s] + 0.01 * int(F)
                r[f"h~@{F}"] = np.mean([r[f"h~#{s}@{F}"] for s in SEEDS], axis=0)
                r[f"z~@{F}"] = np.arange(L, dtype=float) / 10.0
            r["z~@seen"] = np.arange(L, dtype=float) / 10.0
            out.append(r)
    return out


class R21Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        self.rk = {t: 0.5 for t in TASKS}
        every = set(TASKS)
        self.seen = r21.Design(self.cal, self.te, {"all": every}, {"all": every})
        self.unseen = r21.Design(self.cal, self.te, {F: every - HELD[F] for F in HELD}, dict(HELD))

    def test_evaluate_equals_r19_native_band(self):
        """Gate B0's premise: R21's evaluation (explicit halves, cohort takes) is R19's native_band."""
        for d, timers, key in ((self.seen, ("P",), "h~@seen"), (self.unseen, ("G_abs", "G"), None)):
            key_of = (lambda F: "h~@seen") if key else (lambda F: f"h~@{F}")
            ev = r21.evaluate(d, key_of, timers, self.rk, check=True)
            for t in timers:
                nb = rx.native_band(d.parts(), key_of, key_of, t, self.rk)
                self.assertEqual(r21.check_live(ev[t], nb), 0.0)
                self.assertTrue(all(ev[t]["N0_timer_equals_hk_fit"].values()))
                # The timer alone is R19's; the band alone is compared with the timer at the same target.
                self.assertEqual(ev[t]["timer_alone"]["fp"], nb["timer_alone"]["fp"])
                self.assertIn("gain", ev[t]["standalone"])
                m = ev[t]["standalone"]["matched"]
                if m is not None:
                    self.assertLessEqual(m["timer_fp"], ev[t]["standalone"]["fp"])

    def test_seen_band_equals_r12_band_analysis(self):
        """Gate N2's premise: on seen tasks, R21 reproduces R12 B2 (review_analyses_b.band_analysis, its own
        cohort code and P) at R12's 4 decimals; the matched gain differs only by R12's margin grid."""
        norms = {"safe": NORMS}
        rab.band_scores(self.cal, "safe", norms)
        rab.band_scores(self.te, "safe", norms)
        for r in self.cal + self.te:
            self.assertLess(np.max(np.abs(r["safe_band"]["avg"] - r["h~@seen"])), 1e-12)
        Ccal, Cte = cl.Cohort(self.cal, "timer"), cl.Cohort(self.te, "timer")
        b2 = rab.band_analysis(self.cal, self.te, Ccal, Cte, self.rk, "safe", norms, [r for r in self.te if r["failed"]])
        ev = r21.evaluate(self.seen, lambda F: "h~@seen", ("P",), self.rk)
        res = dict(heads={"safe_mlp": {"seen": {"P": {r21.ENSEMBLE: ev["P"]}}}})
        check = r21.check_r12(res, {"B2": {"safe": b2}}, names={"safe": "safe_mlp"})
        self.assertTrue(check["ok"], check)
        bad = dict(b2, standalone=dict(b2["standalone"], mean_lead=b2["standalone"]["mean_lead"] + 1e-3))
        self.assertFalse(r21.check_r12(res, {"B2": {"safe": bad}}, names={"safe": "safe_mlp"})["ok"])

    def test_draw_keeps_halves_tasks_and_classes(self):
        rng = np.random.default_rng(0)
        d = self.unseen
        ri, ci, ti = d.draw(rng)
        self.assertEqual(len(ri), len(d.ref))
        self.assertEqual(len(ci), len(d.conf))
        self.assertEqual(len(ti), len(d.test))
        ref_ids = {d.ref[i]["rollout_id"] for i in ri}
        conf_ids = {d.conf[i]["rollout_id"] for i in ci}
        self.assertFalse(ref_ids & conf_ids)
        self.assertEqual(sum(d.test[i]["failed"] for i in ti) * 2, len(ti))
        for F in d.folds:
            r, c, t = d.fold(F, (ri, ci, ti))
            self.assertFalse({d.ref[i]["task_name"] for i in r} & HELD[F])
            self.assertTrue({d.test[i]["task_name"] for i in t} <= HELD[F])

    def test_curve_and_run_gates(self):
        heads = {"h": {"seen": {r21.ENSEMBLE: lambda F: "h~@seen",
                                **{f"seed_{s}": (lambda F, s=s: f"h~#{s}@seen") for s in SEEDS}},
                       "unseen": {r21.ENSEMBLE: lambda F: f"h~@{F}",
                                  **{f"seed_{s}": (lambda F, s=s: f"h~#{s}@{F}") for s in SEEDS}}},
                 "z": {"seen": {r21.ENSEMBLE: lambda F: "z~@seen"}, "unseen": {r21.ENSEMBLE: lambda F: f"z~@{F}"}}}
        sides = dict(seen=dict(design=self.seen, timers=("P",)), unseen=dict(design=self.unseen, timers=("G_abs", "G")))
        r0 = {f"seed_{s}": r21.running_max_gap(self.cal + self.te, f"h~#{s}@seen", f"h#{s}@seen") for s in SEEDS}
        res, problems = r21.run(sides, heads, self.rk, B=3, contrasts=(("h", "z"),), r0=r0)
        self.assertEqual(problems, [], res["checks"])
        self.assertEqual(res["checks"]["B0_equals_r19_native_band"]["max_abs_diff"], 0.0)
        cv = res["curves"]["h"]["unseen"]
        self.assertEqual(len(cv[r21.ENSEMBLE]), len(r21.CURVE_ALPHAS))
        self.assertEqual(len(cv["mean_std_over_seeds"]), len(r21.CURVE_ALPHAS))
        self.assertIn("mean_std_over_seeds", res["heads"]["h"]["seen"]["P"])
        self.assertIn("fused_matched_gain", res["heads"]["h"]["seen"]["P"]["mean_std_over_seeds"])
        boot = res["bootstrap"]["unseen"]["timers"]["G_abs"]
        self.assertEqual(len(boot["h"]["fused"]["matched_ci95"]), 2)
        self.assertIn("h_minus_z", boot)
        self.assertEqual(res["bootstrap"]["seen"]["samples"], 3)
        # N1 against a reference built from R19's native_band; a changed reference fails.
        nb = {"h": dict(seen=rx.native_band(self.seen.parts(), heads["h"]["seen"][r21.ENSEMBLE],
                                            heads["h"]["seen"][r21.ENSEMBLE], "P", self.rk),
                        unseen={t: rx.native_band(self.unseen.parts(), heads["h"]["unseen"][r21.ENSEMBLE],
                                                  heads["h"]["unseen"][r21.ENSEMBLE], t, self.rk) for t in ("G_abs", "G")})}
        ok = r21.check_r19(res, nb)
        self.assertTrue(ok["ok"], ok)
        self.assertEqual(ok["compared"], 3)
        nb["h"]["unseen"]["G"]["fused"]["gain"] += 1e-9
        self.assertFalse(r21.check_r19(res, nb)["ok"])
        # R0 fails when a raw score is not the table's normalization.
        bad = dict(r0, seed_0=r21.running_max_gap(self.cal, "h~#1@seen", "h#0@seen"))
        _, problems = r21.run(dict(seen=sides["seen"]), {"z": {"seen": heads["z"]["seen"]}}, self.rk, B=0, r0=bad,
                              live=False)
        self.assertIn("R0", problems)

    def test_r21b_draw_variants(self):
        """R21b: the test variant keeps the calibration, the calibration variants keep the test set; `distinct`
        has no duplicate and the expected count of distinct rollouts; `tasks` takes each drawn task whole."""
        rng = np.random.default_rng(1)
        d = self.unseen
        ri, ci, ti = d.draw_test(rng)
        self.assertTrue(np.array_equal(ri, np.arange(len(d.ref))) and np.array_equal(ci, np.arange(len(d.conf))))
        self.assertEqual(len(ti), len(d.test))
        for mode in ("full", "rollouts", "distinct", "tasks"):
            ri, ci, ti = d.draw_calibration(rng, mode)
            self.assertTrue(np.array_equal(ti, np.arange(len(d.test))))
            if mode in ("full", "rollouts"):
                self.assertEqual((len(ri), len(ci)), (len(d.ref), len(d.conf)))
        ri, ci, _ = d.draw_calibration(rng, "distinct")
        self.assertEqual(len(set(ri.tolist())), len(ri))
        for t in TASKS:
            n = len(d.ref_by[t])
            self.assertEqual(int(np.sum(d.ref_task[ri] == t)), r21.distinct_count(n))
        self.assertEqual(r21.distinct_count(5), 3)
        self.assertEqual(r21.distinct_count(10), 7)
        ri, ci, _ = d.draw_calibration(rng, "tasks")
        for t in set(d.ref_task[ri]):
            k = int(np.sum(d.ref_task[ri] == t))
            self.assertEqual(k % len(d.ref_by[t]), 0)
        ri, _, _ = d.draw_calibration(rng, "rollouts")
        self.assertEqual(set(d.ref_task[ri]), set(TASKS))
        with self.assertRaises(ValueError):
            d.draw_calibration(rng, "other")

    def test_r21b_full_variant_replays_r21_bootstrap(self):
        """Gate D0: variant `full` with R21's seed reproduces R21's bootstrap exactly; a changed reference or a
        different number of replicates fails."""
        heads = {"h": {"seen": {r21.ENSEMBLE: lambda F: "h~@seen"}, "unseen": {r21.ENSEMBLE: lambda F: f"h~@{F}"}},
                 "z": {"seen": {r21.ENSEMBLE: lambda F: "z~@seen"}, "unseen": {r21.ENSEMBLE: lambda F: f"z~@{F}"}}}
        sides = dict(seen=dict(design=self.seen, timers=("P",)), unseen=dict(design=self.unseen, timers=("G_abs", "G")))
        ref = {side: r21.bootstrap(s["design"], {h: heads[h][side][r21.ENSEMBLE] for h in heads}, s["timers"], self.rk,
                                   B=4, contrasts=(("h", "z"),))[0] for side, s in sides.items()}
        res, arrays, problems = r21.diagnose(sides, heads, self.rk, B=4, reference=ref)
        self.assertEqual(problems, [], res["checks"])
        d0 = res["checks"]["D0_full_equals_r21_bootstrap"]
        self.assertEqual(d0["by_side"]["unseen"]["compared"], 4)          # 2 heads x (E, G); contrasts skipped
        self.assertEqual(set(res["sides"]["unseen"]), set(r21.VARIANTS))
        row = res["sides"]["unseen"]["test"]["G_abs"]["h"]
        self.assertEqual(row["fused_matched"]["replicates"] + row["fused_matched"]["failed"], 4)
        self.assertIn("spearman_fp_vs_matched", row)
        self.assertEqual(len(arrays["unseen/calibration_distinct/G/h/fused_matched"]), 4)
        bad = {s: dict(v) for s, v in ref.items()}
        bad["seen"] = dict(ref["seen"], timers={"P": {h: {p: dict(v, gain=v["gain"] + 1e-6) for p, v in by.items()}
                                                      if "_minus_" not in h else by
                                                      for h, by in ref["seen"]["timers"]["P"].items()}})
        self.assertIn("D0", r21.diagnose(sides, heads, self.rk, B=4, reference=bad, variants=("full",))[2])
        self.assertIn("D0", r21.diagnose(sides, heads, self.rk, B=3, reference=ref, variants=("full",))[2])

    def test_running_max_gap(self):
        self.assertEqual(r21.running_max_gap(self.te, "h~#0@seen", "h#0@seen"), 0.0)
        self.assertGreater(r21.running_max_gap(self.te, "h~#0@seen", "h#1@seen"), 0.0)
        r = dict(self.te[0])
        r["short"] = r["h#0@seen"][:-1]
        self.assertEqual(r21.running_max_gap([r], "h~#0@seen", "short"), math.inf)


if __name__ == "__main__":
    unittest.main()
