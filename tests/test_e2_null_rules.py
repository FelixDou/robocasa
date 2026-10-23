import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TASKS, H = ["TaskA", "TaskB", "TaskC", "TaskD"], 30
RK = {t: 0.7 for t in TASKS}


def held_rows(seed=0):
    """8 successes and 5 failures per task; failures score higher late (both detectors)."""
    rng = np.random.default_rng(seed)
    rows = []
    for task in TASKS:
        for i in range(13):
            failed = i >= 8
            L = H if failed else int(rng.integers(8, H - 3))
            ramp = np.linspace(0, 1, L)
            rows.append(dict(rollout_id=f"{task}-{i}", task_name=task, failed=failed, L=L, horizon=H,
                             safe=np.maximum.accumulate(rng.standard_normal(L) + (1.0 * ramp if failed else 0)),
                             hide_and_seek=np.maximum.accumulate(rng.standard_normal(L) + (2.0 * ramp if failed else 0))))
    return rows


def seed_pools(rows, seed=1):
    rng = np.random.default_rng(seed)
    return {t: [rng.standard_normal(sum(r["L"] for r in rows if r["task_name"] == t)) for _ in range(3)] for t in TASKS}


class E2NullRulesTests(unittest.TestCase):
    def setUp(self):
        self.m = load("e2_null_rules")
        self.cl = sys.modules["confirm_lead"]
        self.co = sys.modules["e2_coprimary_sizing"]

    def cohort(self, rows):
        return self.cl.Cohort(rows)

    def test_noise_and_resampled_scores(self):
        rows = held_rows()
        C = self.cohort(rows)
        rng = np.random.default_rng(0)
        R = self.m.noise_R(C, rng)
        valid = np.isfinite(C.R)
        self.assertTrue(np.array_equal(np.isfinite(R), valid))
        self.assertTrue(np.all(np.diff(np.where(valid, R, np.inf), axis=1)[valid[:, 1:]] >= 0))
        const = {t: [np.full(5, 2.0), np.full(5, 4.0)] for t in TASKS}
        Rr = self.m.resampled_R(C, rng, const)
        np.testing.assert_allclose(Rr[valid], 3.0)
        D = self.m.with_R(C, Rr)
        np.testing.assert_allclose(D.M, 3.0)
        np.testing.assert_array_equal(D.L, C.L)

    def test_tuned_P_matches_brute_force(self):
        rows = held_rows(2)
        e = self.cohort(rows)
        for fp_max in (0, 1, 2, 4, 10):
            lead, delta = self.m.tuned_P_lead(e, RK, fp_max)
            best = None
            for d in self.m.TUNE_GRID:
                ev, lv = self.cl.evaluate_fast(e, RK, dict(lam=float(d), u=math.inf, s=0.0))
                if ev["fp"] <= fp_max and (best is None or lv.mean() > best[1].mean()):
                    best = (float(d), lv)
            self.assertAlmostEqual(delta, best[0])
            np.testing.assert_allclose(lead, best[1])

    def test_boot_lower_and_mc_p(self):
        rng = np.random.default_rng(0)
        tasks = np.array(["a"] * 5 + ["b"] * 5)
        self.assertAlmostEqual(self.m.boot_lower(np.full(10, 0.02), tasks, rng), 0.02)
        d = rng.normal(0.01, 0.01, 10)
        self.assertLessEqual(self.m.boot_lower(d, tasks, rng), d.mean())
        refs = [0.0] * 39
        self.assertEqual(self.m.mc_p(0.1, refs), 1 / 40)
        self.assertEqual(self.m.mc_p(0.1, refs[:-1] + [0.1]), 2 / 40)     # one tie: still passes at 0.05
        self.assertEqual(self.m.mc_p(0.1, refs[:-2] + [0.2, 0.3]), 3 / 40)  # two above: fails

    def test_studies_reuse_coprimary_draws_and_rules_are_nested(self):
        rows = held_rows()
        pools = self.co.build_pools(rows[:20], rows[20:], TASKS)
        sp = seed_pools(rows)
        self.m.K_NULL = 19
        try:
            studies, nested = self.m.simulate(pools, RK, sp, (10,), 12, 6, 1, (10,), 2, 3, chunk=5)
        finally:
            self.m.K_NULL = 39
        ref = self.co.simulate_pair(pools, RK, (10,), 12, 1)[(10, "within_task")]
        for mine, theirs in zip(studies[10], ref):
            for key in ("safe", "hide_and_seek"):
                self.assertEqual(mine[key]["gain"], theirs[key]["gain"])
                self.assertEqual(mine[key]["fp"], theirs[key]["fp"])
        self.assertEqual(sum("refs" in r for r in studies[10]), 6)
        self.assertEqual(len(studies[10][0]["refs"]["noise"]), 19)
        out = self.m.analyse(studies, (10,))
        for name in self.m.NULLS:
            t = out["type_i"][name]["10"]
            self.assertLessEqual(t["ii"]["rate"], t["i"]["rate"])
            self.assertEqual(t["iii"]["studies"], 6)
        for key in self.m.CO:
            for g, by_n in out["power"][key].items():
                p = by_n["10"]
                self.assertLessEqual(p["ii"], p["i"])
                self.assertLessEqual(p["i"], p["ci_only"])
        res = self.m.nested_check(studies, nested, out)
        self.assertEqual(res["noise"]["10"]["studies"], 2)
        # Determinism, and fork workers agree with serial.
        again, _ = self.m.simulate(pools, RK, sp, (10,), 12, 0, 2, chunk=5)
        self.assertEqual([r["noise"] for r in again[10]], [r["noise"] for r in studies[10]])

    def test_recommendation_logic(self):
        def tab(ti, safe, hs, both):
            sizes = ("10", "20", "30")
            return dict(
                type_i={nm: {n: {r: dict(rate=ti[r][k]) for r in "i ii iii".split()} for k, n in enumerate(sizes)}
                        for nm in self.m.NULLS},
                power={"safe": {"0.034": {n: {r: safe[r][k] for r in ("i", "ii", "iii")} for k, n in enumerate(sizes)}},
                       "hide_and_seek": {"0.052": {n: {r: hs[r][k] for r in ("i", "ii", "iii")} for k, n in enumerate(sizes)}}},
                joint={"safe 0.034 / hs 0.052": {r: {n: dict(both=both[r][k]) for k, n in enumerate(sizes)} for r in ("i", "ii", "iii")}})
        o = tab(ti=dict(i=[0.3] * 3, ii=[0.01] * 3, iii=[0.04, 0.04, 0.06]),
                safe=dict(i=[1] * 3, ii=[0.5, 0.85, 0.9], iii=[0.9] * 3),
                hs=dict(i=[1] * 3, ii=[0.6, 0.9, 0.95], iii=[0.9] * 3),
                both=dict(i=[1] * 3, ii=[0.4, 0.8, 0.9], iii=[0.9] * 3))
        rec = self.m.recommend(o, (10, 20, 30))
        self.assertEqual(rec["eligible_rules"], ["ii"])     # (iii) fails type I at n = 30
        self.assertEqual((rec["rule"], rec["n"]), ("ii", 20))

    def test_seed_pools_from_rldx_files(self):
        from tests.test_rldx_lead_replication import write_fixture
        rhs = sys.modules["rldx_hide_and_seek"]
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d))
            rows, _, tasks = rhs.load_rows(d)
            held = [r for r in rows if r["split"] in ("calibration", "evaluation")]
            sp = self.m.seed_pools_from(d, held)
            self.assertEqual(sorted(sp), tasks)
            for t in tasks:
                self.assertEqual(len(sp[t]), 3)
                self.assertEqual(len(sp[t][0]), sum(r["L"] for r in held if r["task_name"] == t))


if __name__ == "__main__":
    unittest.main()
