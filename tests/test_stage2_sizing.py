import contextlib
import importlib.util
import io
import json
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


TASKS, HORIZON = ["TaskA", "TaskB", "TaskC", "TaskD"], 20


def xr1_fixture(root, shift_atomic=1.5, seed=2):
    """Two cohorts, 4 tasks (B, D atomic), 12/12 retained per task; atomic failures score higher."""
    rng = np.random.default_rng(seed)
    norms = {s: {t: {"location": 0.0, "scale": 1.0} for t in TASKS} for s in ("0", "1", "2")}
    active = [30] + [30 - min(10, max(0, i - 8)) for i in range(1, HORIZON)]
    bundle = {"task_horizons": {t: HORIZON for t in TASKS}, "task_normalizations": norms,
              "time_risk_curves": {t: {"active_counts": active} for t in TASKS}}

    def cohort(label, base):
        rows = []
        for ti, task in enumerate(TASKS):
            for i in range(24):
                failed = i % 2 == 1
                length = HORIZON if failed else int(rng.integers(6, HORIZON - 1))
                shift = shift_atomic if failed and ti % 2 else 0.0
                safe = np.maximum.accumulate(rng.standard_normal(length) + shift * np.linspace(0, 1, length))
                rows.append({"rollout_id": f"{label}-{task}-{i}", "task_name": task,
                             "task_type": "atomic" if ti % 2 else "composite", "failed": failed, "horizon": HORIZON,
                             "environment_seed": base + 100 * ti + i, "environment_reset_index": 0,
                             "safe_scores": safe.tolist(), "time_scores": (np.arange(1, length + 1) / HORIZON).tolist()})
        return rows

    for name, base in (("cal", 1000), ("test", 5000)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cohort(name, base)))
    (root / "bundle.json").write_text(json.dumps(bundle))


def rldx_fixture(root, seed=0):
    rng = np.random.default_rng(seed)
    plan = []
    for ti, task in enumerate(TASKS):
        for split, n_s, n_f in (("train", 20, 18), ("calibration", 3, 0), ("evaluation", 5, 5)):
            for i in range(n_s + n_f):
                failed = i >= n_s
                length = 30 if failed else int(rng.integers(8, 27))
                plan.append(dict(rollout_id=f"{task}-{split}-{i}", task_name=task,
                                 task_type="atomic" if ti % 2 else "composite", failed=failed, split=split,
                                 length=length, environment_seed=1000 * ti + i))
    for s in (0, 1, 2):
        path = root / f"lstm_seed{s}" / "normalized_scores.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w") as stream:
            for p in plan:
                shift = 1.0 if p["failed"] and p["task_type"] == "atomic" else 0.0
                scores = (rng.standard_normal(p["length"]) + shift).tolist()
                stream.write(json.dumps({k: v for k, v in p.items() if k != "length"} | {"full_scores": scores}) + "\n")


def run_cli(module, argv):
    old = sys.argv
    sys.argv = ["stage2_sizing.py"] + [str(a) for a in argv]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            module.main()
    finally:
        sys.argv = old
    return buf.getvalue()


class TestPoolsAndPower(unittest.TestCase):
    def setUp(self):
        self.m = load("stage2_sizing")
        self.cl = load("confirm_lead")

    def rows(self, rng, n_task=3, per=12, shift=2.0):
        rows = []
        for t in range(n_task):
            for i in range(2 * per):
                failed = i % 2 == 1
                L = HORIZON if failed else int(rng.integers(6, HORIZON - 1))
                s = np.maximum.accumulate(rng.standard_normal(L) + (shift if failed else 0) * np.linspace(0, 1, L))
                rows.append(dict(task_name=f"T{t}", L=L, horizon=HORIZON, failed=failed, safe=s))
        return rows

    def test_null_failure_holds_running_max_to_horizon(self):
        r = dict(task_name="T", L=4, horizon=7, failed=False, safe=np.array([0.1, 0.5, 0.2, 0.4]))
        z = self.m.null_failure(r)
        self.assertEqual((z["L"], z["failed"]), (7, True))
        np.testing.assert_allclose(z["safe"], [0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def test_draw_sizes_and_full_dilution(self):
        rng = np.random.default_rng(0)
        pool = self.m.Pool(self.rows(rng), {"T0", "T1", "T2"})
        idx = pool.draw(rng, 7)
        self.assertEqual(len(idx), 3 * 14)
        C = pool.C.take(idx)
        self.assertEqual(int(C.failed.sum()), 21)
        self.assertFalse(any(i in set(np.concatenate([pool.idx[t][2] for t in pool.tasks])) for i in idx))
        full = pool.draw(rng, 7, pi=1.0)
        nulls = set(np.concatenate([pool.idx[t][2] for t in pool.tasks]).tolist())
        fails = [i for i in full if pool.C.failed[i]]
        self.assertTrue(all(i in nulls for i in fails))
        # Task list with repeats is honoured.
        self.assertEqual(len(pool.draw(rng, 2, tasks=["T0", "T0"])), 8)

    def test_resample_within_keeps_task_class_counts(self):
        rng = np.random.default_rng(1)
        pool = self.m.Pool(self.rows(rng), {"T0", "T1", "T2"})
        idx = pool.draw(rng, 5)
        b = self.m.resample_within(pool.C, idx, rng)
        key = lambda ii: sorted((pool.C.task[i], bool(pool.C.failed[i])) for i in ii)
        self.assertEqual(key(idx), key(b))
        self.assertTrue(set(b.tolist()) <= set(idx.tolist()))

    def test_power_rows_shift_and_criteria(self):
        g = np.array([0.04, 0.06, 0.08, 0.10])
        ok = np.array([True, True, False, True])
        p = self.m.power_rows(g, ok, hw_within=0.05, hw_tasks=0.09)
        self.assertEqual(p["point_only"], 0.75)
        self.assertEqual(p["ci_only"], 0.75)           # 0.06, 0.08, 0.10 exceed hw = 0.05
        self.assertEqual(p["ci_and_point"], 0.75)
        self.assertEqual(p["ci_point_fpr"], 0.5)       # 0.08 fails the FPR rule
        self.assertEqual(p["ci_only_task_ci"], 0.25)   # only 0.10 exceeds 0.09
        self.assertEqual(self.m.power_rows(np.array([0.05]), np.array([True]), 0.0, 0.0)["point_only"], 1.0)

    def test_simulation_signal_versus_null(self):
        rng = np.random.default_rng(3)
        rows = self.rows(rng, shift=3.0)
        tasks = {"T0", "T1", "T2"}
        rk = {t: 0.8 for t in tasks}
        pool = self.m.Pool(rows, tasks)
        out = self.m.simulate(pool, pool, rk, gains=(0.05,), sizes=(10, 60), R=40, workers=1,
                              nested_sizes=(10,), nested_outer=4, nested_inner=5)
        g10, g60 = (out["plug_in"][n]["within_task"]["mean_gain"] for n in ("10", "60"))
        self.assertGreater(g10, 0.02)
        # The expected gain rises with n (a less conservative margin) and the SD falls.
        self.assertGreater(g60, g10)
        self.assertLess(out["plug_in"]["60"]["within_task"]["sd"], out["plug_in"]["10"]["within_task"]["sd"])
        self.assertGreaterEqual(out["plug_in"]["10"]["with_tasks"]["sd"], 0)
        d = out["dilute"]["0.05"]["10"]
        self.assertAlmostEqual(d["pi"], 1 - 0.05 / g10, places=3)
        self.assertLess(d["summary"]["mean_gain"], g10)
        self.assertEqual(out["nested"]["10"]["outer"], 4)
        # Shift: the shifted sampling distribution is centred on the assumed gain.
        self.assertIn("ci_and_point", out["shift"]["0.05"]["60"])
        # Determinism: a second run gives identical numbers; fork workers match serial.
        again = self.m.simulate(pool, pool, rk, gains=(0.05,), sizes=(10,), R=40, workers=2,
                                nested_sizes=(), nested_outer=0, nested_inner=0)
        self.assertEqual(again["plug_in"]["10"], out["plug_in"]["10"])

    def test_study_matches_confirm_lead(self):
        rng = np.random.default_rng(4)
        rows = self.rows(rng)
        tasks = {"T0", "T1", "T2"}; rk = {t: 0.7 for t in tasks}
        pool = self.m.Pool(rows, tasks)
        ci, ti = pool.draw(rng, 6), pool.draw(rng, 6)
        res = self.m.study(pool, pool, rk, ci, ti)
        tm, fu = self.cl.fit_fast(pool.C.take(ci), rk)
        _, lP = self.cl.evaluate_fast(pool.C.take(ti), rk, tm); _, lF = self.cl.evaluate_fast(pool.C.take(ti), rk, fu)
        self.assertAlmostEqual(res["gain"], float(np.mean(lF - lP)))


class TestCostAndRetry(unittest.TestCase):
    def setUp(self):
        self.m = load("stage2_sizing")

    def test_attempts(self):
        rng = np.random.default_rng(0)
        a = self.m.attempts(0.9, 10, rng)
        self.assertTrue(np.all(a >= 20))
        self.assertAlmostEqual(a.mean(), 100, delta=6)      # dominated by 10 failures at 0.1
        b = self.m.attempts(0.5, 10, rng)
        self.assertTrue(20 <= b.mean() <= 30)
        self.assertTrue(np.all(np.isinf(self.m.attempts(1.0, 10, rng))))

    def ti(self, alarm_P):
        H = 100.0
        return dict(type="atomic", p=0.5, H=H, D=50.0,
                    fail_alarm=dict(none=np.array([H]), P=np.array([alarm_P]), safe_plus_P=np.array([alarm_P])),
                    fail_hit=dict(none=np.array([True]), P=np.array([True]), safe_plus_P=np.array([True])),
                    succ_L=np.array([50.0]), succ_alarm=dict(none=np.array([np.inf]), P=np.array([np.inf]),
                                                            safe_plus_P=np.array([np.inf])))

    def test_retry_monte_carlo_deterministic_cases(self):
        ti = self.ti(alarm_P=60.0)
        # 2 H: no monitor retries at H and a 50-step success still fits.
        self.assertEqual(self.m.retry_mc(ti, "none", 2.0, 1.0, (0,), M=2000)[0], 1.0)
        # 1.25 H: retry at H needs 150 > 125; an alarm at 60 leaves room (110 <= 125).
        s, p_hat = self.m.retry_mc(ti, "none", 1.25, 1.0, (0,), M=4000)
        self.assertEqual(s, p_hat)                                   # common random numbers: exact
        self.assertAlmostEqual(p_hat, 0.5, delta=0.03)
        self.assertEqual(self.m.retry_mc(ti, "P", 1.25, 1.0, (0,), M=2000)[0], 1.0)
        # A false alarm on every success at step 10 aborts it: with r = 0 nothing is won.
        fa = self.ti(alarm_P=60.0); fa["succ_alarm"]["P"] = np.array([10.0])
        self.assertEqual(self.m.retry_mc(fa, "P", 2.0, 0.0, (0,), M=500)[0], 0.0)

    def test_single_retry_bound_is_flat_at_two_horizons(self):
        out = self.m.e3_policy({"T": self.ti(alarm_P=60.0)}, (1.0, 1.25, 2.0), "1.0", 0)
        s = out["per_task"]["T"]["single"]
        self.assertEqual(s["2.0"]["none"], s["2.0"]["P"])          # D <= H: all leave room at 2 H
        self.assertEqual(s["2.0"]["none"], 0.5)                    # (1 - p) * r * 1
        self.assertEqual(s["1.25"]["none"], 0.0)
        self.assertEqual(s["1.25"]["P"], 0.5)
        self.assertEqual(s["1.0"]["P"], 0.0)                       # 100 - 60 < 50
        a = out["by_type"]["atomic"]["single"]["1.25"]
        self.assertEqual((a["timer_vs_none"], a["safe_vs_timer"]), (0.5, 0.0))
        self.assertTrue(math.isinf(out["by_type"]["all"]["episodes_per_arm_safe_vs_timer"]))
        m = out["per_task"]["T"]["multi"]
        self.assertEqual(m["1.0"]["none"], 0.0)                    # no retry fits: exactly p
        self.assertEqual(m["1.0"]["P"], 0.0)
        self.assertEqual(m["2.0"]["P"], m["2.0"]["none"])          # identical draws, both always fit


class TestRetryPilotSizing(unittest.TestCase):
    def setUp(self):
        self.m = load("stage2_sizing")

    def ti(self):
        H = 100.0
        return dict(type="atomic", p=0.5, H=H, D=50.0,
                    fail_alarm=dict(none=np.array([H]), P=np.array([60.0]), safe_plus_P=np.array([60.0])),
                    fail_hit=dict(none=np.array([True]), P=np.array([True]), safe_plus_P=np.array([True])),
                    succ_L=np.array([50.0]), succ_alarm=dict(none=np.array([np.inf]), P=np.array([np.inf]),
                                                            safe_plus_P=np.array([np.inf])))

    def test_retry_rate_modes(self):
        self.assertEqual(self.m.retry_rate("same", 0.6), 0.6)
        self.assertAlmostEqual(self.m.retry_rate("scale:0.504", 0.6), 0.3024)
        self.assertEqual(self.m.retry_rate("scale:3", 0.6), 1.0)
        self.assertEqual(self.m.retry_rate("0.245", 0.9), 0.245)

    def test_sample_size_formulas(self):
        # Connor (1987): p10 = 0.2, p01 = 0.1 -> (1.96 sqrt(.3) + 0.8416 sqrt(.29))^2 / .01 = 233.1
        self.assertEqual(self.m.n_mcnemar(0.2, 0.1), 234)
        self.assertTrue(math.isinf(self.m.n_mcnemar(0.1, 0.1)))
        self.assertEqual(self.m.n_unpaired(0.5, 0.6), self.m.n_unpaired(0.6, 0.5))
        self.assertTrue(math.isinf(self.m.n_unpaired(0.5, 0.5)))

    def test_late_hazard(self):
        rows = [dict(task_name="T", horizon=100, L=L, failed=f) for L, f in
                ((50, False), (80, False), (90, False), (100, True), (100, True), (70, False))]
        self.assertAlmostEqual(self.m.late_hazard(rows)["T"], 2 / 4)

    def test_reset_versus_continuation(self):
        out = self.m.e3_policy({"T": self.ti()}, (1.0, 2.0), "1.0", 0, c_grid=(0.0, 1.0), contrast_budget=1.25)
        self.assertIn("1.25", out["by_type"]["atomic"]["multi"])         # contrast budget added
        c0 = out["per_task"]["T"]["continuation"]["0.0"]["P"]
        self.assertEqual(c0["reset"], 1.0)                              # alarm at 60, retry of 50 fits in 125
        self.assertAlmostEqual(c0["cont"], 0.5, delta=0.02)             # only first-attempt successes
        self.assertAlmostEqual(c0["discordant"][0], 1 - c0["cont"])
        self.assertEqual(c0["discordant"][1], 0.0)
        c1 = out["by_type"]["atomic"]["continuation"]["1.0"]["P"]
        self.assertEqual((c1["reset"], c1["cont"], c1["delta"]), (1.0, 1.0, 0.0))
        self.assertTrue(math.isinf(c1["paired"]))
        # No retry ever succeeds: continuation at c = 1 wins every first-attempt failure.
        out = self.m.e3_policy({"T": self.ti()}, (1.0,), "0.0", 0, c_grid=(1.0,), contrast_budget=1.25)
        x = out["by_type"]["atomic"]["continuation"]["1.0"]["P"]
        self.assertAlmostEqual(x["delta"], -0.5, delta=0.02)
        self.assertEqual(x["p10"], 0.0)
        e = out["by_type"]["atomic"]["episodes"]["1.25"]["P_vs_none"]
        self.assertEqual(e["delta"], 0.0)

    def test_independent_retry_streams_share_the_first_attempt(self):
        ti = self.ti()
        a_won, a_first = self.m.retry_mc(ti, "P", 2.0, 0.5, (0,), M=4000, paths=True, stream=1)
        b_won, b_first = self.m.retry_mc(ti, "P", 2.0, 0.5, (0,), M=4000, paths=True, stream=2)
        np.testing.assert_array_equal(a_first, b_first)
        self.assertTrue(np.all(a_won[a_first] & b_won[b_first]))   # first-attempt successes win in both
        self.assertFalse(np.array_equal(a_won, b_won))              # retries differ
        _, c_first = self.m.retry_mc(ti, "P", 2.0, 0.5, (0,), M=4000, paths=True)
        _, d_first = self.m.retry_mc(ti, "none", 2.0, 0.5, (0,), M=4000, paths=True)
        np.testing.assert_array_equal(c_first, d_first)              # common random numbers (stream 0)
        # Paired n uses the common-random-number difference: zero difference -> no finite n.
        self.assertTrue(math.isinf(self.m.n_mcnemar_psi(0.3, 0.0)))
        self.assertEqual(self.m.n_mcnemar_psi(0.3, 0.1), self.m.n_mcnemar(0.2, 0.1))


class TestEndToEnd(unittest.TestCase):
    def test_cli_on_synthetic_cohorts(self):
        m = load("stage2_sizing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xr1_fixture(root)
            rldx_fixture(root / "rldx")
            common = ["--sizes", "10,20", "--replicates", "12", "--nested-sizes", "10", "--nested-outer", "3",
                      "--nested-inner", "4"]
            text = run_cli(m, ["e1b", "--calibration", root / "cal.jsonl", "--test", root / "test.jsonl",
                               "--runtime-bundle", root / "bundle.json", "--output", root / "e1b.json"] + common)
            self.assertIn("E1b (all pool)", text)
            out = json.loads((root / "e1b.json").read_text())
            self.assertEqual(out["tasks"], ["TaskB", "TaskD"])
            self.assertEqual(set(out["simulation"]["shift"]), {"0.05", "0.0695", "0.089"})
            self.assertEqual(out["pool_counts"]["calibration"]["TaskB"], dict(successes=12, failures=12))
            self.assertEqual(set(out["cost"]["totals"]), {"10", "20"})
            # Observed-gain check against a dose_response-shaped reference.
            ref = root / "dose.json"
            ref.write_text(json.dumps({"detectors": {"safe": {"by_type": {"atomic": {"stratum_calibration": {
                "gain": out["check"]["observed_gain"]}}}}}}))
            text = run_cli(m, ["e1b", "--calibration", root / "cal.jsonl", "--test", root / "test.jsonl",
                               "--runtime-bundle", root / "bundle.json", "--dose-response", ref, "--pool", "primary",
                               "--output", root / "e1b_primary.json"] + common)
            self.assertIn("MATCH", text)
            self.assertEqual(json.loads((root / "e1b_primary.json").read_text())["pool_counts"]["test"]["TaskD"],
                             dict(successes=10, failures=10))
            with self.assertRaises(SystemExit):
                run_cli(m, ["e1b", "--calibration", root / "cal.jsonl", "--test", root / "test.jsonl",
                            "--runtime-bundle", root / "bundle.json", "--output", root / "e1b.json"] + common)

            run_cli(m, ["e2", "--calibration-dir", root / "rldx", "--output", root / "e2.json"] + common)
            e2 = json.loads((root / "e2.json").read_text())
            self.assertEqual(e2["pool_counts"]["TaskA"], dict(successes=8, failures=5))
            self.assertEqual(set(e2["simulation"]["shift"]), {"0.034", "0.05"})
            self.assertAlmostEqual(e2["success_rates"]["TaskA"]["p"], 28 / 51, places=4)

            text = run_cli(m, ["e3", "--calibration", root / "cal.jsonl", "--test", root / "test.jsonl",
                               "--runtime-bundle", root / "bundle.json", "--rldx-calibration-dir", root / "rldx",
                               "--output", root / "e3.json"])
            e3 = json.loads((root / "e3.json").read_text())
            self.assertEqual(set(e3["policies"]), {"xr1", "rldx1"})
            for pol in e3["policies"].values():
                single2 = pol["by_type"]["all"]["single"]["2.0"]
                self.assertEqual(single2["none"], single2["P"])
                self.assertEqual(set(pol["by_type"]), {"atomic", "composite", "all"})
            self.assertIn("multi-retry", text)


if __name__ == "__main__":
    unittest.main()
