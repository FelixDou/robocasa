"""R9: Stage 2 sizing with the path frozen (E2) and the Hide-and-Seek reset contrast (E3)."""
import argparse
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

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parents[0] / "scripts" / "xr1_lead_time"
sys.path.insert(0, str(SCRIPTS))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts = load("stage2_fixture", TESTS / "test_stage2_sizing.py")
co = load("coprimary_fixture", TESTS / "test_e2_coprimary_sizing.py")
m = load("stage2_frozen_sizing", SCRIPTS / "stage2_frozen_sizing.py")
import confirm_hide_and_seek as chs  # noqa: E402
import confirm_lead as cl  # noqa: E402
import e2_coprimary_sizing as ec  # noqa: E402
import review_analyses as ra  # noqa: E402
import rldx_hide_and_seek as rhs  # noqa: E402
import stage2_sizing as s2  # noqa: E402

RK = co.RK


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = fn(*a)
    return status, buf.getvalue()


def write_hs(root, cal_dir, hs_shift=1.0):
    """Hide-and-Seek score files for the fixture; hs_shift = 0 copies SAFE exactly."""
    rows, _, _ = rhs.load_rows(cal_dir)
    rng = np.random.default_rng(5)
    sd = root / "hs"
    sd.mkdir()
    groups = dict(train=[r for r in rows if r["split"] == "train" and r["failed"]],
                  calibration=[r for r in rows if r["split"] == "calibration"],
                  test=[r for r in rows if r["split"] == "evaluation"])
    for name, rs in groups.items():
        with (sd / f"{name}_predictions.jsonl").open("w") as f:
            for r in rs:
                base = np.asarray(r["safe"], float)
                if hs_shift:
                    base = rng.standard_normal(r["L"]) + (hs_shift * np.linspace(0, 1, r["L"]) if r["failed"] else 0)
                for det in (chs.PRIMARY,) + chs.CONTROLS:
                    v = base if det == chs.PRIMARY else rng.standard_normal(r["L"])
                    f.write(json.dumps(dict(rollout_id=r["rollout_id"], detector=det, failed=r["failed"],
                                            task_name=r["task_name"], safe_scores=list(map(float, v)),
                                            time_scores=[0.0] * r["L"], per_seed_raw={"0": list(map(float, v))})) + "\n")
    return sd


def fixture(root, hs_shift=1.0):
    ts.rldx_fixture(root / "rldx")
    sd = write_hs(root, root / "rldx", hs_shift)
    ns = argparse.Namespace(calibration_dir=str(root / "rldx"), hs_scores_dir=str(sd))
    rows, train, cal_s, test, tasks, rk = m.load(ns)
    g = ec.replication_gains(cal_s, train, test, rk)
    (root / "rep.json").write_text(json.dumps({"primary": {"gain": g["safe"]}}))
    (root / "hs.json").write_text(json.dumps({"detectors": {"hide_and_seek": {"gain": g["hide_and_seek"]}}}))
    common = ["--calibration-dir", root / "rldx", "--hs-scores-dir", sd, "--replication-json", root / "rep.json",
              "--hs-result", root / "hs.json"]
    return [str(x) for x in common], (rows, train, cal_s, test, tasks, rk)


class LoosenAndCertifyTests(unittest.TestCase):
    def cohorts(self, seed=0):
        rows = co.held_rows(seed)
        rows[0] = dict(rows[0], L=co.H, safe=np.zeros(co.H))        # a success that runs to H
        return cl.Cohort(rows[:26], "safe"), cl.Cohort(rows[26:], "safe")

    def test_fast_loosen_equals_review_analyses(self):
        for seed in (0, 1, 2):
            c, e = self.cohorts(seed)
            for te in (c, e):
                n_s = int((~te.failed).sum())
                for target in range(n_s + 1):
                    a, b = m.loosen_fast(te, RK, target), ra.loosen_timer(te, RK, target)
                    self.assertEqual(a is None, b is None, (seed, target))
                    if a is not None:
                        self.assertEqual(a[0], b[0])
                        self.assertEqual(a[1], b[1])
                        np.testing.assert_array_equal(a[2], b[2])
        c, _ = self.cohorts()
        self.assertIsNone(m.loosen_fast(c, RK, 0))          # the success at H always alarms

    def test_frozen_path_at_the_adaptive_path_gives_the_adaptive_gain(self):
        c, e = self.cohorts(3)
        tm, fu = cl.fit_fast(c, RK)
        _, lP = cl.evaluate_fast(e, RK, tm)
        eF, lF = cl.evaluate_fast(e, RK, fu)
        row, matched = m.frozen_study(c, e, RK, fu, lP)
        self.assertEqual(row["lam"], fu["lam"])
        self.assertEqual(row["gain"], float(np.mean(lF - lP)))
        self.assertEqual(row["fp"], eF["fp"])
        loose = ra.loosen_timer(e, RK, eF["fp"])
        self.assertEqual(matched["gain"], float(np.mean(lF - loose[2])))

    def test_frozen_margin_ignores_calibration_failures(self):
        rows = co.held_rows(4)
        c0, e = cl.Cohort(rows[:26], "safe"), cl.Cohort(rows[26:], "safe")
        moved = [dict(r, safe=np.asarray(r["safe"]) + 5.0) if r["failed"] else r for r in rows[:26]]
        c1 = cl.Cohort(moved, "safe")
        path = dict(u=1.0, s=2.0)
        lP = np.zeros(int(e.failed.sum()))
        self.assertEqual(m.frozen_study(c0, e, RK, path, lP), m.frozen_study(c1, e, RK, path, lP))

    def test_infinite_margin_disables_the_monitor(self):
        c, e = self.cohorts(5)
        tm, _ = cl.fit_fast(c, RK)
        _, lP = cl.evaluate_fast(e, RK, tm)
        row, matched = m.frozen_study(c, e, RK, dict(u=-1e9, s=0.0), lP)   # every success alarms
        self.assertTrue(row["disabled"])
        self.assertEqual(row["fp"], 0)
        self.assertAlmostEqual(row["gain"], -float(np.mean(lP)))


class E2SimulationTests(unittest.TestCase):
    def setUp(self):
        rows = co.held_rows(6)
        self.pools = ec.build_pools(rows[:20], rows[20:], co.TASKS)
        cal = cl.Cohort(rows[:30], "safe")
        self.paths = {k: dict(u=p["u"], s=p["s"]) for k, p in
                      ((k, cl.fit_fast(cl.Cohort(rows[:30], k), RK)[1]) for k in m.KEYS)}
        self.assertTrue(all(math.isfinite(p["u"]) for p in self.paths.values()), cal)

    def test_adaptive_rows_are_e2_coprimary_rows(self):
        mine = m.simulate(self.pools, RK, self.paths, (3, 5), 20, 1, chunk=7)
        ref = ec.simulate_pair(self.pools, RK, (3, 5), 20, 1, chunk=7)
        for cell in ref:
            for a, b in zip(mine[cell], ref[cell]):
                for k in m.KEYS:
                    self.assertEqual(a[k], b[k])
                self.assertEqual(a["P_fpr"], b["P_fpr"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "co.json"
            p.write_text(json.dumps(ec.power_tables(ref, (3, 5)) | dict(replicates=20), default=float))
            self.assertTrue(m.check_coprimary(mine, (3, 5), p)["ok"])
            bad = json.loads(p.read_text())
            bad["summary"]["3"]["safe"]["within_task"]["mean_gain"] += 0.1
            p.write_text(json.dumps(bad))
            self.assertFalse(m.check_coprimary(mine, (3, 5), p)["ok"])

    def test_power_orderings(self):
        sims = m.simulate(self.pools, RK, self.paths, (4,), 40, 1)
        t = m.frozen_power(sims, (4,))
        for k in m.KEYS:
            for g, by_l in t["power"][k].items():
                p95, p975 = by_l["0.95"]["4"], by_l["0.975"]["4"]
                self.assertLessEqual(p975["ci_fpr_e2"], p95["ci_fpr_e2"])
                for p in (p95, p975):
                    self.assertLessEqual(p["both_fpr_e2"], p["ci_fpr_e2"])
                    self.assertLessEqual(p["ci_fpr_e2"], p["ci"])
        j = t["joint"]["plug_in"]["0.975"]["4"]
        for c in ("gain", "gain_and_matched"):
            self.assertLessEqual(j[c]["both"], min(j[c]["safe"], j[c]["hide_and_seek"]))
            self.assertGreaterEqual(j[c]["at_least_one"], max(j[c]["safe"], j[c]["hide_and_seek"]))
        rec = m.recommend({k: {"plug_in": {"0.975": {"20": dict(ci_fpr_e2=0.9, both_fpr_e2=0.5),
                                                      "30": dict(ci_fpr_e2=0.95, both_fpr_e2=0.85)}}} for k in m.KEYS},
                          (20, 30))
        self.assertEqual(rec["safe"]["ci_fpr_e2"]["reading"], "n = 20 stands")
        self.assertEqual(rec["safe"]["both_fpr_e2"]["reading"], "n = 30 needed")


class E3RuleTests(unittest.TestCase):
    def test_decision_rule(self):
        self.assertEqual(m.decide(dict(n_branching=900, hw_branching=0.01, hw_independent=0.04, scenes=1000))["decision"],
                         "TESTABLE")
        d = m.decide(dict(n_branching=5000, hw_branching=0.015, hw_independent=0.04, scenes=1000))
        self.assertEqual((d["decision"], d["recommend_branching"]), ("ESTIMATION", True))
        d = m.decide(dict(n_branching=math.inf, hw_branching=0.03, hw_independent=0.04, scenes=1000))
        self.assertEqual((d["decision"], d["recommend_branching"]), ("ESTIMATION", False))

    def test_identical_scores_give_identical_contrasts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, (rows, train, cal_s, test, tasks, rk) = fixture(root, hs_shift=0.0)
            ti, check = m.e3_inputs(rows, train, cal_s, test, tasks, rk)
            self.assertEqual(check["safe"], check["hide_and_seek"])
            res = {k: s2.e3_policy(ti[k], (1.25,), "0.245", 11) for k in m.KEYS}
            c = {k: m.contrast(res[k]["by_type"], (1.25,)) for k in m.KEYS}
            self.assertEqual(c["safe"], c["hide_and_seek"])
            x = c["safe"]["all"]["1.25"]
            self.assertEqual(x["scenes"], 50 * len(tasks))
            self.assertAlmostEqual(x["hw_independent"], round(1.959964 * math.sqrt(
                (x["p_P"] * (1 - x["p_P"]) + x["p_fused"] * (1 - x["p_fused"])) / x["scenes"]), 4), places=3)


class EndToEndTests(unittest.TestCase):
    def test_e2_cli_reproduces_coprimary_and_reports_frozen_power(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common, (rows, train, cal_s, test, tasks, rk) = fixture(root)
            sizes = ["--sizes", "3,5", "--replicates", "12"]
            ref_args = [a for a in common] + sizes + ["--output", str(root / "co.json")]
            self.assertEqual(quiet(ec.main, ref_args)[0], 0)
            status, text = quiet(m.main, ["e2"] + common + sizes + ["--coprimary-json", str(root / "co.json"),
                                                                   "--output", str(root / "e2.json")])
            self.assertEqual(status, 0, text)
            self.assertIn("adaptive rows reproduce e2_coprimary tables MATCH", text)
            out = json.loads((root / "e2.json").read_text())
            self.assertTrue(out["checks"]["coprimary_reproduced"]["ok"])
            self.assertEqual(set(out["frozen"]["power"]), set(m.KEYS))
            self.assertEqual(set(out["recommendation"]), set(m.KEYS))
            for k in m.KEYS:            # frozen once, on held-out successes and training failures
                _, fu = cl.fit_fast(cl.Cohort(cal_s + [r for r in train if r["failed"]], k), rk)
                self.assertEqual((out["frozen_paths"][k]["u"], out["frozen_paths"][k]["s"]), (fu["u"], fu["s"]))
            # A tampered reference fails the check, and the output says so.
            bad = json.loads((root / "co.json").read_text())
            bad["power"]["safe"]["plug_in"]["0.975"]["3"]["ci"] = -1
            (root / "bad.json").write_text(json.dumps(bad))
            status, text = quiet(m.main, ["e2"] + common + sizes + ["--coprimary-json", str(root / "bad.json"),
                                                                   "--output", str(root / "e2_bad.json")])
            self.assertEqual(status, 1)
            self.assertIn("FAILED CHECK", json.loads((root / "e2_bad.json").read_text())["status"])
            # A wrong replication gain stops before simulating.
            (root / "rep_bad.json").write_text(json.dumps({"primary": {"gain": 0.5}}))
            args = list(common)
            args[args.index(str(root / "rep.json"))] = str(root / "rep_bad.json")
            with self.assertRaises(SystemExit):
                quiet(m.main, ["e2"] + args + sizes + ["--output", str(root / "e2_stop.json")])
            self.assertFalse((root / "e2_stop.json").exists())

    def test_e3_cli_reproduces_stage2_sizing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common, _ = fixture(root)
            ts.xr1_fixture(root)
            s2m = ts.load("stage2_sizing")
            ts.run_cli(s2m, ["e3", "--calibration", root / "cal.jsonl", "--test", root / "test.jsonl",
                             "--runtime-bundle", root / "bundle.json", "--rldx-calibration-dir", root / "rldx",
                             "--retry-success", "0.245", "--output", root / "e3_ref.json"])
            status, text = quiet(m.main, ["e3"] + common + ["--retry-success", "0.245", "--reference-json",
                                                            str(root / "e3_ref.json"), "--output", str(root / "e3.json")])
            self.assertEqual(status, 0, text)
            self.assertIn("SAFE + P reproduces e3_ref.json MATCH", text)
            out = json.loads((root / "e3.json").read_text())
            run = out["runs"]["0.245"]
            self.assertTrue(run["shared_arms_identical"])
            self.assertIn(run["decision"]["decision"], ("TESTABLE", "ESTIMATION"))
            self.assertEqual(set(run["contrast"]["hide_and_seek"]), {"atomic", "composite", "all"})
            self.assertIn("0.0405", run["continuation"]["hide_and_seek"]["all"])
            # A reference at another retry rate is not silently skipped.
            status, text = quiet(m.main, ["e3"] + common + ["--retry-success", "0.17", "--reference-json",
                                                            str(root / "e3_ref.json"), "--output", str(root / "e3b.json")])
            self.assertEqual(status, 1)
            self.assertIn("STOP: no run matched", text)


if __name__ == "__main__":
    unittest.main()
