"""R10 part A: E2's frozen decision rule (scripts/xr1_lead_time/e2_frozen_rule.py)."""
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


fz = load("frozen_fixture", TESTS / "test_stage2_frozen_sizing.py")
co = load("coprimary_fixture_r10", TESTS / "test_e2_coprimary_sizing.py")
m = load("e2_frozen_rule", SCRIPTS / "e2_frozen_rule.py")
import confirm_lead as cl  # noqa: E402
import e2_coprimary_sizing as ec  # noqa: E402
import e2_null_rules as en  # noqa: E402
import fixed_path as fp  # noqa: E402
import stage2_frozen_sizing as fs  # noqa: E402

RK = co.RK


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = fn(*a)
    return status, buf.getvalue()


def det_row(gain, matched, fp_=0, n_s=400):
    return ({"gain": gain, "fp": fp_, "n_s": n_s}, {"gain": matched, "fp": fp_, "n_s": n_s})


class RuleLogicTests(unittest.TestCase):
    def rows(self):
        rng = np.random.default_rng(0)
        out = []
        for i in range(200):
            g = 0.03 + 0.01 * rng.standard_normal()
            a, b = det_row(g, g - 0.002, fp_=50 if i % 10 == 0 else 10)
            r = {"safe": a, "safe@matched": b}
            if i % 4:
                r["refs"] = {k: list(rng.normal(0.0, 0.005, m.K_NULL)) for k in m.NULLS}
            out.append(r)
        return out

    def test_full_rule_is_the_conjunction_and_refs_do_not_shift(self):
        rows = self.rows()
        x = m.masks(rows, "safe")
        np.testing.assert_array_equal(x["full"], x["a"] & x["b"] & x["c"] & x["d"])
        np.testing.assert_array_equal(x["no_d"], x["a"] & x["b"] & x["c"])
        np.testing.assert_array_equal(x["no_b"], x["a"] & x["c"] & x["d"])
        self.assertFalse(x["d"][~x["has"]].any())                  # no references, no pass
        self.assertFalse(x["c"][::10].any())                        # 50 of 400 fails the binomial rule
        # A shift far below the references fails (d) everywhere; the references themselves do not move.
        lo = m.masks(rows, "safe", shift=-1.0)
        self.assertFalse(lo["d"].any())
        hi = m.masks(rows, "safe", shift=+1.0)
        self.assertTrue(hi["d"][hi["has"]].all())

    def test_recommendation_rules(self):
        def power(s, h):
            return {"safe": {"0.034": {str(n): {"full": s[i]} for i, n in enumerate((10, 20, 30))}},
                    "hide_and_seek": {"0.052": {str(n): {"full": h[i]} for i, n in enumerate((10, 20, 30))}}}
        ti = {k: {str(n): {"full": {"rate": 0.0}} for n in (10, 20, 30)} for k in m.NULLS}
        rec = m.recommend(dict(type_i=ti, power=power((0.5, 0.9, 1.0), (0.4, 0.85, 0.95))), {}, (10, 20, 30))
        self.assertEqual((rec["n_stands"], rec["smallest_n"], rec["reading"]), (True, 20, "n = 20 stands"))
        rec = m.recommend(dict(type_i=ti, power=power((0.5, 0.9, 1.0), (0.4, 0.7, 0.95))), {}, (10, 20, 30))
        self.assertEqual(rec["reading"], "n = 30 needed")
        ti["noise"]["20"]["full"]["rate"] = 0.06
        nested = {"safe": {"20": {"nested_abc": 0.80, "approx_abc": 0.90}}, "hide_and_seek": {"20": {"nested_abc": 0.9, "approx_abc": 0.92}}}
        rec = m.recommend(dict(type_i=ti, power=power((1, 1, 1), (1, 1, 1))), nested, (10, 20, 30))
        self.assertEqual(rec["type_i_above_0_05"], [("noise", "20")])
        self.assertEqual(rec["quote_nested_for"], ["safe"])


class BootstrapAndStudyTests(unittest.TestCase):
    def test_degenerate_inner_bootstrap_equals_the_point_gain(self):
        rows = co.held_rows(2)
        pick = lambda off: [r for t in co.TASKS for r in (
            [x for x in rows if x["task_name"] == t and not x["failed"]][off],
            [x for x in rows if x["task_name"] == t and x["failed"]][off])]
        c, e = cl.Cohort(pick(0), "safe"), cl.Cohort(pick(1), "safe")
        path = dict(u=1.0, s=2.0)
        tm, _ = fp.certify(c, RK, path)
        _, lP = fs.evaluate_cert(e, RK, tm)
        row, matched = fs.frozen_study(c, e, RK, path, lP)
        lo, lom = m.inner_bounds(c, e, RK, path, np.random.default_rng(1), 7)
        self.assertEqual(lo, row["gain"])
        self.assertEqual(lom, -np.inf if not math.isfinite(matched["gain"]) else matched["gain"])

    def test_nested_job_rebuilds_the_same_null_subjects(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, (rows, train, cal_s, test, tasks, rk) = fz.fixture(root)
            pools = ec.build_pools(cal_s, test, tasks)
            devC = cl.Cohort(cal_s + [r for r in train if r["failed"]], "safe")
            paths = {k: dict(u=1.0, s=1.0) for k in m.CO}
            sp = en.seed_pools_from(root / "rldx", cal_s + test)
            m._G.update(pools=pools, rk=rk, paths=paths, seed_pools=sp, devC=devC)
            a, pa, _, la, _ = m.make_study(10, 3)
            b, pb, _, lb, _ = m.make_study(10, 3)
            np.testing.assert_array_equal(la, lb)
            for k in m.DETS:
                self.assertEqual(pa[k], pb[k])
                for x, y in zip(a[k], b[k]):
                    np.testing.assert_array_equal(x.R, y.R)
            # The subjects are the safe draw's rollouts with fresh null scores.
            np.testing.assert_array_equal(a["noise"][1].L, a["safe"][1].L)
            self.assertFalse(np.array_equal(a["noise"][1].R, a["safe"][1].R))


class EndToEndTests(unittest.TestCase):
    def test_cli_reproduces_r9_and_reports_type_i_and_power(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common, _ = fz.fixture(root)
            sizes = ["--sizes", "10,20", "--replicates", "6"]
            status, text = quiet(fs.main, ["e2"] + common + sizes + ["--output", str(root / "e2_frozen.json")])
            self.assertEqual(status, 0, text)
            args = common + sizes + ["--frozen-json", str(root / "e2_frozen.json"), "--nested-outer", "2",
                                     "--nested-inner", "5"]
            status, text = quiet(m.main, args + ["--output", str(root / "rule.json")])
            self.assertEqual(status, 0, text)
            self.assertIn("co-primary frozen rows reproduce R9 MATCH", text)
            out = json.loads((root / "rule.json").read_text())
            self.assertTrue(out["checks"]["r9_reproduced"]["ok"])
            for kind in m.NULLS:
                for n in ("10", "20"):
                    self.assertEqual(out["type_i"][kind][n]["full"]["studies"], 6)
                    self.assertEqual(out["references"][kind][n]["draws"], 6 * m.K_NULL)
            self.assertEqual(set(out["nested"]), set(m.DETS))
            self.assertEqual(out["nested"]["safe"]["20"]["studies"], 2)
            self.assertEqual(set(out["joint"]), {"plug_in", "safe 0.034 / hs 0.083", "safe 0.034 / hs 0.052"})
            # A different R9 output fails the check and marks the JSON.
            bad = json.loads((root / "e2_frozen.json").read_text())
            bad["frozen"]["summary"]["20"]["safe"]["within_task"]["sd"] += 0.01
            (root / "bad.json").write_text(json.dumps(bad))
            args[args.index(str(root / "e2_frozen.json"))] = str(root / "bad.json")
            status, text = quiet(m.main, args + ["--output", str(root / "rule_bad.json")])
            self.assertEqual(status, 1)
            self.assertIn("FAILED CHECK", json.loads((root / "rule_bad.json").read_text())["status"])


if __name__ == "__main__":
    unittest.main()
