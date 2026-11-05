import argparse
import contextlib
import importlib.util
import io
import json
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


fx = load("hs_fixture", TESTS / "test_xr1_confirm_hide_and_seek.py")
ra = load("review_analyses", SCRIPTS / "review_analyses.py")
import confirm_lead as cl
import quantile_timer as qt


def fixture(root, B=30):
    """E1, Amendment 2 and dose-response results on the Amendment 2 test fixture, plus baselines."""
    bundle = fx.write_fixture(root)
    fx.e1_result(root, bundle, B)
    chs = fx.load("confirm_hide_and_seek")
    with contextlib.redirect_stdout(io.StringIO()):
        assert chs.main(fx.argv(root)[:-4] + ["--bootstrap", str(B), "--null-draws", "2"]) == 0
    (root / "amend2.json").write_text((root / "out.json").read_text())
    (root / "out.json").unlink()
    rng = np.random.default_rng(11)
    for name in ("cal", "test"):
        rows = [json.loads(l) for l in (root / f"{name}.jsonl").read_text().splitlines()]
        for r in rows:
            n = len(r["safe_scores"])
            r["baseline_scores"] = {"knn_cosine": np.maximum.accumulate(
                rng.standard_normal(n) + (0.6 if r["failed"] else 0.0)).tolist()}
        (root / f"{name}_b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    dose = load("dose_response", SCRIPTS / "dose_response.py")
    dose.BOOT, dose.PERMUTATIONS = 20, 200      # descriptive parts only; the gate reads pooled gains
    with contextlib.redirect_stdout(io.StringIO()):
        dose.run(argparse.Namespace(calibration=str(root / "cal_b.jsonl"), test=str(root / "test_b.jsonl"),
                                    runtime_bundle=str(root / "bundle.json"), output=str(root / "dose.json"),
                                    confirm_predictions=None))
    return bundle


def argv(root, *extra, B=30):
    return ["run", "--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
            "--scores-calibration", str(root / "scores_cal.jsonl"), "--scores-test", str(root / "scores_test.jsonl"),
            "--normalizations", str(root / "norms.json"), "--runtime-bundle", str(root / "bundle.json"),
            "--e1-result", str(root / "e1.json"), "--amend2-result", str(root / "amend2.json"),
            "--baselines-calibration", str(root / "cal_b.jsonl"), "--baselines-test", str(root / "test_b.jsonl"),
            "--dose-response-result", str(root / "dose.json"), "--output", str(root / "review.json"),
            "--bootstrap", str(B), "--null-draws", "2", "--resplits", "5", *extra]


def cohorts(root, bundle):
    chs = fx.load("confirm_hide_and_seek")
    _, cal = cl.primary(root / "cal.jsonl", "calibration")
    _, te = cl.primary(root / "test.jsonl", "test")
    chs.attach(cal, chs.load_scores(root / "scores_cal.jsonl"), "cal")
    chs.attach(te, chs.load_scores(root / "scores_test.jsonl"), "test")
    return cal, te, qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)


class ReviewAnalysesTests(unittest.TestCase):
    def test_fit_alpha_equals_fit_fast_and_loosens_with_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cal, _, rk = cohorts(root, fx.write_fixture(root))
            for key in ("safe", "hide_and_seek"):
                c = cl.Cohort(cal, key)
                self.assertEqual(ra.fit_alpha(c, rk, 0.05), cl.fit_fast(c, rk), key)
            c = cl.Cohort(cal, "safe")
            lams = [ra.fit_alpha(c, rk, a)[0]["lam"] for a in (0.05, 0.1, 0.2, 0.4)]
            self.assertEqual(lams, sorted(lams, reverse=True))

    def test_split_path_is_chosen_without_the_certification_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cal, _, rk = cohorts(root, fx.write_fixture(root))
            c = cl.Cohort(cal, "hide_and_seek")
            succ = np.flatnonzero(~c.failed)
            A = np.concatenate([np.flatnonzero(c.failed), succ[::2]])
            Bi = succ[1::2]
            _, fu = ra.fit_split(c.take(A), c.take(Bi), rk, 0.2)
            changed = c.take(Bi)
            changed.M = changed.M + 5.0
            changed.R = changed.R + 5.0
            _, fu2 = ra.fit_split(c.take(A), changed, rk, 0.2)
            self.assertEqual((fu["u"], fu["s"]), (fu2["u"], fu2["s"]))
            S = c.take(Bi)
            want = cl.conformal(cl.V_vec(S, cl.rho_rows(S, rk), fu["u"], fu["s"]), 0.2)
            self.assertEqual(fu["lam"], want)

    def test_matched_detectors_respect_their_false_alarm_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cal, te, rk = cohorts(root, fx.write_fixture(root))
            c, e = cl.Cohort(cal, "hide_and_seek"), cl.Cohort(te, "hide_and_seek")
            tm, fu = ra.fit_alpha(c, rk, 0.1)
            eF, _ = cl.evaluate_fast(e, rk, fu)
            eP, _ = cl.evaluate_fast(e, rk, tm)
            loose = ra.loosen_timer(e, rk, eF["fp"])
            self.assertLessEqual(loose[1]["fp"], eF["fp"])
            tight = ra.tighten_fused(e, rk, fu, eP["fp"])
            if tight is not None:
                self.assertLessEqual(tight[1]["fp"], eP["fp"])

    def test_sign_test_and_per_task_gains(self):
        self.assertEqual(ra.sign_test(8, 2), 0.1094)
        self.assertEqual(ra.sign_test(5, 0), 0.0625)
        self.assertIsNone(ra.sign_test(0, 0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cal, te, rk = cohorts(root, fx.write_fixture(root))
            c, e = cl.Cohort(cal, "hide_and_seek"), cl.Cohort(te, "hide_and_seek")
            tm, fu = ra.fit_alpha(c, rk, 0.05)
            _, lP = cl.evaluate_fast(e, rk, tm)
            _, lF = cl.evaluate_fast(e, rk, fu)
            rows, summary = ra.per_task(e, rk, tm, fu, lF, lP, {r["task_name"]: r["task_type"] for r in te})
            # Every task has the same number of failures, so task gains average to the pooled gain.
            self.assertAlmostEqual(np.mean([v["gain"] for v in rows.values()]), float(np.mean(lF - lP)), places=3)
            self.assertEqual(summary["positive"] + summary["zero"] + summary["negative"], len(rows))
            self.assertEqual(summary["false_alarms"]["P"]["total"], int(ra.success_alarms(e, rk, tm).sum()))

    def test_run_reproduces_references_and_reports_every_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                self.assertEqual(ra.main(argv(root)), 0)
            self.assertIn("REVIEW ANALYSES OK", buffer.getvalue())
            out = json.loads((root / "review.json").read_text())
            for name in ("e1_reference", "amend2_reference", "dose_response_reference", "amend2_decision_ci"):
                self.assertTrue(out["checks"][name]["ok"], name)
            self.assertTrue(out["checks"]["fit_alpha_equals_fit_fast"])
            self.assertEqual(out["counts"]["detectors"][-1], "knn_cosine")
            self.assertEqual(len(out["R1"]["detectors"]["hide_and_seek"]), len(ra.ALPHAS))
            row = out["R2"]["rows"]["hide_and_seek"]
            self.assertEqual(row["twa"], round(0.5 * (row["mean_lead"] + 1 - row["fpr"]), 4))
            self.assertIn(out["R1"]["band_summary"]["safe"]["reading"],
                          {"earlier at every budget", "no earlier warning shown", "budget-dependent",
                           "later than the timer at matched false alarms"})
            self.assertIn(out["R3"]["hide_and_seek"]["summary"]["reading"], {"broad", "concentrated", "mixed",
                                                                               "no pooled gain"})
            self.assertEqual(out["R4"]["resplit"]["resplits"], 5)
            # At alpha = 0.05 the decision CIs are Amendment 2's (same draws, same refits).
            a2 = json.loads((root / "amend2.json").read_text())
            self.assertEqual(out["R1"]["decision_bootstrap"]["by_alpha"]["0.05"]["gain_ci95"]["hide_and_seek"],
                             a2["detectors"]["hide_and_seek"]["gain_decision_ci95"])

    def test_wrong_reference_blocks_every_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            a2 = json.loads((root / "amend2.json").read_text())
            a2["detectors"]["hide_and_seek"]["gain"] += 0.01
            (root / "amend2.json").write_text(json.dumps(a2))
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                self.assertEqual(ra.main(argv(root)), 1)
            self.assertIn("REPRODUCTION FAILED", buffer.getvalue())
            out = json.loads((root / "review.json").read_text())
            self.assertNotIn("R1", out)


if __name__ == "__main__":
    unittest.main()
