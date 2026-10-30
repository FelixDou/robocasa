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


rt = load("review_fixture", TESTS / "test_xr1_review_analyses.py")
rb = load("review_analyses_b", SCRIPTS / "review_analyses_b.py")
import confirm_lead as cl  # noqa: E402
import quantile_timer as qt  # noqa: E402


class AllocationTests(unittest.TestCase):
    def test_budget_is_met_and_fractions_rise_with_tighter_budgets(self):
        rng = np.random.default_rng(0)
        dev = {f"T{k}": rng.uniform(0.2, 0.9, size=rng.integers(5, 30)).tolist() for k in range(8)}
        prev = None
        for beta in rb.BETA_GRID:
            rk, info = rb.allocate(dev, beta)
            fpr = np.mean([np.mean(np.asarray(dev[t]) >= rk[t]) for t in dev])
            self.assertLessEqual(fpr, beta + 1e-9)
            self.assertAlmostEqual(fpr, info["development_fpr"], places=12)
            if prev is not None:      # a looser budget never raises a task's fraction
                self.assertTrue(all(rk[t] <= prev[t] + 1e-12 for t in dev))
            prev = rk
        rk0, _ = rb.allocate(dev, 0.0)
        self.assertTrue(all(np.mean(np.asarray(dev[t]) >= rk0[t]) == 0.0 for t in dev))

    def test_allocation_beats_a_shared_quantile_at_the_same_development_budget(self):
        rng = np.random.default_rng(1)
        # Tasks differ in spread, so a shared quantile spends the budget badly.
        dev = {f"T{k}": (rng.uniform(0.3, 0.35, 20) if k % 2 else rng.uniform(0.1, 0.9, 20)).tolist() for k in range(10)}
        q = qt.rho(dev, 0.95)
        budget = np.mean([np.mean(np.asarray(dev[t]) >= q[t] + 1e-9) for t in dev])
        rk, _ = rb.allocate(dev, budget)
        self.assertGreaterEqual(np.mean([1 - rk[t] for t in dev]), np.mean([1 - (q[t] + 1e-9) for t in dev]) - 1e-9)


class EndToEndTests(unittest.TestCase):
    def test_list_b_runs_and_reproduces_its_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = rt.fixture(root)
            # The real E1 rows carry SAFE's per-seed raw scores and the bundle its normalizations; give the
            # fixture both, consistent with the stored safe_scores (location 0, scale 1, monotone raw).
            seeds = ("0", "1", "2")
            bundle["task_normalizations"] = {s: {t: {"location": 0.0, "scale": 1.0} for t in bundle["task_horizons"]}
                                             for s in seeds}
            (root / "bundle.json").write_text(json.dumps(bundle))
            for name in ("cal", "test"):
                rows = [json.loads(l) for l in (root / f"{name}.jsonl").read_text().splitlines()]
                for r in rows:
                    r["per_seed_raw_safe"] = {s: r["safe_scores"] for s in seeds}
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rt.ra.main(rt.argv(root)), 0)
            # bisection matching equals review_analyses.loosen_timer at every false-alarm count
            cal, te, rk = rt.cohorts(root, bundle)
            T = cl.Cohort(te, "safe")
            for fp in range(0, 8):
                slow, fast = rt.ra.loosen_timer(T, rk, fp), rb.loosen_fast(T, rk, fp)
                self.assertEqual(slow is None, fast is None)
                if slow is not None:
                    self.assertEqual(slow[0], fast[0])
                    np.testing.assert_array_equal(slow[2], fast[2])
            argv = ["run", "--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
                    "--scores-calibration", str(root / "scores_cal.jsonl"), "--scores-test", str(root / "scores_test.jsonl"),
                    "--normalizations", str(root / "norms.json"), "--runtime-bundle", str(root / "bundle.json"),
                    "--e1-result", str(root / "e1.json"), "--amend2-result", str(root / "amend2.json"),
                    "--review-result", str(root / "review.json"), "--output", str(root / "b.json"),
                    "--bootstrap", "30", "--null-draws", "3"]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = rb.main(argv)
            self.assertEqual(code, 0, buf.getvalue())
            self.assertIn("LIST B OK", buf.getvalue())
            out = json.loads((root / "b.json").read_text())
            for gate in ("G1_e1_amend2", "G2_matching", "G3_allocation", "G5_decision_ci"):
                self.assertIn(gate, out["checks"])
            self.assertTrue(out["checks"]["G5_decision_ci"]["ok"])
            self.assertTrue(all(v["ok"] for v in out["checks"]["G2_matching"].values()))
            for label in ("A_fixed", "A_adaptive"):
                self.assertIn(out["B1"][label]["budget"], rb.BETA_GRID)
                self.assertIn("plus_A_against_A", out["B1"][label]["hide_and_seek"])
            for k in ("safe", "hide_and_seek"):
                self.assertIn("standalone", out["B2"][k])
                self.assertIn("fused_bonferroni", out["B2"][k])
                self.assertEqual(sorted(out["B4"]["matched_ci95"]), ["hide_and_seek", "safe"])
            self.assertEqual(sorted(out["B3"]["nulls"]), sorted(["noise", "safe_resampled", "hide_and_seek_resampled"]))
            self.assertIn("all", out["B5"]["seconds"]["P"])
            # refuses to overwrite
            with self.assertRaises(SystemExit):
                rb.main(argv)


class NoReturnTests(unittest.TestCase):
    def test_lead_before_no_return_and_key_check(self):
        te = [dict(rollout_id=f"r{i}") for i in range(3)]
        alarms = {"P": np.array([8.0, 9.0, math.inf]), "safe_plus_P": np.array([5.0, 9.0, 10.0]),
                  "hide_and_seek_plus_P": np.array([3.0, 6.0, 10.0])}
        key = {"videos": {
            "v0": dict(rollout_id="r0", steps_per_inference=2, horizon_inferences=10, length_inferences=10,
                       task_type="atomic", alarm_P_inference=8.0, alarm_safe_plus_P_inference=5.0),
            "v1": dict(rollout_id="r1", steps_per_inference=2, horizon_inferences=10, length_inferences=10,
                       task_type="composite", alarm_P_inference=9.0, alarm_safe_plus_P_inference=9.0),
            "v2": dict(rollout_id="r2", steps_per_inference=2, horizon_inferences=10, length_inferences=10,
                       task_type="composite", alarm_P_inference=None, alarm_safe_plus_P_inference=10.0)}}
        labels = {"labels": {
            "v0": dict(t_u_env_step=10, category="drop", confidence="high"),
            "v1": dict(recoverable_until_timeout=True, category="slow", confidence="high"),
            "v2": dict(t_u_env_step=4, category="stuck", confidence="low")}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "labels.json").write_text(json.dumps(labels))
            (root / "key.json").write_text(json.dumps(key))
            out, check = rb.no_return(te, None, None, alarms, root / "labels.json", root / "key.json")
            self.assertTrue(check["ok"])
            self.assertEqual((out["summary"]["decided"], out["summary"]["recoverable"]), (2, 1))
            # v0: H = 20 steps, t_u = 10; Hide-and-Seek alarms at inference 3 = step 4 (6/20 before).
            row = [r for r in out["rows"] if r["id"] == "v0"][0]
            self.assertAlmostEqual(row["detectors"]["hide_and_seek_plus_P"]["lead_before_no_return"], 0.3)
            self.assertTrue(row["detectors"]["hide_and_seek_plus_P"]["before_no_return"])
            self.assertFalse(row["detectors"]["P"]["before_no_return"])      # step 14 > 10
            self.assertEqual(out["summary"]["hide_and_seek_plus_P"]["before_no_return"], 1)
            key["videos"]["v0"]["alarm_P_inference"] = 7.0
            (root / "key.json").write_text(json.dumps(key))
            _, bad = rb.no_return(te, None, None, alarms, root / "labels.json", root / "key.json")
            self.assertFalse(bad["ok"])


class BandTests(unittest.TestCase):
    def test_split_alternates_within_task_and_band_alarm_is_one_based(self):
        rows = [dict(rollout_id=f"{t}{i}", task_name=t, failed=False) for t in "AB" for i in range(5)]
        ref, conf = rb.split_successes(rows)
        self.assertEqual([r["rollout_id"] for r in ref], ["A0", "A2", "A4", "B0", "B2", "B4"])
        self.assertEqual([r["rollout_id"] for r in conf], ["A1", "A3", "B1", "B3"])
        calib = dict(threshold=[1.0, 1.0, 0.5], alignment="extend")
        r = dict(x={"avg": np.array([0.0, 0.2, 0.6, 0.0])})
        self.assertEqual(rb.band_alarms([r], calib, "x", "avg").tolist(), [3.0])
        r2 = dict(x={"avg": np.array([0.0, 0.2, 0.4])})
        self.assertTrue(math.isinf(rb.band_alarms([r2], calib, "x", "avg")[0]))


if __name__ == "__main__":
    unittest.main()
