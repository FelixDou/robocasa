"""R13: second mock review, list B (ensemble size, SAFE form, truncation, per-task FP, success-optimal reset)."""
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


rt = load("review_fixture_c", TESTS / "test_xr1_review_analyses.py")
rc = load("review_analyses_c", SCRIPTS / "review_analyses_c.py")
import confirm_lead as cl  # noqa: E402
import stage2_sizing as s2  # noqa: E402


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = fn(*a)
    return status, buf.getvalue()


def rows_for(task, H, lengths, failed, key="safe"):
    out = []
    for i, (L, f) in enumerate(zip(lengths, failed)):
        out.append(dict(rollout_id=f"{task}{i}", task_name=task, horizon=H, L=L, failed=f,
                        **{key: np.arange(1, L + 1, dtype=float) + (5.0 if f else 0.0)}))
    return out


class TruncationTests(unittest.TestCase):
    def test_identity_at_one_and_relabel_below(self):
        rows = rows_for("A", 10, [4, 8, 10, 10], [False, False, True, True])
        same = rc.truncate_rows(rows, 1.0, ("safe",))
        for a, b in zip(rows, same):
            self.assertEqual((a["L"], a["horizon"], a["failed"]), (b["L"], b["horizon"], b["failed"]))
            np.testing.assert_array_equal(a["safe"], b["safe"])
        cut = rc.truncate_rows(rows, 0.7, ("safe",))          # H' = 7
        self.assertEqual([r["horizon"] for r in cut], [7] * 4)
        self.assertEqual([r["failed"] for r in cut], [False, True, True, True])
        self.assertEqual([r["L"] for r in cut], [4, 7, 7, 7])
        self.assertEqual(len(cut[1]["safe"]), 7)

    def test_development_fractions_are_rescaled_and_long_successes_dropped(self):
        dev = {"A": [0.3, 0.6, 0.9]}
        self.assertEqual(rc.truncate_dev(dev, {"A": 10}, 1.0), dev)
        got = rc.truncate_dev(dev, {"A": 10}, 0.7)["A"]
        np.testing.assert_allclose(got, [0.3 / 0.7, 0.6 / 0.7])


class FormTests(unittest.TestCase):
    def test_cumulative_and_non_cumulative_series(self):
        rng = np.random.default_rng(0)
        cum = [dict(failed=bool(i % 2), per_seed_raw_safe={"0": np.cumsum(rng.uniform(0, 1, 12)).tolist()}) for i in range(6)]
        f = rc.safe_form(cum, ("0",))
        self.assertTrue(f["cumulative"])
        self.assertEqual(f["cumulative_series"], 6)
        bad = cum + [dict(failed=False, per_seed_raw_safe={"0": [0.5, 0.2, 3.0]})]
        self.assertFalse(rc.safe_form(bad, ("0",))["cumulative"])

    def test_time_free_mean_is_the_running_mean_of_increments(self):
        p = np.array([0.2, 0.8, 0.5, 0.5])
        r = dict(task_name="A", failed=True, per_seed_raw_safe={"0": np.cumsum(p).tolist()})
        s = dict(task_name="A", failed=False, per_seed_raw_safe={"0": np.cumsum(p[::-1]).tolist()})
        key, aff = rc.add_time_free([r, s], [], ("0",), "mean")
        raw = np.maximum.accumulate(np.cumsum(p) / np.arange(1, 5))
        np.testing.assert_allclose(r[key] * aff["scale"] + aff["location"], raw, atol=1e-4)


class TimeOnlyTests(unittest.TestCase):
    def test_alarm_indices_and_cutoff_choice(self):
        fa, sa = rc.time_only(20, [20, 20], [5, 15], 0.5)
        self.assertEqual(fa.tolist(), [10.0, 10.0])
        self.assertEqual(sa.tolist(), [10.0, 10.0])
        fa1, _ = rc.time_only(20, [20], [5], 1.0)
        self.assertEqual(fa1.tolist(), [20.0])
        # Successes all end by 0.3 H and retries are cheap: an early cutoff beats never resetting.
        v = dict(p=0.4, H=20.0)
        a, w = rc.best_cutoff(v, [20.0] * 10, [4, 5, 6] * 4, 2.0, 0.4, (1, 2, 3), s2)
        self.assertIn(a, rc.CUTOFF_GRID)
        self.assertLess(a, 1.0)
        fa, sa = rc.time_only(20, [20.0] * 10, [4, 5, 6] * 4, 1.0)
        never = s2.retry_mc(dict(v, succ_L=np.array([4, 5, 6] * 4, float), fail_alarm={"R": fa}, succ_alarm={"R": sa}),
                            "R", 2.0, 0.4, (1, 2, 3))[0]
        self.assertGreaterEqual(w, never)


class OracleTests(unittest.TestCase):
    def test_oracle_respects_its_false_alarm_target(self):
        rng = np.random.default_rng(3)
        rows = []
        for t in "ABCD":
            H = 40
            L = list(rng.integers(8, 35, 10)) + [H] * 10
            rows += rows_for(t, H, [int(x) for x in L], [False] * 10 + [True] * 10)
        te = cl.Cohort(rows, "safe")
        for target in (0, 2, 5, 9):
            o, lead = rc.oracle_allocation(te, target)
            self.assertLessEqual(o["fp"], target)
            self.assertEqual(len(lead), 40)
        P = rc.per_task_fp(te, {t: 0.5 for t in "ABCD"}, dict(lam=0.1, u=math.inf, s=0.0))
        self.assertEqual(sum(P["per_task"].values()), P["total"])

    def test_successes_ending_at_the_limit_count_against_the_oracle(self):
        # Task A has a success at its final inference, which any timer alarms. Task B gains most from
        # an earlier cutoff. Allocating in fractions treated A's success as free and lowered B, giving
        # 2 false alarms at a target of 1 (the R13 cluster run's H4 failure); alarm indices must not.
        rows = rows_for("A", 40, [10, 20, 30, 40] + [40] * 4, [False] * 4 + [True] * 4)
        rows += rows_for("B", 40, [10, 20, 21, 39] + [40] * 4, [False] * 4 + [True] * 4)
        for t in "CD":
            rows += rows_for(t, 40, [10, 20, 30, 39] + [40] * 4, [False] * 4 + [True] * 4)
        te = cl.Cohort(rows, "safe")
        for target in (1, 2, 3, 5):
            o, _ = rc.oracle_allocation(te, target)
            self.assertLessEqual(o["fp"], target)
            self.assertEqual(o["fp"], o["planned_fp"])
        o, _ = rc.oracle_allocation(te, 0)          # unreachable (A's success): reported, not hidden
        self.assertEqual((o["fp"], o["planned_fp"]), (1, 1))


class EndToEndTests(unittest.TestCase):
    def test_xr1_runs_and_passes_its_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = rt.fixture(root)
            seeds = ("0", "1", "2")
            bundle["task_normalizations"] = {s: {t: {"location": 0.0, "scale": 1.0} for t in bundle["task_horizons"]}
                                             for s in seeds}
            (root / "bundle.json").write_text(json.dumps(bundle))
            for name in ("cal", "test"):
                rows = [json.loads(l) for l in (root / f"{name}.jsonl").read_text().splitlines()]
                for r in rows:
                    r["per_seed_raw_safe"] = {s: r["safe_scores"] for s in seeds}
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(quiet(rt.ra.main, rt.argv(root))[0], 0)
            argv = ["xr1", "--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
                    "--scores-calibration", str(root / "scores_cal.jsonl"), "--scores-test", str(root / "scores_test.jsonl"),
                    "--normalizations", str(root / "norms.json"), "--runtime-bundle", str(root / "bundle.json"),
                    "--e1-result", str(root / "e1.json"), "--amend2-result", str(root / "amend2.json"),
                    "--review-result", str(root / "review.json"), "--output", str(root / "c.json")]
            status, text = quiet(rc.main, argv)
            self.assertEqual(status, 0, text)
            self.assertIn("LIST C OK", text)
            out = json.loads((root / "c.json").read_text())
            for gate in ("H0_trajectories", "H1_e1_amend2", "H2_matched", "H3_truncation_identity", "H4_counts_and_oracle"):
                self.assertTrue(out["checks"][gate]["ok"] if "ok" in out["checks"][gate]
                                else all(v["ok"] for v in out["checks"][gate].values()), gate)
            self.assertEqual(set(out["C1"]), set(rc.DETECTORS))
            n_seeds = len(json.loads((root / "norms.json").read_text())["hide_and_seek"])
            self.assertEqual(set(out["C1"]["hide_and_seek"]["by_size"]), {str(k) for k in range(1, n_seeds + 1)})
            self.assertEqual(sorted(out["C3"]), ["0.8", "0.9", "1.0"])
            self.assertEqual(sorted(out["C2"]["variants"]), ["safe_mean", "safe_win8"])
            with self.assertRaises(SystemExit):
                rc.main(argv)

    def test_reset_reproduces_r9_and_reports_the_optimal_timer(self):
        fz = load("frozen_fixture_c", TESTS / "test_stage2_frozen_sizing.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common, _ = fz.fixture(root)
            status, text = quiet(fz.m.main, ["e3"] + common + ["--output", str(root / "e3_hs_reset.json")])
            self.assertEqual(status, 0, text)
            status, text = quiet(rc.main, ["reset"] + common + ["--reference-json", str(root / "e3_hs_reset.json"),
                                                                "--output", str(root / "reset.json")])
            self.assertEqual(status, 0, text)
            self.assertIn("RESET OK", text)
            out = json.loads((root / "reset.json").read_text())
            self.assertTrue(out["checks"]["H5_timer_identity"]["ok"])
            for r_mode in rc.RETRY:
                self.assertTrue(out["checks"][f"H5_reference_r_{r_mode}"]["ok"])
                a = out["runs"][r_mode]["by_type"]["all"]["budgets"]["1.25"]
                self.assertEqual(set(a["multi"]), set(rc.ARMS))
                # The oracle cutoff is chosen on the test pool, so it cannot lose to R_opt by much.
                self.assertGreaterEqual(a["multi"]["R_oracle"], a["multi"]["R_opt"] - 0.02)


if __name__ == "__main__":
    unittest.main()
