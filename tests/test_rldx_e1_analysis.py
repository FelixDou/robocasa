import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "rldx_e1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time"))


def load(name):
    spec = importlib.util.spec_from_file_location(f"rldx_e1_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ae, se = load("analyze_e1"), load("score_e1")
import confirm_lead as cl  # noqa: E402

TASKS = {"TaskA": 30, "TaskB": 40}
RHO = 0.6


def make_rows(rng, n=20, signal=True, prefix="x", split=None):
    """n successes and n failures per task. Failures run to H; with signal, both scores jump early."""
    rows = []
    for t, H in TASKS.items():
        for i in range(2 * n):
            failed = i >= n
            L = H if failed else int(rng.integers(int(0.3 * H), int(0.55 * H)))
            base = rng.normal(0, 0.3, L)
            s = base.copy()
            if failed and signal:
                s[int(0.3 * H):] += 3.0
            r = dict(rollout_id=f"{prefix}-{t}-{i}", task_name=t, task_type="atomic" if t == "TaskA" else "composite",
                     failed=failed, L=L, horizon=H, safe=s, hide_and_seek=s + rng.normal(0, 0.1, L))
            if split:
                r["split"] = split
            rows.append(r)
    return rows


def rk():
    return {t: RHO for t in TASKS}


def pools(rng):
    return {t: [rng.normal(0, 1, 500) for _ in range(3)] for t in TASKS}


class AnalysisTest(unittest.TestCase):
    def test_signal_passes_and_noise_fails(self):
        rng = np.random.default_rng(0)
        dev = make_rows(rng, n=10, prefix="dev")
        devC = cl.Cohort([r for r in dev if not r["failed"]] + [r for r in dev if r["failed"]], "safe")
        paths = {"safe": dict(u=1.5, s=0.0), "hide_and_seek": dict(u=1.5, s=0.0)}
        cal, test = make_rows(rng, prefix="c"), make_rows(rng, prefix="t")
        res, boot, refs, max_fa, dec = ae.analyze(cal, test, rk(), paths, devC, pools(rng), B=60)
        self.assertGreater(res["hide_and_seek"]["gain"], 0.2)
        self.assertEqual(len(refs["noise"]), 39)
        self.assertTrue(dec["hide_and_seek"]["H3"], dec["hide_and_seek"])
        self.assertTrue(dec["safe"]["d_beats_null_fusions"])
        self.assertGreater(max_fa, 0)                                         # binomial bound computed

        cal0, test0 = make_rows(rng, signal=False, prefix="c0"), make_rows(rng, signal=False, prefix="t0")
        res0, boot0, refs0, _, dec0 = ae.analyze(cal0, test0, rk(), paths, devC, pools(rng), B=60)
        self.assertFalse(dec0["hide_and_seek"]["H3"])
        self.assertFalse(dec0["safe"]["H3"])

    def test_bootstrap_is_paired_and_reproducible(self):
        rng = np.random.default_rng(1)
        cal, test = make_rows(rng, prefix="c"), make_rows(rng, prefix="t")
        paths = {"safe": dict(u=1.5, s=0.0), "hide_and_seek": dict(u=1.5, s=0.0)}
        C = {k: (cl.Cohort(cal, k), cl.Cohort(test, k)) for k in ae.DETS}
        a = ae.bootstrap(C, rk(), paths, 20, np.random.default_rng([5, 1]))
        b = ae.bootstrap(C, rk(), paths, 20, np.random.default_rng([5, 1]))
        self.assertEqual(a, b)

    def test_decide_rule(self):
        res = {k: dict(gain=0.1, test=dict(fp=5)) for k in ae.DETS}
        boot = {k: dict(gain_ci=[0.01, 0.2], matched_ci=[0.02, 0.2]) for k in ae.DETS}
        refs = {kind: [0.0] * 38 + [0.2] for kind in ae.NULLS}            # one reference above: p = 2/40
        dec = ae.decide(res, boot, refs, max_fa=39)
        self.assertTrue(dec["safe"]["H3"])
        refs2 = {kind: [0.0] * 37 + [0.2, 0.3] for kind in ae.NULLS}      # two above: p = 3/40 > 0.05
        self.assertFalse(ae.decide(res, boot, refs2, max_fa=39)["safe"]["d_beats_null_fusions"])
        boot["hide_and_seek"]["matched_ci"] = [-0.01, 0.2]
        self.assertFalse(ae.decide(res, boot, refs, max_fa=39)["hide_and_seek"]["H3"])
        res["safe"]["test"]["fp"] = 40
        self.assertFalse(ae.decide(res, boot, refs, max_fa=39)["safe"]["c_false_alarms_within_bound"])


class E1RowsTest(unittest.TestCase):
    def test_e1_rows_align_detectors(self):
        rng = np.random.default_rng(2)
        rows = make_rows(rng, n=3, prefix="e")
        with tempfile.TemporaryDirectory() as d:
            for det in ("safe_lstm", "hide_and_seek"):
                key = "safe" if det == "safe_lstm" else "hide_and_seek"
                se.write_jsonl(Path(d) / f"test_{det}_predictions.jsonl",
                               [dict(detector=det, rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"],
                                     safe_scores=r[key]) for r in rows])
            got = ae.e1_rows(d, "test", dict(TASKS))
            self.assertEqual(len(got), len(rows))
            self.assertEqual(got[0]["horizon"], TASKS[got[0]["task_name"]])
            lc = ae.limit_check(got, "test")
            self.assertEqual(lc["failures_not_at_limit"], [])
            bad = [dict(r) for r in rows]
            bad[0]["failed"] = not bad[0]["failed"]
            se.write_jsonl(Path(d) / "test_hide_and_seek_predictions.jsonl",
                           [dict(detector="hide_and_seek", rollout_id=r["rollout_id"], task_name=r["task_name"],
                                 failed=r["failed"], safe_scores=r["hide_and_seek"]) for r in bad])
            with self.assertRaises(SystemExit):
                ae.e1_rows(d, "test", dict(TASKS))


class GateTest(unittest.TestCase):
    """Gate R through the real loaders: synthetic recorded calibration dir and Hide-and-Seek scores."""

    def build(self, d, rng):
        d = Path(d)
        rows = (make_rows(rng, n=6, prefix="tr", split="train") + make_rows(rng, n=6, prefix="ca", split="calibration")
                + make_rows(rng, n=6, prefix="ev", split="evaluation"))
        rows = [r for r in rows if not (r["split"] == "calibration" and r["failed"])]
        cal = d / "cal"
        for s in (0, 1, 2):
            (cal / f"lstm_seed{s}").mkdir(parents=True)
            se.write_jsonl(cal / f"lstm_seed{s}" / "normalized_scores.jsonl",
                           [dict(rollout_id=r["rollout_id"], task_name=r["task_name"], task_type=r["task_type"],
                                 failed=r["failed"], split=r["split"], full_scores=r["safe"]) for r in rows])
        hs = d / "hs"
        hs.mkdir()
        for cohort, split in (("calibration", "calibration"), ("test", "evaluation")):
            se.write_jsonl(hs / f"{cohort}_predictions.jsonl",
                           [dict(detector="hide_and_seek", rollout_id=r["rollout_id"], task_name=r["task_name"],
                                 failed=r["failed"], safe_scores=np.maximum.accumulate(r["hide_and_seek"]),
                                 per_seed_raw={str(s): r["hide_and_seek"] for s in (0, 1, 2)})
                            for r in rows if r["split"] == split])
        return cal, hs

    def test_gate_reproduces_and_detects_change(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            cal, hs = self.build(d, rng)
            rows, H, tasks, train, cal_s, test = ae.recorded_rows(cal, hs)
            import quantile_timer as qt
            import rldx_replication as rr
            rho = qt.rho(rr.dev_fractions(train, tasks), rr.Q)
            man = dict(tasks={t: dict(type="atomic", H_k=int(H[t]), rho_k=rho[t]) for t in tasks}, detectors={})
            paths = {"safe": dict(u=1.5, s=0.0), "hide_and_seek": dict(u=1.0, s=2.0)}
            for k, m in ae.DETS.items():
                c, e = cl.Cohort(cal_s, k), cl.Cohort(test, k)
                tm, fz = ae.fp.certify(c, rho, paths[k])
                _, lP = ae.fs.evaluate_cert(e, rho, tm)
                _, lZ = ae.fs.evaluate_cert(e, rho, fz)
                man["detectors"][m] = dict(path=dict(u=paths[k]["u"], s=paths[k]["s"], lam_at_fit=round(fz["lam"], 4)),
                                           replication_gain=round(float(np.mean(lZ - lP)), 4))
            problems, out = ae.gate_recorded(man, H, tasks, train, cal_s, test)
            self.assertEqual(problems, [], out)
            man["detectors"]["hide_and_seek_plus_P"]["replication_gain"] += 0.01
            problems, _ = ae.gate_recorded(man, H, tasks, train, cal_s, test)
            self.assertTrue(any("hide_and_seek" in p for p in problems))
            man["tasks"]["TaskA"]["rho_k"] += 0.01
            problems, _ = ae.gate_recorded(man, H, tasks, train, cal_s, test)
            self.assertTrue(any("rho_k" in p for p in problems))


class ScoringHelpersTest(unittest.TestCase):
    def test_records_normalize_like_score_cohorts(self):
        from robocasa.recovery.safe.baselines.run_baselines import build_records
        rng = np.random.default_rng(4)
        rows = [dict(rollout_id=f"r{i}", task_name="TaskA", failed=bool(i % 2)) for i in range(4)]
        raw = {s: {r["rollout_id"]: rng.normal(0, 1, 7) for r in rows} for s in se.SEEDS}
        norms = {s: {"TaskA": dict(location=float(rng.normal()), scale=float(rng.uniform(0.5, 2)))} for s in se.SEEDS}
        mine = se.records(rows, raw, norms, {"TaskA": 7}, "hide_and_seek")
        ref = build_records(rows, raw, norms, {"TaskA": 7}, {"TaskA": {"risk": [0.0]}})
        for a, b in zip(mine, ref):
            self.assertTrue(np.array_equal(a["safe_scores"], b["safe_scores"]))

    def test_hs_reference_gap(self):
        rows = [dict(rollout_id="a"), dict(rollout_id="b")]
        ref = {"a": {s: [1.0, 2.0] for s in se.SEEDS}, "b": {s: [0.5] for s in se.SEEDS}}
        gap, missing = se.hs_reference_gap(lambda s, r: np.asarray(ref[r["rollout_id"]][s]), rows, ref)
        self.assertEqual((gap, missing), (0.0, []))
        gap, _ = se.hs_reference_gap(lambda s, r: np.asarray(ref[r["rollout_id"]][s]) + 1e-9, rows, ref)
        self.assertGreater(gap, se.EXACT)
        _, missing = se.hs_reference_gap(lambda s, r: np.zeros(1), rows + [dict(rollout_id="c")], ref)
        self.assertEqual(missing, ["c"])

    def test_safe_normalization_and_recorded_check(self):
        rng = np.random.default_rng(5)
        train = [dict(rollout_id=f"t{i}", task_name="TaskA", split="train", scores=rng.normal(0, 1, 9).tolist(),
                      task_min_step=6, failed=False) for i in range(8)]
        norm = se.safe_normalizations(train)
        norms = {s: norm for s in se.SEEDS}
        raw = rng.normal(0, 1, 5).tolist()
        recorded = {s: {"e1": dict(task_name="TaskA", split="evaluation", full_raw_scores=raw,
                                   normalization_location=norm["TaskA"]["location"],
                                   normalization_scale=norm["TaskA"]["scale"])} for s in se.SEEDS}
        rescored = {s: {"e1": dict(scores=raw)} for s in se.SEEDS}
        ng, rg, missing, n = se.check_safe_recorded(norms, recorded, rescored)
        self.assertEqual((ng, rg, missing, n), (0.0, 0.0, [], 3))
        rescored["1"]["e1"] = dict(scores=[x + 1e-6 for x in raw])
        _, rg, _, _ = se.check_safe_recorded(norms, recorded, rescored)
        self.assertGreater(rg, se.EXACT)


if __name__ == "__main__":
    unittest.main()
