"""E1b (scripts/rldx_e1/e1b.py): paths frozen on E1's rollouts, then E1's rule on new sets. Synthetic data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests import test_rldx_e1_analysis as e1t
from tests.test_rldx_e1_analysis import ae, cl, load, make_rows, se

e1b = load("e1b")


def write_scores(d, name, rows):
    """score_e1-format predictions of both detectors for one set (SAFE values already normalized)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    se.write_jsonl(d / f"{name}_safe_lstm_predictions.jsonl",
                   [dict(detector="safe_lstm", rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"],
                         safe_scores=r["safe"], per_seed_raw={s: r["safe"] for s in e1b.SEEDS}) for r in rows])
    se.write_jsonl(d / f"{name}_hide_and_seek_predictions.jsonl",
                   [dict(detector="hide_and_seek", rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"],
                         safe_scores=r["hide_and_seek"]) for r in rows])
    tasks = sorted({r["task_name"] for r in rows})
    (d / "safe_lstm_scoring.json").write_text(json.dumps({"normalizations": {
        s: {t: dict(location=0.0, scale=1.0) for t in tasks} for s in e1b.SEEDS}}))


def quiet(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = fn(argv)
    return status, buf.getvalue()


class E1bTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        rng = np.random.default_rng(11)
        # Recorded rollouts and E1's manifest, as in E1's gate test.
        self.cal_dir, self.hs_dir = e1t.GateTest().build(root, rng)
        rows, H, tasks, train, cal_s, test = ae.recorded_rows(self.cal_dir, self.hs_dir)
        import quantile_timer as qt
        import rldx_replication as rr
        rho = qt.rho(rr.dev_fractions(train, tasks), rr.Q)
        man = dict(tasks={t: dict(type="atomic" if t == "TaskA" else "composite", H_k=int(H[t]), rho_k=rho[t])
                          for t in tasks}, detectors={})
        paths = {"safe": dict(u=1.5, s=0.0), "hide_and_seek": dict(u=1.0, s=2.0)}
        for k, m in ae.DETS.items():
            c, e = cl.Cohort(cal_s, k), cl.Cohort(test, k)
            tm, fz = ae.fp.certify(c, rho, paths[k])
            _, lP = ae.fs.evaluate_cert(e, rho, tm)
            _, lZ = ae.fs.evaluate_cert(e, rho, fz)
            man["detectors"][m] = dict(path=dict(u=paths[k]["u"], s=paths[k]["s"], lam_at_fit=round(fz["lam"], 4)),
                                       replication_gain=round(float(np.mean(lZ - lP)), 4))
        self.man = root / "e1_freeze.json"
        self.man.write_text(json.dumps(man))
        # E1's sets and E1b's sets.
        for name in e1b.SETS:
            write_scores(root / "e1", name, make_rows(rng, n=20, prefix=f"e1{name}"))
            write_scores(root / "e1b", name, make_rows(rng, n=20, prefix=f"e1b{name}"))
        (root / "plan.json").write_text(json.dumps({"seeds": e1b.DEFAULT_SEEDS}))

    def tearDown(self):
        self.tmp.cleanup()

    def freeze(self, out="freeze.json", check=False):
        argv = ["freeze", "--freeze-manifest", str(self.man), "--e1-scores-dir", str(self.root / "e1"),
                "--output", str(self.root / out), "--replicates", "40"]
        return quiet(e1b.main, argv + (["--design-check"] if check else []))

    def analyze(self, out="result.json", e1b_dir="e1b"):
        return quiet(e1b.main, ["analyze", "--freeze-manifest", str(self.man), "--e1b-freeze", str(self.root / "freeze.json"),
                                "--calibration-dir", str(self.cal_dir), "--hs-scores-dir", str(self.hs_dir),
                                "--e1-scores-dir", str(self.root / "e1"), "--e1b-scores-dir", str(self.root / e1b_dir),
                                "--plan", str(self.root / "plan.json"), "--output", str(self.root / out), "--replicates", "40"])

    def test_freeze_chooses_on_all_e1_rollouts(self):
        status, text = self.freeze(check=True)
        self.assertEqual(status, 0, text)
        self.assertIn("FREEZE OK", text)
        fz = json.loads((self.root / "freeze.json").read_text())
        man = json.loads(self.man.read_text())
        rk, H, data = e1b.e1_data(self.root / "e1", man)
        want = e1b.choose(data["calibration"] + data["test"], rk)
        self.assertEqual(fz["paths"], json.loads(json.dumps(want)))
        self.assertEqual((fz["detectors"], list(fz["paths"])), (["hide_and_seek"], ["hide_and_seek"]))
        self.assertEqual(fz["e1_counts"]["calibration"], dict(successes=40, failures=40))
        self.assertIn("design_check", fz)
        self.assertEqual(set(fz["design_check"]["detectors"]), {"hide_and_seek"})
        self.assertEqual(fz["e1b_seeds"], {"calibration": 1601, "test": 1701})

    def test_analysis_runs_the_E1_rule_with_the_frozen_paths(self):
        self.assertEqual(self.freeze()[0], 0)
        status, text = self.analyze()
        self.assertEqual(status, 0, text)
        self.assertIn("ANALYSIS DONE", text)
        out = json.loads((self.root / "result.json").read_text())
        fz = json.loads((self.root / "freeze.json").read_text())
        self.assertEqual(list(out["results"]), ["hide_and_seek"])
        self.assertEqual(out["frozen_paths"], {"hide_and_seek": dict(u=fz["paths"]["hide_and_seek"]["u"],
                                                                     s=fz["paths"]["hide_and_seek"]["s"])})
        self.assertIn("H3", out["results"]["hide_and_seek"])
        self.assertEqual(len(out["null_references"]["noise"]), ae.K_NULL)
        self.assertTrue(out["checks"]["R_recorded"]["ok"])
        self.assertEqual(out["checks"]["S_sets"]["overlap_with_e1"], 0)

    def test_rule_matches_analyze_e1_for_hide_and_seek(self):
        """With E1's random streams, the one-detector rule gives analyze_e1's numbers for Hide-and-Seek and P."""
        rng = np.random.default_rng(5)
        cal, test, dev = make_rows(rng, prefix="c"), make_rows(rng, prefix="t"), make_rows(rng, n=10, prefix="d")
        paths = {"safe": dict(u=1.5, s=0.0), "hide_and_seek": dict(u=1.0, s=2.0)}
        pools = {t: [rng.normal(0, 1, 300) for _ in range(3)] for t in ("TaskA", "TaskB")}
        devC = cl.Cohort(dev, "safe")
        mine = e1b.run_rule(cal, test, e1t.rk(), {"hide_and_seek": paths["hide_and_seek"]}, devC, pools, B=30, K=5)
        ref = ae.analyze(cal, test, e1t.rk(), paths, devC, pools, B=30, K=5, seed=e1b.SEED)
        for i in (0, 1, 2, 4):               # point, bootstrap, null references, decision
            got, want = mine[i], ref[i]
            if i == 0:
                self.assertEqual(got["P"]["test"], want["P"]["test"])
                got, want = got["hide_and_seek"]["gain"], want["hide_and_seek"]["gain"]
            elif i == 2:
                self.assertEqual(got, want)   # same null draws: P and the null cohorts read no detector score
                continue
            else:
                got, want = got["hide_and_seek"], want["hide_and_seek"]
            self.assertEqual(json.dumps(got, sort_keys=True, default=float), json.dumps(want, sort_keys=True, default=float))
        self.assertEqual(mine[3], ref[3])

    def test_gates_refuse(self):
        self.assertEqual(self.freeze()[0], 0)
        # F: an E1 score file changed after the freeze.
        path = self.root / "e1" / "test_hide_and_seek_predictions.jsonl"
        keep = path.read_text()
        rows = [json.loads(l) for l in keep.splitlines()]
        rows[0]["safe_scores"][0] += 1.0
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(SystemExit, "F: E1's score files differ"):
            self.analyze("bad_f.json")
        path.write_text(keep)
        # S: E1b's plan used other seeds.
        (self.root / "plan.json").write_text(json.dumps({"seeds": {"calibration": 1201, "test": 1301}}))
        with self.assertRaisesRegex(SystemExit, "S: plan seeds"):
            self.analyze("bad_seeds.json")
        (self.root / "plan.json").write_text(json.dumps({"seeds": e1b.DEFAULT_SEEDS}))
        # S: E1b's sets are E1's.
        with self.assertRaisesRegex(SystemExit, "E1b rollouts are E1 rollouts"):
            self.analyze("bad_overlap.json", e1b_dir="e1")
        # The freeze refuses to overwrite.
        with self.assertRaises(SystemExit):
            self.freeze()


if __name__ == "__main__":
    unittest.main()
