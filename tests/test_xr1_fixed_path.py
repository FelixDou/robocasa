"""R8: the fused path chosen on the validation set (scripts/xr1_lead_time/fixed_path.py)."""
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


fx = load("hs_fixture", TESTS / "test_xr1_confirm_hide_and_seek.py")
fp = load("fixed_path", SCRIPTS / "fixed_path.py")
ra = load("review_analyses", SCRIPTS / "review_analyses.py")
import confirm_lead as cl  # noqa: E402
import lead_time  # noqa: E402
import quantile_timer as qt  # noqa: E402

B = 30


def add_safe_seeds(rows, rng, bnorms):
    """Per-seed raw SAFE scores; safe_scores becomes their normalized seed mean."""
    for r in rows:
        n = r["source_inferences"]
        shift = 0.6 if r["failed"] and r["task_type"] == "atomic" else 0.0
        raw = {s: (rng.standard_normal(n) + shift * np.linspace(0, 1, n)).tolist() for s in fx.SEEDS}
        per = [(np.maximum.accumulate(raw[s]) - bnorms[s][r["task_name"]]["location"]) / bnorms[s][r["task_name"]]["scale"]
               for s in fx.SEEDS]
        r["per_seed_raw_safe"] = raw
        r["safe_scores"] = np.mean(per, axis=0).tolist()
    return rows


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def fixture(root):
    bundle = fx.write_fixture(root)
    bnorms = {s: {t: {"location": 0.05 * int(s), "scale": 1.0 + 0.2 * int(s)} for t in fx.TASKS} for s in fx.SEEDS}
    bundle["task_normalizations"] = bnorms
    (root / "bundle.json").write_text(json.dumps(bundle))
    rng = np.random.default_rng(21)
    for name in ("cal", "test"):
        rows = add_safe_seeds([json.loads(l) for l in (root / f"{name}.jsonl").read_text().splitlines()], rng, bnorms)
        write(root / f"{name}.jsonl", rows)
        if name == "test":
            write(root / "confirm.jsonl", rows)
    norms = json.loads((root / "norms.json").read_text())
    for name, base, q in (("val_cal", 7000, 3), ("val_test", 8000, 10)):
        rows = add_safe_seeds(fx.cohort(rng, name, base, q), rng, bnorms)
        write(root / f"{name}.jsonl", rows)
        write(root / f"{name}_scores.jsonl", fx.scores_for(rows, rng, norms))
    fx.e1_result(root, bundle, B)
    chs = fx.load("confirm_hide_and_seek")
    with contextlib.redirect_stdout(io.StringIO()):
        assert chs.main(fx.argv(root)[:-4] + ["--bootstrap", str(B), "--null-draws", "2"]) == 0
    (root / "amend2.json").write_text((root / "out.json").read_text())
    (root / "out.json").unlink()
    return bundle


def argv(root, out="r8.json", **over):
    a = dict(calibration="cal.jsonl", test="test.jsonl", scores_calibration="scores_cal.jsonl",
             scores_test="scores_test.jsonl", val_calibration="val_cal.jsonl", val_test="val_test.jsonl",
             val_scores_calibration="val_cal_scores.jsonl", val_scores_test="val_test_scores.jsonl",
             normalizations="norms.json", runtime_bundle="bundle.json", e1_result="e1.json",
             amend2_result="amend2.json")
    a.update(over)
    args = [x for k, v in a.items() for x in (f"--{k.replace('_', '-')}", str(root / v))]
    return args + ["--output", str(root / out), "--bootstrap", str(B)]


def run(root, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = fp.main(argv(root, **kw))
    return status, buf.getvalue(), json.loads((root / kw.get("out", "r8.json")).read_text())


class FixedPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bundle = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reproduces_registered_results_and_reports_fixed_paths(self):
        status, text, out = run(self.root)
        self.assertEqual(status, 0, text)
        self.assertIn("FIXED-PATH ANALYSIS OK", text)
        c = out["checks"]
        self.assertTrue(c["F1_e1"]["ok"] and c["F1_amend2"]["ok"] and c["F2_ok"] and c["F4_P"]["ok"])
        self.assertTrue(c["F1_decision_ci"]["ok"])
        self.assertLessEqual(c["F3_max_abs_diff"], 1e-9)
        a2 = json.loads((self.root / "amend2.json").read_text())
        for k in ("safe", "hide_and_seek"):
            self.assertEqual(out["rows"][k]["registered"]["gain_decision_ci95"], a2["detectors"][k]["gain_decision_ci95"])
        self.assertEqual(out["counts"]["validation"], len(fx.TASKS) * 2 * (3 + 10))
        for n in ("safe", "hide_and_seek"):
            gains = [out["rows"][f"{n}#{s}"]["gain"] for s in fx.SEEDS]
            self.assertAlmostEqual(out["single_models"][n]["gain"]["mean"], float(np.mean(gains)))
        self.assertIn("hide_and_seek_minus_safe", out["contrasts"])

    def test_path_is_chosen_on_validation_and_certified_on_calibration_successes(self):
        _, _, base = run(self.root)
        rk = qt.rho(qt.dev_fractions(self.bundle), cl.PRIMARY_Q)
        val = lead_time.sel(lead_time.load(self.root / "val_cal.jsonl"), 3) + \
            lead_time.sel(lead_time.load(self.root / "val_test.jsonl"), 10)
        _, cal = cl.primary(self.root / "cal.jsonl", "calibration")
        _, fu = ra.fit_alpha(cl.Cohort(val, "safe"), rk, 0.05)
        path = base["rows"]["safe"]["path"]
        self.assertEqual((path["u"], path["s"]), (fu["u"], fu["s"]))
        tm, fixed = fp.certify(cl.Cohort(cal, "safe"), rk, fu)
        self.assertEqual(path["lam"], cl.det_json(fixed)["lam"])
        self.assertEqual(cl.det_json(tm), base["P"]["path"])

    def test_fixed_path_detector_ignores_calibration_failures(self):
        _, _, base = run(self.root)
        rk = qt.rho(qt.dev_fractions(self.bundle), cl.PRIMARY_Q)
        _, cal = cl.primary(self.root / "cal.jsonl", "calibration")
        _, te = cl.primary(self.root / "test.jsonl", "test")
        S = [r for r in cal if not r["failed"]]
        F = [dict(r, safe=np.asarray(r["safe"]) + 3.0) for r in cal if r["failed"]]   # moved far up
        path = base["rows"]["safe"]["path"]
        for c in (cl.Cohort(cal, "safe"), cl.Cohort(S + F, "safe")):
            _, fu = fp.certify(c, rk, path)
            e, _ = cl.evaluate_fast(cl.Cohort(te, "safe"), rk, fu)
            self.assertEqual(e["mean_lead"], base["rows"]["safe"]["test"]["mean_lead"])
            self.assertEqual(e["fp"], base["rows"]["safe"]["test"]["fp"])

    def test_overlapping_validation_cohort_is_refused(self):
        status, text, out = run(self.root, val_calibration="cal.jsonl", val_scores_calibration="scores_cal.jsonl",
                                out="overlap.json")
        self.assertEqual(status, 1)
        self.assertIn("F2", text)
        self.assertNotIn("rows", out)


if __name__ == "__main__":
    unittest.main()
