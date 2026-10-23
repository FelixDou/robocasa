import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time"
sys.path.insert(0, str(SCRIPTS))
TASKS, HORIZON, SEEDS = ("TaskA", "TaskB", "TaskC", "TaskD"), 20, ("0", "1", "2")
NAMES = ("hide_and_seek", "hide_and_seek_first", "hide_and_seek_zero", "hide_and_seek_task_shuffle")


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort(rng, label, base, per_class):
    rows = []
    for ti, task in enumerate(TASKS):
        for i in range(2 * per_class):
            failed = i % 2 == 1
            length = HORIZON if failed else int(rng.integers(6, HORIZON - 1))
            shift = 0.8 if failed and ti % 2 else 0.0
            rows.append({
                "rollout_id": f"{label}-{task}-{i}", "task_name": task,
                "task_type": "atomic" if ti % 2 else "composite", "failed": failed, "horizon": HORIZON,
                "source_inferences": length, "environment_seed": base + 100 * ti + i, "environment_reset_index": 0,
                "safe_scores": np.maximum.accumulate(rng.standard_normal(length) + shift).tolist(),
                "time_scores": (np.arange(1, length + 1) / HORIZON).tolist(),
            })
    return rows


def scores_for(rows, rng, norms):
    out = []
    for name in NAMES:
        for r in rows:
            gain = 1.5 if name == "hide_and_seek" and r["failed"] else 0.0
            raw = {s: (rng.standard_normal(r["source_inferences"])
                       + gain * np.linspace(0, 1, r["source_inferences"])).tolist() for s in SEEDS}
            vals = [(np.maximum.accumulate(raw[s]) - norms[name][s][r["task_name"]]["location"])
                    / norms[name][s][r["task_name"]]["scale"] for s in SEEDS]
            out.append({"detector": name, "rollout_id": r["rollout_id"], "task_name": r["task_name"],
                        "failed": r["failed"], "horizon": r["horizon"], "time_scores": r["time_scores"],
                        "safe_scores": np.mean(vals, axis=0).tolist(), "per_seed_raw": raw})
    return out


def write_fixture(root, per_class_cal=10):
    rng = np.random.default_rng(4)
    cal, test = cohort(rng, "cal", 1000, per_class_cal), cohort(rng, "test", 5000, 10)
    active = [30] + [30 - min(10, max(0, i - 8)) for i in range(1, HORIZON)]
    bundle = {"task_horizons": {t: HORIZON for t in TASKS},
              "time_risk_curves": {t: {"active_counts": active} for t in TASKS}}
    norms = {n: {s: {t: {"location": 0.1 * int(s), "scale": 1.0 + 0.1 * int(s)} for t in TASKS} for s in SEEDS}
             for n in NAMES}
    files = {"cal.jsonl": cal, "test.jsonl": test, "confirm.jsonl": test,
             "scores_cal.jsonl": scores_for(cal, rng, norms), "scores_test.jsonl": scores_for(test, rng, norms)}
    for name, rows in files.items():
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "bundle.json").write_text(json.dumps(bundle))
    (root / "norms.json").write_text(json.dumps(norms))
    return bundle


def e1_result(root, bundle, B):
    """What confirm_lead run reports for SAFE + P on these cohorts (primary q)."""
    import confirm_lead as cl
    import quantile_timer as qt
    _, cal = cl.primary(root / "cal.jsonl", "calibration")
    _, te = cl.primary(root / "test.jsonl", "test")
    rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    c, e = cl.Cohort(cal), cl.Cohort(te)
    tm, fu = cl.fit_fast(c, rk)
    eP, lP = cl.evaluate_fast(e, rk, tm); eF, lF = cl.evaluate_fast(e, rk, fu)
    result = {"by_q": {str(cl.PRIMARY_Q): {"P": eP, "safe_plus_P": eF, "gain": round(float(np.mean(lF - lP)), 4),
                                           "gain_reselect_bootstrap": cl.boot_reselect(c, e, rk, B, cl.BOOT_SEED)}}}
    (root / "e1.json").write_text(json.dumps(result, default=float))
    return result


def argv(root, *extra):
    return ["run", "--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
            "--scores-calibration", str(root / "scores_cal.jsonl"), "--scores-test", str(root / "scores_test.jsonl"),
            "--normalizations", str(root / "norms.json"), "--runtime-bundle", str(root / "bundle.json"),
            "--e1-result", str(root / "e1.json"), "--confirm-predictions", str(root / "confirm.jsonl"),
            "--output", str(root / "out.json"), "--bootstrap", "30", "--null-draws", "2", *extra]


class ConfirmHideAndSeekTests(unittest.TestCase):
    def test_multi_bootstrap_equals_registered_bootstrap_per_score(self):
        module = load("confirm_hide_and_seek")
        import confirm_lead as cl
        import quantile_timer as qt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_fixture(root)
            _, cal = cl.primary(root / "cal.jsonl", "calibration")
            _, te = cl.primary(root / "test.jsonl", "test")
            module.attach(cal, module.load_scores(root / "scores_cal.jsonl"), "cal")
            module.attach(te, module.load_scores(root / "scores_test.jsonl"), "test")
            rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
            keys = ["safe", "hide_and_seek", "hide_and_seek_zero"]
            multi = module.boot_multi(cal, te, rk, keys, [("hide_and_seek", "safe")], B=25, seed=0)
            for key in keys:
                single = cl.boot_reselect(cl.Cohort(cal, key), cl.Cohort(te, key), rk, 25, 0)
                self.assertEqual(multi["gain_ci95"][key], single["gain_ci95"], key)
                self.assertEqual(multi["gain_sd"][key], single["gain_sd"], key)

    def test_run_reproduces_safe_reference_and_reports_rules(self):
        module = load("confirm_hide_and_seek")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_fixture(root)
            e1_result(root, bundle, 30)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(argv(root)), 0)
            out = json.loads((root / "out.json").read_text())
            self.assertTrue(out["checks"]["safe_reference"]["ok"])
            self.assertEqual(out["counts"]["calibration"], 80)
            self.assertEqual(set(out["H1"]), {"ci_lower_above_0", "point_gain_at_least_margin",
                                             "fused_fpr_at_most_0p075", "fused_fpr_wilson_upper_at_most_0p10",
                                             "supported"})
            self.assertEqual(set(out["H2"]["nulls"]), {"noise", "hide_and_seek_resampled"})
            self.assertIn("develops_during_rollout", out["S1_difficulty_prior"])
            self.assertIn("not_elapsed_time_or_task", out["S2_learned_timer"])
            # The planted signal makes Hide-and-Seek beat the zero-input control.
            self.assertGreater(out["S2_learned_timer"]["hs_minus_hide_and_seek_zero"]["point"], 0)

    def test_wrong_safe_reference_blocks_the_decision(self):
        module = load("confirm_hide_and_seek")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_fixture(root)
            result = e1_result(root, bundle, 30)
            result["by_q"]["0.95"]["gain"] += 0.01
            (root / "e1.json").write_text(json.dumps(result, default=float))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(module.main(argv(root)), 1)
            self.assertIn("REPRODUCTION FAILED", buffer.getvalue())
            self.assertNotIn("H1: ", buffer.getvalue())

    def test_misaligned_scores_are_rejected(self):
        module = load("confirm_hide_and_seek")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_fixture(root)
            e1_result(root, bundle, 30)
            rows = [json.loads(l) for l in (root / "scores_test.jsonl").read_text().splitlines()]
            rows[0]["safe_scores"] = rows[0]["safe_scores"][:-1]
            (root / "scores_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                module.main(argv(root))

    def test_calibration_without_ten_per_class_is_refused_in_run_mode(self):
        module = load("confirm_hide_and_seek")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_fixture(root, per_class_cal=3)
            with self.assertRaises((ValueError, SystemExit)):
                e1_result(root, bundle, 5)
                module.main(argv(root))


if __name__ == "__main__":
    unittest.main()
