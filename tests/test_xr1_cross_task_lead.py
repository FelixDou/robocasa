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
ATOMIC, COMPOSITE = ("A1", "A2", "A3", "A4"), ("C1", "C2", "C3", "C4")
TASKS, HORIZON, SEEDS = ATOMIC + COMPOSITE, 20, ("0", "1", "2")
FOLDS = ("0", "1", "2", "3", "all")


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ct = load("cross_task_lead")
import confirm_lead as cl
import quantile_timer as qt


def cohort(rng, label, base):
    rows = []
    for ti, task in enumerate(TASKS):
        for i in range(20):
            failed = i % 2 == 1
            length = HORIZON if failed else int(rng.integers(6, HORIZON - 1))
            rows.append({
                "rollout_id": f"{label}-{task}-{i}", "task_name": task,
                "task_type": "atomic" if task in ATOMIC else "composite", "failed": failed, "horizon": HORIZON,
                "source_inferences": length, "environment_seed": base + 100 * ti + i, "environment_reset_index": 0,
                "safe_scores": np.maximum.accumulate(rng.standard_normal(length)).tolist(),
                "time_scores": (np.arange(1, length + 1) / HORIZON).tolist(),
            })
    return rows


def raw_scores(rng, r, detector, fold):
    signal = {"hide_and_seek": 1.5, "hide_and_seek_zero": 0.3, "hide_and_seek_first": 0.0, "cosine_knn": 0.8}[detector]
    if fold != "all":
        signal *= 0.8  # a weaker detector on tasks it was not trained on
    ramp = signal * np.linspace(0, 1, r["source_inferences"]) if r["failed"] else 0.0
    return {s: (rng.standard_normal(r["source_inferences"]) + ramp).tolist() for s in SEEDS}


def write_fold(root, F, held, cal, test, rng, spec):
    d = root / f"fold_{F}"
    d.mkdir()
    norms = {det: {s: {"pooled": {"location": 0.2 + 0.1 * int(s), "scale": 1.1},
                       "per_task": {t: {"location": 0.1 * (ti % 3) + 0.05 * int(s), "scale": 1.0 + 0.1 * (ti % 2)}
                                    for ti, t in enumerate(TASKS) if t not in held}}
                   for s in SEEDS} for det in ct.DETECTORS}
    for name, rows in (("lead_calibration", cal), ("confirmatory_test", test)):
        with (d / f"{name}_predictions.jsonl").open("w") as stream:
            for det in ct.DETECTORS:
                for r in rows:
                    stream.write(json.dumps({"detector": det, "rollout_id": r["rollout_id"], "task_name": r["task_name"],
                                             "failed": r["failed"], "fold": F, "held_out": r["task_name"] in held,
                                             "per_seed_raw": raw_scores(rng, r, det, F)}) + "\n")
    meta = {"fold": F, "held_out_tasks": sorted(held),
            "training_rollouts_by_task": {t: 30 for t in TASKS if t not in held},
            "folds_sha256": ct.sha256(root / "folds.json"), "train_ids_sha256": "x", "seeds": [0, 1, 2],
            "epochs": 200, "feature_aggregation": "last", "detectors": list(ct.DETECTORS), "code_commit": "test"}
    if F == "all":
        meta["reference_checks"] = {f"{c}/{det}": {"ok": True, "max_abs_diff": 0.0}
                                    for c in ("lead_calibration", "confirmatory_test") for det in ct.AMEND2_DETECTORS}
    (d / "scoring.json").write_text(json.dumps(meta))
    (d / "normalizations.json").write_text(json.dumps(norms))
    return d, norms


def amend2_from(root, bundle, norms_all):
    """What the Amendment 2 analysis reports on these cohorts (per-task normalization, fold all)."""
    _, cal = cl.primary(root / "cal.jsonl", "calibration")
    _, te = cl.primary(root / "test.jsonl", "test")
    scores = {}
    for name in ("lead_calibration", "confirmatory_test"):
        for line in (root / "fold_all" / f"{name}_predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            scores[(row["detector"], row["rollout_id"])] = row["per_seed_raw"]
    rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    out = {"detectors": {}}
    for det in ct.AMEND2_DETECTORS:
        for r in (*cal, *te):
            raw = {s: np.asarray(v, float) for s, v in scores[(det, r["rollout_id"])].items()}
            r[det] = ct.normalized(raw, {s: norms_all[det][s]["per_task"][r["task_name"]] for s in SEEDS})
        c, e = cl.Cohort(cal, det), cl.Cohort(te, det)
        tm, fu = cl.fit_fast(c, rk)
        eP, lP = cl.evaluate_fast(e, rk, tm)
        eF, lF = cl.evaluate_fast(e, rk, fu)
        out["P"] = eP
        out["detectors"][det] = {"test": eF, "gain": round(float(np.mean(lF - lP)), 4)}
    norms = {det: {s: norms_all[det][s]["per_task"] for s in SEEDS} for det in ct.AMEND2_DETECTORS}
    return out, norms


def fixture(root):
    rng = np.random.default_rng(7)
    cal, test = cohort(rng, "cal", 1000), cohort(rng, "test", 5000)
    for name, rows in (("cal.jsonl", cal), ("test.jsonl", test)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = ct.make_folds({t: "atomic" if t in ATOMIC else "composite" for t in TASKS})
    (root / "folds.json").write_text(json.dumps(spec, indent=1) + "\n")
    active = [30] + [30 - min(12, max(0, i - 7)) for i in range(1, HORIZON)]
    bundle = {"task_horizons": {t: HORIZON for t in TASKS},
              "time_risk_curves": {t: {"active_counts": active} for t in TASKS}}
    (root / "bundle.json").write_text(json.dumps(bundle))
    held = {str(f["fold"]): set(f["held_out"]) for f in spec["folds"]}
    held["all"] = set()
    dirs = {}
    for F in FOLDS:
        dirs[F], norms = write_fold(root, F, held[F], cal, test, rng, spec)
        if F == "all":
            norms_all = norms
    amend2, amend2_norms = amend2_from(root, bundle, norms_all)
    (root / "amend2.json").write_text(json.dumps(amend2, default=float))
    (root / "amend2_norms.json").write_text(json.dumps(amend2_norms))
    return spec, held, dirs


def run_args(root, dirs, out="out.json", boot=40):
    return ["run", "--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
            "--fold-dirs", *[str(d) for d in dirs.values()], "--runtime-bundle", str(root / "bundle.json"),
            "--amend2-result", str(root / "amend2.json"), "--amend2-normalizations", str(root / "amend2_norms.json"),
            "--folds", str(root / "folds.json"), "--output", str(root / out), "--bootstrap", str(boot),
            "--null-draws", "3", "--few-draws", "3"]


class FoldTests(unittest.TestCase):
    def test_folds_are_stratified_deterministic_and_cover_every_task_once(self):
        types = {t: "atomic" if t in ATOMIC else "composite" for t in TASKS}
        spec = ct.make_folds(types)
        self.assertEqual(spec, ct.make_folds(dict(reversed(list(types.items())))))
        tasks, held = ct.fold_sets(spec)
        self.assertEqual(tasks, set(TASKS))
        for f in spec["folds"]:
            self.assertEqual(sorted(types[t] for t in f["held_out"]), ["atomic", "composite"])
        with self.assertRaises(ValueError):
            ct.make_folds({**types, "X": None})

    def test_committed_fold_file_is_the_registered_rule(self):
        spec = json.loads(ct.FOLDS_FILE.read_text())
        self.assertEqual((spec["seed"], spec["k"]), (ct.FOLD_SEED, ct.K))
        self.assertEqual(json.loads(json.dumps(ct.make_folds(spec["tasks"]))), spec)
        self.assertEqual(len(spec["tasks"]), 38)
        sizes = sorted(len(f["held_out"]) for f in spec["folds"])
        self.assertEqual(sizes, [9, 9, 10, 10])
        for f in spec["folds"]:
            n_atomic = sum(spec["tasks"][t] == "atomic" for t in f["held_out"])
            self.assertIn(n_atomic, (2, 3))

    def test_folds_command_checks_the_committed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ct.main(["folds", "--predictions", str(root / "cal.jsonl"),
                                          "--folds", str(root / "folds.json")]), 0)
                spec = json.loads((root / "folds.json").read_text())
                spec["folds"][0]["held_out"], spec["folds"][1]["held_out"] = spec["folds"][1]["held_out"], spec["folds"][0]["held_out"]
                (root / "bad.json").write_text(json.dumps(spec))
                self.assertEqual(ct.main(["folds", "--predictions", str(root / "cal.jsonl"),
                                          "--folds", str(root / "bad.json")]), 1)


class HelperTests(unittest.TestCase):
    def test_rho_n_uses_at_most_the_available_successes(self):
        dev = {"a": [0.1, 0.2, 0.3], "b": [], "c": [0.5] * 10}
        rk = ct.rho_n(dev, 5, np.random.default_rng(0))
        self.assertAlmostEqual(rk["a"], float(np.quantile([0.1, 0.2, 0.3], 0.95)))
        self.assertEqual(rk["b"], 1.0)
        self.assertEqual(rk["c"], 0.5)

    def test_unseen_fits_on_seen_tasks_and_tests_on_held_out_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, held, dirs = fixture(root)
            _, cal = cl.primary(root / "cal.jsonl", "calibration")
            _, te = cl.primary(root / "test.jsonl", "test")
            wanted = {"lead_calibration": {r["rollout_id"] for r in cal}, "confirmatory_test": {r["rollout_id"] for r in te}}
            data = {F: ct.load_fold_dir(d, wanted) for F, d in dirs.items()}
            ct.attach(cal, data, "lead_calibration", held)
            ct.attach(te, data, "confirmatory_test", held)
            bundle = json.loads((root / "bundle.json").read_text())
            keys = [f"{d}@{F}" for F in FOLDS for d in ct.DETECTORS]
            D = ct.Design(cal, te, held, keys, qt.dev_fractions(bundle))
            t, f, per_fold = ct.unseen(D, lambda F: f"hide_and_seek@{F}", D.rk0)
            self.assertEqual(len(f["lead"]), sum(r["failed"] for r in te))
            self.assertEqual(f["n"], sum(not r["failed"] for r in te))
            # Fold 0 by hand: calibration without its tasks, test with only its tasks.
            F = "0"
            c_rows = [r for r in cal if r["task_name"] not in held[F]]
            e_rows = [r for r in te if r["task_name"] in held[F]]
            tm, fu = cl.fit_fast(cl.Cohort(c_rows, f"hide_and_seek@{F}"), D.rk0)
            self.assertEqual((tm, fu), per_fold[F][:2])
            S = [r for r in c_rows if not r["failed"]]
            self.assertAlmostEqual(tm["lam"], cl.conformal(np.array([r["L"] / r["horizon"] for r in S]) + cl.EPS))
            _, lead = cl.evaluate_fast(cl.Cohort(e_rows, f"hide_and_seek@{F}"), D.rk0, fu)
            np.testing.assert_array_equal(lead, per_fold[F][3]["lead"])
            # Bootstrap draws keep the same separation.
            cal_sel, te_sel = ct.draw(D, np.random.default_rng(3))
            for F in D.folds:
                self.assertFalse({cal[i]["task_name"] for i in cal_sel[F]} & held[F])
                self.assertTrue({te[i]["task_name"] for i in te_sel[F]} <= held[F])
            self.assertEqual(sum(len(te_sel[F]) for F in D.folds), len(te_sel["all"]))


class RunTests(unittest.TestCase):
    def test_run_reports_every_declared_quantity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, held, dirs = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                status = ct.main(run_args(root, dirs))
            self.assertEqual(status, 0)
            out = json.loads((root / "out.json").read_text())
            self.assertTrue(out["checks"]["amend2_reproduction"]["ok"])
            self.assertEqual(out["counts"]["test"], 160)
            for key in ("U1_zero_shot", "U1_null_sanity", "U2_learned_timer", "U3_difficulty_prior", "U4_transfer_loss",
                        "U5_few_shot", "U6_success_only_knn", "U7_fpr_transfer"):
                self.assertIn(key, out)
            self.assertEqual(set(out["U1_zero_shot"]), {"ci_lower_above_0", "point_gain_at_least_margin",
                                                        "fused_fpr_at_most_0p075", "fused_fpr_wilson_upper_at_most_0p10",
                                                        "supported"})
            self.assertEqual(set(out["U5_few_shot"]["by_n"]), {"1", "3", "5", "all"})
            self.assertGreater(out["detectors"]["hide_and_seek"]["gain"], out["detectors"]["hide_and_seek_first"]["gain"])
            self.assertAlmostEqual(out["U4_transfer_loss"]["point"],
                                   out["detectors"]["hide_and_seek"]["gain"] - out["seen_reference"]["gain"], delta=1.5e-4)
            self.assertEqual(sum(v["failures"] for v in out["descriptive"]["by_task_type"].values()), 80)
            self.assertEqual(out["seen_reference"]["hs_per_task_gain_over_P"],
                             json.loads((root / "amend2.json").read_text())["detectors"]["hide_and_seek"]["gain"])
            # Same inputs, same output.
            with contextlib.redirect_stdout(io.StringIO()):
                ct.main(run_args(root, dirs, "again.json"))
            again = json.loads((root / "again.json").read_text())
            for k in ("inputs",):
                out.pop(k); again.pop(k)
            self.assertEqual(out, again)

    def test_no_result_without_amendment_2_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, held, dirs = fixture(root)
            a = json.loads((root / "amend2.json").read_text())
            a["detectors"]["hide_and_seek"]["gain"] += 0.001
            (root / "amend2.json").write_text(json.dumps(a))
            with contextlib.redirect_stdout(io.StringIO()):
                status = ct.main(run_args(root, dirs))
            self.assertEqual(status, 1)
            out = json.loads((root / "out.json").read_text())
            self.assertTrue(out["reproduction_failed"])
            self.assertNotIn("U1_zero_shot", out)

    def test_fold_scored_with_a_held_out_task_in_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, held, dirs = fixture(root)
            meta = json.loads((dirs["1"] / "scoring.json").read_text())
            meta["training_rollouts_by_task"][sorted(held["1"])[0]] = 3
            (dirs["1"] / "scoring.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                ct.main(run_args(root, dirs))

    def test_fold_all_without_raw_reproduction_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, held, dirs = fixture(root)
            meta = json.loads((dirs["all"] / "scoring.json").read_text())
            meta["reference_checks"]["confirmatory_test/hide_and_seek"]["ok"] = False
            (dirs["all"] / "scoring.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                ct.main(run_args(root, dirs))


if __name__ == "__main__":
    unittest.main()
