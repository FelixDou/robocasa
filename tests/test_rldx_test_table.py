"""R15: RLDX-1 Table II on the test set (scripts/xr1_lead_time/rldx_test_table.py), on synthetic data.

Builds R7's validation-set table with R7's own fixtures, then E1-like calibration and test sets,
new fold directories and a shuffle directory, and checks the gates and the table.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.analyze_time_safe_hybrid import fit_safe_normalization
from tests.test_rldx_hide_and_seek import load
from tests.test_xr1_rldx_unseen import (HEADS, SEEDS, TASKS, assemble, raw_scores, write_fold_dirs, write_safe,
                                        write_safe_folds)

SHUFFLE = "hide_and_seek_task_shuffle"
FOLDS = ("0", "1", "2", "3", "all")


def quiet(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = fn(*args)
    return status, buf.getvalue()


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def normalized_mean(raw, norms, task):
    return np.mean([(np.maximum.accumulate(raw[s]) - norms[s][task]["location"]) / norms[s][task]["scale"]
                    for s in SEEDS], axis=0)


class RldxTestTableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        self.ru, self.ps, self.hs = load("rldx_unseen"), load("rldx_per_seed"), load("rldx_hide_and_seek")
        self.rtt, self.qt, self.rr = load("rldx_test_table"), load("quantile_timer"), load("rldx_replication")
        self.plan, self.safe_raw = write_safe(root / "safe")
        quiet(self.hs.prepare, __import__("argparse").Namespace(calibration_dir=str(root / "safe"),
                                                                output_dir=str(root / "inputs")))
        self.assertEqual(quiet(self.ru.main, ["folds", "--inputs-dir", str(root / "inputs"),
                                              "--write", str(root / "folds.json")])[0], 0)
        self.spec = json.loads((root / "folds.json").read_text())
        self.H = json.loads((root / "inputs" / "rldx_bundle.json").read_text())["task_horizons"]
        self.held = {str(f["fold"]): set(f["held_out"]) for f in self.spec["folds"]}
        self.held["all"] = set()
        # R7: cross-task fold directories, SAFE fold directories, the shuffle scores, per-seed rows, the table.
        (root / "r7").mkdir()
        self.r7 = write_fold_dirs(root / "r7", self.plan, self.spec, root / "folds.json", self.ru.DETECTORS, self.H)
        args = write_safe_folds(root, self.plan, self.safe_raw, self.spec, root / "folds.json", root / "inputs",
                                root / "safe")
        self.r7_safe = {}
        for (head, F), a in args.items():
            status, text = assemble(a)
            self.assertEqual(status, 0, text)
            self.r7_safe[head, F] = root / "safe_folds" / head / f"fold_{F}"
        self.write_rldxhs()
        status, text = quiet(self.ps.main, ["--safe-dir", f"lstm={root / 'safe'}", f"indep={root / 'safe'}",
                                            "--inputs-dir", str(root / "inputs"), "--scores-dir",
                                            str(root / "r7" / "score_cohorts"), str(root / "rldxhs"),
                                            "--output", str(root / "per_seed.json")])
        self.assertEqual(status, 0, text)
        status, text = quiet(self.ru.main, ["run", "--safe-dir", str(root / "safe"), "--inputs-dir", str(root / "inputs"),
                                            "--folds", str(root / "folds.json"), *self.r7_family_args("--fold-dirs"),
                                            "--reference-per-seed", str(root / "per_seed.json"),
                                            "--output", str(root / "r7_table.json")])
        self.assertEqual(status, 0, text)
        self.write_e1()

    def tearDown(self):
        self.tmp.cleanup()

    # ---- fixtures ------------------------------------------------------------------------

    def r7_family_args(self, flag):
        out = [flag, *[str(self.r7[F]) for F in FOLDS]]
        for head in HEADS:
            out += [flag, *[str(self.r7_safe[head, F]) for F in FOLDS]]
        return out

    def write_rldxhs(self):
        """A score_cohorts directory with the recorded shuffle control (report 15.22's format)."""
        d = self.root / "rldxhs"
        d.mkdir()
        raw = {s: raw_scores(self.plan, "shuffle", SHUFFLE, s) for s in SEEDS}
        train = [p for p in self.plan if p["split"] == "train"]
        norms = {s: fit_safe_normalization([{"task_name": p["task_name"], "scores": raw[s][p["rollout_id"]]}
                                            for p in train], self.H) for s in SEEDS}
        (d / "normalizations.json").write_text(json.dumps({SHUFFLE: norms}))
        for cohort, split in (("calibration", "calibration"), ("test", "evaluation"), ("train", "train")):
            write_jsonl(d / f"{cohort}_predictions.jsonl", [dict(
                detector=SHUFFLE, rollout_id=p["rollout_id"], task_name=p["task_name"], failed=p["failed"],
                safe_scores=normalized_mean({s: raw[s][p["rollout_id"]] for s in SEEDS}, norms, p["task_name"]).tolist(),
                per_seed_raw={s: raw[s][p["rollout_id"]].tolist() for s in SEEDS}) for p in self.plan if p["split"] == split])
        self.shuffle_norms = norms

    def write_e1(self):
        """E1-like sets, E1 scores, the new fold directories, the shuffle directory and the E1 files."""
        root, rng = self.root, np.random.default_rng(7)
        self.e1 = {}
        for name, n_s, n_f in (("calibration", 6, 6), ("test", 5, 5)):
            self.e1[name] = []
            for task in TASKS:
                h = self.H[task]
                for i in range(n_s + n_f):
                    failed = i >= n_s
                    self.e1[name].append(dict(rollout_id=f"{task}-e1{name}-{i}", task_name=task, failed=failed,
                                              length=h if failed else int(rng.integers(8, h - 3))))
        e1s = root / "e1_scores"
        e1s.mkdir()
        norm_hs = json.loads((self.r7["all"] / "normalizations.json").read_text())["hide_and_seek"]
        norm_lstm = json.loads((self.r7_safe["lstm", "all"] / "normalizations.json").read_text())["safe_lstm"]
        for name, plan in self.e1.items():
            for det, norms in (("hide_and_seek", norm_hs), ("safe_lstm", norm_lstm)):
                raw = {s: raw_scores(plan, "all", det, s) for s in SEEDS}
                per_task = {s: norms[s]["per_task"] for s in SEEDS}
                write_jsonl(e1s / f"{name}_{det}_predictions.jsonl", [dict(
                    detector=det, rollout_id=p["rollout_id"], task_name=p["task_name"], failed=p["failed"],
                    horizon=self.H[p["task_name"]],
                    safe_scores=normalized_mean({s: raw[s][p["rollout_id"]] for s in SEEDS}, per_task, p["task_name"]).tolist(),
                    per_seed_raw={s: raw[s][p["rollout_id"]].tolist() for s in SEEDS}) for p in plan])
        self.assertEqual(quiet(self.rtt.main, ["ids", "--inputs-dir", str(root / "inputs"), "--e1-scores-dir", str(e1s),
                                               "--output-dir", str(root / "ids")])[0], 0)
        # New cross-task fold directories: E1 sets scored, recorded cohorts and training copied (the same models).
        self.new = {}
        for F in FOLDS:
            d = root / "new" / f"fold_{F}"
            d.mkdir(parents=True)
            r7 = self.r7[F]
            shutil.copyfile(r7 / "normalizations.json", d / "normalizations.json")
            shutil.copyfile(r7 / "train_predictions.jsonl", d / "train_predictions.jsonl")
            shutil.copyfile(r7 / "calibration_predictions.jsonl", d / "rec_calibration_predictions.jsonl")
            shutil.copyfile(r7 / "test_predictions.jsonl", d / "rec_test_predictions.jsonl")
            for name, plan in self.e1.items():
                rows = []
                for det in self.ru.DETECTORS:
                    raw = {s: raw_scores(plan, F, det, s) for s in SEEDS}
                    rows += [dict(detector=det, rollout_id=p["rollout_id"], task_name=p["task_name"], failed=p["failed"],
                                  fold=F, held_out=p["task_name"] in self.held[F],
                                  per_seed_raw={s: raw[s][p["rollout_id"]].tolist() for s in SEEDS}) for p in plan]
                write_jsonl(d / f"{name}_predictions.jsonl", rows)
            meta = json.loads((r7 / "scoring.json").read_text())
            if F == "all":
                meta["reference_checks"] = {f"{c}/{det}": {"ok": True, "max_abs_diff": 0.0}
                                            for c in ("rec_calibration", "rec_test", "train") for det in self.ru.DETECTORS}
            (d / "scoring.json").write_text(json.dumps(meta))
            self.new[F] = d
        # SAFE heads: the R7 models score E1's two exports; safe-fold assembles each fold directory.
        self.new_safe = {}
        for head in HEADS:
            for F in FOLDS:
                meta7 = json.loads((self.r7_safe[head, F] / "scoring.json").read_text())
                args = ["safe-fold", "--r7-fold-dir", str(self.r7_safe[head, F]), "--folds", str(root / "folds.json"),
                        "--cohort", "calibration", str(root / "ids" / "calibration_ids.json"),
                        "--cohort", "test", str(root / "ids" / "test_ids.json"),
                        "--output-dir", str(root / "new_safe" / head / f"fold_{F}")]
                for name, plan in self.e1.items():
                    for s in SEEDS:
                        sc = root / "e1_safe_scores" / head / F / name / f"seed{s}"
                        sc.mkdir(parents=True)
                        x = raw_scores(plan, F, f"safe_{head}", s)
                        write_jsonl(sc / "scores.jsonl", [dict(rollout_id=p["rollout_id"], task_name=p["task_name"],
                                                               failed=p["failed"], scores=x[p["rollout_id"]].tolist(),
                                                               training_task=p["task_name"] not in self.held[F])
                                                          for p in plan])
                        (sc / "provenance.json").write_text(json.dumps(dict(
                            training_run_dir=meta7["training_runs"][s], group="all", unseen_tasks="score",
                            cutoff_rule="training", checkpoint_sha256=meta7["training_checkpoint_sha256"][s],
                            unseen_task_names=sorted(self.held[F]), task_names=list(TASKS), export_dir=f"e1_{name}")))
                        args += ["--scores", name, s, str(sc)]
                if head == "lstm" and F == "all":
                    for name in self.e1:
                        args += ["--e1-reference", name, str(e1s / f"{name}_safe_lstm_predictions.jsonl")]
                status, text = quiet(self.rtt.main, args)
                self.assertEqual(status, 0, text)
                self.assertIn("SAFE FOLD OK", text)
                self.new_safe[head, F] = root / "new_safe" / head / f"fold_{F}"
                self.safe_fold_args = args
        # Shuffle: E1 sets scored, recorded cohorts and training reproduced.
        d = root / "shuffle"
        d.mkdir()
        for f, g in (("normalizations.json", "normalizations.json"), ("train_predictions.jsonl", "train_predictions.jsonl"),
                     ("calibration_predictions.jsonl", "rec_calibration_predictions.jsonl"),
                     ("test_predictions.jsonl", "rec_test_predictions.jsonl")):
            shutil.copyfile(root / "rldxhs" / f, d / g)
        for name, plan in self.e1.items():
            raw = {s: raw_scores(plan, "shuffle-e1", SHUFFLE, s) for s in SEEDS}
            write_jsonl(d / f"{name}_predictions.jsonl", [dict(
                detector=SHUFFLE, rollout_id=p["rollout_id"], task_name=p["task_name"], failed=p["failed"],
                safe_scores=normalized_mean({s: raw[s][p["rollout_id"]] for s in SEEDS}, self.shuffle_norms,
                                            p["task_name"]).tolist(),
                per_seed_raw={s: raw[s][p["rollout_id"]].tolist() for s in SEEDS}) for p in plan])
        (d / "scoring.json").write_text(json.dumps({"reference_checks": {
            f"rec_calibration/{SHUFFLE}": {"ok": True}, f"rec_test/{SHUFFLE}": {"ok": True}}}))
        # E1's freeze manifest (H_k, rho_k) and E1's P on the test set.
        train = [dict(task_name=p["task_name"], failed=p["failed"], L=p["length"], horizon=self.H[p["task_name"]])
                 for p in self.plan if p["split"] == "train"]
        self.rk = self.qt.rho(self.rr.dev_fractions(train, sorted(TASKS)), self.rr.Q)
        (root / "freeze.json").write_text(json.dumps({"tasks": {t: {"H_k": self.H[t], "rho_k": self.rk[t]} for t in TASKS}}))
        self.set_e1_P()

    def set_e1_P(self, shift=0.0):
        """E1's P on its test set: the timer fitted on E1's calibration successes, evaluated as confirm_lead does."""
        cl = load("confirm_lead")
        rows = {n: [dict(task_name=p["task_name"], failed=p["failed"], L=p["length"], horizon=self.H[p["task_name"]],
                         x=np.zeros(p["length"])) for p in plan] for n, plan in self.e1.items()}
        tm, _ = cl.fit_fast(cl.Cohort(rows["calibration"], "x"), self.rk)
        e, _ = cl.evaluate_fast(cl.Cohort(rows["test"], "x"), self.rk, tm)
        (self.root / "e1_result.json").write_text(json.dumps({"P": {"test": {"fpr": e["fpr"] + shift,
                                                                            "mean_lead": e["mean_lead"]}}}))

    def run_table(self, out, new=None, extra=()):
        new = new or self.new
        args = ["run", "--safe-dir", str(self.root / "safe"), "--inputs-dir", str(self.root / "inputs"),
                "--folds", str(self.root / "folds.json"), "--ids-dir", str(self.root / "ids"),
                "--e1-scores-dir", str(self.root / "e1_scores"), "--e1-result", str(self.root / "e1_result.json"),
                "--freeze-manifest", str(self.root / "freeze.json"), "--shuffle-dir", str(self.root / "shuffle"),
                "--rldxhs-scores-dir", str(self.root / "rldxhs"), "--r7-table", str(self.root / "r7_table.json"),
                "--reference-per-seed", str(self.root / "per_seed.json"), "--output", str(self.root / out),
                "--fold-dirs", *[str(new[F]) for F in FOLDS], "--r7-fold-dirs", *[str(self.r7[F]) for F in FOLDS]]
        for head in HEADS:
            args += ["--fold-dirs", *[str(self.new_safe[head, F]) for F in FOLDS],
                     "--r7-fold-dirs", *[str(self.r7_safe[head, F]) for F in FOLDS]]
        status, text = quiet(self.rtt.main, args + list(extra))
        return status, text, json.loads((self.root / out).read_text())

    # ---- tests ---------------------------------------------------------------------------

    def test_gates_pass_and_the_table_uses_the_test_set(self):
        status, text, out = self.run_table("table.json")
        self.assertEqual(status, 0, text)
        self.assertIn("RLDX-1 TEST TABLE OK", text)
        self.assertEqual(out["failures"], [])
        self.assertEqual(out["checks"]["G0_code_path"]["rows_differing"], [])
        self.assertTrue(out["checks"]["G2c_trajectories_equal_E1"]["ok"])
        n_test = {k: sum(p["failed"] == k for p in self.e1["test"]) for k in (False, True)}
        n_cal_f = sum(p["failed"] for p in self.e1["calibration"])
        rows = out["table2"]["calibration"]
        for d in list(self.ru.DETECTORS) + ["safe_lstm", "safe_indep"]:
            u = rows[d]["unseen"]["seed_average"]
            self.assertEqual((u["n_success"], u["n_failure"]), (n_test[False], n_test[True]))
            # The calibration failures choose the path: every fold sees its seen tasks' E1 failures.
            self.assertEqual(sum(p["cal_failures"] for p in u["paths"].values()), 3 * n_cal_f)
            s = rows[d]["seen"]["seed_average"]
            self.assertAlmostEqual(s["twa"], (s["mean_lead"] + 1 - s["fpr"]) / 2)
        self.assertNotIn("unseen", rows[SHUFFLE])
        self.assertIn("timer_G", rows)
        self.assertEqual(rows["timer_G"]["seen"]["gain_over_P"],
                         round(rows["timer_G"]["seen"]["mean_lead"] - rows["timer_P"]["seen"]["mean_lead"], 4))
        # The R7 protocol (training failures choose the path) is reported too, on the same sets.
        train_rows = out["table2"]["train"]
        u = train_rows["hide_and_seek"]["unseen"]["seed_average"]
        self.assertEqual(sum(p["cal_failures"] for p in u["paths"].values()),
                         sum(p["failed"] and p["split"] == "train" and p["task_name"] not in self.held[F]
                             for F in ("0", "1", "2", "3") for p in self.plan))

    def test_a_retrained_fold_that_differs_from_R7_stops_the_table(self):
        path = self.new["2"] / "rec_test_predictions.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        rows[3]["per_seed_raw"]["1"][0] += 1e-9
        write_jsonl(path, rows)
        status, text, out = self.run_table("bad.json")
        self.assertEqual(status, 1)
        self.assertIn("G1_models", out["failures"])
        self.assertNotIn("table2", out)

    def test_normalizations_get_the_trajectory_tolerance(self):
        """G1 accepts normalizations within 1e-12 of R7's (floating-point noise) and refuses larger gaps."""
        path = self.new["1"] / "normalizations.json"
        norms = json.loads(path.read_text())
        norms["rnd"]["0"]["pooled"]["location"] += 1e-14
        path.write_text(json.dumps(norms))
        status, text, out = self.run_table("noise.json")
        self.assertEqual(status, 0, text)
        g = out["checks"]["G1_models"]["families"]["+".join(self.ru.DETECTORS)]["1"]
        self.assertGreater(g["normalizations_max_abs_diff"], 0.0)
        self.assertEqual(list(g["normalizations_by_detector"]), ["rnd"])
        norms["rnd"]["0"]["pooled"]["location"] += 1e-9
        path.write_text(json.dumps(norms))
        status, text, out = self.run_table("drift.json")
        self.assertEqual(status, 1)
        self.assertIn("G1_models", out["failures"])

    def test_E1_P_and_trajectories_are_gates(self):
        self.set_e1_P(shift=0.01)
        status, text, out = self.run_table("bad_p.json")
        self.assertEqual(status, 1)
        self.assertEqual(out["failures"], ["G2b_P_equals_E1"])
        self.set_e1_P()
        path = self.root / "e1_scores" / "test_hide_and_seek_predictions.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        rows[0]["safe_scores"][0] += 1e-6
        write_jsonl(path, rows)
        status, text, out = self.run_table("bad_traj.json")
        self.assertEqual(status, 1)
        self.assertEqual(out["failures"], ["G2c_trajectories_equal_E1"])

    def test_G0_catches_a_changed_published_row(self):
        t = json.loads((self.root / "r7_table.json").read_text())
        t["table2"]["rnd"]["unseen"]["seed_1"]["gain"] += 0.0001
        (self.root / "r7_table.json").write_text(json.dumps(t))
        status, text, out = self.run_table("bad_g0.json")
        self.assertEqual(status, 1)
        self.assertEqual(out["checks"]["G0_code_path"]["rows_differing"], ["rnd"])

    def test_safe_fold_refuses_another_checkpoint(self):
        args = list(self.safe_fold_args)
        args[args.index("--output-dir") + 1] += "_again"
        sc = Path(args[args.index("--scores") + 3])
        prov = json.loads((sc / "provenance.json").read_text())
        (sc / "provenance.json").write_text(json.dumps(dict(prov, checkpoint_sha256="0" * 64)))
        with self.assertRaises(SystemExit) as e:
            quiet(self.rtt.main, args)
        self.assertIn("checkpoint hash differs", str(e.exception))

    def test_ids_refuse_overlap_with_recorded_sets(self):
        path = self.root / "e1_scores" / "test_safe_lstm_predictions.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        rows[0]["rollout_id"] = json.loads((self.root / "inputs" / "test_ids.json").read_text())[0]
        write_jsonl(path, rows)
        with self.assertRaises(SystemExit) as e:
            quiet(self.rtt.main, ["ids", "--inputs-dir", str(self.root / "inputs"),
                                  "--e1-scores-dir", str(self.root / "e1_scores"), "--output-dir", str(self.root / "ids2")])
        self.assertIn("shares", str(e.exception))


if __name__ == "__main__":
    unittest.main()
