"""R7: RLDX-1 seen + P and unseen + G (scripts/xr1_lead_time/rldx_unseen.py), on synthetic data."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.analyze_time_safe_hybrid import fit_safe_normalization
from robocasa.recovery.safe.baselines.score_cross_task import pooled_normalization
from robocasa.recovery.safe.calibrate_seen_tasks import fit_task_normalization
from robocasa.recovery.safe.cross_task_safe import main as cross_task_safe
from tests.test_rldx_hide_and_seek import load

TASKS = tuple(f"Task{c}" for c in "ABCDEFGH")
SEEDS = ("0", "1", "2")
COHORT_OF = {"train": "train", "calibration": "calibration", "evaluation": "test"}


HEADS = ("lstm", "indep")


def write_safe(root, seed=0):
    """Frozen SAFE20-style normalized scores of both heads; the LSTM's define the 15.15 rows.

    Each row has the raw trajectory (full_raw_scores) and SAFE's own task normalization
    (training_task_early_max_z: mean and std of the training maxima before task_min_step);
    full_scores = (full_raw_scores - location) / scale. Returns the plan and the raw scores.
    """
    rng = np.random.default_rng(seed)
    plan = []
    for ti, task in enumerate(TASKS):
        horizon = 24 + ti
        for split, n_s, n_f in (("train", 10, 9), ("calibration", 8, 0), ("evaluation", 5, 5)):
            for i in range(n_s + n_f):
                failed = i >= n_s
                length = horizon if failed else int(rng.integers(8, horizon - 3))
                plan.append(dict(rollout_id=f"{task}-{split}-{i}", task_name=task,
                                 task_type="atomic" if ti % 2 else "composite", failed=failed, split=split,
                                 length=length, environment_seed=1000 * ti + i))
    tms = task_min_steps(plan)
    raw = {}
    for head in HEADS:
        for s in SEEDS:
            raw[head, s] = {p["rollout_id"]: rng.standard_normal(p["length"]) + (0.5 if p["failed"] else 0.0)
                            for p in plan}
            norm = fit_task_normalization([dict(split="train", task_name=p["task_name"], task_min_step=tms[p["task_name"]],
                                                scores=raw[head, s][p["rollout_id"]].tolist())
                                           for p in plan if p["split"] == "train"])
            path = root / f"{head}_seed{s}" / "normalized_scores.jsonl"
            path.parent.mkdir(parents=True)
            with path.open("w") as stream:
                for p in plan:
                    x, n = raw[head, s][p["rollout_id"]], norm[p["task_name"]]
                    stream.write(json.dumps({k: v for k, v in p.items() if k != "length"} | {
                        "full_scores": ((x - n["location"]) / n["scale"]).tolist(), "full_raw_scores": x.tolist(),
                        "normalization_location": n["location"], "normalization_scale": n["scale"]}) + "\n")
    return plan, raw


def task_min_steps(plan):
    """SAFE's cutoff: the shortest training rollout of each task."""
    out = {}
    for p in plan:
        if p["split"] == "train":
            out[p["task_name"]] = min(out.get(p["task_name"], p["length"]), p["length"])
    return out


def raw_scores(plan, fold, detector, seed):
    """A deterministic raw trajectory per (fold, detector, seed, rollout); failures drift upward."""
    out = {}
    for p in plan:
        key = f"{fold}|{detector}|{seed}|{p['rollout_id']}"
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        drift = np.linspace(0, 1.5, p["length"]) if p["failed"] else np.zeros(p["length"])
        out[p["rollout_id"]] = rng.standard_normal(p["length"]) * 0.6 + drift
    return out


def write_fold_dirs(root, plan, spec, folds_path, detectors, horizons):
    """score_cross_task-format directories (with train_predictions) and a score_cohorts-format copy of fold all."""
    held = {str(f["fold"]): set(f["held_out"]) for f in spec["folds"]}
    held["all"] = set()
    dirs = {}
    for F, out in held.items():
        d = root / f"fold_{F}"
        d.mkdir()
        seen = [t for t in TASKS if t not in out]
        train = [p for p in plan if p["split"] == "train" and p["task_name"] in seen]
        raw = {(det, s): raw_scores(plan, F, det, s) for det in detectors for s in SEEDS}
        norms = {}
        for det in detectors:
            norms[det] = {}
            for s in SEEDS:
                records = [{"task_name": p["task_name"], "scores": raw[det, s][p["rollout_id"]]} for p in train]
                norms[det][s] = {"pooled": pooled_normalization(records, horizons),
                                 "per_task": fit_safe_normalization(records, {t: horizons[t] for t in seen})}
        (d / "normalizations.json").write_text(json.dumps(norms))
        streams = {c: (d / f"{c}_predictions.jsonl").open("w") for c in ("calibration", "test", "train")}
        for det in detectors:
            for p in plan:
                cohort = COHORT_OF[p["split"]]
                if cohort == "train" and p["task_name"] in out:
                    continue
                streams[cohort].write(json.dumps({
                    "detector": det, "rollout_id": p["rollout_id"], "task_name": p["task_name"], "failed": p["failed"],
                    "fold": F, "held_out": p["task_name"] in out,
                    "per_seed_raw": {s: raw[det, s][p["rollout_id"]].tolist() for s in SEEDS}}) + "\n")
        for stream in streams.values():
            stream.close()
        meta = {"fold": F, "held_out_tasks": sorted(out), "seeds": [0, 1, 2], "detectors": list(detectors),
                "training_rollouts_by_task": {t: sum(p["task_name"] == t for p in train) for t in seen},
                "folds_sha256": hashlib.sha256(folds_path.read_bytes()).hexdigest(), "score_train": True,
                "code_commit": "test"}
        if F == "all":
            meta["reference_checks"] = {f"{c}/{det}": {"ok": True, "max_abs_diff": 0.0}
                                        for c in ("calibration", "test", "train") for det in detectors}
            sc = root / "score_cohorts"
            sc.mkdir()
            (sc / "normalizations.json").write_text(json.dumps(
                {det: {s: norms[det][s]["per_task"] for s in SEEDS} for det in detectors}))
            files = {c: (sc / f"{c}_predictions.jsonl").open("w") for c in ("calibration", "test", "train")}
            for det in detectors:
                for p in plan:
                    rid, task = p["rollout_id"], p["task_name"]
                    safe = np.mean([(np.maximum.accumulate(raw[det, s][rid]) - norms[det][s]["per_task"][task]["location"])
                                    / norms[det][s]["per_task"][task]["scale"] for s in SEEDS], axis=0)
                    files[COHORT_OF[p["split"]]].write(json.dumps({
                        "detector": det, "rollout_id": rid, "task_name": task, "failed": p["failed"],
                        "safe_scores": safe.tolist(), "per_seed_raw": {s: raw[det, s][rid].tolist() for s in SEEDS}}) + "\n")
            for stream in files.values():
                stream.close()
        (d / "scoring.json").write_text(json.dumps(meta))
        dirs[F] = d
    return dirs


def write_safe_folds(root, plan, raw, spec, folds_path, inputs, frozen_dir):
    """Training runs, whole-export scores and assemble-export fold directories for both SAFE heads."""
    held = {str(f["fold"]): set(f["held_out"]) for f in spec["folds"]}
    held["all"] = set()
    tasks = sorted({p["task_name"] for p in plan})
    tms_all = task_min_steps(plan)
    dirs = {}
    for head in HEADS:
        for F, out in held.items():
            seen = [t for t in tasks if t not in out]
            tms = {t: tms_all[t] for t in seen}
            manifest = root / "safe_manifests" / f"fold_{F}.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}")
            args = ["assemble-export", "--fold", F, "--folds", str(folds_path), "--head", head,
                    "--runtime-bundle", str(inputs / "rldx_bundle.json"),
                    "--cohort", "train", str(inputs / "train_ids.json"),
                    "--cohort", "calibration", str(inputs / "calibration_ids.json"),
                    "--cohort", "test", str(inputs / "test_ids.json"),
                    "--output-dir", str(root / "safe_folds" / head / f"fold_{F}")]
            if F != "all":
                args += ["--fold-manifest", str(manifest)]
            for s in SEEDS:
                x = raw[head, s] if F == "all" else raw_scores(plan, F, f"safe_{head}", s)
                run = root / "safe_runs" / head / F / f"seed{s}"
                run.mkdir(parents=True)
                (run / "model_final.ckpt").write_bytes(f"{head}|{F}|{s}".encode())
                (run / "metrics.json").write_text(json.dumps(dict(seed=int(s), model=head, task_min_steps=tms,
                                                                  selection_manifest=str(manifest))))
                (run / "split_manifest.json").write_text(json.dumps(dict(task_names=seen)))
                with (run / "scores.jsonl").open("w") as stream:
                    for p in plan:
                        if p["task_name"] in seen:
                            stream.write(json.dumps(dict(
                                rollout_id=p["rollout_id"], task_name=p["task_name"], failed=p["failed"],
                                split="train" if p["split"] == "train" else "test",
                                task_min_step=tms[p["task_name"]], scores=x[p["rollout_id"]].tolist())) + "\n")
                sc = root / "safe_scores" / head / F / f"seed{s}"
                sc.mkdir(parents=True)
                with (sc / "scores.jsonl").open("w") as stream:
                    for p in plan:
                        stream.write(json.dumps(dict(rollout_id=p["rollout_id"], task_name=p["task_name"],
                                                     failed=p["failed"], scores=x[p["rollout_id"]].tolist(),
                                                     training_task=p["task_name"] in seen)) + "\n")
                (sc / "provenance.json").write_text(json.dumps(dict(
                    training_run_dir=str(run), group="all", unseen_tasks="score", cutoff_rule="training",
                    checkpoint_sha256=hashlib.sha256((run / "model_final.ckpt").read_bytes()).hexdigest(),
                    unseen_task_names=sorted(out), task_names=tasks, export_dir="export")))
                args += ["--training-run", s, str(run), "--scores", s, str(sc)]
                if F == "all":
                    args += ["--frozen-normalized", s, str(frozen_dir / f"{head}_seed{s}" / "normalized_scores.jsonl")]
            dirs[head, F] = args
    return dirs


def assemble(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = cross_task_safe(args)
    return status, buf.getvalue()


class RldxUnseenTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        self.ru, self.ps, self.hs = load("rldx_unseen"), load("rldx_per_seed"), load("rldx_hide_and_seek")
        self.plan, self.safe_raw = write_safe(root / "safe")
        with contextlib.redirect_stdout(io.StringIO()):
            self.hs.prepare(argparse.Namespace(calibration_dir=str(root / "safe"), output_dir=str(root / "inputs")))
            self.assertEqual(self.ru.main(["folds", "--inputs-dir", str(root / "inputs"),
                                           "--write", str(root / "folds.json")]), 0)
        self.spec = json.loads((root / "folds.json").read_text())
        horizons = json.loads((root / "inputs" / "rldx_bundle.json").read_text())["task_horizons"]
        self.dirs = write_fold_dirs(root, self.plan, self.spec, root / "folds.json", self.ru.DETECTORS, horizons)
        with contextlib.redirect_stdout(io.StringIO()):
            self.ps.main(["--safe-dir", f"lstm={root / 'safe'}", "--inputs-dir", str(root / "inputs"),
                          "--scores-dir", str(root / "score_cohorts"), "--output", str(root / "per_seed.json")])

    def tearDown(self):
        self.tmp.cleanup()

    def run_unseen(self, out, reference=None, dirs=None, families=()):
        dirs = dirs or [self.dirs[F] for F in ("0", "1", "2", "3", "all")]
        extra = [a for family in families for a in ("--fold-dirs", *map(str, family))]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            status = self.ru.main(["run", "--safe-dir", str(self.root / "safe"), "--inputs-dir", str(self.root / "inputs"),
                                   "--folds", str(self.root / "folds.json"), "--fold-dirs", *map(str, dirs), *extra,
                                   "--reference-per-seed", str(reference or self.root / "per_seed.json"),
                                   "--output", str(self.root / out)])
        return status, buf.getvalue(), json.loads((self.root / out).read_text())

    def test_folds_are_stratified_and_written_once(self):
        self.assertEqual((self.spec["seed"], self.spec["k"]), (self.ru.FOLD_SEED, 4))
        self.assertEqual(sorted(t for f in self.spec["folds"] for t in f["held_out"]), list(TASKS))
        for f in self.spec["folds"]:
            types = [self.spec["tasks"][t] for t in f["held_out"]]
            self.assertEqual(sorted(types), ["atomic", "composite"])
        with self.assertRaises(SystemExit):
            self.ru.main(["folds", "--inputs-dir", str(self.root / "inputs"), "--write", str(self.root / "folds.json")])

    def test_seen_reproduces_rldx_per_seed_and_unseen_pools_held_out_tasks(self):
        status, text, out = self.run_unseen("out.json")
        self.assertEqual(status, 0, text)
        self.assertIn("RLDX-1 UNSEEN TABLE OK", text)
        self.assertTrue(out["checks"]["R2_seen_plus_P"]["ok"])
        self.assertEqual(out["checks"]["R2_seen_plus_P"]["max_abs_diff"], 0.0)
        self.assertTrue(out["checks"]["R3_one_timer"]["ok"])
        ref = json.loads((self.root / "per_seed.json").read_text())
        for d in self.ru.DETECTORS:
            got = out["table2"][d]["seen"]
            for s in SEEDS:
                self.assertEqual(got[f"seed_{s}"]["mean_lead"], ref["results"][d]["per_seed"][s]["lead"])
            u = out["table2"][d]["unseen"]
            for v in [u["seed_average"]] + [u[f"seed_{s}"] for s in SEEDS]:
                # Every task is held out once: the pooled test set is the whole test cohort.
                self.assertEqual((v["n_success"], v["n_failure"]), (40, 40))
                for F, p in v["paths"].items():
                    self.assertEqual((p["cal_successes"], p["cal_failures"], p["test"]), (48, 54, 20))
                self.assertAlmostEqual(v["twa"], (v["mean_lead"] + 1 - v["fpr"]) / 2)
            self.assertAlmostEqual(u["mean_std_over_seeds"]["gain"]["mean"],
                                   np.mean([u[f"seed_{s}"]["gain"] for s in SEEDS]))

    def test_unseen_fit_ignores_held_out_calibration_and_uses_fold_scores(self):
        _, _, base = self.run_unseen("base.json")
        held = self.spec["folds"][0]["held_out"][0]
        # Fold 0's scores of its held-out task's calibration successes are not used anywhere.
        path = self.dirs["0"] / "calibration_predictions.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        for r in rows:
            if r["task_name"] == held:
                r["per_seed_raw"] = {s: (np.asarray(v) + 50.0).tolist() for s, v in r["per_seed_raw"].items()}
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        _, _, same = self.run_unseen("same.json")
        self.assertEqual(json.dumps(same["table2"], sort_keys=True), json.dumps(base["table2"], sort_keys=True))
        # Fold 0's scores of its held-out task's test rollouts are what the unseen arm evaluates.
        path = self.dirs["0"] / "test_predictions.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        for r in rows:
            if r["task_name"] == held and r["failed"]:
                r["per_seed_raw"] = {s: (np.asarray(v) + 50.0).tolist() for s, v in r["per_seed_raw"].items()}
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        _, _, moved = self.run_unseen("moved.json")
        d = self.ru.DETECTORS[0]
        self.assertNotEqual(moved["table2"][d]["unseen"]["seed_average"]["mean_lead"],
                            base["table2"][d]["unseen"]["seed_average"]["mean_lead"])
        self.assertEqual(moved["table2"][d]["seen"], base["table2"][d]["seen"])

    def test_gates_refuse(self):
        ref = json.loads((self.root / "per_seed.json").read_text())
        ref["results"]["rnd"]["per_seed"]["1"]["lead"] += 0.001
        bad = self.root / "bad_ref.json"
        bad.write_text(json.dumps(ref))
        status, text, out = self.run_unseen("bad.json", reference=bad)
        self.assertEqual(status, 1)
        self.assertIn("REPRODUCTION FAILED", text)
        self.assertNotIn("table2", out)
        meta_path = self.dirs["2"] / "scoring.json"
        meta = json.loads(meta_path.read_text())
        meta["score_train"] = False
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "score-train"):
            self.run_unseen("no_train.json")
        meta["score_train"] = True
        meta_path.write_text(json.dumps(meta))
        meta_path = self.dirs["all"] / "scoring.json"
        meta = json.loads(meta_path.read_text())
        meta["reference_checks"]["train/rnd"]["ok"] = False
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "did not reproduce"):
            self.run_unseen("no_ref.json")
        with self.assertRaisesRegex(ValueError, "need fold directories"):
            self.run_unseen("missing.json", dirs=[self.dirs[F] for F in ("0", "1", "2", "3")])


    def build_safe(self):
        args = write_safe_folds(self.root, self.plan, self.safe_raw, self.spec, self.root / "folds.json",
                                self.root / "inputs", self.root / "safe")
        out = {}
        for (head, F), a in args.items():
            status, text = assemble(a)
            self.assertEqual(status, 0, text)
            self.assertIn("ASSEMBLY OK", text)
            out[head, F] = self.root / "safe_folds" / head / f"fold_{F}"
        with contextlib.redirect_stdout(io.StringIO()):
            self.ps.main(["--safe-dir", f"lstm={self.root / 'safe'}", f"indep={self.root / 'safe'}",
                          "--inputs-dir", str(self.root / "inputs"), "--scores-dir", str(self.root / "score_cohorts"),
                          "--output", str(self.root / "per_seed_safe.json")])
        return args, out

    def test_safe_family_reproduces_published_rows_and_scores_unseen_tasks(self):
        _, dirs = self.build_safe()
        meta = json.loads((dirs["lstm", "all"] / "scoring.json").read_text())
        self.assertEqual(sorted(meta["reference_checks"]), sorted(
            [f"{c}/safe_lstm" for c in ("calibration", "test", "train")] + ["normalization/safe_lstm"]))
        self.assertTrue(all(v["ok"] and v["max_abs_diff"] == 0.0 for v in meta["reference_checks"].values()))
        held0 = set(self.spec["folds"][0]["held_out"])
        train0 = [json.loads(l) for l in (dirs["indep", "0"] / "train_predictions.jsonl").read_text().splitlines()]
        self.assertTrue(train0 and not any(r["task_name"] in held0 for r in train0))
        test0 = [json.loads(l) for l in (dirs["indep", "0"] / "test_predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(test0), sum(p["split"] == "evaluation" for p in self.plan))
        self.assertTrue(all(r["held_out"] == (r["task_name"] in held0) for r in test0))
        families = [[dirs[h, F] for F in ("0", "1", "2", "3", "all")] for h in HEADS]
        status, text, out = self.run_unseen("safe.json", reference=self.root / "per_seed_safe.json", families=families)
        self.assertEqual(status, 0, text)
        self.assertEqual(out["checks"]["R2_seen_plus_P"]["max_abs_diff"], 0.0)
        self.assertEqual(sorted(out["checks"]["R1"]["families"]), sorted(["safe_lstm", "safe_indep", "+".join(self.ru.DETECTORS)]))
        ref = json.loads((self.root / "per_seed_safe.json").read_text())
        for head in HEADS:
            d = f"safe_{head}"
            for s in SEEDS:
                self.assertEqual(out["table2"][d]["seen"][f"seed_{s}"]["mean_lead"], ref["results"][d]["per_seed"][s]["lead"])
            u = out["table2"][d]["unseen"]["seed_average"]
            self.assertEqual((u["n_success"], u["n_failure"]), (40, 40))
        # The cross-task rows do not change when SAFE families are added.
        _, _, base = self.run_unseen("base.json")
        for d in self.ru.DETECTORS:
            self.assertEqual(out["table2"][d]["unseen"], base["table2"][d]["unseen"])

    def test_assemble_export_refuses_mismatches(self):
        args = write_safe_folds(self.root, self.plan, self.safe_raw, self.spec, self.root / "folds.json",
                                self.root / "inputs", self.root / "safe")
        # Fold all: one frozen raw trajectory differs.
        path = self.root / "safe" / "lstm_seed1" / "normalized_scores.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        rows[5]["full_raw_scores"][0] += 1e-6
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        status, text = assemble(args["lstm", "all"])
        self.assertEqual(status, 1)
        self.assertIn("REFERENCE MISMATCH", text)
        # A wrong group label is refused before anything is written.
        a = args["indep", "1"]
        prov = Path(a[a.index("--scores") + 2]) / "provenance.json"
        p = json.loads(prov.read_text())
        prov.write_text(json.dumps(dict(p, group="confirmatory_test")))
        with self.assertRaisesRegex(ValueError, "not all"):
            assemble(a)
        prov.write_text(json.dumps(p))
        # Fold 0: the external score of a training rollout differs from the training run's by 1e-6.
        a = args["indep", "0"]
        sc = Path(a[a.index("--scores") + 2]) / "scores.jsonl"
        rows = [json.loads(l) for l in sc.read_text().splitlines()]
        held0 = set(self.spec["folds"][0]["held_out"])
        i = next(k for k, r in enumerate(rows) if "-train-" in r["rollout_id"] and r["task_name"] not in held0)
        rows[i]["scores"][0] += 1e-6
        sc.write_text("".join(json.dumps(r) + "\n" for r in rows))
        status, text = assemble(a)
        self.assertEqual(status, 1)
        self.assertIn("MISMATCH", text)
        # The same fold passes only when a larger tolerance is asked for, and records it.
        tol = list(a)
        tol[tol.index("--output-dir") + 1] += "_tol"
        status, text = assemble(tol + ["--train-check-tolerance", "1e-5"])
        self.assertEqual(status, 0, text)
        meta = json.loads((Path(tol[tol.index("--output-dir") + 1]) / "scoring.json").read_text())
        self.assertEqual(meta["checks"]["train_scores_vs_training_run"]["tolerance"], 1e-5)
        # rldx_unseen refuses a fold directory whose own recorded check failed, and accepts the rerun.
        for F in ("1", "2", "3", "all"):
            status, text = assemble(args["indep", F])
            self.assertEqual(status, 0, text)
        base = self.root / "safe_folds" / "indep"
        with contextlib.redirect_stdout(io.StringIO()):
            self.ps.main(["--safe-dir", f"lstm={self.root / 'safe'}", f"indep={self.root / 'safe'}",
                          "--inputs-dir", str(self.root / "inputs"), "--scores-dir", str(self.root / "score_cohorts"),
                          "--output", str(self.root / "per_seed_safe.json")])
        failed = [base / "fold_0"] + [base / f"fold_{F}" for F in ("1", "2", "3", "all")]
        with self.assertRaisesRegex(ValueError, "recorded failed checks"):
            self.run_unseen("failed.json", reference=self.root / "per_seed_safe.json", families=[failed])
        rerun = [base / "fold_0_tol"] + failed[1:]
        status, text, out = self.run_unseen("rerun.json", reference=self.root / "per_seed_safe.json", families=[rerun])
        self.assertEqual(status, 0, text)
        self.assertIn("safe_indep", out["table2"])

if __name__ == "__main__":
    unittest.main()
