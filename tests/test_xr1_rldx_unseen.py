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
from tests.test_rldx_hide_and_seek import load

TASKS = tuple(f"Task{c}" for c in "ABCDEFGH")
SEEDS = ("0", "1", "2")
COHORT_OF = {"train": "train", "calibration": "calibration", "evaluation": "test"}


def write_safe(root, seed=0):
    """SAFE-LSTM normalized scores defining the 15.15 rows (rldx_replication.assemble)."""
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
    for s in SEEDS:
        path = root / f"lstm_seed{s}" / "normalized_scores.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w") as stream:
            for p in plan:
                scores = (rng.standard_normal(p["length"]) + (0.5 if p["failed"] else 0.0)).tolist()
                stream.write(json.dumps({k: v for k, v in p.items() if k != "length"} | {"full_scores": scores}) + "\n")
    return plan


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


class RldxUnseenTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        self.ru, self.ps, self.hs = load("rldx_unseen"), load("rldx_per_seed"), load("rldx_hide_and_seek")
        self.plan = write_safe(root / "safe")
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

    def run_unseen(self, out, reference=None, dirs=None):
        dirs = dirs or [self.dirs[F] for F in ("0", "1", "2", "3", "all")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            status = self.ru.main(["run", "--safe-dir", str(self.root / "safe"), "--inputs-dir", str(self.root / "inputs"),
                                   "--folds", str(self.root / "folds.json"), "--fold-dirs", *map(str, dirs),
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
        with self.assertRaisesRegex(ValueError, "Need fold directories"):
            self.run_unseen("missing.json", dirs=[self.dirs[F] for F in ("0", "1", "2", "3")])


if __name__ == "__main__":
    unittest.main()
