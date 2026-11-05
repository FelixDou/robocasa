import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.analyze_time_safe_hybrid import fit_safe_normalization
from robocasa.recovery.safe.baselines.score_cohorts import main as score_cohorts
from robocasa.recovery.safe.baselines.score_cross_task import main as score_cross_task, pooled_normalization
from tests.test_safe_baselines import make_dataset

TASKS = ("TaskA", "TaskB", "TaskC", "TaskD")
HAS_TORCH = importlib.util.find_spec("torch") is not None


def fixture(root):
    ids = make_dataset(root / "data", per_task=8, tasks=TASKS)
    split = {"train": [], "calibration": [], "evaluation": []}
    for rollout_id, (task, failed, index) in sorted(ids.items()):
        split[("train", "train", "calibration", "evaluation")[(index // 2) % 4]].append(rollout_id)
    bundle = {"task_horizons": {t: 10 for t in TASKS},
              "time_risk_curves": {t: {"risk": np.linspace(0.1, 0.9, 10).tolist()} for t in TASKS},
              "development_rollout_ids": split["train"]}
    (root / "bundle.json").write_text(json.dumps(bundle))
    for name, rows in split.items():
        (root / f"{name}.json").write_text(json.dumps(rows))
    folds = {"seed": 0, "k": 2, "tasks": {t: "atomic" for t in TASKS},
             "folds": [{"fold": 0, "held_out": ["TaskA", "TaskC"]}, {"fold": 1, "held_out": ["TaskB", "TaskD"]}]}
    (root / "folds.json").write_text(json.dumps(folds))
    return split, ids


def common(root):
    return ["--dataset-dir", str(root / "data"), "--runtime-bundle", str(root / "bundle.json"),
            "--train-ids", str(root / "train.json"),
            "--cohort", "calib", str(root / "calibration.json"), "--cohort", "evalc", str(root / "evaluation.json"),
            "--seeds", "0", "1", "--epochs", "3"]


def cross(root, fold, out, detectors, *extra):
    return [*common(root), "--folds", str(root / "folds.json"), "--fold", fold, "--detectors", *detectors,
            "--output-dir", str(root / out), *extra]


def rows_of(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


class ScoreCrossTaskTests(unittest.TestCase):
    def check_fold_all_reproduces_score_cohorts(self, detectors):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(score_cohorts([*common(root), "--detectors", *detectors,
                                                "--output-dir", str(root / "sc")]), 0)
                status = score_cross_task(cross(root, "all", "xt", detectors,
                                                "--reference-raw", "calib", str(root / "sc" / "calib_predictions.jsonl"),
                                                "--reference-raw", "evalc", str(root / "sc" / "evalc_predictions.jsonl")))
            self.assertEqual(status, 0)
            meta = json.loads((root / "xt" / "scoring.json").read_text())
            self.assertEqual(set(meta["reference_checks"]), {f"{c}/{d}" for c in ("calib", "evalc") for d in detectors})
            self.assertTrue(all(v["ok"] and v["max_abs_diff"] == 0.0 for v in meta["reference_checks"].values()))
            ref = json.loads((root / "sc" / "normalizations.json").read_text())
            mine = json.loads((root / "xt" / "normalizations.json").read_text())
            for d in detectors:
                for s in ("0", "1"):
                    self.assertEqual(mine[d][s]["per_task"], ref[d][s])

    def test_fold_all_reproduces_score_cohorts_knn(self):
        self.check_fold_all_reproduces_score_cohorts(["cosine_knn"])

    @unittest.skipUnless(HAS_TORCH, "Hide-and-Seek needs PyTorch")
    def test_fold_all_reproduces_score_cohorts_hide_and_seek(self):
        self.check_fold_all_reproduces_score_cohorts(["hide_and_seek", "hide_and_seek_zero", "hide_and_seek_first"])

    def test_fold_trains_without_held_out_tasks_and_scores_all_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, ids = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(score_cross_task(cross(root, "0", "f0", ["cosine_knn"])), 0)
                self.assertEqual(score_cross_task(cross(root, "all", "fa", ["cosine_knn"])), 0)
            meta = json.loads((root / "f0" / "scoring.json").read_text())
            self.assertEqual(meta["held_out_tasks"], ["TaskA", "TaskC"])
            self.assertEqual(set(meta["training_rollouts_by_task"]), {"TaskB", "TaskD"})
            self.assertEqual(set(meta["dropped_training_rollouts_by_task"]), {"TaskA", "TaskC"})
            self.assertEqual(meta["counts"]["train"], sum(ids[r][0] in ("TaskB", "TaskD") for r in split["train"]))
            self.assertNotIn("reference_checks", {k for k, v in meta.items() if v})
            rows = rows_of(root / "f0" / "evalc_predictions.jsonl")
            self.assertEqual({r["rollout_id"] for r in rows}, set(split["evaluation"]))
            for r in rows:
                self.assertEqual(r["held_out"], r["task_name"] in ("TaskA", "TaskC"))
                self.assertEqual(r["fold"], "0")
                self.assertEqual(set(r["per_seed_raw"]), {"0", "1"})
                self.assertNotIn("safe_scores", r)
            norms = json.loads((root / "f0" / "normalizations.json").read_text())["cosine_knn"]
            for s in ("0", "1"):
                self.assertEqual(set(norms[s]["per_task"]), {"TaskB", "TaskD"})
                self.assertEqual(norms[s]["pooled"]["fit_rollouts"], meta["counts"]["train"])
            # Held-out training rollouts are not in the bank: a fold-0 score differs from the all-task score.
            all_rows = {r["rollout_id"]: r for r in rows_of(root / "fa" / "evalc_predictions.jsonl")}
            differs = [not np.allclose(r["per_seed_raw"]["0"], all_rows[r["rollout_id"]]["per_seed_raw"]["0"])
                       for r in rows if r["held_out"]]
            self.assertTrue(any(differs))

    def test_score_train_writes_in_sample_training_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, ids = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(score_cohorts([*common(root), "--detectors", "cosine_knn", "--score-train",
                                                "--output-dir", str(root / "sc")]), 0)
                status = score_cross_task(cross(root, "all", "fa", ["cosine_knn"], "--score-train",
                                                "--reference-raw", "train", str(root / "sc" / "train_predictions.jsonl"),
                                                "--reference-raw", "calib", str(root / "sc" / "calib_predictions.jsonl")))
                self.assertEqual(score_cross_task(cross(root, "0", "f0", ["cosine_knn"], "--score-train")), 0)
                self.assertEqual(score_cross_task(cross(root, "1", "f1", ["cosine_knn"])), 0)
            self.assertEqual(status, 0)
            meta = json.loads((root / "fa" / "scoring.json").read_text())
            self.assertTrue(meta["score_train"])
            check = meta["reference_checks"]["train/cosine_knn"]
            self.assertEqual((check["ok"], check["max_abs_diff"], check["rollouts_scored"]),
                             (True, 0.0, len(split["train"])))
            rows = rows_of(root / "f0" / "train_predictions.jsonl")
            seen = [r for r in split["train"] if ids[r][0] in ("TaskB", "TaskD")]
            self.assertEqual(sorted(r["rollout_id"] for r in rows), sorted(seen))
            self.assertFalse(any(r["held_out"] for r in rows))
            self.assertEqual(sorted(r["task_name"] for r in rows), sorted(ids[r][0] for r in seen))
            self.assertFalse((root / "f1" / "train_predictions.jsonl").exists())
            self.assertFalse(json.loads((root / "f1" / "scoring.json").read_text())["score_train"])
            with self.assertRaises(ValueError):
                score_cross_task([*common(root), "--cohort", "train", str(root / "evaluation.json"),
                                  "--folds", str(root / "folds.json"), "--fold", "0", "--detectors", "cosine_knn",
                                  "--score-train", "--output-dir", str(root / "bad")])

    def test_pooled_normalization_is_the_robust_formula_over_all_tasks(self):
        rng = np.random.default_rng(0)
        records = [{"task_name": t, "scores": rng.normal(size=12)} for t in ("a", "b") for _ in range(6)]
        horizons = {"a": 10, "b": 12}
        pooled = pooled_normalization(records, horizons)
        maxima = np.array([np.max(r["scores"][: horizons[r["task_name"]]]) for r in records])
        loc = np.median(maxima)
        self.assertAlmostEqual(pooled["location"], loc)
        self.assertAlmostEqual(pooled["scale"], max(1.4826 * np.median(np.abs(maxima - loc)), np.std(maxima), 1e-6))
        # One task alone gives fit_safe_normalization's numbers.
        only_a = [r for r in records if r["task_name"] == "a"]
        self.assertEqual({k: pooled_normalization(only_a, horizons)[k] for k in ("location", "scale")},
                         {k: fit_safe_normalization(only_a, {"a": 10})["a"][k] for k in ("location", "scale")})

    def test_invalid_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            with self.assertRaises(ValueError):
                score_cross_task(cross(root, "0", "a", ["cosine_knn"], "--reference-raw", "calib", str(root / "x.jsonl")))
            with self.assertRaises(ValueError):
                score_cross_task(cross(root, "1", "b", ["hide_and_seek_task_shuffle"]))
            with self.assertRaises(ValueError):
                score_cross_task(cross(root, "7", "c", ["cosine_knn"]))
            folds = json.loads((root / "folds.json").read_text())
            folds["folds"][1]["held_out"] = ["TaskB"]
            (root / "folds.json").write_text(json.dumps(folds))
            with self.assertRaises(ValueError):
                score_cross_task(cross(root, "0", "d", ["cosine_knn"]))


if __name__ == "__main__":
    unittest.main()
