import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.baselines.run_baselines import main as run_baselines
from robocasa.recovery.safe.baselines.score_cohorts import main as score_cohorts
from tests.test_safe_baselines import make_dataset


def fixture(root):
    ids = make_dataset(root / "data", per_task=12)
    split = {"train": [], "calibration": [], "evaluation": []}
    for rollout_id, (task, failed, index) in sorted(ids.items()):
        split[("train", "train", "calibration", "evaluation")[(index // 2) % 4]].append(rollout_id)
    bundle = {
        "task_horizons": {"TaskA": 10, "TaskB": 10},
        "time_risk_curves": {t: {"risk": np.linspace(0.1, 0.9, 10).tolist()} for t in ("TaskA", "TaskB")},
        "target_fpr": 0.2, "safe_eligibility_end_fraction": 0.5, "time_eligibility_start_fraction": 0.6,
        "development_rollout_ids": split["train"], "primary_calibration_rollout_ids": split["calibration"],
    }
    (root / "bundle.json").write_text(json.dumps(bundle))
    for name, rows in split.items():
        (root / f"{name}.json").write_text(json.dumps(rows))
    return split


def score_args(root, out, *extra):
    return ["--dataset-dir", str(root / "data"), "--runtime-bundle", str(root / "bundle.json"),
            "--train-ids", str(root / "train.json"),
            "--cohort", "calib", str(root / "calibration.json"), "--cohort", "evalc", str(root / "evaluation.json"),
            "--detectors", "hide_and_seek", "hide_and_seek_first", "--seeds", "0", "1", "--epochs", "4",
            "--output-dir", str(root / out), *extra]


class ScoreCohortsTests(unittest.TestCase):
    def test_reproduces_run_baselines_and_writes_no_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run_baselines(["--dataset-dir", str(root / "data"), "--runtime-bundle", str(root / "bundle.json"),
                               "--train-ids", str(root / "train.json"),
                               "--evaluation-ids", str(root / "evaluation.json"),
                               "--output-dir", str(root / "rb"), "--detectors", "hide_and_seek",
                               "--seeds", "0", "1", "--epochs", "4", "--test-per-class", "3"])
                status = score_cohorts(score_args(
                    root, "sc", "--reference", "calib", str(root / "rb" / "calibration_predictions.jsonl"),
                    "--reference", "evalc", str(root / "rb" / "evaluation_predictions.jsonl")))
            self.assertEqual(status, 0)
            meta = json.loads((root / "sc" / "scoring.json").read_text())
            self.assertEqual(meta["protocol"], "score_only_no_metrics")
            self.assertTrue(all(c["ok"] and c["max_abs_diff"] == 0.0 for c in meta["reference_checks"].values()))
            self.assertEqual(set(meta["reference_checks"]), {"calib/hide_and_seek", "evalc/hide_and_seek"})
            norms = json.loads((root / "sc" / "normalizations.json").read_text())
            rows = [json.loads(l) for l in (root / "sc" / "evalc_predictions.jsonl").read_text().splitlines()]
            self.assertEqual({(r["detector"], r["rollout_id"]) for r in rows},
                             {(d, rid) for d in ("hide_and_seek", "hide_and_seek_first") for rid in split["evaluation"]})
            for r in rows:
                self.assertEqual(set(r["per_seed_raw"]), {"0", "1"})
                self.assertNotIn("true_positive_rate", json.dumps(r))
                vals = [(np.maximum.accumulate(r["per_seed_raw"][s]) - norms[r["detector"]][s][r["task_name"]]["location"])
                        / norms[r["detector"]][s][r["task_name"]]["scale"] for s in ("0", "1")]
                np.testing.assert_allclose(np.mean(vals, axis=0), r["safe_scores"], atol=1e-12)
            self.assertNotIn("evaluation", meta)

    def test_reference_mismatch_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = fixture(root)
            rid = split["evaluation"][0]
            fake = {"detector": "hide_and_seek", "rollout_id": rid, "safe_scores": [0.0] * 3}
            (root / "fake.jsonl").write_text(json.dumps(fake) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                status = score_cohorts(score_args(root, "sc", "--reference", "evalc", str(root / "fake.jsonl")))
            self.assertEqual(status, 1)

    def test_overlapping_cohorts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            args = score_args(root, "sc")
            args[args.index(str(root / "evaluation.json"))] = str(root / "calibration.json")
            with self.assertRaises(ValueError):
                score_cohorts(args)
            args = score_args(root, "sc2")
            args[args.index(str(root / "calibration.json"))] = str(root / "train.json")
            with self.assertRaises(ValueError):
                score_cohorts(args)


if __name__ == "__main__":
    unittest.main()
