import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.baselines.score_cohorts import main as score_cohorts
from tests.test_rldx_hide_and_seek import load, raw_dataset
from tests.test_rldx_lead_replication import write_fixture


def write_head(root, plan, model, seed=7):
    """Synthetic normalized scores of a second SAFE head for the same rollouts."""
    rng = np.random.default_rng(seed)
    for s in (0, 1, 2):
        path = root / f"{model}_seed{s}" / "normalized_scores.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w") as stream:
            for p in plan:
                shift = 0.4 if p["failed"] else 0.0
                scores = (rng.standard_normal(p["length"]) + shift).tolist()
                stream.write(json.dumps({k: v for k, v in p.items() if k != "length"} | {"full_scores": scores}) + "\n")


class TestRldxPerSeed(unittest.TestCase):
    def test_reproduces_stored_results_and_reports_per_seed(self):
        rr, hs, ps = load("rldx_replication"), load("rldx_hide_and_seek"), load("rldx_per_seed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = write_fixture(root / "cal")
            write_head(root / "cal", plan, "indep")
            raw_dataset(root / "raw", plan)
            argv = sys.argv
            try:
                for model in ("lstm", "indep"):
                    sys.argv = ["x", "--calibration-dir", str(root / "cal"), "--model", model,
                                "--output", str(root / f"ref_{model}.json")]
                    with contextlib.redirect_stdout(io.StringIO()):
                        rr.main()
            finally:
                sys.argv = argv
            with contextlib.redirect_stdout(io.StringIO()):
                hs.main(["prepare", "--calibration-dir", str(root / "cal"), "--output-dir", str(root / "in")])
                common = ["--dataset-dir", str(root / "raw"), "--runtime-bundle", str(root / "in" / "rldx_bundle.json"),
                          "--train-ids", str(root / "in" / "train_ids.json"),
                          "--cohort", "calibration", str(root / "in" / "calibration_ids.json"),
                          "--cohort", "test", str(root / "in" / "test_ids.json"),
                          "--feature-aggregation", "last", "--seeds", "0", "1", "2", "--epochs", "2", "--score-train"]
                self.assertEqual(score_cohorts(common + [
                    "--detectors", "hide_and_seek", "hide_and_seek_first", "hide_and_seek_zero",
                    "hide_and_seek_task_shuffle", "--output-dir", str(root / "scores_hs")]), 0)
                self.assertEqual(score_cohorts(common + [
                    "--detectors", "mahalanobis", "cosine_knn", "pca_kmeans", "rnd",
                    "--output-dir", str(root / "scores_ood")]), 0)
                self.assertEqual(hs.main(["run", "--calibration-dir", str(root / "cal"), "--inputs-dir", str(root / "in"),
                                          "--scores-dir", str(root / "scores_hs"), "--reference-result",
                                          str(root / "ref_lstm.json"), "--bootstrap", "5", "--null-draws", "1",
                                          "--output", str(root / "hs.json")]), 0)

            args = ["--safe-dir", f"lstm={root / 'cal'}", f"indep={root / 'cal'}", "--inputs-dir", str(root / "in"),
                    "--scores-dir", str(root / "scores_hs"), str(root / "scores_ood"),
                    "--reference-hs", str(root / "hs.json"),
                    "--reference-safe", f"lstm={root / 'ref_lstm.json'}", f"indep={root / 'ref_indep.json'}"]
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                self.assertEqual(ps.main(args + ["--output", str(root / "out.json")]), 0)
            text = log.getvalue()
            self.assertIn("REFERENCE OK", text)
            self.assertNotIn("MISMATCH", text)
            out = json.loads((root / "out.json").read_text())
            expected = {"safe_lstm", "safe_indep", "hide_and_seek", "hide_and_seek_first", "hide_and_seek_zero",
                        "hide_and_seek_task_shuffle", "mahalanobis", "cosine_knn", "pca_kmeans", "rnd"}
            self.assertEqual(set(out["summary"]), expected)
            hs_out = json.loads((root / "hs.json").read_text())
            self.assertEqual(out["results"]["hide_and_seek"]["seed_average"]["gain"],
                             hs_out["detectors"]["hide_and_seek"]["gain"])
            for key in expected:
                self.assertEqual(set(out["results"][key]["per_seed"]), {"0", "1", "2"})
                self.assertGreaterEqual(out["summary"][key]["lead"]["std"], 0.0)

            # A stored result that differs stops the run before any mean.
            bad = json.loads((root / "hs.json").read_text())
            bad["detectors"]["hide_and_seek"]["gain"] += 0.001
            (root / "bad.json").write_text(json.dumps(bad))
            bad_args = list(args)
            bad_args[bad_args.index(str(root / "hs.json"))] = str(root / "bad.json")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "do not reproduce"):
                    ps.main(bad_args + ["--output", str(root / "bad_out.json")])


if __name__ == "__main__":
    unittest.main()
