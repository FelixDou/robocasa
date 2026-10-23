import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.baselines.score_cohorts import main as score_cohorts
from robocasa.recovery.safe.dataset import save_rollout
from robocasa.recovery.safe.schema import XIAOMI_ROBOTICS_1_FEATURE_LAYER, SafeRolloutMetadata
from tests.test_rldx_lead_replication import write_fixture
from tests.test_safe_baselines import FLOW, HORIZON, REPLAN, synthetic_rollout

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_dataset(root, plan, seed=0):
    """Raw save_rollout data with the SAFE fixture's IDs, labels and lengths."""
    rng = np.random.default_rng(seed)
    for i, p in enumerate(plan):
        features, chunks = synthetic_rollout(rng, p["failed"], p["length"])
        metadata = SafeRolloutMetadata(
            rollout_id=p["rollout_id"], task_name=p["task_name"], task_instruction="do it",
            environment_seed=p["environment_seed"], policy_id="rldx1", checkpoint="ckpt", failed=p["failed"],
            num_env_steps=p["length"] * REPLAN, inference_env_steps=[t * REPLAN for t in range(p["length"])],
            valid_sequence_length=p["length"], action_horizon=HORIZON, replan_steps=REPLAN,
            feature_layer=XIAOMI_ROBOTICS_1_FEATURE_LAYER, model_family="xiaomi_robotics_1", flow_steps=FLOW,
            timeout_horizon=30 * REPLAN,
        )
        save_rollout(root, metadata, features, policy_action_chunks=chunks)


class TestRldxHideAndSeek(unittest.TestCase):
    def test_end_to_end_reproduces_15_15_and_reads(self):
        rr, hs = load("rldx_replication"), load("rldx_hide_and_seek")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = write_fixture(root / "cal")
            raw_dataset(root / "raw", plan)
            argv = sys.argv
            sys.argv = ["x", "--calibration-dir", str(root / "cal"), "--model", "lstm", "--output", str(root / "ref.json")]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rr.main()
            finally:
                sys.argv = argv
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(hs.main(["prepare", "--calibration-dir", str(root / "cal"), "--output-dir", str(root / "in")]), 0)
                bundle = json.loads((root / "in" / "rldx_bundle.json").read_text())
                self.assertEqual(bundle["task_horizons"], {t: 30 for t in ("TaskA", "TaskB", "TaskC", "TaskD")})
                self.assertEqual(len(json.loads((root / "in" / "train_ids.json").read_text())), 152)
                status = score_cohorts([
                    "--dataset-dir", str(root / "raw"), "--runtime-bundle", str(root / "in" / "rldx_bundle.json"),
                    "--train-ids", str(root / "in" / "train_ids.json"),
                    "--cohort", "calibration", str(root / "in" / "calibration_ids.json"),
                    "--cohort", "test", str(root / "in" / "test_ids.json"),
                    "--detectors", "hide_and_seek", "hide_and_seek_first", "hide_and_seek_zero",
                    "hide_and_seek_task_shuffle", "--feature-aggregation", "last", "--seeds", "0", "1", "2",
                    "--epochs", "2", "--score-train", "--output-dir", str(root / "scores")])
                self.assertEqual(status, 0)
                train_rows = (root / "scores" / "train_predictions.jsonl").read_text().splitlines()
                self.assertEqual(len(train_rows), 4 * 152)
                run = ["run", "--calibration-dir", str(root / "cal"), "--inputs-dir", str(root / "in"),
                       "--scores-dir", str(root / "scores"), "--reference-result", str(root / "ref.json"),
                       "--bootstrap", "20", "--null-draws", "2"]
                self.assertEqual(hs.main(run + ["--output", str(root / "out.json")]), 0)
            out = json.loads((root / "out.json").read_text())
            ref = json.loads((root / "ref.json").read_text())["primary"]
            self.assertTrue(out["checks"]["safe_reference_15_15"]["ok"])
            self.assertEqual(out["detectors"]["safe"]["gain"], ref["gain"])
            self.assertEqual(out["counts"], dict(train=152, calibration_successes=12, path_choice_failures=72,
                                                 test=40, test_failures=20, tasks=4))
            self.assertEqual(set(out["detectors"]), {"safe", "hide_and_seek", "hide_and_seek_first",
                                                     "hide_and_seek_zero", "hide_and_seek_task_shuffle"})
            self.assertEqual(set(out["by_task_type"]), {"atomic", "composite"})
            self.assertIn("R1_replicates_at_margin", out["reading"])
            self.assertEqual(out["nulls"]["noise"]["draws"], 2)
            self.assertLess(out["checks"]["normalization_max_abs_diff"], 1e-9)

            # A different stored 15.15 result blocks the reading.
            bad = json.loads((root / "ref.json").read_text())
            bad["primary"]["gain"] += 0.001
            (root / "bad.json").write_text(json.dumps(bad))
            run[run.index(str(root / "ref.json"))] = str(root / "bad.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(hs.main(run + ["--output", str(root / "bad_out.json")]), 1)

    def test_score_train_refuses_a_cohort_named_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.json").write_text(json.dumps({"task_horizons": {"TaskA": 5}, "time_risk_curves": {}}))
            (root / "ids.json").write_text(json.dumps(["x"]))
            (root / "c.json").write_text(json.dumps(["y"]))
            with self.assertRaisesRegex(ValueError, "named 'train'"):
                score_cohorts(["--dataset-dir", str(root), "--runtime-bundle", str(root / "b.json"),
                               "--train-ids", str(root / "ids.json"), "--cohort", "train", str(root / "c.json"),
                               "--detectors", "hide_and_seek", "--score-train", "--output-dir", str(root / "o")])


if __name__ == "__main__":
    unittest.main()
