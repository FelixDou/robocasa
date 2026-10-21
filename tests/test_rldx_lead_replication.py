import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time" / "rldx_replication.py"


def load():
    spec = importlib.util.spec_from_file_location("rldx_replication", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(root, early_failures=False, seed=0):
    rng = np.random.default_rng(seed)
    tasks, horizon = ["TaskA", "TaskB", "TaskC", "TaskD"], 30
    plan = []
    for ti, task in enumerate(tasks):
        for split, n_s, n_f in (("train", 20, 18), ("calibration", 3, 0), ("evaluation", 5, 5)):
            for i in range(n_s + n_f):
                failed = i >= n_s
                if failed:
                    length = horizon if not (early_failures and split == "evaluation" and i % 2) else 12
                else:
                    length = int(rng.integers(8, horizon - 3))
                plan.append(dict(rollout_id=f"{task}-{split}-{i}", task_name=task,
                                 task_type="atomic" if ti % 2 else "composite", failed=failed, split=split,
                                 length=length, environment_seed=1000 * ti + i))
    for s in (0, 1, 2):
        path = root / f"lstm_seed{s}" / "normalized_scores.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w") as stream:
            for p in plan:
                shift = 0.7 if p["failed"] and p["task_type"] == "atomic" else 0.0
                scores = (rng.standard_normal(p["length"]) + shift).tolist()
                stream.write(json.dumps({k: v for k, v in p.items() if k != "length"} | {"full_scores": scores}) + "\n")
    return plan


class TestRldxReplication(unittest.TestCase):
    def run_script(self, root, out):
        module = load()
        argv = sys.argv
        sys.argv = ["rldx_replication.py", "--calibration-dir", str(root), "--model", "lstm", "--output", str(out)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
        finally:
            sys.argv = argv
        return json.loads(out.read_text())

    def test_end_to_end_and_failure_limit_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "cal")
            out = self.run_script(root / "cal", root / "out.json")
            self.assertEqual(out["counts"], dict(train=152, calibration_successes=12, test=40, test_failures=20,
                                                 path_choice_failures=72))
            c1 = {c["split"]: c for c in out["checks"]["C1"]}
            self.assertEqual(c1["test"]["at_limit"], 1.0)
            self.assertFalse(out["checks"]["early_ending_flag"])
            self.assertEqual(set(out["primary"]["by_type"]), {"atomic", "composite"})
            self.assertEqual(out["null_noise"]["draws"], 20)
            self.assertIn("global_timer", out["secondary"])
            with self.assertRaises(SystemExit):
                self.run_script(root / "cal", root / "out.json")

            write_fixture(root / "early", early_failures=True, seed=1)
            out = self.run_script(root / "early", root / "early.json")
            c1 = {c["split"]: c for c in out["checks"]["C1"]}
            self.assertTrue(out["checks"]["early_ending_flag"])
            self.assertEqual(len(c1["test"]["early_ending"]), 12)

    def test_seed_disagreement_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            path = root / "lstm_seed2" / "normalized_scores.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["failed"] = not rows[0]["failed"]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                self.run_script(root, root / "out.json")


if __name__ == "__main__":
    unittest.main()
