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


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDoseResponse(unittest.TestCase):
    def test_auc_matches_rank_definition_with_ties(self):
        module = load("dose_response")
        rng = np.random.default_rng(0)
        for _ in range(20):
            labels = rng.integers(0, 2, 30).astype(bool)
            scores = rng.integers(0, 6, 30).astype(float)
            pos, neg = scores[labels], scores[~labels]
            if not len(pos) or not len(neg):
                continue
            expected = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
            self.assertAlmostEqual(module.auc(labels, scores), expected)
        self.assertIsNone(module.auc([True, True], [0.1, 0.2]))

    def test_run_end_to_end_on_synthetic_cohorts(self):
        module = load("dose_response")
        rng = np.random.default_rng(2)
        tasks, horizon = ["TaskA", "TaskB", "TaskC", "TaskD"], 20
        norms = {s: {t: {"location": 0.0, "scale": 1.0} for t in tasks} for s in ("0", "1", "2")}
        active = [30] + [30 - min(10, max(0, i - 8)) for i in range(1, horizon)]
        bundle = {
            "task_horizons": {t: horizon for t in tasks},
            "task_normalizations": norms,
            "time_risk_curves": {t: {"active_counts": active} for t in tasks},
        }

        def cohort(label, base):
            rows = []
            for ti, task in enumerate(tasks):
                for i in range(24):
                    failed = i % 2 == 1
                    length = horizon if failed else int(rng.integers(6, horizon - 1))
                    shift = 0.8 if failed and ti % 2 else 0.0
                    safe = np.maximum.accumulate(rng.standard_normal(length) + shift)
                    rows.append({
                        "rollout_id": f"{label}-{task}-{i}", "task_name": task,
                        "task_type": "atomic" if ti % 2 else "composite", "failed": failed, "horizon": horizon,
                        "environment_seed": base + 100 * ti + i, "environment_reset_index": 0,
                        "safe_scores": safe.tolist(), "time_scores": (np.arange(1, length + 1) / horizon).tolist(),
                        "baseline_scores": {
                            "knn_cosine": np.cumsum(np.abs(rng.standard_normal(length)) + shift).tolist(),
                            "mahalanobis": rng.standard_normal(length).tolist(),
                        },
                    })
            return rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, base in (("cal", 1000), ("test", 5000)):
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cohort(name, base)))
            (root / "bundle.json").write_text(json.dumps(bundle))
            argv = sys.argv
            sys.argv = ["dose_response.py", "--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
                        "--runtime-bundle", str(root / "bundle.json"), "--output", str(root / "out.json")]
            module.BOOT = 20; module.PERMUTATIONS = 50
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            finally:
                sys.argv = argv
            out = json.loads((root / "out.json").read_text())
            self.assertEqual(set(out["detectors"]), {"safe", "knn_cosine", "mahalanobis"})
            self.assertEqual(len(out["D3"]), 6)
            for key in ("safe", "knn_cosine"):
                self.assertIn("gain_decision_bootstrap", out["detectors"][key]["by_type"]["atomic"]["stratum_calibration"])
            self.assertIn("proceed_to_amendment_2", out["D4"]["safe"])
            self.assertEqual(set(out["D4"]["safe"]["power"]), {"10", "20", "30", "40"})
            safe = out["detectors"]["safe"]
            self.assertGreater(safe["by_type"]["atomic"]["lm_auc_calibration"], safe["by_type"]["composite"]["lm_auc_calibration"])
            self.assertEqual(len(safe["per_task"]), 4)


if __name__ == "__main__":
    unittest.main()
