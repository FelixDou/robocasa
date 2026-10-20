import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time" / "confirm_lead.py"


def load_module():
    spec = importlib.util.spec_from_file_location("confirm_lead", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_cohort(rng, tasks, horizon, per_class, base_seed, norms, label):
    rows = []
    for task_index, task in enumerate(tasks):
        for index in range(2 * per_class):
            failed = index % 2 == 1
            length = horizon if failed else int(rng.integers(horizon // 3, horizon - 1))
            raw = {s: rng.standard_normal(length) + (0.6 if failed else 0.0) for s in norms}
            safe = np.mean(
                [(np.maximum.accumulate(raw[s]) - norms[s][task]["location"]) / norms[s][task]["scale"] for s in norms],
                axis=0,
            )
            rows.append(
                {
                    "rollout_id": f"{label}-{task}-{index}",
                    "task_name": task,
                    "task_type": "atomic" if task_index % 2 else "composite",
                    "failed": failed,
                    "horizon": horizon,
                    "environment_seed": base_seed + 100 * task_index + index,
                    "environment_reset_index": 0,
                    "safe_scores": safe.tolist(),
                    "time_scores": (np.arange(1, length + 1) / horizon).tolist(),
                    "per_seed_raw_safe": {s: v.tolist() for s, v in raw.items()},
                }
            )
    return rows


class TestConfirmLead(unittest.TestCase):
    def test_run_end_to_end_on_synthetic_cohorts(self):
        module = load_module()
        rng = np.random.default_rng(0)
        tasks, horizon = ["TaskA", "TaskB", "TaskC", "TaskD"], 20
        norms = {s: {t: {"location": 0.1 * i, "scale": 1.0 + 0.1 * i} for i, t in enumerate(tasks)} for s in ("0", "1", "2")}
        active = [30] + [30 - min(10, max(0, i - 8)) for i in range(1, horizon)]
        bundle = {
            "task_horizons": {t: horizon for t in tasks},
            "task_normalizations": norms,
            "time_risk_curves": {t: {"active_counts": active} for t in tasks},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, base in (("cal", 1000), ("test", 5000)):
                rows = synthetic_cohort(rng, tasks, horizon, 12, base, norms, name)
                paths[name] = root / f"{name}.jsonl"
                paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / "bundle.json").write_text(json.dumps(bundle))
            (root / "fused.json").write_text(
                json.dumps({"detectors": {"fused_safe_time": {"thresholds": {"early_safe": 1.5, "late_time": 0.9}}}})
            )
            argv = sys.argv
            sys.argv = [
                "confirm_lead.py", "run",
                "--calibration", str(paths["cal"]), "--test", str(paths["test"]),
                "--runtime-bundle", str(root / "bundle.json"), "--fused-calibration", str(root / "fused.json"),
                "--output", str(root / "out.json"), "--bootstrap", "20", "--null-draws", "2",
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            finally:
                sys.argv = argv
            out = json.loads((root / "out.json").read_text())
            self.assertEqual(out["counts"]["calibration"], 80)
            self.assertEqual(out["counts"]["test"], 80)
            self.assertLess(out["checks"]["safe_pipeline_max_abs_diff"], 1e-9)
            primary = out["by_q"]["0.95"]
            self.assertEqual(primary["gain_reselect_bootstrap"]["samples"], 20)
            self.assertIn("supported", out["H1"])
            self.assertEqual(out["H2"]["nulls"]["noise"]["draws"], 2)
            self.assertEqual(set(out["by_task_type"]), {"atomic", "composite"})
            for name in ("global_timer", "safe_plus_risk_frozen", "P_tuned_on_test"):
                self.assertIn(name, out["secondary"])
            frozen = out["secondary"]["safe_plus_risk_frozen"]
            self.assertEqual(frozen["thresholds"], {"early_safe": 1.5, "late_time": 0.9})
            test_rows = [json.loads(line) for line in paths["test"].read_text().splitlines()]
            chosen = module.lead_time.sel([dict(r, failed=bool(r["failed"])) for r in test_rows], 10)
            expected_fp = sum(
                (not r["failed"]) and (max(r["safe_scores"]) >= 1.5 or max(r["time_scores"]) >= 0.9) for r in chosen
            )
            self.assertEqual(frozen["test"]["fp"], expected_fp)
            with self.assertRaises(SystemExit):
                sys.argv = ["confirm_lead.py", "run", "--calibration", str(paths["cal"]), "--test", str(paths["test"]),
                            "--runtime-bundle", str(root / "bundle.json"), "--fused-calibration", str(root / "fused.json"),
                            "--output", str(root / "out.json")]
                try:
                    module.main()
                finally:
                    sys.argv = argv

    def test_overlapping_cohorts_are_refused(self):
        module = load_module()
        rng = np.random.default_rng(1)
        tasks, horizon = ["TaskA", "TaskB"], 12
        norms = {s: {t: {"location": 0.0, "scale": 1.0} for t in tasks} for s in ("0", "1", "2")}
        bundle = {
            "task_horizons": {t: horizon for t in tasks},
            "task_normalizations": norms,
            "time_risk_curves": {t: {"active_counts": [20] * 5 + [15] * (horizon - 5)} for t in tasks},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_cohort(rng, tasks, horizon, 10, 1000, norms, "same")
            (root / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / "bundle.json").write_text(json.dumps(bundle))
            (root / "fused.json").write_text(json.dumps({"detectors": {"fused_safe_time": {"thresholds": {"score": 1, "time": 1}}}}))
            args = type("Args", (), dict(calibration=str(root / "a.jsonl"), test=str(root / "a.jsonl"),
                                         runtime_bundle=str(root / "bundle.json"), fused_calibration=str(root / "fused.json"),
                                         output=str(root / "o.json"), confirm_predictions=None, bootstrap=5, null_draws=1))
            with self.assertRaisesRegex(ValueError, "overlap"):
                module.run(args)


if __name__ == "__main__":
    unittest.main()
