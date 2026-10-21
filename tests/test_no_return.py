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


def call(module, argv):
    old = sys.argv
    sys.argv = [module.__name__ + ".py"] + argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()
    finally:
        sys.argv = old


class TestNoReturnSample(unittest.TestCase):
    def test_bundle_is_blind_stratified_and_refuses_overwrite(self):
        module = load("no_return_sample")
        rng = np.random.default_rng(0)
        tasks = {"TaskA": "atomic", "TaskB": "composite", "TaskC": "atomic", "TaskD": "composite"}
        horizon = 20
        bundle = {
            "task_horizons": {t: horizon for t in tasks},
            "task_normalizations": {s: {t: {"location": 0.0, "scale": 1.0} for t in tasks} for s in ("0", "1", "2")},
            "time_risk_curves": {t: {"active_counts": [30] + [30 - min(10, max(0, i - 8)) for i in range(1, horizon)]} for t in tasks},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = []
            (root / "coll" / "shard0" / "videos").mkdir(parents=True)
            for name, base in (("cal", 1000), ("test", 5000)):
                rows = []
                for ti, (task, kind) in enumerate(tasks.items()):
                    for i in range(20):
                        failed = i % 2 == 1
                        length = horizon if failed else int(rng.integers(6, horizon - 1))
                        rid = f"{name}-{task}-{i}"
                        rows.append({"rollout_id": rid, "task_name": task, "task_type": kind, "failed": failed,
                                     "horizon": horizon, "environment_seed": base + 100 * ti + i, "environment_reset_index": 0,
                                     "safe_scores": np.maximum.accumulate(rng.standard_normal(length)).tolist(),
                                     "time_scores": (np.arange(1, length + 1) / horizon).tolist()})
                        if name == "test":
                            (root / "coll" / "shard0" / "videos" / f"{rid}.mp4").write_bytes(b"fake")
                            manifest.append({"rollout_id": rid, "video_path": f"videos/{rid}.mp4", "task_instruction": f"do {task}",
                                             "video_frame_stride": 2, "replan_steps": 16})
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / "coll" / "shard0" / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
            (root / "bundle.json").write_text(json.dumps(bundle))
            argv = ["--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
                    "--runtime-bundle", str(root / "bundle.json"), "--collection-root", str(root / "coll"),
                    "--output", str(root / "out")]
            call(module, argv)
            shown = json.loads((root / "out" / "bundle" / "manifest.json").read_text())
            key = json.loads((root / "out" / "key" / "key.json").read_text())
            self.assertEqual(len(shown["videos"]), 6)
            kinds = [key["videos"][v["id"]]["task_type"] for v in shown["videos"]]
            self.assertEqual(kinds.count("atomic"), 4)
            self.assertFalse(any("alarm" in k or "rollout_id" in k for v in shown["videos"] for k in v))
            self.assertTrue(all((root / "out" / "bundle" / v["file"]).is_file() for v in shown["videos"]))
            self.assertTrue(all("alarm_P_inference" in v for v in key["videos"].values()))
            with self.assertRaises(SystemExit):
                call(module, argv)


class TestNoReturnAnalysis(unittest.TestCase):
    def test_timing_arithmetic_and_completeness(self):
        module = load("no_return_analysis")
        key = {"videos": {
            "v01": {"task_type": "atomic", "steps_per_inference": 16, "horizon_inferences": 10, "length_inferences": 10,
                    "alarm_P_inference": 9, "alarm_safe_plus_P_inference": 3},
            "v02": {"task_type": "composite", "steps_per_inference": 16, "horizon_inferences": 10, "length_inferences": 10,
                    "alarm_P_inference": 9, "alarm_safe_plus_P_inference": 9},
        }}
        labels = {"labels": {
            "v01": {"t_e_env_step": 20, "t_u_env_step": 40, "category": "dropped_or_lost_object", "confidence": "high"},
            "v02": {"recoverable_until_timeout": True, "category": "slow_or_nearly_succeeded", "confidence": "medium"},
        }}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "key.json").write_text(json.dumps(key)); (root / "labels.json").write_text(json.dumps(labels))
            call(module, ["--labels", str(root / "labels.json"), "--key", str(root / "key.json"), "--output", str(root / "out.json")])
            out = json.loads((root / "out.json").read_text())
            atomic = out["by_type"]["atomic"]
            self.assertAlmostEqual(atomic["t_u_over_H"]["median"], 0.25)
            self.assertAlmostEqual(atomic["detectors"]["P"]["mean_wasted_after_t_u_over_H"], (128 - 40) / 160)
            self.assertEqual(atomic["detectors"]["P"]["preemptive_fraction"], 0.0)
            self.assertEqual(atomic["detectors"]["safe_plus_P"]["preemptive_fraction"], 1.0)
            self.assertAlmostEqual(atomic["detectors"]["P"]["mean_usable_lead_over_H"], (160 - 128) / 160)
            self.assertAlmostEqual(atomic["detectors"]["safe_plus_P"]["mean_usable_lead_over_H"], (160 - 40) / 160)
            comp = out["by_type"]["composite"]
            self.assertEqual(comp["recoverable_until_timeout"], 1.0)
            self.assertAlmostEqual(comp["detectors"]["P"]["mean_usable_lead_over_H"], (160 - 128) / 160)
            self.assertEqual(out["overall"]["decided_by"]["0.25"], 0.5)
            labels["labels"]["v02"].pop("confidence")
            (root / "labels2.json").write_text(json.dumps(labels))
            with self.assertRaises(SystemExit):
                call(module, ["--labels", str(root / "labels2.json"), "--key", str(root / "key.json"), "--output", str(root / "o2.json")])


if __name__ == "__main__":
    unittest.main()
