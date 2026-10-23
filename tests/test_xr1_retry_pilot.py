import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "xr1_retry_pilot"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(module, argv):
    old = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()
    finally:
        sys.argv = old


def write_collection(root, outcomes_by_task, prefix="orig"):
    """outcomes_by_task[task] = list of failed flags in episode order; rollout ids shared across collections."""
    rows = []
    for ti, (task, flags) in enumerate(sorted(outcomes_by_task.items())):
        for ep, failed in enumerate(flags):
            rows.append({"rollout_id": f"{task}-{ep}", "task_name": task, "task_type": "atomic" if ti % 2 else "composite",
                         "environment_seed": 1000 + 500 * ti + ep, "environment_reset_index": ep, "failed": bool(failed),
                         "video_path": f"videos/{task}-{ep}.mp4"})
    (root / "shard0").mkdir(parents=True, exist_ok=True)
    # Real manifests carry no task_type; the shard summary records it in its config.
    (root / "shard0" / "manifest.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "task_type"}) + "\n" for r in rows))
    (root / "shard0" / "summary.json").write_text(json.dumps({"config": {"task_types": {r["task_name"]: r["task_type"] for r in rows}}}))
    return rows


class TestPlan(unittest.TestCase):
    def test_selection_rule_and_limits(self):
        module = load("plan_pilot")
        rng = np.random.default_rng(0)
        outcomes = {}
        for i in range(14):
            p = 0.1 + 0.06 * i                                   # success rates 0.10 ... 0.88
            outcomes[f"Task{i:02d}"] = list(rng.random(60) > p)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_collection(root / "coll", outcomes)
            call(module, [str(root / "coll"), "--output", str(root / "pilot")])
            sel = json.loads((root / "pilot" / "selection.json").read_text())
            limits = json.loads((root / "pilot" / "attempt_limits.json").read_text())
            self.assertEqual(len(sel["tasks"]), 10)
            ps = {t: 1 - np.mean(f) for t, f in outcomes.items()}
            dropped = sorted(set(outcomes) - set(sel["tasks"]), key=lambda t: abs(ps[t] - 0.5))
            self.assertTrue(all(abs(ps[d] - 0.5) >= max(abs(ps[t] - 0.5) for t in sel["tasks"]) - 1e-12 for d in dropped))
            for t, lim in limits.items():
                fails = [ep for ep, f in enumerate(outcomes[t]) if f]
                self.assertEqual(lim, fails[9] + 1)
            s0 = (root / "pilot" / "tasks_shard0.txt").read_text().split()
            s1 = (root / "pilot" / "tasks_shard1.txt").read_text().split()
            self.assertEqual(sorted(s0 + s1), sorted(limits))
            self.assertEqual(sum(sel["types"].values()), 10)               # task types resolved from summaries
            with self.assertRaises(SystemExit):
                call(module, [str(root / "coll"), "--output", str(root / "pilot")])


class TestAnalyze(unittest.TestCase):
    def setup_case(self, root, frame_noise=0.0, wrong_seed=False):
        orig = {"TaskA": [1, 0, 1, 1, 0, 1], "TaskB": [0, 1, 1, 0, 1, 0]}     # 1 = failed
        retry = {"TaskA": [0, 0, 1, 1, 0, 0], "TaskB": [0, 0, 1, 0, 1, 1]}
        write_collection(root / "orig", orig)
        rows = write_collection(root / "pilot", retry)
        if wrong_seed:
            rows[0]["environment_seed"] += 1
            (root / "pilot" / "shard0" / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        sel = {"tasks": {"TaskA": {"p": 0.5, "task_type": "composite"}, "TaskB": {"p": 0.5, "task_type": "atomic"}}}
        (root / "selection.json").write_text(json.dumps(sel))
        module = load("analyze_pilot")
        rng = np.random.default_rng(1)
        frames = {}

        def fake_first_frame(path):
            key = Path(path).name
            if key not in frames:
                frames[key] = rng.integers(0, 255, (4, 4, 3)).astype(float)
            noise = frame_noise if "pilot" in str(path) else 0.0
            return frames[key] + noise
        module.first_frame = fake_first_frame
        return module

    def test_rates_and_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); module = self.setup_case(root)
            call(module, [str(root / "pilot"), str(root / "orig"), "--selection", str(root / "selection.json"),
                          "--output", str(root / "res.json"), "--bootstrap", "50"])
            res = json.loads((root / "res.json").read_text())
            # Originally failed: A eps 0,2,3,5 -> retry flags 0,1,1,0 -> 2 successes;
            # B eps 1,2,4 -> retry flags 0,1,1 -> 1 success. Total 3/7.
            # Originally successful: A eps 1,4 -> 0,0 -> 2; B eps 0,3,5 -> 0,0,1 -> 2. Total 4/5.
            self.assertEqual((res["r_fail"]["successes"], res["r_fail"]["n"]), (3, 7))
            self.assertEqual((res["r_succ"]["successes"], res["r_succ"]["n"]), (4, 5))
            self.assertTrue(res["gates"]["scene_gate_passed"])

    def test_scene_gate_voids_the_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); module = self.setup_case(root, frame_noise=10.0)
            with self.assertRaises(SystemExit):
                call(module, [str(root / "pilot"), str(root / "orig"), "--selection", str(root / "selection.json"),
                              "--output", str(root / "res.json"), "--bootstrap", "5"])
            self.assertIn("void", json.loads((root / "res.json").read_text())["status"])

    def test_identity_gate_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); module = self.setup_case(root, wrong_seed=True)
            with self.assertRaises(SystemExit):
                call(module, [str(root / "pilot"), str(root / "orig"), "--selection", str(root / "selection.json"),
                              "--output", str(root / "res.json"), "--bootstrap", "5"])
            self.assertFalse((root / "res.json").exists())


if __name__ == "__main__":
    unittest.main()
