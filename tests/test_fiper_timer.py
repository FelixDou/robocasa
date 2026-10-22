import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "published_eval" / "fiper_timer.py"


def write(path, length, success=None, subtype="id"):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"task": path.parts[-4], "rollout_subtype": subtype}
    if success is not None:
        meta["successful"] = success
    path.write_bytes(pickle.dumps({"metadata": meta, "rollout": [{"step": i} for i in range(length)]}))


class Evil:
    def __reduce__(self):
        return (os.system, ("echo should-not-run",))


def run(*args, cwd):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *args], capture_output=True, text=True, cwd=cwd)


class TestFiperTimer(unittest.TestCase):
    def build(self, root):
        d = root / "data_all"
        # sorting: calibration successes of length 10..19 (maxima 9..18), one calibration failure.
        for i in range(10):
            write(d / "sorting/rollouts/calibration" / f"episode_s_{i:04d}.pkl", 10 + i)
        write(d / "sorting/rollouts/calibration/episode_f_0099.pkl", 30)
        for i in range(6):
            write(d / "sorting/rollouts/test" / f"episode_s_{100 + i:04d}.pkl", 12 + i, subtype="ood" if i % 2 else "id")
            write(d / "sorting/rollouts/test" / f"episode_f_{200 + i:04d}.pkl", 30)
        # push_chair: outcome only in metadata.
        for i in range(4):
            write(d / "push_chair/rollouts/calibration" / f"episode_{i:04d}.pkl", 8 + i, success=True)
        for i in range(4):
            write(d / "push_chair/rollouts/test" / f"episode_{10 + i:04d}.pkl", 9 if i < 2 else 20, success=i < 2)
        write(d / "sorting/rollouts/calibration_unused/episode_s_0500.pkl", 99)
        return d

    def test_index_and_timer_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); d = self.build(root); work = root / "work"; work.mkdir()
            r = run("index", str(d), "--output", str(work / "index.jsonl"), "--workers", "1", cwd=work)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            r = run("analyze", str(work / "index.jsonl"), "--output", str(work / "res.json"), cwd=work)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            res = json.loads((work / "res.json").read_text())
            s = res["tasks"]["sorting"]
            self.assertEqual(s["calibration_successes"], 10)
            self.assertEqual(s["calibration_failures_ignored"], 1)
            self.assertEqual(s["T"], 30)                       # calibration_unused (99) ignored
            self.assertEqual(s["auc_length"], 1.0)
            first = s["timer"]["linear"]["per_level"][0]       # level 0.90 over maxima 9..18
            self.assertAlmostEqual(first["gamma"], 17.1)
            self.assertEqual(first["t_alarm"], 18)
            self.assertEqual(first["tpr"], 1.0)
            self.assertEqual(first["tnr"], 1.0)                # test successes have maxima 11..16 < 18
            self.assertAlmostEqual(first["dt"], 18 / 30)
            self.assertAlmostEqual(first["twa"], 0.5 * ((1 - 18 / 30) + 1.0))
            conf_last = s["timer"]["conformal"]["per_level"][-1]   # ceil(11 * 0.99) = 11 > 10 -> never alarms
            self.assertEqual(conf_last["tpr"], 0.0)
            self.assertEqual(set(s["by_subtype"]), {"id", "ood"})
            pc = res["tasks"]["push_chair"]
            self.assertEqual((pc["test_successes"], pc["test_failures"]), (2, 2))

    def test_refused_and_conflicting_files_stop_the_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); d = self.build(root); work = root / "work"; work.mkdir()
            (d / "sorting/rollouts/test/episode_f_0300.pkl").write_bytes(pickle.dumps(Evil()))
            r = run("index", str(d), "--output", str(work / "i1.jsonl"), "--workers", "1", cwd=work)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("STOP", r.stdout)
            self.assertNotIn("should-not-run", r.stdout + r.stderr)
            (d / "sorting/rollouts/test/episode_f_0300.pkl").unlink()
            write(d / "sorting/rollouts/test/episode_s_0301.pkl", 12, success=False)
            r = run("index", str(d), "--output", str(work / "i2.jsonl"), "--workers", "1", cwd=work)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("conflict", r.stdout)


if __name__ == "__main__":
    unittest.main()
