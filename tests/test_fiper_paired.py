import contextlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "published_eval" / "fiper_paired"


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


class TestMetrics(unittest.TestCase):
    def test_metrics_match_fiper_definitions(self):
        m = load("analyze_paired").metrics
        first = np.array([3, -1, 5, -1])            # rollouts 0,1 fail; 2,3 succeed
        success = np.array([False, False, True, True])
        lengths = np.array([11, 11, 7, 6])
        r = m(first, success, lengths, T=11)
        self.assertEqual((r["tpr"], r["tnr"], r["fpr"]), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(r["dt"], 3 / 10)
        self.assertAlmostEqual(r["twa"], 0.5 * ((1 - 0.3) / 2 + 1 / 2))
        self.assertAlmostEqual(r["lead"], ((11 - 1 - 3) / 10 + 0) / 2)
        self.assertAlmostEqual(r["accuracy"], 2 / 4)


class TestAnalysis(unittest.TestCase):
    def test_end_to_end_on_synthetic_dumps(self):
        module = load("analyze_paired")
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("push_t", "sorting"):
                n = 40
                success = np.arange(n) % 2 == 0
                lengths = np.where(success, rng.integers(8, 15, n), 30)
                for seed in (0, 1):
                    out = {}
                    for method in ("time", "rnd_oe_and_entropy", "rnd_oe"):
                        alarms = {}
                        for style in ("ct_quantile", "tvt_quantile"):
                            for q in (0.9, 0.95, 0.99):
                                for w in ("1", "5"):
                                    cut = int(14 + 10 * (q - 0.9) * 10)
                                    if method == "time":
                                        first = np.where(lengths > cut, cut, -1)
                                    else:
                                        early = rng.integers(3, 12, n)
                                        first = np.where(~success, early, np.where(rng.random(n) < 0.05, 4, -1))
                                    alarms[(style, q, w)] = first.astype(np.int64)
                        out[method] = {"first_alarm": alarms, "lengths": lengths.astype(np.int64),
                                       "successful": success, "max_episode_length": 30}
                    (root / f"seed{seed}_{task}.pkl").write_bytes(pickle.dumps(out))
            call(module, [str(root), "--output", str(root / "res.json"), "--bootstrap", "30"])
            res = json.loads((root / "res.json").read_text())
            self.assertEqual(res["tasks"], ["push_t", "sorting"])
            self.assertIn("time", res["selected"])
            self.assertGreater(res["matched_fpr_average"]["diff"], 0)        # synthetic FIPER alarms earlier
            self.assertEqual(len(res["matched_fpr_average"]["ci95"]), 2)
            self.assertIn("within_tol", res["reproduction"]["push_t"])
            self.assertAlmostEqual(res["table"]["time"]["push_t"]["tnr"], 1.0)
            with self.assertRaises(SystemExit):
                call(module, [str(root), "--output", str(root / "res.json")])


class TestDump(unittest.TestCase):
    def test_inserted_dump_writes_what_the_analysis_reads(self):
        setup, analysis = load("setup_fiper"), load("analyze_paired")
        import os
        ns = {"os": os}
        exec(setup.DUMP_FN, ns)
        scores = [np.array([0.5, 1.2, 0.3]), np.array([0.2, 0.4]), np.array([0.9, 1.0, 1.5, 2.0])]
        results = {"time": {"test_scores_by_threshold": {"ct_quantile": {0.9: {1: scores, 5: scores}}},
                            "successful_test_rollouts": [False, True, False], "max_episode_length": 4},
                   "rnd_oe_and_entropy": {"test_scores_by_threshold": {"ct_quantile": {0.9: {1: scores}}},
                                          "successful_test_rollouts": [False, True, False], "max_episode_length": 4}}
        with tempfile.TemporaryDirectory() as directory:
            os.environ["FIPER_FIRST_ALARM_DIR"] = directory
            try:
                ns["_dump_first_alarms"](results, "push_t", 42)
            finally:
                del os.environ["FIPER_FIRST_ALARM_DIR"]
            data = analysis.load(directory)
            d = data["push_t"][42]["time"]
            np.testing.assert_array_equal(d["first_alarm"][("ct_quantile", 0.9, "1")], [1, -1, 2])
            np.testing.assert_array_equal(d["lengths"], [3, 2, 4])
            self.assertEqual(d["successful"].tolist(), [False, True, False])
            self.assertEqual(d["max_episode_length"], 4)


ANCHORS = {
    "evaluation/method_eval_classes/__init__.py": "from .base_eval_class import *\nfrom .logpzo_eval import *\n",
    "configs/default.yaml": 'rnd_models: ["rnd_oe", "rnd_a"]\nmethods: ["entropy", "similarity", "tc", "logpzo"]\n'
                            'implemented_methods: ["rnd_oe", "rnd_a", "entropy", "tc", "similarity", "logpzo"]\n',
    "configs/results/base.yaml": "  Method:\n    - entropy\n    - rnd_oe_and_entropy\n    # - rnd_oe_or_entropy\n",
    "configs/eval/base.yaml": "window_sizes: [1]\n",
    "scripts/run_fiper.py": "import os\n\ndef main():\n    for task in []:\n        if True:\n            total_results[task] = results\n\n\ndef check_inputs(cfg):\n    return\n",
}


class TestSetup(unittest.TestCase):
    def test_setup_edits_anchors_links_data_and_refuses_dirty_clone(self):
        module = load("setup_fiper")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); repo = root / "fiper"
            for rel, text in ANCHORS.items():
                (repo / rel).parent.mkdir(parents=True, exist_ok=True); (repo / rel).write_text(text)
            git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            git("add", "-A"); git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
            module.PINNED = git("rev-parse", "HEAD")
            for task in module.TASKS:
                (root / "data_all" / task / "rollouts" / "test").mkdir(parents=True)
            argv = ["--repo", str(repo), "--rollouts-root", str(root / "data_all"), "--run-dir", str(root / "run")]
            call(module, argv)
            self.assertIn('methods: ["entropy", "time"]', (repo / "configs/default.yaml").read_text())
            self.assertIn("    - time\n", (repo / "configs/results/base.yaml").read_text())
            run_py = (repo / "scripts/run_fiper.py").read_text()
            self.assertIn("_dump_first_alarms(results, task, seed)", run_py)
            compile(run_py, "run_fiper.py", "exec")
            compile((repo / "evaluation/method_eval_classes/time_eval.py").read_text(), "time_eval.py", "exec")
            self.assertTrue((repo / "data").is_symlink())
            self.assertTrue((root / "run/data/push_t/rollouts").resolve() == (root / "data_all/push_t/rollouts").resolve())
            with self.assertRaises(SystemExit):
                call(module, argv)                     # clone now has local changes


if __name__ == "__main__":
    unittest.main()
