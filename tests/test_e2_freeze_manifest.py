"""R10 part B: E2's freeze manifest (scripts/xr1_lead_time/e2_freeze_manifest.py)."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parents[0] / "scripts" / "xr1_lead_time"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS.parents[0]))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fz = load("frozen_fixture_manifest", TESTS / "test_stage2_frozen_sizing.py")
m = load("e2_freeze_manifest", SCRIPTS / "e2_freeze_manifest.py")
import confirm_lead as cl  # noqa: E402
import stage2_sizing as s2  # noqa: E402
from robocasa.recovery.safe.baselines.detectors import content_hash  # noqa: E402

CONFIG = {"name": "hide_and_seek", "seed": 0, "epochs": 200, "margin": 0.2}


def scoring(inputs, extra=None):
    return dict(inputs=inputs, feature_aggregation="last", seeds=[0, 1, 2], epochs=200,
                detectors={"hide_and_seek": {"configs": {"0": CONFIG}, **(extra or {})}})


def freeze_fixture(root):
    """A 15.22-like scores dir with scoring.json and normalizations.json, and a matching freeze run."""
    common, data = fz.fixture(root)
    hs = root / "hs"
    inputs = {k: {"path": f"/x/{k}", "sha256": k * 4} for k in ("runtime_bundle", "train_ids", "cohort_calibration", "cohort_test")}
    norms = {"hide_and_seek": {s: {"TaskA": {"location": 0.1, "scale": 1.0}} for s in m.SEEDS}}
    (hs / "scoring.json").write_text(json.dumps(scoring(inputs)))
    (hs / "normalizations.json").write_text(json.dumps(norms))
    fr = root / "freeze" / "scores"
    (root / "freeze" / "models").mkdir(parents=True)
    fr.mkdir(parents=True)
    rng = np.random.default_rng(0)
    models = {}
    for s in m.SEEDS:
        arrays = {"input_mean": rng.standard_normal(4), "input_std": np.ones(4), "state.lstm.weight_ih_l0": rng.standard_normal((8, 4)).astype(np.float32)}
        p = root / "freeze" / "models" / f"hide_and_seek_seed{s}.npz"
        np.savez(p, meta_json=np.array(json.dumps({"name": "hide_and_seek", "config": dict(CONFIG, seed=int(s)), "pca_params": None})), **arrays)
        models[s] = dict(path=str(p), sha256=s2.sha256(p), content_sha256=content_hash(arrays), reload_identical=True)
    checks = {f"{c}/hide_and_seek": dict(ok=True, max_abs_diff=0.0, rollouts_compared=10, rollouts_scored=10) for c in m.COHORTS}
    meta = scoring(inputs, {"models": models}) | {"reference_checks": checks, "code_commit": "abc"}
    (fr / "scoring.json").write_text(json.dumps(meta))
    (fr / "normalizations.json").write_text(json.dumps(norms))
    tree = root / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "a.bin").write_bytes(b"alpha")
    (tree / "sub" / "b.bin").write_bytes(b"beta")
    os.symlink(tree / "a.bin", tree / "sub" / "link.bin")
    return common, data, fr, tree


def run(common, fr, tree, out):
    argv = common + ["--freeze-scores-dir", str(fr), "--tree", f"safe_lstm_seed0={tree}", "--file",
                     f"bundle={tree / 'a.bin'}", "--git", f"code={TESTS.parents[0]}", "--output", str(out)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = m.main(argv)
    return status, buf.getvalue()


class FreezeManifestTests(unittest.TestCase):
    def test_writes_manifest_with_paths_models_trees_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common, (rows, train, cal_s, test, tasks, rk), fr, tree = freeze_fixture(root)
            status, text = run(common, fr, tree, root / "manifest.json")
            self.assertEqual(status, 0, text)
            self.assertIn(f"MANIFEST {root / 'manifest.json'} sha256 {s2.sha256(root / 'manifest.json')}", text)
            out = json.loads((root / "manifest.json").read_text())
            self.assertEqual(set(out["tasks"]), set(tasks))
            self.assertEqual(out["tasks"]["TaskA"]["H_k"], 30)
            self.assertAlmostEqual(out["tasks"]["TaskA"]["rho_k"], rk["TaskA"])
            for key, det in (("safe", "safe_plus_P"), ("hide_and_seek", "hide_and_seek_plus_P")):
                _, fu = cl.fit_fast(cl.Cohort(cal_s + [r for r in train if r["failed"]], key), rk)
                self.assertEqual((out["detectors"][det]["path"]["u"], out["detectors"][det]["path"]["s"]), (fu["u"], fu["s"]))
            self.assertEqual(set(out["detectors"]["hide_and_seek_plus_P"]["models"]), set(m.SEEDS))
            t = out["trees"]["safe_lstm_seed0"]
            self.assertEqual((t["files"], [e["path"] for e in t["entries"]]), (3, ["a.bin", "sub/b.bin", "sub/link.bin"]))
            self.assertEqual(t["entries"][2]["sha256"], t["entries"][0]["sha256"])      # symlink followed
            self.assertEqual(out["detectors"]["safe_plus_P"]["artifacts"], ["safe_lstm_seed0"])
            self.assertIn("this_repository", out["git"])

    def test_refuses_when_the_freeze_run_differs(self):
        cases = {
            "normalizations": lambda fr: (fr / "normalizations.json").write_text(json.dumps({"hide_and_seek": {}})),
            "reference": lambda fr: (fr / "scoring.json").write_text((fr / "scoring.json").read_text().replace('"max_abs_diff": 0.0', '"max_abs_diff": 1e-07', 1)),
            "model file": lambda fr: Path(json.loads((fr / "scoring.json").read_text())["detectors"]["hide_and_seek"]["models"]["1"]["path"]).write_bytes(b"changed"),
            "inputs": lambda fr: (fr / "scoring.json").write_text((fr / "scoring.json").read_text().replace("train_idstrain_ids", "other", 1)),
        }
        for name, spoil in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                common, _, fr, tree = freeze_fixture(root)
                spoil(fr)
                with self.assertRaises(SystemExit):
                    run(common, fr, tree, root / "manifest.json")
                self.assertFalse((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
