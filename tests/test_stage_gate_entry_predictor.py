import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robocasa.recovery import stage_gate_entry_predictor as ep


def _write_run(root, n_seeds=80, dim=8, signal=True, seed=0):
    """Fake trial-2 run: seeds that stall are rescued by the entry gate, others harmed."""
    rng = np.random.default_rng(seed)
    shard = Path(root) / "shard0"
    (shard / "features").mkdir(parents=True)
    pairs, rollouts = [], []
    for k in range(n_seeds):
        s = 410007 + k
        uid = f"{k:032x}"
        entered = k % 10 != 0
        stall = entered and rng.random() < 0.35
        base_ok = (not stall) and entered and rng.random() < 0.8
        entry_ok = rng.random() < 0.7 if stall else (base_ok and rng.random() < 0.6)
        entry_step = 200 + int(rng.integers(0, 100))
        rows = {}
        for arm in (ep.BASELINE, ep.ENTRY, ep.STALL):
            rows[arm] = {"pair_uid": uid, "task_name": ep.TASK, "environment_seed": s, "arm": arm,
                         "success": {ep.BASELINE: base_ok, ep.ENTRY: entry_ok, ep.STALL: base_ok}[arm],
                         "gate_stages": [], "feature_meta": []}
        if entered:
            rows[ep.STALL]["gate_stages"] = [{"stage": ep.STAGE, "entry_step": entry_step, "switched": stall}]
            n = 3
            feats = rng.normal(size=(n, 5, 30, dim)).astype(np.float16)
            if signal:
                feats[:, -1, -1, 0] += 3.0 if stall else -3.0
            meta = [{"stage": ep.STAGE, "inference_in_stage": 4 * i, "steps_in_stage": 64 * i,
                     "step": entry_step + 64 * i, "reprompted": False} for i in range(n)]
            meta.append({"stage": ep.STAGE, "inference_in_stage": 40, "steps_in_stage": 640,
                         "step": entry_step + 640, "reprompted": True})
            feats = np.concatenate([feats, feats[:1]])
            rel = f"features/{ep.TASK}_{s}_{ep.STALL}_{uid[:8]}.npz"
            np.savez_compressed(shard / rel, features=feats)
            rows[ep.STALL]["feature_meta"] = meta
            rows[ep.STALL]["features_path"] = rel
        rollouts.extend(rows.values())
        pairs.append({"pair_uid": uid})
    (shard / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (shard / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rollouts))


class ThresholdTest(unittest.TestCase):
    def test_picks_best_prefix_and_never_when_all_negative(self):
        scores = np.array([0.9, 0.8, 0.5, 0.1])
        self.assertEqual(ep.fit_threshold(scores, np.array([1.0, 1.0, -1.0, 2.0])), 0.1)
        self.assertEqual(ep.fit_threshold(scores, np.array([1.0, 1.0, -1.0, 1.0])), 0.8)  # tie: fewer re-prompts
        self.assertEqual(ep.fit_threshold(scores, np.array([1.0, 1.0, -1.0, -1.0])), 0.8)
        self.assertEqual(ep.fit_threshold(scores, np.array([-1.0, 0.0, -1.0, 0.0])), np.inf)

    def test_ties_are_one_decision(self):
        scores = np.array([0.5, 0.5, 0.1])
        self.assertEqual(ep.fit_threshold(scores, np.array([1.0, -1.0, 1.0])), 0.1)
        self.assertEqual(ep.fit_threshold(scores, np.array([1.0, -2.0, 0.0])), np.inf)

    def test_holm(self):
        np.testing.assert_allclose(ep.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


class PipelineTest(unittest.TestCase):
    def test_extract_keeps_unswitched_selector_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(tmp, n_seeds=20)
            out = Path(tmp) / "entry.npz"
            ep.extract(tmp, out)
            with np.load(out) as d:
                self.assertEqual(d["x_last"].shape[1], 8)
                self.assertEqual(len(set(d["seed"].tolist())), 18)
                self.assertEqual(len(d["seed"]), 18 * 3)  # reprompted capture dropped
                self.assertTrue(np.isnan(d["safe"]).all())
            with self.assertRaises(SystemExit):
                ep.extract(tmp, out)

    def test_cohort_and_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(tmp, n_seeds=40)
            ep.extract(tmp, Path(tmp) / "entry.npz")
            with np.load(Path(tmp) / "entry.npz") as d:
                feats = {k: d[k] for k in d.files}
            trial = ep.load_trial(tmp)
            c0 = ep.build_cohort(trial, feats, 0)
            self.assertEqual(int(c0["entered"].sum()), 36)
            self.assertTrue(np.isnan(c0["entry_step"][~c0["entered"]]).all())
            ref = ep.references(c0)
            gains = c0["y_entry"] - c0["y_base"]
            self.assertAlmostEqual(ref["oracle_stall_label"], round(gains[c0["label"]].sum() / 40, 4))
            c4 = ep.build_cohort(trial, feats, 4)
            self.assertFalse(np.allclose(c0["x"][c0["entered"]], c4["x"][c4["entered"]]))

    def test_planted_signal_is_found_and_null_is_not(self):
        for planted in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                _write_run(tmp, n_seeds=120, signal=planted, seed=1)
                ep.extract(tmp, Path(tmp) / "entry.npz")
                res = ep.analyze(tmp, Path(tmp) / "entry.npz", Path(tmp) / "out", repeats=2, folds=5,
                                 permutations=9, signals=("probe", "entry_step"))
                probe = res["decisions"]["0"]["signals"][0]
                self.assertIn("holm_p", probe)
                self.assertFalse(res["safe_scored"])
                if planted:
                    self.assertGreater(probe["auroc_stall"], 0.9)
                    self.assertGreater(probe["delta_vs_baseline"], 0.05)
                    self.assertLessEqual(probe["permutation_p"], 0.1)
                    self.assertGreaterEqual(probe["delta_vs_baseline"],
                                            res["decisions"]["0"]["references"]["oracle_stall_label"] - 0.05)
                else:
                    self.assertLess(probe["auroc_stall"], 0.75)
                self.assertTrue((Path(tmp) / "out" / "entry_predictor.json").exists())


if __name__ == "__main__":
    unittest.main()
