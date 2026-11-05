"""Simulator- and PyTorch-free tests for the frozen Hide-and-Seek + P alarm."""

import argparse
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()
from robocasa.recovery import hs_p_alarm as hs
from robocasa.recovery import recoverability_ladder as ladder
from tests.test_recoverability_ladder import MockPolicy, MockRuntime, StageMockEnv, stage_config

sys.path.insert(0, str(hs.LEAD_TIME_DIR))
import confirm_lead as cl  # noqa: E402
import lead_time  # noqa: E402


def make_spec(tasks=("MockTask",), horizon=10, rho=0.5, path=None):
    return hs.AlarmSpec({
        "kind": hs.SPEC_KIND,
        "seeds": [0, 1],
        "feature_aggregation": "last",
        "normalizations": {"0": {t: {"location": 0.0, "scale": 1.0} for t in tasks},
                           "1": {t: {"location": 1.0, "scale": 2.0} for t in tasks}},
        "tasks": {t: {"horizon_inferences": horizon, "rho": rho} for t in tasks},
        "P": {"lam": 0.1, "u": None, "s": 0.0},
        "fused": path or {"lam": 0.1, "u": 0.6, "s": 1.0},
    })


class AlarmRuleTest(unittest.TestCase):
    def test_first_crossing_matches_lead_time(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            values = rng.normal(size=rng.integers(1, 30))
            tau = float(rng.normal())
            self.assertEqual(hs.first_crossing(values, tau), lead_time.first(values, tau))
        self.assertEqual(hs.first_crossing([0.1, 0.2], math.inf), math.inf)

    def test_alarm_matches_registered_vectorized_rule(self):
        rng = np.random.default_rng(1)
        rows = []
        for i in range(60):
            horizon = int(rng.integers(5, 40))
            failed = bool(i % 2)
            length = horizon if failed else int(rng.integers(1, horizon + 1))
            rows.append({"task_name": f"T{i % 3}", "L": length, "horizon": horizon, "failed": failed,
                         "score": rng.normal(size=length)})
        rho = {"T0": 0.4, "T1": 0.7, "T2": 0.95}
        cohort = cl.Cohort(rows, "score")
        for lam, u, s in ((0.05, math.inf, 0.0), (0.0, 0.3, 0.0), (0.12, -0.2, 2.0), (0.6, 1.5, 5.0)):
            want = cl.alarm_vec(cohort, cl.rho_rows(cohort, rho), lam, u, s)
            path = {"lam": lam, "u": None if u == math.inf else u, "s": s}
            got = [hs.alarm_index(r["score"], r["horizon"], rho[r["task_name"]], path)["alarm"] for r in rows]
            np.testing.assert_array_equal(np.asarray(got, float), want)

    def test_components(self):
        self.assertEqual(hs.alarm_component(5, 3), "score")
        self.assertEqual(hs.alarm_component(5, math.inf), "timer")
        self.assertEqual(hs.alarm_component(4, 4), "both")

    def test_combined_score_normalizes_each_seed_then_averages(self):
        spec = make_spec()
        raw = {0: np.array([0.2, 0.1, 0.5]), 1: np.array([3.0, 1.0, 1.0])}
        got = hs.combined_score(raw, spec.payload["normalizations"], "MockTask")
        want = np.mean([[0.2, 0.2, 0.5], [1.0, 1.0, 1.0]], axis=0)
        np.testing.assert_allclose(got, want)

    def test_spec_alarm_fires_only_within_length(self):
        spec = make_spec(horizon=10, rho=0.5)
        quiet = spec.alarm("MockTask", np.zeros(5))
        self.assertFalse(quiet["fired"])  # timer at ceil(0.6 * 10) = 6 > L = 5
        loud = spec.alarm("MockTask", np.array([0.0, 0.8, 0.9]))
        self.assertTrue(loud["fired"])
        self.assertEqual((loud["alarm"], loud["component"]), (2, "score"))
        timed = spec.alarm("MockTask", np.zeros(10))
        self.assertEqual((timed["alarm"], timed["component"]), (6, "timer"))
        with self.assertRaises(ValueError):
            spec.alarm("OtherTask", np.zeros(3))

    def test_rounded_path_comparison(self):
        ok, _, _ = hs._same_rounded({"lam": 0.1235, "u": None, "s": 0.0},
                                    {"lam": 0.12345678, "u": math.inf, "s": 0.0}, cl.det_json)
        self.assertTrue(ok)
        ok, _, _ = hs._same_rounded({"lam": 0.1235, "u": 0.5, "s": 2.0},
                                    {"lam": 0.12345678, "u": 0.5, "s": 1.0}, cl.det_json)
        self.assertFalse(ok)


class FeaturePolicy(MockPolicy):
    def __init__(self):
        super().__init__(feature_shape=(2, 3, 5))


class FeatureRuntime(MockRuntime):
    def make_policy(self, env):
        policy = FeaturePolicy()
        self.policies.append(policy)
        return policy


class ScoreLadderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        config = stage_config(record_policy_features=True)
        ladder.run_collection(config, self.run, runtime=FeatureRuntime(env_class=StageMockEnv), log=lambda *_: None)
        self.parents = ladder.read_jsonl(self.run / ladder.PARENTS_NAME)

    def tearDown(self):
        self.tmp.cleanup()

    def score(self, scorer, spec=None, name="alarms.jsonl"):
        args = argparse.Namespace(run_dir=[self.run], output=self.root / name, device="cpu", spec_dir=None)
        summary = hs.score_ladder(args, spec=spec or make_spec(horizon=10, rho=0.5), scorer=scorer)
        return summary, ladder.read_jsonl(self.root / name)

    def test_alarm_maps_to_the_inference_snapshot(self):
        # Score crosses u + s*lam = 0.7 at the 3rd inference of every parent.
        scorer = lambda task, features: np.where(np.arange(len(features)) >= 2, 1.0, 0.0)
        summary, rows = self.score(scorer)
        self.assertEqual(len(rows), len(self.parents))
        for row, parent in zip(rows, self.parents):
            self.assertTrue(row["alarm_fired"])
            self.assertEqual(row["alarm_inference"], 3)
            self.assertEqual(row["component"], "score")
            self.assertEqual(row["alarm_snapshot_index"], 2)
            self.assertEqual(row["alarm_step"], parent["snapshot_steps"][2])
            self.assertTrue(row["feature_steps_match_snapshots"])
            self.assertEqual(row["feature_dtype"], "float16")
            self.assertEqual(row["alarm_minus_ladder_detection"], 2 - parent["anchors"]["detection"])
        self.assertEqual(summary["checks"]["feature_snapshot_mismatches"], 0)

    def test_features_are_reduced_with_the_frozen_aggregation(self):
        seen = []

        def scorer(task, features):
            seen.append(features)
            return np.zeros(len(features))

        self.score(scorer)
        self.assertEqual(seen[0].shape[1], 5)
        self.assertEqual(seen[0].dtype, np.float32)
        # MockPolicy features hold the 1-based inference index everywhere.
        np.testing.assert_array_equal(seen[0][:, 0], np.arange(1, len(seen[0]) + 1))

    def test_quiet_score_falls_back_to_timer_or_silence(self):
        spec = make_spec(horizon=10, rho=0.5)
        _, rows = self.score(lambda task, features: np.zeros(len(features)), spec=spec)
        for row in rows:
            if row["num_inferences"] >= 6:
                self.assertEqual((row["alarm_inference"], row["component"]), (6, "timer"))
            else:
                self.assertFalse(row["alarm_fired"])
                self.assertIsNone(row["alarm_snapshot_index"])

    def test_ladder_can_store_float32_features(self):
        run = self.root / "run32"
        config = stage_config(record_policy_features=True, policy_feature_dtype="float32")
        ladder.run_collection(config, run, runtime=FeatureRuntime(env_class=StageMockEnv), log=lambda *_: None)
        args = argparse.Namespace(run_dir=[run], output=self.root / "a32.jsonl", device="cpu", spec_dir=None)
        hs.score_ladder(args, spec=make_spec(), scorer=lambda task, features: np.zeros(len(features)))
        self.assertEqual({r["feature_dtype"] for r in ladder.read_jsonl(self.root / "a32.jsonl")}, {"float32"})
        with self.assertRaises(ValueError):
            ladder.resolve_config(stage_config(policy_feature_dtype="int8"))

    def test_refuses_existing_output(self):
        (self.root / "alarms.jsonl").write_text("")
        with self.assertRaises(SystemExit):
            self.score(lambda task, features: np.zeros(len(features)))


@unittest.skipIf(__import__("importlib").util.find_spec("torch") is None, "needs PyTorch")
class DetectorRoundTripTest(unittest.TestCase):
    def test_saved_detector_reproduces_scores(self):
        from robocasa.recovery.safe.baselines.detectors import build_detector

        rng = np.random.default_rng(0)
        rollouts = [{"rollout_id": str(i), "task_name": "T", "failed": bool(i % 2),
                     "features": rng.normal(size=(int(rng.integers(4, 9)), 12)).astype(np.float32)}
                    for i in range(12)]
        detector = build_detector("hide_and_seek", seed=0, epochs=2, pca_components=4).fit(rollouts)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed0.pt"
            hs.save_detector(detector, path)
            reloaded = hs.load_detector(path)
        for rollout in rollouts:
            np.testing.assert_array_equal(detector.score(rollout), reloaded.score(rollout))


if __name__ == "__main__":
    unittest.main()
