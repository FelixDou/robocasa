"""Fold mode of the Section 15.10 OOD baselines (pooled refits for unseen tasks)."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.offline_detector_baselines import (
    AGGREGATIONS,
    DEFAULT_SELECTION_LANDMARKS,
    BaselineScorer,
    aggregate_scores,
    pooled_variant,
)

OOD = ["knn_cosine", "mahalanobis", "pca_kmeans", "rnd"]


class PooledScorerTest(unittest.TestCase):
    def test_pooled_variants_score_unseen_tasks_and_per_task_ones_refuse(self):
        rng = np.random.default_rng(0)
        features = {"A": rng.normal(size=(40, 6)), "B": rng.normal(loc=2.0, size=(40, 6))}
        unseen = rng.normal(loc=5.0, size=(7, 6))
        cases = {
            "knn_cosine": {"k": 3, "scope": "per_task"},
            "mahalanobis": {"pca_dim": 4},
            "pca_kmeans": {"pca_dim": 4, "clusters": 3, "scope": "per_task"},
        }
        for name, params in cases.items():
            per_task = BaselineScorer(name, params, seed=0).fit(features)
            self.assertFalse(per_task.task_agnostic)
            with self.assertRaisesRegex(ValueError, "not fitted for task"):
                per_task.score("C", unseen)
            pooled = BaselineScorer(name, pooled_variant(name, params), seed=0).fit(features)
            self.assertTrue(pooled.task_agnostic)
            values = pooled.score("C", unseen)
            self.assertEqual(values.shape, (7,))
            self.assertTrue(np.all(np.isfinite(values)))
            # The pooled model does not depend on the task name it is asked about.
            np.testing.assert_array_equal(values, pooled.score("A", unseen))
        rnd_params = {"hidden_dim": 8, "output_dim": 4, "max_iter": 3}
        rnd = BaselineScorer("rnd", rnd_params, seed=0).fit(features)
        self.assertFalse(rnd.task_agnostic)
        with self.assertRaisesRegex(ValueError, "not fitted for task"):
            rnd.score("C", unseen)
        pooled_rnd = BaselineScorer("rnd", pooled_variant("rnd", rnd_params), seed=0).fit(features)
        np.testing.assert_array_equal(pooled_rnd.score("C", unseen), rnd.score("A", unseen))
        self.assertEqual(pooled_variant("rnd", {"hidden_dim": 8}), {"hidden_dim": 8, "scope": "pooled"})
        with self.assertRaises(ValueError):
            pooled_variant("action_inconsistency", {})
        # Defaults are unchanged: a Mahalanobis model without a scope is per task.
        far = BaselineScorer("mahalanobis", {"pca_dim": 4}, seed=0).fit(features)
        self.assertEqual(sorted(far.gaussians), ["A", "B"])


class FoldModeTest(unittest.TestCase):
    def test_fold_develop_and_score_fold(self):
        from robocasa.recovery.safe.run_offline_detector_baselines import (
            develop,
            score_fold,
            synthetic_smoke,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_smoke(argparse.Namespace(output_dir=str(root / "smoke"), seed=0, workers=1))
            smoke = root / "smoke"
            fixture = smoke / "fixture"
            source_path = smoke / "develop" / "baseline_bundle.json"
            source = json.loads(source_path.read_text())
            runtime = next(fixture.rglob("runtime_bundle.json"))
            tasks = sorted(json.loads(runtime.read_text())["task_horizons"])
            self.assertEqual(len(tasks), 2)
            folds = root / "folds.json"
            folds.write_text(json.dumps({"folds": [{"fold": 0, "held_out": [tasks[0]]},
                                                   {"fold": 1, "held_out": [tasks[1]]}]}))
            dev_cache = next(
                path.parent for path in fixture.rglob("provenance.json")
                if json.loads(path.read_text()).get("group") == "development"
            )

            def fold_develop(output, seed, **extra):
                options = dict(
                    development_cache=str(dev_cache), runtime_bundle=str(runtime), split_manifest=None,
                    output_dir=str(root / output), baselines=list(OOD), aggregations=list(AGGREGATIONS),
                    grid="smoke", selection_landmarks=list(DEFAULT_SELECTION_LANDMARKS),
                    validation_fraction=0.3, meta_split_seed=0, model_seed=seed, max_fit_rows=50000,
                    fixed_selection=str(source_path), folds=str(folds), fold="0",
                )
                options.update(extra)
                return develop(argparse.Namespace(**options))

            bundles = {}
            for seed in (0, 1):
                status = fold_develop(f"fold0_seed{seed}", seed)
                bundles[seed] = Path(status["baseline_bundle"])
            b0 = json.loads(bundles[0].read_text())
            self.assertEqual((b0["fold"], b0["held_out_tasks"], b0["variant"]), ("0", [tasks[0]], "pooled"))
            self.assertEqual(b0["fit_rollout_ids"], source["fit_rollout_ids"])
            self.assertIsNone(b0["fixed_selection"]["reproduction"])
            for name in OOD:
                selection = b0["selected"][name]
                self.assertEqual(selection["params"], pooled_variant(name, source["selected"][name]["params"]))
                self.assertEqual(selection["aggregation"], source["selected"][name]["aggregation"])
                self.assertEqual(sorted(selection["normalization"]), [tasks[1]])
                self.assertEqual(sorted(selection["fit_summary"]["fit_rows_per_task"]), [tasks[1]])
                self.assertGreater(selection["pooled_normalization"]["scale"], 0)

            with self.assertRaisesRegex(ValueError, "fixed-selection"):
                fold_develop("no_fixed", 0, fixed_selection=None)
            with self.assertRaisesRegex(ValueError, "not in the fold file"):
                fold_develop("bad_fold", 0, fold="7")

            cache = next(
                path.parent for path in fixture.rglob("provenance.json")
                if json.loads(path.read_text()).get("group") == "confirmatory_test"
                and (path.parent / "features.npy").is_file()
            )
            predictions = smoke / "confirm" / "event_predictions.jsonl"
            records = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]

            def run(output, items, folds_path=folds):
                return score_fold(argparse.Namespace(
                    bundle=items, folds=str(folds_path),
                    cohort=[["confirmatory_test", str(predictions), str(cache), "confirmatory_test"]],
                    output_dir=str(root / output)))

            run("scored", [f"0={bundles[0]}", f"1={bundles[1]}"])
            out = root / "scored"
            rows = [json.loads(line) for line in (out / "confirmatory_test_predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), len(records) * len(OOD))
            self.assertEqual({r["detector"] for r in rows}, set(OOD))
            for r in rows:
                self.assertEqual(r["held_out"], r["task_name"] == tasks[0])
                self.assertEqual(sorted(r["per_seed_raw"]), ["0", "1"])
                for values in r["per_seed_raw"].values():
                    self.assertEqual(len(values), r["source_inferences"])
                    self.assertTrue(np.all(np.diff(values) >= 0))
            norms = json.loads((out / "normalizations.json").read_text())
            self.assertEqual(sorted(norms), sorted(OOD))
            self.assertEqual(sorted(norms["rnd"]["1"]["per_task"]), [tasks[1]])
            meta = json.loads((out / "scoring.json").read_text())
            self.assertEqual((meta["fold"], meta["held_out_tasks"], meta["seeds"]), ("0", [tasks[0]], [0, 1]))
            self.assertEqual(sorted(meta["training_rollouts_by_task"]), [tasks[1]])

            # The stored trajectory is the seed's model score, aggregated as selected.
            import pickle
            sample = next(r for r in rows if r["held_out"] and r["detector"] == "mahalanobis")
            from robocasa.recovery.safe.run_offline_detector_baselines import FeatureCache
            fc = FeatureCache(cache)
            row = fc.by_id[sample["rollout_id"]]
            with open(b0["selected"]["mahalanobis"]["model_path"], "rb") as stream:
                model = pickle.load(stream)
            expected = aggregate_scores(model.score(sample["task_name"], fc.features_for(row)),
                                        b0["selected"]["mahalanobis"]["aggregation"])
            np.testing.assert_allclose(sample["per_seed_raw"]["0"], expected)

            with self.assertRaisesRegex(ValueError, "develop --fold"):
                run("not_fold", [f"0={source_path}"])
            with self.assertRaisesRegex(ValueError, "model seed"):
                run("wrong_seed", [f"0={bundles[1]}"])
            other = root / "folds_other.json"
            other.write_text(json.dumps({"folds": [{"fold": 0, "held_out": [tasks[1]]},
                                                   {"fold": 1, "held_out": [tasks[0]]}]}))
            with self.assertRaisesRegex(ValueError, "holds out other tasks"):
                run("other_folds", [f"0={bundles[0]}"], folds_path=other)


if __name__ == "__main__":
    unittest.main()
