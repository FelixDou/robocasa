import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.offline_detector_baselines import (
    AGGREGATIONS,
    SMOKE_GRIDS,
    BaselineScorer,
    aggregate_scores,
    landmark_auc_rows,
    stac_single,
    stratified_meta_split,
)
from robocasa.recovery.safe.prospective_matched_fpr import (
    first_alert,
    paired_failure_comparison,
    select_staged_thresholds,
)


def record(rollout_id, failed, safe, time, task="TaskA", horizon=10):
    return {
        "rollout_id": rollout_id,
        "task_name": task,
        "failed": failed,
        "horizon": horizon,
        "source_inferences": len(safe),
        "safe_scores": np.asarray(safe, dtype=np.float64),
        "time_scores": np.asarray(time, dtype=np.float64),
    }


class TestOfflineBaselineScorers(unittest.TestCase):
    def test_stac_single_matches_official_overlap(self):
        rng = np.random.default_rng(0)
        actions = rng.normal(size=(4, 5, 3))
        values = stac_single(actions, exec_horizon=2)
        self.assertEqual(values[0], 0.0)
        expected = np.mean(np.linalg.norm(actions[0, 2:] - actions[1, :3], axis=-1))
        self.assertAlmostEqual(values[1], expected)
        with self.assertRaises(ValueError):
            stac_single(actions, exec_horizon=5)

    def test_aggregations_are_causal_and_monotone(self):
        raw = np.asarray([0.3, 0.1, 0.9, 0.2])
        for aggregation in AGGREGATIONS:
            full = aggregate_scores(raw, aggregation)
            self.assertTrue(np.all(np.diff(full) >= 0))
            for end in range(1, len(raw) + 1):
                np.testing.assert_allclose(aggregate_scores(raw[:end], aggregation), full[:end])

    def test_every_scorer_is_causal_per_inference(self):
        rng = np.random.default_rng(1)
        fit = {
            "TaskA": rng.normal(size=(60, 8)).astype(np.float32),
            "TaskB": rng.normal(loc=1.0, size=(60, 8)).astype(np.float32),
        }
        features = rng.normal(size=(7, 8)).astype(np.float32)
        actions = rng.normal(size=(7, 4, 2)).astype(np.float32)
        for name, grid in SMOKE_GRIDS.items():
            params = dict(grid[0], **({"pca_dim": 4} if "pca_dim" in grid[0] else {}))
            scorer = BaselineScorer(name, params).fit(fit, exec_horizon=2)
            full = scorer.score("TaskB", features, actions)
            self.assertEqual(full.shape, (7,))
            self.assertTrue(np.all(np.isfinite(full)))
            for end in (1, 3, 6):
                np.testing.assert_allclose(
                    scorer.score("TaskB", features[:end], actions[:end]),
                    full[:end],
                    rtol=1e-5,
                    atol=1e-6,
                )
            with self.assertRaises(ValueError):
                scorer.score("UnknownTask", features, actions)

    def test_success_reference_scores_farther_inputs_higher(self):
        rng = np.random.default_rng(2)
        fit = {"TaskA": rng.normal(size=(200, 6)).astype(np.float32)}
        near = rng.normal(size=(20, 6)).astype(np.float32)
        far = (near + 4.0).astype(np.float32)
        for name, params in (
            ("mahalanobis", {"pca_dim": 4}),
            ("pca_kmeans", {"pca_dim": 4, "clusters": 4, "scope": "per_task"}),
            ("rnd", {"hidden_dim": 32, "output_dim": 8, "max_iter": 30}),
        ):
            scorer = BaselineScorer(name, params).fit(fit)
            self.assertGreater(
                np.mean(scorer.score("TaskA", far)), np.mean(scorer.score("TaskA", near))
            )
        knn = BaselineScorer("knn_cosine", {"k": 3, "scope": "pooled"}).fit(fit)
        self.assertGreater(
            np.mean(knn.score("TaskA", -near)), np.mean(knn.score("TaskA", near))
        )

    def test_meta_split_is_stratified_disjoint_and_deterministic(self):
        rows = [
            {"rollout_id": f"{task}-{failed}-{index}", "task_name": task, "failed": failed}
            for task in ("TaskA", "TaskB")
            for failed, count in ((False, 3), (True, 5))
            for index in range(count)
        ]
        first = stratified_meta_split(rows, 0.3, seed=0)
        second = stratified_meta_split(rows, 0.3, seed=0)
        self.assertEqual(first["fit"], second["fit"])
        self.assertFalse(first["fit"] & first["validation"])
        self.assertEqual(len(first["fit"] | first["validation"]), len(rows))
        for task in ("TaskA", "TaskB"):
            self.assertEqual(first["counts"][task]["success"], {"fit": 2, "validation": 1})
            self.assertGreaterEqual(first["counts"][task]["failure"]["validation"], 1)
        with self.assertRaises(ValueError):
            stratified_meta_split(rows[:2] + rows[3:8], 0.3, seed=0)

    def test_landmark_auc_uses_only_active_rollouts(self):
        rows = [
            {"task_name": "TaskA", "failed": False, "score": [0.1, 0.2]},
            {"task_name": "TaskA", "failed": True, "score": [0.3, 0.4, 0.5, 0.6]},
            {"task_name": "TaskA", "failed": False, "score": [0.0, 0.1, 0.1, 0.1]},
        ]
        values = landmark_auc_rows(rows, {"TaskA": 4}, "score", (0.25, 1.0))
        self.assertEqual(values[0]["at_risk_rollouts"], 3)
        self.assertEqual(values[1]["at_risk_rollouts"], 2)
        self.assertEqual(values[1]["task_macro_roc_auc"], 1.0)


class TestFusedDetector(unittest.TestCase):
    def test_fused_detector_makes_both_sources_eligible_throughout(self):
        row = record("f", True, [0.1, 0.1, 0.1, 0.1, 0.9, 0.9], [0.9] + [0.1] * 5)
        thresholds = {"early_safe": 0.8, "late_time": 0.8}
        self.assertEqual(first_alert(row, "fused_safe_time", thresholds)["step"], 1)
        staged = first_alert(row, "staged_safe_time", thresholds)
        self.assertIsNone(staged)
        rows = [
            record("s0", False, [0.1] * 10, [0.1] * 10),
            record("f0", True, [0.1] * 6 + [0.9] * 4, [0.1] * 10),
        ]
        fused = select_staged_thresholds(rows, target_fpr=0.0, detector="fused_safe_time")
        self.assertEqual(fused["validation_metrics"]["true_positive_rate"], 1.0)
        with self.assertRaises(ValueError):
            select_staged_thresholds(rows, detector="safe_only")


class TestOfflineBaselineStatistics(unittest.TestCase):
    def make_records(self):
        rng = np.random.default_rng(3)
        records = []
        for task in ("TaskA", "TaskB", "TaskC"):
            for index in range(6):
                failed = index % 2 == 1
                length = 10 if failed else int(rng.integers(5, 10))
                records.append(
                    record(
                        f"{task}-{index}",
                        failed,
                        rng.uniform(0, 1, size=length) + (0.3 if failed else 0.0),
                        np.linspace(0.0, 1.0, length) * (1.0 if failed else 0.8),
                        task=task,
                    )
                )
        return records

    def test_fast_bootstrap_and_paired_comparison_reproduce_frozen_code(self):
        from robocasa.recovery.safe.run_offline_detector_baselines import (
            detector_outcomes,
            paired_comparison,
            task_rollout_bootstrap,
        )
        from robocasa.recovery.safe.run_prospective_matched_fpr import _bootstrap

        records = self.make_records()
        thresholds = {
            "staged_safe_time": {"early_safe": 1.0, "late_time": 0.85},
            "time_only": {"time_only": 0.9},
        }
        specs = {
            name: {"name": name, "base": name, "score": "safe", "family": "frozen"}
            for name in thresholds
        }
        outcomes = {
            name: detector_outcomes(records, specs[name], thresholds[name], 0.25, 0.50)
            for name in thresholds
        }
        frozen = _bootstrap(records, thresholds, 0.25, 0.50, 40, 7)
        fast = task_rollout_bootstrap(
            records, outcomes, (("staged_safe_time", "time_only"),), 40, 7
        )[("staged_safe_time", "time_only")]
        self.assertEqual(frozen, fast)
        frozen_paired = paired_failure_comparison(
            records,
            "staged_safe_time",
            thresholds["staged_safe_time"],
            "time_only",
            thresholds["time_only"],
        )
        self.assertEqual(
            frozen_paired,
            paired_comparison(records, outcomes["staged_safe_time"], outcomes["time_only"]),
        )


class TestOfflineBaselineRunner(unittest.TestCase):
    def test_synthetic_pipeline_reproduces_frozen_results_and_rejects_tampering(self):
        from robocasa.recovery.safe.run_offline_detector_baselines import (
            calibrate,
            develop,
            evaluate,
            write_synthetic_fixture,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = write_synthetic_fixture(root / "fixture", seed=0)
            development = develop(
                argparse.Namespace(
                    development_cache=str(fixture["caches"]["development"]),
                    runtime_bundle=str(fixture["runtime_bundle"]),
                    split_manifest=None,
                    output_dir=str(root / "develop"),
                    baselines=["mahalanobis", "action_inconsistency"],
                    aggregations=["running_max", "running_mean"],
                    grid="smoke",
                    selection_landmarks=[0.10, 0.25],
                    validation_fraction=0.3,
                    meta_split_seed=0,
                    model_seed=0,
                    max_fit_rows=50000,
                )
            )
            bundle = json.loads(Path(development["baseline_bundle"]).read_text())
            self.assertEqual(set(bundle["selected"]), {"mahalanobis", "action_inconsistency"})
            fit_ids = set(bundle["fit_rollout_ids"])
            validation_ids = set(bundle["validation_rollout_ids"])
            split = json.loads(Path(bundle["split_manifest"]).read_text())
            self.assertFalse(fit_ids & validation_ids)
            self.assertEqual(fit_ids | validation_ids, set(split["train"]))
            self.assertFalse((fit_ids | validation_ids) & set(split["test"]))

            calibration = calibrate(
                argparse.Namespace(
                    baseline_bundle=development["baseline_bundle"],
                    calibration_cache=str(fixture["caches"]["calibration"]),
                    calibration_predictions=None,
                    output_dir=str(root / "calibrate"),
                    workers=1,
                )
            )
            self.assertTrue(calibration["frozen_thresholds_reproduced"])
            for detector in calibration["detectors"].values():
                self.assertLessEqual(detector["false_positive_rate"], 0.05)

            def evaluation(output, frozen_analysis):
                return evaluate(
                    argparse.Namespace(
                        baseline_calibration=calibration["baseline_calibration"],
                        test_cache=str(fixture["caches"]["prospective_test"]),
                        test_predictions=str(
                            Path(fixture["frozen_analysis"]).parent / "event_predictions.jsonl"
                        ),
                        frozen_analysis=str(frozen_analysis),
                        output_dir=str(root / output),
                        test_per_class=fixture["test_per_class"],
                        bootstrap_samples=fixture["bootstrap_samples"],
                        bootstrap_seed=0,
                    )
                )

            result = evaluation("evaluate", fixture["frozen_analysis"])
            self.assertTrue(result["frozen_results_reproduced"])
            self.assertFalse(result["thresholds_updated_on_test"])
            self.assertEqual(
                set(result["overall"]),
                {
                    "time_only",
                    "safe_only",
                    "staged_safe_time",
                    "fused_safe_time",
                    "mahalanobis_only",
                    "staged_mahalanobis_time",
                    "action_inconsistency_only",
                    "staged_action_inconsistency_time",
                },
            )
            self.assertIn("staged_mahalanobis_time__vs__staged_safe_time", result["paired"])
            self.assertIn("composite", result["by_task_type"])
            self.assertTrue((root / "evaluate" / "detector_summary.csv").is_file())

            frozen = json.loads(Path(fixture["frozen_analysis"]).read_text())
            frozen["overall"]["staged_safe_time"]["true_positive_rate"] += 0.01
            tampered = root / "tampered_analysis.json"
            tampered.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, "not reproduced"):
                evaluation("evaluate_tampered", tampered)

            model_path = Path(bundle["selected"]["mahalanobis"]["model_path"])
            model_path.write_bytes(model_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluation("evaluate_tampered_model", fixture["frozen_analysis"])

    def test_fused_extension_reproduces_reference_runs_and_rejects_drift(self):
        from robocasa.recovery.safe.run_offline_detector_baselines import (
            calibrate,
            develop,
            evaluate,
            write_synthetic_fixture,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = write_synthetic_fixture(root / "fixture", seed=2)
            development = develop(
                argparse.Namespace(
                    development_cache=str(fixture["caches"]["development"]),
                    runtime_bundle=str(fixture["runtime_bundle"]),
                    split_manifest=None,
                    output_dir=str(root / "develop"),
                    baselines=["mahalanobis"],
                    aggregations=["running_max"],
                    grid="smoke",
                    selection_landmarks=[0.10, 0.25],
                    validation_fraction=0.3,
                    meta_split_seed=0,
                    model_seed=0,
                    max_fit_rows=50000,
                )
            )

            def calibration(output, **extra):
                return calibrate(
                    argparse.Namespace(
                        baseline_bundle=development["baseline_bundle"],
                        calibration_cache=str(fixture["caches"]["calibration"]),
                        calibration_predictions=None,
                        output_dir=str(root / output),
                        workers=1,
                        **extra,
                    )
                )

            def evaluation(output, calibration_path, **extra):
                return evaluate(
                    argparse.Namespace(
                        baseline_calibration=calibration_path,
                        test_cache=str(fixture["caches"]["prospective_test"]),
                        test_predictions=None,
                        frozen_analysis=str(fixture["frozen_analysis"]),
                        output_dir=str(root / output),
                        test_per_class=fixture["test_per_class"],
                        bootstrap_samples=fixture["bootstrap_samples"],
                        bootstrap_seed=0,
                        **extra,
                    )
                )

            base = calibration("calibrate")
            self.assertNotIn("fused_mahalanobis_time", base["detectors"])
            evaluation("evaluate", base["baseline_calibration"])
            reference_analysis = root / "evaluate" / "analysis.json"

            fused = calibration(
                "calibrate_fused",
                fused_baselines=True,
                reference_calibration=base["baseline_calibration"],
            )
            self.assertIn("fused_mahalanobis_time", fused["detectors"])
            self.assertEqual(
                set(fused["reference_detectors_reproduced"]), set(base["detectors"])
            )
            note = root / "interpretation_plan_fused.md"
            with self.assertRaisesRegex(ValueError, "Interpretation note"):
                evaluation(
                    "evaluate_no_note",
                    fused["baseline_calibration"],
                    reference_evaluation=str(reference_analysis),
                    interpretation_note=str(note),
                )
            note.write_text("Decision rule declared before evaluation.\n")
            result = evaluation(
                "evaluate_fused",
                fused["baseline_calibration"],
                reference_evaluation=str(reference_analysis),
                interpretation_note=str(note),
            )
            self.assertEqual(
                set(result["reference_detectors_reproduced"]), set(base["detectors"])
            )
            self.assertIn(
                "fused_mahalanobis_time__vs__fused_safe_time",
                result["task_rollout_bootstrap"],
            )
            self.assertIn("fused_mahalanobis_time__vs__time_only", result["paired"])
            self.assertIsNotNone(result["interpretation_note_sha256"])

            drifted = json.loads(Path(base["baseline_calibration"]).read_text())
            drifted["detectors"]["staged_safe_time"]["thresholds"]["early_safe"] += 1e-6
            drifted_path = root / "drifted_calibration.json"
            drifted_path.write_text(json.dumps(drifted))
            with self.assertRaisesRegex(ValueError, "Reference calibration was not reproduced"):
                calibration(
                    "calibrate_drifted",
                    fused_baselines=True,
                    reference_calibration=str(drifted_path),
                )

            drifted = json.loads(reference_analysis.read_text())
            drifted["overall"]["mahalanobis_only"]["false_positive_rate"] += 0.01
            drifted_path = root / "drifted_analysis.json"
            drifted_path.write_text(json.dumps(drifted))
            with self.assertRaisesRegex(ValueError, "Reference evaluation was not reproduced"):
                evaluation(
                    "evaluate_drifted",
                    fused["baseline_calibration"],
                    reference_evaluation=str(drifted_path),
                    interpretation_note=str(note),
                )

    def test_action_baseline_is_skipped_without_chunk_overlap(self):
        from robocasa.recovery.safe.run_offline_detector_baselines import (
            develop,
            write_synthetic_fixture,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = write_synthetic_fixture(root / "fixture", seed=1)
            provenance_path = fixture["caches"]["development"] / "provenance.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["pred_horizon"] = provenance["exec_horizon"]
            provenance_path.write_text(json.dumps(provenance))
            status = develop(
                argparse.Namespace(
                    development_cache=str(fixture["caches"]["development"]),
                    runtime_bundle=str(fixture["runtime_bundle"]),
                    split_manifest=None,
                    output_dir=str(root / "develop"),
                    baselines=["action_inconsistency", "knn_cosine"],
                    aggregations=["running_max"],
                    grid="smoke",
                    selection_landmarks=[0.10, 0.25],
                    validation_fraction=0.3,
                    meta_split_seed=0,
                    model_seed=0,
                    max_fit_rows=50000,
                )
            )
            self.assertIn("action_inconsistency", status["skipped"])
            self.assertEqual(set(status["selected"]), {"knn_cosine"})

    def test_extract_uses_official_loader_reduction_and_action_vectors(self):
        import pickle
        import sys
        import types
        from unittest import mock

        from robocasa.recovery.safe import run_offline_detector_baselines as runner
        from robocasa.recovery.safe import train_seen_tasks

        def namespace(**values):
            return types.SimpleNamespace(**values)

        def load_config(path):
            return namespace(
                dataset=namespace(horizon_idx_rel=1.0, diff_idx_rel=1.0),
                train=namespace(),
            )

        def load_rollouts(export_dir, cfg):
            cfg.dataset.pred_horizon = 4
            cfg.dataset.dim_action = 2
            cfg.dataset.exec_horizon = 2
            return [
                namespace(
                    hidden_states=np.arange(15, dtype=np.float32).reshape(3, 5),
                    action_vectors=np.arange(24, dtype=np.float32).reshape(3, 8),
                    task_id=0,
                    episode_success=success,
                )
                for success in (1, 0)
            ]

        modules = {
            "omegaconf": namespace(OmegaConf=namespace(load=load_config)),
            "failure_prob": namespace(),
            "failure_prob.data": namespace(),
            "failure_prob.data.pizero": namespace(load_rollouts_from_root=load_rollouts),
            "failure_prob.utils": namespace(),
            "failure_prob.utils.random": namespace(seed_everything=lambda seed: None),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            (export / "env_records").mkdir(parents=True)
            (export / "conversion_report.json").write_text(
                json.dumps({"task_ids": {"TaskA": 0}, "task_types": {"TaskA": "atomic"}})
            )
            for index, success in enumerate((1, 0)):
                with (export / "env_records" / f"rollout_{index:08d}--r{index}.pkl").open("wb") as stream:
                    pickle.dump(
                        {
                            "rollout_id": f"r{index}",
                            "task_id": 0,
                            "episode_success": success,
                            "replan_steps": 2,
                            "environment_seed": 7,
                            "environment_reset_index": index,
                            "robocasa_manifest_record": {
                                "inference_environment_steps": [0, 2, 4]
                            },
                        },
                        stream,
                    )
            config = root / "config.yaml"
            config.write_text("frozen\n")
            args = argparse.Namespace(
                export_dir=str(export),
                safe_repo=str(root),
                safe_config=str(config),
                group="calibration",
                output_dir=str(root / "cache"),
                expected_horizon_selector="1.0",
                expected_diffusion_selector="1.0",
            )
            with mock.patch.dict(sys.modules, modules), mock.patch.object(
                train_seen_tasks, "verify_safe_repo", lambda path: Path(path)
            ):
                provenance = runner.extract(args)
                cache = runner.FeatureCache(root / "cache", expected_group="calibration")
                self.assertEqual(provenance["stac_overlap_positions"], 2)
                self.assertTrue(provenance["stac_supported"])
                self.assertEqual(provenance["irregular_inference_step_gaps"], 0)
                self.assertEqual(cache.features_for(cache.by_id["r0"]).shape, (3, 5))
                actions = cache.actions_for(cache.by_id["r1"])
                self.assertEqual(actions.shape, (3, 4, 2))
                np.testing.assert_array_equal(actions[1, 0], [8.0, 9.0])
                self.assertTrue(cache.by_id["r1"]["failed"])
                mismatch = argparse.Namespace(
                    **{**vars(args), "output_dir": str(root / "other"), "expected_diffusion_selector": "0.0"}
                )
                with self.assertRaisesRegex(ValueError, "selectors"):
                    runner.extract(mismatch)

    def test_cli_defaults_match_frozen_protocol(self):
        from robocasa.recovery.safe.run_offline_detector_baselines import build_parser

        parser = build_parser()
        evaluation = parser.parse_args(
            [
                "evaluate",
                "--baseline-calibration",
                "/c",
                "--test-cache",
                "/t",
                "--frozen-analysis",
                "/a",
                "--output-dir",
                "/o",
            ]
        )
        self.assertEqual(evaluation.test_per_class, 10)
        self.assertEqual(evaluation.bootstrap_samples, 2000)
        self.assertEqual(evaluation.bootstrap_seed, 0)
        development = parser.parse_args(
            ["develop", "--development-cache", "/d", "--runtime-bundle", "/r", "--output-dir", "/o"]
        )
        self.assertEqual(development.selection_landmarks, [0.10, 0.25])
        self.assertEqual(development.grid, "full")


if __name__ == "__main__":
    unittest.main()
