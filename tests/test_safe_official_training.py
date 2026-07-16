import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.evaluate_official_safe import (
    _json_value,
    baseline_scores,
    duration_diagnostics,
    evaluate_functional_bands_by_task,
    validate_persisted_rows_against_official,
    write_json,
)
from robocasa.recovery.safe.run_official_grid import (
    GridRun,
    build_parser as build_grid_parser,
    evaluation_command,
    generate_grid,
    train_command,
    validate_export_gate,
)
from robocasa.recovery.safe.report_official_grid import (
    NUMERIC_CP_FIELDS,
    aggregate_conformal,
    build_report,
    config_matches,
)
from robocasa.recovery.safe.summarize_official_grid import summarize_grid


class TestOfficialSafeEvaluationHelpers(unittest.TestCase):
    def test_strict_json_conversion(self):
        value = _json_value({"finite": np.float32(1.25), "nan": np.float64(np.nan)})
        self.assertEqual(value, {"finite": 1.25, "nan": None})
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(Path(tmp) / "result.json", value)
            self.assertNotIn("NaN", path.read_text())

    def test_causal_time_and_constant_baselines(self):
        scores = {"val_seen": [np.ones(2), np.ones(4)]}
        constant = baseline_scores(scores, "constant")["val_seen"]
        time_only = baseline_scores(scores, "time_only")["val_seen"]
        np.testing.assert_array_equal(constant[0], [0.5, 0.5])
        np.testing.assert_array_equal(time_only[1], [0.0, 1.0, 2.0, 3.0])

    def test_duration_leakage_warning(self):
        rollouts = [
            SimpleNamespace(task_id=0, task_description="Task", episode_success=1, hidden_states=np.zeros((2, 1))),
            SimpleNamespace(task_id=0, task_description="Task", episode_success=1, hidden_states=np.zeros((3, 1))),
            SimpleNamespace(task_id=0, task_description="Task", episode_success=0, hidden_states=np.zeros((8, 1))),
            SimpleNamespace(task_id=0, task_description="Task", episode_success=0, hidden_states=np.zeros((9, 1))),
        ]
        diagnostics, warnings = duration_diagnostics({"val_unseen": rollouts}, {0: "Task"})
        self.assertEqual(diagnostics["val_unseen"]["overall"]["duration_only_roc_auc"], 1.0)
        self.assertTrue(warnings)

    def test_global_functional_band_is_reported_overall_and_per_task(self):
        rollouts = [
            SimpleNamespace(task_id=0, episode_success=1, task_min_step=2),
            SimpleNamespace(task_id=0, episode_success=0, task_min_step=2),
            SimpleNamespace(task_id=1, episode_success=1, task_min_step=2),
            SimpleNamespace(task_id=1, episode_success=0, task_min_step=2),
        ]
        scores = [
            np.array([0.1, 0.2]),
            np.array([0.1, 0.8]),
            np.array([0.2, 0.3]),
            np.array([0.9, 0.9]),
        ]
        rows = evaluate_functional_bands_by_task(
            rollouts,
            scores,
            {0.1: np.array([[0.5, 0.5]])},
            method="model",
            task_names={0: "TaskA", 1: "TaskB"},
        )
        self.assertEqual(len(rows), 6)
        overall = next(
            row for row in rows if row["task"] == "all" and row["time"] == "by final end"
        )
        self.assertEqual(overall["tp"], 2)
        self.assertEqual(overall["fp"], 0)
        validate_persisted_rows_against_official(
            rows,
            [
                {
                    key: overall[key]
                    for key in (
                        "time",
                        "alpha",
                        "avg_det_time",
                        "tpr",
                        "tnr",
                        "fpr",
                        "fnr",
                        "acc",
                        "bal_acc",
                        "f1",
                    )
                }
            ],
        )


class TestOfficialSafeGrid(unittest.TestCase):
    def test_exact_official_grid_has_810_unique_runs(self):
        grid = generate_grid()
        self.assertEqual(len(grid), 810)
        self.assertEqual(len({run.slug for run in grid}), 810)

    def test_commands_preserve_official_training_and_evaluation_contract(self):
        run = GridRun("indep", "concat-2", "1.0", "1e-4", "1e-2", 2)
        args = SimpleNamespace(
            export_dir="/data/export",
            safe_repo="/src/SAFE",
            robocasa_repo="/src/robocasa",
            epochs=1000,
            device="cuda",
        )
        root = Path("/results") / run.slug
        train = train_command(run, args, root)
        evaluate = evaluation_command(run, args, root)
        self.assertIn("model.n_epochs=1000", train)
        self.assertIn("dataset.horizon_idx_rel=concat-2", train)
        self.assertIn("train.eval_save_ckpt=true", train)
        self.assertIn(
            "/src/robocasa/robocasa/recovery/safe/evaluate_official_safe.py",
            evaluate,
        )
        self.assertIn(str(root / "artifacts" / "model_final.ckpt"), evaluate)

    def test_grid_requires_validated_balanced_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            export = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Missing official loader gate"):
                validate_export_gate(export)
            (export / "official_loader_validation.json").write_text(
                json.dumps(
                    {
                        "valid": True,
                        "official_safe_loader_compatible": True,
                        "num_rollouts": 100,
                        "successes": 50,
                        "failures": 50,
                    }
                )
            )
            self.assertTrue(validate_export_gate(export).is_file())

    def test_grid_continues_errors_by_default_and_can_retry_or_fail_fast(self):
        base = [
            "--export-dir", "/data/export",
            "--safe-repo", "/src/SAFE",
            "--robocasa-repo", "/src/robocasa",
            "--output-root", "/results",
        ]
        default = build_grid_parser().parse_args(base)
        self.assertFalse(default.fail_fast)
        self.assertFalse(default.retry_errors)
        explicit = build_grid_parser().parse_args(base + ["--fail-fast", "--retry-errors"])
        self.assertTrue(explicit.fail_fast)
        self.assertTrue(explicit.retry_errors)

    def test_selection_uses_mean_val_seen_across_complete_seed_set(self):
        results = []
        for seed in (0, 1, 2):
            results.append(
                {
                    "model": "indep",
                    "horizon_selector": "0.0",
                    "diffusion_selector": "0.0",
                    "learning_rate": 1e-4,
                    "lambda_reg": 1e-2,
                    "seed": seed,
                    "train": 0.9,
                    "val_seen": 0.7 + 0.01 * seed,
                    "val_unseen": 0.6,
                }
            )
            results.append(
                {
                    "model": "indep",
                    "horizon_selector": "1.0",
                    "diffusion_selector": "1.0",
                    "learning_rate": 3e-4,
                    "lambda_reg": 1e-1,
                    "seed": seed,
                    "train": 0.8,
                    "val_seen": 0.8 + 0.01 * seed,
                    "val_unseen": 0.4,
                }
            )
        summary = summarize_grid(results)
        best = summary["best_by_model"]["indep"]
        self.assertEqual(best["horizon_selector"], "1.0")
        self.assertEqual(best["num_runs"], 3)
        self.assertTrue(best["complete_seed_set"])

    def test_final_report_matches_selected_config_and_aggregates_seeds(self):
        provenance = {
            "model": "lstm",
            "dataset_config": {"horizon_idx_rel": 1.0, "diff_idx_rel": "concat-2"},
            "model_config": {"lr": 1e-3, "lambda_reg": 1e-2},
        }
        selected = {
            "model": "lstm",
            "horizon_selector": "1.0",
            "diffusion_selector": "concat-2",
            "learning_rate": 1e-3,
            "lambda_reg": 1e-2,
        }
        self.assertTrue(config_matches(provenance, selected))
        records = []
        for seed, tpr in ((0, 0.6), (1, 0.8), (2, 1.0)):
            record = {
                "model_architecture": "lstm",
                "detect_method": "model",
                "task": "all",
                "time": "by earliest stop",
                "alpha": 0.1,
                "seed": seed,
            }
            for field in (
                "avg_det_time", "tpr", "tnr", "fpr", "fnr", "acc",
                "bal_acc", "f1", "false_alarms_per_successful_rollout",
                "failed_rollouts_detected_fraction",
            ):
                record[field] = tpr
            records.append(record)
        row = aggregate_conformal(records)[0]
        self.assertEqual(row["seeds"], [0, 1, 2])
        self.assertAlmostEqual(row["tpr_mean"], 0.8)
        self.assertEqual(row["num_seed_task_evaluations"], 3)

    def test_final_report_writes_machine_readable_results_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "grid"
            output = Path(tmp) / "report"
            root.mkdir()
            best = {}
            for model in ("indep", "lstm"):
                best[model] = {
                    "model": model,
                    "horizon_selector": "1.0",
                    "diffusion_selector": "0.0",
                    "learning_rate": 1e-4,
                    "lambda_reg": 1e-2,
                    "seeds": [0, 1, 2],
                    "complete_seed_set": True,
                    "num_runs": 3,
                    "train_mean": 0.9,
                    "train_std": 0.01,
                    "val_seen_mean": 0.8,
                    "val_seen_std": 0.02,
                    "val_unseen_mean": 0.6,
                    "val_unseen_std": 0.03,
                }
                for seed in (0, 1, 2):
                    run = root / f"{model}-seed-{seed}"
                    evaluation = run / "evaluation"
                    evaluation.mkdir(parents=True)
                    provenance = {
                        "model": model,
                        "seed": seed,
                        "dataset_config": {
                            "horizon_idx_rel": 1.0,
                            "diff_idx_rel": 0.0,
                        },
                        "model_config": {"lr": 1e-4, "lambda_reg": 1e-2},
                    }
                    (evaluation / "provenance.json").write_text(json.dumps(provenance))
                    metrics = {
                        "scalar_metrics": {
                            "falert_early_roc_auc/model_val_seen": 0.8,
                            "falert_early_roc_auc/model_val_unseen": 0.6,
                        },
                        "duration_diagnostics": {"val_unseen": {"overall": {"duration_only_roc_auc": 1.0}}},
                        "rollout_length_warnings": ["duration warning"],
                    }
                    (evaluation / "metrics.json").write_text(json.dumps(metrics))
                    fields = [
                        "detect_method", "task", "time", "alpha",
                        *NUMERIC_CP_FIELDS,
                    ]
                    with (evaluation / "functional_conformal.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        for method in ("model", "constant", "time_only"):
                            for task in ("all", "TaskA"):
                                for time in ("by final end", "by earliest stop"):
                                    writer.writerow(
                                        {
                                            "detect_method": method,
                                            "task": task,
                                            "time": time,
                                            "alpha": 0.1,
                                            **{field: 0.5 for field in NUMERIC_CP_FIELDS},
                                        }
                                    )
                    score_rows = [
                        {
                            "split": "val_unseen",
                            "failed": False,
                            "task_name": "TaskA",
                            "task_min_step": 2,
                            "scores": [0.1, 0.2],
                        },
                        {
                            "split": "val_unseen",
                            "failed": True,
                            "task_name": "TaskA",
                            "task_min_step": 2,
                            "scores": [0.2, 0.8],
                        },
                    ]
                    (evaluation / "scores.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in score_rows)
                    )
                    np.savez_compressed(
                        evaluation / "functional_bands.npz",
                        **{"model_alpha_0.10": np.array([0.5, 0.5])},
                    )
            summary = {
                "num_completed_runs": 810,
                "num_configurations": 270,
                "selection_rule": "test",
                "selection_metric": "test",
                "best_by_model": best,
            }
            (root / "selection_summary.json").write_text(json.dumps(summary))
            report = build_report(root, output)
            self.assertEqual(report["official_grid"]["completed_runs"], 810)
            self.assertTrue((output / "final_report.json").is_file())
            self.assertTrue((output / "selected_conformal_summary.csv").is_file())
            self.assertGreaterEqual(len(list(output.glob("*.png"))), 6)


if __name__ == "__main__":
    unittest.main()
