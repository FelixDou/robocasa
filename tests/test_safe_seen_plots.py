import json
from pathlib import Path
import tempfile
import unittest


class TestSeenResultPlots(unittest.TestCase):
    def test_creates_static_figures_and_source_tables(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is unavailable")

        from robocasa.recovery.safe.plot_seen_results import create_seen_result_plots

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root = root / "final_refits"
            output_dir = root / "plots"
            for model, base in (("indep", 0.58), ("lstm", 0.61)):
                for seed in range(3):
                    run = final_root / f"{model}_seed{seed}"
                    run.mkdir(parents=True)
                    metrics = {
                        "model": model,
                        "seed": seed,
                        "selected_hyperparameters": {
                            "horizon_selector": 1.0,
                            "diffusion_selector": "concat-2",
                            "learning_rate": 1e-4,
                            "lambda_reg": 1e-2,
                        },
                        "counts": {"test": 30},
                        "duration_only_test_roc_auc": 0.55,
                        "scalar_metrics": {
                            "falert_early_roc_auc/model_test": base + 0.01 * seed,
                            "falert_early_prc_auc/model_test": base + 0.02 + 0.01 * seed,
                        },
                    }
                    (run / "metrics.json").write_text(json.dumps(metrics))
                    scores = []
                    for failed in (False, True):
                        scores.append(
                            {
                                "split": "test",
                                "failed": failed,
                                "scores": [0.1, 0.2, 0.8 if failed else 0.3],
                            }
                        )
                    (run / "scores.jsonl").write_text(
                        "".join(json.dumps(record) + "\n" for record in scores)
                    )
            cv_summary = root / "cv_selection_summary.json"
            cv_summary.write_text(
                json.dumps(
                    {
                        "outer_test_used_for_selection": False,
                        "best_by_model": {
                            model: {"inner_val_mean": mean, "inner_val_std": 0.02}
                            for model, mean in (("indep", 0.70), ("lstm", 0.80))
                        },
                    }
                )
            )
            outputs = create_seen_result_plots(
                final_root,
                cv_summary,
                output_dir,
                formats=("png",),
            )
            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            self.assertTrue((output_dir / "per_seed_metrics.csv").is_file())
            self.assertTrue((output_dir / "summary_metrics.csv").is_file())
            manifest = json.loads((output_dir / "plot_manifest.json").read_text())
            self.assertEqual(manifest["num_final_runs"], 6)
            self.assertEqual(manifest["num_test_rollouts_per_run"], 30)


if __name__ == "__main__":
    unittest.main()
