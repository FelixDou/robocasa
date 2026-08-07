import json
from pathlib import Path
import pickle
import tempfile
import unittest

from robocasa.recovery.safe.build_task_specific_rate_experiment import (
    build_task_specific_experiment,
)
from robocasa.recovery.safe.summarize_task_specific_rate_screen import summarize
from robocasa.recovery.safe.train_seen_tasks import resolve_task_type_selection


class TestTaskSpecificRateExperiment(unittest.TestCase):
    def make_source_experiment(self, root):
        export = root / "official"
        env_dir = export / "env_records"
        env_dir.mkdir(parents=True)
        task_types = {"TaskA": "atomic", "TaskB": "composite"}
        (export / "conversion_report.json").write_text(
            json.dumps(
                {
                    "task_ids": {"TaskA": 0, "TaskB": 1},
                    "task_types": task_types,
                }
            )
        )
        pools = {}
        file_index = 0
        for task_id, task_name in enumerate(task_types):
            pools[task_name] = {0: [], 1: []}
            for success in (0, 1):
                for index in range(5):
                    rollout_id = f"{task_name}-{success}-{index}"
                    record = {
                        "rollout_id": rollout_id,
                        "task_id": task_id,
                        "task_name": task_name,
                        "episode_success": success,
                    }
                    with (env_dir / f"{file_index:03d}.pkl").open("wb") as stream:
                        pickle.dump(record, stream)
                    file_index += 1
                    pools[task_name][success].append(rollout_id)
        plan_root = root / "source_plan"
        manifests = []
        for subset_seed in (0, 1):
            for selection in ("matched_balanced", "natural_rate"):
                train = []
                test = []
                for task_name in task_types:
                    test.extend(
                        [pools[task_name][0][-1], pools[task_name][1][-1]]
                    )
                    if selection == "matched_balanced":
                        train.extend(pools[task_name][0][:2])
                        train.extend(pools[task_name][1][:2])
                    elif task_name == "TaskA":
                        train.extend(pools[task_name][0][:1])
                        train.extend(pools[task_name][1][:3])
                    else:
                        train.extend(pools[task_name][0][:3])
                        train.extend(pools[task_name][1][:1])
                path = plan_root / selection / f"seed_{subset_seed}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "split_seed": 0,
                            "train": sorted(train),
                            "test": sorted(test),
                        }
                    )
                )
                manifests.append(
                    {
                        "regime": selection,
                        "subset_seed": subset_seed,
                        "path": str(path),
                    }
                )
        source_plan = {
            "official_export": str(export),
            "subset_seeds": [0, 1],
            "regimes": {
                "matched_weighted": {
                    "selection": "matched_balanced",
                    "class_weighting": "official_inverse_frequency",
                },
                "natural_weighted": {
                    "selection": "natural_rate",
                    "class_weighting": "official_inverse_frequency",
                },
            },
            "manifests": manifests,
        }
        plan_path = plan_root / "experiment_plan.json"
        plan_path.write_text(json.dumps(source_plan))
        return export, plan_path

    def test_builder_freezes_test_and_holds_training_n_constant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export, plan_path = self.make_source_experiment(root)
            result = build_task_specific_experiment(
                experiment_plan=plan_path,
                official_export=export,
                output_dir=root / "task_specific",
            )
            self.assertEqual(result["tasks"], ["TaskA", "TaskB"])
            self.assertEqual(result["num_task_regime_comparisons"], 4)
            self.assertEqual(len(result["manifests"]), 8)
            for task_name in result["tasks"]:
                records = [
                    item for item in result["manifests"] if item["task_name"] == task_name
                ]
                test_sets = {
                    tuple(json.loads(Path(item["path"]).read_text())["test"])
                    for item in records
                }
                self.assertEqual(len(test_sets), 1)
                self.assertEqual(
                    {item["counts"]["train"]["rollouts"] for item in records},
                    {4},
                )
                balanced = [
                    item for item in records if item["regime"] == "matched_weighted"
                ]
                self.assertTrue(
                    all(item["counts"]["train"]["successes"] == 2 for item in balanced)
                )
                natural = [
                    item for item in records if item["regime"] == "natural_weighted"
                ]
                expected_successes = 3 if task_name == "TaskA" else 1
                self.assertTrue(
                    all(
                        item["counts"]["train"]["successes"] == expected_successes
                        for item in natural
                    )
                )

    def test_exact_task_selection_is_validated_within_task_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export, _ = self.make_source_experiment(root)
            selection = resolve_task_type_selection(export, "all", ["TaskB"])
            self.assertEqual(selection["selected_task_names"], ["TaskB"])
            self.assertEqual(selection["selected_task_ids"], [1])
            self.assertEqual(selection["task_name_filter"], ["TaskB"])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                resolve_task_type_selection(export, "atomic", ["TaskB"])
            with self.assertRaisesRegex(ValueError, "Unknown requested"):
                resolve_task_type_selection(export, "all", ["MissingTask"])

    def test_summary_uses_subset_means_as_resampling_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task_index, task_name in enumerate(("TaskA", "TaskB")):
                for regime_index, regime in enumerate(
                    ("matched_weighted", "natural_weighted")
                ):
                    for subset_seed in (0, 1):
                        manifest = root / "manifests" / (
                            f"{task_name}-{regime}-{subset_seed}.json"
                        )
                        manifest.parent.mkdir(parents=True, exist_ok=True)
                        manifest.write_text(
                            json.dumps(
                                {
                                    "test": [f"{task_name}-test-f", f"{task_name}-test-s"],
                                    "counts": {
                                        "train": {
                                            "rollouts": 4,
                                            "successes": 2 + regime_index,
                                            "failures": 2 - regime_index,
                                        },
                                        "test": {
                                            "rollouts": 2,
                                            "successes": 1,
                                            "failures": 1,
                                        },
                                    },
                                }
                            )
                        )
                        for model_seed in (0, 1):
                            run = root / task_name / (
                                f"indep__{regime}__subset-{subset_seed}"
                                f"__seed-{model_seed}"
                            )
                            run.mkdir(parents=True)
                            (run / "task_screen_run.json").write_text(
                                json.dumps(
                                    {
                                        "task_name": task_name,
                                        "task_type": "atomic" if task_index == 0 else "composite",
                                        "model": "indep",
                                        "regime": regime,
                                        "subset_seed": subset_seed,
                                        "model_seed": model_seed,
                                        "selection_manifest": str(manifest),
                                    }
                                )
                            )
                            value = (
                                0.60
                                + 0.10 * regime_index
                                + 0.02 * task_index
                                + 0.01 * subset_seed
                                + 0.002 * model_seed
                            )
                            (run / "metrics.json").write_text(
                                json.dumps(
                                    {
                                        "scalar_metrics": {
                                            "falert_early_roc_auc/model_test": value,
                                            "falert_early_prc_auc/model_test": value - 0.05,
                                        }
                                    }
                                )
                            )
            result = summarize(
                root,
                expected_subset_seeds=(0, 1),
                expected_model_seeds=(0, 1),
                formats=(),
            )
            self.assertEqual(result["num_completed_runs"], 16)
            effect = result["paired_comparisons"]["indep"][
                "natural_minus_balanced_roc_auc"
            ]
            self.assertAlmostEqual(effect["mean"], 0.1)
            self.assertTrue((root / "task_specific_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
