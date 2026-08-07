from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.online_safe import (
    OnlineSafeConfig,
    prepare_online_splits,
)


def rollout(task, success, length, rollout_id):
    item = SimpleNamespace(
        task_id=task,
        task_description=f"Task{task}",
        episode_success=int(success),
        hidden_states=np.arange(length * 2, dtype=np.float32).reshape(length, 2),
        action_vectors=np.arange(length * 3, dtype=np.float32).reshape(length, 3),
    )
    env = {
        "rollout_id": rollout_id,
        "task_id": task,
        "episode_success": int(success),
        "model_infer_times": length,
        "inference_environment_steps": list(range(0, length * 16, 16)),
    }
    return item, (Path(f"{rollout_id}.pkl"), env)


def dataset(specs):
    values = []
    identity = {}
    for task, success, length, rollout_id in specs:
        item, aligned = rollout(task, success, length, rollout_id)
        values.append(item)
        identity[id(item)] = aligned
    return values, identity


class TestOnlineSafePrefixes(unittest.TestCase):
    def test_success_length_matching_removes_duration_signal_per_task(self):
        train, identity = dataset(
            [
                (0, True, 2, "t-s0"),
                (0, True, 4, "t-s1"),
                (0, False, 8, "t-f0"),
                (0, False, 8, "t-f1"),
                (1, True, 3, "u-s0"),
                (1, True, 5, "u-s1"),
                (1, False, 9, "u-f0"),
                (1, False, 9, "u-f1"),
            ]
        )
        test, test_identity = dataset(
            [
                (0, True, 3, "v-s0"),
                (0, False, 8, "v-f0"),
                (1, True, 4, "w-s0"),
                (1, False, 9, "w-f0"),
            ]
        )
        identity.update(test_identity)
        result = prepare_online_splits(
            train,
            test,
            identity,
            config=OnlineSafeConfig(mode="matched_success_length", seed=7),
        )

        self.assertEqual(len(result["train"]), len(train))
        self.assertEqual(len(result["test"]), len(test))
        for split in ("train", "test"):
            audit = result["duration_audit"][split]
            self.assertEqual(audit["0"]["duration_roc_auc"], 0.5)
            self.assertEqual(audit["1"]["duration_roc_auc"], 0.5)
        failures = [item for item in result["train"] if not item.episode_success]
        self.assertEqual(
            sorted(len(item.hidden_states) for item in failures), [2, 3, 4, 5]
        )
        for item in result["train"] + result["test"]:
            env = result["identity"][id(item)][1]
            self.assertEqual(env["model_infer_times"], len(item.hidden_states))
            self.assertEqual(
                len(env["inference_environment_steps"]), len(item.hidden_states)
            )

    def test_fixed_landmark_uses_training_failures_and_keeps_at_risk_only(self):
        train, identity = dataset(
            [
                (0, True, 2, "t-s-early"),
                (0, True, 6, "t-s-late"),
                (0, False, 10, "t-f0"),
                (0, False, 10, "t-f1"),
                (1, True, 5, "u-s0"),
                (1, False, 8, "u-f0"),
            ]
        )
        test, test_identity = dataset(
            [
                (0, True, 7, "v-s0"),
                (0, False, 10, "v-f0"),
                (1, True, 6, "w-s0"),
                (1, False, 8, "w-f0"),
            ]
        )
        identity.update(test_identity)
        result = prepare_online_splits(
            train,
            test,
            identity,
            config=OnlineSafeConfig(mode="fixed_landmark", landmark_fraction=0.5),
        )

        self.assertEqual(result["protocol"]["training_task_timeouts"], {0: 10, 1: 8})
        self.assertEqual(result["protocol"]["task_landmark_cutoffs"], {0: 5, 1: 4})
        self.assertEqual(result["counts"], {"train": 4, "test": 4})
        excluded = result["protocol"]["excluded_before_landmark"]["train"]
        self.assertEqual([item["rollout_id"] for item in excluded], ["t-s-early"])
        balance_excluded = result["protocol"]["excluded_for_training_balance"]
        self.assertEqual(len(balance_excluded), 1)
        self.assertEqual(balance_excluded[0]["episode_success"], 0)
        for item in result["train"] + result["test"]:
            cutoff = {0: 5, 1: 4}[item.task_id]
            self.assertEqual(len(item.hidden_states), cutoff)
            env = result["identity"][id(item)][1]
            self.assertEqual(env["online_safe_mode"], "fixed_landmark")

    def test_transform_rejects_parent_overlap_and_unsupported_landmark(self):
        values, identity = dataset(
            [
                (0, True, 2, "s"),
                (0, False, 8, "f"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare_online_splits(
                values,
                values,
                identity,
                config=OnlineSafeConfig(mode="matched_success_length"),
            )

        test, test_identity = dataset([(0, True, 2, "v-s"), (0, False, 8, "v-f")])
        identity.update(test_identity)
        with self.assertRaisesRegex(ValueError, "without both outcomes"):
            prepare_online_splits(
                values,
                test,
                identity,
                config=OnlineSafeConfig(mode="fixed_landmark", landmark_fraction=0.75),
            )

    def test_training_and_cv_clis_expose_both_online_modes(self):
        from robocasa.recovery.safe.run_seen_cv_grid import build_parser as cv_parser
        from robocasa.recovery.safe.train_seen_tasks import build_parser as train_parser

        common = [
            "--export-dir",
            "/data/xiaomi",
            "--safe-repo",
            "/src/SAFE",
        ]
        cv = cv_parser().parse_args(
            common
            + [
                "--output-root",
                "/results/cv",
                "--model",
                "indep",
                "--online-safe-mode",
                "matched_success_length",
                "--online-safe-seed",
                "7",
            ]
        )
        self.assertEqual(cv.online_safe_mode, "matched_success_length")
        self.assertEqual(cv.online_safe_seed, 7)

        final = train_parser().parse_args(
            common
            + [
                "--output-dir",
                "/results/final",
                "--model",
                "lstm",
                "--seed",
                "0",
                "--online-safe-mode",
                "fixed_landmark",
                "--online-landmark-fraction",
                "0.25",
            ]
        )
        self.assertEqual(final.online_safe_mode, "fixed_landmark")
        self.assertEqual(final.online_landmark_fraction, 0.25)

    def test_cv_summary_preserves_online_protocol(self):
        from robocasa.recovery.safe.summarize_seen_cv import summarize_seen_cv

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold in range(3):
                run = root / f"run-{fold}"
                run.mkdir()
                (run / "metrics.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "model": "indep",
                            "horizon_selector": "1.0",
                            "diffusion_selector": "1.0",
                            "learning_rate": 1e-4,
                            "lambda_reg": 1e-3,
                            "fold": fold,
                            "selection_metric": "online-prefix",
                            "selection_value": 0.6 + fold * 0.01,
                            "task_type_filter": "all",
                            "selected_task_names": ["TaskA"],
                            "online_safe": {
                                "protocol": {
                                    "mode": "matched_success_length",
                                    "seed": 0,
                                    "landmark_fraction": 0.5,
                                }
                            },
                        }
                    )
                )
            summary = summarize_seen_cv(root)
            self.assertEqual(
                summary["online_safe"],
                {
                    "mode": "matched_success_length",
                    "seed": 0,
                    "landmark_fraction": 0.5,
                },
            )
            self.assertFalse(summary["outer_test_used_for_selection"])

    def test_final_summary_rejects_mixed_online_protocols(self):
        from robocasa.recovery.safe.summarize_seen_tasks import summarize

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in ("indep", "lstm"):
                for seed in range(3):
                    run = root / f"{model}_seed{seed}"
                    run.mkdir()
                    mode = (
                        "fixed_landmark"
                        if model == "lstm" and seed == 2
                        else "matched_success_length"
                    )
                    (run / "metrics.json").write_text(
                        json.dumps(
                            {
                                "model": model,
                                "seed": seed,
                                "num_tasks": 1,
                                "task_type_filter": "all",
                                "task_types": {"TaskA": "atomic"},
                                "counts": {
                                    "train": 2,
                                    "test": 2,
                                    "train_successes": 1,
                                    "train_failures": 1,
                                    "test_successes": 1,
                                    "test_failures": 1,
                                },
                                "online_safe": {
                                    "protocol": {
                                        "mode": mode,
                                        "seed": 0,
                                        "landmark_fraction": 0.5,
                                    }
                                },
                                "scalar_metrics": {
                                    "falert_early_roc_auc/model_test": 0.6,
                                    "falert_early_prc_auc/model_test": 0.6,
                                },
                            }
                        )
                    )
            with self.assertRaisesRegex(ValueError, "mix incompatible online"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
