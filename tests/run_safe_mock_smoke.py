"""Produce a tiny synthetic SAFE artifact tree; never use it as a scientific result."""

from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.conformal import calibrate_functional_threshold, save_calibration
from robocasa.recovery.safe.dataset import generate_splits, load_manifest, save_rollout, save_splits
from robocasa.recovery.safe.evaluate import evaluate_experiment, save_results
from robocasa.recovery.safe.plots import create_plots
from robocasa.recovery.safe.schema import SafeRolloutMetadata
from robocasa.recovery.safe.train import train_model


def main(output_dir="outputs/safe_mock_smoke"):
    root = Path(output_dir)
    dataset_dir = root / "dataset"
    rng = np.random.default_rng(0)
    for task in ("MockSeenA", "MockSeenB", "MockUnseen"):
        for seed in range(6):
            for failed in (False, True):
                length = 3 + seed % 3
                # Failure signal is deliberately simple so the smoke checks plumbing only.
                features = rng.normal(failed * 0.8, 0.1, (length, 2, 4, 8)).astype(np.float32)
                record = SafeRolloutMetadata(
                    rollout_id=f"{task}-{seed}-{'failure' if failed else 'success'}",
                    task_name=task,
                    task_instruction="synthetic smoke only",
                    environment_seed=seed,
                    policy_id="mock-pi0",
                    checkpoint="mock-checkpoint-not-real",
                    failed=failed,
                    num_env_steps=length * 2,
                    inference_env_steps=list(range(0, length * 2, 2)),
                    valid_sequence_length=length,
                    action_horizon=4,
                    replan_steps=2,
                    feature_shape=list(features.shape),
                    flow_steps=2,
                    termination_reason="mock_failure" if failed else "mock_success",
                    timeout_horizon=10,
                )
                save_rollout(dataset_dir, record, features)
    records = load_manifest(dataset_dir)
    splits = generate_splits(records, ["MockUnseen"], seed=0, train_fraction=0.5, calibration_fraction=0.25)
    splits_path = save_splits(splits, root / "splits.json")
    summary = {"synthetic": True, "warning": "Do not interpret as a scientific SAFE result", "models": {}}
    for model_name in ("mlp", "lstm"):
        model_dir = root / model_name
        args = Namespace(
            dataset_dir=str(dataset_dir),
            splits=str(splits_path),
            aggregation="mean",
            model=model_name,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            cumulative_mlp=False,
            seed=0,
            device="cpu",
            learning_rate=1e-3,
            weight_decay=0.0,
            epochs=3,
            batch_size=4,
            output_checkpoint=str(model_dir / "checkpoint.pt"),
            output_scores=str(model_dir / "scores.json"),
        )
        checkpoint, scores_path, history = train_model(args)
        scores = json.loads(scores_path.read_text())
        reference = [x["scores"] for x in scores if x["split"] == "train" and not x["failed"]]
        calibration_scores = [x["scores"] for x in scores if x["split"] == "calibration" and not x["failed"]]
        calibration = calibrate_functional_threshold(reference, calibration_scores, alpha=0.2, normalized_length=20)
        calibration_path = save_calibration(calibration, model_dir / "calibration_alpha_0.20.json")
        results = evaluate_experiment(scores, calibration, f"safe_{model_name}")
        results_path = save_results(results, model_dir / "results_alpha_0.20.json")
        plots = create_plots(scores_path, calibration_path, results_path, model_dir / "plots")
        summary["models"][model_name] = {
            "loss_history": history,
            "checkpoint": str(checkpoint),
            "scores": str(scores_path),
            "calibration": str(calibration_path),
            "results": str(results_path),
            "plots": [str(path) for path in plots],
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/safe_mock_smoke")
