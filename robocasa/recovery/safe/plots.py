"""Create compact raw SAFE evaluation plots from score and result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .conformal import first_detection, load_calibration


def create_plots(scores_path, calibration_path, results_path, output_dir):
    scores = json.loads(Path(scores_path).read_text())
    calibration = load_calibration(calibration_path)
    results = json.loads(Path(results_path).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = [x for x in scores if x["split"] in ("seen_test", "unseen_test")]
    labels = np.asarray([int(x["failed"]) for x in evaluation])
    maxima = np.asarray([max(x["scores"]) for x in evaluation])
    if len(set(labels.tolist())) == 2:
        fpr, tpr, _ = roc_curve(labels, maxima)
        precision, recall, _ = precision_recall_curve(labels, maxima)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(fpr, tpr)
        axes[0].plot([0, 1], [0, 1], "--", color="0.6")
        axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
        axes[1].plot(recall, precision)
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_pr.png", dpi=160)
        plt.close(fig)
    representatives = []
    for failed in (False, True):
        representatives.extend([x for x in evaluation if x["failed"] == failed][:2])
    fig, ax = plt.subplots(figsize=(7, 4))
    for item in representatives:
        x = np.linspace(0, 1, len(item["scores"]))
        ax.plot(x, item["scores"], label=f"{'fail' if item['failed'] else 'success'}:{item['task_name']}")
    threshold = np.asarray(calibration["threshold"])
    ax.plot(np.linspace(0, 1, len(threshold)), threshold, "k--", label="conformal threshold")
    ax.set(xlabel="Normalized rollout time", ylabel="Failure score", title="Representative SAFE trajectories")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "score_trajectories_threshold.png", dpi=160)
    plt.close(fig)
    model_name = next(name for name in results["models"] if name not in ("constant", "time_only"))
    per_task = {}
    for split in ("seen_test", "unseen_test"):
        block = results["models"][model_name].get(split)
        if block:
            for task, metrics in block["per_task"].items():
                per_task[f"{split}:{task}"] = metrics["balanced_accuracy"]
    valid = {k: v for k, v in per_task.items() if v is not None}
    if valid:
        fig, ax = plt.subplots(figsize=(max(7, len(valid) * 0.7), 4))
        ax.bar(range(len(valid)), valid.values())
        ax.set_xticks(range(len(valid)), valid.keys(), rotation=45, ha="right")
        ax.set(ylabel="Balanced accuracy", ylim=(0, 1), title="Per-task seen/unseen comparison")
        fig.tight_layout()
        fig.savefig(output_dir / "per_task_comparison.png", dpi=160)
        plt.close(fig)
    detection_times = []
    for item in evaluation:
        if item["failed"]:
            index = first_detection(item["scores"], calibration)
            if index is not None:
                detection_times.append(index / max(1, len(item["scores"]) - 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    if detection_times:
        ax.hist(detection_times, bins=np.linspace(0, 1, 11))
    else:
        ax.text(0.5, 0.5, "No failed rollout crossed the threshold", ha="center", va="center")
    ax.set(
        xlabel="Normalized detection time",
        ylabel="Failed rollouts detected",
        title="Failure detection times",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "detection_times.png", dpi=160)
    plt.close(fig)
    return sorted(output_dir.glob("*.png"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    create_plots(args.scores, args.calibration, args.results, args.output_dir)


if __name__ == "__main__":
    main()
