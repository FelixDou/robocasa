"""Build the final compact report for validation-selected official SAFE runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


PRIMARY_ALPHAS = (0.05, 0.10, 0.15, 0.20)
NUMERIC_CP_FIELDS = (
    "avg_det_time",
    "tpr",
    "tnr",
    "fpr",
    "fnr",
    "acc",
    "bal_acc",
    "f1",
    "false_alarms_per_successful_rollout",
    "failed_rollouts_detected_fraction",
)


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def config_matches(provenance, selected):
    model = provenance["model"]
    dataset = provenance["dataset_config"]
    model_cfg = provenance["model_config"]
    return (
        model == selected["model"]
        and str(dataset["horizon_idx_rel"]) == str(selected["horizon_selector"])
        and str(dataset["diff_idx_rel"]) == str(selected["diffusion_selector"])
        and np.isclose(float(model_cfg["lr"]), float(selected["learning_rate"]))
        and np.isclose(float(model_cfg["lambda_reg"]), float(selected["lambda_reg"]))
    )


def find_selected_runs(grid_root, summary, expected_seeds=(0, 1, 2)):
    selected_runs = []
    for provenance_path in sorted(Path(grid_root).glob("*/evaluation/provenance.json")):
        provenance = json.loads(provenance_path.read_text())
        selected = summary["best_by_model"].get(provenance["model"])
        if selected and config_matches(provenance, selected):
            selected_runs.append(
                {
                    "model": provenance["model"],
                    "seed": int(provenance["seed"]),
                    "run_dir": str(provenance_path.parents[1]),
                    "provenance": provenance,
                }
            )
    expected_seeds = set(expected_seeds)
    for model in summary["best_by_model"]:
        runs = [run for run in selected_runs if run["model"] == model]
        seeds = {run["seed"] for run in runs}
        if seeds != expected_seeds:
            raise ValueError(
                f"Selected {model} runs have seeds {sorted(seeds)}, "
                f"expected {sorted(expected_seeds)}"
            )
    return sorted(selected_runs, key=lambda run: (run["model"], run["seed"]))


def load_selected_artifacts(selected_runs):
    scalar_records = []
    conformal_records = []
    duration_records = []
    for run in selected_runs:
        evaluation = Path(run["run_dir"]) / "evaluation"
        metrics = json.loads((evaluation / "metrics.json").read_text())
        scalar_records.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "metrics": metrics["scalar_metrics"],
            }
        )
        duration_records.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "diagnostics": metrics["duration_diagnostics"],
                "warnings": metrics["rollout_length_warnings"],
            }
        )
        with (evaluation / "functional_conformal.csv").open() as stream:
            for row in csv.DictReader(stream):
                converted = {
                    **row,
                    "model_architecture": run["model"],
                    "seed": run["seed"],
                    "alpha": float(row["alpha"]),
                }
                for field in NUMERIC_CP_FIELDS:
                    converted[field] = _finite_or_none(row.get(field))
                conformal_records.append(converted)
    return scalar_records, conformal_records, duration_records


def _aggregate_numbers(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "n": len(values),
    }


def aggregate_scalars(records):
    output = {}
    for model in sorted({record["model"] for record in records}):
        selected = [record for record in records if record["model"] == model]
        keys = sorted({key for record in selected for key in record["metrics"]})
        output[model] = {
            key: _aggregate_numbers([record["metrics"].get(key) for record in selected])
            for key in keys
        }
    return output


def aggregate_conformal(records):
    groups = defaultdict(list)
    for record in records:
        key = (
            record["model_architecture"],
            record["detect_method"],
            record["task"],
            record["time"],
            float(record["alpha"]),
        )
        groups[key].append(record)
    output = []
    for key, values in sorted(groups.items()):
        row = {
            "model": key[0],
            "method": key[1],
            "task": key[2],
            "time": key[3],
            "alpha": key[4],
            "seeds": sorted({int(value["seed"]) for value in values}),
            "num_seed_task_evaluations": len(values),
        }
        for field in NUMERIC_CP_FIELDS:
            aggregate = _aggregate_numbers([value[field] for value in values])
            row[f"{field}_mean"] = aggregate["mean"]
            row[f"{field}_std"] = aggregate["std"]
        output.append(row)
    return output


def save_conformal_csv(rows, path):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "seeds": "-".join(map(str, row["seeds"]))})


def _load_scores(run):
    path = Path(run["run_dir"]) / "evaluation" / "scores.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_band(run, method="model", alpha=0.10):
    path = Path(run["run_dir"]) / "evaluation" / "functional_bands.npz"
    with np.load(path) as bands:
        return np.asarray(bands[f"{method}_alpha_{alpha:.2f}"]).reshape(-1)


def _early_score(record):
    scores = np.asarray(record["scores"], dtype=float)
    return float(np.max(scores[: int(record["task_min_step"])]))


def plot_selection(summary, output_dir):
    models = sorted(summary["best_by_model"])
    splits = ("train", "val_seen", "val_unseen")
    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, split in enumerate(splits):
        means = [summary["best_by_model"][model][f"{split}_mean"] for model in models]
        stds = [summary["best_by_model"][model][f"{split}_std"] for model in models]
        ax.bar(x + (index - 1) * width, means, width, yerr=stds, capsize=3, label=split)
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x, ["SAFE-MLP" if model == "indep" else "SAFE-LSTM" for model in models])
    ax.set(ylabel="Matched-earliest ROC-AUC", ylim=(0, 1), title="Validation-selected official SAFE")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_auc.png", dpi=180)
    plt.close(fig)


def plot_conformal_tradeoff(rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), sharex=True, sharey=True)
    for axis, model in zip(axes, ("indep", "lstm")):
        selected = [
            row
            for row in rows
            if row["model"] == model
            and row["task"] == "all"
            and row["time"] == "by earliest stop"
            and row["method"] in ("model", "constant", "time_only")
        ]
        for method in ("model", "constant", "time_only"):
            values = sorted((row for row in selected if row["method"] == method), key=lambda row: row["alpha"])
            axis.plot(
                [row["fpr_mean"] for row in values],
                [row["tpr_mean"] for row in values],
                marker="o",
                markersize=3,
                label=method,
            )
        axis.plot([0, 1], [0, 1], "--", color="0.7", linewidth=1)
        axis.set_title("SAFE-MLP" if model == "indep" else "SAFE-LSTM")
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False)
    fig.suptitle("Functional conformal failure detection on held-out tasks")
    fig.tight_layout()
    fig.savefig(output_dir / "conformal_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_per_task(rows, output_dir, alpha=0.10):
    selected = [
        row
        for row in rows
        if row["method"] == "model"
        and row["task"] != "all"
        and row["time"] == "by earliest stop"
        and np.isclose(row["alpha"], alpha)
    ]
    tasks = sorted({row["task"] for row in selected})
    x = np.arange(len(tasks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.35), 4.4))
    for offset, model in ((-0.5, "indep"), (0.5, "lstm")):
        lookup = {row["task"]: row for row in selected if row["model"] == model}
        means = [lookup.get(task, {}).get("bal_acc_mean", np.nan) for task in tasks]
        ax.bar(
            x + offset * width,
            means,
            width,
            label="SAFE-MLP" if model == "indep" else "SAFE-LSTM",
        )
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    ax.set_xticks(x, tasks, rotation=35, ha="right")
    ax.set(ylabel="Balanced accuracy", ylim=(0, 1), title=f"Per-task conformal detection (alpha={alpha:.2f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "per_task_balanced_accuracy.png", dpi=180)
    plt.close(fig)


def plot_curves_and_trajectories(selected_runs, output_dir, alpha=0.10):
    seed_zero = [run for run in selected_runs if run["seed"] == 0]
    fig_roc, axes_roc = plt.subplots(1, 2, figsize=(9.4, 4.1))
    fig_score, axes_score = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig_det, ax_det = plt.subplots(figsize=(6.5, 4.2))
    for column, run in enumerate(sorted(seed_zero, key=lambda item: item["model"])):
        model = run["model"]
        records = [record for record in _load_scores(run) if record["split"] == "val_unseen"]
        labels = np.asarray([int(record["failed"]) for record in records])
        values = np.asarray([_early_score(record) for record in records])
        fpr, tpr, _ = roc_curve(labels, values)
        precision, recall, _ = precision_recall_curve(labels, values)
        axes_roc[column].plot(fpr, tpr, label="ROC")
        axes_roc[column].plot(recall, precision, label="precision-recall")
        axes_roc[column].plot([0, 1], [0, 1], "--", color="0.7", linewidth=1)
        axes_roc[column].set(
            xlabel="FPR / recall",
            ylabel="TPR / precision",
            xlim=(0, 1),
            ylim=(0, 1),
            title=("SAFE-MLP" if model == "indep" else "SAFE-LSTM") + " seed 0",
        )
        axes_roc[column].legend(frameon=False)

        band = _load_band(run, alpha=alpha)
        success = next(record for record in records if not record["failed"])
        failure = next(record for record in records if record["failed"])
        for record, label in ((success, "success"), (failure, "failure")):
            scores = np.asarray(record["scores"])
            axes_score[column].plot(np.arange(len(scores)), scores, label=f"{label}: {record['task_name']}")
        axes_score[column].plot(np.arange(len(band)), band, "k--", linewidth=1, label=f"band alpha={alpha:.2f}")
        axes_score[column].set(
            xlabel="Policy inference index",
            ylabel="Failure score",
            title="SAFE-MLP" if model == "indep" else "SAFE-LSTM",
        )
        axes_score[column].legend(frameon=False, fontsize=7)

        detection_times = []
        for record in records:
            if not record["failed"]:
                continue
            length = int(record["task_min_step"])
            scores = np.asarray(record["scores"][:length])
            crossings = np.flatnonzero(scores >= band[:length])
            detection_times.append((int(crossings[0]) if len(crossings) else length) / length)
        ax_det.hist(
            detection_times,
            bins=np.linspace(0, 1, 11),
            histtype="step",
            linewidth=2,
            label="SAFE-MLP" if model == "indep" else "SAFE-LSTM",
        )
    fig_roc.suptitle("Held-out-task discrimination at matched earliest stop")
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "roc_pr_seed0.png", dpi=180)
    plt.close(fig_roc)
    fig_score.suptitle("Representative selected-model trajectories and global bands")
    fig_score.tight_layout()
    fig_score.savefig(output_dir / "score_trajectories_threshold.png", dpi=180)
    plt.close(fig_score)
    ax_det.set(
        xlabel="Relative detection time (1 means undetected)",
        ylabel="Failed held-out rollouts",
        title=f"Detection-time distribution, seed 0, alpha={alpha:.2f}",
    )
    ax_det.legend(frameon=False)
    fig_det.tight_layout()
    fig_det.savefig(output_dir / "detection_times_seed0.png", dpi=180)
    plt.close(fig_det)


def build_report(grid_root, output_dir, expected_seeds=(0, 1, 2)):
    grid_root = Path(grid_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((grid_root / "selection_summary.json").read_text())
    if summary["num_completed_runs"] != 810:
        raise ValueError(f"Expected 810 completed runs, found {summary['num_completed_runs']}")
    selected_runs = find_selected_runs(grid_root, summary, expected_seeds)
    scalar_records, conformal_records, duration_records = load_selected_artifacts(selected_runs)
    scalar_summary = aggregate_scalars(scalar_records)
    conformal_summary = aggregate_conformal(conformal_records)
    primary = [row for row in conformal_summary if any(np.isclose(row["alpha"], alpha) for alpha in PRIMARY_ALPHAS)]
    report = {
        "schema_version": 1,
        "grid_root": str(grid_root),
        "official_grid": {
            "completed_runs": summary["num_completed_runs"],
            "configurations": summary["num_configurations"],
            "selection_rule": summary["selection_rule"],
            "selection_metric": summary["selection_metric"],
        },
        "selected_configurations": summary["best_by_model"],
        "selected_runs": [
            {key: value for key, value in run.items() if key != "provenance"}
            for run in selected_runs
        ],
        "scalar_metrics": scalar_summary,
        "primary_functional_conformal": primary,
        "duration_diagnostics": duration_records,
        "interpretation_guardrail": (
            "RoboCasa failures usually reach the horizon while successes terminate early; "
            "use matched-earliest discrimination and by-earliest-stop conformal results as primary."
        ),
    }
    write_json(output_dir / "final_report.json", report)
    save_conformal_csv(conformal_summary, output_dir / "selected_conformal_summary.csv")
    plot_selection(summary, output_dir)
    plot_conformal_tradeoff(conformal_summary, output_dir)
    plot_per_task(conformal_summary, output_dir)
    plot_curves_and_trajectories(selected_runs, output_dir)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = build_report(args.grid_root, args.output_dir, args.expected_seeds)
    if not args.quiet:
        concise = {
            "output_dir": str(Path(args.output_dir).resolve()),
            "completed_runs": report["official_grid"]["completed_runs"],
            "selected_configurations": report["selected_configurations"],
        }
        print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
