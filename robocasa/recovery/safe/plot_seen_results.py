"""Plot the final all-five-seen SAFE evaluation without touching test selection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODEL_ORDER = ("indep", "lstm")
MODEL_LABELS = {"indep": "MLP", "lstm": "LSTM"}
MODEL_COLORS = {"indep": "#2563A6", "lstm": "#D97706"}
OUTCOME_COLORS = {False: "#2563A6", True: "#D97706"}


def load_json(path):
    return json.loads(Path(path).read_text())


def load_records(final_root):
    final_root = Path(final_root).resolve()
    records = [load_json(path) for path in sorted(final_root.glob("*/metrics.json"))]
    if len(records) != 6:
        raise ValueError(f"Expected six final metrics files, found {len(records)}")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)
    for model in MODEL_ORDER:
        seeds = sorted(int(record["seed"]) for record in grouped[model])
        if seeds != [0, 1, 2]:
            raise ValueError(f"{model} seeds are {seeds}, expected [0, 1, 2]")
        grouped[model].sort(key=lambda record: int(record["seed"]))
    return final_root, records, grouped


def metric(record, kind):
    key = f"falert_early_{kind}_auc/model_test"
    return float(record["scalar_metrics"][key])


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#1F2937",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlecolor": "#111827",
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.7,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "text.color": "#111827",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    return plt


def save_figure(fig, output_dir, stem, formats):
    outputs = []
    for extension in formats:
        path = Path(output_dir) / f"{stem}.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None)
        outputs.append(path)
    return outputs


def write_csvs(records, grouped, cv_summary, output_dir):
    output_dir = Path(output_dir)
    run_rows = []
    for record in records:
        selected = record["selected_hyperparameters"]
        run_rows.append(
            {
                "model": record["model"],
                "seed": int(record["seed"]),
                "test_roc_auc": metric(record, "roc"),
                "test_prc_auc": metric(record, "prc"),
                "duration_only_test_roc_auc": float(record["duration_only_test_roc_auc"]),
                "horizon_selector": selected["horizon_selector"],
                "diffusion_selector": selected["diffusion_selector"],
                "learning_rate": selected["learning_rate"],
                "lambda_reg": selected["lambda_reg"],
                "test_rollouts": int(record["counts"]["test"]),
            }
        )
    with (output_dir / "per_seed_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    summary_rows = []
    for model in MODEL_ORDER:
        selected = cv_summary["best_by_model"][model]
        roc = np.asarray([metric(record, "roc") for record in grouped[model]])
        prc = np.asarray([metric(record, "prc") for record in grouped[model]])
        duration = np.asarray(
            [float(record["duration_only_test_roc_auc"]) for record in grouped[model]]
        )
        summary_rows.append(
            {
                "model": model,
                "inner_cv_roc_mean": float(selected["inner_val_mean"]),
                "inner_cv_roc_std": float(selected["inner_val_std"]),
                "test_roc_mean": float(roc.mean()),
                "test_roc_std": float(roc.std()),
                "test_prc_mean": float(prc.mean()),
                "test_prc_std": float(prc.std()),
                "duration_roc_mean": float(duration.mean()),
                "duration_roc_std": float(duration.std()),
                "num_model_seeds": len(roc),
                "test_rollouts_per_seed": int(grouped[model][0]["counts"]["test"]),
            }
        )
    with (output_dir / "summary_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return run_rows, summary_rows


def plot_test_summary(plt, grouped, output_dir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharey=True)
    for ax, kind, title in zip(
        axes,
        ("roc", "prc"),
        ("Held-out ROC-AUC", "Held-out PRC-AUC"),
    ):
        for index, model in enumerate(MODEL_ORDER):
            values = np.asarray([metric(record, kind) for record in grouped[model]])
            ax.errorbar(
                index,
                values.mean(),
                yerr=values.std(),
                color=MODEL_COLORS[model],
                marker="o" if model == "indep" else "s",
                markersize=8,
                capsize=5,
                linewidth=2,
            )
            ax.text(index, values.mean() + 0.035, f"{values.mean():.3f}", ha="center", fontsize=9)
        ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.2, label="Chance")
        ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_ylim(0.45, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.75)
    axes[0].set_ylabel("Area under curve")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("SAFE performance on the fixed 30-rollout test set", fontsize=14, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Mean +/- population SD across three model seeds; 15 failures and 15 successes per seed",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.tight_layout()
    outputs = save_figure(fig, output_dir, "test_auc_summary", formats)
    plt.close(fig)
    return outputs


def plot_per_seed(plt, grouped, output_dir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharey=True)
    for ax, kind, title in zip(
        axes,
        ("roc", "prc"),
        ("ROC-AUC by training seed", "PRC-AUC by training seed"),
    ):
        for model in MODEL_ORDER:
            seeds = [int(record["seed"]) for record in grouped[model]]
            values = [metric(record, kind) for record in grouped[model]]
            ax.plot(
                seeds,
                values,
                color=MODEL_COLORS[model],
                marker="o" if model == "indep" else "s",
                linewidth=2,
                markersize=6,
                label=MODEL_LABELS[model],
            )
        ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.2)
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel("Training seed")
        ax.set_ylim(0.45, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.75)
    axes[0].set_ylabel("Area under curve")
    axes[1].legend(frameon=False)
    fig.suptitle("Seed sensitivity on the identical held-out split", fontsize=14, y=1.02)
    fig.tight_layout()
    outputs = save_figure(fig, output_dir, "per_seed_test_auc", formats)
    plt.close(fig)
    return outputs


def plot_cv_gap(plt, grouped, cv_summary, output_dir, formats):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    positions = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        selected = cv_summary["best_by_model"][model]
        cv_mean = float(selected["inner_val_mean"])
        cv_std = float(selected["inner_val_std"])
        test = np.asarray([metric(record, "roc") for record in grouped[model]])
        ax.plot([index - 0.12, index + 0.12], [cv_mean, test.mean()], color="#9CA3AF", linewidth=1.5)
        ax.errorbar(
            index - 0.12,
            cv_mean,
            yerr=cv_std,
            color="#2563A6",
            marker="o",
            capsize=4,
            linewidth=2,
            label="Inner CV" if index == 0 else None,
        )
        ax.errorbar(
            index + 0.12,
            test.mean(),
            yerr=test.std(),
            color="#D97706",
            marker="s",
            capsize=4,
            linewidth=2,
            label="Held-out test" if index == 0 else None,
        )
        ax.text(index - 0.12, cv_mean + 0.035, f"{cv_mean:.3f}", ha="center", fontsize=9)
        ax.text(index + 0.12, test.mean() - 0.055, f"{test.mean():.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.2)
    ax.set_xticks(positions, [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Inner-CV selection versus held-out performance")
    ax.grid(axis="y", alpha=0.75)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        -0.01,
        "Inner CV selected among 135 configurations per architecture; test set was never used for selection",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.tight_layout()
    outputs = save_figure(fig, output_dir, "cv_to_test_gap", formats)
    plt.close(fig)
    return outputs


def plot_duration_baseline(plt, grouped, output_dir, formats):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    offsets = {"indep": -0.08, "lstm": 0.08}
    for model in MODEL_ORDER:
        records = grouped[model]
        seeds = np.asarray([int(record["seed"]) for record in records], dtype=float)
        safe = np.asarray([metric(record, "roc") for record in records])
        ax.plot(
            seeds + offsets[model],
            safe,
            color=MODEL_COLORS[model],
            marker="o" if model == "indep" else "s",
            linewidth=2,
            label=f"SAFE {MODEL_LABELS[model]}",
        )
    duration_values = np.asarray(
        [float(record["duration_only_test_roc_auc"]) for record in grouped["indep"]]
    )
    ax.plot(
        [0, 1, 2],
        duration_values,
        color="#374151",
        marker="^",
        linestyle="--",
        linewidth=1.8,
        label="Episode duration only",
    )
    ax.axhline(0.5, color="#9CA3AF", linestyle=":", linewidth=1.2, label="Chance")
    ax.set_xticks([0, 1, 2])
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Held-out ROC-AUC")
    ax.set_title("SAFE compared with the episode-duration baseline")
    ax.grid(axis="y", alpha=0.75)
    ax.legend(frameon=False)
    fig.tight_layout()
    outputs = save_figure(fig, output_dir, "safe_vs_duration_baseline", formats)
    plt.close(fig)
    return outputs


def load_score_records(final_root, model, seed=0):
    path = Path(final_root) / f"{model}_seed{seed}" / "scores.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [record for record in records if record["split"] == "test"]


def interpolate_scores(records, failed, points=101):
    target = np.linspace(0.0, 1.0, points)
    values = []
    for record in records:
        if bool(record["failed"]) != failed:
            continue
        scores = np.asarray(record["scores"], dtype=float)
        source = np.linspace(0.0, 1.0, len(scores))
        values.append(np.interp(target, source, scores))
    return target, np.asarray(values)


def plot_score_trajectories(plt, final_root, output_dir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for ax, model in zip(axes, MODEL_ORDER):
        records = load_score_records(final_root, model, seed=0)
        for failed, label, linestyle in (
            (False, "Success", "-"),
            (True, "Failure", "--"),
        ):
            progress, values = interpolate_scores(records, failed)
            if not len(values):
                continue
            median = np.median(values, axis=0)
            lower = np.percentile(values, 25, axis=0)
            upper = np.percentile(values, 75, axis=0)
            color = OUTCOME_COLORS[failed]
            ax.fill_between(progress, lower, upper, color=color, alpha=0.16)
            ax.plot(
                progress,
                median,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{label} (n={len(values)})",
            )
        ax.set_xlabel("Normalized rollout progress")
        ax.set_ylabel("SAFE score")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(alpha=0.65)
        ax.legend(frameon=False)
    fig.suptitle("Held-out SAFE score trajectories (seed 0)", fontsize=14, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Median and interquartile range; descriptive score view with no test-fitted threshold",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.tight_layout()
    outputs = save_figure(fig, output_dir, "test_score_trajectories_seed0", formats)
    plt.close(fig)
    return outputs


def create_seen_result_plots(final_root, cv_summary, output_dir, formats=("png", "pdf")):
    final_root, records, grouped = load_records(final_root)
    cv_summary_path = Path(cv_summary).resolve()
    cv_summary = load_json(cv_summary_path)
    if cv_summary.get("outer_test_used_for_selection") is not False:
        raise ValueError("CV summary does not certify an untouched outer test set")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys(formats))
    unsupported = set(formats) - {"png", "pdf", "svg"}
    if unsupported:
        raise ValueError(f"Unsupported formats: {sorted(unsupported)}")
    run_rows, summary_rows = write_csvs(records, grouped, cv_summary, output_dir)
    plt = configure_matplotlib()
    outputs = []
    outputs += plot_test_summary(plt, grouped, output_dir, formats)
    outputs += plot_per_seed(plt, grouped, output_dir, formats)
    outputs += plot_cv_gap(plt, grouped, cv_summary, output_dir, formats)
    outputs += plot_duration_baseline(plt, grouped, output_dir, formats)
    outputs += plot_score_trajectories(plt, final_root, output_dir, formats)
    manifest = {
        "schema_version": 1,
        "final_root": str(final_root),
        "cv_summary": str(cv_summary_path),
        "num_final_runs": len(records),
        "num_test_rollouts_per_run": int(records[0]["counts"]["test"]),
        "plots": [str(path) for path in outputs],
        "data_files": [
            str(output_dir / "per_seed_metrics.csv"),
            str(output_dir / "summary_metrics.csv"),
        ],
        "summary_rows": summary_rows,
        "per_seed_rows": run_rows,
        "notes": [
            "Error bars are population standard deviations across three model seeds.",
            "The fixed 30-rollout test set contains 15 successes and 15 failures.",
            "No threshold is fitted on held-out test scores.",
        ],
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return outputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--cv-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outputs = create_seen_result_plots(
        args.final_root,
        args.cv_summary,
        args.output_dir,
        args.formats,
    )
    print(json.dumps([str(path) for path in outputs], indent=2))


if __name__ == "__main__":
    main()
