"""Generate figures added in the August 2026 SAFE report updates."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def read_csv(name):
    with (TABLES / name).open() as stream:
        return list(csv.DictReader(stream))


def plot_frozen_prefix32():
    rows = read_csv("subtask_v2_frozen_prefix32.csv")
    labels = [
        "All stages",
        "Dishwasher\nclosed",
        "Sink faucet\non",
        "Cutting board\nscrubbed",
        "Lettuce\nrinsed",
    ]
    x = np.arange(len(rows))
    width = 0.34
    raw = np.asarray([float(row["raw_roc_mean"]) for row in rows])
    raw_std = np.asarray([float(row["raw_roc_std"]) for row in rows])
    temporal = np.asarray([float(row["temporal_roc_mean"]) for row in rows])
    temporal_std = np.asarray(
        [float(row["temporal_roc_std"]) for row in rows]
    )

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(
        x - width / 2,
        raw,
        width,
        yerr=raw_std,
        capsize=3,
        color="#2563A6",
        edgecolor="#183B56",
        linewidth=0.7,
        label="Raw representation",
    )
    ax.bar(
        x + width / 2,
        temporal,
        width,
        yerr=temporal_std,
        capsize=3,
        color="#D17A22",
        edgecolor="#8A4D13",
        linewidth=0.7,
        label="Raw + delta + mean + slope",
    )
    ax.axhline(0.5, color="#4C5661", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 0.72)
    ax.set_ylabel("ROC-AUC at causal prefix 32")
    ax.set_title("Frozen parent-disjoint Subtask-SAFE evaluation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.text(
        0.5,
        0.01,
        "Four stages; 10 successful and 10 failed evaluation parents per stage; mean $\\pm$ population SD across three model seeds.",
        ha="center",
        fontsize=8.5,
        color="#4C5661",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"subtask_v2_frozen_prefix32.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_updated_collection_support():
    rows = read_csv("subtask_v2_updated_collection_support.csv")
    labels = [
        "Dishwasher closed",
        "Sink faucet on",
        "Cutting board scrubbed",
        "Lettuce rinsed",
    ]
    success = np.asarray([int(row["eligible_successes"]) for row in rows])
    failure = np.asarray([int(row["eligible_failures"]) for row in rows])
    eval_success = np.asarray([int(row["evaluation_successes"]) for row in rows])
    eval_failure = np.asarray([int(row["evaluation_failures"]) for row in rows])
    y = np.arange(len(rows))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.2),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    blue = "#2563A6"
    orange = "#D17A22"
    edge = "#183B56"

    axes[0].barh(
        y,
        success,
        color=blue,
        edgecolor=edge,
        linewidth=0.7,
        label="Success",
    )
    axes[0].barh(
        y,
        failure,
        left=success,
        color=orange,
        edgecolor="#8A4D13",
        linewidth=0.7,
        label="Failure",
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Eligible semantic segments")
    axes[0].set_title("Collected support")
    axes[0].grid(axis="x", alpha=0.25)
    for index, (left, right) in enumerate(zip(success, failure)):
        if left:
            axes[0].text(
                left / 2,
                index,
                str(left),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        if right:
            axes[0].text(
                left + right / 2,
                index,
                str(right),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    axes[1].barh(
        y,
        eval_success,
        color=blue,
        edgecolor=edge,
        linewidth=0.7,
    )
    axes[1].barh(
        y,
        eval_failure,
        left=eval_success,
        color=orange,
        edgecolor="#8A4D13",
        linewidth=0.7,
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 21)
    axes[1].set_xlabel("Frozen evaluation segments")
    axes[1].set_title("Parent-disjoint evaluation")
    axes[1].grid(axis="x", alpha=0.25)
    for index in y:
        axes[1].text(
            eval_success[index] / 2,
            index,
            str(eval_success[index]),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
        axes[1].text(
            eval_success[index] + eval_failure[index] / 2,
            index,
            str(eval_failure[index]),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
    )
    fig.suptitle("Updated targeted Subtask-SAFE label support", y=0.995)
    fig.text(
        0.5,
        0.015,
        "400 new-seed parent rollouts; 280 eligible segments across four selected stages. "
        "Evaluation freezes 10 successes and 10 failures per stage; calibration separately uses 5 successes per stage.",
        ha="center",
        fontsize=8.5,
        color="#4C5661",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90), w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"subtask_v2_updated_collection_support.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_natural_composite_distribution_merged():
    rows = read_csv("subtask_natural_composite_distribution_merged.csv")
    success = np.asarray([int(row["successes"]) for row in rows])
    active_failure = np.asarray([int(row["active_failures"]) for row in rows])
    regression_failure = np.asarray(
        [int(row["regression_failures"]) for row in rows]
    )
    no_inference = np.asarray(
        [int(row["no_usable_inference"]) for row in rows]
    )
    not_reached = np.asarray([int(row["not_reached"]) for row in rows])
    labels = [
        f"{row['task']}  {row['stage_order']}. {row['instruction']}"
        for row in rows
    ]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(14.8, 10.5))
    series = [
        (success, "#2563A6", "#183B56", None, "Successful usable segment"),
        (active_failure, "#D17A22", "#8A4D13", None, "Active-stage failure"),
        (
            regression_failure,
            "#B4235A",
            "#7A163E",
            None,
            "Later predicate regression",
        ),
        (
            no_inference,
            "#A78BBA",
            "#6F5A7E",
            "..",
            "Completed/bypassed without inference",
        ),
        (not_reached, "#D9DEE5", "#7B8794", "//", "Truly not reached"),
    ]
    left = np.zeros(len(rows), dtype=int)
    for values, color, edgecolor, hatch, label in series:
        ax.barh(
            y,
            values,
            left=left,
            color=color,
            edgecolor=edgecolor,
            linewidth=0.6,
            hatch=hatch,
            label=label,
        )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.7)
    ax.invert_yaxis()
    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlabel("Parent rollouts per task (n = 30)")
    ax.grid(axis="x", alpha=0.22)

    cumulative = np.zeros(len(rows), dtype=int)
    for values, text_color in (
        (success, "white"),
        (active_failure, "white"),
        (regression_failure, "white"),
        (no_inference, "white"),
        (not_reached, "#344054"),
    ):
        for index, value in enumerate(values):
            if value >= 2:
                ax.text(
                    cumulative[index] + value / 2,
                    index,
                    str(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                    fontweight="bold",
                )
            elif value == 1:
                ax.text(
                    cumulative[index] + value / 2,
                    index,
                    "1",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6.7,
                    fontweight="bold",
                )
        cumulative += values

    task_boundaries = []
    previous_task = rows[0]["task"]
    for index, row in enumerate(rows[1:], start=1):
        if row["task"] != previous_task:
            task_boundaries.append(index - 0.5)
            previous_task = row["task"]
    for boundary in task_boundaries:
        ax.axhline(boundary, color="#667085", linewidth=0.8, alpha=0.55)

    handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle(
        "Natural Subtask-SAFE label distribution across composite tasks",
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        "Failure timing and observation gaps shown separately",
        ha="center",
        fontsize=11,
        color="#4C5661",
    )
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        fontsize=8.8,
    )
    fig.text(
        0.5,
        0.012,
        "Natural collection from 150 parent rollouts (30 per task): 319 successful usable segments, "
        "83 active-stage failures, 2 later regressions, 2 stages crossed without a usable inference, and 74 truly unreached stages. "
        "Placement/release merges and transient-regression corrections are applied; later regression can coexist with a subsequently reached stage.",
        ha="center",
        fontsize=8.4,
        color="#4C5661",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES
            / f"subtask_natural_composite_distribution_merged.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_paper_roc_comparison():
    benchmarks = [
        "OpenVLA\nLIBERO",
        "$\\pi_0$-FAST\nLIBERO",
        "$\\pi_0$\nLIBERO",
        "$\\pi_0^*$\nSimplerEnv",
        "RLDX-1\nRoboCasa",
    ]
    mlp = np.asarray([0.7347, 0.8044, 0.7327, 0.8482, 0.713])
    lstm = np.asarray([0.7247, 0.8448, 0.7109, 0.8011, 0.547])
    x = np.arange(len(benchmarks))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(
        x - width / 2,
        mlp,
        width,
        color="#2563A6",
        edgecolor="#183B56",
        linewidth=0.7,
        label="SAFE-MLP",
    )
    ax.bar(
        x + width / 2,
        lstm,
        width,
        color="#D17A22",
        edgecolor="#8A4D13",
        linewidth=0.7,
        label="SAFE-LSTM",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("SAFE paper unseen-task ROC versus RLDX seen-task result")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.axvline(3.5, color="#4C5661", linestyle="--", linewidth=1)
    for index, (left, right) in enumerate(zip(mlp, lstm)):
        ax.text(index - width / 2, left + 0.012, f"{left:.3f}", ha="center", fontsize=8)
        ax.text(index + width / 2, right + 0.012, f"{right:.3f}", ha="center", fontsize=8)
    fig.text(
        0.5,
        0.005,
        "Paper bars: unseen-task ROC. RLDX bars: train-only task-z ROC on held-out samples from the same ten tasks; values are not directly comparable.",
        ha="center",
        fontsize=8.2,
        color="#4C5661",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"safe_paper_rldx_roc_comparison.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_xr1_prospective_task_scope():
    rows = [
        row
        for row in read_csv("xr1_prospective_matched_fpr_results.csv")
        if row["cohort"] == "primary_balanced"
    ]
    scopes = ["Overall", "Atomic", "Composite"]
    detectors = ["SAFE only", "Staged SAFE + time", "Time only"]
    by_key = {(row["scope"], row["detector"]): row for row in rows}
    x = np.arange(len(scopes))
    width = 0.25
    palette = {
        "SAFE only": "#2563A6",
        "Staged SAFE + time": "#D17A22",
        "Time only": "#A7B0BA",
    }
    edges = {
        "SAFE only": "#183B56",
        "Staged SAFE + time": "#8A4D13",
        "Time only": "#4C5661",
    }
    hatches = {
        "SAFE only": None,
        "Staged SAFE + time": "//",
        "Time only": "..",
    }
    panels = [
        ("tpr", "Failure recall (TPR)", (0.0, 1.0)),
        ("fpr", "Successful-rollout FPR", (0.0, 0.105)),
        (
            "adjusted_detection_fraction",
            "Miss-adjusted alarm fraction",
            (0.0, 1.0),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.7))
    for ax, (field, title, ylim) in zip(axes, panels):
        for detector_index, detector in enumerate(detectors):
            values = np.asarray(
                [float(by_key[(scope, detector)][field]) for scope in scopes]
            )
            bars = ax.bar(
                x + (detector_index - 1) * width,
                values,
                width,
                color=palette[detector],
                edgecolor=edges[detector],
                linewidth=0.8,
                hatch=hatches[detector],
                label=detector,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.018 if field != "fpr" else 0.0025),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    rotation=90 if field == "fpr" else 0,
                    color="#344054",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(scopes)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        if field == "adjusted_detection_fraction":
            ax.text(
                0.02,
                0.97,
                "Lower is earlier",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#4C5661",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
    )
    fig.suptitle("Prospective matched-FPR results by task scope", y=0.995)
    fig.text(
        0.5,
        0.015,
        "Balanced primary cohort: 380 successes and 380 failures across 38 tasks; atomic n=200, composite n=560. "
        "Frozen thresholds; no test-time retuning.",
        ha="center",
        fontsize=8.4,
        color="#4C5661",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.88), w_pad=1.5)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"xr1_prospective_task_scope.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_frozen_prefix32()
    plot_updated_collection_support()
    plot_natural_composite_distribution_merged()
    plot_paper_roc_comparison()
    plot_xr1_prospective_task_scope()


if __name__ == "__main__":
    main()
