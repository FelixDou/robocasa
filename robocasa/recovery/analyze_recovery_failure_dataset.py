"""Create tables and static figures for a recovery failure dataset manifest."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2F6B9A"
ORANGE = "#D9822B"
GOLD = "#C6A15B"
INK = "#24292F"
MUTED = "#667085"
GRID = "#D9DEE5"
LIGHT_BLUE = "#B8D3E6"
LIGHT_ORANGE = "#F2C89E"
GRANULARITY_COLORS = {"atomic": BLUE, "composite": ORANGE, "unknown": MUTED}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
    )
    return parser.parse_args()


def _id(value: Any, *keys: str) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in keys:
            if value.get(key):
                return str(value[key])
    return None


def _successful_ids(values: Any, *keys: str) -> set[str]:
    result: set[str] = set()
    if not isinstance(values, list):
        return result
    for value in values:
        if isinstance(value, dict) and value.get("success") is False:
            continue
        item_id = _id(value, *keys)
        if item_id:
            result.add(item_id)
    return result


def _completion_step_ids(sample: dict[str, Any]) -> set[str]:
    values = sample.get("subtask_completion_steps")
    if isinstance(values, dict):
        return {str(key) for key, step in values.items() if step is not None}
    if isinstance(values, list):
        return _successful_ids(values, "subtask_id", "id")
    return set()


def completed_subtask_ids(sample: dict[str, Any]) -> set[str]:
    result = _completion_step_ids(sample)
    result.update(
        _successful_ids(sample.get("completed_subtasks"), "subtask_id", "id")
    )
    if not result:
        result.update(
            str(entry["subtask_id"])
            for entry in sample.get("subtask_sequence", [])
            if isinstance(entry, dict)
            and entry.get("subtask_id")
            and entry.get("success") is True
        )
    return result


def completed_atomic_ids(sample: dict[str, Any]) -> set[str]:
    result = _successful_ids(
        sample.get("completed_atomic_steps"), "step_id", "atomic_task", "id"
    )
    if not result:
        result.update(
            str(entry.get("step_id") or entry.get("atomic_task"))
            for entry in sample.get("atomic_sequence", [])
            if isinstance(entry, dict)
            and (entry.get("step_id") or entry.get("atomic_task"))
            and entry.get("success") is True
        )
    return result


def ordered_prefix_length(sequence: list[dict[str, Any]], completed: set[str], key: str) -> int:
    prefix = 0
    for entry in sequence:
        if not isinstance(entry, dict) or entry.get("required") is False:
            continue
        entry_id = entry.get(key)
        if entry_id in completed:
            prefix += 1
        else:
            break
    return prefix


def failure_modes(sample: dict[str, Any]) -> list[str]:
    diagnostic = sample.get("failure_diagnostic") or {}
    modes = diagnostic.get("failure_modes") or sample.get("failure_modes") or []
    if isinstance(modes, str):
        return [modes]
    return [str(mode) for mode in modes] if isinstance(modes, list) else []


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def sample_row(sample: dict[str, Any]) -> dict[str, Any]:
    task_name = sample.get("task_name") or sample.get("task") or "unknown"
    granularity = sample.get("task_granularity") or "unknown"
    subtasks = [
        entry
        for entry in sample.get("subtask_sequence", [])
        if isinstance(entry, dict) and entry.get("required") is not False
    ]
    atomic_steps = [
        entry for entry in sample.get("atomic_sequence", []) if isinstance(entry, dict)
    ]
    completed_subtasks = completed_subtask_ids(sample)
    completed_atomics = completed_atomic_ids(sample)
    subtask_prefix = ordered_prefix_length(subtasks, completed_subtasks, "subtask_id")
    atomic_prefix = ordered_prefix_length(atomic_steps, completed_atomics, "step_id")
    failed = sample.get("failed_subtask") or {}
    diagnostic = sample.get("failure_diagnostic") or {}
    horizon = float(sample.get("horizon") or 0)
    failure_step = float(sample.get("failure_step") or sample.get("num_steps") or 0)

    return {
        "sample_id": sample.get("sample_id"),
        "task_name": task_name,
        "task_granularity": granularity,
        "success": bool(sample.get("success", False)),
        "num_steps": sample.get("num_steps"),
        "horizon": sample.get("horizon"),
        "failure_step": sample.get("failure_step"),
        "normalized_failure_step": _safe_ratio(failure_step, horizon),
        "num_atomic_steps": len(atomic_steps),
        "num_completed_atomic_steps": len(completed_atomics),
        "ordered_atomic_prefix": atomic_prefix,
        "atomic_progress": _safe_ratio(atomic_prefix, len(atomic_steps)),
        "num_subtasks": len(subtasks),
        "num_completed_subtasks": len(completed_subtasks),
        "ordered_subtask_prefix": subtask_prefix,
        "ordered_subtask_progress": _safe_ratio(subtask_prefix, len(subtasks)),
        "final_subtask_progress": sample.get("final_subtask_progress"),
        "max_subtask_progress": sample.get("max_subtask_progress"),
        "failed_atomic_step": sample.get("failed_atomic_step")
        or diagnostic.get("failed_atomic_step"),
        "failed_subtask_id": _id(failed, "subtask_id", "id")
        or diagnostic.get("failed_subtask"),
        "failed_subtask_instruction": (
            failed.get("instruction") if isinstance(failed, dict) else None
        ),
        "failure_stage": diagnostic.get("failure_stage") or "unknown",
        "failure_type": diagnostic.get("failure_type_weak_label") or "unknown",
        "failure_modes": "; ".join(failure_modes(sample)) or "unlabeled",
        "label_status": diagnostic.get("label_status") or "unknown",
        "video_present": bool(sample.get("video_path")),
        "actions_present": bool(sample.get("action_trajectory_path")),
    }


def subtask_rows(sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
    task_name = sample.get("task_name") or sample.get("task") or "unknown"
    granularity = sample.get("task_granularity") or "unknown"
    completed = completed_subtask_ids(sample)
    failed = _id(sample.get("failed_subtask"), "subtask_id", "id")
    diagnostic = sample.get("failure_diagnostic") or {}
    failed = failed or diagnostic.get("failed_subtask")

    for position, entry in enumerate(sample.get("subtask_sequence", []), start=1):
        if not isinstance(entry, dict) or entry.get("required") is False:
            continue
        subtask_id = str(entry.get("subtask_id") or f"position_{position}")
        yield {
            "sample_id": sample.get("sample_id"),
            "task_name": task_name,
            "task_granularity": granularity,
            "subtask_position": position,
            "subtask_id": subtask_id,
            "subtask_instruction": entry.get("instruction") or subtask_id,
            "subtask_stage": entry.get("stage") or "unknown",
            "completed": subtask_id in completed,
            "failed_here": subtask_id == failed,
        }


def atomic_rows(sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
    task_name = sample.get("task_name") or sample.get("task") or "unknown"
    granularity = sample.get("task_granularity") or "unknown"
    completed = completed_atomic_ids(sample)
    diagnostic = sample.get("failure_diagnostic") or {}
    failed = sample.get("failed_atomic_step") or diagnostic.get("failed_atomic_step")

    for position, entry in enumerate(sample.get("atomic_sequence", []), start=1):
        if not isinstance(entry, dict):
            continue
        step_id = str(entry.get("step_id") or entry.get("atomic_task") or f"position_{position}")
        yield {
            "sample_id": sample.get("sample_id"),
            "task_name": task_name,
            "task_granularity": granularity,
            "atomic_position": position,
            "atomic_step_id": step_id,
            "atomic_instruction": entry.get("instruction") or step_id,
            "atomic_skill": entry.get("skill_id") or "unknown",
            "completed": step_id in completed,
            "failed_here": step_id == failed,
        }


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    fig.tight_layout()
    for extension in formats:
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_subtitle(ax: plt.Axes, text: str) -> None:
    ax.set_title(ax.get_title(), pad=34)
    ax.text(0, 1.012, text, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")


def plot_task_balance(samples: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    counts = (
        samples.groupby(["task_name", "task_granularity"], as_index=False)
        .size()
        .sort_values(["task_granularity", "size", "task_name"])
    )
    fig, ax = plt.subplots(figsize=(10, 13))
    colors = [GRANULARITY_COLORS.get(value, MUTED) for value in counts["task_granularity"]]
    bars = ax.barh(counts["task_name"], counts["size"], color=colors, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_title("Samples per task")
    add_subtitle(ax, f"Recorded dataset samples; n={len(samples):,}")
    ax.set_xlabel("Samples")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output_dir, "01_samples_per_task", formats)


def plot_progress_distribution(samples: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for granularity in ["atomic", "composite"]:
        values = samples.loc[
            samples["task_granularity"] == granularity, "ordered_subtask_progress"
        ].dropna()
        if values.empty:
            continue
        ax.hist(
            values,
            bins=bins,
            alpha=0.62,
            color=GRANULARITY_COLORS[granularity],
            edgecolor="white",
            label=f"{granularity} (n={len(values):,})",
        )
    ax.set_title("Ordered subtask progress at failure")
    add_subtitle(ax, "Share of the semantic subtask sequence completed in order")
    ax.set_xlabel("Ordered subtask progress")
    ax.set_ylabel("Samples")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output_dir, "02_ordered_subtask_progress_distribution", formats)


def plot_task_progress(samples: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    grouped = (
        samples.groupby(["task_name", "task_granularity"], as_index=False)
        .agg(
            mean_ordered_progress=("ordered_subtask_progress", "mean"),
            samples=("sample_id", "size"),
        )
        .sort_values("mean_ordered_progress")
    )
    fig, ax = plt.subplots(figsize=(10, 13))
    colors = [GRANULARITY_COLORS.get(value, MUTED) for value in grouped["task_granularity"]]
    bars = ax.barh(grouped["task_name"], grouped["mean_ordered_progress"], color=colors)
    labels = [f"{value:.0%}" for value in grouped["mean_ordered_progress"]]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_title("Mean ordered subtask progress by task")
    add_subtitle(ax, "Mean share completed before failure; each task has its own semantic sequence")
    ax.set_xlabel("Mean ordered subtask progress")
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output_dir, "03_mean_progress_by_task", formats)


def plot_top_failed_subtasks(
    opportunities: pd.DataFrame, output_dir: Path, formats: list[str], top_n: int
) -> pd.DataFrame:
    grouped = (
        opportunities.groupby(
            ["task_name", "task_granularity", "subtask_id", "subtask_instruction"],
            as_index=False,
        )
        .agg(
            opportunities=("sample_id", "size"),
            failures=("failed_here", "sum"),
            completions=("completed", "sum"),
        )
    )
    grouped["failure_rate"] = grouped["failures"] / grouped["opportunities"]
    grouped["completion_rate"] = grouped["completions"] / grouped["opportunities"]
    ranked = grouped.sort_values(["failures", "failure_rate"], ascending=False).head(top_n).copy()
    ranked = ranked.sort_values(["failures", "failure_rate"])
    ranked["label"] = ranked.apply(
        lambda row: textwrap.shorten(
            f"{row['task_name']}: {row['subtask_instruction']}", width=72, placeholder="..."
        ),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.42)))
    colors = [GRANULARITY_COLORS.get(value, MUTED) for value in ranked["task_granularity"]]
    bars = ax.barh(ranked["label"], ranked["failures"], color=colors)
    labels = [
        f"{int(row.failures)}/{int(row.opportunities)} ({row.failure_rate:.0%})"
        for row in ranked.itertuples()
    ]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_title("Most frequent failed subtasks")
    add_subtitle(ax, "Count and within-subtask opportunity rate; top ranked task-subtask pairs")
    ax.set_xlabel("Samples failing at this subtask")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output_dir, "04_top_failed_subtasks", formats)
    return grouped.sort_values(["failures", "failure_rate"], ascending=False)


def plot_failure_modes(samples: pd.DataFrame, output_dir: Path, formats: list[str]) -> pd.DataFrame:
    rows = []
    for row in samples.itertuples(index=False):
        for mode in str(row.failure_modes).split("; "):
            rows.append(
                {
                    "task_granularity": row.task_granularity,
                    "failure_mode": mode,
                    "sample_id": row.sample_id,
                }
            )
    modes = pd.DataFrame(rows)
    counts = modes.groupby(["failure_mode", "task_granularity"]).size().unstack(fill_value=0)
    for column in ["atomic", "composite"]:
        if column not in counts:
            counts[column] = 0
    counts = counts.sort_values(["atomic", "composite"], ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.45)))
    y = np.arange(len(counts))
    ax.barh(y, counts["atomic"], color=BLUE, label="atomic")
    ax.barh(y, counts["composite"], left=counts["atomic"], color=ORANGE, label="composite")
    ax.set_yticks(y, counts.index)
    ax.invert_yaxis()
    ax.set_title("Failure-mode labels")
    add_subtitle(ax, "Multi-label counts; one sample may contribute to more than one mode")
    ax.set_xlabel("Label occurrences")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output_dir, "05_failure_modes", formats)
    return modes


def plot_task_matrix(samples: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    matrix = (
        samples.groupby(["task_name", "task_granularity"])
        .agg(
            ordered_subtask_progress=("ordered_subtask_progress", "mean"),
            atomic_progress=("atomic_progress", "mean"),
            normalized_failure_step=("normalized_failure_step", "mean"),
        )
        .reset_index()
        .sort_values(["task_granularity", "ordered_subtask_progress", "task_name"])
    )
    values = matrix[
        ["ordered_subtask_progress", "atomic_progress", "normalized_failure_step"]
    ].to_numpy(float)
    values = np.clip(np.nan_to_num(values), 0, 1)

    fig, ax = plt.subplots(figsize=(8.5, 13))
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(matrix)), matrix["task_name"], fontsize=8)
    ax.set_xticks(
        np.arange(3),
        ["Ordered subtask\nprogress", "Atomic-step\nprogress", "Failure time /\nhorizon"],
    )
    ax.set_title("Task-level progress and failure timing")
    add_subtitle(ax, "Task means; all cells use a common 0-1 scale")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.55 else INK,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean proportion")
    ax.tick_params(length=0)
    save_figure(fig, output_dir, "06_task_progress_timing_matrix", formats)


def task_statistics(samples: pd.DataFrame) -> pd.DataFrame:
    return (
        samples.groupby(["task_name", "task_granularity"], as_index=False)
        .agg(
            samples=("sample_id", "size"),
            mean_num_steps=("num_steps", "mean"),
            mean_failure_time_fraction=("normalized_failure_step", "mean"),
            mean_atomic_steps=("num_atomic_steps", "mean"),
            mean_completed_atomic_steps=("num_completed_atomic_steps", "mean"),
            mean_atomic_progress=("atomic_progress", "mean"),
            mean_subtasks=("num_subtasks", "mean"),
            mean_completed_subtasks=("num_completed_subtasks", "mean"),
            mean_ordered_subtask_progress=("ordered_subtask_progress", "mean"),
            video_coverage=("video_present", "mean"),
            action_coverage=("actions_present", "mean"),
        )
        .sort_values(["task_granularity", "task_name"])
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring the optional tabulate package."""
    columns = [str(column) for column in frame.columns]
    rows = [columns]
    for values in frame.itertuples(index=False, name=None):
        rows.append([str(value).replace("|", "\\|") for value in values])
    widths = [max(len(row[index]) for row in rows) for index in range(len(columns))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(rows[0]), separator, *(render(row) for row in rows[1:])])


def write_markdown_summary(
    manifest: dict[str, Any],
    samples: pd.DataFrame,
    task_stats: pd.DataFrame,
    subtask_stats: pd.DataFrame,
    output: Path,
) -> None:
    granularity = samples.groupby("task_granularity").size().to_dict()
    top = subtask_stats.head(10).copy()
    top["failure_rate"] = top["failure_rate"].map(lambda value: f"{value:.1%}")
    top = top[
        ["task_name", "subtask_instruction", "failures", "opportunities", "failure_rate"]
    ]
    hardest = task_stats.sort_values("mean_ordered_subtask_progress").head(10).copy()
    hardest["mean_ordered_subtask_progress"] = hardest[
        "mean_ordered_subtask_progress"
    ].map(lambda value: f"{value:.1%}")
    easiest = task_stats.sort_values("mean_ordered_subtask_progress", ascending=False).head(10).copy()
    easiest["mean_ordered_subtask_progress"] = easiest[
        "mean_ordered_subtask_progress"
    ].map(lambda value: f"{value:.1%}")

    lines = [
        "# Recovery failure dataset analysis",
        "",
        f"- Dataset type: `{manifest.get('dataset_type', 'unknown')}`",
        f"- Schema version: `{manifest.get('schema_version', 'unknown')}`",
        f"- Samples analyzed: **{len(samples):,}**",
        f"- Tasks covered: **{samples['task_name'].nunique()}**",
        f"- Atomic samples: **{granularity.get('atomic', 0):,}**",
        f"- Composite samples: **{granularity.get('composite', 0):,}**",
        "",
        "## Most frequent failed subtasks",
        "",
        markdown_table(top),
        "",
        "## Least progress before failure",
        "",
        markdown_table(
            hardest[
                [
                    "task_name",
                    "task_granularity",
                    "samples",
                    "mean_ordered_subtask_progress",
                ]
            ]
        ),
        "",
        "## Most progress before failure",
        "",
        markdown_table(
            easiest[
                [
                    "task_name",
                    "task_granularity",
                    "samples",
                    "mean_ordered_subtask_progress",
                ]
            ]
        ),
        "",
        "## Interpretation notes",
        "",
        "- Failure-mode counts are multi-label and can exceed the number of samples.",
        "- Ordered subtask progress counts only the contiguous completed prefix, preserving sequentiality.",
        "- Failed-subtask rates use samples containing that task-subtask pair as the denominator.",
        "- The dataset contains failed rollouts, so progress is not a task success rate.",
    ]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    raw_samples = manifest.get("samples", [])
    if not raw_samples:
        raise ValueError(f"No samples found in {args.manifest}")

    samples = pd.DataFrame(sample_row(sample) for sample in raw_samples)
    subtasks = pd.DataFrame(
        row for sample in raw_samples for row in subtask_rows(sample)
    )
    atomics = pd.DataFrame(
        row for sample in raw_samples for row in atomic_rows(sample)
    )
    tasks = task_statistics(samples)

    overview = pd.DataFrame(
        [
            ("samples", len(samples)),
            ("tasks", samples["task_name"].nunique()),
            ("atomic_samples", int((samples["task_granularity"] == "atomic").sum())),
            ("composite_samples", int((samples["task_granularity"] == "composite").sum())),
            ("manifest_errors", len(manifest.get("errors", []))),
            ("video_coverage", float(samples["video_present"].mean())),
            ("action_coverage", float(samples["actions_present"].mean())),
        ],
        columns=["metric", "value"],
    )

    configure_plotting()
    plot_task_balance(samples, args.output_dir, args.formats)
    plot_progress_distribution(samples, args.output_dir, args.formats)
    plot_task_progress(samples, args.output_dir, args.formats)
    subtask_stats = plot_top_failed_subtasks(
        subtasks, args.output_dir, args.formats, args.top_n
    )
    modes = plot_failure_modes(samples, args.output_dir, args.formats)
    plot_task_matrix(samples, args.output_dir, args.formats)

    atomic_stats = (
        atomics.groupby(
            ["task_name", "task_granularity", "atomic_step_id", "atomic_instruction"],
            as_index=False,
        )
        .agg(
            opportunities=("sample_id", "size"),
            failures=("failed_here", "sum"),
            completions=("completed", "sum"),
        )
    )
    atomic_stats["failure_rate"] = atomic_stats["failures"] / atomic_stats["opportunities"]
    atomic_stats["completion_rate"] = atomic_stats["completions"] / atomic_stats["opportunities"]
    atomic_stats = atomic_stats.sort_values(["failures", "failure_rate"], ascending=False)

    overview.to_csv(args.output_dir / "dataset_overview.csv", index=False)
    samples.to_csv(args.output_dir / "sample_statistics.csv", index=False)
    tasks.to_csv(args.output_dir / "task_statistics.csv", index=False)
    subtask_stats.to_csv(args.output_dir / "subtask_statistics.csv", index=False)
    atomic_stats.to_csv(args.output_dir / "atomic_step_statistics.csv", index=False)
    modes.to_csv(args.output_dir / "failure_mode_occurrences.csv", index=False)
    write_markdown_summary(
        manifest,
        samples,
        tasks,
        subtask_stats,
        args.output_dir / "summary.md",
    )

    print(f"Analyzed {len(samples):,} samples across {samples['task_name'].nunique()} tasks")
    print(f"Wrote tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
