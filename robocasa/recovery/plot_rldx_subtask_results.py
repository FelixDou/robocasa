"""Create static figures for an RLDX RoboCasa subtask evaluation.

The script consumes the canonical ``benchmark_summary.json`` and
``subtask_accuracy.json`` written by the RLDX evaluation workflow. It exports
PNG/PDF/SVG figures plus chart-ready CSV tables and a small provenance
manifest. No policy server or simulator is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


GROUP_ORDER = ("atomic_seen", "composite_seen", "composite_unseen")
GROUP_LABELS = {
    "atomic_seen": "Atomic seen",
    "composite_seen": "Composite seen",
    "composite_unseen": "Composite unseen",
}
GROUP_SHORT = {
    "atomic_seen": "A",
    "composite_seen": "CS",
    "composite_unseen": "CU",
}
GROUP_COLORS = {
    "atomic_seen": "#356FA3",
    "composite_seen": "#D17C3D",
    "composite_unseen": "#7D8F4E",
}
GROUP_MARKERS = {
    "atomic_seen": "o",
    "composite_seen": "s",
    "composite_unseen": "^",
}
METRIC_COLORS = {
    "task_success": "#356FA3",
    "mean_progress": "#D17C3D",
    "subtask_accuracy": "#7D8F4E",
}
INK = "#202733"
MUTED = "#657080"
GRID = "#D8DEE6"
LIGHT_GRID = "#EDF0F3"
BENCHMARK = "#3C4654"
PUBLISHED_RLDX = {
    "atomic_seen": 0.676,
    "composite_seen": 0.279,
    "composite_unseen": 0.085,
    "overall": 0.360,
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _figure_heading(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    title_y: float = 0.97,
    subtitle_y: float = 0.91,
) -> None:
    fig.suptitle(title, y=title_y, fontsize=15, fontweight="semibold", color=INK)
    fig.text(
        0.5,
        subtitle_y,
        subtitle,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=9.5,
    )


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _validate_inputs(benchmark: dict, subtask: dict) -> None:
    task_rows = benchmark.get("tasks", [])
    if not task_rows:
        raise ValueError("benchmark summary contains no task rows")
    if benchmark.get("missing_tasks"):
        raise ValueError(
            f"benchmark summary has missing tasks: {benchmark['missing_tasks']}"
        )
    if benchmark.get("errors"):
        raise ValueError(f"benchmark summary has errors: {benchmark['errors']}")
    partial = [row.get("task") for row in task_rows if row.get("partial")]
    if partial:
        raise ValueError(f"benchmark summary has partial tasks: {partial}")

    subtask_tasks = subtask.get("tasks", [])
    if not subtask_tasks:
        raise ValueError("subtask accuracy JSON contains no task rows")
    benchmark_names = {row["task"] for row in task_rows}
    subtask_names = {row["task"] for row in subtask_tasks}
    if benchmark_names != subtask_names:
        raise ValueError(
            "benchmark and subtask task sets differ: "
            f"benchmark_only={sorted(benchmark_names - subtask_names)}, "
            f"subtask_only={sorted(subtask_names - benchmark_names)}"
        )


def _group_subtask_accuracy(subtask_tasks: list[dict]) -> dict[str, dict]:
    accum = defaultdict(lambda: {"completed": 0, "total": 0})
    for task in subtask_tasks:
        group = task["group"]
        for subtask in task.get("subtasks", []):
            accum[group]["completed"] += int(subtask["completed_rollouts"])
            accum[group]["total"] += int(subtask["rollout_count"])

    result = {}
    for group in GROUP_ORDER:
        completed = accum[group]["completed"]
        total = accum[group]["total"]
        result[group] = {
            "completed": completed,
            "total": total,
            "accuracy": completed / total if total else 0.0,
        }
    completed = sum(value["completed"] for value in accum.values())
    total = sum(value["total"] for value in accum.values())
    result["overall"] = {
        "completed": completed,
        "total": total,
        "accuracy": completed / total if total else 0.0,
    }
    return result


def _task_rows(benchmark: dict, subtask: dict) -> list[dict]:
    subtask_by_name = {row["task"]: row for row in subtask["tasks"]}
    rows = []
    for row in benchmark["tasks"]:
        task = row["task"]
        subtask_row = subtask_by_name[task]
        rows.append(
            {
                "task": task,
                "group": row["group"],
                "num_rollouts": int(row["num_rollouts"]),
                "num_successes": int(row["num_successes"]),
                "task_success": float(row["success_rate"]),
                "mean_progress": float(row["mean_max_subtask_progress"]),
                "subtask_count": len(subtask_row.get("subtasks", [])),
            }
        )
    return rows


def _group_rows(benchmark: dict, subtask_accuracy: dict[str, dict]) -> list[dict]:
    rows = []
    for group in GROUP_ORDER:
        summary = benchmark["groups"][group]
        task_rows = [row for row in benchmark["tasks"] if row["group"] == group]
        rows.append(
            {
                "group": group,
                "label": GROUP_LABELS[group],
                "tasks": int(summary["num_tasks"]),
                "rollouts": sum(int(row["num_rollouts"]) for row in task_rows),
                "task_success": float(summary["mean_task_success_rate"]),
                "mean_progress": float(summary["mean_task_max_subtask_progress"]),
                "subtask_accuracy": float(subtask_accuracy[group]["accuracy"]),
                "subtasks_completed": int(subtask_accuracy[group]["completed"]),
                "subtasks_total": int(subtask_accuracy[group]["total"]),
            }
        )

    rows.append(
        {
            "group": "overall",
            "label": "Overall",
            "tasks": int(benchmark["num_tasks"]),
            "rollouts": sum(int(row["num_rollouts"]) for row in benchmark["tasks"]),
            "task_success": float(benchmark["overall_mean_task_success_rate"]),
            "mean_progress": float(benchmark["overall_mean_task_max_subtask_progress"]),
            "subtask_accuracy": float(subtask_accuracy["overall"]["accuracy"]),
            "subtasks_completed": int(subtask_accuracy["overall"]["completed"]),
            "subtasks_total": int(subtask_accuracy["overall"]["total"]),
        }
    )
    return rows


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> list[str]:
    outputs = []
    for output_format in formats:
        path = output_dir / f"{stem}.{output_format}"
        kwargs = {"dpi": dpi} if output_format == "png" else {}
        fig.savefig(path, **kwargs)
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def plot_leaderboard_comparison(
    group_rows: list[dict],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    labels = [row["label"] for row in group_rows]
    reproduced = [100.0 * row["task_success"] for row in group_rows]
    published = [
        100.0 * PUBLISHED_RLDX[row["group"] if row["group"] != "overall" else "overall"]
        for row in group_rows
    ]
    y_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    for y, actual, reference in zip(y_positions, reproduced, published):
        ax.plot(
            [min(actual, reference), max(actual, reference)],
            [y, y],
            color=GRID,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            actual,
            y,
            s=105,
            color=METRIC_COLORS["task_success"],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            reference,
            y,
            s=105,
            facecolor="white",
            edgecolor=BENCHMARK,
            linewidth=1.8,
            zorder=3,
        )
        ax.text(
            actual - 1.0 if actual > reference else actual + 1.0,
            y - 0.17,
            f"{actual:.1f}%",
            ha="right" if actual > reference else "left",
            va="center",
            color=METRIC_COLORS["task_success"],
            fontweight="semibold",
        )
        ax.text(
            reference + 1.0 if reference >= actual else reference - 1.0,
            y + 0.18,
            f"{reference:.1f}%",
            ha="left" if reference >= actual else "right",
            va="center",
            color=BENCHMARK,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xlabel("Task success rate")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    _figure_heading(
        fig,
        "RLDX-1 RoboCasa success rates",
        "Reproduction uses 10 episodes/task; published leaderboard uses 50 episodes/task.",
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=METRIC_COLORS["task_success"],
                markeredgecolor="white",
                markersize=9,
                label="This run",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=BENCHMARK,
                markeredgewidth=1.6,
                markersize=9,
                label="Published RLDX-1",
            ),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.subplots_adjust(left=0.20, right=0.97, top=0.82, bottom=0.16)
    return _save_figure(fig, output_dir, "01_leaderboard_comparison", formats, dpi)


def plot_group_metrics(
    group_rows: list[dict],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    labels = [row["label"] for row in group_rows]
    y_positions = list(range(len(labels)))
    offsets = (-0.22, 0.0, 0.22)
    metrics = (
        ("task_success", "Task success", "o"),
        ("mean_progress", "Mean maximum progress", "s"),
        ("subtask_accuracy", "Required-subtask accuracy", "^"),
    )

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    for offset, (field, label, marker) in zip(offsets, metrics):
        values = [100.0 * row[field] for row in group_rows]
        positions = [y + offset for y in y_positions]
        ax.scatter(
            values,
            positions,
            marker=marker,
            s=92,
            color=METRIC_COLORS[field],
            edgecolor="white",
            linewidth=0.9,
            label=label,
            zorder=3,
        )
        for x, y in zip(values, positions):
            ax.text(
                x + 1.1,
                y,
                f"{x:.1f}%",
                va="center",
                ha="left",
                fontsize=8.5,
                color=METRIC_COLORS[field],
            )

    for y in y_positions:
        ax.axhline(y + 0.5, color=LIGHT_GRID, linewidth=0.8, zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            f"{row['label']}\n{row['tasks']} tasks, {row['rollouts']} rollouts"
            for row in group_rows
        ]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xlabel("Rate")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    _figure_heading(
        fig,
        "Task success and ordered subtask performance",
        "All metrics use the same 10 episodes/task; subtask accuracy weights required predicates.",
    )
    ax.legend(
        loc="lower right",
        frameon=False,
        ncol=1,
    )
    fig.subplots_adjust(left=0.26, right=0.95, top=0.83, bottom=0.16)
    return _save_figure(fig, output_dir, "02_group_success_and_subtasks", formats, dpi)


def plot_task_scatter(
    task_rows: list[dict],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    label_count: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    for group in GROUP_ORDER:
        rows = [row for row in task_rows if row["group"] == group]
        ax.scatter(
            [100.0 * row["task_success"] for row in rows],
            [100.0 * row["mean_progress"] for row in rows],
            s=72,
            marker=GROUP_MARKERS[group],
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            label=GROUP_LABELS[group],
        )

    ax.plot(
        [0, 100],
        [0, 100],
        linestyle=(0, (4, 4)),
        color=BENCHMARK,
        linewidth=1.1,
        alpha=0.8,
        label="Progress = success",
    )
    candidates = sorted(
        task_rows,
        key=lambda row: (
            row["mean_progress"] - row["task_success"],
            row["mean_progress"],
        ),
        reverse=True,
    )[:label_count]

    def spread_positions(rows: list[dict], min_gap: float = 5.4) -> dict[str, float]:
        ordered = sorted(
            ([row["task"], 100.0 * row["mean_progress"]] for row in rows),
            key=lambda item: item[1],
        )
        if not ordered:
            return {}
        for index in range(1, len(ordered)):
            ordered[index][1] = max(ordered[index][1], ordered[index - 1][1] + min_gap)
        overflow = max(0.0, ordered[-1][1] - 97.0)
        if overflow:
            for item in ordered:
                item[1] -= overflow
        for index in range(len(ordered) - 2, -1, -1):
            ordered[index][1] = min(ordered[index][1], ordered[index + 1][1] - min_gap)
        underflow = max(0.0, 3.0 - ordered[0][1])
        if underflow:
            for item in ordered:
                item[1] += underflow
        return {task: value for task, value in ordered}

    low_success = [row for row in candidates if 100.0 * row["task_success"] <= 15.0]
    other = [row for row in candidates if row not in low_success]
    label_y = {}
    label_y.update(spread_positions(low_success))
    label_y.update(spread_positions(other))

    for row in candidates:
        x = 100.0 * row["task_success"]
        y = 100.0 * row["mean_progress"]
        if x <= 15.0:
            text_x = 4.0
            horizontal_alignment = "left"
        elif x >= 75.0:
            text_x = x - 3.0
            horizontal_alignment = "right"
        else:
            text_x = x + 3.0
            horizontal_alignment = "left"
        ax.annotate(
            row["task"],
            (x, y),
            xytext=(text_x, label_y[row["task"]]),
            textcoords="data",
            ha=horizontal_alignment,
            va="center",
            fontsize=7.8,
            color=INK,
            arrowprops={
                "arrowstyle": "-",
                "color": GRID,
                "linewidth": 0.7,
                "shrinkA": 2,
                "shrinkB": 3,
            },
        )

    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Task success rate")
    ax.set_ylabel("Mean maximum ordered subtask progress")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(color=LIGHT_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    _figure_heading(
        fig,
        "Per-task success versus ordered subtask progress",
        "Each point is one task with 10 episodes; labels mark the largest progress-success gaps.",
        subtitle_y=0.92,
    )
    ax.legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.11)
    return _save_figure(fig, output_dir, "03_task_success_vs_progress", formats, dpi)


def plot_task_heatmap(
    task_rows: list[dict],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    rows = sorted(
        task_rows,
        key=lambda row: (
            GROUP_ORDER.index(row["group"]),
            -row["task_success"],
            -row["mean_progress"],
            row["task"],
        ),
    )
    values = [
        [100.0 * row["task_success"], 100.0 * row["mean_progress"]] for row in rows
    ]
    height = max(13.0, 0.29 * len(rows) + 2.7)
    fig, ax = plt.subplots(figsize=(8.6, height))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Task success", "Maximum progress"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"[{GROUP_SHORT[row['group']]}] {row['task']}" for row in rows],
        fontsize=8.2,
    )
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)

    for row_index, pair in enumerate(values):
        for column_index, value in enumerate(pair):
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= 58 else INK,
                fontweight="semibold" if value >= 70 else "normal",
            )

    group_boundaries = []
    group_starts = {}
    previous = None
    for index, row in enumerate(rows):
        group = row["group"]
        if group != previous:
            group_starts[group] = index
            if index:
                group_boundaries.append(index - 0.5)
            previous = group
    for boundary in group_boundaries:
        ax.axhline(boundary, color="white", linewidth=3.0)

    for group, start in group_starts.items():
        end = max(index for index, row in enumerate(rows) if row["group"] == group)
        ax.add_patch(
            Rectangle(
                (-0.66, start - 0.48),
                0.05,
                end - start + 0.96,
                transform=ax.transData,
                color=GROUP_COLORS[group],
                clip_on=False,
            )
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.06)
    colorbar.set_label("Rate")
    colorbar.ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    _figure_heading(
        fig,
        "RLDX-1 task performance matrix",
        "Tasks sorted by benchmark group and success; 10 episodes/task.",
        title_y=0.995,
        subtitle_y=0.973,
    )
    fig.subplots_adjust(left=0.39, right=0.88, top=0.925, bottom=0.03)
    return _save_figure(fig, output_dir, "04_task_performance_heatmap", formats, dpi)


def _bottleneck_rows(
    subtask: dict, rollout_counts: dict[str, int], top_n: int
) -> list[dict]:
    rows = []
    for row in subtask.get("bottlenecks", []):
        count = int(row.get("most_common_first_blocker_count", 0))
        if count <= 0:
            continue
        task = row["task"]
        total = int(rollout_counts.get(task, 0))
        rows.append(
            {
                "task": task,
                "group": row["group"],
                "blocker": row.get("most_common_first_blocker")
                or row.get("most_common_first_blocker_key")
                or "Unknown blocker",
                "count": count,
                "total": total,
                "rate": count / total if total else 0.0,
                "task_success": float(row.get("task_success", 0.0)),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["rate"],
            row["count"],
            -row["task_success"],
            row["task"],
        ),
        reverse=True,
    )[:top_n]


def plot_top_bottlenecks(
    rows: list[dict],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    if not rows:
        return []
    rows = list(reversed(rows))
    labels = [
        textwrap.fill(
            f"[{GROUP_SHORT[row['group']]}] {row['task']} — {row['blocker']}",
            width=54,
        )
        for row in rows
    ]
    values = [100.0 * row["rate"] for row in rows]
    colors = [GROUP_COLORS[row["group"]] for row in rows]
    y_positions = list(range(len(rows)))
    height = max(7.2, 0.54 * len(rows) + 2.3)

    fig, ax = plt.subplots(figsize=(12.4, height))
    bars = ax.barh(
        y_positions,
        values,
        color=colors,
        edgecolor=INK,
        linewidth=0.45,
        height=0.68,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Rollouts stopped first at this subtask")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for bar, row in zip(bars, rows):
        ax.text(
            min(102.0, bar.get_width() + 1.1),
            bar.get_y() + bar.get_height() / 2,
            f"{row['count']}/{row['total']}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
        )

    _figure_heading(
        fig,
        "Most frequent first blockers by task",
        "Top tasks ranked by first-blocker rate; each task has 10 rollouts.",
        subtitle_y=0.92,
    )
    fig.subplots_adjust(left=0.47, right=0.96, top=0.88, bottom=0.11)
    return _save_figure(fig, output_dir, "05_top_first_blockers", formats, dpi)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_supporting_files(
    *,
    output_dir: Path,
    benchmark_path: Path,
    subtask_path: Path,
    task_rows: list[dict],
    group_rows: list[dict],
    bottleneck_rows: list[dict],
    figure_files: list[str],
) -> None:
    _write_csv(
        output_dir / "task_metrics.csv",
        task_rows,
        [
            "task",
            "group",
            "num_rollouts",
            "num_successes",
            "task_success",
            "mean_progress",
            "subtask_count",
        ],
    )
    _write_csv(
        output_dir / "group_metrics.csv",
        group_rows,
        [
            "group",
            "label",
            "tasks",
            "rollouts",
            "task_success",
            "mean_progress",
            "subtask_accuracy",
            "subtasks_completed",
            "subtasks_total",
        ],
    )
    _write_csv(
        output_dir / "top_first_blockers.csv",
        bottleneck_rows,
        [
            "task",
            "group",
            "blocker",
            "count",
            "total",
            "rate",
            "task_success",
        ],
    )

    manifest = {
        "title": "RLDX-1 RoboCasa subtask evaluation figures",
        "benchmark_summary": str(benchmark_path),
        "subtask_accuracy": str(subtask_path),
        "task_count": len(task_rows),
        "rollout_count": sum(row["num_rollouts"] for row in task_rows),
        "episodes_per_task": sorted({row["num_rollouts"] for row in task_rows}),
        "published_reference": {key: value for key, value in PUBLISHED_RLDX.items()},
        "figures": figure_files,
        "notes": [
            "This run uses 10 episodes per task; published RLDX-1 uses 50.",
            "Task success is the official sparse task predicate.",
            "Mean progress is maximum ordered subtask progress per rollout, task-averaged.",
            "Required-subtask accuracy weights predicate opportunities across tasks.",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    chart_map = """# RLDX-1 figure map

| Figure | Analytical question | Form | Main fields |
|---|---|---|---|
| 01 leaderboard comparison | How does the 10-episode reproduction compare with the published 50-episode result? | Paired dot plot | task success, published reference |
| 02 group success and subtasks | How much ordered progress is made when sparse task success is low? | Grouped dot plot | task success, maximum progress, required-subtask accuracy |
| 03 task success vs progress | Which tasks make partial progress without completing? | Labeled scatter | task success, maximum progress, benchmark group |
| 04 task performance heatmap | Where are the strongest and weakest tasks across both metrics? | Two-column heatmap | task success, maximum progress |
| 05 top first blockers | Which first unmet subtasks stop the most rollouts? | Ranked horizontal bar | first-blocker count/rate, task, group |

All figures use the canonical `benchmark_summary.json` and
`subtask_accuracy.json` from the same run. Percentages are based on 10
episodes per task.
"""
    (output_dir / "README.md").write_text(chart_map)


def build_figures(args: argparse.Namespace) -> list[Path]:
    input_dir = args.input_dir.resolve()
    benchmark_path = (
        args.benchmark_summary.resolve()
        if args.benchmark_summary
        else input_dir / "benchmark_summary.json"
    )
    subtask_path = (
        args.subtask_accuracy.resolve()
        if args.subtask_accuracy
        else input_dir / "subtask_accuracy.json"
    )
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = _load_json(benchmark_path)
    subtask = _load_json(subtask_path)
    _validate_inputs(benchmark, subtask)
    task_rows = _task_rows(benchmark, subtask)
    subtask_accuracy = _group_subtask_accuracy(subtask["tasks"])
    group_rows = _group_rows(benchmark, subtask_accuracy)
    rollout_counts = {row["task"]: row["num_rollouts"] for row in task_rows}
    bottleneck_rows = _bottleneck_rows(subtask, rollout_counts, args.top_bottlenecks)

    _configure_style()
    figure_files = []
    figure_files.extend(
        plot_leaderboard_comparison(group_rows, output_dir, args.formats, args.dpi)
    )
    figure_files.extend(
        plot_group_metrics(group_rows, output_dir, args.formats, args.dpi)
    )
    figure_files.extend(
        plot_task_scatter(
            task_rows,
            output_dir,
            args.formats,
            args.dpi,
            args.scatter_labels,
        )
    )
    figure_files.extend(
        plot_task_heatmap(task_rows, output_dir, args.formats, args.dpi)
    )
    figure_files.extend(
        plot_top_bottlenecks(bottleneck_rows, output_dir, args.formats, args.dpi)
    )
    _write_supporting_files(
        output_dir=output_dir,
        benchmark_path=benchmark_path,
        subtask_path=subtask_path,
        task_rows=task_rows,
        group_rows=group_rows,
        bottleneck_rows=bottleneck_rows,
        figure_files=figure_files,
    )
    return [output_dir / name for name in figure_files]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Run directory containing benchmark_summary.json and subtask_accuracy.json",
    )
    parser.add_argument("--benchmark-summary", type=Path)
    parser.add_argument("--subtask-accuracy", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--top-bottlenecks", type=int, default=15)
    parser.add_argument("--scatter-labels", type=int, default=9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = build_figures(args)
    print(f"Wrote {len(outputs)} figure files to {outputs[0].parent}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
