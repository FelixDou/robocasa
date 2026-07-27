"""Compute detailed statistics and figures from ABot subtask sidecars.

This analysis keeps official binary task success separate from semantic,
ordered subtask progress. It is intentionally simulator-independent so it can
run on a login or CPU node after an evaluation has completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


SPLIT_ORDER = ("atomic_seen", "composite_seen", "composite_unseen")
EXPECTED_TASK_COUNTS = {
    "atomic_seen": 18,
    "composite_seen": 16,
    "composite_unseen": 16,
}
DISPLAY_NAMES = {
    "atomic_seen": "Atomic-Seen",
    "composite_seen": "Composite-Seen",
    "composite_unseen": "Composite-Unseen",
    "overall": "Overall",
}
COLORS = {
    "atomic_seen": "#2F5D8A",
    "composite_seen": "#C47A2C",
    "composite_unseen": "#7B5AA6",
    "success": "#263238",
    "progress": "#4C91C6",
    "reference": "#8A4F45",
}
LEADERBOARD_SUCCESS = {
    "atomic_seen": 0.756,
    "composite_seen": 0.377,
    "composite_unseen": 0.033,
    "overall": 0.403,
}


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _bootstrap_fixed_task_ci(
    task_records: dict[str, list[dict]],
    field: str,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Resample rollouts within each fixed benchmark task."""
    if not task_records or samples <= 0:
        return (0.0, 0.0)
    estimates = []
    for _ in range(samples):
        task_means = []
        for records in task_records.values():
            values = [float(record[field]) for record in records]
            resampled = [rng.choice(values) for _ in values]
            task_means.append(_mean(resampled))
        estimates.append(_mean(task_means))
    estimates.sort()
    return (
        _percentile(estimates, 0.025),
        _percentile(estimates, 0.975),
    )


def load_rollouts(run_root: Path) -> tuple[list[dict], list[str], list[str]]:
    """Load sidecars and return normalized rollout records plus source issues."""
    records: list[dict] = []
    issues: list[str] = []
    source_paths = sorted(
        run_root.glob("*/envs/*/batches/*/subtask_progress.json")
    )
    if not source_paths:
        issues.append(f"no subtask_progress.json sidecars found below {run_root}")

    for source_path in source_paths:
        relative = source_path.relative_to(run_root).parts
        split = relative[0]
        path_env_name = relative[2] if len(relative) > 2 else ""
        batch_name = relative[4] if len(relative) > 4 else ""
        if split not in EXPECTED_TASK_COUNTS:
            continue
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"could not read {source_path}: {exc}")
            continue

        env_name = payload.get("env_name") or path_env_name
        if not env_name:
            issues.append(f"{source_path}: missing env_name")
            continue
        if path_env_name and env_name != path_env_name:
            issues.append(
                f"{source_path}: payload env_name {env_name!r} does not match "
                f"path task {path_env_name!r}"
            )
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            issues.append(f"{source_path}: episodes is not a list")
            continue
        for source_episode_index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                issues.append(
                    f"{source_path}: episode {source_episode_index} is not an object"
                )
                continue
            progress = float(episode.get("max_subtask_progress", 0.0))
            if not 0.0 <= progress <= 1.0:
                issues.append(
                    f"{source_path}: episode {source_episode_index} has "
                    f"out-of-range progress {progress}"
                )
            records.append(
                {
                    "split": split,
                    "env_name": env_name,
                    "batch": batch_name,
                    "source_episode_index": source_episode_index,
                    "episode_index": episode.get("episode_index"),
                    "seed": episode.get("seed"),
                    "steps": episode.get("steps"),
                    "success": bool(episode.get("success", False)),
                    "subtask_eval_available": bool(
                        episode.get("subtask_eval_available", False)
                    ),
                    "max_subtask_progress": progress,
                    "progress_gap": progress
                    - float(bool(episode.get("success", False))),
                    "stuck_subtask": episode.get("stuck_subtask") or "",
                    "failure_modes": list(episode.get("failure_modes", [])),
                    "completed_subtask_count": len(
                        episode.get("ordered_completed_required_subtasks", [])
                    ),
                    "source": str(source_path),
                }
            )

    return records, issues, [str(path) for path in source_paths]


def validate_rollouts(
    records: Sequence[dict],
    expected_episodes: int,
    initial_issues: Iterable[str] = (),
) -> list[str]:
    issues = list(initial_issues)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["env_name"])].append(record)

    for split in SPLIT_ORDER:
        tasks = sorted(
            env_name
            for record_split, env_name in grouped
            if record_split == split
        )
        expected_tasks = EXPECTED_TASK_COUNTS[split]
        if len(tasks) != expected_tasks:
            issues.append(
                f"{split}: expected {expected_tasks} tasks, found {len(tasks)}"
            )
        for env_name in tasks:
            task_records = grouped[(split, env_name)]
            if len(task_records) != expected_episodes:
                issues.append(
                    f"{split}/{env_name}: expected {expected_episodes} rollouts, "
                    f"found {len(task_records)}"
                )
            unavailable = sum(
                not record["subtask_eval_available"] for record in task_records
            )
            if unavailable:
                issues.append(
                    f"{split}/{env_name}: subtask evaluation unavailable for "
                    f"{unavailable} rollouts"
                )
    return issues


def _summarize_group(records: Sequence[dict]) -> dict:
    progress = [float(record["max_subtask_progress"]) for record in records]
    successes = [float(record["success"]) for record in records]
    failed = [record for record in records if not record["success"]]
    failed_partial = [
        record for record in failed if record["max_subtask_progress"] > 0.0
    ]
    return {
        "rollout_count": len(records),
        "success_count": int(sum(successes)),
        "success_rate": _mean(successes),
        "mean_max_subtask_progress": _mean(progress),
        "median_max_subtask_progress": (
            statistics.median(progress) if progress else 0.0
        ),
        "std_max_subtask_progress": _sample_std(progress),
        "progress_minus_success": _mean(progress) - _mean(successes),
        "no_progress_count": sum(value <= 1e-12 for value in progress),
        "partial_progress_count": sum(
            1e-12 < value < 1.0 - 1e-12 for value in progress
        ),
        "full_progress_count": sum(value >= 1.0 - 1e-12 for value in progress),
        "failed_rollout_count": len(failed),
        "failed_with_partial_progress_count": len(failed_partial),
        "failed_with_partial_progress_rate": (
            len(failed_partial) / len(failed) if failed else 0.0
        ),
    }


def compute_statistics(
    records: Sequence[dict],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    by_split_task: dict[str, dict[str, list[dict]]] = {
        split: defaultdict(list) for split in SPLIT_ORDER
    }
    for record in records:
        by_split_task[record["split"]][record["env_name"]].append(record)

    rng = random.Random(bootstrap_seed)
    split_statistics = {}
    task_statistics = []
    for split in SPLIT_ORDER:
        task_groups = by_split_task[split]
        split_records = [
            record for task_records in task_groups.values() for record in task_records
        ]
        summary = _summarize_group(split_records)
        summary["task_count"] = len(task_groups)
        success_ci = _bootstrap_fixed_task_ci(
            task_groups, "success", bootstrap_samples, rng
        )
        progress_ci = _bootstrap_fixed_task_ci(
            task_groups, "max_subtask_progress", bootstrap_samples, rng
        )
        summary["success_rate_ci95"] = list(success_ci)
        summary["mean_max_subtask_progress_ci95"] = list(progress_ci)
        summary["leaderboard_success_reference"] = LEADERBOARD_SUCCESS[split]
        summary["success_minus_leaderboard"] = (
            summary["success_rate"] - LEADERBOARD_SUCCESS[split]
        )
        split_statistics[split] = summary

        for env_name, task_records in sorted(task_groups.items()):
            task_summary = _summarize_group(task_records)
            stuck = Counter(
                record["stuck_subtask"]
                for record in task_records
                if record["stuck_subtask"]
            )
            modes = Counter(
                mode for record in task_records for mode in record["failure_modes"]
            )
            task_summary.update(
                {
                    "split": split,
                    "env_name": env_name,
                    "top_stuck_subtask": stuck.most_common(1)[0][0] if stuck else "",
                    "top_stuck_subtask_count": (
                        stuck.most_common(1)[0][1] if stuck else 0
                    ),
                    "failure_modes": dict(modes.most_common()),
                }
            )
            task_statistics.append(task_summary)

    overall_task_groups = {
        f"{split}/{env_name}": task_records
        for split, tasks in by_split_task.items()
        for env_name, task_records in tasks.items()
    }
    overall_records = list(records)
    overall = _summarize_group(overall_records)
    overall["task_count"] = len(overall_task_groups)
    overall["success_rate_ci95"] = list(
        _bootstrap_fixed_task_ci(
            overall_task_groups, "success", bootstrap_samples, rng
        )
    )
    overall["mean_max_subtask_progress_ci95"] = list(
        _bootstrap_fixed_task_ci(
            overall_task_groups,
            "max_subtask_progress",
            bootstrap_samples,
            rng,
        )
    )
    overall["leaderboard_success_reference"] = LEADERBOARD_SUCCESS["overall"]
    overall["success_minus_leaderboard"] = (
        overall["success_rate"] - LEADERBOARD_SUCCESS["overall"]
    )

    stuck_rows = []
    failure_mode_rows = []
    for split in SPLIT_ORDER:
        split_records = [
            record for record in records if record["split"] == split
        ]
        failed_count = sum(not record["success"] for record in split_records)
        for name, count in Counter(
            record["stuck_subtask"]
            for record in split_records
            if record["stuck_subtask"]
        ).most_common():
            stuck_rows.append(
                {
                    "split": split,
                    "stuck_subtask": name,
                    "count": count,
                    "rate_among_failed_rollouts": (
                        count / failed_count if failed_count else 0.0
                    ),
                }
            )
        for mode, count in Counter(
            mode for record in split_records for mode in record["failure_modes"]
        ).most_common():
            failure_mode_rows.append(
                {
                    "split": split,
                    "failure_mode": mode,
                    "count": count,
                    "rate_among_rollouts": (
                        count / len(split_records) if split_records else 0.0
                    ),
                }
            )

    return {
        "bootstrap": {
            "method": "fixed-task, within-task rollout bootstrap",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "splits": split_statistics,
        "overall": overall,
        "tasks": task_statistics,
        "stuck_subtasks": stuck_rows,
        "failure_modes": failure_mode_rows,
    }


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output_dir: Path,
    records: Sequence[dict],
    statistics_payload: dict,
) -> None:
    split_rows = []
    for split in (*SPLIT_ORDER, "overall"):
        values = (
            statistics_payload["overall"]
            if split == "overall"
            else statistics_payload["splits"][split]
        )
        split_rows.append({"split": split, **values})
    _write_csv(
        output_dir / "split_statistics.csv",
        split_rows,
        (
            "split",
            "task_count",
            "rollout_count",
            "success_count",
            "success_rate",
            "success_rate_ci95",
            "leaderboard_success_reference",
            "success_minus_leaderboard",
            "mean_max_subtask_progress",
            "mean_max_subtask_progress_ci95",
            "median_max_subtask_progress",
            "std_max_subtask_progress",
            "progress_minus_success",
            "no_progress_count",
            "partial_progress_count",
            "full_progress_count",
            "failed_rollout_count",
            "failed_with_partial_progress_count",
            "failed_with_partial_progress_rate",
        ),
    )
    _write_csv(
        output_dir / "task_statistics.csv",
        statistics_payload["tasks"],
        (
            "split",
            "env_name",
            "rollout_count",
            "success_count",
            "success_rate",
            "mean_max_subtask_progress",
            "median_max_subtask_progress",
            "std_max_subtask_progress",
            "progress_minus_success",
            "no_progress_count",
            "partial_progress_count",
            "full_progress_count",
            "failed_rollout_count",
            "failed_with_partial_progress_count",
            "failed_with_partial_progress_rate",
            "top_stuck_subtask",
            "top_stuck_subtask_count",
            "failure_modes",
        ),
    )
    _write_csv(
        output_dir / "stuck_subtasks.csv",
        statistics_payload["stuck_subtasks"],
        ("split", "stuck_subtask", "count", "rate_among_failed_rollouts"),
    )
    _write_csv(
        output_dir / "failure_modes.csv",
        statistics_payload["failure_modes"],
        ("split", "failure_mode", "count", "rate_among_rollouts"),
    )
    rollout_rows = []
    for record in records:
        row = dict(record)
        row["failure_modes"] = "|".join(record["failure_modes"])
        rollout_rows.append(row)
    _write_csv(
        output_dir / "rollout_statistics.csv",
        rollout_rows,
        (
            "split",
            "env_name",
            "batch",
            "episode_index",
            "source_episode_index",
            "seed",
            "steps",
            "success",
            "subtask_eval_available",
            "max_subtask_progress",
            "progress_gap",
            "completed_subtask_count",
            "stuck_subtask",
            "failure_modes",
            "source",
        ),
    )


def _task_label(name: str) -> str:
    label = name.replace("_", " ")
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return label


def _setup_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for figures; install it in the client "
            "environment or rerun with --no-figures"
        ) from exc
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt, PercentFormatter


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")


def create_figures(
    output_dir: Path,
    records: Sequence[dict],
    statistics_payload: dict,
) -> list[str]:
    plt, PercentFormatter = _setup_matplotlib()
    generated = []

    labels = [DISPLAY_NAMES[name] for name in (*SPLIT_ORDER, "overall")]
    success = [
        (
            statistics_payload["overall"]
            if name == "overall"
            else statistics_payload["splits"][name]
        )["success_rate"]
        for name in (*SPLIT_ORDER, "overall")
    ]
    progress = [
        (
            statistics_payload["overall"]
            if name == "overall"
            else statistics_payload["splits"][name]
        )["mean_max_subtask_progress"]
        for name in (*SPLIT_ORDER, "overall")
    ]
    references = [LEADERBOARD_SUCCESS[name] for name in (*SPLIT_ORDER, "overall")]
    x = list(range(len(labels)))
    width = 0.33
    fig, ax = plt.subplots(figsize=(10, 5.8))
    success_bars = ax.bar(
        [value - width / 2 for value in x],
        success,
        width,
        label="Binary task success",
        color=COLORS["success"],
    )
    progress_bars = ax.bar(
        [value + width / 2 for value in x],
        progress,
        width,
        label="Mean maximum ordered progress",
        color=COLORS["progress"],
    )
    ax.scatter(
        x,
        references,
        marker="D",
        s=42,
        color=COLORS["reference"],
        label="ABot-M0.5 leaderboard success",
        zorder=4,
    )
    ax.bar_label(
        success_bars,
        labels=[f"{value:.1%}" for value in success],
        padding=3,
    )
    ax.bar_label(
        progress_bars,
        labels=[f"{value:.1%}" for value in progress],
        padding=3,
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Rate")
    fig.suptitle("ABot-M0.5 success and ordered subtask progress", y=0.98)
    fig.text(
        0.5,
        0.925,
        f"Completed run: n={len(records)} rollouts; progress is diagnostic, "
        "not a replacement for task success.",
        ha="center",
        color="#555555",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save_figure(fig, output_dir, "split_success_vs_progress")
    plt.close(fig)
    generated.append("split_success_vs_progress")

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    task_rows = statistics_payload["tasks"]
    for split in SPLIT_ORDER:
        rows = [row for row in task_rows if row["split"] == split]
        ax.scatter(
            [row["success_rate"] for row in rows],
            [row["mean_max_subtask_progress"] for row in rows],
            s=48,
            alpha=0.86,
            color=COLORS[split],
            label=f"{DISPLAY_NAMES[split]} (n={len(rows)} tasks)",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    largest_gaps = [
        max(
            (row for row in task_rows if row["split"] == split),
            key=lambda row: row["progress_minus_success"],
        )
        for split in SPLIT_ORDER
    ]
    for index, row in enumerate(largest_gaps):
        ax.annotate(
            _task_label(row["env_name"]),
            (row["success_rate"], row["mean_max_subtask_progress"]),
            xytext=(5, 5 + index * 8),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Binary success rate (10 rollouts per task)")
    ax.set_ylabel("Mean maximum ordered subtask progress")
    fig.suptitle("Task-level success versus partial progress", y=0.98)
    fig.text(
        0.5,
        0.925,
        "Points above the diagonal made ordered progress that binary success hides.",
        ha="center",
        color="#555555",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save_figure(fig, output_dir, "task_success_vs_progress")
    plt.close(fig)
    generated.append("task_success_vs_progress")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    thresholds = [index / 100 for index in range(101)]
    for ax, split in zip(axes, SPLIT_ORDER):
        values = sorted(
            float(record["max_subtask_progress"])
            for record in records
            if record["split"] == split
        )
        survival = [
            sum(value >= threshold for value in values) / len(values)
            if values
            else 0.0
            for threshold in thresholds
        ]
        ax.step(
            thresholds,
            survival,
            where="post",
            color=COLORS[split],
            linewidth=2.2,
        )
        ax.set_title(f"{DISPLAY_NAMES[split]}\nn={len(values)} rollouts")
        ax.grid(alpha=0.18)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Maximum ordered progress reached")
    axes[0].set_ylabel("Rollouts reaching at least this progress")
    fig.suptitle("Rollout-level ordered progress survival curves", y=1.03)
    fig.tight_layout()
    _save_figure(fig, output_dir, "rollout_progress_survival")
    plt.close(fig)
    generated.append("rollout_progress_survival")

    ordered_tasks = {}
    max_task_count = 0
    for split in SPLIT_ORDER:
        rows = sorted(
            [row for row in task_rows if row["split"] == split],
            key=lambda row: row["mean_max_subtask_progress"],
        )
        ordered_tasks[split] = rows
        max_task_count = max(max_task_count, len(rows))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, max(9.0, max_task_count * 0.48)),
        sharex=True,
    )
    for ax, split in zip(axes, SPLIT_ORDER):
        rows = ordered_tasks[split]
        positions = list(range(len(rows)))
        for position, row in zip(positions, rows):
            ax.plot(
                [row["success_rate"], row["mean_max_subtask_progress"]],
                [position, position],
                color="#B8B8B8",
                linewidth=1.5,
                zorder=1,
            )
        ax.scatter(
            [row["success_rate"] for row in rows],
            positions,
            color=COLORS["success"],
            s=28,
            label="Success",
            zorder=2,
        )
        ax.scatter(
            [row["mean_max_subtask_progress"] for row in rows],
            positions,
            color=COLORS[split],
            s=34,
            label="Progress",
            zorder=3,
        )
        ax.set_yticks(positions, [_task_label(row["env_name"]) for row in rows])
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Rate")
        ax.set_title(DISPLAY_NAMES[split])
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Per-task binary success and mean maximum ordered progress",
        y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "task_progress_gaps")
    plt.close(fig)
    generated.append("task_progress_gaps")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, split in zip(axes, SPLIT_ORDER):
        rows = [
            row
            for row in statistics_payload["stuck_subtasks"]
            if row["split"] == split
        ][:10]
        rows = list(reversed(rows))
        if rows:
            ax.barh(
                range(len(rows)),
                [row["count"] for row in rows],
                color=COLORS[split],
            )
            ax.set_yticks(
                range(len(rows)),
                [_task_label(row["stuck_subtask"]) for row in rows],
                fontsize=8,
            )
        else:
            ax.text(0.5, 0.5, "No inferred stuck subtasks", ha="center", va="center")
            ax.set_yticks([])
        ax.set_xlabel("Failed rollouts")
        ax.set_title(DISPLAY_NAMES[split])
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("Most frequent inferred stuck subtasks (top 10 per split)", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_dir, "stuck_subtasks")
    plt.close(fig)
    generated.append("stuck_subtasks")

    return generated


def write_markdown_summary(
    output_dir: Path,
    run_root: Path,
    statistics_payload: dict,
    figure_stems: Sequence[str],
) -> None:
    lines = [
        "# ABot-M0.5 subtask-progress analysis",
        "",
        f"- Run root: `{run_root}`",
        f"- Rollouts: {statistics_payload['overall']['rollout_count']}",
        f"- Tasks: {statistics_payload['overall']['task_count']}",
        (
            "- Confidence intervals: 95% fixed-task bootstrap, resampling "
            "rollouts within each task"
        ),
        "",
        "## Split summary",
        "",
        (
            "| Split | Rollouts | Success | vs leaderboard | Mean max progress "
            "| Progress gap | Failed with progress |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLIT_ORDER:
        row = statistics_payload["splits"][split]
        lines.append(
            f"| {DISPLAY_NAMES[split]} | {row['rollout_count']} | "
            f"{row['success_rate']:.2%} | "
            f"{row['success_minus_leaderboard']:+.2%} | "
            f"{row['mean_max_subtask_progress']:.2%} | "
            f"{row['progress_minus_success']:+.2%} | "
            f"{row['failed_with_partial_progress_rate']:.2%} |"
        )
    overall = statistics_payload["overall"]
    lines.extend(
        [
            (
                f"| Overall | {overall['rollout_count']} | "
                f"{overall['success_rate']:.2%} | "
                f"{overall['success_minus_leaderboard']:+.2%} | "
                f"{overall['mean_max_subtask_progress']:.2%} | "
                f"{overall['progress_minus_success']:+.2%} | "
                f"{overall['failed_with_partial_progress_rate']:.2%} |"
            ),
            "",
            "## Interpretation guardrail",
            "",
            (
                "Binary task success is the leaderboard metric. Maximum ordered "
                "subtask progress is a parallel diagnostic and should not be "
                "reported as task success."
            ),
            "",
            "## Figures",
            "",
        ]
    )
    for stem in figure_stems:
        lines.append(f"- `{stem}.png` and `{stem}.svg`")
    lines.extend(
        [
            "",
            "## Tables",
            "",
            "- `rollout_statistics.csv`: one row per rollout",
            "- `task_statistics.csv`: one row per benchmark task",
            "- `split_statistics.csv`: split and overall aggregates",
            "- `stuck_subtasks.csv`: ranked inferred stuck predicates",
            "- `failure_modes.csv`: failure-mode frequencies",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_run(
    run_root: Path,
    expected_episodes: int,
    output_dir: Path | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260727,
    make_figures: bool = True,
) -> dict:
    run_root = run_root.resolve()
    output_dir = (output_dir or run_root / "subtask_analysis").resolve()
    records, load_issues, source_files = load_rollouts(run_root)
    issues = validate_rollouts(records, expected_episodes, load_issues)
    if issues:
        raise ValueError("Subtask-progress integrity checks failed:\n- " + "\n- ".join(issues))

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_payload = compute_statistics(
        records,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    result = {
        "schema_version": 1,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "expected_episodes_per_task": expected_episodes,
        "source_file_count": len(source_files),
        "integrity_checks_passed": True,
        **statistics_payload,
    }
    (output_dir / "subtask_progress_statistics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_tables(output_dir, records, result)
    figure_stems = (
        create_figures(output_dir, records, result) if make_figures else []
    )
    write_markdown_summary(output_dir, run_root, result, figure_stems)
    return result


def _print_summary(result: dict) -> None:
    print("ABot-M0.5 subtask-progress analysis complete")
    print(f"  output: {result['output_dir']}")
    print(
        f"  coverage: {result['overall']['task_count']} tasks, "
        f"{result['overall']['rollout_count']} rollouts"
    )
    for split in SPLIT_ORDER:
        row = result["splits"][split]
        success_ci = row["success_rate_ci95"]
        progress_ci = row["mean_max_subtask_progress_ci95"]
        print(
            f"  {DISPLAY_NAMES[split]}: success={row['success_rate']:.2%} "
            f"[{success_ci[0]:.2%}, {success_ci[1]:.2%}], "
            f"vs leaderboard={row['success_minus_leaderboard']:+.2%}, "
            f"progress={row['mean_max_subtask_progress']:.2%} "
            f"[{progress_ci[0]:.2%}, {progress_ci[1]:.2%}], "
            f"gap={row['progress_minus_success']:+.2%}"
        )
    overall = result["overall"]
    print(
        f"  Overall: success={overall['success_rate']:.2%}, "
        f"vs leaderboard={overall['success_minus_leaderboard']:+.2%}, "
        f"progress={overall['mean_max_subtask_progress']:.2%}, "
        f"gap={overall['progress_minus_success']:+.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze ABot subtask_progress.json sidecars, writing "
            "reproducible statistics, CSV tables, and publication-ready figures."
        )
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.expected_episodes <= 0:
        parser.error("--expected-episodes must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    try:
        result = analyze_run(
            args.run_root,
            expected_episodes=args.expected_episodes,
            output_dir=args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            make_figures=not args.no_figures,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    _print_summary(result)


if __name__ == "__main__":
    main()
