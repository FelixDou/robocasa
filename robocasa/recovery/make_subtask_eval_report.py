"""Generate tables and SVG figures from subtask accuracy reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path


GROUP_LABELS = {
    "atomic_seen": "Atomic seen",
    "composite_seen": "Composite seen",
    "composite_unseen": "Composite unseen",
}

METRIC_LABELS = {
    "task_success": "Task success",
    "mean_ordered_progress": "Ordered progress",
    "mean_hardest_subtask_accuracy": "Hardest subtask",
}

COLORS = {
    "task_success": "#475569",
    "mean_ordered_progress": "#2563eb",
    "mean_hardest_subtask_accuracy": "#dc2626",
    "atomic_seen": "#2563eb",
    "composite_seen": "#0891b2",
    "composite_unseen": "#7c3aed",
}


def _rate(value: str | float | int) -> float:
    return float(value)


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _shorten(text: str, max_chars: int = 30) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "..."


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(rows: list[dict], path: Path, columns: list[tuple[str, str]]) -> None:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---:" if key.endswith("_pct") else "---" for key, _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key, _ in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _load_report(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    tasks = []
    bottlenecks = []
    if args.accuracy_json:
        data = json.loads(args.accuracy_json.read_text())
        tasks = data.get("tasks", [])
        bottlenecks = data.get("bottlenecks", [])
    if args.bottleneck_csv:
        bottlenecks = _read_csv(args.bottleneck_csv)
    if not bottlenecks:
        raise ValueError("Provide --accuracy-json or --bottleneck-csv.")
    return tasks, bottlenecks


def build_group_rows(tasks: list[dict], bottlenecks: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    source_rows = tasks or bottlenecks
    for row in source_rows:
        grouped[row["group"]].append(row)

    rows = []
    for group in sorted(grouped):
        group_rows = grouped[group]
        if tasks:
            task_success = sum(float(row["success_rate"]) for row in group_rows) / len(
                group_rows
            )
            ordered_progress = sum(
                float(row["mean_max_subtask_progress"]) for row in group_rows
            ) / len(group_rows)
            hardest = []
            for row in group_rows:
                subtasks = row.get("subtasks", [])
                if subtasks:
                    hardest.append(min(float(subtask["accuracy"]) for subtask in subtasks))
            hardest_accuracy = sum(hardest) / len(hardest) if hardest else 0.0
        else:
            task_success = sum(_rate(row["task_success"]) for row in group_rows) / len(
                group_rows
            )
            ordered_progress = sum(
                _rate(row["mean_ordered_progress"]) for row in group_rows
            ) / len(group_rows)
            hardest_accuracy = sum(
                _rate(row["hardest_subtask_accuracy"]) for row in group_rows
            ) / len(group_rows)
        rows.append(
            {
                "group": group,
                "group_label": GROUP_LABELS.get(group, group),
                "task_count": len(group_rows),
                "task_success": task_success,
                "mean_ordered_progress": ordered_progress,
                "mean_hardest_subtask_accuracy": hardest_accuracy,
                "task_success_pct": _pct(task_success),
                "mean_ordered_progress_pct": _pct(ordered_progress),
                "mean_hardest_subtask_accuracy_pct": _pct(hardest_accuracy),
            }
        )
    return rows


def build_bottleneck_rows(bottlenecks: list[dict]) -> list[dict]:
    rows = []
    for row in bottlenecks:
        rows.append(
            {
                "task": row["task"],
                "group": row["group"],
                "group_label": GROUP_LABELS.get(row["group"], row["group"]),
                "task_success": _rate(row["task_success"]),
                "mean_ordered_progress": _rate(row["mean_ordered_progress"]),
                "hardest_subtask": row["hardest_subtask"],
                "hardest_subtask_accuracy": _rate(row["hardest_subtask_accuracy"]),
                "most_common_first_blocker": row["most_common_first_blocker"],
                "most_common_first_blocker_count": int(
                    row["most_common_first_blocker_count"]
                ),
                "task_success_pct": _pct(_rate(row["task_success"])),
                "mean_ordered_progress_pct": _pct(_rate(row["mean_ordered_progress"])),
                "hardest_subtask_accuracy_pct": _pct(
                    _rate(row["hardest_subtask_accuracy"])
                ),
            }
        )
    return rows


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}",
        ".title{font-size:22px;font-weight:700}",
        ".label{font-size:13px}",
        ".small{font-size:11px;fill:#475569}",
        ".axis{stroke:#cbd5e1;stroke-width:1}",
        ".grid{stroke:#e2e8f0;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def write_group_bar_svg(rows: list[dict], path: Path) -> None:
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 92, 32, 82, 86
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    metrics = [
        "task_success",
        "mean_ordered_progress",
        "mean_hardest_subtask_accuracy",
    ]
    max_y = 0.55

    svg = _svg_header(width, height)
    svg.append(
        '<text x="32" y="38" class="title">Subtask-Level Performance by Task Group</text>'
    )
    svg.append(
        '<text x="32" y="62" class="small">pi0 reproduction checkpoint 74999, 5 rollouts per task</text>'
    )

    for tick in range(0, 6):
        value = tick * 0.1
        y = margin_top + plot_h - (value / max_y) * plot_h
        svg.append(
            f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" class="small">{int(value * 100)}%</text>'
        )
    svg.append(
        f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{margin_top + plot_h}" y2="{margin_top + plot_h}" class="axis"/>'
    )
    svg.append(
        f'<line x1="{margin_left}" x2="{margin_left}" y1="{margin_top}" y2="{margin_top + plot_h}" class="axis"/>'
    )

    group_w = plot_w / len(rows)
    bar_w = 44
    gap = 10
    for i, row in enumerate(rows):
        center = margin_left + group_w * (i + 0.5)
        start_x = center - (len(metrics) * bar_w + (len(metrics) - 1) * gap) / 2
        for j, metric in enumerate(metrics):
            value = float(row[metric])
            bar_h = (value / max_y) * plot_h
            x = start_x + j * (bar_w + gap)
            y = margin_top + plot_h - bar_h
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{COLORS[metric]}" rx="2"/>'
            )
            svg.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="small">{_pct(value)}</text>'
            )
        svg.append(
            f'<text x="{center:.1f}" y="{height - 44}" text-anchor="middle" class="label">{_escape(row["group_label"])}</text>'
        )

    legend_x = width - 455
    for i, metric in enumerate(metrics):
        x = legend_x + i * 150
        svg.append(
            f'<rect x="{x}" y="28" width="16" height="16" fill="{COLORS[metric]}" rx="2"/>'
        )
        svg.append(
            f'<text x="{x + 22}" y="41" class="small">{_escape(METRIC_LABELS[metric])}</text>'
        )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n")


def write_top_bottleneck_svg(rows: list[dict], path: Path, top_n: int = 18) -> None:
    selected = sorted(
        rows,
        key=lambda row: (
            -row["most_common_first_blocker_count"],
            row["hardest_subtask_accuracy"],
            row["task"],
        ),
    )[:top_n]
    width = 1180
    row_h = 34
    margin_left, margin_right, margin_top, margin_bottom = 290, 48, 74, 48
    height = margin_top + margin_bottom + row_h * len(selected)
    plot_w = width - margin_left - margin_right
    max_count = max((row["most_common_first_blocker_count"] for row in selected), default=1)

    svg = _svg_header(width, height)
    svg.append('<text x="32" y="38" class="title">Most Common First Blockers</text>')
    svg.append(
        '<text x="32" y="62" class="small">Rows sorted by number of blocked rollouts out of 5</text>'
    )
    for i, row in enumerate(selected):
        y = margin_top + i * row_h
        count = row["most_common_first_blocker_count"]
        bar_w = plot_w * (count / max_count if max_count else 0.0)
        color = COLORS.get(row["group"], "#64748b")
        label = f'{row["task"]} ({row["group_label"]})'
        blocker = row["most_common_first_blocker"]
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 20}" text-anchor="end" class="small">{_escape(_shorten(label, 36))}</text>'
        )
        svg.append(
            f'<rect x="{margin_left}" y="{y + 5}" width="{bar_w:.1f}" height="20" fill="{color}" rx="3"/>'
        )
        svg.append(
            f'<text x="{margin_left + bar_w + 8:.1f}" y="{y + 20}" class="small">{count}/5 - {_escape(_shorten(blocker, 46))}</text>'
        )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n")


def write_atomic_pickplace_svg(tasks: list[dict], path: Path) -> None:
    pickplace = [
        task for task in tasks if task["group"] == "atomic_seen" and "PickPlace" in task["task"]
    ]
    if not pickplace:
        return
    width, height = 1080, 420
    margin_left, margin_right, margin_top, margin_bottom = 230, 30, 76, 72
    plot_w = width - margin_left - margin_right
    row_h = (height - margin_top - margin_bottom) / len(pickplace)
    svg = _svg_header(width, height)
    svg.append(
        '<text x="32" y="38" class="title">Atomic Pick-and-Place Subtask Accuracy</text>'
    )
    svg.append(
        '<text x="32" y="62" class="small">Each cell is completion accuracy over 5 rollouts</text>'
    )

    all_subtasks = []
    for task in pickplace:
        for subtask in task.get("subtasks", []):
            label = subtask["description"]
            if label not in all_subtasks:
                all_subtasks.append(label)
    cell_w = plot_w / len(all_subtasks)
    for j, label in enumerate(all_subtasks):
        x = margin_left + j * cell_w + cell_w / 2
        svg.append(
            f'<text x="{x:.1f}" y="{height - 30}" text-anchor="middle" class="small">{_escape(_shorten(label, 16))}</text>'
        )
    for i, task in enumerate(pickplace):
        y = margin_top + i * row_h
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + row_h * 0.62:.1f}" text-anchor="end" class="small">{_escape(task["task"])}</text>'
        )
        by_label = {subtask["description"]: subtask for subtask in task.get("subtasks", [])}
        for j, label in enumerate(all_subtasks):
            subtask = by_label.get(label)
            value = float(subtask["accuracy"]) if subtask else 0.0
            shade = int(245 - value * 165)
            color = f"rgb({shade},{shade + int(value * 34)},{255})"
            x = margin_left + j * cell_w
            svg.append(
                f'<rect x="{x + 2:.1f}" y="{y + 3:.1f}" width="{cell_w - 4:.1f}" height="{row_h - 6:.1f}" fill="{color}" stroke="#ffffff"/>'
            )
            if subtask:
                svg.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + row_h * 0.62:.1f}" text-anchor="middle" class="small">{_pct(value)}</text>'
                )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy-json", type=Path)
    parser.add_argument("--bottleneck-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=18)
    args = parser.parse_args()

    tasks, raw_bottlenecks = _load_report(args)
    group_rows = build_group_rows(tasks, raw_bottlenecks)
    bottleneck_rows = build_bottleneck_rows(raw_bottlenecks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    group_table_rows = [
        {
            "group": row["group_label"],
            "task_count": row["task_count"],
            "task_success_pct": row["task_success_pct"],
            "mean_ordered_progress_pct": row["mean_ordered_progress_pct"],
            "mean_hardest_subtask_accuracy_pct": row[
                "mean_hardest_subtask_accuracy_pct"
            ],
        }
        for row in group_rows
    ]
    _write_csv(group_rows, args.out_dir / "group_metrics.csv", list(group_rows[0]))
    _write_md_table(
        group_table_rows,
        args.out_dir / "group_metrics.md",
        [
            ("group", "Group"),
            ("task_count", "Tasks"),
            ("task_success_pct", "Task Success"),
            ("mean_ordered_progress_pct", "Ordered Progress"),
            ("mean_hardest_subtask_accuracy_pct", "Hardest Subtask Accuracy"),
        ],
    )

    bottleneck_table_rows = [
        {
            "task": row["task"],
            "group": row["group_label"],
            "task_success_pct": row["task_success_pct"],
            "mean_ordered_progress_pct": row["mean_ordered_progress_pct"],
            "hardest_subtask": row["hardest_subtask"],
            "hardest_subtask_accuracy_pct": row["hardest_subtask_accuracy_pct"],
            "first_blocker": row["most_common_first_blocker"],
            "blocked_rollouts": row["most_common_first_blocker_count"],
        }
        for row in sorted(
            bottleneck_rows,
            key=lambda row: (
                -row["most_common_first_blocker_count"],
                row["hardest_subtask_accuracy"],
                row["group"],
                row["task"],
            ),
        )
    ]
    _write_csv(
        bottleneck_table_rows,
        args.out_dir / "task_bottlenecks.csv",
        list(bottleneck_table_rows[0]),
    )
    _write_md_table(
        bottleneck_table_rows,
        args.out_dir / "task_bottlenecks.md",
        [
            ("task", "Task"),
            ("group", "Group"),
            ("task_success_pct", "Task Success"),
            ("mean_ordered_progress_pct", "Ordered Progress"),
            ("hardest_subtask", "Hardest Subtask"),
            ("hardest_subtask_accuracy_pct", "Hardest Accuracy"),
            ("first_blocker", "First Blocker"),
            ("blocked_rollouts", "Blocked Rollouts"),
        ],
    )

    write_group_bar_svg(group_rows, args.out_dir / "group_metrics.svg")
    write_top_bottleneck_svg(
        bottleneck_rows, args.out_dir / "top_bottlenecks.svg", top_n=args.top_n
    )
    if tasks:
        write_atomic_pickplace_svg(tasks, args.out_dir / "atomic_pickplace_subtasks.svg")

    print(f"Wrote report artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
