"""Generate tried-subtask accuracy figures from non-recovery rollout JSONs.

This report differs from the full required-subtask accuracy report: downstream
subtasks that were never reached are not counted in the denominator. A failed
rollout that completed three ordered subtasks and then failed on the fourth
therefore contributes 3 successful subtasks out of 4 tried subtasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TASK_GROUPS = {
    "atomic_seen": {
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    },
    "composite_seen": {
        "DeliverStraw",
        "GetToastedBread",
        "KettleBoiling",
        "LoadDishwasher",
        "MakeIceLemonade",
        "PackIdenticalLunches",
        "PreSoakPan",
        "PrepareCoffee",
        "RinseSinkBasin",
        "ScrubCuttingBoard",
        "SearingMeat",
        "SetUpCuttingStation",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "StirVegetables",
        "StoreLeftoversInBowl",
        "WashLettuce",
    },
}

GROUP_ORDER = ["atomic_seen", "composite_seen", "composite_unseen"]
COLORS = {
    "hl": "#446E9B",
    "subtask": "#D17C3D",
    "avg": "#4E8F68",
    "grid": "#D8DEE6",
    "text": "#202733",
    "muted": "#657080",
}


def task_group(task_name: str) -> str:
    for group, tasks in TASK_GROUPS.items():
        if task_name in tasks:
            return group
    return "composite_unseen"


def is_success(rollout: dict) -> bool:
    return bool(rollout.get("success") or rollout.get("task_success"))


def required_subtasks(rollout: dict) -> list[str]:
    final_eval = rollout.get("final_subtask_eval") or {}
    return list(final_eval.get("required_predicates") or [])


def completed_subtasks(rollout: dict, required: list[str]) -> list[str]:
    completed = rollout.get("ordered_completed_required_subtasks")
    if completed is None:
        completed = rollout.get("completed_subtask_estimate") or []
    if is_success(rollout) and required:
        return list(required)
    return list(completed or [])


def tried_count(rollout: dict, required: list[str], completed_count: int) -> int:
    if is_success(rollout):
        return completed_count
    current = rollout.get("ordered_current_subtask") or rollout.get("current_subtask_estimate")
    if current is not None and completed_count < len(required):
        return completed_count + 1
    if required and completed_count < len(required):
        return completed_count + 1
    return completed_count


def summarize_rollout_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    task_name = data.get("env_name") or path.parent.parent.name
    rollouts = data.get("rollouts") or []

    rollout_count = len(rollouts)
    success_count = 0
    completed_total = 0
    tried_total = 0
    failed_rollout_count = 0
    completed_before_failure_total = 0
    required_count = 0

    for rollout in rollouts:
        required = required_subtasks(rollout)
        if required:
            required_count = max(required_count, len(required))
        completed = completed_subtasks(rollout, required)
        completed_count = len(completed)
        rollout_tried_count = tried_count(rollout, required, completed_count)

        success = is_success(rollout)
        success_count += int(success)
        completed_total += completed_count
        tried_total += rollout_tried_count

        if not success:
            failed_rollout_count += 1
            completed_before_failure_total += completed_count

    return {
        "task": task_name,
        "group": task_group(task_name),
        "rollout_json": str(path),
        "rollouts": rollout_count,
        "task_successes": success_count,
        "required_subtasks": required_count,
        "completed_tried_subtasks": completed_total,
        "tried_subtasks": tried_total,
        "failed_rollouts": failed_rollout_count,
        "completed_before_failure": completed_before_failure_total,
        "hl_accuracy": success_count / rollout_count if rollout_count else 0.0,
        "tried_subtask_accuracy": completed_total / tried_total if tried_total else 0.0,
        "avg_successful_subtasks_before_failure": (
            completed_before_failure_total / failed_rollout_count
            if failed_rollout_count
            else 0.0
        ),
    }


def collect_paths(inputs: list[Path]) -> list[Path]:
    paths = []
    for path in inputs:
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(path.rglob("subtask_rollouts.json"))
        else:
            raise FileNotFoundError(path)
    return sorted(set(paths))


def group_rows(task_rows: list[dict]) -> list[dict]:
    rows = []
    for group in GROUP_ORDER:
        members = [row for row in task_rows if row["group"] == group]
        if not members:
            continue
        rollouts = sum(row["rollouts"] for row in members)
        successes = sum(row["task_successes"] for row in members)
        completed = sum(row["completed_tried_subtasks"] for row in members)
        tried = sum(row["tried_subtasks"] for row in members)
        failed = sum(row["failed_rollouts"] for row in members)
        completed_before_failure = sum(row["completed_before_failure"] for row in members)
        rows.append(
            {
                "group": group,
                "tasks": len(members),
                "rollouts": rollouts,
                "task_successes": successes,
                "completed_tried_subtasks": completed,
                "tried_subtasks": tried,
                "failed_rollouts": failed,
                "completed_before_failure": completed_before_failure,
                "hl_accuracy": successes / rollouts if rollouts else 0.0,
                "tried_subtask_accuracy": completed / tried if tried else 0.0,
                "avg_successful_subtasks_before_failure": (
                    completed_before_failure / failed if failed else 0.0
                ),
            }
        )
    return rows


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def draw_group_bars(rows: list[dict], path: Path) -> None:
    width, height = 1200, 780
    margin_l, margin_r, margin_t, margin_b = 150, 70, 150, 120
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(30, bold=True)
    label_font = font(19)
    small_font = font(16)

    draw.text((margin_l, 35), "High-level success vs tried-subtask accuracy", fill=COLORS["text"], font=title_font)
    draw.text((margin_l, 72), "Tried-subtask accuracy counts only completed subtasks plus the first failed subtask.", fill=COLORS["muted"], font=small_font)

    legend_x = margin_l
    legend_y = 108
    legend_items = (
        ("High-level success", COLORS["hl"]),
        ("Tried-subtask accuracy", COLORS["subtask"]),
    )
    for idx, (label, color) in enumerate(legend_items):
        x = legend_x + idx * 235
        draw.rectangle((x, legend_y, x + 22, legend_y + 16), fill=color)
        draw.text((x + 31, legend_y - 3), label, fill=COLORS["text"], font=small_font)

    for tick in range(0, 101, 20):
        y = margin_t + chart_h - chart_h * tick / 100.0
        draw.line((margin_l, y, width - margin_r, y), fill=COLORS["grid"], width=1)
        draw.text((margin_l - 58, y - 10), f"{tick}%", fill=COLORS["muted"], font=small_font)

    group_w = chart_w / max(1, len(rows))
    bar_w = min(78, group_w * 0.24)
    for i, row in enumerate(rows):
        cx = margin_l + group_w * (i + 0.5)
        vals = [("HL", row["hl_accuracy"], COLORS["hl"]), ("Tried subtask", row["tried_subtask_accuracy"], COLORS["subtask"])]
        for j, (_, value, color) in enumerate(vals):
            x0 = cx + (j - 0.5) * (bar_w + 18)
            x1 = x0 + bar_w
            y0 = margin_t + chart_h - chart_h * value
            draw.rectangle((x0, y0, x1, margin_t + chart_h), fill=color)
            draw.text((x0 - 8, y0 - 25), pct(value), fill=COLORS["text"], font=small_font)
        draw.text((cx - 68, margin_t + chart_h + 22), row["group"], fill=COLORS["text"], font=label_font)
        draw.text((cx - 58, margin_t + chart_h + 50), f"n={row['rollouts']}", fill=COLORS["muted"], font=small_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_task_bars(rows: list[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda row: (GROUP_ORDER.index(row["group"]), row["hl_accuracy"], row["tried_subtask_accuracy"], row["task"]))
    row_h = 34
    width = 1500
    height = 110 + row_h * len(rows) + 70
    margin_l, margin_r, margin_t = 360, 70, 95
    chart_w = width - margin_l - margin_r
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(28, bold=True)
    label_font = font(16)
    small_font = font(14)

    draw.text((margin_l, 30), "Per-task high-level success vs tried-subtask accuracy", fill=COLORS["text"], font=title_font)
    draw.text((margin_l, 62), "Orange can be high even when full-task success is low, because un-reached downstream subtasks are excluded.", fill=COLORS["muted"], font=small_font)

    for tick in range(0, 101, 20):
        x = margin_l + chart_w * tick / 100.0
        draw.line((x, margin_t - 12, x, height - 45), fill=COLORS["grid"], width=1)
        draw.text((x - 16, height - 36), f"{tick}%", fill=COLORS["muted"], font=small_font)

    last_group = None
    for i, row in enumerate(rows):
        y = margin_t + i * row_h
        if row["group"] != last_group:
            draw.text((25, y + 5), row["group"], fill=COLORS["muted"], font=small_font)
            last_group = row["group"]
        draw.text((145, y + 4), row["task"], fill=COLORS["text"], font=label_font)
        hl_w = chart_w * row["hl_accuracy"]
        st_w = chart_w * row["tried_subtask_accuracy"]
        draw.rectangle((margin_l, y + 5, margin_l + st_w, y + 16), fill=COLORS["subtask"])
        draw.rectangle((margin_l, y + 19, margin_l + hl_w, y + 30), fill=COLORS["hl"])
        draw.text((margin_l + max(st_w, hl_w) + 8, y + 7), f"{pct(row['hl_accuracy'])} / {pct(row['tried_subtask_accuracy'])}", fill=COLORS["muted"], font=small_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_avg_completed_before_failure(rows: list[dict], path: Path) -> None:
    rows = sorted(
        [row for row in rows if row["failed_rollouts"]],
        key=lambda row: (GROUP_ORDER.index(row["group"]), row["avg_successful_subtasks_before_failure"], row["task"]),
    )
    row_h = 34
    width = 1450
    height = 110 + row_h * len(rows) + 70
    margin_l, margin_r, margin_t = 360, 70, 95
    max_value = max([row["avg_successful_subtasks_before_failure"] for row in rows] + [1.0])
    tick_max = max(1, math.ceil(max_value))
    chart_w = width - margin_l - margin_r
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(28, bold=True)
    label_font = font(16)
    small_font = font(14)

    draw.text((margin_l, 30), "Average successful ordered subtasks before failure", fill=COLORS["text"], font=title_font)
    draw.text((margin_l, 62), "Computed on failed high-level rollouts only.", fill=COLORS["muted"], font=small_font)

    for tick in range(tick_max + 1):
        x = margin_l + chart_w * tick / tick_max
        draw.line((x, margin_t - 12, x, height - 45), fill=COLORS["grid"], width=1)
        draw.text((x - 5, height - 36), str(tick), fill=COLORS["muted"], font=small_font)

    last_group = None
    for i, row in enumerate(rows):
        y = margin_t + i * row_h
        if row["group"] != last_group:
            draw.text((25, y + 5), row["group"], fill=COLORS["muted"], font=small_font)
            last_group = row["group"]
        draw.text((145, y + 4), row["task"], fill=COLORS["text"], font=label_font)
        value = row["avg_successful_subtasks_before_failure"]
        bar_w = chart_w * value / tick_max
        draw.rectangle((margin_l, y + 7, margin_l + bar_w, y + 27), fill=COLORS["avg"])
        draw.text((margin_l + bar_w + 8, y + 5), f"{value:.2f}", fill=COLORS["muted"], font=small_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "group",
        "rollouts",
        "task_successes",
        "hl_accuracy",
        "completed_tried_subtasks",
        "tried_subtasks",
        "tried_subtask_accuracy",
        "failed_rollouts",
        "completed_before_failure",
        "avg_successful_subtasks_before_failure",
        "required_subtasks",
        "rollout_json",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_group_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group",
        "tasks",
        "rollouts",
        "task_successes",
        "hl_accuracy",
        "completed_tried_subtasks",
        "tried_subtasks",
        "tried_subtask_accuracy",
        "failed_rollouts",
        "completed_before_failure",
        "avg_successful_subtasks_before_failure",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_summary(rows: list[dict], grouped: list[dict], path: Path) -> None:
    lines = [
        "# Tried-subtask accuracy",
        "",
        "Metric: completed ordered subtasks / tried ordered subtasks.",
        "For failed rollouts, tried ordered subtasks = completed ordered subtasks + the first failed current subtask.",
        "",
        "| Group | HL accuracy | Tried-subtask accuracy | Avg successful subtasks before failure | Rollouts |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            f"| {row['group']} | {pct(row['hl_accuracy'])} | {pct(row['tried_subtask_accuracy'])} | "
            f"{row['avg_successful_subtasks_before_failure']:.2f} | {row['rollouts']} |"
        )
    all_rollouts = sum(row["rollouts"] for row in rows)
    all_successes = sum(row["task_successes"] for row in rows)
    all_completed = sum(row["completed_tried_subtasks"] for row in rows)
    all_tried = sum(row["tried_subtasks"] for row in rows)
    all_failed = sum(row["failed_rollouts"] for row in rows)
    all_completed_before_failure = sum(row["completed_before_failure"] for row in rows)
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- HL accuracy: {all_successes}/{all_rollouts} ({pct(all_successes / all_rollouts if all_rollouts else 0.0)})",
            f"- Tried-subtask accuracy: {all_completed}/{all_tried} ({pct(all_completed / all_tried if all_tried else 0.0)})",
            f"- Average successful subtasks before failure: {all_completed_before_failure / all_failed if all_failed else 0.0:.2f}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+", help="subtask_rollouts.json files or directories containing them")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = collect_paths(args.inputs)
    if not paths:
        raise SystemExit("No subtask_rollouts.json files found")

    rows = [summarize_rollout_json(path) for path in paths]
    grouped = group_rows(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "tried_subtask_accuracy_by_task.csv")
    write_group_csv(grouped, args.out_dir / "tried_subtask_accuracy_by_group.csv")
    write_summary(rows, grouped, args.out_dir / "tried_subtask_accuracy_summary.md")
    draw_group_bars(grouped, args.out_dir / "hl_vs_tried_subtask_accuracy_by_group.png")
    draw_task_bars(rows, args.out_dir / "hl_vs_tried_subtask_accuracy_by_task.png")
    draw_avg_completed_before_failure(
        rows,
        args.out_dir / "avg_successful_subtasks_before_failure_by_task.png",
    )

    print(f"Wrote tried-subtask accuracy figures to {args.out_dir}")


if __name__ == "__main__":
    main()
