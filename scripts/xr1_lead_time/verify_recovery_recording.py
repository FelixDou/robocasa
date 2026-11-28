"""Verify passive XR-1 recordings: video sync, decisions, and recovery observations."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from robocasa.recovery.safe.subtask_safe import RECOVERY_PREDICATES


def _count_frames(video_path):
    try:
        from imageio_ffmpeg import count_frames_and_secs
    except ImportError:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        count = 0
        try:
            while cap.read()[0]:
                count += 1
        finally:
            cap.release()
        return count
    frames, _ = count_frames_and_secs(str(video_path))
    return int(frames)


def _expected_frame_steps(num_steps, stride):
    return [i + 1 for i in range(num_steps) if i % stride == 0 or i == num_steps - 1]


def verify(root, expected_tasks=None, expected_per_task=None, stride=None):
    root = Path(root)
    errors = root / "errors.jsonl"
    if errors.exists() and errors.read_text().strip():
        raise ValueError(f"Collection recorded errors in {errors}")

    summary_path = root / "summary.json"
    summary_cfg = json.loads(summary_path.read_text()).get("config", {}) if summary_path.exists() else {}
    if expected_tasks is None:
        expected_tasks = summary_cfg.get("tasks") or ["ArrangeBreadBasket", "CuttingToolSelection"]
    if expected_per_task is None:
        expected_per_task = summary_cfg.get("xiaomi_num_trials") or 2
    if stride is None:
        stride = int(summary_cfg.get("video_frame_stride") or 1)

    lines = (root / "manifest.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    counts = Counter(row["task_name"] for row in rows)
    expected_counts = {task: int(expected_per_task) for task in expected_tasks}
    if dict(counts) != expected_counts:
        raise ValueError(f"Expected counts {expected_counts}, got {dict(counts)}")
    if len({row["rollout_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate rollout IDs")

    per_task_outcomes = Counter()
    for row in rows:
        task, rollout = row["task_name"], row["rollout_id"]
        if task not in RECOVERY_PREDICATES:
            raise ValueError(f"{rollout}: task {task} is not in RECOVERY_PREDICATES")
        trace = json.loads((root / row["subtask_trace_path"]).read_text())
        n = int(row["num_environment_steps"])
        expected_steps = _expected_frame_steps(n, stride)
        frames = _count_frames(root / row["video_path"])
        actual_steps = trace.get("video_frame_environment_steps")
        if n <= 0 or frames != len(expected_steps) or actual_steps != expected_steps:
            raise ValueError(
                f"{rollout}: expected {len(expected_steps)} post-action frames at stride {stride} "
                f"(got {frames} decoded frames, {len(actual_steps or [])} timestamps)"
            )
        decisions = [r["environment_step"] for r in trace["inference_records"]]
        if not decisions or decisions[0] != 0 or decisions != row["inference_environment_steps"]:
            raise ValueError(f"{rollout}: policy decision timestamps disagree")
        history = trace["recovery_observations"]
        for name in RECOVERY_PREDICATES[task]:
            if any(type(item["values"].get(name)) is not bool for item in history):
                raise ValueError(f"{rollout}: missing simulator observation for {name}")
        if not history or history[0]["environment_step"] != 0 or history[-1]["environment_step"] != n:
            raise ValueError(f"{rollout}: recovery history must cover reset through termination")
        outcome = "failed" if row["failed"] else "succeeded"
        per_task_outcomes[(task, outcome)] += 1
        grasp_keys = [k for k in RECOVERY_PREDICATES[task] if "grasped" in k]
        grasp_hits = {
            k: sum(
                1
                for prev, cur in zip(history, history[1:])
                if not prev["values"][k] and cur["values"][k]
            )
            for k in grasp_keys
        }
        print(
            f"{task} {rollout}: {outcome}, {frames}/{n} frames (stride={stride}), "
            f"{len(decisions)} decisions, {len(history)} obs changes, grasps={grasp_hits}"
        )

    summary_line = ", ".join(
        f"{t}: {per_task_outcomes[(t, 'succeeded')]}S/{per_task_outcomes[(t, 'failed')]}F"
        for t in expected_tasks
    )
    print(f"PASS: {len(rows)} synchronized XR-1 recordings ({summary_line})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--expected-per-task", type=int)
    parser.add_argument("--stride", type=int)
    args = parser.parse_args(argv)
    verify(
        args.root,
        expected_tasks=args.tasks,
        expected_per_task=args.expected_per_task,
        stride=args.stride,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
