"""Check the four passive XR-1 videos and their control and decision timestamps."""

import json
from collections import Counter
from pathlib import Path
import sys

from imageio_ffmpeg import count_frames_and_secs

from robocasa.recovery.safe.subtask_safe import RECOVERY_PREDICATES


def verify(root):
    errors = root / "errors.jsonl"
    if errors.exists() and errors.read_text().strip():
        raise ValueError(f"Collection recorded errors in {errors}")
    lines = (root / "manifest.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    counts = Counter(row["task_name"] for row in rows)
    if counts != {"ArrangeBreadBasket": 2, "CuttingToolSelection": 2}:
        raise ValueError(f"Expected two recordings per task, got {counts}")
    if len({row["rollout_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate rollout IDs")
    for row in rows:
        task, rollout = row["task_name"], row["rollout_id"]
        trace = json.loads((root / row["subtask_trace_path"]).read_text())
        n = row["num_environment_steps"]
        frames, _ = count_frames_and_secs(str(root / row["video_path"]))
        if n <= 0 or frames != n or trace.get("video_frame_environment_steps") != list(range(1, n + 1)):
            raise ValueError(f"{rollout}: expected one decoded post-action frame per step (got {frames}/{n})")
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
        print(f"{task} {rollout}: {outcome}, {frames} frames, {len(decisions)} XR-1 decisions")
    print("PASS: four synchronized XR-1 recordings")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
