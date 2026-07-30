"""Overlay causal SAFE scores on held-out RoboCasa rollout videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .conformal import load_calibration, threshold_for_length
except ImportError:
    from conformal import load_calibration, threshold_for_length


def require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Score-video rendering requires OpenCV (opencv-python), available in the SAFE cluster environment."
        ) from error
    return cv2


def draw_overlay(frame, record, frame_index, total_frames, calibration=None):
    cv2 = require_cv2()
    height, width = frame.shape[:2]
    overlay_height = max(150, int(height * 0.27))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - overlay_height), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    scores = np.asarray(record["scores"], dtype=float)
    env_step = frame_index * int(record.get("video_frame_stride", 1))
    steps = np.asarray(record["inference_environment_steps"], dtype=int)
    current = int(np.searchsorted(steps, env_step, side="right") - 1)
    current = min(max(current, 0), len(scores) - 1)
    threshold = (
        threshold_for_length(calibration, len(scores))
        if calibration is not None
        else None
    )
    crossings = (
        np.flatnonzero(scores[: current + 1] >= threshold[: current + 1])
        if threshold is not None
        else []
    )
    first_alert = int(crossings[0]) if len(crossings) else None
    alert_active = first_alert is not None
    outcome = "FAILURE" if record["failed"] else "SUCCESS"
    color = (80, 80, 255) if record["failed"] else (80, 220, 80)
    title = f"{record['task_name']} | GT {outcome} | SAFE {record['model']} seed {record['seed']}"
    cv2.putText(frame, title, (18, height - overlay_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"inference {current + 1}/{len(scores)}   score={scores[current]:.4f}   max-so-far={np.max(scores[:current + 1]):.4f}",
        (18, height - overlay_height + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    if threshold is None:
        status_text = "Score only - no calibrated threshold supplied"
        status_color = (180, 180, 180)
    else:
        status = "ALERT" if alert_active else "MONITORING"
        alert_text = (
            f"first alert inference={first_alert + 1}"
            if first_alert is not None
            else "no alert yet"
        )
        status_text = (
            f"{status} | threshold={threshold[current]:.4f} | {alert_text} | "
            f"alpha={calibration['alpha']:.2f}"
        )
        status_color = (80, 80, 255) if alert_active else (180, 180, 180)
    cv2.putText(
        frame,
        status_text,
        (18, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        status_color,
        1,
        cv2.LINE_AA,
    )
    left, right = 18, width - 18
    top, bottom = height - overlay_height + 72, height - 34
    cv2.rectangle(frame, (left, top), (right, bottom), (100, 100, 100), 1)
    plot_values = scores if threshold is None else np.concatenate((scores, threshold))
    minimum, maximum = float(np.min(plot_values)), float(np.max(plot_values))
    if np.isclose(minimum, maximum):
        maximum = minimum + 1.0
    xs = np.linspace(left, right, len(scores)).astype(int)
    ys = (bottom - (scores - minimum) / (maximum - minimum) * (bottom - top)).astype(int)
    if current >= 1:
        points = np.column_stack((xs[: current + 1], ys[: current + 1])).astype(np.int32)
        cv2.polylines(frame, [points], False, (80, 210, 255), 2, cv2.LINE_AA)
    if threshold is not None:
        threshold_ys = (
            bottom
            - (threshold - minimum) / (maximum - minimum) * (bottom - top)
        ).astype(int)
        if current >= 1:
            points = np.column_stack(
                (xs[: current + 1], threshold_ys[: current + 1])
            ).astype(np.int32)
            cv2.polylines(
                frame,
                [points],
                False,
                (80, 80, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.circle(frame, (xs[current], ys[current]), 4, (0, 255, 255), -1)
    cv2.putText(frame, f"{maximum:.3g}", (left + 3, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(frame, f"{minimum:.3g}", (left + 3, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    return frame


def render_record(record, output_path, calibration=None):
    cv2 = require_cv2()
    source = Path(record["video_path"])
    if not source.is_file():
        raise FileNotFoundError(f"Missing rollout video: {source}")
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open output video writer: {output_path}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(draw_overlay(frame, record, index, frames, calibration))
        index += 1
    capture.release()
    writer.release()
    if index == 0 or not output_path.is_file():
        raise RuntimeError(f"Failed to render {source}")
    return output_path


def render_score_videos(
    scores_path,
    output_dir,
    *,
    split="test",
    max_videos=None,
    calibration_path=None,
):
    records = [
        json.loads(line)
        for line in Path(scores_path).read_text().splitlines()
        if line.strip()
    ]
    records = [record for record in records if record["split"] == split]
    calibration = (
        load_calibration(calibration_path)
        if calibration_path is not None
        else None
    )
    if max_videos is not None:
        records = records[:max_videos]
    outputs = []
    for record in records:
        filename = f"{record['task_name']}--{record['rollout_id']}--scores.mp4"
        outputs.append(
            render_record(
                record,
                Path(output_dir) / filename,
                calibration=calibration,
            )
        )
    return outputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument(
        "--calibration",
        help="Optional functional calibration JSON to overlay as a threshold",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outputs = render_score_videos(
        args.scores,
        args.output_dir,
        split=args.split,
        max_videos=args.max_videos,
        calibration_path=args.calibration,
    )
    print(json.dumps([str(path) for path in outputs], indent=2))


if __name__ == "__main__":
    main()
