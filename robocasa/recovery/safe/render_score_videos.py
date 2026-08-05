"""Overlay causal SAFE scores on held-out RoboCasa rollout videos."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


SCORE_VARIANTS = ("auto", "unnormalized", "normalized")
TIMELINE_SCOPES = ("evaluation-window", "full-rollout")


def resolve_score_variant(record, requested="auto"):
    if requested not in SCORE_VARIANTS:
        raise ValueError(
            f"Unknown score variant {requested!r}; expected one of {SCORE_VARIANTS}"
        )
    if requested != "auto":
        return requested
    explicit = record.get("score_variant")
    if explicit in SCORE_VARIANTS[1:]:
        return explicit
    normalization = record.get("normalization")
    return (
        "normalized"
        if normalization not in (None, "", "none", "raw")
        else "unnormalized"
    )


def _timeline_values(record, timeline_scope="evaluation-window"):
    if timeline_scope not in TIMELINE_SCOPES:
        raise ValueError(
            f"Unknown timeline scope {timeline_scope!r}; expected one of "
            f"{TIMELINE_SCOPES}"
        )
    scores = np.asarray(record["scores"], dtype=float)
    steps = np.asarray(record["inference_environment_steps"], dtype=int)
    if timeline_scope == "evaluation-window":
        cutoff = min(int(record.get("task_min_step", len(scores))), len(scores))
        return scores[:cutoff], steps[:cutoff]
    if "full_scores" in record:
        scores = np.asarray(record["full_scores"], dtype=float)
        steps = np.asarray(
            record.get(
                "full_inference_environment_steps",
                record["inference_environment_steps"],
            ),
            dtype=int,
        )
    elif (
        record.get("normalization") not in (None, "", "none", "raw")
        and len(scores) <= int(record.get("task_min_step", len(scores)))
    ):
        raise ValueError(
            f"Normalized rollout {record.get('rollout_id')} contains only the "
            "matched evaluation window. Regenerate normalized_scores.jsonl "
            "with the current calibrate_seen_tasks implementation before "
            "using --timeline-scope full-rollout."
        )
    return scores, steps


def _score_arrays(
    record,
    calibration=None,
    timeline_scope="evaluation-window",
):
    scores, steps = _timeline_values(record, timeline_scope)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError(
            f"Rollout {record.get('rollout_id')} has an invalid score trajectory"
        )
    if steps.ndim != 1 or len(steps) != len(scores):
        raise ValueError(
            f"Rollout {record.get('rollout_id')} has {len(scores)} scores but "
            f"{len(steps)} inference environment steps"
        )
    if np.any(np.diff(steps) < 0):
        raise ValueError(
            f"Rollout {record.get('rollout_id')} has non-monotonic inference steps"
        )
    threshold = (
        threshold_for_length(calibration, len(scores))
        if calibration is not None
        else None
    )
    return scores, steps, threshold


def detector_prediction(
    record,
    calibration=None,
    timeline_scope="evaluation-window",
):
    if calibration is None:
        return None
    scores, _, threshold = _score_arrays(record, calibration, timeline_scope)
    return bool(np.any(scores >= threshold))


def detector_result_tag(
    record,
    calibration=None,
    *,
    ground_truth_failed=None,
    timeline_scope="evaluation-window",
):
    prediction = detector_prediction(record, calibration, timeline_scope)
    if prediction is None:
        return "detector-no-threshold"
    failed = bool(
        record["failed"] if ground_truth_failed is None else ground_truth_failed
    )
    return "detector-correct" if prediction == failed else "detector-incorrect"


def timeline_x_positions(
    record,
    total_frames,
    left,
    right,
    timeline_scope="evaluation-window",
):
    _, steps, _ = _score_arrays(
        record,
        timeline_scope=timeline_scope,
    )
    stride = int(record.get("video_frame_stride", 1))
    if stride < 1:
        raise ValueError("video_frame_stride must be positive")
    video_last_environment_step = max(1, (int(total_frames) - 1) * stride)
    clipped = np.clip(steps, 0, video_last_environment_step)
    return (left + clipped / video_last_environment_step * (right - left)).astype(int)


def _output_filename(
    task, identity, variant, ground_truth_failed, detector_tag, suffix
):
    ground_truth = "gt-failure" if ground_truth_failed else "gt-success"
    return (
        f"{task}--{identity}--{variant}--{ground_truth}--"
        f"{detector_tag}--{suffix}.mp4"
    )


def draw_overlay(
    frame,
    record,
    frame_index,
    total_frames,
    calibration=None,
    score_variant="auto",
    timeline_scope="evaluation-window",
):
    cv2 = require_cv2()
    height, width = frame.shape[:2]
    subtask_mode = bool(record.get("subtask_instruction"))
    overlay_height = min(
        height,
        max(190 if subtask_mode else 150, int(height * 0.31)),
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - overlay_height), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    scores, steps, threshold = _score_arrays(
        record,
        calibration,
        timeline_scope,
    )
    variant = resolve_score_variant(record, score_variant)
    env_step = frame_index * int(record.get("video_frame_stride", 1))
    current = int(np.searchsorted(steps, env_step, side="right") - 1)
    current = min(max(current, 0), len(scores) - 1)
    crossings = (
        np.flatnonzero(scores[: current + 1] >= threshold[: current + 1])
        if threshold is not None
        else []
    )
    first_alert = int(crossings[0]) if len(crossings) else None
    alert_active = first_alert is not None
    outcome = "FAILURE" if record["failed"] else "SUCCESS"
    color = (80, 80, 255) if record["failed"] else (80, 220, 80)
    parent_task = record.get("parent_task_name") or record["task_name"]
    scope_label = (
        "FULL ROLLOUT"
        if timeline_scope == "full-rollout"
        else "MATCHED EVALUATION WINDOW"
    )
    title = (
        f"{parent_task} | parent GT "
        f"{'FAILURE' if record.get('parent_rollout_failed', record['failed']) else 'SUCCESS'} "
        f"| SAFE {record['model']} seed {record['seed']} | {variant.upper()} | "
        f"{scope_label}"
    )
    cv2.putText(
        frame,
        title,
        (18, height - overlay_height + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    line_offset = 0
    if subtask_mode:
        instruction = str(record["subtask_instruction"])
        if len(instruction) > 78:
            instruction = instruction[:75] + "..."
        cv2.putText(
            frame,
            f"Active subtask: {instruction} | segment GT {outcome}",
            (18, height - overlay_height + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )
        line_offset = 28
    segment = record.get("subtask_safe_segment") or {}
    entry_step = segment.get("entry_environment_step")
    elapsed = None if entry_step is None else max(0, env_step - int(entry_step))
    elapsed_text = "" if elapsed is None else f"   elapsed_env_steps={elapsed}"
    cv2.putText(
        frame,
        f"inference {current + 1}   SAFE risk={scores[current]:.4f}   max-so-far={np.max(scores[:current + 1]):.4f}{elapsed_text}",
        (18, height - overlay_height + 58 + line_offset),
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
    if env_step > int(steps[-1]):
        ended = (
            "full score trajectory ended"
            if timeline_scope == "full-rollout"
            else "matched scored window ended"
        )
        status_text += f" | {ended} at env step {int(steps[-1])}"
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
    top, bottom = height - overlay_height + 72 + line_offset, height - 34
    top = min(top, bottom - 10)
    cv2.rectangle(frame, (left, top), (right, bottom), (100, 100, 100), 1)
    plot_values = scores if threshold is None else np.concatenate((scores, threshold))
    minimum, maximum = float(np.min(plot_values)), float(np.max(plot_values))
    if np.isclose(minimum, maximum):
        maximum = minimum + 1.0
    xs = timeline_x_positions(
        record,
        total_frames,
        left,
        right,
        timeline_scope,
    )
    ys = (bottom - (scores - minimum) / (maximum - minimum) * (bottom - top)).astype(
        int
    )
    if current >= 1:
        points = np.column_stack((xs[: current + 1], ys[: current + 1])).astype(
            np.int32
        )
        cv2.polylines(frame, [points], False, (80, 210, 255), 2, cv2.LINE_AA)
    if threshold is not None:
        threshold_ys = (
            bottom - (threshold - minimum) / (maximum - minimum) * (bottom - top)
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
    video_last_environment_step = max(
        1,
        (int(total_frames) - 1) * int(record.get("video_frame_stride", 1)),
    )
    cursor_x = int(
        left
        + min(env_step, video_last_environment_step)
        / video_last_environment_step
        * (right - left)
    )
    cv2.line(frame, (cursor_x, top), (cursor_x, bottom), (150, 150, 150), 1)
    if int(steps[-1]) < video_last_environment_step:
        cv2.line(
            frame,
            (int(xs[-1]), top),
            (int(xs[-1]), bottom),
            (200, 160, 80),
            1,
            cv2.LINE_AA,
        )
    cv2.circle(frame, (xs[current], ys[current]), 4, (0, 255, 255), -1)
    cv2.putText(
        frame,
        f"{maximum:.3g}",
        (left + 3, top + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        frame,
        f"{minimum:.3g}",
        (left + 3, bottom - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
    )
    return frame


def _segment_bounds(record):
    segment = record.get("subtask_safe_segment") or {}
    return (
        segment.get("entry_environment_step"),
        segment.get("end_environment_step"),
    )


def _active_subtask_record(records, environment_step):
    for record in records:
        entry, end = _segment_bounds(record)
        if entry is None or end is None:
            continue
        if int(entry) <= environment_step <= int(end):
            return record
    return None


def draw_parent_overlay(
    frame,
    records,
    frame_index,
    total_frames,
    calibration=None,
    score_variant="auto",
    timeline_scope="evaluation-window",
):
    stride = int(records[0].get("video_frame_stride", 1))
    environment_step = frame_index * stride
    active = _active_subtask_record(records, environment_step)
    if active is not None:
        return draw_overlay(
            frame,
            active,
            frame_index,
            total_frames,
            calibration,
            score_variant,
            timeline_scope,
        )

    cv2 = require_cv2()
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 92), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    first = records[0]
    parent_task = first.get("parent_task_name") or first["task_name"]
    parent_failed = bool(first.get("parent_rollout_failed", first["failed"]))
    cv2.putText(
        frame,
        f"{parent_task} | parent GT {'FAILURE' if parent_failed else 'SUCCESS'}",
        (18, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 80, 255) if parent_failed else (80, 220, 80),
        2,
        cv2.LINE_AA,
    )
    future = [
        record
        for record in records
        if _segment_bounds(record)[0] is not None
        and int(_segment_bounds(record)[0]) > environment_step
    ]
    status = (
        f"Waiting for subtask: {future[0].get('subtask_instruction', future[0]['task_name'])}"
        if future
        else "No active scored semantic subtask"
    )
    cv2.putText(
        frame,
        status[:110],
        (18, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return frame


def render_record(
    record,
    output_path,
    calibration=None,
    score_variant="auto",
    timeline_scope="evaluation-window",
):
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
        writer.write(
            draw_overlay(
                frame,
                record,
                index,
                frames,
                calibration,
                score_variant,
                timeline_scope,
            )
        )
        index += 1
    capture.release()
    writer.release()
    if index == 0 or not output_path.is_file():
        raise RuntimeError(f"Failed to render {source}")
    return output_path


def render_parent_record(
    records,
    output_path,
    calibration=None,
    score_variant="auto",
    timeline_scope="evaluation-window",
):
    cv2 = require_cv2()
    records = sorted(
        records,
        key=lambda record: (
            _segment_bounds(record)[0]
            if _segment_bounds(record)[0] is not None
            else 10**12,
            record.get("subtask_index", 10**12),
        ),
    )
    parent_ids = {record.get("parent_rollout_id") for record in records}
    if len(parent_ids) != 1 or None in parent_ids:
        raise ValueError("Parent rendering requires one non-empty parent_rollout_id")
    source = Path(records[0]["video_path"])
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
        writer.write(
            draw_parent_overlay(
                frame,
                records,
                index,
                frames,
                calibration,
                score_variant,
                timeline_scope,
            )
        )
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
    group_by_parent=False,
    score_variant="auto",
    timeline_scope="evaluation-window",
):
    if timeline_scope not in TIMELINE_SCOPES:
        raise ValueError(
            f"Unknown timeline scope {timeline_scope!r}; expected one of "
            f"{TIMELINE_SCOPES}"
        )
    records = [
        json.loads(line)
        for line in Path(scores_path).read_text().splitlines()
        if line.strip()
    ]
    records = [record for record in records if record["split"] == split]
    calibration = (
        load_calibration(calibration_path) if calibration_path is not None else None
    )
    if group_by_parent:
        grouped = defaultdict(list)
        for record in records:
            parent = record.get("parent_rollout_id")
            if not parent:
                raise ValueError(
                    "--group-by-parent requires parent_rollout_id on every record"
                )
            grouped[parent].append(record)
        groups = sorted(grouped.items())
        if max_videos is not None:
            groups = groups[:max_videos]
        outputs = []
        for parent, values in groups:
            first = values[0]
            parent_task = first.get("parent_task_name") or "parent"
            variant = resolve_score_variant(first, score_variant)
            parent_failed = bool(first.get("parent_rollout_failed", first["failed"]))
            parent_prediction = (
                None
                if calibration is None
                else any(
                    detector_prediction(value, calibration, timeline_scope)
                    for value in values
                )
            )
            detector_tag = (
                "detector-no-threshold"
                if parent_prediction is None
                else (
                    "detector-correct"
                    if parent_prediction == parent_failed
                    else "detector-incorrect"
                )
            )
            filename = _output_filename(
                parent_task,
                parent,
                variant,
                parent_failed,
                detector_tag,
                f"subtask-safe-{timeline_scope}",
            )
            outputs.append(
                render_parent_record(
                    values,
                    Path(output_dir) / filename,
                    calibration=calibration,
                    score_variant=score_variant,
                    timeline_scope=timeline_scope,
                )
            )
        return outputs
    if max_videos is not None:
        records = records[:max_videos]
    outputs = []
    for record in records:
        variant = resolve_score_variant(record, score_variant)
        detector_tag = detector_result_tag(
            record,
            calibration,
            timeline_scope=timeline_scope,
        )
        filename = _output_filename(
            record["task_name"],
            record["rollout_id"],
            variant,
            bool(record["failed"]),
            detector_tag,
            f"scores-{timeline_scope}",
        )
        outputs.append(
            render_record(
                record,
                Path(output_dir) / filename,
                calibration=calibration,
                score_variant=score_variant,
                timeline_scope=timeline_scope,
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
    parser.add_argument(
        "--group-by-parent",
        action="store_true",
        help="Render one full rollout video with its ordered semantic segments",
    )
    parser.add_argument(
        "--score-variant",
        choices=SCORE_VARIANTS,
        default="auto",
        help=(
            "Label output filenames and overlays as normalized or unnormalized; "
            "auto infers this from score-record metadata"
        ),
    )
    parser.add_argument(
        "--timeline-scope",
        choices=TIMELINE_SCOPES,
        default="evaluation-window",
        help=(
            "Render either the training-derived matched evaluation window or "
            "the complete recorded score trajectory. Full normalized rendering "
            "requires normalized score files generated by the current "
            "calibration code."
        ),
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
        group_by_parent=args.group_by_parent,
        score_variant=args.score_variant,
        timeline_scope=args.timeline_scope,
    )
    print(json.dumps([str(path) for path in outputs], indent=2))


if __name__ == "__main__":
    main()
