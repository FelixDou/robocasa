"""Render synchronized XR1 matched-FPR detector trajectories on rollout videos."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DETECTORS = ("safe_only", "staged_safe_time", "time_only")
DETECTOR_LABELS = {
    "safe_only": "SAFE only",
    "staged_safe_time": "Staged SAFE + time",
    "time_only": "Time only",
}


def require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "XR1 annotated-video rendering requires OpenCV (opencv-python)."
        ) from error
    return cv2


def first_alert(record, detector, thresholds, safe_end, time_start):
    """Return the first causal alert as a one-based inference index."""
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector: {detector}")
    safe = np.asarray(record["safe_scores"], dtype=float)
    time = np.asarray(record["time_scores"], dtype=float)
    if safe.ndim != 1 or time.ndim != 1 or len(safe) != len(time) or not len(safe):
        raise ValueError("SAFE and time score trajectories must be non-empty 1D arrays")
    horizon = int(record["horizon"])
    inference = np.arange(1, len(safe) + 1)
    if detector == "safe_only":
        hits = np.flatnonzero(safe >= float(thresholds[detector][detector]))
        source = detector
    elif detector == "time_only":
        hits = np.flatnonzero(time >= float(thresholds[detector][detector]))
        source = detector
    else:
        early = np.flatnonzero(
            (inference <= int(np.ceil(float(safe_end) * horizon)))
            & (safe >= float(thresholds[detector]["early_safe"]))
        )
        late = np.flatnonzero(
            (inference >= int(np.ceil(float(time_start) * horizon)))
            & (time >= float(thresholds[detector]["late_time"]))
        )
        candidates = []
        if len(early):
            candidates.append((int(early[0]), "early_safe"))
        if len(late):
            candidates.append((int(late[0]), "late_time"))
        if not candidates:
            return None
        index, source = min(candidates)
        return {"step": index + 1, "source": source}
    if not len(hits):
        return None
    return {"step": int(hits[0]) + 1, "source": source}


def alert_fraction(record, detector, thresholds, safe_end, time_start, miss=None):
    alert = first_alert(record, detector, thresholds, safe_end, time_start)
    if alert is None:
        return miss
    return min(1.0, alert["step"] / int(record["horizon"]))


def _validated_trajectory(record):
    safe = np.asarray(record["safe_scores"], dtype=float)
    time = np.asarray(record["time_scores"], dtype=float)
    steps = np.asarray(record["inference_environment_steps"], dtype=int)
    if safe.ndim != 1 or time.ndim != 1 or steps.ndim != 1:
        raise ValueError("Score trajectories and inference steps must be 1D")
    if not len(safe) or len(safe) != len(time) or len(safe) != len(steps):
        raise ValueError("SAFE scores, time scores, and inference steps must align")
    if not np.all(np.isfinite(safe)) or not np.all(np.isfinite(time)):
        raise ValueError("Score trajectories contain non-finite values")
    if np.any(np.diff(steps) < 0):
        raise ValueError("Inference environment steps are not monotonic")
    horizon = int(record["horizon"])
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return safe, time, steps, horizon


def _put_text(cv2, canvas, text, position, scale=0.55, color=(235, 235, 235), thickness=1):
    cv2.putText(
        canvas,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _fit_video_frame(cv2, canvas, frame, bounds):
    left, top, right, bottom = bounds
    target_width, target_height = right - left, bottom - top
    height, width = frame.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    x = left + (target_width - resized_width) // 2
    y = top + (target_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    cv2.rectangle(canvas, (left, top), (right, bottom), (85, 85, 85), 1)


def _chart_point(rect, fraction, value, minimum, maximum):
    left, top, right, bottom = rect
    x = int(round(left + np.clip(fraction, 0.0, 1.0) * (right - left)))
    normalized = (float(value) - minimum) / (maximum - minimum)
    y = int(round(bottom - np.clip(normalized, 0.0, 1.0) * (bottom - top)))
    return x, y


def _draw_chart(
    cv2,
    canvas,
    rect,
    *,
    title,
    values,
    horizon,
    current_index,
    cursor_fraction,
    thresholds,
    active_window=None,
):
    left, top, right, bottom = rect
    cv2.rectangle(canvas, (left, top), (right, bottom), (25, 29, 34), -1)
    if active_window is not None:
        start, end, color = active_window
        x0 = int(left + start * (right - left))
        x1 = int(left + end * (right - left))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, top), (x1, bottom), color, -1)
        canvas[:] = cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0)

    threshold_values = [float(item[1]) for item in thresholds]
    minimum = min(float(np.min(values)), *threshold_values)
    maximum = max(float(np.max(values)), *threshold_values)
    span = maximum - minimum
    padding = 0.09 * span if span > 1e-8 else 0.5
    minimum -= padding
    maximum += padding

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(left + fraction * (right - left))
        cv2.line(canvas, (x, top), (x, bottom), (60, 65, 72), 1)
        _put_text(cv2, canvas, f"{fraction:.2g}", (x - 9, bottom + 18), 0.38, (170, 175, 180))
    for ratio in (0.0, 0.5, 1.0):
        y = int(bottom - ratio * (bottom - top))
        value = minimum + ratio * (maximum - minimum)
        cv2.line(canvas, (left, y), (right, y), (52, 57, 64), 1)
        _put_text(cv2, canvas, f"{value:.2f}", (left + 4, y - 4), 0.36, (170, 175, 180))

    fractions = np.arange(1, len(values) + 1, dtype=float) / horizon
    shown = min(current_index + 1, len(values))
    if shown >= 2:
        points = np.asarray(
            [
                _chart_point(rect, fractions[index], values[index], minimum, maximum)
                for index in range(shown)
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], False, (0, 190, 255), 3, cv2.LINE_AA)
    current = _chart_point(
        rect,
        fractions[shown - 1],
        values[shown - 1],
        minimum,
        maximum,
    )
    cv2.circle(canvas, current, 5, (0, 240, 255), -1, cv2.LINE_AA)

    for label, value, start, end, color in thresholds:
        p0 = _chart_point(rect, start, value, minimum, maximum)
        p1 = _chart_point(rect, end, value, minimum, maximum)
        cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
        _put_text(cv2, canvas, label, (p0[0] + 5, max(top + 14, p0[1] - 5)), 0.38, color)

    cursor_x = int(left + np.clip(cursor_fraction, 0.0, 1.0) * (right - left))
    cv2.line(canvas, (cursor_x, top), (cursor_x, bottom), (225, 225, 225), 1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (105, 110, 118), 1)
    _put_text(cv2, canvas, title, (left, top - 12), 0.58, (245, 245, 245), 1)


def _alert_status(record, detector, alert, environment_step, inference_steps):
    if alert is None:
        return "NO ALARM", False
    alert_environment_step = int(inference_steps[alert["step"] - 1])
    active = environment_step >= alert_environment_step
    source = alert["source"].replace("_", " ").upper()
    fraction = alert["step"] / int(record["horizon"])
    if active:
        return f"ALERT - {source} at {fraction:.3f}", True
    return f"scheduled first alarm at {fraction:.3f}", False


def draw_matched_fpr_frame(
    frame,
    record,
    frame_index,
    total_frames,
    thresholds,
    safe_end,
    time_start,
    *,
    video_frame_stride,
    reason=None,
):
    """Return a 1280x720 frame with causal SAFE/time trajectories."""
    cv2 = require_cv2()
    safe, time, inference_steps, horizon = _validated_trajectory(record)
    canvas = np.full((720, 1280, 3), (17, 20, 24), dtype=np.uint8)
    environment_step = int(frame_index) * int(video_frame_stride)
    current_index = int(np.searchsorted(inference_steps, environment_step, side="right") - 1)
    current_index = min(max(current_index, 0), len(safe) - 1)
    replan_steps = max(1, int(record.get("replan_steps", 16)))
    cursor_fraction = min(1.0, environment_step / max(1, replan_steps * horizon))

    outcome = "FAILURE" if record["failed"] else "SUCCESS"
    outcome_color = (75, 90, 245) if record["failed"] else (85, 210, 100)
    reason_text = "" if not reason else f" | {str(reason).replace('_', ' ')}"
    _put_text(
        cv2,
        canvas,
        f"{record['task_name']} | ground truth: {outcome}{reason_text}",
        (30, 42),
        0.82,
        outcome_color,
        2,
    )
    _put_text(
        cv2,
        canvas,
        (
            f"env step {environment_step} | inference {current_index + 1}/{len(safe)} "
            f"| task-horizon fraction {cursor_fraction:.3f} | frame {frame_index + 1}/{total_frames}"
        ),
        (30, 70),
        0.48,
        (205, 210, 215),
    )

    _fit_video_frame(cv2, canvas, frame, (30, 95, 620, 490))
    alerts = {
        detector: first_alert(record, detector, thresholds, safe_end, time_start)
        for detector in DETECTORS
    }
    y = 535
    for detector in DETECTORS:
        status, active = _alert_status(
            record,
            detector,
            alerts[detector],
            environment_step,
            inference_steps,
        )
        color = (70, 90, 245) if active else (175, 180, 185)
        _put_text(
            cv2,
            canvas,
            f"{DETECTOR_LABELS[detector]}: {status}",
            (45, y),
            0.56,
            color,
            2 if active else 1,
        )
        y += 35
    _put_text(
        cv2,
        canvas,
        "Raw rollout is shown above; plots update only when a genuine policy inference occurs.",
        (45, 675),
        0.43,
        (165, 170, 175),
    )

    _draw_chart(
        cv2,
        canvas,
        (675, 120, 1245, 335),
        title="Normalized SAFE ensemble score",
        values=safe,
        horizon=horizon,
        current_index=current_index,
        cursor_fraction=cursor_fraction,
        thresholds=(
            (
                "SAFE-only threshold",
                float(thresholds["safe_only"]["safe_only"]),
                0.0,
                1.0,
                (0, 150, 255),
            ),
            (
                "staged early threshold",
                float(thresholds["staged_safe_time"]["early_safe"]),
                0.0,
                float(safe_end),
                (70, 210, 110),
            ),
        ),
        active_window=(0.0, float(safe_end), (30, 115, 50)),
    )
    _draw_chart(
        cv2,
        canvas,
        (675, 430, 1245, 645),
        title="Task-conditioned elapsed-time risk",
        values=time,
        horizon=horizon,
        current_index=current_index,
        cursor_fraction=cursor_fraction,
        thresholds=(
            (
                "time-only threshold",
                float(thresholds["time_only"]["time_only"]),
                0.0,
                1.0,
                (235, 145, 45),
            ),
            (
                "staged late threshold",
                float(thresholds["staged_safe_time"]["late_time"]),
                float(time_start),
                1.0,
                (70, 210, 110),
            ),
        ),
        active_window=(float(time_start), 1.0, (90, 75, 25)),
    )
    _put_text(
        cv2,
        canvas,
        (
            f"Staged windows: SAFE 0-{float(safe_end):.2f}; no staged alarm "
            f"{float(safe_end):.2f}-{float(time_start):.2f}; time {float(time_start):.2f}-1"
        ),
        (675, 690),
        0.44,
        (190, 195, 200),
    )
    return canvas


def render_matched_fpr_video(
    record,
    source_path,
    output_path,
    thresholds,
    safe_end,
    time_start,
    *,
    video_frame_stride,
    reason=None,
):
    """Render one rollout and its causal detector trajectories to MP4."""
    cv2 = require_cv2()
    source_path = Path(source_path)
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source rollout video: {source_path}")
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source rollout video: {source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open annotated-video writer: {output_path}")
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(
            draw_matched_fpr_frame(
                frame,
                record,
                written,
                total_frames,
                thresholds,
                safe_end,
                time_start,
                video_frame_stride=video_frame_stride,
                reason=reason,
            )
        )
        written += 1
    capture.release()
    writer.release()
    if written == 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to render annotated video: {source_path}")
    return {
        "path": output_path,
        "frames": written,
        "fps": fps,
        "width": 1280,
        "height": 720,
    }
