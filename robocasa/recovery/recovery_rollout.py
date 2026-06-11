"""Policy-agnostic rollout recovery helpers.

This module adds a thin recovery layer around an existing policy evaluator. It
first runs the original high-level instruction. If that rollout fails, it can
restore the last good state, restore only the robot state, or continue from the
failure state, then retry only the currently failed subtask.
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

import numpy as np

from robocasa.recovery.subtask_eval import (
    build_subtask_trace,
    get_subtask_eval,
    summarize_subtask_rollout,
)


class RecoveryMode(str, Enum):
    """State handling mode used before retrying the failed subtask."""

    EEF_TO_LAST_GOOD = "eef_to_last_good"
    ENV_TO_LAST_GOOD = "env_to_last_good"
    CONTINUE_FROM_FAILURE = "continue_from_failure"


RECOVERY_MODE_ALIASES = {
    "eef": RecoveryMode.EEF_TO_LAST_GOOD,
    "move_eef": RecoveryMode.EEF_TO_LAST_GOOD,
    "move_end_effector": RecoveryMode.EEF_TO_LAST_GOOD,
    "eef_to_last_good": RecoveryMode.EEF_TO_LAST_GOOD,
    "env": RecoveryMode.ENV_TO_LAST_GOOD,
    "reset_env": RecoveryMode.ENV_TO_LAST_GOOD,
    "reset_environment": RecoveryMode.ENV_TO_LAST_GOOD,
    "env_to_last_good": RecoveryMode.ENV_TO_LAST_GOOD,
    "continue": RecoveryMode.CONTINUE_FROM_FAILURE,
    "continue_from_failure": RecoveryMode.CONTINUE_FROM_FAILURE,
    "none": RecoveryMode.CONTINUE_FROM_FAILURE,
}


@dataclass
class RecoveryConfig:
    """Configuration for a recovery-after-failure rollout."""

    mode: object = RecoveryMode.CONTINUE_FROM_FAILURE
    high_level_horizon: int = 400
    subtask_horizon: int = 120
    match_recovery_horizon_to_no_progress: bool = False
    stuck_patience: int = 10
    include_trace: bool = True
    video_separator_frames: int = 80
    video_separator_text: str | None = None


def normalize_recovery_mode(mode):
    if isinstance(mode, RecoveryMode):
        return mode
    try:
        return RECOVERY_MODE_ALIASES[str(mode)]
    except KeyError as exc:
        valid = ", ".join(sorted(RECOVERY_MODE_ALIASES))
        raise ValueError(
            f"Unknown recovery mode {mode!r}. Valid modes: {valid}"
        ) from exc


def call_policy(policy, obs, instruction=None):
    """
    Invoke a policy with the common evaluator call conventions.

    The policy can be a plain callable, or expose ``predict``, ``select_action``,
    or ``get_action``. When possible, the subtask instruction is passed as a
    keyword so VLA policies that separate observations from language can use it.
    """
    if callable(policy):
        try:
            return policy(obs, instruction=instruction)
        except TypeError:
            return policy(obs)

    for method_name in ("predict", "select_action", "get_action"):
        method = getattr(policy, method_name, None)
        if method is None:
            continue
        try:
            result = method(obs, instruction=instruction)
        except TypeError:
            result = method(obs)
        if isinstance(result, tuple):
            return result[0]
        return result

    raise TypeError(
        "policy must be callable or expose predict/select_action/get_action"
    )


def set_observation_instruction(obs, instruction):
    """Return an observation copy with VLA language fields set to instruction."""
    if instruction is None:
        return obs
    obs = dict(obs)
    for key in ("annotation.human.task_description", "language"):
        if key in obs:
            obs[key] = instruction
    return obs


def _inner_env(env):
    return getattr(env, "env", env)


def _sim_env(env):
    inner = _inner_env(env)
    if hasattr(inner, "sim"):
        return inner
    return getattr(inner, "env", inner)


def _get_sim(env):
    sim_holder = _sim_env(env)
    return getattr(sim_holder, "sim", None)


def _refresh_sim_visuals(env, num_forwards=1):
    sim = _get_sim(env)
    sim_holder = _sim_env(env)
    if sim is not None:
        for _ in range(max(1, int(num_forwards))):
            try:
                sim.forward()
            except Exception:
                break
    if hasattr(sim_holder, "update_sites"):
        try:
            sim_holder.update_sites()
        except Exception:
            pass
    if hasattr(sim_holder, "update_state"):
        try:
            sim_holder.update_state()
        except Exception:
            pass


def _capture_state(env):
    if hasattr(env, "get_state"):
        state = env.get_state()
    elif hasattr(_inner_env(env), "get_state"):
        state = _inner_env(env).get_state()
    else:
        sim = _get_sim(env)
        if sim is None:
            return None
        state = {"states": np.array(sim.get_state().flatten())}
        if hasattr(sim.model, "get_xml"):
            state["model"] = sim.model.get_xml()

    snapshot = deepcopy(state)
    sim = _get_sim(env)
    if sim is not None:
        try:
            sim_state = sim.get_state()
            snapshot["_sim_state_qpos"] = np.array(sim_state.qpos).copy()
            snapshot["_sim_state_qvel"] = np.array(sim_state.qvel).copy()
        except Exception:
            pass
    return snapshot


def _reset_to_state(env, state):
    clean_state = {
        key: value for key, value in state.items() if not key.startswith("_sim_state_")
    }
    if hasattr(env, "reset_to"):
        return env.reset_to(clean_state)
    if hasattr(_inner_env(env), "reset_to"):
        return _inner_env(env).reset_to(clean_state)

    sim = _get_sim(env)
    if sim is None:
        raise RuntimeError("Environment does not expose reset_to() or sim state")
    if "states" not in clean_state:
        raise RuntimeError("Captured state does not contain flattened simulator state")
    sim.set_state_from_flattened(clean_state["states"])
    sim.forward()
    sim_holder = _sim_env(env)
    if hasattr(sim_holder, "update_sites"):
        sim_holder.update_sites()
    if hasattr(sim_holder, "update_state"):
        sim_holder.update_state()
    return None


def _get_obs_after_state_change(env, fallback_obs=None):
    if hasattr(env, "get_current_observation"):
        return env.get_current_observation()
    if hasattr(env, "get_observation") and hasattr(_inner_env(env), "_get_observations"):
        raw_obs = _inner_env(env)._get_observations(force_update=True)
        return env.get_observation(raw_obs)
    inner = _inner_env(env)
    if hasattr(inner, "_get_observations"):
        return inner._get_observations(force_update=True)
    return fallback_obs


def _set_env_instruction(env, instruction):
    setter = getattr(env, "set_task_description", None)
    if setter is not None:
        setter(instruction)


def _robot_joint_indices(env):
    sim = _get_sim(env)
    sim_holder = _sim_env(env)
    if sim is None or not hasattr(sim.model, "njnt"):
        return [], []

    prefixes = []
    for robot in getattr(sim_holder, "robots", []):
        robot_model = getattr(robot, "robot_model", None)
        prefix = getattr(robot_model, "naming_prefix", None)
        if prefix:
            prefixes.append(prefix)
    if not prefixes:
        prefixes = ["robot0_", "mobilebase0_", "gripper0_"]

    qpos_indices = []
    qvel_indices = []
    for joint_id in range(sim.model.njnt):
        joint_name = sim.model.joint_id2name(joint_id)
        if joint_name is None or not any(joint_name.startswith(p) for p in prefixes):
            continue

        qpos_addr = sim.model.jnt_qposadr[joint_id]
        qvel_addr = sim.model.jnt_dofadr[joint_id]
        joint_type = sim.model.jnt_type[joint_id]
        if joint_type == 0:  # free joint
            qpos_width, qvel_width = 7, 6
        elif joint_type == 1:  # ball joint
            qpos_width, qvel_width = 4, 3
        else:
            qpos_width, qvel_width = 1, 1
        qpos_indices.extend(range(qpos_addr, qpos_addr + qpos_width))
        qvel_indices.extend(range(qvel_addr, qvel_addr + qvel_width))
    return sorted(set(qpos_indices)), sorted(set(qvel_indices))


def _restore_robot_state_only(env, good_state):
    sim = _get_sim(env)
    if sim is None:
        raise RuntimeError("Environment does not expose simulator state")

    good_qpos = good_state.get("_sim_state_qpos")
    good_qvel = good_state.get("_sim_state_qvel")
    if good_qpos is None or good_qvel is None:
        _reset_to_state(env, good_state)
        return

    qpos_indices, qvel_indices = _robot_joint_indices(env)
    if not qpos_indices and not qvel_indices:
        _reset_to_state(env, good_state)
        return

    current = sim.get_state()
    qpos = np.array(current.qpos).copy()
    qvel = np.array(current.qvel).copy()
    qpos[qpos_indices] = good_qpos[qpos_indices]
    qvel[qvel_indices] = good_qvel[qvel_indices]
    try:
        current.qpos[:] = qpos
        current.qvel[:] = qvel
        sim.set_state(current)
    except Exception:
        if not hasattr(current, "_replace"):
            raise
        sim.set_state(current._replace(qpos=qpos, qvel=qvel))
    sim.forward()
    sim_holder = _sim_env(env)
    if hasattr(sim_holder, "update_sites"):
        sim_holder.update_sites()
    if hasattr(sim_holder, "update_state"):
        sim_holder.update_state()


def apply_recovery_mode(env, mode, last_good_state):
    """Apply the requested recovery mode and return a short metadata payload."""
    mode = normalize_recovery_mode(mode)
    if mode == RecoveryMode.CONTINUE_FROM_FAILURE:
        return {"mode": mode.value, "state_restored": False}
    if last_good_state is None:
        return {"mode": mode.value, "state_restored": False, "reason": "no_state"}
    if mode == RecoveryMode.ENV_TO_LAST_GOOD:
        _reset_to_state(env, last_good_state)
        return {"mode": mode.value, "state_restored": True}
    if mode == RecoveryMode.EEF_TO_LAST_GOOD:
        _restore_robot_state_only(env, last_good_state)
        return {"mode": mode.value, "state_restored": True, "robot_only": True}
    raise ValueError(f"Unhandled recovery mode: {mode}")


def _is_task_success(info, reward=None, env=None):
    if info and bool(info.get("success", False)):
        return True
    if reward is not None and reward > 0:
        return True
    inner = _inner_env(env) if env is not None else None
    return bool(hasattr(inner, "_check_success") and inner._check_success())


def _subtask_is_complete(subtask_eval, subtask_name):
    if subtask_name is None:
        return False
    predicate = (subtask_eval or {}).get("predicates", {}).get(subtask_name, {})
    return bool(predicate.get("value", False))


def _subtask_instruction(subtask_eval, subtask_name):
    if subtask_name is None:
        return None
    predicate = (subtask_eval or {}).get("predicates", {}).get(subtask_name, {})
    return predicate.get("description") or f"Complete this subtask: {subtask_name}."


def _ordered_current_subtask_from_eval(subtask_eval):
    entry = _latest_ordered_trace_entry([subtask_eval])
    if not entry:
        return None
    return entry.get("ordered_current_subtask") or entry.get("current_subtask_estimate")


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, reward, bool(done), info


def _is_likely_corrupt_video_frame(frame):
    """Detect intermittent EGL/offscreen render garbage before writing videos."""
    frame = np.asarray(frame)
    if frame.ndim < 3 or frame.shape[0] < 2 or frame.shape[1] < 2:
        return True
    frame = _normalize_video_frame(frame)
    rgb = frame[..., :3].astype(np.float32)
    luminance = rgb.mean(axis=2)
    channel_spread = rgb.max(axis=2) - rgb.min(axis=2)

    dark_ratio = float((luminance < 12).mean())
    bright_ratio = float((luminance > 50).mean())
    color_speck_ratio = float(((channel_spread > 35) & (luminance > 20)).mean())
    saturated_speck_ratio = float(
        ((channel_spread > 50) & (rgb.max(axis=2) > 80) & (luminance < 120)).mean()
    )

    # EGL framebuffer garbage often appears as a mostly black image with sparse
    # colored RGB speckles or horizontal text-like bands.
    if dark_ratio > 0.80 and (
        bright_ratio > 0.01
        or color_speck_ratio > 0.005
        or saturated_speck_ratio > 0.002
    ):
        return True

    if dark_ratio > 0.98:
        return True

    sample = rgb[::4, ::4]
    dx = np.abs(sample[:, 1:] - sample[:, :-1])
    dy = np.abs(sample[1:] - sample[:-1])
    high_dx = float((dx > 40).mean())
    high_dy = float((dy > 40).mean())
    mean_dx = float(dx.mean())
    mean_dy = float(dy.mean())
    return (
        (high_dx > 0.25 and high_dy > 0.25)
        or (mean_dx > 30 and mean_dy > 30)
        or (high_dx > 0.45 and mean_dx > 20)
        or (high_dy > 0.45 and mean_dy > 20)
    )


def _resize_video_frame(frame, height, width):
    if height is None or width is None:
        return frame
    height = int(height)
    width = int(width)
    if frame.shape[:2] == (height, width):
        return np.ascontiguousarray(frame)
    try:
        from PIL import Image

        image = Image.fromarray(frame)
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        return np.ascontiguousarray(np.asarray(image.resize((width, height), resample)))
    except Exception:
        return np.ascontiguousarray(frame)


def _coerce_obs_video_frame(value):
    try:
        frame = np.asarray(value)
    except Exception:
        return None
    if not np.issubdtype(frame.dtype, np.number):
        return None

    while frame.ndim > 3:
        if frame.shape[0] == 1:
            frame = frame[0]
        else:
            frame = frame[-1]

    if frame.ndim == 3 and frame.shape[-1] in (1, 3, 4):
        return frame
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        return np.moveaxis(frame, 0, -1)
    if frame.ndim == 2:
        return frame
    return None


def _obs_video_key_candidates(obs, camera_name):
    candidates = []
    if camera_name:
        candidates.extend(
            [
                f"video.{camera_name}",
                f"{camera_name}_image",
                camera_name,
            ]
        )
        if camera_name.endswith("_image"):
            candidates.append(camera_name[: -len("_image")])
        if camera_name.startswith("video."):
            candidates.append(camera_name[len("video.") :])

    # The OpenPI Gym wrapper exposes left/right/wrist cameras as observations,
    # while the historical video default used a render-only center camera.
    candidates.extend(
        [
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
            "robot0_agentview_left_image",
            "robot0_agentview_right_image",
            "robot0_eye_in_hand_image",
        ]
    )

    for key in obs.keys():
        if (
            isinstance(key, str)
            and (key.startswith("video.") or key.endswith("_image"))
            and key not in candidates
        ):
            candidates.append(key)
    return candidates


def _extract_video_frame_from_obs(obs, camera_name):
    if not isinstance(obs, dict):
        return None
    for key in _obs_video_key_candidates(obs, camera_name):
        if key not in obs:
            continue
        frame = _coerce_obs_video_frame(obs[key])
        if frame is not None:
            return frame
    return None


def _append_video_frame(env, video_writer, camera_name, height, width, previous_frame=None):
    return _append_video_frame_from_env(
        env,
        video_writer,
        camera_name=camera_name,
        height=height,
        width=width,
        previous_frame=previous_frame,
        prefer_env_render=True,
    )


def _normalize_video_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[2] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            max_value = float(np.nanmax(frame)) if frame.size else 1.0
            if max_value <= 1.0:
                frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _append_video_frame_from_env(
    env,
    video_writer,
    camera_name,
    height,
    width,
    obs=None,
    previous_frame=None,
    prefer_env_render=True,
    reuse_corrupt_previous=True,
    render_attempts=1,
):
    if video_writer is None:
        return None

    video_img = None
    for attempt_i in range(max(1, int(render_attempts))):
        video_img = _extract_video_frame_from_obs(obs, camera_name)
        if attempt_i:
            _refresh_sim_visuals(env)
        if video_img is None and prefer_env_render:
            try:
                video_img = env.render()
            except Exception:
                video_img = None

        if video_img is None:
            sim = _get_sim(env)
            if sim is None:
                return None
            try:
                video_img = sim.render(
                    height=height, width=width, camera_name=camera_name
                )[::-1]
            except Exception:
                video_img = None

        try:
            video_img = _normalize_video_frame(video_img)
        except Exception:
            video_img = None

        if video_img is None:
            continue
        video_img = _resize_video_frame(video_img, height, width)
        if not _is_likely_corrupt_video_frame(video_img):
            break
        if (
            reuse_corrupt_previous
            and previous_frame is not None
            and not _is_likely_corrupt_video_frame(previous_frame)
        ):
            video_img = previous_frame
            break
        video_img = None


    if video_img is None:
        return None

    video_img = _normalize_video_frame(video_img)
    video_img = _resize_video_frame(video_img, height, width)
    video_writer.append_data(video_img)
    return video_img


def _load_separator_font(height, size=None):
    font_size = size or max(18, height // 18)
    try:
        from PIL import ImageFont

        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            from PIL import ImageFont

            return ImageFont.load_default()
        except Exception:
            return None


def _wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if current and bbox[2] - bbox[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def _make_separator_frame(frame, text):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return np.ascontiguousarray(frame)

    frame = np.asarray(frame)
    height, width = frame.shape[:2]
    image = Image.new("RGB", (width, height), color=(18, 20, 22))
    draw = ImageDraw.Draw(image)
    font = None
    lines = []
    line_boxes = []
    line_heights = []
    line_gap = 0
    total_height = 0
    for font_size in range(max(16, height // 20), 11, -2):
        candidate_font = _load_separator_font(height, size=font_size)
        if candidate_font is None:
            continue
        candidate_lines = _wrap_text(
            draw,
            text,
            candidate_font,
            max_width=int(width * 0.84),
        )
        candidate_boxes = [
            draw.textbbox((0, 0), line or " ", font=candidate_font)
            for line in candidate_lines
        ]
        candidate_heights = [box[3] - box[1] for box in candidate_boxes]
        candidate_gap = max(5, font_size // 3)
        candidate_total_height = sum(candidate_heights) + candidate_gap * max(
            0, len(candidate_lines) - 1
        )
        font = candidate_font
        lines = candidate_lines
        line_boxes = candidate_boxes
        line_heights = candidate_heights
        line_gap = candidate_gap
        total_height = candidate_total_height
        if candidate_total_height <= height * 0.82:
            break
    if font is None:
        return np.ascontiguousarray(frame)
    y = (height - total_height) // 2

    for line, box, line_height in zip(lines, line_boxes, line_heights):
        if not line:
            y += line_height + line_gap
            continue
        line_width = box[2] - box[0]
        x = (width - line_width) // 2
        draw.text((x, y), line, fill=(244, 246, 248), font=font)
        y += line_height + line_gap

    return np.ascontiguousarray(np.asarray(image))


def _append_video_separator(video_writer, frame, num_frames=0, text=""):
    if video_writer is None:
        return
    if frame is None or num_frames <= 0:
        return
    separator_frame = _make_separator_frame(frame, text)
    for _ in range(num_frames):
        video_writer.append_data(separator_frame)


def _predicate_description(subtask_eval, name):
    predicate = (subtask_eval or {}).get("predicates", {}).get(name, {})
    return predicate.get("description") or str(name).replace("_", " ")


def _format_description_list(subtask_eval, names, empty="none", max_items=4):
    if not names:
        return empty
    descriptions = [_predicate_description(subtask_eval, name) for name in names]
    if len(descriptions) > max_items:
        shown = descriptions[:max_items]
        shown.append(f"+{len(descriptions) - max_items} more")
        descriptions = shown
    return "; ".join(descriptions)


def _make_recovery_separator_text(
    header_text,
    high_level_summary,
    subtask_name,
    instruction,
):
    final_eval = high_level_summary.get("final_subtask_eval")
    completed = high_level_summary.get("ordered_completed_required_subtasks", [])
    failure_modes = high_level_summary.get("failure_modes", [])
    failed_preconditions = high_level_summary.get("failed_preconditions_ever", [])
    diagnostics = ", ".join(failure_modes) if failure_modes else "none"
    if failed_preconditions:
        diagnostics = (
            f"{diagnostics}; failed preconditions: "
            f"{_format_description_list(final_eval, failed_preconditions)}"
        )
    return "\n".join(
        [
            header_text,
            "",
            "Successful subtasks:",
            _format_description_list(final_eval, completed),
            "",
            "Failure diagnostic:",
            diagnostics,
            "",
            "Recovering now:",
            instruction or _predicate_description(final_eval, subtask_name),
        ]
    )


def _recovery_separator_header(mode):
    if mode == RecoveryMode.ENV_TO_LAST_GOOD:
        return "Environment is resetting to the last successful state"
    if mode == RecoveryMode.EEF_TO_LAST_GOOD:
        return "Robot is moving back to the last successful state"
    if mode == RecoveryMode.CONTINUE_FROM_FAILURE:
        return "Robot is continuing from the failure state"
    return "Recovery is starting"


def _latest_ordered_trace_entry(subtask_evals):
    trace = build_subtask_trace(subtask_evals)
    for entry in reversed(trace):
        if entry.get("subtask_eval_available"):
            return entry
    return None


def run_recovery_after_failed_rollout(
    policy,
    env,
    config=None,
    initial_obs=None,
    video_writer=None,
    video_camera_name="robot0_agentview_center",
    video_height=512,
    video_width=768,
    video_direct_sim_render=False,
):
    """
    Run a high-level rollout, then retry only the failed subtask if needed.

    Args:
        policy: VLA policy callable or object exposing predict/select_action/get_action.
        env: Gymnasium-style or robosuite-style environment.
        config: ``RecoveryConfig`` or dict.
        initial_obs: Optional already-reset observation. If omitted, ``env.reset()``
            is called.
        video_writer: Optional imageio writer. Frames are appended after each
            high-level and recovery step.

    Returns:
        A dictionary with the high-level rollout summary, chosen subtask prompt,
        recovery metadata, and subtask retry result.
    """
    if config is None:
        config = RecoveryConfig()
    elif isinstance(config, dict):
        config = RecoveryConfig(**config)
    mode = normalize_recovery_mode(config.mode)

    if initial_obs is None:
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    else:
        obs = initial_obs

    subtask_evals = [get_subtask_eval(env)]
    last_good_state = _capture_state(env)
    initial_trace_entry = _latest_ordered_trace_entry(subtask_evals) or {}
    best_ordered_count = len(initial_trace_entry.get("ordered_completed_subtasks", []))
    last_good_subtask = (
        initial_trace_entry.get("ordered_completed_subtasks", [])[-1]
        if best_ordered_count
        else None
    )
    high_level_success = False
    high_level_steps = 0
    last_good_step = 0
    last_video_frame = None

    for step_i in range(config.high_level_horizon):
        high_level_steps = step_i + 1
        action = call_policy(policy, obs)
        obs, reward, done, info = _step_env(env, action)
        video_frame = _append_video_frame_from_env(
            env,
            video_writer,
            camera_name=video_camera_name,
            height=video_height,
            width=video_width,
            obs=obs,
            previous_frame=last_video_frame,
            prefer_env_render=not video_direct_sim_render,
        )
        if video_frame is not None:
            last_video_frame = video_frame
        current_eval = info.get("subtask_eval") if info else None
        if current_eval is None:
            current_eval = get_subtask_eval(env)
        subtask_evals.append(current_eval)

        trace_entry = _latest_ordered_trace_entry(subtask_evals) or {}
        ordered_completed = trace_entry.get("ordered_completed_subtasks", [])
        if len(ordered_completed) > best_ordered_count:
            best_ordered_count = len(ordered_completed)
            last_good_subtask = ordered_completed[-1] if ordered_completed else None
            last_good_state = _capture_state(env)
            last_good_step = high_level_steps

        if _is_task_success(info, reward=reward, env=env):
            high_level_success = True
            break
        if done:
            break

    high_level_summary = summarize_subtask_rollout(
        subtask_evals,
        stuck_patience=config.stuck_patience,
        include_trace=config.include_trace,
    )
    high_level_summary["success"] = high_level_success
    high_level_summary["num_steps"] = high_level_steps
    high_level_summary["last_good_ordered_subtask"] = last_good_subtask
    high_level_summary["last_good_ordered_subtask_count"] = best_ordered_count
    high_level_summary["last_good_step"] = last_good_step
    high_level_summary["steps_since_last_good_progress"] = (
        high_level_steps - last_good_step
    )

    result = {
        "high_level": high_level_summary,
        "recovery_attempted": False,
        "recovery": None,
        "subtask": None,
    }
    if high_level_success:
        return result

    final_eval = high_level_summary.get("final_subtask_eval")
    high_level_subtask_name = (
        high_level_summary.get("ordered_current_subtask")
        or high_level_summary.get("current_subtask_estimate")
        or high_level_summary.get("stuck_subtask")
    )
    high_level_instruction = _subtask_instruction(final_eval, high_level_subtask_name)

    separator_header = config.video_separator_text or _recovery_separator_header(mode)
    separator_text = _make_recovery_separator_text(
        separator_header,
        high_level_summary,
        high_level_subtask_name,
        high_level_instruction,
    )
    _append_video_separator(
        video_writer,
        last_video_frame,
        num_frames=config.video_separator_frames,
        text=separator_text,
    )
    recovery_meta = apply_recovery_mode(env, mode, last_good_state)
    _refresh_sim_visuals(env, num_forwards=3)
    obs = _get_obs_after_state_change(env, fallback_obs=obs)
    video_frame = _append_video_frame_from_env(
        env,
        video_writer,
        camera_name=video_camera_name,
        height=video_height,
        width=video_width,
        obs=obs,
        previous_frame=last_video_frame,
        prefer_env_render=not video_direct_sim_render,
        reuse_corrupt_previous=False,
        render_attempts=5,
    )
    if video_frame is not None:
        last_video_frame = video_frame
    recovery_start_eval = get_subtask_eval(env)
    subtask_name = (
        _ordered_current_subtask_from_eval(recovery_start_eval)
        or high_level_subtask_name
    )
    instruction = _subtask_instruction(recovery_start_eval, subtask_name)
    recovery_meta["high_level_target_subtask"] = high_level_subtask_name
    recovery_meta["high_level_target_instruction"] = high_level_instruction
    recovery_meta["recovery_start_target_subtask"] = subtask_name
    recovery_meta["recovery_start_target_instruction"] = instruction
    _set_env_instruction(env, instruction)
    obs = _get_obs_after_state_change(env, fallback_obs=obs)
    obs = set_observation_instruction(obs, instruction)

    retry_evals = [recovery_start_eval]
    retry_success = _subtask_is_complete(retry_evals[-1], subtask_name)
    retry_steps = 0
    retry_horizon = (
        max(1, high_level_steps - last_good_step)
        if config.match_recovery_horizon_to_no_progress
        else config.subtask_horizon
    )
    for step_i in range(retry_horizon):
        if retry_success:
            break
        retry_steps = step_i + 1
        action = call_policy(policy, obs, instruction=instruction)
        obs, reward, done, info = _step_env(env, action)
        _refresh_sim_visuals(env)
        video_frame = _append_video_frame_from_env(
            env,
            video_writer,
            camera_name=video_camera_name,
            height=video_height,
            width=video_width,
            obs=obs,
            previous_frame=last_video_frame,
            prefer_env_render=not video_direct_sim_render,
            reuse_corrupt_previous=False,
            render_attempts=3,
        )
        if video_frame is not None:
            last_video_frame = video_frame
        obs = set_observation_instruction(obs, instruction)
        current_eval = info.get("subtask_eval") if info else None
        if current_eval is None:
            current_eval = get_subtask_eval(env)
        retry_evals.append(current_eval)
        retry_success = _subtask_is_complete(current_eval, subtask_name)
        if done:
            break

    retry_summary = summarize_subtask_rollout(
        retry_evals,
        stuck_patience=config.stuck_patience,
        include_trace=config.include_trace,
    )
    retry_summary["success"] = retry_success
    retry_summary["num_steps"] = retry_steps
    retry_summary["horizon"] = retry_horizon
    retry_summary["horizon_source"] = (
        "steps_since_last_good_progress"
        if config.match_recovery_horizon_to_no_progress
        else "fixed_subtask_horizon"
    )
    retry_summary["target_subtask"] = subtask_name
    retry_summary["target_instruction"] = instruction

    result.update(
        {
            "recovery_attempted": True,
            "recovery": recovery_meta,
            "subtask": retry_summary,
        }
    )
    return result
