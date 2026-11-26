"""Training samples for fine-tuning XR-1 (RoboCasa365), built as the served client builds requests.

Step 0 of ``docs/recovery_harvest_finetune_plan.md`` (decision A, 26 November 2026). The released
RoboCasa365 checkpoint receives, per inference (``XiaomiRobotics1Policy._make_request``, which
mirrors Xiaomi's ``eval_robocasa365/entry.py``):

- each camera of ``CAMERA_KEYS`` as a four-frame video, frames t-6, t-4, t-2, t (clamped at 0),
  each centre-cropped to 0.95 and resized back with bilinear resampling;
- a ``(1, 4, 60)`` state whose first 14 dims are the policy state of those frames;
- the text "Left camera: <video>\\nRight camera: <video>\\nWrist camera: <video>\\n\\nGenerate robot
  actions for the task:\\n{instruction} /no_cot" and the assistant turn "<cot></cot>";
- ``robot_type="robocasa365"`` (identity action statistics on dims 0-11).

A training sample adds the target: the next 16 actions in the environment's 12-d layout
(``convert_action``: end-effector position, rotation, gripper, base motion, control mode) in dims
0-11 of a ``(1, 16, 60)`` array. The RoboCasa365 LeRobot files store actions in another order
(base motion first, ``meta/modality.json``), so they are reordered here.

This module needs neither the simulator nor PyTorch, except where the Hugging Face processor is
called (``processor_inputs``), and the gate ``compare_with_policy`` that checks this builder
against the policy client on the same frames.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robocasa.recovery.xiaomi_robotics_1_policy import (
    CAMERA_KEYS,
    DEFAULT_ACTION_DIM,
    DEFAULT_ROBOT_TYPE,
    DEFAULT_STATE_DIM,
    XiaomiRobotics1Policy,
    observation_to_state,
)

# Frozen with this code: the served preprocessing (XiaomiRobotics1Policy defaults, entry.py).
ROBOT_TYPE = DEFAULT_ROBOT_TYPE
STATE_DIM = DEFAULT_STATE_DIM        # 60, of which the first 14 are used
ACTION_DIM = DEFAULT_ACTION_DIM      # 12
OBS_HISTORY = 4
OBS_INTERVAL = 2
CHUNK = 16                           # action horizon of the RoboCasa365 checkpoint
CROP_RATIO = 0.95
POLICY_STATE_DIM = 14

# convert_action (robocasa/utils/env_utils.py): the order the model's dims 0-11 are executed in.
ENV_ACTION_ORDER = (
    ("end_effector_position", 3),
    ("end_effector_rotation", 3),
    ("gripper_close", 1),
    ("base_motion", 4),
    ("control_mode", 1),
)
STATE_KEYS = (
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
    "base_position",
    "base_rotation",
)
REQUEST_KEYS = ("input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw", "state", "action_mask")
HELDOUT_FRACTION = 0.1


# ---- pure helpers -------------------------------------------------------------------------------

def history_indices(t, length=OBS_HISTORY, interval=OBS_INTERVAL):
    """Frames the client sends at step ``t``: ``sample_history`` over a deque of the last frames."""
    t = int(t)
    if t < 0:
        raise ValueError("t must be non-negative")
    return [max(0, t - (length - 1 - i) * interval) for i in range(length)]


def valid_times(num_frames, chunk=CHUNK):
    """Frames with a full future action chunk (no padding at the episode end)."""
    return range(0, max(0, int(num_frames) - chunk + 1))


def episode_split(task_name, episode, heldout_fraction=HELDOUT_FRACTION):
    """Deterministic held-out split by episode (never by frame)."""
    digest = hashlib.sha256(f"{task_name}:{int(episode)}".encode()).hexdigest()
    return "heldout" if int(digest[:8], 16) % 1000 < round(1000 * heldout_fraction) else "train"


def lerobot_layout(modality):
    """Named slices of ``observation.state`` and ``action`` from LeRobot ``meta/modality.json``."""
    state = {k: (int(v["start"]), int(v["end"])) for k, v in modality["state"].items()}
    action = {k: (int(v["start"]), int(v["end"])) for k, v in modality["action"].items()}
    missing = [k for k in STATE_KEYS if k not in state] + [k for k, _ in ENV_ACTION_ORDER if k not in action]
    if missing:
        raise ValueError(f"modality.json lacks {missing}")
    for key, width in ENV_ACTION_ORDER:
        if action[key][1] - action[key][0] != width:
            raise ValueError(f"action '{key}' has width {action[key][1] - action[key][0]}, expected {width}")
    video = {}
    for name, spec in modality.get("video", {}).items():
        video[f"video.{name}"] = spec.get("original_key", f"observation.images.{name}")
    if any(key not in video for key in CAMERA_KEYS):
        raise ValueError(f"modality.json lacks cameras {[k for k in CAMERA_KEYS if k not in video]}")
    return {"state": state, "action": action, "video": video}


def env_actions(lerobot_actions, layout):
    """LeRobot action rows -> the environment's 12-d order (what the model outputs)."""
    actions = np.asarray(lerobot_actions, dtype=np.float32)
    parts = [actions[:, slice(*layout["action"][key])] for key, _ in ENV_ACTION_ORDER]
    out = np.concatenate(parts, axis=1)
    if out.shape[1] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} action dims, got {out.shape[1]}")
    return np.ascontiguousarray(out)


def named_states(lerobot_states, layout):
    """LeRobot ``observation.state`` rows -> the observation's named state arrays."""
    states = np.asarray(lerobot_states, dtype=np.float32)
    return {key: np.ascontiguousarray(states[:, slice(*layout["state"][key])]) for key in STATE_KEYS}


def observation_at(named, frames, t, instruction=None):
    """The observation dict the gym wrapper gives the policy at frame ``t``."""
    obs = {f"state.{key}": named[key][t] for key in STATE_KEYS}
    obs.update({key: frames[key][t] for key in CAMERA_KEYS})
    if instruction is not None:
        obs["annotation.human.task_description"] = instruction
    return obs


def policy_states(named):
    """(T, 14) policy states through the client's own ``observation_to_state``."""
    num = len(next(iter(named.values())))
    return np.stack([observation_to_state({f"state.{k}": named[k][t] for k in STATE_KEYS}) for t in range(num)])


def center_crop(image, crop_ratio=CROP_RATIO):
    from PIL import Image

    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


def build_messages(image_history, instruction, crop_ratio=CROP_RATIO):
    videos = {key: [center_crop(frame, crop_ratio) for frame in image_history[key]] for key in CAMERA_KEYS}
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Left camera: "},
                {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                {"type": "text", "text": "\nRight camera: "},
                {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                {"type": "text", "text": "\nWrist camera: "},
                {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


# ---- episodes -----------------------------------------------------------------------------------

@dataclass
class Episode:
    task_name: str
    episode: int
    instruction: str
    frames: dict                 # camera key -> (T, H, W, 3) uint8
    named: dict                  # state key -> (T, d) float32, as the observation carries it
    actions: np.ndarray          # (T, 12) float32, environment order
    states: np.ndarray = field(default=None)   # (T, 14) float32 policy state

    def __post_init__(self):
        lengths = {len(self.actions)} | {len(v) for v in self.frames.values()} | {len(v) for v in self.named.values()}
        if len(lengths) != 1:
            raise ValueError(f"{self.task_name} episode {self.episode}: frame counts differ {sorted(lengths)}")
        if self.states is None:
            self.states = policy_states(self.named)

    @property
    def num_frames(self):
        return len(self.actions)


def sample_arrays(episode, t, chunk=CHUNK):
    """Image history, ``(1, 4, 60)`` state and ``(1, chunk, 60)`` action target at frame ``t``."""
    if t not in valid_times(episode.num_frames, chunk):
        raise ValueError(f"t={t} has no full {chunk}-step future in a {episode.num_frames}-frame episode")
    idx = history_indices(t)
    images = {key: np.ascontiguousarray(episode.frames[key][idx]) for key in CAMERA_KEYS}
    state = np.zeros((1, OBS_HISTORY, STATE_DIM), dtype=np.float32)
    state[0, :, :POLICY_STATE_DIM] = episode.states[idx]
    action = np.zeros((1, chunk, STATE_DIM), dtype=np.float32)
    action[0, :, :ACTION_DIM] = episode.actions[t:t + chunk]
    return images, state, action


def processor_inputs(processor, images, state, instruction):
    """The exact processor call of ``XiaomiRobotics1Policy._make_request``."""
    return processor.apply_chat_template(
        build_messages(images, instruction),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_resize=False,
        state=state,
        robot_type=ROBOT_TYPE,
    )


def make_sample(processor, episode, t):
    images, state, action = sample_arrays(episode, t)
    sample = dict(processor_inputs(processor, images, state, episode.instruction))
    sample["action"] = action
    sample["meta"] = {"task_name": episode.task_name, "episode": int(episode.episode), "t": int(t)}
    return sample


# ---- the input-match gate -----------------------------------------------------------------------

class _NoServer:
    def infer(self, request):  # pragma: no cover - never called by the gate
        raise RuntimeError("the input-match gate never queries a server")

    def close(self):
        pass


def policy_request(processor, episode, t):
    """What the deployed client sends at frame ``t``, fed with the episode's observations."""
    policy = XiaomiRobotics1Policy(model_path="input-match", client=_NoServer(), processor=processor)
    for i in range(max(0, t - policy.queue_length + 1), t + 1):
        policy._append_observation(observation_at(episode.named, episode.frames, i))
    return policy._make_request(episode.instruction)


def _as_array(value):
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu()
        if value.is_floating_point():
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def compare_requests(built, reference, keys=REQUEST_KEYS):
    """Names of the keys whose tensors differ (shape, dtype class or any value)."""
    mismatched = []
    for key in keys:
        if key not in built or key not in reference:
            mismatched.append(f"{key} (missing)")
            continue
        a, b = _as_array(built[key]), _as_array(reference[key])
        if a.shape != b.shape or not np.array_equal(a, b):
            mismatched.append(key)
    return mismatched


def compare_with_policy(processor, episode, t):
    images, state, _ = sample_arrays(episode, t)
    built = processor_inputs(processor, images, state, episode.instruction)
    reference = policy_request(processor, episode, t)
    return compare_requests(built, reference)


# ---- LeRobot reading ----------------------------------------------------------------------------

def decode_video(path):
    """All frames of an mp4 as (T, H, W, 3) uint8, with whichever decoder is installed."""
    path = str(path)
    try:
        import av

        with av.open(path) as container:
            return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    except ImportError:
        pass
    try:
        import decord

        reader = decord.VideoReader(path)
        return reader.get_batch(list(range(len(reader)))).asnumpy()
    except ImportError:
        pass
    import imageio.v3 as iio

    return np.asarray(iio.imread(path, plugin="pyav"))


def _read_json(path):
    return json.loads(Path(path).read_text())


def lerobot_tasks(lerobot):
    tasks = {}
    with (Path(lerobot) / "meta" / "tasks.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                tasks[int(row["task_index"])] = row["task"]
    return tasks


def lerobot_episodes(lerobot):
    return sorted(int(p.stem.split("_")[1]) for p in (Path(lerobot) / "data").rglob("episode_*.parquet"))


def read_lerobot_arrays(lerobot, episode):
    """(named states, environment-order actions) of one LeRobot episode, without its videos."""
    import pyarrow.parquet as pq

    lerobot = Path(lerobot)
    info = _read_json(lerobot / "meta" / "info.json")
    layout = lerobot_layout(_read_json(lerobot / "meta" / "modality.json"))
    chunk = int(episode) // int(info.get("chunks_size", 1000))
    table = pq.read_table(lerobot / info["data_path"].format(episode_chunk=chunk, episode_index=int(episode)),
                          columns=["observation.state", "action"])
    states = np.stack([np.asarray(x, dtype=np.float32) for x in table.column("observation.state").to_pylist()])
    actions = np.stack([np.asarray(x, dtype=np.float32) for x in table.column("action").to_pylist()])
    return named_states(states, layout), env_actions(actions, layout)


def read_lerobot_episode(lerobot, episode, task_name, decode=decode_video):
    import pyarrow.parquet as pq

    lerobot = Path(lerobot)
    info = _read_json(lerobot / "meta" / "info.json")
    layout = lerobot_layout(_read_json(lerobot / "meta" / "modality.json"))
    chunk = int(episode) // int(info.get("chunks_size", 1000))
    table = pq.read_table(lerobot / info["data_path"].format(episode_chunk=chunk, episode_index=int(episode)))
    states = np.stack([np.asarray(x, dtype=np.float32) for x in table.column("observation.state").to_pylist()])
    actions = np.stack([np.asarray(x, dtype=np.float32) for x in table.column("action").to_pylist()])
    task_column = "annotation.human.task_description"
    if task_column in table.column_names:
        task_ids = {int(np.asarray(x).reshape(-1)[0]) for x in table.column(task_column).to_pylist()}
    else:
        task_ids = {int(x) for x in table.column("task_index").to_pylist()}
    if len(task_ids) != 1:
        raise ValueError(f"{task_name} episode {episode}: {len(task_ids)} instructions in one episode")
    instruction = lerobot_tasks(lerobot)[task_ids.pop()]
    frames = {}
    for key in CAMERA_KEYS:
        video_key = layout["video"][key]
        path = lerobot / info["video_path"].format(episode_chunk=chunk, video_key=video_key, episode_index=int(episode))
        frames[key] = decode(path)
    return Episode(task_name=task_name, episode=int(episode), instruction=instruction, frames=frames,
                   named=named_states(states, layout), actions=env_actions(actions, layout))
