"""Simulator- and PyTorch-free tests for the XR-1 fine-tuning sample builder (Step 0)."""

import collections
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery import xr1_samples as xs  # noqa: E402
from robocasa.recovery.xiaomi_robotics_1_policy import (  # noqa: E402
    CAMERA_KEYS,
    observation_to_state,
    sample_history,
)

MODALITY = Path(__file__).resolve().parents[1] / "robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json"


class FakeProcessor:
    """Records each call and returns arrays that depend on every input the real processor reads."""

    def __init__(self):
        self.calls = []

    def list_robot_types(self):
        return ["robocasa365"]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        texts, videos = [], []
        for turn in messages:
            for item in turn["content"]:
                if item["type"] == "text":
                    texts.append(item["text"])
                else:
                    videos.append(np.stack([np.asarray(frame) for frame in item["video"]]))
        flat = "|".join(texts) + f"|{kwargs['robot_type']}|{kwargs['do_resize']}|{kwargs['return_tensors']}"
        return {
            "input_ids": np.array([[ord(c) for c in flat]]),
            "attention_mask": np.ones((1, len(flat)), dtype=np.int64),
            "pixel_values_videos": np.concatenate(videos).astype(np.float32),
            "video_grid_thw": np.array([v.shape[:3] for v in videos]),
            "state": np.asarray(kwargs["state"], dtype=np.float32),
            "action_mask": np.ones((1, 16, 60), dtype=np.float32),
        }


def quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def synthetic_episode(num_frames=40, size=12, seed=0, instruction="pick the mug"):
    rng = np.random.default_rng(seed)
    frames = {key: rng.integers(0, 256, size=(num_frames, size, size, 3), dtype=np.uint8) for key in CAMERA_KEYS}
    named = {
        "end_effector_position_relative": rng.normal(size=(num_frames, 3)).astype(np.float32),
        "end_effector_rotation_relative": np.stack([quat(rng) for _ in range(num_frames)]).astype(np.float32),
        "gripper_qpos": rng.normal(size=(num_frames, 2)).astype(np.float32),
        "base_position": rng.normal(size=(num_frames, 3)).astype(np.float32),
        "base_rotation": np.stack([quat(rng) for _ in range(num_frames)]).astype(np.float32),
    }
    actions = rng.uniform(-1, 1, size=(num_frames, 12)).astype(np.float32)
    return xs.Episode(task_name="ArrangeTea", episode=3, instruction=instruction, frames=frames, named=named,
                      actions=actions)


class HelperTest(unittest.TestCase):
    def test_history_indices_match_the_client_deque(self):
        queue = collections.deque(maxlen=(xs.OBS_HISTORY - 1) * xs.OBS_INTERVAL + 1)
        for t in range(25):
            queue.append(t)
            expected = list(sample_history(queue, xs.OBS_HISTORY, xs.OBS_INTERVAL))
            self.assertEqual(xs.history_indices(t), expected)
        self.assertEqual(xs.history_indices(10), [4, 6, 8, 10])
        self.assertEqual(xs.history_indices(3), [0, 0, 1, 3])

    def test_valid_times_need_a_full_chunk(self):
        self.assertEqual(list(xs.valid_times(18)), [0, 1, 2])
        self.assertEqual(list(xs.valid_times(10)), [])

    def test_episode_split_is_deterministic_and_about_a_tenth(self):
        splits = [xs.episode_split("ArrangeTea", e) for e in range(2000)]
        self.assertEqual(splits, [xs.episode_split("ArrangeTea", e) for e in range(2000)])
        self.assertTrue(0.07 < splits.count("heldout") / 2000 < 0.13)

    def test_lerobot_actions_are_reordered_to_the_environment_layout(self):
        layout = xs.lerobot_layout(json.loads(MODALITY.read_text()))
        env = np.arange(24, dtype=np.float32).reshape(2, 12)   # env order: eef pos, rot, gripper, base, mode
        lerobot = np.zeros_like(env)
        start = 0
        for key, width in xs.ENV_ACTION_ORDER:
            lo, hi = layout["action"][key]
            lerobot[:, lo:hi] = env[:, start:start + width]
            start += width
        self.assertFalse(np.array_equal(lerobot, env))    # the two layouts really differ
        np.testing.assert_array_equal(xs.env_actions(lerobot, layout), env)
        # Base motion is LeRobot dims 0-3 and environment dims 7-10.
        self.assertEqual(layout["action"]["base_motion"], (0, 4))

    def test_states_follow_the_client_observation_to_state(self):
        layout = xs.lerobot_layout(json.loads(MODALITY.read_text()))
        rng = np.random.default_rng(1)
        rows = rng.normal(size=(5, 16)).astype(np.float32)
        named = xs.named_states(rows, layout)
        states = xs.policy_states(named)
        self.assertEqual(states.shape, (5, 14))
        for t in range(5):
            obs = {f"state.{k}": rows[t, slice(*layout["state"][k])] for k in xs.STATE_KEYS}
            np.testing.assert_array_equal(states[t], observation_to_state(obs))

    def test_sample_arrays(self):
        ep = synthetic_episode()
        images, state, action = xs.sample_arrays(ep, 10)
        np.testing.assert_array_equal(images[CAMERA_KEYS[1]], ep.frames[CAMERA_KEYS[1]][[4, 6, 8, 10]])
        np.testing.assert_array_equal(state[0, :, :14], ep.states[[4, 6, 8, 10]])
        self.assertFalse(state[0, :, 14:].any())
        np.testing.assert_array_equal(action[0, :, :12], ep.actions[10:26])
        self.assertFalse(action[0, :, 12:].any())
        with self.assertRaises(ValueError):
            xs.sample_arrays(ep, 30)    # only 9 future frames left


class GateTest(unittest.TestCase):
    def test_builder_matches_the_client_request(self):
        ep = synthetic_episode()
        processor = FakeProcessor()
        for t in (0, 1, 5, 6, 13, 24):
            self.assertEqual(xs.compare_with_policy(processor, ep, t), [], f"t={t}")
        builder_kwargs, client_kwargs = processor.calls[-2], processor.calls[-1]
        self.assertEqual({k: v for k, v in builder_kwargs.items() if k != "state"},
                         {k: v for k, v in client_kwargs.items() if k != "state"})

    def test_gate_catches_a_wrong_history(self):
        ep = synthetic_episode()
        processor = FakeProcessor()
        images, state, _ = xs.sample_arrays(ep, 13)
        shifted = {key: ep.frames[key][[6, 8, 10, 12]] for key in CAMERA_KEYS}   # off by one frame
        built = xs.processor_inputs(processor, shifted, state, ep.instruction)
        reference = xs.policy_request(processor, ep, 13)
        self.assertEqual(xs.compare_requests(built, reference), ["pixel_values_videos"])
        built = xs.processor_inputs(processor, images, state, ep.instruction + " now")
        self.assertIn("input_ids", xs.compare_requests(built, reference))

    def test_make_sample_carries_the_target(self):
        ep = synthetic_episode()
        sample = xs.make_sample(FakeProcessor(), ep, 7)
        np.testing.assert_array_equal(sample["action"][0, :, :12], ep.actions[7:23])
        self.assertEqual(sample["meta"], {"task_name": "ArrangeTea", "episode": 3, "t": 7})


class LeRobotReaderTest(unittest.TestCase):
    def test_reads_a_synthetic_lerobot_episode(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        ep = synthetic_episode(num_frames=20)
        layout = xs.lerobot_layout(json.loads(MODALITY.read_text()))
        state_rows = np.zeros((20, 16), dtype=np.float32)
        for key in xs.STATE_KEYS:
            state_rows[:, slice(*layout["state"][key])] = ep.named[key]
        action_rows = np.zeros((20, 12), dtype=np.float32)
        start = 0
        for key, width in xs.ENV_ACTION_ORDER:
            action_rows[:, slice(*layout["action"][key])] = ep.actions[:, start:start + width]
            start += width
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "meta/modality.json").write_text(MODALITY.read_text())
            (root / "meta/info.json").write_text(json.dumps({
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }))
            (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick the mug"}) + "\n")
            table = pa.table({
                "observation.state": [list(map(float, r)) for r in state_rows],
                "action": [list(map(float, r)) for r in action_rows],
                "annotation.human.task_description": [[0]] * 20,
            })
            pq.write_table(table, root / "data/chunk-000/episode_000003.parquet")
            by_name = {layout["video"][key]: ep.frames[key] for key in CAMERA_KEYS}
            decode = lambda path: by_name[Path(path).parent.name]   # noqa: E731
            read = xs.read_lerobot_episode(root, 3, "ArrangeTea", decode=decode)
            self.assertEqual(xs.lerobot_episodes(root), [3])
        self.assertEqual(read.instruction, "pick the mug")
        np.testing.assert_array_equal(read.actions, ep.actions)
        np.testing.assert_array_equal(read.states, ep.states)
        for key in CAMERA_KEYS:
            np.testing.assert_array_equal(read.frames[key], ep.frames[key])


if __name__ == "__main__":
    unittest.main()
