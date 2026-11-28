"""Verify recovery/MP4 alignment without requiring a simulator or a policy checkpoint."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_safe_collect_rollouts import FakePolicy
from robocasa.recovery.safe.collect_rollouts import collect_single_rollout
from robocasa.recovery.safe.subtask_safe import validate_subtask_safe_record

try:
    import cv2
except ImportError:
    cv2 = None


class TimingFixture:
    """Scripted object states and a camera marker for validating frame timestamps."""
    def __init__(self, task):
        self.task = task

    def reset(self):
        self.step_no = 0
        return {}, {}

    def step(self, action):
        self.step_no += 1
        done = self.step_no == 8
        return {}, 0., done, False, {"success": done}

    def get_subtask_progress(self):
        i = self.step_no
        if self.task == "ArrangeBreadBasket":
            required = ["cabinet_open", "bread_in_basket", "basket_on_dining_counter", "gripper_released"]
            values = dict(cabinet_open=True, bread_in_basket=i == 1 or i >= 5,
                          basket_on_dining_counter=i >= 7, gripper_released=i >= 8,
                          bread_grasped=2 <= i <= 4, basket_grasped=i == 6)
        else:
            required = ["drawer_open", "correct_tool_on_cutting_board", "wrong_tool_not_on_cutting_board",
                        "gripper_released"]
            values = dict(drawer_open=True, correct_tool_on_cutting_board=i >= 6,
                          wrong_tool_not_on_cutting_board=i >= 6, gripper_released=i >= 8,
                          correct_tool_grasped=i == 1, correct_tool_present_on_board=i >= 2,
                          wrong_tool_grasped=3 <= i <= 5)
        predicates = {k: dict(value=v, required=k in required, stage="subtask" if k in required else "transient")
                      for k, v in values.items()}
        return dict(task_name=self.task, predicates=predicates, required_predicates=required, task_success=i == 8)

    def marker_frame(self):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :48] = self.step_no * 24
        active = self.step_no in ((2, 3, 4) if self.task == "ArrangeBreadBasket" else (3, 4, 5))
        frame[:, 48:] = 240 if active else 16
        return frame


class TimingTests(unittest.TestCase):
    def collect(self, task, writer, stride, skip=None):
        def render(env, out, obs, previous):
            self.assertIsNone(previous)
            if env.step_no != skip:
                out.append_data(env.marker_frame())
            # A callback may append a frame without returning its pixels.
        return collect_single_rollout(FakePolicy(), TimingFixture(task), horizon=12, video_writer=writer,
                                      frame_fn=render, video_frame_stride=stride, record_subtask_trace=True)

    def test_skipped_render_has_no_timestamp_and_terminal_is_kept(self):
        class Sink:
            def append_data(self, frame):
                pass
        r = self.collect("CuttingToolSelection", Sink(), stride=3, skip=4)
        trace = r["subtask_safe_record"]
        self.assertEqual(trace["video_frame_environment_steps"], [1, 7, 8])
        self.assertEqual(r["num_video_frames"], 3)
        for invalid in ([0, 1], [2, 1], [9]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_subtask_safe_record(dict(trace, video_frame_environment_steps=invalid),
                                             rollout_id=None, rollout_failed=False,
                                             inference_environment_steps=r["inference_env_steps"])

    @unittest.skipIf(cv2 is None, "OpenCV is needed for MP4 round-trip")
    def test_two_task_mp4s_against_stored_predicates(self):
        for task, key in (("ArrangeBreadBasket", "bread_grasped"), ("CuttingToolSelection", "wrong_tool_grasped")):
            for stride, expected in ((1, list(range(1, 9))), (3, [1, 4, 7, 8])):
                with self.subTest(task=task, stride=stride), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "markers.mp4"
                    file = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (96, 64))
                    self.assertTrue(file.isOpened())

                    class Sink:
                        def append_data(self, frame):
                            file.write(frame)

                    try:
                        r = self.collect(task, Sink(), stride)
                    finally:
                        file.release()
                    trace = json.loads(json.dumps(r["subtask_safe_record"]))
                    validate_subtask_safe_record(trace, rollout_id=None, rollout_failed=False,
                                                 inference_environment_steps=r["inference_env_steps"])
                    self.assertEqual(trace["video_frame_environment_steps"], expected)
                    self.assertEqual(r["inference_env_steps"], [0, 2, 4, 6])
                    reader = cv2.VideoCapture(str(path))
                    try:
                        for step in trace["video_frame_environment_steps"]:
                            ok, pixels = reader.read()
                            self.assertTrue(ok)
                            self.assertEqual(round(float(pixels[32, 20].mean()) / 24), step)
                            observation = next(v for v in reversed(trace["recovery_observations"])
                                               if v["environment_step"] <= step)
                            self.assertEqual(bool(pixels[32, 75].mean() > 128), observation["values"][key])
                        self.assertFalse(reader.read()[0])
                    finally:
                        reader.release()
