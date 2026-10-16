import tempfile
import unittest
from pathlib import Path

import numpy as np

from robocasa.recovery.safe.xr1_matched_fpr_video import (
    alert_fraction,
    first_alert,
    render_matched_fpr_video,
)

try:
    import cv2
except ImportError:
    cv2 = None


THRESHOLDS = {
    "safe_only": {"safe_only": 0.60},
    "staged_safe_time": {"early_safe": 0.25, "late_time": 0.75},
    "time_only": {"time_only": 0.80},
}


def example_record():
    return {
        "rollout_id": "rollout-0",
        "task_name": "OpenDrawer",
        "failed": True,
        "horizon": 4,
        "safe_scores": [0.10, 0.30, 0.50, 0.70],
        "time_scores": [0.10, 0.40, 0.78, 0.90],
        "inference_environment_steps": [0, 3, 6, 9],
        "replan_steps": 3,
    }


class TestXr1MatchedFprVideo(unittest.TestCase):
    def test_first_alerts_follow_causal_windows(self):
        record = example_record()
        self.assertEqual(
            first_alert(record, "safe_only", THRESHOLDS, 0.5, 0.5),
            {"step": 4, "source": "safe_only"},
        )
        self.assertEqual(
            first_alert(record, "staged_safe_time", THRESHOLDS, 0.5, 0.5),
            {"step": 2, "source": "early_safe"},
        )
        self.assertEqual(
            first_alert(record, "time_only", THRESHOLDS, 0.5, 0.5),
            {"step": 4, "source": "time_only"},
        )
        self.assertEqual(
            alert_fraction(
                record,
                "staged_safe_time",
                THRESHOLDS,
                0.5,
                0.5,
            ),
            0.5,
        )

    @unittest.skipIf(cv2 is None, "OpenCV is not installed locally")
    def test_annotated_video_is_synchronized_and_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for index in range(12):
                frame = np.full((120, 160, 3), 30 + index * 8, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            output = root / "annotated.mp4"
            result = render_matched_fpr_video(
                example_record(),
                source,
                output,
                THRESHOLDS,
                0.5,
                0.5,
                video_frame_stride=1,
                reason="unit_test",
            )
            self.assertEqual(result["frames"], 12)
            self.assertEqual((result["width"], result["height"]), (1280, 720))
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 12)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1280)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 720)
            capture.release()


if __name__ == "__main__":
    unittest.main()
