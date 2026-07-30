import json
from pathlib import Path
import tempfile
import unittest

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.subtask_safe import (
    SUBTASK_FAILURE_LABEL_SEMANTICS,
    atomic_write_subtask_safe_record,
    build_subtask_safe_record,
    validate_subtask_safe_record,
)
from robocasa.recovery.subtask_eval import get_subtask_eval


def subtask_eval(*, first=False, second=False, task_success=False):
    return {
        "required_predicates": ["first", "second"],
        "predicates": {
            "first": {"value": first, "stage": "subtask"},
            "second": {"value": second, "stage": "subtask"},
        },
        "task_success": task_success,
    }


class TestSubtaskSafe(unittest.TestCase):
    def test_get_subtask_eval_traverses_nested_wrappers(self):
        payload = subtask_eval()

        class Inner:
            def get_subtask_progress(self):
                return payload

        class Wrapper:
            def __init__(self, env):
                self.env = env

        self.assertIs(
            get_subtask_eval(Wrapper(Wrapper(Inner()))),
            payload,
        )

    def test_get_subtask_eval_handles_wrapper_cycle(self):
        class Wrapper:
            pass

        first = Wrapper()
        second = Wrapper()
        first.env = second
        second.env = first
        self.assertIsNone(get_subtask_eval(first))

    def test_successful_segments_use_first_ordered_completion(self):
        record = build_subtask_safe_record(
            [
                subtask_eval(),
                subtask_eval(first=True),
                subtask_eval(first=False),
                subtask_eval(second=True, task_success=True),
            ],
            [0, 2],
            rollout_failed=False,
            rollout_id="success-rollout",
        )

        self.assertEqual(
            record["label_semantics"], SUBTASK_FAILURE_LABEL_SEMANTICS
        )
        self.assertEqual(
            [
                (
                    segment["subtask_name"],
                    segment["completion_environment_step"],
                    segment["failure_label"],
                )
                for segment in record["segments"]
            ],
            [("first", 1, 0), ("second", 3, 0)],
        )
        self.assertEqual(
            [
                item["ordered_current_subtask"]
                for item in record["inference_records"]
            ],
            ["first", "second"],
        )
        self.assertEqual(
            record["transitions"][1]["regressed_predicates"],
            [],
        )
        self.assertIn(
            "first",
            {
                name
                for transition in record["transitions"]
                for name in transition["regressed_predicates"]
            },
        )
        counts = validate_subtask_safe_record(
            record,
            rollout_id="success-rollout",
            rollout_failed=False,
            inference_environment_steps=[0, 2],
        )
        self.assertEqual(
            counts,
            {
                "segments": 2,
                "usable_segments": 2,
                "successful_segments": 2,
                "failed_segments": 0,
                "labeled_without_inference": 0,
            },
        )

    def test_only_terminal_active_segment_is_failure(self):
        record = build_subtask_safe_record(
            [
                subtask_eval(),
                subtask_eval(first=True),
                subtask_eval(first=True),
                subtask_eval(first=True),
            ],
            [0, 2],
            rollout_failed=True,
            rollout_id="failed-rollout",
        )

        self.assertEqual(
            [
                (
                    segment["subtask_name"],
                    segment["eventually_failed"],
                    segment["failure_label"],
                )
                for segment in record["segments"]
            ],
            [("first", False, 0), ("second", True, 1)],
        )
        self.assertEqual(record["terminal_active_subtask"], "second")
        self.assertEqual(record["labeling_status"], "complete")

    def test_unavailable_trace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            build_subtask_safe_record(
                [subtask_eval(), None],
                [0],
                rollout_failed=True,
            )

    def test_atomic_json_round_trip(self):
        record = build_subtask_safe_record(
            [subtask_eval(), subtask_eval(first=True)],
            [0],
            rollout_failed=True,
            rollout_id="round-trip",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subtasks" / "record.json"
            atomic_write_subtask_safe_record(path, record)
            self.assertEqual(json.loads(path.read_text()), record)


if __name__ == "__main__":
    unittest.main()
