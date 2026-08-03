import json
from pathlib import Path
import tempfile
import unittest

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.audit_subtask_safe_dataset import (
    format_report as format_audit_report,
    summarize_subtask_records,
)
from robocasa.recovery.safe.subtask_safe import (
    SUBTASK_FAILURE_LABEL_SEMANTICS,
    atomic_write_subtask_safe_record,
    build_subtask_safe_record,
    validate_subtask_safe_record,
)
from robocasa.recovery.subtask_eval import get_subtask_eval


def subtask_eval(*, first=False, second=False, task_success=False):
    return {
        "task_name": "TestSemanticTask",
        "required_predicates": ["first", "second"],
        "predicates": {
            "first": {
                "value": first,
                "stage": "subtask",
                "description": "Complete the first semantic subtask.",
            },
            "second": {
                "value": second,
                "stage": "subtask",
                "description": "Complete the second semantic subtask.",
            },
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

        self.assertEqual(record["label_semantics"], SUBTASK_FAILURE_LABEL_SEMANTICS)
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
            [item["subtask_id"] for item in record["inference_records"]],
            ["first", "second"],
        )
        self.assertEqual(
            [item["subtask_instruction"] for item in record["inference_records"]],
            [
                "Complete the first semantic subtask.",
                "Complete the second semantic subtask.",
            ],
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
                "excluded_completed_subtasks": 0,
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
        self.assertEqual(
            record["terminal_failure_reason"],
            "active_subtask_never_completed",
        )
        self.assertEqual(record["terminal_unsatisfied_predicate_names"], ["second"])
        self.assertEqual(record["labeling_status"], "complete")

    def test_terminal_regression_relabels_first_unsatisfied_subtask(self):
        record = build_subtask_safe_record(
            [
                subtask_eval(),
                subtask_eval(first=True),
                subtask_eval(first=False, second=True),
            ],
            [0, 1],
            rollout_failed=True,
            rollout_id="regressed-rollout",
        )

        self.assertEqual(record["terminal_active_subtask"], "first")
        self.assertEqual(
            record["terminal_failure_reason"],
            "completed_subtask_regressed_before_task_completion",
        )
        self.assertEqual(record["terminal_unsatisfied_predicate_names"], ["first"])
        self.assertEqual(
            [
                (
                    segment["subtask_id"],
                    segment["completed"],
                    segment["eventually_failed"],
                    segment["failure_label"],
                    segment["first_observed_completion_environment_step"],
                    segment["completion_environment_step"],
                )
                for segment in record["segments"]
            ],
            [
                ("first", False, True, 1, 1, None),
                ("second", True, False, 0, 2, 2),
            ],
        )
        counts = validate_subtask_safe_record(
            record,
            rollout_id="regressed-rollout",
            rollout_failed=True,
            inference_environment_steps=[0, 1],
        )
        self.assertEqual(counts["failed_segments"], 1)
        self.assertEqual(counts["successful_segments"], 1)

    def test_failed_rollout_with_all_terminal_predicates_true_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "mapping does not explain the official task failure",
        ):
            build_subtask_safe_record(
                [
                    subtask_eval(),
                    subtask_eval(first=True, second=True),
                ],
                [0],
                rollout_failed=True,
                rollout_id="unexplained-failure",
            )

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

    def test_pre_normalization_schema_records_are_rejected(self):
        for schema_version in (1, 2):
            with self.subTest(schema_version=schema_version):
                record = build_subtask_safe_record(
                    [subtask_eval(), subtask_eval(first=True)],
                    [0],
                    rollout_failed=True,
                    rollout_id="old-schema",
                )
                record["schema_version"] = schema_version
                with self.assertRaisesRegex(
                    ValueError,
                    "rerun semantic Subtask-SAFE collection",
                ):
                    validate_subtask_safe_record(
                        record,
                        rollout_id="old-schema",
                        rollout_failed=True,
                        inference_environment_steps=[0],
                    )

    def test_coverage_audit_counts_zero_inference_and_deficits(self):
        success_without_first_inference = build_subtask_safe_record(
            [
                subtask_eval(),
                subtask_eval(first=True),
                subtask_eval(first=True, second=True, task_success=True),
            ],
            [1],
            rollout_failed=False,
            rollout_id="success-no-first-inference",
        )
        success_with_both_inferences = build_subtask_safe_record(
            [
                subtask_eval(),
                subtask_eval(first=True),
                subtask_eval(first=True, second=True, task_success=True),
            ],
            [0, 1],
            rollout_failed=False,
            rollout_id="success-both-inferences",
        )
        terminal_second_failure = build_subtask_safe_record(
            [
                subtask_eval(),
                subtask_eval(first=True),
                subtask_eval(first=True),
            ],
            [0, 1],
            rollout_failed=True,
            rollout_id="failure-second",
        )
        result = summarize_subtask_records(
            [
                {
                    "task_name": "CompositeTask",
                    "task_type": "composite",
                    "rollout_id": "success-no-first-inference",
                    "failed": False,
                    "subtask_record": success_without_first_inference,
                },
                {
                    "task_name": "CompositeTask",
                    "task_type": "composite",
                    "rollout_id": "success-both-inferences",
                    "failed": False,
                    "subtask_record": success_with_both_inferences,
                },
                {
                    "task_name": "CompositeTask",
                    "task_type": "composite",
                    "rollout_id": "failure-second",
                    "failed": True,
                    "subtask_record": terminal_second_failure,
                },
            ],
            target_successes=3,
            target_failures=2,
            task_type_filter="composite",
        )

        first, second = result["task_subtask_rows"]
        self.assertEqual(first["subtask_name"], "first")
        self.assertEqual(first["usable_successes"], 2)
        self.assertEqual(first["usable_failures"], 0)
        self.assertEqual(first["labeled_without_inference"], 1)
        self.assertEqual(first["inference_coverage"], 2 / 3)
        self.assertEqual(
            (first["success_deficit"], first["failure_deficit"]),
            (1, 2),
        )
        self.assertEqual(second["subtask_name"], "second")
        self.assertEqual(second["usable_successes"], 2)
        self.assertEqual(second["usable_failures"], 1)
        self.assertEqual(
            (second["success_deficit"], second["failure_deficit"]),
            (1, 1),
        )
        self.assertEqual(
            result["counts"],
            {
                "rollouts": 3,
                "rollout_successes": 2,
                "rollout_failures": 1,
                "tasks": 1,
                "task_subtask_pairs": 2,
                "usable_success_segments": 4,
                "usable_failure_segments": 1,
                "labeled_without_inference": 1,
                "excluded_completed_without_activation": 0,
                "pairs_reaching_target": 0,
            },
        )
        self.assertEqual(result["tasks"][0]["total_deficit"], 5)
        report = format_audit_report(result)
        self.assertIn("CompositeTask", report)
        self.assertIn("diagnostic, not rollout counts", report)

    def test_coverage_audit_filters_before_duplicate_check(self):
        atomic_record = build_subtask_safe_record(
            [subtask_eval(), subtask_eval(first=True)],
            [0],
            rollout_failed=True,
            rollout_id="shared",
        )
        composite_record = build_subtask_safe_record(
            [subtask_eval(), subtask_eval(first=True)],
            [0],
            rollout_failed=True,
            rollout_id="shared",
        )
        result = summarize_subtask_records(
            [
                {
                    "task_name": "AtomicTask",
                    "task_type": "atomic",
                    "rollout_id": "shared",
                    "failed": True,
                    "subtask_record": atomic_record,
                },
                {
                    "task_name": "CompositeTask",
                    "task_type": "composite",
                    "rollout_id": "shared",
                    "failed": True,
                    "subtask_record": composite_record,
                },
            ],
            task_type_filter="composite",
        )
        self.assertEqual(result["counts"]["rollouts"], 1)
        self.assertEqual(result["task_priority"], ["CompositeTask"])

    def test_load_dishwasher_uses_ordered_natural_language_subtasks(self):
        def payload(
            *,
            cup_grasped=False,
            cup_on_rack=False,
            bowl_grasped=False,
            bowl_on_rack=False,
            dishwasher_closed=False,
            task_success=False,
        ):
            dishes_on_rack = cup_on_rack and bowl_on_rack
            values = {
                "dishwasher_rack_accessible": True,
                "cup_grasped": cup_grasped,
                "cup_on_rack": cup_on_rack,
                "bowl_grasped": bowl_grasped,
                "bowl_on_rack": bowl_on_rack,
                "dishes_on_rack": dishes_on_rack,
                "dishwasher_closed": dishwasher_closed,
            }
            predicates = {}
            for name, value in values.items():
                predicates[name] = {
                    "value": value,
                    "required": name not in {"cup_grasped", "bowl_grasped"},
                }
            return {
                "task_name": "LoadDishwasher",
                "required_predicates": [
                    "dishwasher_rack_accessible",
                    "cup_on_rack",
                    "bowl_on_rack",
                    "dishes_on_rack",
                    "dishwasher_closed",
                ],
                "predicates": predicates,
                "task_success": task_success,
            }

        record = build_subtask_safe_record(
            [
                payload(),
                payload(cup_grasped=True),
                payload(cup_on_rack=True),
                payload(cup_on_rack=True, bowl_grasped=True),
                payload(
                    cup_on_rack=True,
                    bowl_on_rack=True,
                ),
                payload(
                    cup_on_rack=True,
                    bowl_on_rack=True,
                    dishwasher_closed=True,
                    task_success=True,
                ),
            ],
            [0, 1, 2, 3, 4],
            rollout_failed=False,
            rollout_id="load-dishwasher-success",
        )

        self.assertEqual(
            [definition["subtask_id"] for definition in record["semantic_subtasks"]],
            [
                "cup_grasped",
                "cup_on_rack",
                "bowl_grasped",
                "bowl_on_rack",
                "dishwasher_closed",
            ],
        )
        self.assertEqual(
            record["semantic_subtasks"][3],
            {
                "subtask_index": 3,
                "subtask_id": "bowl_on_rack",
                "instruction": "Place the bowl on the dishwasher rack.",
                "predicate_names": ["bowl_on_rack", "dishes_on_rack"],
                "source_subtask_ids": [
                    "bowl_on_rack",
                    "bowl_released_on_rack",
                ],
            },
        )
        self.assertEqual(
            [
                (
                    segment["subtask_id"],
                    segment["subtask_instruction"],
                )
                for segment in record["segments"]
            ],
            [
                ("cup_grasped", "Pick the cup from the counter."),
                (
                    "cup_on_rack",
                    "Place the cup on the dishwasher rack.",
                ),
                ("bowl_grasped", "Pick the bowl from the counter."),
                (
                    "bowl_on_rack",
                    "Place the bowl on the dishwasher rack.",
                ),
                ("dishwasher_closed", "Close the dishwasher."),
            ],
        )
        self.assertEqual(
            record["excluded_completed_subtasks"],
            [],
        )
        counts = validate_subtask_safe_record(
            record,
            rollout_id="load-dishwasher-success",
            rollout_failed=False,
            inference_environment_steps=[0, 1, 2, 3, 4],
            task_name="LoadDishwasher",
        )
        self.assertEqual(counts["successful_segments"], 5)
        self.assertEqual(counts["excluded_completed_subtasks"], 0)
        audit = summarize_subtask_records(
            [
                {
                    "task_name": "LoadDishwasher",
                    "task_type": "composite",
                    "rollout_id": "load-dishwasher-success",
                    "failed": False,
                    "subtask_record": record,
                }
            ]
        )
        self.assertEqual(audit["counts"]["task_subtask_pairs"], 5)
        self.assertEqual(
            audit["counts"]["excluded_completed_without_activation"],
            0,
        )
        bowl_placement = next(
            row
            for row in audit["task_subtask_rows"]
            if row["subtask_id"] == "bowl_on_rack"
        )
        self.assertEqual(
            bowl_placement["subtask_instruction"],
            "Place the bowl on the dishwasher rack.",
        )
        self.assertEqual(
            bowl_placement["predicate_names"],
            ["bowl_on_rack", "dishes_on_rack"],
        )
        self.assertEqual(
            bowl_placement["source_subtask_ids"],
            ["bowl_on_rack", "bowl_released_on_rack"],
        )


if __name__ == "__main__":
    unittest.main()
