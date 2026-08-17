import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.replay_dense_subtask_labels import (  # noqa: E402
    DEFAULT_XR1_DENSE_EXPANDED_TASKS,
    DEFAULT_XR1_DENSE_PILOT_TASKS,
    dense_inference_labels,
    load_action_trajectory,
    load_selection_rollout_ids,
    replay_rollout,
    select_records,
)


class FakeEnv:
    def __init__(self):
        self.step_index = 0
        self.closed = False

    def payload(self):
        first = self.step_index >= 1
        second = self.step_index >= 99
        return {
            "task_name": "BreadSelection",
            "required_predicates": [
                "croissant_on_cutting_board",
                "jam_on_cutting_board",
                "gripper_released",
            ],
            "predicates": {
                "croissant_on_cutting_board": {
                    "value": first,
                    "required": True,
                    "stage": "subtask",
                },
                "jam_on_cutting_board": {
                    "value": second,
                    "required": True,
                    "stage": "subtask",
                },
                "gripper_released": {
                    "value": second,
                    "required": True,
                    "stage": "task_success",
                },
            },
            "task_success": first and second,
        }

    def reset(self, **kwargs):
        self.step_index = 0
        return {}, {"subtask_eval": self.payload()}

    def close(self):
        self.closed = True


def failed_bread_record():
    return SimpleNamespace(
        rollout_id="bread-failure",
        task_name="BreadSelection",
        failed=True,
        environment_seed=17,
        environment_reset_index=0,
        seed_protocol="official_xiaomi",
        num_env_steps=2,
        valid_sequence_length=2,
        inference_env_steps=[0, 1],
        action_recording_requested=True,
        action_path="actions/BreadSelection/bread-failure.npz",
    )


class TestDenseSubtaskReplay(unittest.TestCase):
    def test_default_task_sets_are_nested_and_composite(self):
        self.assertEqual(len(DEFAULT_XR1_DENSE_PILOT_TASKS), 5)
        self.assertEqual(len(DEFAULT_XR1_DENSE_EXPANDED_TASKS), 10)
        self.assertEqual(
            DEFAULT_XR1_DENSE_EXPANDED_TASKS[:5],
            DEFAULT_XR1_DENSE_PILOT_TASKS,
        )

    def test_load_vector_and_dictionary_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            vector = Path(tmp) / "vector.npz"
            np.savez_compressed(
                vector,
                num_steps=np.asarray(2),
                action_key_map_json=np.asarray(json.dumps({"action": "action"})),
                action=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
            values = load_action_trajectory(vector)
            np.testing.assert_array_equal(values[1], [3.0, 4.0])

            mapping = Path(tmp) / "mapping.npz"
            np.savez_compressed(
                mapping,
                num_steps=np.asarray(2),
                action_key_map_json=np.asarray(
                    json.dumps({"arm_action": "arm/action", "gripper": "gripper"})
                ),
                arm_action=np.asarray([[1.0], [2.0]], dtype=np.float32),
                gripper=np.asarray([0.0, 1.0], dtype=np.float32),
            )
            values = load_action_trajectory(mapping)
            self.assertEqual(set(values[0]), {"arm/action", "gripper"})
            self.assertEqual(float(values[1]["gripper"]), 1.0)

    def test_dense_labels_use_one_for_completed_and_zero_for_failed_stage(self):
        record = {
            "rollout_id": "parent",
            "task_name": "Task",
            "segments": [
                {"subtask_id": "first", "segment_id": "s0", "failure_label": 0},
                {"subtask_id": "second", "segment_id": "s1", "failure_label": 1},
            ],
            "inference_records": [
                {"inference_index": 0, "subtask_id": "first"},
                {"inference_index": 1, "subtask_id": "first"},
                {"inference_index": 2, "subtask_id": "second"},
                {"inference_index": 3, "subtask_id": None},
            ],
        }
        dense = dense_inference_labels(record)
        self.assertEqual(
            [item["subtask_success_label"] for item in dense],
            [1, 1, 0],
        )
        self.assertEqual([item["failure_label"] for item in dense], [0, 0, 1])

    def test_action_replay_builds_real_ordered_dense_labels(self):
        env = FakeEnv()

        def make_env(*args):
            return env

        def step_fn(environment, action):
            environment.step_index += 1
            payload = environment.payload()
            return {}, 0.0, False, {"subtask_eval": payload, "success": False}

        annotation, audit = replay_rollout(
            failed_bread_record(),
            [np.zeros(12), np.zeros(12)],
            make_env_fn=make_env,
            step_fn=step_fn,
            success_fn=lambda info, reward, environment: False,
            subtask_eval_fn=lambda environment: environment.payload(),
        )
        dense = annotation["dense_inference_labels"]
        self.assertEqual(
            [(row["subtask_id"], row["subtask_success_label"]) for row in dense],
            [
                ("PickPlaceCounterToCounter_1_place", 1),
                ("PickPlaceCabinetToCounter_2_place", 0),
            ],
        )
        self.assertEqual(audit["dense_success_samples"], 1)
        self.assertEqual(audit["dense_failure_samples"], 1)
        self.assertTrue(audit["replay_terminal_outcome_matches"])
        self.assertTrue(env.closed)

    def test_replay_rejects_terminal_outcome_drift(self):
        env = FakeEnv()
        with self.assertRaisesRegex(ValueError, "terminal outcome mismatch"):
            replay_rollout(
                failed_bread_record(),
                [np.zeros(12), np.zeros(12)],
                make_env_fn=lambda *args: env,
                step_fn=lambda environment, action: (
                    {},
                    1.0,
                    False,
                    {"subtask_eval": environment.payload(), "success": True},
                ),
                success_fn=lambda info, reward, environment: True,
                subtask_eval_fn=lambda environment: environment.payload(),
            )

    def test_record_selection_is_deterministic_and_requires_both_outcomes(self):
        records = []
        for task in DEFAULT_XR1_DENSE_PILOT_TASKS:
            for seed, failed in ((3, True), (1, False), (2, True)):
                records.append(
                    SimpleNamespace(
                        task_name=task,
                        environment_seed=seed,
                        environment_reset_index=seed,
                        rollout_id=f"{task}-{seed}",
                        failed=failed,
                    )
                )
        selected = select_records(
            records,
            DEFAULT_XR1_DENSE_PILOT_TASKS,
            max_rollouts_per_task=3,
        )
        self.assertEqual(len(selected), 15)
        self.assertEqual(
            [record.environment_seed for record in selected[:3]],
            [1, 2, 3],
        )
        smoke = select_records(
            records,
            DEFAULT_XR1_DENSE_PILOT_TASKS,
            max_rollouts_per_task=None,
            successes_per_task=1,
            failures_per_task=1,
        )
        self.assertEqual(len(smoke), 10)
        for task in DEFAULT_XR1_DENSE_PILOT_TASKS:
            task_rows = [record for record in smoke if record.task_name == task]
            self.assertEqual({record.failed for record in task_rows}, {False, True})

    def test_official_export_mapping_freezes_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conversion_report.json"
            path.write_text(
                json.dumps(
                    {"mapping": [{"rollout_id": "a"}, {"rollout_id": "b"}]}
                )
            )
            self.assertEqual(load_selection_rollout_ids(path), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
