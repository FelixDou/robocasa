import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa/recovery/create_atomic_failure_dataset.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "create_atomic_failure_dataset", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAtomicFailureDataset(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_save_dict_action_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "actions.npz"
            self.module.save_action_trajectory(
                [
                    {
                        "action.eef_pos_delta": np.array([1.0, 2.0, 3.0]),
                        "action.gripper_close": 1.0,
                    },
                    {
                        "action.eef_pos_delta": np.array([4.0, 5.0, 6.0]),
                        "action.gripper_close": 0.0,
                    },
                ],
                path,
            )

            data = np.load(path)
            self.assertEqual(int(data["num_steps"]), 2)
            self.assertEqual(data["action_eef_pos_delta"].shape, (2, 3))
            self.assertEqual(data["action_gripper_close"].shape, (2,))
            key_map = json.loads(str(data["action_key_map_json"]))
            self.assertEqual(
                key_map["action_eef_pos_delta"], "action.eef_pos_delta"
            )
            self.assertEqual(
                key_map["action_gripper_close"], "action.gripper_close"
            )

    def test_runtime_subtask_sequence_uses_required_predicate_order(self):
        summary = {
            "final_subtask_eval": {
                "required_predicates": [
                    "object_grasped",
                    "object_at_target_and_released",
                ],
                "predicates": {
                    "object_grasped": {
                        "description": "Grasp object.",
                        "stage": "transient",
                        "required": True,
                        "value": True,
                    },
                    "object_at_target_and_released": {
                        "description": "Place and release.",
                        "stage": "task_success",
                        "required": True,
                        "value": False,
                    },
                },
            }
        }

        sequence = self.module.runtime_subtask_sequence(summary)

        self.assertEqual(
            [entry["predicate"] for entry in sequence],
            ["object_grasped", "object_at_target_and_released"],
        )
        self.assertEqual(sequence[1]["stage"], "task_success")
        self.assertFalse(sequence[1]["value_final"])

    def test_observation_instruction_reads_gym_language_field(self):
        self.assertEqual(
            self.module.observation_instruction(
                {"annotation.human.task_description": "Open the drawer."}
            ),
            "Open the drawer.",
        )
        self.assertEqual(
            self.module.observation_instruction({"language": "Turn on the faucet."}),
            "Turn on the faucet.",
        )

    def test_build_sample_records_atomic_failure_schema(self):
        summary = {
            "task_success": False,
            "final_subtask_eval": {
                "subtask_progress": 0.5,
                "required_predicates": [
                    "object_grasped",
                    "object_at_target_and_released",
                ],
                "predicates": {
                    "object_grasped": {
                        "value": True,
                        "required": True,
                        "stage": "transient",
                        "description": "Grasp object.",
                    },
                    "object_at_target_and_released": {
                        "value": False,
                        "required": True,
                        "stage": "task_success",
                        "description": "Place and release.",
                    },
                },
            },
            "subtask_trace": [
                {"subtask_eval_available": True, "ordered_subtask_progress": 0.5}
            ],
            "max_subtask_progress": 0.5,
            "ordered_completed_required_subtasks": ["object_grasped"],
            "failed_required_predicates_final": ["object_at_target_and_released"],
            "ordered_current_subtask": "object_at_target_and_released",
            "failure_modes": ["wrong_receptacle"],
            "failed_preconditions": [],
            "failed_preconditions_ever": [],
            "completed_predicates_ever": ["object_grasped"],
        }

        self.module.summarize_subtask_rollout = lambda *args, **kwargs: summary
        args = Namespace(
            stuck_patience=10,
            include_trace=True,
            task_set="atomic_seen",
            split="test",
            policy_name="unit_policy",
            policy_module="unit:factory",
            policy_arg=[],
            env_interface="gym",
        )
        rollout = {
            "subtask_evals": [{"subtask_progress": 0.0}],
            "instruction": "Pick and place object.",
            "num_steps": 12,
            "success": False,
        }

        sample = self.module.build_sample(
            args=args,
            task_name="PickPlaceCounterToCabinet",
            rollout_i=3,
            seed=42,
            horizon=400,
            rollout=rollout,
            task_plan=None,
            action_path=Path("actions/sample.npz"),
            video_path=Path("videos/sample.mp4"),
        )

        self.assertEqual(sample["task_granularity"], "atomic")
        self.assertEqual(sample["failed_subtask_predicates"], ["object_at_target_and_released"])
        self.assertEqual(sample["failure_diagnostic"]["failed_atomic"], "PickPlaceCounterToCabinet")
        self.assertEqual(sample["failure_diagnostic"]["failed_subtask"], "object_at_target_and_released")
        self.assertEqual(sample["subtask_sequence_source"], "runtime_required_predicates")
        self.assertEqual(sample["subtask_sequence"][0]["predicate"], "object_grasped")
        self.assertEqual(sample["max_subtask_progress"], 0.5)

    def test_validate_atomic_tasks_rejects_unknown_by_default(self):
        self.module.load_registered_atomic_task_names = lambda: {"OpenDrawer"}

        with self.assertRaisesRegex(ValueError, "Non-atomic or unknown"):
            self.module.validate_atomic_tasks(["OpenDrawer", "PrepareCoffee"])

        self.module.validate_atomic_tasks(
            ["PrepareCoffee"], allow_unregistered=True
        )

    def test_run_dataset_creation_with_mock_atomic_env(self):
        class FakeEnv:
            def __init__(self):
                self.steps = 0

            def reset(self):
                self.steps = 0
                return {"obs": 0}

            def step(self, action):
                self.steps += 1
                return {"obs": self.steps}, 0.0, self.steps >= 2, {}

            def close(self):
                pass

        class FakePolicy:
            def __init__(self, env):
                self.env = env

            def __call__(self, obs, instruction=None):
                return {
                    "action.eef_pos_delta": np.array([self.env.steps], dtype=np.float32)
                }

        summary = {
            "task_success": False,
            "final_subtask_eval": {
                "subtask_progress": 0.0,
                "required_predicates": ["drawer_open"],
                "predicates": {
                    "drawer_open": {
                        "value": False,
                        "required": True,
                        "stage": "fixture_state",
                        "description": "Open the drawer.",
                    }
                },
            },
            "subtask_trace": [
                {"subtask_eval_available": True, "ordered_subtask_progress": 0.0}
            ],
            "max_subtask_progress": 0.0,
            "ordered_completed_required_subtasks": [],
            "failed_required_predicates_final": ["drawer_open"],
            "ordered_current_subtask": "drawer_open",
            "failure_modes": ["no_progress"],
            "failed_preconditions": [],
            "failed_preconditions_ever": [],
            "completed_predicates_ever": [],
        }

        self.module.import_rollout_runtime = lambda: {
            "RandomPolicy": FakePolicy,
            "call_factory": lambda factory, env, policy_args: factory(env),
            "json_default": self.module.json_default,
            "load_factory": lambda spec: FakePolicy,
            "make_env": lambda *args, **kwargs: FakeEnv(),
            "open_deferred_video_writer": lambda *args, **kwargs: None,
            "open_video_writer": lambda *args, **kwargs: None,
            "parse_policy_args": lambda values: {},
            "resolve_tasks": lambda args: args.envs or ["OpenDrawer"],
            "_append_video_frame_from_env": lambda *args, **kwargs: None,
            "_env_task_instruction": lambda env: "Open drawer.",
            "_is_task_success": lambda info, reward=None, env=None: False,
            "_step_env": lambda env, action: env.step(action),
            "call_policy": lambda policy, obs, instruction=None: policy(
                obs, instruction=instruction
            ),
            "get_subtask_eval": lambda env: summary["final_subtask_eval"],
            "get_task_horizon": lambda task_name: 2,
            "summarize_subtask_rollout": lambda *args, **kwargs: summary,
        }
        self.module.load_registered_atomic_task_names = lambda: {"OpenDrawer"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = Namespace(
                output_dir=output_dir,
                output=None,
                policy_module=None,
                policy_name="random",
                policy_arg=[],
                random_policy=True,
                env_interface="gym",
                task_set="atomic_seen",
                envs=["OpenDrawer"],
                allow_unregistered_atomic_envs=False,
                split="test",
                num_rollouts=1,
                seed=7,
                horizon=2,
                stuck_patience=10,
                include_trace=True,
                include_successes=False,
                max_failed_per_task=None,
                max_success_per_task=None,
                stop_after_failures=None,
                subtask_plans=None,
                record_actions=True,
                record_videos=False,
                keep_discarded_videos=False,
                enable_render=False,
                video_camera_name="robot0_agentview_center",
                video_height=512,
                video_width=768,
                video_fps=20,
                video_direct_sim_render=False,
                video_render_source="auto",
            )

            self.module.run_dataset_creation(args)

            manifest_path = output_dir / "atomic_failure_dataset.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertFalse(manifest["partial"])
            self.assertEqual(manifest["summary"]["num_failures"], 1)
            sample = manifest["samples"][0]
            self.assertEqual(sample["task_name"], "OpenDrawer")
            self.assertEqual(sample["failure_diagnostic"]["failed_atomic"], "OpenDrawer")
            action_path = output_dir / sample["action_trajectory_path"]
            self.assertTrue(action_path.exists())
            actions = np.load(action_path)
            self.assertEqual(int(actions["num_steps"]), 2)


if __name__ == "__main__":
    unittest.main()
