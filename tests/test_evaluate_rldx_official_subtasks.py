import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa/recovery/evaluate_rldx_official_subtasks.py"
)
SUBTASK_MODULE_PATH = MODULE_PATH.with_name("subtask_eval.py")
PRINT_ACCURACY_MODULE_PATH = MODULE_PATH.with_name("print_subtask_accuracy.py")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_rldx_official_subtasks", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(progress, completed):
    return {
        "predicates": {
            "first": {"value": bool(completed), "required": True},
        },
        "required_predicates": ["first"],
        "subtask_progress": float(progress),
        "task_success": False,
    }


class TestEvaluateRLDXOfficialSubtasks(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_loads_official_task_yaml_shape(self):
        text = """
atomic_seen:
  - AtomicTask
composite_seen:
  - SeenTask
composite_unseen:
  - UnseenTask
task_horizons:
  AtomicTask: 100
  SeenTask: 200
  UnseenTask: 300
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "task_sets.yaml"
            path.write_text(text)
            task_sets, horizons = self.module.load_rldx_task_config(path)

        self.assertEqual(task_sets["atomic_seen"], ["AtomicTask"])
        self.assertEqual(task_sets["composite_unseen"], ["UnseenTask"])
        self.assertEqual(horizons["SeenTask"], 200)

    def test_extracts_every_primitive_step_payload(self):
        values = np.array([[payload(0.0, False), payload(1.0, True)]], dtype=object)
        env_value = self.module._vector_item(values, 0)
        extracted = self.module._extract_subtask_payloads(env_value)

        self.assertEqual(len(extracted), 2)
        self.assertEqual([row["subtask_progress"] for row in extracted], [0.0, 1.0])

    def test_subtask_eval_unwraps_multiple_environment_wrappers(self):
        spec = importlib.util.spec_from_file_location(
            "subtask_eval", SUBTASK_MODULE_PATH
        )
        subtask_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(subtask_module)

        class BaseEnv:
            def get_subtask_progress(self):
                return {"subtask_progress": 0.5}

        class Wrapper:
            def __init__(self, env):
                self.env = env

        wrapped = Wrapper(Wrapper(Wrapper(BaseEnv())))
        self.assertEqual(
            subtask_module.get_subtask_eval(wrapped), {"subtask_progress": 0.5}
        )

    def test_finished_episode_prefers_final_info(self):
        final_payload = payload(1.0, True)
        reset_payload = payload(0.0, False)
        infos = {
            "subtask_eval": np.array([[reset_payload]], dtype=object),
            "final_info": np.array(
                [{"subtask_eval": np.array([final_payload], dtype=object)}],
                dtype=object,
            ),
        }

        step_info = self.module._episode_step_info(infos, 0, episode_finished=True)
        extracted = self.module._extract_subtask_payloads(step_info["subtask_eval"])

        self.assertEqual(extracted, [final_payload])

    def test_primitive_step_count_handles_vector_autoreset_info(self):
        self.assertEqual(
            self.module._primitive_step_count(
                {"subtask_eval": np.array([payload(0.0, False)], dtype=object)},
                fallback=8,
            ),
            1,
        )

    def test_aggregate_uses_task_average_and_flags_missing_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for task, group, successes in (
                ("TaskA", "atomic_seen", [True, False]),
                ("TaskB", "atomic_seen", [True, True]),
            ):
                task_dir = root / task
                task_dir.mkdir()
                rollouts = [
                    {
                        "success": success,
                        "max_subtask_progress": float(success),
                        "failure_modes": [],
                    }
                    for success in successes
                ]
                (task_dir / "subtask_rollouts.json").write_text(
                    json.dumps(
                        {
                            "env_name": task,
                            "task_group": group,
                            "partial": False,
                            "rollouts": rollouts,
                            "summary": self.module.summarize_rollouts(rollouts),
                        }
                    )
                )

            summary = self.module.aggregate_task_outputs(
                root, expected_tasks=["TaskA", "TaskB", "TaskC"]
            )

        self.assertEqual(summary["missing_tasks"], ["TaskC"])
        self.assertAlmostEqual(
            summary["groups"]["atomic_seen"]["mean_task_success_rate"], 0.75
        )

    def test_accuracy_report_uses_current_make_ice_lemonade_split(self):
        spec = importlib.util.spec_from_file_location(
            "print_subtask_accuracy", PRINT_ACCURACY_MODULE_PATH
        )
        accuracy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(accuracy_module)

        self.assertEqual(
            accuracy_module._task_group("MakeIceLemonade"), "composite_unseen"
        )


if __name__ == "__main__":
    unittest.main()
