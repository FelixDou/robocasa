import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from robocasa.scripts.abot_m05.subtask_progress_recorder import (
    SubtaskProgressRecorder,
)
from robocasa.scripts.abot_m05.summarize_results import (
    EXPECTED_TASK_COUNTS,
    summarize_run,
)
from robocasa.scripts.abot_m05.summarize_subtask_progress import (
    summarize_subtask_progress,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = REPO_ROOT / "robocasa/scripts/abot_m05/setup_cluster.sh"
EVAL_SCRIPT = REPO_ROOT / "robocasa/scripts/abot_m05/evaluate_cluster.sh"
ABOT_COMMIT = "7642747ed2817b241dde5df06e17ee80192718ad"


def subtask_eval(first=False, second=False, success=False):
    return {
        "required_predicates": ["first", "second"],
        "predicates": {
            "first": {"value": first, "stage": "pick"},
            "second": {"value": second, "stage": "place"},
        },
        "task_success": success,
    }


class FakeSubtaskEnv:
    def __init__(self):
        self.env_name = "FakeCompositeTask"
        self.step_index = 0

    def reset(self, *, seed=None, options=None):
        self.step_index = 0
        return {}, {"success": False, "subtask_eval": subtask_eval()}

    def step(self, action):
        self.step_index += 1
        success = self.step_index >= 2
        return (
            {},
            float(success),
            success,
            False,
            {
                "success": success,
                "subtask_eval": subtask_eval(
                    first=True,
                    second=success,
                    success=success,
                ),
            },
        )

    def close(self):
        return None


def run_script(script, *args):
    env = os.environ.copy()
    env["CONDA_EXE"] = "conda"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class TestABotM05ClusterScripts(unittest.TestCase):
    def test_shell_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(SETUP_SCRIPT), str(EVAL_SCRIPT)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_setup_dry_run_uses_pinned_public_artifacts(self):
        result = run_script(SETUP_SCRIPT, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(ABOT_COMMIT, result.stdout)
        self.assertIn("acvlab/ABot-M0.5-RoboCasa365", result.stdout)
        self.assertIn("base_checkpoint/", result.stdout)
        self.assertIn("checkpoint_step/", result.stdout)
        self.assertIn("/gs/bs/tga-shinoda/felid/envs/abot_m05", result.stdout)
        self.assertIn("huggingface_hub==0.36.2", result.stdout)

    def test_smoke_dry_run_is_one_pretrain_episode(self):
        result = run_script(
            EVAL_SCRIPT,
            "smoke",
            "--dry-run",
            "--gpus",
            "0,1",
            "--run-tag",
            "unit_smoke",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("episodes per task     : 1", result.stdout)
        self.assertIn("SPLIT=pretrain", result.stdout)
        self.assertIn("ENV_NAME=CloseFridge", result.stdout)
        self.assertIn("/tmp/ut06746/abot_m05", result.stdout)
        self.assertIn("launch_server_env_sweep.sh", result.stdout)

    def test_full_dry_run_explicitly_uses_fifty_episodes_for_all_splits(self):
        result = run_script(
            EVAL_SCRIPT,
            "all",
            "--dry-run",
            "--gpus",
            "0,1,2,3",
            "--run-tag",
            "unit_full",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("episodes per task     : 50", result.stdout)
        self.assertIn("NUM_EPISODES=50", result.stdout)
        self.assertIn("eval_atomic_seen.sh", result.stdout)
        self.assertIn("eval_composite_seen.sh", result.stdout)
        self.assertIn("eval_composite_unseen.sh", result.stdout)
        self.assertNotIn("SPLIT=target", result.stdout)

    def test_two_gpu_subtask_progress_dry_run(self):
        result = run_script(
            EVAL_SCRIPT,
            "all",
            "--dry-run",
            "--gpus",
            "0,1",
            "--episodes",
            "10",
            "--subtask-progress",
            "--run-tag",
            "unit_subtask10",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("episodes per task     : 10", result.stdout)
        self.assertIn("subtask progress       : 1", result.stdout)
        self.assertIn("ROBOCASA_TRACK_SUBTASK_PROGRESS=1", result.stdout)
        self.assertIn("python_startup", result.stdout)
        self.assertIn("summarize_subtask_progress.py", result.stdout)

    def test_all_rejects_detached_parallel_split_launch(self):
        result = run_script(EVAL_SCRIPT, "all", "--dry-run", "--background")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("runs splits sequentially", result.stdout)

    def test_rejects_invalid_episode_count(self):
        result = run_script(EVAL_SCRIPT, "smoke", "--dry-run", "--episodes", "0")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("positive integer", result.stdout)

    def test_summary_accepts_complete_balanced_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            expected_episodes = 5
            expected_task_rates = []
            for split_index, (split_name, task_count) in enumerate(
                EXPECTED_TASK_COUNTS.items()
            ):
                per_env = []
                for task_index in range(task_count):
                    success_count = (task_index + split_index) % (
                        expected_episodes + 1
                    )
                    rate = success_count / expected_episodes
                    expected_task_rates.append(rate)
                    per_env.append(
                        {
                            "env_name": f"{split_name}_task_{task_index}",
                            "num_episodes": expected_episodes,
                            "success_count": success_count,
                            "success_rate": rate,
                        }
                    )
                split_dir = run_root / split_name
                split_dir.mkdir(parents=True)
                (split_dir / "summary.json").write_text(
                    json.dumps({"per_env": per_env}),
                    encoding="utf-8",
                )

            result = summarize_run(run_root, expected_episodes)
            self.assertTrue(result["complete"], result["issues"])
            self.assertEqual(result["unique_task_count"], 50)
            self.assertAlmostEqual(
                result["overall_task_average"],
                sum(expected_task_rates) / len(expected_task_rates),
            )

    def test_summary_rejects_partial_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = summarize_run(Path(temp_dir), expected_episodes=50)
            self.assertFalse(result["complete"])
            self.assertTrue(
                any("missing split summary" in issue for issue in result["issues"])
            )

    def test_subtask_recorder_writes_compact_episode_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subtask_progress.json"
            env = SubtaskProgressRecorder(
                FakeSubtaskEnv(),
                output_path=output_path,
            )
            env.reset(seed=7)
            env.step(None)
            env.step(None)
            env.close()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["env_name"], "FakeCompositeTask")
            self.assertEqual(len(payload["episodes"]), 1)
            episode = payload["episodes"][0]
            self.assertTrue(episode["subtask_eval_available"])
            self.assertEqual(episode["seed"], 7)
            self.assertEqual(episode["steps"], 2)
            self.assertEqual(episode["max_subtask_progress"], 1.0)
            self.assertEqual(
                episode["ordered_completed_required_subtasks"],
                ["first", "second"],
            )
            self.assertGreaterEqual(len(episode["progress_events"]), 2)
            self.assertNotIn("subtask_trace", episode)

    def test_subtask_aggregator_validates_all_fifty_tasks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            for split_name, task_count in EXPECTED_TASK_COUNTS.items():
                for task_index in range(task_count):
                    batch_dir = (
                        run_root
                        / split_name
                        / "envs"
                        / f"task_{task_index}"
                        / "batches"
                        / "ep000_n10_seed0"
                    )
                    batch_dir.mkdir(parents=True)
                    episodes = [
                        {
                            "episode_index": episode_index,
                            "success": False,
                            "subtask_eval_available": True,
                            "max_subtask_progress": 0.5,
                            "stuck_subtask": "second",
                            "failure_modes": ["wrong_receptacle"],
                        }
                        for episode_index in range(10)
                    ]
                    (batch_dir / "subtask_progress.json").write_text(
                        json.dumps(
                            {
                                "env_name": f"task_{task_index}",
                                "split": "pretrain",
                                "episodes": episodes,
                            }
                        ),
                        encoding="utf-8",
                    )

            summary = summarize_subtask_progress(run_root, expected_episodes=10)
            self.assertTrue(summary["complete"], summary["issues"])
            self.assertEqual(summary["rollout_count"], 500)
            self.assertEqual(
                summary["subtask_eval_available_rollouts"],
                500,
            )

            atomic_sidecar = next(
                (run_root / "atomic_seen").rglob("subtask_progress.json")
            )
            payload = json.loads(atomic_sidecar.read_text(encoding="utf-8"))
            payload["episodes"][0]["subtask_eval_available"] = False
            atomic_sidecar.write_text(json.dumps(payload), encoding="utf-8")
            invalid_summary = summarize_subtask_progress(
                run_root,
                expected_episodes=10,
            )
            self.assertFalse(invalid_summary["complete"])
            self.assertTrue(
                any(
                    "atomic_seen: subtask evaluation unavailable" in issue
                    for issue in invalid_summary["issues"]
                )
            )


if __name__ == "__main__":
    unittest.main()
