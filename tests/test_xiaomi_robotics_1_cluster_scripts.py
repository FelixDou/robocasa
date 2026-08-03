import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "robocasa/scripts/xiaomi_robotics_1"
SETUP_SCRIPT = SCRIPT_ROOT / "setup_cluster.sh"
EVAL_SCRIPT = SCRIPT_ROOT / "evaluate_cluster.sh"
SUMMARY_SCRIPT = SCRIPT_ROOT / "summarize_results.py"
XR1_COMMIT = "4da1db0a4deefa6de7ebb4ef0b8754017290f5f7"
XR1_HF_REVISION = "0d1aa76d0d82debc9b611e4d1e231096434d5be4"

spec = importlib.util.spec_from_file_location("xr1_summary", SUMMARY_SCRIPT)
xr1_summary = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = xr1_summary
spec.loader.exec_module(xr1_summary)


def run_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
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


def fake_summary(expected_episodes: int = 2) -> dict:
    tasks = {}
    total_successes = 0
    for task_index, task_name in enumerate(xr1_summary.TARGET50):
        episodes = []
        successes = 0
        for episode_index in range(expected_episodes):
            success = (task_index + episode_index) % 2 == 0
            successes += int(success)
            episodes.append(
                {
                    "episode": episode_index,
                    "global_episode_index": (
                        task_index * expected_episodes + episode_index
                    ),
                    "seed": 7 + task_index * expected_episodes + episode_index,
                    "success": success,
                    "steps": 10,
                }
            )
        total_successes += successes
        tasks[task_name] = {
            "env_name": task_name,
            "split": "pretrain",
            "num_episodes": expected_episodes,
            "successes": successes,
            "success_rate": successes / expected_episodes,
            "horizon": 20,
            "episodes": episodes,
        }
    total_episodes = len(tasks) * expected_episodes
    return {
        "model_path": "/checkpoint",
        "robot_type": "robocasa365",
        "split": "pretrain",
        "task_set": "target50",
        "num_tasks": len(tasks),
        "num_episodes": total_episodes,
        "successes": total_successes,
        "episode_success_rate": total_successes / total_episodes,
        "mean_task_success_rate": total_successes / total_episodes,
        "replan_steps": 16,
        "obs_history": 4,
        "obs_interval": 2,
        "tasks": tasks,
    }


class TestXiaomiRobotics1ClusterScripts(unittest.TestCase):
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

    def test_setup_dry_run_pins_public_artifacts_and_bs_storage(self):
        result = run_script(SETUP_SCRIPT, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(XR1_COMMIT, result.stdout)
        self.assertIn(XR1_HF_REVISION, result.stdout)
        self.assertIn(
            "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365", result.stdout
        )
        self.assertIn("transformers==4.57.1", result.stdout)
        self.assertIn("torch==2.8.0", result.stdout)
        self.assertIn("flash_attn-2.8.3", result.stdout)
        self.assertIn("/gs/bs/tga-shinoda/felid/envs/", result.stdout)
        self.assertIn("--revision", result.stdout)
        self.assertNotIn("robocasa_checkpoints", result.stdout.split("/gs/fs")[0])

    def test_smoke_dry_run_uses_official_protocol_and_one_short_trial(self):
        result = run_script(
            EVAL_SCRIPT,
            "smoke",
            "--dry-run",
            "--gpus",
            "2",
            "--run-tag",
            "unit_smoke",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("episodes per task    : 1", result.stdout)
        self.assertIn("pretrain / target50", result.stdout)
        self.assertIn("CloseBlenderLid", result.stdout)
        self.assertIn("--horizon 20", result.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=2", result.stdout)
        self.assertIn("NUM_TRIALS=1", result.stdout)
        self.assertIn("OBS_HISTORY=4", result.stdout)
        self.assertIn("OBS_INTERVAL=2", result.stdout)
        self.assertIn("REPLAN_STEPS=16", result.stdout)
        self.assertIn("CROP_RATIO=0.95", result.stdout)

    def test_full_dry_run_uses_one_server_per_gpu_and_2500_protocol(self):
        result = run_script(
            EVAL_SCRIPT,
            "run",
            "--dry-run",
            "--gpus",
            "0,1,3",
            "--run-tag",
            "unit_full",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("workers              : 3", result.stdout)
        self.assertIn("ports                : 10086-10088", result.stdout)
        self.assertIn("episodes per task    : 50", result.stdout)
        self.assertEqual(result.stdout.count("deploy/server.py"), 3)
        self.assertIn("NUM_TRIALS=50", result.stdout)
        self.assertNotIn("--task-name", result.stdout)
        self.assertNotIn("SPLIT=target", result.stdout)

    def test_rejects_invalid_or_conflicting_selection(self):
        invalid = run_script(
            EVAL_SCRIPT, "run", "--dry-run", "--episodes", "0"
        )
        self.assertEqual(invalid.returncode, 2, invalid.stdout)
        self.assertIn("positive integer", invalid.stdout)

        conflicting = run_script(
            EVAL_SCRIPT,
            "run",
            "--dry-run",
            "--task",
            "CloseBlenderLid",
            "--max-tasks",
            "1",
        )
        self.assertEqual(conflicting.returncode, 2, conflicting.stdout)
        self.assertIn("either --task or --max-tasks", conflicting.stdout)

    def test_summary_validates_complete_balanced_inventory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(fake_summary()), encoding="utf-8")

            result = xr1_summary.summarize(
                summary_path,
                expected_episodes=2,
                expected_seed=7,
                source_commit=XR1_COMMIT,
                checkpoint_revision=XR1_HF_REVISION,
            )
            self.assertTrue(result["valid"], result["issues"])
            self.assertEqual(result["task_count"], 50)
            self.assertEqual(result["episodes"], 100)
            self.assertEqual(result["splits"]["atomic_seen"]["task_count"], 18)
            self.assertEqual(
                result["splits"]["composite_seen"]["task_count"], 16
            )
            self.assertEqual(
                result["splits"]["composite_unseen"]["task_count"], 16
            )
            self.assertFalse(result["complete_official_protocol"])

    def test_summary_rejects_missing_task_and_bad_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = fake_summary()
            payload["tasks"].pop("WeighIngredients")
            payload["tasks"]["CloseBlenderLid"]["episodes"][0]["seed"] = 99
            payload["num_tasks"] -= 1
            payload["num_episodes"] -= 2
            payload["successes"] -= 1
            payload["episode_success_rate"] = (
                payload["successes"] / payload["num_episodes"]
            )
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(payload), encoding="utf-8")

            result = xr1_summary.summarize(
                summary_path,
                expected_episodes=2,
                expected_seed=7,
            )
            self.assertFalse(result["valid"])
            self.assertTrue(
                any("missing target50 tasks" in issue for issue in result["issues"])
            )
            self.assertTrue(
                any("unexpected seed" in issue for issue in result["issues"])
            )

    def test_queue_validation_rejects_errors_and_incomplete_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = Path(temp_dir)
            for name in ("pending", "running", "results", "errors"):
                (queue / name).mkdir()
            (queue / "manifest.json").write_text("{}", encoding="utf-8")
            (queue / "errors/00000001.json").write_text("{}", encoding="utf-8")

            queue_result, issues = xr1_summary.validate_queue(
                queue, expected_result_count=2500
            )
            self.assertEqual(queue_result["errors"], 1)
            self.assertTrue(any("error records" in issue for issue in issues))
            self.assertTrue(any("expected 2500" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
