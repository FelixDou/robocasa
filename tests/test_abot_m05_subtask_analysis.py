import csv
import json
import tempfile
import unittest
from pathlib import Path

from robocasa.scripts.abot_m05.analyze_subtask_progress import (
    EXPECTED_TASK_COUNTS,
    analyze_run,
)


def write_synthetic_run(run_root: Path, episodes_per_task: int = 2) -> None:
    for split_index, (split, task_count) in enumerate(
        EXPECTED_TASK_COUNTS.items()
    ):
        for task_index in range(task_count):
            env_name = f"{split}_task_{task_index:02d}"
            batch_dir = (
                run_root
                / split
                / "envs"
                / env_name
                / "batches"
                / f"ep000_n{episodes_per_task}_seed0"
            )
            batch_dir.mkdir(parents=True)
            episodes = []
            for episode_index in range(episodes_per_task):
                success = (task_index + episode_index + split_index) % 3 == 0
                progress = (
                    1.0
                    if success
                    else ((task_index + episode_index) % 3) / 3
                )
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "seed": episode_index,
                        "steps": 100,
                        "success": success,
                        "subtask_eval_available": True,
                        "max_subtask_progress": progress,
                        "ordered_completed_required_subtasks": (
                            ["pick", "place"] if success else ["pick"]
                        ),
                        "stuck_subtask": "" if success else "place",
                        "failure_modes": [] if success else ["wrong_receptacle"],
                    }
                )
            (batch_dir / "subtask_progress.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "env_name": env_name,
                        "episodes": episodes,
                    }
                ),
                encoding="utf-8",
            )


class TestABotM05SubtaskAnalysis(unittest.TestCase):
    def test_complete_run_writes_reproducible_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "run"
            write_synthetic_run(run_root)
            result = analyze_run(
                run_root,
                expected_episodes=2,
                bootstrap_samples=100,
                make_figures=False,
            )

            output_dir = run_root / "subtask_analysis"
            self.assertTrue(result["integrity_checks_passed"])
            self.assertEqual(result["overall"]["task_count"], 50)
            self.assertEqual(result["overall"]["rollout_count"], 100)
            self.assertTrue(
                0.0 <= result["overall"]["success_rate"] <= 1.0
            )
            self.assertGreaterEqual(
                result["overall"]["mean_max_subtask_progress"],
                result["overall"]["success_rate"],
            )
            self.assertTrue(
                (output_dir / "subtask_progress_statistics.json").is_file()
            )
            self.assertTrue((output_dir / "README.md").is_file())

            with (output_dir / "rollout_statistics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rollout_rows = list(csv.DictReader(handle))
            with (output_dir / "task_statistics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                task_rows = list(csv.DictReader(handle))
            self.assertEqual(len(rollout_rows), 100)
            self.assertEqual(len(task_rows), 50)

    def test_incomplete_task_fails_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "run"
            write_synthetic_run(run_root)
            missing = next(run_root.glob("atomic_seen/**/subtask_progress.json"))
            missing.unlink()

            with self.assertRaisesRegex(ValueError, "integrity checks failed"):
                analyze_run(
                    run_root,
                    expected_episodes=2,
                    bootstrap_samples=0,
                    make_figures=False,
                )
            self.assertFalse((run_root / "subtask_analysis").exists())


if __name__ == "__main__":
    unittest.main()
