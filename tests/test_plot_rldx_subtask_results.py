from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa/recovery/plot_rldx_subtask_results.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "plot_rldx_subtask_results", MODULE_PATH
)
plots = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(plots)


class TestPlotRldxSubtaskResults(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        tasks = []
        subtask_tasks = []
        bottlenecks = []
        specs = [
            ("AtomicEasy", "atomic_seen", 0.8, 0.9, 2),
            ("AtomicHard", "atomic_seen", 0.2, 0.4, 7),
            ("SeenEasy", "composite_seen", 0.5, 0.8, 4),
            ("SeenHard", "composite_seen", 0.0, 0.3, 9),
            ("UnseenEasy", "composite_unseen", 0.3, 0.6, 5),
            ("UnseenHard", "composite_unseen", 0.0, 0.1, 10),
        ]
        for task, group, success, progress, blocker_count in specs:
            tasks.append(
                {
                    "task": task,
                    "group": group,
                    "partial": False,
                    "num_rollouts": 10,
                    "num_successes": int(10 * success),
                    "success_rate": success,
                    "mean_max_subtask_progress": progress,
                    "subtask_eval_unavailable_rollouts": 0,
                    "path": str(root / task / "subtask_rollouts.json"),
                }
            )
            subtask_tasks.append(
                {
                    "task": task,
                    "group": group,
                    "rollout_count": 10,
                    "success_count": int(10 * success),
                    "success_rate": success,
                    "mean_max_subtask_progress": progress,
                    "subtasks": [
                        {
                            "name": "first_step",
                            "description": "Complete the first step.",
                            "completed_rollouts": int(10 * progress),
                            "rollout_count": 10,
                            "accuracy": progress,
                            "first_blocker_count": blocker_count,
                        }
                    ],
                }
            )
            bottlenecks.append(
                {
                    "task": task,
                    "group": group,
                    "task_success": success,
                    "mean_ordered_progress": progress,
                    "most_common_first_blocker": "Complete the first step.",
                    "most_common_first_blocker_key": "first_step",
                    "most_common_first_blocker_count": blocker_count,
                }
            )

        groups = {}
        for group in plots.GROUP_ORDER:
            members = [row for row in tasks if row["group"] == group]
            groups[group] = {
                "num_tasks": len(members),
                "mean_task_success_rate": sum(row["success_rate"] for row in members)
                / len(members),
                "mean_task_max_subtask_progress": sum(
                    row["mean_max_subtask_progress"] for row in members
                )
                / len(members),
                "subtask_eval_unavailable_rollouts": 0,
            }
        benchmark = {
            "num_tasks": len(tasks),
            "num_complete_tasks": len(tasks),
            "overall_mean_task_success_rate": sum(row["success_rate"] for row in tasks)
            / len(tasks),
            "overall_mean_task_max_subtask_progress": sum(
                row["mean_max_subtask_progress"] for row in tasks
            )
            / len(tasks),
            "groups": groups,
            "missing_tasks": [],
            "errors": [],
            "tasks": tasks,
        }
        (root / "benchmark_summary.json").write_text(json.dumps(benchmark, indent=2))
        (root / "subtask_accuracy.json").write_text(
            json.dumps(
                {"tasks": subtask_tasks, "bottlenecks": bottlenecks},
                indent=2,
            )
        )

    def test_build_figures_and_supporting_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            output_dir = root / "figures"
            args = Namespace(
                input_dir=root,
                benchmark_summary=None,
                subtask_accuracy=None,
                output_dir=output_dir,
                formats=["png"],
                dpi=90,
                top_bottlenecks=6,
                scatter_labels=3,
            )
            outputs = plots.build_figures(args)

            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertTrue(all(path.stat().st_size > 1_000 for path in outputs))
            self.assertTrue((output_dir / "task_metrics.csv").is_file())
            self.assertTrue((output_dir / "group_metrics.csv").is_file())
            self.assertTrue((output_dir / "top_first_blockers.csv").is_file())
            self.assertTrue((output_dir / "figure_manifest.json").is_file())
            self.assertTrue((output_dir / "README.md").is_file())

            manifest = json.loads((output_dir / "figure_manifest.json").read_text())
            self.assertEqual(manifest["task_count"], 6)
            self.assertEqual(manifest["rollout_count"], 60)
            self.assertEqual(len(manifest["figures"]), 5)

    def test_rejects_partial_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            path = root / "benchmark_summary.json"
            benchmark = json.loads(path.read_text())
            benchmark["tasks"][0]["partial"] = True
            path.write_text(json.dumps(benchmark))

            with self.assertRaisesRegex(ValueError, "partial tasks"):
                plots.build_figures(
                    Namespace(
                        input_dir=root,
                        benchmark_summary=None,
                        subtask_accuracy=None,
                        output_dir=root / "figures",
                        formats=["png"],
                        dpi=90,
                        top_bottlenecks=6,
                        scatter_labels=3,
                    )
                )


if __name__ == "__main__":
    unittest.main()
