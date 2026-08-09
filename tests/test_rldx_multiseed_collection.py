import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.run_rldx_multiseed_collection import (
    DEFAULT_TASKS,
    PLAN_NAME,
    build_collection_plan,
    collection_provenance,
    collector_command,
    quarantine_incomplete_block,
    rollout_counts_by_block,
    server_command,
    server_environment,
    write_or_verify_plan,
)


class TestRLDXMultiseedCollection(unittest.TestCase):
    def make_plan(self, root, **overrides):
        values = {
            "output_root": Path(root) / "output",
            "log_root": Path(root) / "logs",
            "rldx_repo": Path(root) / "rldx",
            "robocasa_commit": "robocasa-commit",
            "rldx_commit": "rldx-commit",
            "sim_python": sys.executable,
        }
        values.update(overrides)
        return build_collection_plan(**values)

    def test_exact_balanced_cyclic_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.make_plan(tmp)

        self.assertEqual(plan["total_rollouts"], 1000)
        self.assertEqual(
            plan["per_task_rollouts"], {task: 250 for task in DEFAULT_TASKS}
        )
        self.assertEqual(len(plan["blocks"]), 12)
        self.assertEqual(sum(plan["rollouts_per_block"]), 250)
        self.assertEqual(plan["rollouts_per_block"], [21] * 10 + [20] * 2)
        self.assertEqual(len(set(plan["server_rng_seeds"])), 48)
        self.assertEqual(len(set(plan["environment_seeds"])), 12)

        for block in plan["blocks"]:
            self.assertEqual(
                {lane["task"] for lane in block["lanes"]}, set(DEFAULT_TASKS)
            )
            self.assertEqual(len({lane["gpu"] for lane in block["lanes"]}), 4)
            self.assertEqual(len({lane["port"] for lane in block["lanes"]}), 4)

        for task in DEFAULT_TASKS:
            block_counts = plan["schedule"]["task_gpu_blocks"][task]
            rollout_counts = plan["schedule"]["task_gpu_rollouts"][task]
            self.assertEqual(set(block_counts.values()), {3})
            self.assertLessEqual(
                max(rollout_counts.values()) - min(rollout_counts.values()), 1
            )
            self.assertEqual(sum(rollout_counts.values()), 250)

    def test_plan_fingerprint_is_stable_and_covers_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.make_plan(tmp)
            second = self.make_plan(tmp)
            changed = self.make_plan(tmp, server_seed_start=2000)
        self.assertEqual(first["plan_fingerprint"], second["plan_fingerprint"])
        self.assertNotEqual(first["plan_fingerprint"], changed["plan_fingerprint"])

    def test_requires_complete_task_gpu_rotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "divisible"):
                self.make_plan(tmp, num_blocks=10)

    def test_rollout_count_allocation_is_exact(self):
        self.assertEqual(rollout_counts_by_block(11, 4), [3, 3, 3, 2])
        with self.assertRaisesRegex(ValueError, "at least num_blocks"):
            rollout_counts_by_block(3, 4)

    def test_commands_pin_lane_seed_task_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.make_plan(tmp)
        block = plan["blocks"][0]
        lane = block["lanes"][2]

        server = server_command(plan, lane)
        self.assertIn("RLDX_SERVER_RNG_SEED", server[server.index("-c") + 1])
        self.assertEqual(server[server.index("--port") + 1], str(lane["port"]))
        environment = server_environment(plan, lane)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], str(lane["gpu"]))
        self.assertEqual(
            environment["RLDX_SERVER_RNG_SEED"], str(lane["server_rng_seed"])
        )

        collector = collector_command(plan, block, lane)
        self.assertEqual(collector[collector.index("--tasks") + 1], lane["task"])
        self.assertEqual(
            collector[collector.index("--safe-feature-mode") + 1],
            "action_observation_context",
        )
        policy_config = json.loads(collector[collector.index("--policy-config") + 1])
        self.assertEqual(
            policy_config["collection_provenance"],
            collection_provenance(plan, block, lane),
        )
        self.assertNotIn("--resume", collector)
        self.assertNotIn("--success-quota", collector)
        self.assertNotIn("--failure-quota", collector)

    def test_immutable_plan_and_incomplete_block_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.make_plan(tmp)
            written = write_or_verify_plan(plan)
            self.assertTrue((Path(plan["output_root"]) / PLAN_NAME).is_file())
            self.assertEqual(
                write_or_verify_plan(plan, resume=True)["plan_fingerprint"],
                written["plan_fingerprint"],
            )
            with self.assertRaises(FileExistsError):
                write_or_verify_plan(plan)
            changed = self.make_plan(tmp, environment_seed_start=200)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                write_or_verify_plan(changed, resume=True)

            block = plan["blocks"][0]
            block_dir = Path(block["block_dir"])
            block_dir.mkdir(parents=True)
            marker = block_dir / "partial.txt"
            marker.write_text("preserve me")
            destination = quarantine_incomplete_block(plan, block)
            self.assertIsNotNone(destination)
            self.assertEqual((destination / marker.name).read_text(), "preserve me")
            self.assertFalse(block_dir.exists())


if __name__ == "__main__":
    unittest.main()
