import json
import io
import random
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages


install_lightweight_robocasa_packages()

from robocasa.recovery.counterfactual_branch import (  # noqa: E402
    BranchSpec,
    analyze_phase2_replay,
    run_counterfactual_branch,
)
from robocasa.recovery.full_snapshot import (  # noqa: E402
    capture_full_snapshot,
    load_full_snapshot,
    restore_full_snapshot,
    save_full_snapshot,
    stable_digest,
)
from robocasa.recovery.run_phase2_snapshot_replay import (  # noqa: E402
    build_parser,
    build_plan,
    run,
)
from robocasa.recovery.recovery_rollout import call_policy  # noqa: E402
from robocasa.recovery.subtask_eval import build_subtask_trace  # noqa: E402


def subtask_eval():
    return {
        "task_success": False,
        "required_predicates": ["target_stage"],
        "predicates": {"target_stage": {"value": False, "stage": "task"}},
    }


class FakeSimData:
    def __init__(self):
        self.time = 0.0
        self.qpos = np.zeros(1, dtype=np.float64)
        self.qvel = np.zeros(1, dtype=np.float64)
        self.act = np.zeros(1, dtype=np.float64)
        self.mocap_pos = np.zeros((1, 3), dtype=np.float64)
        self.mocap_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.userdata = np.zeros(1, dtype=np.float64)
        self.ctrl = np.zeros(1, dtype=np.float64)
        self.qfrc_applied = np.zeros(1, dtype=np.float64)
        self.xfrc_applied = np.zeros((1, 6), dtype=np.float64)
        self.qacc_warmstart = np.zeros(1, dtype=np.float64)


class FakeSim:
    def __init__(self):
        self.data = FakeSimData()

    def get_state(self):
        return SimpleNamespace(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
        )

    def forward(self):
        return None


class FakeInterpolator:
    def __init__(self):
        self.step = 3
        self.start = np.asarray([0.125], dtype=np.float64)
        self.goal = np.asarray([0.25], dtype=np.float64)


class FakePartController:
    def __init__(self):
        self.goal = np.asarray([0.25], dtype=np.float64)
        self.interpolator = FakeInterpolator()


class FakeCompositeController:
    def __init__(self):
        self.part_controllers = {"arm": FakePartController()}


class FakeGripper:
    def __init__(self):
        self.current_action = np.asarray([0.2], dtype=np.float64)
        self.speed = 0.004

    def format_action(self, action):
        direction = np.sign(float(np.asarray(action)[0]))
        self.current_action[:] = np.clip(
            self.current_action + self.speed * direction,
            -1.0,
            1.0,
        )
        return self.current_action.copy()


class FakeRobot:
    def __init__(self):
        self.composite_controller = FakeCompositeController()
        self.gripper = {"arm": FakeGripper()}
        self.recent_torques = np.asarray([0.0], dtype=np.float64)


class FakeEnvironment:
    def __init__(self):
        self.value = 0.0
        self.timestep = 0
        self._elapsed_steps = 0
        self.np_random = np.random.default_rng(123)
        self.sim = FakeSim()
        self.robots = [FakeRobot()]

    def get_state(self):
        return {"states": np.asarray([self.value], dtype=np.float64)}

    def reset(self, seed=None):
        self.value = 0.0
        self.timestep = 0
        self._elapsed_steps = 0
        self.np_random = np.random.default_rng(seed)
        self.sim = FakeSim()
        self.robots = [FakeRobot()]
        return self.get_current_observation(), {}

    def reset_to(self, state):
        self.value = float(np.asarray(state["states"])[0])
        # A realistic reset path mutates episode counters; the complete
        # snapshot layer must overwrite them after restoring MuJoCo state.
        self.timestep = 0
        self._elapsed_steps = 0
        # reset_to() intentionally loses the solver history and applied inputs
        # that a complete Phase 2 snapshot must restore separately.
        self.sim.data.qpos[:] = self.value
        self.sim.data.qvel[:] = 99.0
        self.sim.data.qacc_warmstart[:] = -99.0
        self.sim.data.ctrl[:] = -99.0

    def get_current_observation(self):
        return {"value": np.asarray([self.value], dtype=np.float64)}

    def get_subtask_progress(self):
        return subtask_eval()

    def step(self, action):
        controller = self.robots[0].composite_controller.part_controllers["arm"]
        interpolator = controller.interpolator
        command = float(np.asarray(action)[0]) + 0.01 * interpolator.step
        controller.goal[:] = command
        interpolator.start[:] = self.sim.data.ctrl
        interpolator.goal[:] = command
        interpolator.step += 1
        gripper_command = float(
            self.robots[0].gripper["arm"].format_action([command])[0]
        )
        command += gripper_command
        acceleration = command + 0.1 * float(self.sim.data.qacc_warmstart[0])
        self.sim.data.ctrl[:] = command
        self.sim.data.qvel[:] += acceleration
        self.sim.data.qpos[:] += self.sim.data.qvel
        self.sim.data.qacc_warmstart[:] = acceleration
        self.sim.data.time += 0.1
        self.value = float(self.sim.data.qpos[0])
        self.timestep += 1
        self._elapsed_steps += 1
        return self.get_current_observation(), 0.0, False, {"success": False}

    def close(self):
        return None


class FakePolicy:
    def __init__(self):
        self.counter = 0
        self.next_seed = None
        self.pending_requests = []
        self.pending_inference = None

    @property
    def at_inference_boundary(self):
        return True

    @property
    def next_ordinary_sampling_seed(self):
        return 1000 + self.counter

    def set_next_sampling_seed(self, seed):
        self.next_seed = int(seed)

    def __call__(self, observation, instruction=None):
        seed = 1000 + self.counter if self.next_seed is None else self.next_seed
        self.next_seed = None
        action = np.asarray(
            [np.random.default_rng(seed).uniform(-0.25, 0.25)], dtype=np.float64
        )
        request = {
            "sampling_seed": seed,
            "observation": observation,
            "instruction": instruction,
        }
        request_sha256 = stable_digest(request)
        self.pending_requests.append(
            {"sampling_seed": seed, "request_sha256": request_sha256}
        )
        self.pending_inference = {
            "sampling_seed": seed,
            "request_sha256": request_sha256,
            "actions": action.copy(),
        }
        self.counter += 1
        return action

    def pop_request_records(self):
        result = list(self.pending_requests)
        self.pending_requests = []
        return result

    def pop_inference_record(self):
        result = self.pending_inference
        self.pending_inference = None
        return result

    def get_state(self):
        return {
            "schema_version": 1,
            "sampling_config": {"sampling_seed_base": 1000},
            "counter": self.counter,
            "next_seed": self.next_seed,
            "pending_requests": list(self.pending_requests),
            "pending_inference": self.pending_inference,
            "at_inference_boundary": True,
        }

    def set_state(self, state):
        self.counter = int(state["counter"])
        self.next_seed = state["next_seed"]
        self.pending_requests = list(state["pending_requests"])
        self.pending_inference = state["pending_inference"]

    def reset(self):
        self.counter = 0
        self.next_seed = None
        self.pending_requests = []
        self.pending_inference = None

    def close(self):
        return None


class TestPhase2SnapshotReplay(unittest.TestCase):
    def test_stable_digest_uses_raw_bfloat16_storage_without_numpy_conversion(self):
        class FakeView:
            def __init__(self, bits):
                self.bits = bits

            def numpy(self):
                return np.asarray(self.bits, dtype=np.uint16)

        class FakeBFloat16Tensor:
            def __init__(self, bits):
                self.bits = list(bits)
                self.dtype = "torch.bfloat16"
                self.shape = (len(self.bits),)

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                raise TypeError("Got unsupported ScalarType BFloat16")

            def view(self, dtype):
                self.view_dtype = dtype
                return FakeView(self.bits)

        FakeBFloat16Tensor.__module__ = "torch"
        fake_torch = SimpleNamespace(uint16=object())

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            first = FakeBFloat16Tensor([0x3F80, 0xC020])
            same = FakeBFloat16Tensor([0x3F80, 0xC020])
            changed = FakeBFloat16Tensor([0x3F80, 0xC000])
            self.assertEqual(stable_digest(first), stable_digest(same))
            self.assertNotEqual(stable_digest(first), stable_digest(changed))
            self.assertIs(first.view_dtype, fake_torch.uint16)

    def test_stable_digest_hashes_bfloat16_tensor_exactly(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")

        value = torch.tensor([1.0, -2.5, 3.25], dtype=torch.bfloat16)
        same = value.clone()
        changed = value.clone()
        changed[1] = -2.0

        self.assertEqual(stable_digest(value), stable_digest(same))
        self.assertNotEqual(stable_digest(value), stable_digest(changed))
        self.assertNotEqual(
            stable_digest(value),
            stable_digest(value.to(dtype=torch.float32)),
        )

    def make_snapshot(self):
        env = FakeEnvironment()
        env.value = 1.25
        env.sim.data.qpos[:] = env.value
        env.sim.data.qvel[:] = 0.125
        env.sim.data.qacc_warmstart[:] = 0.75
        env.sim.data.ctrl[:] = -0.25
        interpolator = (
            env.robots[0].composite_controller.part_controllers["arm"].interpolator
        )
        interpolator.step = 7
        interpolator.start[:] = -0.25
        interpolator.goal[:] = 0.5
        env.robots[0].gripper["arm"].current_action[:] = 0.35
        env.timestep = 17
        env._elapsed_steps = 17
        policy = FakePolicy()
        snapshot = capture_full_snapshot(
            env,
            policy,
            parent_id="parent-0",
            task_name="ArrangeTea",
            trigger_name="stage_prefix_2",
            environment_step=17,
            observation=env.get_current_observation(),
            subtask_eval=subtask_eval(),
        )
        return env, policy, snapshot

    def test_snapshot_round_trip_restores_sim_policy_rng_and_counters(self):
        env, policy, snapshot = self.make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.pkl.gz"
            save_full_snapshot(snapshot, path)
            loaded = load_full_snapshot(path)

        env.value = 99.0
        env.timestep = 99
        env._elapsed_steps = 99
        env.np_random.random()
        policy.counter = 99
        random.random()
        np.random.random()
        audit = restore_full_snapshot(loaded, env, policy)

        self.assertTrue(audit["valid"], audit)
        self.assertEqual(env.timestep, 17)
        self.assertEqual(env._elapsed_steps, 17)
        self.assertEqual(policy.counter, 0)
        self.assertTrue(audit["simulator_integration_exact"], audit)
        np.testing.assert_array_equal(env.sim.data.qvel, [0.125])
        np.testing.assert_array_equal(env.sim.data.qacc_warmstart, [0.75])
        np.testing.assert_array_equal(env.sim.data.ctrl, [-0.25])
        restored_interpolator = (
            env.robots[0].composite_controller.part_controllers["arm"].interpolator
        )
        self.assertEqual(restored_interpolator.step, 7)
        np.testing.assert_array_equal(restored_interpolator.start, [-0.25])
        np.testing.assert_array_equal(restored_interpolator.goal, [0.5])
        np.testing.assert_array_equal(
            env.robots[0].gripper["arm"].current_action,
            [0.35],
        )

    def test_same_seed_replay_and_candidate_diversity_pass_engineering_gates(self):
        env, policy, snapshot = self.make_snapshot()
        records = []
        for repeat in range(2):
            result = run_counterfactual_branch(
                snapshot,
                env,
                policy,
                BranchSpec(
                    branch_id=f"repeat-{repeat}",
                    kind="same_seed_repeat",
                    sampling_seed=55,
                    suffix_steps=3,
                    repeat_index=repeat,
                ),
            )
            records.append(result["summary"])
        for index, seed in enumerate((100, 101, 102, 103)):
            result = run_counterfactual_branch(
                snapshot,
                env,
                policy,
                BranchSpec(
                    branch_id=f"candidate-{index}",
                    kind="candidate",
                    sampling_seed=seed,
                    suffix_steps=3,
                ),
            )
            records.append(result["summary"])

        analysis = analyze_phase2_replay(records)
        self.assertTrue(analysis["all_pass"], analysis)
        self.assertEqual(analysis["same_seed_suffix_outcome_agreement"], 1.0)
        self.assertEqual(analysis["candidate_diversity_rate"], 1.0)
        for pair in analysis["repeat_pairs"]:
            self.assertEqual(
                pair["transition_components_exact"],
                {
                    "control_state": True,
                    "controller_state": True,
                    "rng_state": True,
                    "simulator_state": True,
                },
            )

    def test_dry_run_plan_uses_frozen_training_horizons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime.json"
            model = root / "checkpoint"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            runtime.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "primary_detector": "stage",
                        "horizons": {
                            "stage": {
                                "ArrangeTea::PickPlaceCabinetToCounter_2_place": 16,
                                "CuttingToolSelection::PickPlaceDrawerToCounter_2_place": 20,
                            }
                        },
                    }
                )
            )
            args = build_parser().parse_args(
                [
                    "--output-dir",
                    str(root / "out"),
                    "--runtime-bundle",
                    str(runtime),
                    "--tasks",
                    "ArrangeTea",
                    "CuttingToolSelection",
                    "--target-stage",
                    "ArrangeTea=PickPlaceCabinetToCounter_2_place",
                    "--target-stage",
                    "CuttingToolSelection=PickPlaceDrawerToCounter_2_place",
                    "--trigger-stage",
                    "ArrangeTea=mug_on_tray",
                    "--trigger-stage",
                    "CuttingToolSelection=correct_tool_on_cutting_board",
                    "--model-path",
                    str(model),
                    "--checkpoint",
                    "Xiaomi/checkpoint",
                    "--dry-run",
                ]
            )
            plan = build_plan(args)

        self.assertEqual(plan["expected_parents"], 10)
        self.assertEqual(plan["expected_snapshots"], 20)
        self.assertEqual(plan["expected_primary_branches"], 120)
        self.assertEqual(plan["expected_total_branches"], 140)
        self.assertEqual(plan["training_stage_horizons"]["CuttingToolSelection"], 20)
        self.assertEqual(
            plan["trigger_stages"]["ArrangeTea"], "ArrangeTea::mug_on_tray"
        )

    def test_runner_executes_complete_fake_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_bundle = root / "runtime.json"
            model = root / "checkpoint"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            runtime_bundle.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "primary_detector": "stage",
                        "horizons": {"stage": {"ArrangeTea::frozen_segment": 8}},
                    }
                )
            )
            args = build_parser().parse_args(
                [
                    "--output-dir",
                    str(root / "out"),
                    "--runtime-bundle",
                    str(runtime_bundle),
                    "--tasks",
                    "ArrangeTea",
                    "--target-stage",
                    "ArrangeTea=frozen_segment",
                    "--trigger-stage",
                    "ArrangeTea=target_stage",
                    "--num-parents-per-task",
                    "1",
                    "--candidate-count",
                    "2",
                    "--suffix-steps",
                    "2",
                    "--parent-horizon",
                    "20",
                    "--model-path",
                    str(model),
                    "--checkpoint",
                    "Xiaomi/checkpoint",
                ]
            )
            fake_runtime = {
                "load_factory": lambda value: object(),
                "parse_policy_args": lambda values: {},
                "call_factory": lambda factory, env, policy_args: FakePolicy(),
                "make_env": lambda *unused: FakeEnvironment(),
                "success_fn": lambda **unused: False,
                "step_fn": lambda env, action: env.step(action),
                "call_policy": call_policy,
                "build_subtask_trace": build_subtask_trace,
                "get_subtask_eval": lambda env: env.get_subtask_progress(),
            }
            with redirect_stdout(io.StringIO()):
                analysis = run(args, runtime=fake_runtime)

            self.assertTrue(analysis["all_pass"], analysis)
            self.assertEqual(analysis["completed_parents"], 1)
            self.assertEqual(analysis["completed_snapshots"], 2)
            self.assertEqual(analysis["saved_snapshot_files"], 2)
            self.assertEqual(analysis["orphan_snapshot_files"], 0)
            self.assertEqual(analysis["primary_records"], 8)
            self.assertEqual(analysis["records"], 10)
            parent_record = json.loads(
                (root / "out" / "parent_records.jsonl").read_text().splitlines()[0]
            )
            self.assertTrue(parent_record["target_stage_reached"])
            self.assertEqual(
                parent_record["target_stage"], "ArrangeTea::frozen_segment"
            )
            self.assertEqual(parent_record["trigger_stage"], "ArrangeTea::target_stage")
            self.assertEqual(parent_record["observed_stage_sequence"], ["target_stage"])
            self.assertGreaterEqual(
                parent_record["target_stage_max_consecutive_policy_inferences"],
                3,
            )
            self.assertEqual(
                parent_record["stage_diagnostics"]["target_stage"]["visits"], 1
            )
            self.assertEqual(
                parent_record["termination_reason"], "snapshot_pair_captured"
            )
            self.assertEqual(
                json.loads((root / "out" / "status.json").read_text())["status"],
                "complete",
            )

            args.resume = True
            with redirect_stdout(io.StringIO()):
                resumed = run(args, runtime=fake_runtime)
            self.assertTrue(resumed["all_pass"], resumed)
            self.assertEqual(resumed["records"], 10)
            self.assertEqual(
                len((root / "out" / "parent_records.jsonl").read_text().splitlines()),
                1,
            )

            # Simulate an allocation ending after snapshots and all but one
            # branch were durably written, but before the parent commit.
            branch_path = root / "out" / "branch_records.jsonl"
            branch_lines = branch_path.read_text().splitlines()
            branch_path.write_text("\n".join(branch_lines[:-1]) + "\n")
            (root / "out" / "parent_records.jsonl").write_text("")
            with redirect_stdout(io.StringIO()):
                recovered = run(args, runtime=fake_runtime)
            self.assertTrue(recovered["all_pass"], recovered)
            self.assertEqual(recovered["records"], 10)
            self.assertEqual(recovered["completed_parents"], 1)


if __name__ == "__main__":
    unittest.main()
