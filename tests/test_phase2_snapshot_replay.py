import json
import io
import random
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

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


class FakeEnvironment:
    def __init__(self):
        self.value = 0.0
        self.timestep = 0
        self._elapsed_steps = 0
        self.np_random = np.random.default_rng(123)

    def get_state(self):
        return {"states": np.asarray([self.value], dtype=np.float64)}

    def reset(self, seed=None):
        self.value = 0.0
        self.timestep = 0
        self._elapsed_steps = 0
        self.np_random = np.random.default_rng(seed)
        return self.get_current_observation(), {}

    def reset_to(self, state):
        self.value = float(np.asarray(state["states"])[0])
        # A realistic reset path mutates episode counters; the complete
        # snapshot layer must overwrite them after restoring MuJoCo state.
        self.timestep = 0
        self._elapsed_steps = 0

    def get_current_observation(self):
        return {"value": np.asarray([self.value], dtype=np.float64)}

    def get_subtask_progress(self):
        return subtask_eval()

    def step(self, action):
        self.value += float(np.asarray(action)[0])
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
    def make_snapshot(self):
        env = FakeEnvironment()
        env.value = 1.25
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
                        "horizons": {"stage": {"ArrangeTea::target_stage": 8}},
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
            self.assertEqual(analysis["primary_records"], 8)
            self.assertEqual(analysis["records"], 10)
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
