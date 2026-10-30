"""Simulator-free tests for the stage-gated re-prompt trial."""

from collections import deque
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()
from robocasa.recovery import recoverability_ladder as ladder
from robocasa.recovery import stage_gated_reprompt as gate


REPO = Path(__file__).resolve().parents[1]
TASK_INSTRUCTION = "Do a, then b, then c."
STAGE_INSTRUCTIONS = {"a": "Do a.", "b": "Do b.", "c": "Do c."}
INSTRUCTION_IDS = {TASK_INSTRUCTION: 0, "Do a.": 1, "Do b.": 2, "Do c.": 3}
HORIZON = 40


class State:
    def __init__(self, time, qpos, qvel):
        self.time = float(time)
        self.qpos = np.array(qpos, dtype=float)
        self.qvel = np.array(qvel, dtype=float)

    def flatten(self):
        return np.concatenate([[self.time], self.qpos, self.qvel])


class Sim:
    def __init__(self):
        self.time = 0.0
        self.qpos = np.zeros(2)
        self.qvel = np.zeros(2)

    def get_state(self):
        return State(self.time, self.qpos.copy(), self.qvel.copy())


class GateEnv:
    """Three ordered stages a, b, c at progress 1, 2, 3; success at 3.

    The policy writes the id of its instruction into ``action[1]``.
    seed % 3 == 0: the task instruction suffices, but the stage-a instruction
    breaks the scene (ungated harm); 1: progress stalls in stage b unless the
    stage-b instruction is given (instruction-recoverable stall); 2: the
    stage-b instruction breaks the scene (gated harm). A broken scene
    (``qpos[1]``) never progresses again.
    """

    def __init__(self, seed=0):
        self.sim = Sim()
        self.mode = 0
        self.override = None
        self.reset_seeds = []

    def reset(self, seed=None):
        self.sim = Sim()
        self.mode = (seed or 0) % 3
        self.override = None
        self.reset_seeds.append(seed)
        # Like a scene layout drawn from the seed.
        self.sim.qvel[:] = [(seed or 0) % 7, 0.0]
        return self.observation(), {"subtask_eval": self.get_subtask_progress()}

    def observation(self):
        return {"annotation.human.task_description": self.override or TASK_INSTRUCTION, "state": self.sim.qpos.copy()}

    def set_task_description(self, text):
        self.override = text

    def clear_task_description(self):
        self.override = None

    def get_ep_meta(self):
        return {"lang": TASK_INSTRUCTION}

    def get_subtask_progress(self):
        progress = self.sim.qpos[0]
        predicates = {
            name: {"value": bool(progress >= level), "required": True, "description": STAGE_INSTRUCTIONS[name]}
            for name, level in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        }
        predicates["task_success"] = {"value": bool(progress >= 3.0), "required": False}
        return {
            "task_name": "GateTask",
            "predicates": predicates,
            "required_predicates": ["a", "b", "c"],
            "task_success": bool(progress >= 3.0),
        }

    def step(self, action):
        qpos = self.sim.qpos
        prompt = int(action[1])
        if self.mode == 0 and qpos[0] < 1.0 and prompt == INSTRUCTION_IDS["Do a."]:
            qpos[1] = 1.0
        stage_b = 1.0 <= qpos[0] < 2.0
        if self.mode == 2 and stage_b and prompt == INSTRUCTION_IDS["Do b."]:
            qpos[1] = 1.0
        blocked = qpos[1] > 0 or (self.mode == 1 and stage_b and prompt != INSTRUCTION_IDS["Do b."])
        if not blocked:
            qpos[0] = min(3.0, qpos[0] + action[0])
        self.sim.time += 0.05
        progress = self.get_subtask_progress()
        success = progress["task_success"]
        return self.observation(), float(success), False, False, {"success": success, "subtask_eval": progress}

    def close(self):
        pass


class XR1LikePolicy:
    """Chunked policy with XR-1's sampling_seed_base contract.

    Like ``XiaomiRobotics1Policy.__call__``, an instruction change resets the
    action plan and the inference index.
    """

    def __init__(self, replan_steps=4):
        self.replan_steps = replan_steps
        self.sampling_seed_base = None
        self.log = []
        self.reset()

    def reset(self):
        self.action_plan = deque()
        self.inference_index = 0
        self.last_instruction = None

    def __call__(self, obs, instruction=None):
        instruction = instruction or obs["annotation.human.task_description"]
        if self.last_instruction is not None and instruction != self.last_instruction:
            self.reset()
        self.last_instruction = instruction
        if not self.action_plan:
            seed = None if self.sampling_seed_base is None else self.sampling_seed_base + self.inference_index
            self.log.append(("inference", seed, instruction))
            self.inference_index += 1
            for _ in range(self.replan_steps):
                self.action_plan.append(np.array([0.25, INSTRUCTION_IDS[instruction]], dtype=float))
        self.log.append(("step", instruction, obs["annotation.human.task_description"]))
        return self.action_plan.popleft()

    def inference_seeds(self):
        return [entry[1] for entry in self.log if entry[0] == "inference"]


class GateRuntime:
    def __init__(self, horizon=HORIZON):
        self.horizon = horizon
        self.envs = []
        self.policies = []

    def task_horizon(self, task_name):
        return self.horizon

    def make_env(self, task_name, seed):
        env = GateEnv(seed)
        self.envs.append(env)
        return env

    def make_policy(self, env):
        policy = XR1LikePolicy()
        self.policies.append(policy)
        return policy


def gate_config(**overrides):
    config = {
        "tasks": ["GateTask", "ControlTask"],
        "arms": ["baseline", "gated_reprompt", "always_reprompt"],
        "reprompt_whitelist": {"GateTask": ["b"]},
        "seeds_per_task": 6,
        "base_seed": 0,
        "seed_block_size": 4,
        "reprompt_step_cap": 8,
        # GateTask is not a registered composite task; "subtask" gives the predicate description.
        "reprompt_level": "subtask",
        "state_hash_stride": 4,
        "policy_module": "tests.mock:make_policy",
        "expected_success_rates": {"GateTask": 0.5, "ControlTask": 0.5},
        "horizons": {"GateTask": HORIZON, "ControlTask": HORIZON},
    }
    config.update(overrides)
    return config


def collect(tmp, **overrides):
    runtime = GateRuntime()
    status = gate.run_collection(gate_config(**overrides), tmp, runtime=runtime, log=lambda *_: None)
    return runtime, status


def rollouts_by_key(root):
    return {(r["task_name"], r["environment_seed"], r["arm"]): r for r in ladder.read_jsonl(Path(root) / gate.ROLLOUTS_NAME)}


class Tracker:
    def __init__(self, required, count=0):
        self.inner = ladder.OrderedProgress(required, count=count)

    def advance(self, values):
        evaluation = {
            "predicates": {name: {"value": value, "description": STAGE_INSTRUCTIONS[name]} for name, value in values.items()},
            "required_predicates": list(self.inner.required),
            "task_success": False,
        }
        self.inner.update(evaluation)
        return evaluation


class StageGateTest(unittest.TestCase):
    def make(self, arm, whitelist=("b",), cap=5):
        return gate.StageGate(
            arm=arm, task_name="GateTask", task_instruction=TASK_INSTRUCTION, whitelist=whitelist,
            step_cap=cap, reprompt_level="subtask",
        )

    def drive(self, controller, timeline):
        """timeline: [(step, {a, b, c values})]; returns the instruction after each update."""
        tracker = Tracker(["a", "b", "c"])
        out = []
        for step, values in timeline:
            evaluation = tracker.advance(values)
            out.append(controller.update(step, tracker.inner, evaluation))
        return out

    def test_gated_switches_at_whitelisted_stage_entry_and_returns_on_completion(self):
        controller = self.make("gated_reprompt")
        out = self.drive(
            controller,
            [
                (0, {"a": False, "b": False, "c": False}),
                (3, {"a": True, "b": False, "c": False}),
                (4, {"a": True, "b": False, "c": False}),
                (6, {"a": True, "b": True, "c": False}),
                (7, {"a": True, "b": True, "c": False}),
            ],
        )
        self.assertEqual(out, [TASK_INSTRUCTION, "Do b.", "Do b.", TASK_INSTRUCTION, TASK_INSTRUCTION])
        self.assertEqual([e["event"] for e in controller.events], ["switch", "return"])
        record = controller.records[0]
        self.assertEqual((record["stage"], record["switch_step"], record["return_step"]), ("b", 3, 6))
        self.assertEqual((record["return_reason"], record["steps_active"], record["target_completed"]), ("target_complete", 3, True))

    def test_step_cap_returns_and_the_stage_never_fires_again(self):
        controller = self.make("gated_reprompt", cap=5)
        timeline = [(0, {"a": True, "b": False, "c": False})] + [(s, {"a": True, "b": False, "c": False}) for s in range(1, 12)]
        out = self.drive(controller, timeline)
        self.assertEqual(out[:5], ["Do b."] * 5)
        self.assertTrue(all(instruction == TASK_INSTRUCTION for instruction in out[5:]))
        self.assertEqual([e["event"] for e in controller.events], ["switch", "return"])
        self.assertEqual(controller.records[0]["return_reason"], "step_cap")
        self.assertEqual(controller.records[0]["steps_active"], 5)
        self.assertEqual(controller.fired, ["b"])

    def test_baseline_never_switches_and_non_whitelisted_stages_do_not_fire(self):
        timeline = [
            (0, {"a": False, "b": False, "c": False}),
            (3, {"a": True, "b": False, "c": False}),
            (6, {"a": True, "b": True, "c": False}),
        ]
        for controller in (self.make("baseline"), self.make("gated_reprompt", whitelist=())):
            self.assertEqual(self.drive(controller, timeline), [TASK_INSTRUCTION] * 3)
            self.assertEqual(controller.events, [])
        with self.assertRaises(ValueError):
            gate.StageGate(arm="sometimes", task_name="T", task_instruction="x")

    def test_always_chains_stages_without_returning_in_between(self):
        controller = self.make("always_reprompt")
        out = self.drive(
            controller,
            [
                (0, {"a": False, "b": False, "c": False}),
                (2, {"a": True, "b": False, "c": False}),
                (5, {"a": True, "b": True, "c": False}),
            ],
        )
        self.assertEqual(out, ["Do a.", "Do b.", "Do c."])
        self.assertEqual(
            [(e["event"], e["stage"]) for e in controller.events],
            [("switch", "a"), ("return", "a"), ("switch", "b"), ("return", "b"), ("switch", "c")],
        )
        controller.finish(9, "horizon")
        self.assertEqual(controller.records[-1]["return_reason"], "horizon")
        self.assertEqual(controller.records[-1]["steps_active"], 4)

    def test_several_stages_latching_at_once_skip_the_intermediate_ones(self):
        controller = self.make("always_reprompt")
        out = self.drive(controller, [(0, {"a": False, "b": False, "c": False}), (4, {"a": True, "b": True, "c": False})])
        self.assertEqual(out, ["Do a.", "Do c."])
        self.assertEqual(controller.fired, ["a", "c"])

    def test_target_equal_to_task_prompt_is_skipped(self):
        # Atomic level on a non-composite task resolves to the task instruction.
        controller = gate.StageGate(
            arm="gated_reprompt", task_name="GateTask", task_instruction=TASK_INSTRUCTION, whitelist=["b"],
            step_cap=5, reprompt_level="atomic",
        )
        out = self.drive(controller, [(0, {"a": True, "b": False, "c": False}), (1, {"a": True, "b": False, "c": False})])
        self.assertEqual(out, [TASK_INSTRUCTION, TASK_INSTRUCTION])
        self.assertEqual(controller.events[0]["event"], "skip")
        self.assertFalse(controller.records[0]["switched"])

    def test_composite_atomic_instruction_for_the_whitelisted_stages(self):
        for task, stage, description in (
            ("ArrangeTea", "mug_on_tray", "Move mug to tray."),
            ("BreadSelection", "jam_on_cutting_board", "Move jam to cutting board."),
        ):
            evaluation = {
                "task_name": task,
                "predicates": {stage: {"value": False, "description": description}},
                "required_predicates": [stage],
            }
            target, instruction, kind = gate.resolve_stage_instruction(
                evaluation, stage, task_instruction="whole task", task_name=task, reprompt_level="atomic"
            )
            self.assertEqual((target, instruction, kind), (stage, description, "atomic_predicate"))


class PairingTest(unittest.TestCase):
    def test_arms_share_env_seed_initial_state_and_policy_seed_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, status = collect(tmp)
            self.assertEqual(status["num_pairs"], 12)
            self.assertEqual(status["num_errors"], 0)
            pairs = ladder.read_jsonl(Path(tmp) / gate.PAIRS_NAME)
            rows = ladder.read_jsonl(Path(tmp) / gate.ROLLOUTS_NAME)
            self.assertEqual(len(rows), 36)
            for pair in pairs:
                self.assertTrue(pair["initial_state_match"])
                self.assertEqual(sorted(pair["arm_order"]), sorted(gate_config()["arms"]))
                arm_rows = [r for r in rows if r["pair_uid"] == pair["pair_uid"]]
                self.assertEqual({r["environment_seed"] for r in arm_rows}, {pair["environment_seed"]})
                self.assertEqual({r["policy_seed"] for r in arm_rows}, {pair["policy_seed"]})
                self.assertEqual({r["seed_control"] for r in arm_rows}, {"policy_sampling_seed_base"})
            # Every reset of the shared env used the rollout's seed.
            for env in runtime.envs:
                self.assertTrue(set(env.reset_seeds) <= set(range(6)))
            # Counterbalanced order: not every seed runs baseline first.
            self.assertGreater(len({tuple(pair["arm_order"]) for pair in pairs}), 1)

    def test_nth_inference_uses_the_same_sampling_seed_in_every_arm(self):
        config = gate.resolve_config(gate_config())
        for arm in config["arms"]:
            env, policy = GateEnv(), XR1LikePolicy()
            row = gate.run_rollout(env, policy, task_name="GateTask", arm=arm, env_seed=1, horizon=HORIZON, config=config)
            base = gate.rollout_policy_seed("GateTask", 1)
            self.assertEqual(policy.inference_seeds(), [base + i for i in range(len(policy.inference_seeds()))], arm)
            for switch in row["instruction_switches"]:
                self.assertEqual(switch["sampling_seed_base"], base + switch["inference_index"])

    def test_arms_are_identical_before_the_first_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            collect(tmp)
            pairs = ladder.read_jsonl(Path(tmp) / gate.PAIRS_NAME)
            for pair in pairs:
                for arm, divergence in pair["first_divergence_step"].items():
                    switch = pair["first_switch_step"][arm]
                    if switch is None:
                        self.assertIsNone(divergence, (pair["task_name"], pair["environment_seed"], arm))
                    elif divergence is not None:
                        # The state after the switch step is the first that can differ.
                        self.assertGreater(divergence, switch)

    def test_seed_shards_partition_the_seeds_and_keep_arms_together(self):
        seeds = list(range(400007, 400057))
        shards = [gate.shard_seeds(seeds, 2, index) for index in range(2)]
        self.assertEqual(sorted(shards[0] + shards[1]), seeds)
        self.assertEqual(len(shards[0]), 25)
        self.assertEqual([len(block) for block in gate.seed_blocks(shards[0], 10)], [10, 10, 5])
        self.assertEqual(gate.arm_order("T", 5, ["baseline", "gated_reprompt"], "fixed"), ["baseline", "gated_reprompt"])
        self.assertEqual(gate.arm_order("T", 5, gate.ARMS), gate.arm_order("T", 5, gate.ARMS))
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(2):
                gate.run_collection(gate_config(), Path(tmp) / f"shard{index}", runtime=GateRuntime(), shard_index=index, num_shards=2, log=lambda *_: None)
            keys = [(p["task_name"], p["environment_seed"]) for index in range(2) for p in ladder.read_jsonl(Path(tmp) / f"shard{index}" / gate.PAIRS_NAME)]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(keys), 12)


class SwitchBehaviourTest(unittest.TestCase):
    def test_gated_switch_and_return_in_a_rollout(self):
        config = gate.resolve_config(gate_config())
        env, policy = GateEnv(), XR1LikePolicy()
        # seed 1: instruction-recoverable stall at stage b.
        row = gate.run_rollout(env, policy, task_name="GateTask", arm="gated_reprompt", env_seed=1, horizon=HORIZON, config=config)
        self.assertTrue(row["success"])
        self.assertEqual(row["n_switches"], 1)
        record = row["gate_stages"][0]
        self.assertEqual((record["stage"], record["instruction"], record["return_reason"]), ("b", "Do b.", "target_complete"))
        self.assertEqual(record["switch_step"], row["stage_latch_steps"]["a"])
        self.assertEqual(record["return_step"], row["stage_latch_steps"]["b"])
        # Policy instructions: task, then stage b from the switch step, then task again.
        steps = [entry for entry in policy.log if entry[0] == "step"]
        instructions = [entry[1] for entry in steps]
        switch, back = record["switch_step"], record["return_step"]
        self.assertEqual(set(instructions[:switch]), {TASK_INSTRUCTION})
        self.assertEqual(set(instructions[switch:back]), {"Do b."})
        self.assertEqual(set(instructions[back:]), {TASK_INSTRUCTION})
        # The observation's language field follows the instruction (env override + observation copy).
        self.assertTrue(all(entry[1] == entry[2] for entry in steps))
        self.assertIsNone(env.override)
        self.assertEqual(row["policy_resets"], 2)
        # The switch takes effect immediately: an inference at the switch step and at the return step.
        self.assertEqual([s["step"] for s in row["instruction_switches"]], [switch, back])
        self.assertEqual(row["predicate_trace"]["a"][0][0], row["stage_latch_steps"]["a"])
        self.assertEqual(row["cap_hits"], 0)

    def test_harmful_switch_hits_the_cap_and_returns(self):
        config = gate.resolve_config(gate_config())
        row = gate.run_rollout(GateEnv(), XR1LikePolicy(), task_name="GateTask", arm="gated_reprompt", env_seed=2, horizon=HORIZON, config=config)
        self.assertFalse(row["success"])
        record = row["gate_stages"][0]
        self.assertEqual((record["return_reason"], record["steps_active"], record["target_completed"]), ("step_cap", 8, False))
        self.assertEqual(row["cap_hits"], 1)
        self.assertEqual(row["end_reason"], "horizon")

    def test_unknown_whitelisted_stage_stops_the_rollout(self):
        config = gate.resolve_config(gate_config(reprompt_whitelist={"GateTask": ["z"]}))
        with self.assertRaisesRegex(ValueError, "not in the ordered list"):
            gate.run_rollout(GateEnv(), XR1LikePolicy(), task_name="GateTask", arm="gated_reprompt", env_seed=1, horizon=HORIZON, config=config)

    def test_excluded_whitelisted_stage_is_rejected_like_an_unknown_one(self):
        config = gate.resolve_config(gate_config(exclude_predicates={"GateTask": ["b"]}))
        with self.assertRaisesRegex(ValueError, "not in the ordered list"):
            gate.run_rollout(GateEnv(), XR1LikePolicy(), task_name="GateTask", arm="gated_reprompt", env_seed=1, horizon=HORIZON, config=config)


class ControlTaskTest(unittest.TestCase):
    def test_non_whitelisted_task_arms_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            collect(tmp)
            rows = rollouts_by_key(tmp)
            for seed in range(6):
                base = rows[("ControlTask", seed, "baseline")]
                gated = rows[("ControlTask", seed, "gated_reprompt")]
                self.assertEqual(gated["n_switches"], 0)
                self.assertEqual(gated["gate_events"], [])
                for key in ("success", "num_steps", "state_hashes", "progress_events", "predicate_trace", "final_ordered_count"):
                    self.assertEqual(base[key], gated[key], key)
                # The ungated arm does switch on the control task.
                self.assertGreater(rows[("ControlTask", seed, "always_reprompt")]["n_switches"], 0)

    def test_expected_outcomes_by_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            collect(tmp)
            rows = rollouts_by_key(tmp)
            expected = {
                0: {"baseline": True, "gated_reprompt": True, "always_reprompt": False},
                1: {"baseline": False, "gated_reprompt": True, "always_reprompt": True},
                2: {"baseline": True, "gated_reprompt": False, "always_reprompt": False},
            }
            for seed in range(6):
                for arm, success in expected[seed % 3].items():
                    self.assertEqual(rows[("GateTask", seed, arm)]["success"], success, (seed, arm))
            # Every GateTask rollout reaches stage b, so the gate fires in all of them.
            self.assertTrue(all(rows[("GateTask", seed, "gated_reprompt")]["fired"] for seed in range(6)))


class ResumeTest(unittest.TestCase):
    def test_resume_skips_complete_seeds_reruns_partial_ones_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            collect(tmp)
            pairs_path = Path(tmp) / gate.PAIRS_NAME
            lines = pairs_path.read_text().splitlines()
            # Drop the last pair row: its rollouts become orphans and the seed reruns.
            dropped = json.loads(lines[-1])
            pairs_path.write_text("\n".join(lines[:-1]) + "\n")
            runtime = GateRuntime()
            status = gate.run_collection(gate_config(), tmp, runtime=runtime, resume=True, log=lambda *_: None)
            self.assertEqual(status["num_pairs"], 12)
            self.assertEqual(sum(len(env.reset_seeds) for env in runtime.envs), 3)
            complete, _, _ = gate.load_run([tmp])
            self.assertEqual(len(complete), 12)
            uids = {pair["pair_uid"] for pair, _ in complete}
            self.assertNotIn(dropped["pair_uid"], uids)
            with self.assertRaises(FileExistsError):
                gate.run_collection(gate_config(), tmp, runtime=GateRuntime(), log=lambda *_: None)
            with self.assertRaisesRegex(ValueError, "differs"):
                gate.run_collection(gate_config(reprompt_step_cap=9), tmp, runtime=GateRuntime(), resume=True, log=lambda *_: None)
            with self.assertRaisesRegex(ValueError, "differs"):
                gate.run_collection(gate_config(), tmp, runtime=GateRuntime(), resume=True, num_shards=2, log=lambda *_: None)

    def test_errors_are_recorded_and_bounded(self):
        class Broken(GateRuntime):
            def make_policy(self, env):
                raise RuntimeError("server down")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "Stopping after 2 errors"):
                gate.run_collection(gate_config(max_errors=2), tmp, runtime=Broken(), log=lambda *_: None)
            self.assertEqual(len(ladder.read_jsonl(Path(tmp) / gate.ERRORS_NAME)), 2)


class StatisticsTest(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(gate.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(gate.mcnemar_exact(7, 0), 2 / 2**7)
        self.assertAlmostEqual(gate.mcnemar_exact(0, 7), 2 / 2**7)
        self.assertAlmostEqual(gate.mcnemar_exact(8, 2), 2 * (1 + 10 + 45) / 2**10)
        self.assertEqual(gate.mcnemar_exact(3, 3), 1.0)

    def test_holm(self):
        self.assertEqual(gate.holm([0.01, 0.04]), [0.02, 0.04])
        self.assertEqual(gate.holm([0.04, 0.01]), [0.04, 0.02])
        self.assertEqual(gate.holm([0.5, 0.6]), [1.0, 1.0])

    def test_power(self):
        # All discordant pairs help: rejection needs >= 6 of them (2 * 0.5**6 < 0.05).
        n, p = 50, 0.15
        from math import comb

        exact = sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(6, n + 1))
        self.assertAlmostEqual(gate.mcnemar_power(n, p, 0.0), exact, places=9)
        self.assertLess(gate.mcnemar_power(50, 0.20, 0.05), gate.mcnemar_power(50, 0.15, 0.0))
        self.assertLess(gate.mcnemar_power(50, 0.15, 0.0), gate.mcnemar_power(100, 0.15, 0.0))
        self.assertLess(gate.mcnemar_power(50, 0.0, 0.0), 0.05 + 1e-9)
        self.assertEqual(gate.minimum_detectable_difference(50), 0.16)

    def test_paired_bootstrap(self):
        out = gate.paired_bootstrap({"A": [1, 1, 0, 0], "B": [0, 0, 0, 0]}, n_boot=500, seed=1)
        self.assertEqual(out["A"][0], 0.5)
        self.assertEqual(out["B"], (0.0, 0.0, 0.0))
        self.assertEqual(out["macro"][0], 0.25)
        self.assertLessEqual(out["macro"][1], 0.25)
        self.assertGreaterEqual(out["macro"][2], 0.25)
        self.assertLessEqual(out["macro"][2], 0.5)


class AnalysisTest(unittest.TestCase):
    def test_analysis_of_mock_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            collect(tmp)
            summary, tables = gate.analyze([tmp], n_boot=300)
            self.assertEqual(summary["n_complete_pairs"], 12)
            self.assertEqual(summary["whitelisted_tasks"], ["GateTask"])
            self.assertEqual(summary["control_tasks"], ["ControlTask"])
            self.assertEqual(summary["initial_state_mismatches"], 0)
            contrasts = {(row["contrast"], row["group"]): row for row in summary["contrasts"]}
            gated = contrasts[("gated_reprompt - baseline", "GateTask")]
            # Seeds 1, 4 rescued; seeds 2, 5 harmed; seeds 0, 3 unchanged.
            self.assertEqual((gated["b_help"], gated["c_harm"], gated["n_pairs"]), (2, 2, 6))
            self.assertEqual(gated["difference"], 0.0)
            self.assertEqual(gated["mcnemar_p"], 1.0)
            self.assertEqual(gated["rescue_rate"], 1.0)
            self.assertEqual(gated["harm_rate_on_b_successes"], 0.5)
            self.assertEqual(gated["n_fired"], 6)
            self.assertIn("mcnemar_p_holm", gated)
            self.assertEqual(summary["primary"]["group"], "macro_whitelisted")
            self.assertEqual(summary["primary"]["tasks"], "GateTask")
            always = contrasts[("always_reprompt - baseline", "GateTask")]
            self.assertAlmostEqual(always["difference"], (2 - 4) / 6)
            control = contrasts[("gated_reprompt - baseline", "ControlTask")]
            self.assertEqual((control["b_help"], control["c_harm"], control["n_fired"]), (0, 0, 0))
            self.assertEqual(contrasts[("gated_reprompt - baseline", "macro_all")]["n_pairs"], 12)
            # On the control task only the always arm switches; every pair still counts as switched.
            versus_always = contrasts[("gated_reprompt - always_reprompt", "ControlTask")]
            self.assertEqual((versus_always["n_fired"], versus_always["discordant_when_not_fired"]), (6, 0))
            checks = summary["control_checks"]
            self.assertEqual(len(checks), 1)
            self.assertEqual((checks[0]["gated_switches"], checks[0]["outcome_concordance"], checks[0]["identical_state_traces"]), (0, 1.0, 1.0))
            switches = {(row["task"], row["arm"], row["stage"]): row for row in summary["switches"]}
            self.assertEqual(switches[("GateTask", "gated_reprompt", "b")]["n_fired"], 6)
            self.assertEqual(switches[("GateTask", "gated_reprompt", "b")]["cap_hits"], 2)
            self.assertNotIn(("ControlTask", "gated_reprompt", "b"), switches)
            stages = {(row["task"], row["arm"], row["stage"]): row for row in summary["stage_completion"]}
            self.assertAlmostEqual(stages[("GateTask", "baseline", "b")]["completed_rate"], 4 / 6)
            self.assertAlmostEqual(stages[("GateTask", "gated_reprompt", "b")]["completed_rate"], 4 / 6)
            arms = {(row["task"], row["arm"]): row for row in summary["arms_by_task"]}
            self.assertEqual(arms[("GateTask", "gated_reprompt")]["fired_rate"], 1.0)
            self.assertEqual(arms[("GateTask", "baseline")]["fired_rate"], 0.0)
            out = Path(tmp) / "analysis"
            gate.write_analysis(summary, tables, out)
            text = (out / "summary.md").read_text()
            self.assertIn("## Primary endpoint", text)
            self.assertIn("Control tasks", text)
            for name in ("rollouts", "contrasts", "arms_by_task", "switches", "stage_completion", "control_checks"):
                self.assertTrue((out / f"{name}.csv").exists())
            self.assertEqual(json.loads((out / "summary.json").read_text())["n_complete_pairs"], 12)

    def test_incomplete_pairs_are_ignored_and_shards_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(2):
                gate.run_collection(gate_config(), Path(tmp) / f"shard{index}", runtime=GateRuntime(), shard_index=index, num_shards=2, log=lambda *_: None)
            orphan = dict(ladder.read_jsonl(Path(tmp) / "shard0" / gate.ROLLOUTS_NAME)[0], pair_uid="orphan")
            ladder.append_jsonl(Path(tmp) / "shard0" / gate.ROLLOUTS_NAME, orphan)
            summary, _ = gate.analyze([Path(tmp) / "shard0", Path(tmp) / "shard1"], n_boot=100)
            self.assertEqual(summary["n_complete_pairs"], 12)
            self.assertEqual(summary["pairs_per_task"], {"GateTask": 6, "ControlTask": 6})


class PlanTest(unittest.TestCase):
    def load(self):
        return json.loads((REPO / "scripts" / "stage_gated_reprompt_xr1.json").read_text())

    def test_default_config_fits_one_allocation_on_two_workers(self):
        plan = gate.build_plan(self.load())
        self.assertEqual(plan["tasks"], ["ArrangeTea", "BreadSelection", "GarnishPancake"])
        self.assertEqual(plan["whitelisted_tasks"], ["ArrangeTea", "BreadSelection"])
        self.assertEqual(plan["control_tasks"], ["GarnishPancake"])
        self.assertEqual(plan["arms"], ["baseline", "gated_reprompt", "always_reprompt"])
        self.assertEqual((plan["seeds_per_task"], plan["seed_range"]), (100, [400007, 400106]))
        self.assertEqual((plan["num_workers"], plan["seconds_per_env_step"]), (2, 0.055))
        self.assertEqual(plan["warnings"], [])
        for kind in ("expected", "upper"):
            self.assertTrue(plan["totals"][kind]["fits_single_allocation"], kind)
            self.assertLess(plan["totals"][kind]["wall_hours_with_workers"], plan["usable_hours_per_allocation"])
        self.assertGreater(plan["totals"]["upper"]["worker_hours"], plan["totals"]["expected"]["worker_hours"])
        self.assertEqual(sum(row["rollouts"] for row in plan["per_task"]), 900)
        self.assertEqual(plan["totals"]["upper"]["env_steps"], 300 * (2250 + 1950 + 2700))
        power = {(row["group"], row["p_help"], row["p_harm"]): row["power"] for row in plan["power"]["rows"]}
        self.assertGreater(power[("pooled_whitelisted", 0.15, 0.0)], power[("per_task", 0.15, 0.0)])
        self.assertGreater(power[("per_task", 0.15, 0.0)], power[("per_task", 0.25, 0.1)])
        resolved = gate.resolve_config(self.load())
        self.assertEqual(resolved["reprompt_whitelist"], {"ArrangeTea": ["mug_on_tray"], "BreadSelection": ["jam_on_cutting_board"]})

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            gate.resolve_config(gate_config(arms=["gated_reprompt", "baseline"]))
        with self.assertRaises(ValueError):
            gate.resolve_config(gate_config(arms=["baseline", "sometimes"]))
        with self.assertRaises(ValueError):
            gate.resolve_config(gate_config(reprompt_whitelist={"Elsewhere": ["b"]}))
        with self.assertRaises(ValueError):
            gate.resolve_config(gate_config(unknown_option=1))
        with self.assertRaises(ValueError):
            gate.resolve_config(gate_config(reprompt_step_cap=0))

    def test_task_subset_keeps_only_its_whitelist_entries(self):
        # The smoke run passes --tasks ArrangeTea with the shipped config.
        import argparse

        config_path = str(Path(__file__).resolve().parents[1] / "scripts/stage_gated_reprompt_xr1.json")
        args = argparse.Namespace(config=config_path, tasks=["ArrangeTea"], arms=None, seeds_per_task=2,
                                  base_seed=None, num_workers=None, policy_arg=None)
        resolved = gate._load_config(args)
        self.assertEqual(resolved["tasks"], ["ArrangeTea"])
        self.assertEqual(resolved["reprompt_whitelist"], {"ArrangeTea": ["mug_on_tray"]})
        full = argparse.Namespace(config=config_path, tasks=None, arms=None, seeds_per_task=None,
                                  base_seed=None, num_workers=None, policy_arg=None)
        self.assertEqual(set(gate._load_config(full)["reprompt_whitelist"]), {"ArrangeTea", "BreadSelection"})

    def test_cli_plan_and_analyze(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                plan = gate.main(["plan", "--config", str(REPO / "scripts" / "stage_gated_reprompt_xr1.json"), "--num-workers", "1", "--output", str(Path(tmp) / "plan.json")])
            self.assertEqual(plan["num_workers"], 1)
            self.assertTrue((Path(tmp) / "plan.json").exists())
            collect(Path(tmp) / "run")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                gate.main(["analyze", "--run-dirs", str(Path(tmp) / "run"), "--output-dir", str(Path(tmp) / "analysis"), "--n-boot", "50"])
            self.assertIn("Primary endpoint", out.getvalue())


if __name__ == "__main__":
    unittest.main()
