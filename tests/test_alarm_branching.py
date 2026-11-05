"""Simulator-free tests for branching ladder parents at the Hide-and-Seek + P alarm."""

import json
from pathlib import Path
import tempfile
import unittest


from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()
from robocasa.recovery import alarm_branching as ab
from robocasa.recovery import recoverability_ladder as ladder
from tests.test_recoverability_ladder import MockRuntime, StageMockEnv, stage_config


def quiet(*_):
    pass


class ShiftedEnv(StageMockEnv):
    """Resets to a different scene than the one the ladder recorded."""

    def reset(self, seed=None):
        result = super().reset(seed=seed)
        self.sim.qpos[3] = 0.25
        return result


def at_alarm(parent):
    return min(5, len(parent["snapshot_steps"]) - 2)


def alarm_rows(parents, index_fn):
    rows = []
    for parent in parents:
        index = index_fn(parent)
        rows.append({
            "parent_id": parent["parent_id"],
            "attempt_uid": parent["attempt_uid"],
            "task_name": parent["task_name"],
            "success": parent["success"],
            "alarm_fired": index is not None,
            "alarm_inference": None if index is None else index + 1,
            "alarm_snapshot_index": index,
            "component": None if index is None else "score",
            "alarm_fraction": None if index is None else parent["snapshot_steps"][index] / parent["horizon"],
        })
    return rows


class AlarmBranchingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "ladder"
        ladder.run_collection(stage_config(), self.source, runtime=MockRuntime(env_class=StageMockEnv), log=quiet)
        self.parents = ladder.read_jsonl(self.source / ladder.PARENTS_NAME)
        # Alarm at the 6th inference (snapshot 5, step 20 of H = 40), or the last
        # inference of shorter (successful) parents.
        self.alarms = self.root / "alarms.jsonl"
        self.write_alarms(at_alarm)

    def tearDown(self):
        self.tmp.cleanup()

    def write_alarms(self, index_fn):
        with self.alarms.open("w") as stream:
            for row in alarm_rows(self.parents, index_fn):
                stream.write(json.dumps(row) + "\n")

    def branch(self, out="branches", runtime=None, **kwargs):
        kwargs.setdefault("k_branches", 2)
        kwargs.setdefault("offsets", (2, 4))
        return ab.run_branching([self.source], self.alarms, self.root / out,
                                runtime=runtime or MockRuntime(env_class=StageMockEnv), log=quiet, **kwargs)

    def test_anchors_are_clipped_at_the_stage_entry_before_the_alarm(self):
        parent = dict(next(p for p in self.parents if not p["success"]))
        parent["progress_events"] = [{"step": 13, "subtask": "reach", "ordered_count": 2}]
        anchors, steps = ab.parent_anchors(parent, 5, (2, 4))
        # Stage entry is the first inference at or after step 13: snapshot 4 (step 16).
        self.assertEqual(anchors["stage_entry"], 4)
        self.assertEqual(anchors["alarm"], 5)
        self.assertEqual(anchors["alarm_minus_2"], 4)
        self.assertEqual(anchors["alarm_minus_4"], 4)
        self.assertEqual(anchors["detection"], 5)

    def test_branches_start_at_the_alarm_with_a_fixed_budget(self):
        status = self.branch()
        self.assertEqual(status["errors"], 0)
        out = self.root / "branches"
        rows = ladder.read_jsonl(out / ab.PARENTS_NAME)
        branches = ladder.read_jsonl(out / ab.BRANCHES_NAME)
        scenes = ladder.read_jsonl(out / ab.SCENES_NAME)
        self.assertEqual(len(rows), len(self.parents))
        self.assertTrue(all(scene["ok"] for scene in scenes))
        expected = sum(sum(cell["run"] for cell in row["cells"]) for row in rows) * 2
        self.assertEqual(len(branches), expected)
        for branch in branches:
            self.assertEqual(branch["rung"], "resample_continue")
            self.assertIn(branch["anchor"], ("alarm", "alarm_minus_2", "alarm_minus_4"))
            self.assertIn(branch["source_index"], (5, 2, 3, 4))
            # Default: ceil(0.5 * H) = 20 steps whatever the clock.
            self.assertEqual(branch["budget_steps"], 20)
            self.assertEqual(branch["remaining_steps"], 40 - branch["budget_start_step"])
            self.assertEqual(branch["success_within_remaining"],
                             branch["success"] and branch["steps_used"] <= branch["remaining_steps"])
            self.assertEqual(branch["policy_context"], "reset")
            self.assertEqual(branch["source_restore_residual"], 0.0)
            self.assertIn(branch["outcome"], ("failure", "false_alarm_success"))
        alarm = [b for b in branches if b["anchor"] == "alarm" and b["outcome"] == "failure"]
        self.assertTrue(alarm)
        self.assertTrue(all(b["anchor_step"] == 20 for b in alarm))
        seeds = {(b["parent_id"], b["anchor"], b["k"]): b["branch_seed"] for b in branches}
        self.assertEqual(len(set(seeds.values())), len(seeds))

    def test_remaining_horizon_budget_is_still_available(self):
        self.branch(budget_mode="remaining_horizon")
        branches = ladder.read_jsonl(self.root / "branches" / ab.BRANCHES_NAME)
        self.assertTrue(branches)
        for branch in branches:
            self.assertEqual(branch["budget_steps"], 40 - branch["budget_start_step"])
            self.assertEqual(branch["success_within_remaining"], branch["success"])
        plan = json.loads((self.root / "branches" / ab.PLAN_NAME).read_text())
        self.assertEqual(plan["budget_mode"], "remaining_horizon")

    def test_resume_runs_nothing_twice(self):
        self.branch()
        before = ladder.read_jsonl(self.root / "branches" / ab.BRANCHES_NAME)
        self.branch(resume=True)
        after = ladder.read_jsonl(self.root / "branches" / ab.BRANCHES_NAME)
        self.assertEqual(len(before), len(after))
        with self.assertRaises(FileExistsError):
            self.branch()
        with self.assertRaises(ValueError):
            self.branch(resume=True, k_branches=3)
        with self.assertRaises(ValueError):
            self.branch(resume=True, budget_fraction=0.25)

    def test_scene_mismatch_is_recorded_and_skipped(self):
        self.branch(runtime=MockRuntime(env_class=ShiftedEnv))
        out = self.root / "branches"
        scenes = ladder.read_jsonl(out / ab.SCENES_NAME)
        self.assertTrue(scenes)
        self.assertFalse(any(scene["ok"] for scene in scenes))
        self.assertEqual(ladder.read_jsonl(out / ab.BRANCHES_NAME), [])

    def test_failures_only_and_silent_parents(self):
        failed = {p["parent_id"] for p in self.parents if not p["success"]}
        first = self.parents[0]["parent_id"]
        self.write_alarms(lambda parent: None if parent["parent_id"] == first else at_alarm(parent))
        self.branch(include_false_alarms=False)
        rows = ladder.read_jsonl(self.root / "branches" / ab.PARENTS_NAME)
        self.assertEqual({r["parent_id"] for r in rows}, failed - {first})

    def test_stale_alarm_rows_are_refused(self):
        rows = alarm_rows(self.parents, at_alarm)
        rows[0]["attempt_uid"] = "other"
        self.alarms.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(ValueError):
            self.branch()

    def test_gate_a_reads_aliases_and_decides(self):
        self.branch()
        result = ab.gate_a([self.root / "branches"], k_branches=2, n_boot=50)
        self.assertIn(result["decision"], ("harvest_at_alarm_and_earlier", "harvest_at_alarm_minus_16_only",
                                           "redesign_expert_data"))
        rows = ladder.read_jsonl(self.root / "branches" / ab.PARENTS_NAME)
        failures = sum(r["outcome"] == "failure" for r in rows)
        self.assertEqual(result["cells"]["failure/alarm"]["parents"], failures)
        # Every requested anchor is reported, aliased or not.
        for anchor in ("alarm", "alarm_minus_2", "alarm_minus_4"):
            self.assertEqual(result["cells"][f"failure/{anchor}"]["parents"], failures)
        path = ab.write_gate_a(result, self.root / "gate")
        self.assertIn("Gate A", path.read_text())

    def test_gate_a_thresholds(self):
        def fake(outcomes):
            rows = {f"p{i}": {"parent_id": f"p{i}", "attempt_uid": "u", "task_name": "T", "outcome": "failure",
                              "horizon_steps": 100,
                              "cells": [{"anchor": a, "canonical": ["resample_continue", a], "run": True}
                                        for a in outcomes]}
                    for i in range(10)}
            return rows

        def run(shares, late=0):
            out = self.root / f"fake_{len(list(self.root.iterdir()))}"
            out.mkdir()
            rows = fake(shares)
            with (out / ab.PARENTS_NAME).open("w") as stream:
                for row in rows.values():
                    stream.write(json.dumps(row) + "\n")
            with (out / ab.SCENES_NAME).open("w") as stream:
                for pid in rows:
                    stream.write(json.dumps({"parent_id": pid, "ok": True}) + "\n")
            with (out / ab.BRANCHES_NAME).open("w") as stream:
                for anchor, share in shares.items():
                    for i, pid in enumerate(rows):
                        # The first `late` successes finish after the natural rollout's time limit.
                        stream.write(json.dumps({"parent_id": pid, "attempt_uid": "u", "anchor": anchor,
                                                 "k": 0, "success": i < share * 10, "budget_start_step": 90,
                                                 "steps_used": 50 if i < late else 5}) + "\n")
            result = ab.gate_a([out], k_branches=1, n_boot=20)
            self.last = result
            return result["decision"]

        self.assertEqual(run({"alarm": 0.3, "alarm_minus_8": 0.1, "alarm_minus_16": 0.0}),
                         "harvest_at_alarm_and_earlier")
        self.assertEqual(run({"alarm": 0.1, "alarm_minus_8": 0.2, "alarm_minus_16": 0.4}),
                         "harvest_at_alarm_minus_16_only")
        self.assertEqual(run({"alarm": 0.1, "alarm_minus_8": 0.2, "alarm_minus_16": 0.2}), "redesign_expert_data")
        # The gate reads the fixed budget; late successes count there but not within the remaining time.
        self.assertEqual(run({"alarm": 0.4, "alarm_minus_8": 0.0, "alarm_minus_16": 0.0}, late=3),
                         "harvest_at_alarm_and_earlier")
        cell = self.last["cells"]["failure/alarm"]
        self.assertEqual(cell["share_with_any_success"], 0.4)
        self.assertEqual(cell["share_with_any_success_within_remaining"], 0.1)

    def test_plan_counts_without_a_simulator(self):
        plan = ab.plan_summary([self.source], self.alarms, k_branches=2, offsets=(2, 4))
        task = plan["by_task"]["MockTask"]
        self.assertEqual(task["failures"] + task["false_alarm_successes"], len(self.parents))
        self.assertEqual(plan["branches"], task["branches"])
        self.assertGreater(task["budget_steps_upper_bound"], 0)
        self.assertGreaterEqual(plan["worker_hours_upper_bound"], 0)


if __name__ == "__main__":
    unittest.main()
