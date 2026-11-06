"""Simulator-free tests for the R0 failure audit at the Hide-and-Seek + P alarm."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()
from robocasa.recovery import failure_audit as fa
from robocasa.recovery import recoverability_ladder as ladder
from tests.test_alarm_branching import alarm_rows, at_alarm
from tests.test_recoverability_ladder import MockRuntime, StageMockEnv, stage_config


def quiet(*_):
    pass


def fake_inspector(env):
    """``obj`` falls to the floor when the mock object is stuck (qpos[3] set at the trigger)."""
    qpos = env.sim.qpos
    stuck = qpos[3] >= 0.5
    objects = {
        "obj": {"pos": [0.0, 0.0, 0.1 if stuck else 0.9], "up_cos": 1.0, "grasped": False,
                "counter_contact": not stuck, "outside_scene": False, "on_floor": bool(stuck),
                "eef_dist": 0.3, "base_xy_dist": 0.5},
    }
    return {"objects": objects, "eef_pos": [0.0, 0.0, 1.0], "holding": []}


def fake_imager(env):
    return {"robot0_agentview": np.zeros((4, 4, 3), dtype=np.uint8)}


class FailureAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "ladder"
        ladder.run_collection(stage_config(), self.source, runtime=MockRuntime(env_class=StageMockEnv), log=quiet)
        self.parents = ladder.read_jsonl(self.source / ladder.PARENTS_NAME)
        self.failed = [p for p in self.parents if not p["success"]]
        self.alarms = self.root / "alarms.jsonl"
        with self.alarms.open("w") as stream:
            for row in alarm_rows(self.parents, at_alarm):
                stream.write(json.dumps(row) + "\n")
        # One failed parent per mock failure mode (stuck, jammed, knocked back).
        self.assertEqual(sorted(p["environment_seed"] % 4 for p in self.failed), [1, 2, 3])
        # One Phase A1 shard: the jammed parent recovered once.
        self.branches = self.root / "a1" / "shard0"
        self.branches.mkdir(parents=True)
        self.recovered_id = next(p["parent_id"] for p in self.failed if p["environment_seed"] % 4 == 2)
        with (self.branches / "branches.jsonl").open("w") as stream:
            for parent in self.failed:
                for k in range(2):
                    success = parent["parent_id"] == self.recovered_id and k == 1
                    stream.write(json.dumps({"parent_id": parent["parent_id"], "anchor": "alarm", "k": k,
                                             "success": success}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self, out="audit", **kwargs):
        kwargs.setdefault("branch_runs", [self.branches])
        kwargs.setdefault("inspector", fake_inspector)
        kwargs.setdefault("imager", fake_imager)
        kwargs.setdefault("runtime", MockRuntime(env_class=StageMockEnv))
        return fa.run_audit([self.source], self.alarms, self.root / out, log=quiet, **kwargs)

    def rows(self, out="audit"):
        return {r["parent_id"]: r for r in ladder.read_jsonl(self.root / out / fa.STATES_NAME)}

    def test_audits_every_alarmed_failure_at_its_alarm(self):
        summary = self.audit()
        rows = self.rows()
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(set(rows), {p["parent_id"] for p in self.failed})
        by_id = {p["parent_id"]: p for p in self.failed}
        for pid, row in rows.items():
            parent = by_id[pid]
            self.assertEqual(row["alarm_snapshot_index"], at_alarm(parent))
            self.assertEqual(row["alarm_step"], parent["snapshot_steps"][at_alarm(parent)])
            self.assertEqual(row["scene_residual"], 0.0)
            self.assertEqual(set(row["restore_residuals"].values()), {0.0})
            self.assertEqual(set(row["facts"]), set(fa.SNAPSHOT_LABELS))
            self.assertEqual(row["saved_live_predicate_mismatch"], [])
            self.assertEqual(row["stage_at_alarm"],
                             ladder.stage_label(parent["required_subtasks"], row["ordered_count_at_alarm"]))
            for label in ("alarm", "final"):
                for name in row["images"][label].values():
                    self.assertTrue((self.root / "audit" / "images" / name).exists())
        self.assertTrue((self.root / "audit" / fa.REVIEW_CSV).exists())
        self.assertTrue((self.root / "audit" / fa.REVIEW_MD).exists())

    def test_provisional_classes_follow_resampling_and_the_floor_rule(self):
        self.audit()
        classes = sorted(r["provisional_class"] for r in self.rows().values())
        self.assertEqual(classes, ["recoverable", "recovered_by_resampling", "unrecoverable"])
        for pid, row in self.rows().items():
            mode = row["environment_seed"] % 4
            if pid == self.recovered_id:
                self.assertEqual(row["provisional_class"], "recovered_by_resampling")
                self.assertEqual(row["a1_successful_branches"], 1)
            elif mode == 1:  # object stuck -> the fake inspector puts it on the floor
                self.assertEqual(row["provisional_class"], "unrecoverable")
                self.assertIn("obj:on_floor", row["flags"])
            else:
                self.assertEqual(row["provisional_class"], "recoverable")
            if mode == 3:  # knocked back after reaching
                self.assertIn("reach", row["regressed_at_alarm"])
                self.assertIn("undone:reach", row["flags"])

    def test_decide_applies_manual_classes_and_the_gate(self):
        self.audit()
        audit_dir = self.root / "audit"
        result = fa.decide(audit_dir)
        rows = self.rows()
        recoverable = sum(r["provisional_class"] == "recoverable" for r in rows.values())
        self.assertEqual(result["recoverable_never_recovered"], recoverable)
        self.assertEqual(result["decision"], "stop_report_detection_and_resampling")  # < 10 states here
        self.assertEqual(result["classes"]["recovered_by_resampling"], 1)
        # Re-class one recoverable state as unrecoverable by hand.
        target = next(pid for pid, r in rows.items() if r["provisional_class"] == "recoverable")
        review = audit_dir / fa.REVIEW_CSV
        lines = list(csv.DictReader(review.open()))
        for line in lines:
            if line["parent_id"] == target:
                line["manual_class"], line["note"] = "3", "kettle wedged"
        with review.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fa.REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(lines)
        result = fa.decide(audit_dir, review)
        self.assertEqual(result["manual_overrides"], 1)
        self.assertEqual(result["recoverable_never_recovered"], recoverable - 1)
        states = {r["parent_id"]: r for r in ladder.read_jsonl(audit_dir / "r0_states.jsonl")}
        self.assertEqual((states[target]["final_class"], states[target]["note"]), ("unrecoverable", "kettle wedged"))
        self.assertTrue((audit_dir / "r0_gate.md").exists())

    def test_gate_passes_at_the_threshold(self):
        self.audit()
        audit_dir = self.root / "audit"
        rows = ladder.read_jsonl(audit_dir / fa.STATES_NAME)
        template = next(r for r in rows if r["provisional_class"] == "recoverable")
        with (audit_dir / fa.STATES_NAME).open("w") as stream:
            for i in range(fa.GATE_R0_MIN_RECOVERABLE):
                stream.write(json.dumps({**template, "parent_id": f"p{i}"}) + "\n")
        self.assertEqual(fa.decide(audit_dir)["decision"], "proceed_to_R1")

    def test_resampled_recoveries_cannot_be_reclassed(self):
        self.audit()
        review = self.root / "audit" / fa.REVIEW_CSV
        lines = list(csv.DictReader(review.open()))
        for line in lines:
            if line["parent_id"] == self.recovered_id:
                line["manual_class"] = "2"
        with review.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fa.REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(lines)
        with self.assertRaises(ValueError):
            fa.decide(self.root / "audit", review)

    def test_resume_skips_done_states_and_refuses_changed_inputs(self):
        self.audit()
        before = len(ladder.read_jsonl(self.root / "audit" / fa.STATES_NAME))
        with self.assertRaises(FileExistsError):
            self.audit()
        self.audit(resume=True)
        self.assertEqual(len(ladder.read_jsonl(self.root / "audit" / fa.STATES_NAME)), before)
        with self.assertRaises(ValueError):
            self.audit(resume=True, floor_z=0.5)

    def test_inspector_errors_are_recorded_and_the_audit_continues(self):
        bad_seed = self.failed[0]["environment_seed"]

        def flaky(env):
            if getattr(env, "fail_mode", None) == bad_seed % 4 and not hasattr(env, "_raised"):
                env._raised = True
                raise RuntimeError("boom")
            return fake_inspector(env)

        summary = self.audit(inspector=flaky)
        self.assertGreaterEqual(summary["errors"], 1)
        errors = ladder.read_jsonl(self.root / "audit" / fa.ERRORS_NAME)
        self.assertIn("boom", errors[0]["error"])
        self.assertEqual(summary["states"] + summary["errors"], len(self.failed))


class HelperTest(unittest.TestCase):
    def test_regressed_at_ignores_release_and_unlatched_predicates(self):
        names = ["a", "b", "gripper_released", "c"]
        values = np.array([[0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]], dtype=bool)
        self.assertEqual(fa.regressed_at(names, values, 2, ["a", "b", "gripper_released", "c"]), ["a"])
        self.assertEqual(fa.regressed_at(names, values, 1, ["a", "b"]), [])
        self.assertEqual(fa.regressed_at([], None, 1, ["a"]), [])

    def test_tipped_only_when_it_started_upright(self):
        facts = {"objects": {"mug": {"up_cos": 0.2}, "knife": {"up_cos": 0.1}}}
        initial = {"objects": {"mug": {"up_cos": 0.99}, "knife": {"up_cos": 0.05}}}
        self.assertEqual(fa.flags_for(facts, initial, []), ["mug:tipped"])

    def test_task_objects_drop_distractors(self):
        self.assertEqual(fa.task_objects(["obj", "distr_pastry", "dstr_dining", "jam"]), ["obj", "jam"])

    def test_atomic_step_of_a_stage(self):
        self.assertEqual(fa.atomic_step("ArrangeTea", "mug_on_tray"), ("PickPlaceCabinetToCounter", "registered"))
        self.assertEqual(fa.atomic_step("GarnishPancake", "strawberry_on_pancake")[1], "derived")
        self.assertEqual(fa.atomic_step("ArrangeTea", "nope"), (None, None))

    def test_camera_images_keep_rgb_uint8_frames(self):
        class Env:
            def get_current_observation(self):
                return {"cam": np.zeros((2, 2, 3), np.uint8), "batched": np.zeros((1, 2, 2, 3), np.uint8),
                        "depth": np.zeros((2, 2, 1), np.float32), "state": np.zeros(3)}

        self.assertEqual(sorted(fa.camera_images(Env())), ["batched", "cam"])


if __name__ == "__main__":
    unittest.main()
