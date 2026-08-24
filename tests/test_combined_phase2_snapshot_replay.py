import json
from pathlib import Path
import tempfile
import unittest

from robocasa.recovery.audit_combined_phase2_snapshot_replay import (
    audit_combined_runs,
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_summary(task, parent_id, snapshot_id, kind, index):
    common = {
        "schema_version": 8,
        "protocol": "phase2_complete_snapshot_branch",
        "branch_id": f"{snapshot_id}-{kind}-{index}",
        "snapshot_id": snapshot_id,
        "parent_id": parent_id,
        "task_name": task,
        "kind": kind,
        "sampling_seed": 10 if kind == "same_seed_repeat" else 100 + index,
        "restore_valid": True,
        "restore_subtask_exact": True,
        "restore_induced_regressed_predicates": [],
        "record_alignment_valid": True,
        "first_request_sha256": "request",
        "first_action_sha256": "action",
        "first_inference_actions_sha256": (
            "repeat-action" if kind == "same_seed_repeat" else f"candidate-{index}"
        ),
        "first_transition_sha256": "transition",
        "first_diagnostic_transition_sha256": "diagnostic-transition",
        "first_transition_component_sha256": {
            "control_state": "control",
            "controller_state": "controller",
            "rng_state": "rng",
            "simulator_state": "simulator",
        },
        "first_diagnostic_transition_component_sha256": {
            "control_state": "control",
            "controller_state": "diagnostic-controller",
            "rng_state": "rng",
            "simulator_state": "simulator",
        },
        "suffix_request_sha256": ["request", "request-2"],
        "suffix_action_sha256": ["action", "action-2"],
        "suffix_observation_sha256": ["observation", "observation-2"],
        "suffix_observation_component_sha256": [
            {"video": "frame-1"},
            {"video": "frame-2"},
        ],
        "suffix_environment_sha256": ["environment", "environment-2"],
        "suffix_diagnostic_environment_sha256": [
            "diagnostic-environment",
            "diagnostic-environment-2",
        ],
        "task_success": False,
        "ordered_completed_subtasks": ["stage-0"],
    }
    if kind == "environment_only":
        common["sampling_seed"] = 500 + index
    common["payload_path"] = f"branches/{common['branch_id']}.pkl.gz"
    return common


def make_source(root, task, *, status="complete"):
    candidate_count = 4
    plan = {
        "tasks": [task],
        "runtime_bundle_sha256": "runtime",
        "checkpoint_provenance_sha256": "checkpoint-provenance",
        "server_repository_commit": "server-commit",
        "robocasa_commit": "robocasa-commit",
        "policy_module": "policy.module",
        "policy_name": "xiaomi",
        "checkpoint": "Xiaomi/checkpoint",
        "checkpoint_revision": None,
        "candidate_count": candidate_count,
        "suffix_steps": 64,
        "snapshot_prefix": 2,
        "landmark_fraction": 0.25,
        "env_interface": "gym",
        "canonical_camera_observations": True,
        "canonical_camera_render_repeats": 2,
        "fresh_branch_contexts": True,
        "branch_policy_connection_mode": "shared_restored",
        "deterministic_environment_construction": True,
        "replan_steps": 16,
        "split": "pretrain",
        "target_stages": {task: f"{task}::frozen-stage"},
        "trigger_stages": {task: f"{task}::live-stage"},
    }
    parent_id = f"{task}-parent"
    snapshot_ids = [f"{task}-snapshot-0", f"{task}-snapshot-1"]
    parent = {
        "parent_id": parent_id,
        "task_name": task,
        "valid": True,
        "captured": {"prefix": snapshot_ids[0], "landmark": snapshot_ids[1]},
    }
    records = []
    for snapshot_id in snapshot_ids:
        records.extend(
            make_summary(task, parent_id, snapshot_id, "same_seed_repeat", index)
            for index in range(2)
        )
        records.extend(
            make_summary(task, parent_id, snapshot_id, "candidate", index)
            for index in range(candidate_count)
        )
        records.append(
            make_summary(task, parent_id, snapshot_id, "environment_only", 0)
        )
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", {"status": status})
    write_jsonl(root / "parent_records.jsonl", [parent])
    write_jsonl(root / "branch_records.jsonl", records)
    write_jsonl(root / "errors.jsonl", [])
    write_jsonl(root / "ineligible_parent_records.jsonl", [])
    for snapshot_id in snapshot_ids:
        path = root / "snapshots" / f"{snapshot_id}.pkl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"snapshot")
    for record in records:
        path = root / record["payload_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")
    return records


class TestCombinedPhase2SnapshotReplay(unittest.TestCase):
    def test_combines_complete_and_stopped_task_roots_without_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrange = root / "arrange"
            cutting = root / "cutting"
            make_source(arrange, "ArrangeTea", status="running")
            make_source(cutting, "CuttingToolSelection")

            analysis = audit_combined_runs(
                [
                    ("ArrangeTea", arrange),
                    ("CuttingToolSelection", cutting),
                ],
                root / "combined",
                parents_per_task=1,
            )

            self.assertTrue(analysis["all_pass"], analysis)
            self.assertEqual(analysis["completed_parents"], 2)
            self.assertEqual(analysis["completed_snapshots"], 4)
            self.assertEqual(analysis["primary_records"], 24)
            self.assertEqual(analysis["records"], 28)
            self.assertFalse(analysis["source_artifacts_copied"])
            self.assertFalse(
                analysis["source_provenance"][0]["source_analysis_present"]
            )
            self.assertEqual(
                json.loads((root / "combined" / "status.json").read_text())["status"],
                "complete",
            )

    def test_rejects_missing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrange = root / "arrange"
            cutting = root / "cutting"
            arrange_records = make_source(arrange, "ArrangeTea")
            make_source(cutting, "CuttingToolSelection")
            (arrange / arrange_records[0]["payload_path"]).unlink()

            with self.assertRaisesRegex(FileNotFoundError, "Missing branch payload"):
                audit_combined_runs(
                    [
                        ("ArrangeTea", arrange),
                        ("CuttingToolSelection", cutting),
                    ],
                    root / "combined",
                    parents_per_task=1,
                )

    def test_rejects_mixed_branch_context_isolation_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrange = root / "arrange"
            cutting = root / "cutting"
            make_source(arrange, "ArrangeTea")
            make_source(cutting, "CuttingToolSelection")
            cutting_plan_path = cutting / "plan.json"
            cutting_plan = json.loads(cutting_plan_path.read_text())
            cutting_plan["fresh_branch_contexts"] = False
            write_json(cutting_plan_path, cutting_plan)

            with self.assertRaisesRegex(
                ValueError, "Incompatible source protocols"
            ):
                audit_combined_runs(
                    [
                        ("ArrangeTea", arrange),
                        ("CuttingToolSelection", cutting),
                    ],
                    root / "combined",
                    parents_per_task=1,
                )


if __name__ == "__main__":
    unittest.main()
