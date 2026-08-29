import json
from pathlib import Path
import tempfile
import unittest

from robocasa.recovery.phase3b_training_horizon import (
    PHASE3B_TASK_HORIZONS,
    build_registration,
    completion_event,
    first64_equality_audit,
    route_phase3b,
    sha256_file,
    validate_registration_source,
)


TASKS = ("ArrangeTea", "CuttingToolSelection")


def _write_phase2_source(root: Path):
    candidate_parents = []
    parent_rows = []
    branch_rows = []
    snapshot_index = 0
    for task_index, task in enumerate(TASKS):
        # Reverse reset order in the ledger so deterministic selection must use
        # the frozen identity, not file order.
        for reset_index in (4, 3, 2, 1, 0):
            parent_id = f"{task}-parent-{reset_index}"
            candidate_parents.append(
                {
                    "parent_id": parent_id,
                    "task_name": task,
                    "environment_seed": 1000 + task_index * 10 + reset_index,
                    "environment_reset_index": reset_index,
                }
            )
            captured = {}
            for boundary in ("prefix", "landmark"):
                snapshot_id = f"snapshot-{snapshot_index:02d}"
                snapshot_index += 1
                captured[boundary] = snapshot_id
                snapshot_path = root / "snapshots" / f"{snapshot_id}.pkl.gz"
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_path.write_bytes(f"snapshot:{snapshot_id}".encode())
                specifications = [
                    ("same_seed_repeat", 5000 + snapshot_index, 0),
                    ("same_seed_repeat", 5000 + snapshot_index, 1),
                    *[
                        ("candidate", 7000 + snapshot_index * 10 + index, None)
                        for index in range(4)
                    ],
                    ("environment_only", 5000 + snapshot_index, None),
                ]
                for branch_number, (kind, seed, repeat) in enumerate(specifications):
                    branch_id = f"{snapshot_id}-{branch_number}"
                    payload_path = root / "branches" / f"{branch_id}.pkl.gz"
                    payload_path.parent.mkdir(parents=True, exist_ok=True)
                    payload_path.write_bytes(f"payload:{branch_id}".encode())
                    branch_rows.append(
                        {
                            "branch_id": branch_id,
                            "snapshot_id": snapshot_id,
                            "parent_id": parent_id,
                            "task_name": task,
                            "kind": kind,
                            "repeat_index": repeat,
                            "sampling_seed": seed,
                            "payload_path": str(payload_path.relative_to(root)),
                            "payload_sha256": f"embedded-{branch_id}",
                            "payload_file_sha256": sha256_file(payload_path),
                        }
                    )
            parent_rows.append(
                {
                    "parent_id": parent_id,
                    "task_name": task,
                    "captured": captured,
                    "valid": True,
                }
            )

    plan = {
        "protocol": "phase2_complete_snapshot_replay",
        "tasks": list(TASKS),
        "target_stages": {task: f"{task}::target" for task in TASKS},
        "trigger_stages": {task: f"{task}::trigger" for task in TASKS},
        "branch_policy_connection_mode": "shared_restored",
        "candidate_count": 4,
        "suffix_steps": 64,
        "expected_total_branches": 140,
        "candidate_parents": candidate_parents,
        "runtime_bundle_sha256": "runtime",
        "checkpoint_provenance_sha256": "checkpoint",
        "robocasa_commit": "robocasa",
        "server_repository_commit": "server",
    }
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "analysis.json").write_text(json.dumps({"all_pass": True, "errors": 0}))
    (root / "errors.jsonl").write_text("")
    with (root / "parent_records.jsonl").open("w") as stream:
        for row in parent_rows:
            stream.write(json.dumps(row) + "\n")
    with (root / "branch_records.jsonl").open("w") as stream:
        for row in branch_rows:
            stream.write(json.dumps(row) + "\n")


class Phase3BTrainingHorizonTest(unittest.TestCase):
    def test_sentinel_selection_and_registered_horizons_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "phase2"
            root.mkdir()
            _write_phase2_source(root)

            registration = build_registration(root, scope="sentinel")

            self.assertEqual(registration["expected_parents"], 2)
            self.assertEqual(registration["expected_snapshots"], 4)
            self.assertEqual(registration["expected_branches"], 28)
            self.assertEqual(registration["expected_scientific_branches"], 16)
            self.assertEqual(
                {row["environment_reset_index"] for row in registration["selected_snapshots"]},
                {0},
            )
            self.assertEqual(
                {
                    row["task_name"]: row["horizon_environment_steps"]
                    for row in registration["selected_snapshots"]
                },
                PHASE3B_TASK_HORIZONS,
            )
            self.assertFalse(
                registration["scientific_outcomes_may_stop_full_progression"]
            )
            source_rows = {
                json.loads(line)["branch_id"]: json.loads(line)
                for line in (root / "branch_records.jsonl").read_text().splitlines()
            }
            self.assertTrue(
                all(
                    row["sampling_seed"]
                    == source_rows[row["source_branch_id"]]["sampling_seed"]
                    for row in registration["registered_branches"]
                )
            )
            validate_registration_source(registration)

            (root / "analysis.json").write_text(
                json.dumps({"all_pass": True, "errors": 0, "changed": True})
            )
            with self.assertRaisesRegex(ValueError, "source artifacts changed"):
                validate_registration_source(registration)

    def test_full_registration_requires_matching_all_pass_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "phase2"
            sentinel = Path(directory) / "sentinel"
            root.mkdir()
            sentinel.mkdir()
            _write_phase2_source(root)
            sentinel_registration = build_registration(root, scope="sentinel")
            (sentinel / "registration.json").write_text(
                json.dumps(sentinel_registration)
            )
            (sentinel / "analysis.json").write_text(
                json.dumps({"engineering_all_pass": True})
            )

            full = build_registration(
                root, scope="full", sentinel_run_dir=sentinel
            )

            self.assertEqual(full["expected_parents"], 10)
            self.assertEqual(full["expected_snapshots"], 20)
            self.assertEqual(full["expected_branches"], 140)
            self.assertEqual(full["expected_scientific_branches"], 80)

    def test_first64_audit_fails_closed_on_any_channel(self):
        action = [f"a-{index}" for index in range(64)]
        observations = [f"o-{index}" for index in range(65)]
        environment = [f"e-{index}" for index in range(64)]
        requests = [f"r-{index}" for index in range(4)]
        trace = [{"step": index} for index in range(65)]
        source_record = {
            "branch_id": "source",
            "sampling_seed": 17,
            "num_steps": 64,
            "num_policy_requests": 4,
            "suffix_action_sha256": action,
            "suffix_observation_sha256": observations,
            "suffix_environment_sha256": environment,
            "suffix_request_sha256": requests,
        }
        source_payload = {"subtask_trace": trace}
        result = {
            "summary": {
                "branch_id": "continued",
                "sampling_seed": 17,
                "num_steps": 480,
                "suffix_action_sha256": action + ["later"],
                "suffix_observation_sha256": observations + ["later"],
                "suffix_environment_sha256": environment + ["later"],
                "suffix_request_sha256": requests + ["later"],
            },
            "payload": {"subtask_trace": trace + [{"step": 65}]},
        }
        self.assertTrue(
            first64_equality_audit(source_record, source_payload, result)[
                "all_exact"
            ]
        )
        result["summary"]["suffix_action_sha256"][10] = "mismatch"
        with self.assertRaisesRegex(ValueError, "actions"):
            first64_equality_audit(source_record, source_payload, result)

    def test_routing_only_unlocks_critic_when_all_registered_gates_pass(self):
        tasks = {
            task: {
                "mixed_outcome_fraction": 0.5,
                "random_minus_nominal": 0.1,
            }
            for task in TASKS
        }
        macro = {"mixed_outcome_fraction": 0.5, "oracle_minus_nominal": 0.2}
        gates = {"support": True, "headroom": True}
        self.assertEqual(
            route_phase3b(tasks, macro, gates),
            "candidate_critic_formal_five_task_screen",
        )
        gates["headroom"] = False
        self.assertEqual(
            route_phase3b(tasks, macro, gates), "higher_level_operator_screen"
        )

        homogeneous = {
            task: {
                "mixed_outcome_fraction": 0.0,
                "random_minus_nominal": 0.25,
            }
            for task in TASKS
        }
        self.assertEqual(
            route_phase3b(homogeneous, macro, {"support": False}),
            "fresh_replan_without_candidate_ranking",
        )
        self.assertEqual(
            route_phase3b(
                tasks,
                {"mixed_outcome_fraction": 0.5, "oracle_minus_nominal": 0.0},
                {"support": False},
            ),
            "do_not_train_candidate_critic",
        )

    def test_completion_event_uses_genuine_request_indices_and_checks_durability(self):
        payload = {
            "summary": {"request_environment_step_indices": [0, 16, 32, 48]},
            "subtask_evals": [
                {"predicates": {"trigger": {"value": False}}}
                for _ in range(34)
            ],
        }
        payload["subtask_evals"][33]["predicates"]["trigger"]["value"] = True

        event = completion_event(payload, "trigger")

        self.assertTrue(event["completed"])
        self.assertEqual(event["first_completion_environment_step"], 32)
        self.assertEqual(event["first_completion_replan"], 3)
        self.assertTrue(event["durable_to_branch_end"])


if __name__ == "__main__":
    unittest.main()
