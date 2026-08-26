import csv
import gzip
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from robocasa.recovery.analyze_phase3_candidate_opportunity import (
    analyze_candidate_opportunity,
)
from robocasa.recovery.full_snapshot import stable_digest


class FakeSafeScorer:
    def score_candidates(self, raw_features, *, task_name):
        self.last_shape = tuple(np.asarray(raw_features).shape)
        self.last_task = task_name
        return {
            "scores": [3.0, 1.0, 4.0, 2.0],
            "provenance": {"protocol": "fake-frozen-safe"},
        }


def _payload(stage, completed, marker):
    final_completed = [stage] if completed else []
    final_stage = "next_stage" if completed else stage
    final_progress = 0.5 if completed else 0.0
    features = np.full((3, 4, 2), float(marker), dtype=np.float32)
    actions = np.full((4, 2), float(marker), dtype=np.float32)
    return {
        "summary": {},
        "first_action": actions[0],
        "actions": [actions[0]],
        "request_records": [],
        "inference_records": [
            {
                "features": features,
                "actions": actions,
                "auxiliary_features": {
                    "observation_state_history": np.zeros((2, 5), dtype=np.float32)
                },
            }
        ],
        "subtask_evals": [],
        "subtask_trace": [
            {
                "ordered_current_subtask": stage,
                "ordered_completed_subtasks": [],
                "ordered_subtask_progress": 0.0,
            },
            {
                "ordered_current_subtask": final_stage,
                "ordered_completed_subtasks": final_completed,
                "ordered_subtask_progress": final_progress,
            },
        ],
        "rewards": [],
        "infos": [],
        "first_transition_fingerprint": {},
        "first_causal_transition_fingerprint": {},
    }


def _write_branch(root, *, snapshot, parent, kind, seed, completed, marker, repeat=0):
    branch_id = f"{snapshot}-{kind}-{seed}-{repeat}"
    payload = _payload("active_stage", completed, marker)
    payload_path = root / "branches" / f"{branch_id}.pkl.gz"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(payload_path, "wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    return {
        "branch_id": branch_id,
        "snapshot_id": snapshot,
        "parent_id": parent,
        "task_name": "ArrangeTea",
        "trigger_name": "prefix_2",
        "kind": kind,
        "sampling_seed": seed,
        "task_success": False,
        "regressed_predicates": [],
        "termination_reason": "suffix_horizon",
        "num_steps": 4,
        "num_policy_requests": 1,
        "payload_path": str(payload_path.relative_to(root)),
        "payload_sha256": stable_digest(payload),
    }


def _write_source(root, *, all_pass=True, disagree_nominal=False):
    plan = {
        "schema_version": 4,
        "protocol": "phase2_complete_snapshot_replay",
        "robocasa_commit": "abc123",
        "runtime_bundle_sha256": "runtime-sha",
        "checkpoint_provenance_sha256": "checkpoint-sha",
        "server_repository_commit": "server123",
        "tasks": ["ArrangeTea"],
        "trigger_stages": {"ArrangeTea": "ArrangeTea::active_stage"},
        "expected_snapshots": 2,
        "candidate_count": 4,
        "suffix_steps": 4,
        "branch_policy_connection_mode": "shared_restored",
    }
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "analysis.json").write_text(json.dumps({"all_pass": all_pass}))
    (root / "errors.jsonl").write_text("")
    (root / "snapshots").mkdir()

    rows = []
    outcomes = {
        "snapshot-a": [True, False, True, False],
        "snapshot-b": [False, False, False, False],
    }
    for snapshot_index, (snapshot, candidate_outcomes) in enumerate(outcomes.items()):
        parent = f"parent-{snapshot_index}"
        (root / "snapshots" / f"{snapshot}.pkl.gz").write_bytes(
            f"snapshot-{snapshot}".encode()
        )
        nominal = snapshot_index == 0
        for repeat in range(2):
            rows.append(
                _write_branch(
                    root,
                    snapshot=snapshot,
                    parent=parent,
                    kind="same_seed_repeat",
                    seed=100 + snapshot_index,
                    completed=(
                        not nominal if disagree_nominal and repeat == 1 else nominal
                    ),
                    marker=10 + repeat,
                    repeat=repeat,
                )
            )
        for candidate_index, completed in enumerate(candidate_outcomes):
            rows.append(
                _write_branch(
                    root,
                    snapshot=snapshot,
                    parent=parent,
                    kind="candidate",
                    seed=1000 + snapshot_index * 10 + candidate_index,
                    completed=completed,
                    marker=candidate_index + 1,
                )
            )
        rows.append(
            _write_branch(
                root,
                snapshot=snapshot,
                parent=parent,
                kind="environment_only",
                seed=9000 + snapshot_index,
                completed=False,
                marker=20,
            )
        )
    with (root / "branch_records.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


class Phase3CandidateOpportunityTest(unittest.TestCase):
    def test_paired_opportunity_analysis_and_frozen_safe_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "phase2"
            output = Path(directory) / "phase3"
            root.mkdir()
            _write_source(root)
            scorer = FakeSafeScorer()

            analysis = analyze_candidate_opportunity(
                root,
                output,
                bootstrap_replicates=100,
                bootstrap_seed=7,
                scorer=scorer,
            )

            self.assertEqual(analysis["status"], "pilot_complete")
            self.assertEqual(analysis["parents"], 2)
            self.assertEqual(analysis["snapshots"], 2)
            self.assertEqual(analysis["candidates"], 8)
            self.assertEqual(analysis["task_macro"]["mixed_outcome_fraction"], 0.5)
            self.assertEqual(analysis["task_macro"]["oracle_headroom"], 0.25)
            self.assertTrue(analysis["bounded_pilot_supports_scaling"])
            self.assertIsNone(analysis["formal_phase3_gate_passed"])
            self.assertEqual(scorer.last_shape, (4, 3, 4, 2))
            self.assertEqual(scorer.last_task, "ArrangeTea")
            self.assertEqual(
                analysis["task_macro"]["safe_selected_stage_completion"],
                0.0,
            )
            self.assertTrue(analysis["phase2_source_unchanged"])

            candidates = [
                json.loads(line)
                for line in (output / "candidate_opportunity.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(candidates), 8)
            self.assertEqual(candidates[0]["robocasa_commit"], "abc123")
            self.assertIn("source_payload_file_sha256", candidates[0])
            self.assertIn("safe_feature_sha256", candidates[0])
            self.assertNotIn("features", candidates[0])
            self.assertTrue((output / "analysis.json").is_file())
            with (output / "snapshot_opportunity.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_rejects_invalid_phase2_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "phase2"
            root.mkdir()
            _write_source(root, all_pass=False)
            with self.assertRaisesRegex(ValueError, "all-pass Phase 2"):
                analyze_candidate_opportunity(root, Path(directory) / "out")

    def test_rejects_disagreeing_nominal_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "phase2"
            root.mkdir()
            _write_source(root, disagree_nominal=True)
            with self.assertRaisesRegex(ValueError, "Nominal repeat outcomes disagree"):
                analyze_candidate_opportunity(root, Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()
