"""Frozen, detector-independent XR1 recovery-operator screen.

The stage-retry goal is supplied by the experimental supervisor, never inferred
from simulator predicates. This measures operator capability at known subgoals,
not deployable stage tracking. Existing Phase 2/3B modules remain unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from robocasa.recovery.exact_replay.counterfactual_branch import load_branch_records
from robocasa.recovery.full_snapshot import stable_digest
from robocasa.recovery.exact_replay.phase3b_training_horizon import (
    PHASE3B_TASK_HORIZONS,
    deterministic_snapshot_selection,
    load_branch_payload,
    load_registered_phase2_snapshot,
    read_jsonl,
    sha256_file,
    sha256_json,
    validate_registration_source,
)


PROTOCOL = "xr1_higher_level_operator_screen_v1"
OPERATORS = ("continue", "history_reset_replan", "stage_retry", "retreat_reset_replan")
ORACLE = "oracle_full_state_rewind"
DEFAULT_SPEC = {
    "operators": list(OPERATORS),
    "continuation_seed_bases": [81000000, 82000000],
    "seed_stride_per_snapshot": 1000,
    "retry_steps": 64,
    "subgoal_source": "fixed_supervisor_supplied_task_goal_no_online_stage_claim",
    "subgoal_prompts": {
        "ArrangeTea": "Pick the mug from the cabinet and place it on the tray.",
        "CuttingToolSelection": "Open the drawer.",
    },
    "retreat": {
        "release_steps": 8,
        "lift_steps": 16,
        "lift_command": 0.1,
        "control_mode": 0.0,
        "gripper_close": 0.0,
        "base_motion": [0.0, 0.0, 0.0, 0.0],
    },
    "gates": {
        "operator_gain": 0.10,
        "routing_gain": 0.10,
        "max_regression_increase": 0.05,
        "min_benefiting_parents": 3,
    },
    "endpoint": "same_environment_step_budget_including_scripted_actions",
    "oracle_policy": "optional_full_state_anchor_rewind_diagnostic_only",
}


def validate_spec(spec):
    if set(spec) != set(DEFAULT_SPEC):
        raise ValueError("Operator specification has missing or unknown fields")
    if spec["operators"] != list(OPERATORS):
        raise ValueError("Exactly the four registered operators are required")
    seeds = spec["continuation_seed_bases"]
    if len(seeds) != 2 or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("Require two nonnegative integer continuation seed bases")
    stride = spec["seed_stride_per_snapshot"]
    if type(stride) is not int or stride < max(PHASE3B_TASK_HORIZONS.values()):
        raise ValueError("Seed stride must exceed any possible per-branch request count")
    if abs(seeds[0] - seeds[1]) < 20 * stride:
        raise ValueError("Continuation seed schedule ranges overlap")
    if type(spec["retry_steps"]) is not int or not 0 < spec["retry_steps"] < 256:
        raise ValueError("Retry duration must be a positive bounded prefix below 256")
    prompts = spec["subgoal_prompts"]
    if set(prompts) != set(PHASE3B_TASK_HORIZONS) or any(
        not isinstance(x, str) or not x.strip() for x in prompts.values()
    ):
        raise ValueError("Require one fixed supervisor-supplied prompt per task")
    retreat = spec["retreat"]
    if set(retreat) != set(DEFAULT_SPEC["retreat"]):
        raise ValueError("Invalid retreat specification")
    if any(type(retreat[k]) is not int or retreat[k] < 1 for k in ("release_steps", "lift_steps")):
        raise ValueError("Retreat release/lift durations must be positive integers")
    if retreat["release_steps"] + retreat["lift_steps"] >= 256:
        raise ValueError("Retreat must leave time for a policy request")
    if not 0 < retreat["lift_command"] <= 0.2:
        raise ValueError("Lift command must be in (0, 0.2]")
    for key in ("control_mode", "gripper_close", "base_motion"):
        if retreat[key] != DEFAULT_SPEC["retreat"][key]:
            raise ValueError(f"Unsupported retreat {key}")
    if set(spec["gates"]) != set(DEFAULT_SPEC["gates"]):
        raise ValueError("Missing or unknown scientific gate")
    for key in ("operator_gain", "routing_gain", "max_regression_increase"):
        value = spec["gates"][key]
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"Invalid scientific margin: {key}")
    if type(spec["gates"]["min_benefiting_parents"]) is not int or spec["gates"]["min_benefiting_parents"] < 2:
        raise ValueError("Require benefit in multiple independent parents")
    for key in ("subgoal_source", "endpoint", "oracle_policy"):
        if spec[key] != DEFAULT_SPEC[key]:
            raise ValueError(f"Unsupported {key}")


def code_hashes():
    root = Path(__file__).resolve().parents[3]
    names = (
        "robocasa/recovery/exact_replay/operator_screen.py",
        "robocasa/recovery/exact_replay/run_operator_screen.py",
        "robocasa/recovery/exact_replay/analyze_operator_screen.py",
        "robocasa/recovery/xiaomi_robotics_1_policy.py",
        "robocasa/recovery/openpi_websocket_policy.py",
        "robocasa/recovery/recovery_rollout.py",
        "robocasa/recovery/subtask_eval.py",
        "robocasa/recovery/exact_replay/run_phase2_snapshot_replay.py",
        "robocasa/recovery/full_snapshot.py",
        "robocasa/recovery/exact_replay/counterfactual_branch.py",
        "robocasa/wrappers/gym_wrapper.py",
        "robocasa/utils/env_utils.py",
    )
    return {name: sha256_file(root / name) for name in names}


def _source(root):
    registration = json.loads((root / "registration.json").read_text())
    validate_registration_source(registration)
    engineering = json.loads((root / "analysis.json").read_text())
    science = json.loads((root / "scientific_analysis.json").read_text())
    if registration["scope"] != "full" or engineering.get("engineering_all_pass") is not True:
        raise ValueError("Operator screen requires the passing full Phase 3B engineering audit")
    if science.get("routing_decision") != "higher_level_operator_screen":
        raise ValueError("Phase 3B did not route to the higher-level operator screen")
    if science.get("engineering_analysis_sha256") != sha256_file(root / "analysis.json"):
        raise ValueError("Phase 3B science does not reference this engineering audit")
    if read_jsonl(root / "errors.jsonl"):
        raise ValueError("Phase 3B source has errors")
    source_records = load_branch_records(root / "branch_records.jsonl")
    expected = {x["branch_id"] for x in registration["registered_branches"]}
    if len(source_records) != len(expected) or {x["branch_id"] for x in source_records} != expected:
        raise ValueError("Phase 3B branch support is incomplete or duplicated")
    for row in source_records:
        if not row.get("payload_file_sha256") or sha256_file(root / row["payload_path"]) != row["payload_file_sha256"]:
            raise ValueError(f"Phase 3B payload hash mismatch: {row['branch_id']}")
    return registration


def build_registration(source_dir, *, scope, spec=None, sentinel_dir=None, oracle_anchors=None):
    spec = deepcopy(DEFAULT_SPEC if spec is None else spec)
    validate_spec(spec)
    root = Path(source_dir).resolve()
    source = _source(root)
    phase2 = Path(source["phase2_run_dir"])
    plan = json.loads((phase2 / "plan.json").read_text())
    all_snapshots = deterministic_snapshot_selection(
        plan, read_jsonl(phase2 / "parent_records.jsonl"), scope="full"
    )
    index = {x["snapshot_id"]: i for i, x in enumerate(all_snapshots)}
    selection = deterministic_snapshot_selection(
        plan, read_jsonl(phase2 / "parent_records.jsonl"), scope=scope
    )
    anchors = {}
    if oracle_anchors is not None:
        manifest = json.loads(Path(oracle_anchors).read_text())
        if set(manifest) != set(index):
            raise ValueError("Oracle anchors must cover all 20 target snapshots exactly")
        for selected in all_snapshots:
            target_id = selected["snapshot_id"]
            entry = manifest[target_id]
            path = Path(entry["path"])
            if not path.is_absolute():
                raise ValueError("Oracle anchor paths must be absolute")
            anchor, _ = load_registered_phase2_snapshot(
                path, expected_file_sha256=entry["sha256"],
                expected_snapshot_id=entry["snapshot_id"],
                expected_parent_id=selected["parent_id"],
                expected_task_name=selected["task_name"],
            )
            target, _ = load_registered_phase2_snapshot(
                phase2 / "snapshots" / f"{target_id}.pkl.gz",
                expected_file_sha256=source["phase2_snapshot_file_sha256"][target_id],
                expected_snapshot_id=target_id, expected_parent_id=selected["parent_id"],
                expected_task_name=selected["task_name"],
            )
            if anchor.environment_step >= target.environment_step:
                raise ValueError("An oracle anchor must precede its target snapshot")
            if anchor.metadata.get("environment_seed") != target.metadata.get("environment_seed"):
                raise ValueError("Oracle anchor belongs to a different environment identity")
            if anchor.environment_state.get("model") != target.environment_state.get("model"):
                raise ValueError("Oracle anchor and target have different model XML")
            if stable_digest(anchor.policy_state.get("structural_config")) != stable_digest(target.policy_state.get("structural_config")):
                raise ValueError("Oracle anchor and target have different policy structure")
            predicate = source["trigger_stages"][selected["task_name"]].split("::", 1)[-1]
            anchor_predicate = (anchor.subtask_eval.get("predicates") or {}).get(predicate)
            if not isinstance(anchor_predicate, dict) or anchor_predicate.get("value"):
                raise ValueError("Oracle anchor must have the same evaluable, incomplete target predicate")
            anchors[target_id] = {**entry, "path": str(path.resolve())}
    sentinel = None
    if scope == "full":
        if sentinel_dir is None:
            raise ValueError("Full screen requires --sentinel-run-dir")
        sroot = Path(sentinel_dir).resolve()
        sreg = json.loads((sroot / "registration.json").read_text())
        validate_registration(sreg)
        from robocasa.recovery.exact_replay.run_operator_screen import engineering_analysis
        audit = engineering_analysis(sroot, sreg)
        if sreg["scope"] != "sentinel" or not audit["engineering_all_pass"]:
            raise ValueError("Full screen blocked by the engineering sentinel")
        if sreg["spec"] != spec or sreg["phase3b_run_dir"] != str(root) or sreg["oracle_anchors"] != anchors:
            raise ValueError("Sentinel/full source or operator specification differs")
        sentinel = {"run_dir": str(sroot), "registration_sha256": sha256_file(sroot / "registration.json"),
                    "ledger_sha256": sha256_file(sroot / "branch_records.jsonl")}
    branches = []
    for selected in selection:
        snapshot_id = selected["snapshot_id"]
        for operator in OPERATORS + ((ORACLE,) if anchors else ()):
            for schedule in range(1 if operator == ORACLE else 2):
                seed = spec["continuation_seed_bases"][schedule] + index[snapshot_id] * spec["seed_stride_per_snapshot"]
                # Sentinel duplicates every arm in schedule 0 as an integrity
                # check. They are excluded from scientific estimates.
                repeats = 2 if scope == "sentinel" and schedule == 0 else 1
                for repeat in range(repeats):
                    branches.append({
                        **selected, "operator": operator, "schedule": schedule,
                        "repeat": repeat, "role": "repeat" if repeat else "primary",
                        "sampling_seed": seed,
                        "branch_id": f"{snapshot_id}--{operator}--s{schedule}--r{repeat}",
                    })
    hashes = {name: sha256_file(root / name) for name in
              ("registration.json", "analysis.json", "scientific_analysis.json", "branch_records.jsonl")}
    result = {
        "protocol": PROTOCOL, "scope": scope, "created_at": datetime.now(timezone.utc).isoformat(),
        "phase3b_run_dir": str(root), "phase3b_hashes": hashes,
        "phase3b_payload_hashes": {row["payload_path"]: row["payload_file_sha256"]
                                  for row in load_branch_records(root / "branch_records.jsonl")},
        "phase2_run_dir": str(phase2), "phase2_registration": source,
        "spec": spec, "oracle_anchors": anchors, "sentinel": sentinel,
        "selected_snapshots": selection, "branches": branches,
        "expected_branches": len(branches), "code_hashes": code_hashes(),
        "scientific_outcomes_used_as_sentinel_gate": False,
    }
    result["registration_sha256"] = sha256_json(result)
    return result


def validate_registration(registration):
    unsigned = dict(registration)
    digest = unsigned.pop("registration_sha256", None)
    if registration.get("protocol") != PROTOCOL or digest != sha256_json(unsigned):
        raise ValueError("Operator registration checksum/protocol mismatch")
    validate_spec(registration["spec"])
    if code_hashes() != registration["code_hashes"]:
        raise ValueError("Operator execution/analysis code changed after registration")
    root = Path(registration["phase3b_run_dir"])
    for name, digest in registration["phase3b_hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Frozen Phase 3B source changed: {name}")
    for name, digest in registration["phase3b_payload_hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Frozen Phase 3B payload changed: {name}")
    validate_registration_source(registration["phase2_registration"])
    for entry in registration["oracle_anchors"].values():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError("Oracle anchor changed")
    if registration["sentinel"]:
        sentinel = registration["sentinel"]
        sroot = Path(sentinel["run_dir"])
        if sha256_file(sroot / "registration.json") != sentinel["registration_sha256"] or sha256_file(sroot / "branch_records.jsonl") != sentinel["ledger_sha256"]:
            raise ValueError("Frozen engineering sentinel changed")


def clear_history(policy, *, instruction=None):
    """Reset temporal context without changing any random schedule or counter."""
    state = deepcopy(policy.get_state())
    state["action_plan"] = []
    state["image_queues"] = {key: [] for key in state["image_queues"]}
    state["state_queue"] = []
    state["last_instruction"] = instruction
    state["at_inference_boundary"] = True
    state["latest_observation_state_history"] = None
    policy.set_state(state)


def validate_retreat_controller(env):
    """A normalized lift is admissible only for a known delta-pose arm."""
    from robocasa.recovery.recovery_rollout import _sim_env
    target = _sim_env(env)
    robots = getattr(target, "robots", [])
    if len(robots) != 1:
        raise ValueError("Retreat requires exactly one mobile manipulation robot")
    composite = getattr(robots[0], "composite_controller", None)
    controller = getattr(composite, "part_controllers", {}).get("right")
    input_type = getattr(controller, "input_type", None)
    if input_type is None and getattr(controller, "control_delta", None) is True:
        input_type = "delta"
    frame = getattr(controller, "input_ref_frame", None)
    if input_type != "delta" or getattr(controller, "control_dim", None) != 6 or frame not in {"base", "world"}:
        raise ValueError("Retreat requires a 6D delta-pose right-arm controller in base/world frame; inspect the controller before registering a different maneuver")
    return {"accepted": True, "controller_class": type(controller).__name__,
            "input_type": input_type, "control_dim": 6, "input_ref_frame": frame}


class OperatorPolicy:
    """Delay mutation until after the existing exact baseline restore audit."""

    def __init__(self, policy, branch, spec):
        self.policy = policy
        self.branch = branch
        self.spec = spec
        self.steps = 0
        self.request_index = 0
        self.events = []

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __call__(self, observation, instruction=None):
        operator = self.branch["operator"]
        original = instruction or observation["annotation.human.task_description"]
        prompt = original
        if operator == "stage_retry" and self.steps < self.spec["retry_steps"]:
            prompt = self.spec["subgoal_prompts"][self.branch["task_name"]]
        if self.steps == 0 and operator in {"history_reset_replan", "stage_retry"}:
            clear_history(self.policy, instruction=prompt)
            self.events.append({"step": self.steps, "event": "history_reset", "instruction": prompt})
        if operator == "stage_retry" and self.steps == self.spec["retry_steps"]:
            clear_history(self.policy, instruction=original)
            self.events.append({"step": self.steps, "event": "original_instruction_restored"})
        retreat = self.spec["retreat"]
        retreat_steps = retreat["release_steps"] + retreat["lift_steps"]
        if operator == "retreat_reset_replan" and self.steps < retreat_steps:
            lift = retreat["lift_command"] if self.steps >= retreat["release_steps"] else 0.0
            action = {
                "action.end_effector_position": np.asarray([0.0, 0.0, lift], dtype=np.float32),
                "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
                "action.gripper_close": np.asarray([0.0], dtype=np.float32),
                "action.base_motion": np.zeros(4, dtype=np.float32),
                "action.control_mode": np.asarray([retreat["control_mode"]], dtype=np.float32),
            }
            if self.steps == 0:
                self.events.append({"step": 0, "event": "scripted_release_then_lift", **retreat})
            state = deepcopy(self.policy.get_state())
            state["environment_step"] += 1
            self.policy.set_state(state)
        else:
            if operator == "retreat_reset_replan" and self.steps == retreat_steps:
                clear_history(self.policy, instruction=original)
                self.events.append({"step": self.steps, "event": "history_reset_after_retreat"})
            if self.policy.at_inference_boundary:
                self.policy.set_next_sampling_seed(self.branch["sampling_seed"] + self.request_index)
                self.request_index += 1
            action = self.policy(observation, instruction=prompt)
        self.steps += 1
        return action


def repeat_channels(record):
    return {key: record.get(key) for key in (
        "suffix_request_sha256", "suffix_action_sha256", "suffix_observation_sha256",
        "suffix_environment_sha256", "termination_reason", "num_steps", "task_success",
        "regressed_predicates", "semantic_sha256", "request_sampling_seeds", "operator_events",
    )}


def unique_records(root, registration):
    records = load_branch_records(Path(root) / "branch_records.jsonl")
    expected = {x["branch_id"]: x for x in registration["branches"]}
    found = {}
    for record in records:
        branch_id = record["branch_id"]
        if branch_id not in expected or branch_id in found:
            raise ValueError("Unknown or duplicate branch ID in operator ledger")
        for key in ("operator", "schedule", "repeat", "role", "sampling_seed", "snapshot_id", "parent_id", "task_name", "boundary"):
            if record.get(key) != expected[branch_id][key]:
                raise ValueError(f"Branch registration mismatch: {branch_id}: {key}")
        payload = load_branch_payload(root, record)
        embedded = payload["summary"]
        for key, value in record.items():
            if key not in {"payload_path", "payload_file_sha256"} and stable_digest(embedded.get(key)) != stable_digest(value):
                raise ValueError(f"Branch ledger/payload mismatch: {branch_id}: {key}")
        found[branch_id] = record
    return found
