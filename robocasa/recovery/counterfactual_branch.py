"""Deterministic branch execution and Phase 2 replay validation.

This module deliberately separates branch validity from recovery quality.  A
branch is useful for later causal analysis only when the complete snapshot
restores, the declared request seed is observed, and same-seed repeats produce
the same request, first action, and first transition.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from robocasa.recovery.full_snapshot import (
    FullSnapshot,
    causal_transition_fingerprint,
    environment_fingerprint,
    restore_full_snapshot,
    restore_simulator_integration_state,
    stable_digest,
)


BRANCH_SCHEMA_VERSION = 8
BRANCH_PROTOCOL = "robocasa_exact_snapshot_counterfactual_branch"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def branch_payload_digest(payload):
    """Hash branch content without recursively hashing its embedded checksum.

    Phase 2 stores the checksum inside ``payload["summary"]`` for artifact
    self-description.  The checksum is defined over the payload before that
    field is inserted.  Temporarily removing only that field makes the digest
    reproducible after loading while leaving the in-memory payload unchanged.
    """
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict) or "payload_sha256" not in summary:
        return stable_digest(payload)
    embedded = summary.pop("payload_sha256")
    try:
        return stable_digest(payload)
    finally:
        summary["payload_sha256"] = embedded


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    kind: str
    sampling_seed: int
    suffix_steps: int
    repeat_index: int | None = None

    def validate(self):
        if self.kind not in {"same_seed_repeat", "candidate", "environment_only"}:
            raise ValueError(f"Unsupported branch kind: {self.kind}")
        if int(self.suffix_steps) < 1:
            raise ValueError("suffix_steps must be positive")


def _jsonable(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().numpy())
    return repr(value)


def _environment_only_restore(snapshot, env):
    from robocasa.recovery.recovery_rollout import _reset_to_state

    _reset_to_state(env, deepcopy(snapshot.environment_state))
    restore_simulator_integration_state(env, snapshot.simulator_integration_state)
    return {
        "snapshot_id": snapshot.snapshot_id,
        "valid": False,
        "environment_exact": True,
        "policy_exact": False,
        "observation_exact": False,
        "negative_control": True,
        "reason": "policy_state_intentionally_not_restored",
    }


def observation_component_sha256(observation):
    """Hash each top-level observation field for bounded replay diagnostics."""
    if not isinstance(observation, dict):
        return {"<root>": stable_digest(observation)}
    return {
        str(key): stable_digest(value)
        for key, value in sorted(observation.items(), key=lambda item: str(item[0]))
    }


def run_counterfactual_branch(
    snapshot: FullSnapshot,
    env,
    policy,
    spec: BranchSpec,
    *,
    instruction=None,
    tracker=None,
    step_fn=None,
    success_fn=None,
    subtask_eval_fn=None,
    restore_atol=1e-8,
    restore_rtol=1e-8,
    verify_restored_observation=True,
):
    """Restore one branch point and execute a declared finite suffix."""
    from robocasa.recovery.recovery_rollout import (
        _is_task_success,
        _step_env,
        call_policy,
    )
    from robocasa.recovery.subtask_eval import (
        build_subtask_trace,
        get_subtask_eval,
    )

    spec.validate()
    snapshot.validate()
    step_fn = step_fn or _step_env
    success_fn = success_fn or _is_task_success
    subtask_eval_fn = subtask_eval_fn or get_subtask_eval
    if spec.kind == "environment_only":
        restore_audit = _environment_only_restore(snapshot, env)
    else:
        restore_audit = restore_full_snapshot(
            snapshot,
            env,
            policy,
            tracker=tracker,
            atol=restore_atol,
            rtol=restore_rtol,
            verify_observation=verify_restored_observation,
        )
        if not restore_audit["valid"]:
            raise RuntimeError(
                f"Complete snapshot restore failed for {snapshot.snapshot_id}: "
                f"{restore_audit}"
            )

    restored_subtask_eval = deepcopy(subtask_eval_fn(env))
    restore_subtask_exact = stable_digest(restored_subtask_eval) == stable_digest(
        snapshot.subtask_eval
    )
    restore_trace = build_subtask_trace(
        [deepcopy(snapshot.subtask_eval), restored_subtask_eval]
    )
    restore_regressions = sorted(
        {
            name
            for entry in restore_trace
            for name in entry.get("regressed_predicates", [])
        }
    )

    boundary = getattr(policy, "at_inference_boundary", None)
    if spec.kind != "environment_only" and not bool(boundary):
        raise RuntimeError("Restored policy is not at an inference boundary")
    seed_setter = getattr(policy, "set_next_sampling_seed", None)
    if not callable(seed_setter):
        raise TypeError("Policy must expose set_next_sampling_seed()")
    if spec.kind != "environment_only":
        seed_setter(spec.sampling_seed)

    obs = deepcopy(snapshot.observation)
    subtask_evals = [deepcopy(snapshot.subtask_eval)]
    action_payloads = []
    action_sha256 = []
    observation_sha256 = [stable_digest(obs)]
    observation_component_digests = [observation_component_sha256(obs)]
    repeat_observations = (
        [deepcopy(obs)] if spec.kind == "same_seed_repeat" else None
    )
    environment_sha256 = []
    diagnostic_environment_sha256 = []
    request_records = []
    request_environment_step_indices = []
    inference_records = []
    inference_environment_step_indices = []
    rewards = []
    infos = []
    termination_reason = "suffix_horizon"
    first_transition_sha256 = None
    first_diagnostic_transition_sha256 = None
    first_transition_component_sha256 = None
    first_diagnostic_transition_component_sha256 = None
    first_transition_fingerprint = None
    first_causal_transition_fingerprint = None
    first_action_sha256 = None
    first_action = None

    for step_index in range(int(spec.suffix_steps)):
        action = call_policy(policy, obs, instruction=instruction)
        if step_index == 0:
            first_action = deepcopy(action)
            first_action_sha256 = stable_digest(action)
        action_payloads.append(deepcopy(action))
        action_sha256.append(stable_digest(action))

        pop_requests = getattr(policy, "pop_request_records", None)
        current_requests = pop_requests() if callable(pop_requests) else []
        request_records.extend(deepcopy(current_requests))
        request_environment_step_indices.extend(
            [int(step_index)] * len(current_requests)
        )
        pop_inference = getattr(policy, "pop_inference_record", None)
        inference_record = pop_inference() if callable(pop_inference) else None
        if inference_record is not None:
            inference_records.append(deepcopy(inference_record))
            inference_environment_step_indices.append(int(step_index))

        obs, reward, done, info = step_fn(env, action)
        rewards.append(float(reward) if reward is not None else None)
        infos.append(deepcopy(info))
        observation_sha256.append(stable_digest(obs))
        observation_component_digests.append(observation_component_sha256(obs))
        if repeat_observations is not None:
            repeat_observations.append(deepcopy(obs))
        transition_fingerprint = environment_fingerprint(env)
        causal_fingerprint = causal_transition_fingerprint(transition_fingerprint)
        transition_sha = stable_digest(causal_fingerprint)
        diagnostic_transition_sha = stable_digest(transition_fingerprint)
        environment_sha256.append(transition_sha)
        diagnostic_environment_sha256.append(diagnostic_transition_sha)
        if first_transition_sha256 is None:
            first_transition_sha256 = transition_sha
            first_diagnostic_transition_sha256 = diagnostic_transition_sha
            first_transition_fingerprint = deepcopy(transition_fingerprint)
            first_causal_transition_fingerprint = deepcopy(causal_fingerprint)
            first_transition_component_sha256 = {
                name: stable_digest(value) for name, value in causal_fingerprint.items()
            }
            first_diagnostic_transition_component_sha256 = {
                name: stable_digest(value)
                for name, value in transition_fingerprint.items()
            }
        subtask_evals.append(deepcopy(subtask_eval_fn(env)))
        if success_fn(info=info, reward=reward, env=env):
            termination_reason = "success"
            break
        if done:
            declared_reason = (
                info.get("termination_reason") if isinstance(info, dict) else None
            )
            explicitly_unsafe = isinstance(info, dict) and bool(
                info.get("safety_termination") or info.get("safety_violation")
            )
            termination_reason = (
                "safety_termination"
                if explicitly_unsafe
                or "safety" in str(declared_reason or "").lower()
                else "environment_done"
            )
            break

    if spec.kind != "environment_only":
        if not request_records:
            raise RuntimeError("Branch produced no true policy request")
        first_request = request_records[0]
        if first_request.get("sampling_seed") != int(spec.sampling_seed):
            raise RuntimeError(
                "Branch request seed does not match declaration: "
                f"declared={spec.sampling_seed} observed="
                f"{first_request.get('sampling_seed')}"
            )

    trace = build_subtask_trace(subtask_evals)
    regressions = sorted(
        {name for entry in trace for name in entry.get("regressed_predicates", [])}
    )
    final_entry = trace[-1] if trace else {}
    first_inference_actions_sha256 = None
    if inference_records:
        first_actions = inference_records[0].get("actions")
        if first_actions is not None:
            first_inference_actions_sha256 = stable_digest(first_actions)
    alignment_pairs_exact = bool(request_records) and len(request_records) == len(
        inference_records
    )
    if alignment_pairs_exact:
        alignment_pairs_exact = all(
            request.get("sampling_seed") == inference.get("sampling_seed")
            and request.get("request_sha256") == inference.get("request_sha256")
            for request, inference in zip(request_records, inference_records)
        )
    summary = {
        "schema_version": BRANCH_SCHEMA_VERSION,
        "protocol": BRANCH_PROTOCOL,
        "created_at": utc_now(),
        "snapshot_id": snapshot.snapshot_id,
        "parent_id": snapshot.parent_id,
        "task_name": snapshot.task_name,
        "trigger_name": snapshot.trigger_name,
        **asdict(spec),
        "restore_valid": bool(restore_audit.get("valid")),
        "restore_audit": _jsonable(restore_audit),
        "restore_subtask_exact": bool(restore_subtask_exact),
        "restore_induced_regressed_predicates": restore_regressions,
        "first_request_sha256": (
            request_records[0]["request_sha256"] if request_records else None
        ),
        "first_request_sampling_seed": (
            request_records[0].get("sampling_seed") if request_records else None
        ),
        "first_action_sha256": first_action_sha256,
        "first_inference_actions_sha256": first_inference_actions_sha256,
        "first_transition_sha256": first_transition_sha256,
        "first_diagnostic_transition_sha256": first_diagnostic_transition_sha256,
        "first_transition_component_sha256": first_transition_component_sha256,
        "first_diagnostic_transition_component_sha256": (
            first_diagnostic_transition_component_sha256
        ),
        "request_sampling_seeds": [
            record.get("sampling_seed") for record in request_records
        ],
        "request_environment_step_indices": request_environment_step_indices,
        "inference_environment_step_indices": inference_environment_step_indices,
        "suffix_request_sha256": [
            stable_digest(record) for record in request_records
        ],
        "common_randomness_contract": {
            "identical_complete_state_before_first_request": (
                spec.kind != "environment_only"
            ),
            "declared_treatment_request_index": (
                0 if spec.kind == "candidate" else None
            ),
            "state_divergence_can_begin_after_action_index": (
                0 if spec.kind == "candidate" else None
            ),
            "post_treatment_policy_seed_schedule_recorded": True,
        },
        "suffix_action_sha256": action_sha256,
        "suffix_observation_sha256": observation_sha256,
        "suffix_observation_component_sha256": observation_component_digests,
        "suffix_environment_sha256": environment_sha256,
        "suffix_diagnostic_environment_sha256": diagnostic_environment_sha256,
        "num_steps": len(action_payloads),
        "num_policy_requests": len(request_records),
        "num_inference_records": len(inference_records),
        "termination_reason": termination_reason,
        "task_success": termination_reason == "success",
        "ordered_completed_subtasks": final_entry.get("ordered_completed_subtasks", []),
        "ordered_current_subtask": final_entry.get("ordered_current_subtask"),
        "ordered_subtask_progress": final_entry.get("ordered_subtask_progress", 0.0),
        "regressed_predicates": regressions,
        "record_alignment_valid": bool(
            alignment_pairs_exact
            and first_action_sha256 is not None
            and first_inference_actions_sha256 is not None
            and first_transition_sha256 is not None
        ),
    }
    payload = {
        "summary": summary,
        "first_action": first_action,
        "actions": action_payloads,
        "request_records": request_records,
        "inference_records": inference_records,
        "subtask_evals": subtask_evals,
        "subtask_trace": trace,
        "rewards": rewards,
        "infos": infos,
        "first_transition_fingerprint": first_transition_fingerprint,
        "first_causal_transition_fingerprint": first_causal_transition_fingerprint,
    }
    if repeat_observations is not None:
        payload["observations"] = repeat_observations
    summary["payload_sha256"] = branch_payload_digest(payload)
    return {"summary": summary, "payload": payload}


def save_branch_result(result, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = deepcopy(result["summary"])
    payload = result["payload"]
    branch_id = summary["branch_id"]
    payload_path = output_dir / "branches" / f"{branch_id}.pkl.gz"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = payload_path.with_suffix(payload_path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    os.replace(temporary, payload_path)
    summary["payload_path"] = str(payload_path.relative_to(output_dir))
    summary["payload_file_sha256"] = _sha256_file(payload_path)
    summary_path = output_dir / "branch_records.jsonl"
    line = (json.dumps(_jsonable(summary), sort_keys=True) + "\n").encode()
    descriptor = os.open(summary_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return summary


def load_branch_records(path: str | Path):
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid branch JSONL line {line_number}: {error}"
                ) from error
    return records


def analyze_phase2_replay(records, errors=None):
    """Evaluate only the predeclared engineering validity gates."""
    errors = list(errors or [])
    primary = [row for row in records if row["kind"] != "environment_only"]
    repeats = defaultdict(list)
    candidates = defaultdict(list)
    for row in primary:
        if row["kind"] == "same_seed_repeat":
            repeats[(row["snapshot_id"], int(row["sampling_seed"]))].append(row)
        elif row["kind"] == "candidate":
            candidates[row["snapshot_id"]].append(row)

    repeat_pairs = []
    for key, rows in sorted(repeats.items()):
        if len(rows) != 2:
            repeat_pairs.append(
                {
                    "snapshot_id": key[0],
                    "sampling_seed": key[1],
                    "valid": False,
                    "reason": "repeat_count",
                }
            )
            continue
        left, right = rows
        left_components = left.get("first_transition_component_sha256") or {}
        right_components = right.get("first_transition_component_sha256") or {}
        left_diagnostic_components = (
            left.get("first_diagnostic_transition_component_sha256") or {}
        )
        right_diagnostic_components = (
            right.get("first_diagnostic_transition_component_sha256") or {}
        )
        left_observation_components = left.get(
            "suffix_observation_component_sha256", []
        )
        right_observation_components = right.get(
            "suffix_observation_component_sha256", []
        )
        observation_component_mismatches = []
        for step_index in range(
            max(len(left_observation_components), len(right_observation_components))
        ):
            left_step = (
                left_observation_components[step_index]
                if step_index < len(left_observation_components)
                else {}
            )
            right_step = (
                right_observation_components[step_index]
                if step_index < len(right_observation_components)
                else {}
            )
            paths = sorted(
                name
                for name in set(left_step) | set(right_step)
                if left_step.get(name) != right_step.get(name)
            )
            if paths:
                observation_component_mismatches.append(
                    {"step_index": step_index, "paths": paths}
                )
        left_requests = left.get("suffix_request_sha256", [])
        right_requests = right.get("suffix_request_sha256", [])
        request_mismatch_indices = [
            request_index
            for request_index in range(max(len(left_requests), len(right_requests)))
            if request_index >= len(left_requests)
            or request_index >= len(right_requests)
            or left_requests[request_index] != right_requests[request_index]
        ]
        repeat_pairs.append(
            {
                "snapshot_id": key[0],
                "sampling_seed": key[1],
                "valid": True,
                "request_exact": left["first_request_sha256"]
                == right["first_request_sha256"],
                "request_sequence_exact": not request_mismatch_indices,
                "request_mismatch_indices": request_mismatch_indices,
                "action_exact": (
                    left["first_action_sha256"] == right["first_action_sha256"]
                    and left.get("first_inference_actions_sha256")
                    == right.get("first_inference_actions_sha256")
                ),
                "transition_exact": left["first_transition_sha256"]
                == right["first_transition_sha256"],
                "diagnostic_transition_exact": (
                    left.get("first_diagnostic_transition_sha256")
                    == right.get("first_diagnostic_transition_sha256")
                ),
                "transition_components_exact": {
                    name: left_components.get(name) == right_components.get(name)
                    for name in sorted(set(left_components) | set(right_components))
                },
                "diagnostic_transition_components_exact": {
                    name: left_diagnostic_components.get(name)
                    == right_diagnostic_components.get(name)
                    for name in sorted(
                        set(left_diagnostic_components)
                        | set(right_diagnostic_components)
                    )
                },
                "observation_exact": (
                    left.get("suffix_observation_sha256")
                    == right.get("suffix_observation_sha256")
                ),
                "observation_component_mismatches": (
                    observation_component_mismatches
                ),
                "action_sequence_exact": (
                    left.get("suffix_action_sha256")
                    == right.get("suffix_action_sha256")
                ),
                "causal_environment_sequence_exact": (
                    left.get("suffix_environment_sha256")
                    == right.get("suffix_environment_sha256")
                ),
                "diagnostic_environment_sequence_exact": (
                    left.get("suffix_diagnostic_environment_sha256")
                    == right.get("suffix_diagnostic_environment_sha256")
                ),
                "suffix_outcome_equal": (
                    left["task_success"] == right["task_success"]
                    and left["ordered_completed_subtasks"]
                    == right["ordered_completed_subtasks"]
                ),
            }
        )
    valid_pairs = [row for row in repeat_pairs if row.get("valid")]
    suffix_agreement = (
        float(np.mean([row["suffix_outcome_equal"] for row in valid_pairs]))
        if valid_pairs
        else 0.0
    )
    candidate_diversity = []
    for snapshot_id, rows in sorted(candidates.items()):
        hashes = {row.get("first_inference_actions_sha256") for row in rows}
        candidate_diversity.append(
            {
                "snapshot_id": snapshot_id,
                "candidate_count": len(rows),
                "unique_first_actions": len(hashes),
                "diverse": len(hashes) > 1,
            }
        )
    diversity_rate = (
        float(np.mean([row["diverse"] for row in candidate_diversity]))
        if candidate_diversity
        else 0.0
    )
    total_attempts = len(primary) + len(errors)
    error_rate = len(errors) / max(1, total_attempts)
    gates = {
        "all_complete_restores_valid": bool(primary)
        and all(row["restore_valid"] for row in primary),
        "same_seed_requests_exact": bool(valid_pairs)
        and all(row["request_exact"] for row in valid_pairs),
        "same_seed_request_sequences_exact": bool(valid_pairs)
        and all(row["request_sequence_exact"] for row in valid_pairs),
        "same_seed_actions_exact": bool(valid_pairs)
        and all(row["action_exact"] for row in valid_pairs),
        "same_seed_first_transitions_exact": bool(valid_pairs)
        and all(row["transition_exact"] for row in valid_pairs),
        "same_seed_observations_exact": bool(valid_pairs)
        and all(row["observation_exact"] for row in valid_pairs),
        "same_seed_action_sequences_exact": bool(valid_pairs)
        and all(row["action_sequence_exact"] for row in valid_pairs),
        "same_seed_causal_environment_sequences_exact": bool(valid_pairs)
        and all(row["causal_environment_sequence_exact"] for row in valid_pairs),
        "suffix_outcome_agreement_at_least_0p95": suffix_agreement >= 0.95,
        "candidate_diversity_at_least_0p90": diversity_rate >= 0.90,
        "record_alignment_exact": bool(primary)
        and all(row["record_alignment_valid"] for row in primary),
        "zero_restore_induced_regressions": bool(primary)
        and all(row.get("restore_subtask_exact", False) for row in primary)
        and not any(
            row.get("restore_induced_regressed_predicates", []) for row in primary
        ),
        "branch_error_rate_below_0p02": error_rate < 0.02,
    }
    return {
        "schema_version": BRANCH_SCHEMA_VERSION,
        "protocol": "phase2_complete_snapshot_replay_audit",
        "status": "complete",
        "records": len(records),
        "primary_records": len(primary),
        "errors": len(errors),
        "error_rate": error_rate,
        "repeat_pairs": repeat_pairs,
        "same_seed_suffix_outcome_agreement": suffix_agreement,
        "candidate_diversity": candidate_diversity,
        "candidate_diversity_rate": diversity_rate,
        "gates": gates,
        "all_pass": all(gates.values()),
    }


__all__ = [
    "BRANCH_PROTOCOL",
    "BRANCH_SCHEMA_VERSION",
    "BranchSpec",
    "analyze_phase2_replay",
    "branch_payload_digest",
    "load_branch_records",
    "run_counterfactual_branch",
    "save_branch_result",
]
