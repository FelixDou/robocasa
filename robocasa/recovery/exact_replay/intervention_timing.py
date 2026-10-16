"""Frozen common-entry timing protocol; existing recovery studies stay immutable."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import gzip
import json
import pickle

import numpy as np

from robocasa.recovery.exact_replay.counterfactual_branch import branch_payload_digest, load_branch_records
from robocasa.recovery.full_snapshot import stable_digest, load_full_snapshot
from robocasa.recovery.exact_replay.operator_screen import (
    OPERATORS, OperatorPolicy, validate_registration as validate_operator_registration,
    code_hashes as operator_code_hashes,
)
from robocasa.recovery.exact_replay.phase3b_training_horizon import (
    read_jsonl, sha256_file, sha256_json, load_branch_payload,
)
from robocasa.recovery.exact_replay.run_phase2_snapshot_replay import utc_now, append_jsonl


PROTOCOL = "xr1_common_entry_intervention_timing_v1"
DEFAULT_SPEC = {
    "delays": [0, 1, 2],
    "continuation_seed_bases": [91000000, 92000000],
    "seed_stride_per_parent": 1000,
    "horizons": {"ArrangeTea": 480, "CuttingToolSelection": 256},
    "endpoint": "absolute_environment_steps_from_common_stage_entry",
    "ineligible_fallback": "continue_without_intervention_keep_assignment",
    "subgoal_source": "fixed_supervisor_supplied_no_online_stage_claim",
    "environment_rng_replay": "distinct_pcg_seed_streams_reconstructed_before_nominal_replay_only_if_reconstructed_streams_remain_idle",
}


def validate_spec(spec):
    if set(spec) != set(DEFAULT_SPEC):
        raise ValueError("Missing or unknown timing specification fields")
    if spec["delays"] != [0, 1, 2] or any(type(x) is not int for x in spec["delays"]):
        raise ValueError("Timings must be zero, one, and two genuine inferences")
    if spec["horizons"] != DEFAULT_SPEC["horizons"]:
        raise ValueError("Common entry endpoints must remain 480/256 environment steps")
    seeds, stride = spec["continuation_seed_bases"], spec["seed_stride_per_parent"]
    if (len(seeds) != 2 or any(type(x) is not int or x < 0 for x in seeds)
            or type(stride) is not int or stride < 480 or abs(seeds[1]-seeds[0]) < 10*stride):
        raise ValueError("Require disjoint bounded per-parent request seed schedules")
    for key in ("endpoint", "ineligible_fallback", "subgoal_source", "environment_rng_replay"):
        if spec[key] != DEFAULT_SPEC[key]:
            raise ValueError(f"Unsupported timing contract: {key}")


def code_hashes():
    root = Path(__file__).resolve().parents[3]
    names = (
        "robocasa/recovery/exact_replay/intervention_timing.py",
        "robocasa/recovery/exact_replay/run_intervention_timing.py",
        "robocasa/recovery/exact_replay/analyze_intervention_timing.py",
        "robocasa/recovery/exact_replay/run_operator_screen.py",
        "robocasa/recovery/exact_replay/counterfactual_branch.py",
        "robocasa/recovery/exact_replay/run_phase3b_training_horizon_continuation.py",
        "robocasa/recovery/exact_replay/phase3b_training_horizon.py",
        "scripts/exact_replay/run_xr1_intervention_timing.sh",
    )
    return {**operator_code_hashes(), **{name: sha256_file(root/name) for name in names}}


def allocate(parents, spec):
    """200 primary regimes and 20 sentinel repeats, with stable global IDs."""
    branches = []
    for parent in parents:
        regimes = [("continue", None)] + [(op, delay) for op in OPERATORS[1:] for delay in spec["delays"]]
        for schedule in (0, 1):
            for operator, delay in regimes:
                for repeat in range(2 if parent["sentinel"] and schedule == 0 else 1):
                    cell = "continue" if delay is None else f"{operator}--d{delay}"
                    branches.append({
                        "branch_id": f"{parent['parent_id']}--{cell}--s{schedule}--r{repeat}",
                        "parent_id": parent["parent_id"], "task_name": parent["task_name"],
                        "operator": operator, "delay": delay, "schedule": schedule,
                        "sampling_seed": spec["continuation_seed_bases"][schedule] + parent["index"]*spec["seed_stride_per_parent"],
                        "repeat": repeat, "role": "repeat" if repeat else "primary",
                        "sentinel": parent["sentinel"],
                        "horizon_environment_steps": spec["horizons"][parent["task_name"]],
                    })
    return branches


def build_registration(operator_run_dir, spec=None):
    root = Path(operator_run_dir).resolve()
    spec = deepcopy(DEFAULT_SPEC if spec is None else spec)
    validate_spec(spec)
    original = json.loads((root/"registration.json").read_text())
    validate_operator_registration(original)
    engineering = json.loads((root/"analysis.json").read_text())
    science = json.loads((root/"scientific_analysis.json").read_text())
    if (original["scope"] != "full" or engineering.get("engineering_all_pass") is not True
            or engineering.get("branches") != 160 or read_jsonl(root/"errors.jsonl")
            or science.get("decision") != "mechanism_redesign_or_earlier_snapshot_study"
            or science.get("source_ledger_sha256") != sha256_file(root/"branch_records.jsonl")):
        raise ValueError("Timing study requires the complete negative full operator screen")
    records = load_branch_records(root/"branch_records.jsonl")
    expected = {b["branch_id"] for b in original["branches"]}
    if len(records) != 160 or {b["branch_id"] for b in records} != expected:
        raise ValueError("Source operator support is incomplete or duplicated")
    payloads = {}
    for record in records:
        name, digest = record["payload_path"], record["payload_file_sha256"]
        if sha256_file(root/name) != digest:
            raise ValueError("Source operator payload changed")
        payloads[name] = digest
    selected = [s for s in original["selected_snapshots"] if s["boundary"] == "prefix"]
    if len(selected) != 10 or Counter(s["task_name"] for s in selected) != {"ArrangeTea": 5, "CuttingToolSelection": 5}:
        raise ValueError("Require all ten original prefix parents, five per task")
    if len({s["parent_id"] for s in selected}) != 10:
        raise ValueError("Duplicate parent in timing source")
    sentinel_ids = {
        min((s for s in selected if s["task_name"] == task), key=lambda s: (s["environment_reset_index"], s["parent_id"]))["parent_id"]
        for task in spec["horizons"]
    }
    parents = [{**s, "index": i, "sentinel": s["parent_id"] in sentinel_ids}
               for i, s in enumerate(sorted(selected, key=lambda s: (s["task_name"], s["environment_reset_index"], s["parent_id"])))]
    # Execute sentinel identities first without changing global seed indices.
    parents.sort(key=lambda p: (not p["sentinel"], p["index"]))
    result = {
        "protocol": PROTOCOL, "created_at": utc_now(), "operator_run_dir": str(root),
        "operator_registration": original,
        "source_hashes": {name: sha256_file(root/name) for name in
                          ("registration.json", "analysis.json", "scientific_analysis.json", "branch_records.jsonl")},
        "source_payload_hashes": payloads,
        "phase2_run_dir": original["phase2_run_dir"],
        "phase2_registration": original["phase2_registration"],
        "operator_spec": original["spec"], "spec": spec, "parents": parents,
        "branches": allocate(parents, spec), "code_hashes": code_hashes(),
        "expected_primary_branches": 200, "expected_total_branches": 220,
        "expected_sentinel_branches": 60,
        "scientific_outcomes_used_as_sentinel_gate": False,
    }
    result["registration_sha256"] = sha256_json(result)
    return result


def validate_registration(registration):
    unsigned = dict(registration)
    digest = unsigned.pop("registration_sha256", None)
    if registration.get("protocol") != PROTOCOL or sha256_json(unsigned) != digest:
        raise ValueError("Timing registration checksum/protocol mismatch")
    validate_spec(registration["spec"])
    if code_hashes() != registration["code_hashes"]:
        raise ValueError("Timing execution or analysis code changed after registration")
    if allocate(registration["parents"], registration["spec"]) != registration["branches"]:
        raise ValueError("Timing allocation changed")
    root = Path(registration["operator_run_dir"])
    for name, expected in {**registration["source_hashes"], **registration["source_payload_hashes"]}.items():
        if sha256_file(root/name) != expected:
            raise ValueError(f"Frozen timing source changed: {name}")
    validate_operator_registration(registration["operator_registration"])


def ordered_trace(evaluations, prefix=()):
    """Carry the actual upstream ordered prefix without modifying raw predicates."""
    from robocasa.recovery.subtask_eval import build_subtask_trace
    if not prefix:
        return build_subtask_trace(evaluations)
    seed = deepcopy(evaluations[0])
    names = seed.get("required_predicates", [])
    if names[:len(prefix)] != list(prefix):
        raise ValueError("Captured ordered prefix is not a required-stage prefix")
    for name in prefix:
        seed["predicates"][name]["value"] = True
    trace = build_subtask_trace([seed, *evaluations])[1:]
    trace[0]["regressed_predicates"] = []  # no synthetic transition is an outcome
    for i, entry in enumerate(trace):
        entry["step"] = i
    return trace


def target_value(evaluation, predicate):
    item = (evaluation.get("predicates") or {}).get(predicate)
    if not isinstance(item, dict) or not isinstance(item.get("value"), (bool, np.bool_)):
        raise ValueError(f"Missing or non-boolean target predicate: {predicate}")
    return bool(item["value"])


class TimingPolicy:
    """One frozen request-index schedule shared across operators and timings."""
    def __init__(self, policy, branch, operator_spec, predicate):
        self.policy, self.branch, self.operator_spec = policy, branch, operator_spec
        self.predicate = predicate
        self.requests = 0
        self.steps = 0
        self.treatment = None
        self.decision = None

    def decide(self, evaluation, trace_entry):
        if self.branch["operator"] == "continue" or self.decision is not None:
            return
        reason = None
        if target_value(evaluation, self.predicate):
            reason = "target_completed_before_application"
        elif trace_entry["ordered_current_subtask"] != self.predicate:
            reason = "target_stage_ineligible_before_application"
        if reason:
            self.decision = {"executed": False, "reason": reason, "environment_step": self.steps,
                             "pre_application_requests": self.requests}
        elif self.policy.at_inference_boundary and self.requests == self.branch["delay"]:
            self.decision = {"executed": True, "reason": "assigned_boundary", "environment_step": self.steps,
                             "pre_application_requests": self.requests}
            branch = {**self.branch, "sampling_seed": self.branch["sampling_seed"] + self.requests}
            self.treatment = OperatorPolicy(self.policy, branch, self.operator_spec)
        elif self.policy.at_inference_boundary and self.requests > self.branch["delay"]:
            raise ValueError("Missed assigned intervention boundary")

    def __call__(self, observation):
        if self.treatment is not None:
            action = self.treatment(observation)
            self.requests = self.decision["pre_application_requests"] + self.treatment.request_index
        else:
            if self.policy.at_inference_boundary:
                self.policy.set_next_sampling_seed(self.branch["sampling_seed"] + self.requests)
                self.requests += 1
            action = self.policy(observation)
        self.steps += 1
        return action

    def final_decision(self, termination):
        if self.branch["operator"] == "continue":
            return {"executed": False, "reason": "shared_continuation", "environment_step": None,
                    "pre_application_requests": None}
        return self.decision or {"executed": False, "reason": f"boundary_not_reached_{termination}",
                                 "environment_step": self.steps, "pre_application_requests": self.requests}


def unique_records(root, registration, *, recover_orphans=False):
    root = Path(root)
    expected = {b["branch_id"]: b for b in registration["branches"]}
    rows = load_branch_records(root/"branch_records.jsonl")
    known = {r["branch_id"] for r in rows}
    if recover_orphans:
        for bid in expected.keys()-known:
            path = root/"branches"/f"{bid}.pkl.gz"
            if path.exists():
                with gzip.open(path, "rb") as stream:
                    payload = pickle.load(stream)  # trusted frozen experiment only
                record = deepcopy(payload["summary"])
                if record.get("branch_id") != bid or branch_payload_digest(payload) != record.get("payload_sha256"):
                    raise ValueError("Orphan timing payload checksum mismatch")
                record.update(payload_path=str(path.relative_to(root)), payload_file_sha256=sha256_file(path))
                append_jsonl(root/"branch_records.jsonl", record)
                rows.append(record)
    result = {}
    for row in rows:
        bid = row["branch_id"]
        if bid not in expected or bid in result:
            raise ValueError("Unknown or duplicate timing branch")
        if any(row.get(key) != value for key, value in expected[bid].items()):
            raise ValueError("Timing branch differs from registration")
        payload = load_branch_payload(root, row)
        if branch_payload_digest(payload) != row["payload_sha256"]:
            raise ValueError("Timing payload content checksum mismatch")
        for key, value in row.items():
            if key not in {"payload_path", "payload_file_sha256"} and stable_digest(payload["summary"].get(key)) != stable_digest(value):
                raise ValueError("Timing ledger/payload mismatch")
        result[bid] = row
    return result


def base_records(root, registration):
    root = Path(root)
    rows = read_jsonl(root/"base_records.jsonl")
    expected = {p["parent_id"]: p for p in registration["parents"]}
    result = {}
    for row in rows:
        pid = row["parent_id"]
        if pid not in expected or pid in result or row.get("replay_audit", {}).get("all_exact") is not True:
            raise ValueError("Invalid/duplicate common-entry base record")
        if row.get("registration_sha256") != registration["registration_sha256"] or row["source_prefix_snapshot_id"] != expected[pid]["snapshot_id"]:
            raise ValueError("Common-entry source registration mismatch")
        marker = root/"bases"/pid/"committed.json"
        if json.loads(marker.read_text()) != row:
            raise ValueError("Common-entry commit marker/ledger mismatch")
        for field, digest in row["artifact_hashes"].items():
            if sha256_file(root/field) != digest:
                raise ValueError("Common-entry artifact changed")
        snapshot = load_full_snapshot(root/row["snapshot_path"])
        if (snapshot.parent_id != pid or snapshot.snapshot_id != row["snapshot_id"]
                or snapshot.task_name != expected[pid]["task_name"]
                or snapshot.metadata["ordered_prefix"] != row["ordered_prefix"]
                or snapshot.environment_step != row["entry_environment_step"]):
            raise ValueError("Common-entry snapshot identity mismatch")
        result[pid] = row
    return result


def prefix_audit(branch_payload, control_payload):
    decision = branch_payload["summary"]["intervention"]
    n = decision["environment_step"] if decision["executed"] else branch_payload["summary"]["num_steps"]
    a, b = branch_payload["summary"], control_payload["summary"]
    channels = {key: a[key][:n] == b[key][:n] for key in ("suffix_action_sha256", "suffix_environment_sha256")}
    channels["observations"] = a["suffix_observation_sha256"][:n+1] == b["suffix_observation_sha256"][:n+1]
    for key in ("suffix_policy_state_sha256", "suffix_global_rng_sha256"):
        channels[key] = a[key][:n+1] == b[key][:n+1]
    for key in ("request_records", "inference_records"):
        steps = "request_environment_step_indices" if key == "request_records" else "inference_environment_step_indices"
        left = [v for v, step in zip(branch_payload[key], a[steps]) if step < n]
        right = [v for v, step in zip(control_payload[key], b[steps]) if step < n]
        channels[key] = stable_digest(left) == stable_digest(right)
    channels["semantics"] = stable_digest(branch_payload["subtask_evals"][:n+1]) == stable_digest(control_payload["subtask_evals"][:n+1])
    if not decision["executed"]:
        channels["unchanged_noop_endpoint"] = all(a[k] == b[k] for k in ("num_steps", "termination_reason", "task_success"))
    return {"pre_intervention_steps": n, "channels": channels, "all_exact": all(channels.values())}


def repeat_channels(record):
    return {key: record[key] for key in (
        "suffix_action_sha256", "suffix_environment_sha256", "suffix_observation_sha256",
        "suffix_policy_state_sha256", "suffix_global_rng_sha256",
        "suffix_request_sha256", "semantic_sha256", "request_sampling_seeds",
        "request_environment_step_indices", "num_steps", "termination_reason", "task_success",
        "intervention", "operator_events", "source_base_snapshot_id",
        "policy_environment_step_start", "policy_environment_step_end",
    )}
