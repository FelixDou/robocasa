"""Registered Phase 3B training-horizon continuation helpers.

Phase 3B is deliberately a continuation of the frozen Phase 2 experiment.  It
does not recollect parents, choose new snapshots, or draw new treatment seeds.
The engineering sentinel replays four deterministically selected snapshots and
must reproduce every causal channel of the original 64-step suffix before the
full 20-snapshot screen is allowed to run.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

from robocasa.recovery.counterfactual_branch import (
    branch_payload_digest,
    load_branch_records,
)
from robocasa.recovery.full_snapshot import stable_digest


PHASE3B_SCHEMA_VERSION = 1
PHASE3B_PROTOCOL = "phase3b_training_horizon_continuation"
PHASE3B_TASK_HORIZONS = {
    "ArrangeTea": 480,
    "CuttingToolSelection": 256,
}
PHASE2_PREFIX_STEPS = 64
BRANCH_KINDS = {"same_seed_repeat", "candidate", "environment_only"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {line_number}: {path}") from error
    return rows


def load_branch_payload(root: str | Path, record: dict) -> dict:
    root = Path(root)
    payload_path = root / record["payload_path"]
    if not payload_path.is_file():
        raise FileNotFoundError(f"Missing branch payload: {payload_path}")
    if record.get("payload_file_sha256"):
        actual = sha256_file(payload_path)
        if actual != record["payload_file_sha256"]:
            raise ValueError(f"Branch payload file changed: {record['branch_id']}")
    with gzip.open(payload_path, "rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - trusted experiment artifact
    embedded = (payload.get("summary") or {}).get("payload_sha256")
    if embedded != record.get("payload_sha256"):
        raise ValueError(f"Embedded branch checksum mismatch: {record['branch_id']}")
    # The legacy embedded digest includes scientific arrays but also a few
    # process-dependent diagnostic objects.  Phase 2 already froze it and the
    # compressed file hash; recomputing it under a different Python/NumPy
    # runtime can yield a false mismatch.  File hash + embedded/ledger equality
    # is therefore the cross-runtime integrity contract.
    return payload


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "plan": root / "plan.json",
        "analysis": root / "analysis.json",
        "parents": root / "parent_records.jsonl",
        "branches": root / "branch_records.jsonl",
        "errors": root / "errors.jsonl",
    }


def _phase3b_code_paths() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    relative = (
        "robocasa/recovery/full_snapshot.py",
        "robocasa/recovery/counterfactual_branch.py",
        "robocasa/recovery/phase3b_training_horizon.py",
        "robocasa/recovery/run_phase3b_training_horizon_continuation.py",
        "robocasa/recovery/analyze_phase3b_training_horizon.py",
    )
    return {name: repository / name for name in relative}


def _validate_source(root: Path) -> tuple[dict, dict, list[dict], list[dict]]:
    paths = _source_paths(root)
    missing = [str(path) for key, path in paths.items() if key != "errors" and not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Phase 2 source artifacts: " + ", ".join(missing))
    plan = json.loads(paths["plan"].read_text())
    analysis = json.loads(paths["analysis"].read_text())
    parents = read_jsonl(paths["parents"])
    records = load_branch_records(paths["branches"])
    errors = read_jsonl(paths["errors"])
    if analysis.get("all_pass") is not True or analysis.get("errors") != 0 or errors:
        raise ValueError("Phase 3B requires a zero-error, all-pass Phase 2 source")
    if plan.get("branch_policy_connection_mode") != "shared_restored":
        raise ValueError("Phase 3B requires the shared-restored policy protocol")
    if int(plan.get("candidate_count", -1)) != 4:
        raise ValueError("Phase 3B requires exactly four candidate branches")
    if int(plan.get("suffix_steps", -1)) != PHASE2_PREFIX_STEPS:
        raise ValueError("Phase 3B requires the frozen 64-step Phase 2 suffix")
    if set(plan.get("tasks") or ()) != set(PHASE3B_TASK_HORIZONS):
        raise ValueError("Phase 3B is registered for ArrangeTea and CuttingToolSelection")
    expected = int(plan.get("expected_total_branches", len(records)))
    if len(records) != expected:
        raise ValueError(f"Phase 2 branch support is incomplete: {len(records)} != {expected}")
    return plan, analysis, parents, records


def _parent_identity_map(plan: dict) -> dict[str, dict]:
    return {row["parent_id"]: row for row in plan.get("candidate_parents") or []}


def deterministic_snapshot_selection(
    plan: dict,
    parents: list[dict],
    *,
    scope: str,
) -> list[dict]:
    """Select the registered sentinel or full Phase 2 snapshot support."""
    if scope not in {"sentinel", "full"}:
        raise ValueError("scope must be sentinel or full")
    identities = _parent_identity_map(plan)
    valid = []
    for parent in parents:
        if not parent.get("valid", True):
            continue
        parent_id = parent["parent_id"]
        identity = identities.get(parent_id)
        if identity is None:
            raise ValueError(f"Parent {parent_id} is absent from the frozen plan")
        captured = parent.get("captured") or {}
        if set(captured) != {"prefix", "landmark"}:
            raise ValueError(f"Parent {parent_id} lacks its frozen snapshot pair")
        valid.append(
            {
                "parent_id": parent_id,
                "task_name": parent["task_name"],
                "environment_seed": int(identity["environment_seed"]),
                "environment_reset_index": int(identity["environment_reset_index"]),
                "snapshots": {
                    "prefix": captured["prefix"],
                    "landmark": captured["landmark"],
                },
            }
        )
    by_task = defaultdict(list)
    for row in valid:
        by_task[row["task_name"]].append(row)
    selected = []
    for task in sorted(PHASE3B_TASK_HORIZONS):
        candidates = sorted(
            by_task[task],
            key=lambda row: (
                row["environment_reset_index"],
                row["environment_seed"],
                row["parent_id"],
            ),
        )
        if not candidates:
            raise ValueError(f"Phase 2 contains no valid parent for {task}")
        selected.extend(candidates[:1] if scope == "sentinel" else candidates)
    expected_parents = 2 if scope == "sentinel" else 10
    if len(selected) != expected_parents:
        raise ValueError(
            f"Registered {scope} parent support differs: {len(selected)} != {expected_parents}"
        )
    rows = []
    for parent in selected:
        for boundary in ("prefix", "landmark"):
            rows.append(
                {
                    **{key: value for key, value in parent.items() if key != "snapshots"},
                    "boundary": boundary,
                    "snapshot_id": parent["snapshots"][boundary],
                    "horizon_environment_steps": PHASE3B_TASK_HORIZONS[
                        parent["task_name"]
                    ],
                }
            )
    return rows


def _validate_branch_group(snapshot_id: str, rows: list[dict]) -> None:
    kinds = defaultdict(list)
    for row in rows:
        if row.get("kind") not in BRANCH_KINDS:
            raise ValueError(f"Unsupported source branch kind for {snapshot_id}")
        kinds[row["kind"]].append(row)
    if (
        len(kinds["same_seed_repeat"]) != 2
        or len(kinds["candidate"]) != 4
        or len(kinds["environment_only"]) != 1
    ):
        raise ValueError(f"Invalid frozen branch composition for {snapshot_id}")
    repeat_seeds = {int(row["sampling_seed"]) for row in kinds["same_seed_repeat"]}
    if len(repeat_seeds) != 1:
        raise ValueError(f"Nominal repeats use different seeds for {snapshot_id}")
    if len({int(row["sampling_seed"]) for row in kinds["candidate"]}) != 4:
        raise ValueError(f"Candidate treatment seeds are not unique for {snapshot_id}")


def build_registration(
    phase2_run_dir: str | Path,
    *,
    scope: str,
    sentinel_run_dir: str | Path | None = None,
) -> dict:
    root = Path(phase2_run_dir).resolve()
    plan, analysis, parents, records = _validate_source(root)
    selection = deterministic_snapshot_selection(plan, parents, scope=scope)
    selected_ids = {row["snapshot_id"] for row in selection}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["snapshot_id"]].append(record)
    if set(selected_ids) - set(grouped):
        raise ValueError("Selected snapshots are absent from the branch ledger")

    paths = _source_paths(root)
    source_hashes = {
        key: (sha256_file(path) if path.is_file() else None)
        for key, path in paths.items()
    }
    branches = []
    payload_hashes = {}
    snapshot_hashes = {}
    for snapshot in selection:
        snapshot_id = snapshot["snapshot_id"]
        snapshot_path = root / "snapshots" / f"{snapshot_id}.pkl.gz"
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"Missing frozen snapshot: {snapshot_path}")
        snapshot_hashes[snapshot_id] = sha256_file(snapshot_path)
        source_rows = grouped[snapshot_id]
        _validate_branch_group(snapshot_id, source_rows)
        for source in sorted(
            source_rows,
            key=lambda row: (
                {"same_seed_repeat": 0, "candidate": 1, "environment_only": 2}[
                    row["kind"]
                ],
                int(row.get("repeat_index") or 0),
                int(row["sampling_seed"]),
                row["branch_id"],
            ),
        ):
            payload_path = root / source["payload_path"]
            actual_file_hash = sha256_file(payload_path)
            if source.get("payload_file_sha256") not in {None, actual_file_hash}:
                raise ValueError(f"Frozen payload changed: {source['branch_id']}")
            payload_hashes[source["branch_id"]] = actual_file_hash
            branch_id = f"{source['branch_id']}--phase3b-{scope}"
            branches.append(
                {
                    "branch_id": branch_id,
                    "source_branch_id": source["branch_id"],
                    "snapshot_id": snapshot_id,
                    "parent_id": snapshot["parent_id"],
                    "task_name": snapshot["task_name"],
                    "boundary": snapshot["boundary"],
                    "kind": source["kind"],
                    "repeat_index": source.get("repeat_index"),
                    "sampling_seed": int(source["sampling_seed"]),
                    "suffix_steps": int(snapshot["horizon_environment_steps"]),
                    "source_payload_path": source["payload_path"],
                    "source_payload_sha256": source.get("payload_sha256"),
                    "source_payload_file_sha256": actual_file_hash,
                }
            )

    sentinel = None
    if scope == "full":
        if sentinel_run_dir is None:
            raise ValueError("Full Phase 3B requires --sentinel-run-dir")
        sentinel_root = Path(sentinel_run_dir).resolve()
        sentinel_analysis_path = sentinel_root / "analysis.json"
        sentinel_plan_path = sentinel_root / "registration.json"
        if not sentinel_analysis_path.is_file() or not sentinel_plan_path.is_file():
            raise ValueError("Sentinel registration or analysis is missing")
        sentinel_analysis = json.loads(sentinel_analysis_path.read_text())
        sentinel_plan = json.loads(sentinel_plan_path.read_text())
        validate_registration_source(sentinel_plan)
        if sentinel_plan.get("scope") != "sentinel":
            raise ValueError("--sentinel-run-dir is not a sentinel registration")
        if sentinel_analysis.get("engineering_all_pass") is not True:
            raise ValueError("Full Phase 3B is blocked by the engineering sentinel")
        if sentinel_plan.get("phase2_source_hashes") != source_hashes:
            raise ValueError("Sentinel and full run reference different Phase 2 artifacts")
        sentinel = {
            "run_dir": str(sentinel_root),
            "registration_sha256": sha256_file(sentinel_plan_path),
            "analysis_sha256": sha256_file(sentinel_analysis_path),
        }

    registration = {
        "schema_version": PHASE3B_SCHEMA_VERSION,
        "protocol": PHASE3B_PROTOCOL,
        "status": "registered",
        "created_at": utc_now(),
        "scope": scope,
        "phase2_run_dir": str(root),
        "phase2_source_hashes": source_hashes,
        "phase2_snapshot_file_sha256": snapshot_hashes,
        "phase2_branch_payload_file_sha256": payload_hashes,
        "phase3b_code_file_sha256": {
            name: sha256_file(path) for name, path in _phase3b_code_paths().items()
        },
        "phase2_runtime_bundle_sha256": plan.get("runtime_bundle_sha256"),
        "checkpoint_provenance_sha256": plan.get("checkpoint_provenance_sha256"),
        "robocasa_commit": plan.get("robocasa_commit"),
        "server_repository_commit": plan.get("server_repository_commit"),
        "tasks": sorted(PHASE3B_TASK_HORIZONS),
        "target_stages": plan.get("target_stages"),
        "trigger_stages": plan.get("trigger_stages"),
        "task_horizons_environment_steps": PHASE3B_TASK_HORIZONS,
        "replan_steps": int(plan.get("replan_steps", 16)),
        "phase2_prefix_environment_steps": PHASE2_PREFIX_STEPS,
        "candidate_count": 4,
        "selected_snapshots": selection,
        "registered_branches": branches,
        "expected_parents": len({row["parent_id"] for row in selection}),
        "expected_snapshots": len(selection),
        "expected_branches": len(branches),
        "expected_scientific_branches": sum(
            row["kind"] == "candidate" for row in branches
        ),
        "sentinel_provenance": sentinel,
        "selection_rule": (
            "lowest_environment_reset_index_valid_parent_per_task_then_both_boundaries"
            if scope == "sentinel"
            else "all_frozen_phase2_parents_and_both_boundaries"
        ),
        "scientific_outcomes_may_stop_full_progression": False,
    }
    registration["registration_sha256"] = sha256_json(registration)
    return registration


def validate_registration_source(registration: dict) -> None:
    frozen_digest = registration.get("registration_sha256")
    unsigned = dict(registration)
    unsigned.pop("registration_sha256", None)
    if frozen_digest != sha256_json(unsigned):
        raise ValueError("Phase 3B registration checksum mismatch")
    root = Path(registration["phase2_run_dir"])
    current = {
        key: (sha256_file(path) if path.is_file() else None)
        for key, path in _source_paths(root).items()
    }
    if current != registration["phase2_source_hashes"]:
        raise ValueError("Frozen Phase 2 source artifacts changed after registration")
    for snapshot_id, expected in registration[
        "phase2_snapshot_file_sha256"
    ].items():
        path = root / "snapshots" / f"{snapshot_id}.pkl.gz"
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen snapshot changed: {snapshot_id}")
    source_by_id = {
        row["branch_id"]: row
        for row in load_branch_records(root / "branch_records.jsonl")
    }
    for source_id, expected in registration[
        "phase2_branch_payload_file_sha256"
    ].items():
        record = source_by_id[source_id]
        if sha256_file(root / record["payload_path"]) != expected:
            raise ValueError(f"Frozen branch payload changed: {source_id}")
    current_code = {
        name: sha256_file(path) for name, path in _phase3b_code_paths().items()
    }
    if current_code != registration.get("phase3b_code_file_sha256"):
        raise ValueError("Phase 3B execution code changed after registration")


def first64_equality_audit(
    source_record: dict,
    source_payload: dict,
    continued_result: dict,
) -> dict:
    """Fail-closed equality audit for every registered causal prefix channel."""
    continued_summary = continued_result["summary"]
    continued_payload = continued_result["payload"]
    source_steps = int(source_record.get("num_steps", 0))
    if source_steps != PHASE2_PREFIX_STEPS:
        raise ValueError(
            f"Source branch {source_record['branch_id']} does not contain 64 steps"
        )
    if int(continued_summary.get("num_steps", 0)) < PHASE2_PREFIX_STEPS:
        raise ValueError(
            f"Continued branch ended before its registered 64-step prefix: "
            f"{continued_summary.get('branch_id')}"
        )
    request_count = int(source_record.get("num_policy_requests", 0))
    channels = {
        "requests": source_record.get("suffix_request_sha256", [])
        == continued_summary.get("suffix_request_sha256", [])[:request_count],
        "actions": source_record.get("suffix_action_sha256", [])
        == continued_summary.get("suffix_action_sha256", [])[:PHASE2_PREFIX_STEPS],
        "observations": source_record.get("suffix_observation_sha256", [])
        == continued_summary.get("suffix_observation_sha256", [])[
            : PHASE2_PREFIX_STEPS + 1
        ],
        "causal_environment": source_record.get("suffix_environment_sha256", [])
        == continued_summary.get("suffix_environment_sha256", [])[
            :PHASE2_PREFIX_STEPS
        ],
        "semantic_trace": stable_digest(
            (source_payload.get("subtask_trace") or [])[: PHASE2_PREFIX_STEPS + 1]
        )
        == stable_digest(
            (continued_payload.get("subtask_trace") or [])[
                : PHASE2_PREFIX_STEPS + 1
            ]
        ),
        "declared_seed": int(source_record["sampling_seed"])
        == int(continued_summary["sampling_seed"]),
    }
    audit = {
        "source_branch_id": source_record["branch_id"],
        "continued_branch_id": continued_summary["branch_id"],
        "prefix_steps": PHASE2_PREFIX_STEPS,
        "channels": channels,
        "all_exact": all(channels.values()),
    }
    if not audit["all_exact"]:
        failed = sorted(name for name, exact in channels.items() if not exact)
        raise ValueError(
            f"Phase 3B first-64 mismatch for {source_record['branch_id']}: {failed}"
        )
    return audit


def annotate_continued_result(result: dict, registration_branch: dict, audit: dict) -> None:
    summary = result["summary"]
    summary.pop("payload_sha256", None)
    summary.update(
        {
            "phase3b_protocol": PHASE3B_PROTOCOL,
            "source_branch_id": registration_branch["source_branch_id"],
            "source_payload_sha256": registration_branch["source_payload_sha256"],
            "boundary": registration_branch["boundary"],
            "first64_exact": bool(audit["all_exact"]),
            "first64_channels": deepcopy(audit["channels"]),
            "registered_horizon_environment_steps": int(
                registration_branch["suffix_steps"]
            ),
        }
    )
    result["payload"]["summary"] = summary
    summary["payload_sha256"] = branch_payload_digest(result["payload"])


def completion_event(payload: dict, predicate: str) -> dict:
    evaluations = payload.get("subtask_evals") or []
    if not evaluations:
        raise ValueError("Branch payload has no semantic evaluations")
    values = []
    for evaluation in evaluations:
        item = (evaluation.get("predicates") or {}).get(predicate)
        values.append(bool(item.get("value")) if isinstance(item, dict) else False)
    if values[0]:
        raise ValueError(
            f"Snapshot trigger predicate {predicate!r} was already complete"
        )
    first = next((index for index, value in enumerate(values[1:], 1) if value), None)
    request_steps = (payload.get("summary") or {}).get(
        "request_environment_step_indices", []
    )
    first_replan = None
    if first is not None:
        completed_step = first - 1
        first_replan = sum(int(step) <= completed_step for step in request_steps)
    return {
        "completed": first is not None,
        "first_completion_environment_step": None if first is None else first - 1,
        "first_completion_replan": first_replan,
        "durable_to_branch_end": bool(first is not None and values[-1]),
        "regressed_after_completion": bool(
            first is not None and any(not value for value in values[first + 1 :])
        ),
    }


def route_phase3b(task_rows: dict, macro: dict, gates: dict) -> str:
    if all(gates.values()):
        return "candidate_critic_formal_five_task_screen"
    homogeneous_success = all(
        row.get("mixed_outcome_fraction", 0.0) == 0.0
        and row.get("random_minus_nominal", 0.0) > 0.0
        for row in task_rows.values()
    )
    if homogeneous_success:
        return "fresh_replan_without_candidate_ranking"
    if macro.get("mixed_outcome_fraction", 0.0) > 0.0 and macro.get(
        "oracle_minus_nominal", 0.0
    ) <= 0.0:
        return "do_not_train_candidate_critic"
    return "higher_level_operator_screen"


__all__ = [
    "PHASE2_PREFIX_STEPS",
    "PHASE3B_PROTOCOL",
    "PHASE3B_SCHEMA_VERSION",
    "PHASE3B_TASK_HORIZONS",
    "annotate_continued_result",
    "atomic_write_json",
    "build_registration",
    "completion_event",
    "deterministic_snapshot_selection",
    "first64_equality_audit",
    "load_branch_payload",
    "read_jsonl",
    "route_phase3b",
    "sha256_file",
    "validate_registration_source",
    "write_jsonl",
]
