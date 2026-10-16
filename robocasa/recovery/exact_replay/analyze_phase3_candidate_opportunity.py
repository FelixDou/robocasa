"""Analyze Phase 2 branches for bounded Phase 3 candidate opportunity.

This command is read-only with respect to the validated Phase 2 source.  It
collapses deterministic nominal repeats, excludes environment-only controls,
and treats the four explicit candidate branches from each complete snapshot as
one paired candidate set.  The primary outcome is completion of the active
semantic stage within the frozen suffix horizon.

Snapshot and branch payloads use pickle and must only be loaded from trusted
experiment directories.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from robocasa.recovery.full_snapshot import stable_digest


ANALYSIS_SCHEMA_VERSION = 3
ANALYSIS_PROTOCOL = "phase3_counterfactual_candidate_opportunity_v3"
SCIENTIFIC_PAYLOAD_PROTOCOL = "phase3_scientific_payload_without_replay_fingerprints_v1"


def sha256_file(path: str | Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text())


def _read_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _source_artifact_hash(path: Path, *, optional=False):
    """Hash a frozen source artifact, preserving an absent optional ledger."""
    if path.is_file():
        return sha256_file(path)
    if optional and not path.exists():
        return None
    raise FileNotFoundError(f"Phase 2 artifact is missing: {path}")


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
    return repr(value)


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _phase3_scientific_payload_digest(payload):
    """Hash fields used by Phase 3 without process-dependent replay blobs."""
    scientific = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "first_transition_fingerprint",
            "first_causal_transition_fingerprint",
        }
    }
    summary = dict(scientific.get("summary") or {})
    summary.pop("payload_sha256", None)
    scientific["summary"] = summary
    return stable_digest(scientific)


def _require_hash_sequence(record, payload, *, record_key, payload_key):
    expected = record.get(record_key)
    if expected is None or payload_key not in payload:
        return
    actual = [stable_digest(value) for value in payload[payload_key]]
    if actual != expected:
        raise ValueError(
            f"Branch {record['branch_id']} failed {payload_key} hash validation"
        )


def _load_payload(root: Path, record):
    path = root / record["payload_path"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing branch payload: {path}")
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - trusted experiment artifact
    file_sha256 = sha256_file(path)
    recorded_file_sha256 = record.get("payload_file_sha256")
    if recorded_file_sha256 is not None and file_sha256 != recorded_file_sha256:
        raise ValueError(
            f"Branch payload file hash mismatch for {record['branch_id']}: "
            f"{file_sha256} != {recorded_file_sha256}"
        )
    expected = record.get("payload_sha256")
    embedded = (payload.get("summary") or {}).get("payload_sha256")
    if expected is None or embedded != expected:
        raise ValueError(
            f"Branch payload embedded digest mismatch for {record['branch_id']}: "
            f"{embedded} != {expected}"
        )
    payload_summary = _jsonable(payload.get("summary") or {})
    for key, value in payload_summary.items():
        if key == "payload_sha256":
            continue
        if key not in record or record[key] != value:
            raise ValueError(f"Branch {record['branch_id']} summary mismatch for {key}")
    if stable_digest(payload.get("first_action")) != record.get("first_action_sha256"):
        raise ValueError(
            f"Branch {record['branch_id']} failed first-action hash validation"
        )
    inference_records = payload.get("inference_records") or []
    if not inference_records:
        raise ValueError(f"Branch {record['branch_id']} has no inference record")
    if stable_digest(inference_records[0].get("actions")) != record.get(
        "first_inference_actions_sha256"
    ):
        raise ValueError(
            f"Branch {record['branch_id']} failed inference-action hash validation"
        )
    _require_hash_sequence(
        record,
        payload,
        record_key="suffix_action_sha256",
        payload_key="actions",
    )
    _require_hash_sequence(
        record,
        payload,
        record_key="suffix_request_sha256",
        payload_key="request_records",
    )
    _require_hash_sequence(
        record,
        payload,
        record_key="suffix_observation_sha256",
        payload_key="observations",
    )
    integrity = {
        "source_payload_file_sha256": file_sha256,
        "source_payload_file_sha256_recorded": recorded_file_sha256,
        "legacy_payload_sha256": expected,
        "legacy_payload_manifest_embedded_equal": True,
        "scientific_payload_protocol": SCIENTIFIC_PAYLOAD_PROTOCOL,
        "scientific_payload_sha256": _phase3_scientific_payload_digest(payload),
    }
    return payload, path, integrity


def _trigger_predicate(plan, task_name):
    value = plan["trigger_stages"][task_name]
    return str(value).split("::", 1)[-1]


def _predicate_value(subtask_eval, name):
    predicate = (subtask_eval.get("predicates") or {}).get(name)
    if not isinstance(predicate, dict) or "value" not in predicate:
        raise ValueError(f"Subtask evaluation lacks trigger predicate {name!r}")
    return bool(predicate["value"])


def _branch_outcome(root, plan, record):
    payload, payload_path, integrity = _load_payload(root, record)
    trace = payload.get("subtask_trace") or []
    if not trace:
        raise ValueError(f"Branch {record['branch_id']} has no subtask trace")
    subtask_evals = [value for value in payload.get("subtask_evals") or [] if value]
    if not subtask_evals:
        raise ValueError(f"Branch {record['branch_id']} has no subtask evaluation")
    active_stage = _trigger_predicate(plan, record["task_name"])
    initial_stage_value = _predicate_value(subtask_evals[0], active_stage)
    if initial_stage_value:
        raise ValueError(
            f"Snapshot {record['snapshot_id']} trigger predicate "
            f"{active_stage!r} was already complete"
        )
    stage_completed = bool(record.get("task_success")) or any(
        _predicate_value(value, active_stage) for value in subtask_evals[1:]
    )
    initial_progress = float(subtask_evals[0].get("subtask_progress", 0.0))
    final_progress = float(subtask_evals[-1].get("subtask_progress", 0.0))
    regressions = sorted(set(record.get("regressed_predicates") or []))
    inference_records = payload.get("inference_records") or []
    if not inference_records:
        raise ValueError(f"Branch {record['branch_id']} has no inference record")
    inference = inference_records[0]
    features = np.asarray(inference.get("features"))
    actions = np.asarray(inference.get("actions"))
    state_context = np.asarray(
        (inference.get("auxiliary_features") or {}).get("observation_state_history", [])
    )
    for name, value in (("features", features), ("actions", actions)):
        if value.size == 0 or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"Branch {record['branch_id']} has invalid {name}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Branch {record['branch_id']} has non-finite {name}")
    return {
        "branch_id": record["branch_id"],
        "snapshot_id": record["snapshot_id"],
        "parent_id": record["parent_id"],
        "task_name": record["task_name"],
        "trigger_name": record.get("trigger_name"),
        "kind": record["kind"],
        "sampling_seed": int(record["sampling_seed"]),
        "active_stage": active_stage,
        "trigger_predicate_initial_value": initial_stage_value,
        "outcome_definition": "live_trigger_predicate_becomes_true_within_suffix",
        "stage_completed": stage_completed,
        "task_success": bool(record.get("task_success")),
        "initial_progress": initial_progress,
        "final_progress": final_progress,
        "progress_gain": final_progress - initial_progress,
        "regression": bool(regressions),
        "regressed_predicates": regressions,
        "termination_reason": record.get("termination_reason"),
        "num_steps": int(record.get("num_steps", 0)),
        "num_policy_requests": int(record.get("num_policy_requests", 0)),
        "action_chunk_shape": list(actions.shape),
        "action_chunk_l2": float(np.linalg.norm(actions.astype(np.float64))),
        "safe_feature_shape": list(features.shape),
        "safe_feature_l2": float(np.linalg.norm(features.astype(np.float64))),
        "state_context_shape": list(state_context.shape),
        "source_payload_path": record["payload_path"],
        **integrity,
        "_source_payload_absolute_path": str(payload_path),
        "action_chunk_sha256": stable_digest(actions),
        "safe_feature_sha256": stable_digest(features),
        "features": features,
        "actions": actions,
    }


def _pairwise_l2(arrays):
    flattened = [np.asarray(value, dtype=np.float64).reshape(-1) for value in arrays]
    distances = [
        float(np.linalg.norm(flattened[left] - flattened[right]))
        for left in range(len(flattened))
        for right in range(left + 1, len(flattened))
    ]
    return {
        "pairwise_l2_min": min(distances) if distances else 0.0,
        "pairwise_l2_mean": float(np.mean(distances)) if distances else 0.0,
    }


def _mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def _aggregate(rows):
    result = {
        "snapshots": len(rows),
        "mixed_outcome_fraction": _mean(rows, "mixed_outcome"),
        "random_of_four_stage_completion": _mean(
            rows, "random_of_four_stage_completion"
        ),
        "oracle_best_of_four_stage_completion": _mean(
            rows, "oracle_best_of_four_stage_completion"
        ),
        "oracle_headroom": _mean(rows, "oracle_headroom"),
        "nominal_stage_completion": _mean(rows, "nominal_stage_completion"),
        "random_minus_nominal": _mean(rows, "random_minus_nominal"),
        "successful_control_harm": _mean(rows, "successful_control_harm"),
        "candidate_regression_rate": _mean(rows, "candidate_regression_rate"),
        "candidate_action_pairwise_l2_mean": _mean(
            rows, "candidate_action_pairwise_l2_mean"
        ),
    }
    if all(row.get("safe_selected_stage_completion") is not None for row in rows):
        result.update(
            {
                "safe_selected_stage_completion": _mean(
                    rows, "safe_selected_stage_completion"
                ),
                "safe_minus_random": _mean(rows, "safe_minus_random"),
                "safe_minus_highest_safe": _mean(rows, "safe_minus_highest_safe"),
                "safe_score_tie_fraction": _mean(rows, "safe_score_tie"),
            }
        )
    return result


def _task_macro(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_name"]].append(row)
    task_results = {
        task: _aggregate(values) for task, values in sorted(grouped.items())
    }
    keys = sorted(set.intersection(*(set(value) for value in task_results.values())))
    macro = {
        key: float(np.mean([value[key] for value in task_results.values()]))
        for key in keys
        if key != "snapshots"
    }
    macro["tasks"] = len(task_results)
    macro["snapshots"] = len(rows)
    return task_results, macro


def _hierarchical_bootstrap(rows, *, replicates, seed):
    if int(replicates) < 1:
        raise ValueError("Bootstrap replicates must be positive")
    by_task = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task[row["task_name"]][row["parent_id"]].append(row)
    tasks = sorted(by_task)
    rng = np.random.default_rng(int(seed))
    samples = defaultdict(list)
    for _ in range(int(replicates)):
        selected_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        replicate_rows = []
        for task in selected_tasks:
            parents = sorted(by_task[task])
            selected_parents = rng.choice(parents, size=len(parents), replace=True)
            task_rows = []
            for parent in selected_parents:
                snapshots = by_task[task][parent]
                indices = rng.integers(0, len(snapshots), size=len(snapshots))
                task_rows.extend(snapshots[index] for index in indices)
            replicate_rows.append(_aggregate(task_rows))
        keys = set.intersection(*(set(item) for item in replicate_rows))
        for key in keys:
            if key == "snapshots":
                continue
            samples[key].append(float(np.mean([item[key] for item in replicate_rows])))
    return {
        key: {
            "mean": float(np.mean(values)),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
        for key, values in sorted(samples.items())
    }


def analyze_candidate_opportunity(
    phase2_run_dir,
    output_dir,
    *,
    bootstrap_replicates=2000,
    bootstrap_seed=0,
    scorer=None,
):
    root = Path(phase2_run_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    plan_path = root / "plan.json"
    analysis_path = root / "analysis.json"
    branches_path = root / "branch_records.jsonl"
    errors_path = root / "errors.jsonl"
    source_hashes_before = {
        plan_path.name: _source_artifact_hash(plan_path),
        analysis_path.name: _source_artifact_hash(analysis_path),
        branches_path.name: _source_artifact_hash(branches_path),
        errors_path.name: _source_artifact_hash(errors_path, optional=True),
    }
    plan = _read_json(plan_path)
    phase2_analysis = _read_json(analysis_path)
    records = _read_jsonl(branches_path)
    errors = _read_jsonl(errors_path) if errors_path.is_file() else []
    if (
        phase2_analysis.get("all_pass") is not True
        or int(phase2_analysis.get("errors", -1)) != 0
        or errors
    ):
        raise ValueError("Phase 3 requires a zero-error, all-pass Phase 2 source")
    if plan.get("branch_policy_connection_mode") != "shared_restored":
        raise ValueError("Phase 3 requires shared-restored Phase 2 policy state")
    expected_candidates = int(plan["candidate_count"])
    if expected_candidates != 4:
        raise ValueError(
            "Phase 3's frozen opportunity estimands require exactly four "
            f"candidates per snapshot, got {expected_candidates}"
        )
    if len(records) != int(plan.get("expected_total_branches", len(records))):
        raise ValueError("Phase 2 branch support differs from its frozen plan")
    frozen_provenance = {
        "phase2_plan_sha256": source_hashes_before["plan.json"],
        "phase2_runtime_bundle_sha256": plan.get("runtime_bundle_sha256"),
        "checkpoint_provenance_sha256": plan.get("checkpoint_provenance_sha256"),
        "robocasa_commit": plan.get("robocasa_commit"),
        "server_repository_commit": plan.get("server_repository_commit"),
        "suffix_steps": int(plan["suffix_steps"]),
    }
    grouped = defaultdict(list)
    for record in records:
        grouped[record["snapshot_id"]].append(record)
    if len(grouped) != int(plan["expected_snapshots"]):
        raise ValueError("Phase 2 snapshot support differs from its frozen plan")

    candidate_manifest = []
    snapshot_rows = []
    scorer_provenance = None
    primary_payload_file_hashes = {}
    primary_recorded_file_hashes = 0
    for snapshot_id, snapshot_records in sorted(grouped.items()):
        kinds = defaultdict(list)
        for record in snapshot_records:
            kinds[record["kind"]].append(record)
        if (
            len(kinds["same_seed_repeat"]) != 2
            or len(kinds["candidate"]) != expected_candidates
            or len(kinds["environment_only"]) != 1
        ):
            raise ValueError(f"Invalid branch composition for snapshot {snapshot_id}")
        repeats = [
            _branch_outcome(root, plan, record)
            for record in sorted(
                kinds["same_seed_repeat"], key=lambda row: row["branch_id"]
            )
        ]
        candidates = [
            _branch_outcome(root, plan, record)
            for record in sorted(
                kinds["candidate"], key=lambda row: row["sampling_seed"]
            )
        ]
        for branch in repeats + candidates:
            primary_payload_file_hashes[
                branch["_source_payload_absolute_path"]
            ] = branch["source_payload_file_sha256"]
            primary_recorded_file_hashes += int(
                branch["source_payload_file_sha256_recorded"] is not None
            )
        repeat_contract = (
            repeats[0]["stage_completed"] == repeats[1]["stage_completed"]
            and repeats[0]["task_success"] == repeats[1]["task_success"]
            and repeats[0]["final_progress"] == repeats[1]["final_progress"]
            and repeats[0]["regression"] == repeats[1]["regression"]
        )
        if not repeat_contract:
            raise ValueError(f"Nominal repeat outcomes disagree for {snapshot_id}")
        nominal = repeats[0]
        outcomes = np.asarray(
            [candidate["stage_completed"] for candidate in candidates],
            dtype=np.float64,
        )
        random_completion = float(np.mean(outcomes))
        oracle_completion = float(np.max(outcomes))
        action_diversity = _pairwise_l2(
            [candidate["actions"] for candidate in candidates]
        )
        feature_diversity = _pairwise_l2(
            [candidate["features"] for candidate in candidates]
        )
        safe_scores = None
        safe_result = None
        if scorer is not None:
            raw_features = np.stack([candidate["features"] for candidate in candidates])
            safe_result = scorer.score_candidates(
                raw_features,
                task_name=nominal["task_name"],
            )
            safe_scores = np.asarray(safe_result["scores"], dtype=np.float64)
            if safe_scores.shape != (expected_candidates,) or not np.all(
                np.isfinite(safe_scores)
            ):
                raise ValueError(
                    f"SAFE scorer returned invalid values for {snapshot_id}"
                )
            scorer_provenance = safe_result.get("provenance")
        for index, candidate in enumerate(candidates):
            row = {
                key: value
                for key, value in candidate.items()
                if key not in {"features", "actions", "_source_payload_absolute_path"}
            }
            row["candidate_index"] = index
            row.update(frozen_provenance)
            row["snapshot_sha256"] = sha256_file(
                root / "snapshots" / f"{snapshot_id}.pkl.gz"
            )
            row["safe_score"] = (
                None if safe_scores is None else float(safe_scores[index])
            )
            candidate_manifest.append(row)
        safe_selected = None
        safe_highest = None
        safe_tie = None
        if safe_scores is not None:
            safe_selected = float(outcomes[int(np.argmin(safe_scores))])
            safe_highest = float(outcomes[int(np.argmax(safe_scores))])
            safe_tie = bool(float(np.ptp(safe_scores)) == 0.0)
        snapshot_rows.append(
            {
                "snapshot_id": snapshot_id,
                "parent_id": nominal["parent_id"],
                "task_name": nominal["task_name"],
                "trigger_name": nominal["trigger_name"],
                "active_stage": nominal["active_stage"],
                "candidate_count": expected_candidates,
                "mixed_outcome": bool(np.min(outcomes) != np.max(outcomes)),
                "random_of_four_stage_completion": random_completion,
                "oracle_best_of_four_stage_completion": oracle_completion,
                "oracle_headroom": oracle_completion - random_completion,
                "nominal_stage_completion": float(nominal["stage_completed"]),
                "random_minus_nominal": (
                    random_completion - float(nominal["stage_completed"])
                ),
                "successful_control_harm": (
                    float(np.mean(1.0 - outcomes))
                    if nominal["stage_completed"]
                    else 0.0
                ),
                "candidate_regression_rate": float(
                    np.mean([candidate["regression"] for candidate in candidates])
                ),
                "candidate_action_pairwise_l2_min": action_diversity["pairwise_l2_min"],
                "candidate_action_pairwise_l2_mean": action_diversity[
                    "pairwise_l2_mean"
                ],
                "candidate_feature_pairwise_l2_min": feature_diversity[
                    "pairwise_l2_min"
                ],
                "candidate_feature_pairwise_l2_mean": feature_diversity[
                    "pairwise_l2_mean"
                ],
                "safe_selected_stage_completion": safe_selected,
                "highest_safe_stage_completion": safe_highest,
                "safe_minus_random": (
                    None if safe_selected is None else safe_selected - random_completion
                ),
                "safe_minus_highest_safe": (
                    None if safe_selected is None else safe_selected - safe_highest
                ),
                "safe_score_tie": safe_tie,
            }
        )

    task_summary, macro = _task_macro(snapshot_rows)
    bootstrap = _hierarchical_bootstrap(
        snapshot_rows,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    tasks_with_mixed = sum(
        value["mixed_outcome_fraction"] > 0.0 for value in task_summary.values()
    )
    formal_estimable = len(task_summary) >= 5
    gates = {
        "mixed_outcome_fraction_at_least_0p20": (
            macro["mixed_outcome_fraction"] >= 0.20
        ),
        "mixed_outcomes_in_at_least_3_tasks": tasks_with_mixed >= 3,
        "oracle_headroom_at_least_0p15": macro["oracle_headroom"] >= 0.15,
    }
    source_hashes_after = {
        plan_path.name: _source_artifact_hash(plan_path),
        analysis_path.name: _source_artifact_hash(analysis_path),
        branches_path.name: _source_artifact_hash(branches_path),
        errors_path.name: _source_artifact_hash(errors_path, optional=True),
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("Phase 2 source artifacts changed during analysis")
    changed_payload_files = [
        path
        for path, expected in primary_payload_file_hashes.items()
        if sha256_file(path) != expected
    ]
    if changed_payload_files:
        raise RuntimeError(
            "Phase 2 branch payloads changed during analysis: "
            + ", ".join(changed_payload_files)
        )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "protocol": ANALYSIS_PROTOCOL,
        "status": "pilot_complete" if not formal_estimable else "complete",
        "evidence_scope": "simulator_oracle_candidate_opportunity",
        "phase2_run_dir": str(root),
        "phase2_source_hashes": source_hashes_before,
        "phase2_source_unchanged": True,
        "phase2_error_ledger_present": errors_path.is_file(),
        "legacy_payload_digest_verification": (
            "manifest_equals_embedded; aggregate rehash not portable because "
            "replay transition fingerprints vary across Python environments"
        ),
        "scientific_payload_protocol": SCIENTIFIC_PAYLOAD_PROTOCOL,
        "primary_payload_files_verified_unchanged": len(primary_payload_file_hashes),
        "recorded_raw_payload_file_hashes_available": primary_recorded_file_hashes,
        "tasks": sorted(task_summary),
        "parents": len({row["parent_id"] for row in snapshot_rows}),
        "snapshots": len(snapshot_rows),
        "candidates": len(candidate_manifest),
        "candidate_count_per_snapshot": expected_candidates,
        "primary_outcome": "live_trigger_predicate_completion_within_frozen_suffix",
        "environment_only_controls_excluded": True,
        "nominal_repeat_pairs_collapsed": True,
        "safe_scoring_enabled": scorer is not None,
        "safe_scoring_provenance": scorer_provenance,
        "task_summary": task_summary,
        "task_macro": macro,
        "tasks_with_mixed_outcomes": tasks_with_mixed,
        "bootstrap": {
            "hierarchy": "task_parent_snapshot",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "metrics": bootstrap,
        },
        "continuation_gates": gates,
        "formal_five_task_gate_estimable": formal_estimable,
        "formal_phase3_gate_passed": (
            all(gates.values()) if formal_estimable else None
        ),
        "bounded_pilot_supports_scaling": bool(
            gates["mixed_outcome_fraction_at_least_0p20"]
            and gates["oracle_headroom_at_least_0p15"]
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "candidate_opportunity.jsonl", candidate_manifest)
    _write_csv(output / "snapshot_opportunity.csv", snapshot_rows)
    _write_csv(
        output / "task_summary.csv",
        [{"task_name": task, **values} for task, values in task_summary.items()],
    )
    _atomic_json(output / "analysis.json", analysis)
    return analysis


def print_analysis(analysis):
    macro = analysis["task_macro"]
    print("PHASE 3 COUNTERFACTUAL OPPORTUNITY PILOT")
    print("status:", analysis["status"])
    print("scope:", analysis["evidence_scope"])
    print("tasks:", " ".join(analysis["tasks"]))
    print("parents:", analysis["parents"])
    print("snapshots:", analysis["snapshots"])
    print("candidates:", analysis["candidates"])
    print("safe scoring:", analysis["safe_scoring_enabled"])
    print("\nTASK-MACRO OPPORTUNITY")
    print(f"mixed-outcome fraction: {macro['mixed_outcome_fraction']:.4f}")
    print(
        "random-of-four completion: " f"{macro['random_of_four_stage_completion']:.4f}"
    )
    print(
        "oracle-best-of-four completion: "
        f"{macro['oracle_best_of_four_stage_completion']:.4f}"
    )
    print(f"oracle headroom: {macro['oracle_headroom']:+.4f}")
    print(f"nominal completion: {macro['nominal_stage_completion']:.4f}")
    if "safe_selected_stage_completion" in macro:
        print(
            "lowest-SAFE completion: " f"{macro['safe_selected_stage_completion']:.4f}"
        )
        print(f"SAFE minus random: {macro['safe_minus_random']:+.4f}")
    print("\nPER TASK")
    for task, values in analysis["task_summary"].items():
        print(
            f"{task:28s} mixed={values['mixed_outcome_fraction']:.4f} "
            f"headroom={values['oracle_headroom']:+.4f} "
            f"random={values['random_of_four_stage_completion']:.4f} "
            f"oracle={values['oracle_best_of_four_stage_completion']:.4f}"
        )
    print("\nCONTINUATION GATES")
    for name, passed in analysis["continuation_gates"].items():
        print(f"{'PASS' if passed else 'FAIL':4s}  {name}")
    if analysis["formal_five_task_gate_estimable"]:
        print(
            "\nFORMAL PHASE 3 GATE:",
            "PASS" if analysis["formal_phase3_gate_passed"] else "FAIL",
        )
    else:
        print("\nFORMAL PHASE 3 GATE: NOT ESTIMABLE (two-task bounded pilot)")
        print(
            "PILOT SCALE DECISION:",
            "CONTINUE" if analysis["bounded_pilot_supports_scaling"] else "REDESIGN",
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--safe-runtime-bundle", type=Path)
    parser.add_argument("--safe-repo", type=Path)
    parser.add_argument("--safe-device", default="cpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scorer = None
    if (args.safe_runtime_bundle is None) != (args.safe_repo is None):
        raise SystemExit(
            "error: --safe-runtime-bundle and --safe-repo must be provided together"
        )
    if args.safe_runtime_bundle is not None:
        from robocasa.recovery.safe.xr1_best_of_k import FrozenXr1SafeEnsemble

        scorer = FrozenXr1SafeEnsemble(
            runtime_bundle=args.safe_runtime_bundle,
            safe_repo=args.safe_repo,
            device=args.safe_device,
        )
    analysis = analyze_candidate_opportunity(
        args.phase2_run_dir,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        scorer=scorer,
    )
    print_analysis(analysis)
    print("\nartifacts:", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
