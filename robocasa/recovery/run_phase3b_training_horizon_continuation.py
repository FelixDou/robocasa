"""Run the registered Phase 3B training-horizon continuation.

Use ``--scope sentinel`` first.  The sentinel deterministically selects one
frozen parent per task (four snapshots total) and gates only replay integrity.
After it passes, ``--scope full --sentinel-run-dir ...`` rebranches all twenty
frozen Phase 2 snapshots without changing snapshots, treatment seeds, or the
first 64 environment steps.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

from robocasa.recovery.counterfactual_branch import (
    BranchSpec,
    analyze_phase2_replay,
    load_branch_records,
    run_counterfactual_branch,
    save_branch_result,
)
from robocasa.recovery.full_snapshot import restore_full_snapshot
from robocasa.recovery.phase3b_training_horizon import (
    annotate_continued_result,
    atomic_write_json,
    build_registration,
    first64_equality_audit,
    load_branch_payload,
    load_registered_phase2_snapshot,
    read_jsonl,
    sha256_file,
    validate_registration_source,
    write_jsonl,
)
from robocasa.recovery.run_phase2_snapshot_replay import (
    _call_with_deterministic_environment_seed,
    _close_branch_environment,
    _make_fresh_branch_environment,
    _reset_env,
    _runtime,
    checkpoint_provenance,
    sha256_json,
    utc_now,
)


def _append_jsonl(path: Path, row: dict) -> None:
    rows = read_jsonl(path)
    rows.append(row)
    write_jsonl(path, rows)


def _record_snapshot_integrity_audit(path: Path, audit: dict) -> None:
    rows = {row["snapshot_id"]: row for row in read_jsonl(path)}
    rows[audit["snapshot_id"]] = audit
    write_jsonl(path, [rows[key] for key in sorted(rows)])


def _load_registered_snapshot(
    source_root: Path,
    registration: dict,
    selected: dict,
    output_dir: Path,
):
    snapshot_id = selected["snapshot_id"]
    snapshot, audit = load_registered_phase2_snapshot(
        source_root / "snapshots" / f"{snapshot_id}.pkl.gz",
        expected_file_sha256=registration["phase2_snapshot_file_sha256"][
            snapshot_id
        ],
        expected_snapshot_id=snapshot_id,
        expected_parent_id=selected["parent_id"],
        expected_task_name=selected["task_name"],
    )
    _record_snapshot_integrity_audit(
        output_dir / "snapshot_integrity_audits.jsonl", audit
    )
    return snapshot


def _source_plan(registration: dict) -> dict:
    return json.loads(
        (Path(registration["phase2_run_dir"]) / "plan.json").read_text()
    )


def _runtime_args(args, source_plan: dict) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=args.output_dir,
        fresh_branch_contexts=True,
        env_interface=source_plan.get("env_interface", "gym"),
        split=source_plan.get("split", "pretrain"),
        canonical_camera_observations=bool(
            source_plan.get("canonical_camera_observations", False)
        ),
        canonical_camera_render_repeats=int(
            source_plan.get("canonical_camera_render_repeats", 2)
        ),
        restore_atol=float(args.restore_atol),
        restore_rtol=float(args.restore_rtol),
    )


def _validate_live_provenance(args, registration: dict, source_plan: dict) -> None:
    validate_registration_source(registration)
    runtime_path = Path(source_plan["runtime_bundle"])
    if not runtime_path.is_file() or sha256_file(runtime_path) != source_plan.get(
        "runtime_bundle_sha256"
    ):
        raise ValueError("Frozen SAFE runtime bundle changed or is missing")
    current_checkpoint = checkpoint_provenance(args.model_path, include_shards=True)
    if sha256_json(current_checkpoint) != source_plan.get(
        "checkpoint_provenance_sha256"
    ):
        raise ValueError("Live checkpoint differs from frozen Phase 2 provenance")
    server_entrypoint = source_plan.get("server_entrypoint")
    if server_entrypoint:
        server_path = Path(server_entrypoint)
        if not server_path.is_file() or sha256_file(server_path) != source_plan.get(
            "server_entrypoint_sha256"
        ):
            raise ValueError("Frozen Xiaomi server entrypoint changed or is missing")


def _registered_by_snapshot(registration: dict) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in registration["registered_branches"]:
        grouped[row["snapshot_id"]].append(row)
    return dict(grouped)


def _make_policy_args(args, source_plan: dict, runtime) -> tuple[object, dict]:
    factory = runtime["load_factory"](source_plan["policy_module"])
    policy_args = runtime["parse_policy_args"](args.policy_arg)
    policy_args.update(
        {
            "model_path": str(args.model_path),
            "host": args.host,
            "port": args.port,
            "replan_steps": args.replan_steps,
            "collect_safe_features": True,
            "policy_name": args.policy_name,
            "policy_checkpoint": source_plan["checkpoint"],
            "safe_best_of_k": 1,
        }
    )
    return factory, policy_args


def _execute_registered_branch(
    *,
    registration_branch: dict,
    snapshot,
    source_record: dict,
    source_payload: dict,
    policy,
    runtime,
    runtime_args,
):
    spec = BranchSpec(
        branch_id=registration_branch["branch_id"],
        kind=registration_branch["kind"],
        sampling_seed=int(registration_branch["sampling_seed"]),
        suffix_steps=int(registration_branch["suffix_steps"]),
        repeat_index=registration_branch.get("repeat_index"),
    )
    branch_env = _make_fresh_branch_environment(snapshot, runtime_args, runtime)
    try:
        if spec.kind == "environment_only":
            restore_full_snapshot(
                snapshot,
                branch_env,
                policy,
                snapshot_integrity_prevalidated=True,
            )
            mutation_action = runtime["call_policy"](
                policy, deepcopy(snapshot.observation)
            )
            runtime["step_fn"](branch_env, mutation_action)
            pop_requests = getattr(policy, "pop_request_records", None)
            if callable(pop_requests):
                pop_requests()
            pop_inference = getattr(policy, "pop_inference_record", None)
            if callable(pop_inference):
                pop_inference()
        result = run_counterfactual_branch(
            snapshot,
            branch_env,
            policy,
            spec,
            step_fn=runtime["step_fn"],
            success_fn=runtime["success_fn"],
            subtask_eval_fn=runtime["get_subtask_eval"],
            restore_atol=runtime_args.restore_atol,
            restore_rtol=runtime_args.restore_rtol,
            snapshot_integrity_prevalidated=True,
        )
        audit = first64_equality_audit(source_record, source_payload, result)
        annotate_continued_result(result, registration_branch, audit)
        return result, audit
    finally:
        _close_branch_environment(branch_env)


def _engineering_analysis(registration: dict, output_dir: Path) -> dict:
    records = load_branch_records(output_dir / "branch_records.jsonl")
    errors = read_jsonl(output_dir / "errors.jsonl")
    snapshot_integrity = read_jsonl(
        output_dir / "snapshot_integrity_audits.jsonl"
    )
    phase2_style = analyze_phase2_replay(records, errors)
    registered = {row["branch_id"]: row for row in registration["registered_branches"]}
    record_ids = {row["branch_id"] for row in records}
    prefix_exact = bool(records) and all(row.get("first64_exact") for row in records)
    first64_audits = [
        {
            "source_branch_id": row.get("source_branch_id"),
            "continued_branch_id": row["branch_id"],
            "prefix_steps": 64,
            "channels": row.get("first64_channels") or {},
            "all_exact": bool(row.get("first64_exact")),
        }
        for row in sorted(records, key=lambda value: value["branch_id"])
    ]
    # Rebuild this derived ledger at finalization. If a process stopped after
    # atomically saving a payload but before appending the streaming audit row,
    # resume still produces a complete one-row-per-branch equality table.
    write_jsonl(output_dir / "first64_audits.jsonl", first64_audits)
    horizons_valid = True
    termination_valid = True
    allowed_termination = {
        "suffix_horizon",
        "success",
        "environment_done",
        "safety_termination",
    }
    for row in records:
        expected = int(registered[row["branch_id"]]["suffix_steps"])
        steps = int(row["num_steps"])
        reason = row["termination_reason"]
        horizons_valid &= 64 <= steps <= expected
        horizons_valid &= reason != "suffix_horizon" or steps == expected
        termination_valid &= reason in allowed_termination
    repeat_gates = {
        key: value
        for key, value in phase2_style["gates"].items()
        if key.startswith("same_seed_")
    }
    gates = {
        "frozen_source_hashes_exact": True,
        "snapshot_integrity_audit_complete": {
            row["snapshot_id"] for row in snapshot_integrity
        }
        == {
            row["snapshot_id"] for row in registration["selected_snapshots"]
        },
        "snapshot_registered_file_hashes_exact": bool(snapshot_integrity)
        and all(
            row.get("registered_file_sha256_exact")
            for row in snapshot_integrity
        ),
        "snapshot_embedded_identities_exact": bool(snapshot_integrity)
        and all(
            row.get("embedded_snapshot_id_exact") for row in snapshot_integrity
        ),
        "snapshot_integrity_records_accepted": bool(snapshot_integrity)
        and all(row.get("accepted") for row in snapshot_integrity),
        "registered_branch_support_complete": record_ids == set(registered),
        "first64_all_channels_exact": prefix_exact,
        "same_seed_repeats_exact": bool(repeat_gates)
        and all(repeat_gates.values()),
        "record_alignment_exact": phase2_style["gates"]["record_alignment_exact"],
        "zero_restore_induced_regressions": phase2_style["gates"][
            "zero_restore_induced_regressions"
        ],
        "task_specific_horizons_valid": bool(horizons_valid),
        "termination_reasons_registered": bool(termination_valid),
        "zero_branch_errors": not errors,
    }
    return {
        "schema_version": 1,
        "protocol": "phase3b_training_horizon_engineering_audit",
        "status": "complete",
        "scope": registration["scope"],
        "created_at": utc_now(),
        "parents": len({row["parent_id"] for row in records}),
        "snapshots": len({row["snapshot_id"] for row in records}),
        "branches": len(records),
        "expected_branches": registration["expected_branches"],
        "errors": len(errors),
        "snapshot_integrity": {
            "audited_snapshots": len(snapshot_integrity),
            "strict_internal_checksum_snapshots": sum(
                bool(row.get("strict_internal_checksums_exact"))
                for row in snapshot_integrity
            ),
            "legacy_schema5_compatibility_snapshots": sum(
                row.get("loading_mode")
                == "legacy_schema5_registered_file_sha256_and_embedded_identity"
                for row in snapshot_integrity
            ),
            "all_accepted": bool(snapshot_integrity)
            and all(row.get("accepted") for row in snapshot_integrity),
        },
        "engineering_gates": gates,
        "engineering_all_pass": all(gates.values()),
        "scientific_outcomes_used_as_engineering_gate": False,
        "repeat_audit": phase2_style["repeat_pairs"],
    }


def run(args, runtime=None):
    output_dir = args.output_dir.resolve()
    registration_path = output_dir / "registration.json"
    if args.resume:
        if not registration_path.is_file():
            raise ValueError(f"Resume registration is missing: {registration_path}")
        registration = json.loads(registration_path.read_text())
        requested = build_registration(
            args.phase2_run_dir,
            scope=args.scope,
            sentinel_run_dir=args.sentinel_run_dir,
        )
        immutable_keys = [
            "scope",
            "phase2_source_hashes",
            "phase2_snapshot_file_sha256",
            "phase2_branch_payload_file_sha256",
            "phase3b_code_file_sha256",
            "selected_snapshots",
            "registered_branches",
            "sentinel_provenance",
        ]
        mismatches = [key for key in immutable_keys if registration.get(key) != requested.get(key)]
        if mismatches:
            raise ValueError(f"Resume differs from frozen Phase 3B registration: {mismatches}")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        registration = build_registration(
            args.phase2_run_dir,
            scope=args.scope,
            sentinel_run_dir=args.sentinel_run_dir,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(registration_path, registration)
    if args.dry_run:
        atomic_write_json(
            output_dir / "status.json",
            {"status": "registered_dry_run", "updated_at": utc_now()},
        )
        print(json.dumps(registration, indent=2, sort_keys=True))
        return registration

    source_plan = _source_plan(registration)
    _validate_live_provenance(args, registration, source_plan)
    atomic_write_json(
        output_dir / "status.json",
        {"status": "running", "updated_at": utc_now()},
    )
    runtime = runtime or _runtime()
    factory, policy_args = _make_policy_args(args, source_plan, runtime)
    runtime_args = _runtime_args(args, source_plan)
    source_root = Path(registration["phase2_run_dir"])
    source_records = {
        row["branch_id"]: row
        for row in load_branch_records(source_root / "branch_records.jsonl")
    }
    grouped = _registered_by_snapshot(registration)
    completed = {
        row["branch_id"]
        for row in load_branch_records(output_dir / "branch_records.jsonl")
    }
    errors = read_jsonl(output_dir / "errors.jsonl")
    failed_ids = {row["branch_id"] for row in errors}
    by_parent = defaultdict(list)
    for snapshot in registration["selected_snapshots"]:
        by_parent[snapshot["parent_id"]].append(snapshot)

    for parent_id, selected_snapshots in sorted(by_parent.items()):
        first_snapshot = _load_registered_snapshot(
            source_root,
            registration,
            selected_snapshots[0],
            output_dir,
        )
        base_env = _make_fresh_branch_environment(first_snapshot, runtime_args, runtime)
        policy = None
        try:
            local_policy_args = dict(policy_args)
            local_policy_args["sampling_seed_base"] = first_snapshot.policy_state[
                "sampling_config"
            ]["sampling_seed_base"]
            policy = runtime["call_factory"](factory, base_env, local_policy_args)
            _call_with_deterministic_environment_seed(
                int(first_snapshot.metadata["environment_seed"]),
                _reset_env,
                base_env,
                int(first_snapshot.metadata["environment_seed"]),
            )
            reset_policy = getattr(policy, "reset", None)
            if callable(reset_policy):
                reset_policy()
            for selected in sorted(
                selected_snapshots,
                key=lambda row: (row["boundary"] != "prefix", row["snapshot_id"]),
            ):
                snapshot = _load_registered_snapshot(
                    source_root,
                    registration,
                    selected,
                    output_dir,
                )
                for registered in grouped[snapshot.snapshot_id]:
                    branch_id = registered["branch_id"]
                    if branch_id in completed:
                        continue
                    if branch_id in failed_ids and not args.retry_errors:
                        continue
                    try:
                        source_record = source_records[registered["source_branch_id"]]
                        source_payload = load_branch_payload(source_root, source_record)
                        result, audit = _execute_registered_branch(
                            registration_branch=registered,
                            snapshot=snapshot,
                            source_record=source_record,
                            source_payload=source_payload,
                            policy=policy,
                            runtime=runtime,
                            runtime_args=runtime_args,
                        )
                        save_branch_result(result, output_dir)
                        _append_jsonl(output_dir / "first64_audits.jsonl", audit)
                        completed.add(branch_id)
                    except Exception as error:  # preserve evidence before stopping
                        error_row = {
                            **registered,
                            "created_at": utc_now(),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                        _append_jsonl(output_dir / "errors.jsonl", error_row)
                        failed_ids.add(branch_id)
                        if args.fail_fast:
                            raise
                    atomic_write_json(
                        output_dir / "status.json",
                        {
                            "status": "running",
                            "updated_at": utc_now(),
                            "branches": len(completed),
                            "expected_branches": registration["expected_branches"],
                            "errors": len(read_jsonl(output_dir / "errors.jsonl")),
                        },
                    )
        finally:
            close = getattr(policy, "close", None)
            if callable(close):
                close()
            _close_branch_environment(base_env)

    validate_registration_source(registration)
    analysis = _engineering_analysis(registration, output_dir)
    atomic_write_json(output_dir / "analysis.json", analysis)
    atomic_write_json(
        output_dir / "status.json",
        {
            "status": "complete" if analysis["engineering_all_pass"] else "failed_gate",
            "updated_at": utc_now(),
            "scope": registration["scope"],
            "branches": analysis["branches"],
            "expected_branches": analysis["expected_branches"],
            "errors": analysis["errors"],
            "engineering_all_pass": analysis["engineering_all_pass"],
        },
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return analysis


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("sentinel", "full"), required=True)
    parser.add_argument("--sentinel-run-dir", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--policy-name", default="Xiaomi-Robotics-1-RoboCasa365")
    parser.add_argument("--policy-arg", action="append", default=[])
    parser.add_argument("--restore-atol", type=float, default=1e-8)
    parser.add_argument("--restore-rtol", type=float, default=1e-8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--fail-fast", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        if args.output_dir.is_dir():
            atomic_write_json(
                args.output_dir / "status.json",
                {
                    "status": "failed",
                    "updated_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise SystemExit(f"error: {error}") from error
    except Exception as error:
        if args.output_dir.is_dir():
            atomic_write_json(
                args.output_dir / "status.json",
                {
                    "status": "failed",
                    "updated_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    main()
