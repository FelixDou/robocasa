"""Branch ladder parents from saved snapshots at the Hide-and-Seek + P alarm (Phase A1).

Plan: docs/recovery_harvest_finetune_plan.md. Inputs: a recoverability-ladder run
(``plan.json``, ``parents.jsonl``, ``snapshots/``) and the alarm rows written by
``python -m robocasa.recovery.hs_p_alarm score-ladder``.

For every parent whose alarm fired (failures; successful parents with a false
alarm are kept as harm data), resample branches start from the saved snapshot at
the alarm (anchor ``alarm``) and ``n`` inferences before it (``alarm_minus_<n>``,
clipped at the stage entry before the alarm). They run under the task
instruction with K fresh sampling seeds, a reset policy context and the
remaining-horizon budget, through the ladder's own ``run_branch``. Cells that
start from the same snapshot alias and run once.

Scene identity. The environment is rebuilt as the ladder built it: one env per
task, then ``reset(seed)`` for every attempt seed in order (as a ladder resume
does). A parent is branched only if the reset state equals its saved first
snapshot within ``scene_tolerance``; otherwise it is recorded as a scene
mismatch and skipped.

Commands: ``plan`` (counts, no simulator), ``branch`` (simulator and policy
server), ``analyze`` (Gate A table and decision).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from robocasa.recovery import recoverability_ladder as ladder


RUNG = "resample_continue"
ALARM = "alarm"
ALARM_MINUS = "alarm_minus_"
DEFAULT_OFFSETS = (8, 16)
GATE_SHARE = 0.30
PLAN_NAME = "plan.json"
PARENTS_NAME = "parents.jsonl"
BRANCHES_NAME = "branches.jsonl"
SCENES_NAME = "scene_checks.jsonl"
ERRORS_NAME = "errors.jsonl"
STATUS_NAME = "status.json"
SOURCE_KEYS = ("base_seed", "seed_protocol", "split", "env_interface", "policy_module")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anchor_names(offsets):
    return [ALARM] + [f"{ALARM_MINUS}{int(n)}" for n in offsets]


# ---- sources and targets ----------------------------------------------------------------------

def load_sources(run_dirs):
    """Source plan config (shared across shards) and {parent_id: (run_dir, parent)}."""
    configs, parents = [], {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir).resolve()
        plan = json.loads((run_dir / ladder.PLAN_NAME).read_text())
        configs.append(plan["config"])
        for parent in ladder.read_jsonl(run_dir / ladder.PARENTS_NAME):
            if parent["parent_id"] in parents:
                raise ValueError(f"Parent {parent['parent_id']} appears in two source runs")
            parents[parent["parent_id"]] = (run_dir, parent)
    if not configs:
        raise ValueError("No source runs")
    first = configs[0]
    for config in configs[1:]:
        diffs = {key: (first.get(key), config.get(key)) for key in SOURCE_KEYS if first.get(key) != config.get(key)}
        if diffs:
            raise ValueError(f"Source runs differ in {diffs}")
    tasks = sorted({task for config in configs for task in config["tasks"]})
    return {**first, "tasks": tasks}, parents


def select_targets(alarms, parents, *, include_false_alarms=True, tasks=None):
    """Alarm rows to branch, joined with their parents; refuses stale or unknown parents."""
    targets = []
    for row in alarms:
        if not row["alarm_fired"]:
            continue
        if row["success"] and not include_false_alarms:
            continue
        if tasks and row["task_name"] not in tasks:
            continue
        if row["parent_id"] not in parents:
            raise ValueError(f"Alarm row for unknown parent {row['parent_id']}")
        run_dir, parent = parents[row["parent_id"]]
        if parent["attempt_uid"] != row["attempt_uid"]:
            raise ValueError(f"Alarm row for {row['parent_id']} is from another attempt")
        if not parent.get("snapshot_path"):
            raise ValueError(f"Parent {row['parent_id']} has no saved snapshots")
        targets.append({"run_dir": run_dir, "parent": parent, "alarm": row})
    return targets


def parent_anchors(parent, alarm_index, offsets):
    """Anchors (snapshot indices) relative to the alarm, as the ladder computes them."""
    steps = [int(s) for s in parent["snapshot_steps"]]
    is_inference = [True] * (len(steps) - 1) + [False]
    base = ladder.compute_anchors(steps, is_inference, parent["progress_events"], int(alarm_index), 2)
    stage_entry = base["last_good"]
    anchors = {"detection": int(alarm_index), "stage_entry": int(stage_entry), ALARM: int(alarm_index)}
    for n in offsets:
        anchors[f"{ALARM_MINUS}{int(n)}"] = int(max(stage_entry, int(alarm_index) - int(n)))
    return anchors, steps


def branch_config(source_config, *, k_branches, policy_args=None):
    overrides = {"budget_mode": "remaining_horizon", "k_branches": int(k_branches),
                 "policy_context": "reset", "parents_only": False}
    if policy_args:
        overrides["policy_args"] = {**(source_config.get("policy_args") or {}), **policy_args}
    return ladder.resolve_config(source_config, overrides)


def parent_schedule(target, config, offsets):
    parent, alarm = target["parent"], target["alarm"]
    anchors, steps = parent_anchors(parent, alarm["alarm_snapshot_index"], offsets)
    horizon, task = int(parent["horizon"]), parent["task_name"]
    budget_fn = lambda clock_step: ladder.branch_budget(horizon, clock_step, config, task)
    cells = [(RUNG, name) for name in anchor_names(offsets)]
    schedule = ladder.schedule_cells(cells, anchors, budget_fn=budget_fn, steps=steps)
    for cell in schedule:
        cell["anchor_step"] = int(steps[cell["anchor_index"]])
        cell["inferences_before_alarm"] = int(anchors[ALARM] - cell["anchor_index"])
    return anchors, schedule


def build_plan_rows(targets, config, offsets):
    rows = []
    for target in targets:
        anchors, schedule = parent_schedule(target, config, offsets)
        parent, alarm = target["parent"], target["alarm"]
        rows.append({
            "parent_id": parent["parent_id"],
            "attempt_uid": parent["attempt_uid"],
            "task_name": parent["task_name"],
            "environment_seed": int(parent["environment_seed"]),
            "parent_success": bool(parent["success"]),
            "outcome": "false_alarm_success" if parent["success"] else "failure",
            "source_run": str(target["run_dir"]),
            "alarm_inference": alarm["alarm_inference"],
            "alarm_snapshot_index": alarm["alarm_snapshot_index"],
            "alarm_component": alarm["component"],
            "alarm_fraction": alarm["alarm_fraction"],
            "anchors": anchors,
            "cells": schedule,
        })
    return rows


# ---- simulator side ---------------------------------------------------------------------------

def load_rollout(run_dir, parent):
    """The parts of a ladder rollout that ``run_branch`` reads, from saved artifacts."""
    with np.load(Path(run_dir) / parent["snapshot_path"], allow_pickle=False) as data:
        steps = np.asarray(data["steps"]).tolist()
        states, qpos, qvel = data["states"], data["qpos"], data["qvel"]
        ordered, durable = data["ordered_count"], data["durable_count"]
        is_inference = np.asarray(data["is_inference"], dtype=bool).tolist()
        snapshots = [
            {
                "states": np.asarray(states[i], dtype=np.float64),
                "qpos": np.asarray(qpos[i], dtype=np.float64),
                "qvel": np.asarray(qvel[i], dtype=np.float64),
                "step": int(steps[i]),
                "ordered_count": int(ordered[i]),
                "durable_count": int(durable[i]),
                "is_inference": bool(is_inference[i]),
                # Not saved; only re-prompt branches read it, and this module runs none.
                "regressed": [],
            }
            for i in range(len(steps))
        ]
    if [int(s) for s in steps] != [int(s) for s in parent["snapshot_steps"]]:
        raise ValueError(f"{parent['parent_id']}: saved snapshot steps differ from parents.jsonl")
    if is_inference != [True] * (len(steps) - 1) + [False]:
        raise ValueError(f"{parent['parent_id']}: unexpected snapshot layout (stride or terminal)")
    return {
        "snapshots": snapshots,
        "instruction": parent.get("instruction"),
        "required_subtasks": list(parent["required_subtasks"]),
        "predicate_stages": parent.get("predicate_stages") or {},
        "policy_states": [None] * len(snapshots),
    }


def _write_status(output_dir, state, rows, branches, scenes, errors):
    by_task = defaultdict(Counter)
    for row in rows:
        by_task[row["task_name"]][row["outcome"]] += 1
    for scene in scenes:
        by_task[scene["task_name"]]["scene_ok" if scene["ok"] else "scene_mismatch"] += 1
    for branch in branches:
        by_task[branch["task_name"]]["branches"] += 1
        by_task[branch["task_name"]]["branch_successes"] += int(branch["success"])
    ladder.atomic_write_json(output_dir / STATUS_NAME, {
        "status": state, "updated_at": ladder.utc_now(), "num_branches": len(branches),
        "num_errors": len(errors), "by_task": {task: dict(c) for task, c in sorted(by_task.items())},
    })


def run_branching(source_runs, alarms_path, output_dir, *, k_branches=8, offsets=DEFAULT_OFFSETS,
                  include_false_alarms=True, tasks=None, scene_tolerance=1e-6, policy_args=None,
                  runtime=None, host=None, port=None, shard_index=0, num_shards=1, resume=False,
                  max_errors=10, log=print):
    output_dir = Path(output_dir)
    source_config, parents = load_sources(source_runs)
    config = branch_config(source_config, k_branches=k_branches, policy_args=policy_args)
    alarms = ladder.read_jsonl(alarms_path)
    targets = select_targets(alarms, parents, include_false_alarms=include_false_alarms, tasks=tasks)
    runtime = runtime or ladder.Runtime(config, host=host, port=port)
    horizons = {t["parent"]["task_name"]: int(t["parent"]["horizon"]) for t in targets}
    shard = set(ladder.shard_tasks(sorted(horizons), horizons, int(num_shards), int(shard_index)))
    targets = [t for t in targets if t["parent"]["task_name"] in shard]
    rows = build_plan_rows(targets, config, offsets)
    plan = {
        "kind": "alarm_branching_plan",
        "plan_doc": "docs/recovery_harvest_finetune_plan.md",
        "source_runs": [str(Path(d).resolve()) for d in source_runs],
        "alarms": {"path": str(Path(alarms_path).resolve()), "sha256": _sha256(alarms_path)},
        "k_branches": int(k_branches),
        "offsets": [int(n) for n in offsets],
        "include_false_alarms": bool(include_false_alarms),
        "scene_tolerance": float(scene_tolerance),
        "rung": RUNG,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "tasks": sorted(shard),
        "parents": [row["parent_id"] for row in rows],
        "config": {key: config[key] for key in ("budget_mode", "max_budget_steps", "policy_context",
                                                "policy_module", "base_seed", "seed_protocol", "split",
                                                "env_interface", "snapshot_stride")},
        "git_commit": ladder.current_git_commit(),
        "created_at": ladder.utc_now(),
    }
    plan_path = output_dir / PLAN_NAME
    frozen = ("source_runs", "alarms", "k_branches", "offsets", "include_false_alarms", "scene_tolerance",
              "shard_index", "num_shards", "parents")
    if plan_path.exists():
        if not resume:
            raise FileExistsError(f"{plan_path} exists; pass --resume or choose a new output dir")
        previous = json.loads(plan_path.read_text())
        diffs = {key: (previous.get(key), plan[key]) for key in frozen if previous.get(key) != plan[key]}
        if diffs:
            raise ValueError(f"Resume differs from the registered plan: {sorted(diffs)}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        ladder.atomic_write_json(plan_path, plan)
        with (output_dir / PARENTS_NAME).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, default=ladder.json_default) + "\n")

    branches = ladder.read_jsonl(output_dir / BRANCHES_NAME)
    scenes = ladder.read_jsonl(output_dir / SCENES_NAME)
    errors = ladder.read_jsonl(output_dir / ERRORS_NAME)
    done = {(b["parent_id"], b["anchor"], int(b["k"])) for b in branches}
    _write_status(output_dir, "running", rows, branches, scenes, errors)
    by_task = defaultdict(dict)
    for target, row in zip(targets, rows):
        by_task[row["task_name"]][int(row["environment_seed"])] = (target, row)
    base_seed = int(config["base_seed"])
    protocol = config["seed_protocol"]

    for task in sorted(by_task):
        wanted = by_task[task]
        env = policy = None

        def rebuild(upto_seed):
            nonlocal env, policy
            ladder._close(env, policy)
            env = runtime.make_env(task, base_seed)
            policy = runtime.make_policy(env)
            if protocol == "repeated_reset":
                for seed in range(base_seed, upto_seed):
                    env.reset(seed=seed)

        try:
            for seed in range(base_seed, max(wanted) + 1):
                if protocol == "fresh_env":
                    if seed not in wanted:
                        continue
                    ladder._close(env, policy)
                    env = runtime.make_env(task, seed)
                    policy = runtime.make_policy(env)
                    env.reset()
                else:
                    if env is None:
                        rebuild(base_seed)
                    env.reset(seed=seed)
                if seed not in wanted:
                    continue
                target, row = wanted[seed]
                pid = row["parent_id"]
                cells = [cell for cell in row["cells"] if cell["run"]]
                todo = [(cell, k) for cell in cells for k in range(int(k_branches))
                        if (pid, cell["anchor"], k) not in done]
                if not todo:
                    continue
                if len(errors) >= int(max_errors):
                    raise RuntimeError(f"Stopping after {len(errors)} errors; inspect {ERRORS_NAME}")
                try:
                    rollout = load_rollout(target["run_dir"], target["parent"])
                    residual = ladder.state_residual(env, rollout["snapshots"][0]["states"])
                    scene = {"parent_id": pid, "task_name": task, "environment_seed": seed,
                             "residual": residual, "ok": bool(residual <= scene_tolerance),
                             "created_at": ladder.utc_now()}
                    ladder.append_jsonl(output_dir / SCENES_NAME, scene)
                    scenes.append(scene)
                    if not scene["ok"]:
                        log(f"[alarm-branch] {task} seed={seed} scene mismatch (residual {residual:.3g}); skipped")
                        continue
                    parent = dict(target["parent"])
                    parent["anchors"] = row["anchors"]
                    parent["role"] = row["outcome"]
                    alarm_count = rollout["snapshots"][row["anchors"][ALARM]]["ordered_count"]
                    parent["stage_at_detection"] = ladder.stage_label(rollout["required_subtasks"], int(alarm_count))
                    for cell, k in todo:
                        record = ladder.run_branch(env, policy, parent=parent, rollout=rollout, cell=cell, k=k,
                                                   config=config, task_name=task)
                        record.update({
                            "source_run": row["source_run"],
                            "outcome": row["outcome"],
                            "alarm_inference": row["alarm_inference"],
                            "alarm_snapshot_index": row["alarm_snapshot_index"],
                            "alarm_component": row["alarm_component"],
                            "inferences_before_alarm": cell["inferences_before_alarm"],
                            "scene_residual": residual,
                        })
                        ladder.append_jsonl(output_dir / BRANCHES_NAME, record)
                        branches.append(record)
                        done.add((pid, cell["anchor"], k))
                    _write_status(output_dir, "running", rows, branches, scenes, errors)
                    successes = sum(b["success"] for b in branches if b["parent_id"] == pid)
                    log(f"[alarm-branch] {task} seed={seed} {row['outcome']} alarm@{row['alarm_snapshot_index']} "
                        f"successes={successes}/{len(cells) * int(k_branches)}")
                except KeyboardInterrupt:
                    raise
                except Exception:
                    event = {"task_name": task, "environment_seed": seed, "parent_id": pid,
                             "error": traceback.format_exc(), "created_at": ladder.utc_now()}
                    ladder.append_jsonl(output_dir / ERRORS_NAME, event)
                    errors.append(event)
                    log(f"[alarm-branch] error on {task} seed {seed}; see {ERRORS_NAME}")
                    if protocol == "repeated_reset":
                        rebuild(seed + 1)
        finally:
            ladder._close(env, policy)
    _write_status(output_dir, "complete", rows, branches, scenes, errors)
    return {"branches": len(branches), "errors": len(errors)}


# ---- plan (no simulator) ----------------------------------------------------------------------

def plan_summary(source_runs, alarms_path, *, k_branches=8, offsets=DEFAULT_OFFSETS, include_false_alarms=True,
                 tasks=None, seconds_per_env_step=0.055, seconds_per_reset=20.0):
    source_config, parents = load_sources(source_runs)
    config = branch_config(source_config, k_branches=k_branches)
    targets = select_targets(ladder.read_jsonl(alarms_path), parents,
                             include_false_alarms=include_false_alarms, tasks=tasks)
    rows = build_plan_rows(targets, config, offsets)
    out = {"k_branches": int(k_branches), "offsets": [int(n) for n in offsets], "by_task": {}}
    total_steps = total_branches = resets = 0
    for task in sorted({r["task_name"] for r in rows}):
        task_rows = [r for r in rows if r["task_name"] == task]
        cells = [c for r in task_rows for c in r["cells"] if c["run"]]
        steps = sum(int(c["budget_steps"]) for c in cells) * int(k_branches)
        attempts = max(int(r["environment_seed"]) for r in task_rows) - int(config["base_seed"]) + 1
        out["by_task"][task] = {
            "failures": sum(r["outcome"] == "failure" for r in task_rows),
            "false_alarm_successes": sum(r["outcome"] != "failure" for r in task_rows),
            "branches": len(cells) * int(k_branches),
            "budget_steps_upper_bound": steps,
        }
        total_steps += steps
        total_branches += len(cells) * int(k_branches)
        resets += attempts
    out["branches"] = total_branches
    out["worker_hours_upper_bound"] = round(
        (total_steps * seconds_per_env_step + resets * seconds_per_reset) / 3600.0, 1)
    return out


# ---- Gate A -----------------------------------------------------------------------------------

def _bootstrap(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(values, len(values)))) for _ in range(int(n_boot))]
    return [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]


def cell_outcomes(run_dirs):
    """{(parent_id, anchor): [success, ...]} with aliased anchors resolved, plus parent rows."""
    rows, branches, scenes = {}, [], {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        for row in ladder.read_jsonl(run_dir / PARENTS_NAME):
            rows[row["parent_id"]] = row
        branches.extend(ladder.read_jsonl(run_dir / BRANCHES_NAME))
        for scene in ladder.read_jsonl(run_dir / SCENES_NAME):
            scenes[scene["parent_id"]] = scene
    by_cell = defaultdict(list)
    for branch in branches:
        row = rows.get(branch["parent_id"])
        if row is not None and row["attempt_uid"] == branch["attempt_uid"]:
            by_cell[(branch["parent_id"], branch["anchor"])].append(bool(branch["success"]))
    outcomes = {}
    for pid, row in rows.items():
        if not scenes.get(pid, {}).get("ok"):
            continue
        for cell in row["cells"]:
            canonical = cell["canonical"][1]
            values = by_cell.get((pid, canonical))
            if values:
                outcomes[(pid, cell["anchor"])] = values
    return rows, outcomes, scenes


def gate_a(run_dirs, *, k_branches=None, n_boot=2000, seed=0):
    rows, outcomes, scenes = cell_outcomes(run_dirs)
    anchors = sorted({anchor for _, anchor in outcomes}, key=lambda a: (a != ALARM, a))
    table = {}
    for outcome in ("failure", "false_alarm_success"):
        for anchor in anchors:
            pids = [pid for pid, row in rows.items() if row["outcome"] == outcome and (pid, anchor) in outcomes]
            if k_branches is not None:
                incomplete = [pid for pid in pids if len(outcomes[(pid, anchor)]) < int(k_branches)]
                if incomplete:
                    raise ValueError(f"{len(incomplete)} parents have fewer than {k_branches} branches at {anchor}")
            any_success = [float(any(outcomes[(pid, anchor)])) for pid in pids]
            mean_success = [float(np.mean(outcomes[(pid, anchor)])) for pid in pids]
            by_task = defaultdict(list)
            for pid, value in zip(pids, any_success):
                by_task[rows[pid]["task_name"]].append(value)
            table[f"{outcome}/{anchor}"] = {
                "parents": len(pids),
                "share_with_any_success": round(float(np.mean(any_success)), 4) if pids else None,
                "share_ci95": _bootstrap(any_success, n_boot, seed),
                "mean_success": round(float(np.mean(mean_success)), 4) if pids else None,
                "mean_success_ci95": _bootstrap(mean_success, n_boot, seed),
                "by_task_share_with_any_success": {t: round(float(np.mean(v)), 4) for t, v in sorted(by_task.items())},
            }
    share = lambda anchor: (table.get(f"failure/{anchor}") or {}).get("share_with_any_success")
    early = [a for a in (ALARM, f"{ALARM_MINUS}8") if share(a) is not None]
    if any(share(a) >= GATE_SHARE for a in early):
        decision = "harvest_at_alarm_and_earlier"
    elif share(f"{ALARM_MINUS}16") is not None and share(f"{ALARM_MINUS}16") >= GATE_SHARE:
        decision = "harvest_at_alarm_minus_16_only"
    else:
        decision = "redesign_expert_data"
    return {
        "gate": "A",
        "rule": f"share of alarmed failure states with >= 1 success; threshold {GATE_SHARE}",
        "decision": decision,
        "scene_checks": dict(Counter("ok" if s["ok"] else "mismatch" for s in scenes.values())),
        "parents_by_outcome": dict(Counter(row["outcome"] for row in rows.values())),
        "cells": table,
    }


def write_gate_a(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ladder.atomic_write_json(output_dir / "gate_a.json", result)
    lines = [
        "# Gate A: recoverability at the Hide-and-Seek + P alarm",
        "",
        f"Decision: **{result['decision']}** ({result['rule']}).",
        f"Scene checks: {result['scene_checks']}. Parents: {result['parents_by_outcome']}.",
        "",
        "| Parents | Anchor | n | Share with any success [95% CI] | Mean success [95% CI] |",
        "|---|---|---:|---|---|",
    ]
    for key, cell in result["cells"].items():
        outcome, anchor = key.split("/")
        fmt = lambda v, ci: "—" if v is None else f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {outcome} | {anchor} | {cell['parents']} | "
                     f"{fmt(cell['share_with_any_success'], cell['share_ci95'])} | "
                     f"{fmt(cell['mean_success'], cell['mean_success_ci95'])} |")
    path = output_dir / "gate_a.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---- CLI --------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--source-run", nargs="+", type=Path, required=True, help="ladder shard dirs")
        p.add_argument("--alarms", type=Path, required=True, help="hs_p_alarm score-ladder output")
        p.add_argument("--k-branches", type=int, default=8)
        p.add_argument("--offsets", nargs="+", type=int, default=list(DEFAULT_OFFSETS))
        p.add_argument("--no-false-alarms", action="store_true", help="branch failures only")
        p.add_argument("--tasks", nargs="+")

    plan = sub.add_parser("plan", help="branch counts and an upper-bound cost, no simulator")
    common(plan)
    branch = sub.add_parser("branch", help="run the branches (simulator + policy server)")
    common(branch)
    branch.add_argument("--output-dir", type=Path, required=True)
    branch.add_argument("--scene-tolerance", type=float, default=1e-6)
    branch.add_argument("--policy-arg", action="append", default=[])
    branch.add_argument("--host")
    branch.add_argument("--port", type=int)
    branch.add_argument("--shard-index", type=int, default=0)
    branch.add_argument("--num-shards", type=int, default=1)
    branch.add_argument("--resume", action="store_true")
    branch.add_argument("--max-errors", type=int, default=10)
    analysis = sub.add_parser("analyze", help="Gate A table and decision")
    analysis.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--k-branches", type=int, default=8)
    analysis.add_argument("--n-boot", type=int, default=2000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        out = plan_summary(args.source_run, args.alarms, k_branches=args.k_branches, offsets=args.offsets,
                           include_false_alarms=not args.no_false_alarms, tasks=args.tasks)
        print(json.dumps(out, indent=2))
        return out
    if args.command == "branch":
        return run_branching(
            args.source_run, args.alarms, args.output_dir, k_branches=args.k_branches, offsets=args.offsets,
            include_false_alarms=not args.no_false_alarms, tasks=args.tasks,
            scene_tolerance=args.scene_tolerance, policy_args=ladder._parse_policy_args(args.policy_arg),
            host=args.host, port=args.port, shard_index=args.shard_index, num_shards=args.num_shards,
            resume=args.resume, max_errors=args.max_errors, log=lambda message: print(message, flush=True),
        )
    result = gate_a(args.run_dirs, k_branches=args.k_branches, n_boot=args.n_boot)
    path = write_gate_a(result, args.output_dir)
    print(path.read_text())
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
