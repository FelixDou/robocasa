"""R0 failure audit: what the alarmed failure states look like (plan step R0).

Plan: docs/recovery_harvest_finetune_plan.md, "Redesign after Gate A". Inputs:
the ladder source runs, the alarm rows from ``hs_p_alarm score-ladder`` and the
Phase A1 branch shards (to know which parents resampling already recovers).

``audit`` rebuilds each scene as Phase A1 did (simulator only, no policy
server), checks it against the saved first snapshot, and restores three saved
states per alarmed failure: the first snapshot, the alarm and the terminal
snapshot. At each it records, for every task object (distractors excluded):
position, uprightness, grasp, counter contact and whether it left the scene; at
the alarm it also records the live predicates, the predicates that had latched
and are false again, and the policy camera images. Each state gets a
provisional class:

1. ``recovered_by_resampling``: some Phase A1 branch of this parent succeeded;
2. ``recoverable``: nothing below applies;
3. ``unrecoverable``: a task object is on the floor or outside the scene.

Tipped objects, undone subtasks and grasps are listed as flags, not classes:
they make a recovery harder but not impossible. ``review.csv`` has an empty
``manual_class`` column; ``decide`` uses a filled-in value over the
provisional class and computes Gate R0.

Commands: ``audit`` (simulator), ``decide`` (no simulator).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

from robocasa.recovery import alarm_branching as ab
from robocasa.recovery import recoverability_ladder as ladder


CLASSES = ("recovered_by_resampling", "recoverable", "unrecoverable")
CLASS_NUMBERS = {"1": CLASSES[0], "2": CLASSES[1], "3": CLASSES[2]}
GATE_R0_MIN_RECOVERABLE = 10
DISTRACTOR_PREFIXES = ("distr", "dstr")
DEFAULT_FLOOR_Z = 0.3
TIPPED_COS = math.cos(math.radians(45.0))
UPRIGHT_COS = 0.9
STATES_NAME = "states.jsonl"
SCENES_NAME = "scene_checks.jsonl"
ERRORS_NAME = "errors.jsonl"
REVIEW_CSV = "review.csv"
REVIEW_MD = "review.md"
MANIFEST_NAME = "audit.json"
REVIEW_COLUMNS = ("parent_id", "task_name", "stage_at_alarm", "atomic_task", "provisional_class",
                  "flags", "manual_class", "note")
SNAPSHOT_LABELS = ("initial", "alarm", "final")


# ---- scene facts --------------------------------------------------------------------------------

def _robocasa_env(env):
    for node in ladder._wrapper_chain(env):
        if hasattr(node, "obj_body_id") and hasattr(node, "objects"):
            return node
    raise RuntimeError("No RoboCasa kitchen env (obj_body_id) beneath the wrappers")


def task_objects(names):
    return [n for n in names if not str(n).startswith(DISTRACTOR_PREFIXES)]


def inspect_scene(env, floor_z=DEFAULT_FLOOR_Z):
    """Object and gripper facts of the current simulator state (RoboCasa kitchen env)."""
    from robocasa.utils import object_utils as OU

    kitchen = _robocasa_env(env)
    data = kitchen.sim.data
    robot = kitchen.robots[0]
    eef = np.asarray(data.site_xpos[robot.eef_site_id["right"]], dtype=float)
    base = None
    try:
        base = np.asarray(data.get_body_xpos(robot.robot_model.root_body), dtype=float)
    except Exception:
        pass
    objects = {}
    for name in task_objects(list(kitchen.objects)):
        pos = np.asarray(data.body_xpos[kitchen.obj_body_id[name]], dtype=float)

        def safe(fn, default=None):
            try:
                return fn()
            except Exception:
                return default

        objects[name] = {
            "pos": [round(float(v), 4) for v in pos],
            "up_cos": safe(lambda: round(float(OU.obj_cos(kitchen, name)), 4)),
            "grasped": safe(lambda: bool(OU.check_obj_grasped(kitchen, name))),
            "counter_contact": safe(lambda: bool(OU.check_obj_any_counter_contact(kitchen, name))),
            "outside_scene": safe(lambda: bool(OU.point_outside_scene(kitchen, pos))),
            "on_floor": bool(pos[2] < float(floor_z)),
            "eef_dist": round(float(np.linalg.norm(eef - pos)), 4),
            "base_xy_dist": None if base is None else round(float(np.linalg.norm((base - pos)[:2])), 4),
        }
    return {"objects": objects, "eef_pos": [round(float(v), 4) for v in eef],
            "holding": sorted(n for n, o in objects.items() if o["grasped"])}


def camera_images(env):
    """HxWx3 uint8 arrays of the live observation (the policy's cameras)."""
    obs = ladder.live_observation(env)
    images = {}
    for key, value in obs.items():
        array = np.asarray(value) if isinstance(value, np.ndarray) else None
        if array is not None and array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array is not None and array.ndim == 3 and array.shape[-1] == 3 and array.dtype == np.uint8:
            images[str(key)] = array
    return images


def save_images(images, out_dir, stem):
    paths = {}
    if not images:
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - PIL ships with the simulator envs
        Image = None
    for key, array in sorted(images.items()):
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        if Image is not None:
            path = out_dir / f"{stem}__{safe_key}.png"
            Image.fromarray(array).save(path)
        else:
            path = out_dir / f"{stem}__{safe_key}.npy"
            np.save(path, array)
        paths[key] = path.name
    return paths


# ---- predicates and stages ----------------------------------------------------------------------

def load_predicates(run_dir, parent):
    with np.load(Path(run_dir) / parent["snapshot_path"], allow_pickle=False) as data:
        names = [str(n) for n in data["predicate_names"]] if "predicate_names" in data else []
        values = np.asarray(data["predicate_values"], dtype=bool) if "predicate_values" in data else None
    return names, values


def regressed_at(names, values, index, required):
    """Required, non-release predicates that were true at some snapshot <= index and are false at it."""
    if values is None or not len(names) or values.shape[0] <= index:
        return []
    column = {name: i for i, name in enumerate(names)}
    out = []
    for name in required:
        if name not in column or ladder.is_release_predicate(name):
            continue
        history = values[: index + 1, column[name]]
        if history.any() and not history[-1]:
            out.append(name)
    return out


def live_predicates(env):
    from robocasa.recovery.subtask_eval import get_subtask_eval

    payload = get_subtask_eval(env) or {}
    return {name: bool((p or {}).get("value", False)) for name, p in (payload.get("predicates") or {}).items()}


def atomic_step(task_name, stage):
    """The registered or derived atomic task whose predicates contain ``stage``."""
    try:
        from robocasa.recovery import eval_composite_predicates as ecp
    except Exception:  # pragma: no cover - module is numpy-only
        return None, None
    for step in (getattr(ecp, "_COMPOSITE_ATOMIC_TASK_OVERRIDES", {}) or {}).get(task_name, []):
        if stage in (step.get("predicate_names") or []):
            return step.get("atomic_task"), step.get("atomic_task_source")
    return None, None


# ---- classification -----------------------------------------------------------------------------

def recovered_parents(branch_runs):
    """{parent_id: number of successful Phase A1 branches} over every anchor."""
    hits = defaultdict(int)
    for run_dir in branch_runs or []:
        for branch in ladder.read_jsonl(Path(run_dir) / ab.BRANCHES_NAME):
            hits[branch["parent_id"]] += int(bool(branch["success"]))
    return dict(hits)


def flags_for(facts, initial, regressed):
    flags = []
    initial_objects = (initial or {}).get("objects", {})
    for name, obj in sorted(facts["objects"].items()):
        start = initial_objects.get(name) or {}
        if obj.get("on_floor"):
            flags.append(f"{name}:on_floor")
        if obj.get("outside_scene"):
            flags.append(f"{name}:outside_scene")
        if (obj.get("up_cos") is not None and obj["up_cos"] < TIPPED_COS
                and (start.get("up_cos") is None or start["up_cos"] >= UPRIGHT_COS)):
            flags.append(f"{name}:tipped")
        if obj.get("grasped"):
            flags.append(f"{name}:grasped")
    flags.extend(f"undone:{name}" for name in regressed)
    return flags


def provisional_class(recovered_branches, flags):
    if recovered_branches:
        return CLASSES[0]
    if any(f.endswith((":on_floor", ":outside_scene")) for f in flags):
        return CLASSES[2]
    return CLASSES[1]


# ---- audit ------------------------------------------------------------------------------------

def _done_ids(path):
    return {row["parent_id"] for row in ladder.read_jsonl(path)}


def run_audit(source_runs, alarms_path, output_dir, *, branch_runs=(), tasks=None, scene_tolerance=1e-6,
              floor_z=DEFAULT_FLOOR_Z, images=True, runtime=None, inspector=None, imager=None,
              resume=False, max_errors=10, log=print):
    output_dir = Path(output_dir)
    source_config, parents = ab.load_sources(source_runs)
    config = ladder.resolve_config(source_config)
    alarms = ladder.read_jsonl(alarms_path)
    targets = [t for t in ab.select_targets(alarms, parents, include_false_alarms=False, tasks=tasks)]
    recovered = recovered_parents(branch_runs)
    if branch_runs and not recovered:
        raise ValueError("Branch runs given but no branches found")
    manifest = {
        "kind": "r0_failure_audit",
        "plan_doc": "docs/recovery_harvest_finetune_plan.md",
        "source_runs": [str(Path(d).resolve()) for d in source_runs],
        "alarms": {"path": str(Path(alarms_path).resolve()), "sha256": ab._sha256(alarms_path)},
        "branch_runs": [str(Path(d).resolve()) for d in branch_runs],
        "parents": sorted(t["parent"]["parent_id"] for t in targets),
        "scene_tolerance": float(scene_tolerance),
        "floor_z": float(floor_z),
        "git_commit": ladder.current_git_commit(),
        "created_at": ladder.utc_now(),
    }
    manifest_path = output_dir / MANIFEST_NAME
    frozen = ("source_runs", "alarms", "branch_runs", "parents", "scene_tolerance", "floor_z")
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; pass --resume or choose a new output dir")
        previous = json.loads(manifest_path.read_text())
        diffs = sorted(k for k in frozen if previous.get(k) != manifest[k])
        if diffs:
            raise ValueError(f"Resume differs from the recorded audit: {diffs}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        ladder.atomic_write_json(manifest_path, manifest)

    inspector = inspector or (lambda env: inspect_scene(env, floor_z=floor_z))
    imager = imager or camera_images
    runtime = runtime or ladder.Runtime(config)
    done = _done_ids(output_dir / STATES_NAME)
    errors = ladder.read_jsonl(output_dir / ERRORS_NAME)
    by_task = defaultdict(dict)
    for target in targets:
        by_task[target["parent"]["task_name"]][int(target["parent"]["environment_seed"])] = target
    base_seed = int(config["base_seed"])
    protocol = config["seed_protocol"]

    for task in sorted(by_task):
        wanted = by_task[task]
        if all(t["parent"]["parent_id"] in done for t in wanted.values()):
            continue
        env = None
        try:
            for seed in range(base_seed, max(wanted) + 1):
                if protocol == "fresh_env":
                    if seed not in wanted:
                        continue
                    ladder._close(env, None)
                    env = runtime.make_env(task, seed)
                    env.reset()
                else:
                    if env is None:
                        env = runtime.make_env(task, base_seed)
                    env.reset(seed=seed)
                if seed not in wanted or wanted[seed]["parent"]["parent_id"] in done:
                    continue
                if len(errors) >= int(max_errors):
                    raise RuntimeError(f"Stopping after {len(errors)} errors; inspect {ERRORS_NAME}")
                target = wanted[seed]
                try:
                    row = audit_state(env, target, recovered, output_dir, scene_tolerance=scene_tolerance,
                                      images=images, inspector=inspector, imager=imager)
                except Exception as exc:  # keep going; the error is recorded for inspection
                    error = {"parent_id": target["parent"]["parent_id"], "task_name": task,
                             "error": repr(exc), "traceback": traceback.format_exc(),
                             "created_at": ladder.utc_now()}
                    ladder.append_jsonl(output_dir / ERRORS_NAME, error)
                    errors.append(error)
                    log(f"[r0-audit] {task} seed={seed} error: {exc!r}")
                    ladder._close(env, None)
                    env = runtime.make_env(task, base_seed)
                    if protocol != "fresh_env":
                        for replay in range(base_seed, seed + 1):
                            env.reset(seed=replay)
                    continue
                if row is None:
                    log(f"[r0-audit] {task} seed={seed} scene mismatch; skipped")
                    continue
                ladder.append_jsonl(output_dir / STATES_NAME, row)
                done.add(row["parent_id"])
                log(f"[r0-audit] {task} seed={seed} stage={row['stage_at_alarm']} "
                    f"class={row['provisional_class']} flags={','.join(row['flags']) or '-'}")
        finally:
            ladder._close(env, None)
    rows = ladder.read_jsonl(output_dir / STATES_NAME)
    write_review(rows, output_dir)
    summary = {"states": len(rows), "errors": len(errors), "targets": len(targets),
               "provisional": dict(Counter(r["provisional_class"] for r in rows))}
    log(json.dumps(summary, indent=2))
    return summary


def audit_state(env, target, recovered, output_dir, *, scene_tolerance, images, inspector, imager):
    parent, alarm = target["parent"], target["alarm"]
    pid = parent["parent_id"]
    rollout = ab.load_rollout(target["run_dir"], parent)
    snaps = rollout["snapshots"]
    residual = ladder.state_residual(env, snaps[0]["states"])
    scene = {"parent_id": pid, "task_name": parent["task_name"], "residual": residual,
             "ok": bool(residual <= scene_tolerance), "created_at": ladder.utc_now()}
    ladder.append_jsonl(Path(output_dir) / SCENES_NAME, scene)
    if not scene["ok"]:
        return None
    a = int(alarm["alarm_snapshot_index"])
    indices = {"initial": 0, "alarm": a, "final": len(snaps) - 1}
    facts, restore, image_paths, live = {}, {}, {}, {}
    for label in SNAPSHOT_LABELS:
        ladder.restore_world(env, snaps[indices[label]])
        restore[label] = ladder.state_residual(env, snaps[indices[label]]["states"])
        facts[label] = inspector(env)
        if label == "alarm":
            live = live_predicates(env)
        if images and label in ("alarm", "final"):
            image_paths[label] = save_images(imager(env), Path(output_dir) / "images", f"{pid}__{label}")
    required = list(rollout["required_subtasks"])
    count = int(snaps[a]["ordered_count"])
    stage = ladder.stage_label(required, count)
    names, values = load_predicates(target["run_dir"], parent)
    regressed = regressed_at(names, values, a, required)
    saved = {}
    if values is not None and values.shape[0] > a:
        saved = {name: bool(values[a, i]) for i, name in enumerate(names)}
    mismatch = sorted(n for n in saved if n in live and saved[n] != live[n])
    flags = flags_for(facts["alarm"], facts["initial"], regressed)
    atomic_task, atomic_source = atomic_step(parent["task_name"], stage)
    hits = int(recovered.get(pid, 0))
    return {
        "parent_id": pid,
        "task_name": parent["task_name"],
        "environment_seed": int(parent["environment_seed"]),
        "source_run": str(target["run_dir"]),
        "alarm_snapshot_index": a,
        "alarm_step": int(snaps[a]["step"]),
        "alarm_fraction": alarm.get("alarm_fraction"),
        "alarm_component": alarm.get("component"),
        "required_subtasks": required,
        "ordered_count_at_alarm": count,
        "stage_at_alarm": stage,
        "atomic_task": atomic_task,
        "atomic_task_source": atomic_source,
        "regressed_at_alarm": regressed,
        "live_predicates_at_alarm": live,
        "saved_live_predicate_mismatch": mismatch,
        "scene_residual": residual,
        "restore_residuals": restore,
        "facts": facts,
        "images": image_paths,
        "a1_successful_branches": hits,
        "flags": flags,
        "provisional_class": provisional_class(hits, flags),
        "created_at": ladder.utc_now(),
    }


def write_review(rows, output_dir):
    output_dir = Path(output_dir)
    with (output_dir / REVIEW_CSV).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["task_name"], r["parent_id"])):
            writer.writerow({"parent_id": row["parent_id"], "task_name": row["task_name"],
                             "stage_at_alarm": row["stage_at_alarm"], "atomic_task": row["atomic_task"],
                             "provisional_class": row["provisional_class"], "flags": ";".join(row["flags"]),
                             "manual_class": "", "note": ""})
    lines = ["# R0 failure audit: review sheet", "",
             "Fill `manual_class` in review.csv (1, 2, 3 or a class name) where the images disagree "
             "with the provisional class; leave it empty otherwise.", ""]
    for row in sorted(rows, key=lambda r: (r["task_name"], r["parent_id"])):
        lines += [f"## {row['task_name']} `{row['parent_id']}`", "",
                  f"- Stage at alarm: `{row['stage_at_alarm']}` (atomic task {row['atomic_task']}, "
                  f"{row['atomic_task_source']}); alarm at {row['alarm_fraction']} of H by {row['alarm_component']}",
                  f"- Provisional class: **{row['provisional_class']}**; A1 successful branches: "
                  f"{row['a1_successful_branches']}",
                  f"- Flags: {', '.join(row['flags']) or 'none'}",
                  f"- Holding at alarm: {', '.join(row['facts']['alarm']['holding']) or 'nothing'}", ""]
        objects = row["facts"]["alarm"]["objects"]
        if objects:
            lines += ["| Object | z | up cos | grasped | counter contact | eef dist |", "|---|---:|---:|---|---|---:|"]
            for name, obj in sorted(objects.items()):
                lines.append(f"| {name} | {obj['pos'][2]:.3f} | {obj['up_cos']} | {obj['grasped']} | "
                             f"{obj['counter_contact']} | {obj['eef_dist']:.3f} |")
            lines.append("")
        for label in ("alarm", "final"):
            for key, name in sorted((row.get("images") or {}).get(label, {}).items()):
                lines.append(f"{label} `{key}`: ![{label} {key}](images/{name})")
        lines.append("")
    (output_dir / REVIEW_MD).write_text("\n".join(lines))


# ---- decide -----------------------------------------------------------------------------------

def _manual_class(value):
    value = (value or "").strip()
    if not value:
        return None
    if value in CLASS_NUMBERS:
        return CLASS_NUMBERS[value]
    if value in CLASSES:
        return value
    raise ValueError(f"Unknown manual_class {value!r}; use 1, 2, 3 or one of {CLASSES}")


def decide(audit_dir, review_path=None):
    audit_dir = Path(audit_dir)
    rows = ladder.read_jsonl(audit_dir / STATES_NAME)
    if not rows:
        raise ValueError(f"No audited states in {audit_dir}")
    manual = {}
    if review_path:
        with Path(review_path).open() as stream:
            for line in csv.DictReader(stream):
                cls = _manual_class(line.get("manual_class"))
                if cls is not None:
                    manual[line["parent_id"]] = (cls, (line.get("note") or "").strip())
    unknown = sorted(set(manual) - {r["parent_id"] for r in rows})
    if unknown:
        raise ValueError(f"Review rows for unaudited parents: {unknown}")
    final = []
    for row in rows:
        cls, note = manual.get(row["parent_id"], (row["provisional_class"], ""))
        if row["a1_successful_branches"] and cls != CLASSES[0]:
            raise ValueError(f"{row['parent_id']} has successful A1 branches; it stays {CLASSES[0]}")
        final.append({**row, "final_class": cls, "manual": row["parent_id"] in manual, "note": note})
    counts = Counter(r["final_class"] for r in final)
    never = [r for r in final if r["final_class"] != CLASSES[0]]
    recoverable = [r for r in never if r["final_class"] == CLASSES[1]]
    passed = len(recoverable) >= GATE_R0_MIN_RECOVERABLE
    by_task = defaultdict(Counter)
    for r in final:
        by_task[r["task_name"]][r["final_class"]] += 1
    stuck = Counter((r["task_name"], r["stage_at_alarm"], r["atomic_task"], r["atomic_task_source"])
                    for r in recoverable)
    errors = ladder.read_jsonl(audit_dir / ERRORS_NAME)
    scenes = ladder.read_jsonl(audit_dir / SCENES_NAME)
    result = {
        "kind": "r0_gate",
        "states": len(final),
        "errors": len(errors),
        "scene_checks": dict(Counter("ok" if s["ok"] else "mismatch" for s in scenes)),
        "manual_overrides": sum(r["manual"] for r in final),
        "classes": {c: counts.get(c, 0) for c in CLASSES},
        "never_recovered": len(never),
        "recoverable_never_recovered": len(recoverable),
        "threshold": GATE_R0_MIN_RECOVERABLE,
        "decision": "proceed_to_R1" if passed else "stop_report_detection_and_resampling",
        "by_task": {t: {c: n for c, n in sorted(v.items())} for t, v in sorted(by_task.items())},
        "recoverable_stuck_at": [
            {"task_name": t, "stage": s, "atomic_task": a, "atomic_task_source": src, "states": n}
            for (t, s, a, src), n in sorted(stuck.items(), key=lambda kv: (-kv[1], kv[0]))],
        "flags": dict(Counter(f.split(":")[0] if f.startswith("undone:") else f.split(":")[1]
                              for r in never for f in r["flags"])),
        "created_at": ladder.utc_now(),
    }
    ladder.atomic_write_json(audit_dir / "r0_gate.json", result)
    with (audit_dir / "r0_states.jsonl").open("w") as stream:
        for row in final:
            stream.write(json.dumps({k: row[k] for k in ("parent_id", "task_name", "stage_at_alarm", "atomic_task",
                                                         "provisional_class", "final_class", "manual", "note",
                                                         "flags")}) + "\n")
    write_gate_md(result, audit_dir / "r0_gate.md")
    return result


def write_gate_md(result, path):
    lines = ["# Gate R0: failure audit at the Hide-and-Seek + P alarm", "",
             f"Decision: **{result['decision']}** ({result['recoverable_never_recovered']} recoverable among "
             f"{result['never_recovered']} never-recovered states; threshold {result['threshold']}).",
             f"States: {result['states']}; errors: {result['errors']}; scene checks: {result['scene_checks']}; "
             f"manual overrides: {result['manual_overrides']}.", "",
             "| Task | " + " | ".join(CLASSES) + " |", "|---|" + "---:|" * len(CLASSES)]
    for task, counts in result["by_task"].items():
        lines.append(f"| {task} | " + " | ".join(str(counts.get(c, 0)) for c in CLASSES) + " |")
    lines.append("| all | " + " | ".join(str(result["classes"][c]) for c in CLASSES) + " |")
    lines += ["", "Recoverable states by the subtask they are stuck on:", "",
              "| Task | Stage | Atomic task | Source | States |", "|---|---|---|---|---:|"]
    for row in result["recoverable_stuck_at"]:
        lines.append(f"| {row['task_name']} | {row['stage']} | {row['atomic_task']} | "
                     f"{row['atomic_task_source']} | {row['states']} |")
    lines += ["", f"Flags among never-recovered states: {result['flags']}", ""]
    Path(path).write_text("\n".join(lines))


# ---- CLI --------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit", help="restore alarmed failure states and record what they look like")
    p.add_argument("--source-run", nargs="+", required=True, type=Path)
    p.add_argument("--alarms", required=True, type=Path)
    p.add_argument("--branch-runs", nargs="*", default=[], type=Path, help="Phase A1 branch shards")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--scene-tolerance", type=float, default=1e-6)
    p.add_argument("--floor-z", type=float, default=DEFAULT_FLOOR_Z)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-errors", type=int, default=10)
    p = sub.add_parser("decide", help="Gate R0 from the audit (and a filled-in review.csv)")
    p.add_argument("--audit-dir", required=True, type=Path)
    p.add_argument("--review", type=Path, help="review.csv with manual_class filled where needed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        run_audit(args.source_run, args.alarms, args.output_dir, branch_runs=args.branch_runs, tasks=args.tasks,
                  scene_tolerance=args.scene_tolerance, floor_z=args.floor_z, images=not args.no_images,
                  resume=args.resume, max_errors=args.max_errors)
    else:
        result = decide(args.audit_dir, args.review)
        print(Path(args.audit_dir / "r0_gate.md").read_text())
        print(f"decision={result['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
