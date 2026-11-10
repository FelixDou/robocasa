"""Check E2's collection and pair its attempts (docs/rldx_e2_plan.md, gate 2). No monitor is scored here.

For every collector output directory:
  1. settings and feature contract: E1's checks (check_e1_collection.check) with E2's base seed and the
     1.25 H rollout horizon;
  2. completeness: a task counts only where all 2 x scenes rollouts exist, with reset indices 0 .. 2S-1 once
     each and no collection error for that task in that directory. Directories use E2's base seed, or a
     reserved rerun seed for a task whose earlier attempt was interrupted; no rollout ID may appear in two
     directories. Every task must be complete in exactly one directory; a partial task elsewhere is not used;
  3. pairs: pairs.jsonl has one line per reset, alternating first / retry; the retry restored the state its
     first attempt began from (same state hash), with restore residual <= RESIDUAL_TOL, the same instruction,
     every observation key present, images within IMAGE_TOL and other observations within STATE_TOL of the
     first attempt's first observation. A pair failing any of these is excluded; more than MAX_BAD_SHARE of
     pairs excluded stops E2 (the restore is not working).
Writes --output (pairs.json): the plan's tasks with their directory, and every pair with its two rollout IDs.

Usage:
  python -I check_e2.py --plan <E2 root>/plan.json --output <E2 root>/pairs.json e2=<dir> [e2=<dir> ...]
  python -I check_e2.py --plan <E2 root>/plan.json --smoke smoke=<dir>
"""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESIDUAL_TOL = 1e-6
IMAGE_TOL = 2.0          # mean |diff| of 0-255 pixels; the continuation pilot saw 0.8-4.0 for re-built scenes
STATE_TOL = 1e-4
MAX_BAD_SHARE = 0.02


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ce1 = _load("_e2_check_e1", REPO / "scripts" / "rldx_e1" / "check_e1_collection.py")
read_jsonl = ce1.read_jsonl


def dir_seed(root):
    summary = Path(root) / "summary.json"
    config = json.loads(summary.read_text()).get("config", {}) if summary.is_file() else {}
    return config.get("base_environment_seed")


def e1_view(plan, name, seed):
    """E2's plan in the shape check_e1_collection.check reads, for one directory's base seed."""
    return dict(settings=plan["settings"], seeds={name: seed}, source=plan["source"],
                tasks={t: dict(H=v["budget_steps"]) for t, v in plan["tasks"].items()})


def pair_problem(first, retry):
    """Why a pair cannot be used, or None."""
    if retry.get("role") != "retry" or first.get("role") != "first":
        return "roles out of order"
    if retry.get("restored_sha") != first.get("state_sha"):
        return "the retry did not restore its first attempt's initial state"
    if retry.get("model_sha") != first.get("model_sha"):
        return "the retry used another scene model"
    if not retry.get("restore_residual", float("inf")) <= RESIDUAL_TOL:
        return f"restore residual {retry.get('restore_residual')}"
    d = retry.get("first_obs_distance")
    if not d:
        return "no first-observation comparison"
    if d.get("missing"):
        return f"observation keys missing: {d['missing'][:3]}"
    if not d.get("text_equal", False):
        return "instruction differs"
    if not d.get("image_max", float("inf")) <= IMAGE_TOL:
        return f"first images differ by {d.get('image_max')}"
    if not d.get("state_max", float("inf")) <= STATE_TOL:
        return f"first observations differ by {d.get('state_max')}"
    return None


def check(plan, dirs, recorded_rows, smoke=False, tensor_fn=ce1.tensor_info):
    problems = []
    allowed = [plan["smoke_seed"]] if smoke else [plan["seed"]] + list(plan.get("rerun_seeds", []))
    ids = Counter()
    for name, root in dirs:
        seed = dir_seed(root)
        if seed not in allowed:
            problems.append(f"{root}: base seed {seed} is not one of the plan's {allowed}")
            continue
        ids.update(r["rollout_id"] for r in read_jsonl(Path(root) / "manifest.jsonl"))
        p, _ = ce1.check(e1_view(plan, name, seed), [(name, root)], recorded_rows, plan["source"], smoke=True,
                         tensor_fn=tensor_fn)
        problems += p
    dup = [rid for rid, c in ids.items() if c > 1]
    if dup:
        problems.append(f"{len(dup)} rollout IDs appear in more than one directory, e.g. {dup[:3]} "
                        "(a task re-run with the same seed: move its partial directory aside or use a rerun seed)")
    n = plan["rollouts_per_task"] if not smoke else None
    complete = defaultdict(list)
    pairs, excluded = [], []
    for name, root in dirs:
        rows = read_jsonl(Path(root) / "manifest.jsonl")
        log = read_jsonl(Path(root) / "pairs.jsonl")
        err_tasks = {e.get("task_name") for e in read_jsonl(Path(root) / "errors.jsonl")}
        by_task = defaultdict(dict)
        for r in rows:
            idx = int(r["environment_reset_index"])
            if idx in by_task[r["task_name"]]:
                problems.append(f"{root}: {r['task_name']} reset index {idx} twice")
            by_task[r["task_name"]][idx] = r
        log_by_task = defaultdict(list)
        for x in log:
            log_by_task[x["task_name"]].append(x)
        for t, recs in by_task.items():
            want = n if n is not None else len(recs) + len(recs) % 2
            if t in err_tasks or sorted(recs) != list(range(want)):
                continue                      # partial: not used from this directory
            entries = log_by_task[t]
            if [x["reset_index"] for x in entries] != list(range(want)):
                problems.append(f"{root}: {t}: pairs.jsonl has resets {[x['reset_index'] for x in entries][:6]}..., "
                                f"want 0..{want - 1} once each")
                continue
            starts = [x.get("state_sha") for x in entries[0::2]]
            dup = len(starts) - len(set(starts))
            if dup and not smoke:
                problems.append(f"{root}: {t}: {dup} of {len(starts)} scenes start from the same state as an "
                                "earlier scene (the scene was not built anew)")
            complete[t].append(root)
            for j in range(want // 2):
                first, retry = entries[2 * j], entries[2 * j + 1]
                a, b = recs[2 * j], recs[2 * j + 1]
                why = pair_problem(first, retry)
                row = dict(task_name=t, scene=j, dir=str(root), first_id=a["rollout_id"], retry_id=b["rollout_id"],
                           restore_residual=retry.get("restore_residual"),
                           image_max=(retry.get("first_obs_distance") or {}).get("image_max"),
                           state_max=(retry.get("first_obs_distance") or {}).get("state_max"), excluded=why)
                (excluded if why else pairs).append(row)
    if not smoke:
        for t in plan["tasks"]:
            if len(complete[t]) != 1:
                problems.append(f"{t}: complete in {len(complete[t])} directories {complete[t]}, want exactly 1")
        total = len(pairs) + len(excluded)
        if total and len(excluded) > MAX_BAD_SHARE * total:
            problems.append(f"{len(excluded)} of {total} pairs fail the restore checks (more than "
                            f"{MAX_BAD_SHARE:.0%}), e.g. {[x['excluded'] for x in excluded[:3]]}")
    return problems, dict(tasks={t: v[0] for t, v in complete.items() if len(v) == 1}, pairs=pairs,
                          excluded=excluded)


def breakdown(dirs):
    """Excluded pairs by task and reason, from pairs.jsonl only (no outcome is read). A task with any
    'instruction differs' pair is marked RERUN (docs/rldx_e2_plan.md, step 7a)."""
    bad, n, lines = Counter(), Counter(), []
    for _, root in dirs:
        log = read_jsonl(Path(root) / "pairs.jsonl")
        for j in range(0, len(log) - 1, 2):
            t = log[j]["task_name"]
            n[t] += 1
            why = pair_problem(log[j], log[j + 1])
            if why:
                bad[(t, why.split(" by ")[0])] += 1
    for (t, why), k in sorted(bad.items()):
        lines.append(f"{t:28s} {k:3d}/{n[t]}  {why}")
    rerun = sorted({t for (t, why) in bad if why == "instruction differs"})
    for _, root in dirs:
        log = read_jsonl(Path(root) / "pairs.jsonl")
        by = defaultdict(list)
        for x in log:
            if x.get("role") == "first":
                by[x["task_name"]].append(x.get("state_sha"))
        for t, starts in sorted(by.items()):
            if len(starts) != len(set(starts)):
                lines.append(f"{t:28s} {len(set(starts)):3d} distinct start states in {len(starts)} scenes")
    lines.append(f"total excluded: {sum(bad.values())} of {sum(n.values())} pairs")
    lines += [f"RERUN {t}" for t in rerun]
    return lines, rerun


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--breakdown", action="store_true", help="excluded pairs by task and reason; RERUN lines (step 7a)")
    p.add_argument("--output", help="pairs.json (required for the full check)")
    p.add_argument("dirs", nargs="+", help="e2=<collector output dir> (smoke=<dir> with --smoke)")
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    dirs = []
    for s in a.dirs:
        name, _, root = s.partition("=")
        if not root or name != ("smoke" if a.smoke else "e2"):
            raise SystemExit(f"error: {s!r}: give {'smoke' if a.smoke else 'e2'}=<dir>")
        dirs.append((name, root))
    if a.breakdown:
        print("\n".join(breakdown(dirs)[0]))
        return
    if not a.smoke and not a.output:
        raise SystemExit("error: --output is required for the full check")
    if a.output and Path(a.output).exists():
        raise SystemExit(f"error: {a.output} exists")
    recorded = read_jsonl(Path(plan["source"]) / "manifest.jsonl")
    problems, out = check(plan, dirs, recorded, smoke=a.smoke)
    c = Counter(r["task_name"] for r in out["pairs"])
    res = [r["restore_residual"] for r in out["pairs"] + out["excluded"] if r["restore_residual"] is not None]
    img = [r["image_max"] for r in out["pairs"] + out["excluded"] if r["image_max"] is not None]
    print(f"pairs usable: {len(out['pairs'])}, excluded: {len(out['excluded'])}, tasks complete: {len(out['tasks'])}")
    if res:
        print(f"restore residual max {max(res):.3g}; first-image difference max {max(img) if img else float('nan'):.3g}")
    for x in out["excluded"][:10]:
        print(f"  excluded {x['task_name']} scene {x['scene']}: {x['excluded']}")
    if problems:
        print("STOP:")
        for x in problems[:50]:
            print("  -", x)
        raise SystemExit(1)
    if a.output:
        Path(a.output).write_text(json.dumps(dict(plan=plan["plan"], seed=plan["seed"], per_task=dict(c), **out),
                                             indent=1, sort_keys=True) + "\n")
    print("CHECK OK" + (" (smoke)" if a.smoke else ""))


if __name__ == "__main__":
    main()
