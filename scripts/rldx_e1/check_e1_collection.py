"""Check E1's new RLDX-1 rollouts against the plan and the recorded collection (docs/rldx_e1_plan.md).

Reads only identities, settings and feature shapes; no monitor is scored here. Outcome counts
are read for the quota check alone.

Checks, per shard directory given as <set>=<dir>:
  1. settings: every row has the plan's base seed for its set and the recorded collection's
     seed protocol, split, replan steps, policy, checkpoint label, policy config, model family,
     video stride, and the task's registry horizon;
  2. feature contract: feature layer, tensor dtype and every tensor dimension except the
     number of inferences equal a recorded rollout's of the same task; features finite and
     nonzero; inference environment steps spaced by the replan steps;
  3. identity: no rollout ID appears twice, in two sets, or in the recorded collection;
  4. quotas (skipped with --smoke): every planned task of the set has exactly 20 retained
     successes and 20 retained failures, and no collection errors.
Prints CHECK OK, or STOP with every problem found.

Usage:
  python -I check_e1_collection.py --plan <E1 root>/plan.json calibration=<dir> test=<dir> [...]
  python -I check_e1_collection.py --plan <E1 root>/plan.json --smoke --smoke-seed 1101 smoke=<smoke dir>
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SETTING_FIELDS = (("seed_protocol", "seed_protocol"), ("environment_split", "split"),
                  ("replan_steps", "replan_steps"), ("policy_id", "policy_name"),
                  ("checkpoint", "checkpoint"), ("policy_config", "policy_config"),
                  ("model_family", "model_family"), ("video_frame_stride", "video_frame_stride"))


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tensor_info(root, row):
    with np.load(Path(root) / row["tensor_path"], allow_pickle=False) as z:
        f = z["features"]
        steps = z["inference_environment_steps"] if "inference_environment_steps" in z.files else None
        return dict(shape=list(f.shape), dtype=str(f.dtype), finite=bool(np.isfinite(f).all()),
                    nonzero=bool(np.any(f)), steps=None if steps is None else [int(s) for s in steps])


def check(plan, shards, recorded_rows, recorded_root, smoke=False, tensor_fn=tensor_info):
    problems = []
    settings = plan["settings"]
    seeds = plan["seeds"]
    rec_ids = {r["rollout_id"] for r in recorded_rows}
    rec_by_task = {}
    for r in recorded_rows:
        rec_by_task.setdefault(r["task_name"], r)
    rec_info = {}
    seen = Counter()
    counts = defaultdict(Counter)
    for name, root in shards:
        if name not in seeds:
            problems.append(f"{root}: unknown set {name!r} (expected {sorted(seeds)})")
            continue
        rows = read_jsonl(Path(root) / "manifest.jsonl")
        errors = read_jsonl(Path(root) / "errors.jsonl")
        if not rows:
            problems.append(f"{root}: no rollouts")
        if errors and not smoke:
            problems.append(f"{root}: {len(errors)} collection errors")
        for r in rows:
            rid, t = r["rollout_id"], r["task_name"]
            seen[rid] += 1
            if rid in rec_ids:
                problems.append(f"{rid}: also in the recorded collection")
            if t not in plan["tasks"]:
                problems.append(f"{rid}: task {t} is not in the plan")
                continue
            if int(r["environment_seed"]) != seeds[name]:    # official_rldx: one base seed per set
                problems.append(f"{rid}: seed {r['environment_seed']} does not belong to {name} (base {seeds[name]})")
            for field, key in SETTING_FIELDS:
                if (r.get(field) or ({} if field == "policy_config" else None)) != settings[key]:
                    problems.append(f"{rid}: {field} {r.get(field)!r} != recorded {settings[key]!r}")
            if int(r["rollout_horizon"]) != plan["tasks"][t]["H"]:
                problems.append(f"{rid}: horizon {r['rollout_horizon']} != {plan['tasks'][t]['H']}")
            ref = rec_by_task.get(t)
            if ref is None:
                problems.append(f"{t}: no recorded rollout to compare features with")
                continue
            if r.get("feature_layer") != ref.get("feature_layer"):
                problems.append(f"{rid}: feature layer {r.get('feature_layer')} != {ref.get('feature_layer')}")
            if t not in rec_info:
                rec_info[t] = tensor_fn(recorded_root, ref)
            info, want = tensor_fn(root, r), rec_info[t]
            if info["shape"][1:] != want["shape"][1:] or info["dtype"] != want["dtype"]:
                problems.append(f"{rid}: features {info['shape']} {info['dtype']} vs recorded "
                                f"{want['shape']} {want['dtype']}")
            if not (info["finite"] and info["nonzero"]):
                problems.append(f"{rid}: features not finite or all zero")
            steps = info["steps"]
            if steps and any(b - a < settings["replan_steps"] for a, b in zip(steps, steps[1:])):
                problems.append(f"{rid}: inference steps closer than {settings['replan_steps']}")
            counts[(name, t)]["failures" if r["failed"] else "successes"] += 1
    dup = [rid for rid, c in seen.items() if c > 1]
    if dup:
        problems.append(f"{len(dup)} rollout IDs appear more than once, e.g. {dup[:3]}")
    if not smoke:
        quota = plan["quota"]
        for name, _ in shards:
            for t in plan["tasks"]:
                c = counts[(name, t)]
                if (c["successes"], c["failures"]) != (quota, quota):
                    problems.append(f"{name} {t}: {c['successes']} successes / {c['failures']} failures, "
                                    f"want {quota} / {quota}")
    return problems, counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--smoke", action="store_true", help="skip the quota and error checks")
    p.add_argument("--smoke-seed", type=int, help="base seed of the smoke run (set name: smoke)")
    p.add_argument("shards", nargs="+", help="<set>=<collector output dir>; give every shard of every set")
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    if a.smoke_seed is not None:
        if not a.smoke or a.smoke_seed in plan["seeds"].values() or a.smoke_seed in plan.get("used_seeds", []):
            raise SystemExit("error: --smoke-seed needs --smoke and a seed used by no set or earlier collection")
        plan["seeds"] = {**plan["seeds"], "smoke": a.smoke_seed}
    shards = []
    for s in a.shards:
        if "=" not in s:
            raise SystemExit(f"error: {s!r} is not <set>=<dir>")
        name, root = s.split("=", 1)
        shards.append((name, root))
    if not a.smoke and {n for n, _ in shards} != set(plan["seeds"]):
        raise SystemExit("error: give the shards of both sets for the full check")
    recorded = [json.loads(x) for x in Path(plan["source"], "manifest.jsonl").read_text().splitlines() if x.strip()]
    problems, counts = check(plan, shards, recorded, plan["source"], smoke=a.smoke)
    by_set = defaultdict(Counter)
    for (name, _), c in counts.items():
        by_set[name].update(c)
    for name, c in sorted(by_set.items()):
        print(f"{name}: {c['successes']} successes, {c['failures']} failures retained")
    if problems:
        print("STOP:")
        for x in problems[:50]:
            print("  -", x)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        raise SystemExit(1)
    print("CHECK OK" + (" (smoke)" if a.smoke else ""))


if __name__ == "__main__":
    main()
