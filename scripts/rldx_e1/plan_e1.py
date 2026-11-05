"""Plan E1's new RLDX-1 rollouts (plan: docs/rldx_e1_plan.md).

E1 tests the frozen SAFE-LSTM + P and Hide-and-Seek + P monitors on new RLDX-1 rollouts: a new
calibration set and a new test set, each with 20 successes and 20 failures per task, on the 20
tasks of the recorded collection the monitors were fixed on.

This script reads that recorded collection (the raw manifest behind
`official_safe_supported20_natural50` and its collector shard summaries) and the E1 freeze
manifest, and writes the collection plan. Every collector setting is copied from the recorded
collection except three: the base seed (a new one per set), the per-class quotas, and the
attempt cap. It stops if
  - the recorded settings are not unique, or a registry horizon changed (via plan_pilot.build);
  - the freeze manifest's tasks or time limits differ from the recorded collection's;
  - a new base seed equals the other set's, or any base seed used by an RLDX-1 collection found
    under --seed-roots.

Writes into --output:
  plan.json                     settings, seeds, tasks (H, p, expected attempts), shards, cost;
  collector_flags_<set>.txt     common collector arguments, one per line (read with mapfile);
  tasks_<set>_shard<k>.txt      the tasks of each shard, balanced by expected env steps;
  shards.tsv                    one line per shard: set, shard, GPU, port.
Outcomes of the recorded collection are read only for its success rates (expected cost).

Usage (outside the data directories):
  python -I plan_e1.py <raw manifest dir> --collection-root <SAFE35 root> \
      --freeze-manifest <e2_freeze_manifest.json> --seed-roots <dir> [<dir> ...] --output <E1 root>
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SETS = ("calibration", "test")
DEFAULT_SEEDS = {"calibration": 1201, "test": 1301}
QUOTA = 20
ATTEMPT_CAP = 300
PORT_BASE = 20150
DRAWS = 20000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = _load("_e1_plan_pilot", REPO / "scripts" / "rldx_continuation_pilot" / "plan_pilot.py")


def used_seeds(roots):
    """Base seeds of every RLDX-1 collector shard under the roots (merged datasets have none)."""
    seeds = {}
    for root in roots:
        for path in sorted(Path(root).rglob("summary.json")):
            try:
                config = json.loads(path.read_text()).get("config") or {}
            except (OSError, ValueError):
                continue
            if config.get("model_family") == "rldx1" and config.get("base_environment_seed") is not None:
                seeds.setdefault(int(config["base_environment_seed"]), []).append(str(path))
    return seeds


def expected_attempts(p, n=QUOTA, cap=ATTEMPT_CAP, draws=DRAWS, seed=0):
    """Attempts until n successes and n failures (retain-only-quota keeps going until both).

    Returns the mean, the 95th percentile and the probability of exceeding the cap."""
    rng = np.random.default_rng(seed)
    p = min(max(p, 1e-6), 1 - 1e-6)
    a = n + rng.negative_binomial(n, p, size=draws)        # attempts until n successes
    b = n + rng.negative_binomial(n, 1 - p, size=draws)    # attempts until n failures
    m = np.maximum(a, b)
    return float(m.mean()), float(np.percentile(m, 95)), float((m > cap).mean())


def task_cost(v, mean_success_len, attempts):
    """Expected env steps: attempts x (p x mean success length + (1 - p) x H)."""
    return attempts * (v["p"] * mean_success_len + (1 - v["p"]) * v["H"])


def balance(tasks, cost, k):
    shards = [[] for _ in range(k)]
    loads = [0.0] * k
    for t in sorted(tasks, key=lambda t: (-cost[t], t)):
        i = min(range(k), key=lambda j: (loads[j], j))
        shards[i].append(t)
        loads[i] += cost[t]
    return [sorted(s) for s in shards], loads


def flags(settings, seed, quota=QUOTA, cap=ATTEMPT_CAP):
    f = pilot.flags({**settings, "seed": seed, "num_rollouts": cap, "horizon_scale": 1.0})
    i = f.index("--horizon-scale")
    del f[i:i + 2]                                      # E1 runs at the registry horizon
    return f + ["--success-quota", str(quota), "--failure-quota", str(quota), "--retain-only-quota"]


def build(rows, source, configs, registry, atomic, freeze, seeds, used, shards_per_set,
          frame_shape_fn=None):
    tasks, out_tasks, settings = pilot.build(rows, source, configs, registry, atomic, frame_shape_fn)
    settings = {k: v for k, v in settings.items() if k not in ("horizon_scale", "num_rollouts")}
    problems = []
    frozen = freeze["tasks"]
    if sorted(frozen) != sorted(tasks):
        problems.append(f"freeze manifest tasks {sorted(frozen)} != recorded tasks {sorted(tasks)}")
    inferences = {}
    for t in tasks:
        if t in frozen:
            H_inf = int(frozen[t]["H_k"])
            # The freeze manifest's H_k is in policy inferences of replan_steps env steps.
            if H_inf != math.ceil(out_tasks[t]["H"] / settings["replan_steps"]):
                problems.append(f"{t}: frozen H_k {H_inf} != ceil({out_tasks[t]['H']} / {settings['replan_steps']})")
            if frozen[t].get("type") and frozen[t]["type"] != out_tasks[t]["type"]:
                problems.append(f"{t}: frozen type {frozen[t]['type']} != registry type {out_tasks[t]['type']}")
            inferences[t] = H_inf
    if seeds["calibration"] == seeds["test"]:
        problems.append("the calibration and test sets need different base seeds")
    for name, s in seeds.items():
        if s == settings["seed"] or s in used:
            where = used.get(s, ["the recorded collection"])
            problems.append(f"{name} seed {s} was used before: {where[:3]}")
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))
    plan_tasks, cost = {}, {}
    for t in tasks:
        v = out_tasks[t]
        lens = [sc["L"] for sc in v["scenes"].values() if not sc["failed"]]
        mean_len = float(np.mean(lens)) if lens else float(v["H"])
        mean_att, p95_att, over = expected_attempts(v["p"])
        cost[t] = task_cost(v, mean_len, mean_att)
        plan_tasks[t] = dict(type=v["type"], H=v["H"], H_inferences=inferences[t], p=v["p"],
                             recorded_n=v["n"], mean_success_length=round(mean_len, 1),
                             expected_attempts=round(mean_att, 1), p95_attempts=round(p95_att, 1),
                             p_over_cap=round(over, 4), expected_env_steps=round(cost[t]))
    shard_tasks, loads = balance(tasks, cost, shards_per_set)
    shard_rows = []
    gpu = 0
    for name in SETS:
        for k, s in enumerate(shard_tasks):
            shard_rows.append(dict(set=name, shard=k, gpu=gpu, port=PORT_BASE + gpu, tasks=s,
                                   expected_env_steps=round(loads[k])))
            gpu += 1
    per_set_steps = sum(cost.values())
    cost_summary = dict(
        rollouts_per_set=round(sum(p["expected_attempts"] for p in plan_tasks.values())),
        rollouts=round(2 * sum(p["expected_attempts"] for p in plan_tasks.values())),
        env_steps=round(2 * per_set_steps),
        inferences=int(math.ceil(2 * per_set_steps / settings["replan_steps"])),
        tasks_over_cap=[t for t, p in plan_tasks.items() if p["p_over_cap"] > 0.01])
    return tasks, plan_tasks, settings, shard_rows, cost_summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="raw manifest dir (source_dataset of the official export)")
    p.add_argument("--collection-root", required=True, help="recorded collection root (summary.json files)")
    p.add_argument("--freeze-manifest", required=True, help="e2_freeze_manifest.json (R10 part B)")
    p.add_argument("--seed-roots", nargs="+", required=True, help="roots searched for used RLDX-1 base seeds")
    p.add_argument("--calibration-seed", type=int, default=DEFAULT_SEEDS["calibration"])
    p.add_argument("--test-seed", type=int, default=DEFAULT_SEEDS["test"])
    p.add_argument("--shards-per-set", type=int, default=2, help="GPUs per set (total GPUs = 2 x this)")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if (out / "plan.json").exists():
        raise SystemExit(f"error: {out} already has a plan")
    rows = pilot.read_jsonl(Path(a.source) / "manifest.jsonl")
    if not rows:
        raise SystemExit("STOP: empty manifest")
    at = pilot.atomic_tasks_module()
    configs = pilot.summaries(a.collection_root, {r["rollout_id"] for r in rows})
    freeze = json.loads(Path(a.freeze_manifest).read_text())
    seeds = {"calibration": a.calibration_seed, "test": a.test_seed}
    used = used_seeds(a.seed_roots)
    tasks, plan_tasks, settings, shard_rows, cost = build(
        rows, a.source, configs, at.registered_safe_task_horizons(), at.registered_atomic_tasks(),
        freeze, seeds, used, a.shards_per_set)
    out.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        (out / f"collector_flags_{name}.txt").write_text("\n".join(flags(settings, seeds[name])) + "\n")
    for r in shard_rows:
        (out / f"tasks_{r['set']}_shard{r['shard']}.txt").write_text("\n".join(r["tasks"]) + "\n")
    (out / "shards.tsv").write_text("".join(f"{r['set']}\t{r['shard']}\t{r['gpu']}\t{r['port']}\n" for r in shard_rows))
    plan = dict(plan="docs/rldx_e1_plan.md", source=str(Path(a.source).resolve()),
                summaries=[str(path) for path, _ in configs], freeze_manifest=str(Path(a.freeze_manifest).resolve()),
                seeds=seeds, used_seeds=sorted(used), quota=QUOTA, attempt_cap=ATTEMPT_CAP,
                settings=settings, tasks=plan_tasks, shards=shard_rows, cost=cost,
                types={k: sum(plan_tasks[t]["type"] == k for t in tasks) for k in ("atomic", "composite")})
    (out / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    for t in tasks:
        v = plan_tasks[t]
        print(f"{t:32s} {v['type']:9s} H={v['H']:5d} ({v['H_inferences']:3d} inf.)  p={v['p']:.2f}  "
              f"attempts/set {v['expected_attempts']:6.1f} (p95 {v['p95_attempts']:5.0f})")
    print(f"settings: recorded seed {settings['seed']}, {settings['seed_protocol']}, split {settings['split']}, "
          f"replan {settings['replan_steps']}, frame {settings.get('video_height')}x{settings.get('video_width')}, "
          f"stride {settings['video_frame_stride']}")
    print(f"new seeds: calibration {seeds['calibration']}, test {seeds['test']} | used RLDX-1 seeds found: {sorted(used)}")
    print(f"cost: ~{cost['rollouts']} rollouts, ~{cost['inferences']} inferences, {cost['env_steps']} env steps "
          f"| over cap: {cost['tasks_over_cap'] or 'none'} | types {plan['types']}")
    for r in shard_rows:
        print(f"{r['set']:11s} shard{r['shard']} GPU {r['gpu']} port {r['port']}: {r['tasks']}")
    if cost["tasks_over_cap"]:
        print("STOP: some tasks exceed the attempt cap in more than 1% of draws; raise ATTEMPT_CAP in the plan first")
        raise SystemExit(1)
    print("PLAN OK")


if __name__ == "__main__":
    main()
