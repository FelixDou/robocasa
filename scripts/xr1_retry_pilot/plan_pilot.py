"""Select tasks and attempt limits for the XR-1 same-scene retry pilot (plan: docs/xr1_retry_pilot_plan.md).

Reads the confirmatory collection's shard manifests, picks the 10 tasks whose all-retained
success rate is closest to 0.5 (ties by name), and for each sets the attempt limit to one past
the episode index of its 10th failure, so the pilot re-attempts every original attempt up to
and including that failure. Writes attempt_limits.json, two shard task lists balanced by
attempts, and selection.json. Reads only outcomes and episode indices.

Usage:
  python3 -I plan_pilot.py <confirmatory_collection dir> --output <pilot root>
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

N_TASKS, N_FAIL = 10, 10


def load(collection):
    """Manifest rows plus task types; manifest rows do not carry task_type, the shard
    summaries record it in their config."""
    rows, types = [], {}
    for shard in sorted(Path(collection).glob("shard*")):
        m = shard / "manifest.jsonl"
        if m.is_file():
            for line in m.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        summary = shard / "summary.json"
        if summary.is_file():
            types.update(json.loads(summary.read_text()).get("config", {}).get("task_types") or {})
    if not rows:
        raise SystemExit(f"error: no shard manifests under {collection}")
    for r in rows:
        r.setdefault("task_type", types.get(r["task_name"]))
    return rows


def select(rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_name"]].append(r)
    stats = {}
    for task, rs in by_task.items():
        rs = sorted(rs, key=lambda r: int(r["environment_reset_index"]))
        fails = [int(r["environment_reset_index"]) for r in rs if r["failed"]]
        stats[task] = dict(n=len(rs), p=sum(not r["failed"] for r in rs) / len(rs), failures=len(fails),
                           task_type=rs[0].get("task_type"),
                           limit=(fails[N_FAIL - 1] + 1) if len(fails) >= N_FAIL else None)
    eligible = sorted((t for t, s in stats.items() if s["limit"] is not None), key=lambda t: (abs(stats[t]["p"] - 0.5), t))
    return stats, sorted(eligible[:N_TASKS])


def shards(tasks, stats):
    a, b = [], []
    for t in sorted(tasks, key=lambda t: -stats[t]["limit"]):
        (a if sum(stats[x]["limit"] for x in a) <= sum(stats[x]["limit"] for x in b) else b).append(t)
    return sorted(a), sorted(b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("collection"); p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if (out / "selection.json").exists():
        raise SystemExit(f"error: {out} already has a selection")
    rows = load(a.collection)
    stats, tasks = select(rows)
    if len(tasks) < N_TASKS:
        raise SystemExit(f"error: only {len(tasks)} tasks have {N_FAIL} failures")
    s0, s1 = shards(tasks, stats)
    out.mkdir(parents=True, exist_ok=True)
    (out / "attempt_limits.json").write_text(json.dumps({t: stats[t]["limit"] for t in tasks}, indent=1, sort_keys=True) + "\n")
    (out / "tasks_shard0.txt").write_text("\n".join(s0) + "\n")
    (out / "tasks_shard1.txt").write_text("\n".join(s1) + "\n")
    sel = dict(plan="docs/xr1_retry_pilot_plan.md", rule=f"{N_TASKS} tasks with success rate closest to 0.5; limit = episode of failure {N_FAIL} + 1",
               tasks={t: stats[t] for t in tasks}, shards=[s0, s1],
               attempts_total=sum(stats[t]["limit"] for t in tasks),
               types={k: sum(stats[t]["task_type"] == k for t in tasks) for k in ("atomic", "composite")})
    (out / "selection.json").write_text(json.dumps(sel, indent=1, sort_keys=True) + "\n")
    for t in tasks:
        s = stats[t]
        print(f"{t:40s} {str(s['task_type']):9s} p={s['p']:.2f}  limit={s['limit']:3d}")
    print(f"attempts: {sel['attempts_total']} | shard0 {s0} | shard1 {s1} | types {sel['types']}")


if __name__ == "__main__":
    main()
