"""Analyse the RLDX-1 continuation and same-scene retry pilot (plan: docs/rldx_e3_pilot_plan.md).

Gates first, before any outcome is read:
  1. identity and coverage: every pilot rollout is a planned scene (task, reset index) with the
     planned seed, protocol, split, policy and replan steps, and a horizon of ceil(1.25 H_k);
     no scene twice; at least 95% of scenes overall and 80% of each task's scenes collected.
     Failing this stops the analysis (resume the collection).
  2. scene: a pair is the same scene if the mean absolute first-frame difference is <= 10
     (0-255 scale). If fewer than 95% of pairs are, r is void (c does not need the pairing).
Endpoints:
  - c: of the pilot rollouts with no success by H_k, the share that succeed by 1.25 H_k
    (pooled; Wilson and task-cluster bootstrap 95% intervals; the decision interval is their
    union);
  - r_fail / r_succ: success by H_k on same-scene retries of originally failed / successful
    scenes;
  - p: original and pilot success by H_k (task-macro), as a stationarity check.

Usage:
  python -I analyze_pilot.py <pilot collection dir> --selection <pilot root>/selection.json \
      --output OUT.json [--bootstrap 2000]
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

FRAME_TOL, FRAME_SHARE = 10.0, 0.95
COVERAGE_ALL, COVERAGE_TASK = 0.95, 0.80
FALLBACK_R, FALLBACK_R_LOW = 0.245, 0.17     # XR-1 retry pilot (report 15.24), if r is void


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pilot_rows(collection):
    rows, configs = [], []
    for shard in sorted(Path(collection).glob("shard*")):
        if (shard / "manifest.jsonl").is_file():
            for r in read_jsonl(shard / "manifest.jsonl"):
                r["_shard"] = str(shard); rows.append(r)
        if (shard / "summary.json").is_file():
            configs.append(json.loads((shard / "summary.json").read_text()).get("config") or {})
    return rows, configs


def first_frame(path):
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(path))
    try:
        return np.asarray(reader.get_data(0), dtype=np.float64)
    finally:
        reader.close()


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def cluster_ci(per, num, den, tasks, reps, seed=0):
    """Task-cluster bootstrap of a pooled ratio sum(num)/sum(den)."""
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        draw = rng.choice(tasks, len(tasks))
        d = sum(per[t][den] for t in draw)
        if d:
            vals.append(sum(per[t][num] for t in draw) / d)
    return np.round(np.percentile(vals, [2.5, 97.5]), 4).tolist() if vals else [None, None]


def gate_identity(rows, configs, sel):
    s = sel["settings"]; tasks = sel["tasks"]
    problems, seen = [], set()
    for c in configs:
        if c.get("horizon_scale") != s["horizon_scale"]:
            problems.append(f"shard horizon_scale {c.get('horizon_scale')}")
    for r in rows:
        t, i = r["task_name"], str(int(r["environment_reset_index"]))
        if t not in tasks or i not in tasks[t]["scenes"]:
            problems.append(f"unplanned scene {t}/{i}"); continue
        if (t, i) in seen:
            problems.append(f"duplicate scene {t}/{i}")
        seen.add((t, i))
        want = dict(environment_seed=s["seed"], seed_protocol=s["seed_protocol"], environment_split=s["split"],
                    replan_steps=s["replan_steps"], policy_id=s["policy_name"], checkpoint=s["checkpoint"],
                    rollout_horizon=tasks[t]["H_scaled"])
        bad = [k for k, v in want.items() if r.get(k) != v]
        if bad:
            problems.append(f"{t}/{i} differs in {bad}")
    total = sum(len(v["scenes"]) for v in tasks.values())
    per_task = {t: sum(1 for (x, _) in seen if x == t) / len(v["scenes"]) for t, v in tasks.items()}
    coverage = dict(collected=len(seen), planned=total, share=round(len(seen) / total, 4),
                    lowest_task=min(per_task.items(), key=lambda kv: kv[1]) if per_task else None)
    if len(seen) < COVERAGE_ALL * total:
        problems.append(f"coverage {len(seen)}/{total} below {COVERAGE_ALL:.0%}")
    low = sorted(t for t, f in per_task.items() if f < COVERAGE_TASK)
    if low:
        problems.append(f"tasks below {COVERAGE_TASK:.0%} coverage: {low}")
    return problems, coverage


def gate_scene(rows, sel, frame_fn=first_frame):
    diffs = {}
    for r in rows:
        t, i = r["task_name"], str(int(r["environment_reset_index"]))
        orig = sel["tasks"][t]["scenes"][i].get("video")
        new = Path(r["_shard"]) / r["video_path"] if r.get("video_path") else None
        if not orig or not new or not Path(orig).is_file() or not new.is_file():
            diffs[(t, i)] = math.inf; continue
        a, b = frame_fn(new), frame_fn(orig)
        if a.shape != b.shape:
            raise SystemExit(f"STOP: frame shapes differ for {t}/{i}: {a.shape} vs {b.shape}")
        diffs[(t, i)] = float(np.mean(np.abs(a - b)))
    vals = np.array(list(diffs.values()))
    finite = vals[np.isfinite(vals)]
    share = float(np.mean(vals <= FRAME_TOL)) if len(vals) else 0.0
    return diffs, dict(tolerance=FRAME_TOL, same_scene_share=round(share, 4), missing_video=int(np.sum(~np.isfinite(vals))),
                       diff_quantiles=np.round(np.quantile(finite, [0, 0.5, 0.95, 1]), 3).tolist() if len(finite) else None,
                       passed=share >= FRAME_SHARE)


def endpoints(rows, sel, diffs, scene_ok, reps):
    tasks_cfg = sel["tasks"]
    per = defaultdict(lambda: defaultdict(int))
    late_fracs, by_type = [], defaultdict(lambda: [0, 0])
    by_orig = {True: [0, 0], False: [0, 0]}
    for r in rows:
        t, i = r["task_name"], str(int(r["environment_reset_index"]))
        H = tasks_cfg[t]["H"]; L = int(r["num_env_steps"]); success = not r["failed"]
        by_H = success and L <= H
        late = success and L > H
        orig_failed = tasks_cfg[t]["scenes"][i]["failed"]
        x = per[t]
        x["n"] += 1; x["by_H"] += by_H
        if not by_H:
            x["c_den"] += 1; x["c_num"] += late
            by_type[tasks_cfg[t]["type"]][0] += late; by_type[tasks_cfg[t]["type"]][1] += 1
            by_orig[orig_failed][0] += late; by_orig[orig_failed][1] += 1
        if late:
            late_fracs.append(L / H)
        if diffs.get((t, i), math.inf) <= FRAME_TOL:
            key = "fail" if orig_failed else "succ"
            x[f"{key}_n"] += 1; x[f"{key}_by_H"] += by_H
    tasks = sorted(per)
    tot = {k: sum(per[t][k] for t in tasks) for k in ("n", "by_H", "c_den", "c_num", "fail_n", "fail_by_H", "succ_n", "succ_by_H")}
    c_hat = tot["c_num"] / tot["c_den"] if tot["c_den"] else None
    c_w = wilson(tot["c_num"], tot["c_den"]); c_cl = cluster_ci(per, "c_num", "c_den", tasks, reps)
    c_lo = min(v for v in (c_w[0], c_cl[0]) if v is not None) if c_hat is not None else None
    c_hi = max(v for v in (c_w[1], c_cl[1]) if v is not None) if c_hat is not None else None
    res = dict(
        c=dict(late_successes=tot["c_num"], no_success_by_H=tot["c_den"], rate=None if c_hat is None else round(c_hat, 4),
               wilson95=c_w, cluster95=c_cl, decision95=[c_lo, c_hi],
               by_type={k: dict(late=v[0], n=v[1], rate=round(v[0] / v[1], 4) if v[1] else None) for k, v in by_type.items()},
               by_original_outcome={("originally failed" if k else "originally succeeded"): dict(late=v[0], n=v[1], rate=round(v[0] / v[1], 4) if v[1] else None)
                                    for k, v in by_orig.items()},
               late_success_L_over_H=np.round(np.quantile(late_fracs, [0, 0.5, 1]), 3).tolist() if late_fracs else None),
        p=dict(original_task_macro=round(float(np.mean([tasks_cfg[t]["p"] for t in tasks])), 4),
               pilot_by_H_task_macro=round(float(np.mean([per[t]["by_H"] / per[t]["n"] for t in tasks])), 4)))
    for key in ("fail", "succ"):
        n, k = tot[f"{key}_n"], tot[f"{key}_by_H"]
        res[f"r_{key}"] = dict(successes=k, n=n, rate=round(k / n, 4) if n else None, wilson95=wilson(k, n),
                               cluster95=cluster_ci(per, f"{key}_by_H", f"{key}_n", tasks, reps),
                               used=bool(scene_ok))
    if scene_ok and res["r_fail"]["rate"] is not None:
        r_use, r_low, r_source = res["r_fail"]["rate"], res["r_fail"]["wilson95"][0], "measured on RLDX-1"
    else:
        r_use, r_low, r_source = FALLBACK_R, FALLBACK_R_LOW, "void on RLDX-1; XR-1 retry pilot (report 15.24)"
    res["sizing_inputs"] = dict(retry_success=r_use, retry_success_low=r_low, retry_source=r_source,
                                continuation=[c_lo, None if c_hat is None else round(c_hat, 4), c_hi])
    res["per_task"] = {t: dict(type=tasks_cfg[t]["type"], p_original=tasks_cfg[t]["p"], **dict(per[t])) for t in tasks}
    return res


def run(args, frame_fn=first_frame):
    out_path = Path(args.output)
    if out_path.exists():
        sys.exit(f"error: output exists: {out_path}")
    sel = json.loads(Path(args.selection).read_text())
    rows, configs = pilot_rows(args.pilot)
    if not rows:
        sys.exit("STOP: no pilot rollouts")
    problems, coverage = gate_identity(rows, configs, sel)
    print("gate 1 (identity, coverage):", "passed" if not problems else "FAILED", coverage)
    if problems:
        print("STOP:", problems[:10]); sys.exit(1)
    diffs, scene = gate_scene(rows, sel, frame_fn)
    print("gate 2 (scene):", scene)
    res = dict(plan="docs/rldx_e3_pilot_plan.md", gates=dict(identity=True, coverage=coverage, scene=scene),
               **endpoints(rows, sel, diffs, scene["passed"], args.bootstrap))
    res["scene_pairs"] = {f"{t}/{i}": (round(d, 3) if math.isfinite(d) else None) for (t, i), d in sorted(diffs.items())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    c = res["c"]
    print(f"c = {c['late_successes']}/{c['no_success_by_H']} = {c['rate']}  Wilson {c['wilson95']}  cluster {c['cluster95']}  "
          f"decision {c['decision95']}")
    print(f"   by type {c['by_type']}\n   by original outcome {c['by_original_outcome']}\n   late L/H {c['late_success_L_over_H']}")
    for key in ("r_fail", "r_succ"):
        v = res[key]
        print(f"{key} = {v['successes']}/{v['n']} = {v['rate']}  Wilson {v['wilson95']}  cluster {v['cluster95']}"
              + ("" if v["used"] else "  (NOT USED: scene gate failed)"))
    print(f"p original {res['p']['original_task_macro']} | pilot by H {res['p']['pilot_by_H_task_macro']}")
    si = res["sizing_inputs"]
    print(f"SIZING INPUTS: --retry-success {si['retry_success']} and {si['retry_success_low']} ({si['retry_source']}); "
          f"continuation c = {si['continuation']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pilot"); p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True); p.add_argument("--bootstrap", type=int, default=2000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
