"""Analyse the XR-1 same-scene retry pilot (plan: docs/xr1_retry_pilot_plan.md).

Gates first: every pilot rollout must match an original rollout ID with the same task, seed and
episode, and first video frames must match (mean absolute pixel difference <= 2.0 for >= 95% of
pairs). Then: r_fail (retry success on originally failed scenes), r_succ (on originally
successful scenes) and the selected tasks' original success rate p, with Wilson and
task-cluster bootstrap intervals.

Usage:
  python3 -I analyze_pilot.py <pilot collection dir> <confirmatory_collection dir> \
      --selection <pilot root>/selection.json --output OUT.json [--bootstrap 2000]
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

FRAME_TOL, FRAME_SHARE = 2.0, 0.95


def manifest(collection):
    rows = {}
    for shard in sorted(Path(collection).glob("shard*")):
        m = shard / "manifest.jsonl"
        if m.is_file():
            for line in m.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); r["_shard"] = str(shard); rows[r["rollout_id"]] = r
    return rows


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


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        sys.exit(f"error: output exists: {out_path}")
    pilot, orig = manifest(args.pilot), manifest(args.original)
    sel = json.loads(Path(args.selection).read_text())
    # Gate 1: identity.
    bad = [rid for rid, r in pilot.items() if rid not in orig or
           (r["task_name"], r["environment_seed"], r["environment_reset_index"]) !=
           (orig[rid]["task_name"], orig[rid]["environment_seed"], orig[rid]["environment_reset_index"])]
    if bad or not pilot:
        print("STOP: identity gate failed:", len(bad), "of", len(pilot), bad[:5]); sys.exit(1)
    # Gate 2: same scene (first frames).
    diffs = []
    for rid, r in pilot.items():
        o = orig[rid]
        if not r.get("video_path") or not o.get("video_path"):
            print("STOP: missing video for", rid); sys.exit(1)
        a = first_frame(Path(r["_shard"]) / r["video_path"]); b = first_frame(Path(o["_shard"]) / o["video_path"])
        if a.shape != b.shape:
            print("STOP: frame shapes differ for", rid, a.shape, b.shape); sys.exit(1)
        diffs.append(float(np.mean(np.abs(a - b))))
    share = float(np.mean(np.array(diffs) <= FRAME_TOL))
    gates = dict(identity=True, frames_within_tol=round(share, 4), frame_diff_quantiles=np.round(np.quantile(diffs, [0, 0.5, 0.95, 1]), 3).tolist(),
                 scene_gate_passed=share >= FRAME_SHARE)
    print("gates:", gates)
    status = "valid"
    if args.post_hoc_exclude_above is not None:
        # Post hoc (see the plan): keep the registered verdict on record, drop only pairs whose
        # first frames differ by more than the given cutoff (different scenes), and label the result.
        cut = float(args.post_hoc_exclude_above)
        excluded = sorted(rid for rid, d in zip(list(pilot), diffs) if d > cut)
        pilot = {rid: r for rid, r in pilot.items() if rid not in excluded}
        gates["post_hoc_excluded"] = excluded
        status = (f"post hoc: registered scene gate {'passed' if share >= FRAME_SHARE else 'failed'} "
                  f"({share:.4f} within {FRAME_TOL}); {len(excluded)} pairs with frame difference > {cut} excluded")
        print("POST HOC:", status)
    elif share < FRAME_SHARE:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dict(status="void: scenes not reproduced", gates=gates), indent=1) + "\n")
        print("STOP: scene gate failed; pilot void"); sys.exit(1)
    # Endpoints.
    per = defaultdict(lambda: dict(fail_n=0, fail_succ=0, succ_n=0, succ_succ=0))
    for rid, r in pilot.items():
        t = per[r["task_name"]]
        retry_success = not r["failed"]
        if orig[rid]["failed"]:
            t["fail_n"] += 1; t["fail_succ"] += retry_success
        else:
            t["succ_n"] += 1; t["succ_succ"] += retry_success
    tasks = sorted(per)
    fn = sum(per[t]["fail_n"] for t in tasks); fs = sum(per[t]["fail_succ"] for t in tasks)
    sn = sum(per[t]["succ_n"] for t in tasks); ss = sum(per[t]["succ_succ"] for t in tasks)
    p = float(np.mean([sel["tasks"][t]["p"] for t in tasks]))
    rng = np.random.default_rng(0); boot = defaultdict(list)
    for _ in range(args.bootstrap):
        draw = rng.choice(tasks, len(tasks))
        f_n = sum(per[t]["fail_n"] for t in draw); f_s = sum(per[t]["fail_succ"] for t in draw)
        s_n = sum(per[t]["succ_n"] for t in draw); s_s = sum(per[t]["succ_succ"] for t in draw)
        pp = np.mean([sel["tasks"][t]["p"] for t in draw])
        if f_n:
            boot["r_fail"].append(f_s / f_n); boot["r_fail_minus_p"].append(f_s / f_n - pp)
        if s_n:
            boot["r_succ"].append(s_s / s_n)
    ci = {k: np.round(np.percentile(v, [2.5, 97.5]), 4).tolist() for k, v in boot.items()}
    res = dict(plan="docs/xr1_retry_pilot_plan.md", status=status, gates=gates,
               r_fail=dict(successes=fs, n=fn, rate=round(fs / fn, 4) if fn else None, wilson95=wilson(fs, fn), cluster95=ci.get("r_fail")),
               r_succ=dict(successes=ss, n=sn, rate=round(ss / sn, 4) if sn else None, wilson95=wilson(ss, sn), cluster95=ci.get("r_succ")),
               p_task_macro=round(p, 4), r_fail_minus_p=dict(point=round(fs / fn - p, 4) if fn else None, cluster95=ci.get("r_fail_minus_p")),
               per_task={t: dict(p=sel["tasks"][t]["p"], type=sel["tasks"][t].get("task_type"), **per[t]) for t in tasks})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"r_fail = {fs}/{fn} = {res['r_fail']['rate']}  Wilson {res['r_fail']['wilson95']}  cluster {res['r_fail']['cluster95']}")
    print(f"r_succ = {ss}/{sn} = {res['r_succ']['rate']}  Wilson {res['r_succ']['wilson95']}  cluster {res['r_succ']['cluster95']}")
    print(f"p (task-macro) = {res['p_task_macro']}  |  r_fail - p = {res['r_fail_minus_p']}")
    for t in tasks:
        v = res["per_task"][t]
        print(f"  {t:40s} {str(v['type']):9s} p={v['p']:.2f}  failed scenes {v['fail_succ']}/{v['fail_n']}  successful scenes {v['succ_succ']}/{v['succ_n']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pilot"); p.add_argument("original"); p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True); p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--post-hoc-exclude-above", type=float,
                   help="post hoc only: exclude pairs whose first-frame difference exceeds this; the result is labelled post hoc")
    run(p.parse_args())


if __name__ == "__main__":
    main()
