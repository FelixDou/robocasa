"""Post hoc extension of the paired FIPER comparison: the elapsed-time curve over a wider grid.

Plan: docs/fiper_paired_plan.md, section "Post hoc extension". In the paired run every FIPER
operating point had a higher test FPR than the timer reached on FIPER's quantile grid
(0.90-0.99), so the matched-FPR interpolation clamped. The pure timer (FIPER's ct_quantile,
window 1, score = step + 1) is a deterministic function of rollout lengths, so its curve can be
extended to quantiles 0.30-0.99 offline. Before using it, the timer recomputed here must
reproduce, rollout by rollout, the `time` method's alarms from FIPER's own pipeline at
0.90-0.99; any mismatch stops the analysis.

Usage:
  python3 -I extend_timer.py <first_alarm dumps> <fiper_timer index.jsonl> --output OUT.json [--bootstrap 1000]
"""
import argparse, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("analyze_paired", here / "analyze_paired.py")
ap = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ap)

LEVELS = tuple(round(0.30 + 0.01 * i, 2) for i in range(70))     # 0.30 ... 0.99
CHECK_LEVELS = tuple(round(0.90 + 0.01 * i, 2) for i in range(10))
SMALL = 1e-6                                                        # FIPER's add_small_score


def alarm_step(cal_lengths, q):
    """FIPER's ct_quantile on score_t = t + 1 (+1e-6), window 1: the first step t whose
    score / threshold > 1, evaluated with FIPER's own float expression. It does not depend on the
    rollout; a rollout of length L is flagged iff that step is <= L - 1."""
    thr = np.quantile(np.asarray(cal_lengths, float) + SMALL, q)
    t = max(0, int(np.ceil(thr - 1 - SMALL)) - 2)
    while not (float(t + 1) + SMALL) / thr > 1:
        t += 1
    return t


def timer_first(test_lengths, cal_lengths, q):
    t = alarm_step(cal_lengths, q)
    lengths = np.asarray(test_lengths)
    return np.where(lengths - 1 >= t, t, -1).astype(np.int64)


def cal_lengths_from_index(path):
    cal = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("split") == "calibration" and r.get("success"):
                cal[r["task"]].append(int(r["L"]))
    return cal


def curve(d, cal, idx=None):
    ii = slice(None) if idx is None else idx
    succ, lens, T = d["successful"][ii], d["lengths"][ii], d["max_episode_length"]
    steps = [alarm_step(cal, q) for q in LEVELS]
    pts = [ap.metrics(np.where(lens - 1 >= t, t, -1), succ, lens, T) for t in steps]
    return np.array([p["fpr"] for p in pts]), np.array([p["lead"] for p in pts])


def fiper_points(data, task, sel, idx=None):
    style, w = sel
    per_q = defaultdict(list)
    for seed in sorted(data[task]):
        d = data[task][seed][ap.FIPER]
        ii = slice(None) if idx is None else idx
        for (s, q, ww), first in d["first_alarm"].items():
            if (s, ww) == (style, w):
                per_q[q].append(ap.metrics(first[ii], d["successful"][ii], d["lengths"][ii], d["max_episode_length"]))
    return {q: (float(np.mean([m["fpr"] for m in v])), float(np.mean([m["lead"] for m in v]))) for q, v in sorted(per_q.items())}


def matched(data, cal, task, sel, idx=None):
    ref = data[task][min(data[task])][ap.TIME]
    tf, tl = curve(ref, cal[task], idx)
    order = np.argsort(tf, kind="mergesort")
    tf, tl = tf[order], tl[order]
    rows, clamped = [], 0
    for q, (f, l) in fiper_points(data, task, sel, idx).items():
        clamped += not (tf.min() <= f <= tf.max())
        rows.append(dict(q=q, fiper_fpr=f, fiper_lead=l, timer_lead=float(np.interp(f, tf, tl)), diff=l - float(np.interp(f, tf, tl))))
    return dict(diff=float(np.mean([r["diff"] for r in rows])), clamped=clamped, n=len(rows), rows=rows,
                timer_fpr_range=[float(tf.min()), float(tf.max())])


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        sys.exit(f"error: output exists: {out_path}")
    data = ap.load(args.dumps)
    cal = cal_lengths_from_index(args.index)
    tasks = sorted(data)
    # 1. Exact-match check against the `time` method inside FIPER's pipeline.
    mismatches = []
    for t in tasks:
        for seed, d in data[t].items():
            tm = d[ap.TIME]
            for q in CHECK_LEVELS:
                key = ("ct_quantile", q, "1")
                if key not in tm["first_alarm"]:
                    sys.exit(f"error: dump lacks {key} for time ({t}, seed {seed})")
                ours = timer_first(tm["lengths"], cal[t], q)
                if not np.array_equal(ours, tm["first_alarm"][key]):
                    mismatches.append((t, seed, q, int(np.sum(ours != tm["first_alarm"][key]))))
    if mismatches:
        print("STOP: recomputed timer differs from FIPER's time method:", mismatches[:10])
        sys.exit(1)
    print(f"exact-match check passed: {len(tasks)} tasks x {len(next(iter(data.values())))} seeds x {len(CHECK_LEVELS)} levels")
    # 2. FIPER's selected configuration (FIPER's rule, as in analyze_paired).
    sel, _ = ap.select(ap.grid(data, ap.FIPER))
    point = {t: matched(data, cal, t, sel) for t in tasks}
    # 3. Bootstrap over test rollouts, within task and outcome.
    rng = np.random.default_rng(0)
    boot = defaultdict(list)
    for _ in range(args.bootstrap):
        diffs = []
        for t in tasks:
            s = data[t][min(data[t])][ap.TIME]["successful"]
            si, fi = np.flatnonzero(s), np.flatnonzero(~s)
            idx = np.concatenate([rng.choice(si, len(si)), rng.choice(fi, len(fi))])
            v = matched(data, cal, t, sel, idx)["diff"]
            boot[t].append(v); diffs.append(v)
        boot["average"].append(float(np.mean(diffs)))
    ci = {k: np.round(np.percentile(v, [2.5, 97.5]), 4).tolist() for k, v in boot.items()}
    out = dict(plan="docs/fiper_paired_plan.md (post hoc extension)", status="exploratory, post hoc",
               timer="ct_quantile, window 1, score = step + 1, levels 0.30-0.99", fiper_selected=dict(style=sel[0], window=sel[1]),
               exact_match_levels=list(CHECK_LEVELS), tasks={t: point[t] for t in tasks},
               ci95={t: ci[t] for t in tasks}, average=dict(diff=round(float(np.mean([point[t]["diff"] for t in tasks])), 4), ci95=ci["average"]),
               bootstrap=args.bootstrap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("FIPER selected:", sel)
    print(f"{'task':11s} {'FIPER FPR':>9s} {'FIPER lead':>10s} {'timer lead @FPR':>15s} {'diff':>7s}  95% CI            clamped  timer FPR range")
    for t in tasks:
        p = point[t]
        print(f"{t:11s} {np.mean([r['fiper_fpr'] for r in p['rows']]):9.3f} {np.mean([r['fiper_lead'] for r in p['rows']]):10.3f} "
              f"{np.mean([r['timer_lead'] for r in p['rows']]):15.3f} {p['diff']:+7.3f}  {ci[t]}  {p['clamped']}/{p['n']}  "
              f"[{p['timer_fpr_range'][0]:.2f}, {p['timer_fpr_range'][1]:.2f}]")
    print("average diff (FIPER - timer at matched FPR):", out["average"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps"); p.add_argument("index"); p.add_argument("--output", required=True); p.add_argument("--bootstrap", type=int, default=1000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
