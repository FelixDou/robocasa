"""Analyse the paired FIPER run: FIPER and an elapsed-time method in FIPER's own pipeline.

Plan: docs/fiper_paired_plan.md. Reads the per-seed first-alarm dumps written by the patched
run_fiper.py (seed<seed>_<task>.pkl), recomputes FIPER's metrics (TPR, TNR, balanced and plain
accuracy, DT, TWA) plus lead, applies FIPER's selection rule (per method: the threshold style
and window with the best TWA averaged over tasks, quantiles and seeds), checks FIPER against
its published Table 1, and compares lead at matched FPR with a bootstrap over test rollouts.

Usage:
  python3 -I analyze_paired.py <dump dir> --output OUT.json [--bootstrap 1000]
"""
import argparse, importlib.util, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("inspect_external", here.parent / "inspect_external.py")
ie = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ie)

FIPER = "rnd_oe_and_entropy"
TIME = "time"
REPORT = (TIME, FIPER, "rnd_oe", "entropy", "rnd_oe_or_entropy")
TABLE1 = {"sorting": (0.54, 0.66, 0.32), "stacking": (0.62, 0.73, 0.28), "push_t": (0.55, 0.71, 0.32),
          "pretzel": (0.68, 0.85, 0.33), "push_chair": (0.83, 0.96, 0.27)}   # TWA, Acc, DT
TOL = 0.05


def load(dump_dir):
    data = defaultdict(dict)       # data[task][seed] = {method: {...}}
    for f in sorted(Path(dump_dir).glob("seed*_*.pkl")):
        m = re.match(r"seed(\d+)_(.+)\.pkl$", f.name)
        data[m.group(2)][int(m.group(1))] = ie.safe_load(f)
    if not data:
        sys.exit(f"error: no seed*_*.pkl dumps in {dump_dir}")
    return data


def metrics(first, success, lengths, T):
    """FIPER's definitions (detection patience 0), plus lead."""
    det = first >= 0
    S, F = success, ~success
    tp, fn = np.sum(det & F), np.sum(~det & F)
    tn, fp = np.sum(~det & S), np.sum(det & S)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    dts = first[det & F] / (T - 1)
    dt = float(np.mean(dts)) if len(dts) else 1.0
    nf, ns = F.sum(), S.sum()
    twa = 0.5 * ((np.sum(1 - dts) / nf if nf else 0.0) + (tn / ns if ns else 0.0))
    lead = np.where(det & F, (lengths - 1 - first) / (T - 1), 0.0)[F]
    return dict(tpr=float(tpr), tnr=float(tnr), fpr=float(1 - tnr), balanced_acc=float(0.5 * (tpr + tnr)),
                accuracy=float((tp + tn) / len(first)), dt=dt, twa=float(twa), lead=float(lead.mean()) if nf else None)


def configs(data, method):
    keys = None
    for task in data:
        for seed in data[task]:
            k = set(data[task][seed][method]["first_alarm"])
            keys = k if keys is None else keys & k
    return sorted(keys, key=str)


def grid(data, method, idx=None, only=None):
    """grid[task][(style, q, w)] = seed-averaged metrics; idx optionally resamples test rollouts;
    only=(style, window) restricts to one configuration (used in the bootstrap)."""
    out = {}
    for task in sorted(data):
        per = defaultdict(list)
        for seed in sorted(data[task]):
            d = data[task][seed][method]
            ii = slice(None) if idx is None else idx[task]
            succ, lens, T = d["successful"][ii], d["lengths"][ii], d["max_episode_length"]
            for key, first in d["first_alarm"].items():
                if only is not None and (key[0], key[2]) != only:
                    continue
                per[key].append(metrics(first[ii], succ, lens, T))
        out[task] = {k: {m: float(np.mean([v[m] for v in vs if v[m] is not None])) for m in vs[0]} for k, vs in per.items()}
    return out


def select(g):
    """FIPER's rule: per (style, window), TWA averaged over tasks and quantiles; take the best."""
    scores = defaultdict(list)
    for task, by_key in g.items():
        for (style, q, w), m in by_key.items():
            scores[(style, w)].append(m["twa"])
    best = max(scores, key=lambda k: (np.mean(scores[k]), -len(str(k[1]))))
    return best, {f"{s}|w={w}": round(float(np.mean(v)), 4) for (s, w), v in sorted(scores.items(), key=lambda kv: -np.mean(kv[1]))[:5]}


def at(g, sel):
    style, w = sel
    out = {}
    for task, by_key in g.items():
        rows = [m for (s, q, ww), m in by_key.items() if s == style and ww == w]
        out[task] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
    return out


def curve(g, sel, task):
    style, w = sel
    pts = sorted(((m["fpr"], m["lead"]) for (s, q, ww), m in g[task].items() if s == style and ww == w))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def matched(gf, sf, gt, st):
    """Per task: FIPER lead minus timer lead interpolated at FIPER's FPR, averaged over FIPER's quantiles."""
    out = {}
    for task in gf:
        tf, tl = curve(gt, st, task)
        rows = [m for (s, q, ww), m in gf[task].items() if s == sf[0] and ww == sf[1]]
        diffs, clamped = [], 0
        for m in rows:
            clamped += not (tf.min() <= m["fpr"] <= tf.max())
            diffs.append(m["lead"] - float(np.interp(m["fpr"], tf, tl)))
        out[task] = dict(diff=float(np.mean(diffs)), clamped=clamped, n=len(rows))
    return out


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        sys.exit(f"error: output exists: {out_path}")
    data = load(args.dumps)
    tasks = sorted(data)
    methods = [m for m in REPORT if all(m in data[t][s] for t in data for s in data[t])]
    if FIPER not in methods or TIME not in methods:
        sys.exit(f"error: dumps lack {FIPER} or {TIME}; found {methods}")
    grids, sels, tops = {}, {}, {}
    for m in methods:
        grids[m] = grid(data, m)
        sels[m], tops[m] = select(grids[m])
    table = {m: at(grids[m], sels[m]) for m in methods}
    for m in methods:
        table[m]["average"] = {k: round(float(np.mean([table[m][t][k] for t in tasks])), 4) for k in table[m][tasks[0]]}
    repro = {}
    for t in tasks:
        if t in TABLE1:
            twa, acc, dt = TABLE1[t]; r = table[FIPER][t]
            repro[t] = dict(published=dict(twa=twa, acc=acc, dt=dt), reproduced=dict(twa=r["twa"], balanced_acc=r["balanced_acc"], accuracy=r["accuracy"], dt=r["dt"]),
                            within_tol=bool(abs(r["twa"] - twa) <= TOL and min(abs(r["balanced_acc"] - acc), abs(r["accuracy"] - acc)) <= TOL))
    point = matched(grids[FIPER], sels[FIPER], grids[TIME], sels[TIME])
    rng = np.random.default_rng(0)
    ref = {t: data[t][min(data[t])][TIME] for t in tasks}
    boot = defaultdict(list)
    for _ in range(args.bootstrap):
        idx = {}
        for t in tasks:
            s = ref[t]["successful"]; si, fi = np.flatnonzero(s), np.flatnonzero(~s)
            idx[t] = np.concatenate([rng.choice(si, len(si)), rng.choice(fi, len(fi))])
        mt = matched(grid(data, FIPER, idx, sels[FIPER]), sels[FIPER], grid(data, TIME, idx, sels[TIME]), sels[TIME])
        for t in tasks:
            boot[t].append(mt[t]["diff"])
        boot["average"].append(np.mean([mt[t]["diff"] for t in tasks]))
    ci = {k: np.round(np.percentile(v, [2.5, 97.5]), 4).tolist() for k, v in boot.items()}
    same_q = {}
    for t in tasks:
        rows = {}
        for (s, q, w), m in grids[FIPER][t].items():
            if (s, w) == sels[FIPER]:
                rows[q] = [m["lead"], m["fpr"]]
        trow = {q: [m["lead"], m["fpr"]] for (s, q, w), m in grids[TIME][t].items() if (s, w) == sels[TIME]}
        common = sorted(set(rows) & set(trow), key=float)
        same_q[t] = dict(lead_diff=round(float(np.mean([rows[q][0] - trow[q][0] for q in common])), 4),
                         fpr_diff=round(float(np.mean([rows[q][1] - trow[q][1] for q in common])), 4), quantiles=len(common))
    out = dict(plan="docs/fiper_paired_plan.md", status="exploratory", tasks=tasks,
               seeds={t: sorted(data[t]) for t in tasks}, selected={m: dict(style=sels[m][0], window=sels[m][1], top5=tops[m]) for m in methods},
               table=table, reproduction=repro,
               matched_fpr_lead_diff_fiper_minus_time={t: dict(**point[t], ci95=ci[t]) for t in tasks},
               matched_fpr_average=dict(diff=round(float(np.mean([point[t]["diff"] for t in tasks])), 4), ci95=ci["average"]),
               same_quantile_fiper_minus_time=same_q, bootstrap=args.bootstrap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("selected (style, window):", {m: sels[m] for m in methods})
    print(f"{'method':20s} {'task':11s}  TWA    bAcc   Acc    DT     FPR    lead")
    for m in methods:
        for t in tasks + ["average"]:
            r = table[m][t]
            print(f"{m:20s} {t:11s}  {r['twa']:.3f}  {r['balanced_acc']:.3f}  {r['accuracy']:.3f}  {r['dt']:.3f}  {r['fpr']:.3f}  {r['lead']:.3f}")
    print("reproduction of FIPER Table 1 (within ±0.05):", {t: v["within_tol"] for t, v in repro.items()})
    print("lead at matched FPR, FIPER - time:", {t: (round(point[t]['diff'], 4), ci[t], point[t]['clamped']) for t in tasks},
          "| average", out["matched_fpr_average"])
    print("same quantile, FIPER - time (lead, FPR):", {t: (v["lead_diff"], v["fpr_diff"]) for t, v in same_q.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps"); p.add_argument("--output", required=True); p.add_argument("--bootstrap", type=int, default=1000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
