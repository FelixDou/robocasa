"""Elapsed-time baseline on FIPER's released rollouts, under FIPER's own protocol and metrics.

Plan and fixed choices: docs/published_eval_reanalysis_plan.md ("Observed structure and fixed
analysis choices"). Two passes:
  index    read every rollout pickle once with the restricted unpickler and keep only task,
           split, outcome, length and sub-type (index.jsonl);
  analyze  from the index: length by outcome, ROC-AUC of length, and the timer's TPR / TNR /
           Acc / DT / TWA averaged over 1 - delta in {0.90, ..., 0.99}, next to FIPER's
           published Table 1 values.

Usage (from outside the data directory):
  python3 -I fiper_timer.py index <data_all dir> --output INDEX.jsonl [--workers 8]
  python3 -I fiper_timer.py analyze INDEX.jsonl --output RESULTS.json
"""
import argparse, importlib.util, json, math, re, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("inspect_external", here / "inspect_external.py")
ie = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ie)

LEVELS = tuple(round(0.90 + 0.01 * i, 2) for i in range(10))      # 1 - delta
FIPER_TABLE1 = {  # FIPER column of Table 1 (mean over five seeds): TWA, Acc, DT
    "sorting": (0.54, 0.66, 0.32), "stacking": (0.62, 0.73, 0.28), "push_t": (0.55, 0.71, 0.32),
    "pretzel": (0.68, 0.85, 0.33), "push_chair": (0.83, 0.96, 0.27), "average": (0.65, 0.78, 0.30)}
NAME = re.compile(r"episode_(?:(?P<o>[sf])_)?(?P<n>\d+)\.pkl$")


def read_one(path):
    path = Path(path)
    m = NAME.match(path.name)
    if not m:
        return dict(path=str(path), skipped="unrecognised name")
    try:
        d = ie.safe_load(path)
    except ie.Refused as e:
        return dict(path=str(path), refused=str(e))
    meta = d.get("metadata", {}) if isinstance(d, dict) else {}
    rollout = d.get("rollout") if isinstance(d, dict) else d
    by_name = None if m.group("o") is None else (m.group("o") == "s")
    by_meta = meta.get("successful")
    if by_name is not None and by_meta is not None and bool(by_meta) != by_name:
        return dict(path=str(path), conflict=f"name says success={by_name}, metadata says {by_meta}")
    success = by_name if by_name is not None else by_meta
    if success is None:
        return dict(path=str(path), skipped="no outcome")
    parts = path.parts
    i = parts.index("rollouts")
    return dict(path=str(path), task=parts[i - 1], split=parts[i + 1], success=bool(success), L=len(rollout),
                subtype=meta.get("rollout_subtype"), meta_task=meta.get("task"), num_steps=meta.get("num_steps"))


def index(args):
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"error: output exists: {out}")
    root = Path(args.data_dir).resolve()
    if Path.cwd().resolve() == root or root in Path.cwd().resolve().parents:
        raise SystemExit("error: run from outside the data directory")
    files = sorted(p for p in root.rglob("*.pkl") if "rollouts" in p.parts)
    with Pool(args.workers) as pool:
        rows = pool.map(read_one, files, chunksize=4)
    bad = [r for r in rows if "refused" in r or "conflict" in r]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"indexed {len(rows)} files; refused {sum('refused' in r for r in rows)}, "
          f"conflicts {sum('conflict' in r for r in rows)}, skipped {sum('skipped' in r for r in rows)}")
    if bad:
        for r in bad[:10]:
            print("STOP:", r)
        raise SystemExit("error: refused or conflicting files; stop and report")


def auc(labels, scores):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def threshold(maxima, level, method):
    m = np.sort(np.asarray(maxima, float))
    if method == "linear":
        return float(np.quantile(m, level))
    k = math.ceil((len(m) + 1) * level)
    return math.inf if k > len(m) else float(m[k - 1])


def timer_metrics(cal_s, test, T, method):
    per = []
    for level in LEVELS:
        g = threshold([r["L"] - 1 for r in cal_s], level, method)
        t_alarm = math.inf if math.isinf(g) else math.floor(g) + 1
        S = [r for r in test if r["success"]]; F = [r for r in test if not r["success"]]
        flagged = lambda r: t_alarm <= r["L"] - 1
        tnr = float(np.mean([not flagged(r) for r in S])) if S else None
        tp = [r for r in F if flagged(r)]
        tpr = len(tp) / len(F) if F else None
        dt = float(t_alarm / T) if tp else None
        twa = 0.5 * (sum(1 - t_alarm / T for _ in tp) / len(F) + tnr) if F and S else None
        per.append(dict(level=level, gamma=g, t_alarm=t_alarm, tpr=tpr, tnr=tnr, acc=0.5 * (tpr + tnr) if F and S else None,
                        dt=dt, twa=twa))
    mean = lambda k: round(float(np.mean([p[k] for p in per if p[k] is not None])), 4) if any(p[k] is not None for p in per) else None
    return dict(mean={k: mean(k) for k in ("tpr", "tnr", "acc", "dt", "twa")}, per_level=per)


def summarize(rows):
    q = lambda xs: np.round(np.quantile(xs, [0, 0.25, 0.5, 0.75, 1]), 1).tolist() if xs else None
    return dict(n=len(rows), L_success=q([r["L"] for r in rows if r["success"]]),
                L_failure=q([r["L"] for r in rows if not r["success"]]))


def analyze(args):
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"error: output exists: {out}")
    rows = [json.loads(l) for l in Path(args.index).read_text().splitlines() if l.strip()]
    if any("refused" in r or "conflict" in r for r in rows):
        raise SystemExit("error: index contains refused or conflicting files")
    rows = [r for r in rows if "skipped" not in r]
    tasks = sorted({r["task"] for r in rows})
    res = dict(plan="docs/published_eval_reanalysis_plan.md", status="exploratory re-analysis of released data",
               levels=list(LEVELS), tasks={})
    for task in tasks:
        tr = [r for r in rows if r["task"] == task]
        cal = [r for r in tr if r["split"] == "calibration"]
        cal_s = [r for r in cal if r["success"]]
        test = [r for r in tr if r["split"] == "test"]
        T = max(r["L"] for r in cal + test)
        entry = dict(T=T, calibration_successes=len(cal_s), calibration_failures_ignored=len(cal) - len(cal_s),
                     test_successes=sum(r["success"] for r in test), test_failures=sum(not r["success"] for r in test),
                     calibration=summarize(cal_s), test=summarize(test),
                     auc_length=auc([not r["success"] for r in test], [r["L"] for r in test]),
                     timer={m: timer_metrics(cal_s, test, T, m) for m in ("linear", "conformal")},
                     fiper_table1=dict(zip(("twa", "acc", "dt"), FIPER_TABLE1[task])) if task in FIPER_TABLE1 else None,
                     by_subtype={})
        for st in sorted({str(r["subtype"]) for r in test}):
            sub = [r for r in test if str(r["subtype"]) == st]
            entry["by_subtype"][st] = dict(n_success=sum(r["success"] for r in sub), n_failure=sum(not r["success"] for r in sub),
                                           auc_length=auc([not r["success"] for r in sub], [r["L"] for r in sub]),
                                           timer=timer_metrics(cal_s, sub, T, "linear")["mean"])
        res["tasks"][task] = entry
    lin = [res["tasks"][t]["timer"]["linear"]["mean"] for t in tasks]
    res["average"] = {k: round(float(np.mean([m[k] for m in lin if m[k] is not None])), 4) for k in ("tpr", "tnr", "acc", "dt", "twa")}
    res["average_fiper_table1"] = dict(zip(("twa", "acc", "dt"), FIPER_TABLE1["average"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"{'task':11s} {'T':>4s} {'M':>3s} {'test S/F':>9s} {'AUC(L)':>7s} | timer TWA  Acc   DT   (TPR TNR) | FIPER TWA Acc DT | L succ median / fail median")
    for t in tasks:
        e = res["tasks"][t]; m = e["timer"]["linear"]["mean"]; f = e["fiper_table1"] or {}
        print(f"{t:11s} {e['T']:4d} {e['calibration_successes']:3d} {e['test_successes']:4d}/{e['test_failures']:<4d} "
              f"{(e['auc_length'] or float('nan')):7.3f} | {m['twa']}  {m['acc']}  {m['dt']}  ({m['tpr']} {m['tnr']}) | "
              f"{f.get('twa')} {f.get('acc')} {f.get('dt')} | {e['test']['L_success'][2] if e['test']['L_success'] else None} / "
              f"{e['test']['L_failure'][2] if e['test']['L_failure'] else None}")
    print("average timer:", res["average"], "| FIPER Table 1 average:", res["average_fiper_table1"])
    print("conformal-rank sensitivity (TWA/Acc/DT):", {t: {k: res['tasks'][t]['timer']['conformal']['mean'][k] for k in ('twa', 'acc', 'dt')} for t in tasks})
    print("calibration failures ignored:", {t: res["tasks"][t]["calibration_failures_ignored"] for t in tasks})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("data_dir"); a.add_argument("--output", required=True); a.add_argument("--workers", type=int, default=8)
    b = sub.add_parser("analyze"); b.add_argument("index"); b.add_argument("--output", required=True)
    args = p.parse_args()
    index(args) if args.cmd == "index" else analyze(args)


if __name__ == "__main__":
    main()
