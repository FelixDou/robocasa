"""Offline mock-review fixes for the RSS Stage 1 draft (exploratory, opened cohorts).
Items: 4 (time-risk misses), 5 (OR vs AND on test lead/recall-vs-FPR frontiers),
6 (bootstrap over calibration AND test, re-selecting thresholds), R1 question (policy-score
catches in the last 10% of the limit). Usage: python3 -I mock_review_fixes.py <review_data dir>
"""
import io, contextlib, runpy, sys, math, json
from collections import defaultdict
from pathlib import Path
import numpy as np
here = Path(__file__).resolve().parent
sys.argv = ["lead_time.py", sys.argv[1]]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(here / "lead_time.py"))
cal, te, ev, search, res, first = g["cal"], g["te"], g["ev"], g["search"], g["res"], g["first"]
out = {}

# ---- item 4: why the time-risk curve misses failures
tau_r = res["risk"]["th"]["risk"]
F = [r for r in te if r["failed"]]
by_task = defaultdict(list)
for r in F: by_task[r["task_name"]].append(r)
never = [t for t, rs in by_task.items() if max(float(np.max(r["risk"])) for r in rs) < tau_r]
miss = sum(1 for r in F if not math.isfinite(first(r["risk"], tau_r)))
miss_never = sum(len(by_task[t]) for t in never)
curve_max = {t: round(float(max(np.max(r["risk"]) for r in rs)), 3) for t, rs in by_task.items()}
out["item4"] = dict(tau_r=tau_r, misses=miss, tasks_curve_below_tau=len(never), misses_in_those_tasks=miss_never,
                    tasks_total=len(by_task), curve_max_quantiles=np.quantile(list(curve_max.values()), [0, .25, .5, .75, 1]).round(3).tolist())

# ---- R1 question: policy-score catches in the last 10% of the limit (SAFE + timer)
th = res["safe_timer"]["th"]
sf = [first(r["safe"], th["safe"]) / r["horizon"] for r in F]
tf = [first(r["timer"], th["timer"]) / r["horizon"] for r in F]
safe_first = [s for s, t in zip(sf, tf) if s < t]
out["r1"] = dict(safe_first=len(safe_first), safe_first_after_0p9=int(sum(s > 0.9 for s in safe_first)))

# ---- item 5: OR vs AND on test frontiers (oracle sweep on test, descriptive)
def frontier(data, a, b, mode, caps=(0.03, 0.05, 0.075, 0.10)):
    S = [r for r in data if not r["failed"]]; Fl = [r for r in data if r["failed"]]
    ga = sorted({float(np.max(r[a])) for r in data}); gb = sorted({float(np.max(r[b])) for r in data})
    ga = ga[::max(1, len(ga)//120)] + [np.inf]; gb = gb[::max(1, len(gb)//120)] + [np.inf]
    def steps(rs, key, grid): return np.array([[first(r[key], x) if math.isfinite(x) else math.inf for x in grid] for r in rs])
    sa, sb, fa, fb = steps(S, a, ga), steps(S, b, gb), steps(Fl, a, ga), steps(Fl, b, gb)
    L = np.array([r["L"] for r in Fl]); H = np.array([r["horizon"] for r in Fl])
    best = {c: (0, 0) for c in caps}
    comb = np.minimum if mode == "or" else np.maximum
    for i in range(len(ga)):
        st = comb(sa[:, [i]], sb); fpr = np.isfinite(st).mean(0)
        ft = comb(fa[:, [i]], fb); al = np.isfinite(ft)
        rec = al.mean(0); lead = np.where(al, (L[:, None] - np.where(al, ft, 0)) / H[:, None], 0).mean(0)
        for c in caps:
            ok = fpr <= c + 1e-12
            if ok.any():
                best[c] = (max(best[c][0], float(rec[ok].max())), max(best[c][1], float(lead[ok].max())))
    return {str(c): {"recall": round(v[0], 3), "mean_lead": round(v[1], 3)} for c, v in best.items()}
out["item5"] = {f"{m}_{b}": frontier(te, "safe", b, m) for m in ("or", "and") for b in ("risk", "timer")}

# ---- item 6: bootstrap calibration and test together, re-selecting thresholds
rng = np.random.default_rng(0)
def resample(data):
    bt = defaultdict(list)
    for r in data: bt[r["task_name"]].append(r)
    tasks = sorted(bt); pick = rng.choice(tasks, len(tasks))
    o = []
    for t in pick:
        for cls in (False, True):  # keep the per-task class quota
            rs = [r for r in bt[t] if r["failed"] == cls]
            o += [rs[j] for j in rng.choice(len(rs), len(rs))]
    return o
B = int(__import__("os").environ.get("BOOT", 300)); gains, fprs, timer_fprs = [], [], []
for b in range(B):
    c2, t2 = resample(cal), resample(te)
    tt = search(c2, ["timer"]); st = search(c2, ["safe", "timer"])
    et, es = ev(t2, tt), ev(t2, st)
    gains.append(es["mean_lead"] - et["mean_lead"]); fprs.append(es["fpr"]); timer_fprs.append(et["fpr"])
out["item6"] = dict(replicates=B, lead_gain_mean=round(float(np.mean(gains)), 4),
                    lead_gain_ci95=np.round(np.percentile(gains, [2.5, 97.5]), 4).tolist(),
                    fused_test_fpr_ci95=np.round(np.percentile(fprs, [2.5, 97.5]), 4).tolist(),
                    timer_test_fpr_ci95=np.round(np.percentile(timer_fprs, [2.5, 97.5]), 4).tolist(),
                    p_gain_le_0=float(np.mean(np.array(gains) <= 0)))
print(json.dumps(out, indent=1))
(here / "mock_review_fixes.json").write_text(json.dumps(out, indent=1))
