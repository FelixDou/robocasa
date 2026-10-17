"""Per-task timer baseline (coordinator/JSAI-thread question, 2026-10-17; exploratory).
A per-task timer alarms at t >= c_k H_k. Every failure ends at H_k, so its lead is 1 - ceil(c_k H_k)/H_k
regardless of the policy. c_k is chosen by dynamic programming to maximise total failure lead under a
pooled false-alarm budget: (A) fitted on calibration (budget = floor(0.05 n_succ)), applied to test;
(B) oracle fitted on the test cohort itself at the test false-alarm count of SAFE + timer (favours the timer).
Usage: python3 -I per_task_timer.py <review_data dir>
"""
import io, contextlib, runpy, sys, math, json
from collections import defaultdict
from pathlib import Path
import numpy as np
here = Path(__file__).resolve().parent
sys.argv = ["lead_time.py", sys.argv[1]]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(here / "lead_time.py"))
cal, te, ev, res = g["cal"], g["te"], g["ev"], g["res"]

def options(rows):
    """Per task: list of (false_alarms, c, lead_sum_on_failures)."""
    S = sorted(r["L"] / r["horizon"] for r in rows if not r["failed"])
    F = [r for r in rows if r["failed"]]
    H = rows[0]["horizon"]
    opts = []
    for f in range(len(S) + 1):
        # alarm index i alarms a success iff i <= L_s; choose smallest index with exactly f successes alarmed
        Ls = sorted((r["L"] for r in rows if not r["failed"]), reverse=True)
        idx = (Ls[f] + 1) if f < len(Ls) else 1
        if f > 0 and f < len(Ls) and Ls[f] == Ls[f - 1]:
            continue
        lead = sum(max(0, r["L"] - idx) / H if idx <= r["L"] else 0.0 for r in F)
        opts.append((f, idx / H, lead))
    return opts

def fit(rows, budget):
    bt = defaultdict(list)
    for r in rows: bt[r["task_name"]].append(r)
    tasks = sorted(bt); opt = {t: options(bt[t]) for t in tasks}
    best = {0: (0.0, {})}
    for t in tasks:
        nb = {}
        for used, (val, ch) in best.items():
            for f, c, lead in opt[t]:
                u = used + f
                if u > budget: continue
                if u not in nb or val + lead > nb[u][0]:
                    nb[u] = (val + lead, {**ch, t: c})
        best = nb
    return max(best.values(), key=lambda x: x[0])[1]

def evaluate(rows, cmap):
    S = [r for r in rows if not r["failed"]]; F = [r for r in rows if r["failed"]]
    def idx(r): return math.ceil(cmap[r["task_name"]] * r["horizon"] - 1e-9)
    fp = sum(idx(r) <= r["L"] for r in S)
    leads = np.array([(r["L"] - idx(r)) / r["horizon"] if idx(r) <= r["L"] else 0.0 for r in F])
    return dict(fpr=round(fp / len(S), 4), fp=fp, recall=float(np.mean([idx(r) <= r["L"] for r in F])), mean_lead=round(float(leads.mean()), 4)), leads

out = {}
nS_cal = sum(not r["failed"] for r in cal)
cA = fit(cal, int(0.05 * nS_cal)); eA, lA = evaluate(te, cA)
out["A_calibrated_per_task_timer"] = eA
safe_timer = res["safe_timer"]["test"]
cB = fit(te, safe_timer["fp"]); eB, lB = evaluate(te, cB)
out["B_oracle_per_task_timer_at_safe_timer_fp"] = eB
out["safe_timer"] = {k: round(safe_timer[k], 4) for k in ("fpr", "recall", "mean_lead")}
out["global_timer"] = {k: round(res["timer"]["test"][k], 4) for k in ("fpr", "recall", "mean_lead")}
# paired bootstrap (task then rollout) of SAFE+timer minus per-task timers
F = [r for r in te if r["failed"]]; ls = safe_timer["lead_vec"]
rng = np.random.default_rng(0); bi = defaultdict(list)
for i, r in enumerate(F): bi[r["task_name"]].append(i)
tasks = sorted(bi)
for name, lv in (("A", lA), ("B", lB)):
    d = ls - lv; bs = []
    for _ in range(2000):
        ii = np.concatenate([rng.choice(bi[t], len(bi[t])) for t in rng.choice(tasks, len(tasks))]); bs.append(d[ii].mean())
    out[f"safe_timer_minus_{name}"] = dict(mean=round(float(d.mean()), 4), ci95=np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist())
# where does the SAFE+timer gain over the global timer sit? split by tasks where time risk never alarms
tau_r = res["risk"]["th"]["risk"]
low = {t for t in tasks if max(float(np.max(r["risk"])) for r in te if r["task_name"] == t and r["failed"]) < tau_r}
gain = ls - res["timer"]["test"]["lead_vec"]
m = np.array([r["task_name"] in low for r in F])
out["gain_split"] = dict(tasks_risk_silent=len(low), gain_in_those=round(float(gain[m].mean()), 4), gain_elsewhere=round(float(gain[~m].mean()), 4),
                         share_of_total_gain_from_those=round(float(gain[m].sum() / gain.sum()), 3))
# SAFE + oracle per-task timer B: does SAFE still add lead on top of the best per-task timer?
pass
(here / "per_task_timer.json").write_text(json.dumps(out, indent=1, default=float))

# ---- realistic per-task timer fitted on ALL retained calibration rollouts (disjoint from test)
load = g["load"]; first = g["first"]
cal_all = load(Path(sys.argv[1]) / "calibration_predictions.jsonl")
bt = defaultdict(list)
for r in cal_all: bt[r["task_name"]].append(r)
succ_frac = {t: [r["L"] / r["horizon"] for r in rs if not r["failed"]] for t, rs in bt.items()}
out["cal_all"] = dict(rollouts=len(cal_all), successes_per_task=np.quantile([len(v) for v in succ_frac.values()], [0, .5, 1]).tolist())
def c_of(t, delta, drop=None):
    v = list(succ_frac[t])
    if drop is not None: v.remove(drop)
    return min(1.0, (max(v) if v else 1.0) + delta)
def loo_fa(delta, tau_s):
    fa = n = 0
    for t, rs in bt.items():
        for r in rs:
            if r["failed"]: continue
            n += 1
            ck = c_of(t, delta, drop=r["L"] / r["horizon"])
            timer_alarm = math.ceil(ck * r["horizon"] - 1e-9) <= r["L"]
            safe_alarm = tau_s is not None and math.isfinite(first(r["safe"], tau_s))
            fa += timer_alarm or safe_alarm
    return fa / n
def cal_lead(delta, tau_s):
    Fc = [r for r in cal_all if r["failed"]]; tot = 0
    for r in Fc:
        a = min(math.ceil(c_of(r["task_name"], delta) * r["horizon"] - 1e-9), first(r["safe"], tau_s) if tau_s is not None else math.inf)
        tot += (r["L"] - a) / r["horizon"] if a <= r["L"] else 0
    return tot / len(Fc)
deltas = np.round(np.arange(0.0, 0.31, 0.01), 2)
best_P = min(d for d in deltas if loo_fa(d, None) <= 0.05)
taus = sorted({float(np.max(r["safe"])) for r in cal_all if not r["failed"]}, reverse=True)[:40]
best_SP = None
for d in deltas:
    for ts in taus:
        if loo_fa(d, ts) <= 0.05:
            v = cal_lead(d, ts)
            if best_SP is None or v > best_SP[0]: best_SP = (v, d, ts)
def ev_test(delta, tau_s):
    S = [r for r in te if not r["failed"]]; Fl = [r for r in te if r["failed"]]
    def a(r): return min(math.ceil(c_of(r["task_name"], delta) * r["horizon"] - 1e-9), first(r["safe"], tau_s) if tau_s is not None else math.inf)
    fp = sum(a(r) <= r["L"] for r in S)
    lv = np.array([(r["L"] - a(r)) / r["horizon"] if a(r) <= r["L"] else 0.0 for r in Fl])
    return dict(delta=float(delta), tau_s=tau_s, fpr=round(fp / len(S), 4), fp=fp, mean_lead=round(float(lv.mean()), 4)), lv
eP, lP = ev_test(best_P, None); eSP, lSP = ev_test(best_SP[1], best_SP[2])
out["P_per_task_timer_retained_cal"] = eP; out["SAFE_plus_P"] = eSP
for name, a_, b_ in (("SAFE+P - P", lSP, lP), ("SAFE+timer - P", ls, lP)):
    d = a_ - b_; bs = []
    for _ in range(2000):
        ii = np.concatenate([rng.choice(bi[t], len(bi[t])) for t in rng.choice(tasks, len(tasks))]); bs.append(d[ii].mean())
    out[name] = dict(mean=round(float(d.mean()), 4), ci95=np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist())
for name, lv in (("oracle_B", lB), ("P", lP), ("SAFE+P", lSP), ("SAFE+timer", ls), ("timer", res["timer"]["test"]["lead_vec"])):
    out.setdefault("lead_by_group", {})[name] = dict(risk_silent_14=round(float(lv[m].mean()), 4), other_24=round(float(lv[~m].mean()), 4))
print(json.dumps({k: out[k] for k in list(out)[-6:]}, indent=1, default=float))
(here / "per_task_timer.json").write_text(json.dumps(out, indent=1, default=float))

# ---- matched-FPR sensitivity: per-task timer with delta tuned on test to SAFE+timer's test FPR (favours P)
target = safe_timer["fpr"]
cands_d = [d for d in np.round(np.arange(0.0, 0.5, 0.005), 3) if ev_test(d, None)[0]["fpr"] <= target + 1e-12]
dm = min(cands_d); eM, lM = ev_test(dm, None)
d_ = ls - lM; bs = []
for _ in range(2000):
    ii = np.concatenate([rng.choice(bi[t], len(bi[t])) for t in rng.choice(tasks, len(tasks))]); bs.append(d_[ii].mean())
out["P_matched_test_fpr"] = eM
out["SAFE+timer - P_matched"] = dict(mean=round(float(d_.mean()), 4), ci95=np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist())
print(json.dumps({k: out[k] for k in ("P_matched_test_fpr", "SAFE+timer - P_matched")}, indent=1, default=float))
(here / "per_task_timer.json").write_text(json.dumps(out, indent=1, default=float))
