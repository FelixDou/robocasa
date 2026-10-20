"""Per-task quantile timer and SAFE + quantile timer, calibrated with Proposition 1.

This is the comparator registered in docs/xr1_confirmatory_amendment_1_lead_time.md.

Per-task quantile timer P_q. For task k, rho_k = q-quantile (numpy, linear) of the
development success durations L/H_k, read from the time-risk curves' active counts in
runtime_bundle.json (every development failure runs to H_k, so a drop in the active count
at inference index i means a success ended there). The timer alarms at the first
inference t >= c_k H_k with c_k = min(1, rho_k + delta). An alarm at the final inference counts.

Fused detector SAFE + P_q (OR rule). It follows the nested path gamma(lam):
delta = lam, tau_s = u + s*lam. (u, s) is chosen on calibration FAILURES only, by the mean
lead at the conformal lam. Calibration successes are never used to pick the path, which
Proposition 1 requires.

Proposition 1. A success i becomes silent at V_i = the smallest lam with
L_i/H < rho_k + lam and M_i < tau_s(lam), where M_i is the running-max SAFE score. lam_hat is
the ceil((n+1)(1-alpha))-th smallest V_i over the n calibration successes.

Usage: python3 -I quantile_timer.py <review_data dir> [calibration jsonl ...]
Default calibration set: the primary balanced cohort, 3 successes and 3 failures per task, as
selected by lead_time.py. For the confirmatory run, pass the 380-success balanced cohort.
"""
import io, contextlib, runpy, sys, math, json
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
D = Path(sys.argv[1])
sys.argv = ["lead_time.py", str(D)]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(here / "lead_time.py"))
cal, te, first, ev, res = g["cal"], g["te"], g["first"], g["ev"], g["res"]
ALPHA = 0.05
QS = (0.75, 0.90, 0.95, 1.00)      # 0.95 is the registered primary; the others are sensitivity
PRIMARY_Q = 0.95
EPS = 1e-9

bundle = json.load(open(D / "runtime_bundle.json"))
def dev_success_fractions(task):
    ac = bundle["time_risk_curves"][task]["active_counts"]; H = len(ac)
    fr = []
    for i in range(1, H):                 # count drops between index i-1 and i: successes of length i
        fr += [i / H] * (ac[i - 1] - ac[i])
    return fr
DEV = {t: dev_success_fractions(t) for t in bundle["time_risk_curves"]}

def rho(q):
    return {t: float(np.quantile(v, q)) if v else 1.0 for t, v in DEV.items()}

def timer_idx(r, rk, delta):
    c = min(1.0, rk[r["task_name"]] + delta)
    return math.ceil(c * r["horizon"] - EPS)

def V(r, rk, u, s):
    """Smallest lam at which success r is silent on the path (inf if never)."""
    if r["L"] >= r["horizon"]:
        return math.inf                    # c is capped at 1, so the final-inference alarm fires
    v_t = r["L"] / r["horizon"] - rk[r["task_name"]]
    M = float(np.max(r["safe"]))
    if u == math.inf:
        v_s = -math.inf
    elif s == 0:
        v_s = -math.inf if u > M else math.inf
    else:
        v_s = (M - u) / s
    return max(v_t, v_s, 0.0) + EPS

def conformal_lam(succ, rk, u, s, alpha=ALPHA):
    vs = sorted(V(r, rk, u, s) for r in succ); n = len(vs)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else vs[k - 1]

def alarm(r, rk, lam, u, s):
    a = timer_idx(r, rk, lam)
    if u != math.inf:
        a = min(a, first(r["safe"], u + s * lam))
    return a

def evaluate(rows, rk, lam, u, s):
    S = [r for r in rows if not r["failed"]]; F = [r for r in rows if r["failed"]]
    fp = sum(alarm(r, rk, lam, u, s) <= r["L"] for r in S)
    lv, final = [], 0
    for r in F:
        a = alarm(r, rk, lam, u, s)
        lv.append((r["L"] - a) / r["horizon"] if a <= r["L"] else 0.0)
        final += a == r["L"]
    lv = np.array(lv)
    return dict(fpr=round(fp / len(S), 4), fp=int(fp), mean_lead=round(float(lv.mean()), 4),
                recall=float(np.mean([alarm(r, rk, lam, u, s) <= r["L"] for r in F])),
                alarms_at_final_inference=int(final)), lv

def cal_fail_lead(calF, rk, lam, u, s):
    return float(np.mean([(r["L"] - a) / r["horizon"] if (a := alarm(r, rk, lam, u, s)) <= r["L"] else 0.0 for r in calF]))

def fit(calrows, q):
    rk = rho(q)
    S = [r for r in calrows if not r["failed"]]; F = [r for r in calrows if r["failed"]]
    timer = dict(u=math.inf, s=0.0, lam=conformal_lam(S, rk, math.inf, 0.0))
    # Path family for the fused detector, built from calibration-failure scores only.
    fm = np.array([float(np.max(r["safe"])) for r in F])
    us = sorted(set(np.round(np.quantile(fm, np.linspace(0, 1, 21)), 4).tolist()))
    best = (cal_fail_lead(F, rk, timer["lam"], math.inf, 0.0), math.inf, 0.0, timer["lam"])
    for u in us:
        for s in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
            lam = conformal_lam(S, rk, u, s)
            if not math.isfinite(lam):
                continue
            v = cal_fail_lead(F, rk, lam, u, s)
            if v > best[0] + 1e-12:
                best = (v, u, s, lam)
    fused = dict(u=best[1], s=best[2], lam=best[3])
    return rk, timer, fused

def boot(d, rows_failed, B=2000, seed=0):
    rng = np.random.default_rng(seed); bi = defaultdict(list)
    for i, r in enumerate(rows_failed): bi[r["task_name"]].append(i)
    tasks = sorted(bi); bs = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(bi[t], len(bi[t])) for t in rng.choice(tasks, len(tasks))]); bs.append(d[ii].mean())
    return dict(mean=round(float(d.mean()), 4), ci95=np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist())

if __name__ == "__main__":
    calrows = cal
    if len(sys.argv) > 2:
        calrows = [r for p in sys.argv[2:] for r in g["load"](Path(p))]
    Fte = [r for r in te if r["failed"]]
    out = dict(alpha=ALPHA, primary_q=PRIMARY_Q, n_cal_successes=sum(not r["failed"] for r in calrows),
               dev_successes_per_task=np.quantile([len(v) for v in DEV.values()], [0, .5, 1]).tolist(), by_q={})
    gt = res["timer"]["test"]; st = res["safe_timer"]["test"]
    out["reference"] = dict(global_timer=dict(fpr=round(gt["fpr"], 4), mean_lead=round(gt["mean_lead"], 4)),
                            safe_global_timer=dict(fpr=round(st["fpr"], 4), mean_lead=round(st["mean_lead"], 4)))
    for q in QS:
        rk, tm, fu = fit(calrows, q)
        eP, lP = evaluate(te, rk, tm["lam"], math.inf, 0.0)
        eF, lF = evaluate(te, rk, fu["lam"], fu["u"], fu["s"])
        out["by_q"][str(q)] = dict(
            timer=dict(delta=round(tm["lam"], 4), **eP),
            safe_plus_timer=dict(delta=round(fu["lam"], 4), u=fu["u"], s=fu["s"], **eF),
            gain_safe_plus_timer_minus_timer=boot(lF - lP, Fte),
            gain_safe_global_timer_minus_timer=boot(st["lead_vec"] - lP, Fte))
    print(json.dumps(out, indent=1, default=float))
    (here / "quantile_timer.json").write_text(json.dumps(out, indent=1, default=float))
