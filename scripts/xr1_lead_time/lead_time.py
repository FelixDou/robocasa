"""Lead-time analysis for the RSS Stage 1 draft (exploratory, opened cohorts).

Alarm index convention follows review_controls.py: a detector alarms at the first
1-based inference index whose score crosses its threshold; an alarm at any index
<= L (rollout length in inferences) counts. Timer: score t/H_k at inference t.
Lead = (L - alarm_index) inferences; lead fraction = lead / H_k (0 for a miss).
Usage: python3 -I lead_time.py <review_data dir>
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ALPHA = 0.05
def load(p):
    out = []
    for l in open(p):
        if l.strip():
            r = json.loads(l); r["failed"] = bool(r["failed"])
            r["safe"] = np.asarray(r["safe_scores"], float); r["risk"] = np.asarray(r["time_scores"], float)
            r["L"] = len(r["safe"]); r["timer"] = np.arange(1, r["L"] + 1) / r["horizon"]
            out.append(r)
    return out
def sel(records, q):
    g = defaultdict(list)
    for r in records: g[(r["task_name"], r["failed"])].append(r)
    out = []
    for t in sorted({r["task_name"] for r in records}):
        for f in (False, True):
            v = sorted(g[(t, f)], key=lambda r: (r.get("environment_seed") if r.get("environment_seed") is not None else -1,
                       r.get("environment_reset_index") if r.get("environment_reset_index") is not None else -1, str(r["rollout_id"])))
            out += v[:q]
    return out
def first(x, tau):
    if tau is None: return math.inf
    m = np.maximum.accumulate(x); i = np.searchsorted(m, tau, side="left")
    return math.inf if i >= len(m) else i + 1
def alarm_idx(r, th):  # th: dict signal->threshold, OR rule
    return min(first(r[k], v) for k, v in th.items())
def ev(data, th):
    S = [r for r in data if not r["failed"]]; F = [r for r in data if r["failed"]]
    fp = sum(math.isfinite(alarm_idx(r, th)) for r in S)
    leads, fr, leadinf = [], [], []
    for r in F:
        a = alarm_idx(r, th)
        if math.isfinite(a):
            leads.append((r["L"] - a) / r["horizon"]); leadinf.append(r["L"] - a); fr.append(a / r["horizon"])
        else:
            leads.append(0.0); fr.append(1.0)
    leads = np.array(leads); fr = np.array(fr)
    return dict(recall=float(np.mean([math.isfinite(alarm_idx(r, th)) for r in F])), fpr=fp / len(S), fp=fp,
                mean_lead=float(leads.mean()), median_lead_inf=float(np.median(leadinf)) if leadinf else None,
                mean_lead_inf=float(np.mean([l for l in leadinf])) if leadinf else None,
                by50=float(np.mean(fr <= 0.5)), by75=float(np.mean(fr <= 0.75)), by85=float(np.mean(fr <= 0.85)),
                lead_vec=leads)
def strip(d): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "lead_vec"}

def cands(data, key):
    mx = sorted({float(np.max(r[key])) for r in data}, reverse=True)
    return [np.nextafter(mx[0], np.inf)] + mx
def search(data, keys):
    """Joint search over per-rollout maxima; maximise mean lead (miss=0) at FPR<=ALPHA."""
    best = None
    grids = [cands(data, k) for k in keys]
    if len(keys) == 1:
        combos = [(a,) for a in grids[0]]
    else:
        combos = [(a, b) for a in grids[0] for b in grids[1]]
    S = [r for r in data if not r["failed"]]; F = [r for r in data if r["failed"]]
    # precompute first-crossing indices
    pre = {k: {g: None for g in []} for k in keys}
    fc = {k: np.array([[first(r[k], g) for g in grids[j]] for r in data]) for j, k in enumerate(keys)}
    failed = np.array([r["failed"] for r in data]); L = np.array([r["L"] for r in data]); H = np.array([r["horizon"] for r in data])
    for j0, a in enumerate(grids[0]):
        if len(keys) == 1:
            step = fc[keys[0]][:, [j0]]; bvals = [None]
        else:
            step = np.minimum(fc[keys[0]][:, [j0]], fc[keys[1]]); bvals = grids[1]
        al = np.isfinite(step)
        fpr = al[~failed].mean(0)
        lead = np.where(al, (L[:, None] - np.where(al, step, 0)) / H[:, None], 0.0)[failed].mean(0)
        for jb in np.flatnonzero(fpr <= ALPHA + 1e-12):
            key = (lead[jb], a, bvals[jb] if bvals[0] is not None else 0)
            if best is None or key > best[0]:
                best = (key, a, bvals[jb])
    th = {keys[0]: best[1]}
    if len(keys) > 1: th[keys[1]] = best[2]
    return th

def run(D):
    """The exploratory August analysis printed by the CLI; returns its cohorts and fits."""
    cal = sel(load(D / "calibration_predictions.jsonl"), 3); te = sel(load(D / "event_predictions.jsonl"), 10)
    res = {}
    frozen = json.load(open(D / "results.json"))["fused_safe_thresholds"]
    res["fused_safe_risk_frozen"] = {"th": frozen, "test": ev(te, {"safe": frozen["score"], "risk": frozen["time"]})}
    for name, keys in [("timer", ["timer"]), ("risk", ["risk"]), ("safe", ["safe"]), ("safe_timer", ["safe", "timer"]), ("safe_risk_leadobj", ["safe", "risk"]), ("risk_timer", ["risk", "timer"])]:
        th = search(cal, keys)
        res[name] = {"th": th, "cal": ev(cal, th), "test": ev(te, th)}
    for k, v in res.items():
        print(k, {a: round(float(b), 4) for a, b in v["th"].items()}, "\n   cal ", strip(v["cal"]) if "cal" in v else "", "\n   test", strip(v["test"]))
    # Oracle timer matched to each detector's test FPR (chosen on test: favours the timer)
    def timer_at_test_fpr(target):
        best = None
        for c in np.arange(0.5, 1.0001, 0.0025):
            e = ev(te, {"timer": c})
            if e["fpr"] <= target + 1e-12 and (best is None or e["mean_lead"] > best[1]["mean_lead"]):
                best = (c, e)
        return best
    for k in ("fused_safe_risk_frozen", "safe_timer"):
        f = res[k]["test"]["fpr"]; c, e = timer_at_test_fpr(f)
        print(f"oracle timer at test FPR<={f:.3f}: c={c:.4f}", strip(e))
    # paired bootstrap (task then rollout) of mean-lead difference safe_timer - timer, and fused frozen - timer
    rng = np.random.default_rng(0)
    tasks = sorted({r["task_name"] for r in te}); F = [r for r in te if r["failed"]]
    idx_by_task = defaultdict(list)
    for i, r in enumerate(F): idx_by_task[r["task_name"]].append(i)
    for k in ("safe_timer", "fused_safe_risk_frozen"):
        d = res[k]["test"]["lead_vec"] - res["timer"]["test"]["lead_vec"]
        bs = []
        for _ in range(2000):
            ts = rng.choice(tasks, len(tasks)); ii = np.concatenate([rng.choice(idx_by_task[t], len(idx_by_task[t])) for t in ts])
            bs.append(d[ii].mean())
        print(k, "- timer mean lead diff", round(float(d.mean()), 4), "CI", np.round(np.percentile(bs, [2.5, 97.5]), 4),
              "earlier than timer:", int((d > 1e-12).sum()), "later:", int((d < -1e-12).sum()), "of", len(d))
    # seconds: control steps per inference = 16; RoboCasa control 20 Hz (verify)
    st = res["safe_timer"]["test"]
    print("horizon stats (inferences): failures", np.percentile([r["horizon"] for r in F], [0, 50, 100]))
    print("all failures end at H_k:", all(r["L"] == r["horizon"] for r in F))
    return dict(cal=cal, te=te, res=res)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
