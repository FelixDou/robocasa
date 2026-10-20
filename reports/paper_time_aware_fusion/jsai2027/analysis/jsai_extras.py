"""JSAI 2027 extras on top of rss2027_stage1/analysis/lead_time.py (same conventions):
per task type (atomic/composite) recall, FPR and mean lead for the timer and SAFE + timer,
and the time-risk misses by whether the task's risk curve can reach its threshold.
Usage: python3 -I jsai_extras.py <review_data dir> <path to lead_time.py>"""
import sys, io, contextlib, runpy, math
import numpy as np
D, LT = sys.argv[1], sys.argv[2]
sys.argv = ["lead_time.py", D]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(LT)
te, ev, res = g["te"], g["ev"], g["res"]
fz = res["fused_safe_risk_frozen"]["th"]
dets = {"timer": res["timer"]["th"], "risk_timer": res["risk_timer"]["th"], "safe_timer": res["safe_timer"]["th"],
        "safe_risk": {"safe": fz["score"], "risk": fz["time"]}}
for tt in ("atomic", "composite"):
    sub = [r for r in te if r["task_type"] == tt]
    print(tt, "tasks", len({r["task_name"] for r in sub}))
    for k, th in dets.items():
        e = ev(sub, th)
        print("  ", k, "recall %.3f fpr %.3f mean_lead %.3f" % (e["recall"], e["fpr"], e["mean_lead"]))
print("thresholds", {k: {a: float(b) for a, b in v.items()} for k, v in dets.items()})
# per-task mean-lead gain of SAFE + timer over the timer, and SAFE + timer false alarms per task
from collections import defaultdict
F = [r for r in te if r["failed"]]
def lead(r, th):
    a = min(g["first"](r[k], v) for k, v in th.items())
    return (r["L"] - a) / r["horizon"] if math.isfinite(a) else 0.0
gain = defaultdict(list)
for r in F:
    gain[r["task_name"]].append(lead(r, dets["safe_timer"]) - lead(r, dets["timer"]))
m = {t: float(np.mean(v)) for t, v in gain.items()}
print("tasks with positive mean gain", sum(v > 1e-9 for v in m.values()), "negative", sum(v < -1e-9 for v in m.values()), "of", len(m))
print("gain quartiles", np.round(np.percentile(list(m.values()), [0, 25, 50, 75, 100]), 3))
top = sorted(m.items(), key=lambda kv: -kv[1])
print("top5", [(t, round(v, 3)) for t, v in top[:5]]); print("bottom5", [(t, round(v, 3)) for t, v in top[-5:]])
S = [r for r in te if not r["failed"]]
fa = defaultdict(int)
for r in S:
    if math.isfinite(min(g["first"](r[k], v) for k, v in dets["safe_timer"].items())): fa[r["task_name"]] += 1
print("SAFE+timer false alarms by task", dict(fa))
share = sum(max(v, 0) for v in m.values())
print("share of total positive gain from top 5 tasks", round(sum(max(v,0) for _, v in top[:5]) / share, 3))
