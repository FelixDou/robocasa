import sys, runpy, numpy as np
sys.argv = ["lead_time.py", sys.argv[1]]
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(str(__import__("pathlib").Path(__file__).with_name("lead_time.py")))
cal, te, ev, search, res = g["cal"], g["te"], g["ev"], g["search"], g["res"]
# noise null fused with timer
rng = np.random.default_rng(1)
leads, fprs = [], []
for d in range(20):
    for r in cal + te:
        r["noise"] = rng.standard_normal(r["L"])
    th = search(cal, ["noise", "timer"]); e = ev(te, th)
    leads.append(e["mean_lead"]); fprs.append(e["fpr"])
print("noise+timer: mean lead %.4f (min %.4f max %.4f), fpr mean %.4f" % (np.mean(leads), min(leads), max(leads), np.mean(fprs)))
# curves: fraction of failures alarmed by elapsed fraction f
import math
F = [r for r in te if r["failed"]]
def curve(th):
    fr = []
    for r in F:
        a = min(g["first"](r[k], v) for k, v in th.items())
        fr.append(a / r["horizon"] if math.isfinite(a) else 9)
    fr = np.array(fr)
    return " ".join(f"({f:.2f},{np.mean(fr <= f + 1e-12):.3f})" for f in np.arange(0, 1.0001, 0.02))
for k in ("timer", "risk_timer", "safe_timer"):
    print(k, curve(res[k]["th"]))
fz = res["fused_safe_risk_frozen"]["th"]
print("fused_frozen", curve({"safe": fz["score"], "risk": fz["time"]}))
# component attribution for safe_timer on test failures
th = res["safe_timer"]["th"]
s_first = [g["first"](r["safe"], th["safe"]) for r in F]; t_first = [g["first"](r["timer"], th["timer"]) for r in F]
print("safe fires first:", sum(s < t for s, t in zip(s_first, t_first)), "timer only:", sum(not math.isfinite(s) or s >= t for s, t in zip(s_first, t_first)))
