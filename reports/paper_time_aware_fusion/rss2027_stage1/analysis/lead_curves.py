"""Curves for Fig. 2: fraction of exploratory test failures alarmed by each elapsed fraction t/H_k.

Same detectors and thresholds as Table I: the global timer and SAFE + timer from lead_time.py,
and the per-task timer P and SAFE + P at q = 0.95 from quantile_timer.py.
Writes ../figures/lead_curves.csv (one column per detector, x = elapsed fraction).

Usage: python3 -I lead_curves.py <review_data dir>
"""
import contextlib, csv, io, math, runpy, sys
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
D = sys.argv[1]
sys.argv = ["quantile_timer.py", D]
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(str(here / "quantile_timer.py"))
first, te, res = q["first"], q["te"], q["res"]
F = [r for r in te if r["failed"]]

def frac_or(th):
    """Alarm fraction a/H_k per failure for an OR rule over thresholds th (9 = missed)."""
    out = []
    for r in F:
        a = min(first(r[k], v) for k, v in th.items())
        out.append(a / r["horizon"] if math.isfinite(a) and a <= r["L"] else 9.0)
    return np.array(out)

rk, tm, fu = q["fit"](q["cal"], q["PRIMARY_Q"])
def frac_p(lam, u, s):
    return np.array([(a / r["horizon"]) if (a := q["alarm"](r, rk, lam, u, s)) <= r["L"] else 9.0 for r in F])

cols = {
    "timer": frac_or(res["timer"]["th"]),
    "safe_timer": frac_or(res["safe_timer"]["th"]),
    "per_task_timer": frac_p(tm["lam"], math.inf, 0.0),
    "safe_per_task_timer": frac_p(fu["lam"], fu["u"], fu["s"]),
}
xs = np.round(np.arange(0, 1.0001, 0.01), 2)
with open(here.parent / "figures" / "lead_curves.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frac", *cols])
    for x in xs:
        w.writerow([f"{x:.2f}", *(f"{np.mean(v <= x + 1e-12):.4f}" for v in cols.values())])
for k, v in cols.items():
    print(f"{k:20s} by .50 {np.mean(v <= .5):.3f}  by .75 {np.mean(v <= .75):.3f}  recall {np.mean(v <= 1):.3f}")
