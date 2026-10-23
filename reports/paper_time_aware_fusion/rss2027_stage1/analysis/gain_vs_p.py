"""Validation-set gains in mean warning lead over the per-task timer P (Table II, validation columns).

Detectors whose scores are in ../review_data: global timer, time risk + global timer,
SAFE + global timer (lead_time.py joint search), frozen SAFE + time risk, and SAFE + P
(quantile_timer.py, q = 0.95). Paired task-then-rollout bootstrap, 2,000 resamples, seed 0,
as in quantile_timer.boot. Hide-and-Seek, ActProbe and the controls are not in review_data;
their validation gains come from docs/safe_robocasa_experiment_report.md, Tables 15.20c-d.

Usage: python3 -I gain_vs_p.py <review_data dir>
"""
import contextlib, io, json, math, runpy, sys
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.argv = ["quantile_timer.py", sys.argv[1]]
with contextlib.redirect_stdout(io.StringIO()):
    q = runpy.run_path(str(here / "quantile_timer.py"))
te, res, ev, boot = q["te"], q["res"], q["ev"], q["boot"]
F = [r for r in te if r["failed"]]
rk, tm, fu = q["fit"](q["cal"], q["PRIMARY_Q"])
eP, lP = q["evaluate"](te, rk, tm["lam"], math.inf, 0.0)
eF, lF = q["evaluate"](te, rk, fu["lam"], fu["u"], fu["s"])
rows = {"per_task_timer": (eP["fpr"], eP["mean_lead"], lP),
        "safe_per_task_timer": (eF["fpr"], eF["mean_lead"], lF)}
for name in ("timer", "risk_timer", "safe_timer", "fused_safe_risk_frozen"):
    t = res[name]["test"]; rows[name] = (t["fpr"], t["mean_lead"], t["lead_vec"])
out = {}
for name, (fpr, lead, lv) in rows.items():
    out[name] = dict(fpr=round(float(fpr), 4), mean_lead=round(float(lead), 4),
                     gain_vs_p=None if name == "per_task_timer" else boot(lv - lP, F))
print(json.dumps(out, indent=1))
(here / "gain_vs_p.json").write_text(json.dumps(out, indent=1))
