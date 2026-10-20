"""Data for the qualitative figure: one failed OpenCabinet test rollout + one successful one.
Writes rollout_curves.csv and frames.json (inference indices and env steps for the frames).
Usage: python3 -I make_qualitative_data.py <review_data dir> <rss2027_stage1/analysis dir>
"""
import runpy, sys, io, contextlib, math, json, csv
from pathlib import Path
import numpy as np
data, ana = sys.argv[1], Path(sys.argv[2])
sys.argv = ["x", data]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(ana / "lead_time.py"))
te, res, first = g["te"], g["res"], g["first"]
FAIL_ID = "ad40212bc300d3f410d0cf4c"
th = res["safe_timer"]["th"]; c_timer = res["timer"]["th"]["timer"]
f = next(r for r in te if r["rollout_id"] == FAIL_ID)
succ = [r for r in te if r["task_name"] == f["task_name"] and not r["failed"]]
quiet = [r for r in succ if float(np.max(r["safe"])) < th["safe"]]
print("successes of this task:", len(succ), "below tau_s:", len(quiet), file=sys.stderr)
s = max(quiet, key=lambda r: r["L"])  # longest success that SAFE leaves silent
H = f["horizon"]
a_safe = first(f["safe"], th["safe"]); a_timer = math.ceil(c_timer * H)
here = Path(__file__).resolve().parent
with open(here / "rollout_curves.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frac", "fail_score", "succ_score"])
    for t in range(1, H + 1):
        w.writerow([round(t / H, 4), round(float(f["safe"][t - 1]), 4),
                    round(float(s["safe"][t - 1]), 4) if t <= s["L"] else "nan"])
frames = {"t_early": 10, "t_safe_alarm": int(a_safe), "t_timer_alarm": int(a_timer), "t_end": int(f["L"])}
meta = dict(task=f["task_name"], horizon=H, fail_rollout=FAIL_ID, fail_video=f["video_path"],
            success_rollout=s["rollout_id"], success_video=s["video_path"], success_length=s["L"],
            tau_s=th["safe"], timer_c=c_timer, safe_alarm=int(a_safe), timer_alarm=int(a_timer),
            lead_safe_inferences=int(f["L"] - a_safe), lead_timer_inferences=int(f["L"] - a_timer),
            frames={k: dict(inference=v, env_step=int(f["inference_environment_steps"][v - 1])) for k, v in frames.items()},
            total_env_steps=int(f["inference_environment_steps"][-1]))
(here / "frames.json").write_text(json.dumps(meta, indent=1, default=float))
print(json.dumps(meta, indent=1, default=float))
