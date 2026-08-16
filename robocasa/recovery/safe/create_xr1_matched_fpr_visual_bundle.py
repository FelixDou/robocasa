"""Create figures, videos, and evidence for XR1 matched-FPR calibration."""

import csv, hashlib, json, math, shutil, sys, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

analysis, matched, out, zip_path = map(lambda x: Path(x).resolve(), sys.argv[1:])
fig_dir, video_dir, data_dir = out / "figures", out / "videos", out / "data"
if out.exists() and any(out.iterdir()):
    raise SystemExit(f"STOP: output is not empty: {out}")
for path in (fig_dir, video_dir, data_dir):
    path.mkdir(parents=True, exist_ok=True)

def read_json(path): return json.loads(path.read_text())
def read_jsonl(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def save(fig, name):
    fig.savefig(fig_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20): h.update(block)
    return h.hexdigest()

sources = {
    "runtime_bundle.json": analysis / "runtime_bundle.json",
    "threshold_curves.json": analysis / "threshold_curves.json",
    "calibration_predictions.jsonl": analysis / "calibration_predictions.jsonl",
    "status.json": analysis / "status.json",
}
missing = [str(p) for p in sources.values() if not p.is_file()]
if missing: raise SystemExit("STOP: missing:\n" + "\n".join(missing))
bundle = read_json(sources["runtime_bundle.json"])
curves = read_json(sources["threshold_curves.json"])
all_rows = read_jsonl(sources["calibration_predictions.jsonl"])
primary_ids = set(bundle["primary_calibration_rollout_ids"])
rows = [r for r in all_rows if r["rollout_id"] in primary_ids]
if bundle.get("phase") != "calibration_frozen": raise SystemExit("STOP: bundle is not frozen")
if (len(all_rows), len(rows), sum(r["failed"] for r in rows)) != (454, 228, 114):
    raise SystemExit(f"STOP: unexpected counts retained={len(all_rows)} primary={len(rows)} failures={sum(r['failed'] for r in rows)}")

names = ["safe_only", "staged_safe_time", "time_only"]
labels = {"safe_only":"SAFE only", "staged_safe_time":"Staged SAFE+time", "time_only":"Time only"}
colors = {"safe_only":"#E69F00", "staged_safe_time":"#009E73", "time_only":"#0072B2"}
sel = bundle["threshold_selection"]
metrics = {n: sel[n]["validation_metrics"] for n in names}
thr = {n: sel[n]["thresholds"] for n in names}
safe_end = bundle["safe_eligibility_end_fraction"]
time_start = bundle["time_eligibility_start_fraction"]

def first_alert(row, name):
    safe, time = np.asarray(row["safe_scores"]), np.asarray(row["time_scores"])
    step = np.arange(1, len(safe) + 1)
    if name == "safe_only":
        hit = np.flatnonzero(safe >= thr[name][name]); source = name
    elif name == "time_only":
        hit = np.flatnonzero(time >= thr[name][name]); source = name
    else:
        early = np.flatnonzero((step <= math.ceil(safe_end * row["horizon"])) & (safe >= thr[name]["early_safe"]))
        late = np.flatnonzero((step >= math.ceil(time_start * row["horizon"])) & (time >= thr[name]["late_time"]))
        candidates = []
        if len(early): candidates.append((early[0], "early_safe"))
        if len(late): candidates.append((late[0], "late_time"))
        if not candidates: return None
        index, source = min(candidates)
        return {"step": int(index + 1), "source": source}
    return None if not len(hit) else {"step": int(hit[0] + 1), "source": source}
def fraction(row, name, miss=None):
    alert = first_alert(row, name)
    return miss if alert is None else min(1.0, alert["step"] / row["horizon"])

# Compact event table.
event_rows = []
for n in names:
    m = metrics[n]
    event_rows.append({"detector":n, "thresholds":json.dumps(m["thresholds"], sort_keys=True),
        "tp":m["confusion"]["tp"], "fn":m["confusion"]["fn"], "fp":m["confusion"]["fp"], "tn":m["confusion"]["tn"],
        "tpr":m["true_positive_rate"], "fpr":m["false_positive_rate"], "balanced_accuracy":m["balanced_accuracy"],
        "roc_auc":m["roc_auc"], "average_precision":m["average_precision"],
        "miss_adjusted_detection_fraction":m["missed_failure_adjusted_detection_fraction"]})
write_csv(data_dir / "calibration_event_metrics.csv", event_rows)

# 1. Matched operating points.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2)); x = np.arange(3)
for ax, key, title in zip(axes, ["true_positive_rate","false_positive_rate","missed_failure_adjusted_detection_fraction"], ["Failure recall (TPR)","False-positive rate","Miss-adjusted alarm fraction"]):
    vals = [metrics[n][key] for n in names]; bars = ax.bar(x, vals, color=[colors[n] for n in names])
    ax.set_xticks(x, [labels[n] for n in names], rotation=15, ha="right"); ax.set_ylim(0, 1.05); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    for b,v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+.02, f"{v:.3f}", ha="center", fontsize=9)
axes[1].axhline(.05, color="#CC3311", ls="--", label="5% cap"); axes[1].legend(frameon=False)
axes[2].text(.02,.97,"lower = earlier",transform=axes[2].transAxes,va="top")
fig.suptitle("Xiaomi matched-FPR calibration (114 successes / 114 failures)", weight="bold"); fig.tight_layout(); save(fig,"01_matched_fpr_operating_points")

# 2. Threshold curves.
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
for ax, xmax in zip(axes,[1,.12]):
    for n in names:
        best = {}
        for p in curves[n]: best[p["false_positive_rate"]] = max(p["true_positive_rate"], best.get(p["false_positive_rate"],-1))
        pts = sorted(best.items()); ax.step([p[0] for p in pts],[p[1] for p in pts],where="post",lw=1.8,color=colors[n],label=labels[n])
        ax.scatter(metrics[n]["false_positive_rate"],metrics[n]["true_positive_rate"],marker="*",s=150,color=colors[n],edgecolor="black",zorder=5)
    ax.axvline(.05,color="#CC3311",ls="--"); ax.set(xlim=(-.005,xmax),ylim=(0,1.03),xlabel="Empirical event FPR",ylabel="Failure recall (TPR)"); ax.grid(alpha=.25)
axes[0].set_title("All empirical FPRs"); axes[1].set_title("Low-FPR region"); axes[0].legend(frameon=False)
fig.suptitle("Threshold trade-off; stars are frozen operating points",weight="bold"); fig.tight_layout(); save(fig,"02_threshold_tradeoff")

# 3. Recall by landmark.
landmarks=["0.1","0.25","0.5"]; fig,ax=plt.subplots(figsize=(8.5,4.8)); width=.24
for i,n in enumerate(names):
    vals=[metrics[n]["failure_recall_by_landmark"][k] for k in landmarks]; pos=np.arange(3)+(i-1)*width
    bars=ax.bar(pos,vals,width,color=colors[n],label=labels[n])
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.015,f"{v:.2f}",ha="center",fontsize=8)
ax.set_xticks(range(3),["10%","25%","50%"]); ax.set(ylim=(0,1.05),ylabel="Fraction of all failures detected",xlabel="Task-horizon landmark"); ax.set_title("Causal failure recall by landmark",weight="bold"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.25); fig.tight_layout(); save(fig,"03_recall_by_landmark")

# 4. Failure alarm-time ECDF and per-rollout CSV.
failures=[r for r in rows if r["failed"]]; fig,ax=plt.subplots(figsize=(8.5,5)); rollout_rows=[]
for n in names:
    vals=np.sort([fraction(r,n,1.0) for r in failures]); ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",lw=2,color=colors[n],label=f"{labels[n]} (mean={vals.mean():.3f})")
for r in rows:
    p={"rollout_id":r["rollout_id"],"task_name":r["task_name"],"task_type":r.get("task_type"),"failed":r["failed"],"horizon":r["horizon"],"video_path":r.get("video_path","")}
    for n in names: p[f"{n}_alert_fraction"]="" if first_alert(r,n) is None else fraction(r,n)
    rollout_rows.append(p)
write_csv(data_dir/"per_rollout_alarm_timing.csv",rollout_rows)
ax.axvline(.25,color="#777",ls=":",label="SAFE window end"); ax.axvline(.5,color="#222",ls="--",label="Time fallback start")
ax.set(xlim=(0,1.01),ylim=(0,1.02),xlabel="First-alarm fraction; misses assigned 1.0",ylabel="Cumulative fraction of failures"); ax.set_title("Failure alarm-time distribution",weight="bold"); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=9); fig.tight_layout(); save(fig,"04_failure_alarm_time_ecdf")

# 5. Per-task retained support.
support=[]
for task,p in sorted(bundle["calibration_counts"]["per_task"].items()): support.append({"task_name":task,"retained_successes":p["successes"]["available"],"retained_failures":p["failures"]["available"],"primary_successes":p["successes"]["selected"],"primary_failures":p["failures"]["selected"]})
write_csv(data_dir/"per_task_collection_support.csv",support); support.sort(key=lambda r:(r["retained_successes"]+r["retained_failures"],r["task_name"]))
fig,ax=plt.subplots(figsize=(10.5,12.5)); y=np.arange(len(support)); s=np.array([r["retained_successes"] for r in support]); f=np.array([r["retained_failures"] for r in support])
ax.barh(y,s,color="#56B4E9",label="Successes"); ax.barh(y,f,left=s,color="#D55E00",label="Failures"); ax.set_yticks(y,[r["task_name"] for r in support],fontsize=8); ax.set_xlabel("All retained calibration rollouts"); ax.set_title("Calibration support and quota overshoot",weight="bold"); ax.legend(frameon=False); ax.grid(axis="x",alpha=.2); fig.tight_layout(); save(fig,"05_per_task_collection_support")

# Representative video selection: two strongest staged gains, one loss, one tie, one staged FP, one all-detector TN.
available=[r for r in rows if r.get("video_path") and Path(r["video_path"]).is_file()]
ff=[r for r in available if r["failed"]]; ss=[r for r in available if not r["failed"]]
if len(ff)<4 or len(ss)<2: raise SystemExit(f"STOP: too few source videos: failures={len(ff)} successes={len(ss)}")
delta=lambda r:fraction(r,"staged_safe_time",1.0)-fraction(r,"time_only",1.0)
chosen=[]; used=set()
def add(reason,candidates,count=1):
    for r in candidates:
        if r["rollout_id"] in used: continue
        chosen.append((reason,r)); used.add(r["rollout_id"])
        if sum(x[0]==reason for x in chosen)>=count: break
add("staged_most_earlier",sorted(ff,key=lambda r:(delta(r),r["rollout_id"])),2)
add("time_most_earlier",sorted(ff,key=lambda r:(-delta(r),r["rollout_id"])))
add("near_tie",sorted(ff,key=lambda r:(abs(delta(r)),r["rollout_id"])))
add("staged_false_positive",[r for r in ss if first_alert(r,"staged_safe_time")])
add("all_detectors_true_negative",[r for r in ss if all(first_alert(r,n) is None for n in names)])
if len(chosen)<6: add("fallback",sorted(available,key=lambda r:r["rollout_id"]),6-len(chosen))
chosen=chosen[:6]; index=[]
for i,(reason,r) in enumerate(chosen,1):
    src=Path(r["video_path"]); dst=video_dir/f"{i:02d}_{r['task_name']}_{reason}_{r['rollout_id']}.mp4"; shutil.copy2(src,dst)
    p={"selection_reason":reason,"rollout_id":r["rollout_id"],"task_name":r["task_name"],"failed":r["failed"],"video_file":f"videos/{dst.name}","source_video":str(src)}
    for n in names:p[f"{n}_alert_fraction"]="" if first_alert(r,n) is None else fraction(r,n)
    index.append(p)
write_csv(data_dir/"representative_video_index.csv",index)

# 6. Score trajectories for the exact six packaged videos.
fig,axes=plt.subplots(6,1,figsize=(11,14),sharex=True)
for ax,(reason,r) in zip(axes,chosen):
    h=r["horizon"]; x=np.arange(1,len(r["safe_scores"])+1)/h; safe=np.asarray(r["safe_scores"]); time=np.asarray(r["time_scores"])
    ax.plot(x,safe-thr["safe_only"]["safe_only"],color=colors["safe_only"],label="SAFE-only margin")
    ax.plot(x,time-thr["time_only"]["time_only"],color=colors["time_only"],label="time-only margin")
    staged=np.full(len(x),np.nan); early=x<=safe_end; late=x>=time_start; staged[early]=safe[early]-thr["staged_safe_time"]["early_safe"]; staged[late]=time[late]-thr["staged_safe_time"]["late_time"]
    ax.plot(x,staged,color=colors["staged_safe_time"],lw=2,label="staged margin"); ax.axhline(0,color="black",lw=.8); ax.axvspan(safe_end,time_start,color="#777",alpha=.08); ax.grid(alpha=.2); ax.set_ylabel("score-thr"); ax.set_title(f"{r['task_name']} | {'FAILURE' if r['failed'] else 'SUCCESS'} | {reason}",loc="left",fontsize=9)
axes[0].legend(frameon=False,ncol=3,fontsize=8); axes[-1].set_xlabel("Training-derived task-horizon fraction"); fig.suptitle("Representative causal trajectories linked to packaged videos",weight="bold"); fig.tight_layout(); save(fig,"06_representative_score_trajectories")

# Copy compact evidence and build checksummed ZIP.
for name,path in sources.items(): shutil.copy2(path,data_dir/name)
for path in [matched/"frozen_tasks.json",matched/"tasks_shard0.txt",matched/"tasks_shard1.txt",matched/"calibration_collection"/"merged_all_retained"/"manifest.jsonl",matched/"calibration_collection"/"official_safe_all_retained"/"conversion_report.json"]:
    if path.is_file(): shutil.copy2(path,data_dir/path.name)
for seed in (0,1,2):
    path=matched/"calibration_collection"/"frozen_mlp_scores"/f"seed{seed}"/"provenance.json"
    if path.is_file(): shutil.copy2(path,data_dir/f"seed{seed}_score_provenance.json")
(data_dir/"visualization_manifest.json").write_text(json.dumps({"scope":"completed matched-FPR calibration only","prospective_test_included":False,"source_analysis":str(analysis),"source_collection":str(matched/"calibration_collection"),"retained_rollouts":len(all_rows),"primary_rollouts":len(rows),"primary_successes":114,"primary_failures":114,"thresholds":thr},indent=2,sort_keys=True)+"\n")
(out/"README.md").write_text(f"""# Xiaomi matched-FPR calibration visuals\n\nCalibration only; no prospective-test claim. Primary cohort: 228 (114/114); retained: 454. All selected detectors have FPR {metrics['time_only']['false_positive_rate']:.6f}. Figures are PNG/PDF. Videos are unmodified representative policy rollouts; `data/representative_video_index.csv` links them to alarm times and `06_representative_score_trajectories` shows their causal score traces. Misses are assigned fraction 1.0.\n""")
files=[p for p in out.rglob("*") if p.is_file()]
(out/"SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.relative_to(out)}" for p in sorted(files))+"\n")
zip_path.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(x for x in out.rglob("*") if x.is_file()): z.write(p,Path(out.name)/p.relative_to(out))
with zipfile.ZipFile(zip_path) as z:
    if z.testzip() is not None: raise SystemExit("STOP: ZIP CRC validation failed")
print("VISUAL BUNDLE: VALID"); print("figures:",len(list(fig_dir.glob("*.png"))),"PNG +",len(list(fig_dir.glob("*.pdf"))),"PDF"); print("videos:",len(list(video_dir.glob("*.mp4")))); print("zip:",zip_path); print("zip_sha256:",sha(zip_path)); print("zip_size_bytes:",zip_path.stat().st_size)
