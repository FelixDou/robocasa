# Copy of paper_review_fixes/analysis/review_controls.py with a fixed-timer comparison appended at the end (JSAI reframe, 2026-10-17).
# Timer convention and threshold search match rss2027_stage1/analysis/lead_time.py (c = 0.903).
# Usage: NULL_DRAWS=1 python -I timer_vs_detectors.py DATA_DIR OUTPUT_DIR REPO_DIR
"""Post-hoc review analyses for the time-aware fusion paper.

Reads the frozen August protocol outputs (calibration_predictions.jsonl,
runtime_bundle.json, event_predictions.jsonl) and never refits SAFE, the time
curves or the frozen thresholds. Every new detector is calibrated on the same
primary calibration cohort with the same FPR cap and ranking rule, then
evaluated once on the primary prospective cohort. All of this is post hoc on an
already-opened cohort.

Usage: python -I review_controls.py DATA_DIR OUTPUT_DIR REPO_DIR
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA, OUT, REPO = (Path(arg).resolve() for arg in sys.argv[1:4])
sys.path.insert(0, str(REPO / "robocasa" / "recovery" / "safe"))
import prospective_matched_fpr as pm  # noqa: E402  (pure numpy/sklearn module)

TARGET_FPR = 0.05
CAL_PER_CLASS = int(__import__("os").environ.get("CAL_PER_CLASS", 3))
TEST_PER_CLASS = int(__import__("os").environ.get("TEST_PER_CLASS", 10))
BOOT_SAMPLES = 2000
NULL_DRAWS = int(__import__("os").environ.get("NULL_DRAWS", 50))
ENV_STEPS_PER_INFERENCE = 16
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- loading


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["failed"] = bool(row["failed"])
        row["horizon"] = int(row["horizon"])
        row["safe_scores"] = np.asarray(row["safe_scores"], dtype=np.float64)
        row["time_scores"] = np.asarray(row["time_scores"], dtype=np.float64)
        row["per_seed_raw_safe"] = {
            k: np.asarray(v, dtype=np.float64) for k, v in row["per_seed_raw_safe"].items()
        }
        rows.append(row)
    return rows


def select_primary(records, per_class):
    """Same deterministic selection as run_prospective_matched_fpr."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["task_name"], row["failed"])].append(row)
    selected = []
    for task in sorted({row["task_name"] for row in records}):
        for failed in (False, True):
            values = sorted(
                groups[(task, failed)],
                key=lambda r: (
                    r.get("environment_seed") if r.get("environment_seed") is not None else -1,
                    r.get("environment_reset_index")
                    if r.get("environment_reset_index") is not None
                    else -1,
                    str(r["rollout_id"]),
                ),
            )
            if len(values) < per_class:
                raise ValueError(f"{task} lacks {per_class} rollouts of class {failed}")
            selected.extend(values[:per_class])
    return selected


bundle = json.loads((DATA / "runtime_bundle.json").read_text())
cal_all = load(DATA / "calibration_predictions.jsonl")
test_all = load(DATA / "event_predictions.jsonl")
cal = select_primary(cal_all, CAL_PER_CLASS)
test = select_primary(test_all, TEST_PER_CLASS)
cal_ids = set(bundle["primary_calibration_rollout_ids"]) if "primary_calibration_rollout_ids" in bundle else None
if cal_ids is not None:
    assert {r["rollout_id"] for r in cal} == cal_ids, "primary calibration selection mismatch"
assert not ({r["rollout_id"] for r in cal_all} & {r["rollout_id"] for r in test_all})
SAFE_END = float(bundle["safe_eligibility_end_fraction"])
TIME_START = float(bundle["time_eligibility_start_fraction"])
FROZEN = {d: bundle["threshold_selection"][d]["thresholds"] for d in pm.DETECTORS}
SEEDS = sorted(bundle["task_normalizations"])
NORM = bundle["task_normalizations"]


# ------------------------------------------------- vectorized two-source search


def first_cross(trajectory, thresholds):
    """1-based first index with value >= threshold, inf when never crossed."""
    running = np.maximum.accumulate(trajectory)
    index = np.searchsorted(running, thresholds, side="left").astype(np.float64)
    index[index >= len(running)] = np.inf
    return index + 1.0


def candidates(maxima):
    maxima = np.asarray(maxima, dtype=np.float64)
    return np.concatenate([[np.nextafter(maxima.max(), np.inf)], np.unique(maxima)[::-1]])


def select_two_source(records, score_key, mode="or", target_fpr=TARGET_FPR):
    """Joint (tau_s, tau_r) search with the frozen ranking, OR or AND fusion."""
    failed = np.asarray([r["failed"] for r in records])
    horizon = np.asarray([r["horizon"] for r in records], dtype=np.float64)
    smax = [float(np.max(r[score_key])) for r in records]
    rmax = [float(np.max(r["time_scores"])) for r in records]
    a_cand, b_cand = candidates(smax), candidates(rmax)
    ts = np.stack([first_cross(r[score_key], a_cand) for r in records])  # n x A
    tr = np.stack([first_cross(r["time_scores"], b_cand) for r in records])  # n x B
    best = None
    n_fail, n_succ = failed.sum(), (~failed).sum()
    for ai, a in enumerate(a_cand):
        if mode == "or":
            step = np.minimum(ts[:, ai : ai + 1], tr)
        else:
            step = np.maximum(ts[:, ai : ai + 1], tr)
        alarm = np.isfinite(step)
        fpr = alarm[~failed].sum(0) / n_succ
        tpr = alarm[failed].sum(0) / n_fail
        frac = np.where(alarm, np.minimum(1.0, step / horizon[:, None]), 1.0)
        adj = frac[failed].mean(0)
        ok = np.flatnonzero(fpr <= target_fpr + 1e-12)
        for bi in ok:
            key = (tpr[bi], -adj[bi], float(a), float(b_cand[bi]))
            if best is None or key > best[0]:
                best = (key, float(a), float(b_cand[bi]))
    if best is None:
        raise ValueError("no feasible pair")
    return {"score": best[1], "time": best[2]}


def outcomes(records, score_key, thresholds, mode="or"):
    """Per-rollout alarm step and component crossings at fixed thresholds."""
    res = []
    for r in records:
        s = first_cross(r[score_key], np.asarray([thresholds["score"]]))[0] if score_key else np.inf
        t = first_cross(r["time_scores"], np.asarray([thresholds["time"]]))[0] if thresholds.get("time") is not None else np.inf
        if mode == "or":
            step = min(s, t)
        elif mode == "and":
            step = max(s, t)
        elif mode == "score":
            step = s
        else:
            step = t
        res.append({"step": step, "s_step": s, "t_step": t})
    return res


def metrics(records, outs):
    failed = np.asarray([r["failed"] for r in records])
    horizon = np.asarray([r["horizon"] for r in records], dtype=np.float64)
    step = np.asarray([o["step"] for o in outs])
    alarm = np.isfinite(step)
    frac = np.where(alarm, np.minimum(1.0, step / horizon), 1.0)
    tpr = alarm[failed].mean()
    fpr = alarm[~failed].mean()
    det = frac[failed & alarm]
    return {
        "recall": float(tpr),
        "fpr": float(fpr),
        "tp": int(alarm[failed].sum()),
        "fp": int(alarm[~failed].sum()),
        "fpr_wilson95": pm.wilson_interval(int(alarm[~failed].sum()), int((~failed).sum())),
        "balanced_accuracy": float(0.5 * (tpr + 1 - fpr)),
        "miss_adjusted": float(frac[failed].mean()),
        "mean_detected_fraction": float(det.mean()) if len(det) else None,
        "recall_by": {f"{f:.2f}": float(((frac <= f + 1e-12) & alarm)[failed].mean()) for f in (0.1, 0.25, 0.5, 0.75, 0.9)},
    }


def as_outcome(records, outs):
    horizon = np.asarray([r["horizon"] for r in records], dtype=np.float64)
    step = np.asarray([o["step"] for o in outs])
    alarm = np.isfinite(step)
    return {"alarm": alarm, "fraction": np.where(alarm, np.minimum(1.0, step / horizon), 1.0)}


def bootstrap(records, outs_by_name, pairs, samples=BOOT_SAMPLES, seed=0):
    """Same task-then-rollout resampling sequence as the frozen evaluator."""
    named = {k: as_outcome(records, v) for k, v in outs_by_name.items()}
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for i, r in enumerate(records):
        by_task[r["task_name"]].append(i)
    tasks = sorted(by_task)
    failed = np.asarray([r["failed"] for r in records])
    vals = {p: defaultdict(list) for p in pairs}
    for _ in range(samples):
        chosen = rng.choice(tasks, size=len(tasks), replace=True)
        idx = []
        for task in chosen:
            group = by_task[str(task)]
            sel = rng.choice(len(group), size=len(group), replace=True)
            idx.extend(group[int(j)] for j in sel)
        idx = np.asarray(idx)
        f = failed[idx]
        if f.all() or not f.any():
            continue
        fi, si = idx[f], idx[~f]
        for p in pairs:
            a, b = named[p[0]], named[p[1]]
            vals[p]["tpr"].append(a["alarm"][fi].mean() - b["alarm"][fi].mean())
            vals[p]["fpr"].append(a["alarm"][si].mean() - b["alarm"][si].mean())
            vals[p]["adj"].append(a["fraction"][fi].mean() - b["fraction"][fi].mean())
    out = {}
    for p in pairs:
        out[f"{p[0]} - {p[1]}"] = {
            k: {"mean": float(np.mean(v)), "ci95": np.quantile(v, [0.025, 0.975]).tolist()}
            for k, v in vals[p].items()
        }
    return out


# ------------------------------------------------------------ null scores


def safe_from_raw(record, raw_by_seed):
    """Frozen SAFE pipeline: per-seed running max, task normalization, seed mean."""
    task = record["task_name"]
    values = []
    for seed in SEEDS:
        raw = raw_by_seed[seed]
        stats = NORM[seed][task]
        values.append((np.maximum.accumulate(raw) - stats["location"]) / stats["scale"])
    return np.mean(values, axis=0)


def add_null_scores(records, rng, label):
    """Attach length-matched null scores; none uses the rollout's outcome."""
    pools = defaultdict(lambda: defaultdict(list))
    for r in records:
        for seed in SEEDS:
            pools[r["task_name"]][seed].append(r["per_seed_raw_safe"][seed])
    pools = {t: {s: np.concatenate(v) for s, v in d.items()} for t, d in pools.items()}
    for r in records:
        n = len(r["safe_scores"])
        r["noise"] = np.maximum.accumulate(rng.standard_normal(n))
        resampled = {s: rng.choice(pools[r["task_name"]][s], size=n, replace=True) for s in SEEDS}
        r["safe_resampled"] = safe_from_raw(r, resampled)


# ------------------------------------------------------------ main analyses

results = {"cohorts": {"calibration": len(cal), "test": len(test)}}

# 1. Reproduce frozen detectors on test with the frozen thresholds.
repro = {}
for det in pm.DETECTORS:
    m = pm.event_metrics(test, det, FROZEN[det], safe_end=SAFE_END, time_start=TIME_START)
    repro[det] = {k: m[k] for k in ("true_positive_rate", "false_positive_rate", "missed_failure_adjusted_detection_fraction")}
results["reproduced_frozen"] = repro
fused_thr = select_two_source(cal, "safe_scores", "or")
results["fused_safe_thresholds"] = fused_thr
time_thr = {"score": None, "time": FROZEN["time_only"]["time_only"]}
safe_thr = {"score": FROZEN["safe_only"]["safe_only"], "time": None}

outs = {
    "time_only": outcomes(test, None, time_thr, "time"),
    "safe_only": outcomes(test, "safe_scores", {"score": safe_thr["score"]}, "score"),
    "fused_safe_time": outcomes(test, "safe_scores", fused_thr, "or"),
}
st = FROZEN["staged_safe_time"]
staged = []
for r in test:
    a = pm.first_alert(r, "staged_safe_time", st, SAFE_END, TIME_START)
    staged.append({"step": np.inf if a is None else float(a["step"])})
outs["staged_safe_time"] = staged

# 2. AND fusion.
and_thr = select_two_source(cal, "safe_scores", "and")
outs["and_safe_time"] = outcomes(test, "safe_scores", and_thr, "and")
results["and_safe_thresholds"] = and_thr

# 3. Null controls, NULL_DRAWS independent draws each.
null_rows = defaultdict(list)
first_draw = {}
for draw in range(NULL_DRAWS):
    rng = np.random.default_rng(1000 + draw)
    add_null_scores(cal, rng, "cal")
    add_null_scores(test, rng, "test")
    for key, name in (("noise", "fused_noise_time"), ("safe_resampled", "fused_resampled_safe_time")):
        thr = select_two_source(cal, key, "or")
        o = outcomes(test, key, thr, "or")
        m = metrics(test, o)
        m["thresholds"] = thr
        null_rows[name].append(m)
        if draw == 0:
            first_draw[name] = o
    print(f"null draw {draw}: noise {null_rows['fused_noise_time'][-1]['recall']:.3f}/"
          f"{null_rows['fused_noise_time'][-1]['fpr']:.3f}  resampled "
          f"{null_rows['fused_resampled_safe_time'][-1]['recall']:.3f}/"
          f"{null_rows['fused_resampled_safe_time'][-1]['fpr']:.3f}", flush=True)
results["null_controls"] = {
    name: {
        "draws": len(rows),
        "recall_mean": float(np.mean([m["recall"] for m in rows])),
        "recall_p05_p95": np.quantile([m["recall"] for m in rows], [0.05, 0.95]).tolist(),
        "recall_min_max": [float(min(m["recall"] for m in rows)), float(max(m["recall"] for m in rows))],
        "fpr_mean": float(np.mean([m["fpr"] for m in rows])),
        "fpr_p05_p95": np.quantile([m["fpr"] for m in rows], [0.05, 0.95]).tolist(),
        "miss_adjusted_mean": float(np.mean([m["miss_adjusted"] for m in rows])),
    }
    for name, rows in null_rows.items()
}
outs.update(first_draw)

results["metrics"] = {name: metrics(test, o) for name, o in outs.items()}
results["bootstrap"] = bootstrap(
    test,
    outs,
    [
        ("fused_safe_time", "time_only"),
        ("staged_safe_time", "time_only"),
        ("fused_safe_time", "fused_noise_time"),
        ("fused_safe_time", "fused_resampled_safe_time"),
        ("fused_noise_time", "time_only"),
        ("fused_resampled_safe_time", "time_only"),
        ("and_safe_time", "time_only"),
        ("fused_safe_time", "and_safe_time"),
    ],
)

# 4. Recall versus elapsed horizon fraction.
grid = np.round(np.linspace(0, 1, 101), 2)
curve_rows = []
for name, o in outs.items():
    oc = as_outcome(test, o)
    failed = np.asarray([r["failed"] for r in test])
    for f in grid:
        curve_rows.append({"detector": name, "fraction": float(f),
                           "recall": float(((oc["fraction"] <= f + 1e-12) & oc["alarm"])[failed].mean())})
with (OUT / "recall_vs_horizon.csv").open("w", newline="") as s:
    w = csv.DictWriter(s, fieldnames=["detector", "fraction", "recall"])
    w.writeheader()
    w.writerows(curve_rows)

# 5. Time saved and timing on jointly detected failures.
timing = {}
for name in ("time_only", "staged_safe_time", "fused_safe_time", "fused_noise_time"):
    left_h, left_end = [], []
    for r, o in zip(test, outs[name]):
        if r["failed"] and np.isfinite(o["step"]):
            left_h.append(r["horizon"] - o["step"])
            left_end.append(len(r["safe_scores"]) - o["step"])
    left_h, left_end = np.asarray(left_h), np.asarray(left_end)
    timing[name] = {
        "detected": int(len(left_h)),
        "inferences_before_horizon_median_iqr": np.quantile(left_h, [0.5, 0.25, 0.75]).tolist(),
        "inferences_before_rollout_end_median_iqr": np.quantile(left_end, [0.5, 0.25, 0.75]).tolist(),
        "env_steps_before_rollout_end_median": float(np.median(left_end) * ENV_STEPS_PER_INFERENCE),
        "fraction_alarm_at_last_inference": float(np.mean(left_end == 0)),
    }
both = [(a["step"] / r["horizon"], b["step"] / r["horizon"]) for r, a, b in zip(test, outs["fused_safe_time"], outs["time_only"])
        if r["failed"] and np.isfinite(a["step"]) and np.isfinite(b["step"])]
both = np.asarray(both)
timing["joint_fused_vs_time"] = {
    "failures_detected_by_both": int(len(both)),
    "mean_fraction_fused": float(both[:, 0].mean()),
    "mean_fraction_time": float(both[:, 1].mean()),
    "fused_earlier": int((both[:, 0] < both[:, 1] - 1e-12).sum()),
    "same": int((np.abs(both[:, 0] - both[:, 1]) <= 1e-12).sum()),
    "fused_later": int((both[:, 0] > both[:, 1] + 1e-12).sum()),
}
results["timing"] = timing

# 6. Matched test FPR (oracle sensitivity on test, not a deployable result).
def best_recall_at(records, key, fpr_cap):
    failed = np.asarray([r["failed"] for r in records])
    mx = np.asarray([np.max(r[key]) for r in records])
    best = 0.0
    for t in candidates(mx):
        alarm = mx >= t
        if alarm[~failed].mean() <= fpr_cap + 1e-12:
            best = max(best, alarm[failed].mean())
    return float(best)

results["matched_test_fpr"] = {
    f"time_only_oracle_recall_at_fpr_{cap}": best_recall_at(test, "time_scores", cap)
    for cap in (0.047, 0.05, 0.058, 0.068, 0.10)
}
fail = np.asarray([r["failed"] for r in test])
roc_rows = []
for key in ("time_scores", "safe_scores"):
    mx = np.asarray([np.max(r[key]) for r in test])
    for t in candidates(mx):
        alarm = mx >= t
        roc_rows.append({"score": key, "threshold": float(t), "fpr": float(alarm[~fail].mean()), "recall": float(alarm[fail].mean())})
with (OUT / "test_recall_fpr_single_scores.csv").open("w", newline="") as s:
    w = csv.DictWriter(s, fieldnames=["score", "threshold", "fpr", "recall"])
    w.writeheader()
    w.writerows(roc_rows)

# 7. Per-task overlap of the two components at the fused thresholds.
per_task = defaultdict(lambda: {"time_only": 0, "score_only": 0, "both": 0, "neither": 0})
for r, o in zip(test, outs["fused_safe_time"]):
    if not r["failed"]:
        continue
    s_hit, t_hit = np.isfinite(o["s_step"]), np.isfinite(o["t_step"])
    key = "both" if s_hit and t_hit else "score_only" if s_hit else "time_only" if t_hit else "neither"
    per_task[r["task_name"]][key] += 1
from sklearn.metrics import roc_auc_score  # noqa: E402

task_rows = []
for task in sorted(per_task):
    rows = [r for r in test if r["task_name"] == task]
    lengths = [len(r["safe_scores"]) for r in rows]
    labels = [int(r["failed"]) for r in rows]
    dur_auc = float(roc_auc_score(labels, lengths)) if len(set(labels)) == 2 else None
    task_rows.append({"task": task, "task_type": rows[0]["task_type"], "duration_auc": dur_auc, **per_task[task]})
with (OUT / "per_task_overlap.csv").open("w", newline="") as s:
    w = csv.DictWriter(s, fieldnames=list(task_rows[0]))
    w.writeheader()
    w.writerows(task_rows)
tot = {k: sum(row[k] for row in task_rows) for k in ("time_only", "score_only", "both", "neither")}
dur = np.asarray([row["duration_auc"] for row in task_rows])
share = np.asarray([row["score_only"] / 10 for row in task_rows])
from scipy.stats import spearmanr  # noqa: E402

rho = spearmanr(dur, share)
results["per_task_overlap"] = {"totals": tot, "spearman_duration_auc_vs_score_only_share": [float(rho.statistic), float(rho.pvalue)]}

# 8. Within-task at-risk ROC-AUC at landmarks (length-matched comparison).
def at_risk(records, key, frac):
    aucs = []
    for task in sorted({r["task_name"] for r in records}):
        rows = [r for r in records if r["task_name"] == task]
        step = max(1, math.ceil(frac * rows[0]["horizon"]))
        active = [r for r in rows if len(r[key]) >= step]
        labels = [int(r["failed"]) for r in active]
        if len(set(labels)) < 2:
            continue
        aucs.append(roc_auc_score(labels, [r[key][step - 1] for r in active]))
    return {"tasks": len(aucs), "task_macro_auc": float(np.mean(aucs)) if aucs else None}

results["at_risk_within_task"] = {
    key: {f"{f:.2f}": at_risk(test, key, f) for f in (0.10, 0.25, 0.50, 0.75)}
    for key in ("safe_scores", "noise", "safe_resampled", "time_scores")
}

(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps({k: results[k] for k in ("reproduced_frozen", "fused_safe_thresholds", "null_controls")}, indent=2, default=float))

# ---- timer comparison (JSAI reframe, 2026-10-17) ----
# One convention for every detector (as review_controls.py and rss2027_stage1/analysis/lead_time.py):
# the timer alarms at inference ceil(c*H_k); an alarm at any index <= L counts, including the last.
def L(r): return len(r["safe_scores"])
def timer_step(r, c): return math.ceil(c * r["horizon"] - 1e-9)
def timer_fpr(data, c): return np.mean([timer_step(r, c) <= L(r) for r in data if not r["failed"]])
grid = [round(x, 4) for x in np.arange(0.5, 1.0001, 0.0005)]
# Same threshold search as lead_time.py: candidates are per-rollout maxima L/H_k on calibration;
# keep FPR <= 0.05 and the largest mean lead fraction (miss = 0).
def mean_lead(data, c):
    F = [r for r in data if r["failed"]]
    return np.mean([(L(r) - timer_step(r, c)) / r["horizon"] if timer_step(r, c) <= L(r) else 0.0 for r in F])
cands = sorted({L(r) / r["horizon"] for r in cal})
C = max((c for c in cands if timer_fpr(cal, c) <= TARGET_FPR + 1e-12), key=lambda c: (mean_lead(cal, c), c))
fails = [i for i, r in enumerate(test) if r["failed"]]
succ = [i for i, r in enumerate(test) if not r["failed"]]
print("TIMER c", C, "cal fpr", timer_fpr(cal, C), "test recall", np.mean([timer_step(test[i], C) <= L(test[i]) for i in fails]),
      "test fpr", timer_fpr(test, C), "mean frac", np.mean([timer_step(test[i], C) / test[i]["horizon"] for i in fails]))
for name in outs:
    o = outs[name]
    earlier = [timer_step(test[i], C) - o[i]["step"] for i in fails if np.isfinite(o[i]["step"]) and o[i]["step"] < timer_step(test[i], C)]
    fa = sum(np.isfinite(o[i]["step"]) for i in succ)
    print("ROW", name, len(earlier), np.median(earlier) if earlier else None,
          np.quantile(earlier, [.25, .75]).tolist() if earlier else None, "fp", fa)
