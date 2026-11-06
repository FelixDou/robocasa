"""E2's analysis: closed-loop reset and retry on new RLDX-1 scenes (plan: docs/rldx_e2_plan.md).

Every rule and number was fixed by plan_e2.py before any E2 rollout. Each usable scene (check_e2.py) has a
first attempt A and a retry R from A's initial state, both run to the budget B = ceil(1.25 H) control steps.
For a reset rule c with alarm inference a on A (E1's alarm code, confirm_lead.alarm_vec, with c's frozen
margin and path; a time-only rule alarms at ceil(cutoff x H_k)):
  - no alarm during A (a > A's inferences): the scene succeeds if A did;
  - otherwise c resets at A's a-th inference, at control step t_a, and charges its latency d_c: the scene
    succeeds if R succeeds within B - t_a - d_c steps. A success of A at or after the alarm inference is
    pre-empted (the reset comes first), as in E1's false-alarm count.
  - "none" never resets: the scene succeeds if A succeeds within B.
The retry is the same for every rule (shared), and is not reset again.

Reported: success rate per rule, with 95% CIs; paired differences (primary: SAFE-LSTM + P minus P), with
95% CIs from scenes resampled within task (2,000 replicates, the plan's seed); resets, unnecessary resets
(of first attempts that succeeded within B) and those that turned a success into a failure; the same with a
full inference of latency; by task type; and the retry's success on scenes whose first attempt succeeded or
failed (a check on the restore). Hide-and-Seek + P is descriptive, and computed only if its scores are given.

Usage:
  python3 -I analyze_e2.py --plan <E2 root>/plan.json --pairs <E2 root>/pairs.json --scores-dir <E2 root>/scores \\
      --output <E2 root>/e2_result.json
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "xr1_lead_time"))
import confirm_lead as cl  # noqa: E402

SCORE_FILES = {"safe": "e2_safe_lstm_predictions.jsonl", "hide_and_seek": "e2_hide_and_seek_predictions.jsonl"}


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


# ---- inputs ------------------------------------------------------------------------------------

def load_scenes(plan, pairs, manifests, scores):
    """One row per usable scene: outcomes and steps of both attempts, A's inference steps and scores."""
    problems, scenes = [], []
    for p in pairs["pairs"]:
        t = p["task_name"]
        A, R = manifests.get(p["first_id"]), manifests.get(p["retry_id"])
        if A is None or R is None:
            problems.append(f"{t} scene {p['scene']}: a rollout is missing from the manifests")
            continue
        task = plan["tasks"][t]
        for r in (A, R):
            if int(r["rollout_horizon"]) != int(task["budget_steps"]):
                problems.append(f"{r['rollout_id']}: horizon {r['rollout_horizon']} != budget {task['budget_steps']}")
        steps = [int(x) for x in A["inference_env_steps"]]
        row = dict(task_name=t, type=task["type"], scene=p["scene"], first_id=A["rollout_id"],
                   retry_id=R["rollout_id"], s0=not A["failed"], n0=int(A["num_env_steps"]),
                   s1=not R["failed"], n1=int(R["num_env_steps"]), inf_steps=steps, B=int(task["budget_steps"]),
                   H=int(task["H_k"]), rho=float(task["rho_k"]))
        for k, by_id in scores.items():
            s = by_id.get(A["rollout_id"])
            if s is None:
                problems.append(f"{A['rollout_id']}: no {k} score")
                continue
            v = np.asarray(s["safe_scores"], float)
            if len(v) != len(steps) or not np.all(np.isfinite(v)):
                problems.append(f"{A['rollout_id']}: {len(v)} {k} scores for {len(steps)} inferences")
            row[k] = v
        scenes.append(row)
    return scenes, problems


# ---- rules -------------------------------------------------------------------------------------

def alarms(scenes, cond):
    """1-based alarm inference on each first attempt (inf: none)."""
    if cond["kind"] == "none":
        return np.full(len(scenes), np.inf)
    rows = [dict(task_name=s["task_name"], L=len(s["inf_steps"]), horizon=s["H"], failed=not s["s0"],
                 score=s.get(cond.get("detector"), np.zeros(len(s["inf_steps"])))) for s in scenes]
    c = cl.Cohort(rows, "score")
    if cond["kind"] == "time":
        rho = np.array([cond["cutoffs"][s["task_name"]] for s in scenes])
        a = cl.timer_idx(c, rho, 0.0)
    else:
        rho = np.array([s["rho"] for s in scenes])
        a = cl.alarm_vec(c, rho, cond["lam"], cond["u"], cond["s"])
    return np.where(a <= c.L, a, np.inf)


def outcomes(scenes, cond, latency=None):
    """Per scene: (won, reset, unnecessary reset, harmful reset)."""
    a = alarms(scenes, cond)
    d = cond.get("latency_steps", 0) if latency is None else latency
    won, reset = np.zeros(len(scenes), bool), np.isfinite(a)
    for i, s in enumerate(scenes):
        if not reset[i]:
            won[i] = s["s0"] and s["n0"] <= s["B"]
        else:
            t_a = s["inf_steps"][int(a[i]) - 1]
            won[i] = s["s1"] and s["n1"] <= s["B"] - t_a - d
    s0 = np.array([s["s0"] and s["n0"] <= s["B"] for s in scenes])
    return won, reset, reset & s0, reset & s0 & ~won


# ---- statistics ----------------------------------------------------------------------------------

def strata(scenes):
    idx = defaultdict(list)
    for i, s in enumerate(scenes):
        idx[s["task_name"]].append(i)
    return [np.array(v) for _, v in sorted(idx.items())]


def bootstrap(W, groups, B, seed, level):
    """Percentile CIs of each column's mean and of every column difference, scenes resampled within task."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    means = np.empty((B, W.shape[1]))
    for b in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        means[b] = W[idx].mean(0)
    q = 100 * (1 - level) / 2
    return means, (lambda v: [float(np.percentile(v, q)), float(np.percentile(v, 100 - q))]), n


def analyze(plan, scenes, has_hs=True):
    conds = {k: v for k, v in plan["conditions"].items() if has_hs or k != "hide_and_seek_plus_P"}
    names = list(conds)
    res = {k: outcomes(scenes, v) for k, v in conds.items()}
    W = np.column_stack([res[k][0] for k in names]).astype(float)
    groups = strata(scenes)
    means, ci, n = bootstrap(W, groups, plan["bootstrap"], plan["analysis_seed"], plan["level"])
    col = {k: i for i, k in enumerate(names)}
    rates = {k: dict(success=round(float(W[:, col[k]].mean()), 4), ci=[round(x, 4) for x in ci(means[:, col[k]])],
                     resets=int(res[k][1].sum()), unnecessary_resets=int(res[k][2].sum()),
                     resets_that_lost_a_success=int(res[k][3].sum())) for k in names}
    contrasts = {}
    for x, y in plan["contrasts"]:
        if x not in col or y not in col:
            continue
        diff = W[:, col[x]] - W[:, col[y]]
        contrasts[f"{x}_minus_{y}"] = dict(
            delta=round(float(diff.mean()), 4), ci=[round(v, 4) for v in ci(means[:, col[x]] - means[:, col[y]])],
            x_wins_only=int((diff > 0).sum()), y_wins_only=int((diff < 0).sum()),
            primary=[x, y] == list(plan["primary"]), descriptive=x in plan["descriptive"] or y in plan["descriptive"])
    sens = {}
    for k, v in conds.items():
        if v.get("latency_steps"):
            sens[k] = round(float(outcomes(scenes, v, plan["latency_sensitivity_steps"])[0].mean()), 4)
    if "P" in col:
        sens = {k: dict(success=v, minus_P=round(v - rates["P"]["success"], 4)) for k, v in sens.items()}
    by_type = {}
    for ty in sorted({s["type"] for s in scenes}):
        m = np.array([s["type"] == ty for s in scenes])
        by_type[ty] = dict(scenes=int(m.sum()), **{k: round(float(W[m, col[k]].mean()), 4) for k in names})
    s0 = np.array([s["s0"] and s["n0"] <= s["B"] for s in scenes])
    s1 = np.array([s["s1"] and s["n1"] <= s["B"] for s in scenes])
    restore = dict(first_attempt_success=round(float(s0.mean()), 4),
                   retry_success_after_success=round(float(s1[s0].mean()), 4) if s0.any() else None,
                   retry_success_after_failure=round(float(s1[~s0].mean()), 4) if (~s0).any() else None)
    return dict(scenes=n, per_task={t: len(g) for t, g in zip(sorted({s["task_name"] for s in scenes}), groups)},
                rates=rates, contrasts=contrasts, latency_sensitivity=sens, by_type=by_type, restore_check=restore)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    if Path(a.output).exists():
        raise SystemExit(f"error: {a.output} exists")
    plan = json.loads(Path(a.plan).read_text())
    pairs = json.loads(Path(a.pairs).read_text())
    dirs = sorted({x["dir"] for x in pairs["pairs"]})
    manifests = {r["rollout_id"]: r for d in dirs for r in jsonl(Path(d) / "manifest.jsonl")}
    scores = {}
    for k, f in SCORE_FILES.items():
        path = Path(a.scores_dir) / f
        if path.is_file():
            scores[k] = {str(r["rollout_id"]): r for r in jsonl(path)}
    if "safe" not in scores:
        raise SystemExit("STOP: no SAFE-LSTM scores")
    scenes, problems = load_scenes(plan, pairs, manifests, scores)
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems[:20]))
    out = analyze(plan, scenes, has_hs="hide_and_seek" in scores)
    out = dict(plan=plan["plan"], status="E2 (rules fixed before collection)", excluded_pairs=len(pairs["excluded"]),
               conditions=plan["conditions"], **out)
    Path(a.output).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"scenes {out['scenes']} (excluded pairs {out['excluded_pairs']}); restore check {out['restore_check']}")
    for k, v in out["rates"].items():
        print(f"{k:22s} success {v['success']:.4f} {v['ci']} | resets {v['resets']}, unnecessary "
              f"{v['unnecessary_resets']}, lost a success {v['resets_that_lost_a_success']}")
    for k, v in out["contrasts"].items():
        tag = " (primary)" if v["primary"] else (" (descriptive)" if v["descriptive"] else "")
        print(f"{k:38s} {v['delta']:+.4f} 95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]{tag}")
    print("latency 8 steps:", out["latency_sensitivity"], "| by type:", out["by_type"])
    print("ANALYSIS DONE")


if __name__ == "__main__":
    main()
