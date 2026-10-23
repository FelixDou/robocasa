"""Per-seed standalone detection metrics for the RSS Table I (mean and std over seeds).

For each detector and each training seed, the score is that seed's task-normalized running
maximum. A single threshold is chosen on the calibration cohort (largest mean lead at
calibration FPR <= 0.05, i.e. lead_time.py's single-score search) and the balanced test
cohort is evaluated once:
  TPR, FPR, bACC = (TPR + 1 - FPR) / 2,
  TWA = (mean lead + 1 - FPR) / 2   (FIPER / Hide-and-Seek time-weighted accuracy, miss = 0),
  DT  = 1 - mean lead               (miss counted as 1).
The seed-averaged score (mean of the per-seed trajectories, as in the paper) is evaluated the
same way and must reproduce the Table I row given with --expect, otherwise the script stops.

Cohort selection copies lead_time.py: 3 successes + 3 failures per task for calibration and
10 + 10 for test, ordered by environment seed, reset index and rollout ID.

Sources of per-seed trajectories:
  --safe                      SAFE-MLP, from the base files' per_seed_normalized_running_max.
  --scores-* / --normalizations / --detectors
                              score_cohorts.py outputs: per_seed_raw plus normalizations.json
                              (detector -> seed -> task -> {location, scale}).

Usage:
  python3 -I per_seed_detection.py --base-calibration CAL.jsonl --base-test TEST.jsonl \\
      [--safe] [--scores-calibration S_CAL.jsonl --scores-test S_TEST.jsonl \\
       --normalizations normalizations.json --detectors NAME ...] \\
      [--expect NAME=TPR,FPR ...] --output OUT.json
"""
import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np

ALPHA = 0.05


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sel(records, q):
    """lead_time.py's balanced selection: first q per (task, outcome) in a fixed order."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["task_name"], bool(r["failed"]))].append(r)
    out = []
    for task in sorted({r["task_name"] for r in records}):
        for failed in (False, True):
            rows = sorted(groups[(task, failed)], key=lambda r: (
                r.get("environment_seed") if r.get("environment_seed") is not None else -1,
                r.get("environment_reset_index") if r.get("environment_reset_index") is not None else -1,
                str(r["rollout_id"])))
            out += rows[:q]
    return out


def first(x, tau):
    """First 1-based inference whose running maximum reaches tau (inf if never)."""
    m = np.maximum.accumulate(x)
    i = np.searchsorted(m, tau, side="left")
    return math.inf if i >= len(m) else i + 1


def evaluate(rows, tau):
    S = [r for r in rows if not r["failed"]]
    F = [r for r in rows if r["failed"]]
    fp = sum(math.isfinite(first(r["score"], tau)) for r in S)
    leads, hits = [], 0
    for r in F:
        a = first(r["score"], tau)
        if math.isfinite(a):
            hits += 1
            leads.append((r["L"] - a) / r["horizon"])
        else:
            leads.append(0.0)
    tpr, fpr, lead = hits / len(F), fp / len(S), float(np.mean(leads))
    return dict(tpr=tpr, fpr=fpr, bacc=(tpr + 1 - fpr) / 2, twa=(lead + 1 - fpr) / 2,
                dt=1 - lead, mean_lead=lead, fp=int(fp), n_success=len(S), n_failure=len(F))


def threshold(cal):
    """lead_time.search for one score: best mean lead among thresholds with FPR <= ALPHA."""
    mx = sorted({float(np.max(r["score"])) for r in cal}, reverse=True)
    best = None
    for tau in [np.nextafter(mx[0], np.inf)] + mx:
        e = evaluate(cal, tau)
        if e["fpr"] <= ALPHA + 1e-12:
            key = (e["mean_lead"], tau)
            if best is None or key > best[0]:
                best = (key, tau)
    return best[1]


def run_detector(cal_traj, test_traj, cal_rows, test_rows):
    """cal_traj/test_traj: rollout_id -> {seed: trajectory}. Returns averaged and per-seed metrics."""
    seeds = sorted(next(iter(cal_traj.values())))

    def rows_for(base, traj, seed):
        out = []
        for r in base:
            per_seed = traj[r["rollout_id"]]
            score = (np.mean([per_seed[s] for s in seeds], axis=0) if seed is None
                     else np.asarray(per_seed[seed], float))
            out.append(dict(score=score, L=len(score), horizon=r["horizon"], failed=bool(r["failed"])))
        return out

    result = {"seeds": seeds}
    for seed in [None] + seeds:
        cal, test = rows_for(cal_rows, cal_traj, seed), rows_for(test_rows, test_traj, seed)
        tau = threshold(cal)
        e = evaluate(test, tau)
        e["threshold"] = float(tau)
        result["seed_average" if seed is None else f"seed_{seed}"] = e
    per = [result[f"seed_{s}"] for s in seeds]
    result["mean_std_over_seeds"] = {
        k: dict(mean=float(np.mean([p[k] for p in per])), std=float(np.std([p[k] for p in per], ddof=1)))
        for k in ("tpr", "fpr", "bacc", "twa", "dt")}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-calibration", required=True)
    ap.add_argument("--base-test", required=True)
    ap.add_argument("--safe", action="store_true")
    ap.add_argument("--scores-calibration")
    ap.add_argument("--scores-test")
    ap.add_argument("--normalizations")
    ap.add_argument("--detectors", nargs="*", default=[])
    ap.add_argument("--expect", nargs="*", default=[], help="NAME=TPR,FPR from Table I")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cal_rows = sel(load_jsonl(args.base_calibration), 3)
    test_rows = sel(load_jsonl(args.base_test), 10)
    print(f"calibration {len(cal_rows)} rollouts, test {len(test_rows)} rollouts", flush=True)
    need = {r["rollout_id"] for r in cal_rows + test_rows}
    out = {"alpha": ALPHA, "n_calibration": len(cal_rows), "n_test": len(test_rows), "detectors": {}}

    if args.safe:
        traj = {r["rollout_id"]: {s: np.asarray(v, float) for s, v in r["per_seed_normalized_running_max"].items()}
                for r in load_jsonl(args.base_calibration) + load_jsonl(args.base_test) if r["rollout_id"] in need}
        out["detectors"]["safe_mlp"] = run_detector(traj, traj, cal_rows, test_rows)

    if args.detectors:
        norms = json.load(open(args.normalizations))
        for name in args.detectors:
            traj, worst = {}, 0.0
            for path in (args.scores_calibration, args.scores_test):
                with open(path) as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if r.get("detector") != name or r["rollout_id"] not in need:
                            continue
                        per = {}
                        for s, raw in r["per_seed_raw"].items():
                            st = norms[name][s][r["task_name"]]
                            per[s] = (np.maximum.accumulate(np.asarray(raw, float)) - st["location"]) / st["scale"]
                        avg = np.mean([per[s] for s in sorted(per)], axis=0)
                        worst = max(worst, float(np.max(np.abs(avg - np.asarray(r["safe_scores"], float)))))
                        traj[r["rollout_id"]] = per
            missing = need - set(traj)
            if missing:
                sys.exit(f"STOP: {name} lacks {len(missing)} selected rollouts, e.g. {sorted(missing)[:3]}")
            if worst > 1e-9:
                sys.exit(f"STOP: {name} per-seed mean differs from its stored score by {worst:.2e}")
            out["detectors"][name] = run_detector(traj, traj, cal_rows, test_rows)

    ok = True
    for item in args.expect:
        name, vals = item.split("=")
        tpr, fpr = (float(v) for v in vals.split(","))
        got = out["detectors"][name]["seed_average"]
        match = round(got["tpr"], 3) == tpr and round(got["fpr"], 3) == fpr
        ok &= match
        print(f"reference {name}: TPR {got['tpr']:.3f} FPR {got['fpr']:.3f} vs Table I {tpr:.3f} {fpr:.3f}"
              f" -> {'OK' if match else 'MISMATCH'}", flush=True)
    for name, d in out["detectors"].items():
        ms = d["mean_std_over_seeds"]
        print(name, " ".join(f"{k} {ms[k]['mean']:.3f}+-{ms[k]['std']:.3f}" for k in ("tpr", "fpr", "bacc", "twa", "dt")))
    json.dump(out, open(args.output, "w"), indent=1)
    print("REFERENCE OK" if ok else "STOP: a seed-averaged row does not reproduce Table I", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
