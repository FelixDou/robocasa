"""Exploratory follow-up to E1 (plan: docs/xr1_lead_time_dose_response_plan.md).

Does the lead gain of a policy score fused with the per-task timer P track the score's
length-matched AUC? Compares SAFE and the frozen success-only baselines, overall, by task
type, and with stratum-only calibration (the design a fresh atomic test would register).
Both cohorts are already open for lead time; nothing here is confirmatory.

Usage:
  python3 -I dose_response.py --calibration CAL_WITH_BASELINES.jsonl --test TEST_WITH_BASELINES.jsonl \
      --runtime-bundle runtime_bundle.json [--confirm-predictions confirm/event_predictions.jsonl] \
      --output OUT.json
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl
import quantile_timer as qt

LANDMARKS = (0.10, 0.25, 0.50, 0.75)
PERMUTATIONS, BOOT, SEED = 10000, 2000, 0
SIZES = (10, 20, 30, 40)


def attach(rows, names):
    for r in rows:
        for n in names:
            v = np.asarray(r["baseline_scores"][n], float)
            if len(v) != r["L"]:
                raise ValueError(f"{n} trajectory length differs for {r['rollout_id']}")
            r[n] = v


def auc(labels, scores):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return None
    order = scores.argsort(kind="mergesort"); ranks = np.empty(len(scores))
    s = scores[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def lm_auc(rows, key):
    """Per-task mean over landmarks of the at-risk AUC of the running-max score."""
    out = {}
    for task in sorted({r["task_name"] for r in rows}):
        tr = [r for r in rows if r["task_name"] == task]
        vals = []
        for f in LANDMARKS:
            k = max(1, math.ceil(f * tr[0]["horizon"]))
            act = [r for r in tr if r["L"] >= k]
            a = auc([r["failed"] for r in act], [np.maximum.accumulate(np.asarray(r[key], float))[k - 1] for r in act])
            if a is not None:
                vals.append(a)
        out[task] = float(np.mean(vals)) if vals else None
    return out


def ranks(x):
    x = np.asarray(x, float); order = x.argsort(kind="mergesort"); r = np.empty(len(x)); s = x[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2; i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_test(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearman(x, y); rng = np.random.default_rng(SEED)
    perm = np.array([spearman(x, rng.permutation(y)) for _ in range(PERMUTATIONS)])
    p = float((np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (PERMUTATIONS + 1))
    rng = np.random.default_rng(SEED); bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(x), len(x)); v = spearman(x[i], y[i])
        if math.isfinite(v):
            bs.append(v)
    return dict(rho=round(rho, 4), p_permutation=round(p, 4), ci95=np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist(),
                tasks=len(x))


def fused_gain(cal, te, key, rk):
    """Fit P and key + P on cal, evaluate on te; per-failure lead differences."""
    calC, teC = cl.Cohort(cal, key), cl.Cohort(te, key)
    tm, fu = cl.fit_fast(calC, rk)
    eP, lP = cl.evaluate_fast(teC, rk, tm); eF, lF = cl.evaluate_fast(teC, rk, fu)
    return dict(P=eP, fused=dict(**cl.det_json(fu), **eF)), lF - lP, (calC, teC)


def per_task_gain(te, d):
    F = [r for r in te if r["failed"]]; g = defaultdict(list)
    for r, v in zip(F, d):
        g[r["task_name"]].append(v)
    return {t: float(np.mean(v)) for t, v in g.items()}


def power(gain, sd, n):
    se = sd * math.sqrt(10 / n); need = max(0.05, 1.959964 * se)
    return round(0.5 * (1 - math.erf((need - gain) / (se * math.sqrt(2)))), 3)


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    bundle = json.load(open(args.runtime_bundle))
    rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    names = sorted(cal[0]["baseline_scores"])
    if sorted(te[0]["baseline_scores"]) != names:
        raise ValueError("Calibration and test carry different baselines")
    attach(cal, names); attach(te, names)
    checks = {}
    if args.confirm_predictions:
        ref = {}
        for line in Path(args.confirm_predictions).read_text().splitlines():
            if line.strip():
                r = json.loads(line); ref[r["rollout_id"]] = r["baseline_scores"]
        worst = 0.0
        for r in te:
            for n in names:
                worst = max(worst, float(np.max(np.abs(np.asarray(ref[r["rollout_id"]][n], float) - r[n]))))
        if worst > 1e-6:
            raise ValueError(f"Test baseline trajectories differ from the confirm run (max diff {worst})")
        checks["test_baselines_match_confirm_max_abs_diff"] = worst
    types = {r["task_name"]: r.get("task_type") for r in cal + te}
    strata = sorted({t for t in types.values()})
    detectors = ["safe"] + names
    out = dict(plan="docs/xr1_lead_time_dose_response_plan.md", status="exploratory (opened cohorts)",
               counts=dict(calibration=len(cal), test=len(te)), checks=checks, detectors={}, D1={}, D3=[])
    for key in detectors:
        res, d, _ = fused_gain(cal, te, key, rk)
        Fte = [r for r in te if r["failed"]]
        lm_cal, lm_te = lm_auc(cal, key), lm_auc(te, key)
        row = dict(pooled=dict(**res, gain=round(float(d.mean()), 4), gain_test_only=qt.boot(d, Fte)),
                   lm_auc_calibration=dict(overall=round(float(np.mean([v for v in lm_cal.values() if v is not None])), 4)),
                   lm_auc_test=dict(overall=round(float(np.mean([v for v in lm_te.values() if v is not None])), 4)),
                   by_type={})
        gains = per_task_gain(te, d)
        for s in strata:
            mask = np.array([types[r["task_name"]] == s for r in Fte])
            Fs = [r for r, m in zip(Fte, mask) if m]
            cal_s = [r for r in cal if types[r["task_name"]] == s]; te_s = [r for r in te if types[r["task_name"]] == s]
            res_s, d_s, (cC, tC) = fused_gain(cal_s, te_s, key, rk)
            Fs_s = [r for r in te_s if r["failed"]]
            lm_s = [lm_cal[t] for t in lm_cal if types[t] == s and lm_cal[t] is not None]
            entry = dict(pooled_calibration_gain=round(float(d[mask].mean()), 4),
                         pooled_calibration_gain_test_only=qt.boot(d[mask], Fs),
                         stratum_calibration=dict(**res_s, gain=round(float(d_s.mean()), 4),
                                                  gain_test_only=qt.boot(d_s, Fs_s)),
                         lm_auc_calibration=round(float(np.mean(lm_s)), 4))
            if key in ("safe", "knn_cosine"):
                entry["stratum_calibration"]["gain_decision_bootstrap"] = cl.boot_reselect(cC, tC, rk, BOOT, SEED)
            row["by_type"][s] = entry
            out["D3"].append(dict(detector=key, stratum=s, lm_auc_calibration=entry["lm_auc_calibration"],
                                  pooled_calibration_gain=entry["pooled_calibration_gain"]))
        tasks = sorted(t for t in gains if lm_cal.get(t) is not None)
        out["D1"][key] = spearman_test([lm_cal[t] for t in tasks], [gains[t] for t in tasks])
        row["per_task"] = {t: dict(type=types[t], lm_auc_calibration=lm_cal[t], lm_auc_test=lm_te[t], gain=round(gains[t], 4))
                           for t in sorted(gains)}
        out["detectors"][key] = row
        print(f"{key:14s} gain {row['pooled']['gain']:+.4f} (FPR {row['pooled']['fused']['fpr']})  LM-AUC cal "
              f"{row['lm_auc_calibration']['overall']:.3f}  D1 rho {out['D1'][key]['rho']:+.3f} "
              f"p {out['D1'][key]['p_permutation']:.3f}", flush=True)
    out["D3_spearman"] = round(spearman([p["lm_auc_calibration"] for p in out["D3"]],
                                        [p["pooled_calibration_gain"] for p in out["D3"]]), 4)

    # D4 and the Amendment 2 gate.
    atomic = "atomic"
    gate = {}
    for key in ("safe", "knn_cosine"):
        if key not in out["detectors"] or atomic not in out["detectors"][key]["by_type"]:
            continue
        sc = out["detectors"][key]["by_type"][atomic]["stratum_calibration"]
        db = sc["gain_decision_bootstrap"]
        assumed = (0.05 + sc["gain"]) / 2
        gate[key] = dict(gain=sc["gain"], decision_ci95=db["gain_ci95"], decision_sd=db["gain_sd"],
                         fused_fpr=sc["fused"]["fpr"], P_fpr=sc["P"]["fpr"],
                         passes=db["gain_ci95"][0] > 0 and sc["gain"] >= 0.05, assumed_gain=round(assumed, 4),
                         power={str(n): power(assumed, db["gain_sd"], n) for n in SIZES})
    if "safe" in gate:
        g = gate["safe"]
        g["proceed_to_amendment_2"] = g["passes"]
        g["n_per_class_per_task"] = next((n for n in SIZES if g["power"][str(n)] >= 0.8), None) if g["passes"] else None
        if "knn_cosine" in gate:
            g["knn_secondary"] = abs(gate["knn_cosine"]["gain"] - g["gain"]) <= 0.02
    out["D4"] = gate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("D3 (detector x stratum) Spearman:", out["D3_spearman"])
    for p in out["D3"]:
        print(f"  {p['detector']:14s} {p['stratum']:10s} LM-AUC {p['lm_auc_calibration']:.3f}  gain {p['pooled_calibration_gain']:+.4f}")
    print("D4:", json.dumps(gate, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--calibration", "--test", "--runtime-bundle", "--output"):
        p.add_argument(name, required=True)
    p.add_argument("--confirm-predictions")
    run(p.parse_args())


if __name__ == "__main__":
    main()
