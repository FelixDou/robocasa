"""R27e (docs/r27e_table1_bands_plan.md): Table I's own-band rows on R27's 20 random 30/70 assignments (XR-1).
Descriptive follow-up to R27c and R27d, called by seen_unseen_tables.py with --r27e.

Table I reports, for XR-1's seen and unseen tasks, each own band's balanced accuracy (an alarm at a failure's final
inference counted as a miss, R26) and mean lead at the 5% target, as the mean and std over the three single models.
Its own-band rows still come from R21's fixed 50/50 split, while Table II, Fig. 1 and Fig. 2 now use R27's 20 random
30/70 assignments. R27e computes the same rows on those assignments:

  per assignment   each single model's band at alpha = 0.05 alone (SAFE's routine, pooled over folds as R21's
                   evaluate and curve do), recounted as R26 does (r26_alarms_at_limit.from_counts), and the mean
                   and std over the three models (r26_alarms_at_limit.with_seed_summary)
  summary          for every value, the median over the assignments of the mean over models and of their std (the
                   Table I cell), and each model's median over the assignments

Everything else is R21's: scores, folds, SAFE's routine, the test rollouts.

Gates: E0, at R19's fixed 50/50 split every head's single-model rows on both sides equal R26's saved own-band rows
(Table I as printed); E1, on each 30/70 assignment the mean over the models of FPR and mean lead equals R27d's saved
mean curve at alpha = 0.05 (Fig. 2's points); E2, every assignment is a 30/70 partition of the calibration successes,
with the same reference rollouts on both sides.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r26_alarms_at_limit as r26  # noqa: E402
import r27d_split_curves as r27d  # noqa: E402

r27c, r27, r21 = r27d.r27c, r27d.r27, r27d.r21
PLAN = "docs/r27e_table1_bands_plan.md (R27e)"
SEED_LABELS = r26.SEED_LABELS
KEYS = r26.KEYS
SPLITS, SPLIT_SEED = r27.SPLITS, r27.SPLIT_SEED
ALPHA = r27.ALPHA
TOL = 1e-12
SUMMARY = "mean_std_over_seeds"


def standalone(d, key_of, alpha=ALPHA):
    """One single model's band alone at alpha, pooled over folds (r19_extras.outcome, as r21_native_bands.evaluate's
    standalone part and r21_native_bands.curve compute it)."""
    alarms, L, H, failed = [], [], [], []
    for F in d.folds:
        ri, ci, ti = d.fold(F, d.identity())
        if len(ti) == 0:
            continue
        key = key_of(F)
        fit = r21.band_fit([d.ref[i][key] for i in ri], [d.conf[i][key] for i in ci], alpha)
        alarms.append(r21.band_alarms([d.test[i][key] for i in ti], fit))
        tc = d.test_cohort.take(ti)
        L.append(tc.L), H.append(tc.H), failed.append(tc.failed)
    o, _ = r21.rx.outcome(np.concatenate(alarms), np.concatenate(L), np.concatenate(H), np.concatenate(failed))
    return o


def rows_on(ds, keys, alpha=ALPHA):
    """keys {head: {side: {seed label: key_of}}} -> {head: {side: {seed label: recount, mean_std_over_seeds}}}."""
    return {h: {side: r26.with_seed_summary({lab: r26.from_counts(standalone(ds[side], k, alpha))
                                             for lab, k in by_seed.items()})
                for side, by_seed in by_side.items()}
            for h, by_side in keys.items()}


def e0_diff(base, ref26):
    """Largest difference between the fixed-split single-model rows and R26's saved own-band rows (every recounted
    value). A missing reference fails the gate."""
    worst, compared, missing = 0.0, 0, []
    rows = (ref26 or {}).get("rows") or {}
    for h, by_side in base.items():
        for side, by_label in by_side.items():
            ref_side = (rows.get(f"{h}_own_band") or {}).get(side) or {}
            for lab in SEED_LABELS:
                if lab not in by_label:
                    continue
                if lab not in ref_side:
                    missing.append(f"{h}/{side}/{lab}")
                    continue
                worst = max(worst, r26.worst_diff(by_label[lab], ref_side[lab], KEYS))
                compared += 1
    if missing:
        worst = math.inf
    return worst, compared, missing


def e1_diff(rows, ref27d, alpha=ALPHA):
    """Each assignment's mean over the models of FPR and mean lead against R27d's mean curve at alpha."""
    saved = (ref27d or {}).get("splits") or []
    worst, compared = 0.0, 0
    for j, row in enumerate(rows):
        if j >= len(saved):
            return math.inf, compared
        for h, by_side in row["rows"].items():
            for side, by_label in by_side.items():
                curve = ((saved[j].get("mean_curves") or {}).get(h) or {}).get(side)
                point = next((q for q in curve or [] if abs(q["alpha"] - alpha) < 1e-9), None)
                if point is None:
                    return math.inf, compared
                m = by_label[SUMMARY]
                worst = max(worst, abs(m["fpr"]["mean"] - point["fpr"]), abs(m["mean_lead"]["mean"] - point["mean_lead"]))
                compared += 1
    return worst, compared


def summarize(rows):
    """{head: {side: {key: cell}}}: the median over the assignments of the mean over models and of their std (Table
    I's cell), the range of the mean, and each model's median over the assignments."""
    out = {}
    first = rows[0]["rows"]
    for h, by_side in first.items():
        out[h] = {}
        for side in by_side:
            per = [r["rows"][h][side] for r in rows]
            cells = {}
            for k in KEYS:
                means = [p[SUMMARY][k]["mean"] for p in per]
                stds = [p[SUMMARY][k]["std"] for p in per]
                cells[k] = dict(mean=float(np.median(means)), std=float(np.median(stds)), mean_min=float(min(means)),
                                mean_max=float(max(means)),
                                by_seed={lab: float(np.median([p[lab][k] for p in per])) for lab in SEED_LABELS})
            out[h][side] = dict(cells, splits=len(rows))
    return out


def run(cal, te, cal_tasks, test_tasks, keys, ref26, ref27d, splits=SPLITS, seed=SPLIT_SEED, progress=None):
    """keys {head: {side: {seed label: key_of}}} (each single model's normalized raw score of each fold); ref26,
    ref27d: the saved r26 and r27d blocks of XR-1's outputs. Returns (result, problems)."""
    base = rows_on(r27c.designs(cal, te, cal_tasks, test_tasks), keys)
    e0, compared0, missing0 = e0_diff(base, ref26)
    rows, e2 = [], True
    for j in range(splits):
        split = lambda rs, j=j: r27.split_30_70(rs, seed + j)
        ds = r27c.designs(cal, te, cal_tasks, test_tasks, split=split)
        for d in ds.values():
            e2 = e2 and r27.check_split(d.cal_rows, d.ref, d.conf)
        e2 = e2 and [str(r["rollout_id"]) for r in ds["unseen"].ref] == [str(r["rollout_id"]) for r in ds["seen"].ref]
        rows.append(dict(split=j, rows=rows_on(ds, keys)))
        if progress:
            progress(f"R27e split {j + 1}/{splits}")
    e1, compared1 = e1_diff(rows, ref27d)
    checks = dict(E0_fixed_split_rows_equal_r26=dict(max_abs_diff=e0, compared=compared0, missing=missing0,
                                                     ok=compared0 > 0 and e0 <= TOL),
                  E1_means_equal_r27d_curves=dict(max_abs_diff=e1, compared=compared1, ok=compared1 > 0 and e1 <= TOL),
                  E2_splits_are_30_70_partitions=dict(max_abs_diff=0.0 if e2 else math.inf, ok=bool(e2)))
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return dict(plan=PLAN, alpha=ALPHA, split_seed=seed, reference_share=r27.REFERENCE_SHARE, fixed_50_50=base,
                splits=rows, summary=summarize(rows), checks=checks), problems


def print_summary(r, label="XR-1"):
    print(f"R27e {label}: Table I's own-band rows on {len(r['splits'])} random 30/70 assignments (median over the "
          "assignments of the mean +- std over the three models; fixed 50/50 for reference)")
    print(f"  {'head':<14} {'side':<6} {'bACC (limit = miss)':>20} {'lead':>14} {'recall before limit':>20} "
          f"{'FPR':>7} | fixed 50/50: bACC, lead, recall")
    for h, by_side in r["summary"].items():
        for side, s in by_side.items():
            c = lambda k: f"{s[k]['mean']:.3f}+-{s[k]['std']:.3f}"
            fx = r["fixed_50_50"][h][side][SUMMARY]
            f = lambda k: f"{fx[k]['mean']:.3f}+-{fx[k]['std']:.3f}"
            print(f"  {h:<14} {side:<6} {c('bacc_before_limit'):>20} {c('mean_lead'):>14} {c('tpr_before_limit'):>20} "
                  f"{100 * s['fpr']['mean']:6.1f}% | {f('bacc_before_limit')}, {f('mean_lead')}, {f('tpr_before_limit')}")
    for k, v in r["checks"].items():
        print(f"R27e {k.split('_', 1)[0]}: {'OK' if v['ok'] else 'FAILED'} (max |diff| {v['max_abs_diff']})")
