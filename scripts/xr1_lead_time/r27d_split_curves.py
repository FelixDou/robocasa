"""R27d (docs/r27d_split_curves_plan.md): Fig. 2's own-band curves on R27's 20 random 30/70 assignments.
Descriptive follow-up to R27c. Policy-agnostic, called by seen_unseen_tables.py (XR-1) and rldx_test_table.py
(RLDX-1) with --r27d.

Fig. 2 draws each own band (SAFE-MLP, SAFE-LSTM, Hide-and-Seek; seen and unseen tasks) as R21's lead-FPR curve on
the fixed 50/50 split: the three single models' bands at every calibration target, averaged over the models at each
target. Table II and the text now use R27's 20 random 30/70 assignments for every band (R27c), and Fig. 1 draws
Hide-and-Seek's unseen-task curve as the median over those assignments (R27b). R27d computes the same curve for every
head and side on the same assignments, so that Fig. 2 can be drawn like Fig. 1:

  per assignment   each single model's band over the calibration targets (r21_native_bands.curve) and their mean
                   at each target (r27_band_split.mean_curve, the points Fig. 2 draws)
  summary          Fig. 1's construction: each assignment's mean curve interpolated on an FPR grid within its
                   observed range, the median (and range) where at least 15 assignments reach the FPR; the target
                   point, the median FPR and lead at alpha = 0.05; the median curve's gap to each timer's curve
                   (R18: P on seen tasks, E and G on unseen tasks) on their overlap up to Fig. 2's axis

Everything else is R21's: scores, folds, SAFE's routine, the timers and the test rollouts.

Gates: D0, at R19's fixed 50/50 split every head's single-model curves on both sides equal R21's saved curves; D1,
Hide-and-Seek's unseen-task mean curve on each 30/70 assignment equals R27b's saved curve (Fig. 1's data); D2, every
assignment is a 30/70 partition of the calibration successes, with the same reference rollouts on both sides.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r27c_split_all_bands as r27c  # noqa: E402

r27, r21 = r27c.r27, r27c.r21
PLAN = "docs/r27d_split_curves_plan.md (R27d)"
SEEDS = r21.SEEDS
SIDES = ("seen", "unseen")
TIMERS = dict(seen=("P",), unseen=("E", "G"))
SPLITS, SPLIT_SEED = r27.SPLITS, r27.SPLIT_SEED
ALPHA = r27.ALPHA
XMAX = r27.XMAX                 # Fig. 2's axis
MIN_ASSIGNMENTS = 15            # Fig. 1's rule: a median point needs at least this many assignments
GRID = np.linspace(0.0, XMAX, 301)
NEAR = 0.02                     # reading: a curve "on" or "near" a timer's stays within this lead of it
HS = r27.FULL
TOL = 1e-12


def head_curves(d, by_seed):
    """One head on one Design: {seed label: single-model curve} and their mean at each target (with alpha)."""
    curves = {lab: r21.curve(d, key_of) for lab, key_of in by_seed.items()}
    first = next(iter(curves.values()))
    mean = [dict(alpha=first[j]["alpha"], fpr=x, mean_lead=y) for j, (x, y) in enumerate(r27.mean_curve(curves))]
    return curves, mean


def all_curves(ds, keys):
    """keys {head: {side: {seed label: key_of}}} -> ({head: {side: single-model curves}}, {head: {side: mean}})."""
    single, mean = {}, {}
    for h, by_side in keys.items():
        single[h], mean[h] = {}, {}
        for side, by_seed in by_side.items():
            single[h][side], mean[h][side] = head_curves(ds[side], by_seed)
    return single, mean


def on_grid(mean_curve, grid=GRID):
    """The mean curve interpolated on the grid within its observed FPR range (NaN outside), as Fig. 1 does."""
    pts = np.array(sorted((q["fpr"], q["mean_lead"]) for q in mean_curve))
    return np.where((grid >= pts[0, 0]) & (grid <= pts[-1, 0]), np.interp(grid, pts[:, 0], pts[:, 1]), np.nan)


def median_curve(mean_curves, grid=GRID, min_n=MIN_ASSIGNMENTS):
    """Median, min and max over the assignments' mean curves where at least min_n reach the FPR."""
    Y = np.array([on_grid(c, grid) for c in mean_curves])
    n = (~np.isnan(Y)).sum(axis=0)
    keep = n >= min_n
    if not keep.any():
        return dict(fpr=[], median=[], min=[], max=[], n=[])
    Z = Y[:, keep]
    return dict(fpr=grid[keep].tolist(), median=np.nanmedian(Z, axis=0).tolist(), min=np.nanmin(Z, axis=0).tolist(),
                max=np.nanmax(Z, axis=0).tolist(), n=n[keep].tolist())


def target_point(mean_curves, alpha=ALPHA):
    """Median FPR and lead over the assignments of the band calibrated for the target (Fig. 1's dot)."""
    pts = [next(q for q in c if abs(q["alpha"] - alpha) < 1e-9) for c in mean_curves]
    return dict(fpr=float(np.median([q["fpr"] for q in pts])), mean_lead=float(np.median([q["mean_lead"] for q in pts])))


def versus(points, timer, xmax=XMAX):
    """points, timer: [(fpr, lead)]. The curve's gap to the timer's on their overlap up to xmax (r27.versus_timer:
    overlap, smallest and largest gap, the FPR above which the curve stays above the timer)."""
    if len(points) < 2:
        return dict(overlap=None, last_not_above=None, above_from=None, share_above=None)
    return r27.versus_timer(points, timer, xmax=xmax)


def describe(mean_curves, timers):
    """The summary of one head on one side. timers {name: [(fpr, lead)]}."""
    med = median_curve(mean_curves)
    pts = list(zip(med["fpr"], med["median"]))
    return dict(median_curve=med, target=target_point(mean_curves), versus={t: versus(pts, c) for t, c in timers.items()})


def describe_fixed(mean_curve, timers):
    """The same gaps for the fixed 50/50 split's mean curve (Fig. 2 as drawn now)."""
    at = next(q for q in mean_curve if abs(q["alpha"] - ALPHA) < 1e-9)
    pts = [(q["fpr"], q["mean_lead"]) for q in mean_curve]
    return dict(target=dict(fpr=at["fpr"], mean_lead=at["mean_lead"]), versus={t: versus(pts, c) for t, c in timers.items()})


def timer_curves(r18):
    """{side: {timer: [(fpr, lead)]}} from R18's saved timer curves."""
    return {side: {t: [(p["fpr"], p["mean_lead"]) for p in r18["timers"][side][t]] for t in TIMERS[side]}
            for side in SIDES}


def d0_diff(single, ref21):
    """Largest difference between the fixed-split single-model curves and R21's saved curves (alpha, FPR, mean lead,
    false alarms). A missing or shorter reference curve fails the gate."""
    worst, compared, missing = 0.0, 0, []
    for h, by_side in single.items():
        for side, by_seed in by_side.items():
            for lab, cv in by_seed.items():
                saved = (((ref21.get("curves") or {}).get(h) or {}).get(side) or {}).get(lab)
                if not saved or len(saved) != len(cv):
                    missing.append(f"{h}/{side}/{lab}")
                    continue
                for a, b in zip(cv, saved):
                    worst = max(worst, abs(a["alpha"] - b["alpha"]), abs(a["fpr"] - b["fpr"]),
                                abs(a["mean_lead"] - b["mean_lead"]), 0.0 if a["fp"] == b["fp"] else math.inf)
                    compared += 1
    if missing:
        worst = math.inf
    return worst, compared, missing


def d1_diff(rows, ref27b, head=HS):
    """Hide-and-Seek's unseen-task mean curves on the 30/70 assignments against R27b's saved ones."""
    saved = ref27b.get("published") or []
    worst, compared = 0.0, 0
    for j, row in enumerate(rows):
        if j >= len(saved) or len(saved[j]["mean_curve"]) != len(row["mean_curves"][head]["unseen"]):
            return math.inf, compared
        for a, b in zip(row["mean_curves"][head]["unseen"], saved[j]["mean_curve"]):
            worst = max(worst, abs(a["alpha"] - b["alpha"]), abs(a["fpr"] - b["fpr"]), abs(a["mean_lead"] - b["mean_lead"]))
            compared += 1
    return worst, compared


def run(cal, te, cal_tasks, test_tasks, rk, keys, r18, ref21, ref27b, splits=SPLITS, seed=SPLIT_SEED, progress=None):
    """keys {head: {side: {seed label: key_of}}} (each single model's normalized raw score of each fold); r18, ref21,
    ref27b: the saved r18, r21 and r27b blocks of this policy's outputs. rk is unused (the curves need no timer) and
    kept for the callers' signature. Returns (result, problems)."""
    timers = timer_curves(r18)
    base_single, base_mean = all_curves(r27c.designs(cal, te, cal_tasks, test_tasks), keys)
    d0, compared0, missing0 = d0_diff(base_single, ref21)
    rows, d2 = [], True
    for j in range(splits):
        split = lambda rs, j=j: r27.split_30_70(rs, seed + j)
        ds = r27c.designs(cal, te, cal_tasks, test_tasks, split=split)
        for d in ds.values():
            d2 = d2 and r27.check_split(d.cal_rows, d.ref, d.conf)
        d2 = d2 and [str(r["rollout_id"]) for r in ds["unseen"].ref] == [str(r["rollout_id"]) for r in ds["seen"].ref]
        _, mean = all_curves(ds, keys)
        rows.append(dict(split=j, mean_curves=mean))
        if progress:
            progress(f"R27d split {j + 1}/{splits}")
    d1, compared1 = d1_diff(rows, ref27b) if HS in keys else (0.0, 0)
    summary, fixed = {}, {}
    for h, by_side in keys.items():
        summary[h], fixed[h] = {}, {}
        for side in by_side:
            summary[h][side] = describe([r["mean_curves"][h][side] for r in rows], timers[side])
            fixed[h][side] = describe_fixed(base_mean[h][side], timers[side])
    checks = dict(D0_fixed_split_curves_equal_r21=dict(max_abs_diff=d0, compared=compared0, missing=missing0,
                                                       ok=compared0 > 0 and d0 <= TOL),
                  D1_hide_and_seek_curves_equal_r27b=dict(max_abs_diff=d1, compared=compared1,
                                                          ok=(compared1 > 0 or HS not in keys) and d1 <= TOL),
                  D2_splits_are_30_70_partitions=dict(max_abs_diff=0.0 if d2 else math.inf, ok=bool(d2)))
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return dict(plan=PLAN, alpha=ALPHA, split_seed=seed, reference_share=r27.REFERENCE_SHARE,
                min_assignments=MIN_ASSIGNMENTS, near=NEAR, xmax=XMAX, timers=timers,
                fixed_50_50=dict(mean_curves=base_mean, summary=fixed), splits=rows, summary=summary,
                checks=checks), problems


def gap_text(v):
    if v.get("overlap") is None:
        return "no overlap"
    lo, hi = v["overlap"]
    above = "never" if v["above_from"] is None else f"{100 * v['above_from']:.1f}%"
    return (f"gap [{v['gap_min']:+.3f}, {v['gap_max']:+.3f}] on {100 * lo:.1f}-{100 * hi:.1f}% FPR, "
            f"above from {above}")


def print_summary(r, label):
    print(f"R27d {label}: own-band curves, median over {len(r['splits'])} random 30/70 assignments "
          f"(where at least {r['min_assignments']} reach the FPR); fixed 50/50 for reference")
    for h, by_side in r["summary"].items():
        for side, s in by_side.items():
            med, tp = s["median_curve"], s["target"]
            span = "none" if not med["fpr"] else f"{100 * med['fpr'][0]:.1f}-{100 * med['fpr'][-1]:.1f}%"
            fx = r["fixed_50_50"]["summary"][h][side]
            print(f"  {h:<14} {side:<6} median curve on FPR {span}; 5% target: FPR {100 * tp['fpr']:.1f}%, "
                  f"lead {tp['mean_lead']:.3f} (fixed 50/50: {100 * fx['target']['fpr']:.1f}%, "
                  f"{fx['target']['mean_lead']:.3f})")
            for t, v in s["versus"].items():
                print(f"    vs {t}: {gap_text(v)} | fixed 50/50: {gap_text(fx['versus'][t])}")
    for k, v in r["checks"].items():
        print(f"R27d {k.split('_', 1)[0]}: {'OK' if v['ok'] else 'FAILED'} (max |diff| {v['max_abs_diff']})")
