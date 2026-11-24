"""R27c (docs/r27c_split_all_bands_plan.md): every monitor's own band on R27's 20 random 30/70 assignments.
Descriptive follow-up to R27 and R27b. Policy-agnostic, called by seen_unseen_tables.py (XR-1) and rldx_test_table.py
(RLDX-1) with --r27c.

R27 refitted only Hide-and-Seek's band on random 30/70 assignments of the calibration successes; SAFE's bands and the
comparisons with G (told H_k) and P stayed on the fixed 50/50 split of R19 to R25. R27c puts every head on the same
assignments (R27's seeds, so the same splits), as a three-seed ensemble at alpha = 0.05:

  unseen  each head's band against E (G_abs) and G, each matched to the band's test false alarms (R21's evaluate)
  seen    each head's band against P, matched the same way (calibration successes of every task, split per task
          with the same seed, hence the same assignment)

Everything else is R21's: scores, folds, SAFE's routine (modulation tfunc, empirical quantile, last-value padding,
alarm at the first score >= the band), the timers and the test rollouts.

Gates: Q0, at R19's fixed 50/50 split every head's ensemble on every side and timer equals R21's saved point (false
alarms, matched gain); Q1, Hide-and-Seek's 30/70 rows against E equal R27's saved rows; Q2, every assignment is a
30/70 partition of the calibration successes, on both sides.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r27_band_split as r27  # noqa: E402

r21 = r27.r21
PLAN = "docs/r27c_split_all_bands_plan.md (R27c)"
ALL = "all"
SIDES = dict(unseen=("G_abs", "G"), seen=("P",))
SPLITS, SPLIT_SEED = r27.SPLITS, r27.SPLIT_SEED
MIN_SPLITS = r27.MIN_SPLITS     # reading: "clearly" means in at least this many of the 20 assignments
ALPHA = r27.ALPHA
TOL = 1e-12


def designs(cal, te, cal_tasks, test_tasks, split=None):
    """The two sides of the table on one assignment (split=None: R19's fixed 50/50 halves)."""
    every = set().union(*test_tasks.values())
    return dict(unseen=r21.Design(cal, te, cal_tasks, test_tasks, split=split),
                seen=r21.Design(cal, te, {ALL: every}, {ALL: every}, split=split))


def point(d, key_of, timers, rk):
    """The ensemble band alone against each timer: false alarms, FPR, mean lead and the gain over the timer matched
    to the band's test false alarms."""
    ev = r21.evaluate(d, key_of, timers, rk, alpha=ALPHA)
    out = {}
    for t in timers:
        sa = ev[t]["standalone"]
        m = sa.get("matched") or {}
        out[t] = dict(fp=int(sa["fp"]), fpr=float(sa["fpr"]), mean_lead=float(sa["mean_lead"]),
                      matched_gain=r21.matched_gain(sa), matched_timer_fp=m.get("timer_fp"),
                      timer_fp=int(ev[t]["timer_alone"]["fp"]))
    return out


def evaluate_all(ds, keys, rk):
    """keys {head: {side: key_of}} -> {head: {side: {timer: point}}}."""
    return {h: {side: point(ds[side], by_side[side], SIDES[side], rk) for side in SIDES if side in by_side}
            for h, by_side in keys.items()}


def value_diff(a, b):
    return r27.value_diff(a, b)


def q0_diff(base, ref21):
    """Largest difference between the fixed-split points and R21's saved bootstrap points (false alarms must be
    equal; matched gains within TOL). Missing reference entries count as a failure."""
    worst, compared, missing = 0.0, 0, []
    for h, by_side in base.items():
        for side, by_timer in by_side.items():
            for t, v in by_timer.items():
                ref = (((ref21.get("bootstrap") or {}).get(side) or {}).get("timers") or {}).get(t, {}).get(h)
                if not ref:
                    missing.append(f"{h}/{side}/{t}")
                    continue
                sa = ref["standalone"]
                worst = max(worst, 0.0 if v["fp"] == sa["fp"] else math.inf,
                            value_diff(v["matched_gain"], sa["matched_gain"]))
                compared += 1
    if missing:
        worst = math.inf
    return worst, compared, missing


def q1_diff(rows, ref27, head=r27.FULL):
    """Hide-and-Seek's 30/70 rows against E against R27's saved rows (false alarms, matched gain)."""
    worst = 0.0
    saved = ref27.get("splits") or []
    for j, row in enumerate(rows):
        if j >= len(saved):
            return math.inf
        mine = row["heads"][head]["unseen"]["G_abs"]
        theirs = saved[j]["matched"]
        worst = max(worst, 0.0 if mine["fp"] == theirs["fp"] else math.inf, value_diff(mine["matched_gain"], theirs["gain"]))
    return worst


def summarize(rows):
    """{head: {side: {timer: spreads and counts}}} over the assignments."""
    out = {}
    first = rows[0]["heads"]
    for h, by_side in first.items():
        out[h] = {}
        for side, by_timer in by_side.items():
            out[h][side] = {}
            for t in by_timer:
                vals = [r["heads"][h][side][t] for r in rows]
                gains = [v["matched_gain"] for v in vals]
                ok = [g for g in gains if g is not None and not math.isnan(g)]
                pos, neg = sum(1 for g in ok if g > 0), sum(1 for g in ok if g < 0)
                out[h][side][t] = dict(
                    matched_gain=r27.spread(gains), fp=r27.spread([v["fp"] for v in vals]),
                    fpr=r27.spread([v["fpr"] for v in vals]), mean_lead=r27.spread([v["mean_lead"] for v in vals]),
                    matched_timer_fp=r27.spread([v["matched_timer_fp"] for v in vals]),
                    splits=len(rows), positive=pos, negative=neg,
                    clearly=dict(earlier=pos >= MIN_SPLITS, later=neg >= MIN_SPLITS))
    return out


def run(cal, te, cal_tasks, test_tasks, rk, keys, ref21, ref27, splits=SPLITS, seed=SPLIT_SEED, progress=None):
    """keys {head: {side: key_of}} (the ensemble's normalized raw score of each fold); ref21, ref27: the saved r21 and
    r27 blocks of this policy's outputs. Returns (result, problems)."""
    base = evaluate_all(designs(cal, te, cal_tasks, test_tasks), keys, rk)
    q0, compared, missing = q0_diff(base, ref21)
    rows, q2 = [], True
    for j in range(splits):
        split = lambda rs, j=j: r27.split_30_70(rs, seed + j)
        ds = designs(cal, te, cal_tasks, test_tasks, split=split)
        for d in ds.values():
            q2 = q2 and r27.check_split(d.cal_rows, d.ref, d.conf)
        q2 = q2 and [str(r["rollout_id"]) for r in ds["unseen"].ref] == [str(r["rollout_id"]) for r in ds["seen"].ref]
        rows.append(dict(split=j, heads=evaluate_all(ds, keys, rk)))
        if progress:
            progress(f"R27c split {j + 1}/{splits}")
    q1 = q1_diff(rows, ref27) if r27.FULL in keys else 0.0
    checks = dict(Q0_fixed_split_equals_r21=dict(max_abs_diff=q0, compared=compared, missing=missing,
                                                  ok=compared > 0 and q0 <= TOL),
                  Q1_hide_and_seek_equals_r27=dict(max_abs_diff=q1, ok=q1 <= TOL),
                  Q2_splits_are_30_70_partitions=dict(max_abs_diff=0.0 if q2 else math.inf, ok=bool(q2)))
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return dict(plan=PLAN, alpha=ALPHA, split_seed=seed, reference_share=r27.REFERENCE_SHARE, sides=SIDES,
                fixed_50_50=base, splits=rows, summary=summarize(rows), checks=checks), problems


def print_summary(r, label):
    s = r["summary"]
    print(f"R27c {label}: own bands on {len(r['splits'])} random 30/70 assignments "
          "(median [min, max]; positive / negative assignments)")
    names = dict(G_abs="E", G="G", P="P")
    for h, by_side in s.items():
        for side, by_timer in by_side.items():
            for t, v in by_timer.items():
                fixed = r["fixed_50_50"][h][side][t]["matched_gain"]
                print(f"  {h:<14} {side:<6} vs {names.get(t, t)}: gain {r27.fmt(v['matched_gain'])} "
                      f"(+{v['positive']}/-{v['negative']}); FPR % {r27.fmt(v['fpr'], pct=True, digits=1)}; "
                      f"fixed 50/50 {fixed:+.3f}")
    for k, v in r["checks"].items():
        print(f"R27c {k.split('_', 1)[0]}: {'OK' if v['ok'] else 'FAILED'} (max |diff| {v['max_abs_diff']})")
