"""R27b (docs/r27b_split_controls_plan.md): why Hide-and-Seek's own-band gain moved under the published 30/70 split.
Descriptive follow-up to R27. Policy-agnostic, called by seen_unseen_tables.py (XR-1) and rldx_test_table.py (RLDX-1)
with --r27b, on unseen tasks only.

R27 found the XR-1 gain over matched E (0.147 with the fixed 50/50 split) above all 20 random 30/70 assignments. That
can mean that the 30/70 ratio performs worse, or that the fixed assignment (rollouts sorted by ID alternate) was
favorable. R27b separates the two and saves what the paper's opening figure needs:

  published   R27's 20 random 30/70 assignments (same seeds), now with each assignment's curve (mean of the three
              models at each calibration target)
  half        20 random 50/50 assignments, stratified by task, with the same quantities
  example     the failed test rollouts' alarms under the representative 30/70 assignment: the one whose matched gain
              is closest to the median (ties: the first)

Everything else is R27's: scores, folds, SAFE's routine, E, and the test rollouts.

Gates: M1, the 30/70 assignments reproduce R27's saved rows; M2, every random 50/50 assignment is a partition of the
calibration successes with half of each task in the reference set; M3, the saved alarms of the representative
assignment give its ensemble's and E's mean lead.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r27_band_split as r27  # noqa: E402

r21 = r27.r21
PLAN = "docs/r27b_split_controls_plan.md (R27b)"
SPLITS = r27.SPLITS
PUBLISHED_SEED, HALF_SEED = r27.SPLIT_SEED, 127
FAVORABLE_SHARE = 0.9           # reading: the fixed 50/50 assignment beats at least 90% of random 50/50 assignments
TOL = 1e-12


def half_count(n):
    return min(n - 1, max(1, n // 2))


def split_half(rows, seed):
    """(reference, nonconformity): per task, a random half of the successes (rollouts sorted by ID first)."""
    by_task = defaultdict(list)
    for r in rows:
        if not r["failed"]:
            by_task[r["task_name"]].append(r)
    rng = np.random.default_rng(seed)
    ref, conf = [], []
    for t in sorted(by_task):
        ordered = sorted(by_task[t], key=lambda r: str(r["rollout_id"]))
        perm = rng.permutation(len(ordered))
        k = half_count(len(ordered))
        ref += [ordered[i] for i in sorted(perm[:k].tolist())]
        conf += [ordered[i] for i in sorted(perm[k:].tolist())]
    return ref, conf


def check_half(rows, ref, conf):
    """Gate M2: a partition of the successes, with half of each task in the reference set."""
    ident = lambda rs: [str(r["rollout_id"]) for r in rs]
    if sorted(ident(ref) + ident(conf)) != sorted(str(r["rollout_id"]) for r in rows if not r["failed"]):
        return False
    n, k = defaultdict(int), defaultdict(int)
    for r in ref + conf:
        n[r["task_name"]] += 1
    for r in ref:
        k[r["task_name"]] += 1
    return all(k[t] == half_count(n[t]) for t in n)


def one(d, rk, timer_curve):
    """R27's three quantities on one Design, with the curve Fig. 2 would draw for this assignment."""
    row, curves = r27.evaluate(d, rk, timer_curve)
    first = next(iter(curves.values()))
    points = r27.mean_curve(curves)
    return dict(row, mean_curve=[dict(alpha=first[j]["alpha"], fpr=x, mean_lead=y) for j, (x, y) in enumerate(points)])


def representative(rows):
    """Index of the assignment whose matched gain is closest to the median (ties: the first)."""
    gains = [r["matched"]["gain"] for r in rows]
    med = float(np.median(gains))
    return min(range(len(gains)), key=lambda i: (round(abs(gains[i] - med), 12), i))


def row_diff(mine, saved):
    """Largest difference between a recomputed assignment and R27's saved row."""
    worst = 0.0
    for part, keys in (("matched", ("fp", "fpr", "mean_lead", "gain", "timer_fp")),
                       ("first_step", ("budget", "matched_fp", "alpha", "delta")),
                       ("curve", ("last_not_above", "above_from", "share_above"))):
        for k in keys:
            worst = max(worst, r27.value_diff(mine[part][k], saved[part][k]))
    return worst


def standing(value, rows, get):
    """How many assignments lie strictly below a value."""
    return sum(1 for r in rows if get(r) is not None and not math.isnan(get(r)) and get(r) < value)


def run(cal, te, cal_tasks, test_tasks, rk, r18, ref27, splits=SPLITS, progress=None):
    """r18, ref27: the saved r18 and r27 blocks of this policy's outputs. Returns (result, problems)."""
    timer_curve = [(p["fpr"], p["mean_lead"]) for p in r18["timers"]["unseen"]["E"]]
    published, half, m1, m2 = [], [], 0.0, True
    designs = []
    for j in range(splits):
        d = r21.Design(cal, te, cal_tasks, test_tasks, split=lambda rs, j=j: r27.split_30_70(rs, PUBLISHED_SEED + j))
        designs.append(d)
        row = one(d, rk, timer_curve)
        published.append(dict(row, split=j))
        m1 = max(m1, row_diff(row, ref27["splits"][j]) if j < len(ref27["splits"]) else math.inf)
        if progress:
            progress(f"R27b published split {j + 1}/{splits}")
    for j in range(splits):
        d = r21.Design(cal, te, cal_tasks, test_tasks, split=lambda rs, j=j: split_half(rs, HALF_SEED + j))
        m2 = m2 and check_half(d.cal_rows, d.ref, d.conf)
        half.append(dict(one(d, rk, timer_curve), split=j))
        if progress:
            progress(f"R27b random 50/50 split {j + 1}/{splits}")
    rep = representative(published)
    alarms = r27.failure_alarms(designs[rep], rk)
    point = r21.evaluate(designs[rep], lambda F: f"{r27.FULL}~@{F}", (r27.TIMER,), rk)[r27.TIMER]
    m3 = max(abs(r27.alarm_leads(alarms, "band_alarm") - point["standalone"]["mean_lead"]),
             abs(r27.alarm_leads(alarms, "timer_alarm") - point["timer_alone"]["mean_lead"]))
    fixed = ref27["reference_50_50"]
    n = len(half)
    below_gain = standing(fixed["matched"]["gain"], half, lambda r: r["matched"]["gain"])
    below_delta = standing(fixed["first_step"]["delta"], half, lambda r: r["first_step"]["delta"])
    comparison = dict(
        fixed_gain=fixed["matched"]["gain"], fixed_first_step=fixed["first_step"]["delta"], random_half_assignments=n,
        random_half_below_fixed_gain=below_gain, random_half_below_fixed_first_step=below_delta,
        fixed_assignment_favorable=dict(gain=below_gain >= FAVORABLE_SHARE * n, first_step=below_delta >= FAVORABLE_SHARE * n))
    checks = dict(M1_published_equals_r27=dict(max_abs_diff=m1, ok=m1 <= TOL),
                  M2_random_halves_are_partitions=dict(max_abs_diff=0.0 if m2 else math.inf, ok=bool(m2)),
                  M3_example_alarms_give_the_mean_leads=dict(max_abs_diff=m3, ok=m3 <= TOL))
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return dict(plan=PLAN, seeds=dict(published=PUBLISHED_SEED, half=HALF_SEED), published=published, half=half,
                summary=dict(published=r27.summarize(published), half=r27.summarize(half)), comparison=comparison,
                example=dict(split=rep, gain=published[rep]["matched"]["gain"], failure_alarms=alarms), checks=checks), problems


def print_summary(r, label):
    p, h, c = r["summary"]["published"], r["summary"]["half"], r["comparison"]
    print(f"R27b {label}: Hide-and-Seek's own band, unseen tasks ({p['splits']} random assignments of each kind; "
          "median [min, max])")
    print(f"  Delta l over matched E: fixed 50/50 {c['fixed_gain']:+.3f} | random 50/50 {r27.fmt(h['matched_gain'])} | "
          f"30/70 {r27.fmt(p['matched_gain'])}")
    print(f"  full - first step:      fixed 50/50 {c['fixed_first_step']:+.3f} | random 50/50 {r27.fmt(h['first_step_delta'])} | "
          f"30/70 {r27.fmt(p['first_step_delta'])}")
    print(f"  curve above E from FPR %: random 50/50 {r27.fmt(h['curve_above_from'], pct=True, digits=1)} | "
          f"30/70 {r27.fmt(p['curve_above_from'], pct=True, digits=1)}")
    print(f"  ensemble FPR %: random 50/50 {r27.fmt(h['matched_fpr'], pct=True, digits=1)} | "
          f"30/70 {r27.fmt(p['matched_fpr'], pct=True, digits=1)}")
    print(f"  positive gain: random 50/50 {h['splits_gain_positive']}/{h['splits']} | 30/70 {p['splits_gain_positive']}/{p['splits']}; "
          f"positive first-step: {h['splits_first_step_positive']}/{h['splits']} | {p['splits_first_step_positive']}/{p['splits']}")
    print(f"  fixed 50/50 above {c['random_half_below_fixed_gain']}/{c['random_half_assignments']} random 50/50 assignments (gain), "
          f"{c['random_half_below_fixed_first_step']}/{c['random_half_assignments']} (first step): favorable assignment: "
          f"gain {c['fixed_assignment_favorable']['gain']}, first step {c['fixed_assignment_favorable']['first_step']}")
    print(f"  representative 30/70 assignment for the example: {r['example']['split']} (gain {r['example']['gain']:+.3f})")
    for k, v in r["checks"].items():
        print(f"R27b {k.split('_', 1)[0]}: {'OK' if v['ok'] else 'FAILED'} (max |diff| {v['max_abs_diff']})")
