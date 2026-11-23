"""R27 (docs/r27_band_split_plan.md): does Hide-and-Seek's own-band result hold under the published 30/70 split?
Sensitivity check. Policy-agnostic, called by seen_unseen_tables.py (XR-1) and rldx_test_table.py (RLDX-1) with
--r27, on unseen tasks only.

Hide-and-Seek's Appendix B.2 fits the band's mean and modulation on 30% of the calibration successes and its width
on the other 70%, without saying how rollouts are assigned. R19 to R25 use SAFE's routine with a 50/50 split (per
task, rollouts sorted by ID alternate). R27 refits the same band on 20 random 30/70 assignments, stratified by task,
and recomputes the three quantities the paper rests on:

  curve       each of the three models' band over the calibration targets, averaged as in Fig. 2, against E's
              lead-FPR curve (R18): the FPR above which the band stays above E
  matched     the three-seed ensemble's band at alpha = 0.05 against E matched to its test false alarms (Delta l)
  first_step  the same ensemble against the band on the first-step retraining, matched to its false alarms
              (r25_band_ablation.paired: the ablation takes its best-lead target within that count)

Everything else is unchanged: scores, folds, SAFE's routine (modulation tfunc, empirical quantile, last-value
padding, alarm at the first score >= the band), E, and the test rollouts.

Gates: K0, the 50/50 ensemble equals R21's (false alarms, Delta l); K1, the 50/50 single-model curves equal R21's
at every target; K2, the 50/50 first-step comparison equals R25's; K3, every 30/70 assignment is a partition of the
calibration successes with 30% of each task in the reference half.

For the paper's opening figure, R27 also saves each failed test rollout's alarm under the 50/50 ensemble band and
under E at alpha = 0.05 (`failure_alarms`), so that an illustrated rollout shows the real alarms of the primary
comparison. Gate K4: these alarms give the ensemble's and E's mean lead.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r21_native_bands as r21  # noqa: E402
import r25_band_ablation as r25  # noqa: E402

PLAN = "docs/r27_band_split_plan.md (R27)"
FULL, FIRST = r25.FULL, "hide_and_seek_first"
HEADS = (FULL, FIRST)
SEEDS = ("0", "1", "2")
ALPHA = r25.ALPHA
TIMER = r25.TIMER
REFERENCE_SHARE = 0.30
SPLITS, SPLIT_SEED = 20, 27
XMAX = 0.15                     # Fig. 2's axis
CURVE_LIMIT = 0.07              # reading: the band stays above E at every observed FPR above this
MIN_SPLITS = 18                 # reading: at least this many of the 20 assignments
TOL = 1e-12


def reference_count(n):
    """Rollouts of a task with n calibration successes in the reference half (at least one in each half)."""
    return min(n - 1, max(1, int(round(REFERENCE_SHARE * n))))


def split_30_70(rows, seed):
    """(reference, nonconformity): per task, a random 30% of the successes (rollouts sorted by ID first)."""
    by_task = defaultdict(list)
    for r in rows:
        if not r["failed"]:
            by_task[r["task_name"]].append(r)
    rng = np.random.default_rng(seed)
    ref, conf = [], []
    for t in sorted(by_task):
        ordered = sorted(by_task[t], key=lambda r: str(r["rollout_id"]))
        perm = rng.permutation(len(ordered))
        k = reference_count(len(ordered))
        ref += [ordered[i] for i in sorted(perm[:k].tolist())]
        conf += [ordered[i] for i in sorted(perm[k:].tolist())]
    return ref, conf


def check_split(rows, ref, conf):
    """Gate K3: a partition of the successes, with the stated share of each task in the reference half."""
    ident = lambda rs: [str(r["rollout_id"]) for r in rs]
    success = sorted(str(r["rollout_id"]) for r in rows if not r["failed"])
    if sorted(ident(ref) + ident(conf)) != success:
        return False
    n, k = defaultdict(int), defaultdict(int)
    for r in ref + conf:
        n[r["task_name"]] += 1
    for r in ref:
        k[r["task_name"]] += 1
    return all(k[t] == reference_count(n[t]) for t in n)


def sorted_curve(points):
    pts = sorted((float(x), float(y)) for x, y in points)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def versus_timer(band, timer, xmax=XMAX, grid=3001):
    """band, timer: [(fpr, lead)]. On their observed overlap up to xmax (linear between sampled points): the
    largest FPR at which the band is not above the timer (None if it is above throughout)."""
    bx, by = sorted_curve(band)
    tx, ty = sorted_curve(timer)
    lo, hi = max(bx[0], tx[0]), min(bx[-1], tx[-1], xmax)
    if not lo < hi:
        return dict(overlap=None, last_not_above=None, above_from=None, share_above=None)
    xs = np.linspace(lo, hi, grid)
    gap = np.interp(xs, bx, by) - np.interp(xs, tx, ty)
    below = xs[gap <= 0]
    last = None if not len(below) else float(below.max())
    return dict(overlap=[float(lo), float(hi)], last_not_above=last, above_from=float(lo) if last is None else last,
                share_above=float(np.mean(gap > 0)), gap_min=float(gap.min()), gap_max=float(gap.max()))


def mean_curve(by_seed):
    """{label: curve}: mean FPR and lead over the seeds at each target, the points Fig. 2 draws."""
    curves = list(by_seed.values())
    return [(float(np.mean([c[j]["fpr"] for c in curves])), float(np.mean([c[j]["mean_lead"] for c in curves])))
            for j in range(len(curves[0]))]


def evaluate(d, rk, timer_curve):
    """The three quantities on one Design. Returns (summary row, single-model curves)."""
    seed_key = lambda s: (lambda F: f"{FULL}~#{s}@{F}")
    ens_key = lambda h: (lambda F: f"{h}~@{F}")
    curves = {f"seed_{s}": r21.curve(d, seed_key(s)) for s in SEEDS}
    cv = versus_timer(mean_curve(curves), timer_curve)
    v = r25.vs_e(d, ens_key(FULL), rk)["standalone"]
    cols = r25.columns(d)
    full = r25.alarm_matrix(d, ens_key(FULL), (ALPHA,))[0]
    abl = r25.alarm_matrix(d, ens_key(FIRST), r25.GRID)
    p = r25.paired(full, abl, np.arange(len(d.test)), cols)
    row = dict(curve=cv, matched=dict(fp=v["fp"], fpr=v["fpr"], mean_lead=v["mean_lead"], gain=v["matched_gain"],
                                      timer_fp=v["matched_timer_fp"]),
               first_step=dict(budget=p["budget"], matched_fp=p["matched_fp"], alpha=p["alpha"], delta=p["delta"]))
    return row, curves


def value_diff(a, b):
    """|a - b|, with two missing values (None or NaN: the control never reached the budget) equal."""
    missing = lambda x: x is None or (isinstance(x, float) and math.isnan(x))
    if missing(a) or missing(b):
        return 0.0 if missing(a) and missing(b) else math.inf
    return abs(float(a) - float(b))


def failure_alarms(d, rk, alpha=ALPHA):
    """One row per failed test rollout: the ensemble band's and E's alarm inference (None: no alarm), on its
    held-out fold, as r21_native_bands.evaluate computes them."""
    band = r25.alarm_matrix(d, lambda F: f"{FULL}~@{F}", (alpha,))[0]
    timer = np.full(len(d.test), math.inf)
    for F in d.folds:
        ri, ci, ti = d.fold(F, d.identity())
        if len(ti) == 0:
            continue
        cu, rc = r21.hk.prepared(d.cal_cohort.take(np.concatenate([ri, d.n_ref + ci])), TIMER, F, None, rk)
        eu, re_ = r21.hk.prepared(d.test_cohort.take(ti), TIMER, F, None, rk)
        timer[ti] = r21.hk.alarm(eu, re_, r21.rx.timer_lam(cu, rc, alpha), math.inf, 0.0)
    out = []
    for i, r in enumerate(d.test):
        if r["failed"]:
            hit = lambda a: None if not a <= r["L"] else int(a)
            out.append(dict(rollout_id=str(r["rollout_id"]), task=r["task_name"], horizon=int(r["horizon"]), L=int(r["L"]),
                            band_alarm=hit(band[i]), timer_alarm=hit(timer[i])))
    return out


def alarm_leads(rows, key):
    """Mean lead over the failures from saved alarm rows (a miss counts as zero)."""
    return float(np.mean([0.0 if r[key] is None else (r["L"] - r[key]) / r["horizon"] for r in rows]))


def spread(values):
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], float)
    if not len(v):
        return dict(n=0, median=None, min=None, max=None)
    return dict(n=int(len(v)), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def summarize(rows):
    gains = [r["matched"]["gain"] for r in rows]
    deltas = [r["first_step"]["delta"] for r in rows]
    above = [r["curve"]["above_from"] for r in rows]
    n_gain = sum(1 for x in gains if x is not None and x > 0)
    n_delta = sum(1 for x in deltas if x is not None and x > 0)
    n_curve = sum(1 for x in above if x is not None and x <= CURVE_LIMIT)
    return dict(
        splits=len(rows), matched_gain=spread(gains), matched_fp=spread([r["matched"]["fp"] for r in rows]),
        matched_fpr=spread([r["matched"]["fpr"] for r in rows]), first_step_delta=spread(deltas),
        curve_above_from=spread(above), curve_share_above=spread([r["curve"]["share_above"] for r in rows]),
        splits_gain_positive=n_gain, splits_first_step_positive=n_delta, splits_curve_above_by_limit=n_curve,
        holds=dict(curve=n_curve >= MIN_SPLITS, matched=n_gain >= MIN_SPLITS, first_step=n_delta >= MIN_SPLITS))


def run(cal, te, cal_tasks, test_tasks, rk, r18, ref21, ref25, splits=SPLITS, seed=SPLIT_SEED, progress=None):
    """r18, ref21, ref25: the saved r18, r21 and r25 blocks of this policy's outputs. Returns (result, problems)."""
    timer_curve = [(p["fpr"], p["mean_lead"]) for p in r18["timers"]["unseen"]["E"]]
    base = r21.Design(cal, te, cal_tasks, test_tasks)
    row, curves = evaluate(base, rk, timer_curve)
    ens = ref21["bootstrap"]["unseen"]["timers"][TIMER][FULL]["standalone"]
    k0 = max(0.0 if row["matched"]["fp"] == ens["fp"] else math.inf, abs(row["matched"]["gain"] - ens["matched_gain"]))
    k1, compared = 0.0, 0
    for lab, cv in curves.items():
        saved = ref21["curves"][FULL]["unseen"][lab]
        if len(saved) != len(cv):
            k1 = math.inf
            continue
        for a, b in zip(cv, saved):
            k1 = max(k1, abs(a["fpr"] - b["fpr"]), abs(a["mean_lead"] - b["mean_lead"]), abs(a["alpha"] - b["alpha"]))
            compared += 1
    sp = ref25["ablations"][FIRST]["paired"]
    fs = row["first_step"]
    k2 = max(value_diff(fs["delta"], sp["delta"]), value_diff(fs["alpha"], sp["alpha"]),
             0.0 if fs["matched_fp"] == sp["matched_fp"] and fs["budget"] == sp["budget"] else math.inf)
    alarms = failure_alarms(base, rk)
    point = r21.evaluate(base, lambda F: f"{FULL}~@{F}", (TIMER,), rk)[TIMER]
    k4 = max(abs(alarm_leads(alarms, "band_alarm") - point["standalone"]["mean_lead"]),
             abs(alarm_leads(alarms, "timer_alarm") - point["timer_alone"]["mean_lead"]))
    rows, k3 = [], True
    for j in range(splits):
        split = lambda rs, j=j: split_30_70(rs, seed + j)
        d = r21.Design(cal, te, cal_tasks, test_tasks, split=split)
        k3 = k3 and check_split(d.cal_rows, d.ref, d.conf)
        one, _ = evaluate(d, rk, timer_curve)
        rows.append(dict(one, split=j))
        if progress:
            progress(f"R27 split {j + 1}/{splits}")
    checks = dict(K0_ensemble_equals_r21=dict(max_abs_diff=k0, ok=k0 <= TOL),
                  K1_curves_equal_r21=dict(max_abs_diff=k1, compared=compared, ok=compared > 0 and k1 <= TOL),
                  K2_first_step_equals_r25=dict(max_abs_diff=k2, ok=k2 <= TOL),
                  K3_splits_are_30_70_partitions=dict(max_abs_diff=0.0 if k3 else math.inf, ok=bool(k3)),
                  K4_failure_alarms_give_the_mean_leads=dict(max_abs_diff=k4, ok=k4 <= TOL))
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return dict(plan=PLAN, alpha=ALPHA, reference_share=REFERENCE_SHARE, split_seed=seed, timer=TIMER,
                reference_50_50=row, splits=rows, summary=summarize(rows), failure_alarms=alarms, checks=checks), problems


def fmt(s, pct=False, digits=3):
    if s["median"] is None:
        return "--"
    k = 100 if pct else 1
    return f"{k * s['median']:.{digits}f} [{k * s['min']:.{digits}f}, {k * s['max']:.{digits}f}]"


def print_summary(r, label):
    b, s = r["reference_50_50"], r["summary"]
    print(f"R27 {label}: Hide-and-Seek's own band, unseen tasks, 50/50 split and {s['splits']} random 30/70 splits "
          "(30/70: median [min, max])")
    print(f"  curve vs E, above from FPR %: 50/50 {100 * b['curve']['above_from']:.1f} | 30/70 "
          f"{fmt(s['curve_above_from'], pct=True, digits=1)} | at most {100 * CURVE_LIMIT:.0f}% in "
          f"{s['splits_curve_above_by_limit']}/{s['splits']} splits")
    print(f"  ensemble false alarms: 50/50 {b['matched']['fp']} ({100 * b['matched']['fpr']:.1f}%) | 30/70 "
          f"{fmt(s['matched_fp'], digits=0)}; FPR % {fmt(s['matched_fpr'], pct=True, digits=1)}")
    print(f"  Delta l over matched E: 50/50 {b['matched']['gain']:+.3f} | 30/70 {fmt(s['matched_gain'])} | positive in "
          f"{s['splits_gain_positive']}/{s['splits']} splits")
    print(f"  full - first step, matched: 50/50 {b['first_step']['delta']:+.3f} | 30/70 {fmt(s['first_step_delta'])} | "
          f"positive in {s['splits_first_step_positive']}/{s['splits']} splits")
    print(f"  reading: curve {'holds' if s['holds']['curve'] else 'DOES NOT HOLD'}; matched gain "
          f"{'holds' if s['holds']['matched'] else 'DOES NOT HOLD'}; first-step control "
          f"{'holds' if s['holds']['first_step'] else 'DOES NOT HOLD'}")
    for k, v in r["checks"].items():
        print(f"R27 {k.split('_', 1)[0]}: {'OK' if v['ok'] else 'FAILED'} (max |diff| {v['max_abs_diff']})")
