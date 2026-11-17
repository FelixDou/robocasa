"""R20 (docs/r20_timer_g_plan.md): the global timer G as a row of Table II. Exploratory.

Table II gives each monitor's gain over the per-task timer P (seen tasks) or the elapsed-time timer E
(unseen tasks; hk_free's G_abs). R20 adds G, the timer told each task's limit H_k (hk_free's G), as one
more row:
  - seen tasks:   G alone against P alone, on Table II's seen cohorts (one fold, "all");
  - unseen tasks: G alone against E alone, on Table II's unseen folds (pooled held-out tests);
at the same 5% target (each timer's own Proposition 1 margin) and at matched false alarms (the
comparator takes the smallest margin with at most G's test false alarms, hk_free.matched_timer, as for
every monitor in Table II). No score is read: every timer alarms on elapsed time only.

Gate T0: each timer alone reproduces R14's (XR-1) or R17's (RLDX-1) `timers_alone` value of the same
timer on the same cohorts (fp exactly, FPR and mean lead within 1e-12). RLDX-1's G on seen tasks has
no hk_free reference; it is compared, for information only, with R16's rounded Table II value.

Called by seen_unseen_tables.py (XR-1) and rldx_test_table.py (RLDX-1) with --r20.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lead as cl  # noqa: E402
import hk_free as hk  # noqa: E402

PLAN = "docs/r20_timer_g_plan.md (R20)"
ALL = "all"
KEY = "r20_no_score"
TOL = 1e-12
ROUNDED_TOL = 5e-5            # R16 stores 4 decimals


def light(rows):
    """Rows reduced to what a timer reads, with an all-zero score so that cl.Cohort can be built."""
    return [dict(task_name=r["task_name"], L=int(r["L"]), horizon=r["horizon"], failed=bool(r["failed"]),
                 **{KEY: np.zeros(int(r["L"]))}) for r in rows]


def seen_parts(cal, te):
    return {ALL: (cl.Cohort(light(cal), KEY), cl.Cohort(light(te), KEY))}


def unseen_parts(cal, te, held, folds):
    lc, lt = light(cal), light(te)
    return {F: (cl.Cohort([r for r in lc if r["task_name"] not in held[F]], KEY),
                cl.Cohort([r for r in lt if r["task_name"] in held[F]], KEY)) for F in folds}


def timer_det(cu, rc):
    """hk_free.fit's timer: Proposition 1's lam on the calibration successes, no score (u = inf)."""
    S = ~cu.failed
    return dict(u=math.inf, s=0.0, lam=cl.conformal(hk.V(hk.sub(cu, S), rc[S], math.inf, 0.0)))


def timer_alone(parts, timer, rk):
    """Per fold, lam on that fold's calibration successes; pooled test outcomes, prepared tests, lams."""
    raw, tests, lams = [], [], {}
    for F, (c, e) in sorted(parts.items()):
        if len(e.L) == 0:
            continue
        cu, rc = hk.prepared(c, timer, F, None, rk)
        eu, re_ = hk.prepared(e, timer, F, None, rk)
        det = timer_det(cu, rc)
        raw.append(hk.raw_eval(eu, re_, det))
        tests.append((eu, re_))
        lams[F] = det["lam"]
    return hk.pool(raw), tests, lams


def versus(parts, timer, comparator, rk):
    """`timer` alone against `comparator` alone on the same cohorts (same failures, same order).
    gain: mean per-failure lead difference at each timer's own margin; matched: the comparator at the
    smallest margin with at most `timer`'s test false alarms (one margin over the pooled folds)."""
    t, _, t_lams = timer_alone(parts, timer, rk)
    c, c_tests, c_lams = timer_alone(parts, comparator, rk)
    if len(t["lead"]) != len(c["lead"]):
        raise ValueError("timer and comparator saw different failures")
    out = hk.summarize(t, c, hk.matched_timer(c_tests, t["fp"]))
    out.update(timer_name=timer, comparator=comparator, lam=dict(timer=t_lams, comparator=c_lams),
               final_inference_alarms=dict(timer=int(t["final"]), comparator=int(c["final"])))
    return out


def alone_values(res):
    """{(side, timer): dict(fp, fpr, mean_lead)} from run()'s result."""
    out = {}
    for side, v in (("seen", res["seen"]), ("unseen", res["unseen"])):
        out[(side, v["timer_name"])] = dict(fp=v["fp"], fpr=v["fpr"], mean_lead=v["mean_lead"])
        out[(side, v["comparator"])] = dict(fp=v["timer"]["fp"], fpr=v["timer"]["fpr"], mean_lead=v["timer"]["mean_lead"])
    return out


def check_t0(res, timers_alone):
    """Gate T0 against a reference `timers_alone` block {side: {timer: {fp, fpr, mean_lead}}}."""
    rows, ok = [], True
    for (side, t), v in sorted(alone_values(res).items()):
        ref = (timers_alone.get(side) or {}).get(t)
        if ref is None:
            rows.append(dict(side=side, timer=t, compared=False))
            continue
        d = max(abs(v["fpr"] - ref["fpr"]), abs(v["mean_lead"] - ref["mean_lead"]))
        good = v["fp"] == ref["fp"] and d <= TOL
        ok &= good
        rows.append(dict(side=side, timer=t, compared=True, fp=v["fp"], ref_fp=ref["fp"], max_abs_diff=d, ok=good))
    compared = [r for r in rows if r["compared"]]
    return dict(ok=bool(ok and compared), compared=len(compared), rows=rows)


def check_rounded(v, ref):
    """Information only: a value against a 4-decimal reference {fpr, mean_lead[, gain_over_P]}."""
    d = max(abs(v["fpr"] - ref["fpr"]), abs(v["mean_lead"] - ref["mean_lead"]))
    if "gain_over_P" in ref:
        d = max(d, abs(v["gain"] - ref["gain_over_P"]))
    return dict(max_abs_diff=d, matches=d <= ROUNDED_TOL)


def run(cal, te, held, folds, rk, reference_timers, rounded_seen_g=None):
    """G against P (seen) and against E (unseen). Returns (result, problems)."""
    res = dict(plan=PLAN,
               seen=versus(seen_parts(cal, te), "G", "P", rk),
               unseen=versus(unseen_parts(cal, te, held, folds), "G", "G_abs", rk))
    t0 = check_t0(res, reference_timers)
    res["checks"] = dict(T0_timers_equal_reference=t0)
    if rounded_seen_g is not None:
        res["checks"]["seen_G_vs_R16_rounded"] = check_rounded(res["seen"], rounded_seen_g)
    problems = [] if t0["ok"] else ["T0"]
    return res, problems


def print_summary(res, label):
    print(f"R20 {label}: timer G as a Table II row (FPR, FP, lead; gain over the comparator, same target | matched)")
    for side, name in (("seen", "P"), ("unseen", "E")):
        v = res[side]
        m = v["matched"]
        mg = f"{m['gain']:+.4f} (comparator at {m['timer_fp']} FP, lead {m['timer_mean_lead']:.4f})" if m else "--"
        print(f"  {side:6s} G {v['fpr']:.4f} {v['fp']:3d} {v['mean_lead']:.4f} | {name} {v['timer']['fpr']:.4f} "
              f"{v['timer']['fp']:3d} {v['timer']['mean_lead']:.4f} | gain {v['gain']:+.4f} | matched {mg}")
    t0 = res["checks"]["T0_timers_equal_reference"]
    print(f"R20 T0: {t0['compared']} timers compared with the reference, "
          f"max |diff| {max([r['max_abs_diff'] for r in t0['rows'] if r['compared']] or [float('nan')]):.2e}: "
          f"{'OK' if t0['ok'] else 'MISMATCH'}")
    for r in t0["rows"]:
        if not r["compared"]:
            print(f"  (no reference for {r['side']} {r['timer']})")
    if "seen_G_vs_R16_rounded" in res["checks"]:
        c = res["checks"]["seen_G_vs_R16_rounded"]
        print(f"  seen G against R16's rounded Table II value (information): max |diff| {c['max_abs_diff']:.1e}, "
              f"{'matches' if c['matches'] else 'differs'}")
