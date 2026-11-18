"""R21 (docs/r21_native_bands_plan.md): SAFE's functional conformal band, the original alarm rule of SAFE
and Hide-and-Seek, against the time-only timers. Exploratory.

Policy-agnostic, called by seen_unseen_tables.py (XR-1) and rldx_test_table.py (RLDX-1) with --r21.
Rows are the scripts' rollout dicts (rollout_id, task_name, L, horizon, failed) carrying each head's
normalized raw score (no running maximum) under keys the caller names per fold.

A Design holds one side of the table: seen tasks (one fold, "all": calibration and test rows of every
task) or unseen tasks (folds F: calibration rows of the other tasks, test rows of F's held-out tasks).
The band is R19's (r19_extras.native_band): SAFE's official upper band, calibrated on the fold's
calibration successes, split per task into reference and nonconformity halves.

  evaluate   per comparator timer, at alpha:
               standalone  the band alone, its gain over the timer at alpha and its matched gain
                           (the timer at the smallest margin with at most the band's test false
                           alarms, hk_free.matched_timer);
               fused       the band at alpha/2 OR the timer at alpha/2 (R19), gain and matched gain;
               timer_alone the timer at alpha.
  curve      the band alone at each calibration target alpha.
  bootstrap  ensembles: tasks with replacement, then calibration successes within task and half (a
             resampled rollout never lands in both halves) and test rollouts within task and class;
             every band and timer refitted, the timer re-matched; percentile 95% CIs.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lead as cl  # noqa: E402
import hk_free as hk  # noqa: E402
import lead_fpr_curves as lfc  # noqa: E402
import r19_extras as rx  # noqa: E402
import review_analyses_b as rab  # noqa: E402

PLAN = "docs/r21_native_bands_plan.md (R21)"
ALPHA = 0.05
ALL = "all"
SEEDS = ("0", "1", "2")
ENSEMBLE = "seed_average"
LABELS = (ENSEMBLE,) + tuple(f"seed_{s}" for s in SEEDS)
CURVE_ALPHAS = tuple(a for a in lfc.ALPHAS if a > 0)          # the band needs 0 < alpha
BOOT_B, BOOT_SEED = 1000, 21
TOL, ROUNDED_TOL = 1e-12, 5e-5
LIGHT = "r21_no_score"
PARTS = ("standalone", "fused")
NAN = float("nan")

_conformal = None


def conformal():
    """robocasa/recovery/safe/conformal.py, loaded once (review_analyses_b loads it by path)."""
    global _conformal
    if _conformal is None:
        _conformal = rab.conformal_module()
    return _conformal


def band_fit(ref, conf, alpha):
    return conformal().calibrate_functional_threshold(ref, conf, alpha=alpha, modulation=rab.BAND_MODULATION,
                                                       alignment=rab.BAND_ALIGNMENT)


def band_alarms(trajectories, fit):
    """Alarm inference (1-based; inf if none) of each trajectory: the first score >= the band."""
    first = conformal().first_detection
    out = np.empty(len(trajectories))
    for i, x in enumerate(trajectories):
        j = first(x, fit)
        out[i] = math.inf if j is None else float(j + 1)
    return out


def cat(parts):
    return np.concatenate(parts) if parts else np.zeros(0, int)


def distinct_count(n):
    """Expected number of distinct rollouts in a bootstrap sample of n, rounded (at least 1)."""
    return max(1, int(round(n * (1 - (1 - 1 / n) ** n))))


def light(r):
    """What a timer reads, with an all-zero score so that a confirm_lead.Cohort can be built."""
    return dict(task_name=r["task_name"], L=int(r["L"]), horizon=r["horizon"], failed=bool(r["failed"]),
                **{LIGHT: np.zeros(int(r["L"]))})


def running_max_gap(rows, raw_key, score_key):
    """Gate R0: max |running maximum of the raw score - the table's score| over rows."""
    worst = 0.0
    for r in rows:
        a, b = np.maximum.accumulate(np.asarray(r[raw_key], float)), np.asarray(r[score_key], float)
        worst = max(worst, math.inf if a.shape != b.shape else float(np.max(np.abs(a - b))))
    return worst


# ---- one side of the table -----------------------------------------------------------------------

class Design:
    """Calibration successes split into halves, test rows, folds, and index maps for the bootstrap."""

    def __init__(self, cal_rows, test_rows, cal_tasks, test_tasks):
        """cal_tasks, test_tasks: {F: set of tasks} (seen tasks: {"all": every task} for both)."""
        self.folds = sorted(test_tasks)
        self.cal_tasks, self.test_tasks = cal_tasks, test_tasks
        self.cal_rows, self.test = list(cal_rows), list(test_rows)
        self.ref, self.conf = rab.split_successes(self.cal_rows)      # R19's split and order
        self.n_ref = len(self.ref)
        task = lambda rows: np.array([r["task_name"] for r in rows])
        self.ref_task, self.conf_task, self.test_task = task(self.ref), task(self.conf), task(self.test)
        self.test_failed = np.array([bool(r["failed"]) for r in self.test])
        self.tasks = sorted(set(self.ref_task) | set(self.conf_task) | set(self.test_task))
        self.ref_by = {t: np.flatnonzero(self.ref_task == t) for t in self.tasks}
        self.conf_by = {t: np.flatnonzero(self.conf_task == t) for t in self.tasks}
        self.test_by = {(t, f): np.flatnonzero((self.test_task == t) & (self.test_failed == f))
                        for t in self.tasks for f in (False, True)}
        self.mask = {F: (np.isin(self.ref_task, sorted(cal_tasks[F])), np.isin(self.conf_task, sorted(cal_tasks[F])),
                         np.isin(self.test_task, sorted(test_tasks[F]))) for F in self.folds}
        self.cal_cohort = cl.Cohort([light(r) for r in self.ref + self.conf], LIGHT)
        self.test_cohort = cl.Cohort([light(r) for r in self.test], LIGHT)

    def identity(self):
        return np.arange(len(self.ref)), np.arange(len(self.conf)), np.arange(len(self.test))

    def draw(self, rng):
        """Tasks with replacement; within each, reference and nonconformity successes within their half
        and test rollouts within class (calibration failures are read by neither band nor timer)."""
        ref, conf, test = [], [], []
        for t in rng.choice(self.tasks, len(self.tasks)):
            for by, pick in ((self.ref_by[t], ref), (self.conf_by[t], conf)):
                if len(by):
                    pick.append(rng.choice(by, len(by)))
            for f in (False, True):
                by = self.test_by[(t, f)]
                if len(by):
                    test.append(rng.choice(by, len(by)))
        return cat(ref), cat(conf), cat(test)

    def draw_test(self, rng):
        """R21b `test`: calibration as it is; tasks with replacement, then test rollouts within class."""
        test = []
        for t in rng.choice(self.tasks, len(self.tasks)):
            for f in (False, True):
                by = self.test_by[(t, f)]
                if len(by):
                    test.append(rng.choice(by, len(by)))
        return np.arange(len(self.ref)), np.arange(len(self.conf)), cat(test)

    def draw_calibration(self, rng, mode):
        """R21b calibration variants (test rows as they are):
          full      tasks with replacement, then successes within task and half with replacement
          rollouts  every task once, successes within task and half with replacement
          distinct  every task once, m of the n successes per task and half without replacement, m the
                    expected number of distinct rollouts in a bootstrap sample of n
          tasks     tasks with replacement, each drawn task's successes as they are"""
        ref, conf = [], []
        tasks = rng.choice(self.tasks, len(self.tasks)) if mode in ("full", "tasks") else self.tasks
        for t in tasks:
            for by, pick in ((self.ref_by[t], ref), (self.conf_by[t], conf)):
                if not len(by):
                    continue
                if mode in ("full", "rollouts"):
                    pick.append(rng.choice(by, len(by)))
                elif mode == "distinct":
                    pick.append(np.sort(rng.choice(by, distinct_count(len(by)), replace=False)))
                elif mode == "tasks":
                    pick.append(by)
                else:
                    raise ValueError(f"unknown calibration mode {mode}")
        return cat(ref), cat(conf), np.arange(len(self.test))

    def fold(self, F, pick):
        ri, ci, ti = pick
        mr, mc, mt = self.mask[F]
        return ri[mr[ri]], ci[mc[ci]], ti[mt[ti]]

    def parts(self):
        """{F: (calibration rows, test rows)}, the form r19_extras.native_band takes."""
        return {F: ([r for r in self.cal_rows if r["task_name"] in self.cal_tasks[F]],
                    [r for r in self.test if r["task_name"] in self.test_tasks[F]]) for F in self.folds}


# ---- evaluation ------------------------------------------------------------------------------------

def matched(tests, fp, lead):
    m = hk.matched_timer(tests, fp)
    if m is None:
        return None
    return dict(timer_fp=m[0]["fp"], gain=float(np.mean(lead - m[0]["lead"])), timer_mean_lead=float(m[0]["lead"].mean()),
                timer_lam=m[1])


def evaluate(d, key_of, timers, rk, alpha=ALPHA, pick=None, check=False):
    """key_of(F): the normalized raw score key of fold F. Returns {timer: result} pooled over folds in
    sorted order; with check, gate N0 per fold (the timer at alpha equals hk_free.fit's)."""
    pick = d.identity() if pick is None else pick
    folds = []
    for F in d.folds:
        ri, ci, ti = d.fold(F, pick)
        if len(ti) == 0:
            continue
        if len(ri) == 0 or len(ci) == 0:
            raise ValueError(f"fold {F}: no reference or nonconformity successes")
        key = key_of(F)
        ref, conf = [d.ref[i][key] for i in ri], [d.conf[i][key] for i in ci]
        te = [d.test[i][key] for i in ti]
        folds.append((F, band_alarms(te, band_fit(ref, conf, alpha)), band_alarms(te, band_fit(ref, conf, alpha / 2)),
                      d.cal_cohort.take(np.concatenate([ri, d.n_ref + ci])), d.test_cohort.take(ti), len(ri), len(ci)))
    out = {}
    for timer in timers:
        a_sa, a_fu, a_tm, cols, tests, info, n0 = [], [], [], [], [], {}, {}
        for F, a_full, a_half, cc, tc, n_ref, n_conf in folds:
            cu, rc = hk.prepared(cc, timer, F, None, rk)
            eu, re_ = hk.prepared(tc, timer, F, None, rk)
            lam_full, lam_half = rx.timer_lam(cu, rc, alpha), rx.timer_lam(cu, rc, alpha / 2)
            if check:
                n0[F] = bool(lam_full == hk.fit(cu, rc)[0]["lam"])
            a_sa.append(a_full)
            a_fu.append(np.minimum(a_half, hk.alarm(eu, re_, lam_half, math.inf, 0.0)))
            a_tm.append(hk.alarm(eu, re_, lam_full, math.inf, 0.0))
            cols.append((eu.L, eu.H, eu.failed))
            tests.append((eu, re_))
            info[F] = dict(reference=n_ref, nonconformity=n_conf, timer_lam=float(lam_full), timer_lam_half=float(lam_half))
        L, H, failed = (np.concatenate([c[i] for c in cols]) for i in range(3))
        tm, l_tm = rx.outcome(np.concatenate(a_tm), L, H, failed)
        res = dict(timer=timer, timer_alone=tm, folds=info)
        for name, a in (("standalone", a_sa), ("fused", a_fu)):
            o, lead = rx.outcome(np.concatenate(a), L, H, failed)
            res[name] = dict(o, gain=float(np.mean(lead - l_tm)), matched=matched(tests, o["fp"], lead))
        if check:
            res["N0_timer_equals_hk_fit"] = n0
        out[timer] = res
    return out


def matched_gain(part):
    return part["matched"]["gain"] if part.get("matched") else NAN


SUMMARY = tuple((p, k) for p in PARTS for k in ("fpr", "mean_lead", "gain", "matched_gain"))


def mean_std(per_seed):
    """Mean and std (ddof 1) over seeds of each summary value."""
    out = {}
    for p, k in SUMMARY:
        v = [matched_gain(r[p]) if k == "matched_gain" else r[p][k] for r in per_seed]
        out[f"{p}_{k}"] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    return out


def curve(d, key_of, alphas=CURVE_ALPHAS):
    """The band alone at each target alpha (test FPR, mean lead, TPR), pooled over folds."""
    parts = []
    for F in d.folds:
        ri, ci, ti = d.fold(F, d.identity())
        if len(ti) == 0:
            continue
        key = key_of(F)
        parts.append(([d.ref[i][key] for i in ri], [d.conf[i][key] for i in ci], [d.test[i][key] for i in ti],
                      d.test_cohort.take(ti)))
    L, H, failed = (np.concatenate([getattr(p[3], k) for p in parts]) for k in ("L", "H", "failed"))
    out = []
    for alpha in alphas:
        a = np.concatenate([band_alarms(te, band_fit(ref, conf, alpha)) for ref, conf, te, _ in parts])
        o, _ = rx.outcome(a, L, H, failed)
        out.append(dict(alpha=alpha, fpr=o["fpr"], mean_lead=o["mean_lead"], tpr=o["tpr"], fp=o["fp"],
                        n_success=o["n_success"], n_failure=o["n_failure"]))
    return out


# ---- bootstrap ------------------------------------------------------------------------------------

def bootstrap(d, key_ofs, timers, rk, B=BOOT_B, seed=BOOT_SEED, contrasts=(), progress=None):
    """key_ofs {head: key_of} (ensembles). Point estimates and percentile 95% CIs of the gain and matched
    gain of the band alone and of the fused rule, per timer; contrasts [(a, b)]: matched gain a - b."""
    point = {h: evaluate(d, k, timers, rk) for h, k in key_ofs.items()}
    rng = np.random.default_rng(seed)
    draws = {t: {h: {p: dict(gain=[], matched=[]) for p in PARTS} for h in key_ofs} for t in timers}
    for b in range(B):
        pick = d.draw(rng)
        for h, k in key_ofs.items():
            try:
                r = evaluate(d, k, timers, rk, pick=pick)
            except (ValueError, ZeroDivisionError):
                r = None
            for t in timers:
                for p in PARTS:
                    v = None if r is None else r[t][p]
                    draws[t][h][p]["gain"].append(NAN if v is None else v["gain"])
                    draws[t][h][p]["matched"].append(NAN if v is None else matched_gain(v))
        if progress and (b + 1) % 100 == 0:
            progress(b + 1, B)
    pct = lambda v: [float(x) for x in np.nanpercentile(np.asarray(v, float), [2.5, 97.5])]
    out = dict(samples=B, seed=seed, timers={})
    for t in timers:
        row = {}
        for h in key_ofs:
            row[h] = {}
            for p in PARTS:
                pt, dr = point[h][t][p], draws[t][h][p]
                row[h][p] = dict(fpr=pt["fpr"], fp=pt["fp"], gain=pt["gain"], gain_ci95=pct(dr["gain"]),
                                 matched_gain=matched_gain(pt), matched_ci95=pct(dr["matched"]),
                                 failed_replicates=int(np.sum(np.isnan(dr["matched"]))))
        for a, b in contrasts:
            if a in key_ofs and b in key_ofs:
                row[f"{a}_minus_{b}"] = {p: dict(
                    matched_gain=row[a][p]["matched_gain"] - row[b][p]["matched_gain"],
                    matched_ci95=pct(np.asarray(draws[t][a][p]["matched"]) - np.asarray(draws[t][b][p]["matched"])))
                    for p in PARTS}
        out["timers"][t] = row
    return out, point


# ---- one policy -----------------------------------------------------------------------------------

def same(mine, ref, tol, parts=(("standalone", ("fpr", "mean_lead", "tpr")),
                                ("fused", ("fpr", "mean_lead", "tpr", "gain")),
                                ("timer_alone", ("fpr", "mean_lead", "tpr")))):
    """Max |diff| of the listed values, inf if a false-alarm count differs; and the matched gains."""
    worst = 0.0
    for p, keys in parts:
        if p not in ref:
            continue
        if mine[p]["fp"] != ref[p]["fp"]:
            return math.inf
        for k in keys:
            if k in ref[p]:
                worst = max(worst, abs(mine[p][k] - ref[p][k]))
    return worst


def check_live(res_block, native, tol=TOL):
    """Gate B0: R21's evaluation equals r19_extras.native_band run on the same rows (all values,
    matched gain and the matched timer's false alarms included)."""
    d = same(res_block, native, tol)
    mf, nf = res_block["fused"]["matched"], native["fused"]["matched"]
    if (mf is None) != (nf is None):
        return math.inf
    if mf is not None:
        d = max(d, abs(mf["gain"] - nf["gain"]), 0.0 if mf["timer_fp"] == nf["timer_fp"] else math.inf)
    return d


def run(sides, heads, rk, B=BOOT_B, seed=BOOT_SEED, contrasts=(), r0=None, live=True, progress=None):
    """sides {side: dict(design=Design, timers=tuple)}; heads {head: {side: {label: key_of}}}, labels from
    LABELS (seed_average: the ensemble). r0: {name: max |diff|} from running_max_gap (gate R0).
    Returns (result, problems); N1 and N2 are checked by the callers (check_r19, check_r12)."""
    res = dict(plan=PLAN, alpha=ALPHA, curve_alphas=list(CURVE_ALPHAS), heads={}, curves={}, bootstrap={})
    n0, b0, c0 = {}, {}, {}
    for h, by_side in heads.items():
        res["heads"][h], res["curves"][h] = {}, {}
        for side, keys in by_side.items():
            d, timers = sides[side]["design"], sides[side]["timers"]
            ev = {lab: evaluate(d, keys[lab], timers, rk, check=True) for lab in keys}
            block = {}
            for t in timers:
                block[t] = {lab: ev[lab][t] for lab in keys}
                seeds = [block[t][f"seed_{s}"] for s in SEEDS if f"seed_{s}" in keys]
                if len(seeds) > 1:
                    block[t]["mean_std_over_seeds"] = mean_std(seeds)
                for lab in keys:
                    n0[f"{h}/{side}/{t}/{lab}"] = all(ev[lab][t]["N0_timer_equals_hk_fit"].values())
                    if live:
                        nb = rx.native_band(d.parts(), keys[lab], keys[lab], t, rk)
                        b0[f"{h}/{side}/{t}/{lab}"] = check_live(ev[lab][t], nb)
            res["heads"][h][side] = block
            cv = {lab: curve(d, keys[lab]) for lab in keys}
            seeds = [cv[f"seed_{s}"] for s in SEEDS if f"seed_{s}" in keys]
            if len(seeds) > 1:
                cv["mean_std_over_seeds"] = lfc.mean_std(seeds)
            res["curves"][h][side] = cv
            for lab in keys:
                p, s = lfc.at_alpha(cv[lab], ALPHA), ev[lab][timers[0]]["standalone"]
                c0[f"{h}/{side}/{lab}"] = max(abs(p["fpr"] - s["fpr"]), abs(p["mean_lead"] - s["mean_lead"]),
                                              0.0 if p["fp"] == s["fp"] else math.inf)
    for side, spec in sides.items():
        key_ofs = {h: by[side][ENSEMBLE] for h, by in heads.items() if side in by and ENSEMBLE in by[side]}
        if not key_ofs or B <= 0:
            continue
        say = None if progress is None else (lambda b, n, side=side: progress(f"R21 bootstrap {side}: {b}/{n}"))
        boot, point = bootstrap(spec["design"], key_ofs, spec["timers"], rk, B=B, seed=seed, contrasts=contrasts,
                                progress=say)
        res["bootstrap"][side] = boot
        for h in key_ofs:
            for t in spec["timers"]:
                b0[f"{h}/{side}/{t}/bootstrap_point"] = check_live(point[h][t], res["heads"][h][side][t][ENSEMBLE])
    worst = lambda v: max(v.values()) if v else 0.0
    checks = dict(R0_raw_running_max_equals_score=dict(max_abs_diff=worst(r0 or {}), by=r0 or {},
                                                       ok=bool(r0) and worst(r0) <= TOL),
                  N0_timer_equals_hk_fit=dict(ok=all(n0.values()), failing=[k for k, v in n0.items() if not v]),
                  B0_equals_r19_native_band=dict(max_abs_diff=worst(b0), compared=len(b0), ok=bool(b0) and worst(b0) <= TOL),
                  C0_curve_at_alpha_equals_standalone=dict(max_abs_diff=worst(c0), ok=worst(c0) <= TOL))
    res["checks"] = checks
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return res, problems


def check_r19(res, native_bands, tol=TOL):
    """Gate N1: the ensemble values equal R19's native_bands {head: {seen: band (P), unseen: {timer: band}}}."""
    rows, ok = [], True
    for h, b in native_bands.items():
        by_side = {"unseen": b.get("unseen") or {}}
        if isinstance(b.get("seen"), dict):
            by_side["seen"] = {"P": b["seen"]}
        for side, by in by_side.items():
            for t, ref in by.items():
                mine = (((res["heads"].get(h) or {}).get(side) or {}).get(t) or {}).get(ENSEMBLE)
                if mine is None:
                    rows.append(dict(head=h, side=side, timer=t, found=False))
                    ok = False
                    continue
                d = check_live(mine, ref)
                ok &= d <= tol
                rows.append(dict(head=h, side=side, timer=t, found=True, max_abs_diff=d, ok=d <= tol))
    return dict(ok=bool(ok and rows), compared=len(rows), rows=rows)


R12_NAMES = {"safe": "safe_mlp", "hide_and_seek": "hide_and_seek"}


def check_r12(res, list_b, names=R12_NAMES):
    """Gate N2: XR-1's seen-task ensemble values equal R12 B2's at its 4 decimals (FP exact). The matched
    gain is information only: R12 matched P on a margin grid."""
    rows, ok = [], True
    for theirs, mine_name in names.items():
        ref = list_b["B2"][theirs]
        mine = res["heads"][mine_name]["seen"]["P"][ENSEMBLE]
        sa, fu = ref["standalone"], ref["fused_bonferroni"]
        d = same(mine, dict(standalone=sa, fused=fu), ROUNDED_TOL)
        good = d <= ROUNDED_TOL
        ok &= good
        mg = mine["fused"]["matched"]
        rows.append(dict(head=mine_name, max_abs_diff=d, ok=good,
                         matched_information=dict(r21=None if mg is None else mg["gain"], r21_timer_fp=None if mg is None else mg["timer_fp"],
                                                  r12=(fu.get("matched") or {}).get("gain"),
                                                  r12_timer_fp=(fu.get("matched") or {}).get("fp"))))
    return dict(ok=bool(ok and rows), rows=rows)


# ---- R21b: where the lopsided CIs come from ----------------------------------------------------------

PLAN_B = "docs/r21b_band_ci_diagnostic_plan.md (R21b)"
VARIANTS = ("full", "test", "calibration", "calibration_rollouts", "calibration_distinct", "calibration_tasks")
CAL_MODES = dict(calibration="full", calibration_rollouts="rollouts", calibration_distinct="distinct",
                 calibration_tasks="tasks")
STATS = ("standalone_gain", "standalone_matched", "fused_gain", "fused_matched", "standalone_fp", "fused_fp",
         "timer_fp", "standalone_matched_timer_fp", "fused_matched_timer_fp")
GAINS = ("standalone_gain", "standalone_matched", "fused_gain", "fused_matched")
SHIFT = 0.02


def replicate_values(r):
    """The saved statistics of one evaluate() result for one timer."""
    sa, fu = r["standalone"], r["fused"]
    mfp = lambda p: p["matched"]["timer_fp"] if p.get("matched") else NAN
    return dict(standalone_gain=sa["gain"], standalone_matched=matched_gain(sa), fused_gain=fu["gain"],
                fused_matched=matched_gain(fu), standalone_fp=sa["fp"], fused_fp=fu["fp"], timer_fp=r["timer_alone"]["fp"],
                standalone_matched_timer_fp=mfp(sa), fused_matched_timer_fp=mfp(fu))


def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return NAN
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(rank(x[m]), rank(y[m]))[0, 1])


def describe(point, v):
    ok = v[~np.isnan(v)]
    if not len(ok):
        return dict(point=point, replicates=0)
    p = np.nanpercentile(v, [2.5, 97.5])
    return dict(point=point, replicates=int(len(ok)), failed=int(len(v) - len(ok)), mean=float(ok.mean()),
                median=float(np.median(ok)), ci95=[float(p[0]), float(p[1])], below_zero=float(np.mean(ok < 0)),
                shift=None if point is None or isinstance(point, float) and math.isnan(point) else float(point - np.median(ok)))


def draw_variant(d, rng, variant):
    if variant == "full":
        return d.draw(rng)
    if variant == "test":
        return d.draw_test(rng)
    return d.draw_calibration(rng, CAL_MODES[variant])


def diagnostic(d, key_ofs, timers, rk, B=BOOT_B, seed=BOOT_SEED, variants=VARIANTS, progress=None):
    """Per variant, B replicates (one generator per variant, seeded alike) of every head's evaluation.
    Returns (summary {variant: {timer: {head: {stat: describe}}}}, replicates {same: {stat: array}}, point)."""
    point = {h: evaluate(d, k, timers, rk) for h, k in key_ofs.items()}
    reps = {v: {t: {h: {s: np.full(B, NAN) for s in STATS} for h in key_ofs} for t in timers} for v in variants}
    for v in variants:
        rng = np.random.default_rng(seed)
        for b in range(B):
            pick = draw_variant(d, rng, v)
            for h, k in key_ofs.items():
                try:
                    r = evaluate(d, k, timers, rk, pick=pick)
                except (ValueError, ZeroDivisionError):
                    continue
                for t in timers:
                    for s, x in replicate_values(r[t]).items():
                        reps[v][t][h][s][b] = x
            if progress and (b + 1) % 250 == 0:
                progress(v, b + 1, B)
    summary = {}
    for v in variants:
        summary[v] = {}
        for t in timers:
            summary[v][t] = {}
            for h in key_ofs:
                pv = replicate_values(point[h][t])
                arr = reps[v][t][h]
                row = {s: describe(pv[s], arr[s]) for s in STATS}
                row["spearman_fp_vs_matched"] = dict(standalone=spearman(arr["standalone_fp"], arr["standalone_matched"]),
                                                     fused=spearman(arr["fused_fp"], arr["fused_matched"]))
                summary[v][t][h] = row
    return summary, reps, point


def check_d0(summary_full, reference, B, tol=TOL):
    """Gate D0: variant `full` reproduces R21's bootstrap block {timers: {t: {h: {part: ...}}}, samples}."""
    rows, ok = [], reference is not None and reference.get("samples") == B
    for t, by_h in (reference or {}).get("timers", {}).items():
        for h, parts in by_h.items():
            if "_minus_" in h:
                continue
            mine = (summary_full.get(t) or {}).get(h)
            if mine is None:
                rows.append(dict(timer=t, head=h, found=False))
                ok = False
                continue
            worst = 0.0
            for p in PARTS:
                ref = parts[p]
                g, m = mine[f"{p}_gain"], mine[f"{p}_matched"]
                pairs = [(g["point"], ref["gain"]), (m["point"], ref["matched_gain"])]
                pairs += list(zip(g.get("ci95", [NAN, NAN]), ref["gain_ci95"]))
                pairs += list(zip(m.get("ci95", [NAN, NAN]), ref["matched_ci95"]))
                for a, b in pairs:
                    worst = max(worst, 0.0 if (a is None and b is None) else abs(float(a) - float(b)))
            ok &= worst <= tol
            rows.append(dict(timer=t, head=h, found=True, max_abs_diff=worst, ok=worst <= tol))
    return dict(ok=bool(ok and rows), compared=len(rows), samples=dict(r21=(reference or {}).get("samples"), r21b=B),
                rows=rows)


def diagnose(sides, heads, rk, B=BOOT_B, seed=BOOT_SEED, reference=None, variants=VARIANTS, progress=None):
    """R21b on one policy: the diagnostic on every side's ensembles. reference: R21's `bootstrap` block
    {side: ...} for gate D0. Returns (result, replicate arrays {name: array}, problems)."""
    res = dict(plan=PLAN_B, variants=list(variants), samples=B, seed=seed, sides={}, checks={})
    arrays, d0 = {}, {}
    for side, spec in sides.items():
        key_ofs = {h: by[side][ENSEMBLE] for h, by in heads.items() if side in by and ENSEMBLE in by[side]}
        say = None if progress is None else (lambda v, b, n, side=side: progress(f"R21b {side} {v}: {b}/{n}"))
        summary, reps, _ = diagnostic(spec["design"], key_ofs, spec["timers"], rk, B, seed, variants, say)
        res["sides"][side] = summary
        for v, by_t in reps.items():
            for t, by_h in by_t.items():
                for h, by_s in by_h.items():
                    for s, a in by_s.items():
                        arrays[f"{side}/{v}/{t}/{h}/{s}"] = a
        if "full" in summary:
            d0[side] = check_d0(summary["full"], (reference or {}).get(side), B)
    res["checks"]["D0_full_equals_r21_bootstrap"] = dict(ok=bool(d0) and all(v["ok"] for v in d0.values()), by_side=d0)
    problems = [] if res["checks"]["D0_full_equals_r21_bootstrap"]["ok"] else ["D0"]
    return res, arrays, problems


def save_replicates(path, arrays):
    """The replicate arrays as one compressed .npz (keys: side__variant__timer__head__stat)."""
    np.savez_compressed(path, **{k.replace("/", "__"): v for k, v in arrays.items()})


def run_diagnostic(args, sides, spec, rk, output, reference_path):
    """Shared by the two hooks: R21b after R21, with R21's output as the D0 reference."""
    ref = json.loads(Path(reference_path).read_text())
    boot = (ref.get("r21") or {}).get("bootstrap")
    if not boot:
        raise SystemExit("error: --r21b-reference must be an R21 output with its bootstrap (r21.bootstrap)")
    res, arrays, problems = diagnose(sides, spec, rk, B=args.r21b_boot, reference=boot,
                                     progress=lambda m: print(m, flush=True))
    npz = Path(output).with_name(Path(output).stem + "_r21b_replicates.npz")
    save_replicates(npz, arrays)
    res["inputs"] = dict(r21=dict(path=str(Path(reference_path).resolve()), sha256=sha256(reference_path)))
    res["replicates_file"] = npz.name
    return res, problems


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def print_diagnostic(res, label, head="hide_and_seek"):
    print(f"R21b {label}: {head}'s matched gains per resampling variant: point | median [2.5%, 97.5%] | share < 0 | "
          f"shift (point - median) | rule false alarms point/median")
    for side, by_v in res["sides"].items():
        timers = next(iter(by_v.values())).keys()
        for t in timers:
            for part in PARTS:
                print(f"  {side} {t} {'band alone' if part == 'standalone' else 'band OR timer'}:")
                for v, by_t in by_v.items():
                    row = by_t[t].get(head)
                    if row is None:
                        continue
                    m, fp = row[f"{part}_matched"], row[f"{part}_fp"]
                    if not m.get("replicates"):
                        print(f"    {v:22s} no replicate")
                        continue
                    shift = NAN if m["shift"] is None else m["shift"]
                    flag = " SHIFT" if shift > SHIFT else ""
                    print(f"    {v:22s} {m['point']:+.3f} | {m['median']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}] | "
                          f"{m['below_zero']:.2f} | {shift:+.3f}{flag} | {fp['point']:.0f}/{fp['median']:.0f} "
                          f"(Spearman {row['spearman_fp_vs_matched'][part]:+.2f})")
    d0 = res["checks"]["D0_full_equals_r21_bootstrap"]
    print(f"R21b D0: {'OK' if d0['ok'] else 'FAILED'} "
          + "; ".join(f"{s}: {v['compared']} compared, samples {v['samples']}" for s, v in d0["by_side"].items()))


# ---- printing -------------------------------------------------------------------------------------

def print_summary(res, label):
    f = lambda x: "  nan " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}"
    print(f"R21 {label}: each head's own band (SAFE's functional conformal band) against the timers")
    print("  ensemble: band alone FPR, lead | alone vs timer: gain, matched (timer FP) | band/2 OR timer/2: FPR, gain, matched")
    for h, by_side in res["heads"].items():
        for side, by_t in by_side.items():
            for t, v in by_t.items():
                e = v[ENSEMBLE]
                sa, fu = e["standalone"], e["fused"]
                print(f"  {h:15s} {side:6s} {t:5s} {sa['fpr']:.3f} {sa['mean_lead']:.3f} | {f(sa['gain'])} "
                      f"{f(matched_gain(sa))} ({(sa['matched'] or {}).get('timer_fp')}) | {fu['fpr']:.3f} {f(fu['gain'])} "
                      f"{f(matched_gain(fu))}")
                m = v.get("mean_std_over_seeds")
                if m:
                    print(f"  {'':15s} {'':6s} {'':5s} single models: alone vs timer matched "
                          f"{m['standalone_matched_gain']['mean']:+.3f}+-{m['standalone_matched_gain']['std']:.3f}; "
                          f"OR timer FPR {m['fused_fpr']['mean']:.3f}, matched {m['fused_matched_gain']['mean']:+.3f}"
                          f"+-{m['fused_matched_gain']['std']:.3f}")
    for side, boot in res["bootstrap"].items():
        print(f"R21 bootstrap, {side} tasks ({boot['samples']} replicates): matched gain [95% CI], alone | OR timer")
        for t, row in boot["timers"].items():
            for h, v in row.items():
                cell = lambda p: (f"{f(v[p]['matched_gain'])} [{v[p]['matched_ci95'][0]:+.3f}, "
                                  f"{v[p]['matched_ci95'][1]:+.3f}]")
                print(f"  {t:5s} {h:36s} {cell('standalone')} | {cell('fused')}")
    for name, c in res["checks"].items():
        print(f"R21 {name.split('_', 1)[0]}: {'OK' if c['ok'] else 'FAILED'} "
              f"{ {k: v for k, v in c.items() if k in ('max_abs_diff', 'compared', 'failing')} }")
