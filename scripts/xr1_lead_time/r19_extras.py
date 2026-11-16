"""R19 (docs/r19_triage_b_plan.md): the draft v3 mock review's offline analyses (triage B). Exploratory.

Policy-agnostic pieces, called by seen_unseen_tables.py (XR-1) and rldx_test_table.py (RLDX-1) with
--r19. Rows are the scripts' rollout dicts (task_name, L, horizon, failed, rollout_id) carrying score
keys; "parts" maps a fold to (calibration rows, test rows), with one fold (ALL) on seen tasks.

  native_band     each monitor with SAFE's official functional conformal upper band
                  (robocasa/recovery/safe/conformal.py, as R12 B2) on its normalized raw score (no
                  running maximum), calibrated per fold on that fold's calibration successes (split per
                  task into reference and nonconformity halves). Standalone at alpha; fused with a
                  timer under a Bonferroni split (band and timer each at alpha/2); gains over the timer
                  at alpha, at the same target and matched (hk_free.matched_timer).
  boot_unseen     full-pipeline bootstrap of the unseen-task gains (hk_free.run_folds): each replicate
                  draws tasks with replacement, then calibration and test rollouts within task and
                  class, refits every fold and re-matches the timer on the pooled resampled test set.
  leads_detail    per-failure leads of a timer, the fused monitor and the matched timer, for the
                  timers' alarms at the limit, lead in seconds by task type and per-task gains.
  saturation      ties among a score's calibration maxima (the zero-input model's plateau).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lead as cl  # noqa: E402
import hk_free as hk  # noqa: E402
import review_analyses_b as rab  # noqa: E402

PLAN = "docs/r19_triage_b_plan.md (R19)"
ALPHA = 0.05
SEEDS = ("0", "1", "2")
BOOT_B, BOOT_SEED = 1000, 19


# ---- shared ------------------------------------------------------------------------------------

def outcome(a, L, H, failed):
    """Pooled metrics of alarm indices a (inf = no alarm); an alarm at the final inference counts, lead 0."""
    hit = a <= L
    lead = np.where(hit, (L - a) / H, 0.0)[failed]
    fp, n = int(hit[~failed].sum()), int((~failed).sum())
    at_limit = int(np.sum(hit[failed] & (a[failed] == L[failed])))
    tpr = float(hit[failed].mean())
    return dict(fpr=fp / n, fp=fp, n_success=n, n_failure=int(failed.sum()), tpr=tpr, mean_lead=float(lead.mean()),
                bacc=(tpr + 1 - fp / n) / 2, alarms_at_limit=at_limit), lead


def prepared(rows, key, timer, F, rk):
    return hk.prepared(cl.Cohort(rows, key), timer, F, None, rk)


def timer_lam(cu, rc, alpha):
    S, rS = hk.sub(cu, ~cu.failed), rc[~cu.failed]
    return cl.conformal(hk.V(S, rS, math.inf, 0.0), alpha)


# ---- native functional bands -------------------------------------------------------------------

def band_fit(cal_rows, raw_key, alpha):
    ref, conf = rab.split_successes(cal_rows)
    fit = rab.conformal_module().calibrate_functional_threshold(
        [np.asarray(r[raw_key], float) for r in ref], [np.asarray(r[raw_key], float) for r in conf], alpha=alpha,
        modulation=rab.BAND_MODULATION, alignment=rab.BAND_ALIGNMENT)
    return fit, len(ref), len(conf)


def band_alarms(rows, raw_key, fit):
    first = rab.conformal_module().first_detection
    out = []
    for r in rows:
        i = first(np.asarray(r[raw_key], float), fit)
        out.append(math.inf if i is None else float(i + 1))
    return np.asarray(out)


def native_band(parts, raw_key_of, key_of, timer, rk, alpha=ALPHA):
    """parts {F: (calibration rows, test rows)}; raw_key_of(F) the normalized raw score key, key_of(F) any
    score key of the rows (only the timer reads the cohort). Pooled over folds in sorted order."""
    a_sa, a_fu, a_tm, cols, tests, info, checks = [], [], [], [], [], {}, {}
    for F, (cal, te) in sorted(parts.items()):
        if not te:
            continue
        raw = raw_key_of(F)
        b_full, n_ref, n_conf = band_fit(cal, raw, alpha)
        b_half, _, _ = band_fit(cal, raw, alpha / 2)
        cu, rc = prepared(cal, key_of(F), timer, F, rk)
        eu, re_ = prepared(te, key_of(F), timer, F, rk)
        lam_full, lam_half = timer_lam(cu, rc, alpha), timer_lam(cu, rc, alpha / 2)
        checks[F] = dict(timer_equals_hk_fit=bool(lam_full == hk.fit(cu, rc)[0]["lam"]))
        a_sa.append(band_alarms(te, raw, b_full))
        a_fu.append(np.minimum(band_alarms(te, raw, b_half), hk.alarm(eu, re_, lam_half, math.inf, 0.0)))
        a_tm.append(hk.alarm(eu, re_, lam_full, math.inf, 0.0))
        cols.append((eu.L, eu.H, eu.failed))
        tests.append((eu, re_))
        info[F] = dict(reference=n_ref, nonconformity=n_conf, timer_lam=float(lam_full), timer_lam_half=float(lam_half))
    L, H, failed = (np.concatenate([c[i] for c in cols]) for i in range(3))
    sa, _ = outcome(np.concatenate(a_sa), L, H, failed)
    fu, l_fu = outcome(np.concatenate(a_fu), L, H, failed)
    tm, l_tm = outcome(np.concatenate(a_tm), L, H, failed)
    m = hk.matched_timer(tests, fu["fp"])
    matched = None if m is None else dict(timer_fp=m[0]["fp"], gain=float(np.mean(l_fu - m[0]["lead"])))
    return dict(timer=timer, standalone=sa, fused=dict(fu, gain=float(np.mean(l_fu - l_tm)), matched=matched),
                timer_alone=tm, folds=info, checks=dict(folds=checks, ok=all(v["timer_equals_hk_fit"] for v in checks.values())))


# ---- full-pipeline bootstrap of the unseen-task gains ------------------------------------------------

class UnseenDesign:
    """Base cohorts per (score key, fold) and the index maps the bootstrap draws through."""

    def __init__(self, cal_rows, test_rows, held, folds):
        self.cal, self.test, self.held, self.folds = cal_rows, test_rows, held, list(folds)
        self.tasks = sorted({r["task_name"] for r in test_rows})
        self.cal_by = {(t, f): [i for i, r in enumerate(cal_rows) if r["task_name"] == t and bool(r["failed"]) == f]
                       for t in self.tasks for f in (False, True)}
        self.te_by = {(t, f): [i for i, r in enumerate(test_rows) if r["task_name"] == t and bool(r["failed"]) == f]
                      for t in self.tasks for f in (False, True)}
        self.cal_idx = {F: [i for i, r in enumerate(cal_rows) if r["task_name"] not in held[F]] for F in self.folds}
        self.te_idx = {F: [i for i, r in enumerate(test_rows) if r["task_name"] in held[F]] for F in self.folds}
        self.cal_pos = {F: {g: j for j, g in enumerate(self.cal_idx[F])} for F in self.folds}
        self.te_pos = {F: {g: j for j, g in enumerate(self.te_idx[F])} for F in self.folds}
        self._cohorts = {}

    def cohorts(self, key_of):
        name = key_of(self.folds[0])
        if name not in self._cohorts:
            self._cohorts[name] = {F: (cl.Cohort([self.cal[i] for i in self.cal_idx[F]], key_of(F)),
                                       cl.Cohort([self.test[i] for i in self.te_idx[F]], key_of(F))) for F in self.folds}
        return self._cohorts[name]

    def pairs(self, key_of, cal_pick=None, te_pick=None):
        """{F: (cal, test)} cohorts; with picks (global row indices, repeats allowed), the resampled ones."""
        base = self.cohorts(key_of)
        if cal_pick is None:
            return base
        out = {}
        for F in self.folds:
            ci = np.array([self.cal_pos[F][g] for g in cal_pick if g in self.cal_pos[F]], int)
            ti = np.array([self.te_pos[F][g] for g in te_pick if g in self.te_pos[F]], int)
            if len(ti):
                out[F] = (base[F][0].take(ci), base[F][1].take(ti))
        return out

    def draw(self, rng):
        cal_pick, te_pick = [], []
        for t in rng.choice(self.tasks, len(self.tasks)):
            for f in (False, True):
                for by, pick in ((self.cal_by, cal_pick), (self.te_by, te_pick)):
                    idx = by[(t, f)]
                    if idx:
                        pick += rng.choice(idx, len(idx)).tolist()
        return cal_pick, te_pick


def boot_unseen(design, key_ofs, timers, rk, B=BOOT_B, seed=BOOT_SEED, contrasts=()):
    """key_ofs {name: key_of}; timers e.g. ("G_abs", "G"). Point estimates (no resampling) and percentile
    95% CIs of each gain (same target, matched); contrasts [(a, b)] give CIs of matched gain a - b."""
    point = {t: {n: hk.run_folds(design.pairs(k), t, None, rk) for n, k in key_ofs.items()} for t in timers}
    rng = np.random.default_rng(seed)
    same = {t: {n: [] for n in key_ofs} for t in timers}
    matched = {t: {n: [] for n in key_ofs} for t in timers}
    for _ in range(B):
        cal_pick, te_pick = design.draw(rng)
        for t in timers:
            for n, k in key_ofs.items():
                r = hk.run_folds(design.pairs(k, cal_pick, te_pick), t, None, rk)
                same[t][n].append(r["gain"])
                matched[t][n].append(r["matched"]["gain"] if r["matched"] else float("nan"))
    pct = lambda v: [float(x) for x in np.nanpercentile(np.asarray(v, float), [2.5, 97.5])]
    out = dict(samples=B, seed=seed, timers={})
    for t in timers:
        row = {}
        for n in key_ofs:
            p = point[t][n]
            row[n] = dict(gain=p["gain"], gain_ci95=pct(same[t][n]),
                          matched_gain=p["matched"]["gain"] if p["matched"] else None, matched_ci95=pct(matched[t][n]),
                          matched_failed_replicates=int(np.sum(np.isnan(matched[t][n]))), fp=p["fp"],
                          timer_fp=p["timer"]["fp"])
        for a, b in contrasts:
            row[f"{a}_minus_{b}"] = dict(
                matched_gain=(row[a]["matched_gain"] - row[b]["matched_gain"])
                if row[a]["matched_gain"] is not None and row[b]["matched_gain"] is not None else None,
                matched_ci95=pct(np.asarray(matched[t][a]) - np.asarray(matched[t][b])))
        out["timers"][t] = row
    return out, point


def check_point(point, reference, tol=1e-12):
    """Gate B0: the un-resampled gains equal an earlier run's hk_free.block seed averages (R14, R17).
    reference: {timer: {name: seed_average dict}}."""
    worst, missing = 0.0, []
    for t, by in point.items():
        for n, p in by.items():
            ref = (reference.get(t) or {}).get(n)
            if ref is None:
                missing.append(f"{t}/{n}")
                continue
            worst = max(worst, abs(p["gain"] - ref["gain"]), abs(p["fpr"] - ref["fpr"]),
                        abs(p["mean_lead"] - ref["mean_lead"]))
            if (p["matched"] is None) != (ref.get("matched") is None):
                worst = math.inf
            elif p["matched"] is not None:
                worst = max(worst, abs(p["matched"]["gain"] - ref["matched"]["gain"]))
    return dict(max_abs_diff=worst, missing=missing, ok=not missing and worst <= tol)


# ---- leads per failure: timers at the limit, seconds, per task ---------------------------------------

def leads_detail(parts, timer, rk):
    """hk_free.run_folds on {F: (cal Cohort, test Cohort)}, keeping per-failure arrays in pooled fold order."""
    task, H, lt, lf, tests, fp, tm_parts = [], [], [], [], [], 0, []
    for F, (c, e) in sorted(parts.items()):
        if len(e.L) == 0:
            continue
        cu, rc = hk.prepared(c, timer, F, None, rk)
        eu, re_ = hk.prepared(e, timer, F, None, rk)
        t_fit, f_fit = hk.fit(cu, rc)
        pt, pf = hk.raw_eval(eu, re_, t_fit), hk.raw_eval(eu, re_, f_fit)
        task.append(eu.task[eu.failed]); H.append(eu.H[eu.failed]); lt.append(pt["lead"]); lf.append(pf["lead"])
        tests.append((eu, re_)); fp += pf["fp"]; tm_parts.append(pt)
    m = hk.matched_timer(tests, fp)
    pooled_timer = hk.pool(tm_parts)
    return dict(task=np.concatenate(task), H=np.concatenate(H), timer=np.concatenate(lt), fused=np.concatenate(lf),
                matched=None if m is None else m[0]["lead"], timer_pooled=pooled_timer)


def timer_at_limit(detail):
    """The timer alone: failures alarmed only at the final inference (lead 0) and bACC if those were misses."""
    p = detail["timer_pooled"]
    n_fail, hits = len(p["lead"]), int(p["hit"].sum())
    fpr = p["fp"] / p["n"]
    strict = (hits - p["final"]) / n_fail
    return dict(failures=n_fail, detected=hits, at_limit=int(p["final"]), share_at_limit=p["final"] / n_fail,
                fpr=fpr, bacc=(hits / n_fail + 1 - fpr) / 2, bacc_limit_as_miss=(strict + 1 - fpr) / 2)


def seconds(detail, types, spi):
    out = {}
    ty = np.array([types[t] for t in detail["task"]])
    for label, mask in (("all", np.ones(len(ty), bool)), ("atomic", ty == "atomic"), ("composite", ty == "composite")):
        if not mask.any():
            continue
        sec = lambda v: float(np.mean(v[mask] * detail["H"][mask] * spi))
        row = dict(failures=int(mask.sum()), timer_lead_seconds=sec(detail["timer"]),
                   gain_fraction=float(np.mean((detail["fused"] - detail["timer"])[mask])),
                   gain_seconds=sec(detail["fused"] - detail["timer"]))
        if detail["matched"] is not None:
            row["matched_gain_fraction"] = float(np.mean((detail["fused"] - detail["matched"])[mask]))
            row["matched_gain_seconds"] = sec(detail["fused"] - detail["matched"])
        out[label] = row
    return out


def per_task(detail, tol=1e-12):
    """Mean gain per task at the same target (and matched), sorted; counts of tasks above and below zero."""
    out = {}
    for label, ref in (("same_target", detail["timer"]), ("matched", detail["matched"])):
        if ref is None:
            continue
        g = detail["fused"] - ref
        by = {t: float(np.mean(g[detail["task"] == t])) for t in sorted(set(detail["task"]))}
        vals = np.array(list(by.values()))
        order = sorted(by.items(), key=lambda kv: -kv[1])
        pos = vals[vals > tol].sum()
        top5 = sum(v for _, v in order[:5] if v > tol)
        out[label] = dict(tasks=len(by), above_zero=int(np.sum(vals > tol)), below_zero=int(np.sum(vals < -tol)),
                          at_zero=int(np.sum(np.abs(vals) <= tol)),
                          top5_share_of_positive=None if pos <= 0 else float(top5 / pos), by_task=dict(order))
    return out


# ---- saturation -------------------------------------------------------------------------------

def saturation(rows, key):
    """Ties among the running maxima of one score: few distinct values means a flat (saturated) score."""
    M = cl.Cohort(rows, key).M
    failed = np.array([bool(r["failed"]) for r in rows])
    out = {}
    for label, mask in (("all", np.ones(len(M), bool)), ("successes", ~failed), ("failures", failed)):
        vals, counts = np.unique(np.round(M[mask], 10), return_counts=True)
        out[label] = dict(rollouts=int(mask.sum()), distinct_maxima=int(len(vals)), largest_tie=int(counts.max()),
                          largest_tie_share=float(counts.max() / mask.sum()))
    return out


# ---- printing ------------------------------------------------------------------------------------

def print_summary(r):
    nan = float("nan")

    def cell(x):
        if not x:
            return "   (R12)                 "
        f = x["fused"]
        return f"{f['fpr']:.3f} {f['mean_lead']:.3f} {f['gain']:+.3f} {(f['matched'] or {}).get('gain', nan):+.3f}"

    print("\nR19 native bands, fused (FPR, lead, gain, matched): seen + P | unseen + E | unseen + G")
    for d, b in r["native_bands"].items():
        print(f"  {d:18s} {cell(b.get('seen'))} | {cell(b['unseen']['G_abs'])} | {cell(b['unseen']['G'])}")
        sa = b["unseen"]["G_abs"]["standalone"]
        print(f"  {'':18s} band alone on unseen tasks: FPR {sa['fpr']:.3f}, lead {sa['mean_lead']:.3f}")
    print(f"R19 unseen bootstrap ({r['bootstrap_unseen']['samples']} replicates): gain [95% CI], matched [95% CI]")
    for t, row in r["bootstrap_unseen"]["timers"].items():
        for n, v in row.items():
            m = v["matched_gain"]
            if "gain" in v:
                print(f"  {t:6s} {n:34s} {v['gain']:+.3f} [{v['gain_ci95'][0]:+.3f}, {v['gain_ci95'][1]:+.3f}]  "
                      f"{nan if m is None else m:+.3f} [{v['matched_ci95'][0]:+.3f}, {v['matched_ci95'][1]:+.3f}]")
            else:
                print(f"  {t:6s} {n:34s} matched {nan if m is None else m:+.3f} "
                      f"[{v['matched_ci95'][0]:+.3f}, {v['matched_ci95'][1]:+.3f}]")
    for k, v in r["timers_at_limit"].items():
        print(f"R19 timer {k}: {v['at_limit']} of {v['failures']} failures flagged only at the limit; "
              f"bACC {v['bacc']:.3f}, {v['bacc_limit_as_miss']:.3f} if those count as misses")
    for d, v in r["leads"].items():
        for side in ("seen", "unseen"):
            if side not in v:
                continue
            a = v[side]["seconds"]["all"]
            pt = v[side]["per_task"]["same_target"]
            print(f"R19 {d} {side}: gain {a['gain_seconds']:+.2f} s (matched {a.get('matched_gain_seconds', nan):+.2f} s); "
                  f"tasks above/below zero {pt['above_zero']}/{pt['below_zero']}, top-5 share {pt['top5_share_of_positive']}")
    for k, v in r["zero_input_saturation"].items():
        print(f"R19 saturation {k}: {v['all']['distinct_maxima']} distinct calibration maxima of "
              f"{v['all']['rollouts']}, largest tie {v['all']['largest_tie']}")
