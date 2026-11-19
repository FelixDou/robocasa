"""R25 (docs/r25_band_ablation_plan.md): does Hide-and-Seek's own band need observations after the first
inference? Exploratory. Policy-agnostic, called by seen_unseen_tables.py (XR-1) and rldx_test_table.py
(RLDX-1) with --r25, on unseen tasks only.

The full band is R21's: Hide-and-Seek's three-seed ensemble raw score under SAFE's functional conformal
band at alpha = 0.05, fitted per fold on the other tasks' calibration successes (R19's halves). An ablation
(Hide-and-Seek retrained on the first inference's features, repeated; or on zero inputs) gets the same band,
fitted the same way, on a grid of alphas. The calibration is held fixed throughout, so every band is fixed and
each test rollout's alarm under each band is computed once.

  paired      at a common false-alarm budget: the ablation's band is matched to the full band's test false
              alarms. Among the grid's alphas with at most that many, it takes the one with the largest mean
              lead; this favors the ablation, as matching favors the timer. Delta is the mean over failures
              of the full band's lead minus the matched ablation band's.
  same_alpha  both bands at alpha = 0.05, each at its own false alarms: the paired difference in mean lead.
  vs_E        each head's band alone, and at alpha/2 fused with E at alpha/2 (r21_native_bands.evaluate,
              timer G_abs): Delta l over E matched to its false alarms.
  bootstrap   R21b's `test` variant (seed 21, 1,000 replicates): calibration as it is, tasks with
              replacement, then test rollouts within task and class; the budget, the match and E are redone
              in each replicate. Percentile 95% CIs.

Gates: G0, each head's raw score has the table's running maximum (callers); G1, the full band against E
equals R21's ensemble (false alarms, Delta l); G2, every bootstrap replicate of the full band against E
equals R21b's saved `test` replicate; G3, the precomputed alarms give evaluate()'s false alarms and mean
lead for every head at alpha.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r21_native_bands as r21  # noqa: E402

PLAN = "docs/r25_band_ablation_plan.md (R25)"
ALPHA = 0.05
GRID = tuple(round(0.0025 * k, 4) for k in range(1, 161))       # 0.0025, 0.005, ..., 0.40
FULL = "hide_and_seek"
ABLATIONS = ("hide_and_seek_first", "hide_and_seek_zero")
HEADS = (FULL,) + ABLATIONS
TIMER = "G_abs"
BOOT_B, BOOT_SEED = r21.BOOT_B, r21.BOOT_SEED                  # R21b's `test` variant: 1,000, seed 21
TOL = 1e-12
NAN = float("nan")
R21B_STATS = ("standalone_matched", "fused_matched")
ALPHA_INDEX = GRID.index(ALPHA)


def alarm_matrix(d, key_of, alphas):
    """alarms[j, i]: the alarm inference (inf if none) of test row i under its fold's band at alphas[j]."""
    A = np.full((len(alphas), len(d.test)), math.inf)
    for F in d.folds:
        ri, ci, ti = d.fold(F, d.identity())
        if len(ti) == 0:
            continue
        key = key_of(F)
        ref, conf = [d.ref[i][key] for i in ri], [d.conf[i][key] for i in ci]
        te = [d.test[i][key] for i in ti]
        for j, a in enumerate(alphas):
            A[j, ti] = r21.band_alarms(te, r21.band_fit(ref, conf, a))
    return A


def columns(d):
    L = np.array([float(r["L"]) for r in d.test])
    H = np.array([float(r["horizon"]) for r in d.test])
    return L, H, d.test_failed.copy()


def rates(A, pick, cols):
    """For the picked test rows (with repeats) and each band (row of A): false alarms, and the lead of each
    picked failure (r19_extras.outcome's rule: an alarm at the final inference counts, lead 0)."""
    L, H, failed = (c[pick] for c in cols)
    a = A[:, pick]
    hit = a <= L
    fa = hit[:, ~failed].sum(axis=1)
    lead = np.where(hit[:, failed], (L[failed] - a[:, failed]) / H[failed], 0.0)
    return fa, lead


def paired(full, abl, pick, cols, grid=GRID):
    """The full band (alarms, shape n) against an ablation's bands on the grid (shape k x n) on one pick."""
    fa_f, lead_f = rates(full[None, :], pick, cols)
    fa_a, lead_a = rates(abl, pick, cols)
    budget = int(fa_f[0])
    out = dict(budget=budget, full_mean_lead=float(lead_f[0].mean()))
    j0 = grid.index(ALPHA)
    out["same_alpha"] = dict(full_fp=budget, ablation_fp=int(fa_a[j0]), ablation_mean_lead=float(lead_a[j0].mean()),
                             delta=float(np.mean(lead_f[0] - lead_a[j0])))
    ok = np.flatnonzero(fa_a <= budget)
    if not len(ok):
        out.update(matched_fp=None, alpha=None, ablation_mean_lead=NAN, delta=NAN)
        return out
    means = lead_a.mean(axis=1)
    best = means[ok].max()
    j = int(ok[means[ok] == best].max())                    # ties: the larger alpha
    out.update(matched_fp=int(fa_a[j]), alpha=float(grid[j]), ablation_mean_lead=float(means[j]),
               delta=float(np.mean(lead_f[0] - lead_a[j])))
    return out


def monotone(abl, cols):
    """Whether the ablation's test false alarms never decrease along the grid (descriptive)."""
    fa, _ = rates(abl, np.arange(abl.shape[1]), cols)
    return bool(np.all(np.diff(fa) >= 0))


def vs_e(d, key_of, rk, pick=None):
    r = r21.evaluate(d, key_of, (TIMER,), rk, pick=pick)[TIMER]
    vals = r21.replicate_values(r)
    return dict(standalone=dict(fp=r["standalone"]["fp"], fpr=r["standalone"]["fpr"],
                                mean_lead=r["standalone"]["mean_lead"], matched_gain=vals["standalone_matched"],
                                matched_timer_fp=vals["standalone_matched_timer_fp"]),
                fused=dict(fp=r["fused"]["fp"], fpr=r["fused"]["fpr"], mean_lead=r["fused"]["mean_lead"],
                           matched_gain=vals["fused_matched"], matched_timer_fp=vals["fused_matched_timer_fp"]),
                timer_fp=vals["timer_fp"])


def pct(v):
    v = np.asarray(v, float)
    ok = v[~np.isnan(v)]
    if not len(ok):
        return dict(ci95=None, replicates=0, failed=int(len(v)))
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return dict(ci95=[float(lo), float(hi)], replicates=int(len(ok)), failed=int(len(v) - len(ok)),
                below_zero=float(np.mean(ok < 0)))


def nan_equal_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return math.inf
    both = np.isnan(a) & np.isnan(b)
    one = np.isnan(a) ^ np.isnan(b)
    if one.any():
        return math.inf
    d = np.abs(a - b)[~both]
    return float(d.max()) if len(d) else 0.0


def run(d, key_ofs, rk, ref21=None, r21b=None, r0=None, B=BOOT_B, seed=BOOT_SEED, progress=None):
    """d: r21_native_bands.Design of the unseen side; key_ofs {head: key_of(F)} for HEADS (ensemble raw score);
    ref21: R21's r21.bootstrap.unseen.timers.G_abs.hide_and_seek; r21b: {stat: R21b's `test` replicates of
    the full band against E}; r0 {name: max |diff|} (gate G0). Returns (result, problems)."""
    cols = columns(d)
    full = alarm_matrix(d, key_ofs[FULL], (ALPHA,))[0]
    abl = {h: alarm_matrix(d, key_ofs[h], GRID) for h in ABLATIONS if h in key_ofs}
    ident = np.arange(len(d.test))
    point = {h: vs_e(d, k, rk) for h, k in key_ofs.items()}
    pairs = {h: paired(full, A, ident, cols) for h, A in abl.items()}
    checks = {}
    checks["G0_raw_scores_have_the_table_running_maximum"] = dict(
        max_abs_diff=max(r0.values()) if r0 else None, ok=bool(r0) and max(r0.values()) <= TOL)
    g1 = math.inf
    if ref21:
        sa, fu = point[FULL]["standalone"], point[FULL]["fused"]
        g1 = max(0.0 if sa["fp"] == ref21["standalone"]["fp"] else math.inf,
                 0.0 if fu["fp"] == ref21["fused"]["fp"] else math.inf,
                 abs(sa["matched_gain"] - ref21["standalone"]["matched_gain"]),
                 abs(fu["matched_gain"] - ref21["fused"]["matched_gain"]))
    checks["G1_full_band_equals_r21"] = dict(max_abs_diff=g1, ok=g1 <= TOL)
    g3 = {}
    for h, k in key_ofs.items():
        A = full[None, :] if h == FULL else abl[h][[ALPHA_INDEX]]
        fa, lead = rates(A, ident, cols)
        sa = point[h]["standalone"]
        g3[h] = max(0.0 if int(fa[0]) == sa["fp"] else math.inf, abs(float(lead[0].mean()) - sa["mean_lead"]))
    checks["G3_precomputed_alarms_equal_evaluate"] = dict(by=g3, max_abs_diff=max(g3.values()), ok=max(g3.values()) <= TOL)
    # Bootstrap: R21b's `test` draws.
    rng = np.random.default_rng(seed)
    reps = {h: {s: np.full(B, NAN) for s in ("standalone_matched", "fused_matched", "standalone_fp", "fused_fp")}
            for h in key_ofs}
    preps = {h: {s: np.full(B, NAN) for s in ("delta", "budget", "matched_fp", "alpha", "same_alpha_delta")}
             for h in abl}
    for b in range(B):
        pick = d.draw_test(rng)
        for h, k in key_ofs.items():
            try:
                v = vs_e(d, k, rk, pick=pick)
            except (ValueError, ZeroDivisionError):
                continue
            reps[h]["standalone_matched"][b] = v["standalone"]["matched_gain"]
            reps[h]["fused_matched"][b] = v["fused"]["matched_gain"]
            reps[h]["standalone_fp"][b] = v["standalone"]["fp"]
            reps[h]["fused_fp"][b] = v["fused"]["fp"]
        for h, A in abl.items():
            p = paired(full, A, pick[2], cols)
            preps[h]["delta"][b] = p["delta"]
            preps[h]["budget"][b] = p["budget"]
            preps[h]["matched_fp"][b] = NAN if p["matched_fp"] is None else p["matched_fp"]
            preps[h]["alpha"][b] = NAN if p["alpha"] is None else p["alpha"]
            preps[h]["same_alpha_delta"][b] = p["same_alpha"]["delta"]
        if progress and (b + 1) % 100 == 0:
            progress(f"R25 bootstrap {b + 1}/{B}")
    g2 = math.inf
    if r21b:
        g2 = max(nan_equal_diff(reps[FULL][s], r21b[s]) for s in R21B_STATS)
    checks["G2_replicates_equal_r21b"] = dict(max_abs_diff=g2, ok=g2 <= TOL)
    heads = {}
    for h in key_ofs:
        heads[h] = dict(point[h], standalone_ci=pct(reps[h]["standalone_matched"]),
                        fused_ci=pct(reps[h]["fused_matched"]))
    ablations = {}
    for h in abl:
        ablations[h] = dict(paired=pairs[h], delta_ci=pct(preps[h]["delta"]),
                            same_alpha_delta_ci=pct(preps[h]["same_alpha_delta"]),
                            matched_in_replicates=int(np.sum(~np.isnan(preps[h]["delta"]))),
                            fp_monotone_in_alpha=monotone(abl[h], cols))
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return dict(plan=PLAN, alpha=ALPHA, grid=[GRID[0], GRID[-1], len(GRID)], samples=B, seed=seed, timer=TIMER,
                heads=heads, ablations=ablations, checks=checks), problems


def r21b_replicates(npz_path, head=FULL):
    """R21b's saved `test` replicates of one head against E on unseen tasks ({stat: array})."""
    z = np.load(npz_path)
    return {s: z[f"unseen__test__{TIMER}__{head}__{s}"] for s in R21B_STATS}


def print_summary(r, label):
    print(f"R25 {label}: Hide-and-Seek's own band against its ablations, unseen tasks (alpha {r['alpha']})")
    for h, v in r["heads"].items():
        sa, fu = v["standalone"], v["fused"]
        print(f"  {h:20s} alone: FA {sa['fp']} FPR {sa['fpr']:.4f} Delta l {sa['matched_gain']:+.4f} "
              f"95% {v['standalone_ci'].get('ci95')} | fused with E: FA {fu['fp']} FPR {fu['fpr']:.4f} "
              f"Delta l {fu['matched_gain']:+.4f} 95% {v['fused_ci'].get('ci95')}")
    for h, v in r["ablations"].items():
        p, s = v["paired"], v["paired"]["same_alpha"]
        print(f"  full - {h:20s} common budget {p['budget']} FA (ablation {p['matched_fp']} at alpha {p['alpha']}): "
              f"{p['delta']:+.4f} 95% {v['delta_ci'].get('ci95')} | same alpha ({s['full_fp']} vs {s['ablation_fp']} FA): "
              f"{s['delta']:+.4f} 95% {v['same_alpha_delta_ci'].get('ci95')} | matched in {v['matched_in_replicates']} "
              f"replicates; FA monotone in alpha: {v['fp_monotone_in_alpha']}")
    for k, v in r["checks"].items():
        print(f"R25 {k.split('_', 1)[0]}: {'OK' if v['ok'] else 'FAILED'} (max |diff| {v['max_abs_diff']})")
