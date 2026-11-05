"""Review-response analyses on the E1 cohorts, pre-specified in docs/xr1_review_analyses_plan.md.

Exploratory. Both E1 cohorts are open: E1 (SAFE + P), Amendment 2 (Hide-and-Seek + P), the
dose-response follow-up (success-only baselines) and the cross-task study have read them.
Nothing here changes a registered decision.

  R1  lead-FPR curves. At each FPR target alpha, every timer and fused detector is refitted with
      Proposition 1 on the calibration cohort only and evaluated once on the test cohort. Gains are
      reported at the same alpha and at matched test false alarms.
  R2  the alpha = 0.05 table, at the registered thresholds and at matched false alarms: P loosened
      on the test cohort to the fused detector's false alarms (favors the timer), and the fused
      detector tightened along its own path to P's false alarms.
  R3  per-task gains over P and per-task false alarms at alpha = 0.05.
  R4  calibration reuse. (a) The fused path is chosen on half of the calibration successes and its
      margin is set on the other half, so Proposition 1 holds for the chosen path. (b) Random
      re-splits of the pooled successes of both cohorts separate the effect of choosing the path
      on the calibration successes from a difference between the two cohorts.

Detectors, timers and fusion are confirm_lead.py's (E1) with alpha as a parameter:
fit_alpha(c, rk, 0.05) equals confirm_lead.fit_fast(c, rk). Before any result is written, the
run reproduces E1, Amendment 2 and (if given) the dose-response gains at alpha = 0.05, and the
decision CIs when the bootstrap size equals Amendment 2's. Otherwise it writes only the checks and
exits with status 1.

Usage:
  python3 -I review_analyses.py run --calibration E1_OUT/calibration_predictions.jsonl \\
      --test E1_OUT/test_predictions.jsonl \\
      --scores-calibration AMEND2_ROOT/scores/lead_calibration_predictions.jsonl \\
      --scores-test AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl \\
      --normalizations AMEND2_ROOT/scores/normalizations.json --runtime-bundle runtime_bundle.json \\
      --e1-result E1_OUT/confirm_lead.json --amend2-result AMEND2_ROOT/confirm_hide_and_seek.json \\
      [--baselines-calibration E1_OUT/calibration_with_baselines.jsonl \\
       --baselines-test E1_OUT/test_with_baselines.jsonl --dose-response-result DOSE.json] \\
      --output OUT.json
"""
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_hide_and_seek as chs
import confirm_lead as cl
import quantile_timer as qt

PLAN = "docs/xr1_review_analyses_plan.md"
ALPHAS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.125, 0.15, 0.20)
BAND = (0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10)     # the R1 summary band
BOOT_ALPHAS = (0.02, 0.05, 0.10)                       # decision bootstrap (refits) at these
REF = 0.05
HS, FIRST, ZERO, SHUFFLE = chs.PRIMARY, chs.FIRST, chs.ZERO, chs.SHUFFLE
POLICY, CONTROLS = ("safe", HS), (FIRST, ZERO, SHUFFLE)
DELTA_GRID = np.arange(-0.5, 1.0001, 0.0025)            # confirm_lead's test-tuned timer grid
LAM_GRID = np.arange(-0.5, 1.5001, 0.0025)              # fused path, tightened on the test cohort
NOISE_DRAWS, NOISE_SEED = 20, 1
SPLIT_DRAWS, SPLIT_SEED = 20, 3
RESPLITS, RESPLIT_SEED = 200, 4
EPS = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r4(x):
    return None if x is None else round(float(x), 4)


# ---- detectors with alpha as a parameter ----------------------------------------------------

def fit_alpha(cal, rk, alpha):
    """confirm_lead.fit_fast with the FPR target alpha as a parameter (same grid and ties)."""
    S, F = cal.take(~cal.failed), cal.take(cal.failed)
    rS, rF = cl.rho_rows(S, rk), cl.rho_rows(F, rk)
    t_lam = cl.conformal(cl.V_vec(S, rS, math.inf, 0.0), alpha)
    us = sorted(set(np.round(np.quantile(F.M, np.linspace(0, 1, 21)), 4).tolist()))
    best = (cl.fail_lead(F, rF, t_lam, math.inf, 0.0), math.inf, 0.0, t_lam)
    for u in us:
        for s in cl.S_GRID:
            lam = cl.conformal(cl.V_vec(S, rS, u, s), alpha)
            if not math.isfinite(lam):
                continue
            v = cl.fail_lead(F, rF, lam, u, s)
            if v > best[0] + EPS:
                best = (v, u, s, lam)
    return dict(u=math.inf, s=0.0, lam=t_lam), dict(u=best[1], s=best[2], lam=best[3])


def fit_split(select, certify, rk, alpha):
    """Path (u, s) chosen on `select` (failures and the selection successes) exactly as fit_alpha
    chooses it; the margins of the timer and of that path come from `certify`'s successes only."""
    _, fu = fit_alpha(select, rk, alpha)
    S = certify.take(~certify.failed)
    rS = cl.rho_rows(S, rk)
    return (dict(u=math.inf, s=0.0, lam=cl.conformal(cl.V_vec(S, rS, math.inf, 0.0), alpha)),
            dict(u=fu["u"], s=fu["s"], lam=cl.conformal(cl.V_vec(S, rS, fu["u"], fu["s"]), alpha)))


def twa(mean_lead, fpr):
    """Time-weighted accuracy (FIPER; Hide-and-Seek Eq. 12) when every failure ends at H_k."""
    return round(0.5 * (mean_lead + 1.0 - fpr), 4)


def evaluate(te, rk, det):
    e, lead = cl.evaluate_fast(te, rk, det)
    return dict(e, twa=twa(e["mean_lead"], e["fpr"])), lead


def success_alarms(te, rk, det):
    a = cl.alarm_vec(te, cl.rho_rows(te, rk), det["lam"], det["u"], det["s"])
    return (a <= te.L) & ~te.failed


def loosen_timer(te, rk, target_fp):
    """P with its margin tuned on the test cohort: largest lead with at most target_fp false alarms."""
    best = None
    for d in DELTA_GRID:
        e, l = cl.evaluate_fast(te, rk, dict(lam=float(d), u=math.inf, s=0.0))
        if e["fp"] <= target_fp and (best is None or e["mean_lead"] > best[1]["mean_lead"]):
            best = (float(d), e, l)
    return best


def tighten_fused(te, rk, det, target_fp):
    """The fused detector moved along its own path (u, s fixed) on the test cohort."""
    best = None
    for lam in LAM_GRID:
        e, l = cl.evaluate_fast(te, rk, dict(lam=float(lam), u=det["u"], s=det["s"]))
        if e["fp"] <= target_fp and (best is None or e["mean_lead"] > best[1]["mean_lead"]):
            best = (float(lam), e, l)
    return best


# ---- bootstrap ------------------------------------------------------------------------------

def boot_alpha(C, rk, keys, alphas, pairs, B, seed):
    """confirm_hide_and_seek.boot_multi at several alphas, on the same replicates.

    The draw sequence is boot_multi's, so at alpha = 0.05 each key's CI equals Amendment 2's
    decision CI for the same B and seed.
    """
    rng = np.random.default_rng(seed)
    cal0, te0 = C[keys[0]]
    sc, st = cl.strata(cal0), cl.strata(te0)
    tasks = sorted(st)
    if sorted(sc) != tasks:
        raise ValueError("Calibration and test task sets differ")
    gains = {(a, k): [] for a in alphas for k in keys}
    for _ in range(B):
        ci, ti = [], []
        for t in rng.choice(tasks, len(tasks)):
            for idx, out in ((sc[t], ci), (st[t], ti)):
                for k in (0, 1):
                    out.append(rng.choice(idx[k], len(idx[k])))
        ci, ti = np.concatenate(ci), np.concatenate(ti)
        for a in alphas:
            tm, _ = fit_alpha(cal0.take(ci), rk, a)
            _, lp = cl.evaluate_fast(te0.take(ti), rk, tm)
            for k in keys:
                _, fu = fit_alpha(C[k][0].take(ci), rk, a)
                _, lf = cl.evaluate_fast(C[k][1].take(ti), rk, fu)
                gains[(a, k)].append(float(np.mean(lf - lp)))
    pct = lambda v: np.round(np.percentile(v, [2.5, 97.5]), 4).tolist()
    out = {"samples": B, "seed": seed, "by_alpha": {}}
    for a in alphas:
        out["by_alpha"][str(a)] = dict(
            gain_ci95={k: pct(gains[(a, k)]) for k in keys},
            contrast_ci95={f"{x}_minus_{y}": pct(np.asarray(gains[(a, x)]) - np.asarray(gains[(a, y)]))
                           for x, y in pairs})
    return out


# ---- per-task summaries (R3) ----------------------------------------------------------------

def sign_test(pos, neg):
    """Exact two-sided sign test, zeros excluded."""
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, p), 4)


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort").astype(float)
    x, y = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def per_task(teC, rk, tm, fu, lF, lP, types):
    """Gain over P per task (mean over the task's failures) and false alarms per task."""
    tf = teC.task[teC.failed]
    d = lF - lP
    fpP, fpF = success_alarms(teC, rk, tm), success_alarms(teC, rk, fu)
    rows = {}
    for t in sorted(set(teC.task)):
        m = tf == t
        dk = d[m]
        rows[t] = dict(type=types.get(t), failures=int(m.sum()), gain=r4(dk.mean()),
                       var_of_mean=float(np.var(dk, ddof=1) / len(dk)) if len(dk) > 1 else None,
                       earlier=int(np.sum(dk > EPS)), later=int(np.sum(dk < -EPS)),
                       successes=int(np.sum((teC.task == t) & ~teC.failed)),
                       fp_P=int(fpP[teC.task == t].sum()), fp_fused=int(fpF[teC.task == t].sum()))
    g = np.array([rows[t]["gain"] for t in rows])
    names = list(rows)
    pos, neg = int(np.sum(g > EPS)), int(np.sum(g < -EPS))
    order = np.argsort(-g, kind="mergesort")
    total = float(g.sum())
    loto = []
    for t in names:
        keep = tf != t
        loto.append((float(d[keep].mean()), t))
    loto.sort()
    within = [v["var_of_mean"] for v in rows.values() if v["var_of_mean"] is not None]
    summary = dict(
        tasks=len(names), pooled_gain=r4(d.mean()), positive=pos, zero=len(names) - pos - neg, negative=neg,
        sign_test_p=sign_test(pos, neg), median=r4(np.median(g)), q25=r4(np.percentile(g, 25)),
        q75=r4(np.percentile(g, 75)),
        top5=[names[i] for i in order[:5]],
        top5_share=r4(sum(g[i] for i in order[:5]) / total) if total > EPS else None,
        loto_min=dict(gain=r4(loto[0][0]), without=loto[0][1]),
        loto_max=dict(gain=r4(loto[-1][0]), without=loto[-1][1]),
        between_task_sd=r4(np.std(g, ddof=1)) if len(g) > 1 else None,
        within_task_se_rms=r4(math.sqrt(np.mean(within))) if within else None,
        by_type={tt: dict(tasks=int(sum(rows[t]["type"] == tt for t in names)),
                          mean_task_gain=r4(np.mean([rows[t]["gain"] for t in names if rows[t]["type"] == tt])),
                          positive=int(sum(rows[t]["gain"] > EPS for t in names if rows[t]["type"] == tt)))
                 for tt in sorted({str(v["type"]) for v in rows.values()})},
        false_alarms={who: dict(total=int(sum(v[f"fp_{who}"] for v in rows.values())),
                                tasks_with_0=int(sum(v[f"fp_{who}"] == 0 for v in rows.values())),
                                tasks_with_2_or_more=int(sum(v[f"fp_{who}"] >= 2 for v in rows.values())),
                                max_per_task=int(max(v[f"fp_{who}"] for v in rows.values())))
                      for who in ("P", "fused")})
    return rows, summary


def read_r3(s):
    if s["pooled_gain"] is None or s["pooled_gain"] <= EPS:
        return "no pooled gain"
    concentrated = (s["top5_share"] is not None and s["top5_share"] > 0.5) or \
        s["loto_min"]["gain"] < 0.5 * s["pooled_gain"]
    nonzero = s["positive"] + s["negative"]
    broad = (nonzero > 0 and s["positive"] >= 0.5 * nonzero and s["sign_test_p"] is not None
             and s["sign_test_p"] < 0.05 and not concentrated)
    return "broad" if broad else "concentrated" if concentrated else "mixed"


# ---- inputs ---------------------------------------------------------------------------------

def attach_baselines(rows, path, label):
    """Copy the frozen success-only baseline trajectories (dose-response files) onto the E1 rows."""
    by_id = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_id[str(r["rollout_id"])] = r
    names = None
    for r in rows:
        o = by_id.get(str(r["rollout_id"]))
        if o is None:
            raise ValueError(f"{label}: no baseline row for {r['rollout_id']}")
        if bool(o["failed"]) != r["failed"] or not np.allclose(np.asarray(o["safe_scores"], float), r["safe"], atol=1e-9):
            raise ValueError(f"{label}: baseline file differs from the E1 row for {r['rollout_id']}")
        got = sorted(o["baseline_scores"])
        if names is None:
            names = got
        elif got != names:
            raise ValueError(f"{label}: baseline names differ for {r['rollout_id']}")
        for n in names:
            v = np.asarray(o["baseline_scores"][n], float)
            if len(v) != r["L"] or not np.all(np.isfinite(v)):
                raise ValueError(f"{label}: {n} trajectory invalid for {r['rollout_id']}")
            r[n] = v
    return names or []


def same(got, want, tol=0.0):
    if isinstance(want, (list, tuple)):
        return len(got) == len(want) and all(same(g, w, tol) for g, w in zip(got, want))
    if isinstance(want, float) or isinstance(got, float):
        return abs(float(got) - float(want)) <= tol
    return got == want


# ---- run ------------------------------------------------------------------------------------

def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    dev = qt.dev_fractions(bundle)
    rkP = qt.rho(dev, cl.PRIMARY_Q)
    rkG = {t: 0.0 for t in rkP}
    norms = json.loads(Path(args.normalizations).read_text())
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    if {r["rollout_id"] for r in cal_all} & {r["rollout_id"] for r in te_all} or cl.identities(cal_all) & cl.identities(te_all):
        raise ValueError("Calibration and test cohorts overlap")
    tasks = set(bundle["task_horizons"])
    if {r["task_name"] for r in cal} != tasks or {r["task_name"] for r in te} != tasks:
        raise ValueError("Cohorts lack exact frozen task coverage")
    chs.attach(cal, chs.load_scores(args.scores_calibration), "calibration")
    chs.attach(te, chs.load_scores(args.scores_test), "test")
    baselines = []
    if bool(args.baselines_calibration) != bool(args.baselines_test):
        raise ValueError("Pass both baseline files or neither")
    if args.baselines_calibration:
        baselines = attach_baselines(cal, args.baselines_calibration, "calibration")
        if attach_baselines(te, args.baselines_test, "test") != baselines:
            raise ValueError("Calibration and test carry different baselines")
    keys = list(POLICY + CONTROLS) + baselines
    types = {r["task_name"]: r.get("task_type") for r in cal + te}
    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in vars(args).items()
              if k in ("calibration", "test", "scores_calibration", "scores_test", "normalizations", "runtime_bundle",
                       "e1_result", "amend2_result", "baselines_calibration", "baselines_test", "dose_response_result")
              and v}
    checks = dict(normalization_max_abs_diff=max(chs.check_normalization(cal, norms), chs.check_normalization(te, norms)),
                  test_failures_end_at_H=all(r["L"] == r["horizon"] for r in te if r["failed"]))
    failures = []
    if not checks["test_failures_end_at_H"]:
        failures.append("a test failure does not end at H_k, so TWA is not 0.5 (lead + 1 - FPR)")

    C = {k: (cl.Cohort(cal, k), cl.Cohort(te, k)) for k in keys}
    Fte = [r for r in te if r["failed"]]

    # ---- gates at alpha = 0.05 -----------------------------------------------------------
    same_path = all(fit_alpha(C[k][0], rkP, REF) == cl.fit_fast(C[k][0], rkP) for k in keys)
    checks["fit_alpha_equals_fit_fast"] = same_path
    if not same_path:
        failures.append("fit_alpha(0.05) differs from confirm_lead.fit_fast")
    tm, _ = fit_alpha(C["safe"][0], rkP, REF)
    eP, lP = evaluate(C["safe"][1], rkP, tm)
    ref_fit = {k: fit_alpha(C[k][0], rkP, REF)[1] for k in keys}
    ref_eval = {k: evaluate(C[k][1], rkP, ref_fit[k]) for k in keys}
    gain_ref = {k: r4(np.mean(ref_eval[k][1] - lP)) for k in keys}
    e1 = json.loads(Path(args.e1_result).read_text())["by_q"][str(cl.PRIMARY_Q)]
    got = (eP["mean_lead"], eP["fp"], ref_eval["safe"][0]["mean_lead"], ref_eval["safe"][0]["fp"], gain_ref["safe"])
    want = (e1["P"]["mean_lead"], e1["P"]["fp"], e1["safe_plus_P"]["mean_lead"], e1["safe_plus_P"]["fp"], e1["gain"])
    checks["e1_reference"] = dict(got=got, want=want, ok=same(got, want))
    if not same(got, want):
        failures.append(f"E1 reference: got {got}, want {want}")
    a2 = json.loads(Path(args.amend2_result).read_text())
    got = [eP["mean_lead"], eP["fp"]] + [x for k in (HS,) + CONTROLS for x in
                                         (ref_eval[k][0]["mean_lead"], ref_eval[k][0]["fp"], gain_ref[k])]
    want = [a2["P"]["mean_lead"], a2["P"]["fp"]] + [x for k in (HS,) + CONTROLS for x in
                                                    (a2["detectors"][k]["test"]["mean_lead"], a2["detectors"][k]["test"]["fp"],
                                                     a2["detectors"][k]["gain"])]
    checks["amend2_reference"] = dict(got=got, want=want, ok=same(got, want))
    if not same(got, want):
        failures.append(f"Amendment 2 reference: got {got}, want {want}")
    if args.dose_response_result:
        dr = json.loads(Path(args.dose_response_result).read_text())["detectors"]
        got = [(gain_ref[n], ref_eval[n][0]["fp"]) for n in baselines]
        want = [(dr[n]["pooled"]["gain"], dr[n]["pooled"]["fused"]["fp"]) for n in baselines]
        checks["dose_response_reference"] = dict(got=got, want=want, ok=same(got, want))
        if not same(got, want):
            failures.append(f"dose-response reference: got {got}, want {want}")
    if failures:
        return finish(out_path, dict(plan=PLAN, inputs=inputs, checks=checks, reproduction_failures=failures))

    # ---- R1 curves -----------------------------------------------------------------------
    r1 = dict(alphas=list(ALPHAS), band=list(BAND), P=[], G=[], detectors={k: [] for k in keys},
              plus_G={k: [] for k in POLICY}, noise=[])
    vec, p_lead = {}, {}
    for a in ALPHAS:
        tP, _ = fit_alpha(C["safe"][0], rkP, a)
        ePa, lPa = evaluate(C["safe"][1], rkP, tP)
        p_lead[a] = lPa
        tG, _ = fit_alpha(C["safe"][0], rkG, a)
        eGa, lGa = evaluate(C["safe"][1], rkG, tG)
        r1["P"].append(dict(alpha=a, path=cl.det_json(tP), test=ePa))
        r1["G"].append(dict(alpha=a, path=cl.det_json(tG), test=eGa))
        for k in keys:
            _, fu = fit_alpha(C[k][0], rkP, a)
            eF, lF = evaluate(C[k][1], rkP, fu)
            loose = loosen_timer(C[k][1], rkP, eF["fp"])
            tight = tighten_fused(C[k][1], rkP, fu, ePa["fp"])
            row = dict(alpha=a, path=cl.det_json(fu), test=eF, gain=r4(np.mean(lF - lPa)),
                       gain_test_only=qt.boot(lF - lPa, Fte),
                       matched_P=dict(delta=loose[0], fp=loose[1]["fp"], mean_lead=loose[1]["mean_lead"],
                                      gain=r4(np.mean(lF - loose[2])), gain_test_only=qt.boot(lF - loose[2], Fte)))
            if tight is None:
                row["tightened_to_P"] = None
            else:
                row["tightened_to_P"] = dict(lam=tight[0], fp=tight[1]["fp"], mean_lead=tight[1]["mean_lead"],
                                             gain=r4(np.mean(tight[2] - lPa)),
                                             gain_test_only=qt.boot(tight[2] - lPa, Fte))
            r1["detectors"][k].append(row)
            vec[(a, k)] = (lF - lPa, lF - loose[2])
        for k in POLICY:
            _, fg = fit_alpha(C[k][0], rkG, a)
            eFG, lFG = evaluate(C[k][1], rkG, fg)
            r1["plus_G"][k].append(dict(alpha=a, path=cl.det_json(fg), test=eFG, gain_over_G=r4(np.mean(lFG - lGa)),
                                        gain_over_P=r4(np.mean(lFG - lPa))))
        print(f"alpha {a:<6}: P {ePa['mean_lead']} (FP {ePa['fp']})  " + "  ".join(
            f"{k} {r1['detectors'][k][-1]['gain']:+.4f}/{r1['detectors'][k][-1]['matched_P']['gain']:+.4f}"
            for k in POLICY + CONTROLS), flush=True)
    rng = np.random.default_rng(NOISE_SEED)
    noise = defaultdict(list)
    for _ in range(args.null_draws):
        for rws in (cal, te):
            for r in rws:
                r["noise"] = np.maximum.accumulate(rng.standard_normal(r["L"]))
        cN, tN = cl.Cohort(cal, "noise"), cl.Cohort(te, "noise")
        for a in ALPHAS:
            _, fu = fit_alpha(cN, rkP, a)
            eN, lN = cl.evaluate_fast(tN, rkP, fu)
            noise[a].append((float(np.mean(lN - p_lead[a])), eN["fp"]))
    r1["noise"] = [dict(alpha=a, draws=len(noise[a]), mean_gain=r4(np.mean([g for g, _ in noise[a]])),
                        max_gain=r4(max(g for g, _ in noise[a])), mean_fp=r4(np.mean([f for _, f in noise[a]])))
                   for a in ALPHAS if noise[a]]
    band = {}
    for k in keys:
        rows = [x for x in r1["detectors"][k] if x["alpha"] in BAND]
        d_same = np.mean([vec[(a, k)][0] for a in BAND], axis=0)
        d_match = np.mean([vec[(a, k)][1] for a in BAND], axis=0)
        b = dict(mean_gain_same_alpha=qt.boot(d_same, Fte), mean_gain_matched=qt.boot(d_match, Fte),
                 matched_positive_at=[x["alpha"] for x in rows if x["matched_P"]["gain"] > EPS],
                 matched_nonpositive_at=[x["alpha"] for x in rows if x["matched_P"]["gain"] <= EPS])
        lo_m, hi_m = b["mean_gain_matched"]["ci95"]
        lo_s, hi_s = b["mean_gain_same_alpha"]["ci95"]
        if not b["matched_nonpositive_at"] and lo_m > 0:
            b["reading"] = "earlier at every budget"
        elif hi_m < 0:
            b["reading"] = "later than the timer at matched false alarms"
        elif lo_m <= 0 <= hi_m and lo_s <= 0 <= hi_s:
            b["reading"] = "no earlier warning shown"
        else:
            b["reading"] = "budget-dependent"
        band[k] = b
    r1["band_summary"] = band
    boot = boot_alpha(C, rkP, list(POLICY + CONTROLS), BOOT_ALPHAS,
                      [(HS, "safe")] + [(HS, c) for c in CONTROLS], args.bootstrap, cl.BOOT_SEED)
    r1["decision_bootstrap"] = boot
    if args.bootstrap == a2.get("bootstrap", {}).get("samples"):
        got = [boot["by_alpha"][str(REF)]["gain_ci95"][k] for k in POLICY + CONTROLS]
        want = [a2["detectors"][k]["gain_decision_ci95"] for k in POLICY + CONTROLS]
        checks["amend2_decision_ci"] = dict(got=got, want=want, ok=same(got, want))
        if not same(got, want):
            failures.append(f"Amendment 2 decision CIs: got {got}, want {want}")
            return finish(out_path, dict(plan=PLAN, inputs=inputs, checks=checks, reproduction_failures=failures))
    else:
        checks["amend2_decision_ci"] = dict(ok=None, note="bootstrap size differs from Amendment 2's; not compared")

    # ---- R2: the alpha = 0.05 table ------------------------------------------------------
    at = lambda lst: next(x for x in lst if x["alpha"] == REF)
    r2 = dict(P=at(r1["P"])["test"], G=at(r1["G"])["test"], rows={})
    for k in keys:
        x = at(r1["detectors"][k])
        r2["rows"][k] = dict(fp=x["test"]["fp"], fpr=x["test"]["fpr"], fpr_wilson95=x["test"]["fpr_wilson95"],
                             mean_lead=x["test"]["mean_lead"], twa=x["test"]["twa"], gain=x["gain"],
                             gain_decision_ci95=boot["by_alpha"][str(REF)]["gain_ci95"].get(k),
                             matched_P=x["matched_P"], tightened_to_P=x["tightened_to_P"])
    for k in POLICY:
        r2["rows"][f"{k}_plus_G"] = at(r1["plus_G"][k])

    # ---- R3: per-task gains --------------------------------------------------------------
    r3 = {}
    for k in keys:
        rows, summary = per_task(C[k][1], rkP, tm, ref_fit[k], ref_eval[k][1], lP, types)
        summary["reading"] = read_r3(summary)
        r3[k] = dict(summary=summary, per_task=rows)
    g_safe = [r3["safe"]["per_task"][t]["gain"] for t in sorted(tasks)]
    g_hs = [r3[HS]["per_task"][t]["gain"] for t in sorted(tasks)]
    r3["spearman_safe_vs_hs_task_gains"] = spearman(g_safe, g_hs)

    # ---- R4a: path chosen on half of the calibration successes ---------------------------
    calS_idx = defaultdict(list)
    for i, r in enumerate(cal):
        if not r["failed"]:
            calS_idx[r["task_name"]].append(i)          # cl.primary order: (seed, reset, id)
    fail_idx = np.array([i for i, r in enumerate(cal) if r["failed"]])

    def split_fit(k, halves, ci=True):
        A = np.concatenate([fail_idx] + [h[0] for h in halves.values()])
        Bi = np.concatenate([h[1] for h in halves.values()])
        tB, fB = fit_split(C[k][0].take(A), C[k][0].take(Bi), rkP, REF)
        ePB, lPB = evaluate(C[k][1], rkP, tB)
        eFB, lFB = evaluate(C[k][1], rkP, fB)
        out = dict(path=cl.det_json(fB), P_half=dict(path=cl.det_json(tB), test=ePB), test=eFB,
                   gain_over_P_half=r4(np.mean(lFB - lPB)), gain_over_P=r4(np.mean(lFB - lP)))
        if ci:
            out["gain_over_P_half_test_only"] = qt.boot(lFB - lPB, Fte)
        return out

    def halves_of(order):
        return {t: (np.array(v[: len(v) // 2]), np.array(v[len(v) // 2:])) for t, v in order.items()}

    r4a = dict(certification_successes=int(sum(len(v) - len(v) // 2 for v in calS_idx.values())), deterministic={},
               random_splits={})
    rng = np.random.default_rng(SPLIT_SEED)
    draws = [halves_of({t: list(rng.permutation(v)) for t, v in sorted(calS_idx.items())}) for _ in range(SPLIT_DRAWS)]
    for k in POLICY + CONTROLS:
        r4a["deterministic"][k] = split_fit(k, halves_of(calS_idx))
        rs = [split_fit(k, h, ci=False) for h in draws]
        fpr = [x["test"]["fpr"] for x in rs]
        gain = [x["gain_over_P_half"] for x in rs]
        r4a["random_splits"][k] = dict(draws=len(rs), fpr_median=r4(np.median(fpr)), fpr_min=min(fpr), fpr_max=max(fpr),
                                       gain_median=r4(np.median(gain)), gain_min=min(gain), gain_max=max(gain))
    hs_split = r4a["deterministic"][HS]
    r4a["reading_hs"] = ("does not depend on reusing calibration successes"
                         if hs_split["gain_over_P_half"] >= cl.MARGIN and hs_split["test"]["fpr"] <= cl.FPR_MAX
                         else "sensitive to reusing calibration successes")

    # ---- R4b: random re-splits of the pooled successes -----------------------------------
    pool = cal + te
    PC = {k: cl.Cohort(pool, k) for k in POLICY}
    n_cal = len(cal)
    pool_S = defaultdict(list)
    for i, r in enumerate(pool):
        if not r["failed"]:
            pool_S[r["task_name"]].append(i)
    cal_F = np.array([i for i, r in enumerate(cal) if r["failed"]])
    te_F = np.array([n_cal + i for i, r in enumerate(te) if r["failed"]])
    per_task_cal = {t: sum(1 for i in v if i < n_cal) for t, v in pool_S.items()}
    rng = np.random.default_rng(RESPLIT_SEED)
    rec = defaultdict(list)
    for _ in range(args.resplits):
        cs, ts, A, Bc = [], [], [], []
        for t in sorted(pool_S):
            p = rng.permutation(pool_S[t])
            c = p[: per_task_cal[t]]
            cs.append(c); ts.append(p[per_task_cal[t]:])
            A.append(c[: len(c) // 2]); Bc.append(c[len(c) // 2:])
        ci = np.concatenate([cal_F] + cs)
        ti = np.concatenate([te_F] + ts)
        Ai = np.concatenate([cal_F] + A)
        Bi = np.concatenate(Bc)
        for k in POLICY:
            calk, tek = PC[k].take(ci), PC[k].take(ti)
            tmk, fuk = fit_alpha(calk, rkP, REF)
            eT, _ = cl.evaluate_fast(tek, rkP, tmk)
            eO, _ = cl.evaluate_fast(tek, rkP, fuk)
            tB, fB = fit_split(PC[k].take(Ai), PC[k].take(Bi), rkP, REF)
            eS, _ = cl.evaluate_fast(tek, rkP, fB)
            rec[("P", k)].append(eT["fpr"]); rec[("original", k)].append(eO["fpr"]); rec[("split", k)].append(eS["fpr"])
    summ = lambda v, actual: dict(mean=r4(np.mean(v)), sd=r4(np.std(v)), p_above_0p05=r4(np.mean(np.array(v) > REF + EPS)),
                                  p_above_0p075=r4(np.mean(np.array(v) > cl.FPR_MAX + EPS)),
                                  actual=actual, actual_percentile=r4(100 * np.mean(np.array(v) < actual - EPS)))
    r4b = dict(resplits=args.resplits, seed=RESPLIT_SEED, by_detector={})
    for k in POLICY:
        b = dict(P=summ(rec[("P", "safe")], eP["fpr"]), original=summ(rec[("original", k)], ref_eval[k][0]["fpr"]),
                 split_path=summ(rec[("split", k)], r4a["deterministic"][k]["test"]["fpr"]))
        sel = b["original"]["mean"] - b["split_path"]["mean"]
        shift = ref_eval[k][0]["fpr"] - b["original"]["mean"]
        b["selection_effect"], b["cohort_effect"] = r4(sel), r4(shift)
        b["reading"] = ("neither raises the test FPR by 0.005 or more" if max(sel, shift) < 0.005 else
                        "mainly path selection on calibration successes" if sel >= shift else
                        "mainly a difference between the calibration and test cohorts")
        r4b["by_detector"][k] = b

    out = dict(plan=PLAN, status="exploratory (opened cohorts)", inputs=inputs,
               counts=dict(calibration=len(cal), test=len(te), tasks=len(tasks), detectors=keys),
               checks=checks, reproduction_failures=[], R1=r1, R2=r2, R3=r3, R4=dict(split_path=r4a, resplit=r4b))
    finish(out_path, out)
    print(json.dumps({k: dict(band=band[k]["reading"], matched_band=band[k]["mean_gain_matched"],
                              r3=r3[k]["summary"]["reading"]) for k in POLICY + CONTROLS}, indent=1))
    print("R4a Hide-and-Seek:", r4a["reading_hs"], "| R4b:",
          {k: r4b["by_detector"][k]["reading"] for k in POLICY})
    print("REVIEW ANALYSES OK")
    return 0


def finish(path, out):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    if out.get("reproduction_failures"):
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(out["reproduction_failures"]))
        print("No result is reported when a reference does not reproduce.")
        return 1
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("run")
    for name in ("--calibration", "--test", "--scores-calibration", "--scores-test", "--normalizations",
                 "--runtime-bundle", "--e1-result", "--amend2-result", "--output"):
        c.add_argument(name, required=True)
    for name in ("--baselines-calibration", "--baselines-test", "--dose-response-result"):
        c.add_argument(name)
    c.add_argument("--bootstrap", type=int, default=cl.BOOT_SAMPLES)
    c.add_argument("--null-draws", type=int, default=NOISE_DRAWS)
    c.add_argument("--resplits", type=int, default=RESPLITS)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
