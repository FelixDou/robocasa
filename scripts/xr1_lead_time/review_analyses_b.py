"""Mock-review list B analyses on the E1 cohorts (R12 of docs/xr1_review_analyses_plan.md). Exploratory.

Both E1 cohorts are open (E1, Amendment 2, R1-R10 read them). Nothing here changes a registered
decision. Detectors, timers, fusion and the alpha = 0.05 fits are confirm_lead.py's, through
review_analyses.fit_alpha; "matched" means P loosened on the test cohort to the fused detector's
false alarms (review_analyses.loosen_timer, here by bisection on the same grid).

  B1  time-only comparators. A_fixed: per-task timer fractions a_k chosen on the DEVELOPMENT
      successes to maximize mean_k(1 - a_k) subject to mean_k FPR_k(a_k) <= alpha (Lagrangian over
      a fixed mu grid), then one shared margin by Proposition 1 on the calibration successes. The
      path does not read the calibration successes, so Proposition 1 holds. A_adaptive: the
      development budget beta is chosen on the calibration cohort exactly as the fused path (u, s)
      is (largest calibration-failure lead, each beta at its own margin), so it is as adaptive as
      the fused detectors. Reported: each timer against P, SAFE + P and Hide-and-Seek + P against
      it (same target and matched), and each score fused with it.
  B2  native time-varying bands. SAFE and Hide-and-Seek with the official SAFE functional
      conformal upper band (robocasa.recovery.safe.conformal, alignment "extend", tfunc
      modulation) on the seed-averaged, per-task-normalized raw score (no running maximum).
      The calibration successes are split per task into reference and nonconformity halves.
      Standalone at alpha; OR-fused with P under a Bonferroni split (band and P each at alpha/2).
  B3  H2 at matched FPR. Gaussian noise and within-task resampled SAFE and Hide-and-Seek scores
      (confirm_lead.null_scores, confirm_hide_and_seek.resample_null), each fused with P like the
      policy scores; their gains at the same target and matched; each policy's matched gain
      ranked among them; a paired test-only CI of (policy matched gain - mean null matched gain).
  B4  full-pipeline bootstrap of the two central matched gains. Every replicate resamples tasks,
      then rollouts within task and class in both cohorts (chs.boot_multi's draw sequence),
      refits P and the fused detector, and re-matches P on the resampled test cohort. The
      same-target CIs at the same seed must equal Amendment 2's decision CIs (gate G5).
  B5  lead in seconds (0.8 s per inference) and by task type, at the same target and matched;
      with --nr-labels and --nr-key, lead before the point of no return on the annotated test
      failures for P, SAFE + P and Hide-and-Seek + P.

Gates (no result without them): G1 E1 and Amendment 2 reproduced at alpha = 0.05 (as review_analyses
does); G2 the bisection loosening equals review_analyses.loosen_timer, and with --review-result the
matched gains and their test-only CIs equal R2's; G3 A_fixed's path is monotone and the allocation code
with P's fractions gives P exactly; G5 above; G6 with --nr-key, the key's P and SAFE + P alarms are
reproduced from the E1 fits.

Usage:
  python3 -I review_analyses_b.py run --calibration E1/calibration_predictions.jsonl \\
      --test E1/test_predictions.jsonl --scores-calibration A2/scores/lead_calibration_predictions.jsonl \\
      --scores-test A2/scores/confirmatory_test_predictions.jsonl --normalizations A2/scores/normalizations.json \\
      --runtime-bundle runtime_bundle.json --e1-result E1/confirm_lead.json --amend2-result A2/confirm_hide_and_seek.json \\
      [--review-result review_analyses.json] [--nr-labels labels.json --nr-key key.json] --output OUT.json
"""
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_hide_and_seek as chs  # noqa: E402
import confirm_lead as cl  # noqa: E402
import quantile_timer as qt  # noqa: E402
import review_analyses as ra  # noqa: E402

PLAN = "docs/xr1_review_analyses_plan.md (R12)"
ALPHA = 0.05
HS = chs.PRIMARY
POLICY = ("safe", HS)
BETA_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
MU_GRID = np.geomspace(1e-4, 1e4, 4001)
SECONDS_PER_INFERENCE = 0.8          # 16 control steps at 20 Hz
NULL_DRAWS, NULL_SEED = 100, 12
BAND_MODULATION, BAND_ALIGNMENT = "tfunc", "extend"
EPS = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conformal_module():
    """robocasa/recovery/safe/conformal.py by path (numpy only), without importing the robocasa package."""
    import importlib.util
    path = here.parents[1] / "robocasa" / "recovery" / "safe" / "conformal.py"
    spec = importlib.util.spec_from_file_location("safe_conformal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def r4(x):
    return None if x is None else round(float(x), 4)


# ---- matching -------------------------------------------------------------------------------

def loosen_fast(te, rk, target_fp):
    """review_analyses.loosen_timer by bisection: the timer's false alarms and lead are monotone
    in its margin, so the best grid margin is the smallest with at most target_fp false alarms."""
    grid = ra.DELTA_GRID

    def ev(i):
        return cl.evaluate_fast(te, rk, dict(lam=float(grid[i]), u=math.inf, s=0.0))

    lo, hi = 0, len(grid) - 1
    if ev(hi)[0]["fp"] > target_fp:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ev(mid)[0]["fp"] <= target_fp:
            hi = mid
        else:
            lo = mid + 1
    e, lead = ev(lo)
    return float(grid[lo]), e, lead


def gains(te, rk, tm, fu, Fte, boot=True):
    """Fused detector against P at the same target and matched (P loosened), with test-only CIs."""
    eP, lP = cl.evaluate_fast(te, rk, tm)
    eF, lF = cl.evaluate_fast(te, rk, fu)
    loose = loosen_fast(te, rk, eF["fp"])
    out = dict(test=dict(eF, twa=ra.twa(eF["mean_lead"], eF["fpr"])), timer=dict(eP, twa=ra.twa(eP["mean_lead"], eP["fpr"])),
               gain=r4(np.mean(lF - lP)), path=cl.det_json(fu))
    if boot:
        out["gain_test_only"] = qt.boot(lF - lP, Fte)
    if loose is not None:
        out["matched"] = dict(delta=loose[0], fp=loose[1]["fp"], timer_lead=loose[1]["mean_lead"],
                              gain=r4(np.mean(lF - loose[2])))
        if boot:
            out["matched"]["gain_test_only"] = qt.boot(lF - loose[2], Fte)
    return out, lF, lP, (None if loose is None else loose[2])


# ---- B1: budget-allocated per-task timer -----------------------------------------------------

def allocate(dev, beta, mus=MU_GRID):
    """Per-task fractions maximizing mean(1 - a_k) s.t. mean_k FPR_k(a_k) <= beta on development
    successes, FPR_k(a) = share of task k's development fractions >= a. Lagrangian: each task takes
    argmin_a a + mu * FPR_k(a) (smallest a on ties), with the smallest grid mu meeting the budget."""
    tasks = sorted(dev)
    cand, fpr = {}, {}
    for k in tasks:
        f = np.sort(np.asarray(dev[k], float))
        c = np.unique(np.concatenate([[0.0], f + 1e-9]))
        cand[k] = c
        fpr[k] = np.array([np.mean(f >= x) for x in c])
    chosen = None
    for mu in mus:
        pick = {k: int(np.argmin(cand[k] + mu * fpr[k])) for k in tasks}
        mean_fpr = float(np.mean([fpr[k][pick[k]] for k in tasks]))
        if mean_fpr <= beta + EPS:
            chosen = (float(mu), pick, mean_fpr)
            break
    if chosen is None:
        mu = float(mus[-1])
        pick = {k: int(np.argmin(cand[k] + mu * fpr[k])) for k in tasks}
        chosen = (mu, pick, float(np.mean([fpr[k][pick[k]] for k in tasks])))
    mu, pick, mean_fpr = chosen
    return {k: float(min(1.0, cand[k][pick[k]])) for k in tasks}, dict(mu=mu, development_fpr=mean_fpr)


def timer_fit(cal, rk, alpha=ALPHA):
    S = cal.take(~cal.failed)
    return dict(u=math.inf, s=0.0, lam=cl.conformal(cl.V_vec(S, cl.rho_rows(S, rk), math.inf, 0.0), alpha))


def adaptive_allocation(cal, dev, alpha=ALPHA):
    """beta chosen as the fused path is: largest calibration-failure lead, each beta at its own margin."""
    F = cal.take(cal.failed)
    best = None
    for beta in BETA_GRID:
        rk, info = allocate(dev, beta)
        tm = timer_fit(cal, rk, alpha)
        if not math.isfinite(tm["lam"]):
            continue
        v = cl.fail_lead(F, cl.rho_rows(F, rk), tm["lam"], math.inf, 0.0)
        if best is None or v > best[0] + EPS:
            best = (v, beta, rk, info, tm)
    return best


# ---- B2: native functional bands -------------------------------------------------------------

def band_scores(rows, name, norms):
    """Seed-averaged per-task-normalized raw score, without the running maximum."""
    for r in rows:
        if name == "safe":
            per = {s: (np.asarray(r["per_seed_raw_safe"][s], float) - norms["safe"][s][r["task_name"]]["location"])
                   / norms["safe"][s][r["task_name"]]["scale"] for s in sorted(norms["safe"])}
        else:
            per = {s: (r["raw"][name][s] - norms[name][s][r["task_name"]]["location"])
                   / norms[name][s][r["task_name"]]["scale"] for s in sorted(norms[name])}
        r[f"{name}_band"] = {"avg": np.mean([per[s] for s in sorted(per)], axis=0), **per}


def split_successes(cal_rows):
    """Per task, successes sorted by rollout ID alternate between reference and nonconformity."""
    by_task = defaultdict(list)
    for r in cal_rows:
        if not r["failed"]:
            by_task[r["task_name"]].append(r)
    ref, conf = [], []
    for t in sorted(by_task):
        rows = sorted(by_task[t], key=lambda r: str(r["rollout_id"]))
        ref += rows[0::2]
        conf += rows[1::2]
    return ref, conf


def band_alarms(rows, calibration, key, which):
    first_detection = conformal_module().first_detection
    out = []
    for r in rows:
        i = first_detection(r[key][which], calibration)
        out.append(math.inf if i is None else float(i + 1))
    return np.asarray(out)


def alarm_metrics(rows, a):
    L = np.array([r["L"] for r in rows], float)
    H = np.array([r["horizon"] for r in rows], float)
    failed = np.array([r["failed"] for r in rows], bool)
    hit = a <= L
    lead = np.where(hit, (L - a) / H, 0.0)[failed]
    fp, n = int(hit[~failed].sum()), int((~failed).sum())
    tpr, fpr = float(hit[failed].mean()), fp / n
    return dict(tpr=r4(tpr), fpr=r4(fpr), fp=fp, mean_lead=r4(lead.mean()), bacc=r4((tpr + 1 - fpr) / 2),
                twa=ra.twa(float(lead.mean()), fpr)), lead


def band_analysis(cal, te, Ccal, Cte, rk, name, norms, Fte, which="avg"):
    calibrate_functional_threshold = conformal_module().calibrate_functional_threshold
    key = f"{name}_band"
    ref, conf = split_successes(cal)
    out = {}
    for label, alpha in (("standalone", ALPHA), ("bonferroni", ALPHA / 2)):
        calib = calibrate_functional_threshold([r[key][which] for r in ref], [r[key][which] for r in conf],
                                               alpha=alpha, modulation=BAND_MODULATION, alignment=BAND_ALIGNMENT)
        a_band = band_alarms(te, calib, key, which)
        if label == "standalone":
            m, _ = alarm_metrics(te, a_band)
            out["standalone"] = dict(m, alpha=alpha, reference=len(ref), nonconformity=len(conf))
            continue
        tm_half = timer_fit(Ccal, rk, alpha)
        a_timer = cl.alarm_vec(Cte, cl.rho_rows(Cte, rk), tm_half["lam"], math.inf, 0.0)
        m, lF = alarm_metrics(te, np.minimum(a_band, a_timer))
        tm = timer_fit(Ccal, rk, ALPHA)
        eP, lP = cl.evaluate_fast(Cte, rk, tm)
        loose = loosen_fast(Cte, rk, m["fp"])
        out["fused_bonferroni"] = dict(m, band_alpha=alpha, timer_alpha=alpha, timer_margin=r4(tm_half["lam"]),
                                       gain=r4(np.mean(lF - lP)), gain_test_only=qt.boot(lF - lP, Fte),
                                       matched=None if loose is None else dict(
                                           fp=loose[1]["fp"], gain=r4(np.mean(lF - loose[2])),
                                           gain_test_only=qt.boot(lF - loose[2], Fte)))
    return out


# ---- B4: full-pipeline bootstrap ---------------------------------------------------------------

def boot_full(C, rk, keys, B, seed, alpha=ALPHA):
    """chs.boot_multi's draws; per replicate refit P and each fused detector, re-match P."""
    rng = np.random.default_rng(seed)
    cal0, te0 = C[keys[0]]
    sc, st = cl.strata(cal0), cl.strata(te0)
    tasks = sorted(st)
    if sorted(sc) != tasks:
        raise ValueError("Calibration and test task sets differ")
    same = {k: [] for k in keys}
    matched = {k: [] for k in keys}
    for _ in range(B):
        ci, ti = [], []
        for t in rng.choice(tasks, len(tasks)):
            for idx, out in ((sc[t], ci), (st[t], ti)):
                for c in (0, 1):
                    out.append(rng.choice(idx[c], len(idx[c])))
        ci, ti = np.concatenate(ci), np.concatenate(ti)
        tm, _ = ra.fit_alpha(cal0.take(ci), rk, alpha)
        _, lp = cl.evaluate_fast(te0.take(ti), rk, tm)
        for k in keys:
            _, fu = ra.fit_alpha(C[k][0].take(ci), rk, alpha)
            teK = C[k][1].take(ti)
            eF, lf = cl.evaluate_fast(teK, rk, fu)
            same[k].append(float(np.mean(lf - lp)))
            loose = loosen_fast(teK, rk, eF["fp"])
            matched[k].append(float(np.mean(lf - loose[2])) if loose is not None else float("nan"))
    pct = lambda v: np.round(np.nanpercentile(v, [2.5, 97.5]), 4).tolist()
    out = dict(samples=B, seed=seed, same_target_ci95={k: pct(same[k]) for k in keys},
               matched_ci95={k: pct(matched[k]) for k in keys},
               matched_failed_replicates={k: int(np.sum(np.isnan(matched[k]))) for k in keys})
    if len(keys) == 2:
        a, b = keys[1], keys[0]
        out["matched_contrast_ci95"] = {f"{a}_minus_{b}": pct(np.asarray(matched[a]) - np.asarray(matched[b]))}
    return out


# ---- B5: seconds, task types, point of no return ------------------------------------------------

def seconds_summary(te_rows, fits, types):
    failed = [r for r in te_rows if r["failed"]]
    H = np.array([r["horizon"] for r in failed], float)
    ty = np.array([types[r["task_name"]] for r in failed])
    out = {}
    for name, (lead, lead_timer, lead_matched) in fits.items():
        row = {}
        for label, mask in (("all", np.ones(len(failed), bool)), ("atomic", ty == "atomic"),
                            ("composite", ty == "composite")):
            if not mask.any():
                continue
            sec = lambda v: r4(np.mean(v[mask] * H[mask] * SECONDS_PER_INFERENCE))
            row[label] = dict(failures=int(mask.sum()), lead_fraction=r4(np.mean(lead[mask])), lead_seconds=sec(lead),
                              gain_fraction=None if lead_timer is None else r4(np.mean((lead - lead_timer)[mask])),
                              gain_seconds=None if lead_timer is None else sec(lead - lead_timer),
                              matched_gain_seconds=None if lead_matched is None else sec(lead - lead_matched),
                              matched_gain_fraction=None if lead_matched is None else r4(np.mean((lead - lead_matched)[mask])))
        out[name] = row
    out["H_seconds"] = dict(min=r4(H.min() * SECONDS_PER_INFERENCE), max=r4(H.max() * SECONDS_PER_INFERENCE))
    return out


def no_return(te, Cte, rk, fits_alarm, labels_path, key_path):
    import no_return_analysis as nra
    labels = json.loads(Path(labels_path).read_text())
    labels = labels.get("labels", labels)          # the labeler's export wraps them
    key = json.loads(Path(key_path).read_text())
    by_id = {str(r["rollout_id"]): i for i, r in enumerate(te)}
    check, rows = dict(compared=0, mismatches=[]), []
    for vid, k in sorted(key["videos"].items()):
        rid = str(k["rollout_id"])
        if rid not in by_id:
            raise ValueError(f"No-return key rollout {rid} is not in the test cohort")
        i = by_id[rid]
        for name, field in (("P", "alarm_P_inference"), ("safe_plus_P", "alarm_safe_plus_P_inference")):
            a = fits_alarm[name][i]
            want = k.get(field)
            got = None if math.isinf(a) else float(a)
            check["compared"] += 1
            if got != (None if want is None else float(want)):
                check["mismatches"].append(dict(video=vid, detector=name, got=got, want=want))
        lab = labels.get(vid, {})
        if not nra.complete(lab):
            continue
        s = int(k["steps_per_inference"]); H = int(k["horizon_inferences"]) * s; L = int(k["length_inferences"]) * s
        rec = bool(lab.get("recoverable_until_timeout"))
        tu = L if rec else int(lab["t_u_env_step"])
        det = {}
        for name in fits_alarm:
            a = fits_alarm[name][i]
            step = L if math.isinf(a) else min((a - 1) * s, L)
            det[name] = dict(alarm_step=step, before_no_return=None if rec else bool(step <= tu),
                             lead_before_no_return=None if rec else (tu - step) / H,
                             lead_to_limit=(L - step) / H)
        rows.append(dict(id=vid, rollout_id=rid, task_type=k.get("task_type"), recoverable=rec, t_u=tu, H=H,
                         detectors=det))
    decided = [r for r in rows if not r["recoverable"]]
    summary = dict(labeled=len(rows), decided=len(decided), recoverable=len(rows) - len(decided))
    for name in fits_alarm:
        if decided:
            lb = [r["detectors"][name]["lead_before_no_return"] for r in decided]
            summary[name] = dict(before_no_return=sum(r["detectors"][name]["before_no_return"] for r in decided),
                                 mean_lead_before_no_return=r4(np.mean(lb)),
                                 mean_positive_lead_before_no_return=r4(np.mean(np.maximum(lb, 0.0))))
    check["ok"] = not check["mismatches"]
    return dict(summary=summary, rows=rows), check


# ---- run ------------------------------------------------------------------------------------

def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    dev = qt.dev_fractions(bundle)
    rkP = qt.rho(dev, cl.PRIMARY_Q)
    norms = json.loads(Path(args.normalizations).read_text())
    norms_all = {"safe": bundle["task_normalizations"], HS: norms[HS]}
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    if {r["rollout_id"] for r in cal_all} & {r["rollout_id"] for r in te_all}:
        raise ValueError("Calibration and test cohorts overlap")
    tasks = set(bundle["task_horizons"])
    if {r["task_name"] for r in cal} != tasks or {r["task_name"] for r in te} != tasks:
        raise ValueError("Cohorts lack exact frozen task coverage")
    chs.attach(cal, chs.load_scores(args.scores_calibration), "calibration")
    chs.attach(te, chs.load_scores(args.scores_test), "test")
    btypes = bundle.get("task_types") or {}
    types = {r["task_name"]: (r.get("task_type") or btypes.get(r["task_name"])) for r in cal + te}
    C = {k: (cl.Cohort(cal, k), cl.Cohort(te, k)) for k in POLICY}
    Fte = [r for r in te if r["failed"]]
    checks, failures = {}, []
    # G0: the stored SAFE and Hide-and-Seek trajectories equal their normalized per-seed raw scores
    # (B2 and B3 read the raw scores).
    checks["G0_pipelines"] = dict(safe=max(cl.check_safe_pipeline(cal, bundle["task_normalizations"]),
                                           cl.check_safe_pipeline(te, bundle["task_normalizations"])),
                                  hide_and_seek=max(chs.check_normalization(cal, norms), chs.check_normalization(te, norms)))

    # G1: E1 and Amendment 2 at alpha = 0.05.
    tm = ra.fit_alpha(C["safe"][0], rkP, ALPHA)[0]
    fits = {k: ra.fit_alpha(C[k][0], rkP, ALPHA)[1] for k in POLICY}
    eP, lP = cl.evaluate_fast(C["safe"][1], rkP, tm)
    res = {k: gains(C[k][1], rkP, tm, fits[k], Fte) for k in POLICY}
    e1 = json.loads(Path(args.e1_result).read_text())["by_q"][str(cl.PRIMARY_Q)]
    a2 = json.loads(Path(args.amend2_result).read_text())
    got = (eP["mean_lead"], eP["fp"], res["safe"][0]["test"]["mean_lead"], res["safe"][0]["test"]["fp"], res["safe"][0]["gain"],
           res[HS][0]["test"]["mean_lead"], res[HS][0]["test"]["fp"], res[HS][0]["gain"])
    want = (e1["P"]["mean_lead"], e1["P"]["fp"], e1["safe_plus_P"]["mean_lead"], e1["safe_plus_P"]["fp"], e1["gain"],
            a2["detectors"][HS]["test"]["mean_lead"], a2["detectors"][HS]["test"]["fp"], a2["detectors"][HS]["gain"])
    checks["G1_e1_amend2"] = dict(got=got, want=want, ok=ra.same(got, want))
    if not checks["G1_e1_amend2"]["ok"]:
        failures.append(f"G1: got {got}, want {want}")

    # G2: bisection loosening equals review_analyses.loosen_timer; R2 matched gains and CIs.
    g2 = {}
    for k in POLICY:
        slow = ra.loosen_timer(C[k][1], rkP, res[k][0]["test"]["fp"])
        fast = loosen_fast(C[k][1], rkP, res[k][0]["test"]["fp"])
        g2[k] = dict(slow=slow[0], fast=fast[0], ok=slow[0] == fast[0] and bool(np.array_equal(slow[2], fast[2])))
    if args.review_result:
        r1 = json.loads(Path(args.review_result).read_text())["R1"]
        i = r1["alphas"].index(ALPHA)
        for k in POLICY:
            ref = r1["detectors"][k][i]["matched_P"]
            mine = res[k][0]["matched"]
            g2[k]["r2"] = dict(got=[mine["gain"], mine["gain_test_only"]["ci95"]], want=[ref["gain"], ref["gain_test_only"]["ci95"]])
            g2[k]["ok"] = g2[k]["ok"] and ra.same(g2[k]["r2"]["got"], g2[k]["r2"]["want"])
    checks["G2_matching"] = g2
    if not all(v["ok"] for v in g2.values()):
        failures.append(f"G2: {g2}")

    # G3: allocation path monotone; P's fractions through the allocation evaluation give P.
    rkA, infoA = allocate(dev, ALPHA)
    mono = all(allocate(dev, b)[0][t] >= allocate(dev, b2)[0][t] - EPS
               for b, b2 in zip(BETA_GRID[:-1], BETA_GRID[1:]) for t in tasks)
    same_P = timer_fit(C["safe"][0], rkP) == tm
    checks["G3_allocation"] = dict(monotone_in_budget=mono, timer_fit_equals_P=same_P, ok=mono and same_P)
    if not checks["G3_allocation"]["ok"]:
        failures.append(f"G3: {checks['G3_allocation']}")

    # G5 (with B4): same-target CIs of the full bootstrap equal Amendment 2's decision CIs.
    b4 = None
    if args.bootstrap > 0:
        b4 = boot_full(C, rkP, list(POLICY), args.bootstrap, cl.BOOT_SEED)
        if args.bootstrap == a2.get("bootstrap", {}).get("samples"):
            want = [a2["detectors"][k]["gain_decision_ci95"] for k in POLICY]
            got = [b4["same_target_ci95"][k] for k in POLICY]
            checks["G5_decision_ci"] = dict(got=got, want=want, ok=ra.same(got, want))
            if not checks["G5_decision_ci"]["ok"]:
                failures.append(f"G5: full bootstrap same-target CIs {got} != Amendment 2 {want}")

    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in vars(args).items()
              if k in ("calibration", "test", "scores_calibration", "scores_test", "normalizations", "runtime_bundle",
                       "e1_result", "amend2_result", "review_result", "nr_labels", "nr_key") and v}
    base = dict(plan=PLAN, status="exploratory (opened cohorts)", inputs=inputs,
                counts=dict(calibration=len(cal), test=len(te), tasks=len(tasks)))
    if failures:
        out_path.write_text(json.dumps(dict(base, checks=checks, reproduction_failures=failures), indent=1,
                                       default=float) + "\n")
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(failures))
        return 1

    # B1.
    b1 = dict(A_fixed=dict(budget=ALPHA, **infoA, fractions={t: r4(v) for t, v in sorted(rkA.items())}))
    best = adaptive_allocation(C["safe"][0], dev)
    b1["A_adaptive"] = dict(budget=best[1], **best[3], fractions={t: r4(v) for t, v in sorted(best[2].items())})
    for label, rk in (("A_fixed", rkA), ("A_adaptive", best[2])):
        tmA = timer_fit(C["safe"][0], rk)
        eA, lA = cl.evaluate_fast(C["safe"][1], rk, tmA)
        row = b1[label]
        row["timer"] = dict(eA, twa=ra.twa(eA["mean_lead"], eA["fpr"]), margin=r4(tmA["lam"]),
                            gain_over_P=r4(np.mean(lA - lP)), gain_over_P_test_only=qt.boot(lA - lP, Fte))
        looseP = loosen_fast(C["safe"][1], rkP, eA["fp"])
        row["timer"]["gain_over_P_matched"] = None if looseP is None else r4(np.mean(lA - looseP[2]))
        for k in POLICY:
            _, lF, _, _ = res[k]
            looseA = loosen_fast(C[k][1], rk, res[k][0]["test"]["fp"])
            plusP = dict(gain=r4(np.mean(lF - lA)), gain_test_only=qt.boot(lF - lA, Fte),
                         matched=None if looseA is None else dict(fp=looseA[1]["fp"], gain=r4(np.mean(lF - looseA[2])),
                                                                  gain_test_only=qt.boot(lF - looseA[2], Fte)))
            fuA = ra.fit_alpha(C[k][0], rk, ALPHA)[1]
            plusA, _, _, _ = gains(C[k][1], rk, tmA, fuA, Fte)
            row[k] = dict(plus_P_against_A=plusP, plus_A_against_A=plusA)

    # B2.
    for rows in (cal, te):
        for name in POLICY:
            band_scores(rows, name, norms_all)
    b2 = {name: band_analysis(cal, te, C["safe"][0], C["safe"][1], rkP, name, norms_all, Fte) for name in POLICY}

    # B3.
    rng = np.random.default_rng(NULL_SEED)
    nulls = defaultdict(list)
    null_leads = defaultdict(list)
    for _ in range(args.null_draws):
        cl.null_scores(cal + te, rng, bundle["task_normalizations"])
        chs.resample_null(cal + te, rng, HS, norms)
        for key in ("noise", "safe_resampled", f"{HS}_resampled"):
            Cc, Ct = cl.Cohort(cal, key), cl.Cohort(te, key)
            fu = ra.fit_alpha(Cc, rkP, ALPHA)[1]
            g, lN, _, lPm = gains(Ct, rkP, tm, fu, Fte, boot=False)
            nulls[key].append(dict(gain=g["gain"], fp=g["test"]["fp"],
                                   matched=None if lPm is None else r4(np.mean(lN - lPm))))
            if lPm is not None:
                null_leads[key].append(lN - lPm)
    b3 = dict(draws=args.null_draws, seed=NULL_SEED, nulls={})
    for key, v in nulls.items():
        m = [x["matched"] for x in v if x["matched"] is not None]
        b3["nulls"][key] = dict(same_target=dict(mean=r4(np.mean([x["gain"] for x in v])), max=r4(max(x["gain"] for x in v))),
                                matched=dict(mean=r4(np.mean(m)), max=r4(max(m)), q95=r4(np.quantile(m, 0.95))),
                                mean_fp=r4(np.mean([x["fp"] for x in v])))
    for k, pair in (("safe", ("noise", "safe_resampled")), (HS, ("noise", f"{HS}_resampled"))):
        _, lF, _, lPm = res[k]
        mine = res[k][0]["matched"]["gain"]
        b3[k] = dict(matched_gain=mine)
        for key in pair:
            m = [x["matched"] for x in nulls[key] if x["matched"] is not None]
            d = (lF - lPm) - np.mean(null_leads[key], axis=0)
            b3[k][key] = dict(rank_p=r4((1 + sum(x >= mine for x in m)) / (1 + len(m))),
                              minus_mean_null=qt.boot(d, Fte))

    # B5.
    leads = {"P": (lP, None, None)}
    for k in POLICY:
        _, lF, lPk, lPm = res[k]
        leads[f"{k}_plus_P"] = (lF, lPk, lPm)
    b5 = dict(seconds=seconds_summary(te, leads, types))
    if args.nr_labels and args.nr_key:
        alarms = {"P": cl.alarm_vec(C["safe"][1], cl.rho_rows(C["safe"][1], rkP), tm["lam"], tm["u"], tm["s"])}
        for k in POLICY:
            alarms[f"{k}_plus_P"] = cl.alarm_vec(C[k][1], cl.rho_rows(C[k][1], rkP), fits[k]["lam"], fits[k]["u"], fits[k]["s"])
        b5["no_return"], g6 = no_return(te, C["safe"][1], rkP, alarms, args.nr_labels, args.nr_key)
        checks["G6_no_return_key"] = g6
        if not g6["ok"]:
            failures.append(f"G6: no-return key alarms not reproduced: {g6['mismatches'][:3]}")

    out = dict(base, checks=checks, reproduction_failures=failures,
               alpha_005=dict(P=dict(eP, twa=ra.twa(eP["mean_lead"], eP["fpr"])), **{k: res[k][0] for k in POLICY}),
               B1=b1, B2=b2, B3=b3, B4=b4, B5=b5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    if failures:
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(failures))
        return 1

    print(f"P: lead {eP['mean_lead']} FP {eP['fp']}")
    for k in POLICY:
        m = res[k][0]["matched"]
        print(f"{k} + P: gain {res[k][0]['gain']:+.4f}, matched {m['gain']:+.4f} {m['gain_test_only']['ci95']}")
    for label in ("A_fixed", "A_adaptive"):
        t = b1[label]["timer"]
        print(f"B1 {label} (budget {b1[label]['budget']}): lead {t['mean_lead']} FP {t['fp']} gain over P "
              f"{t['gain_over_P']:+.4f} (matched {t['gain_over_P_matched']})")
        for k in POLICY:
            pp, pa = b1[label][k]["plus_P_against_A"], b1[label][k]["plus_A_against_A"]
            print(f"   {k}: + P vs {label} {pp['gain']:+.4f} (matched {pp['matched'] and pp['matched']['gain']}); "
                  f"+ {label} vs {label} {pa['gain']:+.4f} (matched {(pa.get('matched') or {}).get('gain')})")
    for k in POLICY:
        s, f = b2[k]["standalone"], b2[k]["fused_bonferroni"]
        print(f"B2 {k} band: standalone TPR {s['tpr']} FPR {s['fpr']} lead {s['mean_lead']} TWA {s['twa']} | "
              f"band OR P (alpha/2 each): FPR {f['fpr']} lead {f['mean_lead']} gain {f['gain']:+.4f} "
              f"matched {f['matched'] and f['matched']['gain']}")
    for key, v in b3["nulls"].items():
        print(f"B3 null {key}: same-target mean {v['same_target']['mean']:+.4f}, matched mean {v['matched']['mean']:+.4f} "
              f"max {v['matched']['max']:+.4f} (FP {v['mean_fp']})")
    for k in POLICY:
        print(f"   {k}: " + "; ".join(f"vs {key}: rank p {b3[k][key]['rank_p']}, minus mean null "
                                     f"{b3[k][key]['minus_mean_null']['mean']:+.4f} {b3[k][key]['minus_mean_null']['ci95']}"
                                     for key in b3[k] if key != "matched_gain"))
    if b4:
        print(f"B4 full-pipeline matched CIs ({b4['samples']} replicates): {b4['matched_ci95']}; "
              f"same target {b4['same_target_ci95']}; contrast {b4.get('matched_contrast_ci95')}")
    for name, row in b5["seconds"].items():
        if name == "H_seconds":
            continue
        print(f"B5 {name}: " + "; ".join(f"{lab} lead {v['lead_seconds']} s"
                                        + ("" if v['gain_seconds'] is None else f", gain {v['gain_seconds']} s")
                                        + ("" if v['matched_gain_seconds'] is None else f" (matched {v['matched_gain_seconds']} s)")
                                        for lab, v in row.items()))
    if "no_return" in b5:
        print(f"B5 no return: {b5['no_return']['summary']}")
    print("LIST B OK")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("run")
    for name in ("--calibration", "--test", "--scores-calibration", "--scores-test", "--normalizations",
                 "--runtime-bundle", "--e1-result", "--amend2-result", "--output"):
        c.add_argument(name, required=True)
    for name in ("--review-result", "--nr-labels", "--nr-key"):
        c.add_argument(name)
    c.add_argument("--bootstrap", type=int, default=cl.BOOT_SAMPLES)
    c.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
