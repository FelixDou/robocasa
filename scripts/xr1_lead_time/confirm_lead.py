"""Confirmatory lead-time analysis registered in docs/xr1_confirmatory_amendment_1_lead_time.md.

Primary: mean warning lead of SAFE + P minus the per-task quantile timer P (q = 0.95), both
calibrated with Proposition 1 on the enlarged calibration cohort (10 successes and 10 failures
per task, fresh seeds). The decision CI comes from a bootstrap that resamples tasks once, then
rollouts within each task and class, in the calibration and test cohorts together, and refits
both detectors in every replicate. Implementation details fixed before scoring are in
docs/xr1_lead_time_e1_implementation_note.md.

The detector definitions are those of quantile_timer.py (registered). `fit_fast` and
`evaluate_fast` vectorise them for the bootstrap; `reproduce` checks on the opened August cohorts
that they match quantile_timer.json and quantile_timer.fit exactly.

Usage:
  python3 -I confirm_lead.py reproduce <review_data dir>
  python3 -I confirm_lead.py run --calibration CAL.jsonl --test TEST.jsonl \
      --runtime-bundle runtime_bundle.json --fused-calibration baseline_calibration.json \
      [--confirm-predictions confirm/event_predictions.jsonl] --output OUT.json
"""
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import lead_time
import quantile_timer as qt

ALPHA = 0.05
PRIMARY_Q = 0.95
QS = (0.75, 0.90, 0.95, 1.00)
S_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
EPS = qt.EPS
PER_CLASS = 10
MARGIN = 0.05            # H1 point-gain margin and H2 bound, as fractions of H_k
FPR_MAX, FPR_WILSON_MAX = 0.075, 0.10
BOOT_SAMPLES, BOOT_SEED, NULL_DRAWS, NULL_SEED = 2000, 0, 20, 1


# ---- cohorts as arrays -------------------------------------------------------------------

class Cohort:
    """Rows of one cohort with a padded running-max matrix for one score key."""

    def __init__(self, rows, key="safe"):
        self.rows = rows
        self.task = np.array([r["task_name"] for r in rows])
        self.L = np.array([r["L"] for r in rows], float)
        self.H = np.array([r["horizon"] for r in rows], float)
        self.failed = np.array([r["failed"] for r in rows], bool)
        n, width = len(rows), max(r["L"] for r in rows)
        self.R = np.full((n, width), -np.inf)
        for i, r in enumerate(rows):
            self.R[i, : r["L"]] = np.maximum.accumulate(np.asarray(r[key], float))
        self.M = self.R.max(1)

    def take(self, idx):
        out = object.__new__(Cohort)
        out.rows = None
        for name in ("task", "L", "H", "failed", "R", "M"):
            setattr(out, name, getattr(self, name)[idx])
        return out


def rho_rows(c, rk):
    return np.array([rk[t] for t in c.task])


def timer_idx(c, rho, delta):
    cap = np.minimum(1.0, rho + delta)
    return np.ceil(cap * c.H - EPS)


def first_vec(c, tau):
    hit = c.R >= tau
    return np.where(hit.any(1), hit.argmax(1) + 1.0, np.inf)


def alarm_vec(c, rho, lam, u, s):
    a = timer_idx(c, rho, lam)
    if u != math.inf:
        a = np.minimum(a, first_vec(c, u + s * lam))
    return a


def V_vec(c, rho, u, s):
    v_t = c.L / c.H - rho
    if u == math.inf:
        v_s = np.full(len(c.L), -np.inf)
    elif s == 0:
        v_s = np.where(u > c.M, -np.inf, np.inf)
    else:
        v_s = (c.M - u) / s
    v = np.maximum(np.maximum(v_t, v_s), 0.0) + EPS
    return np.where(c.L >= c.H, np.inf, v)


def conformal(v, alpha=ALPHA):
    v = np.sort(v); n = len(v); k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else float(v[k - 1])


def fail_lead(c, rho, lam, u, s):
    a = alarm_vec(c, rho, lam, u, s)
    return float(np.mean(np.where(a <= c.L, (c.L - a) / c.H, 0.0)))


def fit_fast(cal, rk):
    """quantile_timer.fit, vectorised: same grid, order, tie-breaking and epsilon."""
    S, F = cal.take(~cal.failed), cal.take(cal.failed)
    rS, rF = rho_rows(S, rk), rho_rows(F, rk)
    t_lam = conformal(V_vec(S, rS, math.inf, 0.0))
    us = sorted(set(np.round(np.quantile(F.M, np.linspace(0, 1, 21)), 4).tolist()))
    best = (fail_lead(F, rF, t_lam, math.inf, 0.0), math.inf, 0.0, t_lam)
    for u in us:
        for s in S_GRID:
            lam = conformal(V_vec(S, rS, u, s))
            if not math.isfinite(lam):
                continue
            v = fail_lead(F, rF, lam, u, s)
            if v > best[0] + 1e-12:
                best = (v, u, s, lam)
    return dict(u=math.inf, s=0.0, lam=t_lam), dict(u=best[1], s=best[2], lam=best[3])


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def evaluate_fast(te, rk, det):
    rho = rho_rows(te, rk)
    a = alarm_vec(te, rho, det["lam"], det["u"], det["s"])
    hit = a <= te.L
    S, F = ~te.failed, te.failed
    lead = np.where(hit, (te.L - a) / te.H, 0.0)[F]
    frac = np.where(hit, a / te.H, 1.0)[F]
    fp = int(hit[S].sum()); n = int(S.sum())
    return dict(fpr=round(fp / n, 4), fp=fp, fpr_wilson95=wilson(fp, n), recall=round(float(hit[F].mean()), 4),
                mean_lead=round(float(lead.mean()), 4), by50=round(float(np.mean(frac <= 0.5)), 4),
                by75=round(float(np.mean(frac <= 0.75)), 4),
                alarms_at_final_inference=int(np.sum(a[F] == te.L[F]))), lead


def det_json(d):
    return dict(lam=round(d["lam"], 4), u=d["u"], s=d["s"])


# ---- bootstrap ----------------------------------------------------------------------------

def strata(c):
    out = defaultdict(lambda: ([], []))
    for i, (t, f) in enumerate(zip(c.task, c.failed)):
        out[t][int(f)].append(i)
    return out


def boot_reselect(cal, te, rk, B=BOOT_SAMPLES, seed=BOOT_SEED):
    """Tasks resampled once per replicate for both cohorts; rollouts within task and class."""
    rng = np.random.default_rng(seed)
    sc, st = strata(cal), strata(te)
    tasks = sorted(st)
    if sorted(sc) != tasks:
        raise ValueError("Calibration and test task sets differ")
    gains, leads_p, leads_f = [], [], []
    for _ in range(B):
        ci, ti = [], []
        for t in rng.choice(tasks, len(tasks)):
            for idx, out in ((sc[t], ci), (st[t], ti)):
                for k in (0, 1):
                    out.append(rng.choice(idx[k], len(idx[k])))
        c, e = cal.take(np.concatenate(ci)), te.take(np.concatenate(ti))
        tm, fu = fit_fast(c, rk)
        _, lp = evaluate_fast(e, rk, tm); _, lf = evaluate_fast(e, rk, fu)
        gains.append(float(np.mean(lf - lp))); leads_p.append(float(lp.mean())); leads_f.append(float(lf.mean()))
    pct = lambda v: np.round(np.percentile(v, [2.5, 97.5]), 4).tolist()
    return dict(samples=B, seed=seed, gain_ci95=pct(gains), P_lead_ci95=pct(leads_p), fused_lead_ci95=pct(leads_f))


# ---- null scores (H2) ----------------------------------------------------------------------

def null_scores(rows, rng, norms):
    """Noise and within-task resampled SAFE, as in review_controls.add_null_scores."""
    seeds = sorted(norms)
    pools = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for s in seeds:
            pools[r["task_name"]][s].append(np.asarray(r["per_seed_raw_safe"][s], float))
    pools = {t: {s: np.concatenate(v) for s, v in d.items()} for t, d in pools.items()}
    for r in rows:
        n = r["L"]
        r["noise"] = np.maximum.accumulate(rng.standard_normal(n))
        vals = []
        for s in seeds:
            raw = rng.choice(pools[r["task_name"]][s], size=n, replace=True)
            st = norms[s][r["task_name"]]
            vals.append((np.maximum.accumulate(raw) - st["location"]) / st["scale"])
        r["safe_resampled"] = np.mean(vals, axis=0)


def check_safe_pipeline(rows, norms):
    seeds = sorted(norms); worst = 0.0
    for r in rows:
        vals = [(np.maximum.accumulate(np.asarray(r["per_seed_raw_safe"][s], float)) - norms[s][r["task_name"]]["location"])
                / norms[s][r["task_name"]]["scale"] for s in seeds]
        worst = max(worst, float(np.max(np.abs(np.mean(vals, axis=0) - r["safe"]))))
    if worst > 1e-9:
        raise ValueError(f"safe_scores do not match the frozen normalisation of per_seed_raw_safe (max diff {worst})")
    return worst


# ---- I/O -----------------------------------------------------------------------------------

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def primary(path, label):
    rows = lead_time.load(Path(path))
    ids = [r["rollout_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate rollout IDs")
    chosen = lead_time.sel(rows, PER_CLASS)
    flagged = {r["rollout_id"] for r in rows if r.get("primary_balanced")}
    if flagged and flagged != {r["rollout_id"] for r in chosen}:
        raise ValueError(f"{label}: first-{PER_CLASS}-per-class selection differs from the primary_balanced flag")
    counts = defaultdict(lambda: [0, 0])
    for r in chosen:
        counts[r["task_name"]][int(r["failed"])] += 1
    short = {t: c for t, c in counts.items() if c != [PER_CLASS, PER_CLASS]}
    if short:
        raise ValueError(f"{label}: tasks without {PER_CLASS}/{PER_CLASS}: {short}")
    return rows, chosen


def identities(rows):
    return {(r["task_name"], r.get("environment_seed"), r.get("environment_reset_index")) for r in rows}


# ---- run -----------------------------------------------------------------------------------

def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    bundle = json.load(open(args.runtime_bundle))
    norms = bundle["task_normalizations"]
    dev = qt.dev_fractions(bundle)
    cal_all, cal = primary(args.calibration, "calibration")
    te_all, te = primary(args.test, "test")
    if {r["rollout_id"] for r in cal_all} & {r["rollout_id"] for r in te_all} or identities(cal_all) & identities(te_all):
        raise ValueError("Calibration and test cohorts overlap")
    if {r["task_name"] for r in cal} != set(bundle["task_horizons"]) or {r["task_name"] for r in te} != set(bundle["task_horizons"]):
        raise ValueError("Cohorts lack exact frozen task coverage")
    checks = dict(safe_pipeline_max_abs_diff=max(check_safe_pipeline(cal, norms), check_safe_pipeline(te, norms)),
                  test_failures_end_at_H=all(r["L"] == r["horizon"] for r in te if r["failed"]))
    if args.confirm_predictions:
        ref = {r["rollout_id"]: r for r in lead_time.load(Path(args.confirm_predictions))}
        for r in te_all:
            o = ref.get(r["rollout_id"])
            if o is None or o["failed"] != r["failed"] or not (np.allclose(o["safe"], r["safe"], atol=1e-9)
                                                               and np.allclose(o["risk"], r["risk"], atol=1e-9)):
                raise ValueError(f"Test predictions differ from the confirm run for {r['rollout_id']}")
        if set(ref) != {r["rollout_id"] for r in te_all}:
            raise ValueError("Test predictions and the confirm run cover different rollouts")
        checks["matches_confirm_predictions"] = True

    calC, teC = Cohort(cal), Cohort(te)
    Fte = [r for r in te if r["failed"]]
    out = dict(plan="docs/xr1_confirmatory_amendment_1_lead_time.md",
               note="docs/xr1_lead_time_e1_implementation_note.md", alpha=ALPHA, primary_q=PRIMARY_Q,
               inputs={k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in
                       (("calibration", args.calibration), ("test", args.test), ("runtime_bundle", args.runtime_bundle),
                        ("fused_calibration", args.fused_calibration))},
               counts=dict(calibration=len(cal), calibration_all_retained=len(cal_all), test=len(te),
                           test_all_retained=len(te_all), tasks=len({r["task_name"] for r in te})),
               checks=checks, by_q={})

    # Primary and quantile sensitivity.
    for q in QS:
        rk = qt.rho(dev, q)
        tm, fu = fit_fast(calC, rk)
        eP, lP = evaluate_fast(teC, rk, tm); eF, lF = evaluate_fast(teC, rk, fu)
        row = dict(P=dict(**det_json(tm), **eP), safe_plus_P=dict(**det_json(fu), **eF),
                   gain=round(float(np.mean(lF - lP)), 4), gain_test_only_bootstrap=qt.boot(lF - lP, Fte))
        if q == PRIMARY_Q:
            primary_fit = (rk, tm, fu, eF, lP, lF)
            row["gain_reselect_bootstrap"] = boot_reselect(calC, teC, rk, args.bootstrap, BOOT_SEED)
        out["by_q"][str(q)] = row
        print(f"q={q}: P {eP['mean_lead']} (FPR {eP['fpr']})  SAFE+P {eF['mean_lead']} (FPR {eF['fpr']})  gain {row['gain']}", flush=True)

    rk, tm, fu, eF, lP, lF = primary_fit
    pr = out["by_q"][str(PRIMARY_Q)]
    ci = pr["gain_reselect_bootstrap"]["gain_ci95"]
    h1 = dict(ci_lower_above_0=ci[0] > 0, point_gain_at_least_margin=pr["gain"] >= MARGIN,
              fused_fpr_at_most_0p075=eF["fpr"] <= FPR_MAX, fused_fpr_wilson_upper_at_most_0p10=eF["fpr_wilson95"][1] <= FPR_WILSON_MAX)
    h1["supported"] = all(h1.values())

    # H2: null scores fused with P, calibrated exactly like SAFE + P.
    rng = np.random.default_rng(NULL_SEED); nulls = {"noise": [], "safe_resampled": []}
    for _ in range(args.null_draws):
        null_scores(cal, rng, norms); null_scores(te, rng, norms)
        for key in nulls:
            c, e = Cohort(cal, key), Cohort(te, key)
            _, f = fit_fast(c, rk)
            ef, l = evaluate_fast(e, rk, f)
            nulls[key].append(dict(gain=round(float(np.mean(l - lP)), 4), fpr=ef["fpr"], mean_lead=ef["mean_lead"]))
    h2 = {k: dict(draws=len(v), mean_gain=round(float(np.mean([d["gain"] for d in v])), 4),
                  max_gain=max(d["gain"] for d in v), mean_fpr=round(float(np.mean([d["fpr"] for d in v])), 4),
                  per_draw=v) for k, v in nulls.items()}
    h2_ok = all(h2[k]["mean_gain"] < MARGIN for k in h2)

    # Secondary detectors (descriptive).
    sec = {}
    for name, keys in (("global_timer", ["timer"]), ("safe_plus_global_timer", ["safe", "timer"]),
                       ("risk_plus_global_timer", ["risk", "timer"]), ("safe_plus_risk_joint_search", ["safe", "risk"])):
        th = lead_time.search(cal, keys)
        e = lead_time.ev(te, th)
        sec[name] = dict(thresholds={k: float(v) for k, v in th.items()}, test=lead_time.strip(e),
                         fpr_wilson95=wilson(e["fp"], sum(not r["failed"] for r in te)),
                         gain_vs_P=qt.boot(e["lead_vec"] - lP, Fte))
    fused = json.load(open(args.fused_calibration))["detectors"]["fused_safe_time"]["thresholds"]
    # prospective_matched_fpr names the fused sources early_safe / late_time; for
    # fused_safe_time both span the whole rollout (SAFE and time risk, alarm on >=).
    keys = {"early_safe": "safe", "late_time": "risk"} if set(fused) == {"early_safe", "late_time"} else \
           {"score": "safe", "time": "risk"} if set(fused) == {"score", "time"} else None
    if keys is None:
        raise ValueError(f"Unexpected frozen fused threshold keys: {sorted(fused)}")
    e = lead_time.ev(te, {keys[k]: float(v) for k, v in fused.items()})
    sec["safe_plus_risk_frozen"] = dict(thresholds=fused, test=lead_time.strip(e),
                                        fpr_wilson95=wilson(e["fp"], sum(not r["failed"] for r in te)),
                                        gain_vs_P=qt.boot(e["lead_vec"] - lP, Fte))
    # Timers tuned on the test cohort to the fused detector's test FPR (favours the timer).
    best_g = best_p = None
    for c in np.arange(0.5, 1.0001, 0.0025):
        e = lead_time.ev(te, {"timer": c})
        if e["fpr"] <= eF["fpr"] + 1e-12 and (best_g is None or e["mean_lead"] > best_g[1]["mean_lead"]):
            best_g = (float(c), e)
    for d in np.arange(-0.5, 1.0001, 0.0025):
        e, l = evaluate_fast(teC, rk, dict(lam=float(d), u=math.inf, s=0.0))
        if e["fpr"] <= eF["fpr"] + 1e-12 and (best_p is None or e["mean_lead"] > best_p[1]["mean_lead"]):
            best_p = (float(d), e, l)
    sec["global_timer_tuned_on_test"] = dict(c=best_g[0], test=lead_time.strip(best_g[1]),
                                             fused_gain=qt.boot(lF - best_g[1]["lead_vec"], Fte))
    sec["P_tuned_on_test"] = dict(delta=best_p[0], test=best_p[1], fused_gain=qt.boot(lF - best_p[2], Fte))

    strata_out = {}
    for tt in sorted({r.get("task_type") for r in te}):
        idx = np.array([r.get("task_type") == tt for r in te])
        c = teC.take(idx); eP_, lP_ = evaluate_fast(c, rk, tm); eF_, lF_ = evaluate_fast(c, rk, fu)
        strata_out[str(tt)] = dict(P=eP_, safe_plus_P=eF_, gain=round(float(np.mean(lF_ - lP_)), 4))

    out.update(H1=h1, H2=dict(nulls=h2, supported=h2_ok), secondary=sec, by_task_type=strata_out,
               original_endpoint="reported by run_offline_detector_baselines confirm (confirm/analysis.json)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(dict(H1=h1, primary=dict(P=pr["P"]["mean_lead"], safe_plus_P=pr["safe_plus_P"]["mean_lead"],
                                               gain=pr["gain"], ci95=ci),
                          H2={k: (v["mean_gain"], v["max_gain"]) for k, v in h2.items()}, H2_supported=h2_ok), indent=1))
    print("H1:", "SUPPORTED" if h1["supported"] else "NOT SUPPORTED", "| H2:", "SUPPORTED" if h2_ok else "NOT SUPPORTED")


# ---- reproduce -----------------------------------------------------------------------------

def reproduce(D, replicates=25):
    """Fast path == registered quantile_timer on the opened August cohorts (dry run).

    D holds the August calibration_predictions.jsonl, event_predictions.jsonl and
    runtime_bundle.json, selected exactly as lead_time.run selects them.
    """
    D = Path(D)
    cal = lead_time.sel(lead_time.load(D / "calibration_predictions.jsonl"), 3)
    te = lead_time.sel(lead_time.load(D / "event_predictions.jsonl"), 10)
    ref = json.load(open(here / "quantile_timer.json"))
    dev = qt.dev_fractions(json.load(open(D / "runtime_bundle.json")))
    calC, teC = Cohort(cal), Cohort(te)
    ok = True
    for q in QS:
        rk = qt.rho(dev, q); tm, fu = fit_fast(calC, rk)
        eP, _ = evaluate_fast(teC, rk, tm); eF, _ = evaluate_fast(teC, rk, fu)
        r = ref["by_q"][str(q)]
        got = dict(timer=(round(tm["lam"], 4), eP["fpr"], eP["fp"], eP["mean_lead"], eP["recall"], eP["alarms_at_final_inference"]),
                   fused=(round(fu["lam"], 4), fu["u"], fu["s"], eF["fpr"], eF["fp"], eF["mean_lead"], eF["recall"], eF["alarms_at_final_inference"]))
        want = dict(timer=tuple(r["timer"][k] for k in ("delta", "fpr", "fp", "mean_lead", "recall", "alarms_at_final_inference")),
                    fused=tuple(r["safe_plus_timer"][k] for k in ("delta", "u", "s", "fpr", "fp", "mean_lead", "recall", "alarms_at_final_inference")))
        same = got == want; ok &= same
        print(f"q={q} matches quantile_timer.json: {same}" + ("" if same else f"\n  got  {got}\n  want {want}"))
    # Random resamples with duplicates: fast fit == reference fit, rollout by rollout.
    rng = np.random.default_rng(123); rk = qt.rho(dev, PRIMARY_Q); n_same = 0
    for _ in range(replicates):
        idx = rng.choice(len(cal), len(cal)); rows = [cal[i] for i in idx]
        _, tm_r, fu_r = qt.fit(rows, dev, PRIMARY_Q)
        tm_f, fu_f = fit_fast(calC.take(idx), rk)
        same = (tm_r["lam"] == tm_f["lam"] and (fu_r["u"], fu_r["s"], fu_r["lam"]) == (fu_f["u"], fu_f["s"], fu_f["lam"]))
        if same:
            _, l_r = qt.evaluate(te, rk, fu_r["lam"], fu_r["u"], fu_r["s"])
            _, l_f = evaluate_fast(teC, rk, fu_f)
            same = np.array_equal(l_r, l_f)
        n_same += same
    print(f"resampled calibration fits identical to quantile_timer.fit: {n_same}/{replicates}")
    ok &= n_same == replicates
    if not ok:
        sys.exit("REPRODUCTION FAILED")
    print("REPRODUCTION OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce"); r.add_argument("review_data")
    r.add_argument("--replicates", type=int, default=25)
    c = sub.add_parser("run")
    for name in ("--calibration", "--test", "--runtime-bundle", "--fused-calibration", "--output"):
        c.add_argument(name, required=True)
    c.add_argument("--confirm-predictions")
    c.add_argument("--bootstrap", type=int, default=BOOT_SAMPLES)
    c.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    a = p.parse_args()
    if a.cmd == "reproduce":
        reproduce(a.review_data, a.replicates)
    else:
        run(a)


if __name__ == "__main__":
    main()
