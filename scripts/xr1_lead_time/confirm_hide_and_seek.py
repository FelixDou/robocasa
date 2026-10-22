"""Amendment 2 analysis: Hide-and-Seek + P against the per-task timer P.

Registered in docs/xr1_confirmatory_amendment_2_hide_and_seek.md. Everything about the
detectors, the timer and the decision rule is taken from confirm_lead.py (Amendment 1,
E1): the same P (q = 0.95, conformal margin on calibration successes), the same nested
fused path (u, s) chosen on calibration failures, the same decision bootstrap (tasks drawn
once for both cohorts, rollouts within task and class, both detectors refitted in every
replicate, 2,000 replicates, seed 0) and the same margin and FPR limits. Only the score is
new: the frozen-protocol Hide-and-Seek trajectories written by
robocasa.recovery.safe.baselines.score_cohorts.

Modes:
  run      the confirmatory analysis on the E1 cohorts (lead calibration 380/380,
           confirmatory test 380/380). It first recomputes SAFE + P and must reproduce the
           registered E1 result (confirm_lead.json) exactly, then reports H1, H2 and the
           declared secondaries.
  dry-run  the same analysis on the opened August cohorts (calibration 114/114, test
           380/380). SAFE + P must reproduce quantile_timer.json. Descriptive only.

Usage:
  python3 -I confirm_hide_and_seek.py run --calibration E1_OUT/calibration_predictions.jsonl \\
      --test E1_OUT/test_predictions.jsonl --scores-calibration SCORES/lead_calibration_predictions.jsonl \\
      --scores-test SCORES/confirmatory_test_predictions.jsonl --normalizations SCORES/normalizations.json \\
      --runtime-bundle runtime_bundle.json --e1-result E1_OUT/confirm_lead.json \\
      --confirm-predictions confirm/event_predictions.jsonl --output OUT.json
  python3 -I confirm_hide_and_seek.py dry-run --calibration AUG/calibration_predictions.jsonl \\
      --test AUG/event_predictions.jsonl --scores-calibration SCORES/august_calibration_predictions.jsonl \\
      --scores-test SCORES/august_test_predictions.jsonl --normalizations SCORES/normalizations.json \\
      --runtime-bundle runtime_bundle.json --output OUT.json
"""
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl
import lead_time
import quantile_timer as qt

PRIMARY = "hide_and_seek"
FIRST, ZERO, SHUFFLE = "hide_and_seek_first", "hide_and_seek_zero", "hide_and_seek_task_shuffle"
CONTROLS = (FIRST, ZERO, SHUFFLE)
SCORES = ("safe", PRIMARY) + CONTROLS
CONTRASTS = ((PRIMARY, FIRST), (PRIMARY, ZERO), (PRIMARY, SHUFFLE), (PRIMARY, "safe"))
PLAN = "docs/xr1_confirmatory_amendment_2_hide_and_seek.md"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scores(path):
    out = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rid = str(row["rollout_id"])
            if rid in out[row["detector"]]:
                raise ValueError(f"{path}: duplicate {row['detector']} row for {rid}")
            out[row["detector"]][rid] = row
    return dict(out)


def attach(rows, scores, label):
    """Copy each detector's trajectory onto the SAFE rows; refuse any misalignment."""
    missing = [n for n in (PRIMARY,) + CONTROLS if n not in scores]
    if missing:
        raise ValueError(f"{label}: score file lacks detectors {missing}")
    for r in rows:
        rid = str(r["rollout_id"])
        r["raw"] = {}
        for name in (PRIMARY,) + CONTROLS:
            row = scores[name].get(rid)
            if row is None:
                raise ValueError(f"{label}: no {name} trajectory for {rid}")
            if bool(row["failed"]) != r["failed"] or row["task_name"] != r["task_name"]:
                raise ValueError(f"{label}: label or task differs for {rid} ({name})")
            v = np.asarray(row["safe_scores"], float)
            if len(v) != r["L"] or not np.all(np.isfinite(v)):
                raise ValueError(f"{label}: {name} has {len(v)} inferences for {rid}, SAFE has {r['L']}")
            if not np.allclose(np.asarray(row["time_scores"], float), r["risk"], atol=1e-9):
                raise ValueError(f"{label}: frozen time risk differs for {rid} ({name})")
            r[name] = v
            r["raw"][name] = {s: np.asarray(x, float) for s, x in row["per_seed_raw"].items()}


def check_normalization(rows, norms):
    """safe_scores of each detector must equal its normalized seed-averaged raw scores."""
    worst = 0.0
    for r in rows:
        for name in (PRIMARY,) + CONTROLS:
            seeds = sorted(norms[name])
            vals = [(np.maximum.accumulate(r["raw"][name][s]) - norms[name][s][r["task_name"]]["location"])
                    / norms[name][s][r["task_name"]]["scale"] for s in seeds]
            worst = max(worst, float(np.max(np.abs(np.mean(vals, axis=0) - r[name]))))
    if worst > 1e-9:
        raise ValueError(f"Detector trajectories do not match their normalized raw scores (max diff {worst})")
    return worst


def resample_null(rows, rng, name, norms):
    """Within-task resampled scores, as confirm_lead.null_scores does for SAFE."""
    seeds = sorted(norms[name])
    pools = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for s in seeds:
            pools[r["task_name"]][s].append(r["raw"][name][s])
    pools = {t: {s: np.concatenate(v) for s, v in d.items()} for t, d in pools.items()}
    for r in rows:
        vals = []
        for s in seeds:
            raw = rng.choice(pools[r["task_name"]][s], size=r["L"], replace=True)
            st = norms[name][s][r["task_name"]]
            vals.append((np.maximum.accumulate(raw) - st["location"]) / st["scale"])
        r[f"{name}_resampled"] = np.mean(vals, axis=0)


def boot_multi(cal, te, rk, keys, pairs, B=cl.BOOT_SAMPLES, seed=cl.BOOT_SEED):
    """confirm_lead.boot_reselect for several scores on the same replicates.

    The draw sequence is boot_reselect's, so each key's gain CI equals what
    boot_reselect gives for that key alone; contrasts are per-replicate differences.
    """
    rng = np.random.default_rng(seed)
    cohorts = {k: (cl.Cohort(cal, k), cl.Cohort(te, k)) for k in keys}
    sc, st = cl.strata(cohorts[keys[0]][0]), cl.strata(cohorts[keys[0]][1])
    tasks = sorted(st)
    if sorted(sc) != tasks:
        raise ValueError("Calibration and test task sets differ")
    gains = {k: [] for k in keys}
    for _ in range(B):
        ci, ti = [], []
        for t in rng.choice(tasks, len(tasks)):
            for idx, out in ((sc[t], ci), (st[t], ti)):
                for k in (0, 1):
                    out.append(rng.choice(idx[k], len(idx[k])))
        ci, ti = np.concatenate(ci), np.concatenate(ti)
        lp = None
        for k in keys:
            c, e = cohorts[k][0].take(ci), cohorts[k][1].take(ti)
            tm, fu = cl.fit_fast(c, rk)
            if lp is None:
                _, lp = cl.evaluate_fast(e, rk, tm)
            _, lf = cl.evaluate_fast(e, rk, fu)
            gains[k].append(float(np.mean(lf - lp)))
    pct = lambda v: np.round(np.percentile(v, [2.5, 97.5]), 4).tolist()
    out = {"samples": B, "seed": seed,
           "gain_ci95": {k: pct(v) for k, v in gains.items()},
           "gain_sd": {k: round(float(np.std(v)), 4) for k, v in gains.items()},
           "contrast_ci95": {}}
    for a, b in pairs:
        out["contrast_ci95"][f"{a}_minus_{b}"] = pct(np.asarray(gains[a]) - np.asarray(gains[b]))
    return out


def select(args):
    if args.cmd == "run":
        cal_all, cal = cl.primary(args.calibration, "calibration")
        te_all, te = cl.primary(args.test, "test")
    else:
        cal_all = lead_time.load(Path(args.calibration)); cal = lead_time.sel(cal_all, 3)
        te_all = lead_time.load(Path(args.test)); te = lead_time.sel(te_all, 10)
    return cal_all, cal, te_all, te


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    bundle = json.load(open(args.runtime_bundle))
    dev = qt.dev_fractions(bundle)
    norms = json.load(open(args.normalizations))
    cal_all, cal, te_all, te = select(args)
    if {r["rollout_id"] for r in cal_all} & {r["rollout_id"] for r in te_all} or cl.identities(cal_all) & cl.identities(te_all):
        raise ValueError("Calibration and test cohorts overlap")
    tasks = set(bundle["task_horizons"])
    if {r["task_name"] for r in cal} != tasks or {r["task_name"] for r in te} != tasks:
        raise ValueError("Cohorts lack exact frozen task coverage")
    attach(cal, load_scores(args.scores_calibration), "calibration")
    attach(te, load_scores(args.scores_test), "test")
    checks = dict(normalization_max_abs_diff=max(check_normalization(cal, norms), check_normalization(te, norms)),
                  test_failures_end_at_H=all(r["L"] == r["horizon"] for r in te if r["failed"]))
    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in
              (("calibration", args.calibration), ("test", args.test), ("scores_calibration", args.scores_calibration),
               ("scores_test", args.scores_test), ("normalizations", args.normalizations),
               ("runtime_bundle", args.runtime_bundle))}
    failures = []
    if args.cmd == "run":
        ref = {r["rollout_id"]: r for r in lead_time.load(Path(args.confirm_predictions))}
        if set(ref) != {r["rollout_id"] for r in te_all}:
            raise ValueError("Test predictions and the confirm run cover different rollouts")
        for r in te_all:
            o = ref[r["rollout_id"]]
            if o["failed"] != r["failed"] or not np.allclose(o["safe"], r["safe"], atol=1e-9):
                raise ValueError(f"Test predictions differ from the confirm run for {r['rollout_id']}")
        checks["matches_confirm_predictions"] = True
        inputs["e1_result"] = dict(path=str(Path(args.e1_result).resolve()), sha256=sha256(args.e1_result))
        inputs["confirm_predictions"] = dict(path=str(Path(args.confirm_predictions).resolve()),
                                             sha256=sha256(args.confirm_predictions))

    rk = qt.rho(dev, cl.PRIMARY_Q)
    C = {k: (cl.Cohort(cal, k), cl.Cohort(te, k)) for k in SCORES}
    Fte = [r for r in te if r["failed"]]
    tm, _ = cl.fit_fast(C["safe"][0], rk)
    eP, lP = cl.evaluate_fast(C["safe"][1], rk, tm)
    rows, leads = {}, {}
    for k in SCORES:
        _, fu = cl.fit_fast(C[k][0], rk)
        eF, lF = cl.evaluate_fast(C[k][1], rk, fu)
        leads[k] = lF
        rows[k] = dict(path=cl.det_json(fu), test=eF, gain=round(float(np.mean(lF - lP)), 4),
                       gain_test_only_bootstrap=qt.boot(lF - lP, Fte))
    boot = boot_multi(cal, te, rk, list(SCORES), CONTRASTS, args.bootstrap, cl.BOOT_SEED)
    for k in SCORES:
        rows[k]["gain_decision_ci95"] = boot["gain_ci95"][k]
    print(f"P {eP['mean_lead']} (FPR {eP['fpr']})", flush=True)
    for k in SCORES:
        print(f"{k:28s} lead {rows[k]['test']['mean_lead']} (FPR {rows[k]['test']['fpr']})  gain {rows[k]['gain']:+.4f} "
              f"CI {rows[k]['gain_decision_ci95']}", flush=True)

    # Reference: SAFE + P must equal the registered result on the same cohorts.
    if args.cmd == "run":
        e1 = json.load(open(args.e1_result))["by_q"][str(cl.PRIMARY_Q)]
        got = (eP["mean_lead"], rows["safe"]["test"]["mean_lead"], rows["safe"]["gain"], boot["gain_ci95"]["safe"])
        want = (e1["P"]["mean_lead"], e1["safe_plus_P"]["mean_lead"], e1["gain"], e1["gain_reselect_bootstrap"]["gain_ci95"])
    else:
        ref = json.load(open(here / "quantile_timer.json"))["by_q"][str(cl.PRIMARY_Q)]
        got = (eP["mean_lead"], eP["fp"], rows["safe"]["test"]["mean_lead"], rows["safe"]["test"]["fp"])
        want = (ref["timer"]["mean_lead"], ref["timer"]["fp"], ref["safe_plus_timer"]["mean_lead"], ref["safe_plus_timer"]["fp"])
    checks["safe_reference"] = dict(got=got, want=want, ok=list(got) == list(want))
    if list(got) != list(want):
        failures.append(f"SAFE + P reference: got {got}, want {want}")

    # H1, as Amendment 1 with Hide-and-Seek in place of SAFE.
    pr = rows[PRIMARY]
    ci = pr["gain_decision_ci95"]
    h1 = dict(ci_lower_above_0=ci[0] > 0, point_gain_at_least_margin=pr["gain"] >= cl.MARGIN,
              fused_fpr_at_most_0p075=pr["test"]["fpr"] <= cl.FPR_MAX,
              fused_fpr_wilson_upper_at_most_0p10=pr["test"]["fpr_wilson95"][1] <= cl.FPR_WILSON_MAX)
    h1["supported"] = all(h1.values())

    # H2: noise and within-task resampled Hide-and-Seek, fitted exactly like Hide-and-Seek + P.
    rng = np.random.default_rng(cl.NULL_SEED)
    nulls = {"noise": [], f"{PRIMARY}_resampled": []}
    for _ in range(args.null_draws):
        for rws in (cal, te):
            for r in rws:
                r["noise"] = np.maximum.accumulate(rng.standard_normal(r["L"]))
        resample_null(cal, rng, PRIMARY, norms); resample_null(te, rng, PRIMARY, norms)
        for key in nulls:
            _, f = cl.fit_fast(cl.Cohort(cal, key), rk)
            ef, l = cl.evaluate_fast(cl.Cohort(te, key), rk, f)
            nulls[key].append(dict(gain=round(float(np.mean(l - lP)), 4), fpr=ef["fpr"], mean_lead=ef["mean_lead"]))
    h2 = {k: dict(draws=len(v), mean_gain=round(float(np.mean([d["gain"] for d in v])), 4),
                  max_gain=max(d["gain"] for d in v), mean_fpr=round(float(np.mean([d["fpr"] for d in v])), 4),
                  per_draw=v) for k, v in nulls.items()}
    h2_ok = all(h2[k]["mean_gain"] < cl.MARGIN for k in h2)

    # Declared secondaries.
    def contrast(a, b):
        d = leads[a] - leads[b]
        return dict(point=round(float(d.mean()), 4), decision_ci95=boot["contrast_ci95"][f"{a}_minus_{b}"],
                    test_only=qt.boot(d, Fte))
    s1 = dict(first_gain_over_P=rows[FIRST]["gain"], first_gain_decision_ci95=rows[FIRST]["gain_decision_ci95"],
              hs_minus_first=contrast(PRIMARY, FIRST))
    s1["develops_during_rollout"] = s1["hs_minus_first"]["decision_ci95"][0] > 0
    s2 = {f"hs_minus_{n}": contrast(PRIMARY, n) for n in (ZERO, SHUFFLE)}
    s2["not_elapsed_time_or_task"] = all(s2[f"hs_minus_{n}"]["decision_ci95"][0] > 0 for n in (ZERO, SHUFFLE))
    s3 = dict(hs_minus_safe=contrast(PRIMARY, "safe"))

    # Descriptive.
    desc = {"by_q": {}, "by_task_type": {}}
    for q in cl.QS:
        rq = qt.rho(dev, q)
        tq, fq = cl.fit_fast(C[PRIMARY][0], rq)
        ep, lp = cl.evaluate_fast(C[PRIMARY][1], rq, tq); ef, lf = cl.evaluate_fast(C[PRIMARY][1], rq, fq)
        desc["by_q"][str(q)] = dict(P=ep, hs_plus_P=ef, gain=round(float(np.mean(lf - lp)), 4))
    _, fu = cl.fit_fast(C[PRIMARY][0], rk)
    for tt in sorted({str(r.get("task_type")) for r in te}):
        idx = np.array([str(r.get("task_type")) == tt for r in te])
        c = C[PRIMARY][1].take(idx)
        ep, lp = cl.evaluate_fast(c, rk, tm); ef, lf = cl.evaluate_fast(c, rk, fu)
        desc["by_task_type"][tt] = dict(P=ep, hs_plus_P=ef, gain=round(float(np.mean(lf - lp)), 4))
    best = None
    for d in np.arange(-0.5, 1.0001, 0.0025):
        e, l = cl.evaluate_fast(C[PRIMARY][1], rk, dict(lam=float(d), u=math.inf, s=0.0))
        if e["fpr"] <= pr["test"]["fpr"] + 1e-12 and (best is None or e["mean_lead"] > best[1]["mean_lead"]):
            best = (float(d), e, l)
    desc["P_tuned_on_test_to_hs_fpr"] = dict(delta=best[0], test=best[1], hs_gain=qt.boot(leads[PRIMARY] - best[2], Fte))

    out = dict(plan=PLAN, mode=args.cmd, status="confirmatory" if args.cmd == "run" else "dry run (opened cohorts)",
               inputs=inputs, counts=dict(calibration=len(cal), calibration_all=len(cal_all), test=len(te),
                                          test_all=len(te_all), tasks=len(tasks)),
               checks=checks, reproduction_failures=failures, P=dict(**cl.det_json(tm), **eP),
               detectors=rows, bootstrap=dict(samples=boot["samples"], seed=boot["seed"], gain_sd=boot["gain_sd"]),
               H1=h1, H2=dict(nulls=h2, supported=h2_ok),
               S1_difficulty_prior=s1, S2_learned_timer=s2, S3_vs_safe=s3, descriptive=desc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(dict(H1=h1, primary=dict(P=eP["mean_lead"], hs_plus_P=pr["test"]["mean_lead"], gain=pr["gain"], ci95=ci),
                          H2={k: (v["mean_gain"], v["max_gain"]) for k, v in h2.items()},
                          S1=dict(first_gain=s1["first_gain_over_P"], hs_minus_first=s1["hs_minus_first"]),
                          S2={k: v for k, v in s2.items()}, S3=s3), indent=1, default=float))
    if failures:
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(failures))
        print("No decision is reported when the SAFE reference does not reproduce.")
        return 1
    print("H1:", "SUPPORTED" if h1["supported"] else "NOT SUPPORTED", "| H2:", "SUPPORTED" if h2_ok else "NOT SUPPORTED",
          "| S1 develops during rollout:", s1["develops_during_rollout"],
          "| S2 not elapsed time or task:", s2["not_elapsed_time_or_task"])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "dry-run"):
        c = sub.add_parser(cmd)
        for name in ("--calibration", "--test", "--scores-calibration", "--scores-test", "--normalizations",
                     "--runtime-bundle", "--output"):
            c.add_argument(name, required=True)
        if cmd == "run":
            c.add_argument("--e1-result", required=True)
            c.add_argument("--confirm-predictions", required=True)
        c.add_argument("--bootstrap", type=int, default=cl.BOOT_SAMPLES)
        c.add_argument("--null-draws", type=int, default=cl.NULL_DRAWS)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
