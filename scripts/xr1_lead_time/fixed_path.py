"""R8 of docs/xr1_review_analyses_plan.md: the fused path chosen on the validation set. Exploratory.

Proposition 1 bounds the FPR of a fused path (u, s) fixed in advance. The registered tests (E1,
Amendment 2) chose the path on the calibration successes that then set its margin, so the bound
covers each candidate path, not the evaluated detector. Here the path is chosen on independent
development data, frozen, certified on the held-out calibration successes and evaluated once:

  select    the validation set (lead_time selection: 3 + 3 calibration and 10 + 10 test rollouts
            per task), E1's score normalization and P's rho_k; the path is chosen as
            confirm_lead.fit_fast chooses it (review_analyses.fit_split)
  certify   the margins of the frozen path and of P from the E1 lead-calibration successes only
            (P is therefore E1's P)
  evaluate  the E1 confirmatory test cohort, once

Scores: SAFE-MLP and Hide-and-Seek, seed-averaged (as registered) and each seed alone; the
Hide-and-Seek controls present in every score file, seed-averaged.

Reported per score: the chosen path; test FPR (Wilson), lead, gain over P, TWA; the decision CI
from Amendment 2's bootstrap (same draws; the path fixed, the margins re-set on the resampled
successes); matched gains (P loosened on the test cohort to the fused detector's false alarms;
the fused detector tightened along its path to P's), test-only CIs; contrasts of Hide-and-Seek
with SAFE and with each control.

Gates (no result without them):
  F1  the registered procedure reproduces E1 and Amendment 2 exactly, including the decision CIs
      when the bootstrap size equals Amendment 2's (same replicates as R8's)
  F2  validation and E1 cohorts are disjoint (rollout IDs; task, environment seed, reset)
  F3  stored seed-averaged trajectories equal the normalized per-seed scores (<= 1e-9)
  F4  P from the certification step equals E1's P

Usage:
  python3 -I fixed_path.py --calibration E1_OUT/calibration_predictions.jsonl --test E1_OUT/test_predictions.jsonl \\
      --scores-calibration A2/scores/lead_calibration_predictions.jsonl \\
      --scores-test A2/scores/confirmatory_test_predictions.jsonl \\
      --val-calibration VAL/calibration_predictions.jsonl --val-test VAL/event_predictions.jsonl \\
      --val-scores-calibration A2/scores/august_calibration_predictions.jsonl \\
      --val-scores-test A2/scores/august_test_predictions.jsonl \\
      --normalizations A2/scores/normalizations.json --runtime-bundle runtime_bundle.json \\
      --e1-result E1_OUT/confirm_lead.json --amend2-result A2/confirm_hide_and_seek.json --output OUT.json
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_hide_and_seek as chs  # noqa: E402
import confirm_lead as cl  # noqa: E402
import lead_time  # noqa: E402
import quantile_timer as qt  # noqa: E402
import review_analyses as ra  # noqa: E402

PLAN = "docs/xr1_review_analyses_plan.md (R8)"
HS = chs.PRIMARY
FAMILY = (HS,) + chs.CONTROLS
SEEDS = ("0", "1", "2")
ALPHA = 0.05


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attach(rows, path, label):
    """Seed-averaged trajectories and per-seed raw scores of the Hide-and-Seek family present in a file."""
    files = chs.load_scores(path)
    present = [d for d in FAMILY if d in files]
    if HS not in present:
        raise ValueError(f"{label}: no {HS} scores in {path}")
    for r in rows:
        rid = str(r["rollout_id"])
        r.setdefault("raw", {})
        for d in present:
            row = files[d].get(rid)
            if row is None:
                raise ValueError(f"{label}: no {d} trajectory for {rid}")
            if bool(row["failed"]) != r["failed"] or row["task_name"] != r["task_name"]:
                raise ValueError(f"{label}: label or task differs for {rid} ({d})")
            v = np.asarray(row["safe_scores"], float)
            if len(v) != r["L"] or not np.all(np.isfinite(v)):
                raise ValueError(f"{label}: {d} has {len(v)} inferences for {rid}, SAFE has {r['L']}")
            r[d] = v
            r["raw"][d] = {str(s): np.asarray(x, float) for s, x in row["per_seed_raw"].items()}
    return present


def per_seed(rows, norms, bundle_norms):
    """Keys safe#s and HS#s: each seed's normalized running maximum. Returns the largest gap
    between the seed mean and the stored seed-averaged trajectory, and the SAFE source used."""
    worst, sources = 0.0, set()
    for r in rows:
        task = r["task_name"]
        if "per_seed_raw_safe" in r:
            per = {str(s): (np.maximum.accumulate(np.asarray(x, float)) - bundle_norms[str(s)][task]["location"])
                   / bundle_norms[str(s)][task]["scale"] for s, x in r["per_seed_raw_safe"].items()}
            sources.add("per_seed_raw_safe")
        elif "per_seed_normalized_running_max" in r:
            per = {str(s): np.asarray(x, float) for s, x in r["per_seed_normalized_running_max"].items()}
            sources.add("per_seed_normalized_running_max")
        else:
            raise ValueError(f"{r['rollout_id']}: no per-seed SAFE scores")
        if sorted(per) != list(SEEDS):
            raise ValueError(f"{r['rollout_id']}: SAFE seeds {sorted(per)}")
        worst = max(worst, float(np.max(np.abs(np.mean([per[s] for s in SEEDS], axis=0) - r["safe"]))))
        hs = {s: (np.maximum.accumulate(r["raw"][HS][s]) - norms[HS][s][task]["location"]) / norms[HS][s][task]["scale"]
              for s in SEEDS}
        worst = max(worst, float(np.max(np.abs(np.mean([hs[s] for s in SEEDS], axis=0) - r[HS]))))
        for s in SEEDS:
            r[f"safe#{s}"] = per[s]
            r[f"{HS}#{s}"] = hs[s]
    return worst, sorted(sources)


def certify(cal, rk, path):
    """Margins of P and of a fixed path (u, s) from the calibration successes (Proposition 1)."""
    S = cal.take(~cal.failed)
    rS = cl.rho_rows(S, rk)
    tm = dict(u=math.inf, s=0.0, lam=cl.conformal(cl.V_vec(S, rS, math.inf, 0.0), ALPHA))
    return tm, dict(u=path["u"], s=path["s"], lam=cl.conformal(cl.V_vec(S, rS, path["u"], path["s"]), ALPHA))


def bootstrap(C, rk, paths, keys, registered_keys, B, seed):
    """Amendment 2's draws (confirm_hide_and_seek.boot_multi); per replicate the registered fit
    (fit_fast) for registered_keys and the fixed-path certification for every key."""
    rng = np.random.default_rng(seed)
    sc, st = cl.strata(C[keys[0]][0]), cl.strata(C[keys[0]][1])
    tasks = sorted(st)
    if sorted(sc) != tasks:
        raise ValueError("Calibration and test task sets differ")
    reg, fix = {k: [] for k in registered_keys}, {k: [] for k in keys}
    for _ in range(B):
        ci, ti = [], []
        for t in rng.choice(tasks, len(tasks)):
            for idx, out in ((sc[t], ci), (st[t], ti)):
                for k in (0, 1):
                    out.append(rng.choice(idx[k], len(idx[k])))
        ci, ti = np.concatenate(ci), np.concatenate(ti)
        lp = None
        for k in keys:
            c, e = C[k][0].take(ci), C[k][1].take(ti)
            tm, fu = certify(c, rk, paths[k])
            if lp is None:
                _, lp = cl.evaluate_fast(e, rk, tm)
            _, lf = cl.evaluate_fast(e, rk, fu)
            fix[k].append(float(np.mean(lf - lp)))
            if k in reg:
                _, fr = cl.fit_fast(c, rk)
                _, lr = cl.evaluate_fast(e, rk, fr)
                reg[k].append(float(np.mean(lr - lp)))
    return reg, fix


def pct(v):
    return np.round(np.percentile(v, [2.5, 97.5]), 4).tolist()


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    norms = json.loads(Path(args.normalizations).read_text())
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    val_cal = lead_time.sel(lead_time.load(Path(args.val_calibration)), args.val_cal_per_class)
    val_test = lead_time.sel(lead_time.load(Path(args.val_test)), args.val_test_per_class)
    val = val_cal + val_test
    tasks = set(bundle["task_horizons"])
    for label, rows in (("calibration", cal), ("test", te), ("validation", val)):
        if {r["task_name"] for r in rows} != tasks:
            raise ValueError(f"{label}: tasks differ from the runtime bundle's")
    ids = {label: {str(r["rollout_id"]) for r in rows} for label, rows in
           (("e1_calibration", cal_all), ("e1_test", te_all), ("validation", val))}
    idents = {label: cl.identities(rows) for label, rows in
              (("e1_calibration", cal_all), ("e1_test", te_all), ("validation", val))}
    overlaps = {f"{a}/{b}": (len(ids[a] & ids[b]), len(idents[a] & idents[b]))
                for a, b in (("validation", "e1_calibration"), ("validation", "e1_test"), ("e1_calibration", "e1_test"))}
    checks = dict(F2_overlaps=overlaps, F2_ok=all(v == (0, 0) for v in overlaps.values()))
    failures = [] if checks["F2_ok"] else [f"F2: cohorts overlap {overlaps}"]

    present = [attach(cal, args.scores_calibration, "calibration"), attach(te, args.scores_test, "test"),
               attach(val_cal, args.val_scores_calibration, "validation calibration"),
               attach(val_test, args.val_scores_test, "validation test")]
    family = [d for d in FAMILY if all(d in p for p in present)]
    gaps, sources = zip(*(per_seed(rows, norms, bundle["task_normalizations"]) for rows in (cal, te, val)))
    checks["F3_max_abs_diff"] = max(gaps)
    checks["safe_per_seed_sources"] = sorted(set().union(*sources))
    if max(gaps) > 1e-9:
        failures.append(f"F3: stored trajectories differ from the normalized per-seed scores ({max(gaps)})")

    keys = ["safe"] + family + [f"{n}#{s}" for n in ("safe", HS) for s in SEEDS]
    C = {k: (cl.Cohort(cal, k), cl.Cohort(te, k)) for k in keys}
    V = {k: cl.Cohort(val, k) for k in keys}
    Fte = [r for r in te if r["failed"]]

    # F1, F4: the registered procedure, and P from certification.
    tm_reg, _ = cl.fit_fast(C["safe"][0], rk)
    eP, lP = ra.evaluate(C["safe"][1], rk, tm_reg)
    reg = {}
    for k in keys:
        _, fu = cl.fit_fast(C[k][0], rk)
        e, l = ra.evaluate(C[k][1], rk, fu)
        reg[k] = dict(path=cl.det_json(fu), test=e, gain=ra.r4(np.mean(l - lP)), lead=l)
    e1 = json.loads(Path(args.e1_result).read_text())["by_q"][str(cl.PRIMARY_Q)]
    got = (eP["mean_lead"], eP["fp"], reg["safe"]["test"]["mean_lead"], reg["safe"]["test"]["fp"], reg["safe"]["gain"])
    want = (e1["P"]["mean_lead"], e1["P"]["fp"], e1["safe_plus_P"]["mean_lead"], e1["safe_plus_P"]["fp"], e1["gain"])
    checks["F1_e1"] = dict(got=got, want=want, ok=ra.same(got, want))
    a2 = json.loads(Path(args.amend2_result).read_text())
    got = [eP["mean_lead"], eP["fp"]] + [x for k in family for x in
                                         (reg[k]["test"]["mean_lead"], reg[k]["test"]["fp"], reg[k]["gain"])]
    want = [a2["P"]["mean_lead"], a2["P"]["fp"]] + [x for k in family for x in
                                                    (a2["detectors"][k]["test"]["mean_lead"], a2["detectors"][k]["test"]["fp"],
                                                     a2["detectors"][k]["gain"])]
    checks["F1_amend2"] = dict(got=got, want=want, ok=ra.same(got, want))
    for name in ("F1_e1", "F1_amend2"):
        if not checks[name]["ok"]:
            failures.append(f"{name}: got {checks[name]['got']}, want {checks[name]['want']}")

    paths, fixed = {}, {}
    for k in keys:
        tmF, fuF = ra.fit_split(V[k], C[k][0], rk, ALPHA)
        paths[k] = dict(u=fuF["u"], s=fuF["s"])
        if k == "safe":
            checks["F4_P"] = dict(certified=cl.det_json(tmF), registered=cl.det_json(tm_reg), ok=tmF == tm_reg)
            if tmF != tm_reg:
                failures.append(f"F4: certified P {tmF} differs from E1's {tm_reg}")
        e, l = ra.evaluate(C[k][1], rk, fuF)
        fixed[k] = dict(path=cl.det_json(fuF), det=fuF, test=e, gain=ra.r4(np.mean(l - lP)), lead=l)
    if failures:
        return finish(out_path, dict(plan=PLAN, checks=checks, reproduction_failures=failures), failures)

    reg_keys = ["safe"] + family
    boot_reg, boot_fix = bootstrap(C, rk, paths, keys, reg_keys, args.bootstrap, cl.BOOT_SEED)
    if args.bootstrap == a2.get("bootstrap", {}).get("samples"):
        got = [pct(boot_reg[k]) for k in reg_keys]
        want = [a2["detectors"][k]["gain_decision_ci95"] for k in reg_keys]
        checks["F1_decision_ci"] = dict(got=got, want=want, ok=ra.same(got, want))
        if not checks["F1_decision_ci"]["ok"]:
            failures.append(f"F1: decision CIs got {got}, want {want}")
            return finish(out_path, dict(plan=PLAN, checks=checks, reproduction_failures=failures), failures)
    else:
        checks["F1_decision_ci"] = dict(ok=None, note="bootstrap size differs from Amendment 2's; not compared")

    rows = {}
    for k in keys:
        f, r = fixed[k], reg[k]
        loose = ra.loosen_timer(C[k][1], rk, f["test"]["fp"])
        tight = ra.tighten_fused(C[k][1], rk, f["det"], eP["fp"])
        rows[k] = dict(
            path=f["path"], test=f["test"], gain=f["gain"], gain_decision_ci95=pct(boot_fix[k]),
            matched_P=None if loose is None else dict(
                fp=loose[1]["fp"], P_lead=loose[1]["mean_lead"], gain=ra.r4(np.mean(f["lead"] - loose[2])),
                gain_test_only=qt.boot(f["lead"] - loose[2], Fte)),
            tightened_to_P=None if tight is None else dict(
                fp=tight[1]["fp"], lead=tight[1]["mean_lead"], gain=ra.r4(np.mean(tight[2] - lP)),
                gain_test_only=qt.boot(tight[2] - lP, Fte)),
            registered=dict(path=r["path"], fpr=r["test"]["fpr"], fp=r["test"]["fp"], lead=r["test"]["mean_lead"],
                            gain=r["gain"], gain_decision_ci95=pct(boot_reg[k]) if k in boot_reg else None))
    contrasts = {}
    for b in ["safe"] + [c for c in family if c != HS]:
        d = np.asarray(boot_fix[HS]) - np.asarray(boot_fix[b])
        contrasts[f"{HS}_minus_{b}"] = dict(point=ra.r4(np.mean(fixed[HS]["lead"] - fixed[b]["lead"])),
                                            decision_ci95=pct(d))
    single = {}
    for n in ("safe", HS):
        per = [rows[f"{n}#{s}"] for s in SEEDS]
        single[n] = {m: dict(mean=float(np.mean([g(p) for p in per])), std=float(np.std([g(p) for p in per], ddof=1)))
                     for m, g in (("fpr", lambda p: p["test"]["fpr"]), ("mean_lead", lambda p: p["test"]["mean_lead"]),
                                  ("gain", lambda p: p["gain"]), ("twa", lambda p: p["test"]["twa"]))}
        single[n]["registered_gain"] = dict(
            mean=float(np.mean([p["registered"]["gain"] for p in per])),
            std=float(np.std([p["registered"]["gain"] for p in per], ddof=1)))

    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in vars(args).items()
              if k in ("calibration", "test", "scores_calibration", "scores_test", "val_calibration", "val_test",
                       "val_scores_calibration", "val_scores_test", "normalizations", "runtime_bundle", "e1_result",
                       "amend2_result")}
    counts = dict(validation=len(val), validation_successes=sum(not r["failed"] for r in val),
                  e1_calibration_successes=sum(not r["failed"] for r in cal), e1_test=len(te), e1_test_failures=len(Fte))
    result = dict(plan=PLAN, status="exploratory (opened cohorts); the registered results are unchanged",
                  inputs=inputs, counts=counts, checks=checks, reproduction_failures=[], alpha=ALPHA,
                  bootstrap=dict(samples=args.bootstrap, seed=cl.BOOT_SEED),
                  P=dict(path=cl.det_json(tm_reg), test=eP), rows=rows, contrasts=contrasts, single_models=single)

    print(f"validation {counts['validation']} rollouts ({counts['validation_successes']} successes); E1 calibration "
          f"successes {counts['e1_calibration_successes']}; E1 test {counts['e1_test']}")
    print(f"P: lead {eP['mean_lead']} FPR {eP['fpr']} ({eP['fp']} FP)")
    print("score: path chosen on validation -> test FPR (Wilson), lead, gain [decision CI], matched | registered")
    for k in ["safe"] + family:
        x = rows[k]
        m = x["matched_P"]
        print(f"  {k:28s} u={x['path']['u']} s={x['path']['s']} lam={x['path']['lam']} -> FPR {x['test']['fpr']} "
              f"{x['test']['fpr_wilson95']} lead {x['test']['mean_lead']} gain {x['gain']:+.4f} {x['gain_decision_ci95']} "
              f"matched {m['gain'] if m else None} {m['gain_test_only']['ci95'] if m else ''} | registered gain "
              f"{x['registered']['gain']:+.4f} {x['registered']['gain_decision_ci95']}")
    for name, c in contrasts.items():
        print(f"  {name}: {c['point']:+.4f} {c['decision_ci95']}")
    for n, v in single.items():
        print(f"  single {n}: gain {v['gain']['mean']:.3f}+-{v['gain']['std']:.3f} fpr {v['fpr']['mean']:.3f} "
              f"twa {v['twa']['mean']:.3f} (registered {v['registered_gain']['mean']:.3f}+-{v['registered_gain']['std']:.3f})")
    return finish(out_path, result, [])


def finish(path, out, failures):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    if failures:
        print("REPRODUCTION FAILED:")
        for f in failures:
            print("  " + f)
        print("No result is reported when a prerequisite fails.")
        return 1
    print("FIXED-PATH ANALYSIS OK")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--calibration", "--test", "--scores-calibration", "--scores-test", "--val-calibration", "--val-test",
                 "--val-scores-calibration", "--val-scores-test", "--normalizations", "--runtime-bundle",
                 "--e1-result", "--amend2-result", "--output"):
        p.add_argument(name, required=True)
    p.add_argument("--val-cal-per-class", type=int, default=3)
    p.add_argument("--val-test-per-class", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=cl.BOOT_SAMPLES)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
