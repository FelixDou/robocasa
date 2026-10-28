"""SAFE-MLP on unseen tasks (R5 Stage B of docs/xr1_review_analyses_plan.md). Exploratory.

The cross-task design of docs/xr1_cross_task_unseen_plan.md with SAFE as the monitor: for each
fold, the timers and the fused path are fitted on the lead-calibration rollouts of the seen tasks
and applied to the confirmatory-test rollouts of the held-out tasks, with the fold's pooled
normalization. Hide-and-Seek's fold scores from the cross-task run are loaded as well, so SAFE and
Hide-and-Seek are compared on the same folds and the same bootstrap replicates.

Prerequisites (no result without them):
  S1  SAFE fold directories: the folds and training tasks of the fold file; fold all reproduced
      E1's raw trajectories and the runtime bundle's normalization (robocasa.recovery.safe.cross_task_safe
      assemble).
  S2  SAFE fold all with per-task normalization, fused with P, equals E1 (confirm_lead.json).
  S3  Hide-and-Seek fold directories pass the cross-task checks, and Hide-and-Seek + G's zero-shot
      gain equals cross_task_lead.json (and its decision CI, when the bootstrap size is the same).

Rules (fixed in the plan before any fold was trained):
  U1-S  SAFE + G over G, unseen arm: decision CI above 0, point gain >= 0.05, FPR <= 0.075, Wilson <= 0.10
  U5-S  SAFE + P_n over P_n, n = 1, 3, 5, all
  U-HS  (Hide-and-Seek + G) - (SAFE + G), unseen arm, decision CI; also over P_n
  U4-S  unseen minus seen gain over G
The matched-false-alarm comparison against G is reported with every G comparison.

Usage:
  python3 -I cross_task_safe.py run --calibration E1_OUT/calibration_predictions.jsonl \\
      --test E1_OUT/test_predictions.jsonl --safe-fold-dirs S0 S1 S2 S3 SALL \\
      --hs-fold-dirs XT/fold_0 XT/fold_1 XT/fold_2 XT/fold_3 XT/fold_all \\
      --runtime-bundle runtime_bundle.json --e1-result E1_OUT/confirm_lead.json \\
      --cross-task-result XT/cross_task_lead.json --output OUT.json
"""
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl
import cross_task_lead as ctl
import quantile_timer as qt

PLAN = "docs/xr1_review_analyses_plan.md"
SAFE, HS = "safe", ctl.PRIMARY
ALL = ctl.ALL


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- SAFE fold directories ------------------------------------------------------------------

def load_safe_dir(path, wanted):
    path = Path(path)
    meta = json.loads((path / "scoring.json").read_text())
    norms = json.loads((path / "normalizations.json").read_text())
    scores = {}
    for cohort in (ctl.CAL_COHORT, ctl.TEST_COHORT):
        out = {}
        with open(path / f"{cohort}_predictions.jsonl") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = str(row["rollout_id"])
                if rid not in wanted[cohort]:
                    continue
                if row["detector"] != SAFE or rid in out:
                    raise ValueError(f"{path}: unexpected or duplicate row for {rid}")
                out[rid] = dict(task_name=row["task_name"], failed=bool(row["failed"]), held_out=bool(row["held_out"]),
                                raw={s: np.asarray(v, float) for s, v in row["per_seed_raw"].items()})
        scores[cohort] = out
    return meta, norms, scores


def check_safe_meta(F, meta, tasks, held, folds_sha, first):
    if str(meta["fold"]) != F or set(meta["held_out_tasks"]) != held[F]:
        raise ValueError(f"SAFE fold directory for {F}: fold or held-out tasks differ from the fold file")
    if set(meta["training_rollouts_by_task"]) != tasks - held[F]:
        raise ValueError(f"SAFE fold {F}: training tasks are not exactly the seen tasks")
    if meta["folds_sha256"] != folds_sha:
        raise ValueError(f"SAFE fold {F}: assembled with a different fold file")
    if meta["seeds"] != first["seeds"] or meta["detectors"] != [SAFE]:
        raise ValueError(f"SAFE fold {F}: seeds or detector differ")
    if F == ALL:
        c = meta["checks"]
        bad = [k for k, v in c.get("reference_predictions", {}).items() if not v["ok"]]
        if set(c.get("reference_predictions", {})) != {ctl.CAL_COHORT, ctl.TEST_COHORT} or bad \
                or not c.get("bundle_normalization", {}).get("ok"):
            raise ValueError("SAFE fold all did not reproduce E1's trajectories and the bundle normalization")
        return dict(reference={k: v["max_abs_diff"] for k, v in c["reference_predictions"].items()},
                    bundle_normalization=c["bundle_normalization"]["max_abs_diff"])
    return None


def attach_safe(rows, folds_data, cohort, held):
    for F, (_, norms, scores) in folds_data.items():
        n = norms[SAFE]
        seeds = sorted(n)
        for r in rows:
            rid = str(r["rollout_id"])
            row = scores[cohort].get(rid)
            if row is None:
                raise ValueError(f"SAFE fold {F}: no trajectory for {rid} ({cohort})")
            if row["failed"] != r["failed"] or row["task_name"] != r["task_name"]:
                raise ValueError(f"SAFE fold {F}: label or task differs for {rid}")
            if row["held_out"] != (r["task_name"] in held[F]):
                raise ValueError(f"SAFE fold {F}: held-out flag wrong for {rid}")
            if sorted(row["raw"]) != seeds or any(len(v) != r["L"] or not np.all(np.isfinite(v))
                                                  for v in row["raw"].values()):
                raise ValueError(f"SAFE fold {F}: trajectory of {rid} has the wrong seeds, length or values")
            r.setdefault("safe_raw", {})[F] = row["raw"]
            r[f"{SAFE}@{F}"] = ctl.normalized(row["raw"], {s: n[s]["pooled"] for s in seeds})
            if F == ALL:
                r[f"{SAFE}@all/task"] = ctl.normalized(row["raw"], {s: n[s]["per_task"][r["task_name"]] for s in seeds})


# ---- statistics -----------------------------------------------------------------------------

def statistics(D, cal_sel=None, te_sel=None, rk_few=None):
    out = {}
    tG, fS, _ = ctl.unseen(D, lambda F: f"{SAFE}@{F}", D.rk0, cal_sel, te_sel)
    _, fH, _ = ctl.unseen(D, lambda F: f"{HS}@{F}", D.rk0, cal_sel, te_sel)
    out["zero_shot_gain/safe"] = ctl.gain(fS, tG)
    out["zero_shot_gain/hide_and_seek"] = ctl.gain(fH, tG)
    out["contrast/hs_minus_safe/G"] = float(np.mean(fH["lead"] - fS["lead"]))
    st, sf, _ = ctl.seen(D, f"{SAFE}@all", D.rk0, cal_sel, te_sel)
    out["seen_gain_over_G/safe"] = ctl.gain(sf, st)
    out["transfer_loss/safe"] = out["zero_shot_gain/safe"] - out["seen_gain_over_G/safe"]
    st, sf, _ = ctl.seen(D, f"{SAFE}@all/task", D.rkP, cal_sel, te_sel)
    out["seen_per_task_gain_over_P/safe"] = ctl.gain(sf, st)
    rks = dict(rk_few or {}); rks["all"] = D.rkP
    for n, rk in rks.items():
        t, f, _ = ctl.unseen(D, lambda F: f"{SAFE}@{F}", rk, cal_sel, te_sel)
        _, fh, _ = ctl.unseen(D, lambda F: f"{HS}@{F}", rk, cal_sel, te_sel)
        out[f"few_shot_gain/safe/{n}"] = ctl.gain(f, t)
        out[f"contrast/hs_minus_safe/P_{n}"] = float(np.mean(fh["lead"] - f["lead"]))
    return out


def bootstrap(D, B, seed):
    """cross_task_lead.bootstrap's draw sequence, so Hide-and-Seek's intervals equal the cross-task run's."""
    rng = np.random.default_rng(seed)
    reps = defaultdict(list)
    for b in range(B):
        cal_sel, te_sel = ctl.draw(D, rng)
        few = {str(n): ctl.rho_n(D.dev, n, rng) for n in ctl.NS}
        for k, v in statistics(D, cal_sel, te_sel, few).items():
            reps[k].append(v)
        if (b + 1) % 200 == 0:
            print(f"bootstrap {b + 1}/{B}", flush=True)
    return {k: dict(ci95=np.round(np.percentile(v, [2.5, 97.5]), 4).tolist(), sd=round(float(np.std(v)), 4))
            for k, v in reps.items()}


def safe_nulls(D, fold_norms, draws, tG):
    """Noise and within-task resampled fold SAFE, pooled-normalized, fitted like SAFE + G."""
    rng = np.random.default_rng(cl.NULL_SEED)
    res = {"noise": [], "safe_resampled": []}
    for _ in range(draws):
        for rows in (D.cal, D.te):
            for r in rows:
                r["noise"] = np.maximum.accumulate(rng.standard_normal(r["L"]))
        for F in D.folds:
            n = fold_norms[F][SAFE]
            seeds = sorted(n)
            for rows in (D.cal, D.te):
                pools = defaultdict(lambda: defaultdict(list))
                for r in rows:
                    for s in seeds:
                        pools[r["task_name"]][s].append(r["safe_raw"][F][s])
                pools = {t: {s: np.concatenate(v) for s, v in d.items()} for t, d in pools.items()}
                for r in rows:
                    raw = {s: rng.choice(pools[r["task_name"]][s], size=r["L"], replace=True) for s in seeds}
                    r[f"safe_resampled@{F}"] = ctl.normalized(raw, {s: n[s]["pooled"] for s in seeds})
        keys = ["noise"] + [f"safe_resampled@{F}" for F in D.folds]
        C = {k: (cl.Cohort(D.cal, k), cl.Cohort(D.te, k)) for k in keys}
        for name, key_of in (("noise", lambda F: "noise"), ("safe_resampled", lambda F: f"safe_resampled@{F}")):
            _, f, _ = ctl.unseen(D, key_of, D.rk0, C=C)
            res[name].append(dict(gain=round(ctl.gain(f, tG), 4), fpr=round(f["fp"] / f["n"], 4)))
    return {k: dict(draws=len(v), mean_gain=round(float(np.mean([d["gain"] for d in v])), 4),
                    max_gain=max(d["gain"] for d in v), per_draw=v) for k, v in res.items()}


def tune_G(D, key, target_fpr):
    """G tuned on the pooled held-out test rollouts to a target FPR (favors the timer)."""
    best = None
    for lam in ctl.LAM_GRID:
        p = ctl.pool([ctl.raw_eval(D.C[key.format(F=F)][1].take(D.te_idx[F]), D.rk0, dict(lam=float(lam), u=math.inf, s=0.0))
                      for F in D.folds])
        if p["fp"] / p["n"] <= target_fpr + 1e-12 and (best is None or p["lead"].mean() > best[1]["lead"].mean()):
            best = (float(lam), p)
    return best


# ---- run ------------------------------------------------------------------------------------

def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    spec = json.loads(Path(args.folds).read_text())
    tasks, held = ctl.fold_sets(spec)
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    dev = qt.dev_fractions(bundle)
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    if {r["rollout_id"] for r in cal_all} & {r["rollout_id"] for r in te_all} or cl.identities(cal_all) & cl.identities(te_all):
        raise ValueError("Calibration and test cohorts overlap")
    if tasks != set(bundle["task_horizons"]) or {r["task_name"] for r in cal} != tasks or {r["task_name"] for r in te} != tasks:
        raise ValueError("Folds, bundle and cohorts do not cover the same tasks")
    wanted = {ctl.CAL_COHORT: {str(r["rollout_id"]) for r in cal}, ctl.TEST_COHORT: {str(r["rollout_id"]) for r in te}}
    folds_sha = sha256(args.folds)

    safe_data, first = {}, None
    checks = {}
    for path in args.safe_fold_dirs:
        meta, norms, scores = load_safe_dir(path, wanted)
        F = str(meta["fold"])
        if F in safe_data:
            raise ValueError(f"Two SAFE fold directories for fold {F}")
        first = first or meta
        ref = check_safe_meta(F, meta, tasks, held, folds_sha, first)
        if ref is not None:
            checks["S1_safe_fold_all"] = ref
        safe_data[F] = (meta, norms, scores)
    hs_data, hs_first = {}, None
    for path in args.hs_fold_dirs:
        meta, norms, scores = ctl.load_fold_dir(path, wanted)
        F = str(meta["fold"])
        if F in hs_data:
            raise ValueError(f"Two Hide-and-Seek fold directories for fold {F}")
        hs_first = hs_first or meta
        ctl.check_meta(F, meta, tasks, held, folds_sha, hs_first)
        hs_data[F] = (meta, norms, scores)
    if set(safe_data) != set(held) or set(hs_data) != set(held):
        raise ValueError(f"Need SAFE and Hide-and-Seek fold directories for {sorted(held)}")
    checks["S3_hs_raw_reference"] = ctl.check_reference(hs_data[ALL][0])
    checks["test_failures_end_at_H"] = all(r["L"] == r["horizon"] for r in te if r["failed"])
    for rows, cohort in ((cal, ctl.CAL_COHORT), (te, ctl.TEST_COHORT)):
        attach_safe(rows, safe_data, cohort, held)
        ctl.attach(rows, hs_data, cohort, held)
    keys = [f"{SAFE}@{F}" for F in sorted(held)] + [f"{SAFE}@all/task"] + [f"{HS}@{F}" for F in sorted(held)]
    D = ctl.Design(cal, te, held, keys, dev)
    fold_norms = {F: v[1] for F, v in safe_data.items()}
    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in
              (("calibration", args.calibration), ("test", args.test), ("runtime_bundle", args.runtime_bundle),
               ("folds", args.folds), ("e1_result", args.e1_result), ("cross_task_result", args.cross_task_result))}
    inputs["safe_fold_dirs"] = {str(json.loads((Path(p) / "scoring.json").read_text())["fold"]): str(Path(p).resolve())
                                for p in args.safe_fold_dirs}
    inputs["safe_fold_code_commits"] = {F: v[0].get("code_commit") for F, v in sorted(safe_data.items())}
    base = dict(plan=PLAN, status="exploratory, pre-specified (opened cohorts)", inputs=inputs,
                counts=dict(calibration=len(cal), test=len(te), tasks=len(tasks)))
    failures = []

    # S2: SAFE fold all, per-task normalization, + P equals E1.
    e1 = json.loads(Path(args.e1_result).read_text())["by_q"][str(cl.PRIMARY_Q)]
    c, e = D.C[f"{SAFE}@all/task"]
    tm, fu = cl.fit_fast(c, D.rkP)
    eP, lP = cl.evaluate_fast(e, D.rkP, tm)
    eF, lF = cl.evaluate_fast(e, D.rkP, fu)
    got = [eP["mean_lead"], eP["fp"], eF["mean_lead"], eF["fp"], round(float(np.mean(lF - lP)), 4)]
    want = [e1["P"]["mean_lead"], e1["P"]["fp"], e1["safe_plus_P"]["mean_lead"], e1["safe_plus_P"]["fp"], e1["gain"]]
    checks["S2_e1_reproduction"] = dict(got=got, want=want, ok=got == want)
    if got != want:
        failures.append(f"S2: SAFE fold all + P gives {got}, E1 {want}")
    # S3: Hide-and-Seek zero-shot point gain equals the cross-task run.
    xt = json.loads(Path(args.cross_task_result).read_text())
    tG, fH, _ = ctl.unseen(D, lambda F: f"{HS}@{F}", D.rk0)
    got_hs = round(ctl.gain(fH, tG), 4)
    checks["S3_hs_zero_shot"] = dict(got=got_hs, want=xt["detectors"][HS]["gain"], ok=got_hs == xt["detectors"][HS]["gain"])
    if not checks["S3_hs_zero_shot"]["ok"]:
        failures.append(f"S3: Hide-and-Seek + G gain {got_hs}, cross-task {xt['detectors'][HS]['gain']}")
    if failures:
        return finish(out_path, dict(base, checks=checks, reproduction_failures=failures))

    # Point estimates.
    point = statistics(D)
    tG, fS, pf = ctl.unseen(D, lambda F: f"{SAFE}@{F}", D.rk0)
    G = ctl.summary(tG)
    safe_G = dict(test=ctl.summary(fS), gain=round(ctl.gain(fS, tG), 4),
                  paths={F: cl.det_json(v[1]) for F, v in pf.items()})
    hs_G = dict(test=ctl.summary(fH), gain=round(ctl.gain(fH, tG), 4))
    Fte = D.pooled_failures()
    safe_G["gain_test_only_bootstrap"] = qt.boot(fS["lead"] - tG["lead"], Fte)
    per_fold = {F: dict(held_out=len(held[F]), G=ctl.summary(a), safe_plus_G=dict(**ctl.summary(b), gain=round(ctl.gain(b, a), 4)))
                for F, (_, _, a, b) in pf.items()}
    st, sf, (_, sfu) = ctl.seen(D, f"{SAFE}@all", D.rk0)
    seen_ref = dict(G=ctl.summary(st), safe_plus_G=dict(path=cl.det_json(sfu), **ctl.summary(sf)), gain=round(ctl.gain(sf, st), 4))
    pt, pfu, _ = ctl.seen(D, f"{SAFE}@all/task", D.rkP)
    seen_ref["safe_per_task_plus_P"] = ctl.summary(pfu); seen_ref["safe_per_task_gain_over_P"] = round(ctl.gain(pfu, pt), 4)

    rng = np.random.default_rng(ctl.FEW_SEED)
    few = {}
    for n in ctl.NS:
        acc = defaultdict(list)
        for _ in range(args.few_draws):
            rk = ctl.rho_n(dev, n, rng)
            t, f, _ = ctl.unseen(D, lambda F: f"{SAFE}@{F}", rk)
            _, fh, _ = ctl.unseen(D, lambda F: f"{HS}@{F}", rk)
            for k, v in (("P_n_lead", t["lead"].mean()), ("P_n_fpr", t["fp"] / t["n"]), ("safe_lead", f["lead"].mean()),
                         ("safe_fpr", f["fp"] / f["n"]), ("gain", ctl.gain(f, t)), ("hs_minus_safe", np.mean(fh["lead"] - f["lead"]))):
                acc[k].append(float(v))
        few[str(n)] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    t, f, _ = ctl.unseen(D, lambda F: f"{SAFE}@{F}", D.rkP)
    _, fh, _ = ctl.unseen(D, lambda F: f"{HS}@{F}", D.rkP)
    few["all"] = dict(P_n_lead=round(float(t["lead"].mean()), 4), P_n_fpr=round(t["fp"] / t["n"], 4),
                      safe_lead=round(float(f["lead"].mean()), 4), safe_fpr=round(f["fp"] / f["n"], 4),
                      gain=round(ctl.gain(f, t), 4), hs_minus_safe=round(float(np.mean(fh["lead"] - f["lead"])), 4))

    print(f"running the decision bootstrap ({args.bootstrap} replicates)", flush=True)
    boot = bootstrap(D, args.bootstrap, cl.BOOT_SEED)
    ci = lambda k: boot[k]["ci95"]
    if args.bootstrap == xt["bootstrap"]["samples"]:
        want_ci = xt["bootstrap"]["intervals"][f"zero_shot_gain/{HS}"]["ci95"]
        checks["S3_hs_decision_ci"] = dict(got=ci("zero_shot_gain/hide_and_seek"), want=want_ci,
                                           ok=ci("zero_shot_gain/hide_and_seek") == want_ci)
        if not checks["S3_hs_decision_ci"]["ok"]:
            failures.append(f"S3: Hide-and-Seek decision CI {ci('zero_shot_gain/hide_and_seek')}, cross-task {want_ci}")
            return finish(out_path, dict(base, checks=checks, reproduction_failures=failures))
    else:
        checks["S3_hs_decision_ci"] = dict(ok=None, note="bootstrap size differs from the cross-task run's; not compared")

    u1 = dict(ci_lower_above_0=ci("zero_shot_gain/safe")[0] > 0, point_gain_at_least_margin=safe_G["gain"] >= cl.MARGIN,
              fused_fpr_at_most_0p075=safe_G["test"]["fpr"] <= cl.FPR_MAX,
              fused_fpr_wilson_upper_at_most_0p10=safe_G["test"]["fpr_wilson95"][1] <= cl.FPR_WILSON_MAX)
    u1["supported"] = all(u1.values())
    null = safe_nulls(D, fold_norms, args.null_draws, tG)
    u1_null_ok = all(v["mean_gain"] < cl.MARGIN for v in null.values())
    best = tune_G(D, SAFE + "@{F}", safe_G["test"]["fpr"])
    matched = dict(delta=best[0], G=ctl.summary(best[1]), safe_gain=qt.boot(fS["lead"] - best[1]["lead"], Fte))
    best_h = tune_G(D, HS + "@{F}", hs_G["test"]["fpr"])
    matched_hs = dict(delta=best_h[0], G=ctl.summary(best_h[1]), hs_gain=qt.boot(fH["lead"] - best_h[1]["lead"], Fte))
    u5 = {n: dict(few[n], gain_decision_ci95=ci(f"few_shot_gain/safe/{n}"),
                  hs_minus_safe_decision_ci95=ci(f"contrast/hs_minus_safe/P_{n}")) for n in [*map(str, ctl.NS), "all"]}
    u_hs = dict(point=round(point["contrast/hs_minus_safe/G"], 4), decision_ci95=ci("contrast/hs_minus_safe/G"),
                hs_zero_shot_gain=hs_G["gain"], safe_zero_shot_gain=safe_G["gain"])
    u_hs["hs_earlier"] = u_hs["decision_ci95"][0] > 0
    u4 = dict(zero_shot_gain=safe_G["gain"], seen_gain=seen_ref["gain"], point=round(point["transfer_loss/safe"], 4),
              decision_ci95=ci("transfer_loss/safe"))
    u4["transfer_loss_detected"] = u4["decision_ci95"][1] < 0
    types = np.array([r["task_type"] for r in Fte])
    by_type = {tt: dict(failures=int((types == tt).sum()), G_lead=round(float(tG["lead"][types == tt].mean()), 4),
                        safe_plus_G_lead=round(float(fS["lead"][types == tt].mean()), 4),
                        gain=round(float(np.mean((fS["lead"] - tG["lead"])[types == tt])), 4)) for tt in ctl.TYPES}

    out = dict(base, checks=checks, reproduction_failures=[], G=G, safe_plus_G=safe_G, hs_plus_G=hs_G,
               seen_reference=seen_ref, bootstrap=dict(samples=args.bootstrap, seed=cl.BOOT_SEED, intervals=boot),
               U1_S=u1, U1_S_nulls=dict(nulls=null, interpretable=u1_null_ok),
               U1_S_matched_G=matched, hs_matched_G=matched_hs, U5_S_few_shot=u5, U_HS=u_hs, U4_S=u4,
               descriptive=dict(per_fold=per_fold, by_task_type=by_type))
    finish(out_path, out)
    print(f"G {G['mean_lead']} (FPR {G['fpr']}, {G['fp']} FP)")
    print(f"SAFE + G lead {safe_G['test']['mean_lead']} (FPR {safe_G['test']['fpr']}, {safe_G['test']['fp']} FP) "
          f"gain {safe_G['gain']:+.4f} CI {ci('zero_shot_gain/safe')}; matched G {matched['safe_gain']}")
    print(f"Hide-and-Seek + G gain {hs_G['gain']:+.4f} CI {ci('zero_shot_gain/hide_and_seek')}; "
          f"H&S - SAFE {u_hs['point']:+.4f} CI {u_hs['decision_ci95']}")
    print(f"seen: SAFE(all) + G gain {seen_ref['gain']:+.4f}; per-task SAFE + P gain {seen_ref['safe_per_task_gain_over_P']:+.4f}")
    for n in [*map(str, ctl.NS), "all"]:
        print(f"few-shot n={n:>3}: P_n {u5[n]['P_n_lead']}  SAFE + P_n {u5[n]['safe_lead']}  gain {u5[n]['gain']:+.4f} "
              f"CI {u5[n]['gain_decision_ci95']}  H&S - SAFE {u5[n]['hs_minus_safe']:+.4f} CI {u5[n]['hs_minus_safe_decision_ci95']}")
    print(json.dumps(dict(U1_S=u1, nulls={k: (v["mean_gain"], v["max_gain"]) for k, v in null.items()}, U_HS=u_hs, U4_S=u4,
                          by_type=by_type), indent=1, default=float))
    print("U1-S:", "SUPPORTED" if u1["supported"] else "NOT SUPPORTED", "| nulls interpretable:", u1_null_ok,
          "| H&S earlier than SAFE on unseen tasks:", u_hs["hs_earlier"], "| transfer loss detected:", u4["transfer_loss_detected"])
    print("CROSS-TASK SAFE OK")
    return 0


def finish(path, out):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    if out.get("reproduction_failures"):
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(out["reproduction_failures"]))
        print("No result is reported when a prerequisite fails.")
        return 1
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for name in ("--calibration", "--test", "--runtime-bundle", "--e1-result", "--cross-task-result", "--output"):
        r.add_argument(name, required=True)
    r.add_argument("--safe-fold-dirs", nargs="+", required=True)
    r.add_argument("--hs-fold-dirs", nargs="+", required=True)
    r.add_argument("--folds", default=str(ctl.FOLDS_FILE))
    r.add_argument("--bootstrap", type=int, default=cl.BOOT_SAMPLES)
    r.add_argument("--null-draws", type=int, default=cl.NULL_DRAWS)
    r.add_argument("--few-draws", type=int, default=ctl.FEW_DRAWS)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
