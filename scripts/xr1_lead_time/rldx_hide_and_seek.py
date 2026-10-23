"""RLDX-1 offline replication of Hide-and-Seek + P against the per-task timer P.

Plan: docs/rldx_hide_and_seek_plan.md. Exploratory (second policy, opened cohorts, no new
rollouts). The protocol is report section 15.15 (rldx_replication.py): the same rows,
splits, H_k, P and fused fit (confirm_lead.fit_fast). Only the scores are new: the
Hide-and-Seek detector and its controls, trained with the Amendment 2 settings by
robocasa.recovery.safe.baselines.score_cohorts.

Modes:
  prepare  write the inputs score_cohorts needs from the 15.15 SAFE files: an RLDX
           "runtime bundle" (H_k = longest training failure, a zero time-risk curve that
           nothing uses, development = training IDs) and the train, calibration and test
           ID lists. No score is computed.
  run      recompute SAFE + P and require it to equal the stored 15.15 result exactly,
           then fit every score + P as 15.15 did and report gains, contrasts and nulls.

Usage:
  python3 -I rldx_hide_and_seek.py prepare --calibration-dir <SAFE20>/calibration/lstm --output-dir IN
  python3 -I rldx_hide_and_seek.py run --calibration-dir <SAFE20>/calibration/lstm --inputs-dir IN \\
      --scores-dir SCORES --reference-result <15.15 run>/rldx_lead_lstm.json --output OUT.json
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_hide_and_seek as chs
import confirm_lead as cl
import quantile_timer as qt
import rldx_replication as rr

PLAN = "docs/rldx_hide_and_seek_plan.md"
PRIMARY, FIRST, ZERO, SHUFFLE = chs.PRIMARY, chs.FIRST, chs.ZERO, chs.SHUFFLE
SCORES = chs.SCORES                      # safe, hide_and_seek, first, zero, shuffle
CONTRASTS = chs.CONTRASTS                # HS minus each control, and HS minus SAFE
MODEL = "lstm"                           # the 15.15 primary SAFE model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_first(c, rk, det):
    """Test failures on which the score part of the fused rule alarms before its timer part."""
    if det["u"] == math.inf:
        return 0
    F = c.take(c.failed)
    t_score = cl.first_vec(F, det["u"] + det["s"] * det["lam"])
    return int(np.sum((t_score < cl.timer_idx(F, cl.rho_rows(F, rk), det["lam"])) & (t_score <= F.L)))


def load_rows(cal_dir):
    """The 15.15 rows exactly: seed-checked SAFE rows, H_k, timer and zero risk."""
    rows = rr.assemble(cal_dir, MODEL)
    H = rr.horizons(rows)
    tasks = sorted({r["task_name"] for r in rows})
    missing = [t for t in tasks if t not in H]
    if missing:
        raise ValueError(f"Tasks without training failures: {missing}")
    rr.finish(rows, H)
    return rows, H, tasks


def prepare(args):
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"error: output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows, H, tasks = load_rows(args.calibration_dir)
    split = {s: sorted(r["rollout_id"] for r in rows if r["split"] == s) for s in ("train", "calibration", "evaluation")}
    if any(r["failed"] for r in rows if r["split"] == "calibration"):
        raise ValueError("Calibration split contains failures")
    bundle = dict(
        note="RLDX-1 inputs for score_cohorts (docs/rldx_hide_and_seek_plan.md). task_horizons are H_k, the "
             "longest training failure (report 15.15). The time-risk curve is zero and is not used.",
        task_horizons={t: int(H[t]) for t in tasks},
        time_risk_curves={t: {"risk": [0.0]} for t in tasks},
        task_types={t: next(r["task_type"] for r in rows if r["task_name"] == t) for t in tasks},
        development_rollout_ids=split["train"],
        source=dict(calibration_dir=str(Path(args.calibration_dir).resolve()),
                    files={f"{MODEL}_seed{s}": sha256(Path(args.calibration_dir) / f"{MODEL}_seed{s}" / "normalized_scores.jsonl")
                           for s in rr.SEEDS}),
    )
    (out / "rldx_bundle.json").write_text(json.dumps(bundle, indent=1) + "\n")
    for name, ids in (("train", split["train"]), ("calibration", split["calibration"]), ("test", split["evaluation"])):
        (out / f"{name}_ids.json").write_text(json.dumps(ids) + "\n")
    unused = sorted({r["split"] for r in rows} - {"train", "calibration", "evaluation"})
    print(f"tasks {len(tasks)} | train {len(split['train'])} | calibration {len(split['calibration'])} | "
          f"test {len(split['evaluation'])} | other splits {unused}")
    print("H_k range", min(H.values()), "-", max(H.values()))
    return 0


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    inputs_dir, scores_dir = Path(args.inputs_dir), Path(args.scores_dir)
    rows, H, tasks = load_rows(args.calibration_dir)
    bundle = json.loads((inputs_dir / "rldx_bundle.json").read_text())
    if bundle["task_horizons"] != {t: int(H[t]) for t in tasks}:
        raise ValueError("The scoring bundle's horizons differ from H_k recomputed now")
    train = [r for r in rows if r["split"] == "train"]
    cal_s = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "evaluation"]
    for name, part in (("train", train), ("calibration", cal_s), ("test", test)):
        if sorted(r["rollout_id"] for r in part) != json.loads((inputs_dir / f"{name}_ids.json").read_text()):
            raise ValueError(f"{name} IDs differ from the prepared ID list")
    # As 15.15: held-out successes set the FPR margin; training failures choose the path.
    cal = cal_s + [r for r in train if r["failed"]]
    chs.attach(cal_s, chs.load_scores(scores_dir / "calibration_predictions.jsonl"), "calibration")
    chs.attach([r for r in train if r["failed"]], chs.load_scores(scores_dir / "train_predictions.jsonl"), "train")
    chs.attach(test, chs.load_scores(scores_dir / "test_predictions.jsonl"), "test")
    norms = json.loads((scores_dir / "normalizations.json").read_text())
    scoring = json.loads((scores_dir / "scoring.json").read_text())
    checks = dict(
        C1=[rr.failure_check(train, "train"), rr.failure_check(test, "test")],
        normalization_max_abs_diff=max(chs.check_normalization(cal, norms), chs.check_normalization(test, norms)),
        scoring=dict(code_commit=scoring.get("code_commit"), seeds=scoring.get("seeds"), epochs=scoring.get("epochs"),
                     feature_aggregation=scoring.get("feature_aggregation"), counts=scoring.get("counts"),
                     configs={k: v["configs"]["0"] for k, v in scoring.get("detectors", {}).items()}),
    )
    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in (
        ("bundle", inputs_dir / "rldx_bundle.json"), ("reference_result", args.reference_result),
        *((f"scores_{n}", scores_dir / f"{n}_predictions.jsonl") for n in ("train", "calibration", "test")),
        ("normalizations", scores_dir / "normalizations.json"))}

    rk = qt.rho(rr.dev_fractions(train, tasks), rr.Q)
    C = {k: (cl.Cohort(cal, k), cl.Cohort(test, k)) for k in SCORES}
    Fte = [r for r in test if r["failed"]]
    tm, _ = cl.fit_fast(C["safe"][0], rk)
    eP, lP = cl.evaluate_fast(C["safe"][1], rk, tm)
    det, leads = {}, {}
    for k in SCORES:
        _, fu = cl.fit_fast(C[k][0], rk)
        eF, lF = cl.evaluate_fast(C[k][1], rk, fu)
        leads[k] = lF
        det[k] = dict(path=cl.det_json(fu), test=eF, gain=round(float(np.mean(lF - lP)), 4),
                      gain_test_only=qt.boot(lF - lP, Fte), score_alarms_first=score_first(C[k][1], rk, fu))

    # Reproduction of the stored 15.15 SAFE (LSTM) + P result; no reading without it.
    ref = json.loads(Path(args.reference_result).read_text())["primary"]
    got = [eP["mean_lead"], det["safe"]["test"]["mean_lead"], det["safe"]["gain"], det["safe"]["gain_test_only"]["ci95"]]
    want = [ref["P"]["mean_lead"], ref["safe_plus_P"]["mean_lead"], ref["gain"], ref["gain_test_only"]["ci95"]]
    checks["safe_reference_15_15"] = dict(got=got, want=want, ok=got == want)

    boot = chs.boot_multi(cal, test, rk, list(SCORES), CONTRASTS, args.bootstrap, cl.BOOT_SEED)
    for k in SCORES:
        det[k]["gain_decision_ci95"] = boot["gain_ci95"][k]

    def contrast(a, b):
        d = leads[a] - leads[b]
        return dict(point=round(float(d.mean()), 4), test_only=qt.boot(d, Fte),
                    decision_ci95=boot["contrast_ci95"][f"{a}_minus_{b}"])
    contrasts = {f"{a}_minus_{b}": contrast(a, b) for a, b in CONTRASTS}

    # Reading rules fixed in the plan (test-only CI, as 15.15; Amendment 2 thresholds).
    pr = det[PRIMARY]
    reading = dict(
        R1_ci_lower_above_0=pr["gain_test_only"]["ci95"][0] > 0,
        R1_point_gain_at_least_0p05=pr["gain"] >= cl.MARGIN,
        R1_fpr_at_most_0p075=pr["test"]["fpr"] <= cl.FPR_MAX,
        R1_wilson_upper_at_most_0p10=pr["test"]["fpr_wilson95"][1] <= cl.FPR_WILSON_MAX,
    )
    reading["R1_replicates_at_margin"] = all(reading.values())
    reading["R2_develops_during_rollout"] = contrasts[f"{PRIMARY}_minus_{FIRST}"]["test_only"]["ci95"][0] > 0
    reading["R3_not_elapsed_time_or_task"] = all(contrasts[f"{PRIMARY}_minus_{n}"]["test_only"]["ci95"][0] > 0
                                                 for n in (ZERO, SHUFFLE))

    # Nulls fitted exactly like Hide-and-Seek + P (Amendment 2 loop, seed 1).
    rng = np.random.default_rng(rr.NULL_SEED)
    nulls = {"noise": [], f"{PRIMARY}_resampled": []}
    for _ in range(args.null_draws):
        for part in (cal, test):
            for r in part:
                r["noise"] = np.maximum.accumulate(rng.standard_normal(r["L"]))
        chs.resample_null(cal, rng, PRIMARY, norms); chs.resample_null(test, rng, PRIMARY, norms)
        for key in nulls:
            _, f = cl.fit_fast(cl.Cohort(cal, key), rk)
            e, l = cl.evaluate_fast(cl.Cohort(test, key), rk, f)
            nulls[key].append(dict(gain=round(float(np.mean(l - lP)), 4), fpr=e["fpr"]))
    null_out = {k: dict(draws=len(v), mean_gain=round(float(np.mean([d["gain"] for d in v])), 4),
                        max_gain=max(d["gain"] for d in v),
                        draws_at_or_above_hs=sum(d["gain"] >= pr["gain"] for d in v), per_draw=v)
                for k, v in nulls.items()}

    # Descriptive: task type, P tuned on test to Hide-and-Seek's FPR.
    by_type = {}
    for tt in sorted({str(r["task_type"]) for r in test}):
        idx = np.array([str(r["task_type"]) == tt for r in test])
        m = np.array([str(r["task_type"]) == tt for r in Fte])
        sub = [r for r, k in zip(Fte, m) if k]
        by_type[tt] = {}
        for k in (PRIMARY, "safe"):
            _, fu = cl.fit_fast(C[k][0], rk)
            ep, _ = cl.evaluate_fast(C[k][1].take(idx), rk, tm); ef, _ = cl.evaluate_fast(C[k][1].take(idx), rk, fu)
            by_type[tt][k] = dict(P_lead=ep["mean_lead"], P_fpr=ep["fpr"], fused_lead=ef["mean_lead"], fused_fpr=ef["fpr"],
                                  gain=round(float((leads[k] - lP)[m].mean()), 4), gain_test_only=qt.boot((leads[k] - lP)[m], sub))
    best = None
    for d in np.arange(-0.5, 1.0001, 0.0025):
        e, l = cl.evaluate_fast(C[PRIMARY][1], rk, dict(lam=float(d), u=math.inf, s=0.0))
        if e["fpr"] <= pr["test"]["fpr"] + 1e-12 and (best is None or e["mean_lead"] > best[1]["mean_lead"]):
            best = (float(d), e, l)
    matched = dict(delta=best[0], test=best[1], hs_gain=qt.boot(leads[PRIMARY] - best[2], Fte))

    out = dict(plan=PLAN, status="exploratory (second policy, opened cohorts)", inputs=inputs,
               counts=dict(train=len(train), calibration_successes=len(cal_s), path_choice_failures=len(cal) - len(cal_s),
                           test=len(test), test_failures=len(Fte), tasks=len(tasks)),
               checks=checks, P=dict(**cl.det_json(tm), **eP), detectors=det, contrasts=contrasts,
               bootstrap=dict(samples=boot["samples"], seed=boot["seed"], gain_sd=boot["gain_sd"]),
               reading=reading, nulls=null_out, by_task_type=by_type, P_tuned_on_test_to_hs_fpr=matched)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")

    for c in checks["C1"]:
        print(f"C1 {c['split']:5s} failures {c['failures']}: at limit {c['at_limit']}  L>H {c['longer_than_H']}")
    print(f"SAFE reference (P lead, SAFE+P lead, gain, CI): got {got} want {want} -> "
          f"{'OK' if got == want else 'MISMATCH'}")
    print(f"P                              lead {eP['mean_lead']:.4f}  FPR {eP['fpr']} {eP['fpr_wilson95']}")
    for k in SCORES:
        d = det[k]
        print(f"{k + ' + P':30s} lead {d['test']['mean_lead']:.4f}  FPR {d['test']['fpr']} {d['test']['fpr_wilson95']}  "
              f"gain {d['gain']:+.4f} test-only {d['gain_test_only']['ci95']} decision {d['gain_decision_ci95']}  "
              f"path {d['path']}  score first {d['score_alarms_first']}")
    for k, v in contrasts.items():
        print(f"{k:42s} {v['point']:+.4f} test-only {v['test_only']['ci95']} decision {v['decision_ci95']}")
    for tt, v in by_type.items():
        for k, s in v.items():
            print(f"{tt:10s} {k:14s} P {s['P_lead']} ({s['P_fpr']})  fused {s['fused_lead']} ({s['fused_fpr']})  "
                  f"gain {s['gain']:+.4f} {s['gain_test_only']['ci95']}")
    for k, v in null_out.items():
        print(f"null {k:24s} mean {v['mean_gain']:+.4f} max {v['max_gain']:+.4f}  >= HS: {v['draws_at_or_above_hs']}/{v['draws']}")
    print(f"P tuned on test to HS FPR: HS gain {matched['hs_gain']}")
    print(json.dumps(reading))
    if got != want:
        print("REPRODUCTION FAILED: SAFE + P does not equal the stored 15.15 result; no reading is reported.")
        return 1
    print("READING:", "R1 replicates at margin" if reading["R1_replicates_at_margin"] else "R1 does not replicate at margin",
          "| R2 develops during rollout:", reading["R2_develops_during_rollout"],
          "| R3 not elapsed time or task:", reading["R3_not_elapsed_time_or_task"])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--calibration-dir", required=True)
    a.add_argument("--output-dir", required=True)
    b = sub.add_parser("run")
    for name in ("--calibration-dir", "--inputs-dir", "--scores-dir", "--reference-result", "--output"):
        b.add_argument(name, required=True)
    b.add_argument("--bootstrap", type=int, default=cl.BOOT_SAMPLES)
    b.add_argument("--null-draws", type=int, default=rr.NULL_DRAWS)
    args = p.parse_args(argv)
    return prepare(args) if args.cmd == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
