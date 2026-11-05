"""E1's analysis: frozen SAFE-LSTM + P and Hide-and-Seek + P on new RLDX-1 rollouts (docs/rldx_e1_plan.md).

The rule was fixed before any E1 rollout existed (R10 of docs/xr1_review_analyses_plan.md). For each
detector, H3 holds only if
  (a) the 97.5% CI of its gain in mean lead over P lies above zero;
  (b) the 97.5% CI of its gain over P loosened on the test set to its own false alarms lies above zero;
  (c) its test false alarms pass the exact binomial rule (true FPR <= 0.075 not rejected at 5%);
  (d) its gain beats at least 38 of 39 random-score and 38 of 39 resampled-score fusions on the same sets.
Every path is frozen: the detectors' paths come from the freeze manifest; each null fusion's path is chosen
on null scores of the recorded data (held-out calibration successes and training failures). Only the margin
lambda-hat is set, on the new calibration successes (Proposition 1). CIs: 2,000 bootstrap replicates, the
calibration and test rollouts resampled within task and class (paired across monitors), margins re-set in
each replicate, paths fixed; two-sided 97.5% percentile intervals (Bonferroni over the two detectors).

Gate R (no E1 result without it): on the recorded rollouts, the frozen paths certified on the recorded
calibration successes must give the freeze manifest's margins and replication gains, and the manifest's H_k
and rho_k must equal those recomputed from the recorded rows.

Usage:
  python3 -I analyze_e1.py --freeze-manifest M.json --calibration-dir <SAFE20>/calibration/lstm \\
      --hs-scores-dir <freeze>/scores_run2 --e1-scores-dir <E1 root>/scores --plan <E1 root>/plan.json --output OUT.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "xr1_lead_time"))
import confirm_lead as cl  # noqa: E402
import e2_coprimary_sizing as ec  # noqa: E402
import e2_frozen_rule as efr  # noqa: E402
import e2_null_rules as en  # noqa: E402
import fixed_path as fp  # noqa: E402
import quantile_timer as qt  # noqa: E402
import rldx_hide_and_seek as rhs  # noqa: E402
import rldx_replication as rr  # noqa: E402
import stage2_frozen_sizing as fs  # noqa: E402
import stage2_sizing as s2  # noqa: E402

PLAN = "docs/rldx_e1_plan.md"
DETS = {"safe": "safe_plus_P", "hide_and_seek": "hide_and_seek_plus_P"}     # row key -> manifest key
FILES = {"safe": "safe_lstm", "hide_and_seek": "hide_and_seek"}              # row key -> E1 score file
LEVEL = 0.975
B_DEFAULT = 2000
K_NULL, MC_ALPHA, NULLS = en.K_NULL, en.MC_ALPHA, en.NULLS
SEED = 20261105                       # fixed with the plan; streams: 1 bootstrap, 2 nulls
SECONDS_PER_INFERENCE = 8 / 20        # RLDX-1 replans every 8 control steps at 20 Hz


# ---- inputs ------------------------------------------------------------------------------------

def recorded_rows(cal_dir, hs_scores_dir):
    """The 15.15 rows (SAFE-LSTM) with the frozen Hide-and-Seek trajectories on calibration and test."""
    rows, H, tasks = rhs.load_rows(cal_dir)
    train = [r for r in rows if r["split"] == "train"]
    cal_s = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "evaluation"]
    sd = Path(hs_scores_dir)
    fp.attach(cal_s, sd / "calibration_predictions.jsonl", "calibration")
    fp.attach(test, sd / "test_predictions.jsonl", "test")
    return rows, H, tasks, train, cal_s, test


def frozen(man):
    rk = {t: float(v["rho_k"]) for t, v in man["tasks"].items()}
    H = {t: int(v["H_k"]) for t, v in man["tasks"].items()}
    paths = {k: dict(u=float(man["detectors"][m]["path"]["u"]), s=float(man["detectors"][m]["path"]["s"]))
             for k, m in DETS.items()}
    return rk, H, paths


def gate_recorded(man, H, tasks, train, cal_s, test):
    rk, Hf, paths = frozen(man)
    problems = []
    if Hf != {t: int(H[t]) for t in tasks}:
        problems.append("frozen H_k differ from the recorded rows'")
    rk_now = qt.rho(rr.dev_fractions(train, tasks), rr.Q)
    if set(rk_now) != set(rk) or max(abs(rk_now[t] - rk[t]) for t in rk) > 1e-12:
        problems.append("frozen rho_k differ from the recorded training successes'")
    out = {}
    for k, m in DETS.items():
        c, e = cl.Cohort(cal_s, k), cl.Cohort(test, k)
        tm, fz = fp.certify(c, rk, paths[k])
        _, lP = fs.evaluate_cert(e, rk, tm)
        _, lZ = fs.evaluate_cert(e, rk, fz)
        gain = round(float(np.mean(lZ - lP)), 4)
        want_lam, want_gain = man["detectors"][m]["path"]["lam_at_fit"], man["detectors"][m]["replication_gain"]
        ok = round(fz["lam"], 4) == want_lam and gain == want_gain
        out[k] = dict(lam=round(fz["lam"], 4), want_lam=want_lam, gain=gain, want_gain=want_gain, ok=ok)
        if not ok:
            problems.append(f"{k}: margin {round(fz['lam'], 4)} / gain {gain}, want {want_lam} / {want_gain}")
    return problems, out


def e1_rows(scores_dir, name, H):
    """One E1 set's rows with both detectors' trajectories; refuses any misalignment."""
    files = {k: {str(r["rollout_id"]): r for r in s2_jsonl(Path(scores_dir) / f"{name}_{f}_predictions.jsonl")}
             for k, f in FILES.items()}
    ids = sorted(files["safe"])
    if sorted(files["hide_and_seek"]) != ids:
        raise SystemExit(f"STOP: {name}: the two detectors scored different rollouts")
    rows = []
    for rid in ids:
        a, b = files["safe"][rid], files["hide_and_seek"][rid]
        if (a["task_name"], bool(a["failed"])) != (b["task_name"], bool(b["failed"])):
            raise SystemExit(f"STOP: {name}: detectors disagree on {rid}")
        sa, sb = np.asarray(a["safe_scores"], float), np.asarray(b["safe_scores"], float)
        if len(sa) != len(sb) or not (np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))):
            raise SystemExit(f"STOP: {name}: {rid} has {len(sa)} SAFE and {len(sb)} Hide-and-Seek inferences")
        t = a["task_name"]
        if t not in H:
            raise SystemExit(f"STOP: {name}: task {t} has no frozen horizon")
        rows.append(dict(rollout_id=rid, task_name=t, failed=bool(a["failed"]), L=len(sa), horizon=H[t],
                         safe=sa, hide_and_seek=sb))
    return rows


def s2_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def limit_check(rows, name):
    F = [r for r in rows if r["failed"]]
    S = [r for r in rows if not r["failed"]]
    return dict(set=name, failures=len(F), successes=len(S),
                failures_not_at_limit=[r["rollout_id"] for r in F if r["L"] != r["horizon"]][:20],
                successes_at_or_past_limit=[r["rollout_id"] for r in S if r["L"] >= r["horizon"]][:20])


# ---- analysis ----------------------------------------------------------------------------------

def point(C, rk, paths):
    """P and each detector certified on the calibration successes, evaluated on the test set."""
    tm, _ = fp.certify(C["safe"][0], rk, paths["safe"])
    eP, lP = fs.evaluate_cert(C["safe"][1], rk, tm)
    res = dict(P=dict(path=cl.det_json(tm), test=eP))
    for k in DETS:
        row, matched = fs.frozen_study(C[k][0], C[k][1], rk, paths[k], lP)
        _, fz = fp.certify(C[k][0], rk, paths[k])
        eZ, lZ = fs.evaluate_cert(C[k][1], rk, fz)
        res[k] = dict(path=cl.det_json(fz), test=eZ, gain=row["gain"], matched_gain=matched["gain"],
                      matched_P_delta=matched["P_delta"], matched_P_fp=matched["P_fp"], disabled=row["disabled"],
                      lead=lZ)
    res["P"]["lead"] = lP
    return res


def bootstrap(C, rk, paths, B, rng):
    """Paired bootstrap: one resample of each set per replicate, shared by P and both detectors."""
    keys = list(DETS)
    c0, e0 = C["safe"]
    g = {k: np.empty(B) for k in keys}
    gm = {k: np.empty(B) for k in keys}
    for b in range(B):
        ci = s2.resample_within(c0, np.arange(len(c0.L)), rng)
        ti = s2.resample_within(e0, np.arange(len(e0.L)), rng)
        tm, _ = fp.certify(C["safe"][0].take(ci), rk, paths["safe"])
        eb0 = C["safe"][1].take(ti)
        _, lP = fs.evaluate_cert(eb0, rk, tm)
        for k in keys:
            cb, eb = C[k][0].take(ci), C[k][1].take(ti)
            _, fz = fp.certify(cb, rk, paths[k])
            eZ, lZ = fs.evaluate_cert(eb, rk, fz)
            g[k][b] = np.mean(lZ - lP)
            loose = fs.loosen_fast(eb, rk, eZ["fp"])
            gm[k][b] = -np.inf if loose is None else np.mean(lZ - loose[2])
    q = 100 * (1 - LEVEL) / 2
    ci = lambda v: [float(np.percentile(v, q)), float(np.percentile(v, 100 - q))]
    return {k: dict(gain_ci=ci(g[k]), matched_ci=ci(gm[k]), matched_infeasible=int(np.sum(~np.isfinite(gm[k]))))
            for k in keys}


def null_references(C, rk, devC, seed_pools, lP, rng, K):
    """K reference gains per null kind: path chosen on recorded-data nulls, margin and gain on E1 nulls."""
    c0, e0 = C["safe"]
    return {kind: [efr.null_gain(kind, devC, c0, e0, rng, rk, seed_pools, lP) for _ in range(K)] for kind in NULLS}


def decide(res, boot, refs, max_fa):
    out = {}
    for k in DETS:
        p = {kind: en.mc_p(res[k]["gain"], refs[kind]) for kind in NULLS}
        a = boot[k]["gain_ci"][0] > 0
        b = boot[k]["matched_ci"][0] > 0
        c = res[k]["test"]["fp"] <= max_fa
        d = all(v <= MC_ALPHA + 1e-12 for v in p.values())
        out[k] = dict(a_gain_ci_above_zero=a, b_matched_ci_above_zero=b, c_false_alarms_within_bound=c,
                      d_beats_null_fusions=d, null_p=p, H3=bool(a and b and c and d))
    return out


def by_type(rows_test, res, rk):
    F = [r for r in rows_test if r["failed"]]
    types = np.array([r.get("task_type") or "" for r in F])
    out = {}
    for t in sorted(set(types)):
        m = types == t
        out[t] = {k: round(float(np.mean(res[k]["lead"][m] - res["P"]["lead"][m])), 4) for k in DETS}
    return out


def seconds(rows_test, res):
    F = [r for r in rows_test if r["failed"]]
    H = np.array([r["horizon"] for r in F], float)
    return {k: dict(gain_seconds=round(float(np.mean((res[k]["lead"] - res["P"]["lead"]) * H)) * SECONDS_PER_INFERENCE, 2))
            for k in DETS}


def analyze(cal, test, rk, paths, devC, seed_pools, B=B_DEFAULT, K=K_NULL, seed=SEED):
    C = {k: (cl.Cohort(cal, k), cl.Cohort(test, k)) for k in DETS}
    res = point(C, rk, paths)
    boot = bootstrap(C, rk, paths, B, np.random.default_rng([seed, 1]))
    refs = null_references(C, rk, devC, seed_pools, res["P"]["lead"], np.random.default_rng([seed, 2]), K)
    n_s = int((~C["safe"][1].failed).sum())
    max_fa = ec.binom_max_false_alarms(n_s)
    dec = decide(res, boot, refs, max_fa)
    return res, boot, refs, max_fa, dec


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    man = json.loads(Path(args.freeze_manifest).read_text())
    plan = json.loads(Path(args.plan).read_text())
    rows, H, tasks, train, cal_s, test = recorded_rows(args.calibration_dir, args.hs_scores_dir)
    problems, gate = gate_recorded(man, H, tasks, train, cal_s, test)
    for k, v in gate.items():
        print(f"gate R {k}: margin {v['lam']} (want {v['want_lam']}), recorded gain {v['gain']} (want {v['want_gain']})"
              f" -> {'OK' if v['ok'] else 'MISMATCH'}", flush=True)
    if problems:
        raise SystemExit("STOP (gate R): " + "; ".join(problems))
    rk, Hf, paths = frozen(man)
    types = {t: v["type"] for t, v in man["tasks"].items()}
    cal = e1_rows(args.e1_scores_dir, "calibration", Hf)
    te = e1_rows(args.e1_scores_dir, "test", Hf)
    for r in cal + te:
        r["task_type"] = types[r["task_name"]]
    seeds = plan["seeds"]
    devC = cl.Cohort(cal_s + [r for r in train if r["failed"]], "safe")
    seed_pools = en.seed_pools_from(args.calibration_dir, cal_s + test)
    res, boot, refs, max_fa, dec = analyze(cal, te, rk, paths, devC, seed_pools, B=args.replicates)
    out = dict(
        plan=PLAN, status="E1 (rule fixed before collection)", seed=SEED, replicates=args.replicates, level=LEVEL,
        k_null=K_NULL, mc_alpha=MC_ALPHA, frozen_paths=paths, e1_seeds=seeds, gate_recorded=gate,
        counts=dict(calibration=limit_check(cal, "calibration"), test=limit_check(te, "test")),
        max_false_alarms=max_fa,
        P={k: v for k, v in res["P"].items() if k != "lead"},
        detectors={k: dict({kk: vv for kk, vv in res[k].items() if kk != "lead"}, **boot[k], **dec[k]) for k in DETS},
        null_references=refs, by_type=by_type(te, res, rk), seconds=seconds(te, res))
    out_path.write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)) + "\n")
    print(f"P: test false alarms {res['P']['test']['fp']}, lead {res['P']['test']['mean_lead']}")
    for k in DETS:
        d = out["detectors"][k]
        print(f"{k:14s} FA {d['test']['fp']:3d} (max {max_fa}) | gain {d['gain']:+.4f} CI [{d['gain_ci'][0]:+.4f}, "
              f"{d['gain_ci'][1]:+.4f}] | matched {d['matched_gain']:+.4f} CI [{d['matched_ci'][0]:+.4f}, "
              f"{d['matched_ci'][1]:+.4f}] | null p {d['null_p']} | a {d['a_gain_ci_above_zero']} b "
              f"{d['b_matched_ci_above_zero']} c {d['c_false_alarms_within_bound']} d {d['d_beats_null_fusions']} "
              f"-> H3 {'HOLDS' if d['H3'] else 'does not hold'}")
    print("by type:", out["by_type"], "| seconds:", out["seconds"])
    print("ANALYSIS DONE")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freeze-manifest", required=True)
    p.add_argument("--calibration-dir", required=True, help="recorded <SAFE20>/calibration/lstm")
    p.add_argument("--hs-scores-dir", required=True, help="freeze run scores (calibration/test predictions)")
    p.add_argument("--e1-scores-dir", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--replicates", type=int, default=B_DEFAULT)
    p.add_argument("--output", required=True)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
