"""Second mock review, list B (R13 of docs/xr1_review_analyses_plan.md). Exploratory, opened cohorts.

Nothing here changes a registered decision. Fits, timers and matching are those of review_analyses
(fit_alpha) and review_analyses_b (gains, loosen_fast, allocate).

  xr1    C1 ensemble size: every subset of the 3 training seeds for SAFE, Hide-and-Seek and its three
            ablations, scored as Eq. 1, refit and evaluated (same target and matched), with AUC,
            calibration-success spread and the share of failures the score catches first.
         C2 SAFE-MLP score form: are the stored per-seed SAFE scores cumulative sums of per-step
            outputs in [0, 1]? Time-free variants (running mean, last-8 mean of the per-step outputs)
            fused with P.
         C3 time-limit truncation at tau in {0.8, 0.9, 1.0}: relabel, refit P and the fused detectors.
         C4 per-task false alarms, and the test-set oracle per-task time-only allocation at each fused
            detector's false-alarm count.
  reset  C5 success-optimal time-only reset on RLDX-1 in E2's retry simulation (R9's model and seeds):
            per-task cutoff chosen on training rollouts (R_opt) or on the test pool (R_oracle),
            against none, P, SAFE-LSTM + P and Hide-and-Seek + P.

Gates: H0 trajectories equal normalized raw scores, full-seed subsets equal the stored trajectories;
H1 E1 and Amendment 2 reproduced; H2 matched gains equal R2's; H3 tau = 1.0 reproduces H1; H4 per-task
counts sum to the false alarms, oracle false alarms within target; H5 (reset) none / P / fused arms
reproduce R9's e3_hs_reset.json, and R_opt at P's cutoffs equals P.

Usage:
  python3 -I review_analyses_c.py xr1 --calibration E1/calibration_predictions.jsonl --test E1/test_predictions.jsonl \\
      --scores-calibration A2/scores/lead_calibration_predictions.jsonl --scores-test A2/scores/confirmatory_test_predictions.jsonl \\
      --normalizations A2/scores/normalizations.json --runtime-bundle runtime_bundle.json --e1-result E1/confirm_lead.json \\
      --amend2-result A2/confirm_hide_and_seek.json [--review-result review_analyses.json] --output OUT.json
  python3 -I review_analyses_c.py reset --calibration-dir <SAFE20>/calibration/lstm --hs-scores-dir <rldx1 H&S>/scores \\
      --replication-json rldx_lead_lstm.json --hs-result rldx_hide_and_seek.json --reference-json e3_hs_reset.json --output OUT.json
"""
import argparse
import hashlib
import itertools
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
import review_analyses_b as rb  # noqa: E402

PLAN = "docs/xr1_review_analyses_plan.md (R13)"
ALPHA = 0.05
HS = chs.PRIMARY
DETECTORS = ("safe", HS) + tuple(chs.CONTROLS)
TAUS = (0.8, 0.9, 1.0)
WINDOW = 8
FORM_TOL = 1e-4                       # float32 storage of a cumulative sum
CUTOFF_GRID = tuple(round(0.05 * i, 2) for i in range(1, 21))
RESET_BUDGETS = (1.25, 1.5, 2.0)
RETRY = ("0.245", "0.17")
ARMS = ("none", "P", "safe_lstm_plus_P", "hs_plus_P", "R_opt", "R_oracle")
Z95 = 1.959964
EPS = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r4(x):
    return None if x is None else round(float(x), 4)


def check_out(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write(path, out):
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")


# ---- scores ---------------------------------------------------------------------------------

def raw_series(r, name, seed):
    return np.asarray(r["per_seed_raw_safe"][seed] if name == "safe" else r["raw"][name][seed], float)


def seed_traj(r, name, seed, norms_all):
    st = norms_all[name][seed][r["task_name"]]
    return (np.maximum.accumulate(raw_series(r, name, seed)) - st["location"]) / st["scale"]


def subset_key(name, subset):
    return f"{name}|{'+'.join(subset)}"


def add_subset(rows, name, subset, norms_all):
    key = subset_key(name, subset)
    for r in rows:
        r[key] = np.mean([seed_traj(r, name, s, norms_all) for s in subset], axis=0)
    return key


def add_max_of_mean(rows, name, seeds, norms_all):
    """Alternative aggregation: running max of the seed-averaged normalized raw score."""
    key = f"{name}|maxmean"
    for r in rows:
        vals = [(raw_series(r, name, s) - norms_all[name][s][r["task_name"]]["location"])
                / norms_all[name][s][r["task_name"]]["scale"] for s in seeds]
        r[key] = np.maximum.accumulate(np.mean(vals, axis=0))
    return key


def step_outputs(raw):
    """Per-step outputs of a cumulative score (the increments; the first is the first value)."""
    return np.diff(np.r_[0.0, raw])


def safe_form(rows, seeds):
    """Is every stored SAFE series a cumulative sum of per-step outputs in [0, 1]?"""
    n = ok = 0
    worst_low, worst_high = 0.0, 0.0
    inc = {True: [], False: []}
    for r in rows:
        for s in seeds:
            d = step_outputs(raw_series(r, "safe", s))
            n += 1
            worst_low, worst_high = min(worst_low, float(d.min())), max(worst_high, float(d.max()))
            ok += bool(d.min() >= -FORM_TOL and d.max() <= 1 + FORM_TOL)
            inc[bool(r["failed"])].append(d)
    summary = {}
    for f, label in ((False, "successes"), (True, "failures")):
        d = np.concatenate(inc[f]) if inc[f] else np.zeros(0)
        summary[label] = dict(steps=int(d.size), mean=r4(d.mean()) if d.size else None,
                              median=r4(np.median(d)) if d.size else None,
                              share_above_0_9=r4(np.mean(d > 0.9)) if d.size else None,
                              share_below_0_1=r4(np.mean(d < 0.1)) if d.size else None)
    return dict(series=n, cumulative_series=ok, cumulative=bool(n and ok == n), min_increment=worst_low,
                max_increment=worst_high, increments=summary)


def add_time_free(cal, te, seeds, kind):
    """safe_mean: running mean of the per-step outputs; safe_win8: mean of the last WINDOW outputs.
    Seed-averaged, running max, then one affine map fitted on the calibration failures' final scores."""
    key = f"safe_{kind}"
    for r in cal + te:
        vals = []
        for s in seeds:
            c = raw_series(r, "safe", s)
            t = np.arange(1, len(c) + 1, dtype=float)
            if kind == "mean":
                vals.append(c / t)
            else:
                prev = np.r_[np.zeros(WINDOW), c][: len(c)]
                vals.append((c - prev) / np.minimum(t, WINDOW))
        r[key] = np.maximum.accumulate(np.mean(vals, axis=0))
    fin = np.array([r[key][-1] for r in cal if r["failed"]])
    loc, scale = float(fin.mean()), float(fin.std()) or 1.0
    for r in cal + te:
        r[key] = (r[key] - loc) / scale
    return key, dict(location=r4(loc), scale=r4(scale))


# ---- diagnostics ----------------------------------------------------------------------------

def auc(c):
    pos, neg = c.M[c.failed], c.M[~c.failed]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return r4(gt + 0.5 * eq)


def score_first_share(te, rk, fu):
    if fu["u"] == math.inf:
        return 0.0
    F = te.take(te.failed)
    a_s = cl.first_vec(F, fu["u"] + fu["s"] * fu["lam"])
    a_t = cl.timer_idx(F, cl.rho_rows(F, rk), fu["lam"])
    return r4(np.mean(a_s < a_t))


def evaluate_key(cal, te, key, rk, tm, Fte=None, boot=False):
    Cc, Ct = cl.Cohort(cal, key), cl.Cohort(te, key)
    fu = ra.fit_alpha(Cc, rk, ALPHA)[1]
    g, lF, lP, lM = rb.gains(Ct, rk, tm, fu, Fte, boot=boot)
    S = Cc.take(~Cc.failed)
    g.update(auc=auc(Ct), calibration_success_sd=r4(np.std(S.M)), score_first=score_first_share(Ct, rk, fu))
    return g, fu, Ct


def summarize(rows):
    out = {}
    for field in ("gain", "matched_gain", "fp", "auc", "calibration_success_sd", "score_first"):
        v = [x[field] for x in rows if x[field] is not None]
        out[field] = dict(mean=r4(np.mean(v)), sd=r4(np.std(v))) if v else None
    return out


def flat(g):
    return dict(gain=g["gain"], matched_gain=(g.get("matched") or {}).get("gain"), fp=g["test"]["fp"],
                fpr=g["test"]["fpr"], lead=g["test"]["mean_lead"], auc=g["auc"],
                calibration_success_sd=g["calibration_success_sd"], score_first=g["score_first"], path=g["path"])


# ---- C3: truncation -------------------------------------------------------------------------

def truncate_rows(rows, tau, keys):
    out = []
    for r in rows:
        H = int(r["horizon"])
        Hn = H if tau >= 1.0 else int(math.floor(tau * H))
        L = min(int(r["L"]), Hn)
        n = dict(rollout_id=r["rollout_id"], task_name=r["task_name"], horizon=Hn, L=L,
                 failed=bool(r["failed"]) or int(r["L"]) > Hn)
        for k in keys:
            n[k] = np.asarray(r[k], float)[:L]
        out.append(n)
    return out


def truncate_dev(dev, horizons, tau):
    if tau >= 1.0:
        return dev
    out = {}
    for t, fr in dev.items():
        H = horizons.get(t)
        if H is None:
            out[t] = fr
            continue
        ratio = math.floor(tau * H) / H
        out[t] = [f / ratio for f in fr if f / ratio <= 1.0 + EPS]
    return out


# ---- C4: per-task false alarms and oracle allocation -----------------------------------------

def per_task_fp(te, rk, det):
    S = te.take(~te.failed)
    a = cl.alarm_vec(S, cl.rho_rows(S, rk), det["lam"], det["u"], det["s"])
    hit = a <= S.L
    counts = defaultdict(int)
    for t, h in zip(S.task, hit):
        counts[str(t)] += int(h)
    tasks = sorted(set(S.task.tolist()))
    v = np.array([counts[t] for t in tasks])
    return dict(total=int(v.sum()), max=int(v.max()), p90=float(np.percentile(v, 90)),
                tasks_ge_3=int((v >= 3).sum()), tasks_ge_4=int((v >= 4).sum()),
                per_task={t: counts[t] for t in tasks})


def oracle_allocation(te, target_fp):
    S = te.take(~te.failed)
    dev = defaultdict(list)
    for t, L, H in zip(S.task, S.L, S.H):
        dev[str(t)].append(float(L / H))
    beta = target_fp / len(S.L)
    rk, info = rb.allocate(dict(dev), beta)
    e, lead = cl.evaluate_fast(te, rk, dict(lam=0.0, u=math.inf, s=0.0))
    return dict(target_fp=int(target_fp), budget=r4(beta), fp=e["fp"], mean_lead=e["mean_lead"]), lead


# ---- xr1 ------------------------------------------------------------------------------------

def run_xr1(args):
    out_path = check_out(args.output)
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    dev = qt.dev_fractions(bundle)
    rkP = qt.rho(dev, cl.PRIMARY_Q)
    norms = json.loads(Path(args.normalizations).read_text())
    norms_all = {"safe": bundle["task_normalizations"], **{k: norms[k] for k in DETECTORS if k != "safe"}}
    _, cal = cl.primary(args.calibration, "calibration")
    _, te = cl.primary(args.test, "test")
    chs.attach(cal, chs.load_scores(args.scores_calibration), "calibration")
    chs.attach(te, chs.load_scores(args.scores_test), "test")
    Fte = [r for r in te if r["failed"]]
    checks, failures = {}, []

    # H0: stored trajectories against normalized raw scores; full subsets against stored trajectories.
    h0 = dict(safe=max(cl.check_safe_pipeline(cal, bundle["task_normalizations"]),
                       cl.check_safe_pipeline(te, bundle["task_normalizations"])),
              hide_and_seek=max(chs.check_normalization(cal, norms), chs.check_normalization(te, norms)))
    full = {}
    for name in DETECTORS:
        seeds = tuple(sorted(norms_all[name]))
        key = add_subset(cal + te, name, seeds, norms_all)
        full[name] = key
        h0[f"{name}_full_subset"] = float(max(np.max(np.abs(r[key] - r[name])) for r in cal + te))
    h0["ok"] = all(v <= 1e-9 for k, v in h0.items() if k != "ok")
    checks["H0_trajectories"] = h0
    if not h0["ok"]:
        failures.append(f"H0: {h0}")

    # H1 / H2: the registered fits through the full subsets.
    tm = ra.fit_alpha(cl.Cohort(cal, "safe"), rkP, ALPHA)[0]
    eP, lP = cl.evaluate_fast(cl.Cohort(te, "safe"), rkP, tm)
    reg = {}
    for name in ("safe", HS):
        g_s, fu_s, _ = evaluate_key(cal, te, name, rkP, tm, Fte, boot=bool(args.review_result))
        g_f, fu_f, _ = evaluate_key(cal, te, full[name], rkP, tm)
        reg[name] = (g_s, fu_s, g_f, fu_f)
    e1 = json.loads(Path(args.e1_result).read_text())["by_q"][str(cl.PRIMARY_Q)]
    a2 = json.loads(Path(args.amend2_result).read_text())
    got = (eP["mean_lead"], eP["fp"], reg["safe"][2]["test"]["mean_lead"], reg["safe"][2]["test"]["fp"], reg["safe"][2]["gain"],
           reg[HS][2]["test"]["mean_lead"], reg[HS][2]["test"]["fp"], reg[HS][2]["gain"])
    want = (e1["P"]["mean_lead"], e1["P"]["fp"], e1["safe_plus_P"]["mean_lead"], e1["safe_plus_P"]["fp"], e1["gain"],
            a2["detectors"][HS]["test"]["mean_lead"], a2["detectors"][HS]["test"]["fp"], a2["detectors"][HS]["gain"])
    same_path = all(reg[k][1] == reg[k][3] for k in reg)
    checks["H1_e1_amend2"] = dict(got=got, want=want, stored_and_subset_paths_equal=same_path,
                                  ok=ra.same(got, want) and same_path)
    if not checks["H1_e1_amend2"]["ok"]:
        failures.append(f"H1: {checks['H1_e1_amend2']}")
    if args.review_result:
        r1 = json.loads(Path(args.review_result).read_text())["R1"]
        i = r1["alphas"].index(ALPHA)
        h2 = {}
        for k in ("safe", HS):
            ref = r1["detectors"][k][i]["matched_P"]
            mine = reg[k][0]["matched"]
            h2[k] = dict(got=[mine["gain"], mine["gain_test_only"]["ci95"]], want=[ref["gain"], ref["gain_test_only"]["ci95"]],
                         subset_matched=reg[k][2]["matched"]["gain"])
            h2[k]["ok"] = ra.same(h2[k]["got"], h2[k]["want"]) and ra.same(h2[k]["subset_matched"], ref["gain"])
        checks["H2_matched"] = h2
        if not all(v["ok"] for v in h2.values()):
            failures.append(f"H2: {h2}")

    # C1: ensemble size.
    c1 = {}
    for name in DETECTORS:
        seeds = tuple(sorted(norms_all[name]))
        rows_k = {}
        for k in range(1, len(seeds) + 1):
            for sub in itertools.combinations(seeds, k):
                key = full[name] if k == len(seeds) else add_subset(cal + te, name, sub, norms_all)
                g, _, _ = evaluate_key(cal, te, key, rkP, tm)
                rows_k.setdefault(str(k), []).append(dict(seeds=list(sub), **flat(g)))
        mm = add_max_of_mean(cal + te, name, seeds, norms_all)
        g, _, _ = evaluate_key(cal, te, mm, rkP, tm)
        c1[name] = dict(subsets=rows_k, by_size={k: summarize(v) for k, v in rows_k.items()},
                        max_of_mean=flat(g))

    # C2: SAFE-MLP score form and time-free variants.
    seeds_safe = tuple(sorted(bundle["task_normalizations"]))
    c2 = dict(form=safe_form(cal + te, seeds_safe), variants={})
    for kind in ("mean", "win8"):
        key, aff = add_time_free(cal, te, seeds_safe, kind)
        g, _, _ = evaluate_key(cal, te, key, rkP, tm)
        c2["variants"][key] = dict(affine=aff, **flat(g))

    # C3: time-limit truncation.
    horizons = {r["task_name"]: int(r["horizon"]) for r in cal + te}
    c3 = {}
    h3 = None
    for tau in TAUS:
        tc, tt = truncate_rows(cal, tau, ("safe", HS)), truncate_rows(te, tau, ("safe", HS))
        rk = qt.rho(truncate_dev(dev, horizons, tau), cl.PRIMARY_Q)
        tmT = ra.fit_alpha(cl.Cohort(tc, "safe"), rk, ALPHA)[0]
        eT, _ = cl.evaluate_fast(cl.Cohort(tt, "safe"), rk, tmT)
        row = dict(counts=dict(calibration_failures=sum(r["failed"] for r in tc), calibration_successes=sum(not r["failed"] for r in tc),
                               test_failures=sum(r["failed"] for r in tt), test_successes=sum(not r["failed"] for r in tt)),
                   P=dict(fp=eT["fp"], fpr=eT["fpr"], mean_lead=eT["mean_lead"]))
        for k in ("safe", HS):
            g, fu, _ = evaluate_key(tc, tt, k, rk, tmT)
            row[k] = flat(g)
        c3[str(tau)] = row
        if tau == 1.0:
            got3 = (eT["mean_lead"], eT["fp"], row["safe"]["gain"], row["safe"]["fp"], row[HS]["gain"], row[HS]["fp"])
            want3 = (eP["mean_lead"], eP["fp"], reg["safe"][0]["gain"], reg["safe"][0]["test"]["fp"],
                     reg[HS][0]["gain"], reg[HS][0]["test"]["fp"])
            h3 = dict(got=got3, want=want3, ok=ra.same(got3, want3))
    checks["H3_truncation_identity"] = h3
    if not h3["ok"]:
        failures.append(f"H3: {h3}")

    # C4: per-task false alarms and the oracle time-only allocation.
    Ct = cl.Cohort(te, "safe")
    c4 = dict(per_task={"P": per_task_fp(Ct, rkP, tm)}, oracle={})
    h4 = dict(ok=True)
    if c4["per_task"]["P"]["total"] != eP["fp"]:
        h4["ok"] = False
    target_P = eP["fp"]
    or_P, l_or_P = oracle_allocation(Ct, target_P)
    c4["oracle"]["at_P_fp"] = dict(or_P, gain_over_P=r4(np.mean(l_or_P - lP)))
    h4["oracle_at_P"] = or_P["fp"] <= target_P
    for k in ("safe", HS):
        g, fu = reg[k][0], reg[k][1]
        CtK = cl.Cohort(te, k)
        c4["per_task"][f"{k}_plus_P"] = per_task_fp(CtK, rkP, fu)
        h4[k] = c4["per_task"][f"{k}_plus_P"]["total"] == g["test"]["fp"]
        if g.get("matched"):
            c4["per_task"][f"P_matched_to_{k}"] = per_task_fp(Ct, rkP, dict(lam=g["matched"]["delta"], u=math.inf, s=0.0))
            h4[f"{k}_matched"] = c4["per_task"][f"P_matched_to_{k}"]["total"] == g["matched"]["fp"]
        _, lF = cl.evaluate_fast(CtK, rkP, fu)
        orK, l_orK = oracle_allocation(Ct, g["test"]["fp"])
        c4["oracle"][f"at_{k}_fp"] = dict(orK, gain_over_P=r4(np.mean(l_orK - lP)),
                                          fused_minus_oracle=r4(np.mean(lF - l_orK)),
                                          fused_minus_oracle_ci95=qt.boot(lF - l_orK, Fte)["ci95"])
        h4[f"oracle_at_{k}"] = orK["fp"] <= g["test"]["fp"]
    h4["ok"] = h4["ok"] and all(v for k, v in h4.items() if k != "ok")
    checks["H4_counts_and_oracle"] = h4
    if not h4["ok"]:
        failures.append(f"H4: {h4}")

    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in vars(args).items()
              if k in ("calibration", "test", "scores_calibration", "scores_test", "normalizations", "runtime_bundle",
                       "e1_result", "amend2_result", "review_result") and v}
    out = dict(plan=PLAN, status="exploratory (opened cohorts)", inputs=inputs, checks=checks,
               reproduction_failures=failures, alpha=ALPHA,
               P=dict(fp=eP["fp"], fpr=eP["fpr"], mean_lead=eP["mean_lead"]),
               C1=c1, C2=c2, C3=c3, C4=c4)
    write(out_path, out)
    if failures:
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(failures))
        return 1
    print_xr1(out)
    print("LIST C OK")
    return 0


def print_xr1(out):
    print(f"P: lead {out['P']['mean_lead']} FP {out['P']['fp']}")
    print("C1 ensemble size (mean over subsets): k | gain | matched | FP | AUC | success SD | score first")
    for name, v in out["C1"].items():
        for k, s in sorted(v["by_size"].items()):
            f = lambda x: "--" if s[x] is None else f"{s[x]['mean']:+.4f}±{s[x]['sd']:.4f}"
            print(f"  {name} k={k}: {f('gain')} | {f('matched_gain')} | {f('fp')} | {f('auc')} | "
                  f"{f('calibration_success_sd')} | {f('score_first')}")
        m = v["max_of_mean"]
        print(f"  {name} max-of-mean: gain {m['gain']:+.4f} matched {m['matched_gain']} FP {m['fp']}")
    form = out["C2"]["form"]
    print(f"C2 SAFE form: cumulative {form['cumulative']} ({form['cumulative_series']}/{form['series']} series; "
          f"increments [{form['min_increment']:.2e}, {form['max_increment']:.6f}]); increments {form['increments']}")
    for key, v in out["C2"]["variants"].items():
        print(f"  {key}: gain {v['gain']:+.4f} matched {v['matched_gain']} FP {v['fp']} AUC {v['auc']}")
    for tau, v in out["C3"].items():
        print(f"C3 tau={tau}: counts {v['counts']}; P lead {v['P']['mean_lead']} FP {v['P']['fp']}; "
              + "; ".join(f"{k} gain {v[k]['gain']:+.4f} matched {v[k]['matched_gain']} FP {v[k]['fp']}" for k in ("safe", HS)))
    for k, v in out["C4"]["per_task"].items():
        print(f"C4 per-task FP {k}: total {v['total']} max {v['max']} p90 {v['p90']} tasks>=3 {v['tasks_ge_3']} tasks>=4 {v['tasks_ge_4']}")
    for k, v in out["C4"]["oracle"].items():
        print(f"C4 oracle {k}: {v}")


# ---- reset (C5) -----------------------------------------------------------------------------

def time_only(H, L_fail, L_succ, a):
    """Alarm indices of a reset at ceil(a * H): failures (capped at their length) and successes."""
    idx = math.ceil(a * H - cl.EPS)
    return np.minimum(np.full(len(L_fail), float(idx)), np.asarray(L_fail, float)), np.full(len(L_succ), float(idx))


def pool(v, fa, sa, arm="R"):
    return dict(p=v["p"], H=v["H"], fail_alarm={arm: fa}, succ_L=v["succ_L"], succ_alarm={arm: sa})


def best_cutoff(v, L_fail, L_succ, B, r_retry, seed, s2):
    """Grid cutoff with the highest simulated success on a pool (ties: the largest cutoff)."""
    if not len(L_succ) or not len(L_fail):
        return 1.0, None
    base = dict(p=v["p"], H=v["H"], succ_L=np.asarray(L_succ, float))
    best = None
    for a in CUTOFF_GRID:
        fa, sa = time_only(v["H"], L_fail, L_succ, a)
        w = s2.retry_mc(dict(base, fail_alarm={"R": fa}, succ_alarm={"R": sa}), "R", B, r_retry, seed)[0]
        if best is None or w >= best[1] - EPS:
            best = (a, w)
    return best


def run_reset(args):
    import stage2_frozen_sizing as sfs
    import stage2_sizing as s2
    out_path = check_out(args.output)
    rows, train, cal_s, test, tasks, rk = sfs.load(args)
    checks, _ = sfs.replication(args, cal_s, train, test, rk)
    ti, leads = sfs.e3_inputs(rows, train, cal_s, test, tasks, rk)
    calF = cal_s + [r for r in train if r["failed"]]
    tm, _ = cl.fit_fast(cl.Cohort(calF, "safe"), rk)
    seed = s2.SEED + 1
    train_s, train_f = defaultdict(list), defaultdict(list)
    for r in train:
        (train_f if r["failed"] else train_s)[r["task_name"]].append(r["L"])
    hs, sf = ti["hide_and_seek"], ti["safe"]
    failures = []
    # H5b: R_opt at P's cutoffs equals the P arm.
    h5b = True
    for t, v in hs.items():
        cP = min(1.0, rk[t] + tm["lam"])
        fa, sa = time_only(v["H"], np.full(len(v["fail_alarm"]["none"]), v["H"]), v["succ_L"], cP)
        h5b &= bool(np.array_equal(fa, v["fail_alarm"]["P"]) and np.array_equal(sa, v["succ_alarm"]["P"]))
    checks["H5_timer_identity"] = dict(ok=h5b)
    if not h5b:
        failures.append("H5: time-only arm at P's cutoffs differs from P")
    ref = json.loads(Path(args.reference_json).read_text()) if args.reference_json else None
    runs = {}
    for r_mode in RETRY:
        per_task, cut = {}, {}
        for k_t, (t, v) in enumerate(hs.items()):
            r_retry = s2.retry_rate(r_mode, v["p"])
            arms = dict(v)
            arms["fail_alarm"] = dict(none=v["fail_alarm"]["none"], P=v["fail_alarm"]["P"],
                                      hs_plus_P=v["fail_alarm"]["safe_plus_P"],
                                      safe_lstm_plus_P=sf[t]["fail_alarm"]["safe_plus_P"])
            arms["succ_alarm"] = dict(none=v["succ_alarm"]["none"], P=v["succ_alarm"]["P"],
                                      hs_plus_P=v["succ_alarm"]["safe_plus_P"],
                                      safe_lstm_plus_P=sf[t]["succ_alarm"]["safe_plus_P"])
            Lf_test = np.full(len(v["fail_alarm"]["none"]), v["H"])
            row = dict(type=v["type"], p=v["p"], c_P=r4(min(1.0, rk[t] + tm["lam"])), budgets={})
            for B in RESET_BUDGETS:
                sel_seed = (seed, k_t, int(round(100 * B)), 99)
                Lf_train = [v["H"]] * len(train_f[t])
                a_opt, _ = best_cutoff(v, Lf_train, train_s[t], B, r_retry, sel_seed, s2)
                a_or, _ = best_cutoff(v, Lf_test, v["succ_L"], B, r_retry, sel_seed, s2)
                for arm, a in (("R_opt", a_opt), ("R_oracle", a_or)):
                    fa, sa = time_only(v["H"], Lf_test, v["succ_L"], a)
                    arms["fail_alarm"][arm], arms["succ_alarm"][arm] = fa, sa
                sd = (seed, k_t, int(round(100 * B)))
                won = {k: s2.retry_mc(arms, k, B, r_retry, sd, paths=True) for k in ARMS}
                p_hat = float(won["none"][1].mean())
                multi = {k: round(float(w.mean()) - p_hat, 4) for k, (w, _) in won.items()}
                ind = {k: s2.retry_mc(arms, k, B, r_retry, sd, paths=True, stream=j + 1)[0] for j, k in enumerate(ARMS)}
                disc = {f"{x}_vs_{y}": s2.paired(ind[x], ind[y])
                        for x, y in (("R_opt", "P"), ("hs_plus_P", "P"), ("hs_plus_P", "R_opt"), ("R_oracle", "P"))}
                row["budgets"][str(B)] = dict(a_opt=a_opt, a_oracle=a_or, multi=multi, discordant=disc,
                                              false_alarm_exposure={k: r4(v["p"] * float(np.mean(arms["succ_alarm"][k] <= v["succ_L"])))
                                                                    for k in ARMS})
            per_task[t] = row
        agg = {}
        for tt in sorted({v["type"] for v in per_task.values()}) + ["all"]:
            sel = [v for v in per_task.values() if tt == "all" or v["type"] == tt]
            a = dict(tasks=len(sel), mean_p=r4(np.mean([v["p"] for v in sel])), budgets={})
            for B in map(str, RESET_BUDGETS):
                m = {k: round(float(np.mean([v["budgets"][B]["multi"][k] for v in sel])), 4) for k in ARMS}
                N = 50 * len(sel)
                cons = {}
                for name, (x, y) in (("R_opt_vs_P", ("R_opt", "P")), ("hs_plus_P_vs_P", ("hs_plus_P", "P")),
                                     ("hs_plus_P_vs_R_opt", ("hs_plus_P", "R_opt")), ("R_oracle_vs_P", ("R_oracle", "P"))):
                    psi = float(np.mean([sum(v["budgets"][B]["discordant"][f"{x}_vs_{y}"]) for v in sel]))
                    d = m[x] - m[y]
                    cons[name] = dict(delta=r4(d), discordant=r4(psi), hw_branching_1000=r4(Z95 * math.sqrt(max(psi - d * d, 0.0) / N)))
                below = [v["budgets"][B]["a_opt"] < v["c_P"] - EPS for v in sel]
                a["budgets"][B] = dict(multi=m, contrasts=cons, tasks_R_opt_below_P=int(sum(below)),
                                       false_alarm_exposure={k: r4(np.mean([v["budgets"][B]["false_alarm_exposure"][k] for v in sel])) for k in ARMS})
            agg[tt] = a
        runs[r_mode] = dict(by_type=agg, per_task=per_task)
        if ref is not None:
            bad = []
            for key_ref, arm in (("hide_and_seek", "hs_plus_P"), ("safe", "safe_lstm_plus_P")):
                want_all = ref["runs"][r_mode]["by_type"][key_ref]["all"]["multi"]
                for B in map(str, RESET_BUDGETS):
                    for want_arm, mine in (("none", "none"), ("P", "P"), ("safe_plus_P", arm)):
                        if want_all[B][want_arm] != agg["all"]["budgets"][B]["multi"][mine]:
                            bad.append((r_mode, key_ref, B, want_arm, want_all[B][want_arm], agg["all"]["budgets"][B]["multi"][mine]))
            checks[f"H5_reference_r_{r_mode}"] = dict(ok=not bad, mismatches=bad[:10])
            if bad:
                failures.append(f"H5 r={r_mode}: {bad[:5]}")
    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in vars(args).items()
              if k in ("replication_json", "hs_result", "reference_json") and v}
    out = dict(plan=PLAN, status="exploratory sizing (opened cohorts, shadow data)", seed=seed, inputs=inputs,
               calibration_dir=str(Path(args.calibration_dir).resolve()), hs_scores_dir=str(Path(args.hs_scores_dir).resolve()),
               cutoff_grid=list(CUTOFF_GRID), budgets=list(RESET_BUDGETS), checks=checks, leads=leads,
               reproduction_failures=failures, runs=runs)
    write(out_path, out)
    if failures:
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(failures))
        return 1
    for r_mode, run in runs.items():
        for tt, a in run["by_type"].items():
            for B, x in a["budgets"].items():
                print(f"C5 r={r_mode} {tt} B={B}: multi {x['multi']}; contrasts "
                      + "; ".join(f"{n} {c['delta']:+.4f} (hw {c['hw_branching_1000']})" for n, c in x["contrasts"].items())
                      + f"; R_opt below P on {x['tasks_R_opt_below_P']}/{a['tasks']} tasks")
    print("RESET OK")
    return 0


# ---- CLI ------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("xr1")
    for name in ("--calibration", "--test", "--scores-calibration", "--scores-test", "--normalizations",
                 "--runtime-bundle", "--e1-result", "--amend2-result", "--output"):
        x.add_argument(name, required=True)
    x.add_argument("--review-result")
    r = sub.add_parser("reset")
    for name in ("--calibration-dir", "--hs-scores-dir", "--replication-json", "--hs-result", "--output"):
        r.add_argument(name, required=True)
    r.add_argument("--reference-json")
    a = p.parse_args(argv)
    return run_xr1(a) if a.cmd == "xr1" else run_reset(a)


if __name__ == "__main__":
    sys.exit(main())
