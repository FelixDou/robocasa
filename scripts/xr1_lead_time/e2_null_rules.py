"""E2 decision rules under null fusions: type I error and co-primary power on RLDX-1.

Plan: docs/stage2_sizing.md, section "E2 rule under null fusions (2026-10-23)". Exploratory:
every input cohort is already open, so these numbers choose and size E2's rule; they test
nothing.

The mock review (concern 2) argues that E2's rule (decision CI > 0 at 97.5%, within task,
plus the binomial FPR rule) can be passed with false-alarm budget alone. This script
simulates, on the same studies as e2_coprimary_sizing.py (same pool, seed and cell numbers):

- two null fusions, each fitted exactly like SAFE + P (confirm_lead.fit_fast):
  noise + P (running max of i.i.d. N(0, 1)) and within-task resampled SAFE + P (per-seed
  normalized SAFE values drawn from all held-out inferences of the task, ignoring outcome);
- three candidate rules:
  (i)   the current E2 rule;
  (ii)  (i) and a CI > 0 for the gain against P tuned on the test cohort to the fused
        detector's own false-alarm count (Amendment 1's registered secondary: thresholds
        fixed, test-only bootstrap, here within task);
  (iii) (i) and a Monte Carlo null test: the gain exceeds the null gains of K fresh draws
        of each null, refitted the same way, with p = (1 + #{null >= gain}) / (K + 1) <= 0.05
        for both nulls;
- type I error of each rule under each null, and the co-primary power (SAFE + P,
  Hide-and-Seek + P) under each rule at assumed gains imposed by shift.

Usage:
  python3 -I e2_null_rules.py --calibration-dir <SAFE20>/calibration/lstm \\
      --hs-scores-dir <rldx1_hide_and_seek run>/scores \\
      [--replication-json rldx_lead_lstm.json] [--hs-result rldx_hide_and_seek.json] \\
      [--coprimary-json e2_coprimary.json] [--workers 8] --output OUT.json
"""
import argparse, json, math, multiprocessing as mp, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_hide_and_seek as chs
import confirm_lead as cl
import e2_coprimary_sizing as co
import quantile_timer as qt
import rldx_hide_and_seek as rhs
import rldx_replication as rr
import stage2_sizing as s2

PLAN = "docs/stage2_sizing.md (E2 rule under null fusions, 2026-10-23)"
SIZES = (10, 20, 30)
LEVEL = 0.975                       # E2's Bonferroni level over the two co-primaries
K_NULL, MC_ALPHA = 39, 0.05         # rule (iii): beat at least 38 of 39 draws of each null
TUNE_GRID = np.arange(-0.5, 1.0001, 0.0025)   # confirm_lead's grid for "P tuned on test"
BOOT_TUNED = 1000
CO = ("safe", "hide_and_seek")
NULLS = ("noise", "safe_resampled")
DETS = CO + NULLS
RULES = ("i", "ii", "iii")
GAINS = co.GAINS                    # safe: 0.034; hide_and_seek: 0.083, 0.052, 0.05
JOINT = co.JOINT                    # (0.034, 0.083), (0.034, 0.052)
DECISION_PAIR = (0.034, 0.052)      # planning gains for the recommendation (written before running)
NESTED_SIZES, NESTED_OUTER, NESTED_INNER = (10, 20, 30), 100, 200


def cell_id(n):
    """The cell of e2_coprimary_sizing / stage2_sizing for this n (within task)."""
    return 1000 * s2.SIZES.index(n) if n in s2.SIZES else 1000 * (100 + n)


# ---- null scores ---------------------------------------------------------------------------

def with_R(C, R):
    out = C.take(np.arange(len(C.L)))
    out.R = R
    out.M = R.max(1)
    return out


def noise_R(C, rng):
    """Running max of i.i.d. N(0, 1), one value per inference."""
    R = np.maximum.accumulate(rng.standard_normal(C.R.shape), axis=1)
    R[~np.isfinite(C.R)] = -np.inf
    return R


def resampled_R(C, rng, seed_pools):
    """Within-task resampled SAFE: per seed, normalized values drawn from all held-out
    inferences of the task (outcome ignored), running max, then the mean over seeds."""
    R = np.full(C.R.shape, -np.inf)
    width = C.R.shape[1]
    for t in np.unique(C.task):
        m = np.flatnonzero(C.task == t)
        acc = np.zeros((len(m), width))
        for v in seed_pools[t]:
            acc += np.maximum.accumulate(v[rng.integers(0, len(v), (len(m), width))], axis=1)
        R[m] = acc / len(seed_pools[t])
    R[~np.isfinite(C.R)] = -np.inf
    return R


def null_cohort(name, C, rng, seed_pools):
    return with_R(C, noise_R(C, rng) if name == "noise" else resampled_R(C, rng, seed_pools))


# ---- rule components -----------------------------------------------------------------------

def tuned_P_lead(e, rk, fp_max):
    """Per-failure lead of P with its margin chosen on this test cohort: the largest mean lead
    whose false alarms do not exceed fp_max (first grid value on ties, as confirm_lead).
    None if no grid value qualifies."""
    rho = cl.rho_rows(e, rk)
    a = np.ceil(np.minimum(1.0, rho[:, None] + TUNE_GRID[None, :]) * e.H[:, None] - cl.EPS)
    hit = a <= e.L[:, None]
    fp = hit[~e.failed].sum(0)
    F = e.failed
    lead = np.where(hit[F], (e.L[F][:, None] - a[F]) / e.H[F][:, None], 0.0)
    ok = np.flatnonzero(fp <= fp_max)
    if not len(ok):
        return None, None
    j = ok[np.argmax(lead.mean(0)[ok])]
    return lead[:, j], float(TUNE_GRID[j])


def boot_lower(d, tasks, rng, B=BOOT_TUNED, level=LEVEL):
    """Lower percentile bound of the mean of d, failures resampled within task (thresholds fixed)."""
    tot = np.zeros(B)
    for t in np.unique(tasks):
        v = d[tasks == t]
        tot += v[rng.integers(0, len(v), (B, len(v)))].sum(1)
    return float(np.percentile(tot / len(d), 100 * (1 - level) / 2))


def mc_p(g, refs):
    refs = np.asarray(refs, float)
    return (1 + int(np.sum(refs >= g - 1e-12))) / (len(refs) + 1)


def fused_gain(c, e, rk, lP):
    _, fu = cl.fit_fast(c, rk)
    eF, lF = cl.evaluate_fast(e, rk, fu)
    return fu, eF, lF


# ---- simulated studies (fork workers inherit the pools) ------------------------------------

_G = {}


def make_study(n, r):
    """The drawn rollouts of study r at size n (exactly e2_coprimary_sizing's within-task draw),
    P's fit, and the null cohorts generated from a separate stream."""
    pools, rk, sp = _G["pools"], _G["rk"], _G["seed_pools"]
    PS = pools["safe"]
    rng = np.random.default_rng([s2.SEED, cell_id(n), r])
    ci, ti = PS.draw(rng, n, 0.0, None), PS.draw(rng, n, 0.0, None)
    rngn = np.random.default_rng([s2.SEED, cell_id(n), r, 1])
    coh = {k: (pools[k].C.take(ci), pools[k].C.take(ti)) for k in CO}
    for name in NULLS:
        c0, e0 = coh["safe"]
        coh[name] = (null_cohort(name, c0, rngn, sp), null_cohort(name, e0, rngn, sp))
    c0, e0 = coh["safe"]
    tm, _ = cl.fit_fast(c0, rk)
    eP, lP = cl.evaluate_fast(e0, rk, tm)
    return coh, eP, lP, rngn


def _job(args):
    n, r0, r1, n_refs = args
    rk, sp = _G["rk"], _G["seed_pools"]
    out = []
    for r in range(r0, r1):
        coh, eP, lP, rngn = make_study(n, r)
        row = dict(P_fpr=eP["fpr"], P_fp=eP["fp"])
        for key in DETS:
            c, e = coh[key]
            fu, eF, lF = fused_gain(c, e, rk, lP)
            lPt, delta = tuned_P_lead(e, rk, eF["fp"])
            dt = lF - (lPt if lPt is not None else 0.0)
            row[key] = dict(gain=float(np.mean(lF - lP)), fp=eF["fp"], n_s=int((~e.failed).sum()), fpr=eF["fpr"],
                            uses_score=bool(math.isfinite(fu["u"])), tuned_gain=float(np.mean(dt)),
                            tuned_lower=boot_lower(dt, e.task[e.failed], rngn), tuned_delta=delta)
        if r < n_refs:
            c0, e0 = coh["safe"]
            row["refs"] = {}
            for name in NULLS:
                g = []
                for _ in range(K_NULL):
                    c, e = null_cohort(name, c0, rngn, sp), null_cohort(name, e0, rngn, sp)
                    g.append(float(np.mean(fused_gain(c, e, rk, lP)[2] - lP)))
                row["refs"][name] = g
        out.append(row)
    return n, out


def _nested(args):
    """Real inner decision bootstraps (rollouts within task and class, refit) for the nulls."""
    n, r0, r1, inner = args
    rk = _G["rk"]
    out = []
    for r in range(r0, r1):
        coh, _, _, _ = make_study(n, r)
        rngi = np.random.default_rng([s2.SEED, cell_id(n), r, 2])
        res = {}
        for name in NULLS:
            c, e = coh[name]
            g = []
            for _ in range(inner):
                cb, eb = c.take(s2.resample_within(c, np.arange(len(c.L)), rngi)), e.take(s2.resample_within(e, np.arange(len(e.L)), rngi))
                tm, fu = cl.fit_fast(cb, rk)
                _, lPb = cl.evaluate_fast(eb, rk, tm); _, lFb = cl.evaluate_fast(eb, rk, fu)
                g.append(float(np.mean(lFb - lPb)))
            res[name] = float(np.percentile(g, 100 * (1 - LEVEL) / 2))
        out.append(res)
    return n, out


def run_pool(fn, jobs, workers):
    if workers <= 1:
        res = [fn(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as p:
            res = p.map(fn, jobs, chunksize=1)
    out = defaultdict(list)
    for n, rows in res:
        out[n] += rows
    return out


def simulate(pools, rk, seed_pools, sizes, R, n_refs, workers, nested_sizes=(), nested_outer=0, nested_inner=0, chunk=10):
    _G.update(pools=pools, rk=rk, seed_pools=seed_pools)
    jobs = [(n, a, min(R, a + chunk), n_refs) for n in sizes for a in range(0, R, chunk)]
    studies = run_pool(_job, jobs, workers)
    jobs = [(n, a, min(nested_outer, a + 5), nested_inner) for n in nested_sizes if n in sizes
            for a in range(0, nested_outer, 5)]
    nested = run_pool(_nested, jobs, workers) if jobs else {}
    return dict(studies), dict(nested)


# ---- rules, type I error and power ---------------------------------------------------------

def rule_masks(rows, key, shift=0.0, hw=None):
    """Per-study pass of rules (i), (ii), (iii) for detector key, its gain moved by shift."""
    g = np.array([r[key]["gain"] for r in rows])
    if hw is None:
        hw = co.half_width(g, LEVEL)
    gs = g + shift
    fpr_ok = np.array([r[key]["fp"] <= co.binom_max_false_alarms(r[key]["n_s"]) for r in rows])
    ci = np.round(gs, 4) - hw > 0
    tuned = np.array([r[key]["tuned_lower"] for r in rows]) + shift > 0
    i = ci & fpr_ok
    has = np.array(["refs" in r for r in rows])
    p = {name: np.array([mc_p(x, r["refs"][name]) if "refs" in r else np.nan for x, r in zip(gs, rows)]) for name in NULLS}
    mc = has & np.all([np.nan_to_num(p[name], nan=1.0) <= MC_ALPHA + 1e-12 for name in NULLS], axis=0)
    return dict(i=i, ii=i & tuned, iii=i & mc, ci=ci, fpr_ok=fpr_ok, tuned=tuned, mc=mc, has_refs=has, hw=hw)


def rate(mask, among=None):
    m = mask if among is None else mask[among]
    k = len(m)
    p = float(np.mean(m)) if k else float("nan")
    return dict(rate=round(p, 4), se=round(math.sqrt(p * (1 - p) / k), 4) if k else None, studies=k)


def analyse(studies, sizes):
    out = dict(null_summary={}, type_i={}, power={}, joint={}, components={})
    for n in sizes:
        rows = studies[n]
        sn = str(n)
        for key in DETS:
            g = np.array([r[key]["gain"] for r in rows])
            out["components"].setdefault(key, {})[sn] = dict(
                mean_gain=round(float(g.mean()), 4), sd=round(float(g.std()), 4), q95=round(float(np.percentile(g, 95)), 4),
                hw=round(co.half_width(g, LEVEL), 4),
                mean_fused_fpr=round(float(np.mean([r[key]["fpr"] for r in rows])), 4),
                mean_P_fpr=round(float(np.mean([r["P_fpr"] for r in rows])), 4),
                mean_tuned_gain=round(float(np.mean([r[key]["tuned_gain"] for r in rows])), 4),
                no_tuned_P=int(sum(r[key]["tuned_delta"] is None for r in rows)),
                uses_score=round(float(np.mean([r[key]["uses_score"] for r in rows])), 3))
        refs = [r for r in rows if "refs" in r]
        for name in NULLS:
            v = np.concatenate([r["refs"][name] for r in refs]) if refs else np.array([])
            out["null_summary"].setdefault(name, {})[sn] = dict(
                reference_draws=int(len(v)), mean=round(float(v.mean()), 4) if len(v) else None,
                q95=round(float(np.percentile(v, 95)), 4) if len(v) else None,
                mean_second_largest=round(float(np.mean([np.sort(r["refs"][name])[-2] for r in refs])), 4) if refs else None)
        for name in NULLS:
            m = rule_masks(rows, name)
            out["type_i"].setdefault(name, {})[sn] = {
                "i": rate(m["i"]), "ii": rate(m["ii"]), "iii": rate(m["iii"], m["has_refs"]),
                "ci_only": rate(m["ci"]), "fpr_rule_only": rate(m["fpr_ok"]), "tuned_ci_only": rate(m["tuned"]),
                "mc_only": rate(m["mc"], m["has_refs"])}
        for key in CO:
            g = np.array([r[key]["gain"] for r in rows])
            for gamma in (None,) + GAINS[key]:
                shift = 0.0 if gamma is None else gamma - g.mean()
                m = rule_masks(rows, key, shift)
                out["power"].setdefault(key, {}).setdefault("plug_in" if gamma is None else str(gamma), {})[sn] = {
                    "i": rate(m["i"])["rate"], "ii": rate(m["ii"])["rate"], "iii": rate(m["iii"], m["has_refs"])["rate"],
                    "ci_only": rate(m["ci"])["rate"], "tuned_ci_only": rate(m["tuned"])["rate"],
                    "mc_only": rate(m["mc"], m["has_refs"])["rate"], "shift": round(shift, 4)}
        gS, gH = (np.array([r[k]["gain"] for r in rows]) for k in CO)
        for gs, gh in JOINT:
            ms = rule_masks(rows, "safe", gs - gS.mean())
            mh = rule_masks(rows, "hide_and_seek", gh - gH.mean())
            for rule in RULES:
                a, b = ms[rule], mh[rule]
                among = ms["has_refs"] if rule == "iii" else None
                out["joint"].setdefault(f"safe {gs} / hs {gh}", {}).setdefault(rule, {})[sn] = dict(
                    both=rate(a & b, among)["rate"], at_least_one=rate(a | b, among)["rate"])
    return out


def nested_check(studies, nested, out):
    res = {}
    for n, inner in nested.items():
        rows = studies[n][: len(inner)]
        for name in NULLS:
            hw = out["components"][name][str(n)]["hw"]
            m = rule_masks(studies[n], name)          # hw from all studies, as in the main tables
            g = np.array([r[name]["gain"] for r in rows])
            fpr_ok = m["fpr_ok"][: len(rows)]
            lo = np.array([x[name] for x in inner])
            res.setdefault(name, {})[str(n)] = dict(
                studies=len(rows), nested_rule_i=round(float(np.mean((lo > 0) & fpr_ok)), 3),
                approx_rule_i=round(float(np.mean((np.round(g, 4) - hw > 0) & fpr_ok)), 3),
                nested_ci_only=round(float(np.mean(lo > 0)), 3), approx_ci_only=round(float(np.mean(np.round(g, 4) - hw > 0)), 3))
    return res


def recommend(out, sizes):
    """Pre-stated rule: among rules with type I <= 0.05 under both nulls at every simulated n,
    take the highest 'both co-primaries pass' power at DECISION_PAIR and n = 20 (ties: the
    simpler rule, in the order i, ii, iii), then the smallest n with SAFE (0.034) and
    Hide-and-Seek (0.052) each at power >= 0.8."""
    ok = [r for r in RULES if all(out["type_i"][name][str(n)][r]["rate"] <= 0.05 for name in NULLS for n in sizes)]
    if not ok:
        return dict(eligible_rules=[], rule=None, n=None)
    pair = f"safe {DECISION_PAIR[0]} / hs {DECISION_PAIR[1]}"
    n_ref = "20" if 20 in sizes else str(sizes[len(sizes) // 2])
    best = max(ok, key=lambda r: (out["joint"][pair][r][n_ref]["both"], -RULES.index(r)))
    n_min = next((n for n in sizes if out["power"]["safe"][str(DECISION_PAIR[0])][str(n)][best] >= 0.8
                  and out["power"]["hide_and_seek"][str(DECISION_PAIR[1])][str(n)][best] >= 0.8), None)
    return dict(eligible_rules=ok, rule=best, n=n_min, decision_pair=list(DECISION_PAIR), reference_n=int(n_ref))


# ---- data ----------------------------------------------------------------------------------

def seed_pools_from(cal_dir, held, model=rhs.MODEL):
    """Per task, per seed: all normalized SAFE values of the held-out rollouts (both outcomes)."""
    ids = {str(r["rollout_id"]): r["task_name"] for r in held}
    pools = defaultdict(list)
    for s in rr.SEEDS:
        rows = rr.load_seed(Path(cal_dir) / f"{model}_seed{s}" / "normalized_scores.jsonl")
        per = defaultdict(list)
        for rid, t in ids.items():
            per[t].append(np.asarray(rows[rid]["full_scores"], float))
        for t, v in per.items():
            pools[t].append(np.concatenate(v))
    return dict(pools)


def run(args):
    out_path = s2.check_out(args.output)
    rows, H, tasks = rhs.load_rows(args.calibration_dir)
    train = [r for r in rows if r["split"] == "train"]
    cal_s = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "evaluation"]
    sd = Path(args.hs_scores_dir)
    chs.attach(cal_s, chs.load_scores(sd / "calibration_predictions.jsonl"), "calibration")
    chs.attach([r for r in train if r["failed"]], chs.load_scores(sd / "train_predictions.jsonl"), "train")
    chs.attach(test, chs.load_scores(sd / "test_predictions.jsonl"), "test")
    rk = qt.rho(rr.dev_fractions(train, tasks), rr.Q)

    checks, stop = dict(replication=co.replication_gains(cal_s, train, test, rk)), []
    for flag, key, path, want_of in (("replication_json", "safe", args.replication_json, lambda d: d["primary"]["gain"]),
                                     ("hs_result", "hide_and_seek", args.hs_result,
                                      lambda d: d["detectors"]["hide_and_seek"]["gain"])):
        if path:
            want = want_of(json.load(open(path)))
            checks[f"{key}_reference"] = want
            if abs(checks["replication"][key] - want) >= 1e-9:
                stop.append(f"{key} replication gain {checks['replication'][key]} != {want}")
    print(f"check: replication gains SAFE {checks['replication']['safe']:+.4f}, Hide-and-Seek "
          f"{checks['replication']['hide_and_seek']:+.4f}" + ("; " + "; ".join(stop) if stop else ""), flush=True)
    if stop:
        sys.exit("STOP: " + "; ".join(stop))

    held = cal_s + test
    pools = co.build_pools(cal_s, test, tasks)
    seed_pools = seed_pools_from(args.calibration_dir, held)
    studies, nested = simulate(pools, rk, seed_pools, args.sizes, args.replicates, args.refs_studies,
                               args.workers, args.nested_sizes, args.nested_outer, args.nested_inner)
    out = analyse(studies, args.sizes)
    if args.coprimary_json:
        ref = json.load(open(args.coprimary_json))["summary"]
        mism = [(n, k) for n in map(str, args.sizes) if n in ref for k in CO
                if (out["components"][k][n]["mean_gain"], out["components"][k][n]["sd"]) !=
                (ref[n][k]["within_task"]["mean_gain"], ref[n][k]["within_task"]["sd"])]
        checks["matches_coprimary_draws"] = not mism
        print("check: co-primary draws " + ("MATCH" if not mism else f"MISMATCH {mism}"), flush=True)
        if mism:
            sys.exit("STOP: the simulated studies differ from e2_coprimary_sizing's")
    out["nested"] = nested_check(studies, nested, out)
    out["recommendation"] = recommend(out, args.sizes)
    res = dict(plan=PLAN, status="exploratory sizing (opened cohorts)", seed=s2.SEED, replicates=args.replicates,
               reference_studies=min(args.refs_studies, args.replicates), k_null=K_NULL, mc_alpha=MC_ALPHA, level=LEVEL,
               sizes=list(args.sizes), max_false_alarms={str(20 * n): co.binom_max_false_alarms(20 * n) for n in args.sizes},
               calibration_dir=str(Path(args.calibration_dir).resolve()), pool_counts=pools["safe"].counts(), checks=checks, **out)
    s2.write(out_path, res)
    print_tables(res)
    return 0


def print_tables(o):
    sizes = [str(n) for n in o["sizes"]]
    print(f"\nE2 FPR rule, most false alarms allowed: {o['max_false_alarms']}")
    print("\n## Detectors fused with P (within task): mean gain (SD) | 97.5% hw | mean fused FPR | tuned-P gain")
    for key, by_n in o["components"].items():
        print(f"{key:15s} " + "  ".join(f"n={n}: {v['mean_gain']:+.4f} ({v['sd']:.4f}) | {v['hw']:.4f} | {v['mean_fused_fpr']:.3f} | "
                                        f"{v['mean_tuned_gain']:+.4f}" for n, v in by_n.items()))
    print("\n## Null reference draws: mean | 95th pct | mean 2nd largest of 39 (the rule (iii) bar)")
    for name, by_n in o["null_summary"].items():
        print(f"{name:15s} " + "  ".join(f"n={n}: {v['mean']:+.4f} | {v['q95']:+.4f} | {v['mean_second_largest']:+.4f}"
                                        for n, v in by_n.items() if v["mean"] is not None))
    print("\n## Type I error (share of null studies passing), MC SE in brackets")
    print("| null | n | rule (i) | rule (ii) | rule (iii) | CI only | FPR rule only | tuned CI only |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, by_n in o["type_i"].items():
        for n in sizes:
            v = by_n[n]
            f = lambda x: f"{x['rate']:.3f} ({x['se']:.3f})"
            print(f"| {name} | {n} | {f(v['i'])} | {f(v['ii'])} | {f(v['iii'])} | {v['ci_only']['rate']:.3f} | "
                  f"{v['fpr_rule_only']['rate']:.3f} | {v['tuned_ci_only']['rate']:.3f} |")
    print("\n## Co-primary power by rule (i) / (ii) / (iii)")
    print("| detector | gain | " + " | ".join(f"n = {n}" for n in sizes) + " |")
    print("|---|---|" + "---|" * len(sizes))
    for key, by_g in o["power"].items():
        for g, by_n in by_g.items():
            print(f"| {key} | {g} | " + " | ".join(f"{by_n[n]['i']:.2f} / {by_n[n]['ii']:.2f} / {by_n[n]['iii']:.2f}" for n in sizes) + " |")
    print("\n## Both co-primaries pass, by rule (i) / (ii) / (iii)")
    for pair, by_r in o["joint"].items():
        print(f"{pair:22s} " + "  ".join(f"n={n}: " + " / ".join(f"{by_r[r][n]['both']:.2f}" for r in RULES) for n in sizes))
    if o["nested"]:
        print("\n## Nested check of rule (i) for the nulls: nested / approx (rule (i); CI only)")
        for name, by_n in o["nested"].items():
            print(f"{name:15s} " + "  ".join(f"n={n}: {v['nested_rule_i']:.3f} / {v['approx_rule_i']:.3f}; "
                                            f"{v['nested_ci_only']:.3f} / {v['approx_ci_only']:.3f}" for n, v in by_n.items()))
    rec = o["recommendation"]
    print(f"\nRecommendation (pre-stated rule): eligible {rec['eligible_rules']}; rule {rec['rule']}; smallest n {rec['n']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration-dir", required=True)
    p.add_argument("--hs-scores-dir", required=True)
    p.add_argument("--replication-json")
    p.add_argument("--hs-result")
    p.add_argument("--coprimary-json")
    p.add_argument("--sizes", type=s2.sizes_arg, default=SIZES)
    p.add_argument("--replicates", type=int, default=s2.REPLICATES)
    p.add_argument("--refs-studies", type=int, default=s2.REPLICATES, help="studies that get the K null reference draws")
    p.add_argument("--nested-sizes", type=s2.sizes_arg, default=NESTED_SIZES)
    p.add_argument("--nested-outer", type=int, default=NESTED_OUTER)
    p.add_argument("--nested-inner", type=int, default=NESTED_INNER)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", required=True)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
