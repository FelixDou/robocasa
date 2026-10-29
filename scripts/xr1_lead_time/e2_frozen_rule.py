"""R10 part A: type I error and power of E2's frozen decision rule (plan: docs/xr1_review_analyses_plan.md, R10).

Exploratory. Every input cohort is already open, so these numbers size E2 and test nothing.

The author's E2 rule (2026-10-29): a detector's H3 holds if
  (a) the 97.5% CI of its gain over P lies above zero,
  (b) the 97.5% CI of its gain over P loosened on the test set to its own false alarms lies above zero,
  (c) its false alarms pass the binomial FPR rule, and
  (d) its gain beats at least 38 of 39 noise fusions and 38 of 39 resampled fusions run on the same cohorts,
with every path frozen: the detectors' paths and each null fusion's path are chosen on the existing RLDX-1
data (held-out calibration successes and training failures), and only the margin is set on the study's
calibration successes (fixed_path.certify).

The studies are R9's (stage2_frozen_sizing e2; the same pool, seed, cells and draws), so the co-primaries'
frozen rows must reproduce R9's summary. Each study also draws one null detector of each kind (the type I
subjects) and K = 39 reference draws of each null, from a separate stream. A nested check gives 100 studies
per n a real inner bootstrap with the path fixed.

Usage:
  python3 -I e2_frozen_rule.py --calibration-dir <SAFE20>/calibration/lstm --hs-scores-dir <rldx1_hide_and_seek>/scores \\
      --replication-json rldx_lead_lstm.json --hs-result rldx_hide_and_seek.json --frozen-json e2_frozen.json \\
      [--workers 8] --output OUT.json
"""
import argparse, json, math, multiprocessing as mp, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl
import e2_coprimary_sizing as ec
import e2_null_rules as en
import fixed_path as fp
import stage2_frozen_sizing as fs
import stage2_sizing as s2

PLAN = "docs/xr1_review_analyses_plan.md (R10 part A, 2026-10-29)"
SIZES = (10, 20, 30)
LEVEL = 0.975
K_NULL, MC_ALPHA = en.K_NULL, en.MC_ALPHA
CO = ("safe", "hide_and_seek")
NULLS = en.NULLS                                  # ("noise", "safe_resampled")
DETS = CO + NULLS
GAINS = ec.GAINS
JOINT = ((None, None),) + ec.JOINT
PLANNING = (0.034, 0.052)
NESTED_OUTER, NESTED_INNER = 100, 200
NULL_STREAM = 3                                   # e2_null_rules used streams 1 and 2


# ---- one study -----------------------------------------------------------------------------

_G = {}


def null_path(kind, devC, rng, rk, seed_pools):
    """A null fusion's path, chosen with fit_fast on null scores for the existing data."""
    _, fu = cl.fit_fast(en.null_cohort(kind, devC, rng, seed_pools), rk)
    return dict(u=fu["u"], s=fu["s"])


def null_gain(kind, devC, c0, e0, rng, rk, seed_pools, lPc):
    """One reference draw: path chosen on existing-data nulls, margin certified and gain evaluated on study nulls."""
    path = null_path(kind, devC, rng, rk, seed_pools)
    c, e = en.null_cohort(kind, c0, rng, seed_pools), en.null_cohort(kind, e0, rng, seed_pools)
    _, fz = fp.certify(c, rk, path)
    _, lZ = fs.evaluate_cert(e, rk, fz)
    return float(np.mean(lZ - lPc))


def make_study(n, r):
    """R9's drawn rollouts for study r at size n, P certified on them, and one null subject of
    each kind (path, calibration and test cohorts) from the null stream."""
    pools, rk, paths, sp, devC = _G["pools"], _G["rk"], _G["paths"], _G["seed_pools"], _G["devC"]
    cid = en.cell_id(n)
    rng = np.random.default_rng([s2.SEED, cid, r])
    ci, ti = pools["safe"].draw(rng, n, 0.0, None), pools["safe"].draw(rng, n, 0.0, None)
    coh = {k: (pools[k].C.take(ci), pools[k].C.take(ti)) for k in CO}
    c0, e0 = coh["safe"]
    tmc, _ = fp.certify(c0, rk, paths["safe"])
    ePc, lPc = fs.evaluate_cert(e0, rk, tmc)
    rngn = np.random.default_rng([s2.SEED, cid, r, NULL_STREAM])
    det_paths = dict(paths)
    for kind in NULLS:
        det_paths[kind] = null_path(kind, devC, rngn, rk, sp)
        coh[kind] = (en.null_cohort(kind, c0, rngn, sp), en.null_cohort(kind, e0, rngn, sp))
    return coh, det_paths, ePc, lPc, rngn


def _job(args):
    n, r0, r1, n_refs = args
    rk, sp, devC = _G["rk"], _G["seed_pools"], _G["devC"]
    out = []
    for r in range(r0, r1):
        coh, det_paths, ePc, lPc, rngn = make_study(n, r)
        c0, e0 = coh["safe"]
        row = dict(P_fp=ePc["fp"], P_disabled=bool(ePc.get("disabled", False)))
        for key in DETS:
            c, e = coh[key]
            row[key], row[f"{key}@matched"] = fs.frozen_study(c, e, rk, det_paths[key], lPc)
            row[key]["path_uses_score"] = bool(math.isfinite(det_paths[key]["u"]))
        if r < n_refs:
            row["refs"] = {kind: [null_gain(kind, devC, c0, e0, rngn, rk, sp, lPc) for _ in range(K_NULL)]
                           for kind in NULLS}
        out.append(row)
    return n, out


def inner_bounds(c, e, rk, path, rng, B, level=LEVEL):
    """Real decision bootstrap with the path fixed: calibration and test rollouts resampled within
    task and class, margins re-set on each replicate. Returns the lower bounds of (gain, matched gain)."""
    g, gm = np.empty(B), np.empty(B)
    for b in range(B):
        cb = c.take(s2.resample_within(c, np.arange(len(c.L)), rng))
        eb = e.take(s2.resample_within(e, np.arange(len(e.L)), rng))
        tm, fz = fp.certify(cb, rk, path)
        _, lP = fs.evaluate_cert(eb, rk, tm)
        eZ, lZ = fs.evaluate_cert(eb, rk, fz)
        g[b] = np.mean(lZ - lP)
        loose = fs.loosen_fast(eb, rk, eZ["fp"])
        gm[b] = -np.inf if loose is None else np.mean(lZ - loose[2])
    q = 100 * (1 - level) / 2
    return float(np.percentile(g, q)), float(np.percentile(gm, q))


def _nested(args):
    n, r0, r1, inner = args
    rk = _G["rk"]
    out = []
    for r in range(r0, r1):
        coh, det_paths, _, _, _ = make_study(n, r)
        rngi = np.random.default_rng([s2.SEED, en.cell_id(n), r, NULL_STREAM + 1])
        out.append({key: inner_bounds(*coh[key], rk, det_paths[key], rngi, inner) for key in DETS})
    return n, out


def simulate(pools, rk, paths, seed_pools, devC, sizes, R, n_refs, workers, nested_outer, nested_inner, chunk=10):
    _G.update(pools=pools, rk=rk, paths=paths, seed_pools=seed_pools, devC=devC)
    jobs = [(n, a, min(R, a + chunk), n_refs) for n in sizes for a in range(0, R, chunk)]
    studies = en.run_pool(_job, jobs, workers)
    jobs = [(n, a, min(nested_outer, a + 5), nested_inner) for n in sizes for a in range(0, nested_outer, 5)]
    nested = en.run_pool(_nested, jobs, workers) if jobs and nested_inner > 0 else {}
    return dict(studies), dict(nested)


# ---- rule, type I error, power -------------------------------------------------------------

def gains(rows, k):
    return np.array([r[k]["gain"] for r in rows], float)


def masks(rows, key, shift=0.0):
    """Per-study conditions (a)-(d) for detector key, its gain and matched gain moved by shift."""
    g, gm = gains(rows, key), gains(rows, f"{key}@matched")
    hw, hwm = fs.half_width(g, LEVEL), fs.half_width(gm, LEVEL)
    a, b = fs.above(g, shift, hw), fs.above(gm, shift, hwm)
    c = fs.e2_rule(rows, key)
    has = np.array(["refs" in r for r in rows])
    d = has.copy()
    for kind in NULLS:
        d &= np.array([en.mc_p(x + shift, r["refs"][kind]) <= MC_ALPHA + 1e-12 if "refs" in r else False
                       for x, r in zip(g, rows)])
    return dict(a=a, b=b, c=c, d=d, has=has, hw=hw, hwm=hwm, full=a & b & c & d, no_d=a & b & c, no_b=a & c & d)


def rate(m, among=None):
    v = m if among is None else m[among]
    k = len(v)
    p = float(np.mean(v)) if k else float("nan")
    return dict(rate=round(p, 4), se=round(math.sqrt(p * (1 - p) / k), 4) if k else None, studies=k)


def summary(rows, key):
    g, gm = gains(rows, key), gains(rows, f"{key}@matched")
    f = [r[key] for r in rows]
    return dict(mean_gain=round(float(g.mean()), 4), sd=round(float(g.std()), 4),
                matched_mean_gain=round(float(np.nanmean(gm)), 4), matched_sd=round(float(np.nanstd(gm)), 4),
                matched_infeasible=int(np.sum(~np.isfinite(gm))), q95=round(float(np.percentile(g, 95)), 4),
                mean_fpr=round(float(np.mean([x["fpr"] for x in f])), 4),
                disabled=round(float(np.mean([x["disabled"] for x in f])), 4),
                path_uses_score=round(float(np.mean([x["path_uses_score"] for x in f])), 4))


def analyse(studies, sizes):
    out = dict(summary={}, references={}, type_i={}, power={}, joint={})
    for n in sizes:
        rows, sn = studies[n], str(n)
        for key in DETS:
            out["summary"].setdefault(key, {})[sn] = summary(rows, key)
        refs = [r for r in rows if "refs" in r]
        for kind in NULLS:
            v = np.concatenate([r["refs"][kind] for r in refs]) if refs else np.array([])
            out["references"].setdefault(kind, {})[sn] = dict(
                draws=int(len(v)), mean=round(float(v.mean()), 4) if len(v) else None,
                q95=round(float(np.percentile(v, 95)), 4) if len(v) else None,
                mean_second_largest=round(float(np.mean([np.sort(r["refs"][kind])[-2] for r in refs])), 4) if refs else None)
        for kind in NULLS:
            m = masks(rows, kind)
            out["type_i"].setdefault(kind, {})[sn] = {
                "full": rate(m["full"], m["has"]), "without_d": rate(m["no_d"]), "without_b": rate(m["no_b"], m["has"]),
                "a": rate(m["a"]), "b": rate(m["b"]), "c": rate(m["c"]), "d": rate(m["d"], m["has"])}
        passes = {}
        for key in CO:
            g = gains(rows, key)
            for gamma in (None,) + GAINS[key]:
                shift = 0.0 if gamma is None else gamma - float(g.mean())
                m = masks(rows, key, shift)
                gname = "plug_in" if gamma is None else str(gamma)
                passes[(key, gname)] = (m["full"], m["has"])
                out["power"].setdefault(key, {}).setdefault(gname, {})[sn] = dict(
                    full=rate(m["full"], m["has"])["rate"], without_d=rate(m["no_d"])["rate"],
                    without_b=rate(m["no_b"], m["has"])["rate"], a=rate(m["a"])["rate"], b=rate(m["b"])["rate"],
                    c=rate(m["c"])["rate"], d=rate(m["d"], m["has"])["rate"], shift=round(shift, 4),
                    hw=round(m["hw"], 4), hw_matched=round(m["hwm"], 4))
        for gs, gh in JOINT:
            ms, has = passes[("safe", "plug_in" if gs is None else str(gs))]
            mh, _ = passes[("hide_and_seek", "plug_in" if gh is None else str(gh))]
            name = "plug_in" if gs is None else f"safe {gs} / hs {gh}"
            out["joint"].setdefault(name, {})[sn] = dict(both=rate(ms & mh, has)["rate"], at_least_one=rate(ms | mh, has)["rate"])
    return out


def nested_check(studies, nested, out):
    res = {}
    for n, inner in nested.items():
        rows = studies[n]
        k = len(inner)
        for key in DETS:
            m = masks(rows, key)
            lo = np.array([x[key][0] for x in inner]); lom = np.array([x[key][1] for x in inner])
            c = m["c"][:k]
            res.setdefault(key, {})[str(n)] = dict(
                studies=k, nested_a=round(float(np.mean(lo > 0)), 3), approx_a=round(float(np.mean(m["a"][:k])), 3),
                nested_b=round(float(np.mean(lom > 0)), 3), approx_b=round(float(np.mean(m["b"][:k])), 3),
                nested_abc=round(float(np.mean((lo > 0) & (lom > 0) & c)), 3),
                approx_abc=round(float(np.mean(m["a"][:k] & m["b"][:k] & c)), 3))
    return res


def recommend(out, nested, sizes):
    """The pre-stated rules: type I of the full rule <= 0.05 everywhere (else a bug), n = 20 stands if
    SAFE (0.034) and Hide-and-Seek (0.052) each reach power 0.8 under the full rule, and the nested
    value is quoted where it is more than 0.05 below the approximation at n = 20."""
    bad = [(kind, n) for kind in NULLS for n in map(str, sizes) if out["type_i"][kind][n]["full"]["rate"] > 0.05]
    ok_n = [n for n in sizes if out["power"]["safe"][str(PLANNING[0])][str(n)]["full"] >= 0.8
            and out["power"]["hide_and_seek"][str(PLANNING[1])][str(n)]["full"] >= 0.8]
    quote_nested = [k for k in CO if "20" in nested.get(k, {})
                    and nested[k]["20"]["nested_abc"] < nested[k]["20"]["approx_abc"] - 0.05]
    return dict(type_i_above_0_05=bad, n_stands=20 in ok_n, smallest_n=min(ok_n) if ok_n else None,
                quote_nested_for=quote_nested, planning_gains=list(PLANNING),
                reading=("n = 20 stands" if 20 in ok_n else f"n = {min(ok_n)} needed" if ok_n else
                         f"no n in {list(sizes)} reaches 0.8"))


def check_r9(out, sizes, path):
    """The co-primaries' frozen rows must give R9's summary (same draws)."""
    ref = json.load(open(path))["frozen"]["summary"]
    bad = []
    for n in map(str, sizes):
        if n not in ref:
            continue
        for k in CO:
            want, got = ref[n][k]["within_task"], out["summary"][k][n]
            for f in ("mean_gain", "sd", "matched_mean_gain", "matched_sd"):
                if want[f] != got[f]:
                    bad.append(f"n={n} {k} {f}: {got[f]} != {want[f]}")
    return dict(ok=not bad and any(str(n) in ref for n in sizes), mismatches=bad[:20])


# ---- run -----------------------------------------------------------------------------------

def run(args):
    out_path = s2.check_out(args.output)
    rows, train, cal_s, test, tasks, rk = fs.load(args)
    checks, paths = fs.replication(args, cal_s, train, test, rk)
    pools = ec.build_pools(cal_s, test, tasks)
    devC = cl.Cohort(cal_s + [r for r in train if r["failed"]], "safe")
    seed_pools = en.seed_pools_from(args.calibration_dir, cal_s + test)
    studies, nested = simulate(pools, rk, paths, seed_pools, devC, args.sizes, args.replicates, args.refs_studies,
                               args.workers, args.nested_outer, args.nested_inner)
    out = analyse(studies, args.sizes)
    checks["r9_reproduced"] = check_r9(out, args.sizes, args.frozen_json)
    print("check: co-primary frozen rows reproduce R9 " + ("MATCH" if checks["r9_reproduced"]["ok"]
                                                          else f"MISMATCH {checks['r9_reproduced']['mismatches']}"), flush=True)
    out["nested"] = nested_check(studies, nested, out)
    out["recommendation"] = recommend(out, out["nested"], args.sizes)
    status = 0 if checks["r9_reproduced"]["ok"] and not out["recommendation"]["type_i_above_0_05"] else 1
    res = dict(plan=PLAN, status="exploratory sizing (opened cohorts)" + ("" if status == 0 else "; FAILED CHECK, do not use"),
               seed=s2.SEED, replicates=args.replicates, reference_studies=min(args.refs_studies, args.replicates),
               k_null=K_NULL, mc_alpha=MC_ALPHA, level=LEVEL, sizes=list(args.sizes), frozen_paths=paths,
               max_false_alarms={str(20 * n): ec.binom_max_false_alarms(20 * n) for n in args.sizes},
               calibration_dir=str(Path(args.calibration_dir).resolve()),
               inputs=fs.input_hashes(args) | {"frozen_json": dict(path=str(Path(args.frozen_json).resolve()),
                                                                   sha256=s2.sha256(args.frozen_json))},
               pool_counts=pools["safe"].counts(), dev_rows=int(len(devC.L)), checks=checks, **out)
    s2.write(out_path, res)
    print_tables(res)
    if out["recommendation"]["type_i_above_0_05"]:
        print(f"STOP: full-rule type I above 0.05 at {out['recommendation']['type_i_above_0_05']} (bounded by construction: a bug)")
    return status


def print_tables(o):
    sizes = [str(n) for n in o["sizes"]]
    print(f"\nfrozen paths: " + ", ".join(f"{k} u={p['u']} s={p['s']}" for k, p in o["frozen_paths"].items()))
    print("\n## Frozen detectors (within task): mean gain (SD) | matched (SD) | FPR | disabled | path uses score")
    for key, by_n in o["summary"].items():
        print(f"{key:15s} " + "  ".join(f"n={n}: {v['mean_gain']:+.4f} ({v['sd']:.4f}) | {v['matched_mean_gain']:+.4f} "
                                        f"({v['matched_sd']:.4f}) | {v['mean_fpr']:.3f} | {v['disabled']:.3f} | "
                                        f"{v['path_uses_score']:.2f}" for n, v in by_n.items()))
    print("\n## Null references: mean | 95th pct | mean 2nd largest of 39 (the bar for (d))")
    for kind, by_n in o["references"].items():
        print(f"{kind:15s} " + "  ".join(f"n={n}: {v['mean']:+.4f} | {v['q95']:+.4f} | {v['mean_second_largest']:+.4f}"
                                        for n, v in by_n.items() if v["mean"] is not None))
    print("\n## Type I error (share of null subjects passing), MC SE in brackets")
    print("| null | n | full rule | without (d) | without (b) | (a) | (b) | (c) | (d) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for kind, by_n in o["type_i"].items():
        for n in sizes:
            v = by_n[n]
            f = lambda x: f"{x['rate']:.3f} ({x['se']:.3f})"
            print(f"| {kind} | {n} | {f(v['full'])} | {f(v['without_d'])} | {f(v['without_b'])} | {v['a']['rate']:.3f} | "
                  f"{v['b']['rate']:.3f} | {v['c']['rate']:.3f} | {v['d']['rate']:.3f} |")
    print("\n## Co-primary power: full rule | without (d) | without (b); conditions (a) (b) (c) (d)")
    print("| detector | gain | " + " | ".join(f"n = {n}" for n in sizes) + " |")
    print("|---|---|" + "---|" * len(sizes))
    for key, by_g in o["power"].items():
        for g, by_n in by_g.items():
            print(f"| {key} | {g} | " + " | ".join(
                f"{by_n[n]['full']:.2f} / {by_n[n]['without_d']:.2f} / {by_n[n]['without_b']:.2f}; "
                f"{by_n[n]['a']:.2f} {by_n[n]['b']:.2f} {by_n[n]['c']:.2f} {by_n[n]['d']:.2f}" for n in sizes) + " |")
    print("\n## Both co-primaries pass the full rule (at least one in brackets)")
    for name, by_n in o["joint"].items():
        print(f"{name:22s} " + "  ".join(f"n={n}: {v['both']:.2f} ({v['at_least_one']:.2f})" for n, v in by_n.items()))
    if o["nested"]:
        print("\n## Nested check (path fixed, 200 inner replicates): nested / approx for (a), (b), (a)-(c)")
        for key, by_n in o["nested"].items():
            print(f"{key:15s} " + "  ".join(f"n={n}: {v['nested_a']:.2f}/{v['approx_a']:.2f}, {v['nested_b']:.2f}/"
                                            f"{v['approx_b']:.2f}, {v['nested_abc']:.2f}/{v['approx_abc']:.2f}"
                                            for n, v in by_n.items()))
    print("\nrecommendation (pre-stated rules): " + json.dumps(o["recommendation"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--calibration-dir", "--hs-scores-dir", "--replication-json", "--hs-result", "--frozen-json", "--output"):
        p.add_argument(name, required=True)
    p.add_argument("--sizes", type=s2.sizes_arg, default=SIZES)
    p.add_argument("--replicates", type=int, default=s2.REPLICATES)
    p.add_argument("--refs-studies", type=int, default=s2.REPLICATES, help="studies that get the K null references")
    p.add_argument("--nested-outer", type=int, default=NESTED_OUTER)
    p.add_argument("--nested-inner", type=int, default=NESTED_INNER)
    p.add_argument("--workers", type=int, default=1)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
