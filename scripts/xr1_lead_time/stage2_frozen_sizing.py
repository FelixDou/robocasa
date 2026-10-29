"""R9: Stage 2 sizing after the author's 2026-10-29 decisions (plan: docs/xr1_review_analyses_plan.md, R9).

Exploratory. Every input cohort is already open, so these numbers size Stage 2 and test nothing.

Subcommands:
  e2  E2's power with each fused path frozen. The path (u, s) of SAFE + P and of Hide-and-Seek + P
      is the one the 15.15 / 15.22 fit chose (held-out calibration successes, training failures),
      chosen once. Each simulated study uses the draws of e2_coprimary_sizing. It sets the margins
      of P and of each frozen path on its drawn calibration successes only (fixed_path.certify;
      an infinite margin disables the monitor). It then records the gain over P, the fused false
      alarms and the matched gain on the drawn test sample. The matched gain is against P
      loosened on that sample to the fused detector's false alarms. The adaptive fit of
      e2_coprimary_sizing is recomputed on the same draws. It is the comparison and, with
      --coprimary-json, a reproduction check.
  e3  E3's primary contrast on RLDX-1: a reset triggered by Hide-and-Seek + P against the same
      reset triggered by P, with stage2_sizing.e3_policy unchanged. SAFE + P is run alongside.
      With --reference-json, it must reproduce an earlier stage2_sizing e3 run.

Usage:
  python3 -I stage2_frozen_sizing.py e2 --calibration-dir <SAFE20>/calibration/lstm \\
      --hs-scores-dir <rldx1_hide_and_seek run>/scores --replication-json rldx_lead_lstm.json \\
      --hs-result rldx_hide_and_seek.json [--coprimary-json e2_coprimary.json] [--workers 8] --output OUT.json
  python3 -I stage2_frozen_sizing.py e3 --calibration-dir ... --hs-scores-dir ... --replication-json ... \\
      --hs-result ... [--reference-json e3_r_0_245.json ...] --output OUT.json
"""
import argparse, json, math, multiprocessing as mp, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_hide_and_seek as chs
import confirm_lead as cl
import e2_coprimary_sizing as ec
import fixed_path as fp
import quantile_timer as qt
import review_analyses as ra
import rldx_hide_and_seek as rhs
import rldx_replication as rr
import stage2_sizing as s2

PLAN = "docs/xr1_review_analyses_plan.md (R9, 2026-10-29)"
KEYS = ec.KEYS
LEVELS = ec.LEVELS
GAINS = ec.GAINS
DECISION_LEVEL, TARGET_POWER, E2_N = 0.975, 0.8, 20
E3_RETRY = ("0.245", "0.17")
E3_CONTINUATION = (0.0, 0.0405, 0.05, 0.07, 0.10, 0.15, 0.20, 0.27, 0.40)   # adds the pilot's c and its upper bound
E3_SCENES_PER_TASK = 50
Z95 = 1.959964


# ---- inputs and checks ---------------------------------------------------------------------

def load(args):
    """The 15.15 rows with the 15.22 Hide-and-Seek trajectories attached (as e2_coprimary_sizing)."""
    rows, _, tasks = rhs.load_rows(args.calibration_dir)
    train = [r for r in rows if r["split"] == "train"]
    cal_s = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "evaluation"]
    sd = Path(args.hs_scores_dir)
    chs.attach(cal_s, chs.load_scores(sd / "calibration_predictions.jsonl"), "calibration")
    chs.attach([r for r in train if r["failed"]], chs.load_scores(sd / "train_predictions.jsonl"), "train")
    chs.attach(test, chs.load_scores(sd / "test_predictions.jsonl"), "test")
    rk = qt.rho(rr.dev_fractions(train, tasks), rr.Q)
    return rows, train, cal_s, test, tasks, rk


def input_hashes(args):
    sd = Path(args.hs_scores_dir)
    files = {f"hs_{n}": sd / f"{n}_predictions.jsonl" for n in ("train", "calibration", "test")}
    files.update(replication_json=args.replication_json, hs_result=args.hs_result)
    for name in ("coprimary_json",):
        if getattr(args, name, None):
            files[name] = getattr(args, name)
    for i, p in enumerate(getattr(args, "reference_json", None) or []):
        files[f"reference_json_{i}"] = p
    return {k: dict(path=str(Path(v).resolve()), sha256=s2.sha256(v)) for k, v in files.items()}


def replication(args, cal_s, train, test, rk):
    """The 15.15 / 15.22 gains must reproduce; returns the checks and the frozen paths."""
    got = ec.replication_gains(cal_s, train, test, rk)
    want = dict(safe=json.load(open(args.replication_json))["primary"]["gain"],
                hide_and_seek=json.load(open(args.hs_result))["detectors"]["hide_and_seek"]["gain"])
    bad = [f"{k} gain {got[k]} != {want[k]}" for k in KEYS if abs(got[k] - want[k]) >= 1e-9]
    cal = cal_s + [r for r in train if r["failed"]]
    paths = {}
    for k in KEYS:
        _, fu = cl.fit_fast(cl.Cohort(cal, k), rk)
        paths[k] = dict(u=fu["u"], s=fu["s"], lam_at_fit=round(fu["lam"], 4))
    print(f"check: replication gains SAFE {got['safe']:+.4f} (want {want['safe']:+.4f}), Hide-and-Seek "
          f"{got['hide_and_seek']:+.4f} (want {want['hide_and_seek']:+.4f}) -> {'OK' if not bad else 'MISMATCH'}", flush=True)
    if bad:
        sys.exit("STOP: " + "; ".join(bad))
    return dict(replication=got, reference=want, ok=True), paths


# ---- fixed-path evaluation -----------------------------------------------------------------

def evaluate_cert(e, rk, det):
    """Proposition 1: a monitor whose margin is infinite is disabled and raises no alarm."""
    if math.isfinite(det["lam"]):
        return cl.evaluate_fast(e, rk, det)
    n = int((~e.failed).sum())
    return dict(fpr=0.0, fp=0, fpr_wilson95=cl.wilson(0, n), mean_lead=0.0, disabled=True), np.zeros(int(e.failed.sum()))


def loosen_fast(te, rk, target_fp, grid=ra.DELTA_GRID):
    """review_analyses.loosen_timer by bisection. The timer's false alarms and its lead both fall
    as its margin grows, so the best margin is the smallest grid value with at most target_fp."""
    ev = lambda i: cl.evaluate_fast(te, rk, dict(lam=float(grid[i]), u=math.inf, s=0.0))
    lo, hi = 0, len(grid) - 1
    if ev(hi)[0]["fp"] > target_fp:
        return None
    if ev(lo)[0]["fp"] <= target_fp:
        hi = lo
    while hi - lo > 1:                     # fp(grid[lo]) > target >= fp(grid[hi])
        mid = (lo + hi) // 2
        if ev(mid)[0]["fp"] <= target_fp:
            hi = mid
        else:
            lo = mid
    e, l = ev(hi)
    return float(grid[hi]), e, l


def frozen_study(c, e, rk, path, lP):
    """One detector, path fixed: margin from c's successes, evaluated on e against P (lead lP)."""
    _, fz = fp.certify(c, rk, path)
    eZ, lZ = evaluate_cert(e, rk, fz)
    n_s = int((~e.failed).sum())
    on = math.isfinite(fz["lam"])
    row = dict(gain=float(np.mean(lZ - lP)), fp=eZ["fp"], n_s=n_s, fpr=eZ["fpr"], wilson_up=eZ["fpr_wilson95"][1],
               uses_score=on, disabled=not on, lam=fz["lam"])
    loose = loosen_fast(e, rk, eZ["fp"])
    matched = dict(row, gain=math.nan if loose is None else float(np.mean(lZ - loose[2])),
                   P_delta=None if loose is None else loose[0], P_fp=None if loose is None else loose[1]["fp"])
    return row, matched


# ---- E2 simulation (fork workers inherit the pools) ----------------------------------------

_G = {}


def _job(args):
    cell, n, mode, r0, r1 = args
    pools, rk, paths = _G["pools"], _G["rk"], _G["paths"]
    PS = pools["safe"]
    out = []
    for r in range(r0, r1):
        # e2_coprimary_sizing._job's stream: tasks (with_tasks only), then calibration, then test.
        rng = np.random.default_rng([s2.SEED, cell, r])
        tasks = list(rng.choice(PS.tasks, len(PS.tasks))) if mode == "with_tasks" else None
        ci, ti = PS.draw(rng, n, 0.0, tasks), PS.draw(rng, n, 0.0, tasks)
        row, lP = {}, None
        for key in KEYS:
            c, e = pools[key].C.take(ci), pools[key].C.take(ti)
            tm, fu = cl.fit_fast(c, rk)
            if lP is None:
                eP, lP = cl.evaluate_fast(e, rk, tm)
                tmc, _ = fp.certify(c, rk, paths[key])
                ePc, lPc = evaluate_cert(e, rk, tmc)
                row["P_fpr"] = eP["fpr"]
                row["P_cert"] = dict(disabled=not math.isfinite(tmc["lam"]), fp=ePc["fp"], lead=ePc["mean_lead"])
            eF, lF = cl.evaluate_fast(e, rk, fu)
            # The adaptive row is e2_coprimary_sizing's, field for field.
            row[key] = dict(gain=float(np.mean(lF - lP)), fp=eF["fp"], n_s=int((~e.failed).sum()), fpr=eF["fpr"],
                            wilson_up=eF["fpr_wilson95"][1], uses_score=bool(math.isfinite(fu["u"])))
            row[f"{key}@frozen"], row[f"{key}@matched"] = frozen_study(c, e, rk, paths[key], lPc)
        out.append(row)
    return cell, out


def simulate(pools, rk, paths, sizes, R, workers, chunk=25):
    _G.update(pools=pools, rk=rk, paths=paths)
    jobs, cells = [], {}
    for i, n in enumerate(sizes):
        for mode in ("within_task", "with_tasks"):
            cid = 1000 * i + (0 if mode == "within_task" else 1)
            cells[cid] = (n, mode)
            jobs += [(cid, n, mode, a, min(R, a + chunk)) for a in range(0, R, chunk)]
    if workers <= 1:
        res = [_job(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as p:
            res = p.map(_job, jobs, chunksize=1)
    out = defaultdict(list)
    for cid, rows in res:
        out[cid] += rows
    return {(n, mode): out[cid] for cid, (n, mode) in cells.items()}


# ---- E2 power ------------------------------------------------------------------------------

def gains(rows, k):
    return np.array([r[k]["gain"] for r in rows], float)


def half_width(g, level):
    g = g[np.isfinite(g)]
    return float(np.median(g) - np.percentile(g, 100 * (1 - level) / 2)) if len(g) else math.nan


def above(g, shift, hw):
    """Per study: decision CI (gain - hw, as e2_coprimary_sizing.ci_pass) above zero."""
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(g), np.round(g + shift, 4) - hw > 0, False)


def e2_rule(rows, k):
    return np.array([r[k]["fp"] <= ec.binom_max_false_alarms(r[k]["n_s"]) for r in rows])


def frozen_summary(rows, key):
    g, gm = gains(rows, f"{key}@frozen"), gains(rows, f"{key}@matched")
    f = [r[f"{key}@frozen"] for r in rows]
    loosened = [r[f"{key}@matched"]["P_fp"] for r in rows if r[f"{key}@matched"]["P_fp"] is not None]
    return dict(replicates=len(rows), mean_gain=round(float(g.mean()), 4), sd=round(float(g.std()), 4),
                adaptive_mean_gain=round(float(gains(rows, key).mean()), 4), adaptive_sd=round(float(gains(rows, key).std()), 4),
                mean_fused_fpr=round(float(np.mean([x["fpr"] for x in f])), 4),
                mean_P_fpr=round(float(np.mean([r["P_fpr"] for r in rows])), 4),
                fpr_pass_e2=round(float(e2_rule(rows, f"{key}@frozen").mean()), 3),
                disabled=round(float(np.mean([x["disabled"] for x in f])), 4),
                P_disabled=round(float(np.mean([r["P_cert"]["disabled"] for r in rows])), 4),
                matched_mean_gain=round(float(np.nanmean(gm)), 4), matched_sd=round(float(np.nanstd(gm)), 4),
                matched_infeasible=int(np.sum(~np.isfinite(gm))),
                mean_P_fp_loosened=round(float(np.mean(loosened)), 2) if loosened else None)


def frozen_power(sims, sizes):
    f = lambda m: round(float(np.mean(m)), 3)
    out = dict(summary={}, power={}, joint={})
    for n in sizes:
        w, t = sims[(n, "within_task")], sims[(n, "with_tasks")]
        out["summary"][str(n)] = {k: dict(within_task=frozen_summary(w, k), with_tasks=frozen_summary(t, k)) for k in KEYS}
        passes = {}
        for key in KEYS:
            g, gm = gains(w, f"{key}@frozen"), gains(w, f"{key}@matched")
            g_t, gm_t = gains(t, f"{key}@frozen"), gains(t, f"{key}@matched")
            fpr = e2_rule(w, f"{key}@frozen")
            for gamma in (None,) + GAINS[key]:
                gname = "plug_in" if gamma is None else str(gamma)
                shift = 0.0 if gamma is None else gamma - float(g.mean())
                for level in LEVELS:
                    hw, hwm = half_width(g, level), half_width(gm, level)
                    hw_t, hwm_t = half_width(g_t, level), half_width(gm_t, level)
                    a, m = above(g, shift, hw), above(gm, shift, hwm)
                    at, mt = above(g, shift, hw_t), above(gm, shift, hwm_t)
                    passes[(key, gname, level)] = (a & fpr, a & m & fpr)
                    out["power"].setdefault(key, {}).setdefault(gname, {}).setdefault(str(level), {})[str(n)] = dict(
                        ci=f(a), ci_fpr_e2=f(a & fpr), matched_ci=f(m), both_fpr_e2=f(a & m & fpr),
                        task_ci_fpr_e2=f(at & fpr), task_both_fpr_e2=f(at & mt & fpr),
                        hw_within=round(hw, 4), hw_matched=round(hwm, 4), hw_tasks=round(hw_t, 4))
        for gs, gh in ((None, None),) + ec.JOINT:
            name = "plug_in" if gs is None else f"safe {gs} / hs {gh}"
            for level in LEVELS:
                ps = passes[("safe", "plug_in" if gs is None else str(gs), level)]
                ph = passes[("hide_and_seek", "plug_in" if gh is None else str(gh), level)]
                out["joint"].setdefault(name, {}).setdefault(str(level), {})[str(n)] = {
                    c: dict(safe=f(ps[i]), hide_and_seek=f(ph[i]), at_least_one=f(ps[i] | ph[i]), both=f(ps[i] & ph[i]))
                    for i, c in enumerate(("gain", "gain_and_matched"))}
    return out


def recommend(power, sizes):
    """The pre-stated E2 rule: n = 20 stands for a detector if plug-in power (a) at 97.5% is at
    least 0.8; otherwise the smallest n that reaches it, or 'underpowered'."""
    out = {}
    for key in KEYS:
        by_n = power[key]["plug_in"][str(DECISION_LEVEL)]
        rec = {}
        for crit in ("ci_fpr_e2", "both_fpr_e2"):
            ok = [n for n in sizes if by_n[str(n)][crit] >= TARGET_POWER]
            at = by_n.get(str(E2_N), {}).get(crit)
            rec[crit] = dict(power_at_20=at, smallest_n=min(ok) if ok else None,
                             reading=("n = 20 stands" if at is not None and at >= TARGET_POWER else
                                      f"n = {min(ok)} needed" if ok else "underpowered at n <= 60"))
        out[key] = rec
    return out


def check_coprimary(sims, sizes, ref_path):
    """The adaptive rows must give e2_coprimary_sizing's tables for every size both runs have."""
    ref = json.load(open(ref_path))
    common = [n for n in sizes if str(n) in ref["summary"]]
    mine = json.loads(json.dumps(ec.power_tables(sims, common), default=float))
    bad = []
    for n in map(str, common):
        for k in KEYS:
            if mine["summary"][n][k] != ref["summary"][n][k]:
                bad.append(f"summary n={n} {k}")
        for k, by_g in ref["power"].items():
            for g, by_l in by_g.items():
                for lvl, by_n in by_l.items():
                    if n in by_n and mine["power"][k][g][lvl][n] != by_n[n]:
                        bad.append(f"power {k} {g} {lvl} n={n}")
    return dict(ok=not bad and bool(common), sizes=common, replicates=ref.get("replicates"), mismatches=bad[:20])


def e2(args):
    out_path = s2.check_out(args.output)
    rows, train, cal_s, test, tasks, rk = load(args)
    checks, paths = replication(args, cal_s, train, test, rk)
    pools = ec.build_pools(cal_s, test, tasks)
    sims = simulate(pools, rk, paths, args.sizes, args.replicates, args.workers)
    tables = frozen_power(sims, args.sizes)
    adaptive = ec.power_tables(sims, args.sizes)
    status = 0
    if args.coprimary_json:
        cc = check_coprimary(sims, args.sizes, args.coprimary_json)
        checks["coprimary_reproduced"] = cc
        print("check: adaptive rows reproduce e2_coprimary tables " + ("MATCH" if cc["ok"] else f"MISMATCH {cc['mismatches']}"),
              flush=True)
        status = 0 if cc["ok"] else 1
    out = dict(plan=PLAN, experiment="E2, path frozen", status="exploratory sizing (opened cohorts)"
               + ("" if status == 0 else "; FAILED CHECK, do not use"), seed=s2.SEED, replicates=args.replicates,
               sizes=list(args.sizes), levels=list(LEVELS), gains={k: list(v) for k, v in GAINS.items()},
               frozen_paths=paths, e2_fpr_rule=dict(bound=ec.FPR_BOUND, alpha=ec.FPR_ALPHA, max_false_alarms={
                   str(20 * n): ec.binom_max_false_alarms(20 * n) for n in args.sizes}),
               calibration_dir=str(Path(args.calibration_dir).resolve()), inputs=input_hashes(args),
               pool_counts=pools["safe"].counts(), checks=checks, frozen=tables, adaptive=adaptive,
               recommendation=recommend(tables["power"], args.sizes))
    s2.write(out_path, out)
    print_e2(out)
    return status


def print_e2(out):
    print("\nfrozen paths: " + ", ".join(f"{k} u={p['u']} s={p['s']}" for k, p in out["frozen_paths"].items()))
    print("\nPlug-in sampling distribution (within task), frozen | adaptive: mean gain (SD); matched; E2 FPR pass; disabled")
    for n, v in out["frozen"]["summary"].items():
        print(f"n={n:>3}  " + "  ".join(
            f"{k}: {s['within_task']['mean_gain']:+.4f} ({s['within_task']['sd']:.4f}) | "
            f"{s['within_task']['adaptive_mean_gain']:+.4f} ({s['within_task']['adaptive_sd']:.4f}); "
            f"matched {s['within_task']['matched_mean_gain']:+.4f} ({s['within_task']['matched_sd']:.4f}); "
            f"{s['within_task']['fpr_pass_e2']:.2f}; {s['within_task']['disabled']:.3f}" for k, s in v.items()))
    print("\nPower, path frozen (E2 FPR rule): (a) gain CI > 0 | (b) gain and matched CI > 0 | task-level (a)")
    for key, by_g in out["frozen"]["power"].items():
        for g, by_l in by_g.items():
            for level, by_n in by_l.items():
                print(f"{key:14s} gain {g:>8s} CI {float(level):.3f}: " + "  ".join(
                    f"n={n}: {p['ci_fpr_e2']:.2f} | {p['both_fpr_e2']:.2f} | {p['task_ci_fpr_e2']:.2f} (hw {p['hw_within']:.4f})"
                    for n, p in by_n.items()))
    print("\nAdaptive path, same draws (E2 FPR rule, 97.5%): " + "  ".join(
        f"{k} n={n}: {p['ci_fpr_e2']:.2f}" for k in KEYS
        for n, p in out["adaptive"]["power"][k]["plug_in"]["0.975"].items()))
    for name, by_l in out["frozen"]["joint"].items():
        p = by_l["0.975"]
        print(f"joint {name} 97.5%: " + "  ".join(
            f"n={n}: one {x['gain']['at_least_one']:.2f}, both {x['gain']['both']:.2f}; with matched one "
            f"{x['gain_and_matched']['at_least_one']:.2f}, both {x['gain_and_matched']['both']:.2f}" for n, x in p.items()))
    print("\nrecommendation (pre-stated rule): " + json.dumps(out["recommendation"]))


# ---- E3 ------------------------------------------------------------------------------------

def e3_inputs(rows, train, cal_s, test, tasks, rk):
    """stage2_sizing.e3's RLDX-1 inputs, once per fused detector (P and none are shared)."""
    calF = cal_s + [r for r in train if r["failed"]]
    rates = s2.success_rates(rows)
    succ = defaultdict(list)
    for r in train:
        if not r["failed"]:
            succ[r["task_name"]].append(r["L"])
    H = {r["task_name"]: r["horizon"] for r in rows}
    types = {r["task_name"]: r["task_type"] for r in rows}
    D = {t: np.median(succ[t]) if succ[t] else H[t] for t in tasks}
    hazard = s2.late_hazard(rows)
    ti, check = {}, {}
    for k in KEYS:
        tm, fu = cl.fit_fast(cl.Cohort(calF, k), rk)
        teC = cl.Cohort(test, k)
        eP, _ = cl.evaluate_fast(teC, rk, tm)
        eF, _ = cl.evaluate_fast(teC, rk, fu)
        check[k] = dict(P_lead=eP["mean_lead"], fused_lead=eF["mean_lead"], P_fpr=eP["fpr"], fused_fpr=eF["fpr"],
                        path=cl.det_json(fu))
        ti[k] = s2.task_inputs(teC, rk, tm, fu, rates, D, types)
        for t, h in hazard.items():
            if t in ti[k]:
                ti[k][t]["late_hazard"] = h
    return ti, check


def contrast(by_type, budgets):
    """Fused + P reset against P reset: effect, n per design and CI half-width at 50 scenes per task."""
    out = {}
    for tt, a in by_type.items():
        out[tt] = {}
        N = E3_SCENES_PER_TASK * a["tasks"]
        for B in map(str, budgets):
            e = a["episodes"][B]["safe_vs_P"]
            pP, pF = a["mean_p"] + a["multi"][B]["P"], a["mean_p"] + a["multi"][B]["safe_plus_P"]
            d, psi = e["delta"], e["discordant"]
            out[tt][B] = dict(p_P=round(pP, 4), p_fused=round(pF, 4), delta=d, discordant=psi,
                              n_independent=e["unpaired"], n_branching=e["paired"], scenes=N,
                              hw_independent=round(Z95 * math.sqrt((pP * (1 - pP) + pF * (1 - pF)) / N), 4),
                              hw_branching=round(Z95 * math.sqrt(max(psi - d * d, 0.0) / N), 4),
                              false_alarm_exposure=dict(P=a["false_alarm_exposure"]["P"],
                                                        fused=a["false_alarm_exposure"]["safe_plus_P"]))
    return out


def decide(c):
    """The pre-stated E3 rule, on all tasks at the contrast budget."""
    n = c["n_branching"]
    if n <= 1000:
        return dict(decision="TESTABLE", note=f"H4 can be registered with the branching design ({n} pairs)")
    half = c["hw_branching"] <= 0.5 * c["hw_independent"]
    return dict(decision="ESTIMATION", recommend_branching=bool(half),
                note=f"{n} branching pairs needed; half-widths at {c['scenes']} scenes: independent "
                     f"{c['hw_independent']}, branching {c['hw_branching']}")


def check_reference(ref_path, safe_res, r_mode):
    ref = json.load(open(ref_path))
    if str(ref.get("retry_success", "same")) != str(r_mode):
        return None
    want = ref["policies"]["rldx1"]["by_type"]
    bad = [(tt, kind, B) for tt, a in want.items() for kind in ("single", "multi") for B, m in a[kind].items()
           if safe_res["by_type"][tt][kind].get(B) is None
           or any(safe_res["by_type"][tt][kind][B][k] != x for k, x in m.items())]
    return dict(path=str(Path(ref_path).resolve()), ok=not bad, mismatches=bad[:10])


def e3(args):
    out_path = s2.check_out(args.output)
    rows, train, cal_s, test, tasks, rk = load(args)
    checks, _ = replication(args, cal_s, train, test, rk)
    ti, leads = e3_inputs(rows, train, cal_s, test, tasks, rk)
    checks["leads"] = leads
    budgets = tuple(args.budgets) + (() if args.contrast_budget in args.budgets else (args.contrast_budget,))
    out = dict(plan=PLAN, experiment="E3, Hide-and-Seek + P reset vs P reset (RLDX-1)",
               status="exploratory sizing (opened cohorts, shadow data)", seed=s2.SEED + 1, budgets=list(budgets),
               contrast_budget=args.contrast_budget, continuation=list(args.continuation),
               scenes_per_task=E3_SCENES_PER_TASK, calibration_dir=str(Path(args.calibration_dir).resolve()),
               inputs=input_hashes(args), checks=checks, runs={})
    status, refs_used = 0, []
    for r_mode in args.retry_success:
        res = {k: s2.e3_policy(ti[k], args.budgets, r_mode, s2.SEED + 1, args.continuation, args.contrast_budget) for k in KEYS}
        # none and P see the same alarms and draws in both runs, so they must agree exactly.
        shared = all(res["safe"]["by_type"][tt][kind][B][d] == res["hide_and_seek"]["by_type"][tt][kind][B][d]
                     for tt in res["safe"]["by_type"] for kind in ("single", "multi")
                     for B in res["safe"]["by_type"][tt][kind] for d in ("none", "P"))
        run = dict(shared_arms_identical=shared, references=[])
        for p in args.reference_json or []:
            c = check_reference(p, res["safe"], r_mode)
            if c is not None:
                run["references"].append(c)
                refs_used.append(p)
                print(f"check r={r_mode}: SAFE + P reproduces {Path(p).name} " + ("MATCH" if c["ok"] else f"MISMATCH {c['mismatches']}"),
                      flush=True)
                status |= 0 if c["ok"] else 1
        print(f"check r={r_mode}: none and P arms identical across the two runs -> {'OK' if shared else 'MISMATCH'}", flush=True)
        status |= 0 if shared else 1
        run["contrast"] = {k: contrast(res[k]["by_type"], budgets) for k in KEYS}
        run["decision"] = decide(run["contrast"]["hide_and_seek"]["all"][str(args.contrast_budget)])
        run["continuation"] = {k: {tt: a["continuation"] for tt, a in res[k]["by_type"].items()} for k in KEYS}
        run["by_type"] = {k: {tt: {x: a[x] for x in ("tasks", "mean_p", "mean_r", "single", "multi", "episodes",
                                                      "false_alarm_exposure")}
                              for tt, a in res[k]["by_type"].items()} for k in KEYS}
        run["per_task"] = {k: res[k]["per_task"] for k in KEYS}
        out["runs"][str(r_mode)] = run
    unused = [p for p in args.reference_json or [] if p not in refs_used]
    if unused:
        print(f"STOP: no run matched the retry rate of {unused}", flush=True)
        status = 1
    if status:
        out["status"] += "; FAILED CHECK, do not use"
    s2.write(out_path, out)
    print_e3(out)
    return status


def print_e3(out):
    B = str(out["contrast_budget"])
    for r_mode, run in out["runs"].items():
        print(f"\n## E3 RLDX-1, retry success r = {r_mode}: fused + P reset vs P reset")
        print("| detector | type | budget x H | p (P) | p (fused) | delta | n independent | n branching | "
              "hw independent | hw branching | scenes |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in KEYS:
            for tt, by_b in run["contrast"][k].items():
                for b, c in by_b.items():
                    print(f"| {k} | {tt} | {b} | {c['p_P']:.4f} | {c['p_fused']:.4f} | {c['delta']:+.4f} | "
                          f"{c['n_independent']} | {c['n_branching']} | {c['hw_independent']:.4f} | {c['hw_branching']:.4f} | {c['scenes']} |")
        print(f"decision at {B} H (Hide-and-Seek, all tasks): {json.dumps(run['decision'])}")
        cont = run["continuation"]["hide_and_seek"]["all"]
        print("reset - continuation (all tasks), P | Hide-and-Seek + P: " + "  ".join(
            f"c={c}: {x['P']['delta']:+.4f} | {x['safe_plus_P']['delta']:+.4f}" for c, x in cont.items()))


# ---- CLI -----------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("e2", "e3"):
        s = sub.add_parser(name)
        s.add_argument("--calibration-dir", required=True)
        s.add_argument("--hs-scores-dir", required=True)
        s.add_argument("--replication-json", required=True)
        s.add_argument("--hs-result", required=True)
        s.add_argument("--output", required=True)
        if name == "e2":
            s.add_argument("--coprimary-json", help="e2_coprimary_sizing output; its tables must reproduce")
            s.add_argument("--sizes", type=s2.sizes_arg, default=s2.SIZES)
            s.add_argument("--replicates", type=int, default=s2.REPLICATES)
            s.add_argument("--workers", type=int, default=1)
        else:
            s.add_argument("--retry-success", nargs="+", default=list(E3_RETRY))
            s.add_argument("--budgets", type=s2.gains_arg, default=s2.BUDGETS)
            s.add_argument("--continuation", type=s2.gains_arg, default=E3_CONTINUATION)
            s.add_argument("--contrast-budget", type=float, default=s2.CONTRAST_BUDGET)
            s.add_argument("--reference-json", action="append",
                           help="a stage2_sizing e3 output; SAFE + P at its retry rate must reproduce its RLDX-1 tables")
    args = p.parse_args(argv)
    return {"e2": e2, "e3": e3}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
