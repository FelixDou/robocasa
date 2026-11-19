"""R22 (docs/r22_fixed_paths_plan.md): every fused path fixed on the validation set. Exploratory.

Policy-agnostic pieces, called by seen_unseen_tables.py (XR-1) and rldx_test_table.py (RLDX-1) with
--r22. A fused monitor is a score OR a timer (hk_free's P, E = G_abs, or G). Its path (u, s) is
  adaptive  chosen by hk_free.fit on the calibration cohort itself (the paper's earlier Table II; gate
            K0 reproduces those rows with this code), or
  fixed     chosen by hk_free.fit on the validation cohort, frozen, and certified on the calibration
            successes: Proposition 1 gives the margins of the timer and of the frozen path; or
  frozen    given (H3's registered path; the bootstrap's replicates).
Every result is evaluated once on the test cohort. Delta l (matched) is the gain in mean lead over the
timer held to the monitor's test false alarms (hk_free.matched_timer).
"""
import math
from collections import defaultdict
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lead as cl  # noqa: E402
import hk_free as hk  # noqa: E402
import quantile_timer as qt  # noqa: E402

PLAN = "docs/r22_fixed_paths_plan.md (R22)"
ALL = "all"
SEEDS = ("0", "1", "2")
LABELS = ("seed_average",) + tuple(f"seed_{s}" for s in SEEDS)
TOL = 1e-12
R8_TOL = 5e-5
BOOT_B, BOOT_SEED = 1000, 22
NULL_DRAWS, NULL_SEED = 100, 22


def path_json(p):
    return dict(u=None if p["u"] == math.inf else float(p["u"]), s=float(p["s"]))


def certify(cu, rc, path):
    """Margins of the timer and of a frozen path (u, s) from the calibration successes (Proposition 1)."""
    S = ~cu.failed
    Sc, rS = hk.sub(cu, S), rc[S]
    tm = dict(u=math.inf, s=0.0, lam=cl.conformal(hk.V(Sc, rS, math.inf, 0.0)))
    fu = dict(u=path["u"], s=path["s"], lam=cl.conformal(hk.V(Sc, rS, path["u"], path["s"])))
    return tm, fu


def run_folds(pairs, timer, rk, val=None, frozen=None, keep=False):
    """pairs {F: (calibration Cohort, test Cohort)}. val {F: validation Cohort}: fixed paths chosen there;
    frozen {F: path}: given paths; neither: hk_free.fit on the calibration cohort (adaptive, which is
    hk_free.run_folds). keep: per-failure arrays in pooled fold order under "_arrays" (not for JSON)."""
    pt, pf, tests, paths, chosen = [], [], [], {}, {}
    tasks, H = [], []
    for F, (c, e) in sorted(pairs.items()):
        if len(e.L) == 0:
            continue
        cu, rc = hk.prepared(c, timer, F, None, rk)
        eu, re_ = hk.prepared(e, timer, F, None, rk)
        if frozen is not None:
            ch = frozen[F]
            tm, fu = certify(cu, rc, ch)
        elif val is not None:
            vu, rv = hk.prepared(val[F], timer, F, None, rk)
            _, ch = hk.fit(vu, rv)
            tm, fu = certify(cu, rc, ch)
        else:
            tm, fu = hk.fit(cu, rc)
            ch = fu
        pt.append(hk.raw_eval(eu, re_, tm))
        pf.append(hk.raw_eval(eu, re_, fu))
        tests.append((eu, re_))
        tasks.append(eu.task[eu.failed])
        H.append(eu.H[eu.failed])
        paths[F] = dict(timer=cl.det_json(tm), fused=cl.det_json(fu))
        chosen[F] = dict(u=ch["u"], s=ch["s"])
    fused, tmr = hk.pool(pf), hk.pool(pt)
    m = hk.matched_timer(tests, fused["fp"])
    out = hk.summarize(fused, tmr, m)
    out["paths"] = paths
    out["chosen"] = {F: path_json(p) for F, p in chosen.items()}
    out["_chosen"] = chosen
    if keep:
        out["_arrays"] = dict(fused=fused["lead"], timer=tmr["lead"], matched=None if m is None else m[0]["lead"],
                              task=np.concatenate(tasks), H=np.concatenate(H))
    return out


def strip(x):
    """A result without its private (underscored) entries, for JSON."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def block(pairs_of, val_of, labels, timers, rk):
    """pairs_of(label), val_of(label) -> {F: ...}; val_of None: adaptive. Per timer and label, and mean +- std."""
    out = {}
    for t in timers:
        res = {lab: run_folds(pairs_of(lab), t, rk, val=None if val_of is None else val_of(lab)) for lab in labels}
        seeds = [lab for lab in labels if lab != "seed_average"]
        if len(seeds) > 1:
            res["mean_std_over_seeds"] = hk.mean_std([res[s] for s in seeds])
        out[t] = res
    return out


def compare(mine, ref, keys=("fpr", "mean_lead", "gain")):
    """Max |diff| of a result against a reference row (inf if a matched gain exists in one only)."""
    d = max(abs(mine[k] - ref[k]) for k in keys)
    rm, mm = ref.get("matched"), mine.get("matched")
    if (rm is None) != (mm is None):
        return math.inf
    if rm is not None:
        d = max(d, abs(mm["gain"] - rm["gain"]))
    return d


def k0(adaptive, reference, timers, labels):
    """Gate K0: this code with the adaptive choice reproduces stored rows {timer: {label: row}}."""
    worst, missing = 0.0, []
    for t in timers:
        for lab in labels:
            ref = ((reference or {}).get(t) or {}).get(lab)
            if ref is None:
                missing.append(f"{t}/{lab}")
                continue
            worst = max(worst, compare(adaptive[t][lab], ref))
    return dict(max_abs_diff=worst, missing=missing, ok=not missing and worst <= TOL)


def ensemble_row(pairs, val, timer, rk, failed_rows):
    """A seen-task ensemble + timer with a fixed path: false alarms, gains and Table III's CIs (tasks, then
    failures, 2,000 paired resamples, conditional on the fitted thresholds and the matched timer)."""
    r = run_folds(pairs, timer, rk, val=val, keep=True)
    a = r["_arrays"]
    if len(failed_rows) != len(a["fused"]):
        raise ValueError("failure rows and leads differ in length")
    row = dict(fp=r["fp"], fpr=r["fpr"], mean_lead=r["mean_lead"], gain=r["gain"], timer=r["timer"],
               gain_ci95=qt.boot(a["fused"] - a["timer"], failed_rows)["ci95"], chosen=r["chosen"], paths=r["paths"])
    if r["matched"] is not None:
        row["matched"] = dict(r["matched"], ci95=qt.boot(a["fused"] - a["matched"], failed_rows)["ci95"])
    else:
        row["matched"] = None
    return row, r


def boot_levels(d, rows_failed, B=2000, seed=0, levels=(95.0, 97.5)):
    """quantile_timer.boot's resampling (tasks, then failures within each drawn task; paired), with
    percentile intervals at several levels. Level 95 reproduces qt.boot's ci95 exactly (R23)."""
    rng = np.random.default_rng(seed)
    bi = defaultdict(list)
    for i, r in enumerate(rows_failed):
        bi[r["task_name"]].append(i)
    tasks = sorted(bi)
    bs = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(bi[t], len(bi[t])) for t in rng.choice(tasks, len(tasks))])
        bs.append(d[ii].mean())
    out = {}
    for lv in levels:
        a = (100.0 - lv) / 2.0
        out[f"ci{lv:g}"] = np.round(np.percentile(bs, [a, 100.0 - a]), 4).tolist()
    return out


def frozen_row(pairs, frozen, timer, rk, failed_rows):
    """A seen-task ensemble + timer on a given (frozen) path, judged like Table III's other ensembles:
    false alarms, gains, and task-then-failure CIs at 95% and 97.5% (R23: H3 under H1's rule)."""
    r = run_folds(pairs, timer, rk, frozen=frozen, keep=True)
    a = r["_arrays"]
    if len(failed_rows) != len(a["fused"]):
        raise ValueError("failure rows and leads differ in length")
    row = dict(fp=r["fp"], fpr=r["fpr"], mean_lead=r["mean_lead"], gain=r["gain"], timer=r["timer"],
               gain_ci=boot_levels(a["fused"] - a["timer"], failed_rows), chosen=r["chosen"], paths=r["paths"])
    row["matched"] = None if r["matched"] is None else dict(
        r["matched"], ci=boot_levels(a["fused"] - a["matched"], failed_rows))
    return row, r


def same_point(row, ref, tol=TOL):
    """R24's gate: false alarms, gain, Delta l and the matched timer's false alarms of a recomputed row
    against a stored one (a missing row, or a matched gain in one only, is inf)."""
    if row is None or ref is None:
        return dict(max_abs_diff=math.inf, ok=False, note="row missing")
    m, rm = row.get("matched") or {}, ref.get("matched") or {}
    d = max(abs(row["gain"] - ref["gain"]), 0.0 if row["fp"] == ref["fp"] else math.inf,
            (abs(m["gain"] - rm["gain"]) if m and rm else (0.0 if not m and not rm else math.inf)),
            0.0 if m.get("timer_fp") == rm.get("timer_fp") else math.inf)
    return dict(fp=row["fp"], ref_fp=ref["fp"], matched_gain=m.get("gain"), ref_matched_gain=rm.get("gain"),
                timer_fp=m.get("timer_fp"), ref_timer_fp=rm.get("timer_fp"), max_abs_diff=d, ok=d <= tol)


def tree_diff(new, ref, path=""):
    """(max |diff|, first path at that diff) over every entry of ref; new may hold more keys. A missing key,
    a changed string, bool or length, or a number against a non-number is inf. NaN equals NaN."""
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            return math.inf, path
        worst = (0.0, None)
        for k, v in ref.items():
            if k not in new:
                return math.inf, f"{path}/{k}"
            d = tree_diff(new[k], v, f"{path}/{k}")
            if d[0] > worst[0]:
                worst = d
        return worst
    if isinstance(ref, list):
        if not isinstance(new, list) or len(new) != len(ref):
            return math.inf, path
        worst = (0.0, None)
        for i, (x, y) in enumerate(zip(new, ref)):
            d = tree_diff(x, y, f"{path}/{i}")
            if d[0] > worst[0]:
                worst = d
        return worst
    if isinstance(ref, bool) or ref is None or isinstance(ref, str):
        return (0.0, None) if new == ref and type(new) is type(ref) else (math.inf, path)
    if isinstance(new, bool) or not isinstance(new, (int, float)):
        return math.inf, path
    if math.isnan(ref) or math.isnan(new):
        return (0.0, None) if math.isnan(ref) and math.isnan(new) else (math.inf, path)
    if math.isinf(ref) or math.isinf(new):
        return (0.0, None) if new == ref else (math.inf, path)
    d = abs(float(new) - float(ref))
    return d, (path if d > 0 else None)


def per_type(result, types, spi):
    """Delta l (matched) by task type and per task, from a run_folds(keep=True) result."""
    import r19_extras as rx
    a = result["_arrays"]
    if a["matched"] is None:
        return None
    detail = dict(task=a["task"], H=a["H"], timer=a["timer"], fused=a["fused"], matched=a["matched"])
    return dict(seconds=rx.seconds(detail, types, spi), per_task=rx.per_task(detail))


# ---- unseen ensembles: full-pipeline bootstrap with the paths frozen ---------------------------

def boot_unseen(design, key_ofs, val_of, timers, rk, B=BOOT_B, seed=BOOT_SEED, contrasts=()):
    """design: r19_extras.UnseenDesign. key_ofs {name: key_of}; val_of(name) -> {F: validation Cohort}.
    Point estimates choose each fold's path on the validation set; every replicate keeps those paths,
    draws tasks, then calibration and test rollouts within task and class, recertifies the margins, refits
    the timer and re-matches it. Percentile 95% CIs of the gain and Delta l."""
    point = {t: {n: run_folds(design.pairs(k), t, rk, val=val_of(n)) for n, k in key_ofs.items()} for t in timers}
    rng = np.random.default_rng(seed)
    same = {t: {n: [] for n in key_ofs} for t in timers}
    matched = {t: {n: [] for n in key_ofs} for t in timers}
    for _ in range(B):
        cal_pick, te_pick = design.draw(rng)
        for t in timers:
            for n, k in key_ofs.items():
                pairs = design.pairs(k, cal_pick, te_pick)
                r = run_folds(pairs, t, rk, frozen=point[t][n]["_chosen"])
                same[t][n].append(r["gain"])
                matched[t][n].append(r["matched"]["gain"] if r["matched"] else float("nan"))
    pct = lambda v: [float(x) for x in np.nanpercentile(np.asarray(v, float), [2.5, 97.5])]
    out = dict(samples=B, seed=seed, timers={})
    for t in timers:
        row = {}
        for n in key_ofs:
            p = point[t][n]
            row[n] = dict(fpr=p["fpr"], fp=p["fp"], gain=p["gain"], gain_ci95=pct(same[t][n]),
                          matched_gain=p["matched"]["gain"] if p["matched"] else None, matched_ci95=pct(matched[t][n]),
                          matched_failed_replicates=int(np.sum(np.isnan(matched[t][n]))), timer_fp=p["timer"]["fp"],
                          chosen=p["chosen"])
        for a, b in contrasts:
            if a in key_ofs and b in key_ofs:
                row[f"{a}_minus_{b}"] = dict(
                    matched_gain=(row[a]["matched_gain"] - row[b]["matched_gain"])
                    if row[a]["matched_gain"] is not None and row[b]["matched_gain"] is not None else None,
                    matched_ci95=pct(np.asarray(matched[t][a]) - np.asarray(matched[t][b])))
        out["timers"][t] = row
    return out, point


# ---- the R8 cross-check -------------------------------------------------------------------------

def k8(results, r8, names):
    """Gate K8: the seen-task fixed paths of SAFE-MLP and Hide-and-Seek (ensembles) are R8's, and their
    same-target gains equal R8's at its 4 decimals. names {r8 key: result key}."""
    rows, ok = [], True
    for theirs, mine in names.items():
        ref = (r8.get("rows") or {}).get(theirs)
        got = results.get(mine)
        if ref is None or got is None:
            rows.append(dict(score=mine, found=False))
            ok = False
            continue
        ch = got["chosen"][ALL]
        rp = ref["path"]
        same_path = (ch["u"] is None and rp.get("u") is None) or (
            ch["u"] is not None and rp.get("u") is not None and abs(ch["u"] - float(rp["u"])) <= R8_TOL)
        same_path = same_path and abs(ch["s"] - float(rp["s"])) <= R8_TOL
        d = abs(got["gain"] - float(ref["gain"]))
        good = same_path and d <= R8_TOL
        ok &= good
        rows.append(dict(score=mine, found=True, path=ch, r8_path=dict(u=rp.get("u"), s=rp.get("s")),
                         gain=got["gain"], r8_gain=ref["gain"], ok=good))
    return dict(ok=bool(ok and rows), rows=rows)


# ---- H2 nulls with fixed paths -------------------------------------------------------------------

def nulls(cal, te, val, rk, safe_norms, hs_norms, hs, monitors, B=NULL_DRAWS, seed=NULL_SEED):
    """Noise and within-task resampled SAFE and Hide-and-Seek scores, built on the calibration and test
    rows (pooled per task over both) and, separately, on the validation rows; each null gets its own fixed
    path. monitors {name: matched gain} are ranked among the nulls of the same family."""
    import confirm_hide_and_seek as chs
    rng = np.random.default_rng(seed)
    keys = ("noise", "safe_resampled", f"{hs}_resampled")
    draws = {k: [] for k in keys}
    for _ in range(B):
        cl.null_scores(cal + te, rng, safe_norms)
        chs.resample_null(cal + te, rng, hs, hs_norms)
        cl.null_scores(val, rng, safe_norms)
        chs.resample_null(val, rng, hs, hs_norms)
        for k in keys:
            r = run_folds({ALL: (cl.Cohort(cal, k), cl.Cohort(te, k))}, "P", rk, val={ALL: cl.Cohort(val, k)})
            draws[k].append(dict(gain=r["gain"], fp=r["fp"], matched=r["matched"]["gain"] if r["matched"] else None))
    out = dict(draws=B, seed=seed, nulls={})
    for k, v in draws.items():
        m = [x["matched"] for x in v if x["matched"] is not None]
        out["nulls"][k] = dict(same_target=dict(mean=float(np.mean([x["gain"] for x in v])), max=float(max(x["gain"] for x in v))),
                               matched=dict(mean=float(np.mean(m)), max=float(max(m)), q95=float(np.quantile(m, 0.95))),
                               mean_fp=float(np.mean([x["fp"] for x in v])))
    for name, (mine, family) in monitors.items():
        out[name] = dict(matched_gain=mine)
        for k in ("noise", family):
            m = [x["matched"] for x in draws[k] if x["matched"] is not None]
            out[name][k] = dict(rank_p=(1 + sum(x >= mine for x in m)) / (1 + len(m)), beats_all=all(mine > x for x in m))
    return out


# ---- budget-allocated timer ------------------------------------------------------------------------

def budget_timer(val_cohort, cal_cohort, te_cohort, dev, rk, fused_result, alpha=0.05):
    """review_analyses_b's per-task budget allocation, its beta chosen on the validation set and its margin
    certified on the calibration successes. Reports its Delta l over P (P held to its false alarms) and the
    fused monitor's Delta l over it (it held to the monitor's false alarms)."""
    import review_analyses_b as rab
    best = rab.adaptive_allocation(val_cohort, dev, alpha)
    if best is None:
        return None
    _, beta, rkA, info, _ = best
    tmA = rab.timer_fit(cal_cohort, rkA, alpha)
    eA, rA = hk.prepared(te_cohort, "P", ALL, None, rkA)
    eP, rP = hk.prepared(te_cohort, "P", ALL, None, rk)
    a = hk.raw_eval(eA, rA, dict(u=math.inf, s=0.0, lam=tmA["lam"]))
    mP = hk.matched_timer([(eP, rP)], a["fp"])
    mA = hk.matched_timer([(eA, rA)], fused_result["fp"])
    lead_f = fused_result["_arrays"]["fused"]
    return dict(beta=beta, lam=float(tmA["lam"]), fp=a["fp"], mean_lead=float(a["lead"].mean()),
                over_P_matched=None if mP is None else float(np.mean(a["lead"] - mP[0]["lead"])),
                fused_over_it_matched=None if mA is None else float(np.mean(lead_f - mA[0]["lead"])))


# ---- Fig. 1 -----------------------------------------------------------------------------------------

def teaser(row, score, rho_k, H, path, lam_fused, lam_P):
    """Alarm times on one rollout: the score's crossing of tau_s = u + s * lam, the fused monitor's own timer,
    and P. score: the running maximum trajectory."""
    eps = hk.EPS
    tau = None if path["u"] == math.inf else path["u"] + path["s"] * lam_fused
    hit = np.flatnonzero(np.asarray(score) >= tau) if tau is not None else np.zeros(0, int)
    return dict(rollout_id=str(row["rollout_id"]), task=row["task_name"], L=int(row["L"]), H=int(H),
                tau_s=tau, lam_fused=float(lam_fused), lam_P=float(lam_P),
                score_alarm=None if not len(hit) else int(hit[0] + 1),
                fused_timer_alarm=int(math.ceil(min(H, (rho_k + lam_fused) * H) - eps)),
                P_alarm=int(math.ceil(min(H, (rho_k + lam_P) * H) - eps)))


# ---- printing -------------------------------------------------------------------------------------

def print_summary(res, label):
    nan = float("nan")
    f = lambda x: "  --  " if x is None else f"{x:+.3f}"
    print(f"R22 {label}: fused monitors with paths fixed on the validation set")
    for side, by in res["table2"].items():
        for name, by_t in by.items():
            for t, v in by_t.items():
                m = v.get("mean_std_over_seeds")
                if not m:
                    continue
                mg = m.get("matched_gain", dict(mean=nan, std=nan))
                print(f"  {side:6s} {name:28s} {t:5s} FPR {m['fpr']['mean']:.3f}  Delta l {mg['mean']:+.3f}+-{mg['std']:.3f}"
                      f"  (same target {m['gain']['mean']:+.3f})")
    for name, row in (res.get("ensembles_seen") or {}).items():
        mm = row["matched"]
        print(f"  ensemble + P {name:28s} FA {row['fp']:3d} (P {row['timer']['fp']})  Delta l "
              f"{f(mm['gain'] if mm else None)} {mm['ci95'] if mm else ''}  same target {row['gain']:+.3f}  path {row['chosen']}")
    for t, row in (res.get("unseen_bootstrap") or {}).get("timers", {}).items():
        for n, v in row.items():
            print(f"  unseen {t:5s} {n:36s} Delta l {f(v.get('matched_gain'))} {[round(x, 3) for x in v['matched_ci95']]}")
    nl = res.get("nulls")
    if nl:
        for k, v in nl["nulls"].items():
            print(f"  null {k:28s} Delta l mean {v['matched']['mean']:+.3f} max {v['matched']['max']:+.3f}; "
                  f"same target max {v['same_target']['max']:+.3f}")
        for name in [k for k in nl if k not in ("draws", "seed", "nulls")]:
            print(f"  null rank {name}: " + ", ".join(f"{k} p {v['rank_p']:.3f} beats all {v['beats_all']}"
                                                     for k, v in nl[name].items() if isinstance(v, dict)))
    if res.get("budget_timer"):
        b = res["budget_timer"]
        print(f"  budget timer: beta {b['beta']}, FA {b['fp']}, over P {f(b['over_P_matched'])}, "
              f"Hide-and-Seek + P over it {f(b['fused_over_it_matched'])}")
    if res.get("teaser"):
        print(f"  Fig. 1: {res['teaser']}")
    for name, c in res["checks"].items():
        ok = c["ok"] if "ok" in c else all(x["ok"] for x in c.values())
        print(f"R22 {name.split('_', 1)[0]}: {'OK' if ok else 'FAILED'}")
