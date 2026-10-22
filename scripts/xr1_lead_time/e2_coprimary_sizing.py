"""E2 co-primary sizing: SAFE + P and Hide-and-Seek + P on a fresh RLDX-1 cohort.

Plan: docs/stage2_sizing.md, section "E2 co-primary sizing (2026-10-23)". Exploratory: every
input cohort is already open, so these numbers size E2 and test nothing.

E2 (RSS Stage 1 draft) makes SAFE + P and Hide-and-Seek + P co-primary against P on RLDX-1's
20 tasks: each is supported if its decision CI lies above zero, at 97.5% (Bonferroni over the
two), under the E2 FPR rule. This script extends the E2 sizing of stage2_sizing.py:

- the same held-out pool (3 calibration successes + 5/5 evaluation rollouts per task), the
  same draws (seed, cell and replicate numbering of stage2_sizing.simulate), so SAFE's
  simulated studies are exactly those of the documented E2 sizing;
- every simulated study is scored with both detectors on the same drawn rollouts (paired),
  with P fitted once per study;
- the decision CI is approximated as in stage2_sizing (g - hw, hw = median minus the lower
  percentile of the sampling distribution), at 95% and 97.5%;
- two FPR rules: Amendment 1's (FPR <= 0.075 and Wilson upper <= 0.10) and E2's (false alarms
  do not reject "true FPR <= 0.075" in a one-sided exact binomial test at level 0.05);
- power per detector, for at least one and for both, at assumed gains imposed by shift.

Usage:
  python3 -I e2_coprimary_sizing.py --calibration-dir <SAFE20>/calibration/lstm \\
      --hs-scores-dir <rldx1_hide_and_seek run>/scores \\
      [--replication-json rldx_lead_lstm.json] [--hs-result rldx_hide_and_seek.json] \\
      [--workers 8] --output OUT.json
"""
import argparse, hashlib, json, math, multiprocessing as mp, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_hide_and_seek as chs
import confirm_lead as cl
import quantile_timer as qt
import rldx_hide_and_seek as rhs
import rldx_replication as rr
import stage2_sizing as s2

PLAN = "docs/stage2_sizing.md (E2 co-primary sizing, 2026-10-23)"
KEYS = ("safe", "hide_and_seek")
LEVELS = (0.95, 0.975)
GAINS = {"safe": (0.034,), "hide_and_seek": (0.083, 0.052, 0.05)}
JOINT = ((0.034, 0.083), (0.034, 0.052))
FPR_BOUND, FPR_ALPHA = 0.075, 0.05


def binom_max_false_alarms(n, p=FPR_BOUND, alpha=FPR_ALPHA):
    """Largest k whose upper tail P(X >= k), X ~ Binomial(n, p), is still >= alpha: the most
    false alarms that do not reject 'true FPR <= p' in a one-sided exact test at level alpha."""
    pmf = [0.0] * (n + 1)
    pmf[0] = (1 - p) ** n
    for k in range(1, n + 1):
        pmf[k] = pmf[k - 1] * (n - k + 1) / k * p / (1 - p)
    tail, kmax = 1.0, 0
    for k in range(n + 1):
        if tail >= alpha - 1e-15:
            kmax = k
        tail -= pmf[k]
    return kmax


# ---- paired simulation (fork workers inherit the pools) -------------------------------------

_G = {}


def _job(args):
    cell, n, mode, r0, r1 = args
    pools, rk = _G["pools"], _G["rk"]
    PS = pools["safe"]
    out = []
    for r in range(r0, r1):
        # Same RNG stream as stage2_sizing._job: tasks (with_tasks only), then cal, then test.
        rng = np.random.default_rng([s2.SEED, cell, r])
        tasks = list(rng.choice(PS.tasks, len(PS.tasks))) if mode == "with_tasks" else None
        ci, ti = PS.draw(rng, n, 0.0, tasks), PS.draw(rng, n, 0.0, tasks)
        row, lP = {}, None
        for key in KEYS:
            c, e = pools[key].C.take(ci), pools[key].C.take(ti)
            tm, fu = cl.fit_fast(c, rk)
            eP, lp = cl.evaluate_fast(e, rk, tm)
            if lP is None:
                lP, P_eval = lp, eP
            eF, lF = cl.evaluate_fast(e, rk, fu)
            row[key] = dict(gain=float(np.mean(lF - lP)), fp=eF["fp"], n_s=int((~e.failed).sum()), fpr=eF["fpr"],
                            wilson_up=eF["fpr_wilson95"][1], uses_score=bool(math.isfinite(fu["u"])))
        row["P_fpr"] = P_eval["fpr"]
        out.append(row)
    return cell, out


def simulate_pair(pools, rk, sizes, R, workers, chunk=25):
    """Paired sampling distributions for both detectors; cells numbered as stage2_sizing."""
    _G.update(pools=pools, rk=rk)
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


# ---- power ---------------------------------------------------------------------------------

def half_width(g, level):
    return float(np.median(g) - np.percentile(g, 100 * (1 - level) / 2))


def masks(rows, key):
    a1 = np.array([r[key]["fpr"] <= cl.FPR_MAX + 1e-12 and r[key]["wilson_up"] <= cl.FPR_WILSON_MAX + 1e-12 for r in rows])
    e2 = np.array([r[key]["fp"] <= binom_max_false_alarms(r[key]["n_s"]) for r in rows])
    return dict(a1=a1, e2=e2)


def ci_pass(rows, key, gamma, level, hw_rows):
    """Per-study 'decision CI above zero' at an assumed gain (shift), CI half-width taken from
    hw_rows (within-task rows for the primary CI, with-task rows for the task-level CI)."""
    g = np.array([r[key]["gain"] for r in rows])
    if gamma is not None:
        g = g + gamma - g.mean()
    hw = half_width(np.array([r[key]["gain"] for r in hw_rows]), level)
    return np.round(g, 4) - hw > 0, hw


def summary(rows, key):
    g = np.array([r[key]["gain"] for r in rows])
    m = masks(rows, key)
    return dict(replicates=len(rows), mean_gain=round(float(g.mean()), 4), sd=round(float(g.std()), 4),
                mean_fused_fpr=round(float(np.mean([r[key]["fpr"] for r in rows])), 4),
                mean_P_fpr=round(float(np.mean([r["P_fpr"] for r in rows])), 4),
                fpr_pass_a1=round(float(m["a1"].mean()), 3), fpr_pass_e2=round(float(m["e2"].mean()), 3),
                uses_score=round(float(np.mean([r[key]["uses_score"] for r in rows])), 3))


def power_tables(sims, sizes):
    f = lambda m: round(float(np.mean(m)), 3)
    out = dict(summary={}, power={}, joint={})
    for n in sizes:
        w, t = sims[(n, "within_task")], sims[(n, "with_tasks")]
        out["summary"][str(n)] = {k: dict(within_task=summary(w, k), with_tasks=summary(t, k)) for k in KEYS}
        for key in KEYS:
            m = masks(w, key)
            for gamma in (None,) + GAINS[key]:
                gname = "plug_in" if gamma is None else str(gamma)
                for level in LEVELS:
                    ci_w, hw_w = ci_pass(w, key, gamma, level, w)
                    ci_t, hw_t = ci_pass(w, key, gamma, level, t)
                    out["power"].setdefault(key, {}).setdefault(gname, {}).setdefault(str(level), {})[str(n)] = dict(
                        ci=f(ci_w), ci_fpr_a1=f(ci_w & m["a1"]), ci_fpr_e2=f(ci_w & m["e2"]),
                        task_ci=f(ci_t), task_ci_fpr_e2=f(ci_t & m["e2"]), hw_within=round(hw_w, 4), hw_tasks=round(hw_t, 4))
        ms, mh = masks(w, "safe"), masks(w, "hide_and_seek")
        for gs, gh in JOINT:
            for level in LEVELS:
                ps = ci_pass(w, "safe", gs, level, w)[0] & ms["e2"]
                ph = ci_pass(w, "hide_and_seek", gh, level, w)[0] & mh["e2"]
                out["joint"].setdefault(f"safe {gs} / hs {gh}", {}).setdefault(str(level), {})[str(n)] = dict(
                    safe=f(ps), hide_and_seek=f(ph), at_least_one=f(ps | ph), both=f(ps & ph))
    return out


# ---- data and checks -----------------------------------------------------------------------

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_pools(cal_s, test, tasks):
    """Held-out pool for each detector; identical row order so draw indices align."""
    held = cal_s + test
    pools = {"safe": s2.Pool(held, tasks),
             "hide_and_seek": s2.Pool([dict(r, safe=np.asarray(r["hide_and_seek"], float)) for r in held], tasks)}
    a, b = pools["safe"].idx, pools["hide_and_seek"].idx
    if any(not all(np.array_equal(x, y) for x, y in zip(a[t], b[t])) for t in tasks):
        raise ValueError("SAFE and Hide-and-Seek pools are not aligned")
    return pools


def replication_gains(cal_s, train, test, rk):
    """The 15.15 / 15.22 fit: held-out successes set the margin, training failures the path."""
    cal = cal_s + [r for r in train if r["failed"]]
    out = {}
    tm, _ = cl.fit_fast(cl.Cohort(cal, "safe"), rk)
    _, lP = cl.evaluate_fast(cl.Cohort(test, "safe"), rk, tm)
    for key in KEYS:
        _, fu = cl.fit_fast(cl.Cohort(cal, key), rk)
        _, lF = cl.evaluate_fast(cl.Cohort(test, key), rk, fu)
        out[key] = round(float(np.mean(lF - lP)), 4)
    return out


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

    checks = dict(replication=replication_gains(cal_s, train, test, rk))
    stop = []
    if args.replication_json:
        want = json.load(open(args.replication_json))["primary"]["gain"]
        checks["safe_matches_15_15"] = abs(checks["replication"]["safe"] - want) < 1e-9
        checks["safe_15_15_gain"] = want
        if not checks["safe_matches_15_15"]:
            stop.append(f"SAFE gain {checks['replication']['safe']} != 15.15 {want}")
    if args.hs_result:
        want = json.load(open(args.hs_result))["detectors"]["hide_and_seek"]["gain"]
        checks["hs_matches_15_22"] = abs(checks["replication"]["hide_and_seek"] - want) < 1e-9
        checks["hs_15_22_gain"] = want
        if not checks["hs_matches_15_22"]:
            stop.append(f"Hide-and-Seek gain {checks['replication']['hide_and_seek']} != 15.22 {want}")
    print(f"check: replication gains SAFE {checks['replication']['safe']:+.4f}, Hide-and-Seek "
          f"{checks['replication']['hide_and_seek']:+.4f}" + ("; " + "; ".join(stop) if stop else ""), flush=True)
    if stop:
        sys.exit("STOP: " + "; ".join(stop))

    pools = build_pools(cal_s, test, tasks)
    sims = simulate_pair(pools, rk, args.sizes, args.replicates, args.workers)
    tables = power_tables(sims, args.sizes)
    inputs = {f"hs_{n}": dict(path=str((sd / f"{n}_predictions.jsonl").resolve()), sha256=sha256(sd / f"{n}_predictions.jsonl"))
              for n in ("train", "calibration", "test")}
    for name, path in (("replication_json", args.replication_json), ("hs_result", args.hs_result)):
        if path:
            inputs[name] = dict(path=str(Path(path).resolve()), sha256=sha256(path))
    out = dict(plan=PLAN, status="exploratory sizing (opened cohorts)", seed=s2.SEED, replicates=args.replicates,
               sizes=list(args.sizes), levels=list(LEVELS), gains={k: list(v) for k, v in GAINS.items()},
               e2_fpr_rule=dict(bound=FPR_BOUND, alpha=FPR_ALPHA,
                                max_false_alarms={str(20 * n): binom_max_false_alarms(20 * n) for n in args.sizes}),
               calibration_dir=str(Path(args.calibration_dir).resolve()), inputs=inputs,
               pool_counts=pools["safe"].counts(), checks=checks, **tables)
    s2.write(out_path, out)
    print_tables(out)
    return 0


def print_tables(out):
    print("\nPlug-in sampling distribution (within task): mean gain (SD); FPR pass A1 / E2")
    for n, v in out["summary"].items():
        print(f"n={n:>3}  " + "  ".join(f"{k}: {s['within_task']['mean_gain']:+.4f} ({s['within_task']['sd']:.4f}); "
                                        f"{s['within_task']['fpr_pass_a1']:.2f} / {s['within_task']['fpr_pass_e2']:.2f}"
                                        for k, s in v.items()))
    for key, by_g in out["power"].items():
        for g, by_l in by_g.items():
            for level, by_n in by_l.items():
                print(f"{key:14s} gain {g:>8s} CI {float(level):.3f}: " + "  ".join(
                    f"n={n}: CI {p['ci']:.2f}, +E2 {p['ci_fpr_e2']:.2f}, +A1 {p['ci_fpr_a1']:.2f}, taskCI+E2 {p['task_ci_fpr_e2']:.2f}"
                    for n, p in by_n.items()))
    for pair, by_l in out["joint"].items():
        for level, by_n in by_l.items():
            print(f"joint {pair} CI {float(level):.3f} (E2 rule): " + "  ".join(
                f"n={n}: one {p['at_least_one']:.2f}, both {p['both']:.2f}" for n, p in by_n.items()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration-dir", required=True)
    p.add_argument("--hs-scores-dir", required=True)
    p.add_argument("--replication-json")
    p.add_argument("--hs-result")
    p.add_argument("--sizes", type=s2.sizes_arg, default=s2.SIZES)
    p.add_argument("--replicates", type=int, default=s2.REPLICATES)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", required=True)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
