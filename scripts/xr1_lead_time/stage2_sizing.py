"""Offline sizing for the RSS 2027 Stage 2 protocol (plan: docs/stage2_sizing.md).

Exploratory. Every input cohort is already open for lead time, so these numbers size
the Stage 2 experiments; they test nothing. No rollout is collected and no registered
analysis is changed: P and SAFE + P are fitted with confirm_lead.fit_fast and evaluated
with confirm_lead.evaluate_fast, exactly as in E1.

Subcommands:
  e1b   Power of an atomic-only XR-1 test (SAFE + P vs P), by rollouts per class per task,
        conditional on the 10 atomic tasks, from the E1 calibration and test cohorts.
  e2    The same power curve for a fresh RLDX-1 cohort with held-out calibration failures,
        from the RLDX-1 SAFE20 held-out scores.
  e3    Bounds on the closed-loop retry benefit of no monitor, P and SAFE + P, from the
        observed alarm times of both policies.
e1b and e2 also report the collection cost (attempts per task) from observed success rates.

Usage:
  python3 -I stage2_sizing.py e1b --calibration CAL.jsonl --test TEST.jsonl --runtime-bundle B.json \
      [--dose-response dose_response.json] [--pool all|primary] [--workers 8] --output OUT.json
  python3 -I stage2_sizing.py e2 --calibration-dir <rldx root>/calibration/lstm \
      [--replication-json rldx_lead_lstm.json] [--workers 8] --output OUT.json
  python3 -I stage2_sizing.py e3 --calibration CAL.jsonl --test TEST.jsonl --runtime-bundle B.json \
      --rldx-calibration-dir <rldx root>/calibration/lstm --output OUT.json
"""
import argparse, hashlib, json, math, multiprocessing as mp, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl
import lead_time
import quantile_timer as qt
import rldx_replication as rr

Q = cl.PRIMARY_Q
MARGIN, FPR_MAX, FPR_WILSON_MAX = cl.MARGIN, cl.FPR_MAX, cl.FPR_WILSON_MAX
SIZES = (10, 20, 30, 40, 50, 60)
E1B_GAINS = (0.05, 0.0695, 0.089)
E2_GAINS = (0.034, 0.05)
REPLICATES, SEED = 2000, 20261021
NESTED_SIZES, NESTED_OUTER, NESTED_INNER = (10, 30, 60), 200, 300
BUDGETS = (1.0, 1.25, 1.5, 2.0)
RETRY_EPISODES = 20000
COST_DRAWS = 20000
ATTEMPT_CAP = 500        # attempts per task allowed by the E1 collection protocol


# ---- pools ---------------------------------------------------------------------------------

def null_failure(r):
    """A failure whose SAFE score behaves like a success: the success's trajectory, held at
    its running maximum up to H (every failure in these cohorts runs to H)."""
    H = int(r["horizon"]); s = np.maximum.accumulate(np.asarray(r["safe"], float))
    traj = np.concatenate([s, np.full(max(0, H - len(s)), s[-1])])[:H]
    return dict(task_name=r["task_name"], L=H, horizon=H, failed=True, safe=traj, null=True)


class Pool:
    """Resampling pool of one cohort: per task, success / failure / null-failure indices."""

    def __init__(self, rows, tasks):
        rows = [r for r in rows if r["task_name"] in tasks]
        nulls = [null_failure(r) for r in rows if not r["failed"]]
        allrows = rows + nulls
        self.C = cl.Cohort(allrows)
        self.tasks = sorted(tasks)
        self.idx = {t: ([], [], []) for t in self.tasks}
        for i, r in enumerate(allrows):
            self.idx[r["task_name"]][2 if r.get("null") else int(r["failed"])].append(i)
        self.idx = {t: tuple(np.array(v, int) for v in vs) for t, vs in self.idx.items()}
        for t, (s, f, _) in self.idx.items():
            if not len(s) or not len(f):
                raise ValueError(f"Task {t} lacks successes or failures in the pool")

    def counts(self):
        return {t: dict(successes=len(s), failures=len(f)) for t, (s, f, _) in self.idx.items()}

    def draw(self, rng, n, pi=0.0, tasks=None):
        """n successes and n failures per task, with replacement; each failure is a null
        failure with probability pi. tasks: a (possibly repeated) task list."""
        out = []
        for t in (self.tasks if tasks is None else tasks):
            s, f, z = self.idx[t]
            out.append(rng.choice(s, n))
            fi = rng.choice(f, n)
            if pi > 0:
                m = rng.random(n) < pi
                fi[m] = rng.choice(z, int(m.sum()))
            out.append(fi)
        return np.concatenate(out)


def resample_within(C, idx, rng):
    """Bootstrap of a drawn sample: rollouts resampled within task and class."""
    g = defaultdict(list)
    for i in idx:
        g[(C.task[i], bool(C.failed[i]))].append(i)
    return np.concatenate([rng.choice(v, len(v)) for _, v in sorted(g.items())])


def study(cal, te, rk, ci, ti):
    """One simulated study: fit on the calibration sample, evaluate on the test sample."""
    c, e = cal.C.take(ci), te.C.take(ti)
    tm, fu = cl.fit_fast(c, rk)
    eP, lP = cl.evaluate_fast(e, rk, tm); eF, lF = cl.evaluate_fast(e, rk, fu)
    return dict(gain=float(np.mean(lF - lP)), fpr=eF["fpr"], wilson_up=eF["fpr_wilson95"][1], P_fpr=eP["fpr"],
                P_lead=eP["mean_lead"], uses_safe=bool(math.isfinite(fu["u"])))


# ---- worker plumbing (fork: pools are inherited, not pickled) -------------------------------

_G = {}


def _job(args):
    kind, cell, n, pi, mode, r0, r1, inner = args
    cal, te, rk = _G["cal"], _G["te"], _G["rk"]
    out = []
    for r in range(r0, r1):
        rng = np.random.default_rng([SEED, cell, r])
        tasks = None
        if mode == "with_tasks":
            tasks = list(rng.choice(cal.tasks, len(cal.tasks)))
        ci, ti = cal.draw(rng, n, pi, tasks), te.draw(rng, n, pi, tasks)
        res = study(cal, te, rk, ci, ti)
        if kind == "nested":
            g = []
            for _ in range(inner):
                g.append(study(cal, te, rk, resample_within(cal.C, ci, rng), resample_within(te.C, ti, rng))["gain"])
            res.update(inner_lower=float(np.percentile(g, 2.5)), inner_sd=float(np.std(g)), inner_median=float(np.median(g)))
        out.append(res)
    return cell, out


def run_jobs(jobs, workers):
    if workers <= 1:
        res = [_job(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as p:
            res = p.map(_job, jobs, chunksize=1)
    out = defaultdict(list)
    for cell, rows in res:
        out[cell] += rows
    return out


def chunks(kind, cell, n, pi, mode, R, inner=0, size=25):
    return [(kind, cell, n, pi, mode, a, min(R, a + size), inner) for a in range(0, R, size)]


# ---- power -------------------------------------------------------------------------------

def summarise(rows):
    g = np.array([r["gain"] for r in rows])
    fpr_ok = np.array([r["fpr"] <= FPR_MAX + 1e-12 and r["wilson_up"] <= FPR_WILSON_MAX + 1e-12 for r in rows])
    return dict(replicates=len(rows), mean_gain=round(float(g.mean()), 4), sd=round(float(g.std()), 4),
                median=round(float(np.median(g)), 4), q025=round(float(np.percentile(g, 2.5)), 4),
                half_width=round(float(np.median(g) - np.percentile(g, 2.5)), 4),
                mean_fused_fpr=round(float(np.mean([r["fpr"] for r in rows])), 4),
                mean_P_fpr=round(float(np.mean([r["P_fpr"] for r in rows])), 4),
                fpr_criteria_pass=round(float(fpr_ok.mean()), 3),
                uses_safe=round(float(np.mean([r["uses_safe"] for r in rows])), 3))


def power_rows(g, fpr_ok, hw_within, hw_tasks):
    """Pass rates of the registered criteria for study estimates g (already at the assumed gain).

    The decision CI's lower bound in a study is approximated by g - hw, where hw is the
    median minus the 2.5th percentile of the simulated sampling distribution (rollouts
    within task: hw_within; tasks and rollouts: hw_tasks)."""
    g = np.round(g, 4)
    point = g >= MARGIN
    ci_w, ci_t = g - hw_within > 0, g - hw_tasks > 0
    f = lambda m: round(float(np.mean(m)), 3)
    return dict(ci_only=f(ci_w), ci_and_point=f(ci_w & point), ci_point_fpr=f(ci_w & point & fpr_ok),
                point_only=f(point), ci_only_task_ci=f(ci_t), ci_and_point_task_ci=f(ci_t & point))


def fpr_mask(rows):
    return np.array([r["fpr"] <= FPR_MAX + 1e-12 and r["wilson_up"] <= FPR_WILSON_MAX + 1e-12 for r in rows])


def simulate(cal, te, rk, gains, sizes, R, workers, nested_sizes, nested_outer, nested_inner):
    """Plug-in, shifted and diluted power curves, plus the nested check of the CI approximation."""
    _G.update(cal=cal, te=te, rk=rk)
    cells, jobs = {}, []
    for i, n in enumerate(sizes):
        for mode in ("within_task", "with_tasks"):
            cid = 1000 * i + (0 if mode == "within_task" else 1)
            cells[cid] = (n, mode, 0.0); jobs += chunks("plain", cid, n, 0.0, mode, R)
    sims = run_jobs(jobs, workers)
    base = {n: {m: sims[c] for c, (n2, m, _) in cells.items() if n2 == n} for n in sizes}
    out = dict(plug_in={}, shift={}, dilute={}, nested={})
    for n in sizes:
        w, t = base[n]["within_task"], base[n]["with_tasks"]
        sw, st = summarise(w), summarise(t)
        g, ok = np.array([r["gain"] for r in w]), fpr_mask(w)
        out["plug_in"][str(n)] = dict(within_task=sw, with_tasks=st,
                                      power=power_rows(g, ok, sw["half_width"], st["half_width"]))
        for gam in gains:
            out["shift"].setdefault(str(gam), {})[str(n)] = power_rows(g + gam - g.mean(), ok, sw["half_width"], st["half_width"])
    # Dilution: a fraction pi of failures carries no SAFE signal, pi = 1 - gamma / plug-in gain.
    jobs, dcells = [], {}
    for i, n in enumerate(sizes):
        g0 = out["plug_in"][str(n)]["within_task"]["mean_gain"]
        for j, gam in enumerate(gains):
            if g0 > 0 and gam < g0:
                cid = 1000 * i + 10 + j; pi = 1 - gam / g0
                dcells[cid] = (n, gam, pi); jobs += chunks("plain", cid, n, pi, "within_task", R)
    sims = run_jobs(jobs, workers)
    for cid, (n, gam, pi) in dcells.items():
        s = summarise(sims[cid]); g = np.array([r["gain"] for r in sims[cid]])
        hw_t = out["plug_in"][str(n)]["with_tasks"]["half_width"]
        out["dilute"].setdefault(str(gam), {})[str(n)] = dict(pi=round(pi, 4), summary=s,
                                                              power=power_rows(g, fpr_mask(sims[cid]), s["half_width"], hw_t))
    # Nested check: real inner bootstraps (rollouts within task, refit) on plug-in studies.
    jobs, ncells = [], {}
    for n in nested_sizes:
        if n not in sizes:
            continue
        cid = 900000 + n; ncells[cid] = n
        jobs += chunks("nested", cid, n, 0.0, "within_task", nested_outer, nested_inner, size=5)
    sims = run_jobs(jobs, workers)
    for cid, n in ncells.items():
        rows = sims[cid]; g = np.round([r["gain"] for r in rows], 4); lo = np.array([r["inner_lower"] for r in rows])
        hw = out["plug_in"][str(n)]["within_task"]["half_width"]
        out["nested"][str(n)] = dict(
            outer=len(rows), inner=nested_inner,
            nested_ci_only=round(float(np.mean(lo > 0)), 3),
            approx_ci_only=round(float(np.mean(g - hw > 0)), 3),
            nested_ci_and_point=round(float(np.mean((lo > 0) & (g >= MARGIN))), 3),
            approx_ci_and_point=round(float(np.mean((g - hw > 0) & (g >= MARGIN))), 3),
            mean_inner_sd=round(float(np.mean([r["inner_sd"] for r in rows])), 4),
            outer_sd=out["plug_in"][str(n)]["within_task"]["sd"],
            mean_inner_half_width=round(float(np.mean([r["inner_median"] - r["inner_lower"] for r in rows])), 4),
            outer_half_width=hw)
    return out


# ---- collection cost ----------------------------------------------------------------------

def success_rates(rows):
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        c[r["task_name"]][int(bool(r["failed"]))] += 1
    return {t: dict(successes=s, failures=f, p=round(s / (s + f), 4)) for t, (s, f) in sorted(c.items())}


def attempts(p, n, rng, draws=COST_DRAWS):
    """Attempts until at least n successes and n failures, everything retained."""
    if p <= 0 or p >= 1:
        return np.full(draws, np.inf)
    ts = n + rng.negative_binomial(n, p, draws)
    tf = n + rng.negative_binomial(n, 1 - p, draws)
    return np.maximum(ts, tf).astype(float)


def cost_table(rates, sizes, cohorts, mean_success_L, H, seed=SEED):
    rng = np.random.default_rng(seed)
    per_task, totals = {}, {}
    for n in sizes:
        tot = np.zeros(COST_DRAWS); inf_tot = np.zeros(COST_DRAWS); over = []
        for t, v in rates.items():
            a = attempts(v["p"], n, rng)
            per_task.setdefault(t, dict(p=v["p"], observed=f"{v['successes']}/{v['failures']}"))[str(n)] = \
                dict(mean=round(float(a.mean()), 1), p95=float(np.percentile(a, 95)))
            tot += cohorts * a
            inf_tot += cohorts * a * (v["p"] * mean_success_L[t] + (1 - v["p"]) * H[t])
            if np.mean(a > ATTEMPT_CAP) > 0.05:
                over.append(t)
        totals[str(n)] = dict(cohorts=cohorts, rollouts_mean=round(float(tot.mean())), rollouts_p95=round(float(np.percentile(tot, 95))),
                              inferences_mean=round(float(inf_tot.mean())), tasks_over_attempt_cap=over)
    return dict(per_task=per_task, totals=totals, attempt_cap=ATTEMPT_CAP)


# ---- shared output ------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_out(path):
    p = Path(path)
    if p.exists():
        raise SystemExit(f"error: output exists: {p}")
    return p


def write(p, out):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float) + "\n")


def print_power(out, gains, label):
    print(f"\n## {label}: plug-in population (no shift)")
    print("| n/class/task | mean gain | SD within | SD with tasks | fused FPR | FPR criteria pass | uses SAFE |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n, v in out["plug_in"].items():
        w, t = v["within_task"], v["with_tasks"]
        print(f"| {n} | {w['mean_gain']:+.4f} | {w['sd']:.4f} | {t['sd']:.4f} | {w['mean_fused_fpr']:.3f} | "
              f"{w['fpr_criteria_pass']:.2f} | {w['uses_safe']:.2f} |")
    for gam in gains:
        print(f"\n## {label}: assumed gain {gam} (shift; dilution check in the last columns)")
        print("| n | CI>0 | CI>0 & point>=0.05 | + FPR rule | CI>0 & point (task CI) | dilute: pi, mean gain, CI>0 & point |")
        print("|---:|---:|---:|---:|---:|---|")
        for n, p in out["shift"][str(gam)].items():
            d = out["dilute"].get(str(gam), {}).get(n)
            dtxt = f"{d['pi']:.2f}, {d['summary']['mean_gain']:+.4f}, {d['power']['ci_and_point']:.2f}" if d else "n/a"
            print(f"| {n} | {p['ci_only']:.2f} | {p['ci_and_point']:.2f} | {p['ci_point_fpr']:.2f} | "
                  f"{p['ci_and_point_task_ci']:.2f} | {dtxt} |")
    if out["nested"]:
        print(f"\n## {label}: nested check of the CI approximation (plug-in)")
        print("| n | CI>0 nested / approx | CI>0 & point nested / approx | inner SD / outer SD | inner / outer half-width |")
        print("|---:|---|---|---|---|")
        for n, v in out["nested"].items():
            print(f"| {n} | {v['nested_ci_only']:.2f} / {v['approx_ci_only']:.2f} | {v['nested_ci_and_point']:.2f} / "
                  f"{v['approx_ci_and_point']:.2f} | {v['mean_inner_sd']:.4f} / {v['outer_sd']:.4f} | "
                  f"{v['mean_inner_half_width']:.4f} / {v['outer_half_width']:.4f} |")


def print_cost(cost, label):
    print(f"\n## {label}: collection cost ({cost['totals'][next(iter(cost['totals']))]['cohorts']} cohorts, everything retained)")
    print("| n/class/task | rollouts (mean) | rollouts (p95) | inferences (mean) | tasks over the 500-attempt cap |")
    print("|---:|---:|---:|---:|---|")
    for n, v in cost["totals"].items():
        print(f"| {n} | {v['rollouts_mean']} | {v['rollouts_p95']} | {v['inferences_mean']} | {', '.join(v['tasks_over_attempt_cap']) or '-'} |")


def sizes_arg(s):
    return tuple(int(x) for x in s.split(",") if x)


def gains_arg(s):
    return tuple(float(x) for x in s.split(",") if x)


# ---- E1b -----------------------------------------------------------------------------------

def e1b(args):
    out_path = check_out(args.output)
    bundle = json.load(open(args.runtime_bundle))
    rk = qt.rho(qt.dev_fractions(bundle), Q)
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    tasks = sorted({r["task_name"] for r in cal if r.get("task_type") == args.stratum})
    if not tasks or tasks != sorted({r["task_name"] for r in te if r.get("task_type") == args.stratum}):
        raise ValueError(f"Calibration and test disagree on the {args.stratum} tasks")
    sub = lambda rows: [r for r in rows if r["task_name"] in tasks]
    # Check: stratum-calibrated observed gain on the primary rows (dose_response D4).
    calC, teC = cl.Cohort(sub(cal)), cl.Cohort(sub(te))
    tm, fu = cl.fit_fast(calC, rk)
    eP, lP = cl.evaluate_fast(teC, rk, tm); eF, lF = cl.evaluate_fast(teC, rk, fu)
    check = dict(observed_gain=round(float(np.mean(lF - lP)), 4), P=eP, fused=dict(**cl.det_json(fu), **eF))
    if args.dose_response:
        ref = json.load(open(args.dose_response))["detectors"]["safe"]["by_type"][args.stratum]["stratum_calibration"]["gain"]
        check.update(dose_response_gain=ref, matches_dose_response=abs(ref - check["observed_gain"]) < 1e-9)
    print(f"check: observed {args.stratum} gain (stratum calibration, primary rows) {check['observed_gain']:+.4f}"
          + (f"; dose_response {check['dose_response_gain']:+.4f} -> {'MATCH' if check['matches_dose_response'] else 'MISMATCH'}"
             if args.dose_response else ""), flush=True)
    if args.dose_response and not check["matches_dose_response"]:
        sys.exit("STOP: the observed atomic gain does not reproduce dose_response.json")
    pc, pt = (cal_all, te_all) if args.pool == "all" else (cal, te)
    calP, teP = Pool(sub(pc), tasks), Pool(sub(pt), tasks)
    sim = simulate(calP, teP, rk, args.gains, args.sizes, args.replicates, args.workers,
                   args.nested_sizes, args.nested_outer, args.nested_inner)
    rates = success_rates(sub(cal_all) + sub(te_all))
    succL = defaultdict(list)
    for r in sub(cal_all) + sub(te_all):
        if not r["failed"]:
            succL[r["task_name"]].append(r["L"])
    H = {r["task_name"]: r["horizon"] for r in sub(cal_all)}
    cost = cost_table(rates, args.sizes, 2, {t: float(np.mean(v)) for t, v in succL.items()}, H)
    out = dict(plan="docs/stage2_sizing.md", status="exploratory sizing (opened cohorts)", experiment="E1b",
               stratum=args.stratum, tasks=tasks, pool=args.pool, q=Q, seed=SEED,
               inputs={k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in
                       (("calibration", args.calibration), ("test", args.test), ("runtime_bundle", args.runtime_bundle))},
               pool_counts=dict(calibration=calP.counts(), test=teP.counts()), check=check,
               gains=list(args.gains), sizes=list(args.sizes), simulation=sim, success_rates=rates, cost=cost)
    write(out_path, out)
    print(f"tasks: {len(tasks)} {args.stratum}; pool {args.pool}: calibration "
          f"{sum(v['successes'] for v in calP.counts().values())}S/{sum(v['failures'] for v in calP.counts().values())}F, "
          f"test {sum(v['successes'] for v in teP.counts().values())}S/{sum(v['failures'] for v in teP.counts().values())}F")
    print_power(sim, args.gains, f"E1b ({args.pool} pool)")
    print_cost(cost, "E1b")
    print("\nper-task success rate (all retained rows, both cohorts):",
          ", ".join(f"{t}: {v['p']:.2f}" for t, v in rates.items()))


# ---- E2 ------------------------------------------------------------------------------------

def rldx_rows(cal_dir, model):
    rows = rr.assemble(cal_dir, model)
    H = rr.horizons(rows)
    tasks = sorted({r["task_name"] for r in rows})
    if [t for t in tasks if t not in H]:
        raise ValueError("RLDX tasks without training failures")
    rr.finish(rows, H)
    train = [r for r in rows if r["split"] == "train"]
    return rows, train, tasks, qt.rho(rr.dev_fractions(train, tasks), Q)


def e2(args):
    out_path = check_out(args.output)
    rows, train, tasks, rk = rldx_rows(args.calibration_dir, args.model)
    cal_s = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "evaluation"]
    # Check: the replication's fit (held-out successes + training failures).
    calC, teC = cl.Cohort(cal_s + [r for r in train if r["failed"]]), cl.Cohort(test)
    tm, fu = cl.fit_fast(calC, rk)
    eP, lP = cl.evaluate_fast(teC, rk, tm); eF, lF = cl.evaluate_fast(teC, rk, fu)
    check = dict(observed_gain=round(float(np.mean(lF - lP)), 4), P=eP, fused=dict(**cl.det_json(fu), **eF))
    if args.replication_json:
        ref = json.load(open(args.replication_json))["primary"]["gain"]
        check.update(replication_gain=ref, matches_replication=abs(ref - check["observed_gain"]) < 1e-9)
    print(f"check: RLDX-1 replication gain {check['observed_gain']:+.4f}"
          + (f"; rldx_lead json {check['replication_gain']:+.4f} -> {'MATCH' if check['matches_replication'] else 'MISMATCH'}"
             if args.replication_json else ""), flush=True)
    if args.replication_json and not check["matches_replication"]:
        sys.exit("STOP: the RLDX-1 gain does not reproduce the replication output")
    # A fresh cohort's scores are held-out: pool = calibration + evaluation rollouts, shared by
    # the simulated calibration and test samples.
    held = cal_s + test
    P = Pool(held, tasks)
    sim = simulate(P, P, rk, args.gains, args.sizes, args.replicates, args.workers,
                   args.nested_sizes, args.nested_outer, args.nested_inner)
    natural = [r for r in rows if r["split"] in ("train", "calibration", "evaluation")]
    rates = success_rates(natural)
    succL = defaultdict(list)
    for r in train:
        if not r["failed"]:
            succL[r["task_name"]].append(r["L"])
    H = {r["task_name"]: r["horizon"] for r in rows}
    cost = cost_table(rates, args.sizes, 2, {t: float(np.mean(succL.get(t, [H[t]]))) for t in tasks}, H)
    out = dict(plan="docs/stage2_sizing.md", status="exploratory sizing (opened cohorts)", experiment="E2",
               model=args.model, tasks=tasks, q=Q, seed=SEED, calibration_dir=str(Path(args.calibration_dir).resolve()),
               pool_counts=P.counts(), check=check, gains=list(args.gains), sizes=list(args.sizes),
               simulation=sim, success_rates=rates, cost=cost,
               by_type={tt: sorted({r["task_name"] for r in rows if r["task_type"] == tt}) for tt in sorted({r["task_type"] for r in rows})})
    write(out_path, out)
    c = P.counts()
    print(f"tasks: {len(tasks)}; held-out pool {sum(v['successes'] for v in c.values())}S/{sum(v['failures'] for v in c.values())}F "
          f"(shared by simulated calibration and test)")
    print_power(sim, args.gains, f"E2 (RLDX-1 {args.model})")
    print_cost(cost, "E2")
    print("\nnatural success rate per task:", ", ".join(f"{t}: {v['p']:.2f}" for t, v in rates.items()))


# ---- E3 ------------------------------------------------------------------------------------

def alarms(C, rk, det):
    return cl.alarm_vec(C, cl.rho_rows(C, rk), det["lam"], det["u"], det["s"])


def task_inputs(teC, rk, tm, fu, rates, D, types):
    """Per task: test failures' and successes' alarm indices for none / P / SAFE + P."""
    a = dict(none=np.where(teC.failed, teC.L, np.inf), P=alarms(teC, rk, tm), safe_plus_P=alarms(teC, rk, fu))
    out = {}
    for t in sorted(set(teC.task)):
        m = teC.task == t; F, S = m & teC.failed, m & ~teC.failed
        out[t] = dict(type=types[t], p=rates[t]["p"], H=float(teC.H[m][0]), D=float(D[t]),
                      fail_alarm={k: np.minimum(v[F], teC.L[F]) for k, v in a.items()},
                      fail_hit={k: v[F] <= teC.L[F] for k, v in a.items()},
                      succ_L=teC.L[S], succ_alarm={k: v[S] for k, v in a.items()})
    return out


def retry_mc(ti, det, B, r_retry, seed, M=RETRY_EPISODES):
    """Episodes under a budget of B*H inferences: an attempt succeeds with p (first) or r_retry
    (later); a failing attempt ends at its alarm (or at H unalarmed); a success ends at its
    duration unless a false alarm aborts it first. Attempts are drawn from the task's test rollouts.

    Common random numbers: attempt a of episode e uses the same draws for every detector, so
    detector differences carry little Monte Carlo noise. Returns (success rate, first-attempt
    success rate of the same draws)."""
    budget = B * ti["H"]; t = np.zeros(M); done = np.zeros(M, bool); won = np.zeros(M, bool)
    fa = ti["fail_alarm"][det]
    sL, sa = ti["succ_L"], ti["succ_alarm"][det]
    a, p_hat = 0, None
    while not done.all():
        rng = np.random.default_rng([*seed, a])
        u, si, fi = rng.random(M), rng.integers(0, len(sL), M), rng.integers(0, len(fa), M)
        ok_all = u < (ti["p"] if a == 0 else r_retry)
        if a == 0:
            p_hat = float(ok_all.mean())
        idx = np.flatnonzero(~done)
        ok, d_s, a_s = ok_all[idx], sL[si[idx]], sa[si[idx]]
        aborted = ok & (a_s <= d_s)
        win = ok & ~aborted & (t[idx] + d_s <= budget)
        t[idx] += np.where(ok & ~aborted, d_s, np.where(aborted, a_s, fa[fi[idx]]))
        won[idx[win]] = True
        # Stop on a win or an exhausted budget. An unalarmed failure ends at the time limit,
        # which is itself observable, so it is retried like a failure under no monitor.
        done[idx[win | (t[idx] >= budget)]] = True
        a += 1
    return float(won.mean()), p_hat


def e3_policy(ti, budgets, r_mode, seed):
    dets = ("none", "P", "safe_plus_P")
    per_task = {}
    for k_t, (t, v) in enumerate(ti.items()):
        r_retry = v["p"] if r_mode == "same" else float(r_mode)
        row = dict(type=v["type"], p=v["p"], D_over_H=round(v["D"] / v["H"], 4), single={}, multi={},
                   false_alarm_exposure={k: round(v["p"] * float(np.mean(v["succ_alarm"][k] <= v["succ_L"])), 4) for k in dets})
        for B in budgets:
            frac = {k: float(np.mean(B * v["H"] - v["fail_alarm"][k] >= v["D"])) for k in dets}
            row["single"][str(B)] = {k: round((1 - v["p"]) * r_retry * frac[k], 4) for k in dets} | \
                                    {f"frac_{k}": round(frac[k], 4) for k in dets}
            mc = {k: retry_mc(v, k, B, r_retry, (seed, k_t, int(round(100 * B)))) for k in dets}
            row["multi"][str(B)] = {k: round(m - ph, 4) for k, (m, ph) in mc.items()}
        per_task[t] = row
    agg = {}
    for tt in sorted({v["type"] for v in per_task.values()}) + ["all"]:
        rows = [v for v in per_task.values() if tt == "all" or v["type"] == tt]
        a = dict(tasks=len(rows), mean_p=round(float(np.mean([v["p"] for v in rows])), 4),
                 false_alarm_exposure={k: round(float(np.mean([v["false_alarm_exposure"][k] for v in rows])), 4) for k in dets})
        for kind in ("single", "multi"):
            a[kind] = {}
            for B in budgets:
                m = {k: float(np.mean([v[kind][str(B)][k] for v in rows])) for k in dets}
                a[kind][str(B)] = {k: round(m[k], 4) for k in dets} | dict(
                    timer_vs_none=round(m["P"] - m["none"], 4), safe_vs_timer=round(m["safe_plus_P"] - m["P"], 4))
        # Episodes per arm to detect SAFE + P vs P on success rate (two-sided 0.05, power 0.8),
        # from the multi-retry bound at 2 H_k. A bound on the effect gives a lower bound on n.
        Bm = str(max(budgets)); p1 = a["mean_p"] + a["multi"][Bm]["P"]; p2 = a["mean_p"] + a["multi"][Bm]["safe_plus_P"]
        d = p2 - p1
        a["episodes_per_arm_safe_vs_timer"] = math.inf if d <= 0 else math.ceil(
            (1.959964 + 0.841621) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / d ** 2)
        agg[tt] = a
    return dict(per_task=per_task, by_type=agg)


def e3(args):
    out_path = check_out(args.output)
    out = dict(plan="docs/stage2_sizing.md", status="exploratory bound (opened cohorts, shadow data)", experiment="E3",
               budgets=list(args.budgets), retry_success=args.retry_success, policies={})
    # XR-1: the registered E1 fit (pooled calibration, 38 tasks).
    bundle = json.load(open(args.runtime_bundle))
    rk = qt.rho(qt.dev_fractions(bundle), Q)
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    tm, fu = cl.fit_fast(cl.Cohort(cal), rk)
    teC = cl.Cohort(te)
    eP, _ = cl.evaluate_fast(teC, rk, tm); eF, _ = cl.evaluate_fast(teC, rk, fu)
    rates = success_rates(cal_all + te_all)
    succ = defaultdict(list)
    for r in cal_all + te_all:
        if not r["failed"]:
            succ[r["task_name"]].append(r["L"])
    types = {r["task_name"]: r.get("task_type") for r in te}
    ti = task_inputs(teC, rk, tm, fu, rates, {t: np.median(v) for t, v in succ.items()}, types)
    out["policies"]["xr1"] = dict(check=dict(P_lead=eP["mean_lead"], safe_plus_P_lead=eF["mean_lead"], P_fpr=eP["fpr"],
                                             safe_plus_P_fpr=eF["fpr"]), **e3_policy(ti, args.budgets, args.retry_success, SEED))
    print(f"check XR-1: P lead {eP['mean_lead']} (FPR {eP['fpr']}), SAFE+P lead {eF['mean_lead']} (FPR {eF['fpr']})", flush=True)
    if args.rldx_calibration_dir:
        rows, train, tasks, rk2 = rldx_rows(args.rldx_calibration_dir, args.model)
        cal_s = [r for r in rows if r["split"] == "calibration"]
        test = [r for r in rows if r["split"] == "evaluation"]
        tm2, fu2 = cl.fit_fast(cl.Cohort(cal_s + [r for r in train if r["failed"]]), rk2)
        teC2 = cl.Cohort(test)
        eP2, _ = cl.evaluate_fast(teC2, rk2, tm2); eF2, _ = cl.evaluate_fast(teC2, rk2, fu2)
        rates2 = success_rates(rows)
        succ2 = defaultdict(list)
        for r in train:
            if not r["failed"]:
                succ2[r["task_name"]].append(r["L"])
        H2 = {r["task_name"]: r["horizon"] for r in rows}
        types2 = {r["task_name"]: r["task_type"] for r in rows}
        ti2 = task_inputs(teC2, rk2, tm2, fu2, rates2, {t: np.median(succ2[t]) if succ2[t] else H2[t] for t in tasks}, types2)
        out["policies"]["rldx1"] = dict(check=dict(P_lead=eP2["mean_lead"], safe_plus_P_lead=eF2["mean_lead"],
                                                   P_fpr=eP2["fpr"], safe_plus_P_fpr=eF2["fpr"]),
                                        **e3_policy(ti2, args.budgets, args.retry_success, SEED + 1))
        print(f"check RLDX-1: P lead {eP2['mean_lead']} (FPR {eP2['fpr']}), SAFE+P lead {eF2['mean_lead']} (FPR {eF2['fpr']})")
    write(out_path, out)
    for pol, v in out["policies"].items():
        for kind, label in (("single", "single-retry bound (1-p) r frac"), ("multi", "multi-retry Monte Carlo, success-rate gain over p")):
            print(f"\n## E3 {pol}: {label}, retry success = {args.retry_success}")
            print("| type | budget x H | none | P | SAFE+P | P - none | SAFE+P - P |")
            print("|---|---:|---:|---:|---:|---:|---:|")
            for tt, a in v["by_type"].items():
                for B, m in a[kind].items():
                    print(f"| {tt} | {B} | {m['none']:+.4f} | {m['P']:+.4f} | {m['safe_plus_P']:+.4f} | "
                          f"{m['timer_vs_none']:+.4f} | {m['safe_vs_timer']:+.4f} |")
        print(f"\n## E3 {pol}: false-alarm exposure p x FPR, and episodes per arm for SAFE+P vs P (multi-retry, {max(args.budgets)} H)")
        print("| type | tasks | mean p | FA exposure P | FA exposure SAFE+P | episodes per arm |")
        print("|---|---:|---:|---:|---:|---:|")
        for tt, a in v["by_type"].items():
            print(f"| {tt} | {a['tasks']} | {a['mean_p']:.3f} | {a['false_alarm_exposure']['P']:.4f} | "
                  f"{a['false_alarm_exposure']['safe_plus_P']:.4f} | {a['episodes_per_arm_safe_vs_timer']} |")


# ---- CLI -----------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def sim_args(s, gains):
        s.add_argument("--sizes", type=sizes_arg, default=SIZES)
        s.add_argument("--gains", type=gains_arg, default=gains)
        s.add_argument("--replicates", type=int, default=REPLICATES)
        s.add_argument("--nested-sizes", type=sizes_arg, default=NESTED_SIZES)
        s.add_argument("--nested-outer", type=int, default=NESTED_OUTER)
        s.add_argument("--nested-inner", type=int, default=NESTED_INNER)
        s.add_argument("--workers", type=int, default=1)
        s.add_argument("--output", required=True)

    a = sub.add_parser("e1b")
    for name in ("--calibration", "--test", "--runtime-bundle"):
        a.add_argument(name, required=True)
    a.add_argument("--dose-response")
    a.add_argument("--stratum", default="atomic")
    a.add_argument("--pool", choices=("all", "primary"), default="all")
    sim_args(a, E1B_GAINS)
    b = sub.add_parser("e2")
    b.add_argument("--calibration-dir", required=True)
    b.add_argument("--model", default="lstm", choices=("lstm", "indep"))
    b.add_argument("--replication-json")
    sim_args(b, E2_GAINS)
    c = sub.add_parser("e3")
    for name in ("--calibration", "--test", "--runtime-bundle"):
        c.add_argument(name, required=True)
    c.add_argument("--rldx-calibration-dir")
    c.add_argument("--model", default="lstm", choices=("lstm", "indep"))
    c.add_argument("--budgets", type=gains_arg, default=BUDGETS)
    c.add_argument("--retry-success", default="same", help="'same' (= task success rate) or a probability")
    c.add_argument("--output", required=True)
    args = p.parse_args()
    {"e1b": e1b, "e2": e2, "e3": e3}[args.cmd](args)


if __name__ == "__main__":
    main()
