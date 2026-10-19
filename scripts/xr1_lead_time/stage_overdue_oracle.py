"""Oracle stage-overdue detector against the per-task quantile timer P (go/no-go).

Plan and decision rule: stage_overdue_oracle_plan.md (written before running).

The P definitions copy quantile_timer.py / lead_time.py from the RSS draft
(reports/paper_time_aware_fusion/rss2027_stage1/analysis/), so this file needs only
numpy and runs on the cluster without those untracked files. `reproduce-p` checks
the copy against the registered dry run on the opened August cohorts.

Usage:
  python3 -I stage_overdue_oracle.py reproduce-p <review_data dir> [quantile_timer.json]
  python3 -I stage_overdue_oracle.py run --runtime-bundle <runtime_bundle.json> \
      --dataset-dir <traced dataset> [<traced dataset> ...] --output <out.json>
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ALPHA = 0.05
EPS = 1e-9
PRIMARY_Q = 0.95
QS = (0.75, 0.90, 0.95, 1.00)
CONFIRMATORY_SEEDS = (1300007, 1325007)    # never read (amendment 1 cohort)
K_FOLDS = 5
B = 2000


# ---- P, copied from quantile_timer.py -------------------------------------------------

def dev_success_fractions(bundle, task):
    ac = bundle["time_risk_curves"][task]["active_counts"]; H = len(ac)
    fr = []
    for i in range(1, H):
        fr += [i / H] * (ac[i - 1] - ac[i])
    return fr

def rho_registered(bundle, q):
    return {t: float(np.quantile(v, q)) if v else 1.0
            for t in bundle["time_risk_curves"] for v in [dev_success_fractions(bundle, t)]}

def timer_idx(r, rk, delta):
    c = min(1.0, rk[r["task_name"]] + delta)
    return math.ceil(c * r["horizon"] - EPS)

def conformal(vs, alpha=ALPHA):
    vs = sorted(vs); n = len(vs); k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else vs[k - 1]


# ---- detectors: P, oracle stage + P, oracle stage only ---------------------------------
# Each detector is (rk, Q, use_timer). rk: task -> rho (None = no timer).
# Q: (task, stage) -> fraction of H (None = no stage rule).

def stage_excess(r, Q):
    """x_t = tau_t/H - Q_{k,s_t} per inference (−inf where the stage has no quantile)."""
    H = r["horizon"]; k = r["task_name"]
    return np.array([tau / H - Q[(k, s)] if (k, s) in Q else -math.inf
                     for s, tau in zip(r["stage"], r["tau"])])

def V(r, rk, Q):
    parts = [0.0]
    if rk is not None:
        if r["L"] >= r["horizon"]:
            return math.inf
        parts.append(r["L"] / r["horizon"] - rk[r["task_name"]])
    if Q is not None:
        parts.append(float(np.max(stage_excess(r, Q))))
    return max(parts) + EPS

def alarm(r, rk, Q, lam):
    a = math.inf
    if rk is not None:
        a = timer_idx(r, rk, lam)
    if Q is not None:
        hit = np.flatnonzero(stage_excess(r, Q) >= lam - 1e-12)
        if hit.size:
            a = min(a, int(hit[0]) + 1)
    return a

def fit_lam(cal_succ, rk, Q):
    return conformal([V(r, rk, Q) for r in cal_succ])

def score(rows, rk, Q, lam):
    """Per-rollout alarm, lead (failures) and fired flag."""
    out = []
    for r in rows:
        a = alarm(r, rk, Q, lam)
        fired = a <= r["L"]
        lead = (r["L"] - a) / r["horizon"] if fired else 0.0
        out.append(dict(rollout_id=r["rollout_id"], task=r["task_name"], failed=r["failed"],
                        fired=bool(fired), alarm=(int(a) if fired else None), lead=float(lead),
                        frac=(a / r["horizon"] if fired else 1.0)))
    return out


# ---- summaries and bootstrap -----------------------------------------------------------

def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]

def summarize(scored):
    S = [s for s in scored if not s["failed"]]; F = [s for s in scored if s["failed"]]
    fp = sum(s["fired"] for s in S); lv = np.array([s["lead"] for s in F]); fr = np.array([s["frac"] for s in F])
    return dict(n_success=len(S), n_failure=len(F), fp=int(fp), fpr=round(fp / max(len(S), 1), 4),
                fpr_wilson95=wilson(fp, len(S)), recall=round(float(np.mean([s["fired"] for s in F])), 4),
                mean_lead=round(float(lv.mean()), 4), by50=round(float(np.mean(fr <= 0.5)), 4),
                by75=round(float(np.mean(fr <= 0.75)), 4))

def boot(d, tasks_of_failures, seed=0):
    rng = np.random.default_rng(seed); bi = defaultdict(list)
    for i, t in enumerate(tasks_of_failures): bi[t].append(i)
    tasks = sorted(bi); hier, strat = [], []
    for _ in range(B):
        ii = np.concatenate([rng.choice(bi[t], len(bi[t])) for t in rng.choice(tasks, len(tasks))]); hier.append(d[ii].mean())
    for _ in range(B):
        ii = np.concatenate([rng.choice(bi[t], len(bi[t])) for t in tasks]); strat.append(d[ii].mean())
    return dict(mean=round(float(d.mean()), 4),
                ci95_task_then_rollout=np.round(np.percentile(hier, [2.5, 97.5]), 4).tolist(),
                ci95_rollout_within_task=np.round(np.percentile(strat, [2.5, 97.5]), 4).tolist())


# ---- reproduce-p: the copied P must match the registered dry run ----------------------

def load_review(path):
    out = []
    for l in open(path):
        if l.strip():
            r = json.loads(l); r["failed"] = bool(r["failed"]); r["L"] = len(r["safe_scores"]); out.append(r)
    return out

def sel(records, q):           # lead_time.sel
    g = defaultdict(list)
    for r in records: g[(r["task_name"], r["failed"])].append(r)
    out = []
    for t in sorted({r["task_name"] for r in records}):
        for f in (False, True):
            v = sorted(g[(t, f)], key=lambda r: (r.get("environment_seed") if r.get("environment_seed") is not None else -1,
                       r.get("environment_reset_index") if r.get("environment_reset_index") is not None else -1, str(r["rollout_id"])))
            out += v[:q]
    return out

def reproduce_p(D, ref_path):
    bundle = json.load(open(D / "runtime_bundle.json"))
    cal = sel(load_review(D / "calibration_predictions.jsonl"), 3)
    te = sel(load_review(D / "event_predictions.jsonl"), 10)
    ref = json.load(open(ref_path))["by_q"] if ref_path else None
    out, ok = {}, True
    for q in QS:
        rk = rho_registered(bundle, q)
        lam = fit_lam([r for r in cal if not r["failed"]], rk, None)
        sc = score(te, rk, None, lam); s = summarize(sc)
        out[str(q)] = dict(delta=round(lam, 4), **s)
        if ref:
            want = ref[str(q)]["timer"]
            same = (abs(want["delta"] - round(lam, 4)) < 1e-4 and want["fp"] == s["fp"]
                    and abs(want["mean_lead"] - s["mean_lead"]) < 1e-4)
            out[str(q)]["matches_quantile_timer_json"] = same; ok &= same
    rk = rho_registered(bundle, PRIMARY_Q)
    lam = fit_lam([r for r in cal if not r["failed"]], rk, None)
    sc = score(te, rk, None, lam)
    out["primary_P_on_August_test_by_task"] = {
        t: summarize([s for s in sc if s["task"] == t])
        for t in ("ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CuttingToolSelection", "GarnishPancake")}
    out["primary_P_on_August_test_5tasks"] = summarize([s for s in sc if s["task"] in out["primary_P_on_August_test_by_task"]])
    print(json.dumps(out, indent=1))
    if ref and not ok:
        sys.exit("REPRODUCTION FAILED: the copied P does not match quantile_timer.json")
    print("REPRODUCTION OK" if ref else "no reference given")


# ---- run: traced cohorts ---------------------------------------------------------------

def load_traced(dataset_dirs, bundle):
    rows, checks = [], defaultdict(int)
    for d in dataset_dirs:
        d = Path(d)
        for line in open(d / "manifest.jsonl"):
            if not line.strip(): continue
            m = json.loads(line)
            seed = m.get("environment_seed")
            if seed is not None and CONFIRMATORY_SEEDS[0] <= int(seed) < CONFIRMATORY_SEEDS[1]:
                sys.exit(f"STOP: rollout {m['rollout_id']} has a confirmatory-cohort seed {seed}")
            if not m.get("subtask_recording_available") or not m.get("subtask_trace_path"):
                sys.exit(f"STOP: rollout {m['rollout_id']} in {d} has no subtask trace")
            if not m.get("collection_complete", True):
                checks["skipped_incomplete"] += 1; continue
            tr = json.load(open(d / m["subtask_trace_path"]))
            recs = tr["inference_records"]; L = len(m["inference_env_steps"])
            if len(recs) != L or tr["rollout_id"] != m["rollout_id"] or bool(tr["rollout_failed"]) != bool(m["failed"]):
                sys.exit(f"STOP: trace/manifest mismatch for {m['rollout_id']}")
            task = m["task_name"]; H = int(bundle["task_horizons"][task]); n_def = len(tr["semantic_subtasks"])
            stage = [rc["segment_index"] if rc["segment_index"] is not None else n_def for rc in recs]
            if any(b < a for a, b in zip(stage, stage[1:])):
                sys.exit(f"STOP: stage index decreased in {m['rollout_id']}")
            tau, run = [], 0
            for i, s in enumerate(stage):
                run = run + 1 if i and s == stage[i - 1] else 1
                tau.append(run)
            failed = bool(m["failed"])
            checks["failures_L_eq_H" if failed and L == H else "failures_L_ne_H" if failed else "successes"] += 1
            checks["L_gt_H"] += int(L > H)
            rows.append(dict(rollout_id=m["rollout_id"], task_name=task, failed=failed, L=L, horizon=H,
                             n_stages=n_def, stage=stage, tau=tau, source=str(d),
                             environment_seed=seed, environment_reset_index=m.get("environment_reset_index")))
    ids = [r["rollout_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("STOP: duplicate rollout IDs across datasets")
    return rows, dict(checks)

def assign_folds(rows, seed=0):
    rng = np.random.default_rng(seed); g = defaultdict(list)
    for r in rows: g[(r["task_name"], r["failed"])].append(r)
    for key in sorted(g):
        v = sorted(g[key], key=lambda r: (r["environment_seed"] or -1, r["environment_reset_index"] or -1, r["rollout_id"]))
        offset = int(rng.integers(K_FOLDS))
        for j, i in enumerate(rng.permutation(len(v))):
            v[i]["fold"] = (j + offset) % K_FOLDS

def stage_quantiles(dev_succ, q):
    per = defaultdict(list)
    by_task = defaultdict(list)
    for r in dev_succ: by_task[r["task_name"]].append(r)
    Q = {}
    for k, rs in by_task.items():
        n = max(r["n_stages"] for r in rs)
        for j in range(n + 1):           # n = post-completion stage
            counts = [sum(s == j for s in r["stage"]) / r["horizon"] for r in rs]
            Q[(k, j)] = float(np.quantile(counts, q))
            per[k].append(round(Q[(k, j)], 4))
    return Q, dict(per)

def rho_local(dev_succ, q):
    by = defaultdict(list)
    for r in dev_succ: by[r["task_name"]].append(r["L"] / r["horizon"])
    return {k: float(np.quantile(v, q)) for k, v in by.items()}

DETECTORS = ("P_registered", "P_local", "oracle_stage_plus_P_registered",
             "oracle_stage_plus_P_local", "oracle_stage_only")

def run(args):
    bundle = json.load(open(args.runtime_bundle))
    rows, checks = load_traced(args.dataset_dir, bundle)
    assign_folds(rows, seed=0)
    tasks = sorted({r["task_name"] for r in rows})
    out = dict(plan="scripts/xr1_lead_time/stage_overdue_oracle_plan.md", alpha=ALPHA, primary_q=PRIMARY_Q,
               datasets=[str(d) for d in args.dataset_dir], runtime_bundle=str(args.runtime_bundle),
               checks=checks, tasks=tasks,
               counts={t: dict(success=sum(1 for r in rows if r["task_name"] == t and not r["failed"]),
                               failure=sum(1 for r in rows if r["task_name"] == t and r["failed"])) for t in tasks},
               by_q={})
    for q in QS:
        scored = {d: [] for d in DETECTORS}; fold_fits = []
        for f in range(K_FOLDS):
            cal_f = {(f + 1) % K_FOLDS, (f + 2) % K_FOLDS}
            test = [r for r in rows if r["fold"] == f]
            cal_s = [r for r in rows if r["fold"] in cal_f and not r["failed"]]
            dev_s = [r for r in rows if r["fold"] not in cal_f and r["fold"] != f and not r["failed"]]
            Q, Qtab = stage_quantiles(dev_s, q)
            rk_reg = rho_registered(bundle, q); rk_loc = rho_local(dev_s, q)
            spec = dict(P_registered=(rk_reg, None), P_local=(rk_loc, None),
                        oracle_stage_plus_P_registered=(rk_reg, Q), oracle_stage_plus_P_local=(rk_loc, Q),
                        oracle_stage_only=(None, Q))
            fit = dict(fold=f, n_cal_successes=len(cal_s), n_dev_successes=len(dev_s), stage_quantiles=Qtab)
            for name, (rk, QQ) in spec.items():
                lam = fit_lam(cal_s, rk, QQ)
                if not math.isfinite(lam):
                    sys.exit(f"STOP: too few calibration successes in fold {f} for alpha={ALPHA}")
                fit[f"delta_{name}"] = round(lam, 4)
                scored[name] += [dict(s, fold=f) for s in score(test, rk, QQ, lam)]
            fold_fits.append(fit)
        res = dict(overall={d: summarize(scored[d]) for d in DETECTORS},
                   by_task={t: {d: summarize([s for s in scored[d] if s["task"] == t]) for d in DETECTORS} for t in tasks},
                   folds=fold_fits)
        key = lambda s: s["rollout_id"]
        lead = {d: np.array([s["lead"] for s in sorted((s for s in scored[d] if s["failed"]), key=key)]) for d in DETECTORS}
        ftasks = [s["task"] for s in sorted((s for s in scored["P_registered"] if s["failed"]), key=key)]
        res["gain_oracle_plus_P_registered_minus_P_registered"] = boot(lead["oracle_stage_plus_P_registered"] - lead["P_registered"], ftasks)
        res["gain_oracle_plus_P_local_minus_P_local"] = boot(lead["oracle_stage_plus_P_local"] - lead["P_local"], ftasks)
        res["gain_oracle_plus_P_registered_minus_P_local"] = boot(lead["oracle_stage_plus_P_registered"] - lead["P_local"], ftasks)
        res["gain_by_task_oracle_plus_P_registered_minus_P_registered"] = {
            t: round(float(np.mean([a - b for a, b, tt in zip(lead["oracle_stage_plus_P_registered"], lead["P_registered"], ftasks) if tt == t])), 4)
            for t in tasks}
        if q == PRIMARY_Q:
            res["per_rollout"] = {d: scored[d] for d in DETECTORS}
            g = res["gain_oracle_plus_P_registered_minus_P_registered"]
            go = g["mean"] >= 0.05 and g["ci95_task_then_rollout"][0] > 0
            loc = res["gain_oracle_plus_P_registered_minus_P_local"]["mean"]
            out["verdict"] = ("GO" if go and loc >= 0.05 else "INCONCLUSIVE" if go else "NO-GO")
            out["verdict_inputs"] = dict(gain_vs_P_registered=g, gain_vs_P_local_point=loc,
                                         rule="GO iff gain >= 0.05 and task-then-rollout CI lower > 0; "
                                              "INCONCLUSIVE if that holds but point gain vs local P < 0.05")
        out["by_q"][str(q)] = res
    Path(args.output).write_text(json.dumps(out, indent=1))
    brief = {k: v for k, v in out.items() if k != "by_q"}
    brief["primary"] = {k: v for k, v in out["by_q"][str(PRIMARY_Q)].items() if k not in ("per_rollout", "folds")}
    brief["sensitivity_gain_vs_P_registered"] = {q: out["by_q"][q]["gain_oracle_plus_P_registered_minus_P_registered"] for q in out["by_q"]}
    print(json.dumps(brief, indent=1))
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("reproduce-p"); a.add_argument("review_data"); a.add_argument("reference", nargs="?")
    b = sub.add_parser("run"); b.add_argument("--runtime-bundle", required=True)
    b.add_argument("--dataset-dir", nargs="+", required=True); b.add_argument("--output", required=True)
    args = p.parse_args()
    if args.cmd == "reproduce-p":
        reproduce_p(Path(args.review_data), args.reference)
    else:
        run(args)
