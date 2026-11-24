"""Seen and unseen columns of RSS Tables I and II (R6 of docs/xr1_review_analyses_plan.md). Exploratory.

Hide-and-Seek reports every detector on seen and unseen tasks. This script does the same on the
held-out E1 cohorts (lead calibration 380/380, confirmatory test 380/380), per training seed:

Table I, standalone detection. One threshold per detector, chosen on the calibration rollouts
(largest mean lead with calibration FPR <= 0.05; per_seed_detection.threshold), applied once.
  seen    calibrate on every calibration rollout, test on every test rollout; scores normalized
          per task, as the tables are.
  unseen  per cross-task fold: calibrate on the seen tasks' calibration rollouts, test on the
          held-out tasks' test rollouts, with the fold's models and pooled normalization; the
          outcomes are pooled over the four folds.
  Reported: TPR, FPR, bACC = (TPR + 1 - FPR)/2, TWA = (lead + 1 - FPR)/2, DT = 1 - lead.
  Timers: the global timer (score t/H, same threshold rule) on both; the time risk and the
  per-task timer P (Proposition 1 margin, confirm_lead.fit_fast) only on seen tasks, since they
  need the task's own rollouts.

Table II, score OR-fused with a timer (confirm_lead.fit_fast), per seed:
  seen    + P on all tasks, per-task normalization. These are the published per-seed rows; they
          are recomputed here and must equal the stored runs (gate T2).
  unseen  + G per fold (cross_task_lead.unseen: G and the path fitted on seen-task calibration
          rollouts, pooled normalization), pooled over folds.
  Reported: FPR, mean lead, gain over the timer, TWA.

Score sources (raw per-seed trajectories, normalized here):
  SAFE-MLP        seen: E1 per_seed_raw_safe + the runtime bundle's task normalizations;
                  unseen: Stage B fold directories (detector "safe").
  Hide-and-Seek, zero-input, first-inference
                  seen: Amendment 2 scores and normalizations; unseen: cross-task fold dirs.
  same-task shuffle  seen only (it needs the scored task's training rollouts).
  ActProbe        seen: fold all of the ActProbe cross-task run (per-task normalization);
                  unseen: its folds 0-3.
  Mahalanobis, cosine k-NN, PCA-KMeans, RND (Section 15.10, tuned)
                  seen: the per-seed held-out files of xr1_ood_per_seed (per-task models);
                  unseen: score-fold directories of the same configuration refit per fold with
                  every task-specific reference pooled.

Gates (no table without them):
  T1  fold directories: the fold file's folds and held-out tasks, training on exactly the seen
      tasks, seeds 0-2, every selected rollout present with the right task, label and length.
  T2  seen + P per seed equals the stored per-seed runs (run C of xr1_per_seed_detection and
      ood_per_seed_fused) to 1e-12.
  T3  seed-averaged unseen + G gains equal the cross-task results (Hide-and-Seek, zero-input,
      first-inference: cross_task_lead.json; SAFE: cross_task_safe.json) at their stored precision.
  T4  ActProbe fold all reproduced the B1 raw trajectories (score_cross_task --reference-raw) and
      its per-task normalization equals B1's.
  T5  OOD fold directories: pooled variant of the frozen Section 15.10 bundle (SHA-256).

Usage:
  python3 -I seen_unseen_tables.py --calibration E1/calibration_predictions.jsonl \\
      --test E1/test_predictions.jsonl --runtime-bundle runtime_bundle.json --folds cross_task_folds.json \\
      --safe-fold-dirs S0 S1 S2 S3 --xt-fold-dirs X0 X1 X2 X3 --actprobe-fold-dirs A0 A1 A2 A3 Aall \\
      --ood-fold-dirs O0 O1 O2 O3 --ood-source-bundle 15_10/develop/baseline_bundle.json \\
      --amend2-scores CAL.jsonl TEST.jsonl --amend2-normalizations normalizations.json \\
      --ood-seen 0 CAL0 TEST0 --ood-seen 1 CAL1 TEST1 --ood-seen 2 CAL2 TEST2 \\
      --run-c per_seed_fused_heldout.json --ood-fused ood_per_seed_fused.json \\
      --cross-task-result cross_task_lead.json --cross-task-safe-result cross_task_safe.json \\
      --actprobe-reference-normalizations B1/normalizations.json --output OUT.json
"""
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl  # noqa: E402
import cross_task_lead as ctl  # noqa: E402
import ood_per_seed_fused as opf  # noqa: E402
import per_seed_detection as psd  # noqa: E402
import quantile_timer as qt  # noqa: E402

PLAN = "docs/xr1_review_analyses_plan.md (R6; R11 rows)"
ALL = ctl.ALL
SEEDS = ("0", "1", "2")
SAFE = "safe_mlp"
HS, ZERO, FIRST, SHUFFLE = "hide_and_seek", "hide_and_seek_zero", "hide_and_seek_first", "hide_and_seek_task_shuffle"
HS_FAMILY = (HS, ZERO, FIRST)
ACTPROBE = "actprobe"
SAFE_LSTM, ACTPROBE_T = "safe_lstm", "actprobe_t"          # R11 rows, from optional fold directories
OOD = ("mahalanobis", "knn_cosine", "pca_kmeans", "rnd")
R24_SEEN = (SAFE_LSTM,)                         # R24: Table III rows without an interval in R22
R24_UNSEEN = ("mahalanobis", FIRST, ZERO)
R24_KEYS = ("table2", "ensembles_seen", "per_type", "unseen_bootstrap", "nulls", "budget_timer", "teaser")
TIMERS = ("time_risk", "global_timer", "per_task_timer")
TABLE1 = OOD + (SAFE, ACTPROBE, HS) + TIMERS
TABLE2 = OOD + (SAFE, ACTPROBE) + HS_FAMILY + (SHUFFLE,)   # R11 adds ActProbe to Table II
STANDALONE_KEYS = ("tpr", "fpr", "bacc", "twa", "dt")
FUSED_KEYS = ("fpr", "mean_lead", "gain", "twa")
TOL = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean_std(per_seed, keys):
    return {k: dict(mean=float(np.mean([p[k] for p in per_seed])), std=float(np.std([p[k] for p in per_seed], ddof=1)))
            for k in keys}


# ---- standalone detection (Table I) ---------------------------------------------------------

def scored(rows, traj, seed):
    """Rows for one score: seed None averages the seeds' normalized trajectories."""
    out = []
    for r in rows:
        per = traj[str(r["rollout_id"])]
        x = np.mean([per[s] for s in sorted(per)], axis=0) if seed is None else per[seed]
        x = np.asarray(x, float)
        if len(x) != r["L"]:
            raise ValueError(f"{r['rollout_id']}: score length {len(x)} != {r['L']}")
        out.append(dict(score=x, L=r["L"], horizon=r["horizon"], failed=bool(r["failed"])))
    return out


def outcomes(rows, tau):
    """Per-rollout outcomes of one threshold, in the form that pools over folds."""
    fp, n, hits, leads = 0, 0, [], []
    for r in rows:
        a = psd.first(r["score"], tau)
        if r["failed"]:
            hit = math.isfinite(a)
            hits.append(hit)
            leads.append((r["L"] - a) / r["horizon"] if hit else 0.0)
        else:
            n += 1
            fp += int(math.isfinite(a))
    return dict(fp=fp, n=n, hits=hits, leads=leads)


def pool_outcomes(parts):
    return dict(fp=sum(p["fp"] for p in parts), n=sum(p["n"] for p in parts),
                hits=[h for p in parts for h in p["hits"]], leads=[v for p in parts for v in p["leads"]])


def standalone_metrics(o):
    tpr, fpr, lead = float(np.mean(o["hits"])), o["fp"] / o["n"], float(np.mean(o["leads"]))
    return dict(tpr=tpr, fpr=fpr, bacc=(tpr + 1 - fpr) / 2, twa=(lead + 1 - fpr) / 2, dt=1 - lead,
                mean_lead=lead, fp=o["fp"], n_success=o["n"], n_failure=len(o["hits"]))


def standalone_seen(cal, te, traj, seeds):
    out = {}
    for seed in [None, *seeds]:
        tau = psd.threshold(scored(cal, traj, seed))
        m = standalone_metrics(outcomes(scored(te, traj, seed), tau))
        m["threshold"] = float(tau)
        out["seed_average" if seed is None else f"seed_{seed}"] = m
    if len(seeds) > 1:
        out["mean_std_over_seeds"] = mean_std([out[f"seed_{s}"] for s in seeds], STANDALONE_KEYS)
    return out


def standalone_unseen(cal, te, held, folds, traj_by_fold, seeds):
    out = {}
    for seed in [None, *seeds]:
        parts, thresholds = [], {}
        for F in folds:
            c = [r for r in cal if r["task_name"] not in held[F]]
            e = [r for r in te if r["task_name"] in held[F]]
            tau = psd.threshold(scored(c, traj_by_fold[F], seed))
            thresholds[F] = float(tau)
            parts.append(outcomes(scored(e, traj_by_fold[F], seed), tau))
        m = standalone_metrics(pool_outcomes(parts))
        m["thresholds"] = thresholds
        out["seed_average" if seed is None else f"seed_{seed}"] = m
    if len(seeds) > 1:
        out["mean_std_over_seeds"] = mean_std([out[f"seed_{s}"] for s in seeds], STANDALONE_KEYS)
    return out


def per_task_timer_seen(cal, te, rk):
    """P alone (Proposition 1 margin), as a standalone detector."""
    calC, teC = cl.Cohort(cal, "timer"), cl.Cohort(te, "timer")
    tm, _ = cl.fit_fast(calC, rk)
    p = ctl.raw_eval(teC, rk, tm)
    m = standalone_metrics(dict(fp=p["fp"], n=p["n"], hits=list(p["hit"]), leads=list(p["lead"])))
    m["lam"] = float(tm["lam"])
    return {"seed_average": m}


# ---- fused with a timer (Table II) ----------------------------------------------------------

def fused_unseen(D, det, seeds):
    """Score + G per fold, pooled; per seed and for the seed average."""
    out = {}
    for seed in [None, *seeds]:
        tag = "" if seed is None else f"#{seed}"
        tG, fD, per_fold = ctl.unseen(D, lambda F: f"{det}{tag}@{F}", D.rk0)
        fpr, lead = fD["fp"] / fD["n"], float(fD["lead"].mean())
        out["seed_average" if seed is None else f"seed_{seed}"] = dict(
            fpr=fpr, mean_lead=lead, gain=ctl.gain(fD, tG), twa=(lead + 1 - fpr) / 2, fp=fD["fp"],
            timer=dict(fpr=tG["fp"] / tG["n"], mean_lead=float(tG["lead"].mean()), fp=tG["fp"]),
            paths={F: cl.det_json(v[1]) for F, v in per_fold.items()})
    out["mean_std_over_seeds"] = mean_std([out[f"seed_{s}"] for s in seeds], FUSED_KEYS)
    return out


# ---- R14: timers without H_k (hk_free.py) ------------------------------------------------------

def seen_cohort(traj, rows, seed):
    """per_seed_detection.run_fused's cohort: one score per row, seed None = seed average."""
    seeds = sorted(next(iter(traj.values())))
    out = []
    for r in rows:
        per = traj[str(r["rollout_id"])]
        score = np.mean([per[s] for s in seeds], axis=0) if seed is None else np.asarray(per[seed], float)
        out.append(dict(task_name=r["task_name"], failed=bool(r["failed"]), horizon=r["horizon"], L=len(score),
                        score=score))
    return cl.Cohort(out, key="score")


def r14_xr1(path, D, cal, te, held, folds, tasks, unseen_names, seen, seen_fused, unseen_fused, bundle, rkP, folds_sha):
    """R14 on XR-1: every unseen row against G, G_abs, G_llm_zero, G_llm_few; seen rows also against P, P_mult.
    Returns (result, problems). K1: G reproduces unseen + G; K5: P reproduces seen + P."""
    import hk_free as hk
    dur = json.loads(Path(path).read_text())
    problems, parse_ok = hk.check_durations(dur, tasks, held, folds_sha)
    if problems:
        return None, problems
    labels = ["seed_average"] + [f"seed_{s}" for s in SEEDS]
    tag = lambda lab: "" if lab == "seed_average" else "#" + lab.split("_", 1)[1]
    unseen, seen_r, k1, k5 = {}, {}, {}, {}
    for name in unseen_names:
        def pairs_of(lab, name=name):
            out = {}
            for F in folds:
                c, e = D.C[f"{name}{tag(lab)}@{F}"]
                out[F] = (c.take(D.cal_idx[F]), e.take(D.te_idx[F]))
            return out
        unseen[name] = hk.block(pairs_of, labels, hk.TIMERS, dur, rkP)
        k1[name] = max(abs(unseen[name]["G"][lab][k] - unseen_fused[name][lab][k])
                       for lab in labels for k in ("fpr", "mean_lead", "gain"))
    for name, traj in seen.items():
        if name not in seen_fused:
            continue
        seeds = sorted(next(iter(traj.values())))
        labs = ["seed_average"] + [f"seed_{s}" for s in seeds]

        def pairs_of(lab, traj=traj):
            seed = None if lab == "seed_average" else lab.split("_", 1)[1]
            return {ALL: (seen_cohort(traj, cal, seed), seen_cohort(traj, te, seed))}
        seen_r[name] = hk.block(pairs_of, labs, hk.SEEN_TIMERS, dur, rkP)
        ref = seen_fused[name]
        tol = TOL if name not in OOD else 1e-4          # the OOD seen rows are stored at 4 decimals
        diffs = [abs(seen_r[name]["P"][lab][k] - ref[lab][k]) for lab in labs if lab in ref
                 for k in ("fpr", "mean_lead", "gain")]
        k5[name] = dict(max_abs_diff=max(diffs) if diffs else None, tol=tol)
    for name, v in k1.items():
        if not v <= TOL:
            problems.append(f"K1: {name}: G here differs from unseen + G by {v}")
    for name, v in k5.items():
        if v["max_abs_diff"] is None or not v["max_abs_diff"] <= v["tol"]:
            problems.append(f"K5: {name}: P here differs from seen + P by {v['max_abs_diff']}")
    H = bundle["task_horizons"]
    ref_s = {t: rkP[t] * H[t] * hk.SECONDS_PER_INFERENCE["xr1"] for t in tasks}
    first = next(iter(unseen.values()))
    first_seen = next(iter(seen_r.values()))
    result = dict(
        plan="docs/xr1_review_analyses_plan.md (R14)",
        durations=dict(path=str(Path(path).resolve()), sha256=sha256(path), model=dur.get("model"),
                       parse_failures=dur.get("parse_failures"), parse_ok=parse_ok),
        llm_accuracy=hk.accuracy(dur, tasks, held, folds, ref_s),
        checks=dict(K1_G_reproduces_unseen_plus_G=k1, K5_P_reproduces_seen_plus_P=k5),
        timers_alone=dict(unseen={t: first[t]["seed_average"]["timer"] for t in hk.TIMERS},
                          seen={t: first_seen[t]["seed_average"]["timer"] for t in hk.SEEN_TIMERS}),
        unseen=unseen, seen=seen_r)
    return result, problems


def print_r14(r):
    import hk_free as hk
    print("R14 timers alone (FPR, FP, lead, TWA): unseen | seen")
    for t in hk.SEEN_TIMERS:
        u, s = r["timers_alone"]["unseen"].get(t), r["timers_alone"]["seen"][t]
        cu = f"{u['fpr']:.3f} {u['fp']:3d} {u['mean_lead']:.3f} {u['twa']:.3f}" if u else "   --"
        print(f"  {t:11s} {cu} | {s['fpr']:.3f} {s['fp']:3d} {s['mean_lead']:.3f} {s['twa']:.3f}")
    a = r["llm_accuracy"]
    for k in ("zero", "few_held_out"):
        print(f"  LLM {k}: Spearman {a[k]['spearman']:.2f}, median |log ratio| {a[k]['median_abs_log_ratio']:.2f}, "
              f"median ratio {a[k]['median_ratio']:.2f}, worst {a[k]['worst']}")
    print("R14 unseen, gain over each timer (seed average; matched in brackets):")
    print("  " + " " * 28 + "".join(f"{t:>20s}" for t in hk.TIMERS))
    for name, by in r["unseen"].items():
        cells = []
        for t in hk.TIMERS:
            v = by[t]["seed_average"]
            m = v["matched"]["gain"] if v["matched"] else float("nan")
            cells.append(f"{v['gain']:+.3f} [{m:+.3f}] {v['fp']:3d}")
        print(f"  {name:28s}" + "".join(f"{c:>20s}" for c in cells))
    print("R14 seen, gain over each timer (seed average):")
    print("  " + " " * 28 + "".join(f"{t:>11s}" for t in hk.SEEN_TIMERS))
    for name, by in r["seen"].items():
        print(f"  {name:28s}" + "".join(f"{by[t]['seed_average']['gain']:+11.3f}" for t in hk.SEEN_TIMERS))


# ---- R18: calibrated lead-FPR curves (lead_fpr_curves.py) -------------------------------------

R18_PLAN = "docs/r18_lead_fpr_curves_plan.md (R18)"


def r18_xr1(cal, te, held, folds, seen, unseen, table1, rkP, names):
    """R18 on XR-1: lead-FPR curves of every named monitor (seen and unseen) and of the timers G, E, P.
    C1: at alpha = 0.05 every curve point equals Table I's row (FPR and lead; seed average and seeds)."""
    import lead_fpr_curves as lfc
    lab = lambda s: "seed_average" if s is None else f"seed_{s}"
    labels = [None, *SEEDS]
    out = dict(plan=R18_PLAN, alphas=list(lfc.ALPHAS), seen={}, unseen={}, timers=dict(seen={}, unseen={}))
    c1 = {}

    def unseen_parts(traj_by_fold, s):
        return [(scored([r for r in cal if r["task_name"] not in held[F]], traj_by_fold[F], s),
                 scored([r for r in te if r["task_name"] in held[F]], traj_by_fold[F], s)) for F in folds]

    for name in names:
        sc = {lab(s): lfc.curve([(scored(cal, seen[name], s), scored(te, seen[name], s))]) for s in labels}
        uc = {lab(s): lfc.curve(unseen_parts(unseen[name], s)) for s in labels}
        sc["mean_std_over_seeds"] = lfc.mean_std([sc[lab(s)] for s in SEEDS])
        uc["mean_std_over_seeds"] = lfc.mean_std([uc[lab(s)] for s in SEEDS])
        out["seen"][name], out["unseen"][name] = sc, uc
        if name in table1:
            c1[name] = max(abs(lfc.at_alpha(cv[lab(s)], 0.05)[k] - table1[name][side][lab(s)][k])
                           for side, cv in (("seen", sc), ("unseen", uc)) for s in labels for k in ("fpr", "mean_lead"))
    for kind in lfc.TIMER_KINDS:
        out["timers"]["seen"][kind] = lfc.curve([(lfc.timer_rows(cal, kind, rkP), lfc.timer_rows(te, kind, rkP))])
    for kind in ("G", "E"):
        out["timers"]["unseen"][kind] = lfc.curve(
            [(lfc.timer_rows([r for r in cal if r["task_name"] not in held[F]], kind),
              lfc.timer_rows([r for r in te if r["task_name"] in held[F]], kind)) for F in folds])
    g = table1["global_timer"]
    c1["global_timer"] = max(abs(lfc.at_alpha(out["timers"][side]["G"], 0.05)[k] - g[side]["seed_average"][k])
                             for side in ("seen", "unseen") for k in ("fpr", "mean_lead"))
    out["checks"] = dict(C1_alpha_005_equals_table1=c1)
    problems = [f"C1: {k}: the alpha = 0.05 point differs from Table I by {v}" for k, v in c1.items() if not v <= TOL]
    return out, problems


def print_r18(r, alphas=(0.02, 0.05, 0.10, 0.20)):
    import lead_fpr_curves as lfc
    print("R18 lead-FPR curves, test FPR and mean lead at calibration targets " + ", ".join(map(str, alphas)))
    for side in ("seen", "unseen"):
        print(f"  {side}:")
        for name, by in r[side].items():
            cells = []
            for a in alphas:
                p = lfc.at_alpha(by["mean_std_over_seeds"], a)
                cells.append(f"{p['fpr']['mean']:.3f}/{p['mean_lead']['mean']:.3f}")
            print(f"    {name:28s} " + "  ".join(cells))
        for kind, pts in r["timers"][side].items():
            cells = [f"{lfc.at_alpha(pts, a)['fpr']:.3f}/{lfc.at_alpha(pts, a)['mean_lead']:.3f}" for a in alphas]
            print(f"    timer {kind:22s} " + "  ".join(cells))


# ---- R19: the draft v3 mock review's offline analyses (r19_extras.py) ----------------------------

def r19_xr1(cal, te, held, folds, raw_unseen, seen, rkP, types, reference, B):
    """R19 on XR-1. raw_unseen {name: {F: {rid: {seed: raw normalized}}}}; seen {name: {rid: {seed: traj}}};
    rows already carry name@F (running maximum, pooled) for every unseen name. reference: R14's output."""
    import r19_extras as rx
    spi = 0.8                                   # 16 control steps per inference at 20 Hz
    for name, by_fold in raw_unseen.items():
        for F in folds:
            for r in cal + te:
                per = by_fold[F][str(r["rollout_id"])]
                r[f"{name}~@{F}"] = np.mean([per[s] for s in sorted(per)], axis=0)
    for name, traj in seen.items():
        for r in cal + te:
            per = traj[str(r["rollout_id"])]
            r[f"{name}@seen"] = np.mean([per[s] for s in sorted(per)], axis=0)
            for s in sorted(per):
                r[f"{name}#{s}@seen"] = np.asarray(per[s], float)
    unseen_parts = {F: ([r for r in cal if r["task_name"] not in held[F]], [r for r in te if r["task_name"] in held[F]])
                    for F in folds}
    bands = {n: dict(unseen={t: rx.native_band(unseen_parts, lambda F, n=n: f"{n}~@{F}", lambda F, n=n: f"{n}@{F}", t, rkP)
                             for t in ("G_abs", "G")}) for n in raw_unseen}
    heads = tuple(raw_unseen)
    design = rx.UnseenDesign(cal, te, held, folds)
    key_ofs = {n: (lambda F, n=n: f"{n}@{F}") for n in heads + (ZERO,)}
    boot, point = rx.boot_unseen(design, key_ofs, ("G_abs", "G"), rkP, B=B,
                                 contrasts=((HS, SAFE), (HS, ZERO)))
    ref = {t: {n: reference["r14"]["unseen"][n][t]["seed_average"] for n in key_ofs} for t in ("G_abs", "G")}
    b0 = rx.check_point(point, ref)
    detail, limits = {}, {}
    for n in heads:
        un_pairs = design.pairs(lambda F, n=n: f"{n}@{F}")
        du = rx.leads_detail(un_pairs, "G_abs", rkP)
        detail[n] = dict(unseen=dict(seconds=rx.seconds(du, types, spi), per_task=rx.per_task(du)))
        if n in seen:
            seen_pairs = {ALL: (cl.Cohort(cal, f"{n}@seen"), cl.Cohort(te, f"{n}@seen"))}
            ds = rx.leads_detail(seen_pairs, "P", rkP)
            detail[n]["seen"] = dict(seconds=rx.seconds(ds, types, spi), per_task=rx.per_task(ds))
            if n == HS:
                limits = dict(P_seen=rx.timer_at_limit(ds), E_unseen=rx.timer_at_limit(du),
                              G_unseen=rx.timer_at_limit(rx.leads_detail(un_pairs, "G", rkP)))
    sat = {}
    for n in (ZERO, HS):
        if n in seen:
            for tag in ("", "#0", "#1", "#2"):
                if f"{n}{tag}@seen" in cal[0]:
                    sat[f"{n}{tag}"] = rx.saturation(cal, f"{n}{tag}@seen")
    band_ok = all(all(u["checks"]["ok"] for u in b["unseen"].values()) for b in bands.values())
    problems = [k for k, v in (("B0", b0["ok"]), ("N0", band_ok)) if not v]
    return dict(plan=rx.PLAN, checks=dict(B0_point_equals_r14=b0, N0_band_timer_equals_hk_fit=band_ok),
                native_bands=bands, seen_bands="R12 B2 (xr1_list_b_20261030_212017): SAFE-MLP and Hide-and-Seek + P",
                bootstrap_unseen=boot, leads=detail, timers_at_limit=limits, zero_input_saturation=sat), problems


# ---- R21: the original alarm rules (r21_native_bands.py) ---------------------------------------------

def seen_amend2_raw(paths, norms_path, name, need):
    """{rid: {seed: per-task normalized raw score}} (no running maximum) from the Amendment 2 files."""
    norms = json.loads(Path(norms_path).read_text())
    out = {}
    for path in paths:
        for r in load_jsonl(path):
            if r.get("detector") != name or str(r["rollout_id"]) not in need:
                continue
            out[str(r["rollout_id"])] = {s: (np.asarray(raw, float) - norms[name][s][r["task_name"]]["location"])
                                         / norms[name][s][r["task_name"]]["scale"] for s, raw in r["per_seed_raw"].items()}
    if need - set(out):
        raise ValueError(f"Amendment 2 {name} lacks {len(need - set(out))} selected rollouts")
    return out


def seen_safe_raw(rows, bundle):
    """{rid: {seed: per-task normalized raw SAFE score}} with the bundle's frozen task normalization."""
    norms = bundle["task_normalizations"]
    return {str(r["rollout_id"]): {s: (np.asarray(raw, float) - norms[s][r["task_name"]]["location"])
                                   / norms[s][r["task_name"]]["scale"] for s, raw in r["per_seed_raw_safe"].items()}
            for r in rows}


def r21_xr1(args, fam, cal, te, held, folds, tasks, bundle, extra, seen, rkP):
    """R21 on XR-1: SAFE-MLP, Hide-and-Seek (and SAFE-LSTM with its fold directories) under SAFE's band, on
    seen tasks against P and unseen tasks against E (G_abs) and G. Gates N1 (R19) and N2 (R12 B2)."""
    import r21_native_bands as r21
    ref19 = json.loads(Path(args.r21_reference).read_text()) if args.r21_reference else None
    if ref19 is None or "native_bands" not in (ref19.get("r19") or {}):
        raise SystemExit("error: --r21 needs --r21-reference, R19's XR-1 output (seen_unseen_tables_r19.json)")
    if not args.r21_r12_reference:
        raise SystemExit("error: --r21 needs --r21-r12-reference, R12's list_b.json (gate N2)")
    list_b = json.loads(Path(args.r21_r12_reference).read_text())
    rows = cal + te
    need = {str(r["rollout_id"]) for r in rows}
    heads = (SAFE, HS) + ((SAFE_LSTM,) if SAFE_LSTM in extra else ())
    seen_raw = {SAFE: seen_safe_raw(rows, bundle),
                HS: seen_amend2_raw(args.amend2_scores, args.amend2_normalizations, HS, need)}
    if SAFE_LSTM in heads:
        a_all = {ALL: fam[SAFE_LSTM][ALL]}
        seen_raw[SAFE_LSTM] = merge(
            trajectories(a_all, SAFE_LSTM, cal, ctl.CAL_COHORT, held, "per_task", running_max=False)[ALL],
            trajectories(a_all, SAFE_LSTM, te, ctl.TEST_COHORT, held, "per_task", running_max=False)[ALL])

    def unseen_raw(family, det):
        data = {F: v for F, v in fam[family].items() if F != ALL}
        c = trajectories(data, det, cal, ctl.CAL_COHORT, held, "pooled", running_max=False)
        e = trajectories(data, det, te, ctl.TEST_COHORT, held, "pooled", running_max=False)
        return {F: merge(c[F], e[F]) for F in folds}

    raw_unseen = {SAFE: unseen_raw("safe", "safe"), HS: unseen_raw("xt", HS)}
    if SAFE_LSTM in heads:
        raw_unseen[SAFE_LSTM] = unseen_raw(SAFE_LSTM, SAFE_LSTM)
    r0 = {}
    for h in heads:
        for r in rows:
            rid = str(r["rollout_id"])
            per = seen_raw[h][rid]
            for s in SEEDS:
                r[f"{h}~#{s}@seen"] = per[s]
                r[f"{h}#{s}@seen"] = np.asarray(seen[h][rid][s], float)
            r[f"{h}~@seen"] = np.mean([per[s] for s in SEEDS], axis=0)
            for F in folds:
                per = raw_unseen[h][F][rid]
                for s in SEEDS:
                    r[f"{h}~#{s}@{F}"] = per[s]
                r[f"{h}~@{F}"] = np.mean([per[s] for s in sorted(per)], axis=0)
        for s in SEEDS:
            r0[f"{h}/seen/seed_{s}"] = r21.running_max_gap(rows, f"{h}~#{s}@seen", f"{h}#{s}@seen")
            for F in folds:
                r0[f"{h}/unseen/{F}/seed_{s}"] = r21.running_max_gap(rows, f"{h}~#{s}@{F}", f"{h}#{s}@{F}")
    every = set(tasks)
    sides = dict(seen=dict(design=r21.Design(cal, te, {ALL: every}, {ALL: every}), timers=("P",)),
                 unseen=dict(design=r21.Design(cal, te, {F: every - held[F] for F in folds}, {F: held[F] for F in folds}),
                             timers=("G_abs", "G")))

    def keys(h, side):
        at = (lambda F: "seen") if side == "seen" else (lambda F: F)
        out = {r21.ENSEMBLE: lambda F, h=h, at=at: f"{h}~@{at(F)}"}
        for s in SEEDS:
            out[f"seed_{s}"] = lambda F, h=h, s=s, at=at: f"{h}~#{s}@{at(F)}"
        return out

    spec = {h: {side: keys(h, side) for side in sides} for h in heads}
    res, problems = r21.run(sides, spec, rkP, B=args.r21_boot, contrasts=((HS, SAFE),), r0=r0,
                            progress=lambda m: print(m, flush=True))
    n1 = r21.check_r19(res, ref19["r19"]["native_bands"])
    n2 = r21.check_r12(res, list_b)
    res["checks"]["N1_ensemble_equals_r19"] = n1
    res["checks"]["N2_seen_ensemble_equals_r12_b2"] = n2
    problems += [k for k, v in (("N1", n1["ok"]), ("N2", n2["ok"])) if not v]
    res["inputs"] = dict(r19=dict(path=str(Path(args.r21_reference).resolve()), sha256=sha256(args.r21_reference)),
                         r12_list_b=dict(path=str(Path(args.r21_r12_reference).resolve()),
                                         sha256=sha256(args.r21_r12_reference)))
    diag, diag_problems = None, []
    if args.r21b:
        if not args.r21b_reference:
            raise SystemExit("error: --r21b needs --r21b-reference, R21's XR-1 output (seen_unseen_tables_r21.json)")
        diag, diag_problems = r21.run_diagnostic(args, sides, spec, rkP, args.output, args.r21b_reference)
    return res, problems, diag, diag_problems


# ---- R22: every fused path fixed on the validation set (r22_fixed_paths.py) ---------------------

R22_VAL_COHORTS = ("august_calibration", "august_test")
R22_TEASER_ROLLOUT = "d5be410e4f86cbc95d554fe7"


def load_fold_cohorts(path, cohorts, wanted):
    """cross_task_lead.load_fold_dir for any cohort names: (meta, norms, scores, path)."""
    path = Path(path)
    meta = json.loads((path / "scoring.json").read_text())
    norms = json.loads((path / "normalizations.json").read_text())
    scores = {}
    for cohort in cohorts:
        out = defaultdict(dict)
        for row in load_jsonl(path / f"{cohort}_predictions.jsonl"):
            rid = str(row["rollout_id"])
            if rid in wanted[cohort]:
                out[row["detector"]][rid] = dict(task_name=row["task_name"], failed=bool(row["failed"]),
                                                 held_out=bool(row["held_out"]),
                                                 raw={s: np.asarray(v, float) for s, v in row["per_seed_raw"].items()})
        scores[cohort] = dict(out)
    return meta, norms, scores, str(path.resolve())


def r22_xr1(args, D, cal, te, held, folds, tasks, bundle, extra, seen, rkP, types):
    """R22 on XR-1: Table II (seen + P, unseen + E and G), the seen-task ensembles, the unseen ensembles'
    bootstrap, H2's nulls, per-type gains, the budget-allocated timer and Fig. 1, all with fixed paths; gates
    V0, V1, K0 (against R14), K8 (against R8) and N0."""
    import confirm_hide_and_seek as chs
    import lead_time
    import r19_extras as rx
    import r22_fixed_paths as r22
    ref14 = json.loads(Path(args.r22_reference).read_text())["r14"]
    r8 = json.loads(Path(args.r22_r8).read_text())
    # Validation rows: R8's selection (3 + 3 August calibration and 10 + 10 August test rollouts per task).
    val_cal = lead_time.sel(lead_time.load(Path(args.r22_val_calibration)), 3)
    val_test = lead_time.sel(lead_time.load(Path(args.r22_val_test)), 10)
    val = val_cal + val_test
    ids = {c: [str(r["rollout_id"]) for r in rows] for c, rows in zip(R22_VAL_COHORTS, (val_cal, val_test))}
    want_ids = {c: set(r22_ids(p)) for c, p in zip(R22_VAL_COHORTS, args.r22_val_ids)}
    checks = {}
    checks["V1_selection_equals_stage_a_ids"] = {c: dict(selected=len(ids[c]), listed=len(want_ids[c]), ok=set(ids[c]) == want_ids[c])
                                                for c in R22_VAL_COHORTS}
    e1_ids = {str(r["rollout_id"]) for r in cal + te}
    val_ids = {str(r["rollout_id"]) for r in val}
    checks["V0_disjoint"] = dict(shared_ids=len(e1_ids & val_ids), shared_identities=len(cl.identities(cal + te) & cl.identities(val)),
                                 tasks_equal={r["task_name"] for r in val} == tasks)
    checks["V0_disjoint"]["ok"] = (checks["V0_disjoint"]["shared_ids"] == 0 and checks["V0_disjoint"]["shared_identities"] == 0
                                   and checks["V0_disjoint"]["tasks_equal"])
    # Seen-task validation scores: SAFE-MLP (R8's per-seed raw scores), the Hide-and-Seek family and the
    # shuffle (Amendment 2's August files), the OOD scores (the per-seed calibrate / evaluate outputs) and
    # SAFE-LSTM (stage A's fold all).
    need = val_ids
    seen_val = {SAFE: seen_safe(val, bundle)}
    for name in (*HS_FAMILY, SHUFFLE):
        seen_val[name] = seen_amend2(args.r22_val_hs_scores, args.amend2_normalizations, name, need)
    seen_val.update(seen_ood(args.r22_ood_val, need))
    root = Path(args.r22_scores_dir)
    wanted = {c: set(ids[c]) for c in R22_VAL_COHORTS}

    def fold_data(family, F):
        return load_fold_cohorts(root / family / f"fold_{F}", R22_VAL_COHORTS, wanted)

    def val_traj(family, det, Fs, norm):
        data = {F: fold_data(family, F) for F in Fs}
        out = {}
        for c, rows in zip(R22_VAL_COHORTS, (val_cal, val_test)):
            t = trajectories(data, det, rows, c, held, norm)
            for F in Fs:
                out.setdefault(F, {}).update(t[F])
        return out

    if SAFE_LSTM in extra:
        seen_val[SAFE_LSTM] = val_traj("safe_lstm", SAFE_LSTM, [ALL], "per_task")[ALL]
    unseen_fam = dict({SAFE: ("safe", "safe"), HS: ("xt", HS), ZERO: ("xt", ZERO), FIRST: ("xt", FIRST)},
                      **{n: ("ood", n) for n in OOD})
    if SAFE_LSTM in extra:
        unseen_fam[SAFE_LSTM] = ("safe_lstm", SAFE_LSTM)
    for name, (family, det) in unseen_fam.items():
        traj = val_traj(family, det, folds, "pooled")
        for F in folds:
            for r in val:
                per = traj[F][str(r["rollout_id"])]
                r[f"{name}@{F}"] = np.mean([per[s] for s in SEEDS], axis=0)
                for s in SEEDS:
                    r[f"{name}#{s}@{F}"] = per[s]
    labels = list(r22.LABELS)
    tag = lambda lab: "" if lab == "seed_average" else "#" + lab.split("_", 1)[1]
    seed_of = lambda lab: None if lab == "seed_average" else lab.split("_", 1)[1]

    def seen_pairs(name):
        return lambda lab: {ALL: (seen_cohort(seen[name], cal, seed_of(lab)), seen_cohort(seen[name], te, seed_of(lab)))}

    def seen_val_of(name):
        return lambda lab: {ALL: seen_cohort(seen_val[name], val, seed_of(lab))}

    def unseen_pairs(name):
        def f(lab):
            out = {}
            for F in folds:
                c, e = D.C[f"{name}{tag(lab)}@{F}"]
                out[F] = (c.take(D.cal_idx[F]), e.take(D.te_idx[F]))
            return out
        return f

    def unseen_val_of(name):
        return lambda lab: {F: cl.Cohort([r for r in val if r["task_name"] not in held[F]], f"{name}{tag(lab)}@{F}")
                            for F in folds}

    seen_names = [n for n in OOD + (SAFE,) + tuple(n for n in extra if n == SAFE_LSTM) + HS_FAMILY + (SHUFFLE,)]
    unseen_names = list(unseen_fam)
    table2, k0, n0 = dict(seen={}, unseen={}), {}, {}
    for name in seen_names:
        fixed = r22.block(seen_pairs(name), seen_val_of(name), labels, ("P",), rkP)
        adaptive = r22.block(seen_pairs(name), None, labels, ("P",), rkP)
        ref = (ref14["seen"].get(name) or {}).get("P")
        k0[f"seen/{name}"] = r22.k0(adaptive, {"P": ref}, ("P",), labels) if ref else dict(ok=True, note="no R14 row")
        n0[f"seen/{name}"] = all(fixed["P"][lab]["paths"][ALL]["timer"] == adaptive["P"][lab]["paths"][ALL]["timer"] for lab in labels)
        table2["seen"][name] = r22.strip(fixed)
    for name in unseen_names:
        fixed = r22.block(unseen_pairs(name), unseen_val_of(name), labels, ("G_abs", "G"), rkP)
        adaptive = r22.block(unseen_pairs(name), None, labels, ("G_abs", "G"), rkP)
        ref = {t: (ref14["unseen"].get(name) or {}).get(t) for t in ("G_abs", "G")}
        k0[f"unseen/{name}"] = r22.k0(adaptive, ref, ("G_abs", "G"), labels)
        n0[f"unseen/{name}"] = all(fixed[t][lab]["paths"][F]["timer"] == adaptive[t][lab]["paths"][F]["timer"]
                                   for t in ("G_abs", "G") for lab in labels for F in fixed[t][lab]["paths"])
        table2["unseen"][name] = r22.strip(fixed)
    checks["K0_adaptive_reproduces_r14"] = dict(by=k0, ok=all(v["ok"] for v in k0.values()))
    checks["N0_timer_margins_unchanged"] = dict(by=n0, ok=all(n0.values()))
    # Seen-task ensembles (Table III) and their per-type gains.
    Fte = [r for r in te if r["failed"]]
    ensembles, per_types, runs = {}, {}, {}
    r24 = bool(getattr(args, "r24", False))           # R24 adds rows; every R22 row stays as it was (gate R0)
    for name in (SAFE, HS, *OOD, FIRST, ZERO, SHUFFLE) + tuple(n for n in R24_SEEN if r24 and n in seen_val):
        row, run_ = r22.ensemble_row(seen_pairs(name)("seed_average"), seen_val_of(name)("seed_average"), "P", rkP, Fte)
        ensembles[name], runs[name] = row, run_
        if name in (SAFE, HS) + OOD:
            per_types[name] = r22.per_type(run_, types, 0.8)
    checks["K8_fixed_paths_equal_r8"] = r22.k8(ensembles, r8, {"safe": SAFE, "hide_and_seek": HS})
    # Unseen ensembles: the full-pipeline bootstrap with the paths frozen.
    design = rx.UnseenDesign(cal, te, held, folds)
    # Replicate draws depend only on the seed (design.draw), so R24's added rows leave R22's intervals unchanged.
    boot_names = [n for n in (HS, SAFE, SAFE_LSTM) + (R24_UNSEEN if r24 else ()) if n in unseen_fam]
    boot, _ = r22.boot_unseen(design, {n: (lambda F, n=n: f"{n}@{F}") for n in boot_names},
                              lambda n: unseen_val_of(n)("seed_average"), ("G_abs", "G"), rkP, B=args.r22_boot,
                              contrasts=((HS, SAFE),))
    # H2's nulls (the raw Hide-and-Seek scores of every row first).
    for path in list(args.amend2_scores) + list(args.r22_val_hs_scores):
        files = chs.load_scores(path)
        for r in cal + te + val:
            row = files.get(HS, {}).get(str(r["rollout_id"]))
            if row is not None:
                r.setdefault("raw", {})[HS] = {str(s): np.asarray(v, float) for s, v in row["per_seed_raw"].items()}
    norms_a2 = json.loads(Path(args.amend2_normalizations).read_text())
    null_out = r22.nulls(cal, te, val, rkP, bundle["task_normalizations"], norms_a2, HS,
                         {SAFE: (ensembles[SAFE]["matched"]["gain"], "safe_resampled"),
                          HS: (ensembles[HS]["matched"]["gain"], f"{HS}_resampled")}, B=args.r22_null_draws)
    budget = r22.budget_timer(cl.Cohort(val, "safe"), cl.Cohort(cal, "safe"), cl.Cohort(te, "safe"),
                              qt.dev_fractions(bundle), rkP, runs[HS])
    # Fig. 1: Hide-and-Seek + P's frozen path on the teaser rollout.
    teaser = None
    row = next((r for r in te if str(r["rollout_id"]) == R22_TEASER_ROLLOUT), None)
    if row is not None:
        chosen = runs[HS]["_chosen"][ALL]
        c = seen_cohort(seen[HS], cal, None)
        cu, rc = hk_prepared(c, rkP)
        tm, fu = r22.certify(cu, rc, chosen)
        score = np.mean([seen[HS][R22_TEASER_ROLLOUT][s] for s in SEEDS], axis=0)
        teaser = r22.teaser(row, score, rkP[row["task_name"]], row["horizon"], chosen, fu["lam"], tm["lam"])
    problems = [k.split("_", 1)[0] for k, v in checks.items()
                if not (v["ok"] if "ok" in v else all(x["ok"] for x in v.values()))]
    return dict(plan=r22.PLAN, checks=checks, validation=dict(counts={c: len(v) for c, v in ids.items()}),
                table2=table2, ensembles_seen=r22.strip(ensembles), per_type=per_types, unseen_bootstrap=r22.strip(boot),
                nulls=null_out, budget_timer=budget, teaser=teaser,
                inputs=dict(r14=dict(path=str(Path(args.r22_reference).resolve()), sha256=sha256(args.r22_reference)),
                            r8=dict(path=str(Path(args.r22_r8).resolve()), sha256=sha256(args.r22_r8)),
                            scores_dir=str(root.resolve()))), problems


def r25_xr1(args, fam, cal, te, held, folds, tasks, rkP):
    """R25 on XR-1 (r25_band_ablation): Hide-and-Seek's band against its first-step and zero-input
    retrainings on unseen tasks, with R21's and R21b's full-band results as gates G1 and G2."""
    import r21_native_bands as r21
    import r25_band_ablation as r25
    if not (args.r25_r21 and args.r25_r21b_replicates):
        raise SystemExit("error: --r25 needs --r25-r21 (seen_unseen_tables_r21.json) and --r25-r21b-replicates "
                         "(seen_unseen_tables_r21b_r21b_replicates.npz)")
    rows = cal + te
    data = {F: v for F, v in fam["xt"].items() if F != ALL}
    r0 = {}
    for h in r25.HEADS:
        c = trajectories(data, h, cal, ctl.CAL_COHORT, held, "pooled", running_max=False)
        e = trajectories(data, h, te, ctl.TEST_COHORT, held, "pooled", running_max=False)
        raw = {F: merge(c[F], e[F]) for F in folds}
        for r in rows:
            rid = str(r["rollout_id"])
            for F in folds:
                per = raw[F][rid]
                for s in SEEDS:
                    r[f"{h}~#{s}@{F}"] = per[s]
                r[f"{h}~@{F}"] = np.mean([per[s] for s in sorted(per)], axis=0)
        for F in folds:
            for s in SEEDS:
                r0[f"{h}/{F}/seed_{s}"] = r21.running_max_gap(rows, f"{h}~#{s}@{F}", f"{h}#{s}@{F}")
    every = set(tasks)
    d = r21.Design(cal, te, {F: every - held[F] for F in folds}, {F: held[F] for F in folds})
    key_ofs = {h: (lambda F, h=h: f"{h}~@{F}") for h in r25.HEADS}
    ref21 = json.loads(Path(args.r25_r21).read_text())["r21"]["bootstrap"]["unseen"]["timers"][r25.TIMER][r25.FULL]
    res, problems = r25.run(d, key_ofs, rkP, ref21=ref21, r21b=r25.r21b_replicates(args.r25_r21b_replicates), r0=r0,
                            B=args.r25_boot, progress=lambda m: print(m, flush=True))
    res["inputs"] = dict(r21=dict(path=str(Path(args.r25_r21).resolve()), sha256=sha256(args.r25_r21)),
                         r21b_replicates=dict(path=str(Path(args.r25_r21b_replicates).resolve()),
                                              sha256=sha256(args.r25_r21b_replicates)))
    return res, problems


def r26_xr1(args, cal, te, held, folds, seen, unseen, table1, rkP, extra):
    """R26 (r26_alarms_at_limit) on XR-1: Table I recounted with an alarm at a failure's final inference as a
    miss. Every row goes through Table I's own code path (thresholds, P's fit, E's hk_free fit), keeping each
    failure's hit and lead. Gates R0 (this run's table1), R1 (R14's table1 and timers_alone), R2 (R21's timers)."""
    import hk_free as hk
    import r21_native_bands as r21
    import r26_alarms_at_limit as r26
    if not (args.r26_r14 and args.r26_r21):
        raise SystemExit("error: --r26 needs --r26-r14 (seen_unseen_tables_r14.json) and --r26-r21 "
                         "(seen_unseen_tables_r21.json)")
    lab = lambda seed: "seed_average" if seed is None else f"seed_{seed}"

    def seen_side(traj, seeds):
        out = {}
        for seed in [None, *seeds]:
            o = outcomes(scored(te, traj, seed), psd.threshold(scored(cal, traj, seed)))
            out[lab(seed)] = r26.recount(o["hits"], o["leads"], o["fp"], o["n"])
        return r26.with_seed_summary(out)

    def unseen_side(traj_by_fold, seeds):
        out = {}
        for seed in [None, *seeds]:
            parts = []
            for F in folds:
                c = [r for r in cal if r["task_name"] not in held[F]]
                e = [r for r in te if r["task_name"] in held[F]]
                parts.append(outcomes(scored(e, traj_by_fold[F], seed), psd.threshold(scored(c, traj_by_fold[F], seed))))
            o = pool_outcomes(parts)
            out[lab(seed)] = r26.recount(o["hits"], o["leads"], o["fp"], o["n"])
        return r26.with_seed_summary(out)

    def timer_alone(parts, timer):
        """hk_free's timer alone (as r14.timers_alone), pooled over folds."""
        ps = []
        for F, (c, e) in sorted(parts.items()):
            cu, rc = hk.prepared(c, timer, F, None, rkP)
            eu, re_ = hk.prepared(e, timer, F, None, rkP)
            ps.append(hk.raw_eval(eu, re_, hk.fit(cu, rc)[0]))
        p = hk.pool(ps)
        return r26.recount(p["hit"], p["lead"], p["fp"], p["n"])

    rows = {}
    for name in OOD + (SAFE, HS) + tuple(n for n in extra if n == SAFE_LSTM):
        rows[name] = dict(seen=seen_side(seen[name], SEEDS), unseen=unseen_side(unseen[name], SEEDS))
    timer_traj = {str(r["rollout_id"]): {"det": r["timer"]} for r in cal + te}
    rows["global_timer"] = dict(seen=seen_side(timer_traj, []), unseen=unseen_side({F: timer_traj for F in folds}, []))
    calC, teC = cl.Cohort(cal, "timer"), cl.Cohort(te, "timer")
    tm, _ = cl.fit_fast(calC, rkP)
    p = ctl.raw_eval(teC, rkP, tm)
    rows["per_task_timer"] = dict(seen=dict(seed_average=r26.recount(p["hit"], p["lead"], p["fp"], p["n"])), unseen=None)
    light = lambda rs: cl.Cohort([r21.light(r) for r in rs], r21.LIGHT)
    e_seen = timer_alone({ALL: (light(cal), light(te))}, "G_abs")
    e_unseen = timer_alone({F: (light([r for r in cal if r["task_name"] not in held[F]]),
                                light([r for r in te if r["task_name"] in held[F]])) for F in folds}, "G_abs")
    rows["elapsed_timer"] = dict(seen=dict(seed_average=e_seen), unseen=dict(seed_average=e_unseen))
    ref21 = json.loads(Path(args.r26_r21).read_text())["r21"]
    heads = tuple(h for h in (SAFE, SAFE_LSTM, HS) if h in ref21["heads"])
    for h, block in r26.bands_from_r21(ref21, heads).items():
        rows[f"{h}_own_band"] = block

    # R0: this run's table1; R1: R14's table1 and timers_alone (E: no tpr there); R2: R21's timers alone.
    r0 = r26.compare({k: v for k, v in rows.items() if k in table1}, table1)
    ref14 = json.loads(Path(args.r26_r14).read_text())
    r1 = r26.compare({k: v for k, v in rows.items() if k in ref14["table1"]}, ref14["table1"])
    ta = ref14["r14"]["timers_alone"]
    for side, mine in (("seen", e_seen), ("unseen", e_unseen)):
        r1[f"elapsed_timer/{side}/seed_average"] = r26.worst_diff(mine, ta[side]["G_abs"], ("fpr", "mean_lead", "fp"))
    keys2 = ("fp", "fpr", "tpr", "mean_lead", "alarms_at_limit")
    r2 = {"per_task_timer/seen": r26.worst_diff(rows["per_task_timer"]["seen"]["seed_average"],
                                                ref21["heads"][HS]["seen"]["P"]["seed_average"]["timer_alone"], keys2),
          "elapsed_timer/unseen": r26.worst_diff(e_unseen,
                                                 ref21["heads"][HS]["unseen"]["G_abs"]["seed_average"]["timer_alone"], keys2)}
    checks = dict(R0_equals_this_runs_table1=r26.gate("R0", r0), R1_equals_r14=r26.gate("R1", r1),
                  R2_timers_equal_r21=r26.gate("R2", r2))
    res = dict(plan=r26.PLAN, rule="an alarm at a failure's final inference (lead 0) counts as a miss; successes, "
                                   "thresholds, FPR and mean lead unchanged",
               rows=rows, checks=checks, ok=all(v["ok"] for v in checks.values()),
               inputs=dict(r14=dict(path=str(Path(args.r26_r14).resolve()), sha256=sha256(args.r26_r14)),
                           r21=dict(path=str(Path(args.r26_r21).resolve()), sha256=sha256(args.r26_r21))))
    return res, [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]


def r27_xr1(args, fam, cal, te, held, folds, tasks, rkP):
    """R27 on XR-1 (r27_band_split): Hide-and-Seek's own band under the published 30/70 calibration split, with
    R21's and R25's 50/50 results as gates K0 to K2."""
    import r27_band_split as r27
    if not (args.r27_r18 and args.r27_r21 and args.r27_r25):
        raise SystemExit("error: --r27 needs --r27-r18, --r27-r21 and --r27-r25 (the XR-1 outputs of R18, R21, R25)")
    rows = cal + te
    data = {F: v for F, v in fam["xt"].items() if F != ALL}
    for h in r27.HEADS:
        c = trajectories(data, h, cal, ctl.CAL_COHORT, held, "pooled", running_max=False)
        e = trajectories(data, h, te, ctl.TEST_COHORT, held, "pooled", running_max=False)
        raw = {F: merge(c[F], e[F]) for F in folds}
        for r in rows:
            rid = str(r["rollout_id"])
            for F in folds:
                per = raw[F][rid]
                for s in SEEDS:
                    r[f"{h}~#{s}@{F}"] = per[s]
                r[f"{h}~@{F}"] = np.mean([per[s] for s in sorted(per)], axis=0)
    every = set(tasks)
    refs = {k: json.loads(Path(p).read_text())[k] for k, p in (("r18", args.r27_r18), ("r21", args.r27_r21),
                                                               ("r25", args.r27_r25))}
    res, problems = r27.run(cal, te, {F: every - held[F] for F in folds}, {F: held[F] for F in folds}, rkP,
                            refs["r18"], refs["r21"], refs["r25"], splits=args.r27_splits,
                            progress=lambda m: print(m, flush=True))
    res["inputs"] = {k: dict(path=str(Path(p).resolve()), sha256=sha256(p))
                     for k, p in (("r18", args.r27_r18), ("r21", args.r27_r21), ("r25", args.r27_r25))}
    return res, problems


def r27b_xr1(args, fam, cal, te, held, folds, tasks, rkP):
    """R27b on XR-1 (r27b_split_controls): random 50/50 controls, each assignment's curve and the example's alarms,
    with R27's 30/70 rows as gate M1."""
    import r27_band_split as r27
    import r27b_split_controls as r27b
    if not (args.r27b_r18 and args.r27b_r27):
        raise SystemExit("error: --r27b needs --r27b-r18 and --r27b-r27 (the XR-1 outputs of R18 and R27)")
    rows = cal + te
    data = {F: v for F, v in fam["xt"].items() if F != ALL}
    for h in r27.HEADS:
        c = trajectories(data, h, cal, ctl.CAL_COHORT, held, "pooled", running_max=False)
        e = trajectories(data, h, te, ctl.TEST_COHORT, held, "pooled", running_max=False)
        raw = {F: merge(c[F], e[F]) for F in folds}
        for r in rows:
            rid = str(r["rollout_id"])
            for F in folds:
                per = raw[F][rid]
                for s in SEEDS:
                    r[f"{h}~#{s}@{F}"] = per[s]
                r[f"{h}~@{F}"] = np.mean([per[s] for s in sorted(per)], axis=0)
    every = set(tasks)
    res, problems = r27b.run(cal, te, {F: every - held[F] for F in folds}, {F: held[F] for F in folds}, rkP,
                             json.loads(Path(args.r27b_r18).read_text())["r18"],
                             json.loads(Path(args.r27b_r27).read_text())["r27"], splits=args.r27b_splits,
                             progress=lambda m: print(m, flush=True))
    res["inputs"] = {k: dict(path=str(Path(p).resolve()), sha256=sha256(p)) for k, p in (("r18", args.r27b_r18),
                                                                                         ("r27", args.r27b_r27))}
    return res, problems


def r27c_xr1(args, fam, cal, te, held, folds, tasks, bundle, extra, rkP):
    """R27c on XR-1 (r27c_split_all_bands): SAFE-MLP's, SAFE-LSTM's and Hide-and-Seek's own bands on R27's 30/70
    assignments, against E and G (unseen) and P (seen), with R21's fixed-split points (Q0) and R27's rows (Q1). The
    scores are R21's (r21_xr1): seen per task, unseen pooled per fold, three-seed average of the normalized raw score."""
    import r27c_split_all_bands as r27c
    if not (args.r27c_r21 and args.r27c_r27):
        raise SystemExit("error: --r27c needs --r27c-r21 and --r27c-r27 (the XR-1 outputs of R21 and R27)")
    if SAFE_LSTM not in extra:
        raise SystemExit("error: --r27c needs --safe-lstm-fold-dirs (SAFE-LSTM's band)")
    rows = cal + te
    need = {str(r["rollout_id"]) for r in rows}
    heads = (SAFE, SAFE_LSTM, HS)
    a_all = {ALL: fam[SAFE_LSTM][ALL]}
    seen_raw = {SAFE: seen_safe_raw(rows, bundle),
                HS: seen_amend2_raw(args.amend2_scores, args.amend2_normalizations, HS, need),
                SAFE_LSTM: merge(
                    trajectories(a_all, SAFE_LSTM, cal, ctl.CAL_COHORT, held, "per_task", running_max=False)[ALL],
                    trajectories(a_all, SAFE_LSTM, te, ctl.TEST_COHORT, held, "per_task", running_max=False)[ALL])}

    def unseen_raw(family, det):
        data = {F: v for F, v in fam[family].items() if F != ALL}
        c = trajectories(data, det, cal, ctl.CAL_COHORT, held, "pooled", running_max=False)
        e = trajectories(data, det, te, ctl.TEST_COHORT, held, "pooled", running_max=False)
        return {F: merge(c[F], e[F]) for F in folds}

    raw_unseen = {SAFE: unseen_raw("safe", "safe"), HS: unseen_raw("xt", HS), SAFE_LSTM: unseen_raw(SAFE_LSTM, SAFE_LSTM)}
    for h in heads:
        for r in rows:
            rid = str(r["rollout_id"])
            per = seen_raw[h][rid]
            r[f"{h}~@seen"] = np.mean([per[s] for s in SEEDS], axis=0)
            for F in folds:
                per = raw_unseen[h][F][rid]
                r[f"{h}~@{F}"] = np.mean([per[s] for s in sorted(per)], axis=0)
    every = set(tasks)
    keys = {h: dict(unseen=lambda F, h=h: f"{h}~@{F}", seen=lambda F, h=h: f"{h}~@seen") for h in heads}
    res, problems = r27c.run(cal, te, {F: every - held[F] for F in folds}, {F: held[F] for F in folds}, rkP, keys,
                             json.loads(Path(args.r27c_r21).read_text())["r21"],
                             json.loads(Path(args.r27c_r27).read_text())["r27"], splits=args.r27c_splits,
                             progress=lambda m: print(m, flush=True))
    res["inputs"] = {k: dict(path=str(Path(p).resolve()), sha256=sha256(p)) for k, p in (("r21", args.r27c_r21),
                                                                                         ("r27", args.r27c_r27))}
    return res, problems


def r24_xr1(r22, ref_path):
    """R24 (docs/r24_table3_ci_seconds_plan.md) on XR-1: R22 rerun with Table III's missing rows added.
    Gate R0: every entry of R22's output is unchanged (the added rows only add keys). Gate R1: each added
    row's point estimate equals R22's Table II seed average for the same detector and path."""
    import r22_fixed_paths as r22m
    ref = json.loads(Path(ref_path).read_text())["r22"]
    new = json.loads(json.dumps(r22, default=float))
    checks = {}
    d, where = r22m.tree_diff({k: new.get(k) for k in R24_KEYS}, {k: ref.get(k) for k in R24_KEYS})
    checks["R0_r22_unchanged"] = dict(max_abs_diff=d, first_difference=where, ok=d <= TOL)
    seen, unseen = {}, {}
    for name in R24_SEEN:
        row = new["ensembles_seen"].get(name)
        ref_row = ((ref["table2"]["seen"].get(name) or {}).get("P") or {}).get("seed_average")
        checks[f"R1_seen_{name}"] = r22m.same_point(row, ref_row, TOL)
        seen[name] = row
    boot = new["unseen_bootstrap"]["timers"]["G_abs"]
    for name in R24_UNSEEN:
        b = boot.get(name)
        sa = ((ref["table2"]["unseen"].get(name) or {}).get("G_abs") or {}).get("seed_average")
        if b is None or sa is None:
            checks[f"R1_unseen_{name}"] = dict(max_abs_diff=math.inf, ok=False, note="row missing")
            continue
        dd = max(0.0 if b["fp"] == sa["fp"] else math.inf, abs(b["gain"] - sa["gain"]),
                 abs(b["matched_gain"] - sa["matched"]["gain"]) if b["matched_gain"] is not None and sa["matched"] else math.inf)
        checks[f"R1_unseen_{name}"] = dict(fp=b["fp"], ref_fp=sa["fp"], matched_gain=b["matched_gain"],
                                           ref_matched_gain=(sa["matched"] or {}).get("gain"),
                                           timer_fp=(sa["matched"] or {}).get("timer_fp"), max_abs_diff=dd, ok=dd <= TOL)
        unseen[name] = dict(b, matched_timer_fp=(sa["matched"] or {}).get("timer_fp"))
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return dict(plan="docs/r24_table3_ci_seconds_plan.md (R24)", checks=checks, seen=seen, unseen=unseen,
                inputs=dict(r22=dict(path=str(Path(ref_path).resolve()), sha256=sha256(ref_path)))), problems


def print_r24(r):
    print("R24 XR-1: Table III intervals for the rows R22 had without one")
    for name, row in r["seen"].items():
        m = (row or {}).get("matched") or {}
        print(f"  seen + P   {name:20s} FA {(row or {}).get('fp')} (matched P {m.get('timer_fp')})  "
              f"Delta l {m.get('gain', float('nan')):+.4f} 95% {m.get('ci95')}")
    for name, b in r["unseen"].items():
        print(f"  unseen + E {name:20s} FA {b['fp']} (matched E {b.get('matched_timer_fp')})  "
              f"Delta l {b['matched_gain']:+.4f} 95% {[round(x, 4) for x in b['matched_ci95']]}")
    for k, v in r["checks"].items():
        extra = f", first at {v['first_difference']}" if v.get("first_difference") else ""
        print(f"R24 {k}: {'OK' if v['ok'] else 'FAILED'} (max |diff| {v['max_abs_diff']:.3g}{extra})")


def r22_ids(path):
    """Rollout IDs from a JSON list, a JSON object with rollout_ids, or a JSONL file."""
    if str(path).endswith(".jsonl"):
        return [str(r["rollout_id"]) for r in load_jsonl(path)]
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        value = value.get("rollout_ids", value.get("evaluation_rollout_ids", value.get("test")))
    return [str(v) for v in value]


def hk_prepared(c, rk):
    import hk_free as hk
    return hk.prepared(c, "P", ALL, None, rk)


# ---- loading ----------------------------------------------------------------------------------

def check_fold_meta(label, F, meta, tasks, held, folds_sha, detectors):
    if str(meta["fold"]) != F or set(meta["held_out_tasks"]) != held[F]:
        raise ValueError(f"{label} fold {F}: fold or held-out tasks differ from the fold file")
    trained = set(meta["training_rollouts_by_task"])
    if trained != tasks - held[F]:
        raise ValueError(f"{label} fold {F}: training tasks are not exactly the seen tasks")
    if meta["folds_sha256"] != folds_sha:
        raise ValueError(f"{label} fold {F}: built with a different fold file")
    if [int(s) for s in meta["seeds"]] != [int(s) for s in SEEDS]:
        raise ValueError(f"{label} fold {F}: seeds {meta['seeds']} are not 0, 1, 2")
    missing = [d for d in detectors if d not in meta["detectors"]]
    if missing:
        raise ValueError(f"{label} fold {F}: detectors {missing} were not scored")


def load_family(label, paths, detectors, tasks, held, folds_sha, wanted, need_all=False):
    """Fold directories in the cross-task format -> {F: (meta, norms, scores)}."""
    data = {}
    for path in paths:
        meta, norms, scores = ctl.load_fold_dir(path, wanted)
        F = str(meta["fold"])
        if F in data:
            raise ValueError(f"{label}: two directories for fold {F}")
        check_fold_meta(label, F, meta, tasks, held, folds_sha, detectors)
        data[F] = (meta, norms, scores, str(Path(path).resolve()))
    want = set(held) if need_all else set(held) - {ALL}
    if set(data) != want:
        raise ValueError(f"{label}: need fold directories {sorted(want)}, got {sorted(data)}")
    return data


def trajectories(data, det, cohort_rows, cohort, held, norm, running_max=True):
    """{F: {rid: {seed: normalized trajectory}}} for one detector of one family (R19's native bands read
    the normalized raw score: running_max=False)."""
    out = {}
    for F, (_, norms, scores, _) in data.items():
        if det not in norms or det not in scores[cohort]:
            raise ValueError(f"fold {F}: no {det} scores for {cohort}")
        n = norms[det]
        out[F] = {}
        for r in cohort_rows:
            rid = str(r["rollout_id"])
            row = scores[cohort][det].get(rid)
            if row is None:
                raise ValueError(f"fold {F}: no {det} trajectory for {rid} ({cohort})")
            if row["failed"] != bool(r["failed"]) or row["task_name"] != r["task_name"]:
                raise ValueError(f"fold {F}: label or task differs for {rid} ({det})")
            if F != ALL and row["held_out"] != (r["task_name"] in held[F]):
                raise ValueError(f"fold {F}: held-out flag wrong for {rid} ({det})")
            if sorted(row["raw"]) != list(SEEDS):
                raise ValueError(f"fold {F}: {det} seeds of {rid} are {sorted(row['raw'])}")
            per = {}
            for s, raw in row["raw"].items():
                if len(raw) != r["L"] or not np.all(np.isfinite(raw)):
                    raise ValueError(f"fold {F}: {det} trajectory of {rid} has the wrong length or values")
                st = n[s]["pooled"] if norm == "pooled" else n[s]["per_task"][r["task_name"]]
                per[s] = ((np.maximum.accumulate(raw) if running_max else np.asarray(raw, float))
                          - st["location"]) / st["scale"]
            out[F][rid] = per
    return out


def merge(cal_traj, te_traj):
    return {**cal_traj, **te_traj}


def seen_safe(rows, bundle):
    """SAFE per seed from per_seed_raw_safe and the bundle's frozen task normalization."""
    norms = bundle["task_normalizations"]
    out = {}
    for r in rows:
        per = {s: (np.maximum.accumulate(np.asarray(raw, float)) - norms[s][r["task_name"]]["location"])
               / norms[s][r["task_name"]]["scale"] for s, raw in r["per_seed_raw_safe"].items()}
        if sorted(per) != list(SEEDS):
            raise ValueError(f"{r['rollout_id']}: SAFE seeds {sorted(per)}")
        avg = np.mean([per[s] for s in SEEDS], axis=0)
        if np.max(np.abs(avg - np.asarray(r["safe_scores"], float))) > 1e-9:
            raise ValueError(f"{r['rollout_id']}: per-seed SAFE mean differs from safe_scores")
        out[str(r["rollout_id"])] = per
    return out


def seen_amend2(paths, norms_path, name, need):
    norms = json.loads(Path(norms_path).read_text())
    out = {}
    for path in paths:
        for r in load_jsonl(path):
            if r.get("detector") != name or str(r["rollout_id"]) not in need:
                continue
            per = {}
            for s, raw in r["per_seed_raw"].items():
                st = norms[name][s][r["task_name"]]
                per[s] = (np.maximum.accumulate(np.asarray(raw, float)) - st["location"]) / st["scale"]
            if np.max(np.abs(np.mean([per[s] for s in sorted(per)], axis=0) - np.asarray(r["safe_scores"], float))) > 1e-9:
                raise ValueError(f"{name} {r['rollout_id']}: per-seed mean differs from its stored score")
            out[str(r["rollout_id"])] = per
    missing = need - set(out)
    if missing:
        raise ValueError(f"Amendment 2 {name} lacks {len(missing)} selected rollouts")
    return out


def seen_ood(ood_seen, need):
    """{name: {rid: {seed: trajectory}}} from the per-seed held-out files (already normalized)."""
    out = {name: {} for name in OOD}
    for seed, cal_path, test_path in ood_seen:
        for path in (cal_path, test_path):
            for r in load_jsonl(path):
                rid = str(r["rollout_id"])
                if rid not in need:
                    continue
                for name in OOD:
                    out[name].setdefault(rid, {})[str(seed)] = np.asarray(r["baseline_scores"][name], float)
    for name in OOD:
        bad = [rid for rid in need if sorted(out[name].get(rid, {})) != list(SEEDS)]
        if bad:
            raise ValueError(f"OOD {name}: {len(bad)} selected rollouts lack seeds 0-2")
    return out


# ---- run ----------------------------------------------------------------------------------------

def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    spec = json.loads(Path(args.folds).read_text())
    tasks, held = ctl.fold_sets(spec)
    folds = sorted(F for F in held if F != ALL)
    folds_sha = sha256(args.folds)
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    dev = qt.dev_fractions(bundle)
    rkP = qt.rho(dev, cl.PRIMARY_Q)
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    if {r["rollout_id"] for r in cal_all} & {r["rollout_id"] for r in te_all}:
        raise ValueError("Calibration and test cohorts overlap")
    if tasks != set(bundle["task_horizons"]) or {r["task_name"] for r in cal} != tasks or {r["task_name"] for r in te} != tasks:
        raise ValueError("Folds, bundle and cohorts do not cover the same tasks")
    wanted = {ctl.CAL_COHORT: {str(r["rollout_id"]) for r in cal}, ctl.TEST_COHORT: {str(r["rollout_id"]) for r in te}}
    need = wanted[ctl.CAL_COHORT] | wanted[ctl.TEST_COHORT]
    checks, failures = {}, []

    # T1: fold directories.
    fam = dict(
        safe=load_family("SAFE", args.safe_fold_dirs, ["safe"], tasks, held, folds_sha, wanted),
        xt=load_family("cross-task", args.xt_fold_dirs, list(HS_FAMILY), tasks, held, folds_sha, wanted),
        actprobe=load_family("ActProbe", args.actprobe_fold_dirs, [ACTPROBE], tasks, held, folds_sha, wanted,
                             need_all=True),
        ood=load_family("OOD", args.ood_fold_dirs, list(OOD), tasks, held, folds_sha, wanted),
    )
    # R11: optional families, each five fold directories (0-3 and all) of one detector.
    extra = []
    for name, paths in ((SAFE_LSTM, args.safe_lstm_fold_dirs), (ACTPROBE_T, args.actprobe_t_fold_dirs)):
        if paths:
            fam[name] = load_family(name, paths, [name], tasks, held, folds_sha, wanted, need_all=True)
            extra.append(name)
    ood_source_sha = sha256(args.ood_source_bundle)
    for F, (meta, _, _, _) in fam["ood"].items():
        if meta.get("variant") != "pooled" or (meta.get("fixed_selection") or {}).get("bundle_sha256") != ood_source_sha:
            raise ValueError(f"OOD fold {F}: not the pooled variant of the frozen Section 15.10 bundle (T5)")
    checks["T1_fold_dirs"] = {k: sorted(v) for k, v in fam.items()}
    checks["T5_ood_source_bundle_sha256"] = ood_source_sha

    # T4: ActProbe fold all reproduced B1.
    ameta, anorms = fam["actprobe"][ALL][0], fam["actprobe"][ALL][1]
    refs = {k: v for k, v in (ameta.get("reference_checks") or {}).items() if k.endswith(f"/{ACTPROBE}")}
    t4 = dict(reference_checks=refs, ok=bool(refs) and all(v.get("ok") for v in refs.values()))
    if args.actprobe_reference_normalizations:
        b1 = json.loads(Path(args.actprobe_reference_normalizations).read_text())[ACTPROBE]
        worst = max(abs(anorms[ACTPROBE][s]["per_task"][t][k] - b1[s][t][k]) / max(abs(b1[s][t][k]), 1e-12)
                    for s in SEEDS for t in tasks for k in ("location", "scale"))
        t4["normalization_max_relative_diff"] = worst
        t4["ok"] = t4["ok"] and worst <= TOL
    checks["T4_actprobe_fold_all"] = t4
    if not t4["ok"]:
        failures.append(f"T4: ActProbe fold all did not reproduce B1 ({t4})")

    # Seen trajectories (per-task normalization).
    seen = {SAFE: seen_safe(cal + te, bundle)}
    for name in (*HS_FAMILY, SHUFFLE):
        seen[name] = seen_amend2(args.amend2_scores, args.amend2_normalizations, name, need)
    def seen_fold_all(family, det):
        a_all = {ALL: fam[family][ALL]}
        return merge(trajectories(a_all, det, cal, ctl.CAL_COHORT, held, "per_task")[ALL],
                     trajectories(a_all, det, te, ctl.TEST_COHORT, held, "per_task")[ALL])

    seen[ACTPROBE] = seen_fold_all("actprobe", ACTPROBE)
    for name in extra:
        seen[name] = seen_fold_all(name, name)
    seen.update(seen_ood(args.ood_seen, need))

    # Unseen trajectories (pooled normalization), per fold.
    def unseen_traj(family, det):
        data = {F: v for F, v in fam[family].items() if F != ALL}
        c = trajectories(data, det, cal, ctl.CAL_COHORT, held, "pooled")
        e = trajectories(data, det, te, ctl.TEST_COHORT, held, "pooled")
        return {F: merge(c[F], e[F]) for F in folds}

    unseen = {SAFE: unseen_traj("safe", "safe"), ACTPROBE: unseen_traj("actprobe", ACTPROBE)}
    for name in extra:
        unseen[name] = unseen_traj(name, name)
    for name in HS_FAMILY:
        unseen[name] = unseen_traj("xt", name)
    for name in OOD:
        unseen[name] = unseen_traj("ood", name)

    # T2: seen + P per seed equals the stored runs.
    run_c = json.loads(Path(args.run_c).read_text())["detectors"]
    seen_fused = {}
    t2 = {}
    for name in (SAFE, *HS_FAMILY, SHUFFLE):
        res = psd.run_fused(seen[name], seen[name], cal, te, bundle)
        seen_fused[name] = res
        diff = max(abs(res[f"seed_{s}"][k] - run_c[name][f"seed_{s}"][k]) for s in SEEDS for k in FUSED_KEYS)
        t2[name] = diff
    ood_ref = json.loads(Path(args.ood_fused).read_text())["per_seed"]
    paths = {str(s): (c, t) for s, c, t in args.ood_seen}
    for s in SEEDS:
        got = opf.seed_metrics(*paths[s], rkP)
        for name in OOD:
            ref = ood_ref[s]["detectors"][name]
            row = got["detectors"][name]
            t2[name] = max(t2.get(name, 0.0), max(abs(row[k] - ref[k]) for k in ("fpr", "lead", "gain", "twa")))
            seen_fused.setdefault(name, {})[f"seed_{s}"] = dict(fpr=row["fpr"], mean_lead=row["lead"],
                                                               gain=row["gain"], twa=row["twa"])
    for name in OOD:
        seen_fused[name]["mean_std_over_seeds"] = mean_std([seen_fused[name][f"seed_{s}"] for s in SEEDS], FUSED_KEYS)
    checks["T2_seen_fused_max_abs_diff"] = t2
    for name in (ACTPROBE, *extra):          # R11 rows: no stored run to compare with
        seen_fused[name] = psd.run_fused(seen[name], seen[name], cal, te, bundle)
    bad = {k: v for k, v in t2.items() if not v <= TOL}
    if bad:
        failures.append(f"T2: seen + P differs from the stored per-seed runs: {bad}")

    # Unseen + G (Table II) with the cross-task design.
    keys = []
    for name, by_fold in unseen.items():
        for F in folds:
            for r in cal + te:
                per = by_fold[F][str(r["rollout_id"])]
                r[f"{name}@{F}"] = np.mean([per[s] for s in SEEDS], axis=0)
                for s in SEEDS:
                    r[f"{name}#{s}@{F}"] = per[s]
            keys += [f"{name}@{F}"] + [f"{name}#{s}@{F}" for s in SEEDS]
    D = ctl.Design(cal, te, held, keys, dev)
    unseen_fused = {name: fused_unseen(D, name, SEEDS) for name in OOD + (SAFE, ACTPROBE) + HS_FAMILY + tuple(extra)}

    # T3: seed-averaged unseen + G reproduces the cross-task runs.
    xt = json.loads(Path(args.cross_task_result).read_text())["detectors"]
    xs = json.loads(Path(args.cross_task_safe_result).read_text())
    t3 = {name: dict(got=round(unseen_fused[name]["seed_average"]["gain"], 4), want=xt[name]["gain"])
          for name in HS_FAMILY}
    t3[SAFE] = dict(got=round(unseen_fused[SAFE]["seed_average"]["gain"], 4), want=xs["safe_plus_G"]["gain"])
    for v in t3.values():
        v["ok"] = v["got"] == v["want"]
    checks["T3_unseen_seed_average"] = t3
    if not all(v["ok"] for v in t3.values()):
        failures.append(f"T3: seed-averaged unseen + G differs from the cross-task runs: {t3}")

    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in (
        ("calibration", args.calibration), ("test", args.test), ("runtime_bundle", args.runtime_bundle),
        ("folds", args.folds), ("run_c", args.run_c), ("ood_fused", args.ood_fused),
        ("cross_task_result", args.cross_task_result), ("cross_task_safe_result", args.cross_task_safe_result),
        ("amend2_normalizations", args.amend2_normalizations), ("ood_source_bundle", args.ood_source_bundle))}
    inputs["amend2_scores"] = [dict(path=str(Path(p).resolve()), sha256=sha256(p)) for p in args.amend2_scores]
    inputs["ood_seen"] = {str(s): [dict(path=str(Path(p).resolve()), sha256=sha256(p)) for p in (c, t)]
                          for s, c, t in args.ood_seen}
    inputs["fold_dirs"] = {k: {F: dict(path=v[3], code_commit=v[0].get("code_commit")) for F, v in sorted(d.items())}
                           for k, d in fam.items()}
    base = dict(plan=PLAN, status="exploratory, pre-specified (opened cohorts)", inputs=inputs,
                counts=dict(calibration=len(cal), test=len(te), tasks=len(tasks),
                            held_out_per_fold={F: len(held[F]) for F in folds}))
    if failures:
        out_path.write_text(json.dumps(dict(base, checks=checks, reproduction_failures=failures), indent=1,
                                       default=float) + "\n")
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(failures))
        print("No table is reported when a prerequisite fails.")
        return 1

    # Table I.
    table1 = {}
    for name in OOD + (SAFE, ACTPROBE, HS) + tuple(extra):
        table1[name] = dict(seen=standalone_seen(cal, te, seen[name], SEEDS),
                            unseen=standalone_unseen(cal, te, held, folds, unseen[name], SEEDS))
    timer_traj = {str(r["rollout_id"]): {"det": r["timer"]} for r in cal + te}
    risk_traj = {str(r["rollout_id"]): {"det": r["risk"]} for r in cal + te}
    table1["global_timer"] = dict(seen=standalone_seen(cal, te, timer_traj, []),
                                  unseen=standalone_unseen(cal, te, held, folds, {F: timer_traj for F in folds}, []))
    table1["time_risk"] = dict(seen=standalone_seen(cal, te, risk_traj, []), unseen=None)
    table1["per_task_timer"] = dict(seen=per_task_timer_seen(cal, te, rkP), unseen=None)

    # Table II.
    table2 = {name: dict(seen=seen_fused[name], unseen=unseen_fused.get(name)) for name in TABLE2 + tuple(extra)}
    g = unseen_fused[SAFE]["seed_average"]["timer"]
    table2["global_timer"] = dict(seen=None, unseen=dict(fpr=g["fpr"], mean_lead=g["mean_lead"], fp=g["fp"],
                                                         twa=(g["mean_lead"] + 1 - g["fpr"]) / 2))

    # T6 (R11): with --r6-result, every row R6 published is reproduced exactly (same inputs, same code
    # path), so adding rows changed nothing else.
    if args.r6_result:
        r6 = json.loads(Path(args.r6_result).read_text())
        worst, compared = 0.0, 0

        def walk(a, b):
            nonlocal worst, compared
            if isinstance(b, dict):
                for k, v in b.items():
                    if k in ("paths", "thresholds"):
                        continue
                    walk(a[k], v)
            elif isinstance(b, (int, float)) and not isinstance(b, bool):
                worst = max(worst, abs(float(a) - float(b)))
                compared += 1

        for tab, mine in (("table1", table1), ("table2", table2)):
            for name, row in r6[tab].items():
                walk(mine[name], row)
        t6 = dict(values=compared, max_abs_diff=worst, ok=compared > 0 and worst <= TOL)
        checks["T6_r6_rows_unchanged"] = t6
        if not t6["ok"]:
            out_path.write_text(json.dumps(dict(base, checks=checks, reproduction_failures=[f"T6: {t6}"]), indent=1,
                                           default=float) + "\n")
            print(f"REPRODUCTION FAILED:\n  T6: R6 rows changed ({t6})")
            return 1

    r14, r14_failures = None, []
    if args.r14_durations:
        r14, r14_failures = r14_xr1(args.r14_durations, D, cal, te, held, folds, tasks,
                                    OOD + (SAFE, ACTPROBE) + HS_FAMILY + tuple(extra), seen, seen_fused,
                                    unseen_fused, bundle, rkP, folds_sha)
    r18, r18_failures = None, []
    if args.curves:
        names = OOD + (SAFE,) + tuple(n for n in extra if n == SAFE_LSTM) + HS_FAMILY
        r18, r18_failures = r18_xr1(cal, te, held, folds, seen, unseen, table1, rkP, names)
    r19, r19_failures = None, []
    if args.r19:
        reference = json.loads(Path(args.r19_reference).read_text()) if args.r19_reference else None
        if reference is None or "r14" not in reference:
            raise SystemExit("error: --r19 needs --r19-reference, R14's output (it holds the r14 block)")

        def unseen_raw(family, det):
            data = {F: v for F, v in fam[family].items() if F != ALL}
            c = trajectories(data, det, cal, ctl.CAL_COHORT, held, "pooled", running_max=False)
            e = trajectories(data, det, te, ctl.TEST_COHORT, held, "pooled", running_max=False)
            return {F: merge(c[F], e[F]) for F in folds}

        raw_unseen = {SAFE: unseen_raw("safe", "safe"), HS: unseen_raw("xt", HS)}
        if SAFE_LSTM in extra:
            raw_unseen[SAFE_LSTM] = unseen_raw(SAFE_LSTM, SAFE_LSTM)
        btypes = bundle.get("task_types") or {}
        types = {r["task_name"]: (r.get("task_type") or btypes.get(r["task_name"])) for r in cal + te}
        seen_for = {n: seen[n] for n in (SAFE, HS, ZERO, SAFE_LSTM) if n in seen}
        r19, r19_failures = r19_xr1(cal, te, held, folds, raw_unseen, seen_for, rkP, types, reference, args.r19_boot)
    r20, r20_failures = None, []
    if args.r20:
        import r20_timer_g as r20m
        reference = json.loads(Path(args.r20_reference).read_text()) if args.r20_reference else None
        if reference is None or "timers_alone" not in (reference.get("r14") or {}):
            raise SystemExit("error: --r20 needs --r20-reference, R14's output (it holds r14.timers_alone)")
        r20, r20_failures = r20m.run(cal, te, held, folds, rkP, reference["r14"]["timers_alone"])
    r21, r21_failures, r21b, r21b_failures = None, [], None, []
    if args.r21b and not args.r21:
        raise SystemExit("error: --r21b runs after R21: give --r21 and its references too")
    if args.r21:
        r21, r21_failures, r21b, r21b_failures = r21_xr1(args, fam, cal, te, held, folds, tasks, bundle, extra, seen, rkP)
    r22, r22_failures = None, []
    if args.r22:
        btypes = bundle.get("task_types") or {}
        types = {r["task_name"]: (r.get("task_type") or btypes.get(r["task_name"])) for r in cal + te}
        r22, r22_failures = r22_xr1(args, D, cal, te, held, folds, tasks, bundle, extra, seen, rkP, types)
    r25, r25_failures = None, []
    if getattr(args, "r25", False):
        r25, r25_failures = r25_xr1(args, fam, cal, te, held, folds, tasks, rkP)
    r24, r24_failures = None, []
    if getattr(args, "r24", False):
        if not (args.r22 and args.r24_reference):
            raise SystemExit("error: --r24 reruns R22 with added rows: give --r22, its inputs and --r24-reference")
        r24, r24_failures = r24_xr1(r22, args.r24_reference)
    r27, r27_failures = None, []
    if getattr(args, "r27", False):
        r27, r27_failures = r27_xr1(args, fam, cal, te, held, folds, tasks, rkP)
    r27b, r27b_failures = None, []
    if getattr(args, "r27b", False):
        r27b, r27b_failures = r27b_xr1(args, fam, cal, te, held, folds, tasks, rkP)
    r27c, r27c_failures = None, []
    if getattr(args, "r27c", False):
        r27c, r27c_failures = r27c_xr1(args, fam, cal, te, held, folds, tasks, bundle, extra, rkP)
    r26, r26_failures = None, []
    if getattr(args, "r26", False):
        r26, r26_failures = r26_xr1(args, cal, te, held, folds, seen, unseen, table1, rkP, extra)
    out = dict(base, checks=checks, reproduction_failures=[], table1=table1, table2=table2,
               notes=dict(std="ddof 1 over training seeds 0-2; timers are deterministic",
                          unseen_normalization="pooled over the fold's seen tasks",
                          seen_normalization="per task"))
    if args.r14_durations:
        out["r14"] = r14
        out["r14_failures"] = r14_failures
    if args.r19:
        out["r19"] = r19
        out["r19_failures"] = r19_failures
    if args.r20:
        out["r20"] = r20
        out["r20_failures"] = r20_failures
    if args.r21:
        out["r21"] = r21
        out["r21_failures"] = r21_failures
    if args.r21b:
        out["r21b"] = r21b
        out["r21b_failures"] = r21b_failures
    if args.r22:
        out["r22"] = r22
        out["r22_failures"] = r22_failures
    if getattr(args, "r24", False):
        out["r24"] = r24
        out["r24_failures"] = r24_failures
    if getattr(args, "r25", False):
        out["r25"] = r25
        out["r25_failures"] = r25_failures
    if getattr(args, "r27", False):
        out["r27"] = r27
        out["r27_failures"] = r27_failures
    if getattr(args, "r27b", False):
        out["r27b"] = r27b
        out["r27b_failures"] = r27b_failures
    if getattr(args, "r27c", False):
        out["r27c"] = r27c
        out["r27c_failures"] = r27c_failures
    if getattr(args, "r26", False):
        out["r26"] = r26
        out["r26_failures"] = r26_failures
    if args.curves:
        out["r18"] = r18
        out["r18_failures"] = r18_failures
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")

    def cell(d, k):
        if d is None:
            return "   --        "
        m = d.get("mean_std_over_seeds")
        if m:
            return f"{m[k]['mean']:.3f}+-{m[k]['std']:.3f}"
        return f"{d['seed_average'][k]:.3f}      "

    print("Table I (bACC, TWA): seen | unseen")
    for name in TABLE1 + tuple(extra):
        s, u = table1[name]["seen"], table1[name]["unseen"]
        print(f"  {name:28s} {cell(s, 'bacc')} {cell(s, 'twa')} | {cell(u, 'bacc')} {cell(u, 'twa')}")
    print("Table II (FPR, gain, TWA): seen + P | unseen + G")
    for name in TABLE2 + tuple(extra):
        s, u = table2[name]["seen"], table2[name]["unseen"]
        print(f"  {name:28s} {cell(s, 'fpr')} {cell(s, 'gain')} {cell(s, 'twa')} | "
              f"{cell(u, 'fpr')} {cell(u, 'gain')} {cell(u, 'twa')}")
    gt = table2["global_timer"]["unseen"]
    print(f"  global timer, unseen: FPR {gt['fpr']:.3f} lead {gt['mean_lead']:.3f} TWA {gt['twa']:.3f}")
    print("SEEN/UNSEEN TABLES OK")
    if args.r14_durations:
        if r14_failures:
            print("R14 FAILED (no R14 result is used):\n  " + "\n  ".join(r14_failures))
            return 1
        print_r14(r14)
        print("R14 OK")
    if args.r19:
        if r19_failures:
            print("R19 FAILED (no R19 result is used): " + ", ".join(r19_failures) + f" {r19['checks']}")
            return 1
        import r19_extras as rx
        print(f"R19 B0: point estimates equal R14's (max |diff| {r19['checks']['B0_point_equals_r14']['max_abs_diff']}); "
              "N0: band timers equal hk_free.fit's")
        rx.print_summary(r19)
        print("R19 OK")
    if args.r20:
        import r20_timer_g as r20m
        r20m.print_summary(r20, "XR-1")
        if r20_failures:
            print("R20 FAILED (no R20 result is used): " + ", ".join(r20_failures))
            return 1
        print("R20 OK")
    if args.r21:
        import r21_native_bands as r21m
        r21m.print_summary(r21, "XR-1")
        if r21_failures:
            print("R21 FAILED (no R21 result is used): " + ", ".join(r21_failures))
            return 1
        print("R21 OK")
    if args.r21b:
        r21m.print_diagnostic(r21b, "XR-1")
        if r21b_failures:
            print("R21b FAILED (no R21b result is used): " + ", ".join(r21b_failures))
            return 1
        print("R21b OK")
    if args.r22:
        import r22_fixed_paths as r22m
        r22m.print_summary(r22, "XR-1")
        if r22_failures:
            print("R22 FAILED (no R22 result is used): " + ", ".join(r22_failures))
            return 1
        print("R22 OK")
    if getattr(args, "r25", False):
        import r25_band_ablation as r25m
        print()
        r25m.print_summary(r25, "XR-1")
        if r25_failures:
            print("R25 FAILED (no R25 result is used): " + ", ".join(r25_failures))
            return 1
        print("\nR25 OK")
    if getattr(args, "r27", False):
        import r27_band_split as r27m
        print()
        r27m.print_summary(r27, "XR-1")
        if r27_failures:
            print("R27 FAILED (no R27 result is used): " + ", ".join(r27_failures))
            return 1
        print("\nR27 OK")
    if getattr(args, "r27b", False):
        import r27b_split_controls as r27bm
        print()
        r27bm.print_summary(r27b, "XR-1")
        if r27b_failures:
            print("R27b FAILED (no R27b result is used): " + ", ".join(r27b_failures))
            return 1
        print("\nR27b OK")
    if getattr(args, "r27c", False):
        import r27c_split_all_bands as r27cm
        print()
        r27cm.print_summary(r27c, "XR-1")
        if r27c_failures:
            print("R27c FAILED (no R27c result is used): " + ", ".join(r27c_failures))
            return 1
        print("\nR27c OK")
    if getattr(args, "r26", False):
        import r26_alarms_at_limit as r26m
        print()
        r26m.print_summary(r26)
        if r26_failures:
            print("R26 FAILED (no R26 result is used): " + ", ".join(r26_failures))
            return 1
    if getattr(args, "r24", False):
        print()
        print_r24(r24)
        if r24_failures:
            print("R24 FAILED (no R24 result is used): " + ", ".join(r24_failures))
            return 1
        print("\nR24 OK")
    if args.curves:
        if r18_failures:
            print("R18 FAILED (no curve is used):\n  " + "\n  ".join(r18_failures))
            return 1
        print("R18 C1: every alpha = 0.05 point equals Table I")
        print_r18(r18)
        print("R18 CURVES OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--calibration", required=True)
    r.add_argument("--test", required=True)
    r.add_argument("--runtime-bundle", required=True)
    r.add_argument("--folds", default=str(ctl.FOLDS_FILE))
    r.add_argument("--safe-fold-dirs", nargs="+", required=True)
    r.add_argument("--xt-fold-dirs", nargs="+", required=True)
    r.add_argument("--actprobe-fold-dirs", nargs="+", required=True)
    r.add_argument("--ood-fold-dirs", nargs="+", required=True)
    r.add_argument("--ood-source-bundle", required=True)
    r.add_argument("--amend2-scores", nargs=2, required=True, metavar=("CALIBRATION", "TEST"))
    r.add_argument("--amend2-normalizations", required=True)
    r.add_argument("--ood-seen", nargs=3, action="append", required=True, metavar=("SEED", "CALIBRATION", "TEST"))
    r.add_argument("--run-c", required=True)
    r.add_argument("--ood-fused", required=True)
    r.add_argument("--cross-task-result", required=True)
    r.add_argument("--cross-task-safe-result", required=True)
    r.add_argument("--actprobe-reference-normalizations")
    r.add_argument("--safe-lstm-fold-dirs", nargs="+", help="R11: SAFE-LSTM fold directories 0-3 and all")
    r.add_argument("--actprobe-t-fold-dirs", nargs="+", help="R11: ActProbe-with-timestep fold directories 0-3 and all")
    r.add_argument("--r14-durations", help="R14: hk_free.py llm output for the XR-1 folds")
    r.add_argument("--curves", action="store_true", help="R18: calibrated lead-FPR curves (lead_fpr_curves.py)")
    r.add_argument("--r19", action="store_true", help="R19: the mock review's offline analyses (r19_extras.py)")
    r.add_argument("--r19-reference", help="R19 gate B0: R14's output (seen_unseen_tables_r14.json)")
    r.add_argument("--r19-boot", type=int, default=1000, help="R19 bootstrap replicates")
    r.add_argument("--r20", action="store_true", help="R20: timer G as a Table II row (r20_timer_g.py)")
    r.add_argument("--r20-reference", help="R20 gate T0: R14's output (seen_unseen_tables_r14.json)")
    r.add_argument("--r21", action="store_true", help="R21: the original alarm rules (r21_native_bands.py)")
    r.add_argument("--r21-reference", help="R21 gate N1: R19's output (seen_unseen_tables_r19.json)")
    r.add_argument("--r21-r12-reference", help="R21 gate N2: R12's list_b.json")
    r.add_argument("--r21-boot", type=int, default=1000, help="R21 bootstrap replicates")
    r.add_argument("--r21b", action="store_true", help="R21b: the band CI diagnostic, after R21")
    r.add_argument("--r21b-reference", help="R21b gate D0: R21's output (seen_unseen_tables_r21.json)")
    r.add_argument("--r21b-boot", type=int, default=1000, help="R21b replicates per variant")
    r.add_argument("--r22", action="store_true", help="R22: every fused path fixed on the validation set")
    r.add_argument("--r22-scores-dir", help="R22: stage A's root (xt, ood, safe, safe_lstm fold directories)")
    r.add_argument("--r22-val-calibration", help="R22: August calibration predictions ($XR1_PS_INPUTS)")
    r.add_argument("--r22-val-test", help="R22: August test (event) predictions ($XR1_PS_INPUTS)")
    r.add_argument("--r22-val-ids", nargs=2, metavar=("CALIBRATION_IDS", "TEST_IDS"), help="R22 gate V1: stage A's ID lists")
    r.add_argument("--r22-val-hs-scores", nargs=2, metavar=("CALIBRATION", "TEST"), help="R22: Amendment 2's August files")
    r.add_argument("--r22-ood-val", nargs=3, action="append", metavar=("SEED", "CALIBRATION", "TEST"),
                   help="R22: the per-seed OOD calibrate / evaluate predictions (August cohorts)")
    r.add_argument("--r22-reference", help="R22 gate K0: R14's output (seen_unseen_tables_r14.json)")
    r.add_argument("--r22-r8", help="R22 gate K8: R8's fixed_path.json")
    r.add_argument("--r22-boot", type=int, default=1000, help="R22 unseen bootstrap replicates")
    r.add_argument("--r22-null-draws", type=int, default=100, help="R22 null fusions per family")
    r.add_argument("--r24", action="store_true", help="R24: R22 with Table III's missing intervals added")
    r.add_argument("--r24-reference", help="R24 gate R0: R22's XR-1 output (seen_unseen_tables_r22.json)")
    r.add_argument("--r25", action="store_true", help="R25: Hide-and-Seek's band against its input ablations")
    r.add_argument("--r25-r21", help="R25 gate G1: R21's XR-1 output (seen_unseen_tables_r21.json)")
    r.add_argument("--r25-r21b-replicates", help="R25 gate G2: R21b's replicates (seen_unseen_tables_r21b_r21b_replicates.npz)")
    r.add_argument("--r25-boot", type=int, default=1000, help="R25 bootstrap replicates (R21b's: 1000)")
    r.add_argument("--r27b", action="store_true", help="R27b: random 50/50 controls, curves and the example's alarms")
    r.add_argument("--r27b-r18", help="R27b: R18's XR-1 output (E's curve)")
    r.add_argument("--r27b-r27", help="R27b gate M1: R27's XR-1 output")
    r.add_argument("--r27b-splits", type=int, default=20, help="R27b random assignments of each kind")
    r.add_argument("--r27c", action="store_true", help="R27c: every own band on R27's 30/70 assignments vs E, G and P")
    r.add_argument("--r27c-r21", help="R27c gate Q0: R21's XR-1 output")
    r.add_argument("--r27c-r27", help="R27c gate Q1: R27's XR-1 output")
    r.add_argument("--r27c-splits", type=int, default=20, help="R27c random 30/70 assignments (R27's seeds)")
    r.add_argument("--r27", action="store_true", help="R27: Hide-and-Seek's band under the published 30/70 split")
    r.add_argument("--r27-r18", help="R27: R18's XR-1 output (E's curve)")
    r.add_argument("--r27-r21", help="R27 gates K0, K1: R21's XR-1 output")
    r.add_argument("--r27-r25", help="R27 gate K2: R25's XR-1 output")
    r.add_argument("--r27-splits", type=int, default=20, help="R27 random 30/70 assignments")
    r.add_argument("--r26", action="store_true", help="R26: Table I with an alarm at the limit counted as a miss")
    r.add_argument("--r26-r14", help="R26 gate R1: R14's output (seen_unseen_tables_r14.json), Table I's source")
    r.add_argument("--r26-r21", help="R26: R21's XR-1 output (own bands; gate R2)")
    r.add_argument("--r6-result", help="R11 gate T6: R6's seen_unseen_tables JSON, reproduced exactly")
    r.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
