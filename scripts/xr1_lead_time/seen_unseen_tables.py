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

PLAN = "docs/xr1_review_analyses_plan.md (R6)"
ALL = ctl.ALL
SEEDS = ("0", "1", "2")
SAFE = "safe_mlp"
HS, ZERO, FIRST, SHUFFLE = "hide_and_seek", "hide_and_seek_zero", "hide_and_seek_first", "hide_and_seek_task_shuffle"
HS_FAMILY = (HS, ZERO, FIRST)
ACTPROBE = "actprobe"
OOD = ("mahalanobis", "knn_cosine", "pca_kmeans", "rnd")
TIMERS = ("time_risk", "global_timer", "per_task_timer")
TABLE1 = OOD + (SAFE, ACTPROBE, HS) + TIMERS
TABLE2 = OOD + (SAFE,) + HS_FAMILY + (SHUFFLE,)
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


def trajectories(data, det, cohort_rows, cohort, held, norm):
    """{F: {rid: {seed: normalized trajectory}}} for one detector of one family."""
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
                per[s] = (np.maximum.accumulate(raw) - st["location"]) / st["scale"]
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
    a_all = {ALL: fam["actprobe"][ALL]}
    seen[ACTPROBE] = merge(trajectories(a_all, ACTPROBE, cal, ctl.CAL_COHORT, held, "per_task")[ALL],
                           trajectories(a_all, ACTPROBE, te, ctl.TEST_COHORT, held, "per_task")[ALL])
    seen.update(seen_ood(args.ood_seen, need))

    # Unseen trajectories (pooled normalization), per fold.
    def unseen_traj(family, det):
        data = {F: v for F, v in fam[family].items() if F != ALL}
        c = trajectories(data, det, cal, ctl.CAL_COHORT, held, "pooled")
        e = trajectories(data, det, te, ctl.TEST_COHORT, held, "pooled")
        return {F: merge(c[F], e[F]) for F in folds}

    unseen = {SAFE: unseen_traj("safe", "safe"), ACTPROBE: unseen_traj("actprobe", ACTPROBE)}
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
    bad = {k: v for k, v in t2.items() if not v <= TOL}
    if bad:
        failures.append(f"T2: seen + P differs from the stored per-seed runs: {bad}")

    # Unseen + G (Table II) with the cross-task design.
    keys = []
    for name, by_fold in unseen.items():
        if name == ACTPROBE:
            continue
        for F in folds:
            for r in cal + te:
                per = by_fold[F][str(r["rollout_id"])]
                r[f"{name}@{F}"] = np.mean([per[s] for s in SEEDS], axis=0)
                for s in SEEDS:
                    r[f"{name}#{s}@{F}"] = per[s]
            keys += [f"{name}@{F}"] + [f"{name}#{s}@{F}" for s in SEEDS]
    D = ctl.Design(cal, te, held, keys, dev)
    unseen_fused = {name: fused_unseen(D, name, SEEDS) for name in OOD + (SAFE,) + HS_FAMILY}

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
    for name in OOD + (SAFE, ACTPROBE, HS):
        table1[name] = dict(seen=standalone_seen(cal, te, seen[name], SEEDS),
                            unseen=standalone_unseen(cal, te, held, folds, unseen[name], SEEDS))
    timer_traj = {str(r["rollout_id"]): {"det": r["timer"]} for r in cal + te}
    risk_traj = {str(r["rollout_id"]): {"det": r["risk"]} for r in cal + te}
    table1["global_timer"] = dict(seen=standalone_seen(cal, te, timer_traj, []),
                                  unseen=standalone_unseen(cal, te, held, folds, {F: timer_traj for F in folds}, []))
    table1["time_risk"] = dict(seen=standalone_seen(cal, te, risk_traj, []), unseen=None)
    table1["per_task_timer"] = dict(seen=per_task_timer_seen(cal, te, rkP), unseen=None)

    # Table II.
    table2 = {name: dict(seen=seen_fused[name], unseen=unseen_fused.get(name)) for name in TABLE2}
    g = unseen_fused[SAFE]["seed_average"]["timer"]
    table2["global_timer"] = dict(seen=None, unseen=dict(fpr=g["fpr"], mean_lead=g["mean_lead"], fp=g["fp"],
                                                         twa=(g["mean_lead"] + 1 - g["fpr"]) / 2))

    out = dict(base, checks=checks, reproduction_failures=[], table1=table1, table2=table2,
               notes=dict(std="ddof 1 over training seeds 0-2; timers are deterministic",
                          unseen_normalization="pooled over the fold's seen tasks",
                          seen_normalization="per task"))
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")

    def cell(d, k):
        if d is None:
            return "   --        "
        m = d.get("mean_std_over_seeds")
        if m:
            return f"{m[k]['mean']:.3f}+-{m[k]['std']:.3f}"
        return f"{d['seed_average'][k]:.3f}      "

    print("Table I (bACC, TWA): seen | unseen")
    for name in TABLE1:
        s, u = table1[name]["seen"], table1[name]["unseen"]
        print(f"  {name:28s} {cell(s, 'bacc')} {cell(s, 'twa')} | {cell(u, 'bacc')} {cell(u, 'twa')}")
    print("Table II (FPR, gain, TWA): seen + P | unseen + G")
    for name in TABLE2:
        s, u = table2[name]["seen"], table2[name]["unseen"]
        print(f"  {name:28s} {cell(s, 'fpr')} {cell(s, 'gain')} {cell(s, 'twa')} | "
              f"{cell(u, 'fpr')} {cell(u, 'gain')} {cell(u, 'twa')}")
    gt = table2["global_timer"]["unseen"]
    print(f"  global timer, unseen: FPR {gt['fpr']:.3f} lead {gt['mean_lead']:.3f} TWA {gt['twa']:.3f}")
    print("SEEN/UNSEEN TABLES OK")
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
    r.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
