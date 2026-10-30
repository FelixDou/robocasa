"""RLDX-1 seen and unseen columns of RSS Table II (R7 of docs/xr1_review_analyses_plan.md). Exploratory.

R6 gave XR-1's Table II an unseen-task column group (score + G per cross-task fold). This script
does the same for RLDX-1's offline cohorts (report sections 15.15 and 15.22): 20 tasks, 740
training rollouts, 60 held-out calibration successes (3 per task), 200 test rollouts (5 + 5 per
task). The rows, splits, H_k (longest training failure), P (q = 0.95) and fit
(confirm_lead.fit_fast) are those of rldx_replication.py; on RLDX-1 the held-out calibration
successes set the FPR and the training failures choose the fused path.

Folds. ``folds`` deals the 20 tasks into 4 folds, stratified by task type, with
cross_task_lead.make_folds and seed FOLD_SEED. robocasa.recovery.safe.baselines.score_cross_task
(--score-train) retrains every detector without each fold's tasks, as score_cohorts trains it.

Table II, score OR-fused with a timer, per training seed and for the seed average:
  seen    fold all, per-task normalization, + P on every task. These are the published
          per-seed rows (rldx_per_seed.py); gate R2 requires them to be reproduced exactly.
  unseen  per fold: G (P with rho_k = 0) and the fused path fitted on the SEEN tasks'
          calibration successes and (in-sample) training failures, with the fold's pooled
          normalization, applied to the HELD-OUT tasks' test rollouts; pooled over folds.
          Every task is held out once, so the pooled test set is all 200 test rollouts.
  Reported: FPR, false alarms, mean lead, gain over the timer, TWA = (lead + 1 - FPR) / 2.

Families. Each --fold-dirs gives one family (folds 0-3 and all) with its own detectors:
  score_cross_task --score-train   the seven cross-task detectors (DETECTORS; required)
  cross_task_safe assemble-export  one SAFE head each (safe_lstm, safe_indep; Stage B')

Gates (no table without them):
  R1  fold directories: the fold file's folds and held-out tasks, training on exactly the seen
      tasks, seeds 0-2, --score-train, every needed rollout present with the right task, label
      and length; fold all reproduced the earlier trajectories of every detector of its family
      in all three cohorts (score_cross_task --reference-raw; for SAFE, the frozen
      normalized_scores.jsonl), and every other reference check it recorded passed; every fold
      directory's own recorded checks passed.
  R2  seen + P, per seed and seed-averaged, equals rldx_per_seed.json (P, lead, FPR, gain) at
      its stored precision, for every detector.
  R3  G is the same timer whichever score it is fused with (it does not read the score).

Usage:
  python3 -I rldx_unseen.py folds --inputs-dir RLDXHS/inputs --write rldx_folds.json
  python3 -I rldx_unseen.py run --safe-dir <SAFE20>/calibration/lstm --inputs-dir RLDXHS/inputs \\
      --folds rldx_folds.json --fold-dirs F0 F1 F2 F3 FALL [--fold-dirs S0 S1 S2 S3 SALL ...] \\
      --reference-per-seed rldx_per_seed.json --output OUT.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl  # noqa: E402
import cross_task_lead as ctl  # noqa: E402
import quantile_timer as qt  # noqa: E402
import rldx_hide_and_seek as hs  # noqa: E402
import rldx_replication as rr  # noqa: E402

PLAN = "docs/xr1_review_analyses_plan.md (R7)"
FOLD_SEED, K = 20261029, 4
ALL = "all"
SEEDS = ("0", "1", "2")
COHORTS = ("calibration", "test", "train")
DETECTORS = ("hide_and_seek", "hide_and_seek_zero", "hide_and_seek_first",
             "mahalanobis", "cosine_knn", "pca_kmeans", "rnd")
KEYS = ("fpr", "mean_lead", "gain", "twa")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_std(per_seed, keys=KEYS):
    return {k: dict(mean=float(np.mean([p[k] for p in per_seed])), std=float(np.std([p[k] for p in per_seed], ddof=1)))
            for k in keys}


# ---- folds ----------------------------------------------------------------------------------

def folds_cmd(args):
    bundle = json.loads((Path(args.inputs_dir) / "rldx_bundle.json").read_text())
    spec = ctl.make_folds(bundle["task_types"], k=K, seed=FOLD_SEED)
    spec["plan"] = PLAN
    if any(not f["held_out"] for f in spec["folds"]):
        raise ValueError("A fold holds out no task")
    out = Path(args.write)
    if out.exists():
        raise SystemExit(f"error: output exists: {out}")
    out.write_text(json.dumps(spec, indent=1) + "\n")
    for f in spec["folds"]:
        types = [spec["tasks"][t] for t in f["held_out"]]
        print(f"fold {f['fold']}: {types.count('atomic')} atomic + {types.count('composite')} composite: "
              + ", ".join(f["held_out"]))
    print(f"wrote {out} (sha256 {sha256(out)})")
    return 0


# ---- rows and fold scores --------------------------------------------------------------------

def load_rows(safe_dir, inputs_dir):
    """The 15.15 rows (rldx_hide_and_seek.load_rows) and the prepared ID lists, checked."""
    rows, H, tasks = hs.load_rows(safe_dir)
    inputs = Path(inputs_dir)
    bundle = json.loads((inputs / "rldx_bundle.json").read_text())
    if bundle["task_horizons"] != {t: int(H[t]) for t in tasks}:
        raise ValueError("The scoring bundle's horizons differ from H_k recomputed now")
    parts = {"train": [r for r in rows if r["split"] == "train"],
             "calibration": [r for r in rows if r["split"] == "calibration"],
             "test": [r for r in rows if r["split"] == "evaluation"]}
    for name, part in parts.items():
        if sorted(r["rollout_id"] for r in part) != json.loads((inputs / f"{name}_ids.json").read_text()):
            raise ValueError(f"{name} IDs differ from the prepared ID list")
    if any(r["failed"] for r in parts["calibration"]):
        raise ValueError("Calibration split contains failures")
    return parts, H, tasks, bundle


def load_fold_dir(path, wanted):
    """scoring.json, normalizations.json and the per-seed raw trajectories of wanted rollouts."""
    path = Path(path)
    meta = json.loads((path / "scoring.json").read_text())
    norms = json.loads((path / "normalizations.json").read_text())
    scores = {}
    for cohort in COHORTS:
        out = {}
        with open(path / f"{cohort}_predictions.jsonl") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = str(row["rollout_id"])
                if rid not in wanted[cohort]:
                    continue
                d = out.setdefault(row["detector"], {})
                if rid in d:
                    raise ValueError(f"{path}: duplicate {row['detector']} row for {rid} ({cohort})")
                d[rid] = dict(task_name=row["task_name"], failed=bool(row["failed"]), held_out=bool(row["held_out"]),
                              raw={s: np.asarray(v, float) for s, v in row["per_seed_raw"].items()})
        scores[cohort] = out
    return meta, norms, scores


def check_meta(F, meta, tasks, held, folds_sha, detectors):
    if str(meta["fold"]) != F or set(meta["held_out_tasks"]) != held[F]:
        raise ValueError(f"Fold {F}: fold or held-out tasks differ from the fold file")
    if set(meta["training_rollouts_by_task"]) != set(tasks) - held[F]:
        raise ValueError(f"Fold {F}: training tasks are not exactly the seen tasks")
    if meta["folds_sha256"] != folds_sha:
        raise ValueError(f"Fold {F}: scored with a different fold file")
    if [str(s) for s in meta["seeds"]] != list(SEEDS):
        raise ValueError(f"Fold {F}: seeds {meta['seeds']} are not 0, 1, 2")
    if not meta.get("score_train"):
        raise ValueError(f"Fold {F}: scored without --score-train")
    failed = sorted(k for k, v in (meta.get("checks") or {}).items() if isinstance(v, dict) and not v.get("ok", True))
    if failed:
        raise ValueError(f"Fold {F}: the directory recorded failed checks {failed}")
    missing = [d for d in detectors if d not in meta["detectors"]]
    if missing:
        raise ValueError(f"Fold {F}: detectors {missing} were not scored")
    if F != ALL:
        return None
    checks = meta.get("reference_checks") or {}
    need = {f"{c}/{d}" for c in COHORTS for d in detectors}
    bad = sorted(k for k in need | set(checks) if not checks.get(k, {}).get("ok"))
    if bad:
        raise ValueError(f"Fold all did not reproduce the earlier trajectories for {bad}")
    return {k: checks[k]["max_abs_diff"] for k in sorted(checks)}


def normalized(raw, stats):
    """Normalized running maximum of one seed's raw trajectory."""
    return (np.maximum.accumulate(raw) - stats["location"]) / stats["scale"]


def attach(parts, F, norms, scores, held, detectors):
    """Keys d#s@F (one seed) and d@F (seed average); fold all also gets d#s@all/task and d@all/task.

    Fold F needs its scores on every calibration success and test rollout, and on its own
    (seen-task, in-sample) training failures.
    """
    for cohort, rows in parts.items():
        for r in rows:
            if cohort == "train" and r["task_name"] in held[F]:
                continue
            rid = str(r["rollout_id"])
            for d in detectors:
                row = scores[cohort].get(d, {}).get(rid)
                if row is None:
                    raise ValueError(f"Fold {F}: no {d} trajectory for {rid} ({cohort})")
                if row["failed"] != r["failed"] or row["task_name"] != r["task_name"]:
                    raise ValueError(f"Fold {F}: label or task differs for {rid} ({d})")
                if row["held_out"] != (r["task_name"] in held[F]):
                    raise ValueError(f"Fold {F}: held-out flag wrong for {rid} ({d})")
                if sorted(row["raw"]) != list(SEEDS) or sorted(norms[d]) != list(SEEDS):
                    raise ValueError(f"Fold {F}: seeds of {rid} ({d}) are not 0, 1, 2")
                if any(len(v) != r["L"] or not np.all(np.isfinite(v)) for v in row["raw"].values()):
                    raise ValueError(f"Fold {F}: {d} trajectory of {rid} has the wrong length or non-finite values")
                per = {s: normalized(row["raw"][s], norms[d][s]["pooled"]) for s in SEEDS}
                for s in SEEDS:
                    r[f"{d}#{s}@{F}"] = per[s]
                r[f"{d}@{F}"] = np.mean([per[s] for s in SEEDS], axis=0)
                if F == ALL:
                    task = {s: normalized(row["raw"][s], norms[d][s]["per_task"][r["task_name"]]) for s in SEEDS}
                    for s in SEEDS:
                        r[f"{d}#{s}@all/task"] = task[s]
                    r[f"{d}@all/task"] = np.mean([task[s] for s in SEEDS], axis=0)


# ---- fits ------------------------------------------------------------------------------------

def seen_plus_P(parts, key, rk):
    """rldx_per_seed.fused on fold all: + P on every task, P alone as the reference."""
    cal = parts["calibration"] + [r for r in parts["train"] if r["failed"]]
    test = parts["test"]
    tm, fu = cl.fit_fast(cl.Cohort(cal, key), rk)
    eP, lP = cl.evaluate_fast(cl.Cohort(test, key), rk, tm)
    e, lF = cl.evaluate_fast(cl.Cohort(test, key), rk, fu)
    fpr, lead = float(e["fpr"]), float(e["mean_lead"])
    return dict(fpr=fpr, mean_lead=lead, gain=round(float(np.mean(lF - lP)), 4), twa=(lead + 1.0 - fpr) / 2.0,
                fp=e["fp"], P=dict(lead=float(eP["mean_lead"]), fpr=float(eP["fpr"])), path=cl.det_json(fu))


def unseen_plus_G(parts, held, folds, key_of, rk0):
    """Per fold: fit on seen-task calibration successes and training failures, test on held-out tasks."""
    pt, pf, paths = [], [], {}
    for F in folds:
        cal = [r for r in parts["calibration"] if r["task_name"] not in held[F]] + \
              [r for r in parts["train"] if r["failed"] and r["task_name"] not in held[F]]
        test = [r for r in parts["test"] if r["task_name"] in held[F]]
        key = key_of(F)
        tm, fu = cl.fit_fast(cl.Cohort(cal, key), rk0)
        e = cl.Cohort(test, key)
        pt.append(ctl.raw_eval(e, rk0, tm))
        pf.append(ctl.raw_eval(e, rk0, fu))
        paths[F] = dict(timer=cl.det_json(tm), fused=cl.det_json(fu), cal_successes=sum(not r["failed"] for r in cal),
                        cal_failures=sum(r["failed"] for r in cal), test=len(test))
    tG, fD = ctl.pool(pt), ctl.pool(pf)
    fpr, lead = fD["fp"] / fD["n"], float(fD["lead"].mean())
    gfpr, glead = tG["fp"] / tG["n"], float(tG["lead"].mean())
    return dict(fpr=fpr, mean_lead=lead, gain=ctl.gain(fD, tG), twa=(lead + 1.0 - fpr) / 2.0, fp=fD["fp"],
                n_success=fD["n"], n_failure=int(len(fD["lead"])),
                timer=dict(fpr=gfpr, mean_lead=glead, fp=tG["fp"], twa=(glead + 1.0 - gfpr) / 2.0), paths=paths)


# ---- run -------------------------------------------------------------------------------------

def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    parts, H, tasks, bundle = load_rows(args.safe_dir, args.inputs_dir)
    spec = json.loads(Path(args.folds).read_text())
    if spec.get("seed") != FOLD_SEED or spec.get("k") != K or spec.get("tasks") != bundle["task_types"]:
        raise ValueError("The fold file is not R7's (seed, k or task types differ)")
    fold_tasks, held = ctl.fold_sets(spec)
    if fold_tasks != set(tasks):
        raise ValueError("The fold file's tasks differ from the RLDX-1 tasks")
    folds = sorted(F for F in held if F != ALL)
    if any(not held[F] for F in folds):
        raise ValueError("A fold holds out no task")
    folds_sha = sha256(args.folds)
    wanted = {c: {str(r["rollout_id"]) for r in parts[c]} for c in COHORTS}

    ref = json.loads(Path(args.reference_per_seed).read_text())
    detectors, families = [], {}
    for family in args.fold_dirs:
        first = json.loads((Path(family[0]) / "scoring.json").read_text())["detectors"]
        name = "+".join(first)
        if set(first) & set(detectors):
            raise ValueError(f"Two families score {sorted(set(first) & set(detectors))}")
        detectors += list(first)
        metas, fold_all_refs = {}, None
        for path in family:
            meta, norms, scores = load_fold_dir(path, wanted)
            F = str(meta["fold"])
            if F in metas:
                raise ValueError(f"{name}: two directories for fold {F}")
            if list(meta["detectors"]) != list(first):
                raise ValueError(f"{name}: fold {F} scored other detectors ({meta['detectors']})")
            refs = check_meta(F, meta, tasks, held, folds_sha, first)
            if F == ALL:
                fold_all_refs = refs
            attach(parts, F, norms, scores, held, first)
            metas[F] = dict(path=str(Path(path).resolve()), code_commit=meta.get("code_commit"),
                            training_rollouts=sum(meta["training_rollouts_by_task"].values()))
        if set(metas) != set(held):
            raise ValueError(f"{name}: need fold directories {sorted(held)}, got {sorted(metas)}")
        families[name] = dict(fold_dirs=metas, fold_all_reference=fold_all_refs)
    missing = [d for d in DETECTORS if d not in detectors]
    if missing:
        raise ValueError(f"No family scores {missing}")
    detectors = list(DETECTORS) + [d for d in detectors if d not in DETECTORS]
    unreferenced = set(args.unreferenced or [])
    if unreferenced - set(detectors):
        raise ValueError(f"--unreferenced names detectors no family scores: {sorted(unreferenced - set(detectors))}")
    unknown = [d for d in detectors if d not in ref["results"] and d not in unreferenced]
    if unknown:
        raise ValueError(f"rldx_per_seed.json has no result for {unknown} (R11: list new rows in --unreferenced)")
    checks = dict(R1=dict(folds={F: sorted(held[F]) for F in folds}, families=families))

    train = parts["train"]
    rk = qt.rho(rr.dev_fractions(train, tasks), rr.Q)
    rk0 = {t: 0.0 for t in tasks}

    # R2: seen + P reproduces the published per-seed rows.
    failures, worst, seen = [], [], {}
    stored_P = (ref["P"]["lead"], ref["P"]["fpr"])
    for d in detectors:
        out = {"seed_average": seen_plus_P(parts, f"{d}@all/task", rk)}
        for s in SEEDS:
            out[f"seed_{s}"] = seen_plus_P(parts, f"{d}#{s}@all/task", rk)
        out["mean_std_over_seeds"] = mean_std([out[f"seed_{s}"] for s in SEEDS])
        seen[d] = out
        if d in unreferenced:            # R11 rows (e.g. ActProbe): no published row to reproduce
            if (out["seed_average"]["P"]["lead"], out["seed_average"]["P"]["fpr"]) != stored_P:
                failures.append(f"R2: {d}: P {out['seed_average']['P']} != stored {stored_P}")
            continue
        stored = ref["results"][d]
        pairs = [("seed_average", stored["seed_average"])] + [(f"seed_{s}", stored["per_seed"][s]) for s in SEEDS]
        for label, want in pairs:
            got = out[label]
            now = (got["mean_lead"], got["fpr"], got["gain"])
            exp = (want["lead"], want["fpr"], want["gain"])
            if now != exp or (got["P"]["lead"], got["P"]["fpr"]) != stored_P:
                failures.append(f"R2: {d} {label}: + P {now} (P {got['P']}) != stored {exp} (P {stored_P})")
            worst.append(max(abs(a - b) for a, b in zip(now, exp)))
    checks["R2_seen_plus_P"] = dict(stored_P=stored_P, max_abs_diff=max(worst) if worst else None,
                                    unreferenced=sorted(unreferenced), ok=not failures)

    # Unseen + G per fold, pooled.
    unseen, timers = {}, set()
    for d in detectors:
        out = {"seed_average": unseen_plus_G(parts, held, folds, lambda F, d=d: f"{d}@{F}", rk0)}
        for s in SEEDS:
            out[f"seed_{s}"] = unseen_plus_G(parts, held, folds, lambda F, d=d, s=s: f"{d}#{s}@{F}", rk0)
        out["mean_std_over_seeds"] = mean_std([out[f"seed_{s}"] for s in SEEDS])
        unseen[d] = out
        for v in [out["seed_average"]] + [out[f"seed_{s}"] for s in SEEDS]:
            timers.add((v["timer"]["fp"], round(v["timer"]["mean_lead"], 12),
                        json.dumps({F: p["timer"] for F, p in v["paths"].items()}, sort_keys=True)))
    if len(timers) != 1:
        failures.append(f"R3: G differs between scores ({len(timers)} versions)")
    checks["R3_one_timer"] = dict(versions=len(timers), ok=len(timers) == 1)
    G = next(iter(unseen.values()))["seed_average"]["timer"]

    inputs = {
        "folds": dict(path=str(Path(args.folds).resolve()), sha256=folds_sha),
        "reference_per_seed": dict(path=str(Path(args.reference_per_seed).resolve()),
                                   sha256=sha256(args.reference_per_seed)),
        **{f"inputs_{n}": dict(path=str((Path(args.inputs_dir) / n).resolve()), sha256=sha256(Path(args.inputs_dir) / n))
           for n in ("rldx_bundle.json", "train_ids.json", "calibration_ids.json", "test_ids.json")},
        **{f"safe_lstm_seed{s}": dict(path=str((Path(args.safe_dir) / f"lstm_seed{s}" / "normalized_scores.jsonl").resolve()),
                                      sha256=sha256(Path(args.safe_dir) / f"lstm_seed{s}" / "normalized_scores.jsonl"))
           for s in SEEDS},
    }
    counts = dict(tasks=len(tasks), train=len(train), calibration_successes=len(parts["calibration"]),
                  train_failures=sum(r["failed"] for r in train), test=len(parts["test"]),
                  test_failures=sum(r["failed"] for r in parts["test"]),
                  held_out_per_fold={F: len(held[F]) for F in folds})
    result = dict(plan=PLAN, status="exploratory (second policy, opened cohorts)", inputs=inputs, counts=counts,
                  checks=checks, reproduction_failures=failures,
                  notes=dict(std="ddof 1 over training seeds 0-2; timers are deterministic",
                             seen="fold all, per-task normalization, + P (the published per-seed rows)",
                             unseen="per fold, pooled normalization, + G fitted on seen-task calibration successes "
                                    "and training failures, pooled over folds"),
                  table2=dict(global_timer=dict(unseen=G),
                              **{d: dict(seen=seen[d], unseen=unseen[d]) for d in detectors}))

    print(f"RLDX-1: {counts['tasks']} tasks, train {counts['train']}, calibration successes "
          f"{counts['calibration_successes']}, test {counts['test']}; folds "
          + ", ".join(f"{F}: {len(held[F])} tasks" for F in folds))
    print(f"P (seen): lead {stored_P[0]} FPR {stored_P[1]} | G (unseen): FPR {G['fpr']:.3f} ({G['fp']} FP) "
          f"lead {G['mean_lead']:.3f} TWA {G['twa']:.3f}")
    print("Table II RLDX-1 (FPR, gain, TWA): seen + P | unseen + G")
    for d in detectors:
        a, b = seen[d]["mean_std_over_seeds"], unseen[d]["mean_std_over_seeds"]
        fps = [unseen[d][f"seed_{s}"]["fp"] for s in SEEDS]
        cell = lambda m: " ".join(f"{m[k]['mean']:.3f}+-{m[k]['std']:.3f}" for k in ("fpr", "gain", "twa"))
        print(f"  {d:22s} {cell(a)} | {cell(b)}  (unseen FP {fps}; seed average gain "
              f"{unseen[d]['seed_average']['gain']:+.4f} at FPR {unseen[d]['seed_average']['fpr']:.3f})")
    if failures:
        print("REPRODUCTION FAILED:")
        for f in failures:
            print("  " + f)
        print("No table is reported when a prerequisite fails.")
        result.pop("table2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, default=float) + "\n")
    if failures:
        return 1
    print("RLDX-1 UNSEEN TABLE OK")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folds", help="write R7's stratified fold assignment")
    f.add_argument("--inputs-dir", required=True, help="rldx_hide_and_seek.py prepare output (rldx_bundle.json)")
    f.add_argument("--write", required=True)
    r = sub.add_parser("run", help="seen + P and unseen + G for every cross-task detector")
    r.add_argument("--safe-dir", required=True, help="<SAFE20>/calibration/lstm (defines the 15.15 rows)")
    r.add_argument("--inputs-dir", required=True)
    r.add_argument("--folds", required=True)
    r.add_argument("--fold-dirs", nargs="+", action="append", required=True,
                   help="one family's fold directories (folds 0-3 and all); repeat for each family")
    r.add_argument("--reference-per-seed", required=True, help="rldx_per_seed.json")
    r.add_argument("--unreferenced", nargs="+", help="R11: detectors with no published row (e.g. actprobe actprobe_t); "
                                                      "R2 checks only that their P equals the stored P")
    r.add_argument("--output", required=True)
    args = p.parse_args(argv)
    return folds_cmd(args) if args.cmd == "folds" else run(args)


if __name__ == "__main__":
    sys.exit(main())
