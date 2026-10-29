"""SAFE-MLP on unseen tasks: fold manifests, fold-directory assembly and score comparison.

Stage B of R5 in docs/xr1_review_analyses_plan.md. Training and scoring use the existing tools
unchanged in their defaults (train_seen_tasks with a fold selection manifest, and
score_seen_checkpoint_external with --unseen-tasks score). This module computes no threshold,
alarm or metric.

  manifests  one selection manifest per fold of scripts/xr1_lead_time/cross_task_folds.json: the
             frozen seen-task split (seen38_split_seed0.json) without the fold's held-out tasks.
             The frozen training IDs must equal the manifest's, so fold all is the final refit.
  assemble   one fold directory in the cross-task format (scoring.json, normalizations.json,
             <cohort>_predictions.jsonl with detector "safe"), from the fold's three training runs
             and their scores of both E1 cohorts. Normalization per seed: pooled over the seen-task
             training rollouts (the cross-task formula) and per task (fit_safe_normalization).
             Fold all also checks that its scores equal the E1 rows' per_seed_raw_safe and that its
             per-task normalization equals the runtime bundle's.
  assemble-export
             RLDX-1 (R7 Stage B'): one SAFE head's fold directory from scores of a whole export,
             split into calibration, test and (seen-task) training cohorts by ID lists.
  compare    largest absolute difference between two score directories (scores.jsonl).

Usage:
  python -m robocasa.recovery.safe.cross_task_safe manifests --selection-manifest SEEN38.json \\
      --folds scripts/xr1_lead_time/cross_task_folds.json --frozen-scores FINAL/indep_seed0/scores.jsonl \\
      --output-dir OUT
  python -m robocasa.recovery.safe.cross_task_safe assemble --fold 0 --folds FOLDS --runtime-bundle B.json \\
      --training-run 0 DIR --training-run 1 DIR --training-run 2 DIR \\
      --scores lead_calibration 0 DIR ... --scores confirmatory_test 2 DIR [--fold-manifest fold_0.json] \\
      [--reference-predictions lead_calibration E1/calibration_predictions.jsonl ...] --output-dir OUT
  python -m robocasa.recovery.safe.cross_task_safe compare A_DIR B_DIR
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .analyze_time_safe_hybrid import fit_safe_normalization
from .baselines.score_cross_task import pooled_normalization

PLAN = "docs/xr1_review_analyses_plan.md"
DETECTOR = "safe"
COHORTS = ("lead_calibration", "confirmatory_test")
ALL = "all"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def code_commit():
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def fold_sets(spec):
    tasks = set(spec["tasks"])
    listed = [t for f in spec["folds"] for t in f["held_out"]]
    if sorted(listed) != sorted(tasks) or len(listed) != len(set(listed)):
        raise ValueError("The fold file does not hold out every task exactly once")
    held = {str(f["fold"]): set(f["held_out"]) for f in spec["folds"]}
    held[ALL] = set()
    return tasks, held


# ---- manifests ------------------------------------------------------------------------------

def manifests(args):
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    source = json.loads(Path(args.selection_manifest).read_text())
    spec = json.loads(Path(args.folds).read_text())
    tasks, held = fold_sets(spec)
    if set(source["included_tasks"]) != tasks:
        raise ValueError("The selection manifest's tasks differ from the fold file's")
    frozen = jsonl(args.frozen_scores)
    task_of = {}
    for r in frozen:
        if task_of.setdefault(str(r["rollout_id"]), r["task_name"]) != r["task_name"]:
            raise ValueError(f"Rollout {r['rollout_id']} has two tasks in the frozen scores")
    by_split = defaultdict(set)
    for r in frozen:
        by_split[r["split"]].add(str(r["rollout_id"]))
    train, test = [str(x) for x in source["train"]], [str(x) for x in source["test"]]
    if set(train) != by_split["train"] or set(test) != by_split["test"]:
        raise ValueError("The frozen final refit was not trained on this selection manifest's split")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for F in sorted(f for f in held if f != ALL):
        seen = sorted(tasks - held[F])
        keep = lambda ids: [i for i in ids if task_of[i] not in held[F]]
        dropped = Counter(task_of[i] for i in train if task_of[i] in held[F])
        manifest = dict(
            protocol="cross_task_safe_fold_selection", plan=PLAN, fold=F, held_out_tasks=sorted(held[F]),
            included_tasks=seen, train=keep(train), test=keep(test), split_seed=source["split_seed"],
            split_unit=source.get("split_unit", "rollout"),
            source_selection_manifest=dict(path=str(Path(args.selection_manifest).resolve()),
                                           sha256=sha256(args.selection_manifest)),
            folds=dict(path=str(Path(args.folds).resolve()), sha256=sha256(args.folds)),
            dropped_train_by_task=dict(sorted(dropped.items())))
        if {task_of[i] for i in manifest["train"]} != set(seen) or {task_of[i] for i in manifest["test"]} != set(seen):
            raise ValueError(f"Fold {F}: the kept rollouts do not cover exactly the seen tasks")
        write_json(out / f"fold_{F}.json", manifest)
        summary[F] = dict(held_out=len(held[F]), seen=len(seen), train=len(manifest["train"]),
                          test=len(manifest["test"]), dropped_train=sum(dropped.values()),
                          dropped_per_held_out_task=sorted(set(dropped.values())))
        print(f"fold {F}: {len(seen)} seen tasks, train {len(manifest['train'])} (dropped {sum(dropped.values())}, "
              f"{sorted(set(dropped.values()))} per held-out task), test {len(manifest['test'])}", flush=True)
    write_json(out / "manifests.json", dict(plan=PLAN, code_commit=code_commit(), folds=summary))
    print("MANIFESTS OK")
    return 0


# ---- assemble -------------------------------------------------------------------------------

def assemble(args):
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    F = str(args.fold)
    spec = json.loads(Path(args.folds).read_text())
    tasks, held = fold_sets(spec)
    if F not in held:
        raise ValueError(f"Fold {F} is not in the fold file")
    seen = tasks - held[F]
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    horizons = {t: int(v) for t, v in bundle["task_horizons"].items()}
    if set(horizons) != tasks:
        raise ValueError("The runtime bundle's tasks differ from the fold file's")

    runs = {str(s): Path(d).resolve() for s, d in args.training_run}
    if len(runs) != len(args.training_run):
        raise ValueError("A training seed is given twice")
    seeds = sorted(runs)
    manifest_path = str(Path(args.fold_manifest).resolve()) if args.fold_manifest else None
    norms, by_task, checks = {}, None, {}
    for s, run in runs.items():
        metrics = json.loads((run / "metrics.json").read_text())
        split = json.loads((run / "split_manifest.json").read_text())
        if str(metrics["seed"]) != s:
            raise ValueError(f"Training run {run} is seed {metrics['seed']}, not {s}")
        if set(split["task_names"]) != seen:
            raise ValueError(f"Training run {run} was not trained on exactly fold {F}'s seen tasks")
        if manifest_path is not None and str(Path(metrics["selection_manifest"]).resolve()) != manifest_path:
            raise ValueError(f"Training run {run} used {metrics['selection_manifest']}, not {manifest_path}")
        train = [r for r in jsonl(run / "scores.jsonl") if r.get("split") == "train"]
        counts = dict(sorted(Counter(r["task_name"] for r in train).items()))
        if set(counts) != seen:
            raise ValueError(f"Training scores of {run} do not cover exactly the seen tasks")
        if by_task is None:
            by_task = counts
        elif counts != by_task:
            raise ValueError("Training rollouts differ between seeds")
        records = [dict(task_name=r["task_name"], scores=np.asarray(r["scores"], np.float64)) for r in train]
        norms[s] = dict(pooled=pooled_normalization(records, horizons),
                        per_task=fit_safe_normalization(records, {t: horizons[t] for t in seen}))
    if F == ALL:
        worst = 0.0
        for s in seeds:
            ref = bundle["task_normalizations"][s]
            for t in tasks:
                mine = norms[s]["per_task"][t]
                worst = max(worst, abs(mine["location"] - ref[t]["location"]), abs(mine["scale"] - ref[t]["scale"]))
        checks["bundle_normalization"] = dict(max_abs_diff=worst, ok=worst <= 1e-12)

    names = [c for c, _, _ in args.scores]
    if sorted(set(names)) != sorted(COHORTS):
        raise ValueError(f"Scores are needed for exactly the cohorts {COHORTS}")
    cohorts = defaultdict(dict)
    provenance = defaultdict(dict)
    for cohort, s, d in args.scores:
        d = Path(d).resolve()
        if str(s) in cohorts[cohort]:
            raise ValueError(f"Two score directories for {cohort} seed {s}")
        if str(s) not in runs:
            raise ValueError(f"Scores for seed {s} have no training run")
        prov = json.loads((d / "provenance.json").read_text())
        if Path(prov["training_run_dir"]).resolve() != runs[str(s)]:
            raise ValueError(f"{d} was scored with {prov['training_run_dir']}, not {runs[str(s)]}")
        if prov["checkpoint_sha256"] != sha256(runs[str(s)] / "model_final.ckpt"):
            raise ValueError(f"{d}: checkpoint hash differs from the training run's")
        if prov["group"] != cohort:
            raise ValueError(f"{d} is group {prov['group']}, not {cohort}")
        if prov.get("unseen_tasks") != "score" or prov.get("cutoff_rule") != "training":
            raise ValueError(f"{d} was not scored with --unseen-tasks score and the training cutoff rule")
        if set(prov["unseen_task_names"]) != held[F] & set(prov["task_names"]):
            raise ValueError(f"{d}: tasks scored as unseen differ from fold {F}'s held-out tasks")
        rows = {}
        for r in jsonl(d / "scores.jsonl"):
            rows[str(r["rollout_id"])] = r
        cohorts[cohort][str(s)] = rows
        provenance[cohort][str(s)] = dict(path=str(d), export_dir=prov["export_dir"],
                                          checkpoint_sha256=prov["checkpoint_sha256"])
    for cohort in COHORTS:
        if sorted(cohorts[cohort]) != seeds:
            raise ValueError(f"{cohort}: scores for seeds {sorted(cohorts[cohort])}, training seeds {seeds}")
        ids = {s: set(v) for s, v in cohorts[cohort].items()}
        if any(v != ids[seeds[0]] for v in ids.values()):
            raise ValueError(f"{cohort}: seeds scored different rollouts")
        exports = {p["export_dir"] for p in provenance[cohort].values()}
        if len(exports) != 1:
            raise ValueError(f"{cohort}: seeds scored different exports")

    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for cohort in COHORTS:
        n = 0
        with (out / f"{cohort}_predictions.jsonl").open("w") as stream:
            for rid in sorted(cohorts[cohort][seeds[0]]):
                rows = {s: cohorts[cohort][s][rid] for s in seeds}
                r0 = rows[seeds[0]]
                if any(r["task_name"] != r0["task_name"] or bool(r["failed"]) != bool(r0["failed"])
                       or len(r["scores"]) != len(r0["scores"]) for r in rows.values()):
                    raise ValueError(f"{cohort}: seeds disagree on {rid}")
                is_held = r0["task_name"] in held[F]
                if any(bool(r.get("training_task")) == is_held for r in rows.values()):
                    raise ValueError(f"{cohort}: training-task flag of {rid} contradicts fold {F}")
                stream.write(json.dumps(dict(detector=DETECTOR, rollout_id=rid, task_name=r0["task_name"],
                                             failed=bool(r0["failed"]), fold=F, held_out=is_held,
                                             source_inferences=len(r0["scores"]),
                                             per_seed_raw={s: rows[s]["scores"] for s in seeds})) + "\n")
                n += 1
        counts[cohort] = n
    write_json(out / "normalizations.json", {DETECTOR: norms})

    status = 0
    reference = {}
    for cohort, path in args.reference_predictions or []:
        if F != ALL:
            raise ValueError("--reference-predictions applies to fold all only")
        if cohort not in COHORTS:
            raise ValueError(f"Unknown cohort {cohort}")
        worst, n = 0.0, 0
        for r in jsonl(path):
            rid = str(r["rollout_id"])
            got = cohorts[cohort][seeds[0]].get(rid)
            if got is None:
                raise ValueError(f"{cohort}: reference rollout {rid} was not scored")
            for s in seeds:
                a = np.asarray(cohorts[cohort][s][rid]["scores"], float)
                b = np.asarray(r["per_seed_raw_safe"][s], float)
                worst = max(worst, float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf"))
            n += 1
        reference[cohort] = dict(reference=str(Path(path).resolve()), rollouts=n, max_abs_diff=worst,
                                 ok=n > 0 and worst <= 1e-12)
        print(f"reference {cohort}: {n} rollouts, max diff {worst} -> {'OK' if reference[cohort]['ok'] else 'MISMATCH'}")
        status |= int(not reference[cohort]["ok"])
    if F == ALL:
        checks["reference_predictions"] = reference
        if set(reference) != set(COHORTS):
            print("fold all needs --reference-predictions for both cohorts")
            status = 1
        bn = checks["bundle_normalization"]
        print(f"bundle normalization: max diff {bn['max_abs_diff']} -> {'OK' if bn['ok'] else 'MISMATCH'}")
        status |= int(not bn["ok"])

    meta = dict(protocol="cross_task_safe_assembled_no_metrics", plan=PLAN, code_commit=code_commit(), fold=F,
                held_out_tasks=sorted(held[F]), training_rollouts_by_task=by_task, folds_sha256=sha256(args.folds),
                detectors=[DETECTOR], seeds=seeds, fold_manifest=manifest_path,
                training_runs={s: str(r) for s, r in runs.items()},
                training_checkpoint_sha256={s: sha256(r / "model_final.ckpt") for s, r in runs.items()},
                scores=provenance, counts=counts, runtime_bundle_sha256=sha256(args.runtime_bundle), checks=checks)
    write_json(out / "scoring.json", meta)
    print(f"fold {F}: " + ", ".join(f"{c} {n}" for c, n in counts.items()) + f", seeds {seeds}")
    print("ASSEMBLY OK" if status == 0 else "REFERENCE MISMATCH: do not analyze this fold")
    return status


# ---- assemble-export (RLDX-1, R7 Stage B') ----------------------------------------------------

EXPORT_COHORTS = ("calibration", "test", "train")


def assemble_export(args):
    """One SAFE head's fold directory from scores of a whole export (every split in one export).

    RLDX-1 keeps all its rollouts in one export, so each training run scores the whole export
    (score_seen_checkpoint_external --group all --unseen-tasks score). The ID lists split it into
    the calibration, test and training cohorts; the training cohort is written for the fold's seen
    tasks only (train_predictions.jsonl), as score_cross_task --score-train does. Normalization per
    seed: pooled over the seen tasks' training rollouts (the cross-task formula) and per task with
    SAFE's own formula (calibrate_seen_tasks.fit_task_normalization), which the published seen rows
    use. Fold all also checks its scores and per-task normalization against the frozen
    normalized_scores.jsonl (--frozen-normalized), and every fold checks that the external scores of
    its training rollouts equal its training run's own scores.
    """
    from .calibrate_seen_tasks import fit_task_normalization

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    F = str(args.fold)
    spec = json.loads(Path(args.folds).read_text())
    tasks, held = fold_sets(spec)
    if F not in held:
        raise ValueError(f"Fold {F} is not in the fold file")
    seen = tasks - held[F]
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    horizons = {t: int(v) for t, v in bundle["task_horizons"].items()}
    if set(horizons) != tasks:
        raise ValueError("The bundle's tasks differ from the fold file's")
    detector = f"safe_{args.head}"

    ids = {}
    for name, path in args.cohort:
        if name not in EXPORT_COHORTS or name in ids:
            raise ValueError(f"--cohort must name each of {EXPORT_COHORTS} once")
        value = json.loads(Path(path).read_text())
        ids[name] = [str(x) for x in value]
    if sorted(ids) != sorted(EXPORT_COHORTS):
        raise ValueError(f"--cohort must name each of {EXPORT_COHORTS} once")
    all_ids = [i for name in EXPORT_COHORTS for i in ids[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("The cohorts share rollouts")

    runs = {str(s): Path(d).resolve() for s, d in args.training_run}
    if len(runs) != len(args.training_run):
        raise ValueError("A training seed is given twice")
    seeds = sorted(runs)
    manifest_path = str(Path(args.fold_manifest).resolve()) if args.fold_manifest else None
    if F != ALL and manifest_path is None:
        raise ValueError("A fold other than all needs --fold-manifest")
    train_rows, by_task, norms = {}, None, {}
    for s, run in runs.items():
        metrics = json.loads((run / "metrics.json").read_text())
        split = json.loads((run / "split_manifest.json").read_text())
        if str(metrics["seed"]) != s:
            raise ValueError(f"Training run {run} is seed {metrics['seed']}, not {s}")
        if str(metrics.get("model")) != args.head:
            raise ValueError(f"Training run {run} is model {metrics.get('model')}, not {args.head}")
        if set(split["task_names"]) != seen:
            raise ValueError(f"Training run {run} was not trained on exactly fold {F}'s seen tasks")
        if manifest_path is not None and str(Path(metrics["selection_manifest"]).resolve()) != manifest_path:
            raise ValueError(f"Training run {run} used {metrics['selection_manifest']}, not {manifest_path}")
        rows = [r for r in jsonl(run / "scores.jsonl") if r.get("split") == "train"]
        counts = dict(sorted(Counter(r["task_name"] for r in rows).items()))
        if set(counts) != seen:
            raise ValueError(f"Training scores of {run} do not cover exactly the seen tasks")
        if by_task is None:
            by_task = counts
        elif counts != by_task:
            raise ValueError("Training rollouts differ between seeds")
        if {str(r["rollout_id"]) for r in rows} - set(ids["train"]):
            raise ValueError(f"{run}: training rollouts outside the training ID list")
        train_rows[s] = {str(r["rollout_id"]): r for r in rows}
        norms[s] = {"per_task": fit_task_normalization(rows)}

    scores, provenance = {}, {}
    for s, d in args.scores:
        d = Path(d).resolve()
        s = str(s)
        if s in scores:
            raise ValueError(f"Two score directories for seed {s}")
        if s not in runs:
            raise ValueError(f"Scores for seed {s} have no training run")
        prov = json.loads((d / "provenance.json").read_text())
        if Path(prov["training_run_dir"]).resolve() != runs[s]:
            raise ValueError(f"{d} was scored with {prov['training_run_dir']}, not {runs[s]}")
        if prov["checkpoint_sha256"] != sha256(runs[s] / "model_final.ckpt"):
            raise ValueError(f"{d}: checkpoint hash differs from the training run's")
        if prov["group"] != "all":
            raise ValueError(f"{d} is group {prov['group']}, not all")
        if prov.get("unseen_tasks") != "score" or prov.get("cutoff_rule") != "training":
            raise ValueError(f"{d} was not scored with --unseen-tasks score and the training cutoff rule")
        if set(prov["unseen_task_names"]) != held[F] & set(prov["task_names"]):
            raise ValueError(f"{d}: tasks scored as unseen differ from fold {F}'s held-out tasks")
        scores[s] = {str(r["rollout_id"]): r for r in jsonl(d / "scores.jsonl")}
        provenance[s] = dict(path=str(d), export_dir=prov["export_dir"], checkpoint_sha256=prov["checkpoint_sha256"])
    if sorted(scores) != seeds:
        raise ValueError(f"Scores for seeds {sorted(scores)}, training seeds {seeds}")
    if len({p["export_dir"] for p in provenance.values()}) != 1:
        raise ValueError("Seeds scored different exports")
    for s in seeds:
        missing = [i for i in all_ids if i not in scores[s]]
        if missing:
            raise ValueError(f"Seed {s}: {len(missing)} listed rollouts were not scored, e.g. {missing[:3]}")

    checks, status = {}, 0
    worst = 0.0
    for s in seeds:
        for rid, r in train_rows[s].items():
            a, b = np.asarray(scores[s][rid]["scores"], float), np.asarray(r["scores"], float)
            worst = max(worst, float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf"))
    checks["train_scores_vs_training_run"] = dict(max_abs_diff=worst, ok=worst <= 1e-12)
    print(f"external vs training-run scores of the training rollouts: max diff {worst} -> "
          f"{'OK' if worst <= 1e-12 else 'MISMATCH'}", flush=True)
    status |= int(worst > 1e-12)
    for s in seeds:
        records = [dict(task_name=scores[s][rid]["task_name"], scores=np.asarray(scores[s][rid]["scores"], np.float64))
                   for rid in ids["train"] if scores[s][rid]["task_name"] in seen]
        norms[s]["pooled"] = pooled_normalization(records, horizons)

    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for cohort in EXPORT_COHORTS:
        n = 0
        with (out / f"{cohort}_predictions.jsonl").open("w") as stream:
            for rid in sorted(ids[cohort]):
                rows = {s: scores[s][rid] for s in seeds}
                r0 = rows[seeds[0]]
                if any(r["task_name"] != r0["task_name"] or bool(r["failed"]) != bool(r0["failed"])
                       or len(r["scores"]) != len(r0["scores"]) for r in rows.values()):
                    raise ValueError(f"{cohort}: seeds disagree on {rid}")
                is_held = r0["task_name"] in held[F]
                if any(bool(r.get("training_task")) == is_held for r in rows.values()):
                    raise ValueError(f"{cohort}: training-task flag of {rid} contradicts fold {F}")
                if cohort == "train" and is_held:
                    continue
                stream.write(json.dumps(dict(detector=detector, rollout_id=rid, task_name=r0["task_name"],
                                             failed=bool(r0["failed"]), fold=F, held_out=is_held,
                                             source_inferences=len(r0["scores"]),
                                             per_seed_raw={s: rows[s]["scores"] for s in seeds})) + "\n")
                n += 1
        counts[cohort] = n
    write_json(out / "normalizations.json", {detector: norms})

    reference = {}
    if F == ALL:
        frozen = {str(s): Path(p) for s, p in args.frozen_normalized or []}
        if sorted(frozen) != seeds:
            raise ValueError("Fold all needs --frozen-normalized for every seed")
        cohort_of = {i: name for name in EXPORT_COHORTS for i in ids[name]}
        diffs = defaultdict(float)
        norm_worst, n = 0.0, defaultdict(int)
        for s in seeds:
            for r in jsonl(frozen[s]):
                rid = str(r["rollout_id"])
                if rid not in cohort_of:
                    continue
                c = cohort_of[rid]
                a, b = np.asarray(scores[s][rid]["scores"], float), np.asarray(r["full_raw_scores"], float)
                diffs[c] = max(diffs[c], float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf"))
                n[c] += 1
                mine = norms[s]["per_task"][r["task_name"]]
                norm_worst = max(norm_worst, abs(mine["location"] - float(r["normalization_location"])),
                                 abs(mine["scale"] - float(r["normalization_scale"])))
        for c in EXPORT_COHORTS:
            full = n[c] == len(ids[c]) * len(seeds)
            reference[f"{c}/{detector}"] = dict(rollouts_scored=n[c] // max(len(seeds), 1), max_abs_diff=diffs[c],
                                                ok=full and diffs[c] <= 1e-12)
        reference[f"normalization/{detector}"] = dict(max_abs_diff=norm_worst, ok=norm_worst <= 1e-12)
        for k, v in reference.items():
            print(f"reference {k}: max diff {v['max_abs_diff']} -> {'OK' if v['ok'] else 'MISMATCH'}", flush=True)
            status |= int(not v["ok"])

    meta = dict(protocol="cross_task_safe_export_assembled_no_metrics", plan=PLAN + " (R7)", code_commit=code_commit(),
                fold=F, held_out_tasks=sorted(held[F]), training_rollouts_by_task=by_task,
                folds_sha256=sha256(args.folds), detectors=[detector], head=args.head, seeds=[int(s) for s in seeds],
                score_train=True, fold_manifest=manifest_path, training_runs={s: str(r) for s, r in runs.items()},
                training_checkpoint_sha256={s: sha256(r / "model_final.ckpt") for s, r in runs.items()},
                scores=provenance, counts=counts, bundle_sha256=sha256(args.runtime_bundle), checks=checks,
                reference_checks=reference)
    write_json(out / "scoring.json", meta)
    print(f"fold {F} {detector}: " + ", ".join(f"{c} {n}" for c, n in counts.items()) + f", seeds {seeds}")
    print("ASSEMBLY OK" if status == 0 else "REFERENCE MISMATCH: do not analyze this fold")
    return status


# ---- compare --------------------------------------------------------------------------------

def compare(args):
    a = {str(r["rollout_id"]): np.asarray(r["scores"], float) for r in jsonl(Path(args.a) / "scores.jsonl")}
    b = {str(r["rollout_id"]): np.asarray(r["scores"], float) for r in jsonl(Path(args.b) / "scores.jsonl")}
    if set(a) != set(b):
        print(f"DIFFERENT ROLLOUTS: {len(set(a) ^ set(b))} not in both")
        return 1
    worst = max(float(np.max(np.abs(a[k] - b[k]))) if a[k].shape == b[k].shape else float("inf") for k in a)
    print(f"{len(a)} rollouts, max abs diff {worst}")
    print("IDENTICAL" if worst == 0.0 else "DIFFERENT")
    return int(worst != 0.0)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifests")
    for name in ("--selection-manifest", "--folds", "--frozen-scores", "--output-dir"):
        m.add_argument(name, required=True)
    a = sub.add_parser("assemble")
    for name in ("--fold", "--folds", "--runtime-bundle", "--output-dir"):
        a.add_argument(name, required=True)
    a.add_argument("--training-run", nargs=2, action="append", required=True, metavar=("SEED", "DIR"))
    a.add_argument("--scores", nargs=3, action="append", required=True, metavar=("COHORT", "SEED", "DIR"))
    a.add_argument("--fold-manifest")
    a.add_argument("--reference-predictions", nargs=2, action="append", metavar=("COHORT", "PREDICTIONS"))
    e = sub.add_parser("assemble-export", help="one SAFE head's fold directory from whole-export scores (RLDX-1, R7)")
    for name in ("--fold", "--folds", "--runtime-bundle", "--output-dir"):
        e.add_argument(name, required=True)
    e.add_argument("--head", required=True, choices=("lstm", "indep"))
    e.add_argument("--cohort", nargs=2, action="append", required=True, metavar=("NAME", "IDS"),
                   help="calibration, test and train, each with a JSON list of rollout IDs")
    e.add_argument("--training-run", nargs=2, action="append", required=True, metavar=("SEED", "DIR"))
    e.add_argument("--scores", nargs=2, action="append", required=True, metavar=("SEED", "DIR"))
    e.add_argument("--fold-manifest")
    e.add_argument("--frozen-normalized", nargs=2, action="append", metavar=("SEED", "NORMALIZED_SCORES"),
                   help="fold all: the frozen normalized_scores.jsonl of each seed")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"manifests": manifests, "assemble": assemble, "assemble-export": assemble_export,
            "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
