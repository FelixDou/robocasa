"""Score cohorts with detectors retrained without one fold's tasks, without computing any metric.

One fold of docs/xr1_cross_task_unseen_plan.md per call. Every detector is trained exactly as
``score_cohorts`` trains it (same detector code, seeds, epochs, feature aggregation and
training-ID order), but only on the training rollouts of the fold's seen tasks: the held-out
tasks' training rollouts are dropped. Every rollout of every named cohort is then scored, from
seen and held-out tasks alike. No threshold, alarm or metric is computed.

``--fold all`` holds out no task. Its raw trajectories must equal those of the Amendment 2
scoring run (``--reference-raw COHORT FILE``), which shows that the fold code trains the
registered detectors.

Normalization. A zero-shot monitor has no statistics for a task it has never seen, so
normalizations.json stores, per detector and seed:
  pooled    one location and scale from the training maxima of all seen-task training
            rollouts (fit_safe_normalization's robust formula, pooled over tasks)
  per_task  fit_safe_normalization over the seen tasks (held-out tasks absent)
The analysis (scripts/xr1_lead_time/cross_task_lead.py) applies them; this step does not.

Outputs in --output-dir:
  <cohort>_predictions.jsonl  one row per (detector, rollout): per_seed_raw, task, label,
                              whether the task is held out in this fold
  normalizations.json         detector -> seed -> {pooled, per_task}
  scoring.json                fold, held-out tasks, training rollouts per task, inputs with
                              SHA-256, detector configs, reference checks

Usage:
  python -u -m robocasa.recovery.safe.baselines.score_cross_task \\
      --dataset-dir DIR [DIR ...] --runtime-bundle runtime_bundle.json \\
      --train-ids scores.jsonl --folds scripts/xr1_lead_time/cross_task_folds.json --fold 0 \\
      --cohort lead_calibration IDS.jsonl --cohort confirmatory_test IDS.jsonl \\
      --detectors hide_and_seek hide_and_seek_zero hide_and_seek_first cosine_knn \\
      --feature-aggregation last --seeds 0 1 2 --output-dir OUT [--reference-raw COHORT FILE ...]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from ..analyze_time_safe_hybrid import fit_safe_normalization
from .detectors import DETECTOR_REGISTRY, build_detector
from .run_baselines import (
    _json_default,
    _write_json,
    detector_kwargs,
    index_datasets,
    load_baseline_rollout,
    load_rollout_ids,
)
from .score_cohorts import _code_commit, _sha256


# Detectors whose inputs need training rollouts of the scored task, so they are undefined
# on a held-out task.
NEEDS_SEEN_TASK = {"hide_and_seek_task_shuffle"}


def pooled_normalization(records, horizons):
    """fit_safe_normalization's location and scale, over all records' maxima at once."""
    maxima = [float(np.max(np.asarray(r["scores"], dtype=np.float64)[: horizons[r["task_name"]]])) for r in records]
    if not maxima:
        raise ValueError("No training scores to normalize")
    array = np.asarray(maxima, dtype=np.float64)
    location = float(np.median(array))
    mad_scale = float(1.4826 * np.median(np.abs(array - location)))
    scale = max(mad_scale, float(np.std(array)), 1e-6)
    return {"location": location, "scale": scale, "fit_rollouts": len(array)}


def held_out_tasks(spec, fold, tasks):
    listed = [t for f in spec["folds"] for t in f["held_out"]]
    if sorted(listed) != sorted(tasks) or len(listed) != len(set(listed)):
        raise ValueError("The fold file does not hold out every bundle task exactly once")
    if fold == "all":
        return set()
    match = [f for f in spec["folds"] if str(f["fold"]) == fold]
    if not match:
        raise ValueError(f"Fold {fold} is not in the fold file")
    return set(match[0]["held_out"])


def _raw_reference(path):
    out = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["detector"]][str(row["rollout_id"])] = {
                str(s): np.asarray(v, float) for s, v in row["per_seed_raw"].items()}
    return out


def run(args):
    started = time.time()
    fold = str(args.fold)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if args.reference_raw and fold != "all":
        raise ValueError("--reference-raw only applies to --fold all")
    bad = sorted(set(args.detectors) & NEEDS_SEEN_TASK)
    if bad and fold != "all":
        raise ValueError(f"{bad} need training rollouts of the scored task; they are undefined on held-out tasks")

    bundle = json.loads(Path(args.runtime_bundle).read_text())
    horizons = {task: int(value) for task, value in bundle["task_horizons"].items()}
    spec = json.loads(Path(args.folds).read_text())
    held = held_out_tasks(spec, fold, horizons)
    seen = sorted(set(horizons) - held)

    all_train_ids = load_rollout_ids(args.train_ids, split=args.train_split)
    if not all_train_ids:
        raise ValueError(f"No {args.train_split!r} rows in {args.train_ids}")
    development = set(bundle.get("development_rollout_ids", all_train_ids))
    if not set(all_train_ids) <= development:
        raise ValueError("Training IDs are not part of the bundle's development rollouts")

    names = [name for name, _ in args.cohort]
    if len(set(names)) != len(names):
        raise ValueError(f"Cohort names repeat: {names}")
    cohort_ids = {name: list(dict.fromkeys(load_rollout_ids(path))) for name, path in args.cohort}
    for name, ids in cohort_ids.items():
        if set(ids) & set(all_train_ids):
            raise ValueError(f"Cohort {name} overlaps the training rollouts")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(cohort_ids[a]) & set(cohort_ids[b]):
                raise ValueError(f"Cohorts {a} and {b} share rollouts")

    index = index_datasets(args.dataset_dir)
    missing = [rid for rid in [*all_train_ids, *(r for ids in cohort_ids.values() for r in ids)] if rid not in index]
    if missing:
        raise ValueError(f"{len(missing)} rollouts are missing from --dataset-dir, e.g. {missing[:3]}")
    # Keep the training-ID order, so fold all reproduces score_cohorts exactly.
    train_ids = [rid for rid in all_train_ids if index[rid][1].task_name not in held]
    dropped = Counter(index[rid][1].task_name for rid in all_train_ids if index[rid][1].task_name in held)

    def load(ids):
        return [load_baseline_rollout(*index[rid], args.feature_aggregation) for rid in ids]

    train = load(train_ids)
    by_task = Counter(r["task_name"] for r in train)
    if set(by_task) != set(seen):
        raise ValueError(f"Training rollouts do not cover exactly the seen tasks: {sorted(set(seen) ^ set(by_task))}")
    if len({bool(r["failed"]) for r in train}) != 2:
        raise ValueError("Training rollouts need both successes and failures")
    cohorts = {name: load(ids) for name, ids in cohort_ids.items()}
    for name, rows in cohorts.items():
        unknown = sorted({r["task_name"] for r in rows} - set(horizons))
        if unknown:
            raise ValueError(f"Cohort {name} has tasks without a frozen horizon: {unknown}")
    print(f"fold {fold}: {len(held)} held-out tasks, train {len(train)} (dropped {sum(dropped.values())}); "
          + ", ".join(f"{n} {len(r)}" for n, r in cohorts.items()), flush=True)

    meta = {
        "protocol": "cross_task_score_only_no_metrics",
        "plan": "docs/xr1_cross_task_unseen_plan.md",
        "code_commit": _code_commit(),
        "fold": fold,
        "held_out_tasks": sorted(held),
        "training_rollouts_by_task": dict(sorted(by_task.items())),
        "dropped_training_rollouts_by_task": dict(sorted(dropped.items())),
        "folds_sha256": _sha256(args.folds),
        "train_ids_sha256": _sha256(args.train_ids),
        "inputs": {
            "runtime_bundle": {"path": str(Path(args.runtime_bundle).resolve()), "sha256": _sha256(args.runtime_bundle)},
            "train_ids": {"path": str(Path(args.train_ids).resolve()), "sha256": _sha256(args.train_ids)},
            "folds": {"path": str(Path(args.folds).resolve()), "sha256": _sha256(args.folds)},
            **{f"cohort_{name}": {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
               for name, path in args.cohort},
        },
        "dataset_dirs": [str(Path(d).resolve()) for d in args.dataset_dir],
        "feature_aggregation": args.feature_aggregation,
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "detectors": list(args.detectors),
        "detector_configs": {},
        "counts": {"train": len(train), **{name: len(rows) for name, rows in cohorts.items()}},
    }
    output.mkdir(parents=True, exist_ok=True)
    normalizations_out = {}
    streams = {name: (output / f"{name}_predictions.jsonl").open("w") for name in cohorts}
    seen_horizons = {t: horizons[t] for t in seen}
    try:
        for detector_name in args.detectors:
            check = getattr(DETECTOR_REGISTRY[detector_name], "unsupported_reason", None)
            reason = check((*train, *(r for rows in cohorts.values() for r in rows))) if check else None
            if reason:
                raise ValueError(f"{detector_name} cannot run on these rollouts: {reason}")
            configs, norms = {}, {}
            cohort_scores = {name: {} for name in cohorts}
            for seed in args.seeds:
                detector = build_detector(detector_name, seed=seed, **detector_kwargs(detector_name, args, sorted(horizons)))
                detector.fit(train)
                configs[str(seed)] = detector.config()
                fit_records = [{"task_name": r["task_name"], "scores": np.asarray(detector.score(r), np.float64)}
                               for r in train]
                norms[str(seed)] = {"pooled": pooled_normalization(fit_records, horizons),
                                    "per_task": fit_safe_normalization(fit_records, seen_horizons)}
                for name, rows in cohorts.items():
                    for r in rows:
                        values = np.asarray(detector.score(r), np.float64)
                        if values.shape != (len(r["features"]),) or not np.all(np.isfinite(values)):
                            raise ValueError(f"{detector_name} produced invalid scores for {r['rollout_id']}")
                        cohort_scores[name].setdefault(r["rollout_id"], {})[str(seed)] = values
                print(f"{detector_name} seed {seed} done", flush=True)
            normalizations_out[detector_name] = norms
            for name, rows in cohorts.items():
                for r in rows:
                    streams[name].write(json.dumps({
                        "detector": detector_name, "rollout_id": r["rollout_id"], "task_name": r["task_name"],
                        "failed": bool(r["failed"]), "fold": fold, "held_out": r["task_name"] in held,
                        "source_inferences": len(r["features"]),
                        "per_seed_raw": cohort_scores[name][r["rollout_id"]],
                    }, default=_json_default) + "\n")
                streams[name].flush()
            meta["detector_configs"][detector_name] = configs
            _write_json(output / "scoring.json", meta)
    finally:
        for stream in streams.values():
            stream.close()
    _write_json(output / "normalizations.json", normalizations_out)

    status = 0
    meta["reference_checks"] = {}
    for name, path in args.reference_raw or []:
        if name not in cohorts:
            raise ValueError(f"--reference-raw names unknown cohort {name}")
        ref = _raw_reference(path)
        now = _raw_reference(output / f"{name}_predictions.jsonl")
        for detector_name in sorted(set(ref) & set(now)):
            diffs = []
            for rid, seeds in now[detector_name].items():
                other = ref[detector_name].get(rid)
                if other is None or set(other) != set(seeds):
                    diffs.append(float("inf"))
                    continue
                diffs += [float(np.max(np.abs(seeds[s] - other[s]))) if seeds[s].shape == other[s].shape else float("inf")
                          for s in seeds]
            diff = max(diffs) if diffs else None
            ok = bool(diffs) and diff <= 1e-12
            meta["reference_checks"][f"{name}/{detector_name}"] = {
                "reference": str(Path(path).resolve()), "rollouts_scored": len(now[detector_name]),
                "max_abs_diff": diff, "ok": ok}
            print(f"reference {name}/{detector_name}: {len(now[detector_name])} rollouts, max diff {diff} -> "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
            status |= int(not ok)
    meta["elapsed_seconds"] = time.time() - started
    _write_json(output / "scoring.json", meta)
    if status:
        print("REFERENCE MISMATCH: fold all differs from the earlier scoring run")
    elif args.reference_raw:
        print("REFERENCE OK")
    return status


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", nargs="+", required=True)
    parser.add_argument("--runtime-bundle", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--folds", required=True, help="cross_task_folds.json")
    parser.add_argument("--fold", required=True, help="fold number in the fold file, or 'all'")
    parser.add_argument("--cohort", nargs=2, action="append", required=True, metavar=("NAME", "IDS"))
    parser.add_argument("--detectors", nargs="+", required=True, choices=sorted(DETECTOR_REGISTRY))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--feature-aggregation", default="flow_last_horizon_mean")
    parser.add_argument("--action-dims", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-raw", nargs=2, action="append", metavar=("COHORT", "PREDICTIONS"),
                        help="earlier per_seed_raw trajectories to reproduce exactly (fold all only)")
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
