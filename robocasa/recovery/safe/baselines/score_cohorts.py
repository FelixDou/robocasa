"""Score cohorts with retrained baseline detectors, without computing any metric.

Trains each detector exactly as ``run_baselines`` does (same training IDs, seeds,
feature aggregation and per-task normalization from training maxima), then writes the
per-inference score trajectories of every rollout in each named cohort. No threshold,
alarm, outcome metric or comparison is computed, so this step can run on a
preregistered cohort before its analysis (docs/xr1_confirmatory_amendment_2_hide_and_seek.md).

Outputs in --output-dir:
  <cohort>_predictions.jsonl  one row per (detector, rollout): normalized seed-averaged
                              running-max scores in ``safe_scores`` (as run_baselines),
                              raw per-seed scores in ``per_seed_raw``, frozen time risk
  normalizations.json         detector -> seed -> task -> {location, scale}
  scoring.json                inputs with SHA-256, counts, configs, reference checks
  train_predictions.jsonl     only with --score-train: the training rollouts' own
                              (in-sample) trajectories, in the same format. The RLDX-1
                              replication (docs/rldx_hide_and_seek_plan.md) chooses the fused
                              path on training failures, as report section 15.15 did for SAFE.

``--save-models DIR`` (Hide-and-Seek only) writes each seed's fitted model to
DIR/<detector>_seed<s>.npz (named arrays, no pickle: PCA, input standardization, LSTM
weights), reloads it and requires identical scores on every cohort rollout. scoring.json
records each file's SHA-256 and a content hash (detectors.content_hash). E2 freezes these
models (docs/xr1_review_analyses_plan.md, R10).

``--reference COHORT FILE`` compares a cohort's trajectories with an earlier
run_baselines predictions file (for every detector present in both) and exits with
status 1 if any differs, after writing the outputs.

Usage:
  python -u -m robocasa.recovery.safe.baselines.score_cohorts \\
      --dataset-dir DIR [DIR ...] --runtime-bundle runtime_bundle.json \\
      --train-ids scores.jsonl --cohort NAME IDS.jsonl [--cohort ...] \\
      --detectors hide_and_seek ... --feature-aggregation last --seeds 0 1 2 \\
      --output-dir OUT [--reference NAME earlier_predictions.jsonl ...]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from ..analyze_time_safe_hybrid import fit_safe_normalization
from .detectors import DETECTOR_REGISTRY, SAVABLE, build_detector, load_detector, save_detector
from .run_baselines import (
    _json_default,
    _write_json,
    build_records,
    detector_kwargs,
    index_datasets,
    load_baseline_rollout,
    load_rollout_ids,
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _reference(path):
    out = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["detector"]][str(row["rollout_id"])] = np.asarray(row["safe_scores"], float)
    return out


def run(args):
    started = time.time()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    bundle = json.loads(Path(args.runtime_bundle).read_text())
    horizons = {task: int(value) for task, value in bundle["task_horizons"].items()}
    time_curves = bundle["time_risk_curves"]
    train_ids = load_rollout_ids(args.train_ids, split=args.train_split)
    if not train_ids:
        raise ValueError(f"No {args.train_split!r} rows in {args.train_ids}")
    development = set(bundle.get("development_rollout_ids", train_ids))
    if not set(train_ids) <= development:
        raise ValueError("Training IDs are not part of the bundle's development rollouts")

    models_dir = None
    if args.save_models:
        unsavable = [d for d in args.detectors if d not in SAVABLE]
        if unsavable:
            raise ValueError(f"--save-models supports {SAVABLE} only, not {unsavable}")
        models_dir = Path(args.save_models).resolve()
        if models_dir.exists() and any(models_dir.iterdir()):
            raise FileExistsError(f"Models directory is not empty: {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)

    names = [name for name, _ in args.cohort]
    if args.score_train and "train" in names:
        raise ValueError("A cohort cannot be named 'train' when --score-train is set")
    if len(set(names)) != len(names):
        raise ValueError(f"Cohort names repeat: {names}")
    cohort_ids = {name: list(dict.fromkeys(load_rollout_ids(path))) for name, path in args.cohort}
    for name, ids in cohort_ids.items():
        if set(ids) & set(train_ids):
            raise ValueError(f"Cohort {name} overlaps the training rollouts")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(cohort_ids[a]) & set(cohort_ids[b]):
                raise ValueError(f"Cohorts {a} and {b} share rollouts")

    index = index_datasets(args.dataset_dir)
    wanted = [*train_ids, *(rid for ids in cohort_ids.values() for rid in ids)]
    missing = [rid for rid in wanted if rid not in index]
    if missing:
        raise ValueError(f"{len(missing)} rollouts are missing from --dataset-dir, e.g. {missing[:3]}")

    def load(ids):
        return [load_baseline_rollout(*index[rid], args.feature_aggregation) for rid in ids]

    train = load(train_ids)
    cohorts = {name: load(ids) for name, ids in cohort_ids.items()}
    for name, rows in cohorts.items():
        unknown = sorted({r["task_name"] for r in rows} - set(horizons))
        if unknown:
            raise ValueError(f"Cohort {name} has tasks without a frozen horizon: {unknown}")
    tasks = sorted(horizons)
    print(f"train {len(train)}; " + ", ".join(f"{n} {len(r)}" for n, r in cohorts.items()), flush=True)

    meta = {
        "protocol": "score_only_no_metrics",
        "code_commit": _code_commit(),
        "inputs": {
            "runtime_bundle": {"path": str(Path(args.runtime_bundle).resolve()),
                               "sha256": _sha256(args.runtime_bundle)},
            "train_ids": {"path": str(Path(args.train_ids).resolve()), "sha256": _sha256(args.train_ids)},
            **{f"cohort_{name}": {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
               for name, path in args.cohort},
        },
        "dataset_dirs": [str(Path(d).resolve()) for d in args.dataset_dir],
        "feature_aggregation": args.feature_aggregation,
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "score_train": bool(args.score_train),
        "counts": {"train": len(train), **{name: len(rows) for name, rows in cohorts.items()}},
        "detectors": {},
        "skipped": {},
    }
    normalizations_out = {}
    streams = {name: (output / f"{name}_predictions.jsonl").open("w") for name in cohorts}
    if args.score_train:
        streams["train"] = (output / "train_predictions.jsonl").open("w")
    try:
        for detector_name in args.detectors:
            check = getattr(DETECTOR_REGISTRY[detector_name], "unsupported_reason", None)
            reason = check((*train, *(r for rows in cohorts.values() for r in rows))) if check else None
            if reason:
                print(f"Skipping {detector_name}: {reason}", flush=True)
                meta["skipped"][detector_name] = reason
                continue
            train_scores, cohort_scores, configs, models = {}, {n: {} for n in cohorts}, {}, {}
            for seed in args.seeds:
                detector = build_detector(detector_name, seed=seed,
                                          **detector_kwargs(detector_name, args, tasks, horizons))
                detector.fit(train)
                configs[str(seed)] = detector.config()
                train_scores[seed] = {r["rollout_id"]: np.asarray(detector.score(r), np.float64) for r in train}
                for name, rows in cohorts.items():
                    cohort_scores[name][seed] = {}
                    for r in rows:
                        values = np.asarray(detector.score(r), np.float64)
                        if values.shape != (len(r["features"]),) or not np.all(np.isfinite(values)):
                            raise ValueError(f"{detector_name} produced invalid scores for {r['rollout_id']}")
                        cohort_scores[name][seed][r["rollout_id"]] = values
                if models_dir is not None:
                    models[str(seed)] = _save_and_check(detector, models_dir / f"{detector_name}_seed{seed}.npz",
                                                        cohorts, cohort_scores, seed)
                print(f"{detector_name} seed {seed} done", flush=True)
            normalizations = {
                seed: fit_safe_normalization(
                    [{"task_name": r["task_name"], "scores": train_scores[seed][r["rollout_id"]]} for r in train],
                    horizons,
                )
                for seed in args.seeds
            }
            normalizations_out[detector_name] = {str(s): v for s, v in normalizations.items()}
            for name, rows in cohorts.items():
                for record in build_records(rows, cohort_scores[name], normalizations, horizons, time_curves):
                    rid = record["rollout_id"]
                    record["per_seed_raw"] = {str(s): cohort_scores[name][s][rid] for s in args.seeds}
                    streams[name].write(json.dumps({"detector": detector_name, **record},
                                                   default=_json_default) + "\n")
                streams[name].flush()
            if args.score_train:
                for record in build_records(train, train_scores, normalizations, horizons, time_curves):
                    rid = record["rollout_id"]
                    record["per_seed_raw"] = {str(s): train_scores[s][rid] for s in args.seeds}
                    streams["train"].write(json.dumps({"detector": detector_name, **record},
                                                      default=_json_default) + "\n")
                streams["train"].flush()
            meta["detectors"][detector_name] = {"configs": configs, **({"models": models} if models else {})}
            _write_json(output / "scoring.json", meta)
    finally:
        for stream in streams.values():
            stream.close()
    _write_json(output / "normalizations.json", normalizations_out)

    status = 0
    meta["reference_checks"] = {}
    for name, path in args.reference or []:
        if name not in cohorts:
            raise ValueError(f"--reference names unknown cohort {name}")
        ref = _reference(path)
        now = _reference(output / f"{name}_predictions.jsonl")
        for detector_name in sorted(set(ref) & set(now)):
            common = sorted(set(ref[detector_name]) & set(now[detector_name]))
            def gap(i):
                a, b = now[detector_name][i], ref[detector_name][i]
                return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")

            diff = max((gap(i) for i in common), default=None)
            ok = bool(common) and len(common) == len(now[detector_name]) and diff is not None and diff <= 1e-12
            meta["reference_checks"][f"{name}/{detector_name}"] = {
                "reference": str(Path(path).resolve()), "rollouts_compared": len(common),
                "rollouts_scored": len(now[detector_name]), "max_abs_diff": diff, "ok": ok}
            print(f"reference {name}/{detector_name}: {len(common)} rollouts, max diff {diff} -> "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
            status |= int(not ok)
    meta["elapsed_seconds"] = time.time() - started
    _write_json(output / "scoring.json", meta)
    if status:
        print("REFERENCE MISMATCH: retrained detectors differ from the earlier run")
    elif args.reference:
        print("REFERENCE OK")
    return status


def _save_and_check(detector, path, cohorts, cohort_scores, seed):
    """Save a fitted detector, reload it and require identical scores on every cohort rollout."""
    content = save_detector(detector, path)
    loaded, reloaded = load_detector(path)
    if reloaded != content:
        raise ValueError(f"{path}: content hash changed on reload")
    for name, rows in cohorts.items():
        for r in rows:
            if not np.array_equal(np.asarray(loaded.score(r), np.float64), cohort_scores[name][seed][r["rollout_id"]]):
                raise ValueError(f"{path}: the reloaded model scores {r['rollout_id']} differently")
    return {"path": str(path), "sha256": _sha256(path), "content_sha256": content, "reload_identical": True}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", nargs="+", required=True)
    parser.add_argument("--runtime-bundle", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--cohort", nargs=2, action="append", required=True, metavar=("NAME", "IDS"),
                        help="cohort name and a file of its rollout IDs (JSON list or JSONL with rollout_id)")
    parser.add_argument("--detectors", nargs="+", required=True, choices=sorted(DETECTOR_REGISTRY))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--feature-aggregation", default="flow_last_horizon_mean")
    parser.add_argument("--action-dims", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-train", action="store_true",
                        help="also write train_predictions.jsonl (in-sample training trajectories)")
    parser.add_argument("--save-models", metavar="DIR",
                        help=f"write each fitted model to DIR/<detector>_seed<s>.npz (supported: {', '.join(SAVABLE)})")
    parser.add_argument("--reference", nargs=2, action="append", metavar=("COHORT", "PREDICTIONS"),
                        help="earlier run_baselines predictions to reproduce exactly for this cohort")
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
