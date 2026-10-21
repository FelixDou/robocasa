"""Evaluate published failure-detection baselines under the frozen matched-FPR protocol.

The runner reuses the frozen Xiaomi runtime bundle from
``run_prospective_matched_fpr calibrate`` so every baseline sees exactly the
same training rollouts, task horizons, elapsed-time risk curves, balanced
calibration rollouts, FPR target and stage windows as SAFE:

1. fit each baseline on SAFE's outer-training rollouts;
2. score every inference of the training, calibration and evaluation rollouts;
3. normalize per task from training maxima, as SAFE scores are normalized;
4. pick one threshold (baseline only) and one staged pair (baseline early,
   elapsed time late) on the primary calibration rollouts at the target FPR;
5. apply them unchanged to the evaluation rollouts.

Example::

    python -m robocasa.recovery.safe.baselines.run_baselines \\
      --dataset-dir "$XR1_RAW_DATASET" \\
      --runtime-bundle "$XR1_PROSPECTIVE_CAL/runtime_bundle.json" \\
      --train-ids "$XR1_FINAL_ROOT/indep_seed0/scores.jsonl" \\
      --evaluation-ids "$XR1_PROSPECTIVE_EVAL/event_predictions.jsonl" \\
      --output-dir "$XR1_BASELINES_OUT" \\
      --detectors action_consistency actprobe mahalanobis cosine_knn \\
                  pca_kmeans rnd hide_and_seek \\
      --seeds 0 1 2 --device cuda
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from ..analyze_time_safe_hybrid import fit_safe_normalization
from ..dataset import aggregate_features, load_manifest
from ..prospective_matched_fpr import (
    event_metrics,
    paired_failure_comparison,
    select_single_threshold,
    select_staged_thresholds,
    validate_stage_windows,
)
from .detectors import DETECTOR_REGISTRY, build_detector


TORCH_DETECTORS = {"actprobe", "hide_and_seek", "rnd"}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def load_rollout_ids(path, split=None):
    """Read rollout IDs from a JSON list, a JSON object, or a JSONL file.

    With ``split``, JSONL rows carrying a ``split`` field are filtered to it
    (SAFE ``scores.jsonl`` files mark outer-training rows as ``train``).
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if split is not None and any("split" in row for row in rows):
            rows = [row for row in rows if row.get("split") == split]
        return [str(row["rollout_id"]) for row in rows]
    value = json.loads(text)
    if isinstance(value, dict):
        for key in ("rollout_ids", "evaluation_rollout_ids", "test"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError(f"{path} has no rollout_ids, evaluation_rollout_ids or test list")
    return [str(item["rollout_id"] if isinstance(item, dict) else item) for item in value]


def index_datasets(dataset_dirs):
    index = {}
    for directory in dataset_dirs:
        directory = Path(directory).resolve()
        for metadata in load_manifest(directory):
            if metadata.rollout_id in index:
                raise ValueError(f"Rollout {metadata.rollout_id} appears in two datasets")
            index[metadata.rollout_id] = (directory, metadata)
    return index


def load_baseline_rollout(directory, metadata, aggregation):
    """Load one raw rollout, keeping only aggregated features and actions."""
    with np.load(directory / metadata.tensor_path, allow_pickle=False) as payload:
        features = payload["features"] if "features" in payload else payload["safe_features"]
        actions = (
            np.asarray(payload["policy_action_chunks"], dtype=np.float32)
            if "policy_action_chunks" in payload
            else None
        )
        failed = bool(payload["failed"])
    if failed != bool(metadata.failed):
        raise ValueError(f"Tensor label disagrees with manifest for {metadata.rollout_id}")
    return {
        "rollout_id": metadata.rollout_id,
        "task_name": metadata.task_name,
        "failed": bool(metadata.failed),
        "features": aggregate_features(features, aggregation),
        "actions": actions,
        "replan_steps": int(metadata.replan_steps),
        "environment_seed": metadata.environment_seed,
        "environment_reset_index": metadata.environment_reset_index,
        "seed_protocol": metadata.seed_protocol,
    }


def select_balanced(records, per_class):
    # Same deterministic per-task/per-outcome selection as the frozen evaluation.
    from ..run_prospective_matched_fpr import _select_per_task_class

    return _select_per_task_class(records, per_class)


def normalized_running_max(scores, stats):
    running = np.maximum.accumulate(np.asarray(scores, dtype=np.float64))
    return (running - float(stats["location"])) / float(stats["scale"])


def build_records(rollouts, scores_by_seed, normalizations, horizons, time_curves):
    records = []
    for rollout in rollouts:
        task = rollout["task_name"]
        if task not in horizons:
            raise ValueError(f"Task {task} has no frozen horizon in the runtime bundle")
        normalized = np.mean(
            [
                normalized_running_max(scores_by_seed[seed][rollout["rollout_id"]],
                                       normalizations[seed][task])
                for seed in sorted(scores_by_seed)
            ],
            axis=0,
        )
        steps = np.arange(1, len(normalized) + 1)
        curve = np.asarray(time_curves[task]["risk"], dtype=np.float64)
        records.append(
            {
                "rollout_id": rollout["rollout_id"],
                "task_name": task,
                "failed": rollout["failed"],
                "horizon": int(horizons[task]),
                "source_inferences": len(normalized),
                # The shared evaluation utilities read the learned score from
                # "safe_scores"; here it holds the baseline's score.
                "safe_scores": normalized,
                "time_scores": curve[np.minimum(steps, len(curve)) - 1],
                "environment_seed": rollout.get("environment_seed"),
                "environment_reset_index": rollout.get("environment_reset_index"),
            }
        )
    return records


def detector_kwargs(name, args, tasks):
    kwargs = {}
    if name in TORCH_DETECTORS:
        kwargs.update(device=args.device, epochs=args.epochs)
    if name in {"action_consistency", "actprobe"} and args.action_dims is not None:
        kwargs["action_dims"] = args.action_dims
    if name == "actprobe":
        kwargs["tasks"] = tasks
    return kwargs


def run(args):
    started = time.time()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    bundle = json.loads(Path(args.runtime_bundle).read_text())
    horizons = {task: int(value) for task, value in bundle["task_horizons"].items()}
    time_curves = bundle["time_risk_curves"]
    target_fpr = float(bundle["target_fpr"] if args.target_fpr is None else args.target_fpr)
    safe_end, time_start = validate_stage_windows(
        bundle["safe_eligibility_end_fraction"] if args.safe_end is None else args.safe_end,
        bundle["time_eligibility_start_fraction"] if args.time_start is None else args.time_start,
    )
    # SAFE's normalization and time curves were fit on the outer-training
    # split of the final refits, so baselines must be fit on the same rows.
    train_ids = load_rollout_ids(args.train_ids, split=args.train_split)
    if not train_ids:
        raise ValueError(f"No {args.train_split!r} rows in {args.train_ids}")
    development = set(bundle.get("development_rollout_ids", train_ids))
    if not set(train_ids) <= development:
        raise ValueError("Training IDs are not part of the bundle's development rollouts")
    calibration_ids = bundle["primary_calibration_rollout_ids"]
    evaluation_ids = load_rollout_ids(args.evaluation_ids)
    for left, right, name in (
        (train_ids, calibration_ids, "train/calibration"),
        (train_ids, evaluation_ids, "train/evaluation"),
        (calibration_ids, evaluation_ids, "calibration/evaluation"),
    ):
        if set(left) & set(right):
            raise ValueError(f"{name} rollout IDs overlap")

    index = index_datasets(args.dataset_dir)
    missing = [rid for rid in (*train_ids, *calibration_ids, *evaluation_ids) if rid not in index]
    if missing:
        raise ValueError(f"{len(missing)} rollouts are missing from --dataset-dir, e.g. {missing[:3]}")

    def load(ids):
        return [load_baseline_rollout(*index[rid], args.feature_aggregation) for rid in ids]

    train, calibration, evaluation = load(train_ids), load(calibration_ids), load(evaluation_ids)
    tasks = sorted(horizons)
    if any(r["actions"] is None for r in (*train, *calibration, *evaluation)):
        print("Some rollouts lack action chunks; action-space baselines will fail on them.")

    results = {
        "protocol": "frozen_matched_fpr_baselines",
        "runtime_bundle": str(Path(args.runtime_bundle).resolve()),
        "target_fpr": target_fpr,
        "safe_eligibility_end_fraction": safe_end,
        "time_eligibility_start_fraction": time_start,
        "feature_aggregation": args.feature_aggregation,
        "seeds": list(args.seeds),
        "counts": {
            "train": len(train),
            "calibration": len(calibration),
            "evaluation_all": len(evaluation),
        },
        "detectors": {},
        "skipped": {},
    }
    summary_rows = []
    predictions = {}
    time_reference = None
    for name in args.detectors:
        check = getattr(DETECTOR_REGISTRY[name], "unsupported_reason", None)
        reason = check((*train, *calibration, *evaluation)) if check else None
        if reason:
            print(f"Skipping {name}: {reason}")
            results["skipped"][name] = reason
            continue
        scores ={"train": defaultdict(dict), "calibration": defaultdict(dict),
                  "evaluation": defaultdict(dict)}
        configs = {}
        for seed in args.seeds:
            detector = build_detector(name, seed=seed, **detector_kwargs(name, args, tasks))
            detector.fit(train)
            configs[str(seed)] = detector.config()
            for split, rollouts in (("train", train), ("calibration", calibration),
                                    ("evaluation", evaluation)):
                for rollout in rollouts:
                    values = np.asarray(detector.score(rollout), dtype=np.float64)
                    if values.shape != (len(rollout["features"]),) or not np.all(np.isfinite(values)):
                        raise ValueError(f"{name} produced invalid scores for {rollout['rollout_id']}")
                    scores[split][seed][rollout["rollout_id"]] = values
        normalizations = {
            seed: fit_safe_normalization(
                [{"task_name": r["task_name"], "scores": scores["train"][seed][r["rollout_id"]]}
                 for r in train],
                horizons,
            )
            for seed in args.seeds
        }
        calibration_records = build_records(
            calibration, scores["calibration"], normalizations, horizons, time_curves
        )
        evaluation_all = build_records(
            evaluation, scores["evaluation"], normalizations, horizons, time_curves
        )
        evaluation_records = evaluation_all
        selection_counts = None
        if args.test_per_class:
            evaluation_records, selection_counts = select_balanced(evaluation_all, args.test_per_class)

        single = select_single_threshold(calibration_records, "safe_only", target_fpr=target_fpr,
                                         safe_end=safe_end, time_start=time_start)
        staged = select_staged_thresholds(calibration_records, target_fpr=target_fpr,
                                          safe_end=safe_end, time_start=time_start)
        if time_reference is None:
            time_reference = select_single_threshold(
                calibration_records, "time_only", target_fpr=target_fpr,
                safe_end=safe_end, time_start=time_start,
            )
        evaluated = {}
        for label, detector_key, thresholds in (
            ("baseline_only", "safe_only", single["thresholds"]),
            ("staged_baseline_time", "staged_safe_time", staged["thresholds"]),
            ("time_only", "time_only", time_reference["thresholds"]),
        ):
            evaluated[label] = event_metrics(evaluation_records, detector_key, thresholds,
                                             safe_end=safe_end, time_start=time_start)
        comparisons = {
            f"{label}_vs_time_only": paired_failure_comparison(
                evaluation_records, key, thresholds, "time_only", time_reference["thresholds"],
                safe_end=safe_end, time_start=time_start,
            )
            for label, key, thresholds in (
                ("baseline_only", "safe_only", single["thresholds"]),
                ("staged_baseline_time", "staged_safe_time", staged["thresholds"]),
            )
        }
        results["detectors"][name] = {
            "configs": configs,
            "calibration_thresholds": {
                "baseline_only": single["thresholds"],
                "staged_baseline_time": staged["thresholds"],
                "time_only": time_reference["thresholds"],
            },
            "calibration_metrics": {
                "baseline_only": single["validation_metrics"],
                "staged_baseline_time": staged["validation_metrics"],
            },
            "evaluation": evaluated,
            "paired_vs_time_only": comparisons,
            "evaluation_selection": selection_counts,
            "evaluation_rollouts": len(evaluation_records),
        }
        predictions[name] = evaluation_all
        for label, metrics in evaluated.items():
            if label == "time_only" and summary_rows and any(r[0] == "time_only" for r in summary_rows):
                continue
            row_name = "time_only" if label == "time_only" else f"{name} ({label.replace('_', ' ')})"
            summary_rows.append((
                row_name if label != "time_only" else "time_only",
                metrics["roc_auc"], metrics["true_positive_rate"], metrics["false_positive_rate"],
                metrics["balanced_accuracy"], metrics["missed_failure_adjusted_detection_fraction"],
                metrics["failure_recall_by_landmark"]["0.25"],
            ))
        _write_json(output / "results.json", results)

    with (output / "evaluation_predictions.jsonl").open("w") as stream:
        for name, records in predictions.items():
            for record in records:
                stream.write(json.dumps({"detector": name, **record}, default=_json_default) + "\n")
    lines = [
        "| Detector | ROC-AUC | TPR | FPR | Bal. acc. | Miss-adj. detection | Recall by 25% |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join([row[0], *[f"{value:.4f}" for value in row[1:]]]) + " |")
    for name, reason in results["skipped"].items():
        lines.append(f"\nSkipped `{name}`: {reason}.")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    results["elapsed_seconds"] = time.time() - started
    _write_json(output / "results.json", results)
    print("\n".join(lines))
    return results


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", nargs="+", required=True,
                        help="Raw SAFE dataset directories containing manifest.jsonl")
    parser.add_argument("--runtime-bundle", required=True,
                        help="Frozen runtime_bundle.json from run_prospective_matched_fpr calibrate")
    parser.add_argument("--evaluation-ids", required=True,
                        help="JSON list/object or JSONL (e.g. event_predictions.jsonl) of evaluation rollouts")
    parser.add_argument("--train-ids", required=True,
                        help="SAFE final-refit scores.jsonl (or an ID list) naming training rollouts")
    parser.add_argument("--train-split", default="train",
                        help="Keep only JSONL rows with this split value")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detectors", nargs="+", default=sorted(DETECTOR_REGISTRY),
                        choices=sorted(DETECTOR_REGISTRY))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--feature-aggregation", default="flow_last_horizon_mean",
                        help="dataset.aggregate_features mode for (T, flow, horizon, D) features")
    parser.add_argument("--action-dims", nargs="+", type=int, default=None,
                        help="Action dimensions used by action-space baselines (default: all)")
    parser.add_argument("--test-per-class", type=int, default=10,
                        help="Balanced evaluation rollouts per task and outcome (0 keeps all)")
    parser.add_argument("--target-fpr", type=float, default=None)
    parser.add_argument("--safe-end", type=float, default=None)
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
