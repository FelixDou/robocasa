"""Freeze and evaluate the prospective Xiaomi matched-FPR shadow protocol.

The ``calibrate`` phase consumes three-seed frozen MLP scores for a new
calibration export and writes an immutable runtime bundle.  The ``evaluate``
phase consumes a disjoint prospective shadow export and never changes model,
normalization, time-risk, stage-window, or threshold parameters.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from .analyze_time_safe_hybrid import (
    fit_safe_normalization,
    fit_time_risk,
    training_task_horizons,
)
from .prospective_matched_fpr import (
    DETECTORS,
    event_metrics,
    paired_failure_comparison,
    select_single_threshold,
    select_staged_thresholds,
    validate_stage_windows,
)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(_json_value(row), sort_keys=True, allow_nan=False) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No score records found in {path}")
    ids = [str(row["rollout_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate rollout IDs in {path}")
    return rows


def _signature(rows):
    return {
        str(row["rollout_id"]): (
            str(row["task_name"]),
            bool(row["failed"]),
            len(row["scores"]),
        )
        for row in rows
    }


def _load_seed_group(root, seeds, model="indep"):
    root = Path(root).resolve()
    output = {}
    reference = None
    provenances = {}
    for seed in seeds:
        run = root / f"{model}_seed{seed}"
        # External score roots use seed0/seed1/seed2 by default; accept both.
        if not run.is_dir():
            run = root / f"seed{seed}"
        rows = _load_jsonl(run / "scores.jsonl")
        signature = _signature(rows)
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Score identities differ for seed {seed} in {root}")
        output[int(seed)] = rows
        provenance_path = run / "provenance.json"
        if provenance_path.is_file():
            provenances[int(seed)] = json.loads(provenance_path.read_text())
    return output, reference, provenances


def _training_records(final_root, seeds):
    runs, _, _ = _load_seed_group(final_root, seeds)
    selected = {
        seed: [row for row in rows if row.get("split") == "train"]
        for seed, rows in runs.items()
    }
    if any(not rows for rows in selected.values()):
        raise ValueError("Every detector seed requires frozen outer-training scores")
    return selected


def _validate_score_provenance(provenances, expected_hashes, group, seeds):
    missing = sorted(set(seeds) - set(provenances))
    if missing:
        raise ValueError(f"{group} scores lack provenance for seeds {missing}")
    for seed in seeds:
        provenance = provenances[seed]
        if provenance.get("group") != group:
            raise ValueError(
                f"Seed {seed} score group is {provenance.get('group')!r}, "
                f"expected {group!r}"
            )
        if provenance.get("checkpoint_sha256") != expected_hashes[str(seed)]:
            raise ValueError(f"Seed {seed} external scores use the wrong checkpoint")
        if provenance.get("checkpoint_updated") is not False:
            raise ValueError(f"Seed {seed} external scoring changed the checkpoint")


def _json_time_curves(curves):
    return {
        task: {
            key: np.asarray(value).tolist()
            for key, value in payload.items()
        }
        for task, payload in curves.items()
    }


def _assemble_external(runs, normalizations, horizons, time_curves):
    seeds = sorted(runs)
    by_seed = {
        seed: {str(row["rollout_id"]): row for row in rows}
        for seed, rows in runs.items()
    }
    records = []
    for rollout_id in sorted(by_seed[seeds[0]]):
        reference = by_seed[seeds[0]][rollout_id]
        task = str(reference["task_name"])
        if task not in horizons:
            raise ValueError(f"External task {task} has no frozen horizon")
        raw_by_seed = {}
        normalized_by_seed = {}
        length = None
        for seed in seeds:
            row = by_seed[seed][rollout_id]
            raw = np.asarray(row["scores"], dtype=np.float64)
            if length is None:
                length = len(raw)
            elif len(raw) != length:
                raise ValueError(f"Seed trajectories differ for {rollout_id}")
            stats = normalizations[str(seed)][task]
            running = np.maximum.accumulate(raw)
            normalized = (running - float(stats["location"])) / float(stats["scale"])
            raw_by_seed[str(seed)] = raw.tolist()
            normalized_by_seed[str(seed)] = normalized.tolist()
        safe = np.mean(
            np.stack(
                [np.asarray(normalized_by_seed[str(seed)]) for seed in seeds]
            ),
            axis=0,
        )
        horizon = int(horizons[task])
        steps = np.arange(1, length + 1, dtype=np.int64)
        curve = np.asarray(time_curves[task]["risk"], dtype=np.float64)
        time = curve[np.minimum(steps, len(curve)) - 1]
        records.append(
            {
                "rollout_id": rollout_id,
                "task_name": task,
                "task_type": reference.get("task_type"),
                "failed": bool(reference["failed"]),
                "horizon": horizon,
                "source_inferences": length,
                "safe_scores": safe,
                "time_scores": time,
                "per_seed_raw_safe": raw_by_seed,
                "per_seed_normalized_running_max": normalized_by_seed,
                "environment_seed": reference.get("environment_seed"),
                "environment_reset_index": reference.get("environment_reset_index"),
                "seed_protocol": reference.get("seed_protocol"),
                "video_path": reference.get("video_path"),
                "inference_environment_steps": reference.get(
                    "inference_environment_steps"
                ),
            }
        )
    return records


def _identity_keys(records):
    return {
        (
            str(row["task_name"]),
            row.get("environment_seed"),
            row.get("environment_reset_index"),
        )
        for row in records
        if row.get("environment_seed") is not None
    }


def _select_per_task_class(records, per_class):
    groups = defaultdict(list)
    for row in records:
        groups[(str(row["task_name"]), bool(row["failed"]))].append(row)
    tasks = sorted({str(row["task_name"]) for row in records})
    selected = []
    counts = {}
    for task in tasks:
        counts[task] = {}
        for failed, outcome in ((False, "successes"), (True, "failures")):
            values = sorted(
                groups[(task, failed)],
                key=lambda row: (
                    row.get("environment_seed")
                    if row.get("environment_seed") is not None
                    else -1,
                    row.get("environment_reset_index")
                    if row.get("environment_reset_index") is not None
                    else -1,
                    str(row["rollout_id"]),
                ),
            )
            if len(values) < int(per_class):
                raise ValueError(
                    f"{task} has {len(values)} {outcome}, needs {per_class}"
                )
            selected.extend(values[: int(per_class)])
            counts[task][outcome] = {
                "available": len(values),
                "selected": int(per_class),
                "overshoot": len(values) - int(per_class),
            }
    return selected, counts


def _threshold_payload(records, target_fpr, safe_end, time_start):
    safe = select_single_threshold(
        records,
        "safe_only",
        target_fpr=target_fpr,
        safe_end=safe_end,
        time_start=time_start,
    )
    time = select_single_threshold(
        records,
        "time_only",
        target_fpr=target_fpr,
        safe_end=safe_end,
        time_start=time_start,
    )
    staged = select_staged_thresholds(
        records,
        target_fpr=target_fpr,
        safe_end=safe_end,
        time_start=time_start,
    )
    return {"safe_only": safe, "time_only": time, "staged_safe_time": staged}


def calibrate(args):
    if not 0.0 <= float(args.target_fpr) < 1.0:
        raise ValueError("--target-fpr must be in [0, 1)")
    if int(args.min_per_class) < 1:
        raise ValueError("--min-per-class must be positive")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in args.seeds)
    safe_end, time_start = validate_stage_windows(args.safe_end, args.time_start)
    training = _training_records(args.final_root, seeds)
    development_runs, _, _ = _load_seed_group(args.final_root, seeds)
    reference_training = training[seeds[0]]
    horizons = training_task_horizons(reference_training)
    normalizations = {
        str(seed): fit_safe_normalization(training[seed], horizons)
        for seed in seeds
    }
    time_curves = fit_time_risk(reference_training, horizons, prior=args.time_prior)
    calibration_runs, _, provenances = _load_seed_group(
        args.calibration_score_root, seeds
    )
    checkpoint_hashes = {
        str(seed): _sha256(
            Path(args.final_root).resolve() / f"indep_seed{seed}" / "model_final.ckpt"
        )
        for seed in seeds
    }
    _validate_score_provenance(
        provenances,
        checkpoint_hashes,
        "calibration",
        seeds,
    )
    records = _assemble_external(
        calibration_runs,
        normalizations,
        horizons,
        time_curves,
    )
    tasks = sorted(horizons)
    if {row["task_name"] for row in records} != set(tasks):
        raise ValueError("Calibration score root lacks exact frozen task coverage")
    development_ids = {
        str(row["rollout_id"]) for row in development_runs[seeds[0]]
    }
    calibration_ids = {str(row["rollout_id"]) for row in records}
    if development_ids & calibration_ids:
        raise ValueError("Development and calibration rollout IDs overlap")
    primary_records, per_task = _select_per_task_class(
        records, args.min_per_class
    )
    selections = _threshold_payload(
        primary_records, args.target_fpr, safe_end, time_start
    )
    threshold_curves = {
        detector: selection.pop("curve")
        for detector, selection in selections.items()
    }
    _write_json(output / "threshold_curves.json", threshold_curves)

    runtime = output / "runtime"
    runtime.mkdir()
    checkpoints = {}
    for seed in seeds:
        source = Path(args.final_root).resolve() / f"indep_seed{seed}"
        destination = runtime / f"seed{seed}"
        destination.mkdir()
        for name in ("model_final.ckpt", "config.yaml", "metrics.json", "split_manifest.json"):
            shutil.copy2(source / name, destination / name)
        checkpoints[str(seed)] = {
            name: {
                "path": str((destination / name).resolve()),
                "sha256": _sha256(destination / name),
            }
            for name in ("model_final.ckpt", "config.yaml", "metrics.json", "split_manifest.json")
        }
    bundle = {
        "schema_version": 1,
        "protocol": "prospective_xiaomi_matched_fpr_shadow",
        "phase": "calibration_frozen",
        "model": "indep",
        "model_seeds": list(seeds),
        "binary_final_rollout_labels_only": True,
        "subtask_safe": False,
        "target_fpr": float(args.target_fpr),
        "safe_eligibility_end_fraction": safe_end,
        "time_eligibility_start_fraction": time_start,
        "task_horizons": horizons,
        "task_normalizations": normalizations,
        "time_risk_curves": _json_time_curves(time_curves),
        "threshold_selection": selections,
        "checkpoint_bundle": checkpoints,
        "source_final_root": str(Path(args.final_root).resolve()),
        "development_rollout_ids": sorted(development_ids),
        "calibration_score_root": str(Path(args.calibration_score_root).resolve()),
        "calibration_rollout_ids": sorted(row["rollout_id"] for row in records),
        "primary_calibration_rollout_ids": sorted(
            row["rollout_id"] for row in primary_records
        ),
        "calibration_seed_reset_identities": sorted(
            [list(value) for value in _identity_keys(records)], key=str
        ),
        "calibration_counts": {
            "retained_all": {
                "rollouts": len(records),
                "successes": sum(not row["failed"] for row in records),
                "failures": sum(row["failed"] for row in records),
            },
            "primary_balanced": {
                "rollouts": len(primary_records),
                "successes": sum(not row["failed"] for row in primary_records),
                "failures": sum(row["failed"] for row in primary_records),
            },
            "per_task": dict(sorted(per_task.items())),
        },
        "external_score_provenance": provenances,
        "frozen_fields": [
            "model checkpoints",
            "task normalization",
            "task horizons",
            "time-risk curves",
            "global event thresholds",
            "SAFE/time stage windows",
        ],
    }
    bundle_path = output / "runtime_bundle.json"
    _write_json(bundle_path, bundle)
    _write_jsonl(output / "calibration_predictions.jsonl", records)
    status = {
        "status": "calibration_complete",
        "runtime_bundle": str(bundle_path),
        "rollouts": len(primary_records),
        "retained_rollouts": len(records),
        "target_fpr": float(args.target_fpr),
    }
    _write_json(output / "status.json", status)
    return status


def _landmark_metrics(records, landmarks=(0.10, 0.25)):
    from sklearn.metrics import roc_auc_score

    rows = []
    for landmark in landmarks:
        selected = []
        for row in records:
            step = max(1, int(math.ceil(float(landmark) * row["horizon"])))
            if row["source_inferences"] >= step:
                selected.append((row, step))
        for detector, key in (("safe_only", "safe_scores"), ("time_only", "time_scores")):
            labels = [int(row["failed"]) for row, _ in selected]
            values = [float(row[key][step - 1]) for row, step in selected]
            pooled = float(roc_auc_score(labels, values)) if len(set(labels)) == 2 else None
            task_values = []
            for task in sorted({row["task_name"] for row, _ in selected}):
                subset = [(row, step) for row, step in selected if row["task_name"] == task]
                task_labels = [int(row["failed"]) for row, _ in subset]
                if len(set(task_labels)) != 2:
                    continue
                task_scores = [float(row[key][step - 1]) for row, step in subset]
                task_values.append(float(roc_auc_score(task_labels, task_scores)))
            rows.append(
                {
                    "detector": detector,
                    "landmark_fraction": float(landmark),
                    "pooled_roc_auc": pooled,
                    "task_macro_roc_auc": float(np.mean(task_values)) if task_values else None,
                    "supported_tasks": len(task_values),
                    "at_risk_rollouts": len(selected),
                }
            )
    return rows


def _bootstrap(records, thresholds, safe_end, time_start, samples, seed):
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for row in records:
        by_task[row["task_name"]].append(row)
    tasks = sorted(by_task)
    values = []
    for _ in range(int(samples)):
        chosen = rng.choice(tasks, size=len(tasks), replace=True)
        sample = []
        for index, task in enumerate(chosen):
            group = by_task[str(task)]
            selected = rng.choice(len(group), size=len(group), replace=True)
            for row_index in selected:
                row = dict(group[int(row_index)])
                row["rollout_id"] = f"bootstrap-{index}-{row['rollout_id']}"
                sample.append(row)
        if {bool(row["failed"]) for row in sample} != {False, True}:
            continue
        staged = event_metrics(
            sample,
            "staged_safe_time",
            thresholds["staged_safe_time"],
            safe_end=safe_end,
            time_start=time_start,
        )
        time = event_metrics(
            sample,
            "time_only",
            thresholds["time_only"],
            safe_end=safe_end,
            time_start=time_start,
        )
        values.append(
            {
                "adjusted_detection_delta": staged["missed_failure_adjusted_detection_fraction"]
                - time["missed_failure_adjusted_detection_fraction"],
                "tpr_delta": staged["true_positive_rate"] - time["true_positive_rate"],
                "fpr_delta": staged["false_positive_rate"] - time["false_positive_rate"],
            }
        )
    result = {}
    for key in ("adjusted_detection_delta", "tpr_delta", "fpr_delta"):
        array = np.asarray([row[key] for row in values], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(array)),
            "ci95": np.quantile(array, [0.025, 0.975]).tolist(),
        }
    result["replicates"] = len(values)
    return result


def evaluate(args):
    if int(args.test_per_class) < 1:
        raise ValueError("--test-per-class must be positive")
    if int(args.bootstrap_samples) < 1:
        raise ValueError("--bootstrap-samples must be positive")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(args.runtime_bundle).resolve()
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("phase") != "calibration_frozen":
        raise ValueError("Runtime bundle is not a frozen calibration artifact")
    seeds = tuple(int(seed) for seed in bundle["model_seeds"])
    safe_end = float(bundle["safe_eligibility_end_fraction"])
    time_start = float(bundle["time_eligibility_start_fraction"])
    # Validate copied runtime artifacts before reading prospective outcomes.
    for files in bundle["checkpoint_bundle"].values():
        for artifact in files.values():
            if _sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Frozen runtime artifact checksum changed: {artifact['path']}")
    runs, _, provenances = _load_seed_group(args.evaluation_score_root, seeds)
    expected_hashes = {
        str(seed): bundle["checkpoint_bundle"][str(seed)]["model_final.ckpt"][
            "sha256"
        ]
        for seed in seeds
    }
    _validate_score_provenance(
        provenances,
        expected_hashes,
        "prospective_test",
        seeds,
    )
    records = _assemble_external(
        runs,
        bundle["task_normalizations"],
        bundle["task_horizons"],
        bundle["time_risk_curves"],
    )
    expected_tasks = set(bundle["task_horizons"])
    observed_tasks = {str(row["task_name"]) for row in records}
    if observed_tasks != expected_tasks:
        raise ValueError(
            "Prospective score root lacks exact frozen task coverage: "
            f"missing={sorted(expected_tasks - observed_tasks)}, "
            f"unexpected={sorted(observed_tasks - expected_tasks)}"
        )
    development_ids = set(bundle.get("development_rollout_ids", []))
    calibration_ids = set(bundle["calibration_rollout_ids"])
    evaluation_ids = {row["rollout_id"] for row in records}
    if development_ids & evaluation_ids:
        raise ValueError("Development and prospective rollout IDs overlap")
    if calibration_ids & evaluation_ids:
        raise ValueError("Calibration and prospective rollout IDs overlap")
    calibration_identity = {
        tuple(value) for value in bundle["calibration_seed_reset_identities"]
    }
    overlap = calibration_identity & _identity_keys(records)
    if overlap:
        raise ValueError(f"Calibration and prospective seed/reset identities overlap: {sorted(overlap)[:3]}")
    thresholds = {
        detector: bundle["threshold_selection"][detector]["thresholds"]
        for detector in DETECTORS
    }
    primary_records, primary_selection = _select_per_task_class(
        records, args.test_per_class
    )
    overall = {
        detector: event_metrics(
            primary_records,
            detector,
            thresholds[detector],
            safe_end=safe_end,
            time_start=time_start,
        )
        for detector in DETECTORS
    }
    by_type = {}
    for task_type in ("atomic", "composite"):
        selected = [
            row for row in primary_records if row.get("task_type") == task_type
        ]
        if {bool(row["failed"]) for row in selected} == {False, True}:
            by_type[task_type] = {
                detector: event_metrics(
                    selected,
                    detector,
                    thresholds[detector],
                    safe_end=safe_end,
                    time_start=time_start,
                )
                for detector in DETECTORS
            }
    paired = paired_failure_comparison(
        primary_records,
        "staged_safe_time",
        thresholds["staged_safe_time"],
        "time_only",
        thresholds["time_only"],
        safe_end=safe_end,
        time_start=time_start,
    )
    bootstrap = _bootstrap(
        primary_records,
        thresholds,
        safe_end,
        time_start,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    calibration_metrics = {
        detector: bundle["threshold_selection"][detector]["validation_metrics"]
        for detector in DETECTORS
    }
    fpr_drift = {
        detector: overall[detector]["false_positive_rate"]
        - calibration_metrics[detector]["false_positive_rate"]
        for detector in DETECTORS
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "prospective_xiaomi_matched_fpr_shadow",
        "runtime_bundle": str(bundle_path),
        "evaluation_score_root": str(Path(args.evaluation_score_root).resolve()),
        "development_ids_disjoint": True,
        "calibration_ids_disjoint": True,
        "calibration_seed_reset_identities_disjoint": True,
        "policy_intervention": False,
        "thresholds_updated_on_test": False,
        "counts": {
            "primary_balanced": {
                "rollouts": len(primary_records),
                "successes": sum(not row["failed"] for row in primary_records),
                "failures": sum(row["failed"] for row in primary_records),
            },
            "retained_all": {
                "rollouts": len(records),
                "successes": sum(not row["failed"] for row in records),
                "failures": sum(row["failed"] for row in records),
            },
            "tasks": len({row["task_name"] for row in records}),
            "primary_selection": primary_selection,
        },
        "overall": overall,
        "all_retained_secondary": {
            detector: event_metrics(
                records,
                detector,
                thresholds[detector],
                safe_end=safe_end,
                time_start=time_start,
            )
            for detector in DETECTORS
        },
        "by_task_type": by_type,
        "landmarks": _landmark_metrics(primary_records),
        "paired_staged_minus_time": paired,
        "task_rollout_bootstrap": bootstrap,
        "calibration_to_test_fpr_drift": fpr_drift,
        "external_score_provenance": provenances,
        "success_criteria": {
            "fpr_at_most_0p05_or_wilson_compatible": (
                overall["staged_safe_time"]["false_positive_rate"] <= 0.05
                or (
                    overall["staged_safe_time"]["fpr_wilson_95"][0] <= 0.05
                    <= overall["staged_safe_time"]["fpr_wilson_95"][1]
                )
            ),
            "tpr_not_more_than_0p05_below_time": overall["staged_safe_time"]["true_positive_rate"]
            >= overall["time_only"]["true_positive_rate"] - 0.05,
            "adjusted_detection_at_least_0p10_earlier": overall["staged_safe_time"]["missed_failure_adjusted_detection_fraction"]
            <= overall["time_only"]["missed_failure_adjusted_detection_fraction"] - 0.10,
            "failure_recall_by_0p25_at_least_0p25": overall["staged_safe_time"]["failure_recall_by_landmark"]["0.25"] >= 0.25,
            "bootstrap_adjusted_detection_favors_staged": bootstrap["adjusted_detection_delta"]["ci95"][1] < 0.0,
        },
    }
    result["success_criteria"]["all_pass"] = all(result["success_criteria"].values())
    _write_json(output / "analysis.json", result)
    _write_jsonl(output / "event_predictions.jsonl", records)
    _write_json(
        output / "status.json",
        {"status": "complete", "analysis": str((output / "analysis.json").resolve())},
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--final-root", required=True)
    calibration.add_argument("--calibration-score-root", required=True)
    calibration.add_argument("--output-dir", required=True)
    calibration.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    calibration.add_argument("--target-fpr", type=float, default=0.05)
    calibration.add_argument("--safe-end", type=float, default=0.25)
    calibration.add_argument("--time-start", type=float, default=0.50)
    calibration.add_argument("--time-prior", type=float, default=0.5)
    calibration.add_argument("--min-per-class", type=int, default=3)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--runtime-bundle", required=True)
    evaluation.add_argument("--evaluation-score-root", required=True)
    evaluation.add_argument("--output-dir", required=True)
    evaluation.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluation.add_argument("--bootstrap-seed", type=int, default=0)
    evaluation.add_argument("--test-per-class", type=int, default=10)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = calibrate(args) if args.phase == "calibrate" else evaluate(args)
    except (FileExistsError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(_json_value(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
