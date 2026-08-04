"""Evaluate causal Subtask-SAFE with parent calibration and honest baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .calibrate_seen_tasks import load_score_records
    from .subtask_safe_evaluation import detection_event, parent_id
    from .train_seen_tasks import write_json
except ImportError:
    from calibrate_seen_tasks import load_score_records
    from subtask_safe_evaluation import detection_event, parent_id
    from train_seen_tasks import write_json


def _safe_auc(labels, scores, metric=roc_auc_score):
    return (
        None
        if len(labels) < 2 or len(set(labels)) < 2
        else float(metric(labels, scores))
    )


def _risk(record):
    values = np.asarray(record["scores"], dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid score trajectory for {record.get('rollout_id')}")
    return float(np.max(values))


def validate_causal_records(records, target_prefix):
    seen = set()
    for record in records:
        source = record.get("source_segment_id")
        prefix = record.get("causal_prefix_inferences")
        parent = record.get("parent_rollout_id")
        if not source or not parent or prefix is None:
            raise ValueError(
                "Causal gate evaluation requires source_segment_id, "
                "parent_rollout_id, and causal_prefix_inferences in scores.jsonl"
            )
        key = (record["split"], str(source), int(prefix))
        if key in seen:
            raise ValueError(f"Duplicate causal source/prefix identity: {key}")
        seen.add(key)
    target = [
        record
        for record in records
        if int(record["causal_prefix_inferences"]) == int(target_prefix)
    ]
    if not target:
        raise ValueError(f"No records exist at causal prefix {target_prefix}")
    return target


def _training_stage_prior(train_records, prefix):
    """Fit P(target failure | active semantic stage, prefix) on training only."""
    grouped = defaultdict(list)
    for record in train_records:
        if int(record["causal_prefix_inferences"]) == int(prefix):
            grouped[record["task_name"]].append(record)
    if not grouped:
        raise ValueError("No training records are available at the requested prefix")
    return {
        stage: {
            "risk": float(
                (sum(bool(item["failed"]) for item in values) + 1) / (len(values) + 2)
            ),
            "support": len(values),
            "source": "training target prevalence by stage and prefix",
        }
        for stage, values in sorted(grouped.items())
    }


def _source_length(record, fallback):
    value = record.get("source_segment_num_inferences")
    return int(value) if value is not None else int(fallback)


def _training_duration_progress_model(train_records):
    """Fit successful-stage duration scales without using evaluation labels."""
    by_source = defaultdict(list)
    for record in train_records:
        by_source[str(record["source_segment_id"])].append(record)
    successful_lengths = defaultdict(list)
    global_lengths = []
    for values in by_source.values():
        maximum_prefix = max(int(value["causal_prefix_inferences"]) for value in values)
        first = values[0]
        source_failed = bool(first.get("source_segment_failed", first["failed"]))
        if source_failed:
            continue
        length = _source_length(first, maximum_prefix)
        successful_lengths[first["task_name"]].append(length)
        global_lengths.append(length)
    if not global_lengths:
        raise ValueError(
            "Duration-progress baseline needs successful training segments"
        )
    global_median = float(np.median(global_lengths))
    return {
        "per_stage_median_success_inferences": {
            stage: float(np.median(values))
            for stage, values in sorted(successful_lengths.items())
        },
        "global_median_success_inferences": global_median,
        "source": "successful training source-segment durations only",
    }


def _duration_progress(record, model):
    scale = model["per_stage_median_success_inferences"].get(
        record["task_name"], model["global_median_success_inferences"]
    )
    return float(int(record["causal_prefix_inferences"]) / max(1.0, scale))


def _calibration_parent_partition(records, *, fraction, seed):
    """Reserve complete successful test parents for threshold calibration."""
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("Calibration parent fraction must lie strictly in (0, 1)")
    grouped = defaultdict(list)
    for record in records:
        if record["split"] == "test":
            grouped[parent_id(record)].append(record)
    strata = defaultdict(list)
    for key, values in grouped.items():
        first = values[0]
        parent_failed = bool(first.get("parent_rollout_failed", first["failed"]))
        if not parent_failed:
            task = first.get("parent_task_name") or first["task_name"]
            strata[str(task)].append(key)
    selected = []
    per_task = {}
    for task, keys in sorted(strata.items()):
        keys = sorted(keys)
        random.Random(f"{int(seed)}:{task}").shuffle(keys)
        count = (
            0
            if len(keys) < 2
            else min(len(keys) - 1, max(1, int(round(len(keys) * float(fraction)))))
        )
        selected.extend(keys[:count])
        per_task[task] = {
            "successful_test_parents": len(keys),
            "calibration_parents": count,
            "remaining_successful_evaluation_parents": len(keys) - count,
        }
    if not selected:
        raise ValueError("No successful test parent can be reserved for calibration")
    selected = sorted(selected)
    evaluation = sorted(set(grouped) - set(selected))
    if not evaluation:
        raise ValueError("Parent calibration left no evaluation parents")
    return {
        "calibration_parent_ids": selected,
        "evaluation_parent_ids": evaluation,
        "per_parent_task": per_task,
        "fraction": float(fraction),
        "seed": int(seed),
        "split_unit": "parent_rollout",
        "calibration_candidates": "successful held-out parents only",
    }


def _quantile_threshold(success_scores, target_fpr):
    values = np.sort(np.asarray(success_scores, dtype=np.float64))
    if not len(values):
        raise ValueError("Threshold fitting requires successful calibration prefixes")
    rank = int(math.ceil((len(values) + 1) * (1.0 - float(target_fpr)))) - 1
    return float(values[min(len(values) - 1, max(0, rank))])


def _metric_row(model, seed, prefix, test, stage_prior, duration_model):
    labels = [int(record["failed"]) for record in test]
    safe_scores = [_risk(record) for record in test]
    prior_scores = [stage_prior[record["task_name"]]["risk"] for record in test]
    duration_scores = [_duration_progress(record, duration_model) for record in test]
    return {
        "model": model,
        "seed": int(seed),
        "prefix": int(prefix),
        "segments": len(test),
        "parents": len({parent_id(record) for record in test}),
        "successes": labels.count(0),
        "failures": labels.count(1),
        "safe_roc_auc": _safe_auc(labels, safe_scores),
        "safe_average_precision": _safe_auc(
            labels, safe_scores, average_precision_score
        ),
        "stage_prior_roc_auc": _safe_auc(labels, prior_scores),
        "stage_prior_average_precision": _safe_auc(
            labels, prior_scores, average_precision_score
        ),
        "duration_progress_roc_auc": _safe_auc(labels, duration_scores),
        "duration_progress_average_precision": _safe_auc(
            labels, duration_scores, average_precision_score
        ),
    }


def _operating_point(model, seed, calibration, test, target_prefix, target_fpr):
    calibration_target = [
        record
        for record in calibration
        if int(record["causal_prefix_inferences"]) == int(target_prefix)
        and not bool(record["failed"])
    ]
    threshold = _quantile_threshold(
        [_risk(record) for record in calibration_target], target_fpr
    )
    failures = [record for record in test if bool(record["failed"])]
    successes = [record for record in test if not bool(record["failed"])]
    # Strict crossing avoids counting calibration-boundary ties as detections.
    detected_failures = [record for record in failures if _risk(record) > threshold]
    false_alarms = [record for record in successes if _risk(record) > threshold]
    leads = []
    lead_steps = []
    for record in detected_failures:
        scores = np.asarray(record["scores"], dtype=np.float64)
        detection_index = int(np.flatnonzero(scores > threshold)[0])
        event = detection_event(record, detection_index)
        if event["normalized_lead_time"] is not None:
            leads.append(event["normalized_lead_time"])
            lead_steps.append(event["lead_environment_steps"])
    tpr = len(detected_failures) / len(failures) if failures else None
    fpr = len(false_alarms) / len(successes) if successes else None
    return {
        "model": model,
        "seed": int(seed),
        "prefix": int(target_prefix),
        "threshold_source": "successful parent-disjoint held-out calibration prefixes",
        "target_calibration_fpr": float(target_fpr),
        "threshold": threshold,
        "calibration_successes": len(calibration_target),
        "evaluation_failures": len(failures),
        "evaluation_successes": len(successes),
        "tpr": tpr,
        "fpr": fpr,
        "balanced_accuracy": (
            None if tpr is None or fpr is None else float((tpr + 1.0 - fpr) / 2.0)
        ),
        "detected_failures": len(detected_failures),
        "false_alarms": len(false_alarms),
        "normalized_lead_mean": float(np.mean(leads)) if leads else None,
        "lead_environment_steps_mean": (
            float(np.mean(lead_steps)) if lead_steps else None
        ),
    }


def _bootstrap_deltas(seed_payloads, target_prefix, replicates, bootstrap_seed):
    rng = np.random.RandomState(bootstrap_seed)
    seeds = sorted(seed_payloads)
    values = defaultdict(list)
    for _ in range(int(replicates)):
        payload = seed_payloads[seeds[int(rng.randint(len(seeds)))]]
        test = payload["test"]
        parents = sorted({parent_id(record) for record in test})
        sampled_parents = rng.choice(parents, size=len(parents), replace=True)
        by_parent = defaultdict(list)
        for record in test:
            by_parent[parent_id(record)].append(record)
        sampled = [record for key in sampled_parents for record in by_parent[str(key)]]
        labels = [int(record["failed"]) for record in sampled]
        scores = {
            "safe": [_risk(record) for record in sampled],
            "stage_prior": [
                payload["stage_prior"][record["task_name"]]["risk"]
                for record in sampled
            ],
            "duration_progress": [
                _duration_progress(record, payload["duration_model"])
                for record in sampled
            ],
        }
        aucs = {name: _safe_auc(labels, score) for name, score in scores.items()}
        if any(value is None for value in aucs.values()):
            continue
        for name, value in aucs.items():
            values[f"{name}_roc_auc"].append(value)
        values["safe_minus_stage_prior_roc_auc"].append(
            aucs["safe"] - aucs["stage_prior"]
        )
        values["safe_minus_duration_progress_roc_auc"].append(
            aucs["safe"] - aucs["duration_progress"]
        )
    if not values["safe_roc_auc"]:
        raise ValueError("Parent bootstrap produced no estimable replicates")

    def block(items):
        items = np.asarray(items, dtype=np.float64)
        return {
            "mean": float(np.mean(items)),
            "ci_95_low": float(np.percentile(items, 2.5)),
            "ci_95_high": float(np.percentile(items, 97.5)),
            "replicates": len(items),
        }

    result = {
        "sampling_unit": "parent_rollout",
        "model_seed_resampled": True,
        "prefix": int(target_prefix),
        **{name: block(items) for name, items in values.items()},
    }
    # Compatibility alias for v2 consumers; it now means the genuine duration
    # progress control, never the stage-prior control.
    result["safe_minus_elapsed_roc_auc"] = result[
        "safe_minus_duration_progress_roc_auc"
    ]
    return result


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _at_least(value, threshold):
    return value is not None and value >= threshold


def evaluate_root(
    final_root,
    *,
    models=("indep", "lstm"),
    seeds=(0, 1, 2),
    prefixes=(1, 2, 4, 8, 16),
    target_prefix=8,
    target_fpr=0.10,
    min_roc=0.65,
    min_delta=0.05,
    min_tpr=0.40,
    min_lead=0.25,
    calibration_parent_fraction=0.30,
    calibration_seed=0,
    bootstrap_replicates=2000,
    bootstrap_seed=0,
):
    root = Path(final_root).resolve()
    prefix_rows = []
    operating_rows = []
    model_results = {}
    reference_train = None
    reference_test = None
    reference_partition = None
    for model in models:
        payloads = {}
        for seed in seeds:
            records = load_score_records(root / f"{model}_seed{seed}" / "scores.jsonl")
            validate_causal_records(records, int(target_prefix))
            train = [record for record in records if record["split"] == "train"]
            partition = _calibration_parent_partition(
                records, fraction=calibration_parent_fraction, seed=calibration_seed
            )
            if reference_partition is None:
                reference_partition = partition
            elif (
                partition["calibration_parent_ids"]
                != reference_partition["calibration_parent_ids"]
                or partition["evaluation_parent_ids"]
                != reference_partition["evaluation_parent_ids"]
            ):
                raise ValueError("Model seeds do not use identical calibration parents")
            calibration_parents = set(partition["calibration_parent_ids"])
            evaluation_parents = set(partition["evaluation_parent_ids"])
            calibration = [
                record
                for record in records
                if record["split"] == "test"
                and parent_id(record) in calibration_parents
            ]
            evaluation = [
                record
                for record in records
                if record["split"] == "test" and parent_id(record) in evaluation_parents
            ]
            duration_model = _training_duration_progress_model(train)
            for prefix in prefixes:
                stage_prior = _training_stage_prior(train, int(prefix))
                available_stages = set(stage_prior)
                test = [
                    record
                    for record in evaluation
                    if int(record["causal_prefix_inferences"]) == int(prefix)
                    and record["task_name"] in available_stages
                ]
                if test:
                    prefix_rows.append(
                        _metric_row(
                            model, seed, prefix, test, stage_prior, duration_model
                        )
                    )
            stage_prior = _training_stage_prior(train, int(target_prefix))
            test_target = [
                record
                for record in evaluation
                if int(record["causal_prefix_inferences"]) == int(target_prefix)
                and record["task_name"] in stage_prior
            ]
            calibration_target = [
                record
                for record in calibration
                if int(record["causal_prefix_inferences"]) == int(target_prefix)
                and record["task_name"] in stage_prior
            ]
            if not test_target or not calibration_target:
                raise ValueError("Calibration/evaluation lacks target-prefix records")
            train_identity = sorted(
                (
                    record["source_segment_id"],
                    parent_id(record),
                    record["task_name"],
                    bool(record["failed"]),
                )
                for record in train
                if int(record["causal_prefix_inferences"]) == int(target_prefix)
            )
            test_identity = sorted(
                (
                    record["source_segment_id"],
                    parent_id(record),
                    record["task_name"],
                    bool(record["failed"]),
                )
                for record in test_target
            )
            if reference_train is None:
                reference_train, reference_test = train_identity, test_identity
            elif train_identity != reference_train or test_identity != reference_test:
                raise ValueError("Model seeds do not use identical causal identities")
            operating_rows.append(
                _operating_point(
                    model,
                    seed,
                    calibration_target,
                    test_target,
                    int(target_prefix),
                    target_fpr,
                )
            )
            payloads[int(seed)] = {
                "test": test_target,
                "stage_prior": stage_prior,
                "duration_model": duration_model,
            }

        target_rows = [
            row
            for row in prefix_rows
            if row["model"] == model and row["prefix"] == int(target_prefix)
        ]
        ops = [row for row in operating_rows if row["model"] == model]
        bootstrap = _bootstrap_deltas(
            payloads, target_prefix, bootstrap_replicates, bootstrap_seed
        )
        safe_roc = _mean([row["safe_roc_auc"] for row in target_rows])
        stage_roc = _mean([row["stage_prior_roc_auc"] for row in target_rows])
        duration_roc = _mean([row["duration_progress_roc_auc"] for row in target_rows])
        observed = {
            "causal_prefix_roc_auc": safe_roc,
            "stage_prior_roc_auc": stage_roc,
            "duration_progress_roc_auc": duration_roc,
            "safe_minus_stage_prior_roc_auc": (
                None if safe_roc is None or stage_roc is None else safe_roc - stage_roc
            ),
            "safe_minus_duration_progress_roc_auc": (
                None
                if safe_roc is None or duration_roc is None
                else safe_roc - duration_roc
            ),
            # Compatibility aliases, with corrected duration semantics.
            "elapsed_roc_auc": duration_roc,
            "safe_minus_elapsed_roc_auc": (
                None
                if safe_roc is None or duration_roc is None
                else safe_roc - duration_roc
            ),
            "tpr": _mean([row["tpr"] for row in ops]),
            "fpr": _mean([row["fpr"] for row in ops]),
            "normalized_lead": _mean([row["normalized_lead_mean"] for row in ops]),
        }
        gates = {
            "causal_prefix_roc_at_least_minimum": _at_least(safe_roc, min_roc),
            "beats_stage_prior_by_minimum_delta": _at_least(
                observed["safe_minus_stage_prior_roc_auc"], min_delta
            ),
            "beats_duration_progress_by_minimum_delta": _at_least(
                observed["safe_minus_duration_progress_roc_auc"], min_delta
            ),
            "stage_prior_parent_bootstrap_excludes_zero": (
                bootstrap["safe_minus_stage_prior_roc_auc"]["ci_95_low"] > 0.0
            ),
            "duration_parent_bootstrap_excludes_zero": (
                bootstrap["safe_minus_duration_progress_roc_auc"]["ci_95_low"] > 0.0
            ),
            "tpr_at_least_minimum": _at_least(observed["tpr"], min_tpr),
            "fpr_at_most_target": (
                observed["fpr"] is not None and observed["fpr"] <= target_fpr
            ),
            "normalized_lead_at_least_minimum": _at_least(
                observed["normalized_lead"], min_lead
            ),
        }
        model_results[model] = {
            "observed": observed,
            "bootstrap": bootstrap,
            "gates": gates,
            "continue_to_recovery_integration": all(gates.values()),
            "recommendation": (
                "proceed to online recovery integration"
                if all(gates.values())
                else "change representation or objective before collecting blindly"
            ),
        }
    return {
        "schema_version": 2,
        "protocol": (
            "causal fixed-prefix evaluation; successful parent-disjoint threshold "
            "calibration; training-only stage-prior and duration-progress baselines"
        ),
        "final_root": str(root),
        "target_prefix": int(target_prefix),
        "identical_target_identities_across_models_and_seeds": True,
        "calibration": reference_partition,
        "thresholds": {
            "minimum_causal_prefix_roc_auc": float(min_roc),
            "minimum_safe_minus_each_baseline_roc_auc": float(min_delta),
            "minimum_tpr": float(min_tpr),
            "maximum_fpr": float(target_fpr),
            "minimum_normalized_lead": float(min_lead),
        },
        "baseline_definitions": {
            "stage_prior": "training target prevalence by semantic stage and prefix",
            "duration_progress": (
                "prefix divided by the median successful source-segment duration "
                "for that stage, fitted on training only"
            ),
        },
        "models": model_results,
        "prefix_metrics": prefix_rows,
        "operating_points": operating_rows,
    }


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    grouped = defaultdict(list)
    for row in summary["prefix_metrics"]:
        grouped[row["model"]].append(row)
    fig, axes = plt.subplots(
        1, len(grouped), figsize=(6 * len(grouped), 4.5), squeeze=False
    )
    curves = (
        ("safe_roc_auc", "Causal SAFE", "#2563EB"),
        ("stage_prior_roc_auc", "Stage prior", "#DC2626"),
        ("duration_progress_roc_auc", "Duration progress", "#D97706"),
    )
    for ax, (model, rows) in zip(axes[0], sorted(grouped.items())):
        for key, label, color in curves:
            by_prefix = defaultdict(list)
            for row in rows:
                if row[key] is not None:
                    by_prefix[int(row["prefix"])].append(row[key])
            x = sorted(by_prefix)
            y = [float(np.mean(by_prefix[value])) for value in x]
            ax.plot(x, y, marker="o", label=label, color=color)
        ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
        ax.axhline(
            summary["thresholds"]["minimum_causal_prefix_roc_auc"],
            color="#059669",
            linestyle=":",
            linewidth=1,
        )
        ax.set_title(model.upper())
        ax.set_xlabel("Causal prefix (policy inferences)")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.25, 1.0)
        ax.legend()
    fig.tight_layout()
    path = output / "causal_prefix_vs_baselines.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(summary, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    _write_csv(output / "prefix_metrics.csv", summary["prefix_metrics"])
    _write_csv(output / "operating_points.csv", summary["operating_points"])
    plot_path = _plot(summary, output)
    lines = ["# Causal Subtask-SAFE continuation gates", ""]
    for model, result in summary["models"].items():
        lines.extend((f"## {model}", ""))
        for name, passed in result["gates"].items():
            lines.append(f"- [{'x' if passed else ' '}] {name}")
        lines.extend(("", f"Decision: **{result['recommendation']}**", ""))
    (output / "gate_report.md").write_text("\n".join(lines) + "\n")
    return {
        "summary": str(output / "summary.json"),
        "prefix_metrics": str(output / "prefix_metrics.csv"),
        "operating_points": str(output / "operating_points.csv"),
        "plot": str(plot_path),
        "gate_report": str(output / "gate_report.md"),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["indep", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--prefixes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--target-prefix", type=int, default=8)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--min-roc", type=float, default=0.65)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--min-tpr", type=float, default=0.40)
    parser.add_argument("--min-lead", type=float, default=0.25)
    parser.add_argument("--calibration-parent-fraction", type=float, default=0.30)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = evaluate_root(
        args.final_root,
        models=args.models,
        seeds=args.seeds,
        prefixes=args.prefixes,
        target_prefix=args.target_prefix,
        target_fpr=args.target_fpr,
        min_roc=args.min_roc,
        min_delta=args.min_delta,
        min_tpr=args.min_tpr,
        min_lead=args.min_lead,
        calibration_parent_fraction=args.calibration_parent_fraction,
        calibration_seed=args.calibration_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(write_outputs(summary, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
