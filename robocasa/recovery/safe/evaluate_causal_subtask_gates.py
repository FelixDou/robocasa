"""Evaluate causal Subtask-SAFE against elapsed time and continuation gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

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
        if int(record["causal_prefix_inferences"]) == target_prefix
    ]
    if not target:
        raise ValueError(f"No records exist at causal prefix {target_prefix}")
    return target


def _training_elapsed_model(train_records, target_prefix):
    grouped = defaultdict(list)
    for record in train_records:
        if int(record["causal_prefix_inferences"]) == target_prefix:
            grouped[record["task_name"]].append(record)
    if not grouped:
        raise ValueError("No training records are available at the target prefix")
    return {
        stage: {
            "risk": float(
                (sum(bool(item["failed"]) for item in values) + 1) / (len(values) + 2)
            ),
            "support": len(values),
        }
        for stage, values in sorted(grouped.items())
    }


def _quantile_threshold(success_scores, target_fpr):
    values = np.sort(np.asarray(success_scores, dtype=np.float64))
    if not len(values):
        raise ValueError("Threshold fitting requires successful training prefixes")
    rank = int(math.ceil((len(values) + 1) * (1.0 - float(target_fpr)))) - 1
    return float(values[min(len(values) - 1, max(0, rank))])


def _metric_row(model, seed, prefix, test, elapsed_model, target_fpr):
    labels = [int(record["failed"]) for record in test]
    safe_scores = [_risk(record) for record in test]
    elapsed_scores = [elapsed_model[record["task_name"]]["risk"] for record in test]
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
        "elapsed_roc_auc": _safe_auc(labels, elapsed_scores),
        "elapsed_average_precision": _safe_auc(
            labels, elapsed_scores, average_precision_score
        ),
    }


def _operating_point(model, seed, train, test, target_prefix, target_fpr):
    train_target = [
        record
        for record in train
        if int(record["causal_prefix_inferences"]) == target_prefix
        and not bool(record["failed"])
    ]
    threshold = _quantile_threshold(
        [_risk(record) for record in train_target], target_fpr
    )
    failures = [record for record in test if bool(record["failed"])]
    successes = [record for record in test if not bool(record["failed"])]
    detected_failures = [record for record in failures if _risk(record) >= threshold]
    false_alarms = [record for record in successes if _risk(record) >= threshold]
    leads = []
    lead_steps = []
    for record in detected_failures:
        scores = np.asarray(record["scores"], dtype=np.float64)
        detection_index = int(np.flatnonzero(scores >= threshold)[0])
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
        "threshold_source": "successful training causal prefixes only",
        "target_training_fpr": float(target_fpr),
        "threshold": threshold,
        "training_successes": len(train_target),
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
        "lead_environment_steps_mean": float(np.mean(lead_steps))
        if lead_steps
        else None,
    }


def _bootstrap_delta(seed_payloads, target_prefix, replicates, bootstrap_seed):
    rng = np.random.RandomState(bootstrap_seed)
    seeds = sorted(seed_payloads)
    deltas = []
    safe_values = []
    elapsed_values = []
    for _ in range(int(replicates)):
        seed = seeds[int(rng.randint(len(seeds)))]
        payload = seed_payloads[seed]
        test = payload["test"]
        parents = sorted({parent_id(record) for record in test})
        sampled_parents = rng.choice(parents, size=len(parents), replace=True)
        by_parent = defaultdict(list)
        for record in test:
            by_parent[parent_id(record)].append(record)
        sampled = [record for key in sampled_parents for record in by_parent[str(key)]]
        labels = [int(record["failed"]) for record in sampled]
        safe = _safe_auc(labels, [_risk(record) for record in sampled])
        elapsed = _safe_auc(
            labels,
            [
                payload["elapsed_model"][record["task_name"]]["risk"]
                for record in sampled
            ],
        )
        if safe is not None and elapsed is not None:
            safe_values.append(safe)
            elapsed_values.append(elapsed)
            deltas.append(safe - elapsed)
    if not deltas:
        raise ValueError("Parent bootstrap produced no estimable replicates")

    def block(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "ci_95_low": float(np.percentile(values, 2.5)),
            "ci_95_high": float(np.percentile(values, 97.5)),
            "replicates": len(values),
        }

    return {
        "sampling_unit": "parent_rollout",
        "model_seed_resampled": True,
        "prefix": int(target_prefix),
        "safe_roc_auc": block(safe_values),
        "elapsed_roc_auc": block(elapsed_values),
        "safe_minus_elapsed_roc_auc": block(deltas),
    }


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


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
    bootstrap_replicates=2000,
    bootstrap_seed=0,
):
    root = Path(final_root).resolve()
    prefix_rows = []
    operating_rows = []
    model_results = {}
    reference_target_train = None
    reference_target_test = None
    for model in models:
        payloads = {}
        for seed in seeds:
            path = root / f"{model}_seed{seed}" / "scores.jsonl"
            records = load_score_records(path)
            validate_causal_records(records, int(target_prefix))
            train = [record for record in records if record["split"] == "train"]
            elapsed_model = _training_elapsed_model(train, int(target_prefix))
            for prefix in prefixes:
                prefix_elapsed_model = _training_elapsed_model(train, int(prefix))
                available_stages = set(prefix_elapsed_model)
                test = [
                    record
                    for record in records
                    if record["split"] == "test"
                    and int(record["causal_prefix_inferences"]) == int(prefix)
                    and record["task_name"] in available_stages
                ]
                if test:
                    prefix_rows.append(
                        _metric_row(
                            model,
                            seed,
                            prefix,
                            test,
                            prefix_elapsed_model,
                            target_fpr,
                        )
                    )
            available_stages = set(elapsed_model)
            test_target = [
                record
                for record in records
                if record["split"] == "test"
                and int(record["causal_prefix_inferences"]) == int(target_prefix)
                and record["task_name"] in available_stages
            ]
            if not test_target:
                raise ValueError(
                    f"No {model} seed {seed} test records at target prefix"
                )
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
            if reference_target_train is None:
                reference_target_train = train_identity
                reference_target_test = test_identity
            elif (
                train_identity != reference_target_train
                or test_identity != reference_target_test
            ):
                raise ValueError(
                    "Model seeds do not use identical causal train/test identities"
                )
            operating_rows.append(
                _operating_point(
                    model, seed, train, test_target, int(target_prefix), target_fpr
                )
            )
            payloads[int(seed)] = {
                "train": train,
                "test": test_target,
                "elapsed_model": elapsed_model,
            }

        target_rows = [
            row
            for row in prefix_rows
            if row["model"] == model and row["prefix"] == int(target_prefix)
        ]
        ops = [row for row in operating_rows if row["model"] == model]
        bootstrap = _bootstrap_delta(
            payloads, target_prefix, bootstrap_replicates, bootstrap_seed
        )
        observed = {
            "causal_prefix_roc_auc": _mean(
                [row["safe_roc_auc"] for row in target_rows]
            ),
            "elapsed_roc_auc": _mean([row["elapsed_roc_auc"] for row in target_rows]),
            "safe_minus_elapsed_roc_auc": _mean(
                [row["safe_roc_auc"] - row["elapsed_roc_auc"] for row in target_rows]
            ),
            "tpr": _mean([row["tpr"] for row in ops]),
            "fpr": _mean([row["fpr"] for row in ops]),
            "normalized_lead": _mean([row["normalized_lead_mean"] for row in ops]),
        }
        delta_ci_low = bootstrap["safe_minus_elapsed_roc_auc"]["ci_95_low"]
        gates = {
            "causal_prefix_roc_at_least_minimum": observed["causal_prefix_roc_auc"]
            >= min_roc,
            "beats_elapsed_by_minimum_delta": observed["safe_minus_elapsed_roc_auc"]
            >= min_delta,
            "paired_parent_bootstrap_excludes_zero": delta_ci_low > 0.0,
            "tpr_at_least_minimum": observed["tpr"] >= min_tpr,
            "fpr_at_most_target": observed["fpr"] <= target_fpr,
            "normalized_lead_at_least_minimum": (
                observed["normalized_lead"] is not None
                and observed["normalized_lead"] >= min_lead
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
        "schema_version": 1,
        "protocol": "causal fixed-prefix evaluation with training-only elapsed baseline",
        "final_root": str(root),
        "target_prefix": int(target_prefix),
        "identical_target_identities_across_models_and_seeds": True,
        "thresholds": {
            "minimum_causal_prefix_roc_auc": float(min_roc),
            "minimum_safe_minus_elapsed_roc_auc": float(min_delta),
            "minimum_tpr": float(min_tpr),
            "maximum_fpr": float(target_fpr),
            "minimum_normalized_lead": float(min_lead),
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
    for ax, (model, rows) in zip(axes[0], sorted(grouped.items())):
        for key, label, color in (
            ("safe_roc_auc", "Causal SAFE", "#2563EB"),
            ("elapsed_roc_auc", "Elapsed time", "#DC2626"),
        ):
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
    path = output / "causal_prefix_vs_elapsed.png"
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
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(write_outputs(summary, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
