"""Analyze raw and task-normalized SAFE scores without refitting on test data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


MODELS = ("indep", "lstm")
MODEL_LABELS = {"indep": "MLP", "lstm": "LSTM"}
VARIANTS = ("raw", "normalized")
OUTCOME_COLORS = {False: "#2563A6", True: "#D97706"}


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    records = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No score records found in {path}")
    ids = [record["rollout_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate rollout IDs in {path}")
    return records


def score_array(record, variant):
    key = "scores"
    values = np.asarray(record[key], dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid {variant} scores for {record.get('rollout_id')}")
    if variant == "raw":
        cutoff = min(int(record["task_min_step"]), len(values))
        if cutoff < 1:
            raise ValueError(f"Invalid task_min_step for {record.get('rollout_id')}")
        values = values[:cutoff]
    return values


def original_split(record, variant):
    return record["split"] if variant == "raw" else record["original_split"]


def duration(record):
    """Match the final-refit duration baseline: number of policy inferences."""
    return int(record.get("full_num_inferences", record["num_inferences"]))


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values_a, values_b):
    if len(values_a) < 2:
        return None
    a = _rankdata(values_a)
    b = _rankdata(values_b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else None


def safe_auc(function, labels, values):
    return None if len(set(int(value) for value in labels)) < 2 else float(
        function(labels, values)
    )


def load_score_sets(final_root, calibration_root):
    final_root = Path(final_root).resolve()
    calibration_root = Path(calibration_root).resolve()
    loaded = {}
    reference_signature = None
    for model in MODELS:
        calibration_summary = load_json(calibration_root / model / "summary.json")
        if calibration_summary["normalization"] != "training_task_early_max_z":
            raise ValueError(f"Unexpected normalization for {model}")
        for seed in (0, 1, 2):
            raw = load_jsonl(final_root / f"{model}_seed{seed}" / "scores.jsonl")
            normalized = load_jsonl(
                calibration_root
                / model
                / f"{model}_seed{seed}"
                / "normalized_scores.jsonl"
            )
            raw_by_id = {record["rollout_id"]: record for record in raw}
            normalized_by_id = {
                record["rollout_id"]: record for record in normalized
            }
            if set(raw_by_id) != set(normalized_by_id):
                raise ValueError(f"Raw/normalized rollout mismatch for {model} seed {seed}")
            for rollout_id, raw_record in raw_by_id.items():
                normalized_record = normalized_by_id[rollout_id]
                if (
                    normalized_record["original_split"] != raw_record["split"]
                    or normalized_record["task_name"] != raw_record["task_name"]
                    or bool(normalized_record["failed"])
                    != bool(raw_record["failed"])
                    or duration(normalized_record) != duration(raw_record)
                ):
                    raise ValueError(
                        f"Raw/normalized metadata mismatch for {rollout_id}"
                    )
            signature = {
                rollout_id: (
                    raw_by_id[rollout_id]["split"],
                    raw_by_id[rollout_id]["task_name"],
                    bool(raw_by_id[rollout_id]["failed"]),
                    duration(raw_by_id[rollout_id]),
                )
                for rollout_id in raw_by_id
            }
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise ValueError("Final runs do not use identical rollout identities")
            loaded[(model, seed, "raw")] = raw
            loaded[(model, seed, "normalized")] = normalized
    return loaded, reference_signature


def rollout_rows(score_sets):
    rows = []
    for (model, seed, variant), records in sorted(score_sets.items()):
        for record in records:
            scores = score_array(record, variant)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "score_variant": variant,
                    "rollout_id": record["rollout_id"],
                    "original_split": original_split(record, variant),
                    "calibration_split": (
                        record["split"] if variant == "normalized" else ""
                    ),
                    "task_name": record["task_name"],
                    "failed": int(bool(record["failed"])),
                    "num_inferences": duration(record),
                    "matched_inferences": len(scores),
                    "max_score": float(np.max(scores)),
                    "mean_score": float(np.mean(scores)),
                    "final_score": float(scores[-1]),
                }
            )
    return rows


def per_task_metrics(score_sets):
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for (model, seed, variant), records in sorted(score_sets.items()):
        test = [
            record
            for record in records
            if original_split(record, variant) == "test"
        ]
        for task in sorted({record["task_name"] for record in test}):
            selected = [record for record in test if record["task_name"] == task]
            labels = [int(bool(record["failed"])) for record in selected]
            maxima = [float(np.max(score_array(record, variant))) for record in selected]
            durations = [duration(record) for record in selected]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "score_variant": variant,
                    "task_name": task,
                    "rollouts": len(selected),
                    "successes": sum(not value for value in labels),
                    "failures": sum(labels),
                    "score_roc_auc": safe_auc(roc_auc_score, labels, maxima),
                    "score_auprc": safe_auc(
                        average_precision_score, labels, maxima
                    ),
                    "duration_roc_auc": safe_auc(
                        roc_auc_score, labels, durations
                    ),
                    "score_duration_spearman": spearman(maxima, durations),
                }
            )
    return rows


def conditional_duration_metrics(score_sets):
    """Fit confound probes on training only and evaluate the fixed outer test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for (model, seed, variant), records in sorted(score_sets.items()):
        tasks = sorted({record["task_name"] for record in records})
        task_index = {task: index for index, task in enumerate(tasks)}

        def matrix(selected, feature_set):
            result = []
            for record in selected:
                score = float(np.max(score_array(record, variant)))
                elapsed = math.log1p(duration(record))
                one_hot = [0.0] * max(0, len(tasks) - 1)
                index = task_index[record["task_name"]]
                if index > 0:
                    one_hot[index - 1] = 1.0
                values = {
                    "safe_only": [score],
                    "duration_only": [elapsed],
                    "duration_safe": [elapsed, score],
                    "duration_task": [elapsed, *one_hot],
                    "duration_safe_task": [elapsed, score, *one_hot],
                }
                result.append(values[feature_set])
            return np.asarray(result, dtype=np.float64)

        train = [
            record
            for record in records
            if original_split(record, variant) == "train"
        ]
        test = [
            record
            for record in records
            if original_split(record, variant) == "test"
        ]
        train_labels = np.asarray(
            [int(bool(record["failed"])) for record in train], dtype=np.int64
        )
        test_labels = np.asarray(
            [int(bool(record["failed"])) for record in test], dtype=np.int64
        )
        aucs = {}
        for feature_set in (
            "safe_only",
            "duration_only",
            "duration_safe",
            "duration_task",
            "duration_safe_task",
        ):
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, solver="lbfgs"),
            )
            estimator.fit(matrix(train, feature_set), train_labels)
            probability = estimator.predict_proba(matrix(test, feature_set))[:, 1]
            aucs[f"{feature_set}_roc_auc"] = float(
                roc_auc_score(test_labels, probability)
            )
        rows.append(
            {
                "model": model,
                "seed": seed,
                "score_variant": variant,
                "train_rollouts": len(train),
                "test_rollouts": len(test),
                **aucs,
                "safe_increment_over_duration": (
                    aucs["duration_safe_roc_auc"]
                    - aucs["duration_only_roc_auc"]
                ),
                "safe_increment_over_duration_and_task": (
                    aucs["duration_safe_task_roc_auc"]
                    - aucs["duration_task_roc_auc"]
                ),
            }
        )
    return rows


def aggregate_rows(rows, group_keys, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, values in sorted(grouped.items()):
        result = dict(zip(group_keys, group))
        result["num_seeds"] = len(values)
        for metric in metric_keys:
            array = np.asarray(
                [value[metric] for value in values if value[metric] is not None],
                dtype=np.float64,
            )
            result[f"{metric}_mean"] = float(np.mean(array)) if len(array) else None
            result[f"{metric}_std"] = float(np.std(array)) if len(array) else None
        output.append(result)
    return output


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(fig, output_dir, stem, formats):
    outputs = []
    for extension in formats:
        path = Path(output_dir) / f"{stem}.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None)
        outputs.append(path)
    return outputs


def resample(values, points=101):
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, source, values)


def averaged_rollout_curves(score_sets, model, variant):
    curves = defaultdict(list)
    metadata = {}
    for seed in (0, 1, 2):
        for record in score_sets[(model, seed, variant)]:
            if original_split(record, variant) != "test":
                continue
            rollout_id = record["rollout_id"]
            curves[rollout_id].append(resample(score_array(record, variant)))
            metadata[rollout_id] = {
                "task_name": record["task_name"],
                "failed": bool(record["failed"]),
                "duration": duration(record),
            }
    if any(len(values) != 3 for values in curves.values()):
        raise ValueError(f"Missing model seeds for {model} {variant} trajectories")
    return {
        rollout_id: {
            **metadata[rollout_id],
            "curve": np.mean(values, axis=0),
        }
        for rollout_id, values in curves.items()
    }


def plot_overall_trajectories(plt, score_sets, output_dir, formats):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    progress = np.linspace(0.0, 1.0, 101)
    for row, variant in enumerate(VARIANTS):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            curves = averaged_rollout_curves(score_sets, model, variant)
            for failed, label, linestyle in (
                (False, "Success", "-"),
                (True, "Failure", "--"),
            ):
                values = np.asarray(
                    [value["curve"] for value in curves.values() if value["failed"] == failed]
                )
                median = np.median(values, axis=0)
                lower = np.percentile(values, 25, axis=0)
                upper = np.percentile(values, 75, axis=0)
                color = OUTCOME_COLORS[failed]
                ax.fill_between(progress, lower, upper, color=color, alpha=0.16)
                ax.plot(
                    progress,
                    median,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    label=f"{label} (n={len(values)})",
                )
            ax.set_title(f"{MODEL_LABELS[model]} — {variant}")
            ax.set_ylabel("SAFE score")
            ax.grid(alpha=0.4)
            ax.legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Matched evaluation-window progress")
    fig.suptitle(
        "Held-out SAFE trajectories averaged across model seeds",
        fontsize=14,
    )
    fig.tight_layout()
    outputs = save_figure(fig, output_dir, "score_trajectories_raw_normalized", formats)
    plt.close(fig)
    return outputs


def plot_per_task_trajectories(plt, score_sets, output_dir, formats):
    outputs = []
    progress = np.linspace(0.0, 1.0, 101)
    for model in MODELS:
        curves = averaged_rollout_curves(score_sets, model, "normalized")
        tasks = sorted({value["task_name"] for value in curves.values()})
        fig, axes = plt.subplots(5, 2, figsize=(12.0, 17.0), sharex=True)
        for ax, task in zip(axes.flat, tasks):
            for failed, label, linestyle in (
                (False, "Success", "-"),
                (True, "Failure", "--"),
            ):
                values = np.asarray(
                    [
                        value["curve"]
                        for value in curves.values()
                        if value["task_name"] == task and value["failed"] == failed
                    ]
                )
                median = np.median(values, axis=0)
                lower = np.percentile(values, 25, axis=0)
                upper = np.percentile(values, 75, axis=0)
                color = OUTCOME_COLORS[failed]
                ax.fill_between(progress, lower, upper, color=color, alpha=0.15)
                ax.plot(
                    progress,
                    median,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=f"{label} (n={len(values)})",
                )
            ax.set_title(task)
            ax.grid(alpha=0.35)
            ax.legend(frameon=False, fontsize=8)
        for ax in list(axes.flat)[len(tasks):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("Matched-window progress")
        for ax in axes[:, 0]:
            ax.set_ylabel("Normalized SAFE score")
        fig.suptitle(
            f"{MODEL_LABELS[model]} normalized trajectories by task",
            fontsize=15,
        )
        fig.tight_layout()
        outputs += save_figure(
            fig,
            output_dir,
            f"normalized_per_task_trajectories_{model}",
            formats,
        )
        plt.close(fig)
    return outputs


def plot_per_task_auc(plt, per_task_summary, output_dir, formats):
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.8), sharex=True)
    for ax, model in zip(axes, MODELS):
        selected = [row for row in per_task_summary if row["model"] == model]
        tasks = sorted({row["task_name"] for row in selected})
        positions = np.arange(len(tasks))
        duration_values = []
        for task in tasks:
            raw = next(
                row for row in selected
                if row["task_name"] == task and row["score_variant"] == "raw"
            )
            normalized = next(
                row for row in selected
                if row["task_name"] == task and row["score_variant"] == "normalized"
            )
            duration_values.append(raw["duration_roc_auc_mean"])
        raw_values = [
            next(
                row for row in selected
                if row["task_name"] == task and row["score_variant"] == "raw"
            )["score_roc_auc_mean"]
            for task in tasks
        ]
        normalized_values = [
            next(
                row for row in selected
                if row["task_name"] == task and row["score_variant"] == "normalized"
            )["score_roc_auc_mean"]
            for task in tasks
        ]
        ax.plot(positions, raw_values, marker="o", label="Raw SAFE")
        ax.plot(positions, normalized_values, marker="s", label="Normalized SAFE")
        ax.plot(
            positions,
            duration_values,
            marker="^",
            linestyle="--",
            color="#374151",
            label="Inference duration",
        )
        ax.axhline(0.5, color="#9CA3AF", linestyle=":")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Per-task ROC-AUC")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(axis="y", alpha=0.4)
        ax.legend(frameon=False, ncol=3)
    axes[-1].set_xticks(positions, tasks, rotation=38, ha="right")
    fig.suptitle("Per-task SAFE and duration discrimination", fontsize=14)
    fig.tight_layout()
    outputs = save_figure(fig, output_dir, "per_task_auc_and_duration", formats)
    plt.close(fig)
    return outputs


def plot_duration_scatter(plt, score_sets, output_dir, formats):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for row, variant in enumerate(VARIANTS):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            curves = averaged_rollout_curves(score_sets, model, variant)
            for failed, label in ((False, "Success"), (True, "Failure")):
                values = [value for value in curves.values() if value["failed"] == failed]
                ax.scatter(
                    [value["duration"] for value in values],
                    [float(np.max(value["curve"])) for value in values],
                    s=20,
                    alpha=0.58,
                    color=OUTCOME_COLORS[failed],
                    label=label,
                )
            all_values = list(curves.values())
            correlation = spearman(
                [value["duration"] for value in all_values],
                [float(np.max(value["curve"])) for value in all_values],
            )
            correlation_label = (
                "NA" if correlation is None else f"{correlation:.3f}"
            )
            ax.set_title(
                f"{MODEL_LABELS[model]} — {variant}; "
                f"Spearman={correlation_label}"
            )
            ax.set_xlabel("Policy-inference sequence length")
            ax.set_ylabel("Maximum matched SAFE score")
            ax.grid(alpha=0.35)
            ax.legend(frameon=False)
    fig.suptitle("SAFE score versus rollout-duration confound", fontsize=14)
    fig.tight_layout()
    outputs = save_figure(fig, output_dir, "score_vs_duration", formats)
    plt.close(fig)
    return outputs


def analyze_seen_diagnostics(
    final_root,
    calibration_root,
    output_dir,
    *,
    formats=("png", "pdf"),
):
    final_root = Path(final_root).resolve()
    calibration_root = Path(calibration_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys(formats))
    unsupported = set(formats) - {"png", "pdf", "svg"}
    if unsupported:
        raise ValueError(f"Unsupported formats: {sorted(unsupported)}")

    score_sets, signature = load_score_sets(final_root, calibration_root)
    rows = rollout_rows(score_sets)
    task_rows = per_task_metrics(score_sets)
    conditional_rows = conditional_duration_metrics(score_sets)
    task_summary = aggregate_rows(
        task_rows,
        ("model", "score_variant", "task_name"),
        (
            "score_roc_auc",
            "score_auprc",
            "duration_roc_auc",
            "score_duration_spearman",
        ),
    )
    conditional_summary = aggregate_rows(
        conditional_rows,
        ("model", "score_variant"),
        (
            "safe_only_roc_auc",
            "duration_only_roc_auc",
            "duration_safe_roc_auc",
            "duration_task_roc_auc",
            "duration_safe_task_roc_auc",
            "safe_increment_over_duration",
            "safe_increment_over_duration_and_task",
        ),
    )

    write_csv(output_dir / "per_rollout_scores.csv", rows)
    write_csv(output_dir / "per_task_metrics.csv", task_rows)
    write_csv(output_dir / "per_task_summary.csv", task_summary)
    write_csv(output_dir / "duration_conditional_metrics.csv", conditional_rows)
    write_csv(
        output_dir / "duration_conditional_summary.csv", conditional_summary
    )

    outputs = []
    if formats:
        plt = configure_matplotlib()
        outputs += plot_overall_trajectories(plt, score_sets, output_dir, formats)
        outputs += plot_per_task_trajectories(plt, score_sets, output_dir, formats)
        outputs += plot_per_task_auc(plt, task_summary, output_dir, formats)
        outputs += plot_duration_scatter(plt, score_sets, output_dir, formats)

    test_signature = [value for value in signature.values() if value[0] == "test"]
    report = {
        "schema_version": 1,
        "protocol": (
            "post-hoc frozen-model diagnostics with training-only task normalization "
            "and training-fitted duration probes"
        ),
        "final_root": str(final_root),
        "calibration_root": str(calibration_root),
        "output_dir": str(output_dir),
        "models": list(MODELS),
        "model_seeds": [0, 1, 2],
        "score_variants": list(VARIANTS),
        "counts": {
            "unique_rollouts": len(signature),
            "outer_test_rollouts": len(test_signature),
            "outer_test_successes": sum(not value[2] for value in test_signature),
            "outer_test_failures": sum(value[2] for value in test_signature),
            "tasks": len({value[1] for value in signature.values()}),
        },
        "per_task_summary": task_summary,
        "duration_conditional_summary": conditional_summary,
        "figures": [str(path) for path in outputs],
        "tables": [
            str(output_dir / name)
            for name in (
                "per_rollout_scores.csv",
                "per_task_metrics.csv",
                "per_task_summary.csv",
                "duration_conditional_metrics.csv",
                "duration_conditional_summary.csv",
            )
        ],
        "notes": [
            "Raw scores use the training-derived matched inference cutoff.",
            "Normalized scores use task affine statistics fitted from outer training only.",
            "Trajectory curves first average the three model seeds per rollout, then summarize unique rollouts.",
            "Duration is the full policy-inference sequence length, matching the final-refit duration baseline.",
            "All conditional logistic probes are fitted on outer training and evaluated on the fixed outer test.",
            "These are post-hoc diagnostics after the final test was opened; they must not be used to retune the frozen experiment.",
        ],
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=("png", "pdf", "svg"),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = analyze_seen_diagnostics(
        args.final_root,
        args.calibration_root,
        args.output_dir,
        formats=args.formats,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
