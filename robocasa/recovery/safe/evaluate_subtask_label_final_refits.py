"""Evaluate frozen terminal and Subtask-SAFE refits on active-subtask labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .subtask_safe_evaluation import (
    parent_group_id,
    semantic_subtask_scores_from_saved_records,
    subtask_fixed_prefix_selection,
    validate_subtask_catalog,
)
from .train_seen_tasks import load_env_records, write_json


def load_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {error}"
                    ) from error
    return rows


def load_frozen_stage_catalog(cv_root, models):
    signatures = []
    sources = []
    for treatment in ("terminal", "subtask"):
        for model in models:
            path = Path(cv_root) / treatment / f"cv_plan_{model}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing CV plan: {path}")
            plan = json.loads(path.read_text())
            catalog = plan.get("common_subtask_stage_catalog")
            if not catalog:
                raise ValueError(f"CV plan has no frozen stage catalog: {path}")
            signatures.append(json.dumps(catalog, sort_keys=True))
            sources.append(str(path.resolve()))
    if len(set(signatures)) != 1:
        raise ValueError("Terminal/Subtask CV plans disagree on the stage catalog")
    return json.loads(signatures[0]), sources


def _aggregate(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "values": [float(value) for value in array],
    }


def evaluate_final_refits(
    final_root,
    cv_root,
    subtask_export_dir,
    output_dir,
    *,
    models=("indep", "lstm"),
    seeds=(0, 1, 2),
    prefixes=(1, 2, 4, 8),
    primary_prefixes=(1, 2),
):
    final_root = Path(final_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(int(value) for value in prefixes)
    primary_prefixes = tuple(int(value) for value in primary_prefixes)
    if not set(primary_prefixes).issubset(prefixes):
        raise ValueError("Primary prefixes must be included in evaluation prefixes")

    stage_catalog, cv_plan_sources = load_frozen_stage_catalog(cv_root, models)
    evaluation_stages = stage_catalog["selected_stages"]
    catalog = validate_subtask_catalog(load_env_records(subtask_export_dir))
    per_seed = []
    test_parent_contract = None

    for treatment in ("terminal", "subtask"):
        for model in models:
            for seed in seeds:
                run_root = final_root / treatment / f"{model}_seed{seed}"
                scores_path = run_root / "scores.jsonl"
                metrics_path = run_root / "metrics.json"
                if not scores_path.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(f"Incomplete final refit: {run_root}")
                source_scores = load_jsonl(scores_path)
                training_scores = [
                    record for record in source_scores if record["split"] == "train"
                ]
                test_scores = [
                    record for record in source_scores if record["split"] == "test"
                ]
                if not training_scores or not test_scores:
                    raise ValueError(
                        f"Final refit has an empty score split: {run_root}"
                    )
                training_parents = {
                    parent_group_id(record) for record in training_scores
                }
                test_parents = {parent_group_id(record) for record in test_scores}
                if training_parents & test_parents:
                    raise ValueError(f"Final refit leaks parent IDs: {run_root}")
                if test_parent_contract is None:
                    test_parent_contract = test_parents
                elif test_parent_contract != test_parents:
                    raise ValueError(
                        "Terminal/Subtask final refits use different outer-test parents"
                    )

                training_catalog = [
                    record
                    for record in catalog
                    if parent_group_id(record) in training_parents
                ]
                aligned_test = semantic_subtask_scores_from_saved_records(
                    test_scores,
                    catalog,
                )
                result = subtask_fixed_prefix_selection(
                    training_catalog,
                    aligned_test,
                    prefixes=prefixes,
                    evaluation_stages=evaluation_stages,
                )
                primary_values = [
                    result["safe"]["per_prefix"][str(prefix)][
                        "task_stage_macro_roc_auc"
                    ]
                    for prefix in primary_prefixes
                ]
                per_seed.append(
                    {
                        "schema_version": 1,
                        "treatment": treatment,
                        "model": model,
                        "seed": int(seed),
                        "run_root": str(run_root),
                        "training_parents": len(training_parents),
                        "outer_test_parents": len(test_parents),
                        "outer_test_segments": len(aligned_test),
                        "evaluation_stages": evaluation_stages,
                        "primary_prefixes": list(primary_prefixes),
                        "primary_value": float(np.mean(primary_values)),
                        "fixed_prefix_evaluation": result,
                    }
                )

    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["treatment"], row["model"])].append(row)
    aggregate = {}
    for (treatment, model), rows in sorted(grouped.items()):
        key = f"{treatment}/{model}"
        aggregate[key] = {
            "primary_prefixes": list(primary_prefixes),
            "primary_task_stage_macro_roc_auc": _aggregate(
                [row["primary_value"] for row in rows]
            ),
            "per_prefix": {},
        }
        for prefix in prefixes:
            prefix_key = str(prefix)
            safe = [
                row["fixed_prefix_evaluation"]["safe"]["per_prefix"][prefix_key][
                    "task_stage_macro_roc_auc"
                ]
                for row in rows
            ]
            elapsed = [
                row["fixed_prefix_evaluation"]["time_only"]["per_prefix"][prefix_key][
                    "task_stage_macro_roc_auc"
                ]
                for row in rows
            ]
            aggregate[key]["per_prefix"][prefix_key] = {
                "safe": _aggregate(safe),
                "time_only": _aggregate(elapsed),
                "safe_minus_time": _aggregate([a - b for a, b in zip(safe, elapsed)]),
            }

    output = {
        "schema_version": 1,
        "status": "complete",
        "protocol": (
            "frozen selected hyperparameters refit on all outer-development parents; "
            "untouched outer test evaluated with active-subtask labels"
        ),
        "final_root": str(final_root),
        "cv_root": str(Path(cv_root).resolve()),
        "cv_plan_sources": cv_plan_sources,
        "subtask_export_dir": str(Path(subtask_export_dir).resolve()),
        "outer_test_used_for_selection": False,
        "thresholds_fitted": False,
        "prefixes": list(prefixes),
        "primary_prefixes": list(primary_prefixes),
        "evaluation_stages": evaluation_stages,
        "outer_test_parents": len(test_parent_contract or []),
        "completed_refits": len(per_seed),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    write_json(output_dir / "outer_subtask_label_results.json", output)
    with (output_dir / "per_seed_results.jsonl").open("w") as stream:
        for row in per_seed:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--subtask-export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["indep", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--prefixes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--primary-prefixes", nargs="+", type=int, default=[1, 2])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = evaluate_final_refits(
        args.final_root,
        args.cv_root,
        args.subtask_export_dir,
        args.output_dir,
        models=args.models,
        seeds=args.seeds,
        prefixes=args.prefixes,
        primary_prefixes=args.primary_prefixes,
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
