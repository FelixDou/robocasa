"""Derive task-specific balanced and natural-rate SAFE split manifests.

The input is the immutable experiment plan produced by
``build_natural_rate_experiment``.  This builder does not resample or relabel
anything: it partitions each existing manifest by task and verifies that every
balanced/natural comparison uses the same fixed held-out task episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle


PRIMARY_REGIMES = ("matched_weighted", "natural_weighted")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_env_index(export_dir):
    export_dir = Path(export_dir).resolve()
    report = json.loads((export_dir / "conversion_report.json").read_text())
    by_id = {}
    for path in sorted((export_dir / "env_records").glob("*.pkl")):
        with path.open("rb") as stream:
            record = pickle.load(stream)
        rollout_id = str(record["rollout_id"])
        if rollout_id in by_id:
            raise ValueError(f"Duplicate rollout ID in official export: {rollout_id}")
        by_id[rollout_id] = record
    if not by_id:
        raise ValueError(f"No official SAFE env records found in {export_dir}")
    return report, by_id


def outcome_counts(rollout_ids, env_by_id):
    successes = sum(bool(env_by_id[value]["episode_success"]) for value in rollout_ids)
    return {
        "rollouts": len(rollout_ids),
        "successes": successes,
        "failures": len(rollout_ids) - successes,
    }


def build_task_specific_experiment(
    *,
    experiment_plan,
    official_export,
    output_dir,
    regimes=PRIMARY_REGIMES,
):
    source_plan_path = Path(experiment_plan).resolve()
    source_plan = json.loads(source_plan_path.read_text())
    export_dir = Path(official_export).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report, env_by_id = load_env_index(export_dir)

    if Path(source_plan["official_export"]).resolve() != export_dir:
        raise ValueError("Natural-rate experiment plan uses a different official export")
    unknown_regimes = sorted(set(regimes) - set(source_plan["regimes"]))
    if unknown_regimes:
        raise ValueError(f"Unknown source regimes: {unknown_regimes}")
    task_names = sorted(report["task_ids"])
    task_types = report.get("task_types", {})
    if set(task_names) != set(task_types):
        raise ValueError("Official export lacks task-type provenance for every task")

    source_manifests = {}
    for item in source_plan["manifests"]:
        key = (str(item["regime"]), int(item["subset_seed"]))
        if key in source_manifests:
            raise ValueError(f"Duplicate source manifest for {key}")
        source_manifests[key] = Path(item["path"]).resolve()

    subset_seeds = [int(value) for value in source_plan["subset_seeds"]]
    output_manifests = []
    frozen_test_by_task = {}
    train_total_by_task = {}
    for subset_seed in subset_seeds:
        for regime in regimes:
            regime_spec = source_plan["regimes"][regime]
            selection_name = str(regime_spec["selection"])
            source_path = source_manifests.get((selection_name, subset_seed))
            if source_path is None:
                raise ValueError(
                    f"Missing {selection_name} source manifest for seed {subset_seed}"
                )
            source = json.loads(source_path.read_text())
            requested = [str(value) for value in source["train"] + source["test"]]
            missing = sorted(set(requested) - set(env_by_id))
            if missing:
                raise ValueError(
                    "Source manifest references IDs absent from the official export: "
                    + ", ".join(missing[:5])
                )
            for task_name in task_names:
                train = sorted(
                    value
                    for value in map(str, source["train"])
                    if str(env_by_id[value]["task_name"]) == task_name
                )
                test = sorted(
                    value
                    for value in map(str, source["test"])
                    if str(env_by_id[value]["task_name"]) == task_name
                )
                train_counts = outcome_counts(train, env_by_id)
                test_counts = outcome_counts(test, env_by_id)
                if min(
                    train_counts["successes"],
                    train_counts["failures"],
                    test_counts["successes"],
                    test_counts["failures"],
                ) <= 0:
                    raise ValueError(
                        f"{task_name}/{regime}/seed-{subset_seed} lacks both outcomes"
                    )
                if regime == "matched_weighted" and (
                    train_counts["successes"] != train_counts["failures"]
                ):
                    raise ValueError(
                        f"Balanced task split is not exactly balanced for {task_name}"
                    )
                expected_test = frozen_test_by_task.setdefault(task_name, test)
                if test != expected_test:
                    raise ValueError(
                        f"Frozen test IDs changed for {task_name}/{regime}/seed-{subset_seed}"
                    )
                expected_total = train_total_by_task.setdefault(
                    task_name, train_counts["rollouts"]
                )
                if train_counts["rollouts"] != expected_total:
                    raise ValueError(
                        f"Training size changed for {task_name}/{regime}/seed-{subset_seed}"
                    )
                manifest = {
                    "schema_version": 1,
                    "protocol": "task_specific_fixed_test_rate_comparison",
                    "task_name": task_name,
                    "task_type": task_types[task_name],
                    "regime": regime,
                    "selection": selection_name,
                    "class_weighting": regime_spec["class_weighting"],
                    "subset_seed": subset_seed,
                    "split_seed": source.get("split_seed"),
                    "split_unit": "rollout",
                    "official_export": str(export_dir),
                    "source_experiment_plan": str(source_plan_path),
                    "source_split_manifest": str(source_path),
                    "source_split_manifest_sha256": file_sha256(source_path),
                    "counts": {"train": train_counts, "test": test_counts},
                    "train": train,
                    "test": test,
                }
                manifest_path = (
                    output_dir
                    / "manifests"
                    / task_name
                    / regime
                    / f"subset_seed_{subset_seed}.json"
                )
                write_json(manifest_path, manifest)
                output_manifests.append(
                    {
                        "task_name": task_name,
                        "task_type": task_types[task_name],
                        "regime": regime,
                        "subset_seed": subset_seed,
                        "path": str(manifest_path),
                        "sha256": file_sha256(manifest_path),
                        "counts": manifest["counts"],
                    }
                )

    plan = {
        "schema_version": 1,
        "protocol": "one normal SAFE model per task at fixed balanced/natural N",
        "scope": "rollout-level normal SAFE; no Subtask-SAFE labels or conditioning",
        "official_export": str(export_dir),
        "source_experiment_plan": str(source_plan_path),
        "tasks": task_names,
        "task_types": {name: task_types[name] for name in task_names},
        "subset_seeds": subset_seeds,
        "model_seeds": [0, 1, 2],
        "regimes": {name: source_plan["regimes"][name] for name in regimes},
        "frozen_test_ids_by_task": frozen_test_by_task,
        "train_rollouts_per_task": train_total_by_task,
        "num_task_regime_comparisons": len(task_names) * len(regimes),
        "manifests": output_manifests,
        "primary_comparison": "natural_weighted minus matched_weighted at fixed N",
        "analysis_note": (
            "Subset seeds vary training composition; every task/regime/model uses "
            "the same frozen task-specific test IDs. Test episodes are not new "
            "independent replicates across subset or model seeds."
        ),
    }
    write_json(output_dir / "task_specific_experiment_plan.json", plan)
    return plan


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-plan", required=True)
    parser.add_argument("--official-export", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=list(PRIMARY_REGIMES),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plan = build_task_specific_experiment(
            experiment_plan=args.experiment_plan,
            official_export=args.official_export,
            output_dir=args.output_dir,
            regimes=args.regimes,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
