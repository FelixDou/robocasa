"""Build immutable matched-size SAFE splits from natural policy outcome rates.

The rate audit uses valid manifest records plus only quota-discarded rollouts for
which the simulator outcome was actually observed. Administrative skips and
collection errors are never treated as policy outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import random


ELIGIBLE_SKIPPED_STATUS = "skipped_class_quota_reached"


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_natural_outcomes(dataset_dirs):
    """Count genuine outcomes without counting merged copies or admin skips."""
    datasets = []
    observations = []
    seen_rollout_ids = {}
    for value in dataset_dirs:
        root = Path(value).resolve()
        manifest_path = root / "manifest.jsonl"
        skipped_path = root / "skipped.jsonl"
        if not manifest_path.is_file():
            raise ValueError(f"Missing manifest: {manifest_path}")
        retained = read_jsonl(manifest_path)
        skipped = read_jsonl(skipped_path)
        eligible_skipped = [
            event
            for event in skipped
            if event.get("status") == ELIGIBLE_SKIPPED_STATUS
        ]
        for source, records in (
            ("retained", retained),
            ("quota_discarded", eligible_skipped),
        ):
            for record in records:
                rollout_id = str(record.get("rollout_id", ""))
                task_name = str(record.get("task_name", ""))
                if not rollout_id or not task_name:
                    raise ValueError(
                        f"{root} has an outcome without rollout_id/task_name"
                    )
                if rollout_id in seen_rollout_ids:
                    raise ValueError(
                        "Duplicate audited rollout ID. Pass only original collection "
                        f"shards, not merged copies: {rollout_id} appears in "
                        f"{seen_rollout_ids[rollout_id]} and {root}"
                    )
                seen_rollout_ids[rollout_id] = str(root)
                if "failed" in record:
                    failed = bool(record["failed"])
                elif "success" in record:
                    failed = not bool(record["success"])
                else:
                    raise ValueError(
                        f"Audited outcome {rollout_id} has no success/failed label"
                    )
                observations.append(
                    {
                        "rollout_id": rollout_id,
                        "task_name": task_name,
                        "success": not failed,
                        "source": source,
                        "dataset_dir": str(root),
                    }
                )
        datasets.append(
            {
                "dataset_dir": str(root),
                "manifest_sha256": file_sha256(manifest_path),
                "retained_outcomes": len(retained),
                "quota_discarded_outcomes": len(eligible_skipped),
                "excluded_skipped_events": len(skipped) - len(eligible_skipped),
            }
        )
    if not observations:
        raise ValueError("No genuine policy outcomes were found")
    counts = defaultdict(
        lambda: {
            "successes": 0,
            "failures": 0,
            "retained_successes": 0,
            "retained_failures": 0,
            "quota_discarded_successes": 0,
            "quota_discarded_failures": 0,
        }
    )
    for record in observations:
        outcome = "successes" if record["success"] else "failures"
        counts[record["task_name"]][outcome] += 1
        counts[record["task_name"]][f"{record['source']}_{outcome}"] += 1
    per_task = {}
    for task_name in sorted(counts):
        values = dict(counts[task_name])
        values["observed_outcomes"] = values["successes"] + values["failures"]
        values["natural_success_rate"] = (
            values["successes"] / values["observed_outcomes"]
        )
        per_task[task_name] = values
    return {
        "schema_version": 1,
        "eligible_outcome_definition": (
            "valid manifest records plus skipped_class_quota_reached events only"
        ),
        "excluded_outcomes": (
            "errors, skipped_quota_reached, and all other administrative skips"
        ),
        "datasets": datasets,
        "num_observed_outcomes": len(observations),
        "per_task": per_task,
    }


def load_official_env_records(export_dir):
    records = []
    for path in sorted((Path(export_dir) / "env_records").glob("*.pkl")):
        with path.open("rb") as stream:
            record = pickle.load(stream)
        records.append(record)
    if not records:
        raise ValueError(f"No official SAFE env records found in {export_dir}")
    return records


def load_fixed_outer_split(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    train = [str(value) for value in manifest.get("train", [])]
    test = [str(value) for value in manifest.get("test", [])]
    if not train or not test or set(train) & set(test):
        raise ValueError("Fixed outer split train/test IDs are empty or overlap")
    return path, manifest, train, test


def natural_class_count(total, success_rate, *, min_per_class=3):
    successes = int(math.floor(total * success_rate + 0.5))
    successes = min(max(successes, min_per_class), total - min_per_class)
    return successes, total - successes


def choose_common_total(per_task_rates, pool_counts, *, min_per_class=3):
    maximum = min(
        values["successes"] + values["failures"]
        for values in pool_counts.values()
    )
    for total in range(maximum, 2 * min_per_class - 1, -1):
        balanced_successes = total // 2
        balanced_failures = total - balanced_successes
        valid = True
        for task_name, rate in per_task_rates.items():
            successes, failures = natural_class_count(
                total,
                rate,
                min_per_class=min_per_class,
            )
            available = pool_counts[task_name]
            valid &= (
                successes <= available["successes"]
                and failures <= available["failures"]
                and balanced_successes <= available["successes"]
                and balanced_failures <= available["failures"]
            )
        if valid:
            return total
    raise ValueError(
        "No common per-task sample count can preserve both classes with the "
        f"requested minimum of {min_per_class}"
    )


def deterministic_pick(values, count, *, seed, regime, task_name, success):
    values = sorted(values)
    rng = random.Random(f"{seed}:{regime}:{task_name}:{int(success)}")
    rng.shuffle(values)
    return sorted(values[:count])


def build_experiment(
    *,
    collection_datasets,
    official_export,
    outer_split_manifest,
    output_dir,
    subset_seeds=(0, 1, 2, 3, 4),
    samples_per_task=None,
    min_per_class=3,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_natural_outcomes(collection_datasets)
    export_dir = Path(official_export).resolve()
    outer_path, outer_manifest, outer_train, outer_test = load_fixed_outer_split(
        outer_split_manifest
    )
    env_records = load_official_env_records(export_dir)
    env_by_id = {str(record["rollout_id"]): record for record in env_records}
    if len(env_by_id) != len(env_records):
        raise ValueError("Official export contains duplicate rollout IDs")
    missing = sorted((set(outer_train) | set(outer_test)) - set(env_by_id))
    if missing:
        raise ValueError(
            "Fixed outer split references IDs absent from the official export: "
            + ", ".join(missing[:5])
        )
    train_pool = defaultdict(lambda: {True: [], False: []})
    for rollout_id in outer_train:
        env = env_by_id[rollout_id]
        train_pool[str(env["task_name"])][bool(env["episode_success"])].append(
            rollout_id
        )
    audit_tasks = set(audit["per_task"])
    pool_tasks = set(train_pool)
    if audit_tasks != pool_tasks:
        raise ValueError(
            "Natural-rate audit and fixed training pool task sets differ: "
            f"audit_only={sorted(audit_tasks - pool_tasks)}, "
            f"pool_only={sorted(pool_tasks - audit_tasks)}"
        )
    pool_counts = {
        task_name: {
            "successes": len(groups[True]),
            "failures": len(groups[False]),
        }
        for task_name, groups in train_pool.items()
    }
    rates = {
        task_name: values["natural_success_rate"]
        for task_name, values in audit["per_task"].items()
    }
    if samples_per_task is None:
        samples_per_task = choose_common_total(
            rates,
            pool_counts,
            min_per_class=min_per_class,
        )
    if samples_per_task < 2 * min_per_class:
        raise ValueError(
            f"samples_per_task must be at least {2 * min_per_class}"
        )
    audit["official_export"] = str(export_dir)
    audit["fixed_outer_split_manifest"] = str(outer_path)
    audit["fixed_outer_split_sha256"] = file_sha256(outer_path)
    audit["fixed_outer_counts"] = {
        "train": len(outer_train),
        "test": len(outer_test),
    }
    audit["available_outer_train_per_task"] = pool_counts
    audit["selected_samples_per_task"] = samples_per_task
    audit_path = output_dir / "natural_rate_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    manifests = []
    for subset_seed in subset_seeds:
        for regime in ("matched_balanced", "natural_rate"):
            selected_train = []
            per_task = {}
            for task_name in sorted(train_pool):
                if regime == "matched_balanced":
                    success_count = samples_per_task // 2
                    failure_count = samples_per_task - success_count
                else:
                    success_count, failure_count = natural_class_count(
                        samples_per_task,
                        rates[task_name],
                        min_per_class=min_per_class,
                    )
                selected_successes = deterministic_pick(
                    train_pool[task_name][True],
                    success_count,
                    seed=subset_seed,
                    regime=regime,
                    task_name=task_name,
                    success=True,
                )
                selected_failures = deterministic_pick(
                    train_pool[task_name][False],
                    failure_count,
                    seed=subset_seed,
                    regime=regime,
                    task_name=task_name,
                    success=False,
                )
                if (
                    len(selected_successes) != success_count
                    or len(selected_failures) != failure_count
                ):
                    raise ValueError(
                        f"Task {task_name} lacks samples for {regime}: "
                        f"requested {success_count} successes and {failure_count} failures"
                    )
                selected_train.extend(selected_successes + selected_failures)
                per_task[task_name] = {
                    "observed_natural_success_rate": rates[task_name],
                    "train_successes": success_count,
                    "train_failures": failure_count,
                    "train_total": samples_per_task,
                }
            selected_train = sorted(selected_train)
            excluded = sorted(set(outer_train) - set(selected_train))
            manifest = {
                "schema_version": 1,
                "protocol": "fixed_outer_test_natural_rate_training_subset",
                "regime": regime,
                "subset_seed": int(subset_seed),
                "split_seed": outer_manifest.get("split_seed"),
                "official_export": str(export_dir),
                "natural_rate_audit": str(audit_path),
                "fixed_outer_split_manifest": str(outer_path),
                "fixed_outer_split_sha256": file_sha256(outer_path),
                "samples_per_task": samples_per_task,
                "min_per_class": min_per_class,
                "per_task": per_task,
                "counts": {
                    "train": len(selected_train),
                    "test": len(outer_test),
                    "excluded_outer_train": len(excluded),
                },
                "train": selected_train,
                "test": sorted(outer_test),
                "excluded_outer_train": excluded,
            }
            manifest_dir = output_dir / f"subset_seed_{subset_seed}" / regime
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "split_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            manifests.append(
                {
                    "regime": regime,
                    "subset_seed": int(subset_seed),
                    "path": str(manifest_path),
                    "sha256": file_sha256(manifest_path),
                    "counts": manifest["counts"],
                }
            )
    plan = {
        "schema_version": 1,
        "protocol": "matched-size natural-rate SAFE weighting experiment",
        "official_export": str(export_dir),
        "natural_rate_audit": str(audit_path),
        "fixed_outer_split_manifest": str(outer_path),
        "fixed_outer_test_ids": sorted(outer_test),
        "samples_per_task": samples_per_task,
        "subset_seeds": [int(seed) for seed in subset_seeds],
        "model_seeds": [0, 1, 2],
        "regimes": {
            "matched_weighted": {
                "selection": "matched_balanced",
                "class_weighting": "official_inverse_frequency",
            },
            "natural_weighted": {
                "selection": "natural_rate",
                "class_weighting": "official_inverse_frequency",
            },
            "natural_unweighted": {
                "selection": "natural_rate",
                "class_weighting": "none",
            },
        },
        "manifests": manifests,
        "scientific_comparisons": {
            "natural_rate_effect_at_fixed_n": "natural_weighted - matched_weighted",
            "inverse_frequency_weighting_effect": (
                "natural_weighted - natural_unweighted"
            ),
        },
    }
    plan_path = output_dir / "experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dataset",
        action="append",
        required=True,
        help="Original collection shard with manifest.jsonl and skipped.jsonl",
    )
    parser.add_argument("--official-export", required=True)
    parser.add_argument("--outer-split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--subset-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--samples-per-task", type=int)
    parser.add_argument("--min-per-class", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plan = build_experiment(
            collection_datasets=args.collection_dataset,
            official_export=args.official_export,
            outer_split_manifest=args.outer_split_manifest,
            output_dir=args.output_dir,
            subset_seeds=args.subset_seeds,
            samples_per_task=args.samples_per_task,
            min_per_class=args.min_per_class,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
