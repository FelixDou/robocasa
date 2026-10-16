"""Offline baseline suite for the frozen Xiaomi staged SAFE + time detector.

The suite reuses the frozen prospective matched-FPR protocol without changing
its runtime bundle, thresholds, or reported numbers.  Phases are ordered:

1. ``extract`` reads one official SAFE export with the pinned official loader
   and the frozen MLP configuration, so every baseline sees the identical
   per-inference feature reduction (and the recorded action chunks);
2. ``develop`` fits success-only baselines on development outer-training
   rollouts, selects hyperparameters on a development meta-validation split,
   fits task normalization there, and freezes the baseline bundle without
   reading calibration or prospective outcomes;
3. ``calibrate`` reproduces the frozen SAFE/time thresholds exactly, then
   selects every baseline threshold on the same primary calibration cohort
   under the same event-level FPR cap and tie-breaking;
4. ``evaluate`` verifies that the frozen prospective results are reproduced
   bit-for-bit and then evaluates every frozen baseline once on the
   prospective shadow cohort.

``synthetic-smoke`` runs all phases after ``extract`` on generated data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import pickle
import warnings

import numpy as np

from .analyze_time_safe_hybrid import fit_safe_normalization
from .offline_detector_baselines import (
    AGGREGATIONS,
    BASELINES,
    DEFAULT_GRIDS,
    DEFAULT_SELECTION_LANDMARKS,
    SMOKE_GRIDS,
    BaselineScorer,
    aggregate_scores,
    event_task_macro_auc,
    landmark_auc_rows,
    normalize_trajectory,
    selection_key,
    stratified_meta_split,
    validate_aggregations,
    validate_baselines,
)
from .prospective_matched_fpr import (
    DETECTORS,
    event_metrics,
    first_alert,
    select_single_threshold,
    select_staged_thresholds,
    validate_stage_windows,
)
from .run_prospective_matched_fpr import (
    _identity_keys,
    _json_value,
    _select_per_task_class,
    _sha256,
    _write_json,
    _write_jsonl,
)
from .train_seen_tasks import load_outer_split_ids


PROTOCOL = "offline_success_only_baselines_xr1_matched_fpr"
CACHE_GROUPS = ("development", "calibration", "prospective_test")
REFERENCE_DETECTORS = ("time_only", "staged_safe_time")
EVALUATION_LANDMARKS = (0.10, 0.25, 0.50)
PREDICTION_KEYS = (
    "rollout_id",
    "task_name",
    "task_type",
    "failed",
    "horizon",
    "source_inferences",
    "environment_seed",
    "environment_reset_index",
    "seed_protocol",
    "video_path",
    "inference_environment_steps",
)


# ---------------------------------------------------------------------------
# Shared I/O
# ---------------------------------------------------------------------------


def _prepare_output(path):
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validate_fractions(values):
    values = tuple(sorted({float(value) for value in values}))
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("Landmark fractions must lie in (0, 1]")
    return values


def _write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(row.get(key)) for key in fields})


class FeatureCache:
    """Inference-aligned features and action chunks for one export."""

    def __init__(self, root, expected_group=None):
        self.root = Path(root).resolve()
        self.provenance = json.loads((self.root / "provenance.json").read_text())
        if expected_group is not None and self.provenance.get("group") != expected_group:
            raise ValueError(
                f"Cache {self.root} has group {self.provenance.get('group')!r}, "
                f"expected {expected_group!r}"
            )
        self.features = np.load(self.root / "features.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.rows = [
            json.loads(line)
            for line in (self.root / "rollouts.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.by_id = {str(row["rollout_id"]): row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError(f"Duplicate rollout IDs in cache {self.root}")
        total = sum(int(row["length"]) for row in self.rows)
        if len(self.features) != total or len(self.actions) != total:
            raise ValueError(f"Cache arrays disagree with rollout lengths in {self.root}")

    def features_for(self, row):
        start, length = int(row["offset"]), int(row["length"])
        return np.asarray(self.features[start : start + length], dtype=np.float32)

    def actions_for(self, row):
        start, length = int(row["offset"]), int(row["length"])
        return np.asarray(self.actions[start : start + length], dtype=np.float32)


def write_cache(output_dir, *, rows, features, actions, provenance):
    """Persist inference-aligned arrays plus one metadata row per rollout."""
    output = _prepare_output(output_dir)
    offset = 0
    metadata = []
    for row, feature, action in zip(rows, features, actions):
        feature = np.asarray(feature, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if feature.ndim != 2 or action.ndim != 3 or len(feature) != len(action):
            raise ValueError(f"Misaligned cache arrays for {row['rollout_id']}")
        if not len(feature) or not np.all(np.isfinite(feature)):
            raise ValueError(f"Invalid features for {row['rollout_id']}")
        metadata.append({**row, "offset": offset, "length": int(len(feature))})
        offset += len(feature)
    np.save(output / "features.npy", np.concatenate(features).astype(np.float32))
    np.save(output / "actions.npy", np.concatenate(actions).astype(np.float32))
    _write_jsonl(output / "rollouts.jsonl", metadata)
    provenance = {
        **provenance,
        "rollouts": len(metadata),
        "successes": sum(not bool(row["failed"]) for row in metadata),
        "failures": sum(bool(row["failed"]) for row in metadata),
        "inferences": offset,
        "features_sha256": _sha256(output / "features.npy"),
        "actions_sha256": _sha256(output / "actions.npy"),
    }
    _write_json(output / "provenance.json", provenance)
    return provenance


def _load_prediction_records(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = {key: row.get(key) for key in PREDICTION_KEYS}
        record["failed"] = bool(record["failed"])
        record["horizon"] = int(record["horizon"])
        record["source_inferences"] = int(record["source_inferences"])
        record["safe_scores"] = np.asarray(row["safe_scores"], dtype=np.float64)
        record["time_scores"] = np.asarray(row["time_scores"], dtype=np.float64)
        records.append(record)
    if not records:
        raise ValueError(f"No frozen prediction records in {path}")
    ids = [record["rollout_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate rollout IDs in {path}")
    return records


def _load_runtime_bundle(path):
    path = Path(path).resolve()
    bundle = json.loads(path.read_text())
    if bundle.get("phase") != "calibration_frozen":
        raise ValueError("Runtime bundle is not a frozen calibration artifact")
    return path, bundle


def _verify_sha(path, expected, label):
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum changed: {path}")
    return actual


# ---------------------------------------------------------------------------
# Phase 1: feature extraction with the pinned official loader
# ---------------------------------------------------------------------------


def extract(args):
    import sys

    from .train_seen_tasks import (
        OFFICIAL_SAFE_COMMIT,
        load_env_records,
        task_catalog,
        validate_alignment,
        verify_safe_repo,
    )

    if args.group not in CACHE_GROUPS:
        raise ValueError(f"--group must be one of {CACHE_GROUPS}")
    export_dir = Path(args.export_dir).resolve()
    config_path = Path(args.safe_config).resolve()
    verify_safe_repo(args.safe_repo)
    if str(Path(args.safe_repo).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(args.safe_repo).resolve()))
    from omegaconf import OmegaConf
    from failure_prob.data.pizero import load_rollouts_from_root
    from failure_prob.utils.random import seed_everything

    cfg = OmegaConf.load(config_path)
    selectors = (str(cfg.dataset.horizon_idx_rel), str(cfg.dataset.diff_idx_rel))
    expected = (str(args.expected_horizon_selector), str(args.expected_diffusion_selector))
    if selectors != expected:
        raise ValueError(f"Frozen SAFE selectors are {selectors}, expected {expected}")
    cfg.dataset.data_path_prefix = ""
    cfg.dataset.data_path = str(export_dir)
    cfg.dataset.data_path_unseen = None
    cfg.dataset.load_to_cuda = False
    cfg.train.log_precomputed = False
    cfg.train.log_precomputed_only = False
    seed_everything(0)
    rollouts = load_rollouts_from_root(export_dir, cfg)
    env_records = load_env_records(export_dir)
    identity = validate_alignment(rollouts, env_records)
    catalog = task_catalog(export_dir)
    pred_horizon = int(cfg.dataset.pred_horizon)
    dim_action = int(cfg.dataset.dim_action)
    exec_horizon = int(cfg.dataset.exec_horizon)

    rows, features, actions = [], [], []
    irregular_gaps = 0
    for rollout in rollouts:
        env_path, env = identity[id(rollout)]
        hidden = rollout.hidden_states
        hidden = hidden.detach().cpu().numpy() if hasattr(hidden, "detach") else hidden
        hidden = np.asarray(hidden, dtype=np.float32).reshape(len(hidden), -1)
        vectors = getattr(rollout, "action_vectors", None)
        if vectors is None:
            raise ValueError("The official loader did not provide action_vectors")
        vectors = vectors.detach().cpu().numpy() if hasattr(vectors, "detach") else vectors
        chunks = np.asarray(vectors, dtype=np.float32).reshape(
            len(vectors), pred_horizon, dim_action
        )
        if len(chunks) != len(hidden):
            raise ValueError(f"Action/feature inference counts differ for {env_path}")
        if int(env["replan_steps"]) != exec_horizon:
            raise ValueError(f"Replan steps disagree with exec_horizon for {env_path}")
        manifest = env.get("robocasa_manifest_record", {})
        steps = manifest.get("inference_environment_steps")
        if steps is not None:
            if len(steps) != len(hidden):
                raise ValueError(f"Inference steps disagree with features for {env_path}")
            irregular_gaps += int(np.sum(np.diff(np.asarray(steps)) != exec_horizon))
        task_name = catalog["task_names"][int(rollout.task_id)]
        rows.append(
            {
                "rollout_id": str(env["rollout_id"]),
                "task_name": task_name,
                "task_type": catalog["task_types"][task_name],
                "failed": not bool(int(rollout.episode_success)),
                "environment_seed": env.get("environment_seed"),
                "environment_reset_index": env.get("environment_reset_index"),
                "replan_steps": int(env["replan_steps"]),
            }
        )
        features.append(hidden)
        actions.append(chunks)
    provenance = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "group": args.group,
        "export_dir": str(export_dir),
        "conversion_report_sha256": _sha256(export_dir / "conversion_report.json"),
        "safe_config": str(config_path),
        "safe_config_sha256": _sha256(config_path),
        "safe_repository": str(Path(args.safe_repo).resolve()),
        "safe_repository_commit": OFFICIAL_SAFE_COMMIT,
        "horizon_selector": selectors[0],
        "diffusion_selector": selectors[1],
        "feature_dim": int(features[0].shape[1]),
        "pred_horizon": pred_horizon,
        "dim_action": dim_action,
        "exec_horizon": exec_horizon,
        "stac_overlap_positions": pred_horizon - exec_horizon,
        "stac_supported": pred_horizon > exec_horizon,
        "irregular_inference_step_gaps": irregular_gaps,
        "tasks": sorted({row["task_name"] for row in rows}),
    }
    return write_cache(
        args.output_dir, rows=rows, features=features, actions=actions, provenance=provenance
    )


# ---------------------------------------------------------------------------
# Phase 2: success-only fitting and development selection
# ---------------------------------------------------------------------------


def _resolve_split_manifest(runtime, override):
    if override is not None:
        path = Path(override).resolve()
        return path, _sha256(path)
    seed = str(sorted(int(value) for value in runtime["model_seeds"])[0])
    artifact = runtime["checkpoint_bundle"][seed]["split_manifest.json"]
    path = Path(artifact["path"])
    return path, _verify_sha(path, artifact["sha256"], "Frozen split manifest")


def develop(args):
    output = _prepare_output(args.output_dir)
    baselines = validate_baselines(args.baselines)
    aggregations = validate_aggregations(args.aggregations)
    landmarks = _validate_fractions(args.selection_landmarks)
    grids = SMOKE_GRIDS if args.grid == "smoke" else DEFAULT_GRIDS
    runtime_path, runtime = _load_runtime_bundle(args.runtime_bundle)
    horizons = {str(task): int(value) for task, value in runtime["task_horizons"].items()}
    split_path, split_sha = _resolve_split_manifest(runtime, args.split_manifest)
    split = load_outer_split_ids(split_path)
    cache = FeatureCache(args.development_cache, expected_group="development")

    missing = sorted(split["train"] - set(cache.by_id))
    if missing:
        raise ValueError(f"Development cache lacks outer-training rollouts: {missing[:5]}")
    rows = [cache.by_id[rollout_id] for rollout_id in sorted(split["train"])]
    if {row["task_name"] for row in rows} != set(horizons):
        raise ValueError("Outer-training rollouts do not cover exactly the frozen tasks")
    frozen_development = set(runtime.get("development_rollout_ids", []))
    if frozen_development and not split["train"] <= frozen_development:
        raise ValueError("Outer-training IDs are not frozen development rollouts")
    meta = stratified_meta_split(rows, args.validation_fraction, args.meta_split_seed)
    fit_rows = [row for row in rows if row["rollout_id"] in meta["fit"]]
    validation_rows = [row for row in rows if row["rollout_id"] in meta["validation"]]

    success_features = defaultdict(list)
    for row in fit_rows:
        if not bool(row["failed"]):
            success_features[row["task_name"]].append(cache.features_for(row))
    success_features = {
        task: np.concatenate(values) for task, values in sorted(success_features.items())
    }
    validation_inputs = [
        (row, cache.features_for(row), cache.actions_for(row)) for row in validation_rows
    ]
    exec_horizon = int(cache.provenance["exec_horizon"])
    pred_horizon = int(cache.provenance["pred_horizon"])
    stac_supported = pred_horizon > exec_horizon

    models_dir = output / "models"
    models_dir.mkdir()
    audit, selected, skipped = [], {}, {}
    validation_scores = {row["rollout_id"]: {} for row in validation_rows}
    for baseline in baselines:
        if baseline == "action_inconsistency" and not stac_supported:
            skipped[baseline] = (
                f"pred_horizon {pred_horizon} does not exceed exec_horizon "
                f"{exec_horizon}; consecutive chunks do not overlap"
            )
            continue
        best = None
        for config_index, params in enumerate(grids[baseline]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = BaselineScorer(baseline, params, seed=args.model_seed).fit(
                    success_features,
                    exec_horizon=exec_horizon,
                    max_fit_rows=args.max_fit_rows,
                )
            raw = {
                row["rollout_id"]: model.score(row["task_name"], features, actions)
                for row, features, actions in validation_inputs
            }
            for aggregation_index, aggregation in enumerate(aggregations):
                trajectories = [
                    {
                        "task_name": row["task_name"],
                        "failed": bool(row["failed"]),
                        "trajectory": aggregate_scores(raw[row["rollout_id"]], aggregation),
                    }
                    for row in validation_rows
                ]
                landmark_rows = landmark_auc_rows(
                    trajectories, horizons, "trajectory", landmarks
                )
                macro = [
                    row["task_macro_roc_auc"]
                    for row in landmark_rows
                    if row["task_macro_roc_auc"] is not None
                ]
                candidate = {
                    "baseline": baseline,
                    "candidate_index": config_index * len(aggregations) + aggregation_index,
                    "params": json.dumps(params, sort_keys=True),
                    "aggregation": aggregation,
                    "selection_score": float(np.mean(macro)) if macro else None,
                    "event_task_macro_roc_auc": event_task_macro_auc(
                        trajectories, horizons, "trajectory"
                    ),
                }
                for row in landmark_rows:
                    key = f"landmark_{row['landmark_fraction']:g}"
                    candidate[f"{key}_task_macro_roc_auc"] = row["task_macro_roc_auc"]
                    candidate[f"{key}_supported_tasks"] = row["supported_tasks"]
                audit.append(candidate)
                if best is None or selection_key(candidate) > selection_key(best[0]):
                    best = (candidate, model, params, aggregation, trajectories)
        candidate, model, params, aggregation, trajectories = best
        normalization = fit_safe_normalization(
            [
                {"task_name": row["task_name"], "scores": row["trajectory"]}
                for row in trajectories
            ],
            horizons,
        )
        for row, trajectory in zip(validation_rows, trajectories):
            validation_scores[row["rollout_id"]][baseline] = normalize_trajectory(
                trajectory["trajectory"], normalization[row["task_name"]]
            )
        model_path = models_dir / f"{baseline}.pkl"
        with model_path.open("wb") as stream:
            pickle.dump(model, stream, protocol=4)
        selected[baseline] = {
            "params": params,
            "aggregation": aggregation,
            "selection_score": candidate["selection_score"],
            "event_task_macro_roc_auc": candidate["event_task_macro_roc_auc"],
            "candidate_index": candidate["candidate_index"],
            "candidates": sum(row["baseline"] == baseline for row in audit),
            "normalization": normalization,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "fit_summary": model.fit_summary,
        }
    if not selected:
        raise ValueError("No baseline could be developed")
    _write_csv(output / "selection_audit.csv", audit)
    _write_jsonl(
        output / "validation_scores.jsonl",
        [
            {
                "rollout_id": row["rollout_id"],
                "task_name": row["task_name"],
                "failed": bool(row["failed"]),
                "baseline_scores": validation_scores[row["rollout_id"]],
            }
            for row in validation_rows
        ],
    )
    bundle = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "phase": "development_frozen",
        "outcomes_read": "development outer-training rollouts only",
        "runtime_bundle": str(runtime_path),
        "runtime_bundle_sha256": _sha256(runtime_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": split_sha,
        "development_cache": str(cache.root),
        "development_cache_provenance": cache.provenance,
        "grid": args.grid,
        "grids": {name: grids[name] for name in baselines},
        "aggregations": list(aggregations),
        "selection_landmarks": list(landmarks),
        "selection_rule": (
            "maximize mean task-macro at-risk ROC-AUC over selection landmarks; "
            "tie-break by task-macro event ROC-AUC, then grid order"
        ),
        "validation_fraction": float(args.validation_fraction),
        "meta_split_seed": int(args.meta_split_seed),
        "model_seed": int(args.model_seed),
        "max_fit_rows": int(args.max_fit_rows),
        "task_horizons": horizons,
        "exec_horizon": exec_horizon,
        "pred_horizon": pred_horizon,
        "stac_supported": stac_supported,
        "fit_rollout_ids": sorted(meta["fit"]),
        "validation_rollout_ids": sorted(meta["validation"]),
        "meta_counts": meta["counts"],
        "fit_success_rollouts": sum(not bool(row["failed"]) for row in fit_rows),
        "unused_fit_failure_rollouts": sum(bool(row["failed"]) for row in fit_rows),
        "selected": selected,
        "skipped": skipped,
    }
    bundle_path = output / "baseline_bundle.json"
    _write_json(bundle_path, bundle)
    status = {
        "status": "development_complete",
        "baseline_bundle": str(bundle_path),
        "selected": {
            name: {
                "params": value["params"],
                "aggregation": value["aggregation"],
                "selection_score": value["selection_score"],
            }
            for name, value in selected.items()
        },
        "skipped": skipped,
    }
    _write_json(output / "status.json", status)
    return status


# ---------------------------------------------------------------------------
# Shared detector machinery
# ---------------------------------------------------------------------------


def detector_specs(baselines):
    specs = [
        {"name": "time_only", "base": "time_only", "score": "safe", "family": "frozen"},
        {"name": "safe_only", "base": "safe_only", "score": "safe", "family": "frozen"},
        {
            "name": "staged_safe_time",
            "base": "staged_safe_time",
            "score": "safe",
            "family": "frozen",
        },
        {
            "name": "fused_safe_time",
            "base": "fused_safe_time",
            "score": "safe",
            "family": "staging_ablation",
        },
    ]
    for baseline in baselines:
        specs.append(
            {
                "name": f"{baseline}_only",
                "base": "safe_only",
                "score": baseline,
                "family": "baseline",
            }
        )
        specs.append(
            {
                "name": f"staged_{baseline}_time",
                "base": "staged_safe_time",
                "score": baseline,
                "family": "staged_baseline",
            }
        )
    return specs


def detector_view(record, spec):
    """Place a spec's score trajectory in the frozen module's SAFE slot."""
    score = (
        record["safe_scores"]
        if spec["score"] == "safe"
        else record["baseline_scores"][spec["score"]]
    )
    return {
        "rollout_id": record["rollout_id"],
        "task_name": record["task_name"],
        "task_type": record.get("task_type"),
        "failed": record["failed"],
        "horizon": record["horizon"],
        "source_inferences": record["source_inferences"],
        "safe_scores": score,
        "time_scores": record["time_scores"],
    }


def _views(records, spec):
    return [detector_view(record, spec) for record in records]


def _select_thresholds(job):
    name, base, views, target_fpr, safe_end, time_start = job
    if base in ("safe_only", "time_only"):
        selection = select_single_threshold(
            views, base, target_fpr=target_fpr, safe_end=safe_end, time_start=time_start
        )
    else:
        selection = select_staged_thresholds(
            views,
            target_fpr=target_fpr,
            safe_end=safe_end,
            time_start=time_start,
            detector=base,
        )
    curve = [
        {
            "detector": name,
            **{f"threshold_{key}": value for key, value in row["thresholds"].items()},
            "true_positive_rate": row["true_positive_rate"],
            "false_positive_rate": row["false_positive_rate"],
            "missed_failure_adjusted_detection_fraction": row[
                "missed_failure_adjusted_detection_fraction"
            ],
        }
        for row in selection.pop("curve")
    ]
    return name, selection, curve


def _run_jobs(jobs, workers):
    if int(workers) <= 1:
        return [_select_thresholds(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(_select_thresholds, jobs))


def _load_baseline_models(bundle):
    models = {}
    for name, selection in bundle["selected"].items():
        path = Path(selection["model_path"])
        _verify_sha(path, selection["model_sha256"], f"Frozen {name} baseline model")
        with path.open("rb") as stream:
            models[name] = pickle.load(stream)
    return models


def score_records(records, cache, bundle, models):
    """Attach frozen, task-normalized baseline trajectories to each record."""
    for record in records:
        row = cache.by_id.get(record["rollout_id"])
        if row is None:
            raise ValueError(f"Cache lacks rollout {record['rollout_id']}")
        if (
            int(row["length"]) != int(record["source_inferences"])
            or row["task_name"] != record["task_name"]
            or bool(row["failed"]) != bool(record["failed"])
        ):
            raise ValueError(f"Cache and frozen predictions disagree for {record['rollout_id']}")
        features = cache.features_for(row)
        actions = cache.actions_for(row)
        record["baseline_scores"] = {}
        for name, model in models.items():
            selection = bundle["selected"][name]
            raw = model.score(record["task_name"], features, actions)
            trajectory = aggregate_scores(raw, selection["aggregation"])
            record["baseline_scores"][name] = normalize_trajectory(
                trajectory, selection["normalization"][record["task_name"]]
            )
    return records


def _primary_per_class(per_task_counts):
    values = {
        int(outcomes[outcome]["selected"])
        for outcomes in per_task_counts.values()
        for outcome in ("successes", "failures")
    }
    if len(values) != 1:
        raise ValueError(f"Frozen primary cohort is not balanced per task: {sorted(values)}")
    return values.pop()


def _same_numbers(expected, actual, tolerance=1e-9, path="result"):
    """Return mismatching paths between two JSON-like numeric structures."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [path]
        mismatches = []
        for key, value in expected.items():
            if key not in actual:
                mismatches.append(f"{path}.{key}")
            else:
                mismatches.extend(_same_numbers(value, actual[key], tolerance, f"{path}.{key}"))
        return mismatches
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [path]
        mismatches = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatches.extend(_same_numbers(left, right, tolerance, f"{path}[{index}]"))
        return mismatches
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return [] if expected == actual else [path]
    if isinstance(expected, (int, float)):
        if actual is None or abs(float(expected) - float(actual)) > tolerance:
            return [path]
        return []
    return [] if expected == actual else [path]


# ---------------------------------------------------------------------------
# Phase 3: calibration thresholds
# ---------------------------------------------------------------------------


def calibrate(args):
    output = _prepare_output(args.output_dir)
    baseline_path = Path(args.baseline_bundle).resolve()
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("phase") != "development_frozen":
        raise ValueError("Baseline bundle is not a frozen development artifact")
    runtime_path, runtime = _load_runtime_bundle(baseline["runtime_bundle"])
    _verify_sha(runtime_path, baseline["runtime_bundle_sha256"], "Frozen runtime bundle")
    models = _load_baseline_models(baseline)
    target_fpr = float(runtime["target_fpr"])
    safe_end, time_start = validate_stage_windows(
        runtime["safe_eligibility_end_fraction"], runtime["time_eligibility_start_fraction"]
    )
    predictions_path = Path(
        args.calibration_predictions
        or runtime_path.parent / "calibration_predictions.jsonl"
    ).resolve()
    records = _load_prediction_records(predictions_path)
    ids = {record["rollout_id"] for record in records}
    if ids != set(runtime["calibration_rollout_ids"]):
        raise ValueError("Calibration predictions differ from the frozen calibration IDs")
    cache = FeatureCache(args.calibration_cache, expected_group="calibration")
    if set(cache.by_id) != ids:
        raise ValueError("Calibration cache differs from the frozen calibration IDs")
    development_ids = set(baseline["fit_rollout_ids"]) | set(baseline["validation_rollout_ids"])
    if development_ids & ids:
        raise ValueError("Baseline development and calibration rollout IDs overlap")
    score_records(records, cache, baseline, models)
    per_class = _primary_per_class(runtime["calibration_counts"]["per_task"])
    primary, _ = _select_per_task_class(records, per_class)
    if {row["rollout_id"] for row in primary} != set(
        runtime["primary_calibration_rollout_ids"]
    ):
        raise ValueError("Primary calibration cohort differs from the frozen cohort")

    specs = detector_specs(tuple(baseline["selected"]))
    jobs = [
        (spec["name"], spec["base"], _views(primary, spec), target_fpr, safe_end, time_start)
        for spec in specs
    ]
    results = _run_jobs(jobs, args.workers)
    selections = {name: selection for name, selection, _ in results}
    reproduction = {}
    for detector in DETECTORS:
        frozen = {
            key: float(value)
            for key, value in runtime["threshold_selection"][detector]["thresholds"].items()
        }
        recomputed = {
            key: float(value) for key, value in selections[detector]["thresholds"].items()
        }
        if frozen != recomputed:
            raise ValueError(
                f"Frozen {detector} thresholds were not reproduced: "
                f"frozen={frozen}, recomputed={recomputed}"
            )
        reproduction[detector] = frozen
    _write_csv(output / "threshold_curves.csv", [row for _, _, curve in results for row in curve])
    detectors = {}
    for spec in specs:
        selection = selections[spec["name"]]
        detectors[spec["name"]] = {
            **spec,
            "thresholds": selection["thresholds"],
            "calibration_metrics": selection["validation_metrics"],
            "search": selection.get("search", {"feasible": selection.get("feasible_thresholds")}),
        }
    artifact = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "phase": "calibration_frozen",
        "baseline_bundle": str(baseline_path),
        "baseline_bundle_sha256": _sha256(baseline_path),
        "runtime_bundle": str(runtime_path),
        "runtime_bundle_sha256": _sha256(runtime_path),
        "calibration_predictions": str(predictions_path),
        "calibration_predictions_sha256": _sha256(predictions_path),
        "calibration_cache": str(cache.root),
        "calibration_cache_provenance": cache.provenance,
        "target_fpr": target_fpr,
        "safe_eligibility_end_fraction": safe_end,
        "time_eligibility_start_fraction": time_start,
        "primary_per_class": per_class,
        "primary_calibration_rollouts": len(primary),
        "frozen_thresholds_reproduced": reproduction,
        "detectors": detectors,
    }
    artifact_path = output / "baseline_calibration.json"
    _write_json(artifact_path, artifact)
    _write_jsonl(output / "calibration_predictions.jsonl", records)
    status = {
        "status": "calibration_complete",
        "baseline_calibration": str(artifact_path),
        "primary_calibration_rollouts": len(primary),
        "frozen_thresholds_reproduced": True,
        "detectors": {
            name: {
                "thresholds": value["thresholds"],
                "true_positive_rate": value["calibration_metrics"]["true_positive_rate"],
                "false_positive_rate": value["calibration_metrics"]["false_positive_rate"],
            }
            for name, value in detectors.items()
        },
    }
    _write_json(output / "status.json", status)
    return status


# ---------------------------------------------------------------------------
# Phase 4: one prospective evaluation
# ---------------------------------------------------------------------------


def detector_outcomes(records, spec, thresholds, safe_end, time_start):
    """Per-rollout first alarms, matching ``event_metrics`` exactly."""
    alarms, fractions, steps, sources = [], [], [], []
    for record in records:
        alert = first_alert(
            detector_view(record, spec), spec["base"], thresholds, safe_end, time_start
        )
        alarms.append(alert is not None)
        fractions.append(
            1.0 if alert is None else min(1.0, int(alert["step"]) / int(record["horizon"]))
        )
        steps.append(None if alert is None else int(alert["step"]))
        sources.append(None if alert is None else alert["source"])
    return {
        "alarm": np.asarray(alarms, dtype=bool),
        "fraction": np.asarray(fractions, dtype=np.float64),
        "step": steps,
        "source": sources,
    }


def paired_comparison(records, outcome_a, outcome_b):
    """Same counts as ``paired_failure_comparison`` for arbitrary score slots."""
    counts = {"earlier": 0, "same": 0, "later": 0, "a_missed": 0, "b_missed": 0}
    differences = []
    for index, record in enumerate(records):
        if not bool(record["failed"]):
            continue
        a_fraction = float(outcome_a["fraction"][index])
        b_fraction = float(outcome_b["fraction"][index])
        differences.append(float(a_fraction - b_fraction))
        counts["a_missed"] += int(not outcome_a["alarm"][index])
        counts["b_missed"] += int(not outcome_b["alarm"][index])
        if a_fraction < b_fraction - 1e-12:
            counts["earlier"] += 1
        elif a_fraction > b_fraction + 1e-12:
            counts["later"] += 1
        else:
            counts["same"] += 1
    return {**counts, "mean_a_minus_b_detection_fraction": float(np.mean(differences))}


def task_rollout_bootstrap(records, outcomes, comparisons, samples, seed):
    """Task-then-rollout bootstrap with the frozen resampling sequence.

    For the staged-versus-time pair this reproduces the frozen ``_bootstrap``
    values exactly; every other pair is evaluated on the same replicates.
    """
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for index, record in enumerate(records):
        by_task[record["task_name"]].append(index)
    tasks = sorted(by_task)
    failed = np.asarray([bool(record["failed"]) for record in records], dtype=bool)
    values = {pair: defaultdict(list) for pair in comparisons}
    replicates = 0
    for _ in range(int(samples)):
        chosen = rng.choice(tasks, size=len(tasks), replace=True)
        indices = []
        for task in chosen:
            group = by_task[str(task)]
            selected = rng.choice(len(group), size=len(group), replace=True)
            indices.extend(group[int(row)] for row in selected)
        indices = np.asarray(indices, dtype=np.int64)
        sample_failed = failed[indices]
        if sample_failed.all() or not sample_failed.any():
            continue
        replicates += 1
        failures = indices[sample_failed]
        successes = indices[~sample_failed]
        for pair in comparisons:
            a, b = (outcomes[name] for name in pair)
            values[pair]["adjusted_detection_delta"].append(
                float(np.mean(a["fraction"][failures])) - float(np.mean(b["fraction"][failures]))
            )
            values[pair]["tpr_delta"].append(
                float(np.mean(a["alarm"][failures])) - float(np.mean(b["alarm"][failures]))
            )
            values[pair]["fpr_delta"].append(
                float(np.mean(a["alarm"][successes])) - float(np.mean(b["alarm"][successes]))
            )
    result = {}
    for pair in comparisons:
        summary = {}
        for key in ("adjusted_detection_delta", "tpr_delta", "fpr_delta"):
            array = np.asarray(values[pair][key], dtype=np.float64)
            summary[key] = {
                "mean": float(np.mean(array)),
                "ci95": np.quantile(array, [0.025, 0.975]).tolist(),
            }
        summary["replicates"] = replicates
        result[pair] = summary
    return result


def _score_source_landmarks(records, horizons, sources):
    rows = []
    for source in sources:
        key = f"__landmark_{source}"
        for record in records:
            record[key] = (
                record["safe_scores"]
                if source == "safe"
                else record["time_scores"]
                if source == "time"
                else record["baseline_scores"][source]
            )
        for row in landmark_auc_rows(records, horizons, key, EVALUATION_LANDMARKS):
            rows.append({"score_source": source, **row})
        for record in records:
            del record[key]
    return rows


def _summary_rows(specs, overall, comparisons, bootstrap):
    rows = []
    for spec in specs:
        metrics = overall[spec["name"]]
        row = {
            "detector": spec["name"],
            "family": spec["family"],
            "base": spec["base"],
            "score_source": spec["score"],
            "thresholds": json.dumps(metrics["thresholds"], sort_keys=True),
            **metrics["confusion"],
            "true_positive_rate": metrics["true_positive_rate"],
            "tpr_wilson_95": metrics["tpr_wilson_95"],
            "false_positive_rate": metrics["false_positive_rate"],
            "fpr_wilson_95": metrics["fpr_wilson_95"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "missed_failure_adjusted_detection_fraction": metrics[
                "missed_failure_adjusted_detection_fraction"
            ],
        }
        for landmark, value in metrics["failure_recall_by_landmark"].items():
            row[f"failure_recall_by_{landmark}"] = value
        for reference in REFERENCE_DETECTORS:
            pair = (spec["name"], reference)
            if pair not in bootstrap:
                continue
            row[f"paired_vs_{reference}"] = json.dumps(comparisons[pair], sort_keys=True)
            for key, value in bootstrap[pair].items():
                if key != "replicates":
                    row[f"{key}_vs_{reference}_mean"] = value["mean"]
                    row[f"{key}_vs_{reference}_ci95"] = value["ci95"]
        rows.append(row)
    return rows


def evaluate(args):
    if int(args.test_per_class) < 1 or int(args.bootstrap_samples) < 1:
        raise ValueError("--test-per-class and --bootstrap-samples must be positive")
    output = _prepare_output(args.output_dir)
    calibration_path = Path(args.baseline_calibration).resolve()
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("phase") != "calibration_frozen":
        raise ValueError("Baseline calibration is not a frozen artifact")
    baseline_path = Path(calibration["baseline_bundle"])
    _verify_sha(baseline_path, calibration["baseline_bundle_sha256"], "Frozen baseline bundle")
    baseline = json.loads(baseline_path.read_text())
    runtime_path, runtime = _load_runtime_bundle(calibration["runtime_bundle"])
    _verify_sha(runtime_path, calibration["runtime_bundle_sha256"], "Frozen runtime bundle")
    models = _load_baseline_models(baseline)
    safe_end = float(calibration["safe_eligibility_end_fraction"])
    time_start = float(calibration["time_eligibility_start_fraction"])
    horizons = {str(task): int(value) for task, value in runtime["task_horizons"].items()}
    frozen_analysis_path = Path(args.frozen_analysis).resolve()
    frozen_analysis = json.loads(frozen_analysis_path.read_text())
    if frozen_analysis.get("status") != "complete":
        raise ValueError("Frozen prospective analysis is not complete")
    predictions_path = Path(
        args.test_predictions or frozen_analysis_path.parent / "event_predictions.jsonl"
    ).resolve()
    records = _load_prediction_records(predictions_path)
    ids = {record["rollout_id"] for record in records}
    if {str(record["task_name"]) for record in records} != set(horizons):
        raise ValueError("Prospective predictions lack exact frozen task coverage")
    cache = FeatureCache(args.test_cache, expected_group="prospective_test")
    if set(cache.by_id) != ids:
        raise ValueError("Prospective cache differs from the frozen prospective predictions")
    development_ids = (
        set(runtime.get("development_rollout_ids", []))
        | set(baseline["fit_rollout_ids"])
        | set(baseline["validation_rollout_ids"])
    )
    if development_ids & ids:
        raise ValueError("Development and prospective rollout IDs overlap")
    if set(runtime["calibration_rollout_ids"]) & ids:
        raise ValueError("Calibration and prospective rollout IDs overlap")
    calibration_identity = {
        tuple(value) for value in runtime["calibration_seed_reset_identities"]
    }
    overlap = calibration_identity & _identity_keys(records)
    if overlap:
        raise ValueError(f"Calibration and prospective identities overlap: {sorted(overlap)[:3]}")
    score_records(records, cache, baseline, models)
    primary, primary_selection = _select_per_task_class(records, args.test_per_class)

    specs = [
        {key: value for key, value in detector.items() if key in ("name", "base", "score", "family")}
        for detector in calibration["detectors"].values()
    ]
    thresholds = {
        name: detector["thresholds"] for name, detector in calibration["detectors"].items()
    }

    def metrics_for(cohort):
        return {
            spec["name"]: event_metrics(
                _views(cohort, spec),
                spec["base"],
                thresholds[spec["name"]],
                safe_end=safe_end,
                time_start=time_start,
            )
            for spec in specs
        }

    overall = metrics_for(primary)
    secondary = metrics_for(records)
    outcomes = {
        spec["name"]: detector_outcomes(
            primary, spec, thresholds[spec["name"]], safe_end, time_start
        )
        for spec in specs
    }
    comparisons = {}
    for spec in specs:
        for reference in REFERENCE_DETECTORS:
            if spec["name"] != reference:
                comparisons[(spec["name"], reference)] = paired_comparison(
                    primary, outcomes[spec["name"]], outcomes[reference]
                )
    bootstrap = task_rollout_bootstrap(
        primary,
        outcomes,
        tuple(comparisons),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    mismatches = []
    for detector in DETECTORS:
        mismatches += _same_numbers(
            frozen_analysis["overall"][detector], overall[detector], path=f"overall.{detector}"
        )
        mismatches += _same_numbers(
            frozen_analysis["all_retained_secondary"][detector],
            secondary[detector],
            path=f"all_retained_secondary.{detector}",
        )
    mismatches += _same_numbers(
        frozen_analysis["paired_staged_minus_time"],
        comparisons[("staged_safe_time", "time_only")],
        path="paired_staged_minus_time",
    )
    mismatches += _same_numbers(
        frozen_analysis["task_rollout_bootstrap"],
        bootstrap[("staged_safe_time", "time_only")],
        path="task_rollout_bootstrap",
    )
    if mismatches:
        raise ValueError(
            "Frozen prospective results were not reproduced; pass the frozen "
            f"--test-per-class/--bootstrap-* values. Mismatches: {mismatches[:8]}"
        )

    by_type = {}
    for task_type in ("atomic", "composite"):
        cohort = [row for row in primary if row.get("task_type") == task_type]
        if {bool(row["failed"]) for row in cohort} == {False, True}:
            by_type[task_type] = metrics_for(cohort)
    calibration_metrics = {
        name: detector["calibration_metrics"] for name, detector in calibration["detectors"].items()
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol": PROTOCOL,
        "baseline_calibration": str(calibration_path),
        "baseline_calibration_sha256": _sha256(calibration_path),
        "baseline_bundle": str(baseline_path),
        "runtime_bundle": str(runtime_path),
        "frozen_analysis": str(frozen_analysis_path),
        "test_predictions": str(predictions_path),
        "test_cache": str(cache.root),
        "test_cache_provenance": cache.provenance,
        "frozen_results_reproduced": True,
        "thresholds_updated_on_test": False,
        "policy_intervention": False,
        "development_ids_disjoint": True,
        "calibration_ids_disjoint": True,
        "calibration_seed_reset_identities_disjoint": True,
        "counts": {
            "primary_balanced": {
                "rollouts": len(primary),
                "successes": sum(not row["failed"] for row in primary),
                "failures": sum(row["failed"] for row in primary),
            },
            "retained_all": {
                "rollouts": len(records),
                "successes": sum(not row["failed"] for row in records),
                "failures": sum(row["failed"] for row in records),
            },
            "tasks": len({row["task_name"] for row in records}),
            "primary_selection": primary_selection,
        },
        "detectors": {
            spec["name"]: {**spec, "thresholds": thresholds[spec["name"]]} for spec in specs
        },
        "baseline_selection": {
            name: {
                key: value
                for key, value in selection.items()
                if key in ("params", "aggregation", "selection_score", "event_task_macro_roc_auc")
            }
            for name, selection in baseline["selected"].items()
        },
        "skipped_baselines": baseline["skipped"],
        "overall": overall,
        "all_retained_secondary": secondary,
        "by_task_type": by_type,
        "landmarks": _score_source_landmarks(
            primary, horizons, ("safe", "time", *baseline["selected"])
        ),
        "paired": {f"{a}__vs__{b}": value for (a, b), value in comparisons.items()},
        "task_rollout_bootstrap": {
            f"{a}__vs__{b}": value for (a, b), value in bootstrap.items()
        },
        "calibration_to_test_fpr_drift": {
            name: overall[name]["false_positive_rate"]
            - calibration_metrics[name]["false_positive_rate"]
            for name in overall
        },
    }
    _write_json(output / "analysis.json", result)
    _write_csv(
        output / "detector_summary.csv", _summary_rows(specs, overall, comparisons, bootstrap)
    )
    primary_ids = {row["rollout_id"] for row in primary}
    _write_jsonl(
        output / "event_predictions.jsonl",
        [
            {
                **{key: record.get(key) for key in PREDICTION_KEYS},
                "primary_balanced": record["rollout_id"] in primary_ids,
                "baseline_scores": record["baseline_scores"],
            }
            for record in records
        ],
    )
    _write_json(
        output / "status.json",
        {"status": "complete", "analysis": str((output / "analysis.json").resolve())},
    )
    return result


# ---------------------------------------------------------------------------
# Synthetic end-to-end smoke
# ---------------------------------------------------------------------------


def write_synthetic_fixture(root, *, seed=0, feature_dim=12):
    """Create a miniature frozen protocol plus caches for all three groups.

    Frozen calibration and evaluation are produced with the real
    ``run_prospective_matched_fpr`` phases so the reproduction checks are
    exercised exactly as on the cluster.
    """
    import hashlib

    from . import run_prospective_matched_fpr as frozen

    root = Path(root).resolve()
    rng = np.random.default_rng(seed)
    tasks = {"SynthAtomic": "atomic", "SynthComposite": "composite"}
    horizon, pred_horizon, exec_horizon, action_dim = 12, 6, 2, 3
    task_means = {task: rng.normal(size=feature_dim) for task in tasks}

    def rollout(task, failed, ident):
        length = horizon if failed else int(rng.integers(6, 10))
        onset = int(rng.integers(2, 5))
        features = task_means[task] + 0.3 * rng.normal(size=(length, feature_dim))
        chunks = np.cumsum(0.05 * rng.normal(size=(length, pred_horizon, action_dim)), axis=1)
        if failed:
            features[onset:] += 1.5
            chunks[onset:] += 0.6 * rng.normal(size=(length - onset, pred_horizon, action_dim))
        safe = 0.2 + 0.05 * rng.normal(size=length) + (0.5 if failed else 0.0) * (
            np.arange(length) >= onset
        )
        return {
            "rollout_id": ident,
            "task_name": task,
            "task_type": tasks[task],
            "failed": bool(failed),
            "features": features.astype(np.float32),
            "actions": chunks.astype(np.float32),
            "safe": safe.tolist(),
        }

    groups = {
        "development": [],
        "calibration": [],
        "prospective_test": [],
    }
    split = {"train": [], "test": []}
    for task in tasks:
        for failed, count in ((False, 10), (True, 8)):
            for index in range(count):
                item = rollout(task, failed, f"dev-{task}-{int(failed)}-{index}")
                groups["development"].append(item)
                split["test" if index == 0 else "train"].append(item["rollout_id"])
        for group, count in (("calibration", 5), ("prospective_test", 6)):
            for failed in (False, True):
                for index in range(count):
                    groups[group].append(rollout(task, failed, f"{group}-{task}-{int(failed)}-{index}"))

    final = root / "final"
    for model_seed in (0, 1, 2):
        run = final / f"indep_seed{model_seed}"
        run.mkdir(parents=True)
        rows = [
            {
                "rollout_id": item["rollout_id"],
                "split": "test" if item["rollout_id"] in split["test"] else "train",
                "task_name": item["task_name"],
                "task_type": item["task_type"],
                "failed": item["failed"],
                "scores": [value + 0.01 * model_seed for value in item["safe"]],
            }
            for item in groups["development"]
        ]
        _write_jsonl(run / "scores.jsonl", rows)
        _write_json(run / "split_manifest.json", {**split, "task_names": sorted(tasks)})
        for name in ("model_final.ckpt", "config.yaml", "metrics.json"):
            (run / name).write_text(f"synthetic seed={model_seed} {name}\n")
        checkpoint_sha = hashlib.sha256((run / "model_final.ckpt").read_bytes()).hexdigest()
        for group, score_group in (
            ("calibration", "calibration"),
            ("prospective_test", "prospective_test"),
        ):
            score_root = root / f"{group}_scores" / f"seed{model_seed}"
            score_root.mkdir(parents=True)
            _write_jsonl(
                score_root / "scores.jsonl",
                [
                    {
                        "rollout_id": item["rollout_id"],
                        "task_name": item["task_name"],
                        "task_type": item["task_type"],
                        "failed": item["failed"],
                        "scores": [value + 0.01 * model_seed for value in item["safe"]],
                        "environment_seed": (1000 if group == "calibration" else 2000) + index,
                        "environment_reset_index": 0,
                    }
                    for index, item in enumerate(groups[group])
                ],
            )
            _write_json(
                score_root / "provenance.json",
                {
                    "group": score_group,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_updated": False,
                },
            )

    calibration_dir = root / "frozen_calibration"
    frozen.calibrate(
        argparse.Namespace(
            final_root=str(final),
            calibration_score_root=str(root / "calibration_scores"),
            output_dir=str(calibration_dir),
            seeds=[0, 1, 2],
            target_fpr=0.05,
            safe_end=0.25,
            time_start=0.50,
            time_prior=0.5,
            min_per_class=3,
        )
    )
    test_dir = root / "frozen_prospective"
    frozen.evaluate(
        argparse.Namespace(
            runtime_bundle=str(calibration_dir / "runtime_bundle.json"),
            evaluation_score_root=str(root / "prospective_test_scores"),
            output_dir=str(test_dir),
            bootstrap_samples=50,
            bootstrap_seed=0,
            test_per_class=4,
        )
    )
    caches = {}
    for group, items in groups.items():
        caches[group] = root / f"cache_{group}"
        write_cache(
            caches[group],
            rows=[
                {
                    key: item[key]
                    for key in ("rollout_id", "task_name", "task_type", "failed")
                }
                | {
                    "environment_seed": None,
                    "environment_reset_index": None,
                    "replan_steps": exec_horizon,
                }
                for item in items
            ],
            features=[item["features"] for item in items],
            actions=[item["actions"] for item in items],
            provenance={
                "schema_version": 1,
                "protocol": PROTOCOL,
                "group": group,
                "synthetic": True,
                "feature_dim": feature_dim,
                "pred_horizon": pred_horizon,
                "dim_action": action_dim,
                "exec_horizon": exec_horizon,
                "stac_supported": True,
            },
        )
    return {
        "runtime_bundle": calibration_dir / "runtime_bundle.json",
        "frozen_analysis": test_dir / "analysis.json",
        "caches": caches,
        "bootstrap_samples": 50,
        "test_per_class": 4,
    }


def synthetic_smoke(args):
    output = _prepare_output(args.output_dir)
    fixture = write_synthetic_fixture(output / "fixture", seed=args.seed)
    development = develop(
        argparse.Namespace(
            development_cache=str(fixture["caches"]["development"]),
            runtime_bundle=str(fixture["runtime_bundle"]),
            split_manifest=None,
            output_dir=str(output / "develop"),
            baselines=list(BASELINES),
            aggregations=list(AGGREGATIONS),
            grid="smoke",
            selection_landmarks=list(DEFAULT_SELECTION_LANDMARKS),
            validation_fraction=0.3,
            meta_split_seed=0,
            model_seed=0,
            max_fit_rows=50000,
        )
    )
    calibration = calibrate(
        argparse.Namespace(
            baseline_bundle=development["baseline_bundle"],
            calibration_cache=str(fixture["caches"]["calibration"]),
            calibration_predictions=None,
            output_dir=str(output / "calibrate"),
            workers=args.workers,
        )
    )
    result = evaluate(
        argparse.Namespace(
            baseline_calibration=calibration["baseline_calibration"],
            test_cache=str(fixture["caches"]["prospective_test"]),
            test_predictions=None,
            frozen_analysis=str(fixture["frozen_analysis"]),
            output_dir=str(output / "evaluate"),
            test_per_class=fixture["test_per_class"],
            bootstrap_samples=fixture["bootstrap_samples"],
            bootstrap_seed=0,
        )
    )
    status = {
        "status": "synthetic_smoke_complete",
        "frozen_results_reproduced": result["frozen_results_reproduced"],
        "detectors": sorted(result["overall"]),
        "output_dir": str(output),
    }
    _write_json(output / "status.json", status)
    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    extraction = subparsers.add_parser("extract")
    extraction.add_argument("--export-dir", required=True)
    extraction.add_argument("--safe-repo", required=True)
    extraction.add_argument("--safe-config", required=True)
    extraction.add_argument("--group", choices=CACHE_GROUPS, required=True)
    extraction.add_argument("--output-dir", required=True)
    extraction.add_argument("--expected-horizon-selector", default="1.0")
    extraction.add_argument("--expected-diffusion-selector", default="1.0")

    development = subparsers.add_parser("develop")
    development.add_argument("--development-cache", required=True)
    development.add_argument("--runtime-bundle", required=True)
    development.add_argument("--split-manifest")
    development.add_argument("--output-dir", required=True)
    development.add_argument("--baselines", nargs="+", default=list(BASELINES))
    development.add_argument("--aggregations", nargs="+", default=list(AGGREGATIONS))
    development.add_argument("--grid", choices=("full", "smoke"), default="full")
    development.add_argument(
        "--selection-landmarks",
        nargs="+",
        type=float,
        default=list(DEFAULT_SELECTION_LANDMARKS),
    )
    development.add_argument("--validation-fraction", type=float, default=0.3)
    development.add_argument("--meta-split-seed", type=int, default=0)
    development.add_argument("--model-seed", type=int, default=0)
    development.add_argument("--max-fit-rows", type=int, default=50000)

    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--baseline-bundle", required=True)
    calibration.add_argument("--calibration-cache", required=True)
    calibration.add_argument("--calibration-predictions")
    calibration.add_argument("--output-dir", required=True)
    calibration.add_argument("--workers", type=int, default=1)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--baseline-calibration", required=True)
    evaluation.add_argument("--test-cache", required=True)
    evaluation.add_argument("--frozen-analysis", required=True)
    evaluation.add_argument("--test-predictions")
    evaluation.add_argument("--output-dir", required=True)
    evaluation.add_argument("--test-per-class", type=int, default=10)
    evaluation.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluation.add_argument("--bootstrap-seed", type=int, default=0)

    smoke = subparsers.add_parser("synthetic-smoke")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--workers", type=int, default=1)
    return parser


PHASES = {
    "extract": extract,
    "develop": develop,
    "calibrate": calibrate,
    "evaluate": evaluate,
    "synthetic-smoke": synthetic_smoke,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = PHASES[args.phase](args)
    except (FileExistsError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    if args.phase in ("evaluate",):
        result = {
            "status": result["status"],
            "frozen_results_reproduced": result["frozen_results_reproduced"],
            "counts": result["counts"]["primary_balanced"],
            "overall": {
                name: {
                    "tpr": row["true_positive_rate"],
                    "fpr": row["false_positive_rate"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "miss_adjusted": row["missed_failure_adjusted_detection_fraction"],
                }
                for name, row in result["overall"].items()
            },
        }
    print(json.dumps(_json_value(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
