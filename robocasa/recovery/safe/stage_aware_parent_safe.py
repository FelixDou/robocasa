"""Full-parent, stage-aware causal SAFE utilities.

The earlier dense Subtask-SAFE export materialized each semantic stage as an
independent pseudo-rollout.  That is convenient for the pinned SAFE loader but
resets temporal context at every stage boundary.  This module instead consumes
the original raw rollout dataset: one complete SAFE tensor and one aligned
semantic trace per parent rollout.  It creates one causal training row per
genuine policy inference while keeping the parent sequence intact.

Future, unattempted stages are censored.  A completed stage is a successful
stage and only the terminal active stage of a failed parent is a failed stage.
No final rollout duration, future observation, or future stage identity enters
an online feature.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random

import numpy as np

from .causal_prefix_residual import summarize_trajectory
from .dataset import aggregate_features, load_manifest
from .subtask_safe import validate_subtask_safe_record


ARMS = (
    "terminal",
    "stage",
    "multihorizon",
    "conditioned",
    "context",
    "prototype",
)
MODEL_ARMS = ARMS[:-1]
DEFAULT_FAILURE_HORIZONS = (4, 8, 16)
DEFAULT_PREFIXES = (1, 2, 4, 8)
DEFAULT_PRIMARY_PREFIXES = (1, 2)


def read_json(path):
    return json.loads(Path(path).read_text())


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(json_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def load_parent_split(path):
    """Load the parent contract produced by ``export_subtask_safe``."""
    path = Path(path).resolve()
    payload = read_json(path)
    train = set(payload.get("parent_train", []))
    test = set(payload.get("parent_test", []))
    if not train or not test or train & test:
        raise ValueError(
            "Parent split must contain nonempty, disjoint parent_train/parent_test"
        )
    return {
        "path": str(path),
        "payload": payload,
        "development": train,
        "locked_opened_outer": test,
    }


def _load_context(payload, key, expected_length):
    if key is None:
        return None
    candidates = (f"auxiliary__{key}", key)
    selected = next((name for name in candidates if name in payload), None)
    if selected is None:
        return None
    values = np.asarray(payload[selected], dtype=np.float32)
    if values.ndim < 2 or len(values) != int(expected_length):
        raise ValueError(
            f"Auxiliary context {selected!r} must start with inference axis "
            f"{expected_length}, got {values.shape}"
        )
    values = values.reshape(len(values), -1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Auxiliary context {selected!r} is not finite")
    return values


def _stage_spans(record, task_name, sequence_length):
    spans = []
    for segment in record.get("segments", []):
        label = segment.get("failure_label")
        start = segment.get("inference_start_index")
        end = segment.get("inference_end_index_exclusive")
        if label not in (0, 1) or start is None or end is None:
            continue
        start, end = int(start), int(end)
        if not 0 <= start < end <= int(sequence_length):
            raise ValueError(
                f"Invalid semantic inference span {start}:{end} for length "
                f"{sequence_length}"
            )
        spans.append(
            {
                "stage_name": f"{task_name}::{segment['subtask_id']}",
                "subtask_id": str(segment["subtask_id"]),
                "stage_index_in_task": int(segment["segment_index"]),
                "start": start,
                "end": end,
                "length": end - start,
                "failed": bool(label),
                "completed": not bool(label),
                "segment_id": str(segment["segment_id"]),
            }
        )
    spans.sort(key=lambda item: (item["start"], item["end"]))
    for left, right in zip(spans, spans[1:]):
        if left["end"] > right["start"]:
            raise ValueError(
                f"Overlapping semantic spans {left['segment_id']} and "
                f"{right['segment_id']}"
            )
    return spans


def load_parent_sequences(
    dataset_dir,
    *,
    parent_ids=None,
    tasks=None,
    aggregation="first",
    context_key="observation_state_history",
    require_context=False,
):
    """Load immutable raw parents and validate inference-aligned stage traces."""
    dataset_dir = Path(dataset_dir).resolve()
    requested = None if parent_ids is None else {str(value) for value in parent_ids}
    tasks = None if tasks is None else {str(value) for value in tasks}
    parents = []
    missing_context = []
    records = load_manifest(dataset_dir)
    record_ids = {record.rollout_id for record in records}
    if requested is not None:
        missing = sorted(requested - record_ids)
        if missing:
            raise ValueError(
                "Requested parent IDs are absent from the raw dataset: "
                + ", ".join(missing[:5])
            )
    for metadata in records:
        if requested is not None and metadata.rollout_id not in requested:
            continue
        if tasks is not None and metadata.task_name not in tasks:
            continue
        if not metadata.subtask_recording_available or not metadata.subtask_trace_path:
            raise ValueError(
                f"Parent {metadata.rollout_id} has no recorded semantic trace"
            )
        trace_path = dataset_dir / metadata.subtask_trace_path
        trace = read_json(trace_path)
        validate_subtask_safe_record(
            trace,
            rollout_id=metadata.rollout_id,
            rollout_failed=metadata.failed,
            inference_environment_steps=metadata.inference_env_steps,
            task_name=metadata.task_name,
        )
        with np.load(dataset_dir / metadata.tensor_path, allow_pickle=False) as payload:
            raw = np.asarray(payload["features"], dtype=np.float32)
            features = aggregate_features(raw, aggregation=aggregation)
            context = _load_context(payload, context_key, len(features))
        if context_key is not None and context is None:
            missing_context.append(metadata.rollout_id)
        spans = _stage_spans(trace, metadata.task_name, len(features))
        if not spans:
            raise ValueError(f"Parent {metadata.rollout_id} has no usable stage spans")
        failures = [span for span in spans if span["failed"]]
        if metadata.failed and len(failures) != 1:
            raise ValueError(
                f"Failed parent {metadata.rollout_id} must have one failed stage"
            )
        if not metadata.failed and failures:
            raise ValueError(
                f"Successful parent {metadata.rollout_id} contains a failed stage"
            )
        assignment = [None] * len(features)
        for span_index, span in enumerate(spans):
            for inference_index in range(span["start"], span["end"]):
                if assignment[inference_index] is not None:
                    raise AssertionError("Semantic span assignment overlaps")
                assignment[inference_index] = span_index
        parents.append(
            {
                "rollout_id": metadata.rollout_id,
                "task_name": metadata.task_name,
                "failed": bool(metadata.failed),
                "environment_seed": int(metadata.environment_seed),
                "environment_reset_index": metadata.environment_reset_index,
                "inference_environment_steps": list(metadata.inference_env_steps),
                "features": features.astype(np.float32, copy=False),
                "context": context,
                "spans": spans,
                "stage_assignment": assignment,
                "tensor_path": str((dataset_dir / metadata.tensor_path).resolve()),
                "trace_path": str(trace_path.resolve()),
            }
        )
    if not parents:
        raise ValueError("No raw parent sequences match the requested selection")
    if require_context and missing_context:
        raise ValueError(
            f"Context key {context_key!r} is missing for {len(missing_context)} "
            f"parents, including {missing_context[:3]}"
        )
    return {
        "parents": parents,
        "dataset_dir": str(dataset_dir),
        "aggregation": aggregation,
        "context_key": context_key,
        "context_available_for_all": not missing_context,
        "missing_context_parent_ids": missing_context,
    }


def parent_identity(parent):
    return (
        str(parent["task_name"]),
        int(parent["environment_seed"]),
        parent.get("environment_reset_index"),
    )


def allocate_development_parents(
    parents,
    *,
    num_folds=5,
    selection_fold=1,
    calibration_fold=0,
    seed=0,
):
    """Create fit/selection/calibration folds inside development parents only."""
    num_folds = int(num_folds)
    selection_fold = int(selection_fold)
    calibration_fold = int(calibration_fold)
    if num_folds < 3:
        raise ValueError("At least three development folds are required")
    if not 0 <= selection_fold < num_folds or not 0 <= calibration_fold < num_folds:
        raise ValueError("Selection/calibration fold index is out of range")
    if selection_fold == calibration_fold:
        raise ValueError("Selection and calibration folds must differ")
    strata = defaultdict(list)
    for parent in parents:
        strata[(parent["task_name"], bool(parent["failed"]))].append(
            parent["rollout_id"]
        )
    folds = [set() for _ in range(num_folds)]
    support = {}
    for stratum, ids in sorted(strata.items()):
        if len(ids) < num_folds:
            raise ValueError(
                f"Development stratum {stratum} has {len(ids)} parents, fewer "
                f"than num_folds={num_folds}"
            )
        values = sorted(ids)
        rng = random.Random(f"{int(seed)}:{stratum[0]}:{int(stratum[1])}")
        rng.shuffle(values)
        for index, rollout_id in enumerate(values):
            folds[index % num_folds].add(rollout_id)
        support[f"{stratum[0]}::{int(stratum[1])}"] = {
            "parents": len(values),
            "per_fold": [sum(value in fold for value in values) for fold in folds],
        }
    selection = set(folds[selection_fold])
    calibration = set(folds[calibration_fold])
    all_ids = {parent["rollout_id"] for parent in parents}
    fit = all_ids - selection - calibration
    if not fit or not selection or not calibration:
        raise ValueError("Development allocation produced an empty partition")
    if fit & selection or fit & calibration or selection & calibration:
        raise AssertionError("Development parent allocation overlaps")
    return {
        "fit": fit,
        "selection": selection,
        "calibration": calibration,
        "folds": [sorted(fold) for fold in folds],
        "counts": {
            "fit": len(fit),
            "selection": len(selection),
            "calibration": len(calibration),
        },
        "strata": support,
        "num_folds": num_folds,
        "selection_fold": selection_fold,
        "calibration_fold": calibration_fold,
        "seed": int(seed),
    }


def make_parent_holdout_split(parents, *, train_fraction=0.68, seed=0):
    """Freeze a task/outcome-stratified development versus opened holdout split.

    This lightweight split is for raw full-parent datasets.  Unlike the
    segmented SAFE exporter it does not duplicate any feature tensors.
    """
    train_fraction = float(train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")
    strata = defaultdict(list)
    for item in parents:
        strata[(item["task_name"], bool(item["failed"]))].append(
            item["rollout_id"]
        )
    too_small = {
        f"{task}::{int(failed)}": len(values)
        for (task, failed), values in strata.items()
        if len(values) < 2
    }
    if too_small:
        raise ValueError(
            "Parent task/outcome strata are too small for a holdout: "
            + repr(too_small)
        )
    development, holdout = set(), set()
    support = {}
    for (task, failed), values in sorted(strata.items()):
        values = sorted(values)
        rng = random.Random(f"{int(seed)}:{task}:{int(failed)}")
        rng.shuffle(values)
        count = max(1, min(len(values) - 1, int(len(values) * train_fraction)))
        development.update(values[:count])
        holdout.update(values[count:])
        support[f"{task}::{int(failed)}"] = {
            "source": len(values),
            "development": count,
            "holdout": len(values) - count,
        }
    all_ids = {item["rollout_id"] for item in parents}
    if development & holdout or development | holdout != all_ids:
        raise AssertionError("Parent holdout split overlaps or loses parents")
    return {
        "schema_version": 1,
        "protocol": "full_parent_task_outcome_stratified_holdout",
        "split_unit": "parent_rollout",
        "stratified_by": ["task_name", "rollout_failed"],
        "split_seed": int(seed),
        "train_fraction": train_fraction,
        "parent_train": sorted(development),
        "parent_test": sorted(holdout),
        "counts": {
            "parent_train": len(development),
            "parent_test": len(holdout),
            "total": len(all_ids),
        },
        "per_parent_stratum": support,
        "opened_outer_scored": False,
    }


def select_parents(parents, ids):
    ids = {str(value) for value in ids}
    return [parent for parent in parents if parent["rollout_id"] in ids]


def stage_support(parents):
    support = defaultdict(
        lambda: {"successes": 0, "failures": 0, "parents": set(), "samples": 0}
    )
    for parent in parents:
        for span in parent["spans"]:
            row = support[span["stage_name"]]
            row["failures" if span["failed"] else "successes"] += 1
            row["parents"].add(parent["rollout_id"])
            row["samples"] += span["length"]
    return {
        name: {
            "successes": value["successes"],
            "failures": value["failures"],
            "parents": len(value["parents"]),
            "samples": value["samples"],
        }
        for name, value in sorted(support.items())
    }


def select_supported_stages(
    fit_parents,
    *,
    min_successes=3,
    min_failures=2,
    requested_stages=None,
):
    support = stage_support(fit_parents)
    requested = None if requested_stages is None else set(requested_stages)
    if requested is not None:
        unknown = sorted(requested - set(support))
        if unknown:
            raise ValueError("Requested stages are absent from fit: " + ", ".join(unknown))
    selected, excluded = [], {}
    for name, counts in support.items():
        reasons = []
        if requested is not None and name not in requested:
            reasons.append("not_requested")
        if counts["successes"] < int(min_successes):
            reasons.append("insufficient_successes")
        if counts["failures"] < int(min_failures):
            reasons.append("insufficient_failures")
        if reasons:
            excluded[name] = reasons
        else:
            selected.append(name)
    if not selected:
        raise ValueError("No stage satisfies the fit-only support thresholds")
    return {
        "selected_stages": selected,
        "stage_catalog": {name: index for index, name in enumerate(selected)},
        "support": support,
        "excluded": excluded,
        "min_successes": int(min_successes),
        "min_failures": int(min_failures),
        "source": "fit parents only",
    }


def training_horizons(parents, selected_stages):
    """Fit stage and task time scales without using held-out durations."""
    selected = set(selected_stages)
    successful = defaultdict(list)
    task_lengths = defaultdict(list)
    for parent in parents:
        task_lengths[parent["task_name"]].append(len(parent["features"]))
        for span in parent["spans"]:
            if span["stage_name"] in selected and not span["failed"]:
                successful[span["stage_name"]].append(span["length"])
    missing = sorted(selected - set(successful))
    if missing:
        raise ValueError("Training has no successful horizon for: " + ", ".join(missing))
    return {
        "stage": {
            name: max(1, int(math.ceil(float(np.median(successful[name])))))
            for name in sorted(selected)
        },
        "task": {
            name: max(1, int(math.ceil(float(np.median(lengths)))))
            for name, lengths in sorted(task_lengths.items())
        },
    }


def _global_causal_features(values, window):
    values = np.asarray(values, dtype=np.float32)
    base = summarize_trajectory(values, window=window)
    cumulative = np.cumsum(values, axis=0, dtype=np.float64)
    denominator = np.arange(1, len(values) + 1, dtype=np.float64)[:, None]
    cumulative_mean = (cumulative / denominator).astype(np.float32)
    return np.concatenate((base, cumulative_mean), axis=1).astype(np.float32)


def build_inference_rows(
    parents,
    *,
    selected_stages,
    horizons,
    failure_horizons=DEFAULT_FAILURE_HORIZONS,
    temporal_window=4,
    include_unlabeled_terminal=True,
):
    """Build causal per-inference rows while preserving full-parent summaries."""
    selected = set(selected_stages)
    failure_horizons = tuple(sorted({int(value) for value in failure_horizons}))
    if not failure_horizons or failure_horizons[0] <= 0:
        raise ValueError("Failure horizons must be positive")
    rows = []
    for parent in parents:
        common = _global_causal_features(parent["features"], temporal_window)
        context_summary = (
            None
            if parent.get("context") is None
            else _global_causal_features(parent["context"], temporal_window)
        )
        total_stages = max(1, len(parent["spans"]))
        for inference_index in range(len(parent["features"])):
            span_index = parent["stage_assignment"][inference_index]
            span = None if span_index is None else parent["spans"][span_index]
            if span is None:
                if not include_unlabeled_terminal:
                    continue
                rows.append(
                    {
                        "parent_rollout_id": parent["rollout_id"],
                        "task_name": parent["task_name"],
                        "stage_name": None,
                        "segment_id": None,
                        "inference_index": inference_index,
                        "local_index": None,
                        "terminal_failed": bool(parent["failed"]),
                        "stage_failed": None,
                        "within_horizon": {},
                        "common_features": common[inference_index],
                        "stage_anchor_delta": np.zeros_like(parent["features"][0]),
                        "context_features": (
                            None if context_summary is None else context_summary[inference_index]
                        ),
                        "conditioning": np.zeros(4, dtype=np.float32),
                    }
                )
                continue
            if span["stage_name"] not in selected:
                # Keep unsupported stages for the terminal arm only.
                stage_target = None
            else:
                stage_target = bool(span["failed"])
            local_step = inference_index - span["start"] + 1
            remaining = span["end"] - inference_index
            stage_horizon = int(horizons["stage"].get(span["stage_name"], span["length"]))
            task_horizon = int(horizons["task"][parent["task_name"]])
            conditioning = np.asarray(
                [
                    min(1.0, local_step / max(1, stage_horizon)),
                    span["stage_index_in_task"] / max(1, total_stages - 1),
                    min(1.0, (inference_index + 1) / max(1, task_horizon)),
                    span["stage_index_in_task"] / total_stages,
                ],
                dtype=np.float32,
            )
            rows.append(
                {
                    "parent_rollout_id": parent["rollout_id"],
                    "task_name": parent["task_name"],
                    "stage_name": span["stage_name"],
                    "segment_id": span["segment_id"],
                    "inference_index": inference_index,
                    "local_index": local_step,
                    "stage_length": span["length"],
                    "terminal_failed": bool(parent["failed"]),
                    "stage_failed": stage_target,
                    "within_horizon": {
                        str(horizon): bool(span["failed"] and remaining <= horizon)
                        if stage_target is not None
                        else None
                        for horizon in failure_horizons
                    },
                    "common_features": common[inference_index],
                    "stage_anchor_delta": (
                        parent["features"][inference_index]
                        - parent["features"][span["start"]]
                    ).astype(np.float32),
                    "context_features": (
                        None if context_summary is None else context_summary[inference_index]
                    ),
                    "conditioning": conditioning,
                }
            )
    if not rows:
        raise ValueError("No per-inference rows were constructed")
    return rows


def model_rows(rows, arm):
    if arm not in MODEL_ARMS:
        raise ValueError(f"Unknown model arm {arm!r}")
    if arm == "terminal":
        return list(rows)
    selected = [row for row in rows if row.get("stage_failed") is not None]
    if arm == "context":
        selected = [row for row in selected if row.get("context_features") is not None]
    if not selected:
        raise ValueError(f"No usable rows remain for arm {arm}")
    return selected


def row_feature(row, arm):
    values = [np.asarray(row["common_features"], dtype=np.float32)]
    if arm in ("conditioned", "context"):
        values.extend(
            (
                np.asarray(row["stage_anchor_delta"], dtype=np.float32),
                np.asarray(row["conditioning"], dtype=np.float32),
            )
        )
    if arm == "context":
        context = row.get("context_features")
        if context is None:
            raise ValueError("Context arm received a row without context features")
        values.append(np.asarray(context, dtype=np.float32))
    return np.concatenate(values).astype(np.float32, copy=False)


def fit_scaler(rows, arm):
    matrix = np.stack([row_feature(row, arm) for row in rows]).astype(np.float64)
    weights = parent_stage_weights(rows, target="terminal" if arm == "terminal" else "stage")
    weights = weights / weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum((matrix - mean) ** 2 * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def fit_success_scaler(rows, arm="stage"):
    """Fit a stage/parent-balanced scaler using successful stages only."""
    selected = [row for row in rows if row.get("stage_failed") is False]
    if not selected:
        raise ValueError("Success-only scaler has no successful stage rows")
    grouped = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(selected):
        grouped[row["stage_name"]][row["parent_rollout_id"]].append(index)
    weights = np.zeros(len(selected), dtype=np.float64)
    for parents in grouped.values():
        for indices in parents.values():
            weights[indices] = 1.0 / (
                len(grouped) * len(parents) * len(indices)
            )
    weights /= weights.sum()
    matrix = np.stack([row_feature(row, arm) for row in selected]).astype(np.float64)
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum((matrix - mean) ** 2 * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def transform_row_matrix(rows, arm, scaler):
    matrix = np.stack([row_feature(row, arm) for row in rows]).astype(np.float32)
    return (matrix - scaler["mean"]) / scaler["scale"]


def row_target(row, target):
    if target == "terminal":
        return int(bool(row["terminal_failed"]))
    if target == "stage":
        value = row.get("stage_failed")
    elif target.startswith("horizon_"):
        value = row.get("within_horizon", {}).get(target.split("_", 1)[1])
    else:
        raise ValueError(f"Unknown target {target!r}")
    if value is None:
        raise ValueError(f"Row has no label for target {target}")
    return int(bool(value))


def parent_stage_weights(rows, *, target):
    """Equalize task/stage-outcome mass, parents, then inference samples."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, row in enumerate(rows):
        label = row_target(row, target)
        group = (
            row["task_name"]
            if target == "terminal"
            else row["stage_name"]
        )
        grouped[group][label][row["parent_rollout_id"]].append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    valid_groups = 0
    for group, outcomes in grouped.items():
        if set(outcomes) != {0, 1}:
            # A head is never taught from deterministic strata.
            continue
        valid_groups += 1
        for label in (0, 1):
            parents = outcomes[label]
            for indices in parents.values():
                per_row = 1.0 / (2.0 * len(parents) * len(indices))
                weights[indices] = per_row
    if valid_groups == 0 or not np.any(weights):
        raise ValueError(f"Target {target} has no supported two-outcome groups")
    weights /= valid_groups
    weights *= len(rows) / weights.sum()
    return weights.astype(np.float32)


def task_stage_indices(rows, task_catalog, stage_catalog):
    task = np.asarray([task_catalog[row["task_name"]] for row in rows], dtype=np.int64)
    stage = np.asarray(
        [stage_catalog.get(row.get("stage_name"), 0) for row in rows],
        dtype=np.int64,
    )
    return task, stage


def binary_auc(labels, scores):
    """Dependency-free ROC-AUC with average ranks for ties."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    negatives = labels == 0
    if not np.any(positives) or not np.any(negatives):
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fixed_prefix_metrics(rows, scores, *, prefixes=DEFAULT_PREFIXES):
    if len(rows) != len(scores):
        raise ValueError("Rows and scores are not aligned")
    prefixes = tuple(int(value) for value in prefixes)
    output = {}
    for prefix in prefixes:
        grouped = defaultdict(list)
        for row, score in zip(rows, scores):
            if row.get("stage_failed") is None or int(row["local_index"]) != prefix:
                continue
            grouped[(row["task_name"], row["stage_name"])].append(
                (int(row["stage_failed"]), float(score))
            )
        per_stage = {}
        per_task = defaultdict(list)
        support = 0
        for (task, stage), values in sorted(grouped.items()):
            auc = binary_auc([value[0] for value in values], [value[1] for value in values])
            if auc is None:
                continue
            per_stage[stage] = {"roc_auc": auc, "segments": len(values)}
            per_task[task].append(auc)
            support += len(values)
        task_values = {
            task: float(np.mean(values)) for task, values in sorted(per_task.items())
        }
        output[str(prefix)] = {
            "task_stage_macro_roc_auc": (
                float(np.mean(list(task_values.values()))) if task_values else None
            ),
            "tasks": len(task_values),
            "stages": len(per_stage),
            "segments": support,
            "per_task": task_values,
            "per_stage": per_stage,
            "at_risk_definition": "stage remains active at the exact prefix",
        }
    return output


def primary_prefix_score(metrics, primary_prefixes=DEFAULT_PRIMARY_PREFIXES):
    values = [
        metrics[str(int(prefix))]["task_stage_macro_roc_auc"]
        for prefix in primary_prefixes
        if metrics.get(str(int(prefix)), {}).get("task_stage_macro_roc_auc") is not None
    ]
    if not values:
        raise ValueError("No estimable primary fixed-prefix ROC-AUC")
    return float(np.mean(values))


def fit_stage_time_curves(rows, selected_stages, *, prior=0.5):
    curves = {}
    for stage in selected_stages:
        segments = {}
        for row in rows:
            if row.get("stage_name") == stage and row.get("stage_failed") is not None:
                segments.setdefault(
                    row["segment_id"],
                    {
                        "failed": bool(row["stage_failed"]),
                        "length": int(row["stage_length"]),
                    },
                )
        horizon = max(value["length"] for value in segments.values())
        risk = []
        for step in range(1, horizon + 1):
            active = [value for value in segments.values() if value["length"] >= step]
            failures = sum(value["failed"] for value in active)
            risk.append((failures + prior) / (len(active) + 2 * prior))
        curves[stage] = np.maximum.accumulate(np.asarray(risk, dtype=np.float64))
    return curves


def stage_time_scores(rows, curves):
    output = []
    for row in rows:
        stage = row.get("stage_name")
        if stage not in curves or row.get("local_index") is None:
            output.append(float("nan"))
            continue
        curve = curves[stage]
        index = min(len(curve), int(row["local_index"])) - 1
        output.append(float(curve[index]))
    return np.asarray(output, dtype=np.float64)


def fit_success_prototypes(rows, scaler, horizons, *, bins=4):
    rows = [row for row in rows if row.get("stage_failed") is False]
    matrix = transform_row_matrix(rows, "stage", scaler)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        fraction = min(
            1.0,
            int(row["local_index"]) / max(1, horizons["stage"][row["stage_name"]]),
        )
        bin_index = min(int(bins) - 1, int(fraction * int(bins)))
        grouped[(row["stage_name"], bin_index)].append(matrix[index])
    stage_grouped = defaultdict(list)
    for row, value in zip(rows, matrix):
        stage_grouped[row["stage_name"]].append(value)
    return {
        "bins": int(bins),
        "prototypes": {
            f"{stage}@@{bin_index}": np.mean(values, axis=0).astype(np.float32)
            for (stage, bin_index), values in sorted(grouped.items())
        },
        "stage_fallback": {
            stage: np.mean(values, axis=0).astype(np.float32)
            for stage, values in sorted(stage_grouped.items())
        },
    }


def prototype_scores(rows, scaler, horizons, runtime):
    matrix = transform_row_matrix(rows, "stage", scaler)
    scores = []
    bins = int(runtime["bins"])
    for row, value in zip(rows, matrix):
        fraction = min(
            1.0,
            int(row["local_index"]) / max(1, horizons["stage"][row["stage_name"]]),
        )
        bin_index = min(bins - 1, int(fraction * bins))
        key = f"{row['stage_name']}@@{bin_index}"
        prototype = runtime["prototypes"].get(
            key, runtime["stage_fallback"][row["stage_name"]]
        )
        scores.append(float(np.mean((value - prototype) ** 2)))
    return np.asarray(scores, dtype=np.float64)


def group_score_trajectories(rows, score_by_detector):
    grouped = {}
    for index, row in enumerate(rows):
        if row.get("stage_failed") is None:
            continue
        key = row["segment_id"]
        event = grouped.setdefault(
            key,
            {
                "parent_rollout_id": row["parent_rollout_id"],
                "task_name": row["task_name"],
                "stage_name": row["stage_name"],
                "segment_id": key,
                "failed": bool(row["stage_failed"]),
                "scores": defaultdict(list),
            },
        )
        for detector, scores in score_by_detector.items():
            value = float(scores[index])
            if not math.isfinite(value):
                raise ValueError(f"Detector {detector} produced a non-finite score")
            event["scores"][detector].append(value)
    return [
        {**event, "scores": {key: np.asarray(value) for key, value in event["scores"].items()}}
        for event in grouped.values()
    ]


def fit_score_normalizer(events, detector, *, shrinkage=5.0):
    global_values = [
        float(np.max(event["scores"][detector]))
        for event in events
        if not event["failed"]
    ]
    if len(global_values) < 2:
        raise ValueError(f"Detector {detector} has too few successful fit events")
    global_mean = float(np.mean(global_values))
    global_std = max(float(np.std(global_values)), 1e-6)
    by_stage = defaultdict(list)
    for event in events:
        if not event["failed"]:
            by_stage[event["stage_name"]].append(
                float(np.max(event["scores"][detector]))
            )
    stages = {}
    for stage, values in sorted(by_stage.items()):
        weight = len(values) / (len(values) + float(shrinkage))
        mean = weight * float(np.mean(values)) + (1 - weight) * global_mean
        std = weight * max(float(np.std(values)), 1e-6) + (1 - weight) * global_std
        stages[stage] = {"mean": mean, "scale": max(std, 1e-6), "support": len(values)}
    return {
        "global": {"mean": global_mean, "scale": global_std, "support": len(global_values)},
        "stages": stages,
        "shrinkage": float(shrinkage),
        "fit_on": "successful training-stage event maxima only",
    }


def apply_score_normalizer(events, detector, normalizer):
    output = []
    for event in events:
        item = copy.deepcopy(event)
        parameters = normalizer["stages"].get(
            event["stage_name"], normalizer["global"]
        )
        item["scores"][detector] = (
            np.asarray(event["scores"][detector], dtype=np.float64)
            - float(parameters["mean"])
        ) / float(parameters["scale"])
        output.append(item)
    return output


def conformal_success_threshold(
    events, detector, *, target_fpr=0.05, unit="parent"
):
    """Calibrate a strict-crossing threshold on successful calibration units.

    The deployed intervention unit is a parent rollout: one false alarm in any
    completed stage makes that successful parent a false positive.  Parent is
    therefore the default conformal unit.  Event-level calibration remains
    available for diagnostics only.
    """
    if unit == "event":
        values = [
            float(np.max(event["scores"][detector]))
            for event in events
            if not event["failed"]
        ]
    elif unit == "parent":
        grouped = defaultdict(list)
        parent_failed = defaultdict(bool)
        for event in events:
            parent_id = event["parent_rollout_id"]
            grouped[parent_id].append(float(np.max(event["scores"][detector])))
            parent_failed[parent_id] = parent_failed[parent_id] or bool(event["failed"])
        values = [
            max(parent_values)
            for parent_id, parent_values in grouped.items()
            if not parent_failed[parent_id]
        ]
    else:
        raise ValueError("Conformal threshold unit must be 'parent' or 'event'")
    values = sorted(values)
    if not values:
        raise ValueError("Threshold calibration has no successful stage events")
    rank = int(math.ceil((len(values) + 1) * (1.0 - float(target_fpr))))
    if rank > len(values):
        maximum = float(values[-1])
        threshold = maximum + max(1.0, abs(maximum)) * 1e-6
    else:
        threshold = float(values[rank - 1])
    empirical = float(np.mean(np.asarray(values) > threshold))
    return {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "calibration_success_units": len(values),
        "calibration_unit": unit,
        "empirical_success_unit_fpr_strict_crossing": empirical,
        "crossing": "score > threshold",
        "finite_sample_rank": rank,
    }


def stage_event_metrics(events, detector, threshold, horizons):
    predictions = []
    for event in events:
        scores = np.asarray(event["scores"][detector], dtype=np.float64)
        crossings = np.flatnonzero(scores > float(threshold))
        alarm = int(crossings[0]) if len(crossings) else None
        horizon = int(horizons["stage"][event["stage_name"]])
        fraction = None if alarm is None else min(1.0, (alarm + 1) / max(1, horizon))
        predictions.append(
            {
                **{key: event[key] for key in (
                    "parent_rollout_id", "task_name", "stage_name", "segment_id", "failed"
                )},
                "detector": detector,
                "alarm_inference_index": alarm,
                "alarm_fraction": fraction,
                "detected": alarm is not None,
                "adjusted_detection_fraction": (
                    1.0 if event["failed"] and alarm is None else fraction
                    if event["failed"]
                    else None
                ),
                "max_score": float(np.max(scores)),
            }
        )

    def aggregate(selected):
        labels = np.asarray([int(row["failed"]) for row in selected], dtype=np.int64)
        predicted = np.asarray([int(row["detected"]) for row in selected], dtype=np.int64)
        tp = int(np.sum((labels == 1) & (predicted == 1)))
        fn = int(np.sum((labels == 1) & (predicted == 0)))
        fp = int(np.sum((labels == 0) & (predicted == 1)))
        tn = int(np.sum((labels == 0) & (predicted == 0)))
        adjusted = [
            row["adjusted_detection_fraction"] for row in selected if row["failed"]
        ]
        recall = {}
        for landmark in (0.10, 0.25, 0.50):
            failures = [row for row in selected if row["failed"]]
            recall[str(landmark)] = (
                float(np.mean([
                    row["alarm_fraction"] is not None
                    and row["alarm_fraction"] <= landmark
                    for row in failures
                ]))
                if failures
                else None
            )
        return {
            "events": len(selected),
            "successes": int(np.sum(labels == 0)),
            "failures": int(np.sum(labels == 1)),
            "true_positive_rate": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "balanced_accuracy": (
                0.5 * (tp / (tp + fn) + tn / (tn + fp))
                if tp + fn and tn + fp
                else None
            ),
            "missed_failure_adjusted_detection_fraction": (
                float(np.mean(adjusted)) if adjusted else None
            ),
            "failure_recall_by_landmark": recall,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        }

    per_stage = {
        stage: aggregate([row for row in predictions if row["stage_name"] == stage])
        for stage in sorted({row["stage_name"] for row in predictions})
    }
    per_task = {
        task: aggregate([row for row in predictions if row["task_name"] == task])
        for task in sorted({row["task_name"] for row in predictions})
    }
    macro_fields = (
        "true_positive_rate",
        "false_positive_rate",
        "balanced_accuracy",
        "missed_failure_adjusted_detection_fraction",
    )
    macro = {}
    for field in macro_fields:
        values = [value[field] for value in per_task.values() if value[field] is not None]
        macro[field] = float(np.mean(values)) if values else None
    return {
        "detector": detector,
        "threshold": float(threshold),
        "pooled": aggregate(predictions),
        "task_macro": macro,
        "per_task": per_task,
        "per_stage": per_stage,
        "predictions": predictions,
    }


def parent_event_metrics(stage_predictions, parent_outcomes=None):
    """Aggregate stage alarms without relabeling unsupported failed parents.

    A failed parent is estimable only when its failed terminal stage belongs to
    the fit-supported stage catalog and therefore has a prediction.  Completed
    stages from a failed parent must never make that parent look successful.
    """
    grouped = defaultdict(list)
    for row in stage_predictions:
        grouped[row["parent_rollout_id"]].append(row)
    metadata = {}
    if parent_outcomes is not None:
        for parent_id, value in parent_outcomes.items():
            if isinstance(value, dict):
                metadata[str(parent_id)] = {
                    "failed": bool(value["failed"]),
                    "task_name": str(value["task_name"]),
                }
            else:
                metadata[str(parent_id)] = {
                    "failed": bool(value),
                    "task_name": (
                        grouped[str(parent_id)][0]["task_name"]
                        if grouped.get(str(parent_id))
                        else "unknown"
                    ),
                }
    else:
        for parent_id, values in grouped.items():
            metadata[parent_id] = {
                "failed": any(row["failed"] for row in values),
                "task_name": values[0]["task_name"],
            }
    rows = []
    for parent_id, parent in sorted(metadata.items()):
        values = grouped.get(parent_id, [])
        failed_stages = [row for row in values if row["failed"]]
        parent_failed = bool(parent["failed"])
        evaluable = bool(failed_stages) if parent_failed else bool(values)
        completed_false_alarms = sum(
            row["detected"] for row in values if not row["failed"]
        )
        detected = bool(evaluable and failed_stages and failed_stages[0]["detected"])
        rows.append(
            {
                "parent_rollout_id": parent_id,
                "task_name": parent["task_name"],
                "failed": parent_failed,
                "evaluable": evaluable,
                "exclusion_reason": (
                    None
                    if evaluable
                    else "failed_stage_not_fit_supported"
                    if parent_failed
                    else "no_fit_supported_completed_stage"
                ),
                "detected_failed_stage": detected,
                "completed_stage_false_alarms": int(completed_false_alarms),
                "any_alarm": bool(evaluable and (detected or completed_false_alarms)),
                "adjusted_detection_fraction": (
                    failed_stages[0]["adjusted_detection_fraction"]
                    if evaluable and failed_stages
                    else None
                ),
            }
        )
    evaluable_rows = [row for row in rows if row["evaluable"]]
    labels = np.asarray([int(row["failed"]) for row in evaluable_rows])
    positive = np.asarray(
        [int(row["detected_failed_stage"]) for row in evaluable_rows]
    )
    false_alarm = np.asarray(
        [int(row["any_alarm"]) if not row["failed"] else 0 for row in evaluable_rows]
    )
    return {
        "parents": len(evaluable_rows),
        "source_parents": len(rows),
        "excluded_parents": len(rows) - len(evaluable_rows),
        "excluded_failed_stage_not_fit_supported": sum(
            row["exclusion_reason"] == "failed_stage_not_fit_supported"
            for row in rows
        ),
        "excluded_success_no_fit_supported_stage": sum(
            row["exclusion_reason"] == "no_fit_supported_completed_stage"
            for row in rows
        ),
        "failures": int(labels.sum()),
        "successes": int(np.sum(labels == 0)),
        "failed_stage_tpr": (
            float(np.mean(positive[labels == 1])) if np.any(labels == 1) else None
        ),
        "successful_parent_fpr": (
            float(np.mean(false_alarm[labels == 0])) if np.any(labels == 0) else None
        ),
        "parents_with_completed_stage_false_alarm": sum(
            row["completed_stage_false_alarms"] > 0 for row in evaluable_rows
        ),
        "missed_failure_adjusted_detection_fraction": (
            float(np.mean([
                row["adjusted_detection_fraction"]
                for row in evaluable_rows
                if row["failed"]
            ]))
            if np.any(labels == 1)
            else None
        ),
        "predictions": rows,
    }


def paired_parent_bootstrap(
    predictions_by_detector,
    candidate,
    *,
    baseline="time_only",
    replicates=2000,
    seed=0,
):
    """Task/parent bootstrap of miss-adjusted detection-time differences."""
    by_detector = {
        detector: {
            row["parent_rollout_id"]: row
            for row in payload
            if row["failed"] and row.get("evaluable", True)
        }
        for detector, payload in predictions_by_detector.items()
    }
    common = set(by_detector[candidate]) & set(by_detector[baseline])
    if not common:
        raise ValueError("Paired bootstrap has no common failed parents")
    by_task = defaultdict(list)
    for parent_id in common:
        by_task[by_detector[candidate][parent_id]["task_name"]].append(parent_id)
    tasks = sorted(by_task)
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(replicates)):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_deltas = []
        for task in sampled_tasks:
            parents = by_task[str(task)]
            sampled = rng.choice(parents, size=len(parents), replace=True)
            task_deltas.append(
                np.mean([
                    by_detector[candidate][str(parent)]["adjusted_detection_fraction"]
                    - by_detector[baseline][str(parent)]["adjusted_detection_fraction"]
                    for parent in sampled
                ])
            )
        values.append(float(np.mean(task_deltas)))
    point = float(np.mean([
        by_detector[candidate][parent]["adjusted_detection_fraction"]
        - by_detector[baseline][parent]["adjusted_detection_fraction"]
        for parent in sorted(common)
    ]))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "definition": "candidate minus baseline; negative is earlier",
        "point": point,
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "replicates": int(replicates),
        "tasks": len(tasks),
        "failed_parents": len(common),
    }


__all__ = [
    "ARMS",
    "DEFAULT_FAILURE_HORIZONS",
    "DEFAULT_PREFIXES",
    "DEFAULT_PRIMARY_PREFIXES",
    "MODEL_ARMS",
    "allocate_development_parents",
    "apply_score_normalizer",
    "binary_auc",
    "build_inference_rows",
    "conformal_success_threshold",
    "fit_scaler",
    "fit_score_normalizer",
    "fit_success_scaler",
    "fit_stage_time_curves",
    "fit_success_prototypes",
    "fixed_prefix_metrics",
    "group_score_trajectories",
    "json_value",
    "load_parent_sequences",
    "load_parent_split",
    "make_parent_holdout_split",
    "model_rows",
    "paired_parent_bootstrap",
    "parent_event_metrics",
    "parent_identity",
    "parent_stage_weights",
    "primary_prefix_score",
    "prototype_scores",
    "row_target",
    "select_parents",
    "select_supported_stages",
    "stage_event_metrics",
    "stage_time_scores",
    "task_stage_indices",
    "training_horizons",
    "transform_row_matrix",
    "write_json",
]
