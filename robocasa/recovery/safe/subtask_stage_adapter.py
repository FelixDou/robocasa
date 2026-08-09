"""Parent-grouped utilities for stage-adapter Subtask-SAFE diagnostics.

This module deliberately keeps the locked outer test out of model selection.
It builds a three-way split only from outer-training parents, estimates stage
horizons and elapsed-time risk from meta-fit parents, and evaluates causal
landmarks using one record per genuine policy inference.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import math
import random

import numpy as np

from .causal_subtask_safe import stage_name


DEFAULT_LANDMARKS = (0.10, 0.25, 0.50)
DEFAULT_PRIMARY_LANDMARKS = (0.25, 0.50)
ARCHITECTURES = (
    "shared",
    "shared_one_hot",
    "stage_adapter",
    "stage_specific",
)
OBJECTIVES = ("bce", "temporal_contrastive")
DETECTORS = ("stage_safe", "stage_safe_time")


def validate_landmarks(values):
    landmarks = tuple(float(value) for value in values)
    if not landmarks or any(not 0.0 < value <= 1.0 for value in landmarks):
        raise ValueError("Causal stage landmarks must lie in (0, 1]")
    if len(set(landmarks)) != len(landmarks):
        raise ValueError("Causal stage landmarks must be unique")
    return tuple(sorted(landmarks))


def fit_time_risk(rollouts, horizons, *, prior=0.5):
    """Fit monotone P(stage failure | still active, stage) curves."""
    from sklearn.isotonic import IsotonicRegression

    curves = {}
    for task_id, horizon in sorted(horizons.items()):
        selected = [item for item in rollouts if int(item.task_id) == task_id]
        raw = []
        active_counts = []
        progress = []
        for step in range(1, int(horizon) + 1):
            active = [item for item in selected if len(item.hidden_states) >= step]
            failures = sum(not bool(int(item.episode_success)) for item in active)
            risk = (failures + float(prior)) / (len(active) + 2.0 * float(prior))
            raw.append(float(risk))
            active_counts.append(len(active))
            progress.append(step / int(horizon))
        weights = np.maximum(1, np.asarray(active_counts, dtype=np.int64))
        fitted = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        ).fit_transform(progress, raw, sample_weight=weights)
        curves[int(task_id)] = {
            "risk": np.asarray(fitted, dtype=np.float64),
            "raw_risk": np.asarray(raw, dtype=np.float64),
            "active_counts": np.asarray(active_counts, dtype=np.int64),
        }
    return curves


def time_risk_at(curves, task_id, step):
    risk = curves[int(task_id)]["risk"]
    index = min(max(1, int(step)), len(risk)) - 1
    return float(risk[index])


def summarize_prefix(sequence, step, *, window=4):
    """Return current, delta, recent mean, and recent slope feature blocks."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError(
            f"Expected a nonempty [time, feature] array, got {values.shape}"
        )
    step = int(step)
    if step < 1 or step > len(values):
        raise ValueError(
            f"Prefix step {step} is invalid for sequence length {len(values)}"
        )
    end = step - 1
    start = max(0, step - int(window))
    current = values[end]
    previous = values[max(0, end - 1)]
    recent = values[start:step]
    slope = current - values[start]
    return np.concatenate(
        [current, current - previous, recent.mean(axis=0), slope], axis=0
    ).astype(np.float32, copy=False)


def summarize_trajectory(sequence, *, horizon=None, window=4):
    """Vectorize causal stage summaries over every available inference."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError(
            f"Expected a nonempty [time, feature] array, got {values.shape}"
        )
    length = len(values) if horizon is None else min(len(values), int(horizon))
    values = values[:length]
    current = values
    previous = np.concatenate([values[:1], values[:-1]], axis=0)
    delta = current - previous
    cumulative = np.concatenate(
        [
            np.zeros((1, values.shape[1]), dtype=np.float64),
            np.cumsum(values, axis=0),
        ],
        axis=0,
    )
    means = []
    starts = []
    for end in range(1, length + 1):
        start = max(0, end - int(window))
        starts.append(start)
        means.append((cumulative[end] - cumulative[start]) / (end - start))
    means = np.asarray(means, dtype=np.float32)
    slopes = current - values[np.asarray(starts, dtype=np.int64)]
    return np.concatenate([current, delta, means, slopes], axis=1).astype(
        np.float32,
        copy=False,
    )


def fit_feature_scaler(rows):
    matrix = np.stack([row["features"] for row in rows]).astype(np.float64)
    weights = np.asarray(
        [float(row.get("sample_weight", 1.0)) for row in rows],
        dtype=np.float64,
    )
    weights /= weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum(((matrix - mean) ** 2) * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def transform_features(matrix, scaler):
    values = np.asarray(matrix, dtype=np.float32)
    return (values - scaler["mean"]) / scaler["scale"]


def logit(values, epsilon=1e-4):
    values = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(values / (1.0 - values))


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    if len(set(labels.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def safe_average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    if len(set(labels.tolist())) < 2:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def parent_id(rollout, identity):
    value = str(identity[id(rollout)][1].get("parent_rollout_id", ""))
    if not value:
        raise ValueError("Stage-adapter Subtask-SAFE requires parent_rollout_id")
    return value


def segment_id(rollout, identity):
    value = str(identity[id(rollout)][1].get("rollout_id", ""))
    if not value:
        raise ValueError("Stage-adapter Subtask-SAFE requires rollout_id")
    return value


def validate_stage_identity(rollouts, identity):
    """Return stable task-id/name mappings and reject ambiguous exports."""
    id_to_name = {}
    name_to_id = {}
    for rollout in rollouts:
        task_id = int(rollout.task_id)
        name = stage_name(identity[id(rollout)][1])
        previous_name = id_to_name.setdefault(task_id, name)
        previous_id = name_to_id.setdefault(name, task_id)
        if previous_name != name or previous_id != task_id:
            raise ValueError(
                "Subtask export has a non-bijective task-id/stage-name mapping"
            )
    if not id_to_name:
        raise ValueError("No semantic subtask segments were provided")
    return {
        "id_to_name": dict(sorted(id_to_name.items())),
        "name_to_id": dict(sorted(name_to_id.items())),
    }


def three_way_parent_split(
    rollouts,
    identity,
    *,
    num_folds=5,
    selection_fold=1,
    diagnostic_fold=0,
    seed=0,
):
    """Split outer-training parents into fit, selection, and diagnostic sets.

    Parents are stratified by parent task and parent rollout outcome.  The
    diagnostic parents are never used for optimization, early stopping,
    regularization selection, architecture selection, or threshold fitting.
    """
    num_folds = int(num_folds)
    selection_fold = int(selection_fold)
    diagnostic_fold = int(diagnostic_fold)
    if num_folds < 3:
        raise ValueError("Stage-adapter diagnostics require at least three folds")
    if not 0 <= selection_fold < num_folds:
        raise ValueError("selection_fold is outside the parent-fold range")
    if not 0 <= diagnostic_fold < num_folds:
        raise ValueError("diagnostic_fold is outside the parent-fold range")
    if selection_fold == diagnostic_fold:
        raise ValueError("Selection and diagnostic folds must differ")

    parents = {}
    for rollout in rollouts:
        env = identity[id(rollout)][1]
        key = parent_id(rollout, identity)
        stratum = (
            str(env.get("parent_task_name", "")),
            int(bool(env.get("parent_rollout_failed", False))),
        )
        if not stratum[0]:
            raise ValueError("Parent-grouped diagnostics require parent_task_name")
        current = parents.setdefault(
            key,
            {"stratum": stratum, "segments": []},
        )
        if current["stratum"] != stratum:
            raise ValueError(f"Parent metadata changed within {key}")
        current["segments"].append(rollout)
    if not parents:
        raise ValueError("No outer-training parents were provided")

    by_stratum = defaultdict(list)
    for key, payload in parents.items():
        by_stratum[payload["stratum"]].append(key)
    folds = [set() for _ in range(num_folds)]
    fold_order = [
        diagnostic_fold,
        selection_fold,
        *[
            index
            for index in range(num_folds)
            if index not in {diagnostic_fold, selection_fold}
        ],
    ]
    stratum_counts = {}
    for stratum, values in sorted(by_stratum.items()):
        values = sorted(values)
        if len(values) < 3:
            raise ValueError(
                f"Parent stratum {stratum} has {len(values)} parents, fewer than "
                "the three-way minimum=3"
            )
        rng = random.Random(f"{int(seed)}:{stratum[0]}:{stratum[1]}")
        rng.shuffle(values)
        for index, key in enumerate(values):
            folds[fold_order[index % num_folds]].add(key)
        stratum_counts[f"{stratum[0]}::{stratum[1]}"] = {
            "parents": len(values),
            "per_fold": [sum(key in fold for key in values) for fold in folds],
        }

    selected = set(folds[selection_fold])
    diagnostic = set(folds[diagnostic_fold])
    fit = set(parents) - selected - diagnostic
    if not fit or not selected or not diagnostic:
        raise ValueError("Three-way parent split produced an empty partition")
    if fit & selected or fit & diagnostic or selected & diagnostic:
        raise AssertionError("Three-way parent partitions overlap")
    if fit | selected | diagnostic != set(parents):
        raise AssertionError("Three-way parent split lost source parents")
    return {
        "fit": fit,
        "selection": selected,
        "diagnostic": diagnostic,
        "folds": [sorted(fold) for fold in folds],
        "counts": {
            "fit": len(fit),
            "selection": len(selected),
            "diagnostic": len(diagnostic),
        },
        "strata": stratum_counts,
        "num_folds": num_folds,
        "selection_fold": selection_fold,
        "diagnostic_fold": diagnostic_fold,
        "seed": int(seed),
    }


def select_parent_rollouts(rollouts, identity, parent_ids):
    selected = {str(value) for value in parent_ids}
    return [rollout for rollout in rollouts if parent_id(rollout, identity) in selected]


def select_supported_stages(
    fit_rollouts,
    identity,
    *,
    requested_stages=None,
    min_successes=10,
    min_failures=10,
):
    """Select stages using fit support only and create contiguous model indices."""
    mapping = validate_stage_identity(fit_rollouts, identity)
    support = defaultdict(lambda: {"successes": 0, "failures": 0, "parents": set()})
    for rollout in fit_rollouts:
        name = stage_name(identity[id(rollout)][1])
        key = "successes" if int(rollout.episode_success) else "failures"
        support[name][key] += 1
        support[name]["parents"].add(parent_id(rollout, identity))
    requested = (
        None if requested_stages is None else {str(value) for value in requested_stages}
    )
    if requested is not None:
        unknown = sorted(requested - set(support))
        if unknown:
            raise ValueError(
                "Requested stages are absent from meta-fit: " + ", ".join(unknown)
            )
    selected = []
    excluded = {}
    for name in sorted(support):
        counts = support[name]
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
        raise ValueError(
            "No stage meets fit-only support thresholds: "
            f"successes={min_successes}, failures={min_failures}"
        )
    catalog = {name: index for index, name in enumerate(selected)}
    return {
        "selected_stages": selected,
        "catalog": catalog,
        "task_ids": {name: mapping["name_to_id"][name] for name in selected},
        "support": {
            name: {
                "successes": counts["successes"],
                "failures": counts["failures"],
                "parents": len(counts["parents"]),
            }
            for name, counts in sorted(support.items())
        },
        "excluded": excluded,
        "thresholds": {
            "min_successes": int(min_successes),
            "min_failures": int(min_failures),
        },
        "requested_stages": None if requested is None else sorted(requested),
    }


def filter_stages(rollouts, identity, selected_stages):
    selected = set(selected_stages)
    return [
        rollout
        for rollout in rollouts
        if stage_name(identity[id(rollout)][1]) in selected
    ]


def training_stage_horizons(rollouts, identity, *, quantile=0.5):
    """Estimate deployable stage horizons from fit successes only."""
    quantile = float(quantile)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Stage-horizon quantile must lie in (0, 1]")
    grouped = defaultdict(list)
    names = {}
    for rollout in rollouts:
        task_id = int(rollout.task_id)
        names[task_id] = stage_name(identity[id(rollout)][1])
        if int(rollout.episode_success):
            grouped[task_id].append(len(rollout.hidden_states))
    missing = sorted(task for task in names if not grouped[task])
    if missing:
        raise ValueError(f"Meta-fit has no successful duration for stage IDs {missing}")
    return {
        task_id: max(1, int(math.ceil(float(np.quantile(grouped[task_id], quantile)))))
        for task_id in sorted(names)
    }


def assign_parent_weights(rows):
    """Give every source parent equal total weight across its retained prefixes."""
    rows = [copy.copy(row) for row in rows]
    counts = defaultdict(int)
    for row in rows:
        counts[str(row["parent_rollout_id"])] += 1
    for row in rows:
        row["sample_weight"] = 1.0 / counts[str(row["parent_rollout_id"])]
    total = sum(float(row["sample_weight"]) for row in rows)
    if not rows or total <= 0:
        raise ValueError("No prefix rows remain for parent weighting")
    scale = len(rows) / total
    for row in rows:
        row["sample_weight"] *= scale
    return rows


def build_stage_landmark_rows(
    rollouts,
    identity,
    *,
    stage_catalog,
    horizons,
    time_curves,
    landmarks=DEFAULT_LANDMARKS,
    window=4,
    split,
):
    """Construct causal, relative-landmark examples for active stage segments."""
    landmarks = validate_landmarks(landmarks)
    rows = []
    excluded = []
    for rollout in rollouts:
        task_id = int(rollout.task_id)
        env = identity[id(rollout)][1]
        name = stage_name(env)
        if name not in stage_catalog:
            continue
        if task_id not in horizons:
            raise ValueError(f"Stage {name} is absent from fit-only horizons")
        source_length = len(rollout.hidden_states)
        for landmark in landmarks:
            step = max(1, int(math.ceil(float(landmark) * horizons[task_id])))
            if source_length < step:
                excluded.append(
                    {
                        "parent_rollout_id": parent_id(rollout, identity),
                        "segment_id": segment_id(rollout, identity),
                        "stage_name": name,
                        "task_id": task_id,
                        "landmark_fraction": float(landmark),
                        "source_inferences": source_length,
                        "required_inferences": step,
                        "reason": "stage_terminated_before_landmark",
                    }
                )
                continue
            rows.append(
                {
                    "parent_rollout_id": parent_id(rollout, identity),
                    "segment_id": segment_id(rollout, identity),
                    "stage_name": name,
                    "stage_index": int(stage_catalog[name]),
                    "task_id": task_id,
                    "failed": not bool(int(rollout.episode_success)),
                    "split": str(split),
                    "landmark_fraction": float(landmark),
                    "step": step,
                    "horizon": int(horizons[task_id]),
                    "source_inferences": source_length,
                    "time_risk": time_risk_at(time_curves, task_id, step),
                    "features": summarize_prefix(
                        rollout.hidden_states,
                        step,
                        window=window,
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No causal stage rows were constructed for split={split}")
    return assign_parent_weights(rows), excluded


def balance_bce_rows(rows, *, seed=0):
    """Balance outcome within stage/landmark, then restore parent-equal weights."""
    groups = defaultdict(lambda: {False: [], True: []})
    for index, row in enumerate(rows):
        groups[(str(row["stage_name"]), float(row["landmark_fraction"]))][
            bool(row["failed"])
        ].append(index)
    rng = np.random.default_rng(int(seed))
    selected = set()
    support = {}
    excluded = []
    for key, outcomes in sorted(groups.items()):
        successes = list(outcomes[False])
        failures = list(outcomes[True])
        target = min(len(successes), len(failures))
        support[f"{key[0]}@{key[1]:g}"] = {
            "available_successes": len(successes),
            "available_failures": len(failures),
            "selected_per_outcome": target,
        }
        if target:
            selected.update(int(value) for value in rng.permutation(successes)[:target])
            selected.update(int(value) for value in rng.permutation(failures)[:target])
        for index in successes + failures:
            if index not in selected:
                excluded.append(
                    {
                        "parent_rollout_id": rows[index]["parent_rollout_id"],
                        "segment_id": rows[index]["segment_id"],
                        "stage_name": key[0],
                        "landmark_fraction": key[1],
                        "failed": bool(rows[index]["failed"]),
                        "reason": (
                            "unsupported_outcome_stratum"
                            if target == 0
                            else "deterministic_outcome_balance"
                        ),
                    }
                )
    balanced = [row for index, row in enumerate(rows) if index in selected]
    if not balanced:
        raise ValueError("No supported balanced Subtask-SAFE rows remain")
    return assign_parent_weights(balanced), support, excluded


def rows_to_arrays(rows, scaler, *, residual_time):
    matrix = np.stack([row["features"] for row in rows])
    features = transform_features(matrix, scaler).astype(np.float32)
    targets = np.asarray([int(bool(row["failed"])) for row in rows], dtype=np.float32)
    weights = np.asarray(
        [float(row.get("sample_weight", 1.0)) for row in rows],
        dtype=np.float32,
    )
    stage_indices = np.asarray(
        [int(row["stage_index"]) for row in rows], dtype=np.int64
    )
    offsets = (
        logit([row["time_risk"] for row in rows]).astype(np.float32)
        if residual_time
        else np.zeros(len(rows), dtype=np.float32)
    )
    return features, targets, weights, stage_indices, offsets


def make_stage_model(
    torch,
    *,
    architecture,
    input_dim,
    num_stages,
    hidden_dim=128,
    adapter_dim=16,
    dropout=0.1,
):
    """Create a lightweight shared, conditioned, adapter, or isolated head."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown stage architecture {architecture!r}")
    if int(num_stages) < 1:
        raise ValueError("Stage model requires at least one stage")

    class StageRiskModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = architecture
            self.num_stages = int(num_stages)
            if architecture == "shared_one_hot":
                trunk_input = int(input_dim) + int(num_stages)
            else:
                trunk_input = int(input_dim)
            if architecture == "stage_specific":
                self.stage_networks = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(int(input_dim), int(hidden_dim)),
                            torch.nn.GELU(),
                            torch.nn.Dropout(float(dropout)),
                            torch.nn.Linear(int(hidden_dim), 1),
                        )
                        for _ in range(int(num_stages))
                    ]
                )
                return
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(trunk_input, int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
            )
            self.head = torch.nn.Linear(int(hidden_dim), 1)
            if architecture == "stage_adapter":
                self.adapters = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(int(hidden_dim), int(adapter_dim)),
                            torch.nn.GELU(),
                            torch.nn.Linear(int(adapter_dim), int(hidden_dim)),
                        )
                        for _ in range(int(num_stages))
                    ]
                )
                self.stage_bias = torch.nn.Embedding(int(num_stages), 1)
                torch.nn.init.zeros_(self.stage_bias.weight)

        def forward(self, features, stage_indices):
            if self.architecture == "stage_specific":
                output = features.new_zeros((len(features),))
                for index, network in enumerate(self.stage_networks):
                    values = network(features).squeeze(-1)
                    output = torch.where(stage_indices == index, values, output)
                return output
            if self.architecture == "shared_one_hot":
                one_hot = torch.nn.functional.one_hot(
                    stage_indices,
                    num_classes=self.num_stages,
                ).to(dtype=features.dtype)
                features = torch.cat((features, one_hot), dim=-1)
            hidden = self.trunk(features)
            if self.architecture != "stage_adapter":
                return self.head(hidden).squeeze(-1)
            output = hidden.new_zeros((len(hidden),))
            for index, adapter in enumerate(self.adapters):
                adapted = hidden + adapter(hidden)
                values = self.head(adapted).squeeze(-1)
                output = torch.where(stage_indices == index, values, output)
            return output + self.stage_bias(stage_indices).squeeze(-1)

    return StageRiskModel()


def temporal_contrastive_loss(
    torch,
    logits,
    rows,
    *,
    rank_margin=0.20,
    onset_margin=0.10,
    onset_weight=1.0,
):
    """Weakly localize failure evidence from segment-level outcomes.

    The inter-segment term ranks each failed segment's maximum score above the
    hardest successful segment from the same stage.  The intra-segment term
    locates the largest causal score increase in a failed segment and separates
    the mean score after that proxy onset from the mean score before it.
    """
    scores = torch.sigmoid(logits)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row["stage_name"]), str(row["segment_id"]))].append(index)
    segments = defaultdict(lambda: {False: [], True: []})
    intra_losses = []
    for (name, _), indices in sorted(grouped.items()):
        indices = sorted(
            indices, key=lambda value: float(rows[value]["landmark_fraction"])
        )
        values = scores[torch.as_tensor(indices, device=scores.device)]
        failed = bool(rows[indices[0]]["failed"])
        segments[name][failed].append(torch.max(values))
        if failed and len(indices) >= 2:
            deltas = values[1:] - values[:-1]
            onset = int(torch.argmax(deltas.detach()).item()) + 1
            before = torch.mean(values[:onset])
            after = torch.mean(values[onset:])
            intra_losses.append(
                torch.nn.functional.relu(float(onset_margin) - after + before)
            )
    inter_losses = []
    for outcomes in segments.values():
        if not outcomes[False] or not outcomes[True]:
            continue
        success = torch.stack(outcomes[False])
        failure = torch.stack(outcomes[True])
        inter_losses.append(
            torch.nn.functional.relu(
                float(rank_margin) - failure[:, None] + success[None, :]
            ).mean()
        )
    if not inter_losses:
        raise ValueError(
            "Temporal contrastive loss has no supported stage outcome pairs"
        )
    inter = torch.stack(inter_losses).mean()
    intra = torch.stack(intra_losses).mean() if intra_losses else inter.new_tensor(0.0)
    return inter + float(onset_weight) * intra, {
        "inter": inter,
        "intra": intra,
        "stage_pairs": len(inter_losses),
        "failure_sequences": len(intra_losses),
    }


def landmark_metrics(scored_rows, detectors=("time_only", "candidate")):
    output = []
    landmarks = sorted({float(row["landmark_fraction"]) for row in scored_rows})
    stages = sorted({str(row["stage_name"]) for row in scored_rows})
    for landmark in landmarks:
        landmark_rows = [
            row for row in scored_rows if float(row["landmark_fraction"]) == landmark
        ]
        for detector in detectors:
            for scope, name, selected in [
                ("pooled", None, landmark_rows),
                *[
                    (
                        "stage",
                        stage,
                        [row for row in landmark_rows if row["stage_name"] == stage],
                    )
                    for stage in stages
                ],
            ]:
                if not selected:
                    continue
                labels = [int(bool(row["failed"])) for row in selected]
                scores = [float(row["scores"][detector]) for row in selected]
                output.append(
                    {
                        "scope": scope,
                        "stage_name": name,
                        "landmark_fraction": landmark,
                        "detector": detector,
                        "segments": len(selected),
                        "parents": len({row["parent_rollout_id"] for row in selected}),
                        "successes": sum(not value for value in labels),
                        "failures": sum(labels),
                        "roc_auc": safe_auc(labels, scores),
                        "average_precision": safe_average_precision(labels, scores),
                    }
                )
    return output


def primary_macro_auc(metrics, detector, primary_landmarks):
    primary = set(validate_landmarks(primary_landmarks))
    values = [
        float(row["roc_auc"])
        for row in metrics
        if row["scope"] == "stage"
        and row["detector"] == detector
        and float(row["landmark_fraction"]) in primary
        and row["roc_auc"] is not None
    ]
    if not values:
        raise ValueError(f"Detector {detector} has no supported primary stage AUC")
    return float(np.mean(values)), len(values)


def score_landmark_rows(rows, candidate_scores):
    if len(rows) != len(candidate_scores):
        raise ValueError("Candidate score count does not match landmark rows")
    output = []
    for row, score in zip(rows, candidate_scores):
        item = {key: value for key, value in row.items() if key != "features"}
        item["scores"] = {
            "time_only": float(row["time_risk"]),
            "candidate": float(score),
        }
        output.append(item)
    return output


def parent_bootstrap_delta(
    scored_rows,
    *,
    primary_landmarks=DEFAULT_PRIMARY_LANDMARKS,
    replicates=1000,
    seed=0,
):
    """Bootstrap candidate-minus-time macro AUC by complete parent rollout."""
    metrics = landmark_metrics(scored_rows)
    candidate, support = primary_macro_auc(metrics, "candidate", primary_landmarks)
    time_only, time_support = primary_macro_auc(metrics, "time_only", primary_landmarks)
    if support != time_support:
        raise AssertionError("Candidate and time-only landmark support differ")
    by_parent = defaultdict(list)
    for row in scored_rows:
        by_parent[str(row["parent_rollout_id"])].append(row)
    parents = sorted(by_parent)
    if len(parents) < 2:
        raise ValueError("Parent bootstrap requires at least two diagnostic parents")
    rng = np.random.default_rng(int(seed))
    deltas = []
    for _ in range(int(replicates)):
        sampled = rng.choice(parents, size=len(parents), replace=True)
        rows = []
        for draw, key in enumerate(sampled):
            for source in by_parent[str(key)]:
                item = copy.copy(source)
                item["parent_rollout_id"] = f"draw-{draw:04d}::{key}"
                rows.append(item)
        replicate_metrics = landmark_metrics(rows)
        try:
            candidate_value, candidate_support = primary_macro_auc(
                replicate_metrics,
                "candidate",
                primary_landmarks,
            )
            time_value, replicate_time_support = primary_macro_auc(
                replicate_metrics,
                "time_only",
                primary_landmarks,
            )
        except ValueError:
            continue
        if candidate_support != replicate_time_support:
            raise AssertionError("Bootstrap detector supports differ")
        deltas.append(candidate_value - time_value)
    if not deltas:
        raise ValueError("All parent bootstrap replicates lacked two-class support")
    return {
        "candidate_macro_auc": candidate,
        "time_only_macro_auc": time_only,
        "delta": candidate - time_only,
        "supported_stage_landmarks": support,
        "replicates_requested": int(replicates),
        "replicates_valid": len(deltas),
        "delta_ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def score_full_stage_trajectories(
    rollouts,
    identity,
    *,
    stage_catalog,
    horizons,
    time_curves,
    scaler,
    predict,
    residual_time,
    window=4,
):
    """Score complete causal stage trajectories with a supplied predictor."""
    output = []
    for rollout in rollouts:
        env = identity[id(rollout)][1]
        name = stage_name(env)
        if name not in stage_catalog:
            continue
        task_id = int(rollout.task_id)
        features = summarize_trajectory(
            rollout.hidden_states,
            horizon=horizons[task_id],
            window=window,
        )
        features = transform_features(features, scaler).astype(np.float32)
        steps = np.arange(1, len(features) + 1, dtype=np.int64)
        curve = np.asarray(time_curves[task_id]["risk"], dtype=np.float64)
        time_scores = curve[np.minimum(steps, len(curve)) - 1]
        offsets = (
            logit(time_scores).astype(np.float32)
            if residual_time
            else np.zeros(len(features), dtype=np.float32)
        )
        candidate = np.asarray(
            predict(
                features,
                np.full(len(features), stage_catalog[name], dtype=np.int64),
                offsets,
            ),
            dtype=np.float64,
        )
        output.append(
            {
                "parent_rollout_id": parent_id(rollout, identity),
                "segment_id": segment_id(rollout, identity),
                "stage_name": name,
                "task_id": task_id,
                "failed": not bool(int(rollout.episode_success)),
                "source_inferences": len(rollout.hidden_states),
                "horizon": int(horizons[task_id]),
                "trajectories": {
                    "time_only": time_scores,
                    "candidate": candidate,
                },
            }
        )
    if not output:
        raise ValueError("No supported stage trajectories were scored")
    return output


def select_stage_thresholds(
    scored,
    detector,
    *,
    target_fpr=0.05,
    min_successes=5,
):
    """Fit a separate event threshold for each stage on selection segments."""
    grouped = defaultdict(list)
    for item in scored:
        grouped[str(item["stage_name"])].append(item)
    thresholds = {}
    audit = {}
    for name, values in sorted(grouped.items()):
        successes = [item for item in values if not item["failed"]]
        failures = [item for item in values if item["failed"]]
        if len(successes) < int(min_successes) or not failures:
            audit[name] = {
                "supported": False,
                "successes": len(successes),
                "failures": len(failures),
                "reason": "insufficient_selection_support",
            }
            continue
        maxima = np.asarray(
            [float(np.max(item["trajectories"][detector])) for item in values],
            dtype=np.float64,
        )
        labels = np.asarray(
            [int(bool(item["failed"])) for item in values], dtype=np.int64
        )
        causal_scores = np.concatenate(
            [
                np.asarray(item["trajectories"][detector], dtype=np.float64)
                for item in values
            ]
        )
        if not np.all(np.isfinite(causal_scores)):
            raise ValueError(f"Stage {name} has non-finite causal scores")
        if np.any(causal_scores < 0.0) or np.any(causal_scores > 1.0):
            raise ValueError(f"Stage {name} has scores outside [0, 1]")
        epsilon = np.finfo(np.float64).eps
        candidates = np.concatenate(
            [[1.0 + epsilon], np.unique(causal_scores)[::-1], [-epsilon]]
        )
        feasible = []
        for threshold in candidates:
            predicted = maxima >= threshold
            fpr = float(np.mean(predicted[labels == 0]))
            tpr = float(np.mean(predicted[labels == 1]))
            if fpr <= float(target_fpr) + 1e-12:
                feasible.append((tpr, -fpr, -float(threshold), float(threshold)))
        if not feasible:
            raise ValueError(f"Stage {name} has no threshold with defined TPR/FPR")
        threshold = max(feasible)[3]
        thresholds[name] = threshold
        audit[name] = {
            "supported": True,
            "successes": len(successes),
            "failures": len(failures),
            "threshold": threshold,
            "abstains_on_selection": bool(threshold > np.max(causal_scores)),
            "target_fpr": float(target_fpr),
            "conformal_resolution_proxy": 1.0 / (len(successes) + 1.0),
        }
    if not thresholds:
        raise ValueError("No stage has enough selection support for threshold fitting")
    return thresholds, audit


def stage_event_metrics(scored, detector, thresholds):
    """Evaluate first alarms per active semantic stage segment."""
    grouped = defaultdict(list)
    predictions = []
    for item in scored:
        name = str(item["stage_name"])
        if name not in thresholds:
            continue
        trajectory = np.asarray(item["trajectories"][detector], dtype=np.float64)
        threshold = float(thresholds[name])
        crossings = np.flatnonzero(trajectory >= threshold)
        first = None if not len(crossings) else int(crossings[0]) + 1
        source_length = int(item["source_inferences"])
        predictions.append(
            {
                "parent_rollout_id": item["parent_rollout_id"],
                "segment_id": item["segment_id"],
                "stage_name": name,
                "failed": bool(item["failed"]),
                "maximum": float(np.max(trajectory)),
                "threshold": threshold,
                "first_detection_inference": first,
                "normalized_detection_fraction": (
                    None if first is None else min(1.0, first / source_length)
                ),
                "normalized_lead": (
                    None
                    if first is None
                    else max(0.0, (source_length - first) / source_length)
                ),
            }
        )
        grouped[name].append(predictions[-1])
    metrics = []
    for name, values in sorted(grouped.items()):
        labels = np.asarray([int(item["failed"]) for item in values], dtype=np.int64)
        predicted = np.asarray(
            [item["first_detection_inference"] is not None for item in values],
            dtype=bool,
        )
        successes = labels == 0
        failures = labels == 1
        tpr = float(np.mean(predicted[failures])) if np.any(failures) else None
        fpr = float(np.mean(predicted[successes])) if np.any(successes) else None
        detected_lead = [
            float(item["normalized_lead"])
            for item in values
            if item["failed"] and item["normalized_lead"] is not None
        ]
        adjusted_fraction = [
            (
                1.0
                if item["first_detection_inference"] is None
                else float(item["normalized_detection_fraction"])
            )
            for item in values
            if item["failed"]
        ]
        metrics.append(
            {
                "detector": detector,
                "stage_name": name,
                "segments": len(values),
                "parents": len({item["parent_rollout_id"] for item in values}),
                "successes": int(np.sum(successes)),
                "failures": int(np.sum(failures)),
                "true_positive_rate": tpr,
                "false_positive_rate": fpr,
                "balanced_accuracy": (
                    None if tpr is None or fpr is None else 0.5 * (tpr + 1.0 - fpr)
                ),
                "mean_detected_lead": (
                    float(np.mean(detected_lead)) if detected_lead else None
                ),
                "missed_failure_adjusted_detection_fraction": (
                    float(np.mean(adjusted_fraction)) if adjusted_fraction else None
                ),
            }
        )
    if not metrics:
        raise ValueError("No diagnostic stage has a fitted threshold")
    macro = {}
    for field in (
        "true_positive_rate",
        "false_positive_rate",
        "balanced_accuracy",
        "mean_detected_lead",
        "missed_failure_adjusted_detection_fraction",
    ):
        values = [float(row[field]) for row in metrics if row[field] is not None]
        macro[field] = float(np.mean(values)) if values else None
    macro["max_stage_false_positive_rate"] = max(
        float(row["false_positive_rate"])
        for row in metrics
        if row["false_positive_rate"] is not None
    )
    macro["supported_stages"] = len(metrics)
    return {"per_stage": metrics, "macro": macro, "predictions": predictions}


def continuation_gate(
    bootstrap,
    event_metrics,
    *,
    min_roc=0.65,
    min_delta=0.05,
    min_tpr=0.40,
    max_fpr=0.05,
    min_lead=0.25,
):
    macro = event_metrics["macro"]
    checks = {
        "minimum_macro_roc": bootstrap["candidate_macro_auc"] >= float(min_roc),
        "minimum_delta_over_time": bootstrap["delta"] >= float(min_delta),
        "bootstrap_delta_excludes_zero": bootstrap["delta_ci95"][0] > 0.0,
        "minimum_event_tpr": (
            macro["true_positive_rate"] is not None
            and macro["true_positive_rate"] >= float(min_tpr)
        ),
        "maximum_stage_fpr": (
            macro["max_stage_false_positive_rate"] <= float(max_fpr) + 1e-12
        ),
        "minimum_detected_lead": (
            macro["mean_detected_lead"] is not None
            and macro["mean_detected_lead"] >= float(min_lead)
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_roc": float(min_roc),
            "min_delta": float(min_delta),
            "min_tpr": float(min_tpr),
            "max_fpr": float(max_fpr),
            "min_lead": float(min_lead),
        },
    }


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_LANDMARKS",
    "DEFAULT_PRIMARY_LANDMARKS",
    "DETECTORS",
    "OBJECTIVES",
    "assign_parent_weights",
    "balance_bce_rows",
    "build_stage_landmark_rows",
    "continuation_gate",
    "filter_stages",
    "fit_feature_scaler",
    "fit_time_risk",
    "landmark_metrics",
    "make_stage_model",
    "parent_bootstrap_delta",
    "parent_id",
    "primary_macro_auc",
    "rows_to_arrays",
    "score_full_stage_trajectories",
    "score_landmark_rows",
    "segment_id",
    "select_parent_rollouts",
    "select_stage_thresholds",
    "select_supported_stages",
    "stage_event_metrics",
    "temporal_contrastive_loss",
    "three_way_parent_split",
    "training_stage_horizons",
    "validate_landmarks",
    "validate_stage_identity",
]
