"""Score a new parent allocation with a frozen causal Subtask-SAFE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .causal_subtask_safe import (
        append_causal_conditioning,
        expand_causal_prefixes,
        stage_name,
        transform_temporal_representation,
    )
    from .train_seen_tasks import (
        load_env_records,
        save_scores,
        score_splits,
        validate_alignment,
        verify_safe_repo,
        write_json,
    )
except ImportError:
    from causal_subtask_safe import (
        append_causal_conditioning,
        expand_causal_prefixes,
        stage_name,
        transform_temporal_representation,
    )
    from train_seen_tasks import (
        load_env_records,
        save_scores,
        score_splits,
        validate_alignment,
        verify_safe_repo,
        write_json,
    )


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _training_protocol(training_run):
    run = Path(training_run).resolve()
    metrics = json.loads((run / "metrics.json").read_text())
    causal = metrics.get("causal_subtask_safe")
    if not causal:
        raise ValueError("Training run is not a causal Subtask-SAFE refit")
    protocol = causal["protocol"]
    representation = causal["temporal_representation"]
    conditioning = causal["conditioning"]
    if protocol.get("label_mode") != "within_horizon":
        raise ValueError("Frozen scoring requires a finite-horizon training run")
    if not conditioning.get("catalog"):
        raise ValueError("Training run does not preserve its stage catalog")
    return run, metrics, protocol, representation, conditioning


def score_frozen_checkpoint(
    export_dir,
    training_run,
    allocation_manifest,
    safe_repo,
    output_dir,
    *,
    device="cuda",
):
    """Apply a frozen checkpoint and prepend its immutable training scores."""
    export_dir = Path(export_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run, metrics, protocol, representation, conditioning = _training_protocol(
        training_run
    )
    allocation_path = Path(allocation_manifest).resolve()
    allocation = json.loads(allocation_path.read_text())
    if allocation.get("protocol") != "causal_subtask_finite_horizon_parent_allocation":
        raise ValueError("Unknown target-aware allocation protocol")
    if not allocation.get("complete"):
        raise ValueError("Target-aware allocation is incomplete")
    target = allocation["target_definition"]
    if int(target["failure_horizon_inferences"]) != int(
        protocol["failure_horizon"]
    ):
        raise ValueError("Allocation and frozen checkpoint use different horizons")
    selected_stages = set(allocation["selected_stages"])
    catalog = {str(key): int(value) for key, value in conditioning["catalog"].items()}
    unknown = sorted(selected_stages - set(catalog))
    if unknown:
        raise ValueError(
            "Allocation stages are absent from the frozen catalog: "
            + ", ".join(unknown)
        )
    target_prefix = int(target["causal_prefix_inferences"])
    if target_prefix not in {int(value) for value in protocol["horizons"]}:
        raise ValueError("Allocation prefix is absent from the frozen prefix protocol")
    allocated_parents = {
        str(value)
        for key in ("calibration_parent_ids", "evaluation_parent_ids")
        for value in allocation[key]
    }

    verify_safe_repo(safe_repo)
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from failure_prob.data.pizero import load_rollouts_from_root
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.model import get_model

    cfg = OmegaConf.load(run / "config.yaml")
    source_rollouts = load_rollouts_from_root(export_dir, cfg)
    source_env_records = load_env_records(export_dir)
    source_identity = validate_alignment(source_rollouts, source_env_records)
    source_rollouts = [
        rollout
        for rollout in source_rollouts
        if stage_name(source_identity[id(rollout)][1]) in selected_stages
        and str(source_identity[id(rollout)][1].get("parent_rollout_id"))
        in allocated_parents
    ]
    observed_parents = {
        str(source_identity[id(rollout)][1].get("parent_rollout_id"))
        for rollout in source_rollouts
    }
    missing = sorted(allocated_parents - observed_parents)
    if missing:
        raise ValueError(
            "Allocated parents are absent from the scoring export: "
            + ", ".join(missing[:5])
        )
    rollouts, identity, prefix_counts = expand_causal_prefixes(
        source_rollouts,
        source_identity,
        mode="fixed",
        horizons=tuple(int(value) for value in protocol["horizons"]),
        label_mode=protocol["label_mode"],
        failure_horizon=int(protocol["failure_horizon"]),
    )
    transformed = transform_temporal_representation(
        rollouts,
        mode=representation["mode"],
        window=int(representation["window"]),
    )
    conditioned = append_causal_conditioning(
        rollouts,
        identity,
        mode=conditioning["mode"],
        catalog=catalog,
        elapsed_scale=int(
            conditioning.get("elapsed_scale") or max(protocol["horizons"])
        ),
    )
    for rollout in rollouts:
        rollout.task_min_step = len(rollout.hidden_states)
    if cfg.dataset.load_to_cuda:
        rollouts = [rollout.to(device) for rollout in rollouts]
    dataset = RolloutDataset(cfg, rollouts)
    loader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = get_model(cfg, int(rollouts[0].hidden_states.shape[-1]))
    try:
        state = torch.load(
            run / "model_final.ckpt", map_location=device, weights_only=True
        )
    except TypeError:
        state = torch.load(run / "model_final.ckpt", map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    scores = score_splits(model, {"test": loader})

    names = {}
    task_types = {}
    for rollout in rollouts:
        env = identity[id(rollout)][1]
        task_id = int(rollout.task_id)
        name = stage_name(env)
        if task_id in names and names[task_id] != name:
            raise ValueError(f"Task ID {task_id} maps to multiple semantic stages")
        names[task_id] = name
        task_types[name] = str(env.get("task_type", "composite"))
    external_path = output / "external_scores.jsonl"
    save_scores(
        external_path,
        {"test": rollouts},
        scores,
        identity,
        names,
        task_types,
        metrics["model"],
        int(metrics["seed"]),
    )
    training_scores = [
        record
        for record in _load_jsonl(run / "scores.jsonl")
        if record["split"] == "train" and record["task_name"] in selected_stages
    ]
    external_scores = _load_jsonl(external_path)
    if (
        {record["parent_rollout_id"] for record in training_scores}
        & allocated_parents
    ):
        raise ValueError("Frozen training parents overlap the new allocation")
    combined_path = output / "scores.jsonl"
    combined_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in training_scores + external_scores
        )
    )
    result = {
        "schema_version": 1,
        "protocol": "frozen_causal_subtask_checkpoint_external_scoring",
        "training_run": str(run),
        "checkpoint": str(run / "model_final.ckpt"),
        "export_dir": str(export_dir),
        "allocation_manifest": str(allocation_path),
        "model": metrics["model"],
        "seed": int(metrics["seed"]),
        "selected_stages": sorted(selected_stages),
        "target_definition": target,
        "prefix_counts": prefix_counts,
        "temporal_representation": transformed,
        "conditioning": conditioned,
        "counts": {
            "immutable_training_scores": len(training_scores),
            "external_test_scores": len(external_scores),
            "external_parents": len(allocated_parents),
        },
        "scores": str(combined_path),
        "checkpoint_updated": False,
    }
    write_json(output / "scoring_report.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--allocation-manifest", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            score_frozen_checkpoint(
                args.export_dir,
                args.training_run,
                args.allocation_manifest,
                args.safe_repo,
                args.output_dir,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
