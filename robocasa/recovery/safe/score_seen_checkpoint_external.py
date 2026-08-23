"""Score a new SAFE export with one frozen seen-task detector checkpoint.

This command never trains or calibrates a model.  It applies the task cutoffs,
feature selectors, and detector weights stored by a completed final refit to a
new calibration or prospective shadow export and persists full causal score
trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def score_external_export(
    *,
    export_dir,
    safe_repo,
    training_run_dir,
    output_dir,
    group,
    device="cuda",
    allow_task_subset=False,
):
    import sys

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    export_dir = Path(export_dir).resolve()
    training_run = Path(training_run_dir).resolve()
    checkpoint = training_run / "model_final.ckpt"
    config_path = training_run / "config.yaml"
    metrics_path = training_run / "metrics.json"
    split_path = training_run / "split_manifest.json"
    for path in (checkpoint, config_path, metrics_path, split_path):
        if not path.is_file():
            raise ValueError(f"Frozen training artifact is missing: {path}")

    from .train_seen_tasks import (
        OFFICIAL_SAFE_COMMIT,
        load_env_records,
        task_catalog,
        validate_alignment,
        verify_safe_repo,
    )

    verify_safe_repo(safe_repo)
    if str(Path(safe_repo).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(safe_repo).resolve()))

    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from failure_prob.data.pizero import load_rollouts_from_root
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.model import get_model
    from failure_prob.utils.random import seed_everything
    from failure_prob.utils.routines import model_forward_dataloader

    cfg = OmegaConf.load(config_path)
    cfg.dataset.data_path_prefix = ""
    cfg.dataset.data_path = str(export_dir)
    cfg.dataset.data_path_unseen = None
    cfg.dataset.load_to_cuda = str(device).startswith("cuda")
    cfg.train.log_precomputed = False
    cfg.train.log_precomputed_only = False
    seed = int(cfg.train.seed)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested {device}, but CUDA is unavailable")

    seed_everything(0)
    rollouts = load_rollouts_from_root(export_dir, cfg)
    env_records = load_env_records(export_dir)
    identity = validate_alignment(rollouts, env_records)
    catalog = task_catalog(export_dir)
    training_metrics = json.loads(metrics_path.read_text())
    split_manifest = json.loads(split_path.read_text())
    expected_tasks = set(split_manifest["task_names"])
    actual_tasks = {
        catalog["task_names"][int(rollout.task_id)] for rollout in rollouts
    }
    unexpected = sorted(actual_tasks - expected_tasks)
    missing = sorted(expected_tasks - actual_tasks)
    if unexpected or (missing and not allow_task_subset):
        raise ValueError(
            f"External export task coverage differs from training: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if not actual_tasks:
        raise ValueError("External export contains no scored tasks")
    cutoffs = training_metrics.get("task_min_steps", {})
    if set(cutoffs) != expected_tasks:
        raise ValueError("Frozen training metrics lack exact task cutoff coverage")
    for rollout in rollouts:
        task = catalog["task_names"][int(rollout.task_id)]
        rollout.task_min_step = min(int(cutoffs[task]), len(rollout.hidden_states))
    if cfg.dataset.load_to_cuda:
        rollouts = [rollout.to(device) for rollout in rollouts]

    dataset = RolloutDataset(cfg, rollouts)
    loader = DataLoader(
        dataset,
        batch_size=dataset.cfg.model.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = get_model(cfg, int(rollouts[0].hidden_states.shape[-1]))
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        scores, masks, _ = model_forward_dataloader(model, loader)
    scores = scores.detach().cpu().numpy()
    lengths = masks.sum(dim=-1).detach().cpu().numpy().astype(int)
    trajectories = [scores[index, :length] for index, length in enumerate(lengths)]
    if any(not len(score) or not np.all(np.isfinite(score)) for score in trajectories):
        raise RuntimeError("Frozen checkpoint produced an invalid score trajectory")

    records = []
    for rollout, score in zip(rollouts, trajectories):
        env_path, env = identity[id(rollout)]
        manifest = env.get("robocasa_manifest_record", {})
        task_name = catalog["task_names"][int(rollout.task_id)]
        records.append(
            {
                "rollout_id": str(env["rollout_id"]),
                "group": str(group),
                "task_id": int(rollout.task_id),
                "task_name": task_name,
                "task_type": catalog["task_types"][task_name],
                "failed": not bool(int(rollout.episode_success)),
                "model": str(cfg.model.name),
                "seed": seed,
                "scores": np.asarray(score, dtype=np.float64).tolist(),
                "num_inferences": len(score),
                "task_min_step": int(rollout.task_min_step),
                "environment_seed": env.get("environment_seed"),
                "environment_reset_index": env.get("environment_reset_index"),
                "seed_protocol": env.get("seed_protocol"),
                "video_path": str(env_path.with_suffix(".mp4")),
                "video_frame_stride": int(env.get("video_frame_stride", 1)),
                "inference_environment_steps": manifest.get(
                    "inference_environment_steps",
                    list(range(0, len(score) * int(env["replan_steps"]), int(env["replan_steps"]))),
                ),
            }
        )
    ids = [row["rollout_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("External scoring produced duplicate rollout IDs")
    with (output / "scores.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    provenance = {
        "schema_version": 1,
        "protocol": "frozen_seen_safe_external_scoring",
        "group": str(group),
        "export_dir": str(export_dir),
        "training_run_dir": str(training_run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "safe_repository": str(Path(safe_repo).resolve()),
        "safe_repository_commit": OFFICIAL_SAFE_COMMIT,
        "model": str(cfg.model.name),
        "seed": seed,
        "task_names": sorted(actual_tasks),
        "training_task_names": sorted(expected_tasks),
        "task_subset_allowed": bool(allow_task_subset),
        "rollouts": len(records),
        "successes": sum(not row["failed"] for row in records),
        "failures": sum(row["failed"] for row in records),
        "checkpoint_updated": False,
        "thresholds_fitted": False,
    }
    _write_json(output / "provenance.json", provenance)
    return provenance


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--training-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", choices=("calibration", "prospective_test"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-task-subset",
        action="store_true",
        help=(
            "Allow the external export to contain a strict subset of the "
            "checkpoint's training tasks. Unexpected tasks remain forbidden."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = score_external_export(**vars(args))
    except (FileExistsError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
