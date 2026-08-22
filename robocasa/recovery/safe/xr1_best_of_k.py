"""Frozen Xiaomi SAFE scoring for bounded recovery-time best-of-K reranking.

The prospective detector accumulated scores across a rollout. Candidate chunks
share no executed history, so this module deliberately compares their
single-inference SAFE contributions after the same frozen per-task
normalization. It is a ranking probe, not the calibrated online alarm score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SCORING_PROTOCOL = "xr1_safe_single_inference_normalized_ensemble_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def select_candidate_features(raw_features):
    """Apply the frozen horizon=1.0, diffusion=1.0 selectors."""
    features = np.asarray(raw_features)
    if features.ndim != 4:
        raise ValueError(
            "Candidate SAFE features must have shape "
            "(candidates, flow_steps, action_horizon, feature_dim), got "
            f"{features.shape}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(
            f"Candidate SAFE features must be floating, got {features.dtype}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Candidate SAFE features contain NaN or infinite values")
    return np.ascontiguousarray(features[:, -1, -1, :], dtype=np.float32)


def normalize_seed_scores(raw_scores, *, task_name, seed, task_normalizations):
    seed_stats = task_normalizations.get(str(seed))
    if not isinstance(seed_stats, dict) or task_name not in seed_stats:
        raise ValueError(
            f"Runtime bundle lacks normalization for seed {seed}, task {task_name!r}"
        )
    stats = seed_stats[task_name]
    location = float(stats["location"])
    scale = float(stats["scale"])
    if not np.isfinite(location) or not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"Invalid normalization for seed {seed}, task {task_name!r}: "
            f"location={location}, scale={scale}"
        )
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("SAFE model returned invalid candidate scores")
    return (scores - location) / scale


class FrozenXr1SafeEnsemble:
    """Load the frozen three-seed official SAFE MLP ensemble lazily."""

    def __init__(self, *, runtime_bundle, safe_repo, device="cpu"):
        self.runtime_bundle_path = Path(runtime_bundle).resolve()
        if not self.runtime_bundle_path.is_file():
            raise ValueError(
                f"SAFE runtime bundle is missing: {self.runtime_bundle_path}"
            )
        self.bundle = json.loads(self.runtime_bundle_path.read_text())
        self.runtime_bundle_sha256 = sha256_file(self.runtime_bundle_path)
        if int(self.bundle.get("schema_version", -1)) != 1:
            raise ValueError("Expected SAFE runtime bundle schema_version=1")
        if self.bundle.get("model") != "indep":
            raise ValueError("Best-of-K pilot requires the frozen indep SAFE model")
        if self.bundle.get("binary_final_rollout_labels_only") is not True:
            raise ValueError(
                "Best-of-K pilot requires binary final-rollout SAFE labels"
            )
        if self.bundle.get("subtask_safe") is not False:
            raise ValueError("Best-of-K pilot cannot use a Subtask-SAFE runtime bundle")
        checkpoint_bundle = self.bundle.get("checkpoint_bundle")
        if not isinstance(checkpoint_bundle, dict) or not checkpoint_bundle:
            raise ValueError("SAFE runtime bundle has no checkpoint_bundle")
        self.seeds = tuple(sorted(int(seed) for seed in checkpoint_bundle))
        if len(self.seeds) < 2:
            raise ValueError(
                "SAFE reranking requires an ensemble with at least two seeds"
            )
        recorded_seeds = tuple(
            sorted(int(seed) for seed in self.bundle.get("model_seeds", ()))
        )
        if recorded_seeds != self.seeds:
            raise ValueError("SAFE runtime model_seeds and checkpoint seeds disagree")
        self.task_normalizations = self.bundle.get("task_normalizations", {})
        if set(map(str, self.seeds)) != set(self.task_normalizations):
            raise ValueError("SAFE checkpoint seeds and normalization seeds disagree")
        self.safe_repo = Path(safe_repo).resolve()
        self.device = str(device)
        self.artifact_paths = {
            seed: {
                name: self._resolve_artifact(seed, name)
                for name in ("config.yaml", "model_final.ckpt")
            }
            for seed in self.seeds
        }
        self._models = None
        self._torch = None

    def _resolve_artifact(self, seed, name):
        entry = self.bundle["checkpoint_bundle"][str(seed)][name]
        recorded = Path(entry["path"])
        local = self.runtime_bundle_path.parent / "runtime" / f"seed{seed}" / name
        path = recorded if recorded.is_file() else local
        if not path.is_file():
            raise ValueError(
                f"Frozen SAFE artifact is missing for seed {seed}: {recorded}; "
                f"local fallback also missing: {local}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(
                f"Frozen SAFE artifact hash mismatch for {path}: "
                f"{actual_hash} != {entry['sha256']}"
            )
        return path

    @staticmethod
    def _selector(cfg, name):
        value = float(getattr(cfg.dataset, name))
        if value != 1.0:
            raise ValueError(
                f"Frozen best-of-K scorer requires {name}=1.0, got {value}"
            )

    def _load_models(self, input_dim):
        from .train_seen_tasks import OFFICIAL_SAFE_COMMIT, verify_safe_repo

        verify_safe_repo(self.safe_repo)
        self.safe_commit = OFFICIAL_SAFE_COMMIT
        if str(self.safe_repo) not in sys.path:
            sys.path.insert(0, str(self.safe_repo))

        import torch
        from omegaconf import OmegaConf
        import failure_prob.model as safe_model

        model_module_path = Path(safe_model.__file__).resolve()
        if self.safe_repo not in model_module_path.parents:
            raise ValueError(
                "Imported failure_prob.model from the wrong SAFE checkout: "
                f"{model_module_path}"
            )
        get_model = safe_model.get_model

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested {self.device}, but CUDA is unavailable")
        models = []
        for seed in self.seeds:
            config_path = self.artifact_paths[seed]["config.yaml"]
            checkpoint_path = self.artifact_paths[seed]["model_final.ckpt"]
            cfg = OmegaConf.load(config_path)
            if str(cfg.model.name) != "indep":
                raise ValueError(
                    f"Seed {seed} config uses {cfg.model.name!r}, expected 'indep'"
                )
            self._selector(cfg, "horizon_idx_rel")
            self._selector(cfg, "diff_idx_rel")
            model = get_model(cfg, int(input_dim))
            try:
                state = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=True
                )
            except TypeError:
                state = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state, strict=True)
            model.to(self.device)
            model.eval()
            models.append(model)
        self._models = tuple(models)
        self._torch = torch

    def score_candidates(self, raw_features, *, task_name):
        selected = select_candidate_features(raw_features)
        if self._models is None:
            self._load_models(selected.shape[-1])
        if not selected.shape[-1]:
            raise ValueError("Candidate SAFE feature dimension cannot be zero")

        tensor = self._torch.as_tensor(selected, device=self.device).unsqueeze(1)
        normalized_by_seed = []
        raw_by_seed = []
        with self._torch.no_grad():
            for seed, model in zip(self.seeds, self._models):
                output = model({"features": tensor})
                scores = output.detach().cpu().numpy().reshape(len(selected), -1)
                if scores.shape[1] != 1:
                    raise RuntimeError(
                        "Frozen SAFE model did not return one score per candidate: "
                        f"{scores.shape}"
                    )
                raw_scores = scores[:, 0].astype(np.float64, copy=False)
                normalized = normalize_seed_scores(
                    raw_scores,
                    task_name=task_name,
                    seed=seed,
                    task_normalizations=self.task_normalizations,
                )
                raw_by_seed.append(raw_scores)
                normalized_by_seed.append(normalized)
        ensemble = np.mean(np.stack(normalized_by_seed, axis=0), axis=0)
        if not np.all(np.isfinite(ensemble)):
            raise RuntimeError("Frozen SAFE ensemble returned invalid candidate scores")
        return {
            "protocol": SCORING_PROTOCOL,
            "scores": ensemble.tolist(),
            "seeds": list(self.seeds),
            "raw_scores_by_seed": np.stack(raw_by_seed).tolist(),
            "normalized_scores_by_seed": np.stack(normalized_by_seed).tolist(),
            "task_name": str(task_name),
            "provenance": {
                "runtime_bundle": str(self.runtime_bundle_path),
                "runtime_bundle_sha256": self.runtime_bundle_sha256,
                "safe_repository": str(self.safe_repo),
                "safe_repository_commit": self.safe_commit,
                "model": "indep",
                "horizon_selector": 1.0,
                "diffusion_selector": 1.0,
                "checkpoint_sha256_by_seed": {
                    str(seed): self.bundle["checkpoint_bundle"][str(seed)][
                        "model_final.ckpt"
                    ]["sha256"]
                    for seed in self.seeds
                },
            },
        }
