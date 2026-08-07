"""Online SAFE scoring for policy-inference records.

The policy adapters expose one raw feature record per genuine model inference
through ``pop_inference_record()``.  This module turns those records into an
online failure decision without scoring cached actions more than once.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

from .conformal import load_calibration, threshold_for_length
from .dataset import aggregate_features
from .models import load_checkpoint, require_torch, torch


OFFICIAL_SAFE_COMMIT = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
RLDX_SAFE_FEATURE_LAYER = "action_model_msat_action_suffix_pre_action_decoder"


class CheckpointSafeMonitor:
    """Score raw SAFE records with a RoboCasa SAFE checkpoint and calibration.

    ``checkpoint`` must use the format written by
    :func:`robocasa.recovery.safe.models.save_checkpoint`.  The feature
    aggregation defaults to the value stored in the checkpoint metadata.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        calibration: str | Path,
        *,
        device: str = "cpu",
        aggregation: str | None = None,
    ):
        require_torch()
        self.checkpoint_path = Path(checkpoint)
        self.calibration_path = Path(calibration)
        self.device = str(device)
        self.model, self.model_config, extra = load_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )
        self.aggregation = aggregation or extra.get("aggregation")
        if not self.aggregation:
            raise ValueError(
                "SAFE checkpoint does not declare feature aggregation; pass "
                "aggregation explicitly"
            )
        self.calibration = load_calibration(self.calibration_path)
        if "threshold" not in self.calibration:
            raise ValueError("SAFE calibration is missing 'threshold'")
        self.reset()

    def reset(self, task_name=None):
        """Start a new independent rollout score trajectory."""
        self._features = []
        self._decisions = []

    def describe(self):
        return {
            "type": "checkpoint_safe_monitor",
            "checkpoint": str(self.checkpoint_path),
            "calibration": str(self.calibration_path),
            "device": self.device,
            "model_type": self.model_config.model_type,
            "input_dim": int(self.model_config.input_dim),
            "aggregation": self.aggregation,
            "crossing_rule": self.calibration.get("crossing_rule"),
        }

    @property
    def decisions(self):
        return list(self._decisions)

    def observe_inference(self, record):
        """Score one genuine policy inference and return a JSON-safe decision."""
        if not isinstance(record, dict):
            raise TypeError("SAFE inference record must be a dictionary")
        if "features" not in record:
            raise ValueError("SAFE inference record is missing raw 'features'")

        raw = np.asarray(record["features"], dtype=np.float32)
        if raw.ndim != 3:
            raise ValueError(
                "Online SAFE features must have shape "
                f"(flow_steps, action_horizon, feature_dim), got {raw.shape}"
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("Online SAFE features contain NaN or infinite values")
        feature = aggregate_features(raw[None], self.aggregation)[0]
        if feature.shape != (int(self.model_config.input_dim),):
            raise ValueError(
                "Aggregated online SAFE feature shape does not match checkpoint: "
                f"{feature.shape} != ({self.model_config.input_dim},)"
            )
        self._features.append(np.asarray(feature, dtype=np.float32))

        sequence = np.stack(self._features)
        tensor = torch.from_numpy(sequence[None]).to(self.device)
        lengths = torch.tensor([len(sequence)], device=self.device)
        with torch.no_grad():
            scores = self.model(tensor, lengths)[0].detach().cpu().numpy()
        score = float(scores[len(sequence) - 1])
        threshold = float(
            threshold_for_length(self.calibration, len(sequence))[-1]
        )
        if not np.isfinite(score) or not np.isfinite(threshold):
            raise RuntimeError("Online SAFE produced a non-finite score or threshold")

        decision = {
            "schema_version": 1,
            "inference_index": int(
                record.get("inference_index", len(sequence) - 1)
            ),
            "environment_step": int(
                record.get("environment_step", record.get("env_step", 0))
            ),
            "trajectory_length": int(len(sequence)),
            "score": score,
            "threshold": threshold,
            "failure_detected": bool(score >= threshold),
            "aggregation": self.aggregation,
        }
        self._decisions.append(decision)
        return dict(decision)


class OfficialSafeCheckpointMonitor:
    """Run a checkpoint produced by pinned, official SAFE training code.

    The ten-task RLDX refit stores a raw official ``state_dict`` plus the
    official OmegaConf config, rather than the repository-native checkpoint
    format consumed by :class:`CheckpointSafeMonitor`.  This monitor imports
    the pinned SAFE checkout, applies its exact horizon/diffusion selectors,
    and evaluates the complete causal feature prefix after each RLDX inference.

    Exactly one threshold source is required: a functional calibration JSON or
    a fixed scalar.  The fixed scalar is useful for the original uncalibrated
    10/10 pilot, but carries no conformal false-alert guarantee.
    """

    def __init__(
        self,
        safe_repo: str | Path,
        checkpoint: str | Path,
        config: str | Path,
        *,
        device: str = "cpu",
        calibration: str | Path | None = None,
        fixed_threshold: float | None = None,
        task_normalization: str | Path | None = None,
        expected_commit: str | None = OFFICIAL_SAFE_COMMIT,
    ):
        if (calibration is None) == (fixed_threshold is None):
            raise ValueError(
                "Provide exactly one SAFE threshold source: calibration or "
                "fixed_threshold"
            )
        from .evaluate_official_safe import verify_safe_checkout

        self.safe_repo, self.safe_commit = verify_safe_checkout(
            safe_repo, expected_commit
        )
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        self.config_path = Path(config).expanduser().resolve()
        if not self.checkpoint_path.is_file() or not self.config_path.is_file():
            raise ValueError("Official SAFE checkpoint and config must both exist")
        if str(self.safe_repo) not in sys.path:
            sys.path.insert(0, str(self.safe_repo))
        try:
            import torch as official_torch
            from omegaconf import OmegaConf

            from failure_prob.data.utils import process_tensor_idx_rel
            from failure_prob.model import get_model
        except ImportError as exc:
            raise RuntimeError(
                "Activate the SAFE environment before loading an official SAFE "
                "checkpoint"
            ) from exc

        self.torch = official_torch
        self.process_tensor_idx_rel = process_tensor_idx_rel
        self.get_model = get_model
        self.cfg = OmegaConf.load(self.config_path)
        self.device = str(device)
        try:
            self._checkpoint_state = official_torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            self._checkpoint_state = official_torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )
        self.model = None
        self.input_dim = None

        self.calibration_path = (
            Path(calibration).expanduser().resolve()
            if calibration is not None
            else None
        )
        self.calibration = (
            load_calibration(self.calibration_path)
            if self.calibration_path is not None
            else None
        )
        if self.calibration is not None and "threshold" not in self.calibration:
            raise ValueError("SAFE calibration is missing 'threshold'")
        self.fixed_threshold = (
            float(fixed_threshold) if fixed_threshold is not None else None
        )
        if self.fixed_threshold is not None and not np.isfinite(self.fixed_threshold):
            raise ValueError("SAFE fixed threshold must be finite")

        self.task_normalization_path = (
            Path(task_normalization).expanduser().resolve()
            if task_normalization is not None
            else None
        )
        self.task_normalization = None
        if self.task_normalization_path is not None:
            self.task_normalization = json.loads(
                self.task_normalization_path.read_text()
            )
            if not isinstance(self.task_normalization, dict):
                raise ValueError("SAFE task normalization must be a JSON object")
        self.reset()

    def reset(self, task_name=None):
        """Start a fresh RLDX rollout and bind its seen-task normalization."""
        if self.task_normalization is not None:
            if not task_name:
                raise ValueError(
                    "A task name is required with task-normalized SAFE scores"
                )
            if task_name not in self.task_normalization:
                raise ValueError(
                    f"No SAFE normalization statistics exist for task {task_name!r}"
                )
        self.task_name = task_name
        self._features = []
        self._decisions = []

    @property
    def decisions(self):
        return list(self._decisions)

    def describe(self):
        model_name = getattr(self.cfg.model, "name", None)
        if model_name is None:
            model_name = getattr(self.cfg.model, "model_type", None)
        return {
            "type": "official_safe_checkpoint_monitor",
            "safe_repo": str(self.safe_repo),
            "safe_commit": self.safe_commit,
            "checkpoint": str(self.checkpoint_path),
            "config": str(self.config_path),
            "device": self.device,
            "model": str(model_name),
            "horizon_selector": str(self.cfg.dataset.horizon_idx_rel),
            "diffusion_selector": str(self.cfg.dataset.diff_idx_rel),
            "calibration": (
                str(self.calibration_path)
                if self.calibration_path is not None
                else None
            ),
            "fixed_threshold": self.fixed_threshold,
            "task_normalization": (
                str(self.task_normalization_path)
                if self.task_normalization_path is not None
                else None
            ),
            "task_name": self.task_name,
        }

    def _validate_record(self, record):
        if not isinstance(record, dict) or "features" not in record:
            raise ValueError("Official SAFE requires an RLDX raw feature record")
        raw = np.asarray(record["features"])
        if raw.ndim == 4:
            if raw.shape[0] != 1:
                raise ValueError("Official SAFE runtime supports one RLDX env")
            raw = raw[0]
        if raw.ndim != 3:
            raise ValueError(
                "RLDX SAFE features must be "
                f"(denoising_steps, action_horizon, feature_dim), got {raw.shape}"
            )
        if raw.dtype != np.float32:
            raise ValueError(f"RLDX SAFE features must be float32, got {raw.dtype}")
        if not np.all(np.isfinite(raw)) or not np.any(raw):
            raise ValueError("RLDX SAFE features must be finite and non-zero")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("RLDX SAFE feature metadata is required")
        if metadata.get("model_family") != "rldx1":
            raise ValueError("SAFE runtime record must come from RLDX-1")
        if metadata.get("feature_layer") != RLDX_SAFE_FEATURE_LAYER:
            raise ValueError(
                "RLDX SAFE runtime record uses the wrong hidden-state layer"
            )
        aggregation = metadata.get(
            "feature_aggregation", metadata.get("aggregation")
        )
        if aggregation != "raw":
            raise ValueError("RLDX SAFE runtime features must preserve raw axes")
        return raw

    def _select_feature(self, raw):
        selected = self.process_tensor_idx_rel(
            raw, self.cfg.dataset.horizon_idx_rel
        )
        selected = self.process_tensor_idx_rel(
            selected, self.cfg.dataset.diff_idx_rel
        )
        feature = np.asarray(selected, dtype=np.float32)
        if feature.ndim != 1 or not np.all(np.isfinite(feature)):
            raise ValueError(
                "Official SAFE selectors must produce one finite feature vector; "
                f"got {feature.shape}"
            )
        return feature

    def _ensure_model(self, input_dim):
        input_dim = int(input_dim)
        if self.model is not None:
            if self.input_dim != input_dim:
                raise ValueError(
                    "RLDX SAFE selected feature dimension changed within rollout"
                )
            return
        self.model = self.get_model(self.cfg, input_dim)
        self.model.load_state_dict(self._checkpoint_state, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.input_dim = input_dim

    def observe_inference(self, record):
        """Score one causal RLDX inference prefix before its actions execute."""
        raw = self._validate_record(record)
        feature = self._select_feature(raw)
        self._ensure_model(feature.shape[0])
        self._features.append(feature)
        sequence = np.stack(self._features).astype(np.float32, copy=False)
        batch = {
            "features": self.torch.from_numpy(sequence[None]).to(self.device)
        }
        with self.torch.no_grad():
            scores = self.model(batch)
        raw_score = float(scores[0, len(sequence) - 1, 0].detach().cpu().item())
        score = raw_score
        normalization = None
        if self.task_normalization is not None:
            normalization = self.task_normalization[self.task_name]
            location = float(normalization["location"])
            scale = float(normalization["scale"])
            if not np.isfinite(location) or not np.isfinite(scale) or scale <= 0:
                raise ValueError("SAFE task normalization is invalid")
            score = (raw_score - location) / scale
        threshold = (
            float(threshold_for_length(self.calibration, len(sequence))[-1])
            if self.calibration is not None
            else float(self.fixed_threshold)
        )
        if not np.isfinite(raw_score) or not np.isfinite(score) or not np.isfinite(threshold):
            raise RuntimeError("Official SAFE produced a non-finite decision")
        decision = {
            "schema_version": 1,
            "inference_index": int(
                record.get("inference_index", len(sequence) - 1)
            ),
            "environment_step": int(
                record.get("environment_step", record.get("env_step", 0))
            ),
            "trajectory_length": int(len(sequence)),
            "raw_score": raw_score,
            "score": float(score),
            "threshold": threshold,
            "failure_detected": bool(score >= threshold),
            "task_name": self.task_name,
            "task_normalized": normalization is not None,
        }
        self._decisions.append(decision)
        return dict(decision)
