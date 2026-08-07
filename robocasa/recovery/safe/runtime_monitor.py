"""Online SAFE scoring for policy-inference records.

The policy adapters expose one raw feature record per genuine model inference
through ``pop_inference_record()``.  This module turns those records into an
online failure decision without scoring cached actions more than once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .conformal import load_calibration, threshold_for_length
from .dataset import aggregate_features
from .models import load_checkpoint, require_torch, torch


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

    def reset(self):
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
