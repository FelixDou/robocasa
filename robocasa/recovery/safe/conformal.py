"""Functional conformal upper bands adapted from the official SAFE source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resample_score_trajectory(scores, length=100):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Score trajectory must be a finite non-empty vector")
    if len(scores) == 1:
        return np.repeat(scores, length)
    return np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, len(scores)),
        scores,
    )


def _functional_modulation(reference, mean, alpha, kind):
    eps = 1e-8
    if kind == "constant":
        return np.ones_like(mean) / mean.shape[-1]
    if kind == "stdev":
        ddof = 1 if len(reference) > 1 else 0
        return np.std(reference, axis=0, ddof=ddof, keepdims=True) + eps
    if kind != "tfunc":
        raise ValueError(f"Unknown functional modulation {kind!r}")
    deviations = np.abs(reference - mean)
    rank = int(np.ceil((len(reference) + 1) * (1 - alpha))) - 1
    if rank >= len(reference):
        return np.max(deviations, axis=0, keepdims=True) + eps
    gamma = np.sort(np.max(deviations, axis=1))[rank]
    return np.max(deviations[np.max(deviations, axis=1) <= gamma], axis=0, keepdims=True) + eps


def calibrate_functional_threshold(
    reference_success_scores,
    calibration_success_scores,
    *,
    alpha=0.2,
    normalized_length=100,
    modulation="tfunc",
):
    """Fit SAFE's one-sided functional upper prediction band.

    Failure is detected at the first inference where score is strictly greater
    than the corresponding upper threshold.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not reference_success_scores or not calibration_success_scores:
        raise ValueError("Reference and calibration successes must both be non-empty")
    reference = np.stack(
        [resample_score_trajectory(x, normalized_length) for x in reference_success_scores]
    )
    calibration = np.stack(
        [resample_score_trajectory(x, normalized_length) for x in calibration_success_scores]
    )
    mean = np.mean(reference, axis=0, keepdims=True)
    scale = _functional_modulation(reference, mean, alpha, modulation)
    nonconformity = np.max((calibration - mean) / scale, axis=1)
    # Finite-sample split-conformal quantile; clamp when alpha is too small for n.
    rank = min(
        len(nonconformity) - 1,
        int(np.ceil((len(nonconformity) + 1) * (1 - alpha))) - 1,
    )
    width = np.sort(nonconformity)[rank]
    threshold = (mean + width * scale).reshape(-1)
    return {
        "schema_version": 1,
        "method": "safe_functional_one_sided_upper",
        "alpha": float(alpha),
        "normalized_length": int(normalized_length),
        "modulation": modulation,
        "reference_size": int(len(reference)),
        "calibration_size": int(len(calibration)),
        "threshold": threshold.tolist(),
        "crossing_rule": "first score strictly greater than normalized-time threshold",
    }


def threshold_for_length(calibration, length):
    return resample_score_trajectory(calibration["threshold"], length)


def first_detection(scores, calibration):
    scores = np.asarray(scores, dtype=float)
    crossings = np.flatnonzero(scores > threshold_for_length(calibration, len(scores)))
    return None if not len(crossings) else int(crossings[0])


def save_calibration(calibration, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    return path


def load_calibration(path: str | Path):
    return json.loads(Path(path).read_text())


def build_parser():
    parser = argparse.ArgumentParser(description="Calibrate a raw SAFE functional threshold")
    parser.add_argument("--scores", required=True, help="Training output score JSON")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--normalized-length", type=int, default=100)
    parser.add_argument("--modulation", choices=["tfunc", "stdev", "constant"], default="tfunc")
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data = json.loads(Path(args.scores).read_text())
    reference = [x["scores"] for x in data if x["split"] == "train" and not x["failed"]]
    calibration = [
        x["scores"] for x in data if x["split"] == "calibration" and not x["failed"]
    ]
    result = calibrate_functional_threshold(
        reference,
        calibration,
        alpha=args.alpha,
        normalized_length=args.normalized_length,
        modulation=args.modulation,
    )
    save_calibration(result, args.output)


if __name__ == "__main__":
    main()
