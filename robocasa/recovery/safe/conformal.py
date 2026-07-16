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


def _align_score_trajectories(reference, calibration, alignment, normalized_length):
    trajectories = [np.asarray(x, dtype=np.float64) for x in reference + calibration]
    if any(x.ndim != 1 or len(x) < 1 or not np.all(np.isfinite(x)) for x in trajectories):
        raise ValueError("Score trajectories must be finite non-empty vectors")
    if alignment == "extend":
        length = max(len(x) for x in trajectories)
        aligned = [np.pad(x, (0, length - len(x)), mode="edge") for x in trajectories]
    elif alignment == "normalized_time":
        if normalized_length is None or normalized_length < 1:
            raise ValueError("normalized_time alignment requires a positive normalized_length")
        length = int(normalized_length)
        aligned = [resample_score_trajectory(x, length) for x in trajectories]
    else:
        raise ValueError(f"Unknown SAFE trajectory alignment {alignment!r}")
    split = len(reference)
    return np.stack(aligned[:split]), np.stack(aligned[split:]), length


def calibrate_functional_threshold(
    reference_success_scores,
    calibration_success_scores,
    *,
    alpha=0.2,
    normalized_length=None,
    modulation="tfunc",
    alignment="extend",
):
    """Fit the official SAFE one-sided functional upper prediction band.

    ``alignment="extend"`` and the ``>=`` crossing rule match the official SAFE
    implementation. ``normalized_time`` is retained as an explicit
    rollout-length-leakage sensitivity analysis.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not reference_success_scores or not calibration_success_scores:
        raise ValueError("Reference and calibration successes must both be non-empty")
    reference, calibration, aligned_length = _align_score_trajectories(
        list(reference_success_scores),
        list(calibration_success_scores),
        alignment,
        normalized_length,
    )
    mean = np.mean(reference, axis=0, keepdims=True)
    scale = _functional_modulation(reference, mean, alpha, modulation)
    nonconformity = np.max((calibration - mean) / scale, axis=1)
    # This deliberately matches official SAFE's one-sided implementation.
    width = np.quantile(nonconformity, 1 - alpha)
    threshold = (mean + width * scale).reshape(-1)
    return {
        "schema_version": 1,
        "method": "official_safe_functional_one_sided_upper",
        "alpha": float(alpha),
        "alignment": alignment,
        "aligned_length": int(aligned_length),
        "normalized_length": (
            int(normalized_length) if normalized_length is not None else None
        ),
        "modulation": modulation,
        "reference_size": int(len(reference)),
        "calibration_size": int(len(calibration)),
        "threshold": threshold.tolist(),
        "crossing_rule": "first score greater than or equal to the aligned-time threshold",
    }


def threshold_for_length(calibration, length):
    threshold = np.asarray(calibration["threshold"], dtype=np.float64)
    alignment = calibration.get("alignment", "normalized_time")
    if alignment == "normalized_time":
        return resample_score_trajectory(threshold, length)
    if alignment == "extend":
        if length <= len(threshold):
            return threshold[:length]
        return np.pad(threshold, (0, length - len(threshold)), mode="edge")
    raise ValueError(f"Unknown SAFE trajectory alignment {alignment!r}")


def first_detection(scores, calibration):
    scores = np.asarray(scores, dtype=float)
    crossings = np.flatnonzero(scores >= threshold_for_length(calibration, len(scores)))
    return None if not len(crossings) else int(crossings[0])


def save_calibration(calibration, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    return path


def load_calibration(path: str | Path):
    return json.loads(Path(path).read_text())


def select_protocol_trajectories(data, protocol, seed=0):
    held_out_successes = [
        x["scores"] for x in data if x["split"] == "calibration" and not x["failed"]
    ]
    if protocol == "official_safe":
        if len(held_out_successes) < 2:
            raise ValueError("Official SAFE calibration needs at least two held-out successes")
        order = np.random.RandomState(seed).permutation(len(held_out_successes))
        shuffled = [held_out_successes[index] for index in order]
        reference_size = int(len(shuffled) * 0.3)
        if reference_size < 1:
            raise ValueError("Official SAFE's 30/70 split produced an empty reference set")
        return shuffled[:reference_size], shuffled[reference_size:], "extend"
    if protocol == "normalized_time_sensitivity":
        reference = [
            x["scores"] for x in data if x["split"] == "train" and not x["failed"]
        ]
        return reference, held_out_successes, "normalized_time"
    raise ValueError(f"Unknown SAFE calibration protocol {protocol!r}")


def build_parser():
    parser = argparse.ArgumentParser(description="Calibrate a raw SAFE functional threshold")
    parser.add_argument("--scores", required=True, help="Training output score JSON")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--protocol",
        choices=["official_safe", "normalized_time_sensitivity"],
        default="official_safe",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalized-length", type=int, default=100)
    parser.add_argument("--modulation", choices=["tfunc", "stdev", "constant"], default="tfunc")
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data = json.loads(Path(args.scores).read_text())
    reference, calibration, alignment = select_protocol_trajectories(
        data, args.protocol, args.seed
    )
    result = calibrate_functional_threshold(
        reference,
        calibration,
        alpha=args.alpha,
        normalized_length=(
            args.normalized_length if alignment == "normalized_time" else None
        ),
        modulation=args.modulation,
        alignment=alignment,
    )
    result.update(
        {
            "protocol": args.protocol,
            "seed": args.seed,
            "source_score_file": str(Path(args.scores).resolve()),
        }
    )
    save_calibration(result, args.output)


if __name__ == "__main__":
    main()
