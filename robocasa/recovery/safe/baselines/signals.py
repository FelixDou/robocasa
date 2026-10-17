"""Causal action-chunk signals used by action-space failure detectors.

All functions take the recorded policy action chunks of one rollout, shaped
``(inferences, action_horizon, action_dim)``, and return one value per
inference that depends only on the current and earlier chunks.
"""

from __future__ import annotations

import numpy as np


def _validate_chunks(chunks):
    chunks = np.asarray(chunks, dtype=np.float64)
    if chunks.ndim != 3 or chunks.shape[0] < 1:
        raise ValueError("action chunks must be (inferences, horizon, action_dim)")
    if not np.all(np.isfinite(chunks)):
        raise ValueError("action chunks contain NaN or infinity")
    return chunks


def select_dims(chunks, dims=None):
    chunks = _validate_chunks(chunks)
    if dims is None:
        return chunks
    dims = [int(dim) for dim in dims]
    if not dims or min(dims) < 0 or max(dims) >= chunks.shape[2]:
        raise ValueError(f"action dims {dims} are outside action_dim={chunks.shape[2]}")
    return chunks[:, :, dims]


def temporal_consistency_error(chunks, replan_steps, *, velocity_normalize=False):
    """Disagreement between overlapping parts of consecutive action chunks.

    Chunk ``t-1`` was predicted ``replan_steps`` actions before chunk ``t``,
    so its unexecuted tail ``chunks[t-1, replan:]`` and the head of the new
    chunk ``chunks[t, :H-replan]`` describe the same future steps. This is
    ActProbe's TCE, VLA-FAIL's ACC and the single-sample form of STAC.
    The first inference has no predecessor and scores 0.
    """
    chunks = _validate_chunks(chunks)
    replan = int(replan_steps)
    horizon = chunks.shape[1]
    if replan < 1:
        raise ValueError("replan_steps must be positive")
    overlap = horizon - replan
    if overlap < 1:
        raise ValueError(
            f"action_horizon={horizon} and replan_steps={replan} leave no "
            "overlap between consecutive chunks"
        )
    errors = np.zeros(chunks.shape[0], dtype=np.float64)
    if chunks.shape[0] == 1:
        return errors
    previous_tail = chunks[:-1, replan:]
    current_head = chunks[1:, :overlap]
    distance = np.linalg.norm(previous_tail - current_head, axis=-1).mean(axis=-1)
    if velocity_normalize:
        speed = np.linalg.norm(np.diff(chunks[1:], axis=1), axis=-1).mean(axis=-1)
        distance = distance / (speed + 1e-6)
    errors[1:] = distance
    return errors


def chunk_magnitude(chunks):
    """Mean action norm over the whole predicted chunk (ActProbe's ACM)."""
    chunks = _validate_chunks(chunks)
    return np.linalg.norm(chunks, axis=-1).mean(axis=-1)


def executed_magnitude(chunks, replan_steps):
    """Mean action norm over the part of the chunk that is executed."""
    chunks = _validate_chunks(chunks)
    replan = max(1, min(int(replan_steps), chunks.shape[1]))
    return np.linalg.norm(chunks[:, :replan], axis=-1).mean(axis=-1)


def exponential_moving_average(values, alpha):
    """Causal EMA where ``alpha`` weights the previous average."""
    values = np.asarray(values, dtype=np.float64)
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("EMA alpha must be in [0, 1)")
    out = np.empty_like(values)
    running = None
    for index, value in enumerate(values):
        running = value if running is None else alpha * running + (1.0 - alpha) * value
        out[index] = running
    return out


def action_signal_matrix(chunks, replan_steps, *, dims=None):
    """Stack TCE, ACM and executed magnitude into ``(inferences, 3)``."""
    chunks = select_dims(chunks, dims)
    return np.stack(
        [
            temporal_consistency_error(chunks, replan_steps),
            chunk_magnitude(chunks),
            executed_magnitude(chunks, replan_steps),
        ],
        axis=1,
    )
