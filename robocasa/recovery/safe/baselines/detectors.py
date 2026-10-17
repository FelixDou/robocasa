"""Baseline failure detectors that share SAFE's rollout inputs.

A rollout is a dict with ``rollout_id``, ``task_name``, ``failed``,
``features`` (``(inferences, D)`` aggregated VLA features), ``actions``
(``(inferences, horizon, action_dim)`` recorded chunks, or ``None``) and
``replan_steps``. ``fit`` sees training rollouts only; ``score`` returns one
causal score per inference (higher means more likely to fail).

Implemented, with the paper each one follows:

* ``action_consistency``: training-free EMA of chunk disagreement
  (STAC-Single as used by SAFE, ACC from VLA-FAIL).
* ``actprobe``: LSTM on chunk consistency and magnitude, trajectory labels
  (ActProbe, Huang et al. 2026).
* ``mahalanobis``, ``cosine_knn``, ``pca_kmeans``: success-only distance to
  training features (embedding baselines in SAFE; LLMD in VLA-FAIL).
* ``rnd``: success-only random network distillation on features (the OOD
  half of FIPER; also a FAIL-Detect score).
* ``hide_and_seek``: LSTM on features with inter-trajectory peak margin and
  intra-trajectory onset losses (Park et al. 2026).

Sampling-based scores (full STAC, FIPER action entropy, Tri-Info, ACE) need
several action samples per inference and are not recoverable from recorded
rollouts.
"""

from __future__ import annotations

import numpy as np

from .signals import action_signal_matrix, exponential_moving_average, temporal_consistency_error


def _require_actions(rollout, name):
    if rollout.get("actions") is None:
        raise ValueError(
            f"{name} needs recorded action chunks; rollout {rollout['rollout_id']} "
            "was collected without --record-actions"
        )
    return rollout["actions"]


def _successes(rollouts):
    selected = [rollout for rollout in rollouts if not bool(rollout["failed"])]
    if not selected:
        raise ValueError("Success-only detectors need successful training rollouts")
    return selected


def _stack_features(rollouts, max_rows=None, seed=0):
    rows = np.concatenate([np.asarray(r["features"], dtype=np.float64) for r in rollouts])
    if max_rows is not None and len(rows) > int(max_rows):
        rng = np.random.default_rng(seed)
        rows = rows[np.sort(rng.choice(len(rows), int(max_rows), replace=False))]
    return rows


class Detector:
    name = "base"
    uses_failures = False
    uses_actions = False
    uses_features = False

    def __init__(self, seed=0):
        self.seed = int(seed)

    def fit(self, rollouts):
        return self

    def score(self, rollout):
        raise NotImplementedError

    def config(self):
        return {"name": self.name, "seed": self.seed}


class ActionConsistencyDetector(Detector):
    name = "action_consistency"
    uses_actions = True

    def __init__(self, seed=0, ema_alpha=0.9, action_dims=None, velocity_normalize=False):
        super().__init__(seed)
        self.ema_alpha = float(ema_alpha)
        self.action_dims = action_dims
        self.velocity_normalize = bool(velocity_normalize)

    def score(self, rollout):
        chunks = _require_actions(rollout, self.name)
        if self.action_dims is not None:
            chunks = np.asarray(chunks)[:, :, list(self.action_dims)]
        error = temporal_consistency_error(
            chunks, rollout["replan_steps"], velocity_normalize=self.velocity_normalize
        )
        return exponential_moving_average(error, self.ema_alpha)

    def config(self):
        return {
            **super().config(),
            "ema_alpha": self.ema_alpha,
            "action_dims": self.action_dims,
            "velocity_normalize": self.velocity_normalize,
        }


class _ReducedFeatureDetector(Detector):
    """Shared standardization + PCA for success-only feature detectors."""

    uses_features = True

    def __init__(self, seed=0, pca_components=64, max_fit_rows=50000):
        super().__init__(seed)
        self.pca_components = pca_components
        self.max_fit_rows = max_fit_rows

    def _fit_reduction(self, rows):
        from sklearn.decomposition import PCA

        self.mean_ = rows.mean(axis=0)
        self.std_ = rows.std(axis=0) + 1e-6
        standardized = (rows - self.mean_) / self.std_
        components = self.pca_components
        if components is not None:
            components = int(min(components, standardized.shape[0] - 1, standardized.shape[1]))
            self.pca_ = PCA(n_components=components, random_state=self.seed).fit(standardized)
            return self.pca_.transform(standardized)
        self.pca_ = None
        return standardized

    def _reduce(self, features):
        standardized = (np.asarray(features, dtype=np.float64) - self.mean_) / self.std_
        return standardized if self.pca_ is None else self.pca_.transform(standardized)

    def config(self):
        return {
            **super().config(),
            "pca_components": self.pca_components,
            "max_fit_rows": self.max_fit_rows,
        }


class MahalanobisDetector(_ReducedFeatureDetector):
    name = "mahalanobis"

    def fit(self, rollouts):
        from sklearn.covariance import LedoitWolf

        rows = self._fit_reduction(
            _stack_features(_successes(rollouts), self.max_fit_rows, self.seed)
        )
        self.covariance_ = LedoitWolf().fit(rows)
        return self

    def score(self, rollout):
        reduced = self._reduce(rollout["features"])
        return np.sqrt(np.maximum(self.covariance_.mahalanobis(reduced), 0.0))


class CosineKNNDetector(_ReducedFeatureDetector):
    name = "cosine_knn"

    def __init__(self, seed=0, k=5, max_fit_rows=20000):
        super().__init__(seed, None, max_fit_rows)
        self.k = int(k)

    # Raw (uncentered) features, as in SAFE's cosine kNN baseline: centering
    # would turn cosine distance into a direction test around the success mean.
    def fit(self, rollouts):
        rows = _stack_features(_successes(rollouts), self.max_fit_rows, self.seed)
        self.bank_ = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)
        return self

    def score(self, rollout):
        reduced = np.asarray(rollout["features"], dtype=np.float64)
        reduced = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)
        similarity = reduced @ self.bank_.T
        k = min(self.k, similarity.shape[1])
        top = np.partition(similarity, -k, axis=1)[:, -k:]
        return 1.0 - top.mean(axis=1)

    def config(self):
        return {**super().config(), "k": self.k}


class PCAKMeansDetector(_ReducedFeatureDetector):
    name = "pca_kmeans"

    def __init__(self, seed=0, clusters=32, pca_components=64, max_fit_rows=50000):
        super().__init__(seed, pca_components, max_fit_rows)
        self.clusters = int(clusters)

    def fit(self, rollouts):
        from sklearn.cluster import KMeans

        rows = self._fit_reduction(
            _stack_features(_successes(rollouts), self.max_fit_rows, self.seed)
        )
        clusters = min(self.clusters, len(rows))
        self.kmeans_ = KMeans(n_clusters=clusters, n_init=10, random_state=self.seed).fit(rows)
        return self

    def score(self, rollout):
        return self.kmeans_.transform(self._reduce(rollout["features"])).min(axis=1)

    def config(self):
        return {**super().config(), "clusters": self.clusters}


# ----------------------------------------------------------------------------
# Torch detectors
# ----------------------------------------------------------------------------


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError("This baseline needs PyTorch (use the SAFE environment)") from error
    return torch


class _TorchDetector(Detector):
    def __init__(self, seed=0, epochs=200, learning_rate=1e-3, weight_decay=1e-4,
                 hidden_dim=128, batch_size=32, device="cpu"):
        super().__init__(seed)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.device = device

    def config(self):
        return {
            **super().config(),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
        }


def _pad(sequences, torch, device):
    lengths = [len(sequence) for sequence in sequences]
    width = sequences[0].shape[1]
    batch = np.zeros((len(sequences), max(lengths), width), dtype=np.float32)
    mask = np.zeros((len(sequences), max(lengths)), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        batch[index, : len(sequence)] = sequence
        mask[index, : len(sequence)] = 1.0
    return (
        torch.as_tensor(batch, device=device),
        torch.as_tensor(mask, device=device),
        torch.as_tensor(lengths, device=device),
    )


def _lstm_scorer(torch, input_dim, hidden_dim):
    nn = torch.nn

    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))

        def forward(self, x):
            hidden, _ = self.lstm(x)
            return torch.sigmoid(self.head(hidden).squeeze(-1))

    return Scorer()


class _SequenceDetector(_TorchDetector):
    """LSTM over per-inference inputs; subclasses choose inputs and loss."""

    uses_failures = True

    def _inputs(self, rollout):
        raise NotImplementedError

    def _loss(self, torch, scores, mask, labels):
        raise NotImplementedError

    def fit(self, rollouts):
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        raw = [self._inputs(rollout) for rollout in rollouts]
        rows = np.concatenate(raw)
        self.mean_ = rows.mean(axis=0)
        self.std_ = rows.std(axis=0) + 1e-6
        sequences = [((sequence - self.mean_) / self.std_).astype(np.float32) for sequence in raw]
        labels = np.asarray([float(bool(r["failed"])) for r in rollouts], dtype=np.float32)
        if len(set(labels.tolist())) != 2:
            raise ValueError(f"{self.name} needs successful and failed training rollouts")
        self.model_ = _lstm_scorer(torch, sequences[0].shape[1], self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(len(sequences))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                index = order[start : start + self.batch_size]
                batch_labels = torch.as_tensor(labels[index], device=self.device)
                x, mask, _ = _pad([sequences[i] for i in index], torch, self.device)
                loss = self._loss(torch, self.model_(x), mask, batch_labels)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            self.history_.append(total)
        self.model_.eval()
        return self

    def score(self, rollout):
        torch = _torch()
        sequence = ((self._inputs(rollout) - self.mean_) / self.std_).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(sequence[None], device=self.device)
            return self.model_(x)[0].cpu().numpy().astype(np.float64)


class ActProbeDetector(_SequenceDetector):
    """Per-step BCE on action-space signals with trajectory labels."""

    name = "actprobe"
    uses_actions = True

    def __init__(self, seed=0, action_dims=None, tasks=None, **kwargs):
        super().__init__(seed, **kwargs)
        self.action_dims = action_dims
        self.tasks = None if tasks is None else sorted(tasks)

    def fit(self, rollouts):
        if self.tasks is None:
            self.tasks = sorted({str(r["task_name"]) for r in rollouts})
        return super().fit(rollouts)

    def _inputs(self, rollout):
        signals = action_signal_matrix(
            _require_actions(rollout, self.name), rollout["replan_steps"], dims=self.action_dims
        )
        # ActProbe conditions on the task; unseen tasks get an all-zero code.
        task = np.zeros((len(signals), len(self.tasks)))
        if str(rollout["task_name"]) in self.tasks:
            task[:, self.tasks.index(str(rollout["task_name"]))] = 1.0
        return np.concatenate([signals, task], axis=1)

    def _loss(self, torch, scores, mask, labels):
        target = labels[:, None].expand_as(scores)
        loss = torch.nn.functional.binary_cross_entropy(
            scores.clamp(1e-6, 1 - 1e-6), target, reduction="none"
        )
        return (loss * mask).sum() / mask.sum()

    def config(self):
        return {**super().config(), "action_dims": self.action_dims, "tasks": self.tasks}


class HideAndSeekDetector(_SequenceDetector):
    """Trajectory-label LSTM with peak-margin and proxy-onset losses."""

    name = "hide_and_seek"
    uses_features = True

    def __init__(self, seed=0, margin=0.2, intra_weight=1.0, pca_components=128, **kwargs):
        super().__init__(seed, **kwargs)
        self.margin = float(margin)
        self.intra_weight = float(intra_weight)
        self.pca_components = pca_components

    def fit(self, rollouts):
        self.pca_ = None
        if self.pca_components is not None:
            from sklearn.decomposition import PCA

            rows = _stack_features(rollouts, 50000, self.seed)
            components = int(min(self.pca_components, rows.shape[0] - 1, rows.shape[1]))
            self.pca_ = PCA(n_components=components, random_state=self.seed).fit(rows)
        return super().fit(rollouts)

    def _inputs(self, rollout):
        features = np.asarray(rollout["features"], dtype=np.float64)
        return features if self.pca_ is None else self.pca_.transform(features)

    def _loss(self, torch, scores, mask, labels):
        masked = scores.masked_fill(mask == 0, -1.0)
        peaks = masked.max(dim=1).values
        failed = labels > 0.5
        losses = []
        if failed.any() and (~failed).any():
            gap = peaks[failed][:, None] - peaks[~failed][None, :]
            losses.append(torch.relu(self.margin - gap).mean())
        if failed.any() and self.intra_weight > 0:
            intra = []
            for row in torch.nonzero(failed).flatten().tolist():
                length = int(mask[row].sum())
                if length < 2:
                    continue
                trace = scores[row, :length]
                onset = int(torch.argmax(trace[1:] - trace[:-1]).detach()) + 1
                gap = trace[onset:].mean() - trace[:onset].mean()
                intra.append(torch.relu(self.margin - gap))
            if intra:
                losses.append(self.intra_weight * torch.stack(intra).mean())
        return sum(losses) if losses else None

    def config(self):
        return {
            **super().config(),
            "margin": self.margin,
            "intra_weight": self.intra_weight,
            "pca_components": self.pca_components,
        }


class RNDDetector(_TorchDetector):
    """Success-only random network distillation on per-inference features."""

    name = "rnd"
    uses_features = True

    def __init__(self, seed=0, output_dim=64, pca_components=128, **kwargs):
        super().__init__(seed, **kwargs)
        self.output_dim = int(output_dim)
        self.pca_components = pca_components

    def _prepare(self, features):
        standardized = (np.asarray(features, dtype=np.float64) - self.mean_) / self.std_
        reduced = standardized if self.pca_ is None else self.pca_.transform(standardized)
        return reduced.astype(np.float32)

    def fit(self, rollouts):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        rows = _stack_features(_successes(rollouts), 50000, self.seed)
        self.mean_ = rows.mean(axis=0)
        self.std_ = rows.std(axis=0) + 1e-6
        self.pca_ = None
        if self.pca_components is not None:
            from sklearn.decomposition import PCA

            components = int(min(self.pca_components, rows.shape[0] - 1, rows.shape[1]))
            self.pca_ = PCA(n_components=components, random_state=self.seed).fit(
                (rows - self.mean_) / self.std_
            )
        data = torch.as_tensor(self._prepare(rows), device=self.device)
        width = data.shape[1]

        def network():
            return nn.Sequential(nn.Linear(width, self.hidden_dim), nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.output_dim))

        self.target_ = network().to(self.device)
        for parameter in self.target_.parameters():
            parameter.requires_grad_(False)
        self.predictor_ = network().to(self.device)
        optimizer = torch.optim.AdamW(
            self.predictor_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        batch = max(self.batch_size, 256)
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=generator)
            for start in range(0, len(order), batch):
                x = data[order[start : start + batch].to(self.device)]
                loss = ((self.predictor_(x) - self.target_(x)) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.predictor_.eval()
        return self

    def score(self, rollout):
        torch = _torch()
        with torch.no_grad():
            x = torch.as_tensor(self._prepare(rollout["features"]), device=self.device)
            error = ((self.predictor_(x) - self.target_(x)) ** 2).mean(dim=1)
        return error.cpu().numpy().astype(np.float64)

    def config(self):
        return {**super().config(), "output_dim": self.output_dim,
                "pca_components": self.pca_components}


DETECTOR_REGISTRY = {
    detector.name: detector
    for detector in (
        ActionConsistencyDetector,
        ActProbeDetector,
        MahalanobisDetector,
        CosineKNNDetector,
        PCAKMeansDetector,
        RNDDetector,
        HideAndSeekDetector,
    )
}


def build_detector(name, **kwargs):
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown baseline {name!r}; choose from {sorted(DETECTOR_REGISTRY)}")
    return DETECTOR_REGISTRY[name](**kwargs)
