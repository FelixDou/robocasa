"""Success-only offline failure-detection baselines for frozen SAFE features.

Every scorer maps one rollout's causal per-inference inputs to one raw score
per genuine policy inference; the score at inference ``t`` reads only inputs
available through ``t``.  The embedding scorers follow the ``use_success_only``
variants of the official SAFE embedding baselines (Mahalanobis, cosine k-NN,
PCA-KMeans) plus a feature-space random network distillation (RND) scorer.
They are fitted on successful development rollouts only.  The action scorer is
the official SAFE ``compute_stac_metrics_single`` temporal-consistency metric.

The module is independent of RoboCasa, PyTorch, and the official SAFE checkout
so the same code can run in tests, on login nodes, and inside the SAFE
environment.
"""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


BASELINES = (
    "mahalanobis",
    "knn_cosine",
    "pca_kmeans",
    "rnd",
    "action_inconsistency",
)
AGGREGATIONS = ("running_max", "running_mean", "cumulative_sum")
DEFAULT_SELECTION_LANDMARKS = (0.10, 0.25)

DEFAULT_GRIDS = {
    "mahalanobis": [{"pca_dim": value} for value in (16, 64, 256)],
    "knn_cosine": [
        {"k": k, "scope": scope}
        for scope in ("per_task", "pooled")
        for k in (1, 5, 10, 25)
    ],
    "pca_kmeans": [
        {"pca_dim": pca_dim, "clusters": clusters, "scope": scope}
        for scope in ("per_task", "pooled")
        for pca_dim in (16, 64)
        for clusters in (8, 32)
    ],
    "rnd": [
        {"hidden_dim": 128, "output_dim": 32},
        {"hidden_dim": 512, "output_dim": 128},
    ],
    "action_inconsistency": [{}],
}
SMOKE_GRIDS = {
    "mahalanobis": [{"pca_dim": 16}],
    "knn_cosine": [{"k": 5, "scope": "per_task"}],
    "pca_kmeans": [{"pca_dim": 16, "clusters": 8, "scope": "per_task"}],
    "rnd": [{"hidden_dim": 64, "output_dim": 16, "max_iter": 5}],
    "action_inconsistency": [{}],
}


def validate_baselines(names):
    names = tuple(dict.fromkeys(str(name) for name in names))
    unknown = sorted(set(names) - set(BASELINES))
    if unknown or not names:
        raise ValueError(f"Unknown or empty baseline selection: {unknown}")
    return names


def validate_aggregations(names):
    names = tuple(dict.fromkeys(str(name) for name in names))
    unknown = sorted(set(names) - set(AGGREGATIONS))
    if unknown or not names:
        raise ValueError(f"Unknown or empty aggregation selection: {unknown}")
    return names


def stac_single(actions, exec_horizon):
    """Official SAFE STAC-Single for one rollout of predicted action chunks.

    ``actions`` has shape ``(inferences, pred_horizon, action_dim)``.  The part
    of chunk ``t-1`` that remains after executing ``exec_horizon`` actions is
    compared with the matching prefix of chunk ``t``; the score is the mean
    per-position Euclidean distance, and the first inference scores zero.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 3:
        raise ValueError("Action chunks must be (inferences, pred_horizon, action_dim)")
    exec_horizon = int(exec_horizon)
    if exec_horizon < 1:
        raise ValueError("exec_horizon must be positive")
    if actions.shape[1] <= exec_horizon:
        raise ValueError(
            f"STAC-Single requires pred_horizon {actions.shape[1]} > "
            f"exec_horizon {exec_horizon}"
        )
    if len(actions) < 2:
        return np.zeros(len(actions), dtype=np.float64)
    old = actions[:-1, exec_horizon:]
    new = actions[1:, :-exec_horizon]
    distances = np.linalg.norm(old - new, axis=-1).mean(axis=-1)
    return np.concatenate([[0.0], distances])


def aggregate_scores(values, aggregation):
    """Return a causal, non-decreasing detector trajectory."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Raw baseline scores must be a finite 1-D trajectory")
    if aggregation == "running_max":
        aggregated = values
    elif aggregation == "running_mean":
        aggregated = np.cumsum(values) / np.arange(1, len(values) + 1)
    elif aggregation == "cumulative_sum":
        aggregated = np.cumsum(values)
    else:
        raise ValueError(f"Unknown aggregation {aggregation!r}")
    # A first-crossing alarm is unchanged by the running maximum, and SAFE's
    # frozen ensemble uses the same monotone alarm-state construction.
    return np.maximum.accumulate(aggregated)


def normalize_trajectory(trajectory, stats):
    return (np.asarray(trajectory, dtype=np.float64) - float(stats["location"])) / float(
        stats["scale"]
    )


def stratified_meta_split(rows, validation_fraction=0.3, seed=0):
    """Split development rollouts by task and outcome before any fitting.

    ``rows`` are dictionaries with ``rollout_id``, ``task_name`` and ``failed``.
    Every task/outcome group keeps at least one validation rollout; every task
    keeps at least two successful rollouts for success-only reference fitting.
    """
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["task_name"]), bool(row["failed"]))].append(
            str(row["rollout_id"])
        )
    rng = np.random.default_rng(int(seed))
    fit_ids, validation_ids = set(), set()
    counts = {}
    for (task, failed), ids in sorted(groups.items()):
        ids = sorted(ids)
        count = max(1, int(round(float(validation_fraction) * len(ids))))
        minimum_fit = 0 if failed else 2
        if len(ids) - count < minimum_fit:
            raise ValueError(
                f"{task} outcome={int(failed)} has {len(ids)} development "
                f"rollouts; cannot keep {minimum_fit} for success-only fitting"
            )
        order = rng.permutation(len(ids))
        selected = {ids[int(index)] for index in order[:count]}
        validation_ids.update(selected)
        fit_ids.update(set(ids) - selected)
        counts.setdefault(task, {})["failure" if failed else "success"] = {
            "fit": len(ids) - len(selected),
            "validation": len(selected),
        }
    tasks = {task for task, _ in groups}
    for task in tasks:
        if not groups.get((task, False)) or not groups.get((task, True)):
            raise ValueError(f"{task} lacks one development outcome")
    if fit_ids & validation_ids:
        raise AssertionError("Fit and validation rollout IDs overlap")
    return {"fit": fit_ids, "validation": validation_ids, "counts": counts}


def _subsample_rows(matrix, limit, seed):
    if limit is None or len(matrix) <= int(limit):
        return np.asarray(matrix, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    rows = np.sort(rng.choice(len(matrix), size=int(limit), replace=False))
    return np.asarray(matrix[rows], dtype=np.float32)


class _Standardizer:
    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        self.mean = matrix.mean(axis=0).astype(np.float32)
        self.scale = np.maximum(matrix.std(axis=0), 1e-6).astype(np.float32)

    def __call__(self, matrix):
        return (np.asarray(matrix, dtype=np.float32) - self.mean) / self.scale


def _pooled(task_features):
    return np.concatenate([task_features[task] for task in sorted(task_features)])


def _fit_pca(standardized, dim, seed):
    from sklearn.decomposition import PCA

    dim = int(min(int(dim), standardized.shape[0] - 1, standardized.shape[1]))
    if dim < 1:
        raise ValueError("PCA requires at least two fit rows")
    return PCA(n_components=dim, svd_solver="randomized", random_state=int(seed)).fit(
        standardized
    )


class BaselineScorer:
    """A fitted success-only scorer; ``score`` returns raw per-inference values."""

    def __init__(self, name, params, seed=0):
        if name not in BASELINES:
            raise ValueError(f"Unknown baseline {name!r}")
        self.name = name
        self.params = dict(params)
        self.seed = int(seed)
        self.tasks = []
        self.exec_horizon = None
        self.fit_summary = {}

    # -- fitting -----------------------------------------------------------
    def fit(self, task_features, *, exec_horizon=None, max_fit_rows=50000):
        """Fit from ``{task: (rows, feature_dim)}`` successful-inference matrices."""
        self.tasks = sorted(task_features)
        if not self.tasks:
            raise ValueError("Baseline fitting requires at least one task")
        self.fit_summary = {
            "fit_rows_per_task": {
                task: int(len(task_features[task])) for task in self.tasks
            }
        }
        if self.name == "action_inconsistency":
            if exec_horizon is None:
                raise ValueError("Action inconsistency requires exec_horizon")
            self.exec_horizon = int(exec_horizon)
            return self
        pooled = _pooled(task_features)
        sample = _subsample_rows(pooled, max_fit_rows, self.seed)
        self.fit_summary["pooled_fit_rows"] = int(len(pooled))
        self.fit_summary["sampled_fit_rows"] = int(len(sample))
        if self.name == "knn_cosine":
            self._fit_knn(task_features, pooled)
        elif self.name == "mahalanobis":
            self._fit_mahalanobis(task_features, sample)
        elif self.name == "pca_kmeans":
            self._fit_pca_kmeans(task_features, sample)
        elif self.name == "rnd":
            self._fit_rnd(sample)
        return self

    def _fit_knn(self, task_features, pooled):
        k = int(self.params["k"])
        scope = self.params.get("scope", "per_task")

        def unit(matrix):
            matrix = np.asarray(matrix, dtype=np.float32)
            norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            return matrix / norms

        if scope == "pooled":
            reference = _subsample_rows(
                pooled, self.params.get("max_reference_rows", 100000), self.seed
            )
            self.references = {"__pooled__": unit(reference)}
        elif scope == "per_task":
            self.references = {
                task: unit(task_features[task]) for task in self.tasks
            }
        else:
            raise ValueError(f"Unknown k-NN scope {scope!r}")
        for key, reference in self.references.items():
            if len(reference) < k:
                raise ValueError(f"k-NN reference {key} has {len(reference)} < k={k} rows")

    def _fit_mahalanobis(self, task_features, sample):
        from sklearn.covariance import LedoitWolf

        self.standardizer = _Standardizer(sample)
        self.pca = _fit_pca(self.standardizer(sample), self.params["pca_dim"], self.seed)
        self.gaussians = {}
        scope = self.params.get("scope", "per_task")
        if scope == "pooled":
            # One Gaussian over the pooled fit sample, for tasks without their own successes.
            estimate = LedoitWolf().fit(self.pca.transform(self.standardizer(sample)))
            self.gaussians["__pooled__"] = {
                "mean": estimate.location_.astype(np.float64),
                "precision": estimate.precision_.astype(np.float64),
            }
            return
        if scope != "per_task":
            raise ValueError(f"Unknown Mahalanobis scope {scope!r}")
        for task in self.tasks:
            projected = self.pca.transform(self.standardizer(task_features[task]))
            if len(projected) < 2:
                raise ValueError(f"{task} has fewer than two success fit inferences")
            estimate = LedoitWolf().fit(projected)
            self.gaussians[task] = {
                "mean": estimate.location_.astype(np.float64),
                "precision": estimate.precision_.astype(np.float64),
            }

    def _fit_pca_kmeans(self, task_features, sample):
        from sklearn.cluster import KMeans

        self.standardizer = _Standardizer(sample)
        self.pca = _fit_pca(self.standardizer(sample), self.params["pca_dim"], self.seed)
        scope = self.params.get("scope", "per_task")
        clusters = int(self.params["clusters"])

        def centroids(matrix):
            projected = self.pca.transform(self.standardizer(matrix))
            count = int(min(clusters, len(projected)))
            model = KMeans(n_clusters=count, n_init=4, random_state=self.seed)
            return model.fit(projected).cluster_centers_.astype(np.float64)

        if scope == "pooled":
            self.centroids = {"__pooled__": centroids(sample)}
        elif scope == "per_task":
            self.centroids = {task: centroids(task_features[task]) for task in self.tasks}
        else:
            raise ValueError(f"Unknown PCA-KMeans scope {scope!r}")

    def _rnd_target(self, standardized):
        hidden = np.maximum(standardized @ self.target_w1, 0.0)
        return hidden @ self.target_w2

    def _fit_rnd(self, sample):
        from sklearn.neural_network import MLPRegressor

        hidden_dim = int(self.params["hidden_dim"])
        output_dim = int(self.params["output_dim"])
        self.standardizer = _Standardizer(sample)
        standardized = self.standardizer(sample)
        rng = np.random.default_rng(self.seed)
        width = standardized.shape[1]
        self.target_w1 = (rng.standard_normal((width, hidden_dim)) / math.sqrt(width)).astype(
            np.float32
        )
        self.target_w2 = (
            rng.standard_normal((hidden_dim, output_dim)) / math.sqrt(hidden_dim)
        ).astype(np.float32)
        target = self._rnd_target(standardized)
        self.predictor = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, hidden_dim),
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=int(self.params.get("max_iter", 50)),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=self.seed,
        ).fit(standardized, target)
        self.fit_summary["rnd_iterations"] = int(self.predictor.n_iter_)

    # -- scoring -----------------------------------------------------------
    def requires_actions(self):
        return self.name == "action_inconsistency"

    @property
    def task_agnostic(self):
        """True for a pooled variant (pooled_variant), which may score tasks it was not fitted on.

        RND has no task component, but only its pooled variant is marked, so the frozen
        per-task pipeline keeps refusing unknown tasks.
        """
        return self.name in ("knn_cosine", "pca_kmeans", "mahalanobis", "rnd") and (
            self.params.get("scope", "per_task") == "pooled"
        )

    def score(self, task, features=None, actions=None, *, chunk_rows=2048):
        task = str(task)
        if task not in self.tasks and not self.task_agnostic:
            raise ValueError(f"Baseline {self.name} was not fitted for task {task}")
        if self.name == "action_inconsistency":
            values = stac_single(actions, self.exec_horizon)
        else:
            features = np.asarray(features, dtype=np.float32)
            if features.ndim != 2 or not len(features):
                raise ValueError("Features must be a nonempty (inferences, dim) matrix")
            if self.name == "knn_cosine":
                values = self._score_knn(task, features, chunk_rows)
            elif self.name == "mahalanobis":
                gaussian = self.gaussians["__pooled__" if "__pooled__" in self.gaussians else task]
                centered = self.pca.transform(self.standardizer(features)) - gaussian["mean"]
                squared = np.einsum("ij,jk,ik->i", centered, gaussian["precision"], centered)
                values = np.sqrt(np.maximum(squared, 0.0))
            elif self.name == "pca_kmeans":
                key = "__pooled__" if "__pooled__" in self.centroids else task
                projected = self.pca.transform(self.standardizer(features))
                differences = projected[:, None, :] - self.centroids[key][None, :, :]
                values = np.sqrt(np.min(np.sum(differences**2, axis=-1), axis=1))
            else:
                standardized = self.standardizer(features)
                error = self.predictor.predict(standardized) - self._rnd_target(standardized)
                values = np.linalg.norm(np.asarray(error).reshape(len(features), -1), axis=1)
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Baseline {self.name} produced a non-finite score")
        return values

    def _score_knn(self, task, features, chunk_rows):
        key = "__pooled__" if "__pooled__" in self.references else task
        reference = self.references[key]
        k = int(self.params["k"])
        norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        queries = features / norms
        output = []
        for start in range(0, len(queries), int(chunk_rows)):
            distance = 1.0 - queries[start : start + int(chunk_rows)] @ reference.T
            nearest = np.partition(distance, k - 1, axis=1)[:, :k]
            output.append(nearest.mean(axis=1))
        return np.clip(np.concatenate(output), 0.0, 2.0)


def pooled_variant(name, params):
    """The same tuned configuration with every task-specific reference pooled over tasks.

    Used to score tasks without successes of their own (unseen tasks). RND has no task
    component; its "scope" only marks the model as allowed to score unseen tasks.
    """
    if name not in ("knn_cosine", "pca_kmeans", "mahalanobis", "rnd"):
        raise ValueError(f"{name} has no task-agnostic variant")
    return dict(params, scope="pooled")


def safe_roc_auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    if len(set(labels.tolist())) != 2:
        return None
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def landmark_auc_rows(rows, horizons, score_key, landmarks):
    """Causal at-risk ROC-AUC at fixed task-horizon landmarks.

    Each row needs ``task_name``, ``failed`` and a trajectory under
    ``score_key``.  Only rollouts still active at the landmark inference are
    scored, so completed duration is never used.
    """
    output = []
    for landmark in landmarks:
        selected = []
        for row in rows:
            step = max(1, int(math.ceil(float(landmark) * int(horizons[row["task_name"]]))))
            values = row[score_key]
            if len(values) >= step:
                selected.append((row, float(values[step - 1])))
        pooled = safe_roc_auc(
            [int(row["failed"]) for row, _ in selected],
            [value for _, value in selected],
        )
        task_values = []
        for task in sorted({row["task_name"] for row, _ in selected}):
            subset = [(row, value) for row, value in selected if row["task_name"] == task]
            auc = safe_roc_auc(
                [int(row["failed"]) for row, _ in subset],
                [value for _, value in subset],
            )
            if auc is not None:
                task_values.append(auc)
        output.append(
            {
                "landmark_fraction": float(landmark),
                "pooled_roc_auc": pooled,
                "task_macro_roc_auc": float(np.mean(task_values)) if task_values else None,
                "supported_tasks": len(task_values),
                "at_risk_rollouts": len(selected),
            }
        )
    return output


def event_task_macro_auc(rows, horizons, score_key):
    """Task-macro ROC-AUC of each rollout's maximum score within its horizon."""
    by_task = defaultdict(list)
    for row in rows:
        values = np.asarray(row[score_key], dtype=np.float64)
        limit = min(len(values), int(horizons[row["task_name"]]))
        by_task[row["task_name"]].append((int(row["failed"]), float(np.max(values[:limit]))))
    task_values = []
    for task in sorted(by_task):
        auc = safe_roc_auc([label for label, _ in by_task[task]], [value for _, value in by_task[task]])
        if auc is not None:
            task_values.append(auc)
    return float(np.mean(task_values)) if task_values else None


def selection_key(row):
    """Rank candidates: early landmark ROC, then event ROC, then grid order."""
    primary = row["selection_score"]
    secondary = row["event_task_macro_roc_auc"]
    return (
        -math.inf if primary is None else float(primary),
        -math.inf if secondary is None else float(secondary),
        -int(row["candidate_index"]),
    )
