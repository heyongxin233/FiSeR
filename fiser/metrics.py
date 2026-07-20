from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors


def tpr_at_fpr(
    labels: Iterable[int], scores: Iterable[float], target_fpr: float = 0.05
) -> tuple[float, float]:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels_array, scores_array, pos_label=1)
    # Match the released experiment code: select the discrete ROC operating
    # point nearest to the requested FPR (rather than interpolating or taking
    # only points on the <= side).
    index = int(np.argmin(np.abs(fpr - float(target_fpr))))
    return float(tpr[index]), float(thresholds[index])


def binary_metrics(
    labels: Iterable[int], scores: Iterable[float], target_fpr: float = 0.05
) -> dict[str, float | None]:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have the same shape")
    if np.unique(labels_array).size != 2:
        raise ValueError("binary metrics require both natural and synthetic samples")
    tpr, threshold = tpr_at_fpr(labels_array, scores_array, target_fpr)
    return {
        "auroc": float(roc_auc_score(labels_array, scores_array)),
        "average_precision": float(average_precision_score(labels_array, scores_array)),
        "tpr_at_fpr": tpr,
        "tpr_at_fpr_threshold": threshold if np.isfinite(threshold) else None,
    }


def knn_graph_homophily(
    embeddings: np.ndarray,
    source_labels: Iterable[int],
    k: int = 10,
    metric: str = "cosine",
) -> dict[str, float]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(list(source_labels))
    if embeddings.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and source_labels have incompatible shapes")
    if not 0 < k < embeddings.shape[0]:
        raise ValueError("k must be between 1 and n_samples - 1")

    neighbors = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(embeddings)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    observed = float((labels[indices] == labels[:, None]).mean())

    _, counts = np.unique(labels, return_counts=True)
    n = int(labels.shape[0])
    null = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
    normalized = float((observed - null) / max(1.0 - null, np.finfo(float).eps))
    return {"homophily": observed, "null_homophily": null, "normalized_homophily": normalized}
