from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .metrics import binary_metrics


def l2_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, np.finfo(np.float32).eps)


def prototype_scores(
    support_embeddings: np.ndarray,
    support_labels: Iterable[int],
    query_embeddings: np.ndarray,
) -> np.ndarray:
    support = l2_normalize(support_embeddings)
    query = l2_normalize(query_embeddings)
    labels = np.asarray(list(support_labels), dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Prototype classification requires both binary classes")
    prototypes = np.stack([support[labels == label].mean(axis=0) for label in (0, 1)])
    prototypes = l2_normalize(prototypes)
    logits = query @ prototypes.T
    return logits[:, 1] - logits[:, 0]


def fit_parametric_head(
    kind: str,
    support_embeddings: np.ndarray,
    support_labels: Iterable[int],
    seed: int = 42,
    svm_backend: str = "sklearn",
):
    support = np.asarray(support_embeddings, dtype=np.float32)
    labels = np.asarray(list(support_labels), dtype=np.int64)
    if kind == "svm":
        if svm_backend not in {"sklearn", "cuml", "auto"}:
            raise ValueError(f"Unsupported SVM backend: {svm_backend}")
        if svm_backend in {"cuml", "auto"}:
            try:
                from cuml.svm import SVC as CuMLSVC
            except ImportError as exc:
                if svm_backend == "cuml":
                    raise RuntimeError(
                        "cuML SVM requested but cuml is unavailable; install `.[gpu-svm]`"
                    ) from exc
            else:
                variance = max(float(support.var()), np.finfo(np.float32).eps)
                gamma = 1.0 / (float(support.shape[1]) * variance)
                model = CuMLSVC(
                    C=1.0,
                    kernel="rbf",
                    gamma=gamma,
                    probability=False,
                    output_type="numpy",
                )
                return model.fit(support, labels)
        model = SVC(C=1.0, kernel="rbf", gamma="scale", probability=False, random_state=seed)
    elif kind == "linear":
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"Unsupported parametric head: {kind}")
    return model.fit(support, labels)


def parametric_scores(model, query_embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embeddings, dtype=np.float32)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(query)
        if hasattr(scores, "get"):
            scores = scores.get()
        return np.asarray(scores, dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(query), dtype=np.float64)
    return probabilities[:, 1]


def legacy_knn_probabilities(
    similarities: np.ndarray,
    neighbor_labels: np.ndarray,
    temperature: float = 0.05,
) -> np.ndarray:
    """Vectorized implementation of the paper experiment's cumulative kNN rule.

    Returns one score column for every k from 1 through K.
    """
    similarities = np.asarray(similarities, dtype=np.float32)
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    if similarities.shape != labels.shape or similarities.ndim != 2:
        raise ValueError("similarities and neighbor_labels must both have shape [N, K]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("neighbor labels must be binary")

    weighted = similarities * float(temperature)
    class_zero = np.cumsum(weighted * (labels == 0), axis=1)
    class_one = np.cumsum(weighted * (labels == 1), axis=1)
    delta = np.clip(class_zero - class_one, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(delta))


def exponential_knn_probabilities(
    similarities: np.ndarray,
    neighbor_labels: np.ndarray,
    temperature: float = 0.07,
) -> np.ndarray:
    similarities = np.asarray(similarities, dtype=np.float32)
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    weights = np.exp(np.clip(similarities / float(temperature), -80.0, 80.0))
    positive = np.cumsum(weights * (labels == 1), axis=1)
    total = np.cumsum(weights, axis=1)
    return positive / np.maximum(total, np.finfo(np.float32).eps)


@dataclass(frozen=True)
class KNNSelection:
    k: int
    metrics: dict[str, float]


def select_best_k(
    labels: Iterable[int], probability_by_k: np.ndarray, target_fpr: float = 0.05
) -> KNNSelection:
    labels = np.asarray(list(labels), dtype=np.int64)
    best: KNNSelection | None = None
    for column in range(probability_by_k.shape[1]):
        metrics = binary_metrics(labels, probability_by_k[:, column], target_fpr=target_fpr)
        candidate = KNNSelection(k=column + 1, metrics=metrics)
        if best is None or candidate.metrics["auroc"] > best.metrics["auroc"]:
            best = candidate
    assert best is not None
    return best
