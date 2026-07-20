import numpy as np
import pytest

from fiser.heads import fit_parametric_head, legacy_knn_probabilities, prototype_scores


def test_legacy_knn_probability_shape():
    scores = np.asarray([[0.9, 0.8, 0.1], [0.1, 0.2, 0.8]], dtype=np.float32)
    labels = np.asarray([[1, 1, 0], [0, 0, 1]], dtype=np.int64)
    probabilities = legacy_knn_probabilities(scores, labels)
    assert probabilities.shape == (2, 3)
    assert np.isfinite(probabilities).all()


def test_prototype_scores_rank_positive_class():
    support = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], dtype=np.float32)
    labels = [1, 1, 0, 0]
    queries = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    scores = prototype_scores(support, labels, queries)
    assert scores[0] > scores[1]


def test_unknown_svm_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported SVM backend"):
        fit_parametric_head(
            "svm",
            np.asarray([[0.0], [1.0]], dtype=np.float32),
            [0, 1],
            svm_backend="unknown",
        )
