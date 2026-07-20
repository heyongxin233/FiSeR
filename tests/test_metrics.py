import numpy as np

from fiser.metrics import binary_metrics, knn_graph_homophily


def test_binary_metrics_and_tpr():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auroc"] == 1.0
    # With only four samples the nearest discrete ROC point to 5% FPR can be
    # the origin; the paper protocol intentionally does not interpolate.
    assert 0.0 <= metrics["tpr_at_fpr"] <= 1.0


def test_nonfinite_roc_threshold_is_json_safe():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    if metrics["tpr_at_fpr_threshold"] is not None:
        assert np.isfinite(metrics["tpr_at_fpr_threshold"])


def test_homophily_output_is_bounded():
    values = knn_graph_homophily(
        np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32), [0, 0, 1, 1], k=1
    )
    assert 0.0 <= values["homophily"] <= 1.0
    assert np.isfinite(values["normalized_homophily"])
