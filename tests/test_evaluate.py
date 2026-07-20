import numpy as np
import pytest
import torch

import evaluate


def _numpy_inner_product_search(database, queries, max_k, **_kwargs):
    database = np.asarray(database, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    similarities = queries @ database.T
    indices = np.argsort(-similarities, axis=1)[:, :max_k]
    return np.take_along_axis(similarities, indices, axis=1), indices


def test_fewshot_knn_splits_support_and_runs_requested_layer(monkeypatch):
    embeddings = torch.tensor(
        [
            [-1.0, 0.0],
            [-0.9, 0.1],
            [-0.8, -0.2],
            [-0.7, 0.2],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, -0.2],
            [0.7, 0.2],
        ],
        dtype=torch.float32,
    )
    archive = {
        "ids": torch.arange(8),
        "labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "binary_labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "embeddings": {18: embeddings, 19: embeddings.clone()},
    }
    monkeypatch.setattr(evaluate, "_search_gpu", _numpy_inner_product_search)

    result = evaluate.evaluate_fewshot(
        archive,
        head="knn",
        shots_per_source=None,
        shots_per_class=1,
        seed=42,
        target_fpr=0.05,
        requested_layers=[19],
        devices=[0, 1],
        max_k=2,
    )

    assert result["head"] == "knn"
    assert result["support_size"] == 2
    assert result["query_size"] == 6
    assert result["best"]["layer"] == 19
    assert 1 <= result["best"]["k"] <= 2
    assert result["best"]["auroc"] == 1.0


def test_fewshot_trials_aggregate_mean_and_std(monkeypatch):
    monkeypatch.setattr(evaluate, "evaluate_fewshot", lambda _archive, seed, **_kwargs: {
        "best": {
            "layer": 18,
            "auroc": seed / 100.0,
            "average_precision": 0.9,
            "tpr_at_fpr": 0.8,
        }
    })

    result = evaluate.evaluate_fewshot_trials(
        {},
        num_trials=3,
        head="svm",
        shots_per_source=10,
        shots_per_class=None,
        seed=42,
    )

    assert result["num_trials"] == 3
    assert result["summary"]["auroc"]["mean"] == pytest.approx(0.43)
    assert result["summary"]["auroc"]["std"] == pytest.approx(np.std([0.42, 0.43, 0.44]))
