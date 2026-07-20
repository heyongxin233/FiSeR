from __future__ import annotations

import numpy as np
import pytest
import torch

from fiser.inference import _load_archive, classify_feature, load_head, move_head


def test_linear_head_checkpoint_and_prediction(tmp_path):
    path = tmp_path / "linear.pt"
    torch.save(
        {
            "head_type": "linear",
            "layer": 18,
            "weight": torch.tensor([[2.0, 0.0]]),
            "bias": torch.tensor([-0.5]),
        },
        path,
    )
    head = load_head(path)
    result = classify_feature(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        head_type="linear",
        head=head,
    )
    assert result["prediction"] == "synthetic"
    assert result["probability_synthetic"] > 0.8


def test_two_logit_linear_head_uses_synthetic_margin(tmp_path):
    path = tmp_path / "two_logit.pt"
    torch.save(
        {
            "head_type": "linear",
            "weight": torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
            "bias": torch.tensor([0.0, 0.0]),
        },
        path,
    )
    result = classify_feature(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        head_type="linear",
        head=load_head(path),
    )
    assert result["prediction"] == "synthetic"


def test_historical_layered_probe_checkpoint(tmp_path):
    path = tmp_path / "probes.pt"
    torch.save(
        {"probes": {18: {"weight": torch.tensor([[2.0, 0.0]]), "bias": torch.tensor([-0.5])}}},
        path,
    )
    head = load_head(path, layer=18)
    result = classify_feature(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        head_type="linear",
        head=head,
    )
    assert head["layer"] == 18
    assert result["prediction"] == "synthetic"


def test_tensor_feature_and_head_device_path(tmp_path):
    path = tmp_path / "linear.pt"
    torch.save(
        {
            "head_type": "linear",
            "weight": torch.tensor([[2.0, 0.0]]),
            "bias": torch.tensor([-0.5]),
        },
        path,
    )
    head = move_head(load_head(path), "cpu")
    result = classify_feature(
        torch.tensor([[1.0, 0.0]], device="cpu"),
        head_type="linear",
        head=head,
        device="cpu",
    )
    assert result["prediction"] == "synthetic"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_tensor_head_path(tmp_path):
    path = tmp_path / "linear_cuda.pt"
    torch.save(
        {
            "head_type": "linear",
            "weight": torch.tensor([[2.0, 0.0]]),
            "bias": torch.tensor([-0.5]),
        },
        path,
    )
    head = move_head(load_head(path), "cuda")
    feature = torch.tensor([[1.0, 0.0]], device="cuda")
    result = classify_feature(feature, head_type="linear", head=head, device="cuda")
    assert result["prediction"] == "synthetic"


def test_prototype_head_prediction(tmp_path):
    path = tmp_path / "prototype.pt"
    torch.save(
        {
            "head_type": "prototype",
            "layer": 18,
            "prototypes": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        },
        path,
    )
    result = classify_feature(
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        head_type="prototype",
        head=load_head(path),
    )
    assert result["prediction"] == "synthetic"


def test_archive_derives_binary_labels_from_source_classes(tmp_path):
    path = tmp_path / "archive.pt"
    torch.save(
        {
            "embeddings": {18: torch.randn(3, 4)},
            "labels": torch.tensor([0, 1, 2]),
            "classes": ["source_a", "nature", "source_b"],
        },
        path,
    )
    archive = _load_archive(path)
    assert archive["binary_labels"].tolist() == [1, 0, 1]
