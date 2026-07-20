import pytest
import torch
from torch import nn

from baselines.models import build_baseline_model


def test_resnet50_baseline_output_shape():
    model = build_baseline_model({"architecture": "resnet50"}).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2,)


def test_unknown_baseline_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported baseline architecture"):
        build_baseline_model({"architecture": "unknown"})


def test_clip_detection_baseline_uses_projected_image_features(monkeypatch):
    class FakeBackbone(nn.Module):
        config = type("Config", (), {"projection_dim": 4})()

        def forward(self, pixel_values, return_dict=True):
            assert return_dict
            features = pixel_values.mean(dim=(2, 3))
            features = torch.cat([features, features[:, :1]], dim=1)
            return type("Output", (), {"image_embeds": features})()

    monkeypatch.setattr(
        "baselines.models.CLIPVisionModelWithProjection.from_pretrained",
        lambda *_args, **_kwargs: FakeBackbone(),
    )
    model = build_baseline_model(
        {"architecture": "clip_detection", "model_name": "fake/clip"}
    )
    output = model(torch.randn(2, 3, 8, 8))
    assert output.shape == (2,)
