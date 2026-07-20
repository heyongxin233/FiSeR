from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection


DEFAULT_IMAGE_PROCESSOR = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class ResNet50Baseline(nn.Module):
    """Randomly initialized ResNet-50 baseline used in the paper."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50

        self.backbone = resnet50(weights=None)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(pixel_values)).squeeze(-1)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values)


class CLIPDetectionBaseline(nn.Module):
    """End-to-end CLIP ViT-L/14 with a randomly initialized binary head."""

    def __init__(self, model_name: str, local_files_only: bool = False) -> None:
        super().__init__()
        self.backbone = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        feature_dim = int(self.backbone.config.projection_dim)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixel_values, return_dict=True)
        return output.image_embeds

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(pixel_values)).squeeze(-1)


def build_baseline_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("architecture", "resnet50")).lower()
    if architecture == "resnet50":
        return ResNet50Baseline()
    if architecture in {"clip", "clip_detection", "clip-vit-l-14"}:
        model_name = str(config.get("model_name", "openai/clip-vit-large-patch14"))
        return CLIPDetectionBaseline(
            model_name,
            local_files_only=bool(config.get("local_files_only", False)),
        )
    raise ValueError(f"Unsupported baseline architecture: {architecture}")
