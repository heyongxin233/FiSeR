import torch

from fiser.losses import HierarchicalContrastiveLoss
from train import resolve_accumulation_steps


def test_hierarchical_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    anchors = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0])
    sources = torch.tensor([0, 0, 0, 1, 1, 2, 2, 0])
    values = HierarchicalContrastiveLoss()(
        anchors,
        anchors.detach(),
        labels,
        labels,
        sources,
        sources,
        torch.arange(8),
    )
    assert torch.isfinite(values["loss"])
    values["loss"].backward()
    assert anchors.grad is not None
    assert torch.isfinite(anchors.grad).all()


def test_no_synthetic_samples_has_zero_fine_loss():
    features = torch.randn(4, 8, requires_grad=True)
    labels = torch.zeros(4, dtype=torch.long)
    sources = torch.zeros(4, dtype=torch.long)
    values = HierarchicalContrastiveLoss()(features, features.detach(), labels, labels, sources, sources)
    assert values["fine_loss"].item() == 0.0


def test_global_batch_derives_accumulation_from_world_size():
    config = {"global_batch_size": 1280, "per_device_batch_size": 40}
    assert resolve_accumulation_steps(config, world_size=2) == 16
    assert resolve_accumulation_steps(config, world_size=4) == 8
