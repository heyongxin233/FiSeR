from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _aggregated_positive_contrastive_loss(
    anchors: torch.Tensor,
    bank: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Equation 5 in the paper, with one averaged positive logit per anchor."""
    if anchors.ndim != 2 or bank.ndim != 2:
        raise ValueError("anchors and bank must have shape [N, D]")
    if positive_mask.shape != (anchors.shape[0], bank.shape[0]):
        raise ValueError("positive_mask has an incompatible shape")
    if negative_mask.shape != positive_mask.shape:
        raise ValueError("negative_mask has an incompatible shape")

    anchors = F.normalize(anchors.float(), dim=-1)
    bank = F.normalize(bank.float(), dim=-1)
    similarities = anchors @ bank.transpose(0, 1)

    positive_count = positive_mask.sum(dim=1)
    positive_sum = (similarities * positive_mask).sum(dim=1)
    positive_mean = torch.where(
        positive_count > 0,
        positive_sum / positive_count.clamp_min(1),
        torch.zeros_like(positive_sum),
    )
    positive_logit = positive_mean / temperature

    negative_logits = similarities / temperature
    negative_logits = negative_logits.masked_fill(~negative_mask, float("-inf"))
    denominator = torch.logsumexp(
        torch.cat([positive_logit[:, None], negative_logits], dim=1), dim=1
    )
    losses = denominator - positive_logit
    valid = negative_mask.any(dim=1)
    if not valid.any():
        return anchors.sum() * 0.0
    return losses[valid].mean()


class HierarchicalContrastiveLoss(nn.Module):
    """Coarse natural/synthetic plus fine synthetic-source contrastive loss."""

    def __init__(
        self,
        temperature: float = 0.07,
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.coarse_weight = float(coarse_weight)
        self.fine_weight = float(fine_weight)

    def forward(
        self,
        anchors: torch.Tensor,
        bank: torch.Tensor,
        anchor_labels: torch.Tensor,
        bank_labels: torch.Tensor,
        anchor_sources: torch.Tensor,
        bank_sources: torch.Tensor,
        self_bank_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        anchor_labels = anchor_labels.long().view(-1)
        bank_labels = bank_labels.long().view(-1)
        anchor_sources = anchor_sources.long().view(-1)
        bank_sources = bank_sources.long().view(-1)
        if anchors.shape[0] != anchor_labels.numel() or anchors.shape[0] != anchor_sources.numel():
            raise ValueError("Anchor labels/sources do not match the anchor batch")
        if bank.shape[0] != bank_labels.numel() or bank.shape[0] != bank_sources.numel():
            raise ValueError("Bank labels/sources do not match the bank batch")

        self_mask = torch.zeros(
            (anchors.shape[0], bank.shape[0]), dtype=torch.bool, device=anchors.device
        )
        if self_bank_indices is not None:
            rows = torch.arange(anchors.shape[0], device=anchors.device)
            self_mask[rows, self_bank_indices.long()] = True

        same_binary = anchor_labels[:, None] == bank_labels[None, :]
        coarse_positive = same_binary & ~self_mask
        coarse_negative = ~same_binary & ~self_mask
        coarse = _aggregated_positive_contrastive_loss(
            anchors, bank, coarse_positive, coarse_negative, self.temperature
        )

        synthetic_anchor = anchor_labels == 1
        if synthetic_anchor.any():
            synthetic_bank = bank_labels == 1
            fine_positive = (
                anchor_sources[:, None] == bank_sources[None, :]
            ) & synthetic_bank[None, :] & ~self_mask
            fine_negative = (
                anchor_sources[:, None] != bank_sources[None, :]
            ) & synthetic_bank[None, :] & ~self_mask
            fine = _aggregated_positive_contrastive_loss(
                anchors[synthetic_anchor],
                bank,
                fine_positive[synthetic_anchor],
                fine_negative[synthetic_anchor],
                self.temperature,
            )
        else:
            fine = anchors.sum() * 0.0

        total = self.coarse_weight * coarse + self.fine_weight * fine
        return {"loss": total, "coarse_loss": coarse, "fine_loss": fine}
