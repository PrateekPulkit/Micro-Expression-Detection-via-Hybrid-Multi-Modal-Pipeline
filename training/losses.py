"""
training/losses.py
Custom loss functions for class-imbalanced ME detection.
Focal loss significantly reduces false positives on the majority (non-ME) class.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss — reduces relative loss for easy negatives,
    putting more focus on hard positive examples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Args:
        alpha:  Weighting factor for positive class (0–1).
                Set < 0.5 to down-weight positive, > 0.5 to up-weight.
        gamma:  Focusing parameter. 0 = standard BCE. 2 is typical.
        reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1) raw logits.
            targets: (B,) or (B, 1) float binary targets {0, 1}.
        """
        targets = targets.view(-1, 1).float()
        logits = logits.view(-1, 1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)               # probability of the correct class
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy — simpler class weighting alternative.

    Args:
        pos_weight: Scalar weight for the positive class.
                    pos_weight > 1 → recall↑, precision↓
    """

    def __init__(self, pos_weight: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("pw", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view(-1, 1).float()
        logits = logits.view(-1, 1)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pw)


def build_loss(cfg: dict) -> nn.Module:
    """Factory function from config dict."""
    t_cfg = cfg.get("training", {})
    if t_cfg.get("focal_loss", {}).get("enabled", True):
        return FocalLoss(
            alpha=t_cfg["focal_loss"].get("alpha", 0.75),
            gamma=t_cfg["focal_loss"].get("gamma", 2.0),
        )
    else:
        weights = t_cfg.get("class_weights", [1.0, 3.0])
        return WeightedBCELoss(pos_weight=weights[1])
