from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6, include_background: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        if not self.include_background:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        denominator = torch.sum(probs + one_hot, dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes, include_background=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


class TopologyLoss(nn.Module):
    """Differentiable penalty for cup probability outside disc probability."""

    def __init__(self, disc_class: int = 1, cup_class: int = 2) -> None:
        super().__init__()
        self.disc_class = disc_class
        self.cup_class = cup_class

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        overflow = torch.relu(probs[:, self.cup_class] - probs[:, self.disc_class])
        return overflow.mean()


def build_loss(name: str, num_classes: int, topology_weight: float = 0.0) -> nn.Module:
    base: nn.Module
    if name == "ce":
        base = nn.CrossEntropyLoss()
    elif name == "dice":
        base = DiceLoss(num_classes=num_classes, include_background=False)
    elif name in {"ce_dice", "dice_ce"}:
        base = CombinedLoss(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown loss name: {name}")

    if topology_weight <= 0:
        return base

    topology = TopologyLoss()

    class LossWithTopology(nn.Module):
        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return base(logits, target) + topology_weight * topology(logits)

    return LossWithTopology()

