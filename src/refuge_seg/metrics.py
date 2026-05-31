from __future__ import annotations

import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    scores = []
    for class_id in range(num_classes):
        pred_c = pred == class_id
        target_c = target == class_id
        intersection = (pred_c & target_c).sum().float()
        denom = pred_c.sum().float() + target_c.sum().float()
        scores.append((2 * intersection + smooth) / (denom + smooth))
    return torch.stack(scores)


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    scores = []
    for class_id in range(num_classes):
        pred_c = pred == class_id
        target_c = target == class_id
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        scores.append((intersection + smooth) / (union + smooth))
    return torch.stack(scores)

