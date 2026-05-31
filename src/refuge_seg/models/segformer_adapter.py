from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerAdapter(nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("SegFormerAdapter requires transformers. Install requirements.txt on the A100 environment.") from exc

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = self.model(pixel_values=x)
        return F.interpolate(out.logits, size=size, mode="bilinear", align_corners=False)

