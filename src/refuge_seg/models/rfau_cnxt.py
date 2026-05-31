from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip_gate = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = skip * self.skip_gate(skip)
        return self.conv(torch.cat([x, skip], dim=1))


class RFAUCNxtUNet(nn.Module):
    """RFAU-CNxt-style ConvNeXt encoder + attention UNet decoder.

    This is an implementation-compatible project model inspired by the paper
    architecture. It uses timm ConvNeXt feature maps and an attention-gated
    decoder for optic disc/cup segmentation.
    """

    def __init__(
        self,
        encoder_name: str = "convnext_large",
        num_classes: int = 3,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("RFAUCNxtUNet requires timm. Install requirements.txt on the A100 environment.") from exc

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            checkpoint_path=checkpoint_path or "",
        )
        channels = self.encoder.feature_info.channels()
        self.center = ConvBlock(channels[-1], channels[-1])
        self.dec3 = DecoderBlock(channels[-1], channels[-2], channels[-2])
        self.dec2 = DecoderBlock(channels[-2], channels[-3], channels[-3])
        self.dec1 = DecoderBlock(channels[-3], channels[-4], channels[-4])
        self.head = nn.Conv2d(channels[-4], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        x = self.center(features[-1])
        x = self.dec3(x, features[-2])
        x = self.dec2(x, features[-3])
        x = self.dec1(x, features[-4])
        logits = self.head(x)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
