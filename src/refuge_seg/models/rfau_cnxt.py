from __future__ import annotations

from pathlib import Path

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


def remap_convnext_feature_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert original timm ConvNeXt keys to FeatureListNet wrapper keys."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("head."):
            continue
        new_key = key
        if new_key.startswith("stem."):
            new_key = new_key.replace("stem.0.", "stem_0.", 1)
            new_key = new_key.replace("stem.1.", "stem_1.", 1)
        if new_key.startswith("stages."):
            for stage_id in range(4):
                new_key = new_key.replace(f"stages.{stage_id}.", f"stages_{stage_id}.", 1)
        remapped[new_key] = value
    return remapped


def load_local_convnext_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Local checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading .safetensors checkpoints requires safetensors. Run: pip install safetensors") from exc
        state_dict = load_file(str(path))
    else:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    state_dict = remap_convnext_feature_keys(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(incompatible.unexpected_keys)
    if loaded <= 0:
        raise RuntimeError(f"No compatible ConvNeXt encoder weights were loaded from {path}")
    print(
        f"Loaded local encoder checkpoint from {path} "
        f"({loaded} tensors, {len(incompatible.missing_keys)} missing, {len(incompatible.unexpected_keys)} unexpected)."
    )


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
        )
        if checkpoint_path:
            load_local_convnext_checkpoint(self.encoder, checkpoint_path)
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
