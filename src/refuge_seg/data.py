from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MASK_TO_CLASS = {255: 0, 128: 1, 0: 2}
CLASS_TO_MASK = {0: 255, 1: 128, 2: 0}


def mask_values_to_classes(mask: np.ndarray) -> np.ndarray:
    """Map REFUGE BMP values to contiguous class ids."""
    values = set(np.unique(mask).tolist())
    unknown = sorted(values.difference(MASK_TO_CLASS))
    if unknown:
        raise ValueError(f"Unexpected REFUGE label values: {unknown}")

    result = np.zeros(mask.shape, dtype=np.int64)
    for value, class_id in MASK_TO_CLASS.items():
        result[mask == value] = class_id
    return result


def classes_to_mask_values(classes: np.ndarray) -> np.ndarray:
    result = np.zeros(classes.shape, dtype=np.uint8)
    for class_id, value in CLASS_TO_MASK.items():
        result[classes == class_id] = value
    return result


@dataclass(frozen=True)
class RefugeSample:
    image: torch.Tensor
    mask: Optional[torch.Tensor]
    image_id: str


class RefugeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 512,
        with_masks: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.with_masks = with_masks
        self.image_dir = self.root / split / "Images"
        self.mask_dir = self.root / split / "gts"
        self.images = sorted(self.image_dir.glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(f"No REFUGE images found in {self.image_dir}")
        if with_masks and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.asarray(image).transpose(2, 0, 1)).float() / 255.0

        item: dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "image_id": image_path.stem,
        }
        if self.with_masks:
            mask_path = self.mask_dir / f"{image_path.stem}.bmp"
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(mask_values_to_classes(np.asarray(mask))).long()
            item["mask"] = mask_tensor
        return item

