import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from refuge_seg.data import mask_values_to_classes


COLORS = np.array(
    [
        [0, 0, 0],
        [40, 180, 255],
        [255, 80, 80],
    ],
    dtype=np.uint8,
)


def overlay(image: np.ndarray, mask_classes: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color_mask = COLORS[mask_classes]
    return (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create REFUGE dataset preview figure.")
    parser.add_argument("--root", default="REFUGE")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", default="reports/dataset_preview.png")
    args = parser.parse_args()

    root = Path(args.root)
    image_paths = sorted((root / args.split / "Images").glob("*.jpg"))[: args.count]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(image_paths), 3, figsize=(9, 3 * len(image_paths)))
    if len(image_paths) == 1:
        axes = np.expand_dims(axes, 0)
    for row, image_path in enumerate(image_paths):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(root / args.split / "gts" / f"{image_path.stem}.bmp").convert("L"))
        mask_classes = mask_values_to_classes(mask)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(image_path.name)
        axes[row, 1].imshow(mask_classes, vmin=0, vmax=2)
        axes[row, 1].set_title("label: bg/disc/cup")
        axes[row, 2].imshow(overlay(image, mask_classes))
        axes[row, 2].set_title("overlay")
        for col in range(3):
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved preview to {out}")


if __name__ == "__main__":
    main()
