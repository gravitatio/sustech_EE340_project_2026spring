from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import RefugeDataset, classes_to_mask_values
from .postprocess import postprocess_prediction
from .train import build_model


def predict_from_checkpoint(config: dict, checkpoint_path: str | Path, split: str = "val", postprocess: bool = True) -> None:
    out_dir = Path(config["eval"].get("output_dir", "outputs/predictions"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dataset = RefugeDataset(config["data"]["root"], split, image_size=int(config["data"]["image_size"]), with_masks=split != "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = build_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            logits = model(image)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            if postprocess:
                pred = postprocess_prediction(pred)
            mask_values = classes_to_mask_values(pred)
            Image.fromarray(mask_values).save(out_dir / f"{batch['image_id'][0]}.bmp")

