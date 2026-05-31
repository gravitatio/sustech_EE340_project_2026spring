from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import require_section
from .data import RefugeDataset
from .losses import build_loss
from .metrics import dice_score, iou_score
from .models import RFAUCNxtUNet, SegFormerAdapter


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = require_section(config, "model")
    name = model_cfg["name"]
    if name == "rfau_cnxt":
        return RFAUCNxtUNet(
            encoder_name=model_cfg.get("encoder_name", "convnext_large"),
            num_classes=model_cfg.get("num_classes", 3),
            pretrained=model_cfg.get("pretrained", True),
            checkpoint_path=model_cfg.get("checkpoint_path"),
        )
    if name == "segformer":
        return SegFormerAdapter(
            model_name=model_cfg.get("model_name", "nvidia/segformer-b5-finetuned-ade-640-640"),
            num_classes=model_cfg.get("num_classes", 3),
        )
    raise ValueError(f"Unknown model name: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], steps_per_epoch: int):
    train_cfg = require_section(config, "train")
    scheduler = train_cfg.get("scheduler", "cosine")
    epochs = int(train_cfg["epochs"])
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(train_cfg["lr"]),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
    if scheduler in {"none", None}:
        return None
    raise ValueError(f"Unknown scheduler: {scheduler}")


def run_epoch(model, loader, criterion, optimizer, scaler, device, num_classes, train: bool) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    dice_total = torch.zeros(num_classes, device=device)
    iou_total = torch.zeros(num_classes, device=device)
    count = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        pred = logits.argmax(dim=1)
        total_loss += float(loss.detach().cpu()) * images.size(0)
        dice_total += dice_score(pred, masks, num_classes=num_classes).to(device) * images.size(0)
        iou_total += iou_score(pred, masks, num_classes=num_classes).to(device) * images.size(0)
        count += images.size(0)

    result = {"loss": total_loss / count}
    for class_id, name in [(1, "disc"), (2, "cup")]:
        result[f"{name}_dice"] = float((dice_total[class_id] / count).detach().cpu())
        result[f"{name}_iou"] = float((iou_total[class_id] / count).detach().cpu())
    return result


def train_from_config(config: dict[str, Any]) -> None:
    data_cfg = require_section(config, "data")
    train_cfg = require_section(config, "train")
    out_dir = Path(train_cfg.get("output_dir", "outputs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = int(config["model"].get("num_classes", 3))
    train_set = RefugeDataset(data_cfg["root"], "train", image_size=int(data_cfg["image_size"]), with_masks=True)
    val_set = RefugeDataset(data_cfg["root"], "val", image_size=int(data_cfg["image_size"]), with_masks=True)
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg.get("workers", 4)), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg.get("workers", 4)), pin_memory=True)

    model = build_model(config).to(device)
    criterion = build_loss(train_cfg["loss"], num_classes=num_classes, topology_weight=float(train_cfg.get("topology_weight", 0.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if bool(train_cfg.get("amp", True)) and device.type == "cuda" else None

    rows = []
    best = -1.0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, num_classes, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, None, device, num_classes, train=False)
        if scheduler is not None:
            scheduler.step()
        score = (val_metrics["disc_dice"] + val_metrics["cup_dice"]) / 2
        row = {"epoch": epoch, "seconds": round(time.time() - start, 2)}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        rows.append(row)
        _write_csv(out_dir / "metrics.csv", rows)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "score": score}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "score": score}, out_dir / "last.pt")
        _plot_curves(rows, out_dir / "curves.png")


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(rows: list[dict[str, float]], path: Path) -> None:
    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [r["train_loss"] for r in rows], label="train")
    plt.plot(epochs, [r["val_loss"] for r in rows], label="val")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [r["val_disc_dice"] for r in rows], label="disc Dice")
    plt.plot(epochs, [r["val_cup_dice"] for r in rows], label="cup Dice")
    plt.legend()
    plt.title("Validation Dice")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
