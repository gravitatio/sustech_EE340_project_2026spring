from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


EXPERIMENTS = {
    "rfau_cnxt_large_ce": {
        "name": "CE",
        "title": "Cross Entropy",
        "loss": "Cross Entropy",
        "color": "#4C78A8",
    },
    "rfau_cnxt_large_dice": {
        "name": "Dice",
        "title": "Dice Loss",
        "loss": "Dice Loss",
        "color": "#F58518",
    },
    "rfau_cnxt_large_ce_dice": {
        "name": "CE+Dice+Topo",
        "title": "CE + Dice + Topology",
        "loss": "CE + Dice + Topology",
        "color": "#54A24B",
    },
}


LABEL_COLORS = np.array(
    [
        [0, 0, 0],
        [30, 144, 255],
        [255, 80, 80],
    ],
    dtype=np.uint8,
)


def read_metrics(root: Path) -> dict[str, list[dict[str, float]]]:
    metrics: dict[str, list[dict[str, float]]] = {}
    for exp in EXPERIMENTS:
        path = root / "outputs" / exp / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Copy trained outputs/ into the project root before generating the presentation report."
            )
        with path.open(newline="", encoding="utf-8") as f:
            rows = []
            for row in csv.DictReader(f):
                parsed = {k: float(v) for k, v in row.items()}
                parsed["mean_dice"] = (parsed["val_disc_dice"] + parsed["val_cup_dice"]) / 2
                rows.append(parsed)
            metrics[exp] = rows
    return metrics


def best_rows(metrics: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {exp: max(rows, key=lambda r: r["mean_dice"]) for exp, rows in metrics.items()}


def save_metric_bar_chart(best: dict[str, dict[str, float]], out: Path) -> None:
    labels = [EXPERIMENTS[e]["name"] for e in EXPERIMENTS]
    x = np.arange(len(labels))
    width = 0.24
    values = {
        "Disc Dice": [best[e]["val_disc_dice"] for e in EXPERIMENTS],
        "Cup Dice": [best[e]["val_cup_dice"] for e in EXPERIMENTS],
        "Mean Dice": [best[e]["mean_dice"] for e in EXPERIMENTS],
    }

    plt.figure(figsize=(10, 5.5))
    for i, (name, vals) in enumerate(values.items()):
        plt.bar(x + (i - 1) * width, vals, width, label=name)
        for j, val in enumerate(vals):
            plt.text(x[j] + (i - 1) * width, val + 0.015, f"{val:.3f}", ha="center", fontsize=9)
    plt.xticks(x, labels)
    plt.ylim(0, 1.02)
    plt.ylabel("Dice")
    plt.title("Validation Dice Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def save_iou_bar_chart(best: dict[str, dict[str, float]], out: Path) -> None:
    labels = [EXPERIMENTS[e]["name"] for e in EXPERIMENTS]
    x = np.arange(len(labels))
    width = 0.32
    disc = [best[e]["val_disc_iou"] for e in EXPERIMENTS]
    cup = [best[e]["val_cup_iou"] for e in EXPERIMENTS]

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, disc, width, label="Disc IoU", color="#4C78A8")
    plt.bar(x + width / 2, cup, width, label="Cup IoU", color="#E45756")
    for i, val in enumerate(disc):
        plt.text(x[i] - width / 2, val + 0.015, f"{val:.3f}", ha="center", fontsize=9)
    for i, val in enumerate(cup):
        plt.text(x[i] + width / 2, val + 0.015, f"{val:.3f}", ha="center", fontsize=9)
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("IoU")
    plt.title("Validation IoU Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def save_training_curve(metrics: dict[str, list[dict[str, float]]], out: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for exp, rows in metrics.items():
        plt.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], marker="o", label=EXPERIMENTS[exp]["name"])
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.subplot(1, 2, 2)
    for exp, rows in metrics.items():
        plt.plot([r["epoch"] for r in rows], [r["mean_dice"] for r in rows], marker="o", label=EXPERIMENTS[exp]["name"])
    plt.title("Validation Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def mask_values_to_classes(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[mask == 128] = 1
    out[mask == 0] = 2
    return out


def colorize(mask: np.ndarray) -> np.ndarray:
    return LABEL_COLORS[mask]


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize(mask)
    return (image * (1 - alpha) + color * alpha).astype(np.uint8)


def load_image(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img


def save_sample_comparison(root: Path, sample_id: str, out: Path) -> None:
    image = load_image(root / "REFUGE" / "val" / "Images" / f"{sample_id}.jpg")
    image_np = np.asarray(image)
    gt = Image.open(root / "REFUGE" / "val" / "gts" / f"{sample_id}.bmp").convert("L")
    gt = gt.resize(image.size, Image.NEAREST)
    gt_np = mask_values_to_classes(np.asarray(gt))

    panels = [("Image", image_np), ("Ground Truth", overlay(image_np, gt_np))]
    for exp in EXPERIMENTS:
        pred = Image.open(root / "outputs" / exp / "predictions" / f"{sample_id}.bmp").convert("L")
        pred = pred.resize(image.size, Image.NEAREST)
        pred_np = mask_values_to_classes(np.asarray(pred))
        panels.append((EXPERIMENTS[exp]["name"], overlay(image_np, pred_np)))

    plt.figure(figsize=(16, 4))
    for i, (title, panel) in enumerate(panels, start=1):
        ax = plt.subplot(1, len(panels), i)
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def save_prediction_grid(root: Path, sample_ids: list[str], out: Path) -> None:
    plt.figure(figsize=(16, 3.2 * len(sample_ids)))
    titles = ["Image", "Ground Truth", "CE", "Dice", "CE+Dice+Topo"]
    for row, sample_id in enumerate(sample_ids):
        image = load_image(root / "REFUGE" / "val" / "Images" / f"{sample_id}.jpg")
        image_np = np.asarray(image)
        gt = Image.open(root / "REFUGE" / "val" / "gts" / f"{sample_id}.bmp").convert("L")
        gt = gt.resize(image.size, Image.NEAREST)
        panels = [image_np, overlay(image_np, mask_values_to_classes(np.asarray(gt)))]
        for exp in EXPERIMENTS:
            pred = Image.open(root / "outputs" / exp / "predictions" / f"{sample_id}.bmp").convert("L")
            pred = pred.resize(image.size, Image.NEAREST)
            panels.append(overlay(image_np, mask_values_to_classes(np.asarray(pred))))
        for col, panel in enumerate(panels):
            ax = plt.subplot(len(sample_ids), len(panels), row * len(panels) + col + 1)
            ax.imshow(panel)
            if row == 0:
                ax.set_title(titles[col])
            ax.axis("off")
            if col == 0:
                ax.text(0, -35, sample_id, fontsize=11, color="black")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def write_report(root: Path, out_dir: Path, metrics: dict[str, list[dict[str, float]]], best: dict[str, dict[str, float]]) -> None:
    report = root / "reports" / "pre_detailed_report.md"
    rows = []
    for exp in EXPERIMENTS:
        total_seconds = sum(r["seconds"] for r in metrics[exp])
        b = best[exp]
        rows.append(
            {
                "exp": exp,
                "name": EXPERIMENTS[exp]["title"],
                "best": b,
                "epochs": len(metrics[exp]),
                "minutes": total_seconds / 60,
            }
        )

    combo = next(r for r in rows if r["exp"] == "rfau_cnxt_large_ce_dice")
    ce = next(r for r in rows if r["exp"] == "rfau_cnxt_large_ce")
    dice = next(r for r in rows if r["exp"] == "rfau_cnxt_large_dice")

    report.write_text(
        f"""# REFUGE 视盘/视杯分割项目汇报报告

## 1. 汇报主线

本项目完成 REFUGE 眼底图像中的视盘和视杯分割。汇报重点包括：数据集认识、分割任务和分类任务的差异、网络结构选择、损失函数实验、H100 训练设置、评估指标、预测可视化，以及最终结论。

本次最终训练使用 **H100 GPU**。H100 的显存和 Tensor Core 能力使 ConvNeXt-Large 级别模型可以使用较高输入分辨率 `768 x 768` 和 batch size 12 进行训练。训练阶段开启 AMP 混合精度以降低显存占用并提升吞吐。

## 2. 任务与数据集

REFUGE 是眼底图像视盘/视杯分割数据集。本项目将标签统一映射成三类：

| 类别 | 含义 | 原始标签值 |
| --- | --- | --- |
| 0 | 背景 | 255 |
| 1 | 视盘 optic disc | 128 |
| 2 | 视杯 optic cup | 0 |

视盘通常是图像中较亮、近圆形、血管汇聚的区域；视杯位于视盘内部，面积更小。视杯/视盘结构关系非常重要：正常情况下视杯应包含在视盘内部，因此本项目还加入了拓扑约束和后处理。

## 3. 模型选择

本项目采用 **RFAU-CNxt 风格 ConvNeXt-UNet**：

- 编码器使用 ConvNeXt-Large，负责提取强语义特征。
- 解码器采用 U-Net 风格逐级上采样，恢复空间分辨率。
- 跳跃连接保留低层边界和纹理信息。
- 注意力式 skip fusion 强化关键区域。

选择该模型的原因：

1. 医学图像分割常需要精细边界，U-Net 类编码器-解码器结构非常适合。
2. ConvNeXt 比普通 CNN 编码器表达能力更强。
3. 相比纯 Transformer，ConvNeXt-UNet 训练更稳定、工程实现更直接。
4. 对 REFUGE 这种视盘/视杯任务，局部纹理和全局结构都重要。

## 4. 分割任务和分类任务的差异

分类任务输出通常是 `[B, C]`，表示整张图片属于哪个类别。分割任务输出是 `[B, C, H, W]`，表示每个像素属于哪个类别。本项目输出 3 类像素级 logits：背景、视盘、视杯。

分割任务的关键差异：

- 标签是二维 mask，而不是单个类别 id。
- loss 按像素或区域计算。
- 指标使用 Dice、IoU 等区域重叠指标。
- 小目标类别容易被背景淹没，需要处理类别不均衡。
- 医学结构可能有拓扑约束，例如视杯必须在视盘内部。

## 5. 实验设置

所有已完成实验均使用 H100 训练，主要设置如下：

| 设置项 | 值 |
| --- | --- |
| 模型 | RFAU-CNxt 风格 ConvNeXt-UNet |
| GPU | H100 |
| 输入尺寸 | 768 x 768 |
| Epoch | 10 |
| Batch Size | 12 |
| Workers | 24 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Scheduler | Cosine |
| AMP | 开启 |

实验对比三种损失函数：

| 实验 | 损失函数 | 目的 |
| --- | --- | --- |
| CE | Cross Entropy | 观察逐像素分类监督效果 |
| Dice | Dice Loss | 观察区域重叠优化效果 |
| CE+Dice+Topo | CE + Dice + Topology | 综合像素监督、区域重叠和结构先验 |

## 6. 定量结果

最佳轮次按照 `Mean Dice = (Disc Dice + Cup Dice) / 2` 选择。

| 实验 | 最佳轮 | Val Loss | Disc Dice | Cup Dice | Mean Dice | Disc IoU | Cup IoU | 训练时长 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {r['name']} | {int(r['best']['epoch'])} | {r['best']['val_loss']:.4f} | {r['best']['val_disc_dice']:.4f} | {r['best']['val_cup_dice']:.4f} | {r['best']['mean_dice']:.4f} | {r['best']['val_disc_iou']:.4f} | {r['best']['val_cup_iou']:.4f} | {r['minutes']:.2f} min |" for r in rows)}

![Validation Dice Comparison](pre_assets/metric_dice_comparison.png)

![Validation IoU Comparison](pre_assets/metric_iou_comparison.png)

## 7. 训练过程分析

下图对比了三种损失函数的验证集 loss 和 mean Dice 变化。

![Training Curves](pre_assets/training_curves_comparison.png)

从曲线和结果可以看出：

- CE 的训练较稳定，Cup Dice 表现较好，但 Disc Dice 不如组合损失。
- Dice Loss 对视盘区域有一定效果，但视杯 Dice 接近 0，说明单独 Dice 对小目标视杯优化不稳定。
- CE+Dice+Topology 在 Disc 和 Cup 两类上都较均衡，取得最高 Mean Dice。

## 8. 损失函数分析

### 8.1 Cross Entropy

CE 是逐像素分类损失，每个像素都会贡献梯度，因此训练早期更稳定。实验中 CE 的 Cup Dice 达到 {ce['best']['val_cup_dice']:.4f}，说明它能有效学习视杯类别。但 CE 不直接优化区域重叠，因此 Disc Dice 只有 {ce['best']['val_disc_dice']:.4f}，低于组合损失。

### 8.2 Dice Loss

Dice Loss 直接优化预测区域和真实区域的重叠度，理论上适合医学分割。但本实验中 Dice-only 的 Cup Dice 只有 {dice['best']['val_cup_dice']:.4f}，说明它对视杯这种小区域非常不稳定。可能原因是训练初期预测区域质量差，Dice 梯度对小目标较敏感，导致模型更偏向学习较大的视盘区域。

### 8.3 CE + Dice + Topology

组合损失取得最佳效果，Mean Dice 为 {combo['best']['mean_dice']:.4f}。其中 CE 提供稳定像素监督，Dice 优化区域重叠，Topology 约束利用视杯必须位于视盘内部这一医学先验。该组合使 Disc Dice 达到 {combo['best']['val_disc_dice']:.4f}，Cup Dice 达到 {combo['best']['val_cup_dice']:.4f}。

## 9. 预测可视化

下图展示验证集样本的原图、真实标签和三种损失函数下的预测结果。

![Prediction Grid](pre_assets/prediction_grid.png)

单样本详细对比：

![Sample V0001](pre_assets/sample_V0001_comparison.png)

![Sample V0050](pre_assets/sample_V0050_comparison.png)

图中蓝色区域表示视盘，红色区域表示视杯。CE+Dice+Topology 的预测整体更接近真实标签，结构也更合理。Dice-only 的结果更容易出现视杯缺失或不稳定。

## 10. 后处理与拓扑约束

本项目实现了三类后处理：

1. 孔洞填补：修复预测区域内部出现的空洞。
2. 最大连通域：每类只保留最大区域，去除孤立噪声。
3. 视杯-视盘约束：去除脱离视盘区域的视杯预测。

拓扑损失采用可微形式，对 `P(cup) > P(disc)` 的区域施加惩罚，因此能够参与反向传播。这个约束符合医学先验：视杯应该位于视盘内部。

## 11. 项目不足

当前实验主要完成了课程要求中最核心的 loss 对比。仍有两点不足：

- 参数调优实验没有完整训练输出，报告中只保留实验设计，不纳入定量比较。
- SegFormer 对照实验没有完整产物，暂不能进行模型结构级定量比较。

此外，训练只进行了 10 个 epoch。若进一步增加 epoch、加入 ROI 裁剪和更系统的数据增强，模型性能仍可能提升。

## 12. 最终结论

本项目完成了 REFUGE 视盘/视杯分割任务，构建了 RFAU-CNxt 风格 ConvNeXt-UNet，并比较了 CE、Dice 和 CE+Dice+Topology 三种损失函数。实验表明，CE+Dice+Topology 取得最佳综合效果，Mean Dice = {combo['best']['mean_dice']:.4f}，Disc Dice = {combo['best']['val_disc_dice']:.4f}，Cup Dice = {combo['best']['val_cup_dice']:.4f}。

最终结论是：对于视盘/视杯这类医学图像小目标分割任务，仅使用单一损失函数不够稳定；结合像素级监督、区域重叠优化和医学拓扑先验，能够获得更可靠的分割结果。

## 13. Pre 汇报建议

建议汇报顺序：

1. 介绍 REFUGE 任务和视盘/视杯定义。
2. 解释为什么选择 ConvNeXt-UNet。
3. 说明分割任务和分类任务代码差异。
4. 展示三种损失函数设置。
5. 展示 Dice/IoU 指标柱状图。
6. 展示训练曲线。
7. 展示预测可视化。
8. 总结 CE+Dice+Topology 最优，并解释原因。
""",
        encoding="utf-8",
    )


def main() -> None:
    root = Path(".")
    out_dir = root / "reports" / "pre_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(root)
    best = best_rows(metrics)
    save_metric_bar_chart(best, out_dir / "metric_dice_comparison.png")
    save_iou_bar_chart(best, out_dir / "metric_iou_comparison.png")
    save_training_curve(metrics, out_dir / "training_curves_comparison.png")
    save_prediction_grid(root, ["V0001", "V0050", "V0100"], out_dir / "prediction_grid.png")
    save_sample_comparison(root, "V0001", out_dir / "sample_V0001_comparison.png")
    save_sample_comparison(root, "V0050", out_dir / "sample_V0050_comparison.png")
    write_report(root, out_dir, metrics, best)
    print("Generated reports/pre_detailed_report.md and images in reports/pre_assets")


if __name__ == "__main__":
    main()
