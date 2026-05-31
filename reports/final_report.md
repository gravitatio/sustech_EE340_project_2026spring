# REFUGE 视盘与视杯分割课程报告

## 1. 数据集认识

本项目使用 REFUGE 数据集完成眼底图像中的 optic disc 和 optic cup 分割。目录包含训练集、验证集和测试集，训练集与验证集提供图像和 BMP 标签，测试集只提供图像。

标签像素值映射如下：

| 原始像素值 | 训练类别 | 含义 |
| --- | --- | --- |
| 255 | 0 | 背景 |
| 128 | 1 | 视盘 optic disc |
| 0 | 2 | 视杯 optic cup |

视盘通常是眼底图像中较明亮的近圆形区域，位于血管汇聚处；视杯位于视盘内部，面积更小，颜色通常更亮或纹理更浅。运行 `scripts/explore_dataset.py` 可生成图像、标签和叠加图。

## 2. 网络调研与选择

常见图像分割网络包括：

- FCN：将分类网络改造成全卷积结构，是语义分割基础模型。
- U-Net：编码器-解码器结构配合跳跃连接，适合医学图像小数据场景。
- DeepLab 系列：使用空洞卷积和 ASPP 捕获多尺度上下文。
- Transformer 分割模型：例如 SegFormer、Swin-UNet、TransUNet，擅长全局建模。
- ConvNeXt-UNet 系列：用现代卷积骨干提升 U-Net 表达能力，在医学图像中兼顾稳定性和性能。

本项目选择 RFAU-CNxt 风格 ConvNeXt-UNet 作为主模型，原因是它面向视盘/视杯分割任务，使用强 ConvNeXt 编码器和注意力跳连融合，适合 A100 上进行高分辨率训练。SegFormer-B5 作为 Transformer 对照模型。

## 3. 分割任务代码特点

与分类任务相比，分割任务输出不是一个图像级类别，而是 `[B, C, H, W]` 的像素级 logits。本项目输出 3 类：背景、视盘、视杯。

分割任务常用损失函数：

- Cross Entropy：逐像素分类优化，稳定但对小目标类别不均衡敏感。
- Dice Loss：直接优化区域重叠，更适合医学分割小目标。
- CE + Dice：兼顾像素分类稳定性和区域重叠目标。

评估指标：

- Dice：衡量预测区域与真实区域的重叠程度。
- IoU：交并比，更严格地评估区域重叠。

## 4. 损失函数实验设计

计划比较以下配置：

| 实验 | 配置文件 |
| --- | --- |
| CE | `configs/rfau_cnxt_large_ce.yaml` |
| Dice | `configs/rfau_cnxt_large_dice.yaml` |
| CE + Dice + topology | `configs/rfau_cnxt_large_ce_dice.yaml` |

训练脚本自动记录 `metrics.csv` 和 `curves.png`。正式训练后在此处填入每个实验的 disc Dice、cup Dice、disc IoU、cup IoU 和 loss 曲线分析。

## 5. 参数调优实验设计

可调参数包括：

- Learning Rate：控制参数更新步长。
- Batch Size：影响显存占用、梯度稳定性和训练速度。
- Scheduler：控制学习率随训练过程变化。
- Image Size：影响细节保留和显存需求。
- Topology Weight：控制 cup-in-disc 约束强度。

调参对照配置为 `configs/rfau_cnxt_large_ce_dice_lr3e4_bs16.yaml`，使用更大学习率、更大 batch size 和 OneCycle 调度器。正式训练后比较训练时长、收敛速度和验证指标。

## 6. 后处理与拓扑约束

后处理包含：

- 孔洞填补：修复预测区域内部空洞。
- 最大连通域：每类保留最大的连通区域，去除孤立噪声。
- 拓扑修复：去除脱离视盘区域的视杯预测。

拓扑损失使用 `relu(P(cup) - P(disc))` 作为惩罚项，可以参与反向传播；当模型把视杯预测到视盘外时，损失会增大。

## 7. 训练与复现实验命令

```bash
PYTHONPATH=src python scripts/train.py --config configs/rfau_cnxt_large_ce.yaml
PYTHONPATH=src python scripts/train.py --config configs/rfau_cnxt_large_dice.yaml
PYTHONPATH=src python scripts/train.py --config configs/rfau_cnxt_large_ce_dice.yaml
PYTHONPATH=src python scripts/train.py --config configs/rfau_cnxt_large_ce_dice_lr3e4_bs16.yaml
PYTHONPATH=src python scripts/train.py --config configs/segformer_b5_ce_dice.yaml
```

## 8. 结果记录表

| 模型 | Loss | LR | Batch | Scheduler | Disc Dice | Cup Dice | Disc IoU | Cup IoU | 训练时长 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RFAU-CNxt-Large | CE | 1e-4 | 8 | Cosine | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 |
| RFAU-CNxt-Large | Dice | 1e-4 | 8 | Cosine | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 |
| RFAU-CNxt-Large | CE+Dice+Topo | 1e-4 | 8 | Cosine | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 |
| RFAU-CNxt-Large | CE+Dice+Topo | 3e-4 | 16 | OneCycle | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 |
| SegFormer-B5 | CE+Dice | 6e-5 | 8 | Cosine | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 |

## 9. 结论分析模板

正式训练后根据曲线和指标分析：

- CE 通常收敛稳定，但 cup 小目标 Dice 可能偏低。
- Dice Loss 更关注区域重叠，可能改善 cup，但早期训练可能更不稳定。
- CE+Dice 通常兼顾稳定性和重叠指标。
- 加入拓扑约束后，困难样本中 cup 超出 disc 的错误应减少。
- 更大 batch size 在 A100 上可提升吞吐，但也可能影响泛化，需要结合验证 Dice 判断。

