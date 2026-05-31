# REFUGE 视盘/视杯分割实验报告

## 1. 项目目标

本项目面向 REFUGE 眼底图像数据集，完成视盘（optic disc）和视杯（optic cup）的像素级语义分割。课程要求包括数据集认识、分割网络调研与选择、分割任务代码实现、不同损失函数对比、模型参数分析，以及可选的后处理和拓扑结构约束。

本次实验重点比较三种损失函数设置：Cross Entropy、Dice Loss、CE+Dice+Topology，并使用验证集 Dice 和 IoU 评价模型分割性能。

## 2. 数据集与任务设置

REFUGE 数据集包含眼底彩照和对应分割标签。本项目使用训练集训练模型，验证集评估模型表现并生成预测掩码。标签像素值被映射为三类：

| 原始标签值 | 训练类别 | 含义 |
| --- | --- | --- |
| 255 | 0 | 背景 |
| 128 | 1 | 视盘 optic disc |
| 0 | 2 | 视杯 optic cup |

视盘通常是眼底图像中较明亮、近圆形、血管汇聚的区域；视杯位于视盘内部，面积更小，是青光眼筛查等任务中的关键结构。

## 3. 网络选择

常见分割网络包括 FCN、U-Net、DeepLab、TransUNet、Swin-UNet、SegFormer 和 ConvNeXt-UNet 等。医学图像数据规模通常有限，且目标区域边界细节重要，因此编码器-解码器结构仍然非常适合该任务。

本项目选择 RFAU-CNxt 风格的 ConvNeXt-UNet 作为主模型。该模型以 ConvNeXt 作为强编码器，结合 U-Net 式解码器和跳跃连接，能够同时利用高层语义信息和低层空间细节。相比普通 U-Net，ConvNeXt 编码器具有更强的特征表达能力；相比纯 Transformer 模型，训练稳定性和工程可控性更好。

## 4. 分割任务与分类任务的代码差异

分类任务通常输出 `[B, C]` 的图像级 logits，而分割任务输出 `[B, C, H, W]` 的像素级 logits。本项目中 `C=3`，分别对应背景、视盘和视杯。训练时每个像素都有一个类别标签，因此损失函数按像素或区域进行计算。

本项目实现并比较了以下损失函数：

- Cross Entropy：逐像素分类损失，训练稳定，但对类别不均衡较敏感。
- Dice Loss：直接优化预测区域与真实区域的重叠度，常用于医学图像分割。
- CE + Dice + Topology：结合逐像素监督、区域重叠优化和视杯必须位于视盘内部的拓扑先验。

## 5. 实验设置

| 实验 | 模型 | Loss | 输入尺寸 | Epoch | Batch Size | LR | Scheduler |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| rfau_cnxt_large_ce | RFAU-CNxt-Large | Cross Entropy | 768 | 10 | 12 | 1e-4 | Cosine |
| rfau_cnxt_large_dice | RFAU-CNxt-Large | Dice Loss | 768 | 10 | 12 | 1e-4 | Cosine |
| rfau_cnxt_large_ce_dice | RFAU-CNxt-Large | CE + Dice + Topology | 768 | 10 | 12 | 1e-4 | Cosine |

每个实验均保存 `metrics.csv`、`curves.png`、`best.pt`、`last.pt`，并在验证集上生成 400 张预测掩码。

## 6. 定量结果

下表展示每个实验在验证集上的最佳结果，最佳轮次按 `Mean Dice = (Disc Dice + Cup Dice) / 2` 选择。

| 实验 | 最佳轮 | Val Loss | Disc Dice | Cup Dice | Mean Dice | Disc IoU | Cup IoU | 总时长(min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rfau_cnxt_large_ce | 5 | 0.1737 | 0.8193 | 0.8621 | 0.8407 | 0.6952 | 0.7582 | 6.87 |
| rfau_cnxt_large_dice | 4 | 0.6707 | 0.8856 | 0.0152 | 0.4504 | 0.7948 | 0.0077 | 6.86 |
| rfau_cnxt_large_ce_dice | 7 | 0.9570 | 0.8889 | 0.8749 | 0.8819 | 0.8003 | 0.7785 | 6.84 |

## 7. 结果分析

CE+Dice+Topology 取得了最佳综合效果，Mean Dice 为 0.8819，Disc Dice 为 0.8889，Cup Dice 为 0.8749。这说明组合损失能够同时兼顾像素级分类稳定性、区域重叠质量和结构合理性。

单独使用 Cross Entropy 时，Cup Dice 为 0.8621，说明 CE 对小目标视杯仍提供了稳定的像素级监督；但 Disc Dice 为 0.8193，低于组合损失。

单独使用 Dice Loss 时，Disc Dice 为 0.8856，但 Cup Dice 只有 0.0152。这说明单独 Dice Loss 在本实验中对视杯这类小区域结构优化不稳定，可能原因包括视杯区域面积小、类别不均衡明显，以及训练早期预测区域质量较差导致 Dice 梯度不够稳定。

从三组结果可以看出，医学图像分割中单一损失函数往往难以同时兼顾稳定性和结构精度。CE+Dice 的组合能够缓解类别不均衡和区域重叠优化之间的冲突；额外加入拓扑约束后，可减少视杯越出视盘边界这类不符合解剖结构的预测。

## 8. 曲线与可视化结果

训练曲线文件位于：

- `outputs/rfau_cnxt_large_ce/curves.png`
- `outputs/rfau_cnxt_large_dice/curves.png`
- `outputs/rfau_cnxt_large_ce_dice/curves.png`

验证集预测结果位于：

- `outputs/rfau_cnxt_large_ce/predictions/`
- `outputs/rfau_cnxt_large_dice/predictions/`
- `outputs/rfau_cnxt_large_ce_dice/predictions/`

每个预测目录均包含 400 张 BMP 掩码。汇报时建议展示 CE+Dice+Topology 的若干预测结果，并选择 Dice-only 的失败样本作为对比，用于说明损失函数选择对视杯分割的影响。

## 9. 后处理与拓扑约束

本项目实现了孔洞填补、最大连通域筛选和视杯-视盘拓扑修复。孔洞填补用于修复分割区域内部空洞；最大连通域用于去除孤立噪声；拓扑修复用于减少视杯脱离视盘或越界的情况。拓扑损失采用可微形式，对 `P(cup) > P(disc)` 的区域施加惩罚，因此可以参与反向传播。

## 10. 不足与改进方向

当前实验完成了核心 loss 对比，但仍有两个不足：第一，参数调优实验 `rfau_cnxt_large_ce_dice_lr3e4_bs16` 没有完整输出，因此没有纳入定量表格；第二，SegFormer 对照实验没有完整训练产物，无法进行模型结构级比较。后续可以补充更大学习率、更大 batch size、OneCycle scheduler 的实验，并训练 SegFormer-B5 作为 Transformer baseline。

此外，当前实验只训练 10 个 epoch，结果已经体现出不同损失函数的显著差异；若进一步增加训练轮数，并结合更系统的数据增强和 ROI 裁剪，视盘/视杯分割精度仍有提升空间。

## 11. 汇报结论

本项目完成了 REFUGE 视盘/视杯分割任务，并基于 RFAU-CNxt 风格 ConvNeXt-UNet 比较了三种损失函数。实验结果表明，CE+Dice+Topology 的综合表现最好，验证集 Mean Dice 达到 0.8819，Disc Dice 达到 0.8889，Cup Dice 达到 0.8749。单独 Dice Loss 在视杯分割上表现较差，说明小目标医学图像分割需要稳定的像素级监督和结构先验辅助。最终，本项目选择 CE+Dice+Topology 作为最佳训练方案。
