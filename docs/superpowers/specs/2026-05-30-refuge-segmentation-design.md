# REFUGE 视盘视杯分割项目设计

## 目标

完成课程大作业 Task2 Segmentation 的工程材料：基于 REFUGE 数据集实现视盘/视杯分割项目，覆盖数据认识、网络调研与选择、AI 辅助代码、损失函数对比、参数调优、评估指标、可视化、后处理和中文报告。

本轮只生成训练脚本、配置、测试和报告模板，不执行正式训练。

## 模型路线

主模型选择 RFAU-CNxt 风格的 ConvNeXt-UNet。该路线适合 A100 训练资源，使用 ConvNeXt 编码器、UNet 解码器和注意力跳连融合，针对眼底视盘/视杯这类小目标分割比普通 U-Net 更有表达能力。

对照模型提供 SegFormer 适配脚本。SegFormer 是成熟的 Transformer 分割架构，可作为强基线或迁移学习起点。

## 数据与标签

REFUGE 目录结构：

- `REFUGE/train/Images` 和 `REFUGE/train/gts`
- `REFUGE/val/Images` 和 `REFUGE/val/gts`
- `REFUGE/test/Images`

标签 BMP 像素值映射为三类：

- `0`: cup
- `128`: disc
- `255`: background

训练中统一映射为类别索引：

- `0`: background
- `1`: disc
- `2`: cup

## 实验设计

损失函数实验：

- Cross Entropy
- Dice Loss
- Cross Entropy + Dice Loss

参数调优实验：

- 学习率：`1e-4`、`3e-4`
- 批次大小：`8`、`16`
- 调度器：CosineAnnealingLR、OneCycleLR

输出内容：

- 每轮训练/验证 loss
- disc/cup Dice
- disc/cup IoU
- loss 曲线和指标曲线
- 预测结果可视化
- 后处理前后对比图

## 后处理与拓扑约束

后处理包含：

- 填补分割孔洞
- 每类仅保留最大连通域
- 强制 cup 位于 disc 内部

拓扑约束损失包含一个可微惩罚项：若 cup 概率分布超出 disc 概率分布，则增加损失。该损失可反向传播，适合课程可选任务说明。

## 交付文件

- `requirements.txt`: Python 依赖
- `configs/*.yaml`: 主实验和对照实验配置
- `src/refuge_seg/*`: 数据、模型、损失、指标、训练与评估代码
- `scripts/*.py`: 数据展示、训练、评估、预测、实验计划生成脚本
- `tests/*.py`: 核心逻辑测试
- `reports/final_report.md`: 中文报告模板
- `README.md`: 使用说明

