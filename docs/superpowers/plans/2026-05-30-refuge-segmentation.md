# REFUGE Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成 REFUGE 视盘/视杯分割项目代码、训练脚本、配置、测试和中文报告模板，不执行正式训练。

**Architecture:** 项目采用 Python 包 `src/refuge_seg`，将数据集、损失、指标、模型、后处理和训练循环拆分为独立模块。主模型为 RFAU-CNxt 风格 ConvNeXt-UNet，对照模型为 SegFormer 适配器；通过 YAML 配置驱动不同损失和调参实验。

**Tech Stack:** Python 3.10+、PyTorch、torchvision、timm、transformers、PyYAML、Pillow、OpenCV、NumPy、Matplotlib。

---

### Task 1: 项目骨架与依赖

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `src/refuge_seg/__init__.py`

- [x] 创建 Python 包结构。
- [x] 写入依赖说明，包含 A100 训练推荐依赖。
- [x] 写入 README 使用入口。

### Task 2: 测试先行定义核心行为

**Files:**
- Create: `tests/test_label_mapping.py`
- Create: `tests/test_losses_metrics_postprocess.py`

- [x] 写标签值到类别索引映射测试。
- [x] 写 Dice/IoU、DiceLoss、拓扑损失、后处理测试。
- [x] 先运行测试确认失败，之后实现代码。

### Task 3: 数据集与可视化

**Files:**
- Create: `src/refuge_seg/data.py`
- Create: `scripts/explore_dataset.py`

- [x] 实现 REFUGE 数据集加载、标签映射、resize、tensor 转换。
- [x] 实现图像和标签叠加展示脚本。

### Task 4: 损失、指标、后处理

**Files:**
- Create: `src/refuge_seg/losses.py`
- Create: `src/refuge_seg/metrics.py`
- Create: `src/refuge_seg/postprocess.py`

- [x] 实现 CE、Dice、CE+Dice、拓扑约束损失。
- [x] 实现 per-class Dice 和 IoU。
- [x] 实现填洞、最大连通域、cup-in-disc 修复。

### Task 5: 模型与训练脚本

**Files:**
- Create: `src/refuge_seg/models/rfau_cnxt.py`
- Create: `src/refuge_seg/models/segformer_adapter.py`
- Create: `src/refuge_seg/train.py`
- Create: `src/refuge_seg/evaluate.py`
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`

- [x] 实现 RFAU-CNxt 风格模型，优先使用 timm ConvNeXt encoder，缺少 timm 时给出明确错误。
- [x] 实现 SegFormer 适配器，缺少 transformers 时给出明确错误。
- [x] 实现 AMP、scheduler、checkpoint、CSV 日志、曲线保存。
- [x] 实现评估和预测可视化入口。

### Task 6: 实验配置与报告

**Files:**
- Create: `configs/rfau_cnxt_large_ce.yaml`
- Create: `configs/rfau_cnxt_large_dice.yaml`
- Create: `configs/rfau_cnxt_large_ce_dice.yaml`
- Create: `configs/segformer_b5_ce_dice.yaml`
- Create: `scripts/make_experiment_plan.py`
- Create: `reports/final_report.md`

- [x] 写主模型和对照模型配置。
- [x] 写实验命令生成脚本。
- [x] 写中文报告模板并覆盖课程任务点。

### Task 7: 验证

**Files:**
- Modify: all created files

- [x] 运行核心单元测试。
- [x] 运行脚本 `--help` 或 dry-run。
- [x] 不启动正式训练。

