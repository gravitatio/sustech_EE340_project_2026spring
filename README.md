# REFUGE 视盘/视杯分割课程项目

本项目用于完成“数据科学中的机器学习”课程大作业 Task2 Segmentation：基于 REFUGE 数据集进行眼底图像中的视盘（optic disc）和视杯（optic cup）分割。

当前仓库已经包含完整的项目脚本、配置文件、测试、实验报告和汇报材料。本项目最终使用 H100 GPU 完成训练，主实验采用 RFAU-CNxt 风格 ConvNeXt-UNet，并对 Cross Entropy、Dice Loss、Cross Entropy + Dice + Topology 三种损失设置进行了对比。

## 项目内容

- 主模型：RFAU-CNxt 风格的 ConvNeXt-UNet，使用 ConvNeXt 编码器和注意力跳连融合。
- 对照模型：SegFormer-B5 适配器。
- 分割类别：`0 背景`、`1 视盘`、`2 视杯`。
- 损失函数实验：Cross Entropy、Dice Loss、Cross Entropy + Dice Loss。
- 调参实验：学习率、batch size、scheduler。
- 可选任务：孔洞填补、最大连通域筛选、视杯必须位于视盘内部的拓扑约束。

## 目录结构

```text
.
├── configs/                  # 训练实验配置
├── docs/superpowers/         # 设计文档和实施计划
├── reports/                  # 实验报告、汇报材料和结果图
├── scripts/                  # 命令行入口脚本
├── src/refuge_seg/           # 项目核心代码
├── tests/                    # 单元测试
├── REFUGE/                   # 数据集目录
├── requirements.txt          # Python 依赖
└── README.md                 # 项目说明
```

## 当前实验结论

已完成的三组 H100 主实验结果如下，最佳模型为 `rfau_cnxt_large_ce_dice`：

| 实验 | 最佳轮 | Disc Dice | Cup Dice | Mean Dice | Disc IoU | Cup IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross Entropy | 5 | 0.8193 | 0.8621 | 0.8407 | 0.6952 | 0.7582 |
| Dice Loss | 4 | 0.8856 | 0.0152 | 0.4504 | 0.7948 | 0.0077 |
| CE + Dice + Topology | 7 | 0.8889 | 0.8749 | 0.8819 | 0.8003 | 0.7785 |

结论：单独 Dice Loss 对小目标视杯不稳定；`CE + Dice + Topology` 同时兼顾像素分类、区域重叠和视杯位于视盘内部的结构先验，因此取得最高 Mean Dice。

## 环境配置

建议在 H100/A100 等 CUDA 训练服务器上创建独立 Python 环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

如果服务器需要手动安装 CUDA 版 PyTorch，请先根据服务器 CUDA 版本安装对应的 PyTorch，再安装其余依赖。例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

本地 Windows/PowerShell 运行脚本前，需要设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH='src'
```

Linux/macOS 或服务器 Bash 环境可以这样运行：

```bash
export PYTHONPATH=src
```

## 数据集格式

项目默认数据集位于 `REFUGE/`，目录结构如下：

```text
REFUGE/
  train/
    Images/*.jpg
    gts/*.bmp
  val/
    Images/*.jpg
    gts/*.bmp
  test/
    Images/*.jpg
```

REFUGE 原始标签像素值映射如下：

| 原始像素值 | 训练类别 | 含义 |
| --- | --- | --- |
| `255` | `0` | 背景 |
| `128` | `1` | 视盘 optic disc |
| `0` | `2` | 视杯 optic cup |

## 常用命令

### 1. 生成数据集预览图

PowerShell：

```powershell
$env:PYTHONPATH='src'
python scripts/explore_dataset.py --root REFUGE --split train
```

Bash：

```bash
PYTHONPATH=src python scripts/explore_dataset.py --root REFUGE --split train
```

输出文件默认保存到：

```text
reports/dataset_preview.png
```

### 2. 检查训练配置

该命令只加载并打印配置，不会开始训练：

```bash
PYTHONPATH=src python scripts/train.py --config configs/rfau_cnxt_large_ce_dice.yaml --dry-run
```

PowerShell：

```powershell
$env:PYTHONPATH='src'
python scripts/train.py --config configs/rfau_cnxt_large_ce_dice.yaml --dry-run
```

### 3. 在 H100/A100 上训练主模型

```bash
PYTHONPATH=src python scripts/train.py --config configs/rfau_cnxt_large_ce_dice.yaml
```

训练输出默认保存到：

```text
outputs/rfau_cnxt_large_ce_dice/
```

RFAU-CNxt 配置默认从项目根目录读取本地 ConvNeXt 权重：

```text
model.safetensors
```

该文件较大，不提交到 Git。训练前需要把它放到项目根目录；如果没有本地权重，可以把配置中的 `checkpoint_path` 删除，并将 `pretrained` 设为 `false` 从随机初始化训练。

主要输出包括：

- `best.pt`：验证集 Dice 最优 checkpoint
- `last.pt`：最后一个 epoch checkpoint
- `metrics.csv`：每轮 loss、Dice、IoU 指标
- `curves.png`：loss 和验证 Dice 曲线

### 4. 运行验证集预测

```bash
PYTHONPATH=src python scripts/evaluate.py --config configs/rfau_cnxt_large_ce_dice.yaml --checkpoint outputs/rfau_cnxt_large_ce_dice/best.pt --split val
```

预测结果默认保存到：

```text
outputs/rfau_cnxt_large_ce_dice/predictions/
```

### 5. 生成汇报报告和结果图

如果本地已经有 `outputs/` 中的三组实验结果，可以运行：

```bash
PYTHONPATH=src python scripts/generate_presentation_report.py
```

该脚本会读取各实验的 `metrics.csv`、`curves.png` 和预测结果，生成：

- `reports/presentation_report.md`
- `reports/pre_detailed_report.md`
- `reports/pre_detailed_report.pdf`
- `reports/pre_assets/*.png`

### 6. 打印所有计划实验命令

```bash
PYTHONPATH=src python scripts/make_experiment_plan.py
```

## 实验配置

当前提供的配置文件：

| 配置文件 | 作用 |
| --- | --- |
| `configs/rfau_cnxt_large_ce.yaml` | RFAU-CNxt-Large + Cross Entropy |
| `configs/rfau_cnxt_large_dice.yaml` | RFAU-CNxt-Large + Dice Loss |
| `configs/rfau_cnxt_large_ce_dice.yaml` | RFAU-CNxt-Large + CE + Dice + 拓扑约束 |
| `configs/rfau_cnxt_large_ce_dice_lr3e4_bs16.yaml` | 调参实验：更大学习率、更大 batch size、OneCycle scheduler |
| `configs/segformer_b5_ce_dice.yaml` | SegFormer-B5 对照实验 |

如果需要重新复现实验，建议训练顺序：

1. `rfau_cnxt_large_ce.yaml`
2. `rfau_cnxt_large_dice.yaml`
3. `rfau_cnxt_large_ce_dice.yaml`
4. `rfau_cnxt_large_ce_dice_lr3e4_bs16.yaml`
5. `segformer_b5_ce_dice.yaml`

## 本地验证

运行轻量单元测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

PowerShell：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

测试覆盖内容：

- REFUGE 标签值到训练类别的映射
- Dice 和 IoU 指标
- Dice Loss
- 拓扑约束损失
- 孔洞填补
- 最大连通域筛选
- 移除脱离视盘区域的视杯预测

## 报告材料

主要报告和汇报材料位于：

- `reports/final_report.md`：课程报告基础版本
- `reports/analysis_report.md`：实验结果分析
- `reports/experiment_summary.md`：实验摘要
- `reports/presentation_report.md`：用于 pre 的 Markdown 汇报稿
- `reports/pre_detailed_report.md`：更详细的图文汇报稿
- `reports/pre_detailed_report.pdf`：由详细汇报稿导出的 PDF
- `reports/pre_assets/`：汇报中使用的曲线、柱状图和预测可视化图片

数据预览图位于：

```text
reports/dataset_preview.png
```

`outputs/`、checkpoint 和本地权重文件不会提交到 Git；仓库中保留的是代码、配置、报告和可复现实验流程。
