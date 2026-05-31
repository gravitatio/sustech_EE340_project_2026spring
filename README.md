# REFUGE 视盘/视杯分割课程项目

本项目用于完成“数据科学中的机器学习”课程大作业 Task2 Segmentation：基于 REFUGE 数据集进行眼底图像中的视盘（optic disc）和视杯（optic cup）分割。

当前仓库已经包含完整的项目脚本、配置文件、测试和报告模板。训练部分只提供可运行脚本和实验配置，不在本地执行正式训练；正式训练建议放到 A100 环境中运行。

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
├── reports/                  # 报告模板和数据预览图
├── scripts/                  # 命令行入口脚本
├── src/refuge_seg/           # 项目核心代码
├── tests/                    # 单元测试
├── REFUGE/                   # 数据集目录
├── requirements.txt          # Python 依赖
└── README.md                 # 项目说明
```

## 环境配置

建议在 A100 训练服务器上创建独立 Python 环境，然后安装依赖：

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

### 3. 在 A100 上训练主模型

```bash
PYTHONPATH=src python scripts/train.py --config configs/rfau_cnxt_large_ce_dice.yaml
```

训练输出默认保存到：

```text
outputs/rfau_cnxt_large_ce_dice/
```

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

### 5. 打印所有计划实验命令

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

建议正式训练顺序：

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

中文报告模板位于：

```text
reports/final_report.md
```

数据预览图位于：

```text
reports/dataset_preview.png
```

正式训练完成后，将各实验的 `metrics.csv`、`curves.png` 和预测可视化结果填入报告中的结果表格和分析部分即可。

