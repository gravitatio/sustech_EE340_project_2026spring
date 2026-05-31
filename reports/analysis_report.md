# REFUGE 视盘/视杯分割实验分析报告

## 1. 结果文件完整性

| 实验 | metrics.csv | curves.png | best.pt(MB) | last.pt(MB) | predictions |
| --- | --- | --- | ---: | ---: | ---: |
| rfau_cnxt_large_ce | 有 | 有 | 891.2 | 891.5 | 400 |
| rfau_cnxt_large_dice | 有 | 有 | 891.0 | 891.5 | 400 |
| rfau_cnxt_large_ce_dice | 有 | 有 | 891.2 | 891.5 | 400 |

以下实验未形成完整训练指标，暂不纳入定量比较：rfau_cnxt_large_ce_dice_lr3e4_bs16、segformer_b5_ce_dice。

当前三个 RFAU-CNxt 实验都包含 metrics、训练曲线、best/last checkpoint 和 400 张验证集预测掩码，已经满足报告中的 loss 对比与预测展示需求。

## 2. 最佳验证集指标

| 实验 | Loss | LR | Batch | Scheduler | 最佳轮 | Disc Dice | Cup Dice | Mean Dice | Disc IoU | Cup IoU | 总时长(min) |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rfau_cnxt_large_ce | CE | 1e-4 | 12 | Cosine | 5 | 0.8193 | 0.8621 | 0.8407 | 0.6952 | 0.7582 | 6.87 |
| rfau_cnxt_large_dice | Dice | 1e-4 | 12 | Cosine | 4 | 0.8856 | 0.0152 | 0.4504 | 0.7948 | 0.0077 | 6.86 |
| rfau_cnxt_large_ce_dice | CE+Dice+Topology | 1e-4 | 12 | Cosine | 7 | 0.8889 | 0.8749 | 0.8819 | 0.8003 | 0.7785 | 6.84 |

## 3. 损失函数对比分析

三组实验中，最佳综合结果来自 `rfau_cnxt_large_ce_dice`，其 mean Dice 为 0.8819，Disc Dice 为 0.8889，Cup Dice 为 0.8749。

单独使用 CE 时，Cup Dice 达到 0.8621，说明像素级交叉熵对视杯类别仍有较好的监督效果；但 Disc Dice 为 0.8193，低于组合损失。

单独使用 Dice Loss 时，Disc Dice 达到 0.8856，但 Cup Dice 只有 0.0152，说明在本实验设置下 Dice Loss 对小目标视杯的优化不稳定，可能受到类别不均衡、初始预测质量和视杯区域较小的影响。

CE+Dice+Topology 同时利用了 CE 的逐像素分类稳定性、Dice 的区域重叠优化目标和视杯应位于视盘内的拓扑先验，因此 Disc/Cup 两类更均衡，Cup Dice 提升到 0.8749，Mean Dice 达到 0.8819。

## 4. 曲线与预测结果使用建议

报告中建议放入以下材料：

- `outputs/rfau_cnxt_large_ce/curves.png`：CE 训练曲线。
- `outputs/rfau_cnxt_large_dice/curves.png`：Dice Loss 训练曲线。
- `outputs/rfau_cnxt_large_ce_dice/curves.png`：组合损失训练曲线。
- 从 `outputs/rfau_cnxt_large_ce_dice/predictions/` 中选择若干验证集样本，与原图和标签叠加展示。

建议优先展示 CE+Dice+Topology 的预测图，因为该实验综合指标最好。可补充展示 Dice-only 的失败样本，用于说明小目标类别优化困难。

## 5. 当前不足与后续补充

- 当前已完成 loss 对比实验，但 `rfau_cnxt_large_ce_dice_lr3e4_bs16` 调参实验没有 metrics，无法写入参数调优的定量表格。
- `segformer_b5_ce_dice` 没有训练产物，暂不能作为模型结构对照。
- 如果时间允许，建议至少补跑一个调参实验；如果时间不够，可以在报告中说明由于计算资源/时间限制，对参数调优部分采用配置设计和预期分析。

## 6. 可直接写入报告的结论

本实验比较了 CE、Dice Loss 以及 CE+Dice+Topology 三种损失设置。结果显示，组合损失取得最好的综合分割性能，验证集 Mean Dice 为 0.8819，Disc Dice 为 0.8889，Cup Dice 为 0.8749。单独 Dice Loss 虽然能较好优化视盘区域，但视杯 Dice 明显偏低，说明小目标分割在类别不均衡条件下仅依赖区域重叠损失不够稳定。CE 提供了更稳定的像素级监督，而 Dice 和拓扑约束进一步提升区域一致性与结构合理性，因此 CE+Dice+Topology 是本项目最终采用的最佳方案。
