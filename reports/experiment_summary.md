# 实验结果汇总

## 完整性检查

- rfau_cnxt_large_ce: metrics/curves/best/last = True/True/True, predictions = 0
- rfau_cnxt_large_dice: metrics/curves/best/last = True/True/True, predictions = 0
- rfau_cnxt_large_ce_dice: metrics/curves/best/last = True/True/True, predictions = 0

缺失或未完成实验：rfau_cnxt_large_ce_dice_lr3e4_bs16, segformer_b5_ce_dice

## 最佳验证集指标

| 实验 | 模型 | Loss | LR | Batch | Scheduler | 实际轮数 | 最佳轮 | Disc Dice | Cup Dice | Mean Dice | Disc IoU | Cup IoU | 总时长(min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rfau_cnxt_large_ce | RFAU-CNxt-Large | CE | 1e-4 | 12 | Cosine | 10 | 5 | 0.8193 | 0.8621 | 0.8407 | 0.6952 | 0.7582 | 6.87 |
| rfau_cnxt_large_dice | RFAU-CNxt-Large | Dice | 1e-4 | 12 | Cosine | 10 | 4 | 0.8856 | 0.0152 | 0.4504 | 0.7948 | 0.0077 | 6.86 |
| rfau_cnxt_large_ce_dice | RFAU-CNxt-Large | CE+Dice+Topology | 1e-4 | 12 | Cosine | 10 | 7 | 0.8889 | 0.8749 | 0.8819 | 0.8003 | 0.7785 | 6.84 |

## 结论要点

- 当前已完成 CE、Dice、CE+Dice+Topology 三个 RFAU-CNxt loss 对比实验。
- CE+Dice+Topology 的平均 Dice 最高，说明组合损失比单独 CE 或单独 Dice 更适合该任务。
- 单独 Dice 对 cup 的学习失败明显，cup Dice 接近 0，说明类别不均衡和优化不稳定需要 CE 辅助。
- SegFormer 和 lr3e4_bs16 调参实验目录缺少 metrics.csv，当前不能纳入定量比较。
- 当前没有 predictions 目录，若报告需要可视化预测图，还需要对 best.pt 运行 evaluate.py。
