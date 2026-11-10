# 实验索引

代码入口位于`scripts/`；已发布结果位于`results/`，本地运行输出位于`runs/`。

| 实验 | 状态 | 结果 | 说明 |
|---|---|---|---|
| formal2023 | 完成：10种子 | [results/2023](../results/2023) | 2023为探索性；242日；最终结论取10种子，不取旧3种子 |
| formal2024 | 完成：10种子 | [results/2024](../results/2024) | 冻结年度滚动方案；已查看区间不再称独立盲测 |
| formal2025 | 完成：10种子 | [results/2025](../results/2025) | 2025两组IC接近零 |
| common2023 | 完成：同样本比较 | [results/experiments/compare_gru_ridge_2023](../results/experiments/compare_gru_ridge_2023) | 237日；GRU来自A实验baseline，不是旧242日表 |
| common2024 | 完成：同样本比较 | [results/experiments/compare_gru_ridge_2024](../results/experiments/compare_gru_ridge_2024) | 保存逐日IC和覆盖率，逐股票预测由本地运行生成 |
| common2025 | 完成：同样本比较 | [results/experiments/compare_gru_ridge_2025](../results/experiments/compare_gru_ridge_2025) | 全部增量检验不显著 |
| A2023 | 完成：40次训练 | [results/experiments/experiment_A_csrank/test_2023](../results/experiments/experiment_A_csrank/test_2023) | MSE和rank各10种子，两种特征；排名拼接未改善 |
| ridge_valid | 完成：开发诊断 | [results/experiments/ridge_diagnostic_valid2022](../results/experiments/ridge_diagnostic_valid2022) | 2022验证选lambda，有选模偏差 |
| feature_valid | 完成：开发诊断 | [results/experiments/feature_diagnostic_valid2022](../results/experiments/feature_diagnostic_valid2022) | 固定lambda=10；筛选删除振幅；不代表独立评估 |
| ridge_years | 完成：跨年探索 | [results/experiments/ridge_crossyear_fixed](../results/experiments/ridge_crossyear_fixed) | 固定三方案；删除振幅逐年改善Ridge |
| window | 完成：6次训练 | [results/experiments/window_diagnostic_valid2023_rank](../results/experiments/window_diagnostic_valid2023_rank) | 验证2023；3种子；近期窗口无平均改善 |
| stage1 | 历史：流程检查 | [results/stage1](../results/stage1) | 单seed，不与正式实验混用 |
| stage2 | 历史：结构探索 | [results/stage2](../results/stage2) | MLP/GRU；训练采样和选模规则与正式对照不同 |
| stage2_shuffle | 历史：时间扰动 | [results/stage2_shuffled](../results/stage2_shuffled) | 描述性实验，不证明时序无用 |
| stage3_attn | 历史：结构探索 | [results/stage3_attention](../results/stage3_attention) | Attention |
| stage3_pool | 历史：结构探索 | [results/stage3_pool](../results/stage3_pool) | Mean Pooling |

## 实验分层

正式学习目标对照：2023—2025年各10种子。

基线与消融：Ridge、负换手率、截面排名拼接和训练窗口。均属于历史探索。

历史结构研究：MLP、GRU最后状态、Attention、Mean Pooling与时间扰动；训练采样和选模方式不同，不能跨阶段直接归因。

早期3种子结论已由10种子实验替代。旧零margin hinge实验存在常数解退化，不作为有效对照。A实验2024年中断，未纳入结果。

查看脚本参数：`python project.py run 脚本名 --help`。部分历史入口没有参数解析，具体用法见运行指南。
