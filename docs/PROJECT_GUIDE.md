# 数据与运行指南

## 1. 环境

在仓库根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

训练使用CPU。核心依赖为numpy、pandas、torch、scipy、matplotlib、pyarrow。依赖版本未精确锁定；跨环境可能存在数值差异。

## 2. 导入已有数据

推荐直接使用已有特征缓存，避免重复构建：

```bat
python project.py data --features "C:\path\to\ml\seq_features.parquet"
```

命令会检查字段、重复股票日期、标签结束日期及无穷值，并复制为仓库根目录的seq_features.parquet。这个检查不验证股票池、复权或所有特征的因果正确性。

也可以手动把seq_features.parquet复制到与project.py同级的位置，不要放入scripts/。数据导入不覆盖目标；如需更新数据，先备份旧文件再移除它。

仅有原始行情时：

```bat
python project.py data --klines "C:\path\to\ml\klines.pkl"
python project.py build-features
```

klines.pkl为字典：键是股票代码，值是pandas DataFrame。使用date、close、high、low、volume、amount、turn字段；date可转为日期，其余为数值列。仅反序列化可信来源的pickle。build-features只构建缓存，不训练模型。

缓存字段：code、date、flag_tradable、label_fwd5_ret、label_end、ret_1d、ma5_dev、ma20_dev、log_vol_ratio、turnover、amplitude、vol_20d。日历、标签和缺失处理口径见README。

特征缓存存在时，训练脚本优先使用它，不会自动根据新klines.pkl刷新。更换数据应先备份旧缓存并明确重建。不要把旧数据对应的runs缓存与新数据混用。

## 3. 查看已发布结果

```bash
python project.py list
python project.py report
python project.py verify
```

这些命令不需要行情、预测文件或权重。verify从results/复算已有逐日IC的汇总、配对关系和HAC；不是从原始行情重建结果。results/final/sources.json记录每张最终表的来源。

## 4. 复现正式GRU实验

以下命令会训练模型，输出至本地runs/：

```bash
python project.py run run_stage4_10seeds
python project.py run compare_stage4_10seeds
python project.py run run_crossyear --years 2024 2025
python project.py run compare_crossyear --year 2024
python project.py run compare_crossyear --year 2025
```

2023输出为runs/stage4_fixed；2024/2025输出为runs/stage5_crossyear/test_YEAR。完整运行保存checkpoint.pt、predictions.parquet、history.csv、daily_ic.csv等文件。

数据或代码改变后使用全新runs目录/独立工程副本，避免旧缓存混入。不要手改manifest绕过不匹配检查。未保存完整预测时，完成标记不代表可以直接续用。

## 5. 简单基线与特征诊断

```bash
python project.py run run_ridge_diagnostic
python project.py run run_feature_diagnostic
python project.py run run_ridge_crossyear
```

前两项使用2020—2021训练、2022验证；第三项固定lambda=10，比较7特征Ridge、删除振幅的6特征Ridge及负换手率。数据划分与原GRU年度训练范围对应，间隔年不用于重新调参。

这些脚本遇到已有输出目录会拒绝覆盖；新实验可使用--out指定新目录。评价年份已经参与历史研究，结果按探索性报告。

## 6. 特征拼接与训练窗口实验

```bash
python project.py run run_experiment_A --years 2023 --out runs/A_new
python project.py run run_window_diagnostic --out runs/window_new
```

A比较原始7特征与附加7个截面排名特征。窗口实验默认rank损失、3种子、验证2023，对比训练2020—2022与2021—2022。验证集同时用于早停与对照，因此不构成独立测试。

## 7. 同样本比较

需要先生成Ridge与GRU逐股票预测。默认读取runs中的本地结果，已发布汇总则在results/experiments中：

```bash
python project.py run compare_gru_ridge --year 2024
python project.py run compare_gru_ridge --year 2025
python project.py run compare_gru_ridge --year 2023 --gru-root runs/experiment_A_csrank/test_2023/baseline
```

2023若使用上述A实验的其他输出路径，--gru-root也需相应调整。若使用旧stage4的242日预测，脚本将取共同样本；来源与配置应在报告中说明，不能假设它与A重跑逐位一致。

脚本默认要求MSE和rank各10种子齐全，按共同股票/日期核对标签后重算IC。增量为GRU减基线。GRU先算各seed逐日IC再平均，不是预测集成IC。

## 8. Git提交范围

提交README、project.py、scripts、docs、requirements.txt、.gitignore与results。原始数据、runs、权重、逐股票预测由.gitignore排除；不要强制添加这些文件。

首次克隆后，已发布结果可直接核验。训练重现需要另行提供合适的数据来源与权限。原始行情由使用者按数据来源许可自行提供，不随仓库分发。
