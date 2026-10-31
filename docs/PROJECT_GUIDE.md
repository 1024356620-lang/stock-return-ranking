# 本地项目整理与历史说明

## 使用整理包

推荐先解压到新文件夹检查，再将正式脚本、README、requirements.txt、.gitignore、docs及results复制到原项目。不要删除或覆盖原来的klines.pkl、seq_features.parquet和runs目录。本包不含这些大型产物。

原训练脚本保持在根目录，其ROOT由脚本位置推导；不要直接移入scripts子目录。_crossyear_data.py及run_crossyear.py按原字节复制，以保留既有manifest中的代码指纹；不要因整理而重排代码或换行。

已有run_stage4.py为修正版3种子入口，继续保留在本地即可；整理包以run_stage4_10seeds.py作为正式入口，未重复打包旧入口。已有compare_stage4.py同样可保留，但最终结果使用10种子比较。

旧paired_compare.py、paired_compare_v2.py、paired_compare_v3.py仅作历史版本；正式第三阶段入口为paired_compare_final.py。旧runs/stage4为退化排序损失实验，不能用于正式结论。无需删除旧文件，也不要再次运行旧入口覆盖结果。

## 历史阶段命令

```bash
python run_stage1.py
python run_stage2.py
python run_stage2.py --shuffle
python run_stage3.py
python run_stage3b.py
python rebuild_stage2_daily_ic.py
python paired_compare_final.py
```

这些命令可能重训或重写对应阶段产物，仅在需要复现实验时运行。

## 运行结果与静态结果

2023正式运行在runs/stage4_fixed，汇总为summary_10seeds.csv；比较结果在runs/stage4_fixed_compare_10seeds。
2024/2025运行在runs/stage5_crossyear/test_YEAR，比较结果在该目录comparison下。
本包results/YEAR保存用户上传、已核验的CSV快照。训练脚本不会直接读取或覆盖results快照。新实验完成并核验后才手动替换快照。

2023未收到20次训练的完整summary_10seeds.csv，故该年目录不伪造training_summary.csv；其模型均值来自seed_summary.csv和逐日文件。2024/2025已附原训练汇总。

## Git整理

.gitignore忽略本地原始数据、特征缓存、runs、权重、Python缓存；results中的轻量CSV会保留。忽略规则不会自动移除已经被Git跟踪的文件。先使用git status检查，再明确暂存需要发布的文件，不要盲目git add全部数据。

未设置远程地址或执行提交、推送；本包不假定用户已有ML仓库URL。未添加许可证，因为代码及数据的发布许可需由所有者确定。

## 报告措辞

可以写：实现GRU序列建模、排序损失及公平对照；修复退化目标；用10个随机种子与年度滚动评估检验稳健性；使用HAC和分块bootstrap进行配对分析。
不能写：排序损失显著优于MSE、跨年稳定有效、实现稳定超额收益。最终三个年份的逐日增量检验均不显著。

未完成研究不应借“项目整理”补成既成事实：原始数据来源及股票池偏差审计、逐股票样本对账、精确环境锁定、实盘成本评估均有待独立工作。
