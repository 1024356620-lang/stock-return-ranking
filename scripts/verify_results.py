"""只读核验已保存结果的数值、配对关系与汇总一致性。

数据格式、参数和运行方法见 docs/PROJECT_GUIDE.md。
"""
from pathlib import Path
import ast
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent

def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10)

def hac(x):
    x=np.asarray(x);u=x-x.mean();n=len(x)
    lr=u@u/n+sum(2*(1-k/6)*(u[k:]@u[:-k]/n) for k in range(1,6))
    se=math.sqrt(max(lr,0)/n)
    return math.erfc(abs(x.mean()/se)/math.sqrt(2)) if se else float(x.mean()==0)

def main():
    checks=[]
    for p in (ROOT/'scripts').glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    checks.append('全部脚本语法解析通过')
    formal=pd.read_csv(ROOT/'results/annual_summary.csv').set_index('year')
    for year in [2023,2024,2025]:
        pair=pd.read_csv(ROOT/f'results/{year}/paired_seed_daily.csv')
        assert not pair.duplicated(['seed','date']).any()
        assert set(pair.seed)==set(range(10))
        assert pair.groupby('date').seed.nunique().eq(10).all()
        close(pair['diff'],pair.rank_ic-pair.mse_ic)
        daily=pair.groupby('date')['diff'].mean()
        row=formal.loc[year]
        close(row.mse_rank_ic,pair.groupby('seed').mse_ic.mean().mean())
        close(row.rank_rank_ic,pair.groupby('seed').rank_ic.mean().mean())
        close(row.delta,daily.mean());close(row.hac_p,hac(daily))
        assert len(daily)==row.n_days
    checks.append('正式三年实验的种子日期配对、均值、增量和HAC重算通过')
    annual=pd.read_csv(ROOT/'results/experiments/ridge_crossyear_fixed/annual_summary.csv')
    pairs=pd.read_csv(ROOT/'results/experiments/ridge_crossyear_fixed/paired_daily.csv')
    close(pairs.delta_ic,pairs.candidate_ic-pairs.raw7_ic)
    for (year,variant),g in pairs.groupby(['year','variant']):
        assert not g.date.duplicated().any()
        row=annual.loc[annual.year.eq(year)&annual.variant.eq(variant)].iloc[0]
        close(g.candidate_ic.mean(),row.rank_ic);close(g.delta_ic.mean(),row.delta_vs_raw7)
    checks.append('Ridge跨年均值与配对增量一致')
    for year in [2023,2024,2025]:
        folder=ROOT/f'results/experiments/compare_gru_ridge_{year}'
        daily=pd.read_csv(folder/'daily_ic.csv')
        summary=pd.read_csv(folder/'summary.csv')
        comparison=pd.read_csv(folder/'comparison.csv')
        for row in summary.itertuples():close(daily[row.model].mean(),row.rank_ic)
        for row in comparison.itertuples():
            x=daily[row.model]-daily[row.baseline]
            close(x.mean(),row.delta_ic);close(hac(x),row.hac_p)
        for mode in ['mse','rank']:
            cols=[f'{mode}_seed{i}' for i in range(10)]
            close(daily[cols].mean(axis=1),daily[f'gru_{mode}_seedmean'])
        coverage=pd.read_csv(folder/'coverage.csv')
        assert coverage.retained_fraction.ge(.95).all()
    checks.append('同样本GRU/Ridge的逐日IC、种子平均、汇总和HAC一致')
    folder=ROOT/'results/experiments/experiment_A_csrank/test_2023'
    pairs=pd.read_csv(folder/'paired_daily.csv')
    close(pairs.delta_ic,pairs.rank_ic_A-pairs.rank_ic_baseline)
    summary=pd.read_csv(folder/'comparison.csv')
    for row in summary.itertuples():
        g=pairs.loc[pairs['mode'].eq(row.mode)&pairs['split'].eq(row.split)]
        close(g.groupby('date').delta_ic.mean().mean(),row.delta_ic)
        assert g.seed.nunique()==10
    checks.append('A实验2023年十种子配对汇总一致')
    folder=ROOT/'results/experiments/window_diagnostic_valid2023_rank'
    p=pd.read_csv(folder/'paired_valid_daily.csv')
    close(p.delta_ic,p.rank_ic_recent2y-p.rank_ic_expanding)
    row=pd.read_csv(folder/'comparison.csv').iloc[0]
    close(p.groupby('date').delta_ic.mean().mean(),row.delta_ic)
    assert p.seed.nunique()==3
    checks.append('训练窗口实验三种子验证汇总一致')
    for line in checks:print('通过：',line)
    print('核验范围：已保存逐日IC与汇总的一致性；原始行情和逐股票预测需另行核验。')

if __name__=='__main__':main()
