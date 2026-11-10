"""固定Ridge与负换手率方案的跨年份探索性评价。
冻结三个候选的跨年份探索性比较，不调参、不挑种子。

在项目根目录查看运行参数：
python project.py run run_ridge_crossyear --help

数据与运行说明：docs/PROJECT_GUIDE.md
实验定义与索引：docs/EXPERIMENT_INDEX.md
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import run_ridge_diagnostic as base
import run_feature_diagnostic as diagnostic


def prepare_year(frame, year):
    # 复用相同的样本资格、前向填充和标签规则。
    prior, evaluation = base.prepare(frame, year)
    cutoff = pd.Timestamp(f'{year-1}-01-01')
    train = prior.loc[(prior.date < cutoff) & (prior.label_end < cutoff)].reset_index(drop=True)
    if train.empty:
        raise ValueError(f'{year}: empty training split')
    if evaluation.date.dt.month.nunique() != 12:
        raise ValueError(f'{year}: evaluation data does not cover all 12 months')
    return train, evaluation


def run_year(frame, year, out):
    tr, te = prepare_year(frame, year)
    print(f'YEAR {year}: train 2020..{year-2}, n={len(tr):,}; evaluation n={len(te):,}', flush=True)
    ytr, yte = tr[base.LABEL].to_numpy(), te[base.LABEL].to_numpy()
    predictions = te[['date','code','label_end',base.LABEL]].copy()
    params, summaries, daily_list, monthly_list, pairs = {}, [], [], [], []
    benchmark = None
    for variant in ['raw7','drop_amplitude','negative_turnover']:
        if variant == 'negative_turnover':
            pred = -te.turnover.to_numpy(float)
            n_features = 1
        else:
            cols = [f for f in base.FEATURES if variant == 'raw7' or f != 'amplitude']
            mu,sd,coef,intercept = base.ridge_fit(tr[cols].to_numpy(float),ytr,10.)
            pred = ((te[cols].to_numpy(float)-mu)/sd)@coef+intercept
            n_features = len(cols)
            for k,v in dict(mu=mu,sd=sd,coef=coef,intercept=intercept,features=cols).items():
                params[f'{variant}__{k}'] = np.asarray(v)
        if not np.isfinite(pred).all():
            raise ValueError(f'{variant}: nonfinite prediction')
        predictions[variant] = pred
        d = base.daily_ic(te,yte,pred)
        if benchmark is None:
            benchmark = d.copy()
        pd.testing.assert_frame_equal(d[['date','n']],benchmark[['date','n']])
        if not d.rank_ic.isna().equals(benchmark.rank_ic.isna()):
            raise ValueError('Different evaluable date sets')
        d['delta_vs_raw7'] = d.rank_ic-benchmark.rank_ic
        m = diagnostic.month_table(d)
        m['delta_vs_raw7'] = m.rank_ic-diagnostic.month_table(benchmark).rank_ic
        summaries.append(dict(year=year,variant=variant,rank_ic=d.rank_ic.mean(),
                              delta_vs_raw7=d.delta_vs_raw7.mean(),n_days=int(d.rank_ic.count()),
                              positive_months=int(m.rank_ic.gt(0).sum()),
                              improved_months=int(m.delta_vs_raw7.gt(0).sum()),
                              train_n=len(tr),evaluation_n=len(te),n_features=n_features))
        daily_list.append(d.assign(year=year,variant=variant))
        monthly_list.append(m.assign(year=year,variant=variant))
        if variant != 'raw7':
            pairs.append(pd.DataFrame({'year':year,'variant':variant,'date':d.date,
                                       'raw7_ic':benchmark.rank_ic,'candidate_ic':d.rank_ic,
                                       'delta_ic':d.delta_vs_raw7,'n':d.n}))
    out.mkdir(parents=True,exist_ok=False)
    predictions.to_parquet(out/'predictions.parquet',index=False)
    np.savez(out/'models.npz',**params)
    pd.concat([tr[['date','code','label_end']].assign(split='train'),
               te[['date','code','label_end']].assign(split='evaluation')]).to_parquet(out/'samples.parquet',index=False)
    summary = pd.DataFrame(summaries)
    print(summary[['year','variant','rank_ic','delta_vs_raw7','positive_months','improved_months']].to_string(index=False),flush=True)
    return summary,pd.concat(daily_list,ignore_index=True),pd.concat(monthly_list,ignore_index=True),pd.concat(pairs,ignore_index=True)


def run(frame, years, out):
    if out.exists():
        raise FileExistsError(f'Output exists: {out}; use a new --out')
    # 创建输出文件前先检查全部年份。
    for year in years:
        prepare_year(frame,year)
    collected = [[],[],[],[]]
    for year in years:
        tables = run_year(frame,year,out/f'test_{year}')
        for bucket,table in zip(collected,tables):
            bucket.append(table)
    for name,bucket in zip(['annual_summary','daily_ic','monthly_ic','paired_daily'],collected):
        pd.concat(bucket,ignore_index=True).to_csv(out/f'{name}.csv',index=False)


def self_test():
    import tempfile
    rng = np.random.default_rng(47)
    dates = pd.bdate_range('2020-01-01','2025-12-31')
    d = pd.DataFrame({'date':np.tile(dates,21),'code':np.repeat(np.arange(21),len(dates)), 'flag_tradable':1})
    for f in base.FEATURES:
        d[f] = rng.normal(size=len(d))
    d[base.LABEL] = -.02*d.turnover+rng.normal(0,.001,len(d))
    d['label_end'] = d.groupby('code').date.shift(-5)
    d.loc[d.label_end.isna(),base.LABEL] = np.nan
    for year in [2023,2024,2025]:
        tr,te = prepare_year(d,year)
        cutoff=pd.Timestamp(f'{year-1}-01-01')
        assert (tr.label_end<cutoff).all() and (tr.date<cutoff).all()
        assert (te.label_end<=pd.Timestamp(f'{year}-12-31')).all()
        changed=d.copy()
        changed.loc[changed.date>=cutoff,base.FEATURES] *= -100
        tr2,_=prepare_year(changed,year)
        pd.testing.assert_frame_equal(tr,tr2)
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'result'
        run(d,[2023,2024,2025],out)
        s=pd.read_csv(out/'annual_summary.csv')
        assert len(s)==9
        assert (s.loc[s.variant.eq('negative_turnover'),'rank_ic']>.9).all()
        pairs=pd.read_csv(out/'paired_daily.csv')
        np.testing.assert_allclose(pairs.delta_ic,pairs.candidate_ic-pairs.raw7_ic,atol=1e-12)
        for year in [2023,2024,2025]:
            pred=pd.read_parquet(out/f'test_{year}'/'predictions.parquet')
            _,te=prepare_year(d,year)
            np.testing.assert_array_equal(pred.negative_turnover,-te.turnover)
        try:
            run(d,[2023],out)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Overwrite guard failed')
    print('PASS: all three years, frozen sign, train/gap/evaluation isolation, label boundaries, paired deltas, output guard')


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,default=root/'seq_features.parquet')
    p.add_argument('--years',nargs='+',type=int,default=[2023,2024,2025])
    p.add_argument('--out',type=Path,default=root/'runs'/'ridge_crossyear_fixed')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if len(set(args.years))!=len(args.years) or any(y not in [2023,2024,2025] for y in args.years):
        p.error('Use distinct evaluation years from 2023, 2024, 2025')
    if args.out.exists():
        p.error(f'Output exists: {args.out}; choose a new --out')
    if not args.data.is_file():
        p.error(f'Missing {args.data}')
    started=time.time()
    frame=diagnostic.read_development(args.data,max(args.years))
    run(frame,sorted(args.years),args.out)
    import pyarrow as pa
    manifest=dict(years=sorted(args.years),reg_lambda=10.,negative_turnover_direction=-1,
                  feature_sets={'raw7':base.FEATURES,'drop_amplitude':[f for f in base.FEATURES if f!='amplitude']},
                  train_rule='2020 to evaluation_year-2; label_end before gap year; gap year not fitted',
                  script_sha256=base.sha256(__file__),ridge_code_sha256=base.sha256(base.__file__),
                  diagnostic_code_sha256=base.sha256(diagnostic.__file__),data_sha256=base.sha256(args.data),
                  versions={'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__},
                  note='Exploratory historical evaluation, not independent blind testing',elapsed_seconds=time.time()-started)
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\nDone in {time.time()-started:.1f}s: {args.out.resolve()}')


if __name__=='__main__':
    main()
