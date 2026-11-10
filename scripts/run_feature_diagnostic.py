"""单特征与逐个删除特征诊断：仅使用训练集和验证集。
单特征与逐个删除特征诊断（探索性，仅训练/验证）。

在项目根目录查看运行参数：
python project.py run run_feature_diagnostic --help

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


def train_direction(mean_ic):
    return -1 if mean_ic < 0 else 1


def month_table(daily):
    return daily.assign(month=daily.date.dt.strftime('%Y-%m')).groupby('month').agg(
        rank_ic=('rank_ic','mean'), n_days=('rank_ic','count')).reset_index()


def analyze(frame, year, out):
    if out.exists():
        raise FileExistsError(f'Output exists: {out}; choose another --out')
    tr, va = base.prepare(frame, year)
    features = base.FEATURES
    ytr, yva = tr[base.LABEL].to_numpy(), va[base.LABEL].to_numpy()
    singles, single_days, single_months = [], [], []
    print(f'train={len(tr):,}; valid={len(va):,}; lambda=10; no test evaluation', flush=True)
    for feature in features:
        td = base.daily_ic(tr, ytr, tr[feature].to_numpy())
        train_ic = float(td.rank_ic.mean())
        direction = train_direction(train_ic)
        vd = base.daily_ic(va, yva, direction * va[feature].to_numpy())
        monthly = month_table(vd)
        singles.append(dict(feature=feature, train_raw_ic=train_ic, direction=direction,
                            valid_directed_ic=vd.rank_ic.mean(), n_days=int(vd.rank_ic.count()),
                            positive_months=int(monthly.rank_ic.gt(0).sum())))
        single_days.append(vd.assign(feature=feature, direction=direction))
        single_months.append(monthly.assign(feature=feature, direction=direction))
    rows, days, months, coefficients, model_params = [], [], [], [], {}
    predictions = va[['date','code','label_end',base.LABEL]].copy()
    baseline_day = None
    for removed in [None] + features:
        name = 'baseline' if removed is None else f'drop_{removed}'
        cols = [f for f in features if f != removed]
        X, V = tr[cols].to_numpy(float), va[cols].to_numpy(float)
        mu, sd, coef, intercept = base.ridge_fit(X, ytr, 10.)
        pred = ((V-mu)/sd) @ coef + intercept
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite predictions')
        predictions[name] = pred
        d = base.daily_ic(va, yva, pred)
        if baseline_day is None:
            baseline_day = d.copy()
        pd.testing.assert_frame_equal(d[['date','n']], baseline_day[['date','n']])
        assert d.rank_ic.isna().equals(baseline_day.rank_ic.isna())
        d['delta_vs_baseline'] = d.rank_ic - baseline_day.rank_ic
        monthly = month_table(d)
        monthly['delta_vs_baseline'] = monthly.rank_ic - month_table(baseline_day).rank_ic
        rows.append(dict(variant=name, removed_feature=removed or '', n_features=len(cols),
                         valid_ic=d.rank_ic.mean(), delta_vs_baseline=d.delta_vs_baseline.mean(),
                         n_days=int(d.rank_ic.count()), positive_months=int(monthly.rank_ic.gt(0).sum()),
                         improved_months=int(monthly.delta_vs_baseline.gt(0).sum())))
        days.append(d.assign(variant=name))
        months.append(monthly.assign(variant=name))
        for f,c,m,s in zip(cols,coef,mu,sd):
            coefficients.append(dict(variant=name,feature=f,coefficient=c,train_mu=m,train_sd=s))
        for k,v in dict(mu=mu,sd=sd,coef=coef,intercept=intercept,features=cols).items():
            model_params[f'{name}__{k}'] = np.asarray(v)
    out.mkdir(parents=True, exist_ok=False)
    single_summary, ablation_summary = pd.DataFrame(singles), pd.DataFrame(rows)
    for name, table in [('single_summary',single_summary),('single_daily',pd.concat(single_days)),
                        ('single_monthly',pd.concat(single_months)),('ablation_summary',ablation_summary),
                        ('ablation_daily',pd.concat(days)),('ablation_monthly',pd.concat(months)),
                        ('coefficients',pd.DataFrame(coefficients))]:
        table.to_csv(out/f'{name}.csv', index=False)
    tr[features].corr(method='pearson').to_csv(out/'train_pearson.csv')
    tr[features].corr(method='spearman').to_csv(out/'train_spearman.csv')
    predictions.to_parquet(out/'valid_predictions.parquet', index=False)
    np.savez(out/'models.npz', **model_params)
    pd.concat([tr[['date','code','label_end']].assign(split='train'),
               va[['date','code','label_end']].assign(split='valid')]).to_parquet(out/'samples.parquet',index=False)
    print('\nSingle features (direction fixed on training):')
    print(single_summary.to_string(index=False))
    print('\nLeave-one-out Ridge (fixed lambda=10; exploratory):')
    print(ablation_summary.to_string(index=False))
    return single_summary, ablation_summary


def read_development(path, year):
    import pyarrow as pa
    import pyarrow.parquet as pq
    dtype = pq.read_schema(path).field('date').type
    cutoff = f'{year}-12-31'
    if pa.types.is_timestamp(dtype):
        bound = pd.Timestamp(cutoff).to_pydatetime()
    elif pa.types.is_date(dtype):
        bound = pd.Timestamp(cutoff).date()
    elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        bound = cutoff
    else:
        raise ValueError(f'Unsupported date type: {dtype}')
    return pd.read_parquet(path, columns=['date','code','flag_tradable',base.LABEL,'label_end']+base.FEATURES,
                           engine='pyarrow',filters=[('date','<=',bound)])


def self_test():
    import tempfile
    rng = np.random.default_rng(31)
    dates = pd.bdate_range('2020-01-01','2023-02-01')
    d = pd.DataFrame({'date':np.tile(dates,25),'code':np.repeat(np.arange(25),len(dates)),
                      'flag_tradable':1})
    for f in base.FEATURES:
        d[f] = rng.normal(size=len(d))
    d[base.LABEL] = -.03*d.ret_1d + rng.normal(0,.001,len(d))
    # 反转验证期信号，检查模型不会依据验证结果翻转特征方向。
    d.loc[d.date >= '2022-01-01', base.LABEL] *= -1
    d['label_end'] = d.groupby('code').date.shift(-5)
    d.loc[d.label_end.isna(),base.LABEL] = np.nan
    assert train_direction(0.) == 1
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d.to_parquet(root/'data.parquet',index=False)
        frame = read_development(root/'data.parquet',2022)
        assert frame.date.max() <= pd.Timestamp('2022-12-31')
        s,a = analyze(frame,2022,root/'out')
        target = s.loc[s.feature.eq('ret_1d')].iloc[0]
        assert target.direction == -1 and target.valid_directed_ic < -.9
        assert len(a) == 8 and len(s) == 7
        assert a.iloc[0].delta_vs_baseline == 0
        baseline = a.iloc[0].valid_ic
        np.testing.assert_allclose(a.delta_vs_baseline,a.valid_ic-baseline,atol=1e-12)
        tr,va = base.prepare(frame,2022)
        mu,sd,w,b = base.ridge_fit(tr[base.FEATURES].to_numpy(),tr[base.LABEL].to_numpy(),10.)
        direct = base.daily_ic(va,va[base.LABEL],((va[base.FEATURES].to_numpy()-mu)/sd)@w+b)
        assert abs(direct.rank_ic.mean()-baseline)<1e-12
        assert (tr.label_end < pd.Timestamp('2022-01-01')).all()
        assert (va.label_end <= pd.Timestamp('2022-12-31')).all()
        try:
            analyze(frame,2022,root/'out')
        except FileExistsError:
            pass
        else:
            raise AssertionError('Overwrite guard failed')
    print('PASS: parquet input, train-only direction, reversed validation, baseline reproduction, paired deltas, boundary purge, overwrite guard')


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,default=root/'seq_features.parquet')
    p.add_argument('--valid-year',type=int,default=2022)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--self-test',action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.valid_year < 2021:
        p.error('valid-year must be >=2021')
    out = args.out or root/'runs'/f'feature_diagnostic_valid{args.valid_year}'
    if out.exists():
        p.error(f'Output exists: {out}; use a new --out')
    if not args.data.is_file():
        p.error(f'Missing data file: {args.data}')
    started = time.time()
    frame = read_development(args.data,args.valid_year)
    analyze(frame,args.valid_year,out)
    import pyarrow as pa
    manifest = dict(script_sha256=base.sha256(__file__),dependency_sha256=base.sha256(base.__file__),
                    data_sha256=base.sha256(args.data),valid_year=args.valid_year,reg_lambda=10.,
                    train_start='2020-01-01',train_end=f'{args.valid_year-1}-12-31',
                    features=base.FEATURES,versions={'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__},
                    note='Exploratory validation diagnosis, no test evaluation; no recursive deletion',
                    elapsed_seconds=time.time()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\nDone in {time.time()-started:.1f}s: {out.resolve()}')


if __name__ == '__main__':
    main()
