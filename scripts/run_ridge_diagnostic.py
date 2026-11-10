"""Ridge开发期诊断：比较原始特征、排名特征及两者组合。
快速Ridge诊断：仅训练/验证，不计算测试集表现。

在项目根目录查看运行参数：
python project.py run run_ridge_diagnostic --help

数据与运行说明：docs/PROJECT_GUIDE.md
实验定义与索引：docs/EXPERIMENT_INDEX.md
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ['ret_1d', 'ma5_dev', 'ma20_dev', 'log_vol_ratio',
            'turnover', 'amplitude', 'vol_20d']
RANKS = [f'{c}_csrank' for c in FEATURES]
LABEL = 'label_fwd5_ret'
GROUPS = {'raw7': FEATURES, 'rank7': RANKS, 'raw_plus_rank14': FEATURES + RANKS}


def prepare(frame, valid_year):
    start = pd.Timestamp(f'{valid_year}-01-01')
    end = pd.Timestamp(f'{valid_year}-12-31')
    d = frame.copy()
    d['date'] = pd.to_datetime(d['date'], errors='raise')
    d['label_end'] = pd.to_datetime(d['label_end'], errors='raise')
    d = d.loc[d.date <= end].sort_values(['code', 'date']).reset_index(drop=True)
    if d[['date', 'code']].isna().any().any() or d.duplicated(['date', 'code']).any():
        raise ValueError('Missing or duplicate date/code')
    if not d.flag_tradable.isin([0, 1]).all():
        raise ValueError('flag_tradable must be 0 or 1')
    d[FEATURES + [LABEL]] = d[FEATURES + [LABEL]].apply(pd.to_numeric, errors='raise')
    if np.isinf(d[FEATURES + [LABEL]].to_numpy()).any():
        raise ValueError('Infinite original feature/label: fix input data first')
    # 在标签可用性和历史窗口筛选之前计算截面排名。
    for col, target in zip(FEATURES, RANKS):
        eligible = d.flag_tradable.eq(1) & d[col].notna()
        g = d[col].where(eligible).groupby(d.date)
        ranks = g.rank(method='average', pct=True) - 0.5
        d[target] = ranks.where(eligible & g.transform('nunique').gt(1), 0).fillna(0)
    d[FEATURES] = d.groupby('code', sort=False)[FEATURES].ffill().fillna(0)
    eligible = (d.groupby('code', sort=False).cumcount().ge(59)
                & d.flag_tradable.eq(1) & d[LABEL].notna())
    if (eligible & (d.label_end.isna() | (d.label_end <= d.date))).any():
        raise ValueError('Eligible rows have invalid label_end')
    train = eligible & d.date.between('2020-01-01', start - pd.Timedelta(days=1)) & (d.label_end < start)
    valid = eligible & d.date.between(start, end) & (d.label_end <= end)
    tr, va = d.loc[train].copy(), d.loc[valid].copy()
    if tr.empty or va.empty:
        raise ValueError('Empty training/validation split')
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def ridge_fit(X, y, lam):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.)
    z = (X - mu) / sd
    intercept = float(y.mean())
    coef = np.linalg.solve(z.T @ z / len(y) + lam * np.eye(X.shape[1]),
                           z.T @ (y - intercept) / len(y))
    return mu, sd, coef, intercept


def daily_ic(meta, y, pred, min_stocks=20):
    d = meta[['date', 'code']].copy()
    d['y'], d['pred'] = y, pred
    g = d.groupby('date')
    ry, rp = g.y.rank(), g.pred.rank()
    yc = ry - ry.groupby(d.date).transform('mean')
    pc = rp - rp.groupby(d.date).transform('mean')
    numerator = (yc * pc).groupby(d.date).sum()
    denominator = np.sqrt((yc ** 2).groupby(d.date).sum() * (pc ** 2).groupby(d.date).sum())
    n = g.size()
    ic = numerator / denominator.replace(0, np.nan)
    ic.loc[g.pred.nunique().lt(2) & g.y.nunique().ge(2)] = 0.
    result = pd.DataFrame({'rank_ic': ic, 'n': n})
    result = result.loc[result.n >= min_stocks].reset_index()
    if not result.rank_ic.notna().any():
        raise ValueError('No evaluable validation dates with >=20 stocks and nonconstant labels')
    return result


def run(frame, year, lambdas, out):
    tr, va = prepare(frame, year)
    print(f'train={len(tr):,}; valid={len(va):,}; no test evaluation', flush=True)
    y, yv = tr[LABEL].to_numpy(), va[LABEL].to_numpy()
    rows, all_daily, all_monthly, saved = [], [], [], {}
    for variant, cols in GROUPS.items():
        X, V = tr[cols].to_numpy(dtype=float), va[cols].to_numpy(dtype=float)
        for lam in lambdas:
            mu, sd, coef, intercept = ridge_fit(X, y, lam)
            pred = (V - mu) / sd @ coef + intercept
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite predictions')
            daily = daily_ic(va, yv, pred)
            monthly = daily.assign(month=daily.date.dt.strftime('%Y-%m')).groupby('month').agg(
                rank_ic=('rank_ic', 'mean'), n_days=('rank_ic', 'count')).reset_index()
            row = dict(variant=variant, reg_lambda=lam, valid_ic=daily.rank_ic.mean(),
                       n_days=int(daily.rank_ic.count()), positive_day_ratio=float(daily.rank_ic.dropna().gt(0).mean()),
                       positive_months=int(monthly.rank_ic.gt(0).sum()), n_months=int(monthly.rank_ic.notna().sum()),
                       valid_mse=float(np.mean((pred-yv)**2)), n_features=len(cols))
            rows.append(row)
            all_daily.append(daily.assign(variant=variant, reg_lambda=lam))
            all_monthly.append(monthly.assign(variant=variant, reg_lambda=lam))
            prefix = f'{variant}__{lam:g}'
            for key, val in dict(mu=mu, sd=sd, coef=coef, intercept=intercept, features=cols).items():
                saved[f'{prefix}__{key}'] = np.asarray(val)
            print(f'{variant:16s} lambda={lam:g} valid_IC={row["valid_ic"]:+.6f}', flush=True)
    summary = pd.DataFrame(rows)
    selected = summary.sort_values(['variant','valid_ic','reg_lambda'], ascending=[True,False,False]).groupby('variant', sort=False).head(1).copy()
    base = float(selected.loc[selected.variant.eq('raw7'), 'valid_ic'].iloc[0])
    selected['delta_vs_selected_raw7'] = selected.valid_ic - base
    out.mkdir(parents=True, exist_ok=False)
    summary.to_csv(out/'summary.csv', index=False)
    selected.to_csv(out/'selected.csv', index=False)
    pd.concat(all_daily, ignore_index=True).to_csv(out/'daily_ic.csv', index=False)
    pd.concat(all_monthly, ignore_index=True).to_csv(out/'monthly_ic.csv', index=False)
    predictions = va[['date','code',LABEL,'label_end']].copy()
    for row in selected.itertuples():
        prefix = f'{row.variant}__{row.reg_lambda:g}'
        V = va[GROUPS[row.variant]].to_numpy(dtype=float)
        predictions[row.variant] = ((V-saved[prefix+'__mu'])/saved[prefix+'__sd']) @ saved[prefix+'__coef'] + saved[prefix+'__intercept']
    predictions.to_parquet(out/'selected_predictions.parquet', index=False)
    pd.concat([tr[['date','code','label_end']].assign(split='train'),
               va[['date','code','label_end']].assign(split='valid')]).to_parquet(out/'samples.parquet', index=False)
    np.savez(out/'models.npz', **saved)
    print('\nSelected on validation (exploratory; not out-of-sample evidence):')
    print(selected.to_string(index=False))
    return tr, va


def self_test():
    import tempfile
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2020-09-01','2022-12-30')
    frame = pd.DataFrame({'date':np.tile(dates,25), 'code':np.repeat(np.arange(25),len(dates)), 'flag_tradable':1})
    for c in FEATURES:
        frame[c] = rng.normal(size=len(frame))
    frame[LABEL] = .02*frame.ret_1d + rng.normal(0,.002,len(frame))
    frame['label_end'] = frame.groupby('code').date.shift(-5)
    frame.loc[frame.label_end.isna(),LABEL] = np.nan
    tr, va = prepare(frame,2022)
    assert (tr.label_end < pd.Timestamp('2022-01-01')).all()
    assert (va.label_end <= pd.Timestamp('2022-12-31')).all()
    changed = frame.copy()
    changed.loc[changed.date >= '2022-01-01', FEATURES] *= 100
    tr2, _ = prepare(changed,2022)
    pd.testing.assert_frame_equal(tr, tr2)
    # 修改标签不能改变两次运行共同保留样本的特征排名。
    changed_labels = frame.copy()
    changed_labels.loc[changed_labels.code.eq(0), LABEL] = np.nan
    t3, v3 = prepare(changed_labels,2022)
    for original, modified in [(tr,t3),(va,v3)]:
        expected=original.loc[original.code.ne(0)].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected[['date','code']+RANKS], modified[['date','code']+RANKS])
    X, y = tr[FEATURES].to_numpy(), tr[LABEL].to_numpy()
    mu, sd, coef, intercept = ridge_fit(X,y,.1)
    z = (X-mu)/sd
    np.testing.assert_allclose(z.T@(z@coef+intercept-y)/len(y)+.1*coef,0,atol=1e-12)
    perfect = daily_ic(va,va[LABEL],va[LABEL])
    np.testing.assert_allclose(perfect.rank_ic,1)
    flat = daily_ic(va,va[LABEL],np.zeros(len(va)))
    assert flat.rank_ic.eq(0).all()
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'result'
        run(frame,2022,[.001,.1,10.],out)
        assert len(pd.read_csv(out/'summary.csv'))==9
        assert len(pd.read_csv(out/'selected.csv'))==3
        assert set(pd.read_parquet(out/'samples.parquet').split)=={'train','valid'}
    print('PASS: full synthetic pipeline, training isolation, boundary purge, ridge equation, IC conventions, outputs')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root=Path(__file__).resolve().parent.parent
    p.add_argument('--data',type=Path,default=root/'seq_features.parquet')
    p.add_argument('--valid-year',type=int,default=2022)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.valid_year < 2021:
        p.error('Validation year must be >=2021')
    out=args.out or root/'runs'/f'ridge_diagnostic_valid{args.valid_year}'
    if out.exists():
        p.error(f'Output already exists: {out}; use a new --out directory')
    if not args.data.is_file():
        p.error(f'Missing {args.data}; use the existing project seq_features.parquet')
    started=time.time()
    import pyarrow.parquet as pq
    import pyarrow as pa
    dtype=pq.read_schema(args.data).field('date').type
    cutoff=f'{args.valid_year}-12-31'
    if pa.types.is_timestamp(dtype):
        bound=pd.Timestamp(cutoff).to_pydatetime()
    elif pa.types.is_date(dtype):
        bound=pd.Timestamp(cutoff).date()
    elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        bound=cutoff  # 源日期字符串应采用YYYY-MM-DD格式
    else:
        raise ValueError(f'Unsupported parquet date type: {dtype}')
    frame=pd.read_parquet(args.data,columns=['date','code','flag_tradable',LABEL,'label_end']+FEATURES,
                          engine='pyarrow',filters=[('date','<=',bound)])
    tr, va=run(frame,args.valid_year,[.001,.1,10.],out)
    manifest=dict(script_sha256=sha256(__file__), input_sha256=sha256(args.data),
                  input_path=str(args.data.resolve()), train_start='2020-01-01',
                  train_end=f'{args.valid_year-1}-12-31', valid_year=args.valid_year,
                  lambdas=[.001,.1,10.], train_n=len(tr), valid_n=len(va),
                  versions={'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__},
                  note='Validation-selected exploratory diagnostic; no test predictions; signal-day inputs only',
                  elapsed_seconds=time.time()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\nDone in {time.time()-started:.1f}s: {out.resolve()}')


if __name__=='__main__':
    main()
