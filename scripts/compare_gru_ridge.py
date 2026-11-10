"""在共同股票和日期上比较GRU与简单基线，核对标签并计算配对统计量。
比较已有GRU预测与Ridge：不训练，只重算同股票同日期Rank IC。

在项目根目录查看运行参数：
python project.py run compare_gru_ridge --help

数据与运行说明：docs/PROJECT_GUIDE.md
实验定义与索引：docs/EXPERIMENT_INDEX.md
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd


def indexed(frame, year, label):
    d=frame.copy()
    d['date']=pd.to_datetime(d['date'],errors='raise')
    if d['code'].isna().any():
        raise ValueError(f'{label}: missing stock code')
    d['code']=d['code'].astype(str)
    d=d.loc[d.date.dt.year.eq(year)].copy()
    if d.empty or d[['date','code']].isna().any().any() or d.duplicated(['date','code']).any():
        raise ValueError(f'{label}: empty/missing/duplicate keys')
    return d.set_index(['date','code']).sort_index()


def daily_ic(y,pred):
    dates=y.index.get_level_values('date')
    yr=y.groupby(dates).rank(method='average')
    pr=pred.groupby(dates).rank(method='average')
    yc=yr-yr.groupby(dates).transform('mean')
    pc=pr-pr.groupby(dates).transform('mean')
    num=(yc*pc).groupby(dates).sum()
    den=np.sqrt((yc**2).groupby(dates).sum()*(pc**2).groupby(dates).sum())
    ic=num/den.replace(0,np.nan)
    ic.loc[pred.groupby(dates).nunique().lt(2)&y.groupby(dates).nunique().ge(2)]=0.
    return ic


def inference(x):
    x=np.asarray(x,dtype=float);n=len(x)
    if n<30 or not np.isfinite(x).all():
        raise ValueError('Need >=30 finite daily paired differences')
    mean=float(x.mean());u=x-mean
    lr=float(u@u/n)
    for k in range(1,6):
        lr+=2*(1-k/6)*(u[k:]@u[:-k]/n)
    se=math.sqrt(max(lr,0)/n)
    p=math.erfc(abs(mean/se)/math.sqrt(2)) if se>0 else (1. if mean==0 else 0.)
    rng=np.random.default_rng(20261110)
    boot=[]
    for _ in range(20):
        starts=rng.integers(0,n-10+1,size=(500,(n+9)//10))
        indices=(starts[:,:,None]+np.arange(10)).reshape(500,-1)[:,:n]
        boot.extend(x[indices].mean(axis=1))
    boot=np.asarray(boot)
    low,high=np.quantile(boot,[.025,.975])
    pb=(1+np.sum(abs(boot-mean)>=abs(mean)))/(len(boot)+1)
    return dict(delta_ic=mean,n_days=n,hac_se=se,hac_p=p,bootstrap_low=low,bootstrap_high=high,bootstrap_p=pb)


def holm(p):
    values=np.asarray(p);order=np.argsort(values)
    result=np.empty(len(values));result[order]=np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return result


def compare(ridge, gru, year):
    r=indexed(ridge,year,'ridge')
    required=['label_fwd5_ret','raw7','drop_amplitude','negative_turnover']
    if not np.isfinite(r[required].to_numpy(float)).all():
        raise ValueError('Nonfinite Ridge labels/predictions')
    frames={};common=r.index
    for key,frame in gru.items():
        if 'split' not in frame:
            raise ValueError(f'{key}: missing split column')
        d=indexed(frame.loc[frame.split.eq('test')],year,key)
        if not np.isfinite(d[['score','y']].to_numpy(float)).all():
            raise ValueError(f'{key}: nonfinite labels/predictions')
        frames[key]=d;common=common.intersection(d.index)
    common=common.sort_values()
    if len(common)==0:
        raise ValueError('No shared samples; check stock-code formatting and year')
    counts=pd.Series(1,index=common).groupby(level='date').sum()
    dates=counts[counts>=20].index
    common=common[common.get_level_values('date').isin(dates)]
    # 对所有模型一致排除真实标签为常数的日期。
    yy=r.loc[common,'label_fwd5_ret']
    dates=yy.groupby(level='date').nunique()
    common=common[common.get_level_values('date').isin(dates[dates>=2].index)]
    coverage=[]
    for name,frame in {'ridge':r,**frames}.items():
        fraction=len(common)/len(frame)
        coverage.append(dict(source=name,original_rows=len(frame),common_rows=len(common),retained_fraction=fraction))
        if fraction<.95:
            raise ValueError(f'{name}: shared eligible sample retention {fraction:.1%} <95%; inspect sample differences first')
    y=r.loc[common,'label_fwd5_ret']
    for name,frame in frames.items():
        if not np.allclose(y.to_numpy(),frame.loc[common,'y'].to_numpy(),rtol=1e-5,atol=1e-7):
            raise ValueError(f'{name}: labels disagree on common stocks/dates')
    daily=pd.DataFrame({name:daily_ic(y,r.loc[common,name]) for name in required[1:]})
    seed_rows=[]
    for name,frame in frames.items():
        daily[name]=daily_ic(y,frame.loc[common,'score'])
        seed_rows.append(dict(model=name,rank_ic=daily[name].mean(),n_days=len(daily)))
    for mode in ['mse','rank']:
        cols=[name for name in frames if name.startswith(mode+'_seed')]
        if not cols:
            raise ValueError(f'Missing {mode} seeds')
        daily[f'gru_{mode}_seedmean']=daily[cols].mean(axis=1)
    results=[];paired=[]
    for mode in ['mse','rank']:
        model=f'gru_{mode}_seedmean'
        for baseline in required[1:]:
            delta=daily[model]-daily[baseline]
            results.append(dict(year=year,model=model,baseline=baseline,
                                gru_ic=daily[model].mean(),baseline_ic=daily[baseline].mean(),**inference(delta)))
            paired.append(pd.DataFrame({'date':daily.index,'model':model,'baseline':baseline,'delta_ic':delta.to_numpy()}))
    result=pd.DataFrame(results)
    result['hac_p_holm_6']=holm(result.hac_p)
    result['bootstrap_p_holm_6']=holm(result.bootstrap_p)
    summary=daily[['raw7','drop_amplitude','negative_turnover','gru_mse_seedmean','gru_rank_seedmean']].mean().rename('rank_ic').rename_axis('model').reset_index()
    summary['n_days']=len(daily)
    return result,daily,pd.DataFrame(seed_rows),pd.DataFrame(coverage),pd.concat(paired,ignore_index=True),common,summary


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def self_test():
    rng=np.random.default_rng(3)
    dates=pd.bdate_range('2024-01-01',periods=80)
    r=pd.DataFrame({'date':np.repeat(dates,25),'code':np.tile(np.arange(25).astype(str),80)})
    y=rng.normal(size=len(r))
    r['label_fwd5_ret']=y
    for name in ['raw7','drop_amplitude','negative_turnover']:
        r[name]=y+rng.normal(size=len(r))
    g={}
    for mode in ['mse','rank']:
        for seed in [0,1]:
            d=r[['date','code']].copy();d['split']='test';d['y']=y;d['score']=y+rng.normal(size=len(r))
            g[f'{mode}_seed{seed}']=d.sample(frac=1,random_state=seed)
    result,daily,_,coverage,_,_,summary=compare(r,g,2024)
    assert len(result)==6 and len(summary)==5 and (coverage.retained_fraction==1).all()
    np.testing.assert_allclose(daily.gru_mse_seedmean,(daily.mse_seed0+daily.mse_seed1)/2)
    np.testing.assert_allclose(result.delta_ic,result.gru_ic-result.baseline_ic,atol=1e-12)
    bad={k:v.copy() for k,v in g.items()};bad['mse_seed0'].loc[:,'y']+=1
    try:
        compare(r,bad,2024)
    except ValueError as e:
        assert 'labels disagree' in str(e)
    else:
        raise AssertionError('Label mismatch not rejected')
    assert inference(np.zeros(80))['hac_p']==1
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    print('PASS: alignment, seed-mean IC, paired differences, label guard, zero-difference test, Holm correction')


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--year',type=int,choices=[2023,2024,2025],default=2024)
    p.add_argument('--gru-root',type=Path)
    p.add_argument('--ridge-file',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test();return
    if len(set(args.seeds))!=len(args.seeds) or any(s<0 for s in args.seeds):
        p.error('Use distinct nonnegative seeds')
    gru_root=args.gru_root or (root/'runs'/'stage4_fixed' if args.year==2023 else root/'runs'/'stage5_crossyear'/f'test_{args.year}')
    ridge_path=args.ridge_file or root/'runs'/'ridge_crossyear_fixed'/f'test_{args.year}'/'predictions.parquet'
    out=args.out or root/'runs'/f'compare_gru_ridge_{args.year}'
    if out.exists():
        p.error(f'Output exists: {out}; use another --out')
    paths={f'{mode}_seed{s}':gru_root/f'{mode}_seed{s}'/'predictions.parquet' for mode in ['mse','rank'] for s in args.seeds}
    missing=[str(x) for x in [ridge_path,*paths.values()] if not x.is_file()]
    if missing:
        p.error('Missing prediction files (no training is launched):\n'+'\n'.join(missing)+'\nUse --gru-root to select the directory containing mse_seed*/rank_seed*.')
    started=time.time()
    ridge=pd.read_parquet(ridge_path)
    gru={name:pd.read_parquet(path) for name,path in paths.items()}
    result,daily,seeds,coverage,paired,common,summary=compare(ridge,gru,args.year)
    out.mkdir(parents=True,exist_ok=False)
    for name,table in [('comparison',result),('seed_summary',seeds),('coverage',coverage),('paired_daily',paired),('summary',summary)]:
        table.to_csv(out/f'{name}.csv',index=False)
    daily.to_csv(out/'daily_ic.csv',index_label='date')
    common.to_frame(index=False).to_parquet(out/'common_samples.parquet',index=False)
    manifest=dict(year=args.year,seeds=args.seeds,gru_root=str(gru_root.resolve()),script_sha256=digest(__file__),
                  inputs={str(path.resolve()):digest(path) for path in [ridge_path,*paths.values()]},
                  hac_lag=5,bootstrap_block=10,bootstrap_reps=10000,bootstrap_seed=20261110,
                  versions={'numpy':np.__version__,'pandas':pd.__version__},
                  note='Fixed-seed average daily IC; same-sample exploratory comparison; training provenance not audited')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nCommon-sample mean Rank IC (GRU = average daily IC across seeds):')
    print(summary.to_string(index=False))
    print('\nGRU minus baseline; positive delta favors GRU:')
    print(result.to_string(index=False))
    print(f'\nDone in {time.time()-started:.1f}s: {out.resolve()}')


if __name__=='__main__':
    main()
