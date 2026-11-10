"""GRU训练窗口诊断：扩展窗口与最近两年窗口在同一验证集上比较。
GRU训练窗口对照：只训练/验证，不调用测试集评价。

在项目根目录查看运行参数：
python project.py run run_window_diagnostic --help

数据与运行说明：docs/PROJECT_GUIDE.md
实验定义与索引：docs/EXPERIMENT_INDEX.md
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(daily, start, year, data):
    valid_start=pd.Timestamp(f'{year}-01-01')
    valid_end=pd.Timestamp(f'{year}-12-31')
    d=daily.loc[daily.date<=valid_end].copy()
    mu,sd=data.fit_scaler(d,start,f'{year-1}-12-31')
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<=0).any():
        raise ValueError('Invalid feature scaler')
    X,y,meta=data.make_windows(d,mu,sd)
    meta['date']=pd.to_datetime(meta.date)
    meta['label_end']=pd.to_datetime(meta.label_end)
    if meta.duplicated(['date','code']).any() or meta.label_end.isna().any() or (meta.label_end<=meta.date).any():
        raise ValueError('Duplicate samples or invalid label_end')
    masks={'train':meta.date.between(start,f'{year-1}-12-31') & (meta.label_end<valid_start),
           'valid':meta.date.between(valid_start,valid_end) & (meta.label_end<=valid_end)}
    splits={}
    for name,mask in masks.items():
        xx,yy=X[mask.to_numpy()],y[mask.to_numpy()]
        mm=meta.loc[mask].reset_index(drop=True)
        if len(yy)==0 or not np.isfinite(xx).all() or not np.isfinite(yy).all():
            raise ValueError(f'{name}: empty/nonfinite samples')
        splits[name]=(xx,yy,mm)
    return splits,mu,sd


def summarize(out,seeds):
    pairs=[]
    for seed in seeds:
        a=pd.read_csv(out/'expanding'/f'seed{seed}'/'daily_ic.csv')
        b=pd.read_csv(out/'recent2y'/f'seed{seed}'/'daily_ic.csv')
        z=a.merge(b,on=['date','split'],how='outer',suffixes=('_expanding','_recent2y'),validate='one_to_one',indicator=True)
        if not z['_merge'].eq('both').all() or not z.n_expanding.eq(z.n_recent2y).all():
            raise ValueError('Validation dates/counts differ')
        if not z.rank_ic_expanding.isna().equals(z.rank_ic_recent2y.isna()):
            raise ValueError('Non-evaluable dates differ')
        z['delta_ic']=z.rank_ic_recent2y-z.rank_ic_expanding
        z['seed']=seed
        pairs.append(z[['seed','date','rank_ic_expanding','rank_ic_recent2y','delta_ic']])
    paired=pd.concat(pairs,ignore_index=True)
    paired.to_csv(out/'paired_valid_daily.csv',index=False)
    cols=['rank_ic_expanding','rank_ic_recent2y','delta_ic']
    per_seed=paired.groupby('seed')[cols].mean().reset_index()
    per_seed.to_csv(out/'comparison_by_seed.csv',index=False)
    daily=paired.groupby('date')[cols].mean().reset_index()
    daily['month']=pd.to_datetime(daily.date).dt.strftime('%Y-%m')
    daily.groupby('month')[cols].mean().to_csv(out/'monthly_comparison.csv')
    summary=pd.DataFrame([dict(valid_ic_expanding=daily.rank_ic_expanding.mean(),
                               valid_ic_recent2y=daily.rank_ic_recent2y.mean(),
                               delta_ic=daily.delta_ic.mean(),n_days=int(daily.delta_ic.count()),
                               n_seeds=len(seeds),positive_seeds=int(per_seed.delta_ic.gt(0).sum()))])
    summary.to_csv(out/'comparison.csv',index=False)
    print('\nRecent two years minus expanding (validation-selected, exploratory):')
    print(per_seed.to_string(index=False))
    print(summary.to_string(index=False))


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--valid-year',type=int,default=2023)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--loss',choices=['rank','mse'],default='rank')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.valid_year not in [2023,2024] or args.epochs<1 or len(set(args.seeds))!=len(args.seeds) or any(s<0 for s in args.seeds):
        p.error('Use valid-year 2023/2024, positive epochs and distinct nonnegative seeds')
    import torch
    import run_crossyear as engine
    import _crossyear_data as data
    if list(data.FEATURES)!=['ret_1d','ma5_dev','ma20_dev','log_vol_ratio','turnover','amplitude','vol_20d'] or data.WINDOW!=60:
        raise ValueError('Unexpected data configuration')
    if not data.SEQ_PATH.is_file():
        raise FileNotFoundError(f'Use existing feature cache: {data.SEQ_PATH}')
    engine.CONFIG['epochs']=args.epochs
    out=args.out or root/'runs'/f'window_diagnostic_valid{args.valid_year}_{args.loss}'
    manifest=dict(valid_year=args.valid_year,seeds=args.seeds,loss=args.loss,config=engine.CONFIG,
                  data_sha256=digest(data.SEQ_PATH),script_sha256=digest(__file__),
                  engine_sha256=digest(engine.__file__),data_code_sha256=digest(data.__file__),
                  versions={'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__},
                  threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                  windows={'expanding':'2020-01-01','recent2y':f'{args.valid_year-2}-01-01'},
                  note='Early stopping and comparison share validation; no independent test performance')
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:
            raise ValueError('Settings/data/code changed; use a new --out')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Nonempty output without manifest; use a new --out')
        out.mkdir(parents=True,exist_ok=True)
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    daily=data.load_daily()
    if daily.duplicated(['date','code']).any():
        raise ValueError('Duplicate source date/code')
    daily=daily.loc[daily.date<=pd.Timestamp(f'{args.valid_year}-12-31')].copy()
    if daily.loc[daily.date.dt.year.eq(args.valid_year),'date'].dt.month.nunique()!=12:
        raise ValueError('Validation year lacks 12 months of source data')
    engine.check_ranking_loss()
    validation_reference=None
    rows=[]
    for variant,start in manifest['windows'].items():
        splits,mu,sd=prepare(daily,start,args.valid_year,data)
        if validation_reference is None:
            validation_reference=(splits['valid'][1].copy(),splits['valid'][2].copy())
        else:
            np.testing.assert_array_equal(validation_reference[0],splits['valid'][1])
            pd.testing.assert_frame_equal(validation_reference[1],splits['valid'][2])
        print(f'\n{variant}: train={len(splits["train"][1]):,}, valid={len(splits["valid"][1]):,}',flush=True)
        ymu=float(splits['train'][1].mean());ysd=float(splits['train'][1].std())
        if not np.isfinite(ysd) or ysd<=0:
            raise ValueError('Invalid training label standard deviation')
        for seed in args.seeds:
            directory=out/variant/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            done=directory/'result.json'
            if done.exists():
                required=['history.csv','metrics.csv','predictions.parquet','daily_ic.csv','checkpoint.pt']
                if not all((directory/f).is_file() for f in required):
                    raise ValueError(f'Incomplete cached run: {directory}')
                result=json.loads(done.read_text(encoding='utf-8'))
                if result['variant']!=variant or result['seed']!=seed:
                    raise ValueError('Cached identity mismatch')
                print(f'[skip] {variant} seed={seed}',flush=True)
            else:
                engine.set_seed(seed)
                model=engine.GRUPool(len(data.FEATURES),hidden=engine.CONFIG['gru_hidden'],dropout=engine.CONFIG['dropout']).to(engine.DEVICE)
                print(f'[run] {variant} {args.loss} seed={seed}',flush=True)
                best,elapsed=engine.fit(model,args.loss,seed,splits['train'],splits['valid'],ymu,ysd,directory)
                metrics=engine.evaluate_and_save(model,args.loss,{'valid':splits['valid']},ymu,ysd,directory)
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                features=data.FEATURES,window=data.WINDOW,scaler_mu=mu,scaler_sd=sd,
                                label_mu=ymu,label_std=ysd,config=engine.CONFIG,seed=seed,loss=args.loss,
                                train_start=start,train_end=f'{args.valid_year-1}-12-31',valid_year=args.valid_year,best_epoch=best),directory/'checkpoint.pt')
                result=dict(variant=variant,seed=seed,loss=args.loss,best_epoch=best,
                            elapsed_sec=elapsed,valid_ic=metrics['valid']['rank_ic'])
                done.write_text(json.dumps(result,indent=2),encoding='utf-8')
                del model
            rows.append(result)
            pd.DataFrame(rows).to_csv(out/'run_summary.csv',index=False)
        del splits
        gc.collect()
    summarize(out,args.seeds)
    print(f'\nDone: {out.resolve()}')


if __name__=='__main__':
    main()
