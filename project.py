"""项目统一入口：数据导入、特征构建、实验运行与结果核验。

数据格式、参数和运行方法见 docs/PROJECT_GUIDE.md。
"""
from pathlib import Path
import argparse
import csv
import subprocess
import sys
import shutil
import tempfile
import os

def import_data(source, kind):
    source=source.expanduser().resolve()
    target=ROOT/('seq_features.parquet' if kind=='features' else 'klines.pkl')
    if not source.is_file():
        raise FileNotFoundError(source)
    if kind=='features':
        import pandas as pd
        import numpy as np
        frame=pd.read_parquet(source)
        columns=['date','code','flag_tradable','label_fwd5_ret','label_end',
                 'ret_1d','ma5_dev','ma20_dev','log_vol_ratio','turnover','amplitude','vol_20d']
        missing=set(columns)-set(frame.columns)
        if missing:raise ValueError(f'缺少特征字段： {sorted(missing)}')
        dates=pd.to_datetime(frame.date,errors='raise')
        ends=pd.to_datetime(frame.label_end,errors='raise')
        if frame.empty or dates.isna().any() or frame.code.isna().any():
            raise ValueError('数据为空，或日期/股票代码缺失')
        if pd.DataFrame({'date':dates,'code':frame.code}).duplicated().any():
            raise ValueError('日期与股票代码重复')
        if not frame.flag_tradable.isin([0,1]).all():
            raise ValueError('flag_tradable必须取0或1')
        if np.isinf(frame[columns[3:4]+columns[5:]].to_numpy(dtype=float)).any():
            raise ValueError('特征或标签包含无穷值')
        labeled=frame.label_fwd5_ret.notna()
        if (labeled & (ends.isna() | (ends<=dates))).any():
            raise ValueError('有效标签的结束日期不合法')
        print(f'特征格式检查通过： {len(frame):,}行，{frame.code.nunique()}只股票，{dates.min().date()}..{dates.max().date()}')
    elif (ROOT/'seq_features.parquet').exists():
        raise ValueError('特征缓存已存在。导入不同原始行情前，请先备份并移除旧缓存。')
    if source==target:
        print(f'数据已在目标位置： {target}');return
    if target.exists():
        raise FileExistsError(f'{target}已存在；替换前请先备份。')
    fd,tmp=tempfile.mkstemp(prefix='.data_import_',dir=ROOT);os.close(fd)
    try:
        shutil.copyfile(source,tmp)
        # 独占创建目标文件，避免静默覆盖已有缓存。
        with open(tmp,'rb') as inp, target.open('xb') as dest:shutil.copyfileobj(inp,dest)
    finally:
        os.unlink(tmp)
    print(f'已导入： {target}')
    if kind=='klines':print('下一步：python project.py build-features')

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='查看实验状态、路径和脚本')
    sub.add_parser('report',help='查看最终结果表，不训练模型')
    sub.add_parser('verify',help='核验已保存结果，不训练模型')
    sub.add_parser('build-features',help='从可信的klines.pkl构建特征缓存，不训练模型')
    data=sub.add_parser('data',help='导入特征缓存或可信原始行情，不训练模型')
    choice=data.add_mutually_exclusive_group(required=True)
    choice.add_argument('--features',type=Path)
    choice.add_argument('--klines',type=Path)
    run=sub.add_parser('run',help='运行指定脚本，具体用法见docs/PROJECT_GUIDE.md')
    run.add_argument('script',choices=sorted(p.stem for p in (ROOT/'scripts').glob('*.py') if not p.name.startswith('_')))
    run.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.command=='data':
        import_data(args.features or args.klines,'features' if args.features else 'klines')
    elif args.command=='build-features':
        if (ROOT/'seq_features.parquet').exists():
            parser.error('seq_features.parquet已存在；重新构建前请先备份并移除')
        if not (ROOT/'klines.pkl').is_file():
            parser.error('请先导入行情：python project.py data --klines 文件路径')
        sys.path.insert(0,str(ROOT/'scripts'))
        from _crossyear_data import build_seq_features
        build_seq_features()
    elif args.command=='list':
        with (ROOT/'docs/experiment_index.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):print(f"{row['id']:16s} {row['status']:22s} {row['result_path']}")
    elif args.command=='report':
        for name in ['formal_gru.csv','common_sample_models.csv','ridge_crossyear.csv','A_2023.csv','window_validation.csv']:
            print('\n'+name)
            print((ROOT/'results/final'/name).read_text(encoding='utf-8-sig').strip())
    else:
        script='verify_results' if args.command=='verify' else args.script
        rest=[] if args.command=='verify' else args.arguments
        if rest[:1]==['--']:rest=rest[1:]
        # 历史脚本没有参数解析器，收到帮助参数时不能直接启动实验。
        no_parser={'run_stage1','run_stage3','run_stage3b','run_stage4_10seeds',
                   'compare_stage4_10seeds','paired_compare_final','rebuild_stage2_daily_ic','verify_results'}
        if script in no_parser and rest:
            if rest==['--help']:
                print(f'{script}不接收命令行参数；输入输出说明见docs/PROJECT_GUIDE.md。')
                return
            parser.error(f'{script}不接收命令行参数')
        raise SystemExit(subprocess.call([sys.executable,str(ROOT/'scripts'/f'{script}.py'),*rest],cwd=ROOT))

if __name__=='__main__':main()
