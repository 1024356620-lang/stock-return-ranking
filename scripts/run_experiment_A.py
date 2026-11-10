"""截面排名特征实验：原始七特征与附加七个排名特征的配对对照。
比较原始七特征与附加七个当日截面排名特征。

在项目根目录查看运行参数：
python project.py run run_experiment_A --help

数据与运行说明：docs/PROJECT_GUIDE.md
实验定义与索引：docs/EXPERIMENT_INDEX.md
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURES = ["ret_1d", "ma5_dev", "ma20_dev", "log_vol_ratio",
                 "turnover", "amplitude", "vol_20d"]
RANK_FEATURES = [f"{c}_csrank" for c in BASE_FEATURES]


def add_cross_sectional_ranks(daily):
    """此处不使用标签，保留行顺序和全部原始列。"""
    d = daily.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    if d[["date", "code"]].isna().any().any():
        raise ValueError("Missing date/code")
    if d.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate date/code")
    for feature, new_col in zip(BASE_FEATURES, RANK_FEATURES):
        v = pd.to_numeric(d[feature], errors="raise")
        eligible = d["flag_tradable"].eq(1) & np.isfinite(v)
        values = v.where(eligible)
        grouped = values.groupby(d["date"], sort=False)
        ranks = grouped.rank(method="average", pct=True) - 0.5
        # 全体取值相同或只有一只股票的截面不提供相对位置信息。
        informative = grouped.transform("nunique").gt(1)
        d[new_col] = ranks.where(eligible & informative, 0.0).fillna(0.0).astype("float32")
    return d


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_arrays(X, y, meta, year):
    m = meta.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["label_end"] = pd.to_datetime(m["label_end"])
    valid_start = pd.Timestamp(f"{year-1}-01-01")
    test_start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    masks = {
        "train": m.date.between("2020-01-01", f"{year-2}-12-31") & (m.label_end < valid_start),
        "valid": m.date.between(valid_start, f"{year-1}-12-31") & (m.label_end < test_start),
        "test": m.date.between(test_start, end) & (m.label_end <= end),
    }
    splits = {}
    for name, mask in masks.items():
        xx, yy = X[mask.to_numpy()], y[mask.to_numpy()]
        mm = m.loc[mask].reset_index(drop=True)
        if not len(yy) or not np.isfinite(xx).all() or not np.isfinite(yy).all():
            raise ValueError(f"{name}: empty or non-finite samples")
        if mm.duplicated(["date", "code"]).any():
            raise ValueError(f"{name}: duplicate samples")
        if ((mm.label_end <= mm.date) | mm.label_end.isna()).any():
            raise ValueError(f"{name}: invalid label_end")
        splits[name] = (xx, yy, mm)
    return splits


def prepare(daily, year, variant, data_module):
    # 原始特征的标准化参数沿用固定实验代码的拟合方式。
    original = data_module.FEATURES
    try:
        data_module.FEATURES = BASE_FEATURES
        mu, sd = data_module.fit_scaler(daily, "2020-01-01", f"{year-2}-12-31")
        if not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any():
            raise ValueError("Invalid original feature scaler")
        frame = daily
        if variant == "plus_csrank":
            frame = add_cross_sectional_ranks(daily)
            data_module.FEATURES = BASE_FEATURES + RANK_FEATURES
            mu = np.r_[mu, np.zeros(len(RANK_FEATURES), dtype=np.float32)]
            sd = np.r_[sd, np.ones(len(RANK_FEATURES), dtype=np.float32)]
        X, y, meta = data_module.make_windows(frame, mu, sd)
    finally:
        data_module.FEATURES = original
    return split_arrays(X, y, meta, year), mu, sd


def paired_summary(year_dir, modes, seeds):
    paired = []
    for mode in modes:
        for seed in seeds:
            a = pd.read_csv(year_dir / "baseline" / f"{mode}_seed{seed}" / "daily_ic.csv")
            b = pd.read_csv(year_dir / "plus_csrank" / f"{mode}_seed{seed}" / "daily_ic.csv")
            key = ["split", "date"]
            z = a.merge(b, on=key, suffixes=("_baseline", "_A"),
                        how="outer", validate="one_to_one", indicator=True)
            if not z["_merge"].eq("both").all() or not z.n_baseline.eq(z.n_A).all():
                raise ValueError("Evaluation dates/counts differ between variants")
            if not z.rank_ic_baseline.isna().equals(z.rank_ic_A.isna()):
                raise ValueError("NaN IC date sets differ")
            z["delta_ic"] = z.rank_ic_A - z.rank_ic_baseline
            z["mode"], z["seed"] = mode, seed
            paired.append(z[["mode", "seed", "split", "date", "rank_ic_baseline", "rank_ic_A", "delta_ic"]])
    p = pd.concat(paired, ignore_index=True)
    p.to_csv(year_dir / "paired_daily.csv", index=False)
    s = p.groupby(["mode", "seed", "split"])[["rank_ic_baseline", "rank_ic_A", "delta_ic"]].mean().reset_index()
    s.to_csv(year_dir / "paired_seed_summary.csv", index=False)
    # 先在每日对种子取平均，不能把种子与日期的组合当作独立样本。
    day = p.groupby(["mode", "split", "date"])[["rank_ic_baseline", "rank_ic_A", "delta_ic"]].mean().reset_index()
    day.to_csv(year_dir / "seed_averaged_daily.csv", index=False)
    rows = []
    for (mode, split), g in day.groupby(["mode", "split"]):
        ss = s.loc[s["mode"].eq(mode) & s["split"].eq(split)]
        rows.append(dict(mode=mode, split=split, baseline_ic=g.rank_ic_baseline.mean(),
                         A_ic=g.rank_ic_A.mean(), delta_ic=g.delta_ic.mean(),
                         n_days=int(g.delta_ic.notna().sum()), n_seeds=len(seeds),
                         positive_seeds=int(ss.delta_ic.gt(0).sum())))
    result = pd.DataFrame(rows)
    result.to_csv(year_dir / "comparison.csv", index=False)
    print(result.to_string(index=False), flush=True)


def self_test():
    d = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 5 + ["2024-01-03"] * 5),
                      "code": list("abcde") * 2, "flag_tradable": [1, 1, 1, 0, 1] * 2})
    for f in BASE_FEATURES:
        d[f] = [1., 2., 2., 999., np.nan, 10., 20., 30., -999., np.nan]
    d["label_fwd5_ret"] = np.nan  # 未来标签缺失不能影响当日特征排名所使用的股票集合。
    a = add_cross_sectional_ranks(d)
    np.testing.assert_allclose(a[RANK_FEATURES[0]].iloc[:5], [-1/6, 1/3, 1/3, 0, 0], atol=1e-7)
    pd.testing.assert_frame_equal(a[d.columns], d)
    future = d.copy()
    future.loc[future.date.eq(pd.Timestamp("2024-01-03")), BASE_FEATURES] = -500.
    pd.testing.assert_frame_equal(a.iloc[:5], add_cross_sectional_ranks(future).iloc[:5])
    altered_labels = d.assign(label_fwd5_ret=np.arange(len(d)))
    pd.testing.assert_frame_equal(a[RANK_FEATURES], add_cross_sectional_ranks(altered_labels)[RANK_FEATURES])
    same = d.copy()
    same[BASE_FEATURES] = 1.
    assert (add_cross_sectional_ranks(same)[RANK_FEATURES] == 0).all().all()
    shuffled = add_cross_sectional_ranks(d.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(a.sort_values(["date", "code"]).reset_index(drop=True),
                                  shuffled.sort_values(["date", "code"]).reset_index(drop=True))
    try:
        add_cross_sectional_ranks(pd.concat([d, d.iloc[:1]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate key not rejected")
    print("PASS: ties, missingness, tradability, constant columns, row order, future/label independence, duplicates")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", nargs="+", type=int, default=[2024])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--modes", nargs="+", choices=["mse", "rank"], default=["mse", "rank"])
    p.add_argument("--epochs", type=int, default=None, help="仅用于独立目录的试跑；正式保持原配置")
    p.add_argument("--out", default="runs/experiment_A_csrank")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if (any(y not in [2023, 2024, 2025] for y in args.years)
            or len(set(args.years)) != len(args.years)
            or any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.modes)) != len(args.modes)
            or (args.epochs is not None and args.epochs < 1)):
        p.error("Use distinct years 2023/2024/2025, distinct nonnegative seeds/modes, positive epochs")
    # 延迟导入训练依赖，自检只需要numpy和pandas。
    import run_crossyear as engine
    import _crossyear_data as data
    if list(data.FEATURES) != BASE_FEATURES or data.WINDOW != 60:
        raise ValueError("Unexpected project feature/window configuration; review adapter first")
    if args.epochs is not None:
        engine.CONFIG["epochs"] = args.epochs
    engine.check_ranking_loss()
    output = Path(args.out)
    if not output.is_absolute():
        output = data.ROOT / output
    source = data.load_daily()
    if source.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate date/code in source")
    if not np.isfinite(source[BASE_FEATURES].replace(np.nan, 0).to_numpy()).all():
        raise ValueError("Original features contain infinity: fix upstream before comparing")
    hashes = {"data_sha256": digest(data.SEQ_PATH), "adapter_sha256": digest(__file__),
              "engine_sha256": digest(engine.__file__), "data_code_sha256": digest(data.__file__)}
    import torch
    import scipy
    for year in args.years:
        year_dir = output / f"test_{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"year": year, "seeds": args.seeds, "modes": args.modes,
                    "config": engine.CONFIG, "features": BASE_FEATURES,
                    "new_features": RANK_FEATURES, "rank_rule": "same_date_tradable_finite_average_pct_minus_half_constant_zero",
                    "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                 "torch": torch.__version__, "scipy": scipy.__version__}, **hashes}
        path = year_dir / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"Configuration/data changed. Use another --out: {path}")
        elif any(year_dir.iterdir()):
            raise ValueError(f"Nonempty output lacks manifest: {year_dir}")
        else:
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        engine.TRAIN_START, engine.TRAIN_END = "2020-01-01", f"{year-2}-12-31"
        engine.VALID_START, engine.VALID_END = f"{year-1}-01-01", f"{year-1}-12-31"
        engine.TEST_START, engine.TEST_END = f"{year}-01-01", f"{year}-12-31"
        daily = source.loc[source.date <= pd.Timestamp(engine.TEST_END)].copy()
        if daily.date.max() < pd.Timestamp(engine.TEST_START):
            raise ValueError(f"No data in {year}")
        references = {}
        rows = []
        for variant in ["baseline", "plus_csrank"]:
            print(f"\n=== {year} {variant} ===", flush=True)
            splits, mu, sd = prepare(daily, year, variant, data)
            for split, (X, y, meta) in splits.items():
                if variant == "baseline":
                    references[split] = (y.copy(), meta.copy())
                else:
                    np.testing.assert_array_equal(y, references[split][0])
                    pd.testing.assert_frame_equal(meta, references[split][1])
                print(f"{split}: {X.shape}", flush=True)
            engine.FEATURES = BASE_FEATURES if variant == "baseline" else BASE_FEATURES + RANK_FEATURES
            engine.OUT_BASE = year_dir / variant
            for mode in args.modes:
                for seed in args.seeds:
                    result = engine.run_one(mode, seed, splits, mu, sd)
                    rows.append(dict(result, year=year, variant=variant))
                    pd.DataFrame(rows).to_csv(year_dir / "summary.csv", index=False)
            del splits, X, y, meta
            gc.collect()
        paired_summary(year_dir, args.modes, args.seeds)
    print(f"\nDone: {output}\nExploratory comparison only; positive delta is not proof of significance.")


if __name__ == "__main__":
    main()
