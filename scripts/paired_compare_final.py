from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent

DIRS = {
    "last": ROOT / "runs" / "stage2",
    "attn": ROOT / "runs" / "stage3_gru_attn",
    "pool": ROOT / "runs" / "stage3_gru_pool",
}

SEEDS = [0, 1, 2]
PAIRS = [
    ("attn", "last"),
    ("pool", "last"),
    ("attn", "pool"),
]

HAC_LAG = 5
BLOCK_SIZE = 10
BOOT_B = 10000
BOOT_SEED = 0
ALPHA = 0.05
REFERENCE_THRESHOLD = 0.01

OUT_DIR = ROOT / "runs" / "paired_compare_final"


def load_test_ic(tag):
    """读取各种子的测试期逐日IC，并检查唯一性和完整性。"""
    frames = []

    for seed in SEEDS:
        candidates = [
            DIRS[tag] / f"gru_seed{seed}" / "daily_ic.csv",
            DIRS[tag] / f"seed{seed}" / "daily_ic.csv",
        ]
        existing = [p for p in candidates if p.exists()]

        if len(existing) != 1:
            raise RuntimeError(
                f"{tag}, seed={seed}: 应找到且只找到一个 daily_ic.csv，"
                f"实际找到 {len(existing)} 个。\n候选路径：{candidates}"
            )

        path = existing[0]
        df = pd.read_csv(path)
        required = {"date", "split", "rank_ic"}

        if not required.issubset(df.columns):
            raise ValueError(f"{path}: 缺少列 {required - set(df.columns)}")

        df = df.loc[df["split"] == "test", ["date", "rank_ic"]].copy()

        if df.empty:
            raise ValueError(f"{path}: 没有 test 数据")

        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df["rank_ic"] = pd.to_numeric(df["rank_ic"], errors="raise")

        if df["date"].isna().any():
            raise ValueError(f"{path}: 日期存在缺失")

        if not np.isfinite(df["rank_ic"].to_numpy()).all():
            raise ValueError(f"{path}: IC存在NaN或无穷值")

        if (df["rank_ic"].abs() > 1 + 1e-8).any():
            raise ValueError(f"{path}: IC超出[-1, 1]")

        if df["date"].duplicated().any():
            raise ValueError(f"{path}: 存在重复日期")

        df["seed"] = seed
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_aligned_data():
    """要求所有模型、种子的测试日期完全一致，避免静默丢弃样本。"""
    data = {tag: load_test_ic(tag) for tag in DIRS}
    reference = pd.DatetimeIndex(
        data["last"].loc[
            data["last"]["seed"] == SEEDS[0], "date"
        ].sort_values()
    )

    for tag, df in data.items():
        for seed in SEEDS:
            dates = pd.DatetimeIndex(
                df.loc[df["seed"] == seed, "date"].sort_values()
            )
            if not dates.equals(reference):
                missing = reference.difference(dates)
                extra = dates.difference(reference)
                raise ValueError(
                    f"{tag}, seed={seed}: 测试日期不一致；"
                    f"缺少{len(missing)}天，额外{len(extra)}天"
                )

    return data


def hac_se_mean(x, lag):
    """均值的Newey-West标准误：Bartlett核，自协方差统一除以n。"""
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 2 or not 0 <= lag < n:
        raise ValueError(f"需要 n>=2 且 0<=lag<n；当前 n={n}, lag={lag}")
    if not np.isfinite(x).all():
        raise ValueError("HAC输入存在非有限值")

    centered = x - x.mean()
    long_run_var = np.dot(centered, centered) / n

    for j in range(1, lag + 1):
        weight = 1.0 - j / (lag + 1.0)
        gamma = np.dot(centered[j:], centered[:-j]) / n
        long_run_var += 2.0 * weight * gamma

    return float(np.sqrt(max(long_run_var, 0.0) / n))


def hac_stats(x, lag):
    mean = float(np.mean(x))
    se = hac_se_mean(x, lag)

    if se <= 0:
        raise ValueError("HAC标准误为0，无法进行常规推断")

    z = mean / se
    p = float(2.0 * stats.norm.sf(abs(z)))
    critical = stats.norm.ppf(1.0 - ALPHA / 2.0)

    return {
        "mean": mean,
        "se": se,
        "z": z,
        "p": p,
        "lo": mean - critical * se,
        "hi": mean + critical * se,
    }


def block_bootstrap(x, block_size, B=BOOT_B, seed=BOOT_SEED):
    """
    移动分块自助法：
    - 原始序列重采样得到百分位置信区间；
    - 中心化序列重采样得到对零双侧p值。
    两者不是严格互为反演的同一种检验，临界处可能略有不一致。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if not 1 <= block_size < n:
        raise ValueError("block_size必须满足1<=block_size<n")
    if B < 1:
        raise ValueError("B必须为正整数")

    rng = np.random.default_rng(seed)
    mean = float(x.mean())
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)
    boot_means = np.empty(B)

    for b in range(B):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        indices = (starts[:, None] + offsets).ravel()[:n]
        boot_means[b] = x[indices].mean()

    lo, hi = np.quantile(
        boot_means, [ALPHA / 2.0, 1.0 - ALPHA / 2.0]
    )

    # 使用同一组重采样索引时，中心化样本均值等于下面的差值。
    boot_null = boot_means - mean
    exceed = np.count_nonzero(np.abs(boot_null) >= abs(mean))

    # 有限次模拟修正，避免输出精确的0。
    p = float((exceed + 1) / (B + 1))

    return {"lo": float(lo), "hi": float(hi), "p": p}


def holm_adjust(p_values):
    """Holm多重比较校正；返回与原始输入顺序一致的调整后p值。"""
    p = np.asarray(p_values, dtype=float)

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p值必须是[0,1]内的有限数")

    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0

    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)

    return adjusted


def model_summary(data):
    rows = []

    for tag, df in data.items():
        per_seed = df.groupby("seed")["rank_ic"].mean()
        daily = df.groupby("date")["rank_ic"].mean()

        row = {
            "model": tag,
            "n_days_per_seed": len(daily),
            "mean_across_seeds": per_seed.mean(),
            "std_across_seeds": per_seed.std(ddof=1),
            "daily_mean_of_seeds": daily.mean(),
            "daily_std_of_seeds": daily.std(ddof=1),
            "ic_pos_ratio_of_daily_mean": (daily > 0).mean(),
        }
        row.update({f"seed{s}": per_seed.loc[s] for s in SEEDS})
        rows.append(row)

    columns = (
        ["model", "n_days_per_seed"]
        + [f"seed{s}" for s in SEEDS]
        + [
            "mean_across_seeds",
            "std_across_seeds",
            "daily_mean_of_seeds",
            "daily_std_of_seeds",
            "ic_pos_ratio_of_daily_mean",
        ]
    )
    return pd.DataFrame(rows)[columns]


def pair_differences(data, a, b):
    left = data[a].rename(columns={"rank_ic": "ic_a"})
    right = data[b].rename(columns={"rank_ic": "ic_b"})

    merged = left.merge(
        right,
        on=["seed", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged["diff"] = merged["ic_a"] - merged["ic_b"]

    daily = merged.groupby("date")["diff"].mean().sort_index()
    per_seed = merged.groupby("seed")["diff"].mean().reindex(SEEDS)

    return daily, per_seed


def comparison_tables(data):
    rows = []
    seed_rows = []
    daily_series = {}

    for a, b in PAIRS:
        name = f"{a} - {b}"
        daily, per_seed = pair_differences(data, a, b)
        daily_series[name] = daily
        x = daily.to_numpy()

        hac = hac_stats(x, HAC_LAG)
        boot = block_bootstrap(x, BLOCK_SIZE)
        signs = np.sign(per_seed.to_numpy())
        consistent = int(np.all(signs == signs[0]))

        rows.append({
            "comparison": name,
            "n_days": len(x),
            "daily_diff_mean": hac["mean"],
            "daily_diff_std": x.std(ddof=1),
            "se_hac": hac["se"],
            "z_hac": hac["z"],
            "p_zero_hac": hac["p"],
            "hac_ci_lo": hac["lo"],
            "hac_ci_hi": hac["hi"],
            "boot_ci_lo": boot["lo"],
            "boot_ci_hi": boot["hi"],
            "p_boot": boot["p"],
            "seed_signs_consistent": consistent,
        })

        seed_row = {
            "comparison": name,
            "seed_signs_consistent": consistent,
        }
        seed_row.update({
            f"diff_seed{s}": per_seed.loc[s] for s in SEEDS
        })
        seed_rows.append(seed_row)

    main = pd.DataFrame(rows)
    main["p_zero_holm"] = holm_adjust(main["p_zero_hac"])
    main["p_boot_holm"] = holm_adjust(main["p_boot"])

    return main, pd.DataFrame(seed_rows), daily_series


def sensitivity_table(daily_series):
    rows = []

    for name, daily in daily_series.items():
        x = daily.to_numpy()

        for lag in [3, 5, 10]:
            result = hac_stats(x, lag)
            rows.append({
                "comparison": name,
                "method": f"HAC_lag{lag}",
                "diff_mean": result["mean"],
                "lo_95": result["lo"],
                "hi_95": result["hi"],
                "se": result["se"],
                "p_raw": result["p"],
            })

        for block in [5, 10, 20]:
            result = block_bootstrap(x, block)
            rows.append({
                "comparison": name,
                "method": f"boot_block{block}",
                "diff_mean": x.mean(),
                "lo_95": result["lo"],
                "hi_95": result["hi"],
                "se": np.nan,
                "p_raw": result["p"],
            })

    result = pd.DataFrame(rows)

    # 每个参数设置内，对三项模型比较分别做Holm校正。
    result["p_holm_within_setting"] = (
        result.groupby("method")["p_raw"]
        .transform(lambda values: holm_adjust(values.to_numpy()))
    )
    return result


def print_table(title, frame):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(frame.to_string(
        index=False,
        float_format=lambda value: f"{value:.6g}",
    ))


def main():
    data = load_aligned_data()
    summary = model_summary(data)
    comparisons, seeds, daily_series = comparison_tables(data)
    sensitivity = sensitivity_table(daily_series)

    print_table("A. 单模型汇总（test）", summary)
    print(
        "\n注：daily指标先对同日种子IC取均值，再跨日统计；"
        "不是对集成预测重新计算IC。"
    )

    print_table(
        "B. 同日种子平均差值：HAC + 分块bootstrap + Holm校正",
        comparisons,
    )
    print_table("C. 逐种子差值：描述训练随机性", seeds)
    print_table("D. 参数敏感性检查", sensitivity)

    print("\n说明：")
    print("1. 主配置：HAC lag=5，bootstrap块长=10；不按最小p值挑参数。")
    print("2. HAC和bootstrap使用同一批数据，不是独立验证。")
    print("3. Holm分别校正三项HAC检验和三项bootstrap检验。")
    print("4. 区间是逐项区间，未进行多重比较校正。")
    print("5. 敏感性表的Holm校正在每个参数设置内进行，未覆盖所有设置。")
    print("6. 不显著不代表等价；种子符号不一致只说明方向对初始化敏感。")
    print("7. 当前推断针对这三个已训练种子的平均表现，不覆盖所有训练随机性。")
    print("8. 日期一致不保证股票样本一致；本比较以前序统一样本构建为前提。")
    print(
        f"9. ±{REFERENCE_THRESHOLD}仅为本轮参考阈值。"
        "对零显著且点估计超过阈值，不代表真实差异超过阈值。"
    )
    print("10. 2023年已经用于观察和选择方案，结果按探索性分析报告。")
    print(
        "11. Percentile区间与中心化bootstrap p值不是严格互为反演，"
        "临界处可能出现轻微不一致。"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_summary.csv": summary,
        "paired_comparisons.csv": comparisons,
        "seed_differences.csv": seeds,
        "sensitivity.csv": sensitivity,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT_DIR / filename, index=False, encoding="utf-8-sig")

    pd.DataFrame(daily_series).rename_axis("date").to_csv(
        OUT_DIR / "daily_differences.csv",
        encoding="utf-8-sig",
    )

    print(f"\n结果已保存：{OUT_DIR}")


if __name__ == "__main__":
    main()