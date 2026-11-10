"""从年度滚动实验的逐日IC生成配对统计结果。

数据格式、参数和运行方法见 docs/PROJECT_GUIDE.md。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True, choices=[2024, 2025, 2026])
args = parser.parse_args()
DIR = ROOT / "runs" / "stage5_crossyear" / f"test_{args.year}"

SEEDS = list(range(10))
HAC_LAG    = 5
BLOCK_SIZE = 10
BOOT_B     = 2000
BOOT_SEED  = 0
THRESHOLD  = 0.01


def load_mode(mode):
    dfs = []
    for seed in SEEDS:
        seed_dir = DIR / f"{mode}_seed{seed}"
        f = seed_dir / "daily_ic.csv"
        if not f.is_file() or not (seed_dir / "result.json").is_file():
            raise FileNotFoundError(f"Run incomplete: {seed_dir}; finish all 10 seeds first")
        d = pd.read_csv(f); d["seed"] = seed
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def get_test_ic(mode):
    df = load_mode(mode)
    return (df[df["split"] == "test"][["seed","date","rank_ic"]]
              .rename(columns={"rank_ic": f"{mode}_ic"}))


def hac_se_mean(x, lag):
    x = np.asarray(x, dtype=float); n = len(x)
    x_dm = x - x.mean()
    s = np.dot(x_dm, x_dm) / n
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        s += 2.0 * w * np.dot(x_dm[j:], x_dm[:-j]) / n
    return float(np.sqrt(max(s / n, 0.0)))


def block_bootstrap_p(x, block_size, B, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); n = len(x)
    nb = int(np.ceil(n / block_size))
    obs = x.mean(); xc = x - obs
    means = np.empty(B); nulls = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block_size + 1, size=nb)
        s_raw = np.concatenate([x[s:s+block_size]   for s in starts])[:n]
        s_ctr = np.concatenate([xc[s:s+block_size] for s in starts])[:n]
        means[b] = s_raw.mean(); nulls[b] = s_ctr.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float((np.abs(nulls) >= abs(obs)).mean())
    return lo, hi, p


a = get_test_ic("rank")
b = get_test_ic("mse")
for tag, frame in [("rank", a), ("mse", b)]:
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if set(frame["seed"]) != set(SEEDS):
        raise ValueError(f"{tag}: expected exactly seeds 0 through 9")
    if frame.duplicated(["seed", "date"]).any():
        raise ValueError(f"{tag}: duplicate seed/date")
    if not np.isfinite(frame[f"{tag}_ic"].to_numpy(dtype=float)).all():
        raise ValueError(f"{tag}: nonfinite IC")
    reference = set(frame.loc[frame["seed"] == 0, "date"])
    if not reference or any(set(frame.loc[frame["seed"] == s, "date"]) != reference for s in SEEDS[1:]):
        raise ValueError(f"{tag}: seed date sets differ")
m = a.merge(b, on=["seed","date"], how="outer", validate="one_to_one", indicator=True)
if not m["_merge"].eq("both").all():
    raise ValueError("rank/mse date sets differ")
m = m.drop(columns="_merge")
m["diff"] = m["rank_ic"] - m["mse_ic"]

daily = m.groupby("date")["diff"].mean().sort_index().values
n = len(daily); mean_d = float(daily.mean()); std_d = float(daily.std(ddof=1))

if n <= max(20, HAC_LAG) or not np.isfinite(daily).all():
    raise ValueError("Too few dates or nonfinite daily differences")
se  = hac_se_mean(daily, HAC_LAG)
if se <= 0:
    raise ValueError("Zero HAC SE; normal-approximation inference undefined")
t0  = mean_d / se
p0  = 2 * (1 - stats.norm.cdf(abs(t0)))
t_thr = (mean_d - THRESHOLD) / se
p_thr = 2 * (1 - stats.norm.cdf(abs(t_thr)))
lo, hi, p_boot = block_bootstrap_p(daily, BLOCK_SIZE, BOOT_B, BOOT_SEED)
by_seed = m.groupby("seed")["diff"].mean()

print("=" * 80)
print(f"Cross-year: rank - mse (test, {n} days, {len(by_seed)} seeds)")
print("=" * 80)
print(f"daily_diff_mean = {mean_d:+.4f}")
print(f"daily_diff_std  = {std_d:.4f}")
print(f"HAC(n) SE = {se:.4f},  t_zero = {t0:+.3f},  p_zero = {p0:.4f}")
print(f"t_threshold({THRESHOLD:+.2f}) = {t_thr:+.3f},  p_threshold = {p_thr:.4f}")
print(f"bootstrap 95% CI = [{lo:+.4f}, {hi:+.4f}],  p_boot = {p_boot:.4f}")
print("per-seed diff: " + "  ".join(f"seed{s}={v:+.5f}" for s, v in by_seed.items()))
print(f"positive_seeds = {(by_seed > 0).sum()}/{len(by_seed)}")
print(f"seed_diff_std = {by_seed.std(ddof=1):.6f}; median = {by_seed.median():+.6f}")
print(f"seed_signs_consistent = "
      f"{int((np.sign(by_seed.values) == np.sign(by_seed.values[0])).all())}")

print("\n判读（探索性分析；十个固定种子的平均表现）：")
print("  · 正均值、对零检验显著且区间高于零：支持固定种子平均差异为正。")
print("  · p_threshold 是 H0: mean=0.01 的双侧检验，不能直接作为超过阈值的单侧检验。")
print("  · 种子符号不一致 → 方向对初始化敏感")
print("  · 种子不同号不推翻平均差异显著；不显著也不代表等价。")
print("\n敏感性检查 (HAC_lag, block_size)：")
for lag in [3, 5, 10]:
    s = hac_se_mean(daily, lag)
    print(f"  HAC_lag={lag:2d}: SE={s:.4f}, 95% CI=[{mean_d-1.96*s:+.4f}, {mean_d+1.96*s:+.4f}]")
for bs in [5, 10, 20]:
    l_, h_, _ = block_bootstrap_p(daily, bs, 1000, BOOT_SEED)
    print(f"  boot_block={bs:2d}: 95% CI=[{l_:+.4f}, {h_:+.4f}]")
# 保存实际配对序列，便于复核；不改动训练产物。
out_dir = DIR / "comparison"
out_dir.mkdir(parents=True, exist_ok=True)
m.to_csv(out_dir / "paired_seed_daily.csv", index=False)
m.groupby("date")["diff"].mean().to_csv(out_dir / "daily_differences.csv")

seed_table = m.groupby("seed")[["mse_ic", "rank_ic", "diff"]].mean()
seed_table.to_csv(out_dir / "seed_summary.csv")
pd.DataFrame([{
    "n_days": n, "n_seeds": len(by_seed), "daily_diff_mean": mean_d,
    "daily_diff_std": std_d, "hac_se": se, "hac_p_two_sided": p0,
    "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "bootstrap_p": p_boot,
    "threshold_p_two_sided": p_thr, "positive_seeds": int((by_seed > 0).sum()),
    "seed_diff_std": by_seed.std(ddof=1), "seed_diff_median": by_seed.median(),
}]).to_csv(out_dir / "comparison_summary.csv", index=False)
print(f"Saved: {out_dir}")
