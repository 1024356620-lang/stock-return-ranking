"""年度滚动实验的数据处理与模型定义。特征缓存和原始行情位于项目根目录。

数据格式、参数和运行方法见 docs/PROJECT_GUIDE.md。
"""

import os, time, argparse, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. 配置
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
KLINES_PATH = ROOT / "klines.pkl"
SEQ_PATH    = ROOT / "seq_features.parquet"

FEATURES = ["ret_1d","ma5_dev","ma20_dev","log_vol_ratio",
            "turnover","amplitude","vol_20d"]
WINDOW = 60
LABEL  = "label_fwd5_ret"

TRAIN_START, TRAIN_END = "2020-01-01", "2021-12-31"
VALID_START, VALID_END = "2022-01-01", "2022-12-31"
TEST_START,  TEST_END  = "2023-01-01", "2023-12-31"

CFG = dict(
    mlp_hidden=256, gru_hidden=64,
    dropout=0.2, batch_size=512,
    lr=1e-3, weight_decay=1e-4,
    epochs=50, patience=10,     # 早停耐心轮数设为10。
    grad_clip=1.0, seed=0,
)
SEEDS = [0, 1, 2]

# 运行时容器（避免全局变量）
RT = {"shuffle": False, "out_base": ROOT / "runs" / "stage2"}


# ============================================================
# 2. 数据构建
# ============================================================
def build_seq_features():
    print(f"[data] 构建 {SEQ_PATH.name} ...")
    with open(KLINES_PATH, "rb") as f:
        raw = pickle.load(f)
    d = pd.concat([df.assign(code=code) for code, df in raw.items()],
                  ignore_index=True)
    d = d.rename(columns={"turn":"turnover","pctChg":"pct_chg"})
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["code","date"]).reset_index(drop=True)
    g = d.groupby("code", group_keys=False)
    d["ret_1d"] = g["close"].pct_change()
    for n in (5, 20):
        ma = g["close"].transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
        d[f"ma{n}_dev"] = d["close"] / ma - 1
    d["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=12).mean())
    d["log_vol_ratio"] = np.log((d["volume"] / d["vol_ma20"]).clip(lower=1e-3))
    d["amplitude"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["vol_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=15).std())
    FWD = 5
    d[LABEL] = g["close"].transform(lambda s: s.shift(-FWD)) / d["close"] - 1
    d["label_end"] = g["date"].transform(lambda s: s.shift(-FWD))
    d["flag_tradable"] = ((d["volume"] > 0) & (d["amount"] > 0)).astype("int8")
    keep = ["code","date","flag_tradable",LABEL,"label_end"] + FEATURES
    d[keep].to_parquet(SEQ_PATH, index=False)
    print(f"[data] {SEQ_PATH.name}: {len(d):,} 行")


def load_daily():
    if not SEQ_PATH.exists():
        build_seq_features()
    d = pd.read_parquet(SEQ_PATH)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["code","date"]).reset_index(drop=True)


def fit_scaler(daily, s, e):
    m = (daily["date"] >= s) & (daily["date"] <= e)
    mu = daily.loc[m, FEATURES].mean().values.astype(np.float32)
    sd = daily.loc[m, FEATURES].std().replace(0, 1.0).values.astype(np.float32)
    return mu, sd


def make_windows(daily, mu, sd):
    """窗口末日（t 日）必须 flag_tradable==1；窗口内部保留停牌行，用前向填充处理"""
    X_list, y_list, dates, codes, ends = [], [], [], [], []
    for code, g in daily.groupby("code", sort=False):
        g = g.reset_index(drop=True)
        arr = ((g[FEATURES].ffill().fillna(0.0)).values - mu) / sd
        labels = g[LABEL].values
        end_dates = g["label_end"].values
        tradable = g["flag_tradable"].values
        for t in range(WINDOW - 1, len(g)):
            if np.isnan(labels[t]): continue
            if tradable[t] != 1: continue        # 窗口末日必须当天可交易
            X_list.append(arr[t - WINDOW + 1:t + 1])
            y_list.append(labels[t])
            dates.append(g["date"].iloc[t])
            codes.append(code)
            ends.append(end_dates[t])
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame({"date": dates, "code": codes, "label_end": ends})
    return X, y, meta


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ============================================================
# 3. 模型
# ============================================================
class MLP(nn.Module):
    def __init__(self, window, n_feat, hidden=256, dropout=0.2):
        super().__init__()
        in_dim = window * n_feat
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, n_feat, hidden=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


class GRUAttn(nn.Module):
    def __init__(self, n_feat, hidden=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.score = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, return_attn=False):
        out, _ = self.gru(x)
        w = torch.softmax(self.score(out).squeeze(-1), dim=1)
        ctx = (out * w.unsqueeze(-1)).sum(dim=1)
        pred = self.head(self.drop(ctx)).squeeze(-1)
        return (pred, w) if return_attn else pred


def build_model(name, window, n_feat):
    if name == "mlp":
        return MLP(window, n_feat, CFG["mlp_hidden"], CFG["dropout"])
    if name == "gru":
        return GRUModel(n_feat, CFG["gru_hidden"], CFG["dropout"])
    if name == "gru_attn":
        return GRUAttn(n_feat, CFG["gru_hidden"], CFG["dropout"])
    raise ValueError(name)


# ============================================================
# 4. 训练 / 评估
# ============================================================
def train_one_epoch(model, loader, opt, loss_fn, clip):
    model.train(); total, n = 0.0, 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        if torch.isnan(loss):
            raise RuntimeError("loss is NaN")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item() * len(yb); n += len(yb)
    return total / n


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn):
    model.eval(); total, n = 0.0, 0
    for xb, yb in loader:
        total += loss_fn(model(xb), yb).item() * len(yb); n += len(yb)
    return total / n


@torch.no_grad()
def predict(model, X, batch=1024):
    model.eval(); outs = []
    for i in range(0, len(X), batch):
        outs.append(model(torch.from_numpy(X[i:i+batch])).numpy())
    return np.concatenate(outs)


def fit(model, X_tr, y_tr, X_va, y_va, out_dir):
    """返回 (model, hist_df, best_epoch, elapsed_sec)。
    注意：y_tr / y_va 必须是归一化后的标签。"""
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    tr_loader = DataLoader(SeqDataset(X_tr, y_tr),
                           batch_size=CFG["batch_size"], shuffle=True)
    va_loader = DataLoader(SeqDataset(X_va, y_va),
                           batch_size=CFG["batch_size"], shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                           weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    best, best_state, best_epoch, bad, hist = np.inf, None, 0, 0, []
    t0 = time.time()
    for ep in range(1, CFG["epochs"] + 1):
        tr = train_one_epoch(model, tr_loader, opt, loss_fn, CFG["grad_clip"])
        va = evaluate_loss(model, va_loader, loss_fn)
        sched.step(va)
        hist.append({"epoch": ep, "train_loss": tr, "valid_loss": va})
        print(f"    epoch {ep:02d} | train={tr:.5f} valid={va:.5f}")
        if va < best - 1e-6:
            best, best_epoch, bad = va, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                print(f"    early stop at epoch {ep}")
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    return model, pd.DataFrame(hist), best_epoch, elapsed


def rank_ic_per_day(pred, y, meta):
    d = meta.copy(); d["pred"] = pred; d["y"] = y
    ics = []
    for _, g in d.groupby("date"):
        if len(g) < 20: continue
        rho, _ = spearmanr(g["pred"], g["y"])
        if not np.isnan(rho): ics.append(rho)
    return (float(np.mean(ics)) if ics else np.nan), len(ics)


def report(name, pred_raw, y_raw, meta):
    ic, nd = rank_ic_per_day(pred_raw, y_raw, meta)
    m, z = float(np.mean((pred_raw - y_raw) ** 2)), float(np.mean(y_raw ** 2))
    print(f"    {name:6s} | RankIC={ic:+.4f} (n={nd}) "
          f"MSE={m:.5f} zeroMSE={z:.5f} pred_std={pred_raw.std():.5f}")
    return {"split": name, "rank_ic": ic, "n_days": nd, "mse": m,
            "zero_mse": z, "pred_std": float(pred_raw.std())}


# ============================================================
# 5. 单次运行
# ============================================================
def run_one(model_name, seed, X_all, y_all, meta, mu, sd):
    out_dir = RT["out_base"] / f"{model_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[run] {model_name} seed={seed} shuffle={RT['shuffle']}")

    # 切分
    va_start = pd.Timestamp(VALID_START); te_start = pd.Timestamp(TEST_START)
    tr_mask = ((meta["date"] >= TRAIN_START) & (meta["date"] <= TRAIN_END)
               & (meta["label_end"] < va_start)).values
    va_mask = ((meta["date"] >= VALID_START) & (meta["date"] <= VALID_END)
               & (meta["label_end"] < te_start)).values
    te_mask = ((meta["date"] >= TEST_START) & (meta["date"] <= TEST_END)).values

    X_tr, y_tr_raw = X_all[tr_mask], y_all[tr_mask]
    X_va, y_va_raw = X_all[va_mask], y_all[va_mask]
    X_te, y_te_raw = X_all[te_mask], y_all[te_mask]
    m_va = meta[va_mask].reset_index(drop=True)
    m_te = meta[te_mask].reset_index(drop=True)

    # 可选：时间打乱（每次运行使用相同的随机数种子 0，保证可复现）
    if RT["shuffle"]:
        rng = np.random.default_rng(0)
        def shuffle_time(X):
            idx = np.argsort(rng.random((len(X), WINDOW)), axis=1)
            return np.take_along_axis(X, idx[..., None], axis=1)
        X_tr, X_va, X_te = shuffle_time(X_tr), shuffle_time(X_va), shuffle_time(X_te)
        print("    [shuffle] 时间维度已打乱")

    # 标签标准化（用训练集拟合）
    y_mu = float(y_tr_raw.mean())
    y_std = float(y_tr_raw.std())
    y_tr = (y_tr_raw - y_mu) / y_std
    y_va = (y_va_raw - y_mu) / y_std
    y_te = (y_te_raw - y_mu) / y_std
    print(f"    label: mu={y_mu:.5f} std={y_std:.5f}")

    # 创建模型（种子在创建之前设）
    CFG["seed"] = seed
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(model_name, WINDOW, len(FEATURES))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    n_params={n_params:,}  train={len(y_tr):,} "
          f"valid={len(y_va):,} test={len(y_te):,}")

    # 训练（用归一化标签）
    model, hist, best_epoch, elapsed = fit(model, X_tr, y_tr, X_va, y_va, out_dir)

    # 评估（恢复原始量纲）
    rows = []
    preds_raw = {}
    for name, X, y_raw, m in [
        ("valid", X_va, y_va_raw, m_va),
        ("test",  X_te, y_te_raw, m_te),
    ]:
        p_norm = predict(model, X)
        p_raw = p_norm * y_std + y_mu
        preds_raw[name] = p_raw
        rows.append(report(name, p_raw, y_raw, m))
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    # 完整模型检查点
    torch.save({
        "state_dict": model.state_dict(),
        "features": FEATURES, "window": WINDOW,
        "scaler_mu": mu, "scaler_sd": sd,
        "label_mu": y_mu, "label_std": y_std,     # ← 存标签标准化参数
        "model_name": model_name, "n_params": n_params,
        "config": CFG, "seed": seed,
        "best_epoch": best_epoch, "elapsed_sec": elapsed,
        "shuffle_time": RT["shuffle"],
        "split": {"train": (TRAIN_START, TRAIN_END),
                  "valid": (VALID_START, VALID_END),
                  "test":  (TEST_START, TEST_END)},
    }, out_dir / "checkpoint.pt")

    # 训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["valid_loss"], label="valid")
    axes[0].set_title("loss (normalized)"); axes[0].legend()
    axes[1].hist(preds_raw["valid"], bins=50, alpha=0.7)
    axes[1].set_title(f"valid pred (std={preds_raw['valid'].std():.5f})")
    axes[2].hist(preds_raw["test"], bins=50, alpha=0.7)
    axes[2].set_title(f"test pred (std={preds_raw['test'].std():.5f})")
    plt.tight_layout(); plt.savefig(out_dir / "curves.png", dpi=120); plt.close()

    return {
        "model": model_name, "seed": seed, "shuffle_time": RT["shuffle"],
        "n_params": n_params, "best_epoch": best_epoch, "elapsed_sec": elapsed,
        "train_n": len(y_tr), "valid_n": len(y_va), "test_n": len(y_te),
        "label_mu": y_mu, "label_std": y_std,
        "valid_rank_ic": rows[0]["rank_ic"],
        "valid_mse": rows[0]["mse"],
        "test_rank_ic": rows[1]["rank_ic"],
        "test_mse": rows[1]["mse"],
        "test_pred_std": rows[1]["pred_std"],
    }


# ============================================================
# 6. 主入口
# ============================================================
def main(shuffle_time):
    RT["shuffle"] = shuffle_time
    RT["out_base"] = ROOT / "runs" / ("stage2_shuffled" if shuffle_time else "stage2")
    RT["out_base"].mkdir(parents=True, exist_ok=True)
    print(f"[config] shuffle={shuffle_time}  out={RT['out_base']}  seeds={SEEDS}")

    daily = load_daily()
    print(f"[data] {len(daily):,} 行, {daily['code'].nunique()} 股")

    mu, sd = fit_scaler(daily, TRAIN_START, TRAIN_END)
    print("[scaler] " + ", ".join(
        f"{f}={m:.3f}±{s:.3f}" for f, m, s in zip(FEATURES, mu, sd)))

    X_all, y_all, meta = make_windows(daily, mu, sd)
    print(f"[windows] 总样本 {len(y_all):,}")

    summary = []
    for model_name in ["mlp", "gru"]:
        for seed in SEEDS:
            summary.append(run_one(model_name, seed, X_all, y_all, meta, mu, sd))

    df = pd.DataFrame(summary)
    df.to_csv(RT["out_base"] / "summary.csv", index=False)

    print(f"\n===== 汇总（shuffle={shuffle_time}） =====")
    agg = df.groupby("model").agg(
        n_params=("n_params", "first"),
        valid_rank_ic_mean=("valid_rank_ic", "mean"),
        valid_rank_ic_std=("valid_rank_ic", "std"),
        test_rank_ic_mean=("test_rank_ic", "mean"),
        test_rank_ic_std=("test_rank_ic", "std"),
        test_mse_mean=("test_mse", "mean"),
        test_pred_std_mean=("test_pred_std", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        elapsed_mean=("elapsed_sec", "mean"),
    ).round(4)
    print(agg.to_string())
    print(f"\n[done] 结果保存在 {RT['out_base']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", action="store_true",
                        help="打乱窗口内时间顺序（对照实验）")
    args = parser.parse_args()
    main(shuffle_time=args.shuffle)