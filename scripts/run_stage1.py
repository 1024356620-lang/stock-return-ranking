"""第一阶段：使用多层感知机检查数据、训练和评价流程。

数据格式、参数和运行方法见 docs/PROJECT_GUIDE.md。
"""

import pickle
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


# 1. 配置常量
ROOT = Path(__file__).resolve().parent.parent
KLINES_PATH = ROOT / "klines.pkl"
SEQ_PATH    = ROOT / "seq_features.parquet"
OUT_DIR     = ROOT / "runs" / "stage1_mlp"

FEATURES = ["ret_1d","ma5_dev","ma20_dev","log_vol_ratio",
            "turnover","amplitude","vol_20d"]
WINDOW = 60
LABEL  = "label_fwd5_ret"

# 第一折：train 2020-2021 | valid 2022 | test 2023
TRAIN_START, TRAIN_END = "2020-01-01", "2021-12-31"
VALID_START, VALID_END = "2022-01-01", "2022-12-31"
TEST_START,  TEST_END  = "2023-01-01", "2023-12-31"

CFG = dict(hidden=256, dropout=0.2, batch_size=512,
           lr=1e-3, weight_decay=1e-4, epochs=30,
           patience=5, grad_clip=1.0, seed=0)



# 2. 数据构建
def build_seq_features():
    print(f"[data] 从 {KLINES_PATH.name} 构建日度特征 ...")
    with open(KLINES_PATH, "rb") as f:
        raw = pickle.load(f)
    d = pd.concat([df.assign(code=code) for code, df in raw.items()],
                  ignore_index=True)
    d = d.rename(columns={"turn": "turnover", "pctChg": "pct_chg"})
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["code", "date"]).reset_index(drop=True)

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
    print(f"[data] 写入 {SEQ_PATH.name}: {len(d):,} 行")


def load_daily():
    if not SEQ_PATH.exists():
        build_seq_features()
    d = pd.read_parquet(SEQ_PATH)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["code","date"]).reset_index(drop=True)
    return d


# 3. 窗口切分与标准化器（只在训练期拟合）
def fit_scaler(daily, train_start, train_end):
    m = (daily["date"] >= train_start) & (daily["date"] <= train_end)
    mu = daily.loc[m, FEATURES].mean().values.astype(np.float32)
    sd = daily.loc[m, FEATURES].std().replace(0, 1.0).values.astype(np.float32)
    return mu, sd


def make_windows(daily, mu, sd):
    X_list, y_list, dates, codes, ends = [], [], [], [], []
    for code, g in daily.groupby("code", sort=False):
        g = g.reset_index(drop=True)
        arr = ((g[FEATURES].ffill().fillna(0.0)).values - mu) / sd
        labels = g[LABEL].values
        end_dates = g["label_end"].values
        for t in range(WINDOW - 1, len(g)):
            if np.isnan(labels[t]):
                continue
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
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]



# 4. 三个模型（阶段 1 只用 MLP；GRU/Attn 为后续阶段预留）
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



# 5. 训练 / 评估
def train_one_epoch(model, loader, opt, loss_fn, clip):
    model.train(); total, n = 0.0, 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        if torch.isnan(loss):
            raise RuntimeError("loss is NaN — 检查 lr / 特征缩放")
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
        outs.append(model(torch.from_numpy(X[i:i + batch])).numpy())
    return np.concatenate(outs)


def fit(model, X_tr, y_tr, X_va, y_va, out_dir):
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    tr_loader = DataLoader(SeqDataset(X_tr, y_tr),
                           batch_size=CFG["batch_size"], shuffle=True)
    va_loader = DataLoader(SeqDataset(X_va, y_va),
                           batch_size=CFG["batch_size"], shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                           weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, CFG["epochs"] + 1):
        tr = train_one_epoch(model, tr_loader, opt, loss_fn, CFG["grad_clip"])
        va = evaluate_loss(model, va_loader, loss_fn)
        sched.step(va)
        hist.append({"epoch": ep, "train_loss": tr, "valid_loss": va})
        print(f"  epoch {ep:02d} | train={tr:.5f} valid={va:.5f}")
        if va < best - 1e-6:
            best = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= CFG["patience"]:
                print(f"  early stop at epoch {ep}"); break
    model.load_state_dict(best_state)
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    return model, pd.DataFrame(hist)


def rank_ic_per_day(pred, y, meta):
    d = meta.copy(); d["pred"] = pred; d["y"] = y
    ics = []
    for _, g in d.groupby("date"):
        if len(g) < 20: continue
        rho, _ = spearmanr(g["pred"], g["y"])
        if not np.isnan(rho): ics.append(rho)
    return (float(np.mean(ics)) if ics else np.nan), len(ics)


def report(name, pred, y, meta):
    ic, nd = rank_ic_per_day(pred, y, meta)
    m, z = float(np.mean((pred - y) ** 2)), float(np.mean(y ** 2))
    print(f"  {name:6s} | RankIC={ic:+.4f} (n={nd}) "
          f"MSE={m:.5f} zeroMSE={z:.5f} pred_std={pred.std():.5f}")
    return {"split": name, "rank_ic": ic, "n_days": nd, "mse": m,
            "zero_mse": z, "pred_std": float(pred.std())}



# 6. 主入口
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[stage1] 输出目录: {OUT_DIR}")

    daily = load_daily()
    print(f"[data] {len(daily):,} 行, {daily['code'].nunique()} 股")

    mu, sd = fit_scaler(daily, TRAIN_START, TRAIN_END)
    print("[scaler] " + ", ".join(
        f"{f}={m:.3f}±{s:.3f}" for f, m, s in zip(FEATURES, mu, sd)))

    X_all, y_all, meta = make_windows(daily, mu, sd)
    va_start = pd.Timestamp(VALID_START); te_start = pd.Timestamp(TEST_START)
    tr_mask = ((meta["date"] >= TRAIN_START) & (meta["date"] <= TRAIN_END) &
               (meta["label_end"] < va_start)).values
    va_mask = ((meta["date"] >= VALID_START) & (meta["date"] <= VALID_END) &
               (meta["label_end"] < te_start)).values
    te_mask = ((meta["date"] >= TEST_START) & (meta["date"] <= TEST_END)).values

    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_va, y_va = X_all[va_mask], y_all[va_mask]
    X_te, y_te = X_all[te_mask], y_all[te_mask]
    m_va = meta[va_mask].reset_index(drop=True)
    m_te = meta[te_mask].reset_index(drop=True)
    print(f"[split] train={len(y_tr):,} valid={len(y_va):,} test={len(y_te):,}")

    model = MLP(WINDOW, len(FEATURES), CFG["hidden"], CFG["dropout"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] mlp params={n_params:,}")

    model, hist = fit(model, X_tr, y_tr, X_va, y_va, OUT_DIR)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")

    rows, preds = [], {}
    for name, X, y, m in [("valid", X_va, y_va, m_va),
                          ("test", X_te, y_te, m_te)]:
        p = predict(model, X); preds[name] = p
        rows.append(report(name, p, y, m))
    pd.DataFrame(rows).to_csv(OUT_DIR / "metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["valid_loss"], label="valid")
    axes[0].set_title("loss"); axes[0].legend()
    axes[1].hist(preds["valid"], bins=50, alpha=0.7)
    axes[1].set_title(f"valid pred (std={preds['valid'].std():.5f})")
    axes[2].hist(preds["test"], bins=50, alpha=0.7)
    axes[2].set_title(f"test pred (std={preds['test'].std():.5f})")
    plt.tight_layout(); plt.savefig(OUT_DIR / "stage1.png", dpi=120); plt.close()

    # 塌缩检查
    for name in ["valid", "test"]:
        if preds[name].std() < 0.1 * y_all.std():
            print(f"  [警告] {name} 预测 std={preds[name].std():.5f} "
                  f"远小于标签 std={y_all.std():.5f}，可能塌缩")
    print(f"[done] 结果保存在 {OUT_DIR}")


if __name__ == "__main__":
    main()