"""第三阶段：使用均值池化聚合GRU序列状态。

数据格式、参数和运行方法见 docs/PROJECT_GUIDE.md。
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 复用 run_stage2 的数据/训练原语（主程序有入口保护，导入时不会执行训练）
from run_stage2 import (
    ROOT, CFG, SEEDS, FEATURES, WINDOW,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
    load_daily, fit_scaler, make_windows, SeqDataset,
    train_one_epoch, evaluate_loss, rank_ic_per_day,
)

OUT_BASE = ROOT / "runs" / "stage3_gru_pool"


# ============================================================
# 新增模型：GRU 与均值池化
# ============================================================
class GRUPool(nn.Module):
    """形状说明：B为批次样本数，H为隐藏维数。
        out, _ = GRU(x)  ->  out: (B, 60, H)
    ctx = out.mean(dim=1)  ->  (B, H)
    pred = head(drop(ctx))  ->  (B,)"""
    def __init__(self, n_feat, hidden=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)                  # (B, 60, H)
        ctx = out.mean(dim=1)                 # (B, H)
        return self.head(self.drop(ctx)).squeeze(-1)


# ============================================================
# 训练（与 stage2/3 相同，唯一区别是不需要 return_attn）
# ============================================================
def fit_pool(model, X_tr, y_tr, X_va, y_va, out_dir):
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


@torch.no_grad()
def predict_only(model, X, batch=1024):
    model.eval(); outs = []
    for i in range(0, len(X), batch):
        outs.append(model(torch.from_numpy(X[i:i+batch])).numpy())
    return np.concatenate(outs)


def daily_ic_df(pred, y, meta, split):
    d = meta.copy(); d["pred"] = pred; d["y"] = y
    rows = []
    for dt, g in d.groupby("date"):
        if len(g) < 20: continue
        rho, _ = spearmanr(g["pred"], g["y"])
        if not np.isnan(rho):
            rows.append({"date": dt, "split": split,
                         "rank_ic": float(rho), "n": int(len(g))})
    return pd.DataFrame(rows)


# ============================================================
# 单次运行
# ============================================================
def run_gru_pool(seed, X_all, y_all, meta, mu, sd):
    out_dir = OUT_BASE / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[gru_pool] seed={seed}")

    # 切分（与 stage2/3 完全一致）
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

    # 标签标准化（训练集拟合）
    y_mu = float(y_tr_raw.mean())
    y_std = float(y_tr_raw.std())
    y_tr = (y_tr_raw - y_mu) / y_std
    y_va = (y_va_raw - y_mu) / y_std
    y_te = (y_te_raw - y_mu) / y_std
    print(f"    label: mu={y_mu:.5f} std={y_std:.5f}")

    # 模型（种子在创建前设）
    CFG["seed"] = seed
    torch.manual_seed(seed); np.random.seed(seed)
    model = GRUPool(len(FEATURES), CFG["gru_hidden"], CFG["dropout"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    n_params={n_params:,}  train={len(y_tr):,} "
          f"valid={len(y_va):,} test={len(y_te):,}")

    model, hist, best_epoch, elapsed = fit_pool(model, X_tr, y_tr, X_va, y_va, out_dir)

    # 预测
    p_norm_va = predict_only(model, X_va)
    p_norm_te = predict_only(model, X_te)
    p_raw_va = p_norm_va * y_std + y_mu
    p_raw_te = p_norm_te * y_std + y_mu

    ic_va, nd_va = rank_ic_per_day(p_raw_va, y_va_raw, m_va)
    ic_te, nd_te = rank_ic_per_day(p_raw_te, y_te_raw, m_te)
    mse_va  = float(np.mean((p_raw_va - y_va_raw) ** 2))
    mse_te  = float(np.mean((p_raw_te - y_te_raw) ** 2))
    zero_va = float(np.mean(y_va_raw ** 2))
    zero_te = float(np.mean(y_te_raw ** 2))
    print(f"    valid | RankIC={ic_va:+.4f} (n={nd_va}) "
          f"MSE={mse_va:.5f} zeroMSE={zero_va:.5f} pred_std={p_raw_va.std():.5f}")
    print(f"    test  | RankIC={ic_te:+.4f} (n={nd_te}) "
          f"MSE={mse_te:.5f} zeroMSE={zero_te:.5f} pred_std={p_raw_te.std():.5f}")

    # 逐样本预测
    pred_df = pd.concat([
        pd.DataFrame({"date": m_va["date"].values, "code": m_va["code"].values,
                      "split": "valid", "pred": p_raw_va, "y": y_va_raw}),
        pd.DataFrame({"date": m_te["date"].values, "code": m_te["code"].values,
                      "split": "test",  "pred": p_raw_te, "y": y_te_raw}),
    ], ignore_index=True)
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)

    # 逐日 IC
    ic_df = pd.concat([
        daily_ic_df(p_raw_va, y_va_raw, m_va, "valid"),
        daily_ic_df(p_raw_te, y_te_raw, m_te, "test"),
    ], ignore_index=True)
    ic_df.to_csv(out_dir / "daily_ic.csv", index=False)

    # 训练曲线 + 预测分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["valid_loss"], label="valid")
    axes[0].set_title("loss (normalized)"); axes[0].legend()
    axes[1].hist(p_raw_va, bins=50, alpha=0.7)
    axes[1].set_title(f"valid pred (std={p_raw_va.std():.5f})")
    axes[2].hist(p_raw_te, bins=50, alpha=0.7)
    axes[2].set_title(f"test pred (std={p_raw_te.std():.5f})")
    plt.tight_layout(); plt.savefig(out_dir / "curves.png", dpi=120); plt.close()

    # 模型检查点
    torch.save({
        "state_dict": model.state_dict(),
        "features": FEATURES, "window": WINDOW,
        "scaler_mu": mu, "scaler_sd": sd,
        "label_mu": y_mu, "label_std": y_std,
        "model_name": "gru_pool", "n_params": n_params,
        "config": CFG, "seed": seed,
        "best_epoch": best_epoch, "elapsed_sec": elapsed,
        "split": {"train": (TRAIN_START, TRAIN_END),
                  "valid": (VALID_START, VALID_END),
                  "test":  (TEST_START, TEST_END)},
    }, out_dir / "checkpoint.pt")

    return {
        "model": "gru_pool", "seed": seed, "n_params": n_params,
        "best_epoch": best_epoch, "elapsed_sec": elapsed,
        "valid_rank_ic": ic_va, "valid_mse": mse_va,
        "test_rank_ic": ic_te, "test_mse": mse_te,
        "test_pred_std": float(p_raw_te.std()),
    }


# ============================================================
# 主入口
# ============================================================
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"[config] out={OUT_BASE}  seeds={SEEDS}")

    daily = load_daily()
    mu, sd = fit_scaler(daily, TRAIN_START, TRAIN_END)
    X_all, y_all, meta = make_windows(daily, mu, sd)
    print(f"[windows] 总样本 {len(y_all):,}")

    summary = []
    for seed in SEEDS:
        summary.append(run_gru_pool(seed, X_all, y_all, meta, mu, sd))

    df = pd.DataFrame(summary)
    df.to_csv(OUT_BASE / "summary.csv", index=False)

    print("\n===== GRU + Mean Pooling 汇总 =====")
    cols = ["seed","n_params","best_epoch","elapsed_sec",
            "valid_rank_ic","test_rank_ic","test_mse","test_pred_std"]
    print(df[cols].to_string(index=False))
    print("\n===== 与种子均值 =====")
    print(df[["test_rank_ic","test_mse","test_pred_std","best_epoch"]]
            .agg(["mean","std"]).round(4).to_string())
    print(f"\n[done] 结果保存在 {OUT_BASE}")


if __name__ == "__main__":
    main()