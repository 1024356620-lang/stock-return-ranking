# ============================================================
# run_stage3.py
# Stage 3: GRU + Attention
# 复用 run_stage2.py 的数据/训练/模型/评估逻辑
# 新增：逐样本预测、逐日 IC、完整注意力权重、平均注意力曲线
# 输出：runs/stage3_gru_attn/（不覆盖 stage2）
# 用法：python run_stage3.py
# ============================================================
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 复用 run_stage2.py（main 有 if __name__ guard，import 安全）
from run_stage2 import (
    ROOT, CFG, SEEDS, FEATURES, WINDOW,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
    load_daily, fit_scaler, make_windows, SeqDataset,
    build_model, train_one_epoch, evaluate_loss, rank_ic_per_day,
)

OUT_BASE = ROOT / "runs" / "stage3_gru_attn"


# ============================================================
# 训练循环（与 stage2 相同，只是多返回一步）
# ============================================================
def fit_gru_attn(model, X_tr, y_tr, X_va, y_va, out_dir):
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    tr_loader = DataLoader(SeqDataset(X_tr, y_tr),
                           batch_size=CFG["batch_size"], shuffle=True)
    va_loader = DataLoader(SeqDataset(X_va, y_va),
                           batch_size=CFG["batch_size"], shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                           weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = torch.nn.MSELoss()

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


# ============================================================
# 带注意力的预测
# ============================================================
@torch.no_grad()
def predict_with_attn(model, X, batch=1024):
    """返回 (preds, attns)，形状 (N,), (N, WINDOW)"""
    model.eval()
    preds, attns = [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i+batch])
        p, w = model(xb, return_attn=True)
        preds.append(p.numpy())
        attns.append(w.numpy())
    return np.concatenate(preds), np.concatenate(attns)


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
def run_gru_attn(seed, X_all, y_all, meta, mu, sd):
    out_dir = OUT_BASE / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[gru_attn] seed={seed}")

    # 切分（与 stage2 完全一致）
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

    # 标签标准化（与 stage2 一致）
    y_mu = float(y_tr_raw.mean())
    y_std = float(y_tr_raw.std())
    y_tr = (y_tr_raw - y_mu) / y_std
    y_va = (y_va_raw - y_mu) / y_std
    y_te = (y_te_raw - y_mu) / y_std
    print(f"    label: mu={y_mu:.5f} std={y_std:.5f}")

    # 模型（种子在创建前设）
    CFG["seed"] = seed
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model("gru_attn", WINDOW, len(FEATURES))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    n_params={n_params:,}  train={len(y_tr):,} "
          f"valid={len(y_va):,} test={len(y_te):,}")

    model, hist, best_epoch, elapsed = fit_gru_attn(model, X_tr, y_tr, X_va, y_va, out_dir)

    # 预测 + 注意力
    p_norm_va, attn_va = predict_with_attn(model, X_va)
    p_norm_te, attn_te = predict_with_attn(model, X_te)
    p_raw_va = p_norm_va * y_std + y_mu
    p_raw_te = p_norm_te * y_std + y_mu

    # 指标
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

    # 逐样本预测（便于与 GRU/MLP 做同日对比）
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

    # 完整注意力权重
    np.savez_compressed(
        out_dir / "attention.npz",
        valid_attn=attn_va.astype(np.float32),
        valid_dates=m_va["date"].values,
        valid_codes=m_va["code"].values,
        test_attn=attn_te.astype(np.float32),
        test_dates=m_te["date"].values,
        test_codes=m_te["code"].values,
    )

    # 平均注意力（第 1 列 = 60 天前；第 60 列 = 今天）
    mean_va = attn_va.mean(axis=0)
    mean_te = attn_te.mean(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    xs = np.arange(1, WINDOW + 1)
    ax.plot(xs, mean_va, label="valid", marker="o", markersize=3)
    ax.plot(xs, mean_te, label="test",  marker="o", markersize=3)
    ax.axvline(WINDOW, color="red", ls="--", lw=1, label="today")
    ax.set_xlabel("time step (1 = 60 days ago, 60 = today)")
    ax.set_ylabel("mean attention weight")
    ax.set_title(f"GRU-Attn mean attention (seed {seed})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "mean_attention.png", dpi=120); plt.close()

    # checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "features": FEATURES, "window": WINDOW,
        "scaler_mu": mu, "scaler_sd": sd,
        "label_mu": y_mu, "label_std": y_std,
        "model_name": "gru_attn", "n_params": n_params,
        "config": CFG, "seed": seed,
        "best_epoch": best_epoch, "elapsed_sec": elapsed,
        "split": {"train": (TRAIN_START, TRAIN_END),
                  "valid": (VALID_START, VALID_END),
                  "test":  (TEST_START, TEST_END)},
    }, out_dir / "checkpoint.pt")

    return {
        "model": "gru_attn", "seed": seed, "n_params": n_params,
        "best_epoch": best_epoch, "elapsed_sec": elapsed,
        "valid_rank_ic": ic_va, "valid_mse": mse_va,
        "test_rank_ic": ic_te, "test_mse": mse_te,
        "test_pred_std": float(p_raw_te.std()),
        # 注意力统计：最近 5 天 / 最早 5 天 / 分布离散度
        "attn_today_mean":    float(mean_te[-1]),
        "attn_recent5_mean":  float(mean_te[-5:].mean()),
        "attn_early5_mean":   float(mean_te[:5].mean()),
        "attn_std_across_t":  float(mean_te.std()),
    }


# ============================================================
# main
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
        summary.append(run_gru_attn(seed, X_all, y_all, meta, mu, sd))

    df = pd.DataFrame(summary)
    df.to_csv(OUT_BASE / "summary.csv", index=False)

    print("\n===== GRU-Attn 汇总 =====")
    cols = ["seed","n_params","best_epoch","elapsed_sec",
            "valid_rank_ic","test_rank_ic","test_mse","test_pred_std",
            "attn_today_mean","attn_recent5_mean","attn_early5_mean","attn_std_across_t"]
    print(df[cols].to_string(index=False))
    print("\n===== 与种子均值 =====")
    print(df[["test_rank_ic","test_pred_std","best_epoch"]].agg(["mean","std"]).round(4).to_string())
    print(f"\n[done] 结果保存在 {OUT_BASE}")


if __name__ == "__main__":
    main()