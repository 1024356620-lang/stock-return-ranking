# ============================================================
# rebuild_stage2_daily_ic.py
# 从 Stage 2 的 gru_seed*/checkpoint.pt 重新推理，生成
# predictions.parquet 和 daily_ic.csv，不重跑训练。
# 用法：python rebuild_stage2_daily_ic.py
# ============================================================
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr

from run_stage2 import (
    ROOT, CFG, FEATURES, WINDOW,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
    load_daily, fit_scaler, make_windows, build_model,  # build_model 已存在
)

# 如果 run_stage2.py 里 build_model 没导出（被 __main__ guard 内定义），
# 在 import 失败时手工定义 GRUModel。
try:
    from run_stage2 import build_model  # noqa: F811
except ImportError:
    import torch.nn as nn
    class GRUModel(nn.Module):
        def __init__(self, n_feat, hidden=64, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(n_feat, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)
    def build_model(name, window, n_feat):
        if name == "gru":
            return GRUModel(n_feat, CFG["gru_hidden"], CFG["dropout"])
        raise ValueError(name)


@torch.no_grad()
def predict(model, X, batch=1024):
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


def main():
    daily = load_daily()
    mu, sd = fit_scaler(daily, TRAIN_START, TRAIN_END)
    X_all, y_all, meta = make_windows(daily, mu, sd)

    va_start = pd.Timestamp(VALID_START); te_start = pd.Timestamp(TEST_START)
    va_mask = ((meta["date"] >= VALID_START) & (meta["date"] <= VALID_END)
               & (meta["label_end"] < te_start)).values
    te_mask = ((meta["date"] >= TEST_START) & (meta["date"] <= TEST_END)).values
    X_va, y_va = X_all[va_mask], y_all[va_mask]
    X_te, y_te = X_all[te_mask], y_all[te_mask]
    m_va = meta[va_mask].reset_index(drop=True)
    m_te = meta[te_mask].reset_index(drop=True)

    # 顺带把 MLP 也重建（如果你也想要 MLP 的逐日 IC 做对照）
    models_to_rebuild = ["gru", "mlp"]

    for model_name in models_to_rebuild:
        for seed in [0, 1, 2]:
            ckpt_path = ROOT / "runs" / "stage2" / f"{model_name}_seed{seed}" / "checkpoint.pt"
            if not ckpt_path.exists():
                print(f"[skip] {ckpt_path}")
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            model = build_model(model_name, ckpt["window"], len(ckpt["features"]))
            model.load_state_dict(ckpt["state_dict"])
            y_mu = float(ckpt["label_mu"]); y_std = float(ckpt["label_std"])

            p_raw_va = predict(model, X_va) * y_std + y_mu
            p_raw_te = predict(model, X_te) * y_std + y_mu

            out_dir = ckpt_path.parent
            pred_df = pd.concat([
                pd.DataFrame({"date": m_va["date"].values, "code": m_va["code"].values,
                              "split": "valid", "pred": p_raw_va, "y": y_va}),
                pd.DataFrame({"date": m_te["date"].values, "code": m_te["code"].values,
                              "split": "test",  "pred": p_raw_te, "y": y_te}),
            ], ignore_index=True)
            pred_df.to_parquet(out_dir / "predictions.parquet", index=False)

            ic_df = pd.concat([
                daily_ic_df(p_raw_va, y_va, m_va, "valid"),
                daily_ic_df(p_raw_te, y_te, m_te, "test"),
            ], ignore_index=True)
            ic_df.to_csv(out_dir / "daily_ic.csv", index=False)

            # 与 metrics.csv 自检：重建的 test_ic 应接近 metrics.csv 里的值
            old = pd.read_csv(out_dir / "metrics.csv")
            old_test = float(old[old["split"] == "test"]["rank_ic"].iloc[0])
            new_test = float(ic_df[ic_df["split"] == "test"]["rank_ic"].mean())
            old_valid = float(old[old["split"] == "valid"]["rank_ic"].iloc[0])
            new_valid = float(ic_df[ic_df["split"] == "valid"]["rank_ic"].mean())
            print(f"[ok] {model_name}_seed{seed}: "
                  f"valid {new_valid:+.4f} (old {old_valid:+.4f}) | "
                  f"test {new_test:+.4f} (old {old_test:+.4f})")


if __name__ == "__main__":
    main()