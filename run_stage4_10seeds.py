# run_stage4_10seeds.py
# GRU + mean pooling:
#   1. 标准化收益 MSE
#   2. 同日 pairwise logistic ranking
#
# 两组统一：
#   模型初始化、每日股票抽样、日期顺序、优化器、
#   验证 Rank IC 早停、学习率调度、最大训练轮数。
#
# 依赖同目录 run_stage2.py 的数据构建和配置。
# 输出到 runs/stage4_fixed，不覆盖此前实验。

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from run_stage2 import (
    ROOT, CFG, FEATURES, WINDOW,
    TRAIN_START, TRAIN_END,
    VALID_START, VALID_END,
    TEST_START, TEST_END,
    load_daily, fit_scaler, make_windows,
)


SEEDS = list(range(10))

OUT_BASE = ROOT / "runs" / "stage4_fixed"

CONFIG = dict(CFG)
CONFIG.update({
    "daily_max_size": 100,
    "min_stocks": 20,
    "ic_min_delta": 1e-4,
    "scheduler_patience": 2,
    "scheduler_factor": 0.5,
    "predict_batch_size": 1024,
})

# 使用CPU，与前面阶段保持一致。
DEVICE = torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class GRUPool(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        states, _ = self.gru(x)
        context = states.mean(dim=1)
        return self.head(self.drop(context)).squeeze(-1)


class DailySampler:
    """每轮每个日期出现一次；两组同seed、同epoch使用相同样本。"""

    def __init__(self, meta, max_size, min_stocks, seed):
        self.max_size = max_size
        self.seed = seed
        self.groups = []

        # meta必须已reset_index，与X、y的位置一致。
        for _, group in meta.groupby("date", sort=True):
            indices = group.index.to_numpy(dtype=np.int64)
            if len(indices) >= min_stocks:
                self.groups.append(indices)

        if not self.groups:
            raise ValueError("没有满足最低股票数要求的训练日期")

    def epoch_batches(self, epoch):
        # 每轮独立生成，不依赖其他函数消耗随机数的次数。
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, 314159])
        )

        for group_id in rng.permutation(len(self.groups)):
            indices = self.groups[group_id]
            size = min(len(indices), self.max_size)
            yield rng.choice(indices, size=size, replace=False)


def pairwise_logistic_loss(pred, target):
    """同日所有非并列标签对的平均logistic损失。"""
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    i, j = torch.triu_indices(
        pred.numel(), pred.numel(),
        offset=1, device=pred.device,
    )

    target_diff = target[i] - target[j]
    valid = target_diff != 0

    if not valid.any():
        return pred.sum() * 0.0

    direction = target_diff[valid].sign()
    score_diff = pred[i[valid]] - pred[j[valid]]

    return F.softplus(-direction * score_diff).mean()


def check_ranking_loss():
    """检查常数输出不再获得零损失，且正确排序优于错误排序。"""
    target = torch.tensor([0.0, 1.0, 2.0])
    constant = torch.zeros(3)
    correct = torch.tensor([-1.0, 0.0, 1.0])
    wrong = -correct

    loss_constant = pairwise_logistic_loss(constant, target)
    loss_correct = pairwise_logistic_loss(correct, target)
    loss_wrong = pairwise_logistic_loss(wrong, target)

    assert torch.isclose(
        loss_constant,
        torch.tensor(np.log(2.0), dtype=torch.float32),
        atol=1e-6,
    )
    assert loss_correct < loss_constant < loss_wrong

    pred = torch.zeros(3, requires_grad=True)
    pairwise_logistic_loss(pred, target).backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


@torch.no_grad()
def predict(model, X):
    model.eval()
    outputs = []

    for start in range(0, len(X), CONFIG["predict_batch_size"]):
        xb = torch.from_numpy(
            X[start:start + CONFIG["predict_batch_size"]]
        ).to(DEVICE)

        outputs.append(model(xb).cpu().numpy())

    scores = np.concatenate(outputs)

    if not np.isfinite(scores).all():
        raise RuntimeError("预测出现NaN或无穷值")

    return scores


def daily_metrics(scores, targets, meta, split):
    frame = meta[["date", "code"]].copy()
    frame["score"] = np.asarray(scores, dtype=np.float64)
    frame["y"] = np.asarray(targets, dtype=np.float64)

    rows = []
    for date, group in frame.groupby("date", sort=True):
        if len(group) < CONFIG["min_stocks"]:
            continue

        unique_scores = group["score"].nunique()
        constant_pred = unique_scores < 2
        constant_target = group["y"].nunique() < 2

        # 常数标签没有可评价的排序；保留记录，不用于平均IC。
        # 常数预测按0计入，避免删除模型失效日期抬高平均IC。
        if constant_target:
            ic = np.nan
        elif constant_pred:
            ic = 0.0
        else:
            ic = float(spearmanr(group["score"], group["y"])[0])

        rows.append({
            "date": date,
            "split": split,
            "rank_ic": ic,
            "n": len(group),
            "score_std": group["score"].std(ddof=0),
            "unique_score_ratio": unique_scores / len(group),
            "constant_pred": int(constant_pred),
            "constant_target": int(constant_target),
        })

    result = pd.DataFrame(rows)
    if result.empty or result["rank_ic"].notna().sum() == 0:
        raise RuntimeError(f"{split}: 没有可评价日期")

    return result


def train_epoch(model, mode, X, y, sampler, epoch, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for indices in sampler.epoch_batches(epoch):
        xb = torch.from_numpy(X[indices]).to(DEVICE)
        yb = torch.from_numpy(y[indices]).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        scores = model(xb)

        if mode == "mse":
            loss = F.mse_loss(scores, yb)
        elif mode == "rank":
            loss = pairwise_logistic_loss(scores, yb)
        else:
            raise ValueError(mode)

        if not torch.isfinite(loss):
            raise RuntimeError(f"{mode}: loss出现非有限值")

        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            CONFIG["grad_clip"],
            error_if_nonfinite=True,
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def fit(model, mode, seed, train, valid, y_mu, y_std, out_dir):
    X_tr, y_tr, meta_tr = train
    X_va, y_va, meta_va = valid

    # 两组标签处理相同；正的标准化不会改变排序方向。
    y_tr_norm = ((y_tr - y_mu) / y_std).astype(np.float32)

    sampler = DailySampler(
        meta_tr,
        max_size=CONFIG["daily_max_size"],
        min_stocks=CONFIG["min_stocks"],
        seed=seed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
        threshold=CONFIG["ic_min_delta"],
        threshold_mode="abs",
    )

    best_ic = -np.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []
    started = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        # 使两组对应轮次的Dropout随机流一致，
        # 不受另一组何时早停等因素影响。
        torch.manual_seed(seed + 100003 * epoch)

        lr_used = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(
            model, mode, X_tr, y_tr_norm,
            sampler, epoch, optimizer,
        )

        scores_va = predict(model, X_va)
        daily_va = daily_metrics(
            scores_va, y_va, meta_va, split="valid"
        )
        valid_ic = float(daily_va["rank_ic"].mean())

        if not np.isfinite(valid_ic):
            raise RuntimeError("验证IC不是有限值")

        scheduler.step(valid_ic)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_rank_ic": valid_ic,
            "valid_score_std": float(scores_va.std()),
            "valid_daily_score_std_mean":
                float(daily_va["score_std"].mean()),
            "valid_constant_pred_days":
                int(daily_va["constant_pred"].sum()),
            "lr_used": lr_used,
            "lr_next": optimizer.param_groups[0]["lr"],
        })

        print(
            f"    [{mode}] epoch {epoch:02d} "
            f"loss={train_loss:.6f} "
            f"valid_ic={valid_ic:+.5f} "
            f"daily_score_std={daily_va['score_std'].mean():.6e} "
            f"lr={lr_used:.2e}",
            flush=True,
        )

        if valid_ic > best_ic + CONFIG["ic_min_delta"]:
            best_ic = valid_ic
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        if bad_epochs >= CONFIG["patience"]:
            print(f"    early stop at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("没有可用的最佳模型")

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(
        out_dir / "history.csv", index=False
    )

    return best_epoch, time.time() - started


def evaluate_and_save(model, mode, splits, y_mu, y_std, out_dir):
    prediction_frames = []
    daily_frames = []
    metrics = []

    for split_name, (X, y, meta) in splits.items():
        scores = predict(model, X)
        daily = daily_metrics(scores, y, meta, split_name)

        # score用于排序；只有MSE模型有收益预测的尺度含义。
        if mode == "mse":
            return_pred = scores.astype(np.float64) * y_std + y_mu
            mse = float(np.mean((return_pred - y) ** 2))
        else:
            return_pred = np.full(len(scores), np.nan)
            mse = np.nan

        row = {
            "split": split_name,
            "rank_ic": float(daily["rank_ic"].mean()),
            "n_days": int(daily["rank_ic"].notna().sum()),
            "mse": mse,
            "zero_mse": float(np.mean(y.astype(np.float64) ** 2)),
            "train_mean_mse":
                float(np.mean((y.astype(np.float64) - y_mu) ** 2)),
            "score_std": float(scores.std()),
            "daily_score_std_mean": float(daily["score_std"].mean()),
            "unique_score_ratio_mean":
                float(daily["unique_score_ratio"].mean()),
            "constant_pred_days": int(daily["constant_pred"].sum()),
        }
        metrics.append(row)
        daily_frames.append(daily)

        prediction_frames.append(pd.DataFrame({
            "date": meta["date"].to_numpy(),
            "code": meta["code"].to_numpy(),
            "split": split_name,
            "score": scores,
            "pred_return": return_pred,
            "y": y,
        }))

        mse_text = f"{mse:.6f}" if mode == "mse" else "N/A"
        print(
            f"    {split_name:5s} | "
            f"RankIC={row['rank_ic']:+.5f} "
            f"days={row['n_days']} "
            f"MSE={mse_text} "
            f"score_std={row['score_std']:.6e} "
            f"constant_days={row['constant_pred_days']}",
            flush=True,
        )

    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        out_dir / "predictions.parquet", index=False
    )
    pd.concat(daily_frames, ignore_index=True).to_csv(
        out_dir / "daily_ic.csv", index=False
    )
    pd.DataFrame(metrics).to_csv(
        out_dir / "metrics.csv", index=False
    )

    return {row["split"]: row for row in metrics}


def prepare_data():
    daily = load_daily()
    mu, sd = fit_scaler(daily, TRAIN_START, TRAIN_END)
    X_all, y_all, meta = make_windows(daily, mu, sd)

    meta = meta.copy()
    meta["date"] = pd.to_datetime(meta["date"])
    meta["label_end"] = pd.to_datetime(meta["label_end"])

    masks = {
        "train": (
            meta["date"].between(TRAIN_START, TRAIN_END)
            & (meta["label_end"] < pd.Timestamp(VALID_START))
        ),
        "valid": (
            meta["date"].between(VALID_START, VALID_END)
            & (meta["label_end"] < pd.Timestamp(TEST_START))
        ),
        "test": meta["date"].between(TEST_START, TEST_END),
    }

    splits = {}
    for name, mask in masks.items():
        mask = mask.to_numpy()
        X = np.asarray(X_all[mask], dtype=np.float32)
        y = np.asarray(y_all[mask], dtype=np.float32)
        m = meta.loc[mask].reset_index(drop=True)

        if len(y) == 0:
            raise ValueError(f"{name}: 样本为空")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"{name}: 特征或标签包含NaN/无穷值")
        if m[["date", "code"]].duplicated().any():
            raise ValueError(f"{name}: 存在重复date/code")

        splits[name] = (X, y, m)

    return splits, mu, sd


def run_one(mode, seed, splits, mu, sd):
    out_dir = OUT_BASE / f"{mode}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 完成的运行直接复用，方便中断后继续。
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"[skip] 已完成 {mode} seed={seed}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("mode") != mode or result.get("seed") != seed:
            raise ValueError(f"Cached result identity mismatch: {result_path}")
        required = ["daily_ic.csv", "metrics.csv", "checkpoint.pt", "predictions.parquet", "history.csv"]
        missing = [name for name in required if not (out_dir / name).is_file()]
        if missing:
            raise ValueError(f"Incomplete cached run {out_dir}: {missing}")
        return result

    print(f"\n[stage4_fixed] {mode} seed={seed}", flush=True)

    y_train = splits["train"][1]
    y_mu = float(y_train.mean())
    y_std = float(y_train.std())

    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError("训练标签标准差必须为正")

    set_seed(seed)
    model = GRUPool(
        len(FEATURES),
        hidden=CONFIG["gru_hidden"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    best_epoch, elapsed = fit(
        model, mode, seed,
        splits["train"], splits["valid"],
        y_mu, y_std, out_dir,
    )

    metrics = evaluate_and_save(
        model, mode,
        {"valid": splits["valid"], "test": splits["test"]},
        y_mu, y_std, out_dir,
    )

    torch.save({
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "model_name": "gru_pool",
        "loss_mode": mode,
        "ranking_loss": "pairwise_logistic" if mode == "rank" else None,
        "features": list(FEATURES),
        "window": WINDOW,
        "scaler_mu": mu,
        "scaler_sd": sd,
        "label_mu": y_mu,
        "label_std": y_std,
        "seed": seed,
        "config": dict(CONFIG),
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "selection_metric": "valid_rank_ic",
        "split": {
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test": (TEST_START, TEST_END),
        },
        "score_note": (
            "MSE输出为标准化收益；rank输出为未校准排序分数"
        ),
    }, out_dir / "checkpoint.pt")

    result = {
        "mode": mode,
        "seed": seed,
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "valid_rank_ic": metrics["valid"]["rank_ic"],
        "test_rank_ic": metrics["test"]["rank_ic"],
        "test_mse": metrics["test"]["mse"] if mode == "mse" else None,
        "test_score_std": metrics["test"]["score_std"],
        "test_daily_score_std_mean":
            metrics["test"]["daily_score_std_mean"],
        "test_unique_score_ratio_mean":
            metrics["test"]["unique_score_ratio_mean"],
        "test_constant_pred_days":
            metrics["test"]["constant_pred_days"],
    }

    # 最后写完成标记，避免把中断的运行当作已完成。
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main():
    check_ranking_loss()
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    splits, mu, sd = prepare_data()
    print(
        "[split] "
        + " ".join(f"{name}={len(item[1]):,}"
                   for name, item in splits.items())
    )

    rows = []
    for mode in ["mse", "rank"]:
        for seed in SEEDS:
            rows.append(
                run_one(mode, int(seed), splits, mu, sd)
            )
            pd.DataFrame(rows).to_csv(
                OUT_BASE / "summary_10seeds.csv", index=False
            )

    summary = pd.DataFrame(rows)

    print("\n===== 各种子结果 =====")
    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    ))

    print("\n===== 跨种子Rank IC =====")
    print(
        summary.groupby("mode")[
            ["valid_rank_ic", "test_rank_ic"]
        ].agg(["mean", "std"]).to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print(f"\n结果目录：{OUT_BASE}")
    print("rank的score是排序分数，不作为收益预测计算MSE。")
    print("本轮仍属于已查看2023年数据后的探索性实验。")


if __name__ == "__main__":
    main()