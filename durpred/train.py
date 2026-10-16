"""训练入口：python -m durpred.train --data data/teacher.csv --out models/dur_v1"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .audio_feat import AudioFeatCache
from .data import DurationDataset, check_audio_files, collate, load_table, norm_stats
from .features import FEAT_NAMES
from .model import N_AF, DurationPredictor
from .vocab import CharVocab


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def build_parser():
    ap = argparse.ArgumentParser(description="训练 TTS 时长蒸馏预测器")
    ap.add_argument("--data", required=True, help="训练 CSV（列：text,duration,spk_audio[,…]）")
    ap.add_argument("--out", required=True, help="输出目录，保存 model.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache-dir", default=".durpred_cache", help="音频特征缓存目录")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, dest="batch_size")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    return ap


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    abs_err, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        pred = torch.exp(model(b["ids"], b["mask"], b["feat"], b["spk_af"],
                               b["emo_af"], b["emo_vec"], b["emo_alpha"],
                               b["lang_id"], b["log_df"])) * b["dfac"]
        abs_err += (pred.cpu() - b["duration"].cpu()).abs().sum().item()
        n += len(pred)
    return abs_err / max(n, 1)


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    cfg = {}
    if os.path.exists(a.config):
        cfg = yaml.safe_load(open(a.config, encoding="utf-8")) or {}
    mcfg, tcfg = cfg.get("model", {}), cfg.get("train", {})
    epochs = a.epochs or tcfg.get("epochs", 300)
    batch_size = a.batch_size or tcfg.get("batch_size", 512)
    lr = a.lr or tcfg.get("lr", 3e-4)
    val_ratio, patience = tcfg.get("val_ratio", 0.1), tcfg.get("patience", 25)
    seed, min_freq = tcfg.get("seed", 42), tcfg.get("min_freq", 2)
    max_len = mcfg.get("max_len", 256)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    df = load_table(a.data)
    print(f"[数据] {len(df)} 行")
    check_audio_files(df)
    cache = AudioFeatCache(a.cache_dir)

    n_val = max(1, int(len(df) * val_ratio))
    idx = torch.randperm(len(df), generator=torch.Generator().manual_seed(seed)).tolist()
    train_df = df.iloc[idx[: len(df) - n_val]].reset_index(drop=True)
    val_df = df.iloc[idx[len(df) - n_val:]].reset_index(drop=True)

    vocab = CharVocab.build(train_df["text"], min_freq=min_freq)
    fmean, fstd, afmean, afstd = norm_stats(train_df, cache)
    print(f"[词表] {len(vocab)} 字")
    print("[特征] 正在提取音频特征（含缓存）…")
    train_ds = DurationDataset(train_df, vocab, max_len, cache, fmean, fstd, afmean, afstd)
    val_ds = DurationDataset(val_df, vocab, max_len, cache, fmean, fstd, afmean, afstd)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate)

    model = DurationPredictor(
        vocab_size=len(vocab), d_model=mcfg.get("d_model", 128),
        n_layers=mcfg.get("n_layers", 3), nhead=mcfg.get("nhead", 4),
        d_ff=mcfg.get("d_ff", 256), dropout=mcfg.get("dropout", 0.1),
        max_len=max_len, n_feat=len(FEAT_NAMES),
    ).to(device)
    print(f"[模型] {model.n_params()/1e6:.2f}M 参数，device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_mae, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            pred_log = model(b["ids"], b["mask"], b["feat"], b["spk_af"],
                             b["emo_af"], b["emo_vec"], b["emo_alpha"],
                             b["lang_id"], b["log_df"])
            loss = loss_fn(pred_log, b["target"])
            loss.backward()
            opt.step()
            tot += loss.item() * len(pred_log)
            n += len(pred_log)
        mae = evaluate(model, val_loader, device)
        improved = mae < best_mae - 1e-6
        if improved:
            best_mae, bad = mae, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0 or improved:
            print(f"epoch {ep:3d}  train_mse(log)={tot/max(n,1):.4f}  val_MAE={mae:.3f}s"
                  + ("  *" if improved else ""))
        if bad >= patience:
            print(f"[早停] {patience} 轮无提升，停止。")
            break

    os.makedirs(a.out, exist_ok=True)
    ckpt = {
        "model_cfg": {
            "d_model": mcfg.get("d_model", 128), "n_layers": mcfg.get("n_layers", 3),
            "nhead": mcfg.get("nhead", 4), "d_ff": mcfg.get("d_ff", 256),
            "dropout": mcfg.get("dropout", 0.1), "max_len": max_len,
            "n_feat": len(FEAT_NAMES),
        },
        "vocab": vocab.stoi,
        "feat_mean": fmean.tolist(), "feat_std": fstd.tolist(),
        "af_mean": afmean.tolist(), "af_std": afstd.tolist(),
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "best_val_mae": best_mae,
        "n_train": len(train_df),
    }
    torch.save(ckpt, os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"train_rows={len(train_df)} val_rows={len(val_df)} "
                f"best_val_MAE={best_mae:.4f}s params={model.n_params()}\n")
    print(f"[完成] 最佳 val MAE = {best_mae:.3f}s，模型已存 {a.out}/model.pt")


if __name__ == "__main__":
    main()
