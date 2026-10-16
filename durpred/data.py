"""CSV 读取 / Dataset / collate。

训练 CSV 列（必填）：text, duration, spk_audio
可选：emo_audio, emo_alpha(默认1.0), duration_factor(默认1.0), lang(默认ZH),
      emo_happy..emo_calm（8列，默认0）
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio_feat import AudioFeatCache, N_AF
from .features import FEAT_NAMES, engineered
from .model import lang_to_id

EMO_COLS = [
    "emo_happy", "emo_angry", "emo_sad", "emo_afraid",
    "emo_disgusted", "emo_melancholic", "emo_surprised", "emo_calm",
]

REQUIRED = ["text", "duration", "spk_audio"]


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in REQUIRED:
        assert c in df.columns, f"CSV 缺少必填列: {c}"
    df = df.dropna(subset=["text", "duration", "spk_audio"]).copy()
    df["text"] = df["text"].astype(str)
    df["duration"] = df["duration"].astype(float)
    df["spk_audio"] = df["spk_audio"].astype(str)
    df = df[df["duration"] > 0].reset_index(drop=True)
    if "emo_audio" not in df.columns:
        df["emo_audio"] = ""
    df["emo_audio"] = df["emo_audio"].fillna("").astype(str)
    for c, default in (("emo_alpha", 1.0), ("duration_factor", 1.0)):
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default).astype(float)
    if "lang" not in df.columns:
        df["lang"] = "ZH"
    df["lang"] = df["lang"].fillna("ZH").astype(str)
    for c in EMO_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    assert len(df) > 0, "CSV 为空或没有合法行"
    return df


def check_audio_files(df: pd.DataFrame):
    import os

    missing = [p for p in df["spk_audio"].unique() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"找不到 {len(missing)} 个音色参考音频，例如: {missing[:3]}")


class DurationDataset(Dataset):
    def __init__(self, df, vocab, max_len, cache: AudioFeatCache,
                 fmean, fstd, afmean, afstd):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        # 预计算音频特征（带磁盘缓存，首个 epoch 后极快）
        spk_af, emo_af = [], []
        for _, r in df.iterrows():
            f = cache.get(r["spk_audio"])
            if f is None:
                raise FileNotFoundError(f"音色音频不存在: {r['spk_audio']}")
            spk_af.append(f)
            ef = cache.get(r["emo_audio"]) if r["emo_audio"] else None
            emo_af.append(ef if ef is not None else np.zeros(N_AF, dtype=np.float32))
        self.spk_af = (np.stack(spk_af).astype(np.float32) - afmean) / afstd
        self.emo_af = (np.stack(emo_af).astype(np.float32) - afmean) / afstd
        self.fmean, self.fstd = fmean, fstd

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        ids = self.vocab.encode(r["text"], self.max_len)
        feat = (engineered(r["text"]) - self.fmean) / self.fstd
        dfac = float(r["duration_factor"])
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "feat": torch.from_numpy(feat.astype(np.float32)),
            "spk_af": torch.from_numpy(self.spk_af[i]),
            "emo_af": torch.from_numpy(self.emo_af[i]),
            "emo_vec": torch.tensor(r[EMO_COLS].values.astype(np.float32)),
            "emo_alpha": torch.tensor([float(r["emo_alpha"])], dtype=torch.float32),
            "lang_id": torch.tensor(lang_to_id(r["lang"]), dtype=torch.long),
            "log_df": torch.tensor([float(np.log(dfac))], dtype=torch.float32),
            "target": torch.tensor(float(np.log(r["duration"] / dfac)), dtype=torch.float32),
            "duration": torch.tensor(float(r["duration"]), dtype=torch.float32),
            "dfac": torch.tensor(dfac, dtype=torch.float32),
        }


def collate(batch):
    maxl = max(len(b["ids"]) for b in batch)
    B = len(batch)
    ids = torch.zeros(B, maxl, dtype=torch.long)
    mask = torch.zeros(B, maxl, dtype=torch.float32)
    for i, b in enumerate(batch):
        l = len(b["ids"])
        ids[i, :l] = b["ids"]
        mask[i, :l] = 1.0
    out = {"ids": ids, "mask": mask}
    for k in ("feat", "spk_af", "emo_af", "emo_vec", "emo_alpha",
              "lang_id", "log_df", "target", "duration", "dfac"):
        out[k] = torch.stack([b[k] for b in batch])
    return out


def norm_stats(df, cache: AudioFeatCache):
    feats = np.stack([engineered(t) for t in df["text"]]).astype(np.float64)
    fmean, fstd = feats.mean(0).astype(np.float32), feats.std(0).astype(np.float32)
    fstd[fstd < 1e-6] = 1.0
    afs = []
    for p in df["spk_audio"].unique():
        f = cache.get(p)
        if f is not None:
            afs.append(f)
    afs = np.stack(afs).astype(np.float64)
    afmean, afstd = afs.mean(0).astype(np.float32), afs.std(0).astype(np.float32)
    afstd[afstd < 1e-6] = 1.0
    return fmean, fstd, afmean, afstd
