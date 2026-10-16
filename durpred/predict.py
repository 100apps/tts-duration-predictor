"""推理：参数与 IndexTTS2.5 infer() 对齐。

python -m durpred.predict --model models/dur_v1 \\
    --text "你好" --spk-audio ref.wav --duration-factor 1.2
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from .audio_feat import AudioFeatCache, N_AF
from .data import EMO_COLS, load_table
from .features import engineered
from .model import N_EMO, DurationPredictor, lang_to_id
from .vocab import CharVocab


class Predictor:
    def __init__(self, ckpt: dict, cache_dir=".durpred_cache", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = CharVocab(ckpt["vocab"])
        self.fmean = np.array(ckpt["feat_mean"], dtype=np.float32)
        self.fstd = np.array(ckpt["feat_std"], dtype=np.float32)
        self.afmean = np.array(ckpt["af_mean"], dtype=np.float32)
        self.afstd = np.array(ckpt["af_std"], dtype=np.float32)
        cfg = ckpt["model_cfg"]
        self.max_len = cfg.get("max_len", 256)
        self.model = DurationPredictor(vocab_size=len(self.vocab), **cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.cache = AudioFeatCache(cache_dir)

    @classmethod
    def load(cls, ckpt_dir: str, cache_dir=".durpred_cache", device=None) -> "Predictor":
        ckpt = torch.load(os.path.join(ckpt_dir, "model.pt"),
                          map_location="cpu", weights_only=False)
        return cls(ckpt, cache_dir=cache_dir, device=device)

    def _tensors(self, texts, spk_audios, emo_audios, emo_vecs,
                 emo_alphas, langs, dfacs):
        B = len(texts)
        ids_list = [self.vocab.encode(t, self.max_len) for t in texts]
        L = max(len(x) for x in ids_list)
        ids = torch.zeros(B, L, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.float32)
        for i, x in enumerate(ids_list):
            ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1.0
        feats, spk_af, emo_af = [], [], []
        for t, sp, ea in zip(texts, spk_audios, emo_audios):
            feats.append((engineered(t) - self.fmean) / self.fstd)
            f = self.cache.get(sp)
            if f is None:
                raise FileNotFoundError(f"音色音频不存在: {sp}")
            spk_af.append((f - self.afmean) / self.afstd)
            ef = self.cache.get(ea) if ea else None
            if ef is None:
                ef = np.zeros(N_AF, dtype=np.float32)
            emo_af.append((ef - self.afmean) / self.afstd)
        return {
            "ids": ids.to(self.device), "mask": mask.to(self.device),
            "feat": torch.tensor(np.stack(feats), dtype=torch.float32).to(self.device),
            "spk_af": torch.tensor(np.stack(spk_af), dtype=torch.float32).to(self.device),
            "emo_af": torch.tensor(np.stack(emo_af), dtype=torch.float32).to(self.device),
            "emo_vec": torch.tensor(np.stack(emo_vecs).astype(np.float32)).to(self.device),
            "emo_alpha": torch.tensor(np.array(emo_alphas, dtype=np.float32).reshape(-1, 1)).to(self.device),
            "lang_id": torch.tensor([lang_to_id(l) for l in langs], dtype=torch.long).to(self.device),
            "log_df": torch.tensor(np.log(np.array(dfacs, dtype=np.float32)).reshape(-1, 1)).to(self.device),
            "dfac": torch.tensor(np.array(dfacs, dtype=np.float32)).to(self.device),
        }

    @torch.no_grad()
    def predict_batch(self, texts, spk_audios, emo_audios=None,
                      emo_vecs=None, emo_alphas=None, langs=None, dfacs=None):
        n = len(texts)
        emo_audios = emo_audios or [""] * n
        emo_vecs = emo_vecs or [[0.0] * N_EMO for _ in range(n)]
        emo_alphas = emo_alphas or [1.0] * n
        langs = langs or ["ZH"] * n
        dfacs = dfacs or [1.0] * n
        b = self._tensors(texts, spk_audios, emo_audios, emo_vecs,
                          emo_alphas, langs, dfacs)
        log_base = self.model(b["ids"], b["mask"], b["feat"], b["spk_af"],
                              b["emo_af"], b["emo_vec"], b["emo_alpha"],
                              b["lang_id"], b["log_df"])
        return (torch.exp(log_base) * b["dfac"]).cpu().tolist()

    def predict(self, text: str, spk_audio_prompt: str,
                emo_audio_prompt: str = "", emo_vector=None,
                emo_alpha: float = 1.0, duration_factor: float = 1.0,
                lang: str = "ZH") -> float:
        return self.predict_batch(
            [text], [spk_audio_prompt], [emo_audio_prompt],
            [emo_vector or [0.0] * N_EMO], [emo_alpha], [lang],
            [duration_factor])[0]

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        emo_vecs = df[EMO_COLS].values.tolist() if all(c in df.columns for c in EMO_COLS) else None
        out = self.predict_batch(
            df["text"].astype(str).tolist(),
            df["spk_audio"].astype(str).tolist(),
            df["emo_audio"].astype(str).tolist() if "emo_audio" in df.columns else None,
            emo_vecs,
            df["emo_alpha"].tolist() if "emo_alpha" in df.columns else None,
            df["lang"].astype(str).tolist() if "lang" in df.columns else None,
            df["duration_factor"].tolist() if "duration_factor" in df.columns else None,
        )
        df["pred_duration"] = out
        return df


def _parse_emo_vec(s):
    if not s:
        return None
    v = [float(x) for x in s.split(",")]
    assert len(v) == 8, "emo-vector 需要 8 个逗号分隔的数"
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description="时长预测推理（参数对齐 IndexTTS2.5 infer）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--spk-audio", default=None, dest="spk_audio")
    ap.add_argument("--emo-audio", default="", dest="emo_audio")
    ap.add_argument("--emo-vector", default=None, dest="emo_vector")
    ap.add_argument("--emo-alpha", type=float, default=1.0, dest="emo_alpha")
    ap.add_argument("--duration-factor", type=float, default=1.0, dest="dfac")
    ap.add_argument("--lang", default="ZH")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--cache-dir", default=".durpred_cache")
    a = ap.parse_args(argv)

    p = Predictor.load(a.model, cache_dir=a.cache_dir)
    if a.text:
        if not a.spk_audio:
            ap.error("单句预测需要 --spk-audio")
        sec = p.predict(a.text, a.spk_audio, a.emo_audio,
                        _parse_emo_vec(a.emo_vector), a.emo_alpha, a.dfac, a.lang)
        print(f"预估时长: {sec:.2f}s")
    elif a.csv:
        df = load_table(a.csv)
        out = p.predict_df(df)
        cols = ["text", "pred_duration"] + (["duration"] if "duration" in out.columns else [])
        print(out[cols].to_string(index=False))
        if a.csv_out:
            out.to_csv(a.csv_out, index=False, encoding="utf-8-sig")
            print(f"[保存] {a.csv_out}")
    else:
        ap.error("需要 --text 或 --csv")


if __name__ == "__main__":
    main()
