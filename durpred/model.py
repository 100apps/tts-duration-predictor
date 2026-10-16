"""蒸馏时长预测网络。

输入（与 IndexTTS2.5 infer() 参数对齐）：
  text              文字
  spk_audio         音色参考音频 -> 8维音频统计（语速/能量先验，零样本泛化）
  emo_audio         情绪参考音频（可选）-> 8维音频统计
  emo_vector        8维情绪向量（可选）
  emo_alpha         情绪强度（可选）
  lang              语言（ZH/EN/JA/ES/AR）
  duration_factor   语速倍率（>1变慢），输出端直接相乘

输出 log(基础时长)，最终时长 = exp(out) * duration_factor。
"""
import math

import torch
import torch.nn as nn

N_AF = 8
N_EMO = 8
LANGS = {"ZH": 0, "EN": 1, "JA": 2, "ES": 3, "AR": 4}
N_LANG = 8
LANG_OTHER = 5


def lang_to_id(lang: str) -> int:
    return LANGS.get(str(lang).upper(), LANG_OTHER)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DurationPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 3,
        nhead: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        n_feat: int = 7,
    ):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.text_proj = nn.Linear(d_model, 48)
        self.spk_mlp = nn.Sequential(nn.Linear(N_AF, 16), nn.ReLU())
        self.emo_mlp = nn.Sequential(nn.Linear(N_AF + N_EMO + 1, 16), nn.ReLU())
        self.lang_emb = nn.Embedding(N_LANG, 8)
        self.head = nn.Sequential(
            nn.Linear(48 + n_feat + 16 + 16 + 8 + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, ids, mask, feat, spk_af, emo_af, emo_vec, emo_alpha, lang_ids, log_df):
        # ids (B,L) long; mask (B,L) float; feat (B,7); spk_af/emo_af (B,8);
        # emo_vec (B,8); emo_alpha (B,1); lang_ids (B,); log_df (B,1)
        x = self.pos(self.emb(ids))
        x = self.encoder(x, src_key_padding_mask=(mask < 0.5))
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(1) / mask.sum(1).clamp_min(1.0).unsqueeze(-1)
        t = self.text_proj(pooled)
        s = self.spk_mlp(spk_af)
        e = self.emo_mlp(torch.cat([emo_af, emo_vec, emo_alpha], dim=1))
        l = self.lang_emb(lang_ids)
        h = torch.cat([t, feat, s, e, l, log_df], dim=1)
        return self.head(h).squeeze(-1)  # log(基础时长秒)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
