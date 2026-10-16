"""轻量 wav 特征提取（只用标准库 wave + numpy，无重型依赖），带磁盘缓存。

从参考音频中提取说话人语速/能量等韵律先验，是零样本泛化的关键：
新音色不需要重新训练，直接传音频即可。
"""
import hashlib
import os
import wave

import numpy as np

N_AF = 8
AF_NAMES = ["wav_dur", "active_ratio", "active_dur", "syl_rate",
            "rms_mean", "rms_std", "onsets", "zcr"]


def read_wav_mono(path: str):
    """读 wav（8/16/24/32bit PCM），转单声道 float32。"""
    with wave.open(path, "rb") as w:
        nch, sw, sr, nfr = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nfr)
    if sw == 1:
        y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif sw == 2:
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)
             | (a[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 0x800000, v - 0x1000000, v)
        y = v.astype(np.float32) / 8388608.0
    elif sw == 4:
        y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"不支持的位宽 {sw * 8}bit: {path}")
    if nch > 1:
        y = y.reshape(-1, nch).mean(axis=1)
    return y.astype(np.float32), sr


def _frame_rms(y: np.ndarray, sr: int, win_s=0.02, hop_s=0.01):
    win, hop = int(win_s * sr), int(hop_s * sr)
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    frames = y[idx]
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12), hop / sr, win, hop


def audio_stats(path: str) -> np.ndarray:
    """8 维音频统计特征：时长、浊音占比、有效时长、音节率估计、能量均值/方差、起音数、过零率。"""
    y, sr = read_wav_mono(path)
    dur = len(y) / float(sr)
    rms, hop_s, win, hop = _frame_rms(y, sr)
    thr = 0.06 * float(rms.max())
    active = rms > thr
    active_ratio = float(active.mean())
    active_dur = float(active.sum() * hop_s)
    d = np.diff(np.log(rms + 1e-6))
    onsets = int((d > 1.0).sum())
    syl_rate = onsets / max(active_dur, 0.5)
    n = len(rms)
    idx = np.clip(np.arange(n)[:, None] * hop + np.arange(win)[None, :], 0, len(y) - 1)
    fr = y[idx]
    zcr = np.abs(np.diff(np.sign(fr), axis=1)).sum(axis=1) / 2 / win
    zcr_m = float(zcr[active].mean()) if active.any() else 0.0
    return np.array([dur, active_ratio, active_dur, syl_rate,
                     float(rms.mean()), float(rms.std()),
                     float(onsets), zcr_m], dtype=np.float32)


class AudioFeatCache:
    """按文件内容哈希缓存音频特征，避免每个 epoch 重复解码。"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, path: str) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def get(self, path: str):
        if not path or not os.path.exists(path):
            return None
        cp = os.path.join(self.cache_dir, self._key(path) + ".npy")
        if os.path.exists(cp):
            return np.load(cp)
        feat = audio_stats(path)
        np.save(cp, feat)
        return feat
