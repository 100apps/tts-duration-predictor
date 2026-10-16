"""合成演示数据（含合成参考音频，仅用于冒烟测试流程）。

为 3 个虚拟说话人各合成一段参考 wav（不同"语速"：振幅调制脉冲的密度不同），
再生成文本，时长公式：duration = 字数/语速 + 标点*0.12，叠加情绪/倍率/噪声。
"""
import argparse
import os
import random
import wave

import numpy as np
import pandas as pd

SR = 16000
# 虚拟说话人：每秒字符数（语速）
SPEAKERS = {"narrator": 3.6, "host": 4.4, "actor": 2.9}

TEMPLATES = [
    "你好，欢迎来到{place}。",
    "今天我们来讲一讲{topic}的故事。",
    "他说：{quote}",
    "注意！{warn}千万要小心。",
    "这是第{num}次尝试，我们一定会成功的。",
    "{greet}吃了吗？最近{topic}怎么样？",
    "为什么{topic}总是这样？真是让人{feeling}！",
    "在{place}的那个下午，一切都改变了……",
    "如果{cond}，那么{result}。",
    "他花了{num}年时间，终于完成了{topic}。",
]
WORDS = {
    "place": ["配音工作室", "上海", "录音棚", "直播间", "会议室"],
    "topic": ["时长预测", "语音合成", "配音对齐", "情绪控制", "视频剪辑"],
    "quote": ["明天见", "加油", "没问题", "再想想", "就这么定了"],
    "warn": ["前方高能", "前方施工", "内有剧透"],
    "greet": ["嗨", "喂", "嘿"],
    "feeling": ["头疼", "无奈", "哭笑不得"],
    "cond": ["明天下雨", "时间不够", "模型不收敛"],
    "result": ["就改天再说", "只能加班了", "先做个小模型验证"],
    "num": ["三", "五", "十", "一百"],
}
EMO_COLS = ["emo_happy", "emo_angry", "emo_sad", "emo_afraid",
            "emo_disgusted", "emo_melancholic", "emo_surprised", "emo_calm"]


def synth_wav(path: str, syl_rate: float, dur_s=6.0, seed=0):
    """合成"说话声"：按音节率随机 burst 的调幅噪声。"""
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    y = np.zeros(n, dtype=np.float32)
    t = 0.0
    while t < dur_s - 0.3:
        blen = rng.uniform(0.08, 0.22)
        s0 = int(t * SR)
        s1 = min(n, s0 + int(blen * SR))
        burst = rng.standard_normal(s1 - s0).astype(np.float32)
        env = np.hanning(max(s1 - s0, 1)).astype(np.float32)
        y[s0:s1] = burst * env * 0.5
        t += blen + rng.uniform(0.02, 1.2 / syl_rate)
    y = y / max(np.abs(y).max(), 1e-6) * 0.8
    pcm = (y * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo.csv")
    ap.add_argument("--wav-dir", default="data/demo_wavs")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.wav_dir, exist_ok=True)

    wav_paths = {}
    for i, (spk, cps) in enumerate(SPEAKERS.items()):
        # 音节率 ≈ 字速/1.6（每音节约1.6字，仅合成用）
        p = os.path.join(a.wav_dir, f"{spk}.wav")
        synth_wav(p, syl_rate=cps / 1.6, seed=100 + i)
        wav_paths[spk] = p
    print(f"[合成音频] {len(wav_paths)} 个 -> {a.wav_dir}")

    rows = []
    for _ in range(a.n):
        tpl = rng.choice(TEMPLATES)
        text = tpl
        for k, vs in WORDS.items():
            text = text.replace("{" + k + "}", rng.choice(vs))
        if rng.random() < 0.3:
            text += rng.choice(["真的吗？", "太好了！", "然后呢……", "对吧？"])
        spk = rng.choice(list(SPEAKERS))
        cps = SPEAKERS[spk] * rng.uniform(0.92, 1.08)
        emo = [0.0] * 8
        emo_name = rng.choices(["calm", "happy", "angry", "sad"], [0.7, 0.1, 0.1, 0.1])[0]
        emo[{"happy": 0, "angry": 1, "sad": 2, "calm": 7}[emo_name]] = round(rng.uniform(0.5, 1.0), 2)
        dfac = rng.choice([0.8, 1.0, 1.0, 1.25])
        n_punct = sum(1 for ch in text if ch in "，。！？；：、…—")
        dur = (len(text) / cps + n_punct * 0.12)
        if emo_name == "angry":
            dur *= 0.88
        elif emo_name == "sad":
            dur *= 1.18
        dur = dur * dfac * rng.uniform(0.95, 1.05)
        row = {"text": text, "duration": round(max(dur, 0.5), 2),
               "spk_audio": wav_paths[spk], "emo_audio": "",
               "emo_alpha": 1.0, "duration_factor": dfac, "lang": "ZH"}
        for i, c in enumerate(EMO_COLS):
            row[c] = emo[i]
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"[演示数据] {len(df)} 行 -> {a.out}（合成数据，仅冒烟测试）")


if __name__ == "__main__":
    main()
