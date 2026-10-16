"""用 IndexTTS2.5 当 teacher 生成真实训练数据。

语料 CSV 列：text（必填）
可选：spk_audio, emo_audio, emo_alpha, duration_factor, lang, emo_happy..emo_calm
（缺省用命令行参数 --ref-audio / --lang）

输出列：text,duration,spk_audio,emo_audio,emo_alpha,duration_factor,lang,
         emo_happy..emo_calm,sem_tokens(若能探测到)

注意：这一步慢（每条约一次完整 TTS），1万条在 5060 Ti 上约数小时，
建议 nohup 后台跑。
"""
import argparse
import csv
import os
import sys
import time
import wave

EMO_COLS = [
    "emo_happy", "emo_angry", "emo_sad", "emo_afraid",
    "emo_disgusted", "emo_melancholic", "emo_surprised", "emo_calm",
]

_TOKEN_ATTR_GUESSES = [
    "last_semantic_tokens", "last_sem_tokens", "_last_tokens",
    "semantic_tokens", "last_tokens",
]


def probe_token_count(tts):
    for attr in _TOKEN_ATTR_GUESSES:
        v = getattr(tts, attr, None)
        if v is not None:
            try:
                return int(len(v))
            except TypeError:
                pass
    return ""


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ref-audio", default="", help="默认音色参考音频")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-tts-dir", required=True, help="index-tts 仓库目录")
    ap.add_argument("--checkpoints", required=True, help="IndexTTS-2.5 checkpoints 目录")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--lang", default="ZH")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, a.index_tts_dir)
    try:
        from indextts.infer_v2_5 import IndexTTS2
    except ImportError as e:
        print(f"[错误] 无法导入 indextts.infer_v2_5：{e}")
        sys.exit(1)

    cfg_path = a.cfg or os.path.join(a.checkpoints, "config.yaml")
    print("[加载] IndexTTS2.5 模型…")
    tts = IndexTTS2(cfg_path=cfg_path, model_dir=a.checkpoints, use_bf16=True)
    print("[就绪]")

    import pandas as pd

    df = pd.read_csv(a.corpus)
    assert "text" in df.columns, "语料需要 text 列"
    if a.limit:
        df = df.iloc[: a.limit]
    for c in EMO_COLS:
        if c not in df.columns:
            df[c] = 0.0
    for c, d in (("emo_alpha", 1.0), ("duration_factor", 1.0), ("lang", a.lang),
                 ("spk_audio", ""), ("emo_audio", "")):
        if c not in df.columns:
            df[c] = d
    df = df.fillna({"emo_audio": "", "spk_audio": ""})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp_wav = os.path.join(os.path.dirname(a.out) or ".", "_teacher_tmp.wav")
    t_start, done = time.time(), 0
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["text", "duration", "spk_audio", "emo_audio", "emo_alpha",
                    "duration_factor", "lang"] + EMO_COLS + ["sem_tokens"])
        for i, r in df.iterrows():
            ref = r["spk_audio"] or a.ref_audio
            if not ref:
                print("[错误] 缺少音色参考音频（--ref-audio 或语料 spk_audio 列）")
                sys.exit(1)
            emo = [float(r[c]) for c in EMO_COLS]
            kw = {}
            if r["emo_audio"]:
                kw["emo_audio_prompt"] = r["emo_audio"]
            elif any(v > 0 for v in emo):
                kw["emo_vector"] = emo
            if float(r.get("emo_alpha", 1.0)) != 1.0:
                kw["emo_alpha"] = float(r["emo_alpha"])
            dfac = float(r.get("duration_factor", 1.0))
            if dfac != 1.0:
                kw["duration_factor"] = dfac
            try:
                tts.infer(spk_audio_prompt=ref, text=str(r["text"]),
                          lang=str(r.get("lang", a.lang)),
                          output_path=tmp_wav, verbose=False, **kw)
                dur = wav_duration(tmp_wav)
            except Exception as e:  # noqa: BLE001
                print(f"[跳过 #{i}] {e}")
                continue
            w.writerow([r["text"], round(dur, 3), ref, r["emo_audio"],
                        r["emo_alpha"], dfac, r["lang"],
                        *[round(v, 3) for v in emo], probe_token_count(tts)])
            done += 1
            if done % 20 == 0:
                el = time.time() - t_start
                print(f"[进度] {done}/{len(df)}  已用 {el/60:.1f}min  "
                      f"预计剩余 {(len(df)-done)*(el/max(done,1))/60:.1f}min")
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
    print(f"[完成] {done} 条 -> {a.out}")


if __name__ == "__main__":
    main()
