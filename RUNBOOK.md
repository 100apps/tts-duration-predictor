# 5060 Ti 运行手册（训练 / 推理 / 控制台）

## 0. 环境准备

- GPU：RTX 5060 Ti 16GB（Blackwell 架构，需 CUDA 12.8+ 的 PyTorch）
- Python 3.10+，建议 venv / conda 独立环境

```bash
git clone <你的仓库地址> tts-duration-predictor && cd tts-duration-predictor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 先装 torch（cu128+ 才支持 5060 Ti），再装其余依赖
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出：2.x.x+cu128 True
```

> 若 `torch.cuda.is_available()` 为 False：先 `nvidia-smi` 确认驱动 ≥ 570，
> 再重装 cu128 的 torch。CPU 也能跑（单句约 3ms），只是训练慢。

## 1. 控制台（推荐入口）

```bash
python -m durpred.console
```

菜单：1 造演示数据 → 2 生成 teacher 数据 → 3 训练 → 4 单句预测 → 5 批量预测 → 6 评估。

## 2. 冒烟测试（5 分钟跑通全流程）

```bash
python tools/make_demo_data.py --out data/demo.csv --wav-dir data/demo_wavs --n 600
python -m durpred.train --data data/demo.csv --out models/demo --epochs 20
python -m durpred.predict --model models/demo \
    --text "你好，欢迎来到配音工具。" --spk-audio data/demo_wavs/narrator.wav
```

看到 `预估时长: X.XXs` 即跑通。注意：演示数据是合成的，精度无意义，
只验证流程。

## 3. 生成 teacher 真数据（关键步骤，慢）

用 IndexTTS2.5 当 teacher，对每条语料实测时长。语料 CSV：

```
text,spk_audio[,emo_audio,emo_alpha,duration_factor,lang,emo_happy,...,emo_calm]
```

```bash
# 先准备好官方 index-tts 仓库和 IndexTTS-2.5 checkpoints
nohup python tools/make_teacher_data.py \
  --corpus data/corpus.csv \
  --ref-audio /path/to/voice.wav \
  --out data/teacher.csv \
  --index-tts-dir ~/index-tts \
  --checkpoints ~/checkpoints \
  > teacher.log 2>&1 &
```

- 1 万条在 5060 Ti 上约数小时，可 `tail -f teacher.log` 看进度（每 20 条打印 ETA）
- 建议：直接用你的配音译文当语料（领域一致，效果最好）；
  情绪/语速参数尽量覆盖你实际会用的组合
- 中途中断可重跑（输出是追加写 CSV，已生成的行保留——重跑前先备份）

## 4. 正式训练

```bash
python -m durpred.train --data data/teacher.csv --out models/dur_v1 --epochs 300
```

- 默认早停（patience 25），1 万条在 5060 Ti 上约几分钟（音频特征首次提取后缓存）
- 看 `val_MAE`：合成数据上约 0.3~0.5s，真数据上 0.5~1.0s 属正常；
  目标是**明显优于"字数×系数"**，不是零误差
- 调参：数据 < 3000 条时把 `configs/default.yaml` 里 batch_size 改小、patience 改大

训练完把 `models/dur_v1` 发我，我帮你看收敛曲线和误差分布，决定要不要加数据或调参。

## 5. 推理与集成

```bash
# 单句（参数与 IndexTTS2.5 infer() 对齐）
python -m durpred.predict --model models/dur_v1 \
  --text "这段译文念出来多长？" --spk-audio ref.wav \
  --emo-audio emo_angry.wav --duration-factor 1.1 --lang ZH

# 批量（CSV 需 text,spk_audio 列；输出加 pred_duration 列）
python -m durpred.predict --model models/dur_v1 \
  --csv data/batch.csv --csv-out data/batch_pred.csv
```

Python 集成见 README。典型用法：100 段先批量预筛，
超时段改写/调 `duration_factor`，通过的再走 T2S 精确门控。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | 重装 cu128 torch，检查驱动 ≥ 570 |
| `FileNotFoundError: 音色音频不存在` | CSV 里 `spk_audio` 写绝对路径或相对仓库的正确路径；只支持 wav |
| teacher 生成某条跳过 | 看日志的 `[跳过]` 行，多为文本过长或音频问题，不影响整体 |
| val_MAE 不降 | 数据量太小或文本/时长几乎无相关性（检查 CSV 的 duration 列是否真实） |
| 推理报 shape 错误 | 模型与代码版本不一致，重新训练；不要混用不同版本的 model.pt |
