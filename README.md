# TTS 时长蒸馏预测器（tts-duration-predictor）

给 IndexTTS2.5 配音管线用的**毫秒级时长预估网络**。

## 输入（与 IndexTTS2.5 `infer()` 参数对齐，原模型支持的都传）

| 参数 | 说明 |
|---|---|
| `text` | 文字（必填） |
| `spk_audio_prompt` | 音色参考音频 wav（必填）→ 提取语速/能量先验，**新音色零样本泛化，无需重训** |
| `emo_audio_prompt` | 情绪参考音频（可选） |
| `emo_vector` | 8 维情绪向量（可选） |
| `emo_alpha` | 情绪强度（可选，默认 1.0） |
| `duration_factor` | 语速倍率，>1 变慢（可选，默认 1.0，输出端直接相乘） |
| `lang` | ZH / EN / JA / ES / AR（可选） |

输出：预估音频时长（秒）。约 2M 参数，CPU 单句 < 10ms。

设计思路：IndexTTS2.5 的时长在 T2S（GPT）阶段由语义 token 数完全决定。
本仓库训练一个蒸馏小模型拟合 `(上述参数) -> 时长`，用于批量预筛；
通过门控的段再走 T2S-only 精确确认（token 数即精确时长），零算力浪费。

## 快速开始（RTX 5060 Ti 16GB）

```bash
# 1. 安装 PyTorch（5060 Ti 是 Blackwell 架构，需要 CUDA 12.8+ 的 torch）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. 启动控制台（交互菜单：造数据 / teacher 数据 / 训练 / 预测 / 评估）
python -m durpred.console

# 或直接命令行（冒烟测试）：
python tools/make_demo_data.py --out data/demo.csv --wav-dir data/demo_wavs --n 600
python -m durpred.train --data data/demo.csv --out models/demo --epochs 20
python -m durpred.predict --model models/demo \
    --text "你好，欢迎来到配音工具。" --spk-audio data/demo_wavs/narrator.wav
```

## 真实训练数据（teacher 数据）

演示数据只是为了跑通流程。真正有用的是用 IndexTTS2.5 当 teacher 生成的
全参数 pairs。语料 CSV 列：`text`（必填），可选
`spk_audio, emo_audio, emo_alpha, duration_factor, lang, emo_happy..emo_calm`：

```bash
python tools/make_teacher_data.py \
  --corpus data/corpus.csv \
  --ref-audio /path/to/voice.wav \
  --out data/teacher.csv \
  --index-tts-dir /path/to/index-tts \
  --checkpoints /path/to/checkpoints

python -m durpred.train --data data/teacher.csv --out models/dur_v1 --epochs 300
```

建议 teacher 数据量 1 万条起；你的配音文本本身就是最好的语料（领域一致）。
注意：teacher 生成慢（每条一次完整 TTS），建议 `nohup` 后台跑。

## 训练 CSV 格式

```
text,duration,spk_audio[,emo_audio,emo_alpha,duration_factor,lang,emo_happy,...,emo_calm]
```

- `duration`：实测音频时长（秒）
- 音频特征带磁盘缓存（`.durpred_cache/`），首个 epoch 后训练极快

## 集成到配音工具

```python
from durpred.predict import Predictor
p = Predictor.load("models/dur_v1")
sec = p.predict(
    "这段译文念出来大概多长？",
    spk_audio_prompt="ref.wav",
    emo_audio_prompt="emo_angry.wav",
    duration_factor=1.1,
    lang="ZH",
)
# 批量：p.predict_df(df) -> DataFrame 新增 pred_duration 列
```

## 模型

- 文本分支：字符 embedding(128) + 3 层 TransformerEncoder + 均值池化 → 48 维
- 音色分支：参考音频 8 维统计（时长/浊音占比/音节率估计/能量…）→ 16 维
- 情绪分支：情绪音频统计 + 8 维向量 + alpha → 16 维；语言 embedding 8 维
- 拼接 7 维人工特征（字数/标点/数字/句末标点…）+ log(duration_factor)
- MLP 回归 log-domain MSE；`duration_factor` 在输出端直接相乘

## 精度预期（诚实版）

- 明显优于"字数×系数"线性回归；音色语速、情绪、标点带来的偏离能学到一部分
- 但到不了 T2S 自报的精确度：时长本质是自回归采样的 emergent 属性，
  强情绪、特殊标点等长尾会有误差
- 推荐用法：蒸馏版做 100 段批量粗筛 → T2S-only 做精确门控 → 通过的直接进 S2M+声码器
