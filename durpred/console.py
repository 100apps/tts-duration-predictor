"""交互式控制台：python -m durpred.console"""
import os
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .predict import Predictor

console = Console()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    subprocess.run(cmd, cwd=ROOT)


def ask(prompt, default=""):
    s = console.input(f"[bold cyan]{prompt}[/bold cyan]" + (f" [dim](默认 {default})[/dim]" if default else "") + ": ")
    s = s.strip()
    return s or default


def menu():
    t = Table(title="TTS 时长蒸馏预测器 · 控制台", show_header=False)
    t.add_column("选项", style="bold yellow")
    t.add_column("说明")
    t.add_row("1", "生成演示数据（含合成参考音频，仅冒烟测试）")
    t.add_row("2", "用 IndexTTS2.5 生成 teacher 数据（真数据，慢）")
    t.add_row("3", "训练模型")
    t.add_row("4", "单句预测（参数对齐 IndexTTS2.5 infer）")
    t.add_row("5", "批量预测 CSV")
    t.add_row("6", "评估模型（val MAE）")
    t.add_row("0", "退出")
    console.print(t)


def do_demo():
    out = ask("输出 CSV 路径", "data/demo.csv")
    wavdir = ask("合成音频目录", "data/demo_wavs")
    n = ask("条数", "600")
    run([sys.executable, "tools/make_demo_data.py", "--out", out,
         "--wav-dir", wavdir, "--n", n])


def do_teacher():
    corpus = ask("语料 CSV 路径", "data/corpus.csv")
    ref = ask("默认参考音频 wav 路径（语料无 spk_audio 列时用）", "")
    out = ask("输出 CSV 路径", "data/teacher.csv")
    idx_dir = ask("index-tts 仓库目录", os.path.expanduser("~/index-tts"))
    ckpt = ask("IndexTTS-2.5 checkpoints 目录", os.path.expanduser("~/checkpoints"))
    cmd = [sys.executable, "tools/make_teacher_data.py", "--corpus", corpus,
           "--out", out, "--index-tts-dir", idx_dir, "--checkpoints", ckpt]
    if ref:
        cmd += ["--ref-audio", ref]
    run(cmd)


def do_train():
    data = ask("训练 CSV 路径", "data/demo.csv")
    out = ask("模型输出目录", "models/dur_v1")
    epochs = ask("epochs（空=用配置）", "")
    cmd = [sys.executable, "-m", "durpred.train", "--data", data, "--out", out]
    if epochs:
        cmd += ["--epochs", epochs]
    run(cmd)


def _parse_vec(s):
    if not s.strip():
        return None
    v = [float(x) for x in s.split(",")]
    assert len(v) == 8, "需要 8 个逗号分隔的数"
    return v


def do_predict_one():
    model = ask("模型目录", "models/dur_v1")
    if not os.path.exists(os.path.join(model, "model.pt")):
        console.print("[red]模型不存在，先训练（选项 3）[/red]")
        return
    p = Predictor.load(model)
    console.print("[dim]输入空文本返回主菜单[/dim]")
    while True:
        text = ask("文字 text")
        if not text:
            return
        spk = ask("音色参考音频 spk_audio_prompt（wav 路径）")
        if not spk:
            console.print("[red]音色音频必填[/red]")
            continue
        emo_a = ask("情绪参考音频 emo_audio_prompt（空=无）", "")
        emo_v = ask("情绪向量 emo_vector（8个逗号分隔，空=无）", "")
        emo_al = float(ask("情绪强度 emo_alpha", "1.0") or 1.0)
        dfac = float(ask("语速 duration_factor（>1变慢）", "1.0") or 1.0)
        lang = ask("语言 lang", "ZH")
        try:
            sec = p.predict(text, spk, emo_a, _parse_vec(emo_v), emo_al, dfac, lang)
            console.print(Panel(f"预估时长：[bold green]{sec:.2f}s[/bold green]", title="预测结果"))
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]出错：{e}[/red]")


def do_predict_batch():
    model = ask("模型目录", "models/dur_v1")
    csv = ask("输入 CSV（列：text,duration?,spk_audio[,…]）")
    out = ask("输出 CSV", "data/pred_out.csv")
    if not os.path.exists(os.path.join(model, "model.pt")):
        console.print("[red]模型不存在，先训练（选项 3）[/red]")
        return
    run([sys.executable, "-m", "durpred.predict", "--model", model,
         "--csv", csv, "--csv-out", out])


def do_eval():
    from .data import load_table
    model = ask("模型目录", "models/dur_v1")
    csv = ask("评估 CSV（需 text,duration,spk_audio 列）", "data/demo.csv")
    p = Predictor.load(model)
    df = load_table(csv)
    out = p.predict_df(df)
    err = (out["pred_duration"] - out["duration"]).abs()
    console.print(Panel(
        f"样本数：{len(out)}\nMAE：[bold green]{err.mean():.3f}s[/bold green]\n"
        f"中位误差：{err.median():.3f}s\n90分位误差：{err.quantile(0.9):.3f}s",
        title="评估结果"))


def main():
    console.print(Panel("[bold]TTS 时长蒸馏预测器[/bold]\n"
                        "输入与 IndexTTS2.5 infer() 对齐：文字 / 音色音频 / 情绪音频 / 情绪向量 / 语速",
                        title="durpred console"))
    actions = {"1": do_demo, "2": do_teacher, "3": do_train,
               "4": do_predict_one, "5": do_predict_batch, "6": do_eval}
    while True:
        menu()
        c = ask("选择", "0").strip()
        if c == "0":
            console.print("再见 👋")
            return
        fn = actions.get(c)
        if fn:
            try:
                fn()
            except KeyboardInterrupt:
                console.print("\n[dim]已取消[/dim]")
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]出错：{e}[/red]")
        else:
            console.print("[red]无效选项[/red]")


if __name__ == "__main__":
    main()
