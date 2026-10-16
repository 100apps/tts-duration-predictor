"""人工特征：字数、标点数、数字、句末标点等。给小模型一个强先验。"""
import numpy as np

PUNCT = set("，。！？；：、…—·「」『』（）《》〈〉,.!?;:\"'()[]-～·＂＇")
END_EXCL = set("!！")
END_QUES = set("?？")

FEAT_NAMES = [
    "n_chars",
    "n_punct",
    "n_digits",
    "end_exclaim",
    "end_question",
    "has_ellipsis",
    "punct_ratio",
]


def engineered(text: str) -> np.ndarray:
    t = text.strip()
    n_chars = len(t)
    n_punct = sum(1 for ch in t if ch in PUNCT)
    n_digits = sum(1 for ch in t if ch.isdigit())
    end_exclaim = 1.0 if t and t[-1] in END_EXCL else 0.0
    end_question = 1.0 if t and t[-1] in END_QUES else 0.0
    has_ellipsis = 1.0 if ("…" in t or "..." in t) else 0.0
    punct_ratio = n_punct / max(n_chars, 1)
    return np.array(
        [n_chars, n_punct, n_digits, end_exclaim, end_question, has_ellipsis, punct_ratio],
        dtype=np.float32,
    )
