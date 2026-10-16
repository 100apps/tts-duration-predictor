"""字符级词表。"""
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class CharVocab:
    def __init__(self, stoi: dict):
        self.stoi = dict(stoi)
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.stoi)

    @classmethod
    def build(cls, texts, min_freq: int = 2) -> "CharVocab":
        from collections import Counter

        cnt = Counter()
        for t in texts:
            cnt.update(t)
        stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for ch, c in cnt.most_common():
            if c < min_freq:
                break
            if ch not in stoi:
                stoi[ch] = len(stoi)
        return cls(stoi)

    def encode(self, text: str, max_len: int):
        return [self.stoi.get(ch, self.unk_id) for ch in text[:max_len]]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharVocab":
        with open(path, encoding="utf-8") as f:
            stoi = json.load(f)
        # json keys are strings for the chars themselves (fine); values are ints
        return cls({k: int(v) for k, v in stoi.items()})
