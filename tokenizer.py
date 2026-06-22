"""
SMILES Tokenizer — 基于字符遍历的 SMILES 分词器。

支持：
- 方括号内 token：如 [Dy], [C@@H], [O-], [NH3+] 等
- 双字符原子：Cl, Br
- 单字符原子：B, C, N, O, S, P, F, I（大写）+ b, c, n, o, s, p（芳香小写）
- % 两位数环编号：如 %12
- 一位数环编号/同位素
- 键：=, #, -
- 括号：( )
- 手性：@, @@
- 方向键：\, /
- 电荷：+, -
- 分隔符：.

特殊 token：
    [PAD]  (id=0) — 填充
    [UNK]  (id=1) — 未知
    [CLS]  (id=2) — 分类头
"""

import json
from typing import List, Dict, Optional

# ── 特殊 token ──────────────────────────────────────────────────────
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
CLS_IDX = 2


class SMILESTokenizer:
    """SMILES 分词器 + 词表管理（字符遍历实现）"""

    def __init__(self, max_length: int = 256, vocab: Optional[Dict[str, int]] = None):
        self.max_length = max_length
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}

        if vocab is not None:
            self._vocab = dict(vocab)
            self._reverse_vocab = {v: k for k, v in vocab.items()}

    # ── 分词（核心） ─────────────────────────────────────────────
    def tokenize(self, smiles: str) -> List[str]:
        """
        将 SMILES 字符串分割为 token 列表。

        规则（按优先级）：
        1. [ ... ]  方括号内所有内容作为一个 token
        2. %\d\d     % 后跟两位数字作为环编号 token
        3. Br, Cl    双字符原子
        4. 单个字符   其余所有字符分别作为一个 token
        """
        tokens = []
        i = 0
        n = len(smiles)

        while i < n:
            c = smiles[i]

            # 1) 方括号 [ ... ]
            if c == "[":
                j = smiles.index("]", i) + 1
                tokens.append(smiles[i:j])
                i = j

            # 2) % 两位数环编号
            elif c == "%" and i + 2 < n and smiles[i + 1].isdigit() and smiles[i + 2].isdigit():
                tokens.append(smiles[i : i + 3])
                i += 3

            # 3) 双字符原子 Br / Cl
            elif i + 1 < n and smiles[i : i + 2] in ("Br", "Cl"):
                tokens.append(smiles[i : i + 2])
                i += 2

            # 4) 单字符
            else:
                tokens.append(c)
                i += 1

        return tokens

    # ── 词表构建 ──────────────────────────────────────────────────
    def build_vocab(self, smiles_list: List[str]) -> int:
        """从 SMILES 列表收集所有唯一 token 并构建词表"""
        tokens_set = set()
        for smi in smiles_list:
            tokens_set.update(self.tokenize(smi))

        # 特殊 token 固定在前面
        self._vocab = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self._vocab[tok] = i

        for tok in sorted(tokens_set):
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)

        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
        return len(self._vocab)

    # ── 编码 / 解码 ───────────────────────────────────────────────
    def encode(
        self, smiles: str, add_cls: bool = True, return_tensors: bool = False
    ):
        """
        SMILES → token ids.

        Args:
            smiles: SMILES 字符串
            add_cls: 是否在开头添加 [CLS] token
            return_tensors: 是否返回 torch.Tensor

        Returns:
            list[int] 或 torch.Tensor
        """
        tokens = self.tokenize(smiles)

        # 截断（为 CLS 留一个位置）
        max_token_len = self.max_length - (1 if add_cls else 0)
        tokens = tokens[:max_token_len]

        ids = [self._vocab.get(t, UNK_IDX) for t in tokens]

        if add_cls:
            ids = [CLS_IDX] + ids

        if return_tensors:
            import torch
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Token ids → SMILES 字符串"""
        tokens = []
        for i in ids:
            if skip_special and i in (PAD_IDX, UNK_IDX, CLS_IDX):
                continue
            if i in self._reverse_vocab:
                tokens.append(self._reverse_vocab[i])
        return "".join(tokens)

    # ── 属性 ──────────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    # ── 序列化 ────────────────────────────────────────────────────
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"max_length": self.max_length, "vocab": self._vocab},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "SMILESTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(max_length=data["max_length"], vocab=data["vocab"])
        return tok


# ── 测试 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    smiles_list = [
        "C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)N[Dy])n2)cc1",
        "BrC1=CC=CC=C1",
        "ClCCCl",
        "CCO",
        "C1CCCCC1",
        "[O-][N+](=O)c1ccccc1",
        "F[C@@H](C)O",
        "C%12CC%12C",
        "CC(=O)O",
        "C#CCOc1ccc(CN)cc1.Cl",
    ]

    all_ok = True
    for smi in smiles_list:
        tokenizer = SMILESTokenizer(max_length=256)
        tokenizer._vocab = {}  # 空词表，只测试 tokenize
        tokens = tokenizer.tokenize(smi)
        reconstructed = "".join(tokens)
        ok = (reconstructed == smi)
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {smi[:60]}")
        if not ok:
            print(f"        Reconstructed: {reconstructed[:80]}")
            print(f"        Tokens: {tokens}")
        else:
            print(f"        {len(tokens)} tokens: {tokens[:15]}{'...' if len(tokens)>15 else ''}")

    print(f"\nAll round-trip OK: {all_ok}")

    # 测试词表 + 编码
    tokenizer = SMILESTokenizer(max_length=256)
    vocab_size = tokenizer.build_vocab(smiles_list)
    print(f"\nVocab size: {vocab_size}")
    print(f"Sample vocab: {list(tokenizer._vocab.keys())[:30]}")

    encoded = tokenizer.encode(smiles_list[0])
    decoded = tokenizer.decode(encoded)
    print(f"Encoded length: {len(encoded)}")
    print(f"Decoded == original: {decoded == smiles_list[0]}")
