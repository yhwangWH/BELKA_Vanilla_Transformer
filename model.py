"""
Vanilla Transformer Encoder 多任务分类模型。

架构：
    Embedding → Positional Encoding → Transformer Encoder → [CLS] Pooling → MLP Head

输出：3 个 logits（BRD4, HSA, sEH）
"""

import math
import torch
import torch.nn as nn


# ── Sin/Cos Positional Encoding ─────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    sin/cos 位置编码（不可学习）。
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── Transformer Encoder 模型 ────────────────────────────────────────
class BELKATransformer(nn.Module):
    """
    SMILES → Transformer Encoder → 多任务分类头
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_length: int = 256,
        pad_idx: int = 0,
        num_targets: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # ── Embedding ──
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx
        )
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Pooling（[CLS] token）──
        # CLS token 在位置 0，直接用第一个 token 的 hidden state

        # ── MLP Head ──
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_targets),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) token ids

        Returns:
            logits: (B, 3) 三个 target 的 logits
        """
        # Padding mask: True = 忽略该位置
        padding_mask = input_ids == self.pad_idx  # (B, seq_len)

        # Embedding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)  # (B, S, D)
        x = self.pos_encoding(x)

        # Transformer Encoder
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )  # (B, S, D)

        # [CLS] Pooling: 取第一个 token
        cls_hidden = x[:, 0, :]  # (B, D)

        # MLP Head
        logits = self.head(cls_hidden)  # (B, 3)
        return logits


# ── 损失函数 ─────────────────────────────────────────────────────────
def create_loss_fn(pos_weight: torch.Tensor) -> nn.Module:
    """
    创建带 pos_weight 的 BCEWithLogitsLoss。

    Args:
        pos_weight: shape (3,) — 每个 target 的正样本权重
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ── 测试 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vocab_size = 50
    model = BELKATransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        max_length=128,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch = torch.randint(1, vocab_size, (4, 64))  # (B, S)
    # 加入 padding
    batch[:, -10:] = 0

    logits = model(batch)
    print(f"Input: {batch.shape} → Output: {logits.shape}")
    print(f"Logits: {logits}")

    # 测试损失
    pos_weight = torch.tensor([10.0, 5.0, 8.0])
    loss_fn = create_loss_fn(pos_weight)
    labels = torch.randint(0, 2, (4, 3)).float()
    loss = loss_fn(logits, labels)
    print(f"Loss: {loss.item():.4f}")
