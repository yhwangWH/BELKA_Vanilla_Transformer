"""
# BELKA Vanilla Transformer — Kaggle Notebook 单文件版本

直接在 Kaggle Notebook 中 Run All 即可完成：训练 → 验证 → 推理 → 生成 submission。

## 本地测试（用小数据验证）:
    python kaggle_notebook.py

## Kaggle 使用:
    将本文件内容复制到 Notebook 中，Run All 即可。

---

遵循 General prompt.txt 重构要求：
- 单文件架构，禁止跨文件 import
- Section 清晰分割
- CFG 统一配置
- 支持 Kaggle 路径自动发现
"""

# ============================================================
# Section 1: Imports & Environment Setup
# ============================================================
import os
import sys
import json
import math
import time
import logging
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── 日志 ──
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("BELKA")

# ── 设备 ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 检测 Kaggle 环境 ──
# Kaggle Notebook 会设置 KAGGLE_URL_BASE 等环境变量
IS_KAGGLE = (
    os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
    or os.environ.get("KAGGLE_URL_BASE", "") != ""
)
INPUT_DIR  = "/kaggle/input"  if IS_KAGGLE else "data"
OUTPUT_DIR = "/kaggle/working" if IS_KAGGLE else "ckpt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info(f"{'='*60}")
logger.info(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
logger.info(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
logger.info(f"PyTorch: {torch.__version__}")
logger.info(f"Input dir:  {INPUT_DIR}")
logger.info(f"Output dir: {OUTPUT_DIR}")
logger.info(f"{'='*60}")


# ============================================================
# Section 2: CFG — 统一配置
# ============================================================
class CFG:
    """所有超参数集中管理，禁止魔法数字散落代码中。"""

    # ── 随机种子 ──
    seed = 42

    # ── 数据 ──
    max_len = 128           # SMILES 最大 token 数
    max_negatives = None    # 负样本上限（None=全量，本地测试建议设小值）
    sample_frac = None      # 随机采样比例（None=全量）
    train_ratio = 0.9       # 训练/验证分割比例

    # ── 模型 ──
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    num_targets = 3

    # ── 训练 ──
    batch_size = 64
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-2
    grad_clip = 1.0
    num_workers = 0         # Notebook 建议设为 0 (Windows/WSL兼容)
    use_amp = True          # 混合精度（T4/P100 支持）
    patience = 5            # Early stopping

    # ── 路径（运行时自动设置） ──
    train_path = ""
    test_path = ""
    vocab_path = ""
    model_path = ""
    submission_path = ""


# ============================================================
# Section 3: SMILES Tokenizer
# ============================================================

# 特殊 token
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
CLS_IDX = 2


class SmilesTokenizer:
    """
    SMILES 分词器 — 基于字符遍历的 SMILES tokenization。

    支持：
    - 方括号内 token：如 [Dy], [C@@H], [O-], [NH3+]
    - 双字符原子：Cl, Br
    - % 两位数环编号：如 %12
    - 单字符 token（原子、键、括号等）

    特殊 token：
        [PAD] (id=0) — 填充
        [UNK] (id=1) — 未知
        [CLS] (id=2) — 分类头
    """

    def __init__(self, max_length: int = 128, vocab: Optional[Dict[str, int]] = None):
        self.max_length = max_length
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}

        if vocab is not None:
            self._vocab = dict(vocab)
            self._reverse_vocab = {v: k for k, v in vocab.items()}

    # ── 分词（核心） ──
    @staticmethod
    def tokenize(smiles: str) -> List[str]:
        """
        将 SMILES 字符串分割为 token 列表。

        规则（按优先级）：
        1. [ ... ]  方括号内所有内容作为一个 token
        2. %\\d\\d    % 后跟两位数字作为环编号 token
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

    # ── 词表构建 ──
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

    # ── 编码 / 解码 ──
    def encode(self, smiles: str, add_cls: bool = True) -> List[int]:
        """
        SMILES → token ids.

        Args:
            smiles: SMILES 字符串
            add_cls: 是否在开头添加 [CLS] token

        Returns:
            List[int] token ids
        """
        tokens = self.tokenize(smiles)

        # 截断（为 CLS 留一个位置）
        max_token_len = self.max_length - (1 if add_cls else 0)
        tokens = tokens[:max_token_len]

        ids = [self._vocab.get(t, UNK_IDX) for t in tokens]

        if add_cls:
            ids = [CLS_IDX] + ids

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

    # ── 属性 ──
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    # ── 序列化 ──
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"max_length": self.max_length, "vocab": self._vocab},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "SmilesTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(max_length=data["max_length"], vocab=data["vocab"])


# ============================================================
# Section 4: Dataset & DataLoader
# ============================================================

class BELKADataset(Dataset):
    """
    BELKA 分子数据集。

    输入: SMILES 字符串列表
    输出: (token_ids, label_vector)
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[np.ndarray],
        tokenizer: SmilesTokenizer,
        max_length: int = 128,
    ):
        self.smiles_list = smiles_list
        self.labels = labels  # (N, 3) or None (for test)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        smiles = self.smiles_list[idx]
        token_ids = self.tokenizer.encode(smiles, add_cls=True)
        padded = self._pad(token_ids, self.max_length, PAD_IDX)
        input_ids = torch.tensor(padded, dtype=torch.long)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return input_ids, label
        else:
            return input_ids, torch.tensor(0)

    @staticmethod
    def _pad(ids: List[int], max_len: int, pad_id: int) -> List[int]:
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [pad_id] * (max_len - len(ids))


def create_dataloaders(
    smiles_train: List[str],
    labels_train: np.ndarray,
    smiles_val: List[str],
    labels_val: np.ndarray,
    tokenizer: SmilesTokenizer,
    batch_size: int = 64,
    max_length: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader"""
    train_ds = BELKADataset(smiles_train, labels_train, tokenizer, max_length)
    val_ds   = BELKADataset(smiles_val,   labels_val,   tokenizer, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )
    return train_loader, val_loader


def create_test_dataloader(
    smiles_test: List[str],
    tokenizer: SmilesTokenizer,
    batch_size: int = 128,
    max_length: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """创建测试 DataLoader"""
    test_ds = BELKADataset(smiles_test, None, tokenizer, max_length)
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )


# ============================================================
# Section 5: Model — Vanilla Transformer Encoder
# ============================================================

class PositionalEncoding(nn.Module):
    """sin/cos 位置编码（不可学习）。"""

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


class SmilesTransformer(nn.Module):
    """
    SMILES → Transformer Encoder → 多任务分类头。

    输出：3 个 logits（BRD4, HSA, sEH）
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_length: int = 128,
        pad_idx: int = 0,
        num_targets: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)

        # Transformer Encoder (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_targets),
        )

        self._init_weights()

    def _init_weights(self):
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
        padding_mask = input_ids == self.pad_idx  # (B, seq_len)

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # [CLS] Pooling: 取第一个 token
        cls_hidden = x[:, 0, :]
        return self.head(cls_hidden)


# ============================================================
# Section 6: Metrics — AUC 计算
# ============================================================

PROTEIN_NAMES = ["BRD4", "HSA", "sEH"]


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """
    计算 BCEWithLogitsLoss 的 pos_weight。
    pos_weight = num_negatives / num_positives
    """
    n_pos = labels.sum(axis=0).clip(min=1)
    n_neg = labels.shape[0] - n_pos
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算每个 target 的 ROC-AUC 及平均值。

    Args:
        y_true: (N, 3) 真实标签
        y_pred: (N, 3) 预测概率

    Returns:
        dict: {protein_name: auc, ..., "mean": mean_auc}
    """
    results = {}
    aucs = []
    for i, name in enumerate(PROTEIN_NAMES):
        unique_vals = np.unique(y_true[:, i])
        if len(unique_vals) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        results[name] = auc
        if not np.isnan(auc):
            aucs.append(auc)

    results["mean"] = np.mean(aucs) if aucs else float("nan")
    return results


def format_auc(auc_dict: dict) -> str:
    """格式化 AUC 字典为字符串"""
    parts = [f"{name}={auc_dict.get(name, float('nan')):.4f}" for name in PROTEIN_NAMES]
    parts.append(f"mean={auc_dict.get('mean', float('nan')):.4f}")
    return " | ".join(parts)


# ============================================================
# Section 7: Data Loading — 长格式→宽格式 pivot
# ============================================================

def load_train_data(path: str, max_negatives: Optional[int] = None) -> pd.DataFrame:
    """
    加载训练数据并 pivot 为宽格式（每个分子一行，3 个蛋白标签）。

    输入（长格式）: 每行 = (molecule, protein) pair
    输出（宽格式）: DataFrame with columns: molecule_smiles, BRD4, HSA, sEH

    对 parquet 文件使用流式读取，保留全部正样本 + 限制负样本数。
    """
    logger.info(f"Loading training data from {path} ...")

    if path.endswith(".parquet"):
        df = _load_parquet_streaming(path, max_negatives)
    else:
        df = pd.read_csv(path)

    total = len(df)
    positive = df["binds"].sum()
    logger.info(
        f"Train data: {total:,} rows, positive={positive:,} "
        f"({100 * positive / max(total, 1):.2f}%)"
    )

    # 统计每个蛋白
    for p in PROTEIN_NAMES:
        subset = df[df["protein_name"] == p]
        logger.info(f"  {p}: {len(subset):,} rows, binds={subset['binds'].sum():,}")

    # Pivot: long → wide
    pivot = df.pivot_table(
        index="molecule_smiles",
        columns="protein_name",
        values="binds",
        aggfunc="first",
    ).reset_index()

    for p in PROTEIN_NAMES:
        if p not in pivot.columns:
            pivot[p] = 0

    pivot = pivot[["molecule_smiles"] + PROTEIN_NAMES]
    pivot[PROTEIN_NAMES] = pivot[PROTEIN_NAMES].fillna(0).astype(int)

    for p in PROTEIN_NAMES:
        pos = pivot[p].sum()
        logger.info(f"  Wide {p}: {pos} positive out of {len(pivot)} molecules")

    logger.info(f"Pivoted to {len(pivot):,} unique molecules")
    return pivot


def _load_parquet_streaming(
    path: str, max_negatives: Optional[int] = None
) -> pd.DataFrame:
    """流式读取 parquet，保留全部正样本 + 随机采样负样本。"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    logger.info(f"Parquet file: {total_rows:,} total rows, streaming read...")

    positives = []
    negatives = []
    n_pos_total = 0
    n_neg_total = 0
    rng = np.random.RandomState(CFG.seed)

    # 需要的列（减少内存）
    keep_cols = ["molecule_smiles", "protein_name", "binds"]

    batch_size = 500_000
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
        # 只保留需要的列，并用 numpy 类型避免 PyArrow 兼容问题
        df_batch = batch.to_pandas()[keep_cols].copy()
        df_batch["binds"] = df_batch["binds"].astype("int8")

        pos_mask = df_batch["binds"] == 1
        neg_mask = ~pos_mask

        pos_batch = df_batch[pos_mask]
        neg_batch = df_batch[neg_mask]

        if len(pos_batch) > 0:
            positives.append(pos_batch)
            n_pos_total += len(pos_batch)

        n_neg_total += len(neg_batch)

        if max_negatives is not None and max_negatives > 0:
            neg_sample_rate = max_negatives / max(n_neg_total, 1)
            if len(neg_batch) > 0:
                n_neg_sample = max(1, int(len(neg_batch) * neg_sample_rate))
                neg_sampled = neg_batch.sample(
                    n=min(n_neg_sample, len(neg_batch)),
                    random_state=rng.randint(0, 2**31),
                )
                negatives.append(neg_sampled)
        else:
            if len(neg_batch) > 0:
                negatives.append(neg_batch)

        if (batch_idx + 1) % 50 == 0:
            n_neg_collected = sum(len(n) for n in negatives)
            logger.info(
                f"  Batch {batch_idx + 1}: {n_pos_total:,} positives, "
                f"{n_neg_collected:,} negatives collected"
            )

    pos_df = pd.concat(positives, ignore_index=True) if positives else pd.DataFrame()
    neg_df = pd.concat(negatives, ignore_index=True) if negatives else pd.DataFrame()

    if max_negatives is not None and len(neg_df) > max_negatives:
        neg_df = neg_df.sample(n=max_negatives, random_state=CFG.seed)

    logger.info(f"Streaming done: {len(pos_df):,} positives, {len(neg_df):,} negatives")
    return pd.concat([pos_df, neg_df], ignore_index=True)


def load_test_data(path: str) -> pd.DataFrame:
    """
    加载测试数据，提取去重后的 molecule_smiles。

    Returns:
        DataFrame with columns: molecule_id, molecule_smiles
    """
    logger.info(f"Loading test data from {path} ...")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    logger.info(f"Test data: {len(df):,} rows")
    grouped = df.groupby("molecule_smiles").agg(
        molecule_id=("id", "first"),
    ).reset_index()
    logger.info(f"Unique test molecules: {len(grouped):,}")
    return grouped


# ============================================================
# Section 8: Training Engine
# ============================================================
class Timer:
    """简单计时器上下文管理器"""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> float:
    """单 epoch 训练"""
    model.train()
    total_loss = 0.0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                logits = model(input_ids)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """验证"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(input_ids)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    auc = calculate_auc(y_true, y_pred)

    return {
        "loss": total_loss / len(loader),
        "auc": auc,
        "y_pred": y_pred,
        "y_true": y_true,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    tokenizer: SmilesTokenizer,
    cfg: type,
) -> float:
    """
    完整训练循环，包含 early stopping 和 checkpoint 管理。

    Returns:
        best_auc (float)
    """
    model_path = cfg.model_path
    best_auc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        with Timer() as t:
            train_loss = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler,
                device, cfg.grad_clip, cfg.use_amp,
            )
        scheduler.step()

        val_results = validate(model, val_loader, loss_fn, device)
        val_loss = val_results["loss"]
        val_auc = val_results["auc"]["mean"]

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={format_auc(val_results['auc'])} | "
            f"lr={lr_now:.2e} | "
            f"time={t.elapsed:.1f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "best_auc": best_auc,
                    "epoch": epoch,
                    "config": {
                        "d_model": cfg.d_model,
                        "nhead": cfg.nhead,
                        "num_layers": cfg.num_layers,
                        "dim_feedforward": cfg.dim_feedforward,
                        "dropout": cfg.dropout,
                        "max_len": cfg.max_len,
                    },
                },
                model_path,
            )
            logger.info(f"  -> Best model saved (auc={best_auc:.4f})")
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Training complete. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    return best_auc


# ============================================================
# Section 9: Inference
# ============================================================

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    对测试集进行推理。

    Returns:
        probs: (N_molecules, 3) sigmoid 概率，已 clamp 到 [0, 1]
    """
    model.eval()
    all_probs = []

    for input_ids, _ in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        logits = model(input_ids)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    probs = np.clip(probs, 0.0, 1.0)
    return probs


def load_best_model(
    model_path: str,
    vocab_path: str,
    cfg: type,
    device: torch.device,
) -> Tuple[SmilesTransformer, SmilesTokenizer]:
    """
    加载最佳模型和 tokenizer。

    Returns:
        (model, tokenizer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    tokenizer = SmilesTokenizer.load(vocab_path)
    logger.info(f"Loaded vocab: {tokenizer.vocab_size} tokens")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from {model_path}")
    logger.info(f"  Best AUC: {checkpoint.get('best_auc', 'N/A')}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    model = SmilesTransformer(
        vocab_size=checkpoint["vocab_size"],
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_length=cfg.max_len,
        pad_idx=0,
        num_targets=3,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def run_inference(
    model: SmilesTransformer,
    tokenizer: SmilesTokenizer,
    test_path: str,
    cfg: type,
    device: torch.device,
) -> np.ndarray:
    """
    完整推理流程：加载测试数据 → 推理 → 返回概率数组。
    """
    test_df = load_test_data(test_path)
    smiles_test = test_df["molecule_smiles"].tolist()

    test_loader = create_test_dataloader(
        smiles_test, tokenizer,
        batch_size=cfg.batch_size * 2,
        max_length=cfg.max_len,
        num_workers=cfg.num_workers,
    )

    logger.info("Running inference ...")
    probs = predict(model, test_loader, device)
    logger.info(f"Predictions shape: {probs.shape}")

    # 统计
    for i, name in enumerate(PROTEIN_NAMES):
        p = probs[:, i]
        logger.info(
            f"  {name}: mean={p.mean():.4f}, min={p.min():.4f}, "
            f"max={p.max():.4f}, pos_rate={np.mean(p > 0.5):.4f}"
        )

    return probs


# ============================================================
# Section 10: Submission
# ============================================================

def generate_submission(probs: np.ndarray, test_path: str) -> pd.DataFrame:
    """
    生成长格式 submission DataFrame，匹配 Kaggle 提交格式。

    Kaggle BELKA 提交格式:
        id, binds

    每个 (molecule, protein) pair 一行，按原始测试数据行顺序。

    Args:
        probs: (N_molecules, 3) 预测概率，列顺序 [BRD4, HSA, sEH]
        test_path: 测试数据路径

    Returns:
        DataFrame with columns: id, binds
    """
    test_df = (
        pd.read_parquet(test_path)
        if test_path.endswith(".parquet")
        else pd.read_csv(test_path)
    )

    # molecule_smiles → probs 行索引
    smiles_unique = list(dict.fromkeys(test_df["molecule_smiles"]))
    smile_to_idx = {s: i for i, s in enumerate(smiles_unique)}

    protein_order = {"BRD4": 0, "HSA": 1, "sEH": 2}

    binds_list = []
    for _, row in test_df.iterrows():
        smiles = row["molecule_smiles"]
        protein = row["protein_name"]
        idx = smile_to_idx[smiles]
        prob_idx = protein_order[protein]
        binds_list.append(probs[idx, prob_idx])

    df = pd.DataFrame({"id": test_df["id"], "binds": binds_list})
    return df


# ============================================================
# Section 11: Main Pipeline — 一键运行
# ============================================================

def auto_detect_data():
    """
    自动发现数据文件路径。

    优先级:
    1. 环境变量 TRAIN_PATH / TEST_PATH
    2. Kaggle 标准路径 /kaggle/input/leash-belka/
    3. 本地 data/ 目录
    """
    train_path = os.environ.get("TRAIN_PATH", "")
    test_path = os.environ.get("TEST_PATH", "")

    if not train_path:
        if IS_KAGGLE:
            # Kaggle 环境：优先 parquet
            kaggle_train = "/kaggle/input/leash-belka/train.parquet"
            if os.path.exists(kaggle_train):
                train_path = kaggle_train
            else:
                # 搜索 Kaggle input 目录
                for root, dirs, files in os.walk(INPUT_DIR):
                    for f in sorted(files):
                        if "train" in f.lower() and (f.endswith(".parquet") or f.endswith(".csv")):
                            train_path = os.path.join(root, f)
                            break
                    if train_path:
                        break
        else:
            # 本地环境：优先小数据集 CSV
            for fname in ["train_300.csv", "train_sample.csv", "train.csv", "train.parquet"]:
                candidate = os.path.join(INPUT_DIR, fname)
                if os.path.exists(candidate):
                    train_path = candidate
                    break
            if not train_path:
                for root, dirs, files in os.walk(INPUT_DIR):
                    for f in sorted(files):
                        if "train" in f.lower() and (f.endswith(".csv") or f.endswith(".parquet")):
                            train_path = os.path.join(root, f)
                            break
                    if train_path:
                        break

    if not test_path:
        if IS_KAGGLE:
            kaggle_test = "/kaggle/input/leash-belka/test.parquet"
            if os.path.exists(kaggle_test):
                test_path = kaggle_test
            else:
                for root, dirs, files in os.walk(INPUT_DIR):
                    for f in sorted(files):
                        if "test" in f.lower() and (f.endswith(".parquet") or f.endswith(".csv")):
                            test_path = os.path.join(root, f)
                            break
                    if test_path:
                        break
        else:
            for fname in ["test_300.csv", "test.csv", "test.parquet"]:
                candidate = os.path.join(INPUT_DIR, fname)
                if os.path.exists(candidate):
                    test_path = candidate
                    break
            if not test_path:
                for root, dirs, files in os.walk(INPUT_DIR):
                    for f in sorted(files):
                        if "test" in f.lower() and (f.endswith(".csv") or f.endswith(".parquet")):
                            test_path = os.path.join(root, f)
                            break
                    if test_path:
                        break

    if not train_path:
        raise FileNotFoundError(
            f"Training data not found. Searched in {INPUT_DIR}/. "
            f"Set TRAIN_PATH environment variable."
        )
    if not test_path:
        raise FileNotFoundError(
            f"Test data not found. Searched in {INPUT_DIR}/. "
            f"Set TEST_PATH environment variable."
        )

    return train_path, test_path


def main():
    """主流程：训练 → 推理 → 生成 submission"""
    # ── 1. 路径配置 ──
    CFG.train_path, CFG.test_path = auto_detect_data()
    CFG.vocab_path     = os.path.join(OUTPUT_DIR, "vocab.json")
    CFG.model_path     = os.path.join(OUTPUT_DIR, "best_model.pt")
    CFG.submission_path = os.path.join(OUTPUT_DIR, "submission.csv")

    # 本地小数据集时自动调小参数
    is_small_data = CFG.train_path.endswith(".csv") and not IS_KAGGLE
    if is_small_data:
        logger.info("Local CSV detected — using small-data config")
        CFG.epochs = 5
        CFG.batch_size = 16
        CFG.max_negatives = None

    logger.info(f"{'='*60}")
    logger.info("BELKA Vanilla Transformer — Pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"Train:  {CFG.train_path}")
    logger.info(f"Test:   {CFG.test_path}")
    logger.info(f"Output: {OUTPUT_DIR}/")
    logger.info(f"Model:  d_model={CFG.d_model}, layers={CFG.num_layers}, heads={CFG.nhead}")
    logger.info(f"Train:  epochs={CFG.epochs}, batch={CFG.batch_size}, lr={CFG.lr}")
    logger.info(f"{'='*60}")

    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)

    use_amp = CFG.use_amp and DEVICE.type == "cuda"
    logger.info(f"AMP: {'enabled' if use_amp else 'disabled'}")

    # ── 2. 加载训练数据 ──
    df = load_train_data(CFG.train_path, max_negatives=CFG.max_negatives)
    smiles_list = df["molecule_smiles"].tolist()
    labels = df[PROTEIN_NAMES].values.astype(np.float32)

    # Train / Val split
    smiles_train, smiles_val, y_train, y_val = train_test_split(
        smiles_list, labels,
        test_size=1 - CFG.train_ratio,
        random_state=CFG.seed,
        stratify=(labels.sum(axis=1) > 0),
    )
    logger.info(f"Train: {len(smiles_train):,} | Val: {len(smiles_val):,}")

    # ── 3. Tokenizer ──
    if os.path.exists(CFG.vocab_path):
        tokenizer = SmilesTokenizer.load(CFG.vocab_path)
        logger.info(f"Loaded vocab: {tokenizer.vocab_size} tokens")
    else:
        tokenizer = SmilesTokenizer(max_length=CFG.max_len)
        vocab_size = tokenizer.build_vocab(smiles_train)
        tokenizer.save(CFG.vocab_path)
        logger.info(f"Built vocab: {vocab_size} tokens -> {CFG.vocab_path}")

    # ── 4. DataLoader ──
    train_loader, val_loader = create_dataloaders(
        smiles_train, y_train, smiles_val, y_val,
        tokenizer,
        batch_size=CFG.batch_size,
        max_length=CFG.max_len,
        num_workers=CFG.num_workers,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── 5. 模型 ──
    model = SmilesTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CFG.d_model,
        nhead=CFG.nhead,
        num_layers=CFG.num_layers,
        dim_feedforward=CFG.dim_feedforward,
        dropout=CFG.dropout,
        max_length=CFG.max_len,
        pad_idx=0,
        num_targets=3,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ── 6. 损失 & 优化器 ──
    pos_weight = compute_pos_weight(y_train).to(DEVICE)
    logger.info(f"pos_weight: {pos_weight.tolist()}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.lr * 0.01,
    )
    scaler = GradScaler(enabled=use_amp)

    # ── 7. 训练 ──
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info("Starting Training")
    logger.info(f"{'='*60}")
    best_auc = fit(
        model, train_loader, val_loader, loss_fn,
        optimizer, scheduler, scaler, DEVICE, tokenizer, CFG,
    )

    # ── 8. 推理 ──
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info("Starting Inference")
    logger.info(f"{'='*60}")

    # 加载最佳模型
    model, tokenizer = load_best_model(CFG.model_path, CFG.vocab_path, CFG, DEVICE)
    probs = run_inference(model, tokenizer, CFG.test_path, CFG, DEVICE)

    # ── 9. 生成 Submission ──
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info("Generating Submission")
    logger.info(f"{'='*60}")

    sub_df = generate_submission(probs, CFG.test_path)
    sub_df.to_csv(CFG.submission_path, index=False)

    logger.info(f"Submission saved to {CFG.submission_path}")
    logger.info(f"  Shape: {sub_df.shape}")
    logger.info(f"  Columns: {sub_df.columns.tolist()}")
    logger.info(f"  Preview:\n{sub_df.head(12).to_string(index=False)}")

    # ── 10. 验证 submission 格式 ──
    logger.info("")
    logger.info("Validating submission format ...")
    if list(sub_df.columns) != ["id", "binds"]:
        logger.error(f"WRONG columns! Expected ['id', 'binds'], got {sub_df.columns.tolist()}")
    else:
        logger.info("  Columns OK: ['id', 'binds']")
    logger.info(f"  binds range: [{sub_df['binds'].min():.6f}, {sub_df['binds'].max():.6f}]")
    logger.info(f"  binds mean: {sub_df['binds'].mean():.6f}")

    logger.info("")
    logger.info(f"{'='*60}")
    logger.info("ALL DONE!")
    logger.info(f"  Submission: {CFG.submission_path}")
    logger.info(f"  Best AUC:   {best_auc:.4f}")
    logger.info(f"  Download and submit to Kaggle!")
    logger.info(f"{'='*60}")

    return sub_df


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    submission = main()
