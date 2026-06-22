"""
通用工具函数：日志、指标、配置管理。
"""

import logging
import sys
import time
from typing import Optional
import numpy as np
from sklearn.metrics import roc_auc_score


# ── 日志配置 ────────────────────────────────────────────────────────
def setup_logging(name: Optional[str] = None, level: int = logging.INFO):
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


log = setup_logging("BELKA")


# ── 计时器 ──────────────────────────────────────────────────────────
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


# ── AUC 指标 ─────────────────────────────────────────────────────────
PROTEIN_NAMES = ["BRD4", "HSA", "sEH"]


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
        # 过滤掉全为同一类的列（AUC 无法计算）
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
    parts = []
    for name in PROTEIN_NAMES:
        val = auc_dict.get(name, float("nan"))
        parts.append(f"{name}={val:.4f}")
    parts.append(f"mean={auc_dict.get('mean', float('nan')):.4f}")
    return " | ".join(parts)


# ── 配置类 ──────────────────────────────────────────────────────────
class Config:
    """训练/模型配置"""

    def __init__(self, **kwargs):
        # ── 模型参数 ──
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 4
        self.dim_feedforward = 512
        self.dropout = 0.1
        self.max_length = 256

        # ── 训练参数 ──
        self.batch_size = 128
        self.epochs = 20
        self.lr = 1e-4
        self.weight_decay = 1e-2
        self.grad_clip = 1.0

        # ── 数据参数 ──
        self.train_ratio = 0.9
        self.num_workers = 4

        # ── 混合精度 ──
        self.use_amp = True

        # ── 路径 ──
        self.train_path = "data/train.parquet"
        self.test_path = "data/test.parquet"
        self.sample_sub_path = "data/sample_submission.csv"
        self.vocab_path = "ckpt/vocab.json"
        self.model_path = "ckpt/best_model.pt"
        self.submission_path = "ckpt/submission.csv"

        # 覆盖默认值
        for k, v in kwargs.items():
            setattr(self, k, v)


if __name__ == "__main__":
    # 测试 AUC 计算
    y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    y_pred = np.array([[0.1, 0.2, 0.9], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    auc = compute_auc(y_true, y_pred)
    print(format_auc(auc))
