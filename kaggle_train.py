"""
Kaggle Notebook 专用 — 全量训练 + 推理一体化脚本。

用法：直接上传所有项目文件到 Kaggle，在 Notebook 中运行：
    %run kaggle_train.py

或在 Notebook cell 中直接执行此文件的内容。

数据路径说明：
    先在 Kaggle Notebook 的 "Add Input" 中添加：
        leashbio2024/leash-belka  (LEASH - BELKA competition data)

依赖安装（Notebook 第一个 cell）:
    !pip install torch pandas pyarrow scikit-learn -q
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

# ============================================================
# 项目文件导入
# ============================================================
from data import load_and_pivot_train, compute_pos_weight
from dataset import create_dataloaders, create_test_dataloder
from tokenizer import SMILESTokenizer
from model import BELKATransformer, create_loss_fn
from utils import (
    Config,
    log,
    Timer,
    PROTEIN_NAMES,
    compute_auc,
    format_auc,
)

# ============================================================
# Kaggle 路径配置（直接修改这里的参数）
# ============================================================

# 数据路径（Kaggle 默认）
TRAIN_PATH = "/kaggle/input/leash-belka/train.parquet"
TEST_PATH = "/kaggle/input/leash-belka/test.parquet"

# 输出路径（Kaggle 工作目录）
OUTPUT_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(OUTPUT_DIR, "ckpt")

# 创建目录
os.makedirs(CKPT_DIR, exist_ok=True)

# ============================================================
# 可调参数（可直接修改这些变量）
# ============================================================

EPOCHS = 15                # 训练轮数
BATCH_SIZE = 128           # 批大小（T4 16GB 建议 128-256）
LR = 1e-4                  # 学习率
D_MODEL = 256              # 模型维度
NUM_LAYERS = 4             # Transformer 层数
NHEAD = 8                  # 注意力头数
DIM_FEEDFORWARD = 512      # FFN 维度
DROPOUT = 0.1              # Dropout 比例
MAX_LENGTH = 256           # SMILES 最大长度
WEIGHT_DECAY = 1e-2        # 权重衰减
GRAD_CLIP = 1.0            # 梯度裁剪
NUM_WORKERS = 2            # DataLoader 进程数（Kaggle 建议 2）

# 负样本控制：None = 全量，或设置数字限制（如 500000）
# 注意：全量约 5900 万负样本，可能 OOM
MAX_NEGATIVES = None       # 改为数字限制负样本数
# MAX_NEGATIVES = 500000   # 取消注释此行以限制负样本

USE_AMP = True             # 混合精度训练（T4 支持 FP16，提速+省显存）

# ============================================================
# 训练函数（复用 train.py 的逻辑）
# ============================================================

def train_epoch_fn(model, loader, loss_fn, optimizer, scaler, device, grad_clip, use_amp):
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
def validate_fn(model, loader, loss_fn, device):
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
    auc = compute_auc(y_true, y_pred)

    return {"loss": total_loss / len(loader), "auc": auc, "y_pred": y_pred, "y_true": y_true}


def train():
    """全量训练主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    use_amp = USE_AMP and device.type == "cuda"
    log.info(f"AMP: {'enabled' if use_amp else 'disabled'}")
    log.info(f"Max negatives: {MAX_NEGATIVES if MAX_NEGATIVES else 'ALL'}")

    # ── 1. 加载训练数据 ──
    log.info(f"Loading training data from {TRAIN_PATH} ...")
    df = load_and_pivot_train(TRAIN_PATH, max_negatives=MAX_NEGATIVES)
    smiles_list = df["molecule_smiles"].tolist()
    labels = df[PROTEIN_NAMES].values.astype(np.float32)

    # Train / Val split
    smiles_train, smiles_val, y_train, y_val = train_test_split(
        smiles_list, labels,
        test_size=0.1,
        random_state=42,
        stratify=(labels.sum(axis=1) > 0),
    )
    log.info(f"Train molecules: {len(smiles_train):,}  |  Val molecules: {len(smiles_val):,}")

    # ── 2. Tokenizer ──
    vocab_path = os.path.join(CKPT_DIR, "vocab.json")
    if os.path.exists(vocab_path):
        tokenizer = SMILESTokenizer.load(vocab_path)
        log.info(f"Loaded vocab: {tokenizer.vocab_size} tokens")
    else:
        tokenizer = SMILESTokenizer(max_length=MAX_LENGTH)
        vocab_size = tokenizer.build_vocab(smiles_train)
        tokenizer.save(vocab_path)
        log.info(f"Built vocab: {vocab_size} tokens")

    # ── 3. DataLoader ──
    train_loader, val_loader = create_dataloaders(
        smiles_train, y_train,
        smiles_val, y_val,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        num_workers=NUM_WORKERS,
    )
    log.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── 4. 模型 ──
    model = BELKATransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=0,
        num_targets=3,
    ).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 5. 损失 & 优化器 ──
    pos_weight = compute_pos_weight(y_train).to(device)
    log.info(f"pos_weight: {pos_weight.tolist()}")

    loss_fn = create_loss_fn(pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    scaler = GradScaler(enabled=use_amp)

    # ── 6. 训练循环 ──
    model_path = os.path.join(CKPT_DIR, "best_model.pt")
    best_auc = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        with Timer() as t:
            train_loss = train_epoch_fn(
                model, train_loader, loss_fn, optimizer, scaler,
                device, GRAD_CLIP, use_amp,
            )
        scheduler.step()

        val_results = validate_fn(model, val_loader, loss_fn, device)
        val_loss = val_results["loss"]
        val_auc = val_results["auc"]["mean"]

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch:3d}/{EPOCHS} | "
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
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "num_layers": NUM_LAYERS,
                        "dim_feedforward": DIM_FEEDFORWARD,
                        "dropout": DROPOUT,
                        "max_length": MAX_LENGTH,
                    },
                },
                model_path,
            )
            log.info(f"  -> Best model saved (auc={best_auc:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Training done. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    return model, tokenizer, device, model_path


# ============================================================
# 推理函数（复用 inference.py 的逻辑）
# ============================================================

@torch.no_grad()
def predict_fn(model, loader, device):
    """对测试集推理"""
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


def generate_submission_fn(probs, test_path):
    """
    生成长格式 submission DataFrame，匹配 Kaggle 提交格式。

    Args:
        probs: (N_molecules, 3) 预测概率，列顺序 [BRD4, HSA, sEH]
        test_path: 测试数据路径

    Returns:
        DataFrame with columns: id, binds
    """
    test_df = pd.read_parquet(test_path) if test_path.endswith(".parquet") else pd.read_csv(test_path)

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


def inference(model=None, tokenizer=None, device=None, model_path=None):
    """推理并生成 submission.csv"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果模型未传入，从 checkpoint 加载
    if model is None or tokenizer is None or model_path is None:
        model_path = os.path.join(CKPT_DIR, "best_model.pt")
        vocab_path = os.path.join(CKPT_DIR, "vocab.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab not found: {vocab_path}. Run training first.")

        tokenizer = SMILESTokenizer.load(vocab_path)
        log.info(f"Loaded vocab: {tokenizer.vocab_size} tokens")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = BELKATransformer(
            vocab_size=checkpoint["vocab_size"],
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            max_length=MAX_LENGTH,
            pad_idx=0,
            num_targets=3,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info(f"Loaded model from {model_path}")
        log.info(f"  Best AUC: {checkpoint.get('best_auc', 'N/A')}")
        log.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    # ── 加载测试数据 ──
    log.info(f"Loading test data from {TEST_PATH} ...")
    test_df = load_test_simple(TEST_PATH)

    smiles_test = test_df["molecule_smiles"].tolist()
    log.info(f"Unique test molecules: {len(smiles_test):,}")

    # ── DataLoader ──
    test_loader = create_test_dataloder(
        smiles_test, tokenizer,
        batch_size=BATCH_SIZE * 2,
        max_length=MAX_LENGTH,
        num_workers=NUM_WORKERS,
    )

    # ── 推理 ──
    log.info("Running inference ...")
    probs = predict_fn(model, test_loader, device)
    log.info(f"Predictions shape: {probs.shape}")

    # 统计
    for i, name in enumerate(PROTEIN_NAMES):
        p = probs[:, i]
        log.info(
            f"  {name}: mean={p.mean():.4f}, min={p.min():.4f}, "
            f"max={p.max():.4f}, pos_rate={np.mean(p > 0.5):.4f}"
        )

    # ── 生成 submission ──
    sub_df = generate_submission_fn(probs, TEST_PATH)
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub_df.to_csv(submission_path, index=False)
    log.info(f"Submission saved to {submission_path}")
    log.info(f"  Shape: {sub_df.shape}")
    log.info(f"  Columns: {sub_df.columns.tolist()}")
    log.info(f"  Preview:\n{sub_df.head(12).to_string(index=False)}")

    return sub_df


def load_test_simple(path):
    """加载测试数据，提取去重后的 molecule_smiles"""
    log.info(f"Loading test data from {path} ...")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    log.info(f"Test data: {len(df):,} rows")
    grouped = df.groupby("molecule_smiles").agg(
        molecule_id=("id", "first"),
    ).reset_index()
    log.info(f"Unique test molecules: {len(grouped):,}")
    return grouped


# ============================================================
# 主函数
# ============================================================

def main():
    """运行完整流程：训练 -> 推理"""
    log.info("=" * 60)
    log.info("BELKA Vanilla Transformer — Kaggle Full Training")
    log.info("=" * 60)
    log.info(f"Train path:  {TRAIN_PATH}")
    log.info(f"Test path:   {TEST_PATH}")
    log.info(f"Output dir:  {OUTPUT_DIR}")
    log.info(f"")
    log.info(f"Model config: d_model={D_MODEL}, layers={NUM_LAYERS}, heads={NHEAD}")
    log.info(f"Train config: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")
    log.info(f"Max length:   {MAX_LENGTH}")
    log.info(f"Max negatives: {MAX_NEGATIVES if MAX_NEGATIVES else 'ALL (full dataset)'}")
    log.info("=" * 60)

    # ── 1. 训练 ──
    model, tokenizer, device, model_path = train()

    # ── 2. 推理 ──
    log.info("")
    log.info("=" * 60)
    log.info("Starting inference ...")
    log.info("=" * 60)
    sub_df = inference(model=model, tokenizer=tokenizer, device=device, model_path=model_path)

    log.info("")
    log.info("=" * 60)
    log.info("ALL DONE! Download submission.csv from /kaggle/working/")
    log.info("=" * 60)

    return sub_df


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    _submission = main()
