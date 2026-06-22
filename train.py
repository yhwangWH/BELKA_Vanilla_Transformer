"""
训练脚本 — 完整的训练 + 验证循环。

用法：
    python train.py                    # 使用默认配置
    python train.py --epochs 30 --batch_size 64 --lr 5e-5
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

from data import load_and_pivot_train, compute_pos_weight
from dataset import create_dataloaders
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


# ── 单 epoch 训练 ───────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (input_ids, labels) in enumerate(loader):
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


# ── 验证 ────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

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
    avg_loss = total_loss / len(loader)
    return {"loss": avg_loss, "auc": auc, "y_pred": y_pred, "y_true": y_true}


# ── 主训练函数 ──────────────────────────────────────────────────────
def train(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(f"AMP: {'enabled' if config.use_amp and device.type == 'cuda' else 'disabled'}")

    use_amp = config.use_amp and device.type == "cuda"

    # ── 1. 加载数据 ──
    # 对 parquet 文件使用流式读取，控制负样本数以节省内存
    max_neg = getattr(config, "max_negatives", None)
    sample_frac = getattr(config, "sample_frac", None)
    df = load_and_pivot_train(
        config.train_path,
        sample_frac=sample_frac,
        max_negatives=max_neg,
    )

    smiles_list = df["molecule_smiles"].tolist()
    labels = df[PROTEIN_NAMES].values.astype(np.float32)

    # Train / Val split
    smiles_train, smiles_val, y_train, y_val = train_test_split(
        smiles_list, labels,
        test_size=1 - config.train_ratio,
        random_state=42,
        stratify=(labels.sum(axis=1) > 0),  # 分层：有正样本 vs 无正样本
    )
    log.info(f"Train molecules: {len(smiles_train):,}, Val molecules: {len(smiles_val):,}")

    # ── 2. 构建 / 加载 tokenizer ──
    if os.path.exists(config.vocab_path):
        tokenizer = SMILESTokenizer.load(config.vocab_path)
        log.info(f"Loaded vocabulary from {config.vocab_path} (size={tokenizer.vocab_size})")
    else:
        tokenizer = SMILESTokenizer(max_length=config.max_length)
        vocab_size = tokenizer.build_vocab(smiles_train)
        os.makedirs(os.path.dirname(config.vocab_path), exist_ok=True)
        tokenizer.save(config.vocab_path)
        log.info(f"Built vocabulary: {vocab_size} tokens, saved to {config.vocab_path}")

    # ── 3. 创建 DataLoader ──
    train_loader, val_loader = create_dataloaders(
        smiles_train, y_train,
        smiles_val, y_val,
        tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_workers=config.num_workers,
    )
    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── 4. 创建模型 ──
    model = BELKATransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_length=config.max_length,
        pad_idx=0,
        num_targets=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # ── 5. 损失 & 优化器 ──
    pos_weight = compute_pos_weight(y_train).to(device)
    log.info(f"pos_weight: {pos_weight.tolist()}")

    loss_fn = create_loss_fn(pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
    )
    scaler = GradScaler(enabled=use_amp)

    # ── 6. 训练循环 ──
    best_auc = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        with Timer() as t:
            train_loss = train_epoch(
                model, train_loader, loss_fn, optimizer, scaler,
                device, config.grad_clip, use_amp,
            )
        scheduler.step()

        val_results = validate(model, val_loader, loss_fn, device)
        val_loss = val_results["loss"]
        val_auc = val_results["auc"]["mean"]

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={format_auc(val_results['auc'])} | "
            f"lr={lr_now:.2e} | "
            f"time={t.elapsed:.1f}s"
        )

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
                    "vocab_size": tokenizer.vocab_size,
                    "best_auc": best_auc,
                    "epoch": epoch,
                },
                config.model_path,
            )
            log.info(f"  → Best model saved (auc={best_auc:.4f})")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    log.info(f"Training complete. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_path", type=str, default="data/train.parquet")
    parser.add_argument("--max_negatives", type=int, default=200000)
    parser.add_argument("--sample_frac", type=float, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        train_path=args.train_path,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        max_negatives=args.max_negatives,
        sample_frac=args.sample_frac,
    )
    train(config)
