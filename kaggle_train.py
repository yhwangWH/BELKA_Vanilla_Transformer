"""
Kaggle Notebook 专用 — 全量训练 + 推理一体化脚本。

两种使用方式：

方式 1 — Pipeline API（推荐，Kaggle Notebook 用）:
    from kaggle_train import BELKAPipeline
    pipeline = BELKAPipeline(quick_mode=False)
    pipeline.run()

方式 2 — 直接运行（向后兼容）:
    import os
    os.environ["TRAIN_PATH"]  = "/kaggle/input/leash-belka/train.parquet"
    os.environ["TEST_PATH"]   = "/kaggle/input/leash-belka/test.parquet"
    os.environ["OUTPUT_DIR"]  = "/kaggle/working"
    %run kaggle_train.py

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
from typing import Optional, Tuple

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
# BELKAPipeline — 一体化训练 + 推理封装
# ============================================================

class BELKAPipeline:
    """
    BELKA 比赛全流程 Pipeline：训练 → 推理 → 生成 submission。

    用法:
        # 全量训练
        pipeline = BELKAPipeline(quick_mode=False)
        submission = pipeline.run()

        # 快速验证
        pipeline = BELKAPipeline(quick_mode=True)
        submission = pipeline.run()

        # 自定义参数
        pipeline = BELKAPipeline(
            quick_mode=False,
            epochs=20,
            batch_size=256,
            d_model=512,
            num_layers=6,
        )
        submission = pipeline.run()
    """

    def __init__(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        quick_mode: bool = False,
        # 模型参数
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_length: int = 256,
        # 训练参数
        epochs: int = 15,
        batch_size: int = 128,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        grad_clip: float = 1.0,
        num_workers: int = 2,
        # 数据参数
        max_negatives: Optional[int] = None,
        use_amp: bool = True,
    ):
        # ── 路径配置 ──
        self.train_path = train_path or os.environ.get(
            "TRAIN_PATH", "/kaggle/input/leash-belka/train.parquet"
        )
        self.test_path = test_path or os.environ.get(
            "TEST_PATH", "/kaggle/input/leash-belka/test.parquet"
        )
        self.output_dir = output_dir or os.environ.get(
            "OUTPUT_DIR", "/kaggle/working"
        )
        self.ckpt_dir = os.path.join(self.output_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ── 模式参数 ──
        self.quick_mode = quick_mode

        if quick_mode:
            # 快速验证预设：小模型 + 少轮数 + 限制负样本
            self.epochs = 3
            self.batch_size = 64
            self.lr = 1e-4
            self.d_model = 128
            self.num_layers = 2
            self.nhead = 4
            self.dim_feedforward = 256
            self.dropout = 0.1
            self.max_length = 128
            self.weight_decay = 1e-2
            self.grad_clip = 1.0
            self.num_workers = 2
            self.max_negatives = 100_000
            self.use_amp = True
        else:
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.d_model = d_model
            self.num_layers = num_layers
            self.nhead = nhead
            self.dim_feedforward = dim_feedforward
            self.dropout = dropout
            self.max_length = max_length
            self.weight_decay = weight_decay
            self.grad_clip = grad_clip
            self.num_workers = num_workers
            self.max_negatives = max_negatives
            self.use_amp = use_amp

        # ── 运行时状态 ──
        self.model: Optional[BELKATransformer] = None
        self.tokenizer: Optional[SMILESTokenizer] = None
        self.device: Optional[torch.device] = None
        self.model_path: Optional[str] = None
        self.best_auc: float = 0.0

    # ────────────────────────────────────────────────────────────
    # 公共 API
    # ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        运行完整流程：训练 → 推理 → 生成 submission。

        Returns:
            submission DataFrame (id, binds)
        """
        self._print_header()
        self.train()
        log.info("")
        log.info("=" * 60)
        log.info("Starting inference ...")
        log.info("=" * 60)
        sub_df = self.inference()
        log.info("")
        log.info("=" * 60)
        log.info("ALL DONE! Download submission.csv from the output directory.")
        log.info("=" * 60)
        return sub_df

    def train(self):
        """
        执行全量训练，训练完成后 model、tokenizer、device、model_path 会被设置。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            log.info(
                f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
            )

        use_amp = self.use_amp and self.device.type == "cuda"
        log.info(f"AMP: {'enabled' if use_amp else 'disabled'}")
        log.info(
            f"Max negatives: {self.max_negatives if self.max_negatives else 'ALL'}"
        )

        # ── 1. 加载训练数据 ──
        log.info(f"Loading training data from {self.train_path} ...")
        df = load_and_pivot_train(
            self.train_path, max_negatives=self.max_negatives
        )
        smiles_list = df["molecule_smiles"].tolist()
        labels = df[PROTEIN_NAMES].values.astype(np.float32)

        # Train / Val split
        smiles_train, smiles_val, y_train, y_val = train_test_split(
            smiles_list,
            labels,
            test_size=0.1,
            random_state=42,
            stratify=(labels.sum(axis=1) > 0),
        )
        log.info(
            f"Train molecules: {len(smiles_train):,}  |  "
            f"Val molecules: {len(smiles_val):,}"
        )

        # ── 2. Tokenizer ──
        vocab_path = os.path.join(self.ckpt_dir, "vocab.json")
        if os.path.exists(vocab_path):
            self.tokenizer = SMILESTokenizer.load(vocab_path)
            log.info(f"Loaded vocab: {self.tokenizer.vocab_size} tokens")
        else:
            self.tokenizer = SMILESTokenizer(max_length=self.max_length)
            vocab_size = self.tokenizer.build_vocab(smiles_train)
            self.tokenizer.save(vocab_path)
            log.info(f"Built vocab: {vocab_size} tokens")

        # ── 3. DataLoader ──
        train_loader, val_loader = create_dataloaders(
            smiles_train,
            y_train,
            smiles_val,
            y_val,
            self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            num_workers=self.num_workers,
        )
        log.info(
            f"Train batches: {len(train_loader)}  |  "
            f"Val batches: {len(val_loader)}"
        )

        # ── 4. 模型 ──
        self.model = BELKATransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_length=self.max_length,
            pad_idx=0,
            num_targets=3,
        ).to(self.device)
        log.info(
            f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        # ── 5. 损失 & 优化器 ──
        pos_weight = compute_pos_weight(y_train).to(self.device)
        log.info(f"pos_weight: {pos_weight.tolist()}")

        loss_fn = create_loss_fn(pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        scaler = GradScaler(enabled=use_amp)

        # ── 6. 训练循环 ──
        self.model_path = os.path.join(self.ckpt_dir, "best_model.pt")
        best_auc = 0.0
        best_epoch = 0
        patience = 5
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            with Timer() as t:
                train_loss = self._train_epoch(
                    self.model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    scaler,
                    self.device,
                    self.grad_clip,
                    use_amp,
                )
            scheduler.step()

            val_results = self._validate(
                self.model, val_loader, loss_fn, self.device
            )
            val_loss = val_results["loss"]
            val_auc = val_results["auc"]["mean"]

            lr_now = optimizer.param_groups[0]["lr"]
            log.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
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
                        "model_state_dict": self.model.state_dict(),
                        "vocab_size": self.tokenizer.vocab_size,
                        "best_auc": best_auc,
                        "epoch": epoch,
                        "config": {
                            "d_model": self.d_model,
                            "nhead": self.nhead,
                            "num_layers": self.num_layers,
                            "dim_feedforward": self.dim_feedforward,
                            "dropout": self.dropout,
                            "max_length": self.max_length,
                        },
                    },
                    self.model_path,
                )
                log.info(f"  -> Best model saved (auc={best_auc:.4f})")
            else:
                no_improve += 1

            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

        self.best_auc = best_auc
        log.info(
            f"Training done. Best AUC: {best_auc:.4f} at epoch {best_epoch}"
        )

    def inference(
        self,
        model: Optional[BELKATransformer] = None,
        tokenizer: Optional[SMILESTokenizer] = None,
        device: Optional[torch.device] = None,
        model_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        推理并生成 submission.csv。

        如果不传参数，则使用 train() 后已设置好的模型。
        也可以独立传入模型参数以跳过训练。

        Returns:
            submission DataFrame (id, binds)
        """
        device = device or self.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 如果模型未传入，从 checkpoint 加载
        if model is None or tokenizer is None or model_path is None:
            model_path = model_path or os.path.join(
                self.ckpt_dir, "best_model.pt"
            )
            vocab_path = os.path.join(self.ckpt_dir, "vocab.json")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found: {model_path}. Run training first."
                )
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(
                    f"Vocab not found: {vocab_path}. Run training first."
                )

            tokenizer = SMILESTokenizer.load(vocab_path)
            log.info(f"Loaded vocab: {tokenizer.vocab_size} tokens")

            checkpoint = torch.load(
                model_path, map_location=device, weights_only=False
            )
            model = BELKATransformer(
                vocab_size=checkpoint["vocab_size"],
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                max_length=self.max_length,
                pad_idx=0,
                num_targets=3,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            log.info(f"Loaded model from {model_path}")
            log.info(f"  Best AUC: {checkpoint.get('best_auc', 'N/A')}")
            log.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

        # ── 加载测试数据 ──
        log.info(f"Loading test data from {self.test_path} ...")
        test_df = self._load_test_simple(self.test_path)
        smiles_test = test_df["molecule_smiles"].tolist()
        log.info(f"Unique test molecules: {len(smiles_test):,}")

        # ── DataLoader ──
        test_loader = create_test_dataloder(
            smiles_test,
            tokenizer,
            batch_size=self.batch_size * 2,
            max_length=self.max_length,
            num_workers=self.num_workers,
        )

        # ── 推理 ──
        log.info("Running inference ...")
        probs = self._predict(model, test_loader, device)
        log.info(f"Predictions shape: {probs.shape}")

        # 统计
        for i, name in enumerate(PROTEIN_NAMES):
            p = probs[:, i]
            log.info(
                f"  {name}: mean={p.mean():.4f}, min={p.min():.4f}, "
                f"max={p.max():.4f}, pos_rate={np.mean(p > 0.5):.4f}"
            )

        # ── 生成 submission ──
        sub_df = self._generate_submission(probs, self.test_path)
        submission_path = os.path.join(self.output_dir, "submission.csv")
        sub_df.to_csv(submission_path, index=False)
        log.info(f"Submission saved to {submission_path}")
        log.info(f"  Shape: {sub_df.shape}")
        log.info(f"  Columns: {sub_df.columns.tolist()}")
        log.info(f"  Preview:\n{sub_df.head(12).to_string(index=False)}")

        return sub_df

    # ────────────────────────────────────────────────────────────
    # 私有静态方法（训练/验证/推理逻辑）
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _train_epoch(
        model, loader, loss_fn, optimizer, scaler, device, grad_clip, use_amp
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

    @staticmethod
    @torch.no_grad()
    def _validate(model, loader, loss_fn, device) -> dict:
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

        return {
            "loss": total_loss / len(loader),
            "auc": auc,
            "y_pred": y_pred,
            "y_true": y_true,
        }

    @staticmethod
    @torch.no_grad()
    def _predict(model, loader, device) -> np.ndarray:
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

    @staticmethod
    def _generate_submission(probs: np.ndarray, test_path: str) -> pd.DataFrame:
        """
        生成长格式 submission DataFrame，匹配 Kaggle 提交格式。

        Args:
            probs: (N_molecules, 3) 预测概率，列顺序 [BRD4, HSA, sEH]
            test_path: 测试数据路径
        """
        test_df = (
            pd.read_parquet(test_path)
            if test_path.endswith(".parquet")
            else pd.read_csv(test_path)
        )

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

        return pd.DataFrame({"id": test_df["id"], "binds": binds_list})

    @staticmethod
    def _load_test_simple(path: str) -> pd.DataFrame:
        """加载测试数据，提取去重后的 molecule_smiles"""
        log.info(f"Loading test data from {path} ...")
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        log.info(f"Test data: {len(df):,} rows")
        grouped = (
            df.groupby("molecule_smiles")
            .agg(molecule_id=("id", "first"))
            .reset_index()
        )
        log.info(f"Unique test molecules: {len(grouped):,}")
        return grouped

    # ────────────────────────────────────────────────────────────
    # 工具方法
    # ────────────────────────────────────────────────────────────

    def _print_header(self):
        """打印运行配置信息"""
        log.info("=" * 60)
        log.info("BELKA Vanilla Transformer — Kaggle Full Training")
        log.info("=" * 60)
        log.info(f"Mode:         {'QUICK (fast validation)' if self.quick_mode else 'FULL'}")
        log.info(f"Train path:   {self.train_path}")
        log.info(f"Test path:    {self.test_path}")
        log.info(f"Output dir:   {self.output_dir}")
        log.info(f"")
        log.info(
            f"Model config: d_model={self.d_model}, layers={self.num_layers}, "
            f"heads={self.nhead}, ff={self.dim_feedforward}"
        )
        log.info(
            f"Train config: epochs={self.epochs}, batch_size={self.batch_size}, "
            f"lr={self.lr}"
        )
        log.info(f"Max length:   {self.max_length}")
        log.info(
            f"Max negatives: {self.max_negatives if self.max_negatives else 'ALL (full dataset)'}"
        )
        log.info("=" * 60)


# ============================================================
# 向后兼容入口（%run kaggle_train.py 方式）
# ============================================================

def main():
    """向后兼容的入口函数：使用全局配置运行完整流程"""
    pipeline = BELKAPipeline(quick_mode=False)
    return pipeline.run()


if __name__ == "__main__":
    _submission = main()
