"""
推理脚本 — 加载模型并生成 Kaggle submission。

用法：
    python inference.py [--model_path ckpt/best_model.pt]

输出格式（长格式，匹配 sample_submission.csv）:
    id, binds
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from data import load_test
from dataset import create_test_dataloder
from tokenizer import SMILESTokenizer
from model import BELKATransformer
from utils import Config, log, PROTEIN_NAMES


@torch.no_grad()
def predict(
    model: BELKATransformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    对测试集进行推理。

    Returns:
        probs: (N, 3) sigmoid 概率，已 clamp 到 [0, 1]
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


def generate_submission(probs: np.ndarray, test_path: str) -> pd.DataFrame:
    """
    生成长格式 submission DataFrame，匹配 sample_submission.csv 格式。

    测试数据中每个 molecule 有 3 行（BRD4, HSA, sEH），
    输出保持与原始测试数据相同的行顺序。

    Args:
        probs: (N_molecules, 3) 预测概率，列顺序为 [BRD4, HSA, sEH]
        test_path: 测试数据路径（csv 或 parquet）

    Returns:
        DataFrame with columns: id, binds
    """
    test_df = pd.read_parquet(test_path) if test_path.endswith(".parquet") else pd.read_csv(test_path)

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

    df = pd.DataFrame({
        "id": test_df["id"],
        "binds": binds_list,
    })
    return df


def generate_wide_submission(probs: np.ndarray, molecule_ids: np.ndarray) -> pd.DataFrame:
    """
    生成宽格式 submission DataFrame（辅助用）。

    Args:
        probs: (N, 3) 预测概率
        molecule_ids: (N,) 分子 ID

    Returns:
        DataFrame with columns: id, BRD4, HSA, sEH
    """
    df = pd.DataFrame(probs, columns=PROTEIN_NAMES)
    df.insert(0, "id", molecule_ids)
    return df


def inference(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── 1. 加载 tokenizer ──
    if not os.path.exists(config.vocab_path):
        raise FileNotFoundError(f"Vocabulary not found: {config.vocab_path}. Run train.py first.")
    tokenizer = SMILESTokenizer.load(config.vocab_path)
    log.info(f"Loaded vocabulary: {tokenizer.vocab_size} tokens")

    # ── 2. 加载模型 ──
    checkpoint = torch.load(config.model_path, map_location=device, weights_only=False)
    log.info(f"Loaded checkpoint from {config.model_path}")
    log.info(f"  Best AUC: {checkpoint.get('best_auc', 'N/A')}")
    log.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    model = BELKATransformer(
        vocab_size=checkpoint["vocab_size"],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_length=config.max_length,
        pad_idx=0,
        num_targets=3,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # ── 3. 加载测试数据 ──
    test_df = load_test(config.test_path)
    smiles_test = test_df["molecule_smiles"].tolist()
    log.info(f"Test molecules: {len(smiles_test):,}")

    # ── 4. 创建 DataLoader ──
    test_loader = create_test_dataloder(
        smiles_test, tokenizer,
        batch_size=config.batch_size * 2,
        max_length=config.max_length,
        num_workers=config.num_workers,
    )

    # ── 5. 推理 ──
    log.info("Running inference...")
    probs = predict(model, test_loader, device)
    log.info(f"Predictions shape: {probs.shape}")

    # 统计
    for i, name in enumerate(PROTEIN_NAMES):
        p = probs[:, i]
        log.info(
            f"  {name}: mean={p.mean():.4f}, "
            f"min={p.min():.4f}, max={p.max():.4f}, "
            f"positive_rate={np.mean(p > 0.5):.4f}"
        )

    # ── 6. 生成长格式 submission（匹配 sample_submission.csv）──
    sub_df = generate_submission(probs, config.test_path)

    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
    sub_df.to_csv(config.submission_path, index=False)
    log.info(f"Submission saved to {config.submission_path}")
    log.info(f"  Shape: {sub_df.shape}")
    log.info(f"  Columns: {sub_df.columns.tolist()}")
    log.info(f"  Sample:\n{sub_df.head(10).to_string(index=False)}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpt/best_model.pt")
    parser.add_argument("--vocab_path", type=str, default="ckpt/vocab.json")
    parser.add_argument("--test_path", type=str, default="data/test_300.csv")
    parser.add_argument("--submission_path", type=str, default="ckpt/submission.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    config = Config(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        test_path=args.test_path,
        submission_path=args.submission_path,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_length=args.max_length,
    )
    inference(config)
