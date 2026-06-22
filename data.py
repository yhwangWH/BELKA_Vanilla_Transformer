"""
数据加载与预处理：将 BELKA 长格式数据 pivot 为分子级宽格式。

训练数据格式（长格式）：
    id, buildingblock1_smiles, buildingblock2_smiles, buildingblock3_smiles,
    molecule_smiles, protein_name, binds

预处理后（宽格式）：
    molecule_smiles, label_BRD4, label_HSA, label_sEH

测试数据格式（长格式）：
    id, buildingblock1_smiles, ..., molecule_smiles, protein_name

预处理后：
    molecule_id, molecule_smiles
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from typing import Tuple, Optional
from utils import log, PROTEIN_NAMES


def load_and_pivot_train(
    path: str,
    sample_frac: Optional[float] = None,
    max_negatives: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    加载训练数据并 pivot 为宽格式。

    对于 .parquet 文件，自动使用流式读取以避免 OOM（全量数据 295M 行）。
    默认保留所有正样本 + 最多 max_negatives 条负样本。

    输入（长格式）: 每行 = (molecule, protein) pair
    输出（宽格式）: 每行 = 一个 molecule，标签为 [BRD4, HSA, sEH] 的 0/1

    Args:
        path: 数据文件路径
        sample_frac: 采样比例（仅对 .csv 生效）
        max_negatives: 最多保留的负样本数（默认 None = 全量，对 .parquet 推荐设置此参数）
        random_state: 随机种子

    Returns:
        DataFrame with columns: molecule_smiles, BRD4, HSA, sEH
    """
    log.info(f"Loading training data from {path} ...")

    if path.endswith(".parquet"):
        df = _load_parquet_streaming(path, max_negatives, random_state)
    else:
        df = pd.read_csv(path)
        if sample_frac is not None and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
            log.info(f"Sampled {len(df):,} rows (frac={sample_frac})")

    # 统计
    total = len(df)
    positive = df["binds"].sum()
    log.info(
        f"Train data: {total:,} rows, positive={positive:,} "
        f"({100 * positive / max(total, 1):.2f}%)"
    )

    for p in PROTEIN_NAMES:
        subset = df[df["protein_name"] == p]
        log.info(f"  {p}: {len(subset):,} rows, binds={subset['binds'].sum():,}")

    # Pivot: long → wide
    pivot = df.pivot_table(
        index="molecule_smiles",
        columns="protein_name",
        values="binds",
        aggfunc="first",
    ).reset_index()

    # 确保三列都存在
    for p in PROTEIN_NAMES:
        if p not in pivot.columns:
            pivot[p] = 0

    pivot = pivot[["molecule_smiles"] + PROTEIN_NAMES]
    pivot[PROTEIN_NAMES] = pivot[PROTEIN_NAMES].fillna(0).astype(int)

    # 统计 pivot 后
    for p in PROTEIN_NAMES:
        pos = pivot[p].sum()
        log.info(f"  Wide {p}: {pos} positive out of {len(pivot)} molecules")

    log.info(f"Pivoted to {len(pivot):,} unique molecules")
    return pivot


def _load_parquet_streaming(
    path: str, max_negatives: Optional[int], random_state: int
) -> pd.DataFrame:
    """
    流式读取 parquet，保留全部正样本 + 随机采样负样本。

    Args:
        path: parquet 文件路径
        max_negatives: 最多保留负样本数（None = 不限制）
        random_state: 随机种子
    """
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    log.info(f"Parquet file: {total_rows:,} total rows, streaming read...")

    positives = []
    negatives = []
    n_pos_total = 0
    n_neg_total = 0
    rng = np.random.RandomState(random_state)

    batch_size = 500_000
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
        df_batch = batch.to_pandas()

        pos_mask = df_batch["binds"] == 1
        neg_mask = ~pos_mask

        pos_batch = df_batch[pos_mask]
        neg_batch = df_batch[neg_mask]

        if len(pos_batch) > 0:
            positives.append(pos_batch)
            n_pos_total += len(pos_batch)

        n_neg_total += len(neg_batch)

        # 负样本按比例采样
        if max_negatives is not None and max_negatives > 0:
            # 动态调整采样率：让负样本均匀分布在各批次中
            neg_sample_rate = max_negatives / max(n_neg_total, 1)
            if len(neg_batch) > 0:
                n_neg_sample = max(1, int(len(neg_batch) * neg_sample_rate))
                if len(neg_batch) <= n_neg_sample:
                    negatives.append(neg_batch)
                else:
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
            log.info(
                f"  Batch {batch_idx + 1}: {n_pos_total:,} positives, "
                f"{n_neg_collected:,} negatives collected"
            )

    pos_df = pd.concat(positives, ignore_index=True) if positives else pd.DataFrame()
    neg_df = pd.concat(negatives, ignore_index=True) if negatives else pd.DataFrame()

    if max_negatives is not None and len(neg_df) > max_negatives:
        neg_df = neg_df.sample(n=max_negatives, random_state=random_state)

    log.info(
        f"Streaming done: {len(pos_df):,} positives, {len(neg_df):,} negatives"
    )

    return pd.concat([pos_df, neg_df], ignore_index=True)


def load_test(path: str) -> pd.DataFrame:
    """
    加载测试数据，提取每个分子的信息。

    Returns:
        DataFrame with columns: molecule_id, molecule_smiles
    """
    log.info(f"Loading test data from {path} ...")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    log.info(f"Test data: {len(df):,} rows")

    # 测试数据中每个 molecule_smiles 出现 3 次（对应 3 个 protein）
    # 按 molecule_smiles 分组，取第一个 id 作为分子 id
    grouped = df.groupby("molecule_smiles").agg(
        molecule_id=("id", "first"),
    ).reset_index()

    log.info(f"Unique test molecules: {len(grouped):,}")
    return grouped


def compute_pos_weight(labels: np.ndarray) -> np.ndarray:
    """
    计算 BCEWithLogitsLoss 的 pos_weight。

    pos_weight = num_negatives / num_positives

    Args:
        labels: (N, 3) 二值标签

    Returns:
        Tensor of shape (3,) — 每个 target 的 pos_weight
    """
    import torch
    n_pos = labels.sum(axis=0).clip(min=1)
    n_neg = labels.shape[0] - n_pos
    pos_weight = n_neg / n_pos
    return torch.tensor(pos_weight, dtype=torch.float32)


if __name__ == "__main__":
    # 快速测试
    df = load_and_pivot_train("data/train_300.csv")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Labels: \n{df[PROTEIN_NAMES].sum()}")

    labels = df[PROTEIN_NAMES].values
    pw = compute_pos_weight(labels)
    print(f"\npos_weight: {pw}")
