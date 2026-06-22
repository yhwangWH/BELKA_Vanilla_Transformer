"""
PyTorch Dataset 类：将 SMILES 字符串转换为模型输入。
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional
from tokenizer import SMILESTokenizer, PAD_IDX


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
        tokenizer: SMILESTokenizer,
        max_length: int = 256,
    ):
        self.smiles_list = smiles_list
        self.labels = labels  # (N, 3) or None (for test)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        smiles = self.smiles_list[idx]

        # Encode + pad
        token_ids = self.tokenizer.encode(smiles, add_cls=True)
        padded = self._pad(token_ids, self.max_length, PAD_IDX)

        input_ids = torch.tensor(padded, dtype=torch.long)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return input_ids, label
        else:
            return input_ids, torch.tensor(0)  # dummy

    def _pad(self, ids: List[int], max_len: int, pad_id: int) -> List[int]:
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [pad_id] * (max_len - len(ids))


def create_dataloaders(
    smiles_train: List[str],
    labels_train: np.ndarray,
    smiles_val: List[str],
    labels_val: np.ndarray,
    tokenizer: SMILESTokenizer,
    batch_size: int = 128,
    max_length: int = 256,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """创建训练和验证 DataLoader"""

    train_ds = BELKADataset(smiles_train, labels_train, tokenizer, max_length)
    val_ds = BELKADataset(smiles_val, labels_val, tokenizer, max_length)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def create_test_dataloder(
    smiles_test: List[str],
    tokenizer: SMILESTokenizer,
    batch_size: int = 256,
    max_length: int = 256,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """创建测试 DataLoader"""
    test_ds = BELKADataset(smiles_test, None, tokenizer, max_length)
    return torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    from tokenizer import SMILESTokenizer

    # 测试用 SMILES
    smiles = [
        "C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)N[Dy])n2)cc1",
        "CCO",
        "c1ccccc1",
    ]
    labels = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.float32)

    tok = SMILESTokenizer(max_length=128)
    tok.build_vocab(smiles)

    ds = BELKADataset(smiles, labels, tok, max_length=128)
    ids, lbl = ds[0]
    print(f"Input shape: {ids.shape}, Label: {lbl}")
    print(f"Tokens: {ids[:10]}")
