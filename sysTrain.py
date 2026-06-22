"""
sysTrain.py
从 data/train.parquet 流式提取 300 行（30 正样本 + 270 负样本），输出 data/train_300.csv
正样本按三个蛋白质均分：BRD4=10, HSA=10, sEH=10
"""
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = "data/train.parquet"
OUTPUT_PATH = "data/train_300.csv"
TOTAL_ROWS = 300
POSITIVE_COUNT = 30       # 30 个正样本
NEGATIVE_COUNT = 270      # 270 个负样本
PER_PROTEIN_POS = POSITIVE_COUNT // 3  # 每个蛋白 10 个
RANDOM_STATE = 42

PROTEINS = ["BRD4", "HSA", "sEH"]


def main():
    rng = np.random.RandomState(RANDOM_STATE)
    pf = pq.ParquetFile(INPUT_PATH)
    num_row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"[sysTrain] File: {INPUT_PATH}")
    print(f"[sysTrain] Total rows: {total_rows:,}, row_groups: {num_row_groups}")

    # 收集正负样本
    pos_samples = {p: [] for p in PROTEINS}  # 按蛋白分开
    neg_samples = []
    pos_needed = {p: PER_PROTEIN_POS for p in PROTEINS}
    neg_needed = NEGATIVE_COUNT

    for rg_idx in range(num_row_groups):
        any_needed = neg_needed > 0 or any(v > 0 for v in pos_needed.values())
        if not any_needed:
            break

        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()
        print(f"[sysTrain] Processing row_group {rg_idx}/{num_row_groups}, "
              f"rows={len(df)}, pos_left=[{', '.join(f'{k}={v}' for k,v in pos_needed.items())}], "
              f"neg_left={neg_needed}")

        # 正样本：按蛋白分开
        for prot in PROTEINS:
            if pos_needed[prot] <= 0:
                continue
            mask = (df["binds"] == 1) & (df["protein_name"] == prot)
            pos_df = df[mask]

            if len(pos_df) > pos_needed[prot]:
                # 如果该 row_group 中正样本足够，随机抽取
                idxs = rng.choice(len(pos_df), size=pos_needed[prot], replace=False)
                pos_samples[prot].append(pos_df.iloc[idxs])
                pos_needed[prot] = 0
            else:
                pos_samples[prot].append(pos_df)
                pos_needed[prot] -= len(pos_df)

        # 负样本
        if neg_needed > 0:
            neg_df = df[df["binds"] == 0]
            if len(neg_df) > neg_needed:
                idxs = rng.choice(len(neg_df), size=neg_needed, replace=False)
                neg_samples.append(neg_df.iloc[idxs])
                neg_needed = 0
            else:
                neg_samples.append(neg_df)
                neg_needed -= len(neg_df)

    # 合并
    all_pos = []
    for prot in PROTEINS:
        if pos_samples[prot]:
            all_pos.append(pd.concat(pos_samples[prot], ignore_index=True))
    pos_combined = pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()
    neg_combined = pd.concat(neg_samples, ignore_index=True) if neg_samples else pd.DataFrame()

    print(f"\n[sysTrain] Collected: positive={len(pos_combined)}, negative={len(neg_combined)}")
    if len(pos_combined) > 0:
        print(f"[sysTrain] Positives per protein: {pos_combined['protein_name'].value_counts().to_dict()}")

    # 确保正好 300 行
    final = pd.concat([pos_combined, neg_combined], ignore_index=True)
    if len(final) > TOTAL_ROWS:
        final = final.iloc[:TOTAL_ROWS]
    print(f"[sysTrain] Final rows: {len(final)}")

    # 打乱并保存
    final = final.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"[sysTrain] Saved to {OUTPUT_PATH}")
    print(f"[sysTrain] Columns: {list(final.columns)}")
    print(f"[sysTrain] binds distribution: {final['binds'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
