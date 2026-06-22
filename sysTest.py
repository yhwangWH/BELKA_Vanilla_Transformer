"""
sysTest.py
从 data/test.parquet 随机采样 300 行，输出 data/test_300.csv
注意：test.parquet 无 binds 列，无法按正负样本筛选，此处纯随机采样。
"""
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = "data/test.parquet"
OUTPUT_PATH = "data/test_300.csv"
SAMPLE_SIZE = 300
RANDOM_STATE = 42


def main():
    pf = pq.ParquetFile(INPUT_PATH)
    num_row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    target_per_rg = max(1, SAMPLE_SIZE // num_row_groups)

    print(f"[sysTest] File: {INPUT_PATH}")
    print(f"[sysTest] Total rows: {total_rows:,}, row_groups: {num_row_groups}")
    print(f"[sysTest] Target ~{target_per_rg} rows per row_group")

    rng = np.random.RandomState(RANDOM_STATE)
    samples = []

    for rg_idx in range(num_row_groups):
        if len(samples) >= SAMPLE_SIZE:
            break
        # 计算本 row_group 应采多少
        remaining = SAMPLE_SIZE - len(samples)
        rg_remaining = num_row_groups - rg_idx
        take = min(remaining, max(1, remaining // rg_remaining))

        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()
        if len(df) <= take:
            samples.append(df)
        else:
            idxs = rng.choice(len(df), size=take, replace=False)
            samples.append(df.iloc[idxs])
        print(f"[sysTest] row_group {rg_idx}/{num_row_groups}: took {min(take, len(df))}, total collected={sum(len(s) for s in samples)}")

    final = pd.concat(samples, ignore_index=True)
    if len(final) > SAMPLE_SIZE:
        final = final.iloc[:SAMPLE_SIZE]

    final = final.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[sysTest] Final rows: {len(final)}")
    print(f"[sysTest] Saved to {OUTPUT_PATH}")
    print(f"[sysTest] Columns: {list(final.columns)}")
    if "binds" in final.columns:
        print(f"[sysTest] binds distribution: {final['binds'].value_counts().to_dict()}")
    else:
        print("[sysTest] No 'binds' column in test data (expected).")


if __name__ == "__main__":
    main()
