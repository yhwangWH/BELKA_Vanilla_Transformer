# BELKA Vanilla Transformer Baseline

基于 **SMILES + Transformer Encoder** 的多任务二分类模型，用于 Kaggle BELKA 竞赛。

## 任务

预测小分子对三个蛋白靶点的结合概率：
- **BRD4** (Bromodomain-containing protein 4)
- **HSA** (Human Serum Albumin)  
- **sEH** (Soluble Epoxide Hydrolase / EPHX2)

数据量：~295M 行训练数据（长格式），约 98M 个唯一分子。

---

## 项目结构

```
├── data.py          # 数据加载（流式读取 parquet）+ pivot long→wide
├── tokenizer.py     # SMILES 分词器 + 词表构建
├── dataset.py       # PyTorch Dataset + DataLoader 工厂
├── model.py         # Transformer Encoder 模型 + 损失函数
├── train.py         # 训练脚本（AMP + early stopping）
├── inference.py     # 推理 + submission 生成
├── utils.py         # 日志、指标、Config
├── requirements.txt
└── data/            # 数据目录
    ├── train.parquet       # 训练数据（~295M 行）
    ├── test.parquet        # 测试数据（~1.67M 行）
    └── sample_submission.csv
```

---

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
# 完整训练（默认最大 200k 负样本 + 全部正样本）
python train.py

# 使用 CSV 格式小样本测试
python train.py --train_path data/train_sample.csv --epochs 10 --batch_size 64

# 自定义参数
python train.py \
    --epochs 30 \
    --batch_size 128 \
    --lr 1e-4 \
    --d_model 256 \
    --num_layers 4 \
    --nhead 8 \
    --max_negatives 500000 \
    --dropout 0.1
```

## 推理

```bash
python inference.py
# 输出：
#   ckpt/submission.csv       # 宽格式 (id, BRD4, HSA, sEH)
#   ckpt/submission_long.csv  # 长格式 (id, binds) 兼容官方格式
```

---

## 模型架构

```
SMILES → Tokenizer → Embedding + Positional Encoding
       → Transformer Encoder (Pre-LN, GELU)
       → [CLS] Pooling
       → MLP Head (Linear → GELU → Dropout → Linear)
       → 3 logits [BRD4, HSA, sEH]
```

**推荐配置：**
| 参数 | 值 |
|------|-----|
| d_model | 256 |
| nhead | 8 |
| num_layers | 4 |
| dim_feedforward | 512 |
| dropout | 0.1 |
| max_length | 256 |

**损失函数：** `BCEWithLogitsLoss` + `pos_weight`（处理极度不平衡）

---

## 设计约束

- ✅ 仅使用 SMILES 字符串作为输入
- ✅ 仅使用 PyTorch Transformer Encoder
- ✅ 自定义 tokenizer（非预训练）
- ❌ 不使用 GNN / ECFP / MACCS / RDKit descriptors / ChemBERTa
