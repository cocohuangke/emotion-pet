# Data

本目录存放数据集。**大文件不提交到 git**（见 `.gitignore`）。

## 目录结构

```
data/
├── raw/           # 原始数据（不提交）
│   ├── text/      # 文本情感数据（GoEmotions 英文 + ESD 中文转录）
│   ├── audio/     # 语音情感数据（RAVDESS/CASIA/EMO-DB/ESD，16kHz WAV）
│   └── labels.csv # 统一标签映射（~7,444 行）
├── processed/     # 预处理后数据（不提交）
└── README.md      # 本文件
```

## 数据集

本项目使用 5 个数据集的精简方案（约 7,444 行）：

| 数据集 | 模态 | 行数 | 语言 |
|--------|------|------|------|
| GoEmotions（子采样） | 文本 | 3,000 | 英文 |
| ESD 中文子集（子采样） | 文本 + 语音 | 1,500 | 中文 |
| RAVDESS Speech | 语音 | 1,440 | 英文 |
| CASIA | 语音 | 1,200 | 中文 |
| EMO-DB | 语音 | 304 | 德文 |

数据集选型与剔除理由详见 `docs/dataset.md`。

## 数据获取

```bash
# 下载并预处理数据集
python scripts/download_data.py --dataset all --target ./data/raw
```

国内网络环境需先设置 HF 镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```
