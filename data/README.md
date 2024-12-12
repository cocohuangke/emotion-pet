# Data

本目录存放数据集。**大文件不提交到 git**（见 `.gitignore`）。

## 目录结构

```
data/
├── raw/           # 原始数据（不提交）
│   ├── text/      # 文本情感数据
│   ├── audio/     # 语音情感数据
│   └── labels.csv # 标签
├── processed/     # 预处理后数据（不提交）
└── README.md      # 本文件
```

## 数据获取

运行 `python scripts/download_data.py` 下载并预处理数据集。

详见 `docs/dataset.md`。
