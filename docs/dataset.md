# 数据集

> 本文档说明 Emotion Pet 使用的情感识别数据集、预处理流程和伦理规范。

---

## 数据集总览

项目采用多模态数据策略，覆盖文本、语音两个通道。推荐数据集如下：

### 文本情感数据集

| 数据集 | 规模 | 标签 | 语言 | 用途 |
|--------|------|------|------|------|
| SST-2 | ~67k 句 | 2 类 (pos/neg) | 英文 | 文本分支预训练、基线对比 |
| GoEmotions | ~58k 句 | 27 类 → 映射至 7 类 | 英文 | 主训练集，细粒度情感 |
| CH-SIMS | ~3k 条 | 7 类情感 + 强度 | 中文 | 中文场景适配 |

**来源**：

- SST-2: https://nlp.stanford.edu/sentiment/
- GoEmotions: https://github.com/google-research/google-research/tree/master/goemotions
- CH-SIMS: https://github.com/thuiar/MMSA

### 语音情感数据集

| 数据集 | 规模 | 标签 | 语言 | 用途 |
|--------|------|------|------|------|
| RAVDESS | 1440 条 | 8 类情绪 | 英文 | 主训练集 |
| EMO-DB | 535 条 | 7 类情绪 | 德文 | 数据增强、跨语言验证 |
| IEMOCAP | ~12h | 6 类情绪 | 英文 | 自然对话场景 |

**来源**：

- RAVDESS: https://zenodo.org/record/1188976
- EMO-DB: https://www.emodb.org/
- IEMOCAP: https://sail.usc.edu/iemocap/ (需申请)

### 多模态数据集（融合实验用）

| 数据集 | 模态 | 规模 | 用途 |
|--------|------|------|------|
| MELD | 文本+语音+视频 | ~13k 句 | 多模态融合验证 |
| IEMOCAP | 文本+语音+视频 | ~12h | 多模态融合主实验 |

**来源**：

- MELD: https://affective-meld.github.io/

---

## 标签体系

项目使用 7 类情感标签，对齐 config.yaml 设定：

```
happy, sad, angry, fear, surprise, disgust, neutral
```

各数据集原始标签需映射到统一体系。映射规则示例：

| 原始标签 (GoEmotions) | 映射目标 |
|------------------------|----------|
| joy, amusement, excitement | → happy |
| sadness, grief | → sad |
| anger, annoyance | → angry |
| fear, nervousness | → fear |
| surprise | → surprise |
| disgust | → disgust |
| neutral | → neutral |

RAVDESS 的 "calm" 映射到 neutral，"happy" 映射到 happy。

---

## 预处理流程

### 文本数据

```
原始文本
    │
    ▼
清洗：去 HTML 标签、特殊字符、URL
    │
    ▼
分词：BERT Tokenizer (bert-base-chinese / bert-base-uncased)
    │
    ▼
截断/填充：max_length = 128
    │
    ▼
输出：input_ids, attention_mask, token_type_ids
    │
    ▼
保存：data/processed/text/train.pt, val.pt, test.pt
```

### 语音数据

```
原始音频 (.wav)
    │
    ▼
重采样：16kHz, 单声道 (librosa.load)
    │
    ▼
特征提取：
  ├── MFCC: 40 维, 帧长 25ms, 帧移 10ms
  └── Mel 频谱: 128 维
    │
    ▼
归一化：CMVN (均值方差归一化)
    │
    ▼
对齐/截断：固定长度 3 秒 (pad/truncate)
    │
    ▼
输出：feature tensor [1, 128, T]
    │
    ▼
保存：data/processed/audio/train.pt, val.pt, test.pt
```

---

## 数据划分

所有数据集按 **8:1:1** 比例划分训练集/验证集/测试集：

```
data/
├── raw/
│   ├── text/
│   │   ├── train.csv      # 80%
│   │   ├── val.csv         # 10%
│   │   └── test.csv        # 10%
│   ├── audio/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels.csv
└── processed/
    ├── text/
    │   ├── train.pt
    │   ├── val.pt
    │   └── test.pt
    └── audio/
        ├── train.pt
        ├── val.pt
        └── test.pt
```

划分策略：

- **SST-2 / GoEmotions**：使用官方划分，若官方无 val 则从 train 中随机抽 10% 作为 val
- **RAVDESS / EMO-DB**：按说话人划分 (speaker-independent split)，避免同一说话人的数据同时出现在 train 和 test
- **IEMOCAP**：使用 Leave-One-Session-Out (LOSO) 交叉验证

---

## 数据集统计

> 以下为数据集原始规模。实际预处理后的样本数可能因过滤、标签映射而略有变化。

| 数据集 | 原始样本数 | 用于本项目 | 划分 (train/val/test) |
|--------|------------|------------|----------------------|
| SST-2 | 67,349 | 待确认 | 待补充 |
| GoEmotions | 58,054 | 待确认 | 待补充 |
| RAVDESS | 1,440 | 全部 | 待补充 |
| EMO-DB | 535 | 全部 | 待补充 |
| MELD | 13,708 | 待确认 | 待补充 |
| IEMOCAP | ~10,039 | 待确认 | 待补充 |

> 注：具体统计数字待数据预处理完成后补充。

---

## 隐私与伦理

### 数据敏感性

本项目的目标用户是大学生群体，关注点是心理健康。这涉及以下敏感问题：

1. **情感数据属于个人敏感信息**。即使用户同意，也必须最小化数据收集范围。
2. **心理健康筛查结果（如 PHQ-9、GAD-7）高度敏感**。这些数据仅在用户测试阶段收集，且严格匿名化。

### 本项目的数据策略

- **训练数据**：全部使用公开数据集（SST-2、GoEmotions、RAVDESS 等），不涉及真实用户数据
- **用户测试数据**：
  - 收集前签署知情同意书
  - 所有数据匿名化处理，用随机 ID 替代真实身份
  - 仅存储情感标签和交互日志，不存储原始语音/文本
  - 用户可随时退出并删除自己的数据
- **存储**：所有数据本地存储，不上传任何云端服务
- **访问控制**：仅研究团队成员可访问原始数据

### 伦理审查

本项目涉及人类被试（大学生用户测试），已提交伦理审查申请。遵循以下原则：

- **自愿参与**：被试可随时退出，不影响任何权益
- **知情同意**：明确告知数据用途、存储方式、销毁时间
- **最小伤害**：系统不会诊断或评判用户心理状态，仅提供陪伴
- **数据安全**：实验结束后，原始交互数据在 6 个月内销毁

### 使用公开数据集的合规性

| 数据集 | 许可证 | 合规说明 |
|--------|--------|----------|
| SST-2 | 公开可用 | 学术用途，需引用原始论文 |
| GoEmotions | CC BY 4.0 | 可商用，需归属 |
| RAVDESS | CC BY-NC-SA 4.0 | 非商用，需归属 |
| EMO-DB | 学术免费 | 需申请，仅限研究用途 |
| IEMOCAP | 需签署协议 | 需申请，严格保密 |
| MELD | 公开可用 | 学术用途，需引用原始论文 |
