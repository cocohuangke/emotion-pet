# 数据集

> 本文档说明 Emotion Pet 使用的情感识别数据集、标签映射、预处理流程、下载方式与伦理规范。

---

## 数据集总览

项目采用多模态数据策略，覆盖文本、语音两个通道，目标用户为中文大学生群体。

考虑到本项目为本科生的假期实践项目，核心目标是**跑通完整端到端管线并验证情感识别能力**，训练需在免费 Colab GPU 上约 20 分钟内完成，因此对数据规模做了精简：从最初调研的 8 个数据集（~16 万行）收敛到 **5 个数据集、约 7,400 行**，剔除与项目目标不匹配的 SST-2（二元情感噪声）、RAVDESS Song（唱歌≠说话情感）、MELD（FLAC 转换瓶颈、ROI 低）。

### 精简方案选型

| 数据集 | 模态 | 规模 | 标签 | 语言 | 采样率 | 许可证 | 用途 |
|--------|------|------|------|------|--------|--------|------|
| GoEmotions | 文本 | 3,000（子采样） | 27 类 → 7 类 | 英文 | — | Apache-2.0 | 英文文本分支训练 |
| ESD（中文子集） | 文本 + 语音 | 1,500（子采样） | 5 类 | 中文 | 16kHz/mono/16-bit | 研究用途（需签协议） | 中文文本 + 中文语音（配对） |
| RAVDESS Speech | 语音 | 1,440 | 8 类 → 7 类 | 英文 | 48kHz → 16kHz | CC BY-NC-SA 4.0 | 英文语音训练 |
| CASIA | 语音 | 1,200 | 6 类（无 disgust） | 中文 | 16kHz/mono/16-bit | 研究免费（HF 镜像） | 中文语音主训练集 |
| EMO-DB | 语音 | 304 | 7 类 | 德文 | 16kHz/mono/16-bit | 研究免费 | 补充 fear/disgust |
| **合计** | — | **~7,444** | **7 类** | 中英德 | — | — | — |

### 剔除的数据集及原因

| 数据集 | 剔除原因 |
|--------|----------|
| SST-2 | 二元情感（pos/neg）强行映射成 happy/sad，对 7 类任务是噪声；67k 行占比过大，淹没细粒度信号 |
| RAVDESS Song | 唱歌情感与说话情感特征差异大，不能直接迁移；增量小（1,012 行）ROI 低 |
| MELD | 13,847 个 FLAC 文件转换 WAV 耗时超 30 分钟（超时 3 次），HF 镜像 CSV 还被截断；视频原始数据 10.8GB 不可用 |
| IEMOCAP | 需 Google Form 申请 + 3-5 天审批 + 学术邮箱，许可证限制严格，仍为 acted 英文数据，不填补中文缺口 |

来源：

- GoEmotions: https://github.com/google-research/google-research/tree/master/goemotions
- RAVDESS: https://zenodo.org/record/1188976
- EMO-DB: http://emodb.bilderbar.info/download/download.zip
- CASIA: https://hf-mirror.com/datasets/BillyLin/CASIA_speech_emotion_recognition_preload
- ESD: https://hf-mirror.com/datasets/jspaulsen/esd

---

## 中文数据集选型说明

项目目标用户是中文大学生，语音分支需要中文情感数据。选择 CASIA + ESD 中文子集的理由：

**CASIA 中文情感语音库**

- 中科院自动化所录制，6 种情感（anger/fear/happy/neutral/sad/surprise），与项目目标标签高度对齐
- 16kHz/16-bit/mono 采样，棚录高质量数据
- 4 位说话人，共 1,200 条
- HuggingFace 镜像可免费获取，商用需向 chineseldc.org 购买正式版
- 缺点：规模较小（1,200 条），说话人少（4 人），无 disgust 类别

**ESD（Emotional Speech Dataset）中文子集**

- 5 种情感（anger/happiness/neutral/sadness/surprise），无 fear 和 disgust
- 10 位中文说话人 + 10 位英文说话人（同一批人双语录制）
- 中文子集原始约 17,500 条，本项目子采样到 1,500 条（每类 300）
- 16kHz/16-bit/mono
- 研究用途，需签署数据使用协议
- 优点：**同时提供中文语音和对应的中文文本转录**，是唯一能做文本+语音配对训练的中文数据集
- 缺点：缺少 fear/disgust 两个类别

**互补策略**

- fear/disgust 主要由 EMO-DB 补充（disgust 20 条、anxiety→fear 36 条）和 RAVDESS 补充
- CASIA 的 6 类 + ESD 的 5 类 + EMO-DB 的 7 类 + RAVDESS 的 8 类 → 项目 7 类全覆盖
- 中文数据共 2,700 条（CASIA 1,200 + ESD 1,500），约占总量 36%

---

## 标签体系

项目使用 7 类情感标签，对齐 config.yaml 设定：

```
happy, sad, angry, fear, surprise, disgust, neutral
```

各数据集原始标签需映射到统一体系。

### GoEmotions（27 类 → 7 类）

| 原始标签 | 映射目标 |
|----------|----------|
| joy, amusement, excitement, gratification, optimism, relief, pride, admiration, love, desire, anticipation | → happy |
| sadness, grief, disappointment, remorse, embarrassment | → sad |
| anger, annoyance, rage | → angry |
| fear, nervousness | → fear |
| surprise (startle, surprise) | → surprise |
| disgust | → disgust |
| neutral | → neutral |

### RAVDESS Speech

| 原始标签 | 映射目标 |
|----------|----------|
| angry | → angry |
| fearful | → fear |
| disgust | → disgust |
| sad | → sad |
| surprised | → surprise |
| happy | → happy |
| calm | → neutral |
| neutral | → neutral |

### EMO-DB（文件名第 5 字符为情感字母）

| 字母 | 德语 | 映射目标 |
|------|------|----------|
| W | Wut (anger) | → angry |
| L | Langeweile (boredom) | → neutral |
| E | Ekel (disgust) | → disgust |
| A | Angst (fear) | → fear |
| F | Freude (joy) | → happy |
| T | Trauer (sadness) | → sad |
| N | Neutral | → neutral |

> 注：boredom（L）映射到 neutral，因为项目无独立 boredom 类别。

### CASIA

| 原始标签 | 映射目标 |
|----------|----------|
| anger | → angry |
| fear | → fear |
| happy | → happy |
| neutral | → neutral |
| sad | → sad |
| surprise | → surprise |

> 注：CASIA 无 disgust 类别。

### ESD

| 原始标签 | 映射目标 |
|----------|----------|
| anger | → angry |
| happiness | → happy |
| neutral | → neutral |
| sadness | → sad |
| surprise | → surprise |

> 注：ESD 无 fear 和 disgust 类别。

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
分词：BERT Tokenizer
  ├── 英文文本 (GoEmotions) → bert-base-uncased
  └── 中文文本 (ESD 转录) → bert-base-chinese
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

> 决策说明：GoEmotions 为英文，ESD 中文子集提供中文转录，因此文本分支同时存在中英两种语言。使用 `bert-base-chinese` 作为文本分支主模型（中文是目标用户语言），英文文本通过 tokenizer 自动处理。

### 语音数据

```
原始音频 (.wav)
    │
    ▼
重采样：16kHz, 单声道 (librosa.load)
  ├── RAVDESS: 48kHz → 16kHz (需重采样)
  ├── EMO-DB: 已是 16kHz (跳过)
  ├── CASIA: 已是 16kHz (跳过)
  └── ESD: 已是 16kHz (跳过)
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

| 数据集 | 划分策略 |
|--------|----------|
| GoEmotions | 官方 train/dev/test，子采样仅从 train 取 3,000 |
| RAVDESS Speech | 按说话人划分（speaker-independent split） |
| EMO-DB | 按说话人划分（10 actors） |
| CASIA | 按说话人划分（4 speakers，说话人少需谨慎） |
| ESD | 按说话人划分（10 Chinese speakers） |

```
data/
├── raw/
│   ├── text/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
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

语音数据集统一按说话人划分，避免同一说话人的数据同时出现在 train 和 test，防止说话人特征泄露。

---

## 数据集统计

> 以下为精简方案的实际使用规模。

| 数据集 | 原始规模 | 本项目使用 | 语言 | 模态 | 划分策略 |
|--------|----------|-----------|------|------|----------|
| GoEmotions | 43,410 train | 3,000（子采样） | 英文 | 文本 | 官方 train 子集 |
| ESD（中文） | ~17,500 | 1,500（子采样） | 中文 | 文本 + 语音 | 按说话人划分 |
| RAVDESS Speech | 1,440 | 1,440 | 英文 | 语音 | 按说话人划分 |
| CASIA | 1,200 | 1,200 | 中文 | 语音 | 按说话人划分 |
| EMO-DB | 535 | 304* | 德文 | 语音 | 按说话人划分 |
| **合计** | — | **~7,444** | **中英德** | — | — |

> *EMO-DB 完整版 535 条，本项目使用的 HF 镜像 `confit/emodb-parquet` 仅含 304 条（shard 0），差异源于镜像只导出了部分数据。304 条已覆盖 7 类情感，足以补充 fear/disgust 稀疏类别。

---

## 跨语言泛化问题与对策

### 问题

外语数据训练的语音情感模型用于中文用户时，准确率会大幅下降。

已有研究结论：

- EMO-DB → CASIA 零迁移（德→中）：仅 24.8% UAR（同语言测试通常 85-95%）
- 英语→英语跨语料库：仍下降 10-40%（Braunschweiler 2022：matched 89.8% vs mismatched 65.3%）

下降原因：

1. **韵律是语言特定的**。arousal（唤醒度）可跨语言迁移，valence（效价）下降严重
2. **普通话声调干扰**。音高同时承载词汇声调（四声）和情感韵律，模型容易混淆
3. **acted vs natural 不匹配**。实验室录制与自然表达差距大
4. **语料库特定不匹配**。录音条件、说话人、标注文化差异

### 本项目的对策

1. **引入中文数据**（已采纳）：CASIA 1,200 + ESD 1,500 = 2,700 条中文语音。研究显示 160 秒目标语言数据可提升 +15.6% WA（Tang 2023）。
2. **多语言拼接训练**：中英德三语数据联合训练，优于单源 zero-shot。
3. **按说话人划分**：防止说话人特征泄露，更贴近真实泛化场景。

### 局限性承认

即便引入中文数据，跨语言泛化仍存在上限。本项目作为本科实践项目，接受这一局限：

- 训练集以英文为主（RAVDESS 1,440 + GoEmotions 3,000 = 4,440 英文行 vs 2,700 中文行）
- 测试用户为中文大学生，预期中文语音识别准确率低于英文
- 这一差距本身就是实验报告的重要观察点

---

## 模型架构

当前架构：BERT（text）+ CNN+LSTM（audio）→ 7-class fused logits。

### 文本分支

- 使用 `bert-base-chinese` 作为主模型（目标用户为中文大学生）
- 英文文本（GoEmotions）通过同一 tokenizer 处理
- 输入：text → tokenize → BERT → [CLS] → FC → 7-class logits

### 语音分支

- **CNN+LSTM 从零训练**（不从外部加载预训练 SER 模型）
- 输入：Mel 频谱（128 维）→ Conv2D（3 层，[64, 128, 256]）→ BiLSTM（2 层，hidden=128）→ FC → 7-class logits
- 设计理由：项目核心目标之一是**学习并展示 CNN+LSTM 在 SER 任务上的从零训练能力**，直接使用 emotion2vec+ 等预训练模型相当于"调用 API"，失去教育价值
- 预期：作为 baseline，在 RAVDESS+CASIA+EMO-DB+ESD 上训练 20 epochs，观察收敛情况

### 融合层

- 对文本和语音分支的隐层表示做注意力加权拼接，输出最终情感分布

### 未来改进方向（不在当前实现范围内）

若 baseline CNN+LSTM 效果不理想，可考虑：

- **emotion2vec+**（阿里巴巴达摩院，ACL 2024）：中文原生 SER 基础模型，训练数据 42,526 小时，9 类输出与项目 7 类兼容。免费获取：ModelScope / HuggingFace `emotion2vec/emotion2vec_plus_large`。使用方式：(a) 直接作为分类器 fine-tune；(b) 作为特征提取器替代 Mel 频谱输入 CNN+LSTM。
- **wav2vec2 / HuBERT / WavLM**：通用 SSL 语音预训练模型，可减少跨语言下降。
- **领域自适应**（DANN / CAAM）：+10-15pt 提升。

> 以上均为后续迭代方向，当前版本聚焦 CNN+LSTM baseline。

---

## 备选数据集

以下数据集不自动下载，仅作参考。

| 数据集 | 说明 | 为何不采用 |
|--------|------|-----------|
| SST-2 | 67k 二元情感 | 已剔除：二元 pos/neg 映射成 happy/sad 是噪声 |
| RAVDESS Song | 1,012 唱歌情感 | 已剔除：唱歌情感≠说话情感 |
| MELD | 9,989+1,109+2,610 | 已剔除：FLAC 转换超时，CSV 镜像截断 |
| IEMOCAP | 10,039 turns, 12h, 6 类, 英文 | 需 Google Form 申请 + 3-5 天审批 + 学术邮箱；仍为 acted 数据；不填补中文缺口。**可选/可跳过** |
| CHEAVD 2.0 | 238 说话人, 8 类, 中文自然 | 影视采集，含背景噪声；CASIA/ESD 已够用 |
| M3ED | 626 说话人, 7 类, 中文 TV | ACL 2022，百度云分发，流程繁琐 |
| EmotionTalk | 19 说话人, 7 类, 中文 | HF gated（需 token + 协议），无法自动下载 |
| EmoDialogCN | 119 说话人, 18 类, 400h | 2025 新发布，类别过多需重映射 |
| MER2023/2024 | 5,030 标注 + 115k 未标注 | HF gated + EULA，仅学术 |
| CH-SIMS | 5 类 sentiment | 标签为 sentiment（极性）而非离散情感，无法替代分类 SER |
| AISHELL-3 | 中文 TTS 语料 | 无情感标签，不可用于 SER |

> 注：IEMOCAP 虽为经典 SER 数据集，但申请流程长且许可证限制严格（不得再分发、不得商用、需咨询 USC 后方可公开报告性能）。本项目数据规模已够用，暂不纳入。

---

## 下载命令

以下命令供 `scripts/download_data.py` 参考，所有 URL 已验证。

```bash
# 0. 国内网络准备 (PowerShell)
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 1. GoEmotions (通过 HF datasets, simplified 配置去 PII)
python -c "from datasets import load_dataset; load_dataset('google-research-datasets/go_emotions', 'simplified')"

# 2. RAVDESS Speech (HF 镜像 parquet, 2 shards)
curl.exe -L -o ravdess_0.parquet "https://hf-mirror.com/api/datasets/xbgoose/ravdess/parquet/default/train/0.parquet"
curl.exe -L -o ravdess_1.parquet "https://hf-mirror.com/api/datasets/xbgoose/ravdess/parquet/default/train/1.parquet"

# 3. EMO-DB (HF 镜像 parquet, 1 shard, 304 rows)
curl.exe -L -o emodb_0.parquet "https://hf-mirror.com/api/datasets/confit/emodb-parquet/parquet/default/train/0.parquet"

# 4. CASIA (HF 镜像 parquet, 12 shards, ~71MB)
0..11 | ForEach-Object { curl.exe -L -o "casia_$_.parquet" "https://hf-mirror.com/api/datasets/BillyLin/CASIA_speech_emotion_recognition_preload/parquet/default/train/$_.parquet" }

# 5. ESD (HF 镜像 parquet, 7 shards, ~3.3GB)
0..6 | ForEach-Object { curl.exe -L -o "esd_$_.parquet" "https://hf-mirror.com/api/datasets/jspaulsen/esd/parquet/default/train/$_.parquet" }
```

> 一键下载：`python scripts/download_data.py --dataset all --target ./data/raw`

---

## 隐私与伦理

### 数据敏感性

本项目的目标用户是大学生群体，关注点是心理健康。这涉及以下敏感问题：

1. **情感数据属于个人敏感信息**。即使用户同意，也必须最小化数据收集范围。
2. **心理健康筛查结果（如 PHQ-9、GAD-7）高度敏感**。这些数据仅在用户测试阶段收集，且严格匿名化。

### 本项目的数据策略

- **训练数据**：全部使用公开数据集（GoEmotions、RAVDESS、EMO-DB、CASIA、ESD），不涉及真实用户数据
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
| GoEmotions | Apache-2.0 | 可商用，需归属。simplified 配置已去除 Reddit 用户名（PII） |
| RAVDESS | CC BY-NC-SA 4.0 | 非商用，需归属，需引用 Livingstone & Russo 2018 |
| EMO-DB | 研究免费（无正式许可证） | 需引用 Burkhardt et al. 2005 |
| CASIA | 研究免费（HF 镜像）；商用需购买 | 引用 CLDC-SPC-2005-010 |
| ESD | 研究用途，需签署协议 | 需引用 Zhou et al. 2021 |

> GoEmotions 注意事项：原始数据包含 Reddit 用户名（个人身份信息）。项目使用 `simplified` 配置加载，该配置已去除 PII。

---

## 参考文献

```bibtex
@inproceedings{demszky2020goemotions,
  title={GoEmotions: A Dataset of Fine-Grained Emotions},
  author={Demszky, Dorottya and Movshovitz-Attias, Dana and Kim, Jeongwoo and others},
  booktitle={ACL},
  year={2020}
}

@article{livingstone2018ravdess,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R. and Russo, Frank A.},
  journal={PLoS ONE},
  year={2018}
}

@inproceedings{burkhardt2005emodb,
  title={A Database of German Emotional Speech},
  author={Burkhardt, Felix and Paeschke, Astrid and Rolfes, Miriam and Sendlmeier, Walter F. and Weiss, Benjamin},
  booktitle={Interspeech},
  year={2005}
}

@misc{casia,
  title={CASIA Chinese Emotional Speech Corpus},
  author={Chinese Academy of Sciences, Institute of Automation},
  howpublished={CLDC-SPC-2005-010},
  year={2005}
}

@inproceedings{zhou2021esd,
  title={Emotional Speech Dataset (ESD): A Multilingual Emotional Speech Corpus},
  author={Zhou, Kun and Zhao, Xinyu and Ouyang, Shan and others},
  booktitle={ACL},
  year={2021}
}
```
