# 数据集

> 本文档说明 Emotion Pet 使用的情感识别数据集、标签映射、预处理流程、获取方式与伦理规范。

---

## 数据集总览

项目最终采用 **MELD（Multimodal EmotionLines Dataset）** 作为唯一训练数据集。

MELD 是 CMU Poria 团队于 ACL 2019 发布的多模态对话情感数据集，基于美剧《老友记》(Friends)。每一句台词同时提供**文本转录 + 语音音频 + 视频画面**，天然 text+audio 对齐，适合多模态情感识别。

| 属性 | 值 |
|------|-----|
| 模态 | 文本 + 语音（+ 视频，本项目不使用） |
| 语言 | 英文 |
| 原始规模 | 13,708 条（train 9,989 / dev 1,109 / test 2,610） |
| 本项目使用 | 13,706 条（对齐音频后，缺失 2 个音频文件） |
| 标签 | 7 类：neutral / joy / sadness / anger / surprise / fear / disgust |
| 音频格式 | FLAC（从原视频音轨提取） |
| 许可证 | CC BY-NC-SA 4.0（非商用） |

### 选型过程：从多数据集收敛到 MELD

项目早期尝试过「多数据集拼接」方案（GoEmotions 文本 + RAVDESS/CASIA/EMO-DB/ESD 语音，约 7,400 行），实践中暴露出两个难以克服的问题：

1. **文本与语音不对齐**：GoEmotions 是纯文本，RAVDESS/CASIA/EMO-DB 是纯语音，ESD 虽提供中文配对但子采样后类别缺失。多模态融合训练时，文本与语音来自**不同样本**，模型无法学习真正的跨模态对应关系。
2. **语言混杂**：中英德三语混合，跨语言泛化损失大，且中文数据量不足以支撑目标用户群体。

MELD 一次性解决上述问题：

- **text+audio 严格对齐**（同一句台词）；
- **单一语言**（英文），BERT uncased 无语言不匹配问题；
- **官方 train/dev/test 划分**，含真实分布的不平衡（更贴近真实场景）；
- **7 类标签天然对齐**，无需复杂映射。

### 类别分布（不平衡，代表真实对话分布）

| 类别 | train | dev | test | 合计 |
|------|-------|-----|------|------|
| neutral | 4,710 | 470 | 1,256 | 6,434 |
| joy → happy | 1,743 | 163 | 402 | 2,308 |
| surprise | 1,205 | 150 | 281 | 1,636 |
| anger | 1,109 | 153 | 345 | 1,607 |
| sadness → sad | 683 | 111 | 208 | 1,002 |
| disgust | 271 | 22 | 68 | 361 |
| fear | 268 | 40 | 50 | 358 |

> neutral 约占 47%，fear/disgust 各约 2.6%，相差约 17 倍。
> 为缓解类别不平衡，训练时采用**反频率类别加权损失**（fear/disgust 高权重、neutral 低权重），而非下采样丢弃数据。

---

## 标签体系

MELD 原始 7 类标签仅需两处重命名即可对齐 config.yaml 的 `emotion_labels`：

| MELD 原始 | 项目标签 |
|-----------|----------|
| joy | happy |
| sadness | sad |
| neutral | neutral |
| anger | angry |
| surprise | surprise |
| fear | fear |
| disgust | disgust |

---

## 预处理流程

### 文本数据

```
原始英文台词
    │
    ▼
BERT Tokenizer（bert-base-uncased）
    │
    ▼
截断/填充：max_length = 64
    │
    ▼
输出：input_ids, attention_mask
```

### 语音数据

```
原始音频（.flac）
    │
    ▼
重采样：16kHz 单声道（librosa.load）
    │
    ▼
特征提取：MFCC 40 维（帧长 25ms，帧移 10ms）
    │
    ▼
一阶差分：delta 40 维 → 拼接为 80 维
    │
    ▼
标准化：CMVN（均值方差归一化，沿时间维）
    │
    ▼
对齐/截断：固定 300 帧（超长截断、短音频边缘填充）
    │
    ▼
输出：feature tensor (300, 80)
```

### 加速：MFCC 预提取缓存

- `scripts/precompute_mfcc.py`：多线程预提取所有唯一音频的 MFCC，写入 `data/raw/mfcc_cache.pkl`（约 427MB）。
- `dataset.py` 的 `__getitem__` 命中缓存后直接读内存，训练数据加载从 20+ 分钟降到秒级。

---

## 数据划分

MELD 官方提供 train/dev/test 划分。本项目训练时对 train 集再做 8:1:1 分层切分（`split_dataframe`，seed=42），评估在独立 test 集上进行，避免数据泄露。

---

## 跨语言泛化问题与对策

### 问题

MELD 为英文数据集，而目标用户是中文大学生。外语数据训练的语音情感模型用于中文用户时，准确率会下降。

已有研究结论：

- EMO-DB → CASIA 零迁移（德→中）：仅 24.8% UAR（同语言测试通常 85-95%）
- 英语→英语跨语料库：仍下降 10-40%（Braunschweiler 2022：matched 89.8% vs mismatched 65.3%）

下降原因：

1. **韵律是语言特定的**。arousal（唤醒度）可跨语言迁移，valence（效价）下降严重
2. **普通话声调干扰**。音高同时承载词汇声调（四声）和情感韵律，模型容易混淆
3. **acted vs natural 不匹配**。实验室录制与自然表达差距大

### 本项目的处境与对策

- **文本分支**：BERT uncased 对英文文本建模充分，文本情感识别不受语言迁移影响；
- **语音分支**：英文语音训练的 CNN+LSTM 对中文语音存在跨语言下降，这是已知局限；
- **实验观察**：当前消融结果（语音分支 test acc 0.0773，低于随机基线 0.14）表明语音分支本身尚未学好，跨语言问题叠加从零训练的欠拟合，是下一步改进方向。

### 局限性承认

本项目作为本科实践项目，接受这一局限：

- 训练数据为英文（MELD），测试用户为中文大学生，预期中文语音识别准确率低于英文
- 这一差距本身就是实验报告的重要观察点

---

## 模型架构

当前架构：BERT（text）+ CNN+LSTM（audio）→ 特征层拼接融合 → 7 类。

### 文本分支

- 使用 `bert-base-uncased`（与 MELD 英文数据匹配）
- 骨干冻结（仅训练分类头），避免 1 万级样本上的过拟合
- 输入：text → tokenize → BERT → [CLS] pooler（768 维）

### 语音分支

- **CNN+LSTM 从零训练**（不从外部加载预训练 SER 模型）
- 输入：80 维 MFCC（40 静态 + 40 delta）→ Conv1d（64→128）→ BiLSTM（2 层，hidden=128）→ FC → 128 维特征
- 设计理由：项目核心目标之一是**学习并展示 CNN+LSTM 在 SER 任务上的从零训练能力**，直接使用 emotion2vec+ 等预训练模型相当于"调用 API"，失去教育价值

### 融合层

- 文本特征（768 维）+ 语音特征（128 维）在特征维拼接为 896 维 → MLP（896 → 256 → 7）→ 情感分布
- 保留 `text_fc` / `audio_fc` 单模态分类头，用于消融实验

### 未来改进方向（不在当前实现范围内）

- **语音分支退化**（当前 test acc 0.0773 < 随机基线 0.14）是首要改进点，方向：
  - **wav2vec2 / HuBERT / WavLM**：通用 SSL 语音预训练模型，可减少跨语言下降；
  - **emotion2vec+**（阿里巴巴达摩院，ACL 2024）：中文原生 SER 基础模型，9 类输出与项目 7 类兼容；
  - **领域自适应**（DANN / CAAM）：+10-15pt 提升。
- **fear/disgust 少数类**：可尝试 focal loss 或数据增强。

---

## 备选数据集

以下数据集经调研后未采用，仅作参考。

| 数据集 | 说明 | 为何不采用 |
|--------|------|-----------|
| GoEmotions | 58k 英文文本，27 类 | 纯文本，无配对音频，无法做多模态对齐 |
| RAVDESS Speech | 1,440 英文语音，8 类 | 纯语音，无文本转录 |
| CASIA | 1,200 中文语音，6 类 | 纯语音，无文本；无 disgust 类 |
| EMO-DB | 535 德文语音，7 类 | 纯语音；德文跨语言下降大 |
| ESD | 中英双语语音+转录 | 需签协议；子采样后类别缺失 |
| IEMOCAP | 10,039 条英文多模态 | 需申请 + 审批；仍为 acted 数据 |
| SST-2 | 67k 二元情感 | 二元 pos/neg 映射成 happy/sad 是噪声 |
| CHEAVD 2.0 / M3ED / EmoDialogCN | 中文影视/自然 | 获取流程繁琐，或类别过多需重映射 |

> 早期「5 数据集拼接」方案（GoEmotions + RAVDESS + CASIA + EMO-DB + ESD，约 7,400 行）因文本语音不对齐、语言混杂而被 MELD 取代，记录于此作为选型过程的一部分。

---

## 获取方式

MELD 的获取需要一定流程（非一键下载）：

1. **CSV 标签**：官方 GitHub `declare-lab/MELD` 提供 train/dev/test 的 CSV（含文本转录、情绪标签、说话人、剧集信息）。
2. **音频**：需从《老友记》视频提取（视频受版权保护，需自行获取；每句台词对应一段音轨，转存为 FLAC/WAV）。
3. **本项目已准备好预处理数据**：`data/raw/meld_csv/{meld_train,meld_dev,meld_test}.csv` + `data/raw/meld_audio/{train,dev,test}/`（13,847 个 FLAC）。复现时只需将 `labels.csv`（13,706 行，text/audio_path/label 三列）与 `meld_audio/` 放到 `data/raw/` 下。

> HuggingFace 上存在 MELD 的镜像仓库（含音频），可作为替代获取渠道，但不同镜像的字段与完整性不一，本项目未采用镜像加载。

---

## 隐私与伦理

### 数据敏感性

本项目的目标用户是大学生群体，关注点是心理健康。这涉及以下敏感问题：

1. **情感数据属于个人敏感信息**。即使用户同意，也必须最小化数据收集范围。
2. **心理健康筛查结果（如 PHQ-9、GAD-7）高度敏感**。这些数据仅在用户测试阶段收集，且严格匿名化。

### 本项目的数据策略

- **训练数据**：使用公开数据集 MELD（CC BY-NC-SA 4.0，非商用），不涉及真实用户数据
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
| MELD | CC BY-NC-SA 4.0 | 非商用，需归属，需引用 Poria et al. 2019 |

---

## 参考文献

```bibtex
@inproceedings{poria2019meld,
  title={MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations},
  author={Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2019}
}
```
