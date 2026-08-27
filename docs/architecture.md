# 系统架构

> 本文档对应论文 4.1 节「系统总体架构设计」。

---

## 概览

Emotion Pet 采用分层模块化架构。五个核心模块各司其职，通过明确定义的接口通信：

```
┌──────────────────────────────────────────────────────────────────┐
│                        desktop_pet (GUI 层)                      │
│   PyQt5 透明窗口 · 帧动画 · 表情渲染 · 对话气泡                    │
└──────────────┬───────────────────────────────┬───────────────────┘
               │ action_request                │ user_interaction
               ▼                               │
┌──────────────────────────┐    ┌──────────────┴───────────────────┐
│   rl_agent (决策层)       │    │   emotion_recognition (感知层)    │
│   DQN · ε-greedy         │◄───│   BERT · CNN+LSTM · Fusion       │
│   经验回放 · 目标网络     │    │   文本 + 语音 → 情感标签           │
└──────────────┬───────────┘    └──────────────────────────────────┘
               │ state_update
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    growth_system (状态层)                         │
│   PetStats · LevelSystem · BehaviorEngine · SQLAlchemy 持久化     │
└──────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    data (数据层)                                   │
│   raw/ · processed/ · checkpoints/ · logs/                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 模块详解

### 1. 情感识别模块 (`emotion_recognition/`)

**职责**：接收文本和语音输入，输出 7 类情感概率分布。

| 子模块 | 文件 | 说明 |
|--------|------|------|
| 文本分支 | `models/bert_text.py` | 加载预训练 BERT，取 [CLS] 向量经全连接层输出 7 类情感概率 |
| 语音分支 | `models/cnn_lstm_audio.py` | Mel 频谱（128 维）→ 2D-CNN 提取局部特征 → BiLSTM 捕获时序依赖 → FC 分类。**从零训练**，不加载外部 SER 预训练权重，保留完整学习链路以体现教育价值 |
| 融合层 | `models/fusion.py` | 对文本和语音分支的隐层表示做注意力加权拼接，输出最终情感分布 |
| 数据加载 | `dataset.py` | 统一读取 text/audio 数据，做 tokenize、特征提取、标签对齐 |
| 训练 | `train.py` | 训练入口，支持混合精度、梯度累积、TensorBoard 日志 |
| 评估 | `evaluate.py` | 在测试集上计算 Accuracy、F1-macro、混淆矩阵 |

**输出格式**：

```python
{
    "emotion_label": "sad",          # 7 类之一
    "confidence": 0.83,              # 概率值
    "distribution": {                # 完整分布
        "happy": 0.02, "sad": 0.83, "angry": 0.01,
        "fear": 0.05, "surprise": 0.03, "disgust": 0.01, "neutral": 0.05
    },
    "modality": "text"               # 或 "audio" / "multimodal"
}
```

**7 类情感标签**（对齐 config.yaml）：

happy, sad, angry, fear, surprise, disgust, neutral

---

### 2. 强化学习智能体 (`rl_agent/`)

**职责**：根据当前情感状态和宠物属性，选择最优行为动作。

| 子模块 | 文件 | 说明 |
|--------|------|------|
| DQN 网络 | `dqn.py` | 在线网络 + 目标网络，ε-greedy 探索策略 |
| 经验回放 | `replay_buffer.py` | 存储 (s, a, r, s') 元组，均匀采样 mini-batch |
| 交互环境 | `environment.py` | 封装宠物与用户的交互循环，定义状态/动作/奖励 |
| 训练 | `train.py` | DQN 训练入口，支持 TensorBoard 记录奖励曲线 |

**状态空间 (State)**，共 12 维：

```
state = [
    emotion_one_hot,       # 7 维，当前情感标签 one-hot
    history,               # 3 维，最近 3 次模拟用户反馈，归一化到 [-1, 1]
    time_features,         # 2 维，[当天交互时刻(0~1), 会话进度(步数/最大步数)]
]
```

**动作空间 (Action)**，共 5 种反馈策略：

```
actions = [
    "comfort",        # 安慰
    "encourage",      # 鼓励
    "distract",       # 转移注意力
    "listen",         # 倾听
    "accompany",      # 陪伴
]
```

**奖励设计**：

```
reward = 1.0 × 情绪改善度 + 0.5 × 模拟用户反馈
```

- **情绪改善度**：转移前后情绪效价 (valence) 之差，为主信号，驱动智能体将用户情绪导向更积极方向；
- **模拟用户反馈**：由情绪改善度叠加高斯噪声后裁剪到 [-1, 1]，为辅助信号，模拟真实用户打分的主观性。

---

### 3. 成长系统 (`growth_system/`)

**职责**：维护宠物的长期状态，驱动成长曲线和行为解锁。

| 子模块 | 文件 | 说明 |
|--------|------|------|
| 属性管理 | `stats.py` | PetStats 类：心情、好感度、精力、经验值，带时间衰减 |
| 等级系统 | `level.py` | 经验值→等级映射，成长曲线定义，解锁条件判定 |
| 行为引擎 | `behaviors.py` | 根据当前属性组合，过滤可用动作集合 |

**属性衰减机制**：

宠物属性随时间自然变化，模拟真实宠物体验：

- 心情：每小时自然衰减，与用户正向交互后回升
- 好感度：长期不互动缓慢下降，互动后上升
- 精力：交互消耗精力，闲置时缓慢恢复
- 经验值：只增不减，由每次有效交互累积

**持久化**：

所有属性通过 SQLAlchemy 写入本地 SQLite 数据库。用户关闭程序后再次打开，宠物状态保留。

---

### 4. 桌面 GUI 模块 (`desktop_pet/`)

**职责**：将系统行为渲染为用户可见的桌面宠物窗口。

| 子模块 | 文件 | 说明 |
|--------|------|------|
| 透明窗口 | `pet_window.py` | 无边框、透明背景、置顶的 PyQt5 窗口 |
| 帧动画 | `pet_animation.py` | 管理精灵图序列帧，按行为切换动画集 |
| 表情映射 | `emotion_display.py` | emotion_label → 对应表情帧/动画 |
| 入口 | `main.py` | 启动 GUI 事件循环，初始化各模块 |

**窗口特性**：

- 透明背景（`Qt.WindowTransparentForInput` 可选）
- 始终置顶（`Qt.WindowStaysOnTopHint`）
- 可拖拽移动位置
- 右键菜单：设置 / 退出
- 对话气泡：QLabel 浮层显示回应文本

---

### 5. 数据管理 (`data/`)

**职责**：管理训练数据、模型权重、日志。

```
data/
├── raw/              # 原始数据集（不提交 git）
│   ├── text/         #   文本情感数据（GoEmotions 英文 + ESD 中文转录）
│   ├── audio/        #   语音情感数据（RAVDESS/CASIA/EMO-DB/ESD，16kHz WAV）
│   └── labels.csv    #   统一标签映射（~7,444 行，5 个数据集）
├── processed/        # 预处理后的张量/特征文件
├── README.md
│
checkpoints/          # 模型权重 (.pt)
logs/                 # TensorBoard 事件文件
```

> 数据集规模说明：GoEmotions 3,000（英文文本，子采样）+ ESD 1,500（中文文本+语音配对，子采样）+ RAVDESS 1,440（英文语音）+ CASIA 1,200（中文语音）+ EMO-DB 304（德文语音）= 约 7,444 行。详见 `docs/dataset.md`。

---

## 模块间接口约定

### 数据流方向

```
用户输入
    │
    ├──[文本]──▶ emotion_recognition.bert_text
    │                    │
    ├──[语音]──▶ emotion_recognition.cnn_lstm_audio
    │                    │
    │                    ▼
    │            emotion_recognition.fusion
    │                    │
    │                    │ emotion_result (dict)
    │                    ▼
    │            rl_agent.environment.step(emotion)
    │                    │
    │                    │ action (str)
    │                    ▼
    │            growth_system.behaviors.execute(action)
    │                    │
    │                    │ state_update
    │                    ▼
    │            desktop_pet.render(action, emotion)
    │
    └──[鼠标]──▶ desktop_pet 直接处理
```

### 关键接口定义

**emotion_recognition → rl_agent**：

```python
def observe(emotion_result: dict) -> None:
    """接收情感识别结果，更新环境状态。"""
    # emotion_result 格式见情感识别模块输出
```

**rl_agent → growth_system**：

```python
def execute_action(action: str) -> dict:
    """执行动作，返回状态变化。"""
    # 返回 {"mood_delta": +0.1, "affection_delta": +0.05, ...}
```

**growth_system → desktop_pet**：

```python
def get_render_state() -> dict:
    """返回当前渲染所需的宠物状态。"""
    # 返回 {"animation": "happy_dance", "level": 3, "stats": {...}}
```

---

## 设计决策

1. **为什么用 DQN 而不是规则引擎？**
   规则引擎无法从用户反馈中自适应学习。DQN 让宠物随使用时间逐渐优化行为策略，提升长期互动体验。

2. **为什么多模态而不只用文本？**
   大学生使用场景多样：打字聊天时走文本，语音倾诉时走语音。双通道覆盖更多交互场景，也为心理健康监测提供更丰富信号。

3. **为什么成长系统需要时间衰减？**
   没有衰减的宠物不需要持续关心，互动黏性会快速下降。衰减机制模拟真实宠物体验，让用户感到「宠物需要我」。

4. **为什么本地部署？**
   心理健康数据高度敏感。所有推理和存储均在本地完成，不依赖云端 API，消除数据泄露风险。

5. **为什么 CNN+LSTM 从零训练而不直接用 emotion2vec+？**
   本项目为本科实践项目，核心目标之一是学习并展示 CNN+LSTM 在 SER 任务上的完整训练流程。直接加载预训练 SER 模型（如 emotion2vec+）等同于调用 API，失去教育价值。当前版本先跑 CNN+LSTM baseline，观察收敛情况；若效果不理想，再在"未来改进"阶段引入 emotion2vec+ 作为特征提取器或 fine-tune 起点。
