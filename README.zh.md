# Emotion Pet 🐾

**基于情感交互的智能桌面宠物系统** — 多模态情感识别 × DQN 强化学习 × 成长系统

---

> 🏆 **桌面智能宠物最佳学术奖** — 2025 寒假社会实践
>
> 论文：《基于情感交互的智能桌面宠物设计与应用——关注大学生心理健康的社会实践研究》

---

## 项目简介

大学生心理健康问题日益受到关注。本项目设计并实现了一个能够感知用户情绪、主动提供陪伴反馈的智能桌面宠物。系统通过文本和语音双通道实时识别用户情感状态，由 DQN 强化学习智能体决定宠物的回应行为，再结合成长系统维持长期互动黏性。

整个系统跑在一台普通笔记本上，PyQt5 做透明置顶窗口，宠物就蹲在你的任务栏旁边。

---

## 技术栈

| 领域 | 技术 | 用途 |
|------|------|------|
| 深度学习框架 | PyTorch ≥ 2.0 | 模型训练与推理 |
| 预训练语言模型 | HuggingFace Transformers (BERT) | 文本情感编码 |
| 音频处理 | librosa, soundfile | 语音特征提取 (MFCC/Mel) |
| 强化学习 | OpenAI Gym, DQN | 宠物行为决策 |
| 桌面 GUI | PyQt5 | 透明置顶窗口、动画渲染 |
| 数据持久化 | SQLAlchemy | 用户交互日志、宠物状态存储 |
| 实验管理 | TensorBoard | 训练曲线可视化 |
| 数据处理 | pandas, scikit-learn | 数据加载、评估指标计算 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户 (大学生)                           │
│              文本输入 / 语音输入 / 鼠标交互                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ① 多模态情感识别模块                             │
│                                                             │
│   文本分支: BERT → [CLS] → FC → 7 类情感概率                 │
│   语音分支: MFCC → CNN → LSTM → FC → 7 类情感概率            │
│   融合层:   注意力加权拼接 → Softmax → emotion_label          │
└──────────────────────┬──────────────────────────────────────┘
                       │ emotion_label + confidence
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ② DQN 强化学习智能体                             │
│                                                             │
│   State:  [emotion, pet_stats, time, interaction_history]   │
│   Action: [安慰, 撒娇, 跳舞, 独处, 提醒休息, 播放音乐, ...]   │
│   Reward: 用户正向反馈 (+) / 负向反馈 (-) / 好感度变化        │
│   Network: DQN + Experience Replay + Target Network          │
└──────────────────────┬──────────────────────────────────────┘
                       │ selected_action
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ③ 桌面宠物 GUI + 行为执行                        │
│                                                             │
│   PyQt5 透明窗口 → 帧动画切换 → 表情/动作渲染                 │
│   对话模板引擎 → 生成回应文本气泡                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ action_result
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ④ 成长系统                                       │
│                                                             │
│   PetStats: 心情 / 好感度 / 精力 / 经验值                     │
│   LevelSystem: 等级提升 → 解锁新行为/新外观                    │
│   BehaviorEngine: 状态驱动的行为策略调整                       │
│   SQLAlchemy 持久化 → 关闭后状态保留                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心特性

- 🧠 **多模态情感识别** — BERT 处理文本，CNN+LSTM 处理语音，注意力机制融合双通道
- 🎮 **DQN 行为决策** — 宠物根据用户情绪和自身状态自主选择回应方式，越用越懂你
- 🌱 **成长系统** — 心情、好感度、精力、经验值四维属性，等级解锁新行为
- 🖥️ **桌面透明窗口** — PyQt5 置顶渲染，宠物蹲在任务栏旁，不挡工作
- 💬 **情感对话** — 基于情绪标签的模板对话引擎，回应贴合当下情境
- 📊 **可复现实验** — 固定随机种子，TensorBoard 记录训练全流程
- 🔒 **隐私优先** — 所有数据本地存储，不上传云端

---

## 项目结构

```
emotion-pet/
├── README.md                 # 英文文档
├── README.zh.md              # 本文件（中文文档）
├── config.yaml               # 全局配置（随机种子、设备、情绪标签）
├── requirements.txt          # Python 依赖
├── pyproject.toml            # 包元数据
├── LICENSE                   # MIT
│
├── emotion_recognition/      # 多模态情感识别模块
│   ├── models/
│   │   ├── bert_text.py      #   BERT 文本分支
│   │   ├── cnn_lstm_audio.py #   CNN+LSTM 语音分支
│   │   └── fusion.py         #   多模态融合头
│   ├── dataset.py            #   数据加载
│   ├── train.py              #   训练入口
│   └── evaluate.py           #   评估入口
│
├── rl_agent/                 # DQN 强化学习模块
│   ├── dqn.py                #   DQN 网络 + 目标网络
│   ├── replay_buffer.py      #   经验回放池
│   ├── environment.py        #   宠物交互环境
│   └── train.py              #   训练入口
│
├── growth_system/            # 宠物成长系统
│   ├── stats.py              #   属性管理
│   ├── level.py              #   等级与成长曲线
│   └── behaviors.py          #   状态→行为策略
│
├── desktop_pet/              # PyQt5 桌面 GUI
│   ├── assets/               #   动画帧资源
│   ├── pet_window.py         #   透明置顶窗口
│   ├── pet_animation.py      #   帧动画管理
│   ├── emotion_display.py    #   情绪→表情映射
│   └── main.py               #   GUI 入口
│
├── data/                     # 数据集（不提交到 git）
│   ├── raw/                  #   原始数据
│   └── processed/            #   预处理后数据
│
├── checkpoints/              # 模型权重
├── logs/                     # TensorBoard 日志
├── scripts/
│   ├── download_data.py      # 数据下载脚本
│   └── run_all.sh            # 一键训练流水线
├── tests/                    # 单元测试
└── docs/
    ├── architecture.md       # 架构详解
    ├── dataset.md            # 数据集说明
    └── experiments.md        # 实验记录
```

---

## 快速开始

### 环境要求

- Python ≥ 3.9
- CUDA 11.8+（GPU 训练，可选；CPU 也能跑推理）

### 安装

```bash
# 克隆仓库
git clone <repo-url> emotion-pet
cd emotion-pet

# 创建虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量（可选）
cp .env.example .env
# 编辑 .env，设置 HF_HOME、DEVICE 等
```

### 下载数据

```bash
# 下载并预处理公开情感数据集
python scripts/download_data.py --dataset all --target ./data/raw

# 详见 docs/dataset.md 了解数据集选择和手动下载方式
```

### 训练模型

```bash
# 方式一：一键流水线
bash scripts/run_all.sh

# 方式二：分步训练

# 1) 训练情感识别模型
python -m emotion_recognition.train --config config.yaml --epochs 20

# 2) 训练 DQN 智能体
python -m rl_agent.train --config config.yaml --episodes 1000

# 3) 评估模型
python -m emotion_recognition.evaluate --checkpoint checkpoints/emotion_best.pt
```

### 启动桌面宠物

```bash
python -m desktop_pet.main
```

宠物窗口会以透明背景出现在桌面上。对它说话、或者用麦克风输入，它会识别你的情绪并做出回应。

---

## 实验结果

> 完整实验数据详见 [docs/experiments.md](docs/experiments.md)。

| 实验项 | 指标 | 结果 |
|--------|------|------|
| 文本情感识别 (BERT) | Accuracy / F1-macro | 待补充 |
| 语音情感识别 (CNN+LSTM) | Accuracy / F1-macro | 待补充 |
| 多模态融合 | Accuracy / F1-macro | 待补充 |
| DQN 训练 | 平均奖励收敛曲线 | 待补充 |
| 用户测试 (N=300, 3 个月) | PHQ-9 / GAD-7 变化 | 待补充 |

实验设置、超参数、复现步骤均记录在 [docs/experiments.md](docs/experiments.md)。

---

## 论文引用

```bibtex
@misc{emotion-pet-2025,
  title  = {基于情感交互的智能桌面宠物设计与应用——关注大学生心理健康的社会实践研究},
  author = {Emotion Pet Research Group},
  year   = {2025},
  note   = {桌面智能宠物最佳学术奖 — 2025 寒假社会实践},
}
```

---

## 路线图

- [x] 项目骨架搭建、模块接口定义
- [x] 多模态情感识别模型实现（BERT + CNN+LSTM + Fusion）
- [x] DQN 强化学习智能体实现
- [x] 成长系统与属性管理
- [x] PyQt5 桌面窗口与动画框架
- [ ] 完成模型训练，补充实验数据
- [ ] 执行 300 人对照用户测试
- [ ] 中文语音情感数据增强
- [ ] 宠物外观自定义系统
- [ ] 打包为 Windows 安装程序

---

## License

[MIT](LICENSE) © 2025 Emotion Pet Research Group
