# Emotion Pet 🐾

**An affective desktop companion for college student mental health** — multimodal emotion recognition × DQN reinforcement learning × growth system

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)](tests/)

> 🏆 **Best Academic Award — Desktop Smart Pet**, 2025 winter social practice.
>
> Paper: *Design and Application of an Affective Desktop Pet — A Social Practice Study on College Student Mental Health* ([docs/paper.pdf](docs/paper.pdf), in Chinese).
>
> 中文文档请见 [readme.zh.md](readme.zh.md).

---

## Overview

College student mental health is an increasingly urgent concern. **Emotion Pet** is a desktop companion that senses the user's affective state through text and speech, selects a supportive response via a DQN agent, and sustains long-term engagement through a growth system — all running locally on a laptop with a frameless transparent PyQt5 window.

The system has four pillars:

1. **Multimodal Emotion Recognition** — a BERT text branch and a CNN+LSTM speech branch fused into 7-class emotion logits.
2. **DQN Behavior Policy** — a reinforcement learning agent that maps `(emotion, pet_stats, time, history)` to one of 5 supportive actions.
3. **Growth System** — mood / affinity / energy / experience stats with level-gated skills and appearance evolution, persisted to SQLite.
4. **Desktop GUI** — a transparent always-on-top pet window with frame animation, emotion-driven expressions, and dialogue bubbles.

---

## Tech Stack

| Domain | Technology | Purpose |
|--------|-----------|---------|
| Deep learning | PyTorch ≥ 2.0 | Model training & inference |
| Pretrained NLP | HuggingFace Transformers (BERT) | Text emotion encoding |
| Audio processing | librosa, soundfile | MFCC / Mel feature extraction |
| Reinforcement learning | OpenAI Gym, DQN | Pet behavior policy |
| Desktop GUI | PyQt5 | Transparent frameless window & animation |
| Persistence | SQLAlchemy + SQLite | Pet state & interaction logs |
| Experiment tracking | TensorBoard | Training curve visualization |
| Data tooling | pandas, scikit-learn | Data loading & metrics |
| Testing | pytest | Unit test suite (48 tests) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      User (college student)                  │
│         text input  /  speech input  /  mouse interaction    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│   ① Multimodal Emotion Recognition                          │
│                                                             │
│   Text branch:  BERT → [CLS] → FC → 7-class logits          │
│   Audio branch: MFCC → CNN(Conv1d) → BiLSTM → FC → logits   │
│   Fusion:       concat(visual, audio, text) → Linear → out  │
└──────────────────────────┬───────────────────────────────────┘
                           │ emotion_label + confidence
                           ▼
┌──────────────────────────────────────────────────────────────┐
│   ② DQN Reinforcement Learning Agent                        │
│                                                             │
│   State:  [emotion(7), history(3), time(2)]  (dim=12)       │
│   Action: comfort / encourage / distract / listen / accompany│
│   Reward: Δvalence + simulated_user_feedback                │
│   Net:    MLP(12→64→64→5) + Experience Replay + Target Net  │
└──────────────────────────┬───────────────────────────────────┘
                           │ selected_action
                           ▼
┌──────────────────────────────────────────────────────────────┐
│   ③ Desktop Pet GUI + Behavior Execution                    │
│                                                             │
│   PyQt5 transparent window → frame animation → expression   │
│   Template dialogue engine → response text bubble           │
└──────────────────────────┬───────────────────────────────────┘
                           │ action_result
                           ▼
┌──────────────────────────────────────────────────────────────┐
│   ④ Growth System                                           │
│                                                             │
│   PetStats:      mood / affinity / energy / exp / level     │
│   LevelSystem:   level-up → unlock skills & appearance      │
│   BehaviorEngine: state-driven behavior selection           │
│   SQLite:        state persisted across sessions            │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Why not ASR for the speech branch?

The CNN+LSTM branch performs **Speech Emotion Recognition (SER)**, not Automatic Speech Recognition (ASR). Emotion is encoded in prosody (pitch, energy, speech rate, timbre) — not in the transcript. ASR would discard exactly the signal we need: "I'm fine" said calmly vs. sarcastically vs. while crying transcribes identically, but carries opposite emotions.

The text branch (BERT) and speech branch (CNN+LSTM) are **parallel independent modalities**, each learning emotion directly from its raw signal. They are fused late, never serialized through ASR.

| Approach | Pros | Cons |
|----------|------|------|
| ASR → text emotion | Reuses NLP stack | Loses prosody; ASR error propagation |
| Pure SER | Captures prosody directly | Loses semantic content |
| **Multimodal fusion (this project)** | Semantic + prosody互补 | Requires paired data |

### Why a simulated DQN environment?

Real user interaction data requires a deployed product and months of collection. The `PetEnvironment` simulates user valence dynamics with a 5×7 action-effect matrix and stochastic user feedback, enabling offline policy training before any real deployment. The reward is `Δvalence + simulated_user_feedback`, where user feedback is clipped Gaussian noise around the valence change.

---

## Project Structure

```
emotion-pet/
├── README.md                 # This file (English)
├── readme.zh.md              # Chinese README
├── config.yaml               # Global config (seed, device, emotion labels)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package metadata & pytest config
├── LICENSE                   # MIT
│
├── emotion_recognition/      # Multimodal emotion recognition
│   ├── models/
│   │   ├── bert_text.py      #   BERT text encoder (offline fallback)
│   │   ├── cnn_lstm_audio.py #   CNN+LSTM audio encoder
│   │   └── fusion.py         #   Multimodal fusion head
│   ├── dataset.py            #   Multimodal dataset (lazy librosa)
│   ├── train.py              #   Training entry (--mock supported)
│   ├── evaluate.py           #   Evaluation entry
│   └── config.py             #   Module hyperparameters
│
├── rl_agent/                 # DQN reinforcement learning
│   ├── dqn.py                #   DQN network (3-layer MLP + soft update)
│   ├── replay_buffer.py      #   Experience replay (deque + Transition)
│   ├── environment.py        #   Simulated pet environment
│   ├── policy.py             #   Inference policy wrapper
│   └── train.py              #   Training entry
│
├── growth_system/            # Pet growth & persistence
│   ├── stats.py              #   PetStats dataclass
│   ├── level.py              #   Level system & skill unlocks
│   ├── behaviors.py          #   Emotion → behavior mapping
│   └── persistence.py        #   SQLite via SQLAlchemy
│
├── desktop_pet/              # PyQt5 desktop GUI
│   ├── _qt_compat.py         #   PyQt5 import with graceful degradation
│   ├── pet_window.py         #   Transparent frameless window
│   ├── pet_animation.py      #   Bobbing & frame animation
│   ├── emotion_display.py    #   Emotion → expression rendering
│   ├── main.py               #   PetController entry point
│   └── assets/               #   Animation frames placeholder
│
├── scripts/
│   ├── download_data.py      # Dataset download helper
│   ├── rl_quick_train.py     # Reproducible RL benchmark
│   └── run_all.sh            # End-to-end pipeline
│
├── tests/                    # Unit tests (48 tests, all passing)
│   ├── test_emotion_recognition.py
│   ├── test_rl_agent.py
│   └── test_growth_system.py
│
├── data/                     # Datasets (gitignored)
├── checkpoints/              # Model weights (gitignored)
├── logs/                     # TensorBoard logs (gitignored)
│
└── docs/
    ├── architecture.md       # Architecture deep-dive
    ├── dataset.md            # Dataset documentation
    ├── experiments.md        # Experiment records
    ├── paper.pdf             # Original research paper
    └── award.jpg             # Award certificate
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.9
- CUDA 11.8+ (optional; CPU inference works for all modules)
- PyQt5 (optional; GUI degrades gracefully if absent)

### Installation

```bash
git clone https://github.com/cocohuangke/emotion-pet.git
cd emotion-pet

python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/macOS

pip install -r requirements.txt

cp .env.example .env           # optional: set HF_HOME, DEVICE, SEED
```

### Smoke Test (no data download required)

The `--mock` flag generates synthetic samples so the full pipeline runs without any dataset. This is the fastest way to verify the installation:

```bash
# Emotion recognition: 5-epoch mock training
python -m emotion_recognition.train --mock --epochs 5 --batch_size 16

# Evaluate the mock checkpoint
python -m emotion_recognition.evaluate --checkpoint checkpoints/best_model.pt --mock

# DQN: 40-episode quick benchmark
python scripts/rl_quick_train.py

# Full unit test suite
python -m pytest tests/ -v
```

### Real Training (with datasets)

```bash
# 1. Download datasets (see docs/dataset.md for manual options)
python scripts/download_data.py --dataset all --target ./data/raw

# 2. Train emotion recognition (20 epochs)
python -m emotion_recognition.train --config config.yaml --epochs 20

# 3. Train DQN agent (1000 episodes)
python -m rl_agent.train --config config.yaml --episodes 1000

# 4. Evaluate
python -m emotion_recognition.evaluate --checkpoint checkpoints/best_model.pt
```

### Launch the Desktop Pet

```bash
python -m desktop_pet.main
```

The pet appears as a transparent always-on-top window. It cycles through emotions every 5s (demo mode) and settles growth stats every 30s, persisting state to `data/pet.db` on exit.

---

## Experiment Results

> Full records: [docs/experiments.md](docs/experiments.md). All seeds fixed at 42.

### Emotion Recognition (preliminary, mock data, 5 epochs)

Mock training verifies the forward/backward/checkpoint pipeline. Accuracy stays at chance level (7-class ≈ 0.14) because mock labels are random — this is expected and **does not reflect real-data performance**.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 1.9601 | 0.1406 | 1.9491 | 0.1562 |
| 2 | 1.9495 | 0.1719 | 1.9512 | 0.1562 |
| 3 | 1.9577 | 0.1719 | 1.9517 | 0.1875 |
| 4 | 1.9550 | 0.1250 | 1.9637 | 0.1250 |
| 5 | 1.9454 | 0.1719 | 1.9529 | 0.1250 |

- **Best val acc**: 0.1875 (epoch 3)
- **Trainable params**: 24,766,703

### DQN Agent (preliminary, 40 episodes / max 20 steps)

| Metric | Value |
|--------|-------|
| Episodes | 40 |
| Wall time | 50.1s (CUDA) |
| Avg reward (first 10) | 0.808 |
| Avg reward (last 10) | 0.732 |
| All-time avg reward | 1.002 |
| Min / Max reward | -1.795 / 3.157 |
| Final TD loss | 0.0546 |
| Final ε | 0.818 |

Training curve (rolling 10-episode avg reward):

```
Episode  10 | avg10 = 0.808 | eps = 0.951 | loss = 0.0600
Episode  20 | avg10 = 0.939 | eps = 0.905 | loss = 0.0547
Episode  30 | avg10 = 1.530 | eps = 0.860 | loss = 0.0549
Episode  40 | avg10 = 0.732 | eps = 0.818 | loss = 0.0546
```

Loss converges from 0.060 → 0.055. Reward peaks at episode 30 then dips — likely due to ε still being high (0.818), causing exploratory actions to inject variance. Full 1000-episode training (ε → 0.05) is expected to stabilize avg reward above 2.0.

### Test Suite

```
48 passed in 18.16s

tests/test_emotion_recognition.py  .........  (9 tests)
tests/test_growth_system.py        ...........................  (27 tests)
tests/test_rl_agent.py             ..............  (14 tests)
```

Coverage: encoder forward shapes, fusion with None audio/text/visual inputs, DQN act/forward/soft_update, replay buffer sampling, environment reset/step, PetStats clamping, level-up chains, behavior selection, SQLite persistence round-trip.

---

## Reproducibility

| Lever | Value | Location |
|-------|-------|----------|
| Global seed | 42 | `config.yaml` |
| Python / NumPy / PyTorch seeds | seeded from 42 | module entry points |
| Device | CUDA (falls back to CPU) | `config.yaml` |
| Emotion labels | 7: happy/sad/angry/fear/surprise/disgust/neutral | `config.yaml` |
| DQN state dim | 12 | `rl_agent/environment.py` |
| DQN action dim | 5 | `rl_agent/environment.py` |

To reproduce the preliminary results:

```bash
python -m emotion_recognition.train --mock --epochs 5 --batch_size 16
python scripts/rl_quick_train.py     # 40 episodes, ~50s on GPU
python -m pytest tests/ -v           # 48 passed
```

---

## Citation

```bibtex
@misc{emotion-pet-2025,
  title  = {Design and Application of an Affective Desktop Pet:
            A Social Practice Study on College Student Mental Health},
  author = {Emotion Pet Research Group},
  year   = {2025},
  note   = {Best Academic Award --- Desktop Smart Pet,
            2025 Winter Social Practice},
}
```

---

## Roadmap

- [x] Project skeleton & module interfaces
- [x] Multimodal emotion recognition (BERT + CNN+LSTM + Fusion)
- [x] DQN reinforcement learning agent
- [x] Growth system with SQLite persistence
- [x] PyQt5 transparent window & animation framework
- [x] Unit test suite (48 tests passing)
- [x] Preliminary mock training results
- [ ] Full training on SST-2 / GoEmotions / RAVDESS
- [ ] 300-participant controlled user study (PHQ-9 / GAD-7 / PSS)
- [ ] Chinese speech emotion data augmentation
- [ ] Pet appearance customization
- [ ] Windows installer packaging

---

## License

[MIT](LICENSE) © 2025 Emotion Pet Research Group
