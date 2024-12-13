#!/bin/bash
# 一键训练流水线
# 用法: bash scripts/run_all.sh

set -e

echo "[1/4] Training emotion recognition model..."
python -m emotion_recognition.train --config config.yaml --epochs 20

echo "[2/4] Training DQN feedback agent..."
python -m rl_agent.train --config config.yaml --episodes 1000

echo "[3/4] Evaluating models..."
python -m emotion_recognition.evaluate --checkpoint checkpoints/emotion_best.pt

echo "[4/4] Launching desktop pet..."
python -m desktop_pet.main
