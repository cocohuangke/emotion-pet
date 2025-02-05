"""DQN 训练入口

实现经典 DQN 训练循环，完整流程为：

    interact (与环境交互) -> remember (存入回放池) -> replay (采样学习)
    -> update target (软更新目标网络)

用法：
    python -m rl_agent.train --config config.yaml --episodes 1000 --batch_size 64

训练产物：
    - checkpoint: checkpoints/dqn_best.pt （平均奖励最优时的模型）
    - tensorboard 日志: logs/dqn/ （奖励、损失、探索率等曲线）

说明：
    训练使用模拟环境 (PetEnvironment)，目的是验证 DQN 流程与策略收敛，
    不涉及真实用户数据。真实数据接入后可直接替换环境观测来源。
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from .dqn import DQN
from .environment import ACTIONS, EMOTIONS, PetEnvironment
from .replay_buffer import ReplayBuffer


def resolve_device(requested: str) -> torch.device:
    """解析训练设备，若请求的设备不可用则回退到 CPU。

    Args:
        requested: 配置文件中指定的设备名（cuda / cpu / mps）。

    Returns:
        实际可用的 ``torch.device``。
    """
    requested = (requested or "cpu").lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置；若文件不存在则返回空配置（使用默认超参数）。

    Args:
        config_path: YAML 配置文件路径。

    Returns:
        配置字典。
    """
    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] 配置文件不存在: {path}，使用默认超参数。")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def set_seed(seed: int) -> None:
    """固定 Python / numpy / torch 随机种子，保证训练可复现。

    Args:
        seed: 全局随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    config: Dict[str, Any],
    episodes: int,
    batch_size: int,
) -> Tuple[DQN, float]:
    """执行 DQN 训练主循环。

    Args:
        config:     配置字典（可含 seed / device / checkpoint_root / log_root）。
        episodes:   训练 episode 数。
        batch_size: 每次重放采样的批大小。

    Returns:
        (训练好的策略网络, 最优窗口平均奖励)。
    """
    # ------------------------ 超参数 ------------------------
    seed: int = int(config.get("seed", 42))
    device: torch.device = resolve_device(str(config.get("device", "cpu")))

    # DQN 相关超参数（经典取值，见 Nature 2015 DQN 论文）
    gamma: float = 0.99              # 折扣因子
    tau: float = 1e-3                # 目标网络软更新系数
    lr: float = 1e-3                 # Adam 学习率
    buffer_capacity: int = 10000     # 回放池容量
    target_update_freq: int = 10     # 每多少步软更新一次目标网络
    epsilon_start: float = 1.0       # 初始探索率
    epsilon_end: float = 0.05        # 最小探索率
    epsilon_decay: float = 0.995     # 每 episode 探索率衰减系数
    gradient_clip: float = 10.0      # 梯度裁剪阈值

    env_max_steps: int = int(config.get("max_steps", 50))

    # ------------------------ 可复现 ------------------------
    set_seed(seed)

    # ------------------------ 环境与网络 ------------------------
    env = PetEnvironment(max_steps=env_max_steps, seed=seed)
    state_size: int = PetEnvironment.state_size
    action_size: int = PetEnvironment.action_size

    policy_net = DQN(state_size, action_size, hidden_sizes=(64, 64), seed=seed).to(device)
    target_net = DQN(state_size, action_size, hidden_sizes=(64, 64), seed=seed).to(device)
    # 初始化时目标网络与策略网络完全一致
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Huber 损失 (SmoothL1)：对离群 TD 误差更鲁棒，训练更稳定
    criterion = nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    # ------------------------ 输出目录 ------------------------
    checkpoint_dir = Path(config.get("checkpoint_root", "./checkpoints"))
    log_dir = Path(config.get("log_root", "./logs")) / "dqn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "dqn_best.pt"

    writer = SummaryWriter(log_dir=str(log_dir))

    # ------------------------ 训练状态 ------------------------
    reward_window: Deque[float] = deque(maxlen=50)   # 记录最近 50 个 episode 的奖励
    best_avg_reward: float = float("-inf")
    global_step: int = 0

    print(f"[INFO] 设备: {device} | episodes: {episodes} | batch_size: {batch_size}")
    print(f"[INFO] 状态维度: {state_size} | 动作维度: {action_size} | 动作: {ACTIONS}")
    print(f"[INFO] checkpoint: {checkpoint_path} | tensorboard: {log_dir}")

    # ------------------------ 训练主循环 ------------------------
    for episode in range(1, episodes + 1):
        state: np.ndarray = env.reset()
        episode_reward: float = 0.0

        # ε 随训练进行逐步衰减（指数衰减），初期多探索、后期多利用
        epsilon: float = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        while True:
            # --- interact: ε-greedy 选择动作并与环境交互 ---
            action: int = policy_net.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # --- remember: 存入经验回放池 ---
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            global_step += 1

            # --- replay: 经验足够后开始采样学习 ---
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

                # 当前 Q 值: Q(s, a)
                q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                # 目标 Q 值: r + γ * max_{a'} Q_target(s', a') * (1 - done)
                with torch.no_grad():
                    next_q_values = target_net(next_states_t).max(dim=1).values
                    target_q = rewards_t + gamma * next_q_values * (1.0 - dones_t)

                loss = criterion(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪，防止 Q 值爆炸导致训练震荡
                nn.utils.clip_grad_norm_(policy_net.parameters(), gradient_clip)
                optimizer.step()

                writer.add_scalar("train/loss", loss.item(), global_step)

                # --- update target: 周期性软更新目标网络 ---
                if global_step % target_update_freq == 0:
                    policy_net.soft_update(target_net, tau)

            if done:
                break

        # ------------------------ 记录与日志 ------------------------
        reward_window.append(episode_reward)
        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/epsilon", epsilon, episode)

        # 每 50 个 episode（或最后一个 episode）打印窗口平均奖励
        if episode % 50 == 0 or episode == episodes:
            avg_reward: float = float(np.mean(reward_window))
            print(
                f"[Episode {episode:5d}/{episodes}] "
                f"avg_reward={avg_reward:+.3f} "
                f"(window={len(reward_window)}) epsilon={epsilon:.3f}"
            )

            # 保存历史最优 checkpoint（用于 policy.py 推断）
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                checkpoint: Dict[str, Any] = {
                    "state_dict": policy_net.state_dict(),
                    "state_size": state_size,
                    "action_size": action_size,
                    "hidden_sizes": (64, 64),
                    "emotions": EMOTIONS,
                    "actions": ACTIONS,
                    "episode": episode,
                    "avg_reward": avg_reward,
                    "hyperparams": {
                        "gamma": gamma,
                        "tau": tau,
                        "lr": lr,
                        "batch_size": batch_size,
                        "epsilon_end": epsilon_end,
                        "epsilon_decay": epsilon_decay,
                        "buffer_capacity": buffer_capacity,
                    },
                    "env_note": (
                        "训练环境为模拟环境 (PetEnvironment)，"
                        "用户情绪转移与反馈均由 numpy 随机数模拟，非真实用户数据。"
                    ),
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"[INFO] 保存最优 checkpoint -> {checkpoint_path} "
                      f"(avg_reward={best_avg_reward:+.3f})")

    writer.close()
    print(f"[DONE] 训练完成，最优窗口平均奖励: {best_avg_reward:+.3f}")
    return policy_net, best_avg_reward


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="训练 DQN 动态反馈智能体（论文 3.3 节）"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="YAML 配置文件路径（默认 config.yaml）",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="训练 episode 数（默认 1000）",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="经验回放批大小（默认 64）",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config=config, episodes=args.episodes, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
