"""DQN 强化学习模块

子模块：
    dqn            - DQN 网络 + 目标网络 + ε-greedy
    replay_buffer  - 经验回放池
    environment    - 桌面宠物交互环境（状态/动作/奖励）
    train          - 训练入口
"""
from .dqn import DQN
from .replay_buffer import ReplayBuffer
from .environment import PetEnvironment

__all__ = ["DQN", "ReplayBuffer", "PetEnvironment"]
