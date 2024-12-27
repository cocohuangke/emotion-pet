"""经验回放缓冲 (Experience Replay Buffer)

经典 DQN 的关键组件之一：将智能体与环境交互产生的经验元组
``(state, action, reward, next_state, done)`` 存入固定容量的双端队列，
训练时随机采样一批数据进行学习。

作用：
    1. 打破连续样本之间的时序相关性，使随机梯度下降近似独立同分布假设；
    2. 每条经验可被多次复用，显著提升样本利用率。
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, NamedTuple, Optional, Tuple

import numpy as np


class Transition(NamedTuple):
    """单条经验元组。

    Attributes:
        state:      当前状态向量，shape (state_size,)。
        action:     执行的动作下标。
        reward:     获得的即时奖励。
        next_state: 转移后的下一状态向量。
        done:       是否结束当前 episode（终止标志）。
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """基于 ``collections.deque`` 的固定容量经验回放池。

    当存储的经验数量超过 ``capacity`` 时，最旧的经验会被自动丢弃
    （``deque(maxlen=...)`` 的天然行为），保证回放池始终保留近期样本。
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None) -> None:
        """初始化回放池。

        Args:
            capacity: 最大容量（能存储的经验条数）。
            seed:     可选随机种子，用于采样复现。
        """
        self.capacity: int = int(capacity)
        # deque 的 maxlen 会自动淘汰最旧元素，天然实现"先进先出"的环形缓冲
        self.memory: Deque[Transition] = deque(maxlen=self.capacity)
        # 使用独立的 random.Random 实例，避免污染全局随机状态，且支持复现
        self._rng: random.Random = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """存入一条经验。

        Args:
            state:      当前状态。
            action:     执行的动作。
            reward:     即时奖励。
            next_state: 下一状态。
            done:       是否终止。
        """
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )
        self.memory.append(transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机采样一批经验。

        Args:
            batch_size: 采样数量，必须不超过当前存储的经验数。

        Returns:
            五元组 ``(states, actions, rewards, next_states, dones)``，
            每个都是 numpy 数组，shape 分别为
            ``(batch_size, state_size)`` / ``(batch_size,)`` 等。

        Raises:
            ValueError: 当 ``batch_size`` 超过当前存储数量时抛出。
        """
        if batch_size > len(self.memory):
            raise ValueError(
                f"采样数量 {batch_size} 超过当前经验数 {len(self.memory)}"
            )

        # deque 支持下标访问，random.sample 可从中无放回随机抽样
        batch: list = self._rng.sample(self.memory, batch_size)

        states = np.stack([t.state for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """返回当前存储的经验条数。"""
        return len(self.memory)
