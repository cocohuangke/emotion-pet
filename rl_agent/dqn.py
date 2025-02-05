"""DQN 深度 Q 网络 (Deep Q-Network)

    - 3 层 MLP: state_size -> 64 -> 64 -> action_size
    - 支持目标网络 (target network) 软更新 (soft update)，稳定训练
    - ε-greedy 动作选择策略，平衡探索 (exploration) 与利用 (exploitation)

DQN 通过逼近动作价值函数 Q(s, a)，为桌面宠物在给定用户情绪状态下
选择最优的情感反馈策略（安慰 / 鼓励 / 转移注意力 / 倾听 / 陪伴）。
"""

from __future__ import annotations

import random
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """3 层 MLP 深度 Q 网络。

    Attributes:
        state_size:   状态空间维度（本项目为 12 维，见 environment.PetEnvironment）。
        action_size:  动作空间维度（本项目为 5 种反馈策略）。
        hidden_sizes: 隐藏层宽度序列，默认 (64, 64)。
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (64, 64),
        seed: Optional[int] = None,
    ) -> None:
        """初始化 DQN 网络。

        Args:
            state_size:   输入状态维度。
            action_size:  输出动作维度（Q 值个数）。
            hidden_sizes: 每个隐藏层的神经元数量。
            seed:         可选随机种子，用于权重初始化复现。
        """
        super().__init__()
        self.state_size: int = int(state_size)
        self.action_size: int = int(action_size)
        self.hidden_sizes: tuple = tuple(hidden_sizes)

        # 固定权重初始化种子，保证实验可复现（对齐 config.yaml seed: 42 的意图）
        if seed is not None:
            torch.manual_seed(seed)

        # 构建 3 层 MLP: Linear -> ReLU -> Linear -> ReLU -> Linear
        # 输出层不加激活函数，直接输出每个动作的 Q 值（可为任意实数）
        layers: list = []
        input_dim: int = self.state_size
        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.action_size))
        self.net: nn.Sequential = nn.Sequential(*layers)

    @property
    def device(self) -> torch.device:
        """返回网络参数所在的设备（cpu / cuda / mps）。"""
        return next(self.parameters()).device

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，计算每个动作的 Q 值。

        Args:
            state: 状态张量，shape (batch_size, state_size)。

        Returns:
            Q 值张量，shape (batch_size, action_size)。
        """
        return self.net(state)

    @torch.no_grad()
    def act(self, state: Union[np.ndarray, list], epsilon: float = 0.0) -> int:
        """ε-greedy 策略选择动作。

        以概率 ``epsilon`` 随机选择一个动作（探索），
        否则选择当前 Q 值最大的动作（贪心利用）。

        Args:
            state:   单个状态向量，shape (state_size,)，numpy 数组或列表。
            epsilon: 探索概率，取值范围 [0, 1]。

        Returns:
            选中的动作下标，范围 [0, action_size)。
        """
        # 探索：均匀随机采样一个动作
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        # 利用：选择 Q 值最大的动作。将状态张量移动到网络所在设备，避免 CPU/CUDA 不匹配。
        state_tensor: torch.Tensor = (
            torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        q_values: torch.Tensor = self.forward(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def soft_update(self, target_net: "DQN", tau: float = 1e-3) -> None:
        """对目标网络执行软更新（Polyak averaging）。

        .. math::
            \\theta_{\\text{target}} \\leftarrow
            \\tau \\cdot \\theta + (1 - \\tau) \\cdot \\theta_{\\text{target}}

        相比硬拷贝（``tau=1``，直接复制全部权重），软更新让目标 Q 值变化更
        平滑，显著缓解 DQN 训练中的目标漂移与震荡问题，是经典 DQN 的标配技巧。

        Args:
            target_net: 目标网络实例（结构与当前网络一致）。
            tau:        软更新系数，取值 (0, 1]。越小目标网络更新越缓慢。
        """
        for target_param, param in zip(target_net.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
