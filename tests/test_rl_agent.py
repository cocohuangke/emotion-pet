"""tests/test_rl_agent.py - DQN 强化学习模块单元测试。

覆盖：
    - ReplayBuffer 的 push/sample 行为；
    - DQN 网络 forward 与 act (ε-greedy)；
    - PetEnvironment 的 reset / step / reward 计算。
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_agent import DQN, PetEnvironment, ReplayBuffer


# ----------------------------------------------------------------------
# ReplayBuffer
# ----------------------------------------------------------------------
class TestReplayBuffer:
    """经验回放池。"""

    def test_push_and_len(self) -> None:
        buf = ReplayBuffer(capacity=10)
        assert len(buf) == 0

        transition = (np.zeros(12), 0, 1.0, np.zeros(12), False)
        buf.push(*transition)
        assert len(buf) == 1

    def test_capacity_overflow(self) -> None:
        """超过容量后保留最新的 capacity 条。"""
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push(np.array([i], dtype=np.float32), 0, float(i),
                     np.array([i + 1], dtype=np.float32), False)
        assert len(buf) == 3

    def test_sample_batch_size(self) -> None:
        buf = ReplayBuffer(capacity=20)
        for i in range(10):
            buf.push(np.zeros(12, dtype=np.float32), i % 5, 1.0,
                     np.zeros(12, dtype=np.float32), False)

        states, actions, rewards, next_states, dones = buf.sample(batch_size=4)
        assert states.shape == (4, 12)
        assert actions.shape == (4,)
        assert rewards.shape == (4,)
        assert next_states.shape == (4, 12)
        assert dones.shape == (4,)

    def test_sample_insufficient_raises(self) -> None:
        """样本不足 batch_size 时应抛出异常或给出清晰提示。"""
        buf = ReplayBuffer(capacity=10)
        buf.push(np.zeros(12, dtype=np.float32), 0, 1.0,
                 np.zeros(12, dtype=np.float32), False)
        with pytest.raises(Exception):
            buf.sample(batch_size=4)


# ----------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------
class TestDQN:
    """DQN 网络结构。"""

    def test_forward_output_shape(self) -> None:
        import torch

        net = DQN(state_size=12, action_size=5, hidden_sizes=(64, 64))
        state = torch.randn(8, 12)
        q_values = net(state)
        assert q_values.shape == (8, 5)

    def test_forward_single_state(self) -> None:
        import torch

        net = DQN(state_size=12, action_size=5)
        state = torch.randn(1, 12)
        q = net(state)
        assert q.shape == (1, 5)

    def test_act_returns_valid_action(self) -> None:
        net = DQN(state_size=12, action_size=5)
        state = np.random.randn(12).astype(np.float32)
        action = net.act(state, epsilon=0.0)
        assert 0 <= action < 5

    def test_act_greedy_deterministic(self) -> None:
        """epsilon=0 时对同一 state 总返回相同 action。"""
        net = DQN(state_size=12, action_size=5)
        state = np.random.randn(12).astype(np.float32)
        a1 = net.act(state, epsilon=0.0)
        a2 = net.act(state, epsilon=0.0)
        assert a1 == a2

    def test_act_random_when_epsilon_one(self) -> None:
        """epsilon=1 时为纯随机，action 仍在合法区间。"""
        net = DQN(state_size=12, action_size=5)
        state = np.zeros(12, dtype=np.float32)
        actions = {net.act(state, epsilon=1.0) for _ in range(20)}
        # 20 次随机应至少出现 2 种不同 action
        assert len(actions) >= 2
        for a in actions:
            assert 0 <= a < 5


# ----------------------------------------------------------------------
# PetEnvironment
# ----------------------------------------------------------------------
class TestPetEnvironment:
    """模拟交互环境。"""

    def test_reset_returns_state(self) -> None:
        env = PetEnvironment()
        state = env.reset()
        assert state.shape == (12,) or state.shape == (1, 12) or len(state) == 12

    def test_step_returns_transition(self) -> None:
        env = PetEnvironment()
        env.reset()
        result = env.step(0)
        # 至少包含 next_state, reward, done
        assert len(result) >= 3

    def test_action_space_valid(self) -> None:
        env = PetEnvironment()
        assert env.action_size == 5

    def test_step_all_actions_valid(self) -> None:
        """5 种 action 均能正常执行不报错。"""
        env = PetEnvironment()
        for action in range(5):
            env.reset()
            result = env.step(action)
            assert len(result) >= 3

    def test_reward_is_float(self) -> None:
        env = PetEnvironment()
        env.reset()
        _, reward, _, _ = env.step(0)
        assert isinstance(reward, (float, np.floating))
