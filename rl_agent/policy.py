"""训练后策略推断 (Inference)

加载训练好的 DQN checkpoint，根据当前用户情绪状态输出最优反馈动作，
并将动作映射为自然语言反馈文本，供桌面宠物 GUI 或其他下游模块调用。

用法示例：
    >>> from rl_agent.policy import FeedbackPolicy
    >>> policy = FeedbackPolicy("checkpoints/dqn_best.pt")
    >>> state = build_state_from_emotion("sad")   # 12 维状态向量
    >>> action, text = policy.respond(state)
    >>> print(text)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .dqn import DQN
from .environment import ACTIONS, EMOTIONS


# 动作名称 -> 反馈文本映射（与 environment.ACTIONS 对齐）
FEEDBACK_TEXTS: Dict[str, str] = {
    "comfort": "别担心，我陪着你，一切都会慢慢好起来的。",
    "encourage": "你已经做得很棒了，再坚持一下，我相信你！",
    "distract": "要不要听个笑话，或者一起玩个小游戏放松一下？",
    "listen": "我在听，慢慢说，我一直都在这里。",
    "accompany": "我会一直在这里陪着你。",
}


def load_model(checkpoint_path: Union[str, Path]) -> Tuple[DQN, Dict]:
    """从 checkpoint 加载训练好的 DQN 模型。

    Args:
        checkpoint_path: checkpoint 文件路径（``train.py`` 产出的 ``dqn_best.pt``）。

    Returns:
        二元组 ``(model, meta)``：
            - model: 已加载权重并切到 ``eval`` 模式的 DQN 网络；
            - meta:  checkpoint 中的元信息字典（含 state_size / actions 等）。
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    # weights_only=False：checkpoint 内包含字符串/列表等自定义元信息，
    # 不属于纯张量，故关闭安全加载限制（本项目自产自用，无外部来源风险）。
    checkpoint: Dict = torch.load(path, map_location="cpu", weights_only=False)

    state_size: int = int(checkpoint["state_size"])
    action_size: int = int(checkpoint["action_size"])
    hidden_sizes: tuple = tuple(checkpoint.get("hidden_sizes", (64, 64)))

    model = DQN(state_size, action_size, hidden_sizes=hidden_sizes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict(model: DQN, state: np.ndarray) -> int:
    """根据当前状态预测最优反馈动作（贪心策略，epsilon=0）。

    Args:
        model: 已加载的 DQN 模型。
        state: 状态向量，shape (state_size,)。

    Returns:
        动作下标，范围 [0, action_size)。
    """
    return model.act(state, epsilon=0.0)


def action_to_feedback(
    action: int,
    actions: Optional[List[str]] = None,
) -> str:
    """将动作下标映射为自然语言反馈文本。

    Args:
        action:  动作下标。
        actions: 动作名称列表（默认使用 environment.ACTIONS，与训练时一致）。

    Returns:
        对应的反馈文本。
    """
    action_names: List[str] = actions if actions is not None else ACTIONS
    if not 0 <= action < len(action_names):
        raise ValueError(f"非法动作下标: {action}")
    name: str = action_names[action]
    return FEEDBACK_TEXTS.get(name, "我会一直在这里陪着你。")


class FeedbackPolicy:
    """封装「模型 + 元信息」的策略推断器，提供开箱即用的接口。

    Attributes:
        model:   底层 DQN 模型。
        meta:    checkpoint 元信息。
        actions: 动作名称列表（优先取 checkpoint 中保存的，保证与训练一致）。
    """

    def __init__(self, checkpoint_path: Union[str, Path]) -> None:
        """加载 checkpoint 并初始化推断器。

        Args:
            checkpoint_path: checkpoint 文件路径。
        """
        self.model: DQN
        self.meta: Dict
        self.model, self.meta = load_model(checkpoint_path)
        # 优先使用 checkpoint 内保存的动作列表，确保动作下标语义一致
        self.actions: List[str] = list(self.meta.get("actions", ACTIONS))

    def predict(self, state: np.ndarray) -> int:
        """预测最优动作下标。

        Args:
            state: 状态向量。

        Returns:
            动作下标。
        """
        return predict(self.model, state)

    def feedback_text(self, action: int) -> str:
        """将动作下标映射为反馈文本。

        Args:
            action: 动作下标。

        Returns:
            反馈文本。
        """
        return action_to_feedback(action, self.actions)

    def respond(self, state: np.ndarray) -> Tuple[int, str]:
        """一步到位：输入状态，输出 (动作下标, 反馈文本)。

        Args:
            state: 状态向量。

        Returns:
            二元组 ``(action, feedback_text)``。
        """
        action: int = self.predict(state)
        return action, self.feedback_text(action)


def build_state_from_emotion(
    emotion: str,
    history: Optional[List[float]] = None,
    time_of_day: float = 0.5,
    session_progress: float = 0.0,
) -> np.ndarray:
    """便捷工具：由情绪标签 + 少量特征构造 12 维状态向量，方便快速试跑。

    状态结构与 ``PetEnvironment._build_state`` 保持一致：
    情绪 one-hot (7) + 历史反馈 (3) + 时间特征 (2)。

    Args:
        emotion:          情绪标签（EMOTIONS 之一）。
        history:          最近 3 次用户反馈值（缺省为全 0）。
        time_of_day:      当天时刻，归一化到 [0, 1]。
        session_progress: 会话进度，归一化到 [0, 1]。

    Returns:
        12 维 float32 状态向量。
    """
    if emotion not in EMOTIONS:
        raise ValueError(f"未知情绪标签: {emotion}，应为 {EMOTIONS} 之一")

    one_hot = np.zeros(len(EMOTIONS), dtype=np.float32)
    one_hot[EMOTIONS.index(emotion)] = 1.0

    hist = np.asarray(history if history is not None else [0.0, 0.0, 0.0], dtype=np.float32)
    if hist.shape[0] != 3:
        raise ValueError(f"history 长度应为 3，实际 {hist.shape[0]}")

    time_feat = np.array([time_of_day, session_progress], dtype=np.float32)
    return np.concatenate([one_hot, hist, time_feat]).astype(np.float32)


def main() -> None:
    """命令行演示：加载 checkpoint 并对若干典型情绪状态输出反馈。"""
    import argparse

    parser = argparse.ArgumentParser(description="DQN 反馈策略推断演示")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/dqn_best.pt",
        help="checkpoint 文件路径",
    )
    args = parser.parse_args()

    policy = FeedbackPolicy(args.checkpoint)
    print(f"[INFO] 已加载 checkpoint: {args.checkpoint}")
    print(f"[INFO] 动作空间: {policy.actions}")

    for emotion in ["sad", "angry", "fear", "happy"]:
        state = build_state_from_emotion(emotion)
        action, text = policy.respond(state)
        print(f"  情绪={emotion:<8s} -> 动作={policy.actions[action]:<10s} | {text}")


if __name__ == "__main__":
    main()
