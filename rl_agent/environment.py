"""桌面宠物交互环境 (PetEnvironment)

模拟桌面宠物与用户之间的情感交互，为 DQN 智能体提供训练与评估环境。

状态空间 (state，共 12 维)：
    1. 用户情绪 one-hot (7 维)：
       happy / sad / angry / fear / surprise / disgust / neutral
    2. 历史交互 (3 维)：
       最近 3 次「模拟用户反馈」值，归一化到 [-1, 1]，刻画近期交互质量。
    3. 时间特征 (2 维)：
       [当天交互时刻 (0~1), 会话进度 (当前步数 / 最大步数，0~1)]

动作空间 (action，共 5 种反馈策略)：
    0 -> comfort    安慰
    1 -> encourage  鼓励
    2 -> distract   转移注意力
    3 -> listen     倾听
    4 -> accompany  陪伴

奖励设计 (reward)：
    reward = 1.0 * 情绪改善度 + 0.5 * 模拟用户反馈
    - 情绪改善度：转移前后情绪效价 (valence) 之差，是**主信号**，驱动智能体
      将用户情绪导向更积极的方向；
    - 模拟用户反馈：由情绪改善度叠加高斯噪声后裁剪到 [-1, 1]，模拟真实用户
      打分的主观性与波动，是**辅助信号**，权重更低。
================================================================================
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 全局常量定义（对齐项目根目录 config.yaml 的 emotion_labels）
# ---------------------------------------------------------------------------

# 7 类情绪标签，顺序固定，对应 one-hot 向量各维度
EMOTIONS: List[str] = [
    "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral",
]

# 5 种情感反馈策略，顺序固定，对应动作下标
ACTIONS: List[str] = ["comfort", "encourage", "distract", "listen", "accompany"]

# 情绪效价 (valence)：用于量化"情绪积极程度"，取值 [-1, 1]，越大越积极。
# 它是奖励信号的核心依据——智能体应尽量把用户推向高 valence 情绪。
EMOTION_VALENCE: Dict[str, float] = {
    "happy": 1.0,
    "surprise": 0.4,
    "neutral": 0.0,
    "disgust": -0.5,
    "sad": -0.6,
    "fear": -0.7,
    "angry": -0.8,
}

# 情绪按效价升序排列：改善 = 向列表右侧移动一档，恶化 = 向列表左侧移动一档。
# 该序列定义了环境中的情绪转移方向，简单直观且可复现。
VALENCE_ORDER: List[str] = [
    "angry", "fear", "sad", "disgust", "neutral", "surprise", "happy",
]

# 动作有效性矩阵 EFFECT[action][emotion]：
# 表示"该反馈策略对处于该情绪的用户产生改善"的概率 (0~1)。
# 这是模拟环境内置的先验规律——针对性反馈更有效，例如：
#   - 安慰 (comfort) 对悲伤 (sad)、恐惧 (fear) 更有效；
#   - 转移注意力 (distract) 对愤怒 (angry)、厌恶 (disgust) 更有效；
#   - 倾听 (listen) 对悲伤、恐惧、愤怒都较稳妥。
# 列顺序与 EMOTIONS 一致: [happy, sad, angry, fear, surprise, disgust, neutral]
EFFECT: np.ndarray = np.array(
    [
        # happy  sad    angry  fear   surprise  disgust  neutral
        [0.2,   0.8,   0.3,   0.6,   0.2,      0.3,     0.3],  # comfort 安慰
        [0.4,   0.7,   0.2,   0.5,   0.3,      0.2,     0.5],  # encourage 鼓励
        [0.2,   0.4,   0.8,   0.5,   0.4,      0.6,     0.3],  # distract 转移注意力
        [0.3,   0.7,   0.5,   0.7,   0.3,      0.4,     0.4],  # listen 倾听
        [0.5,   0.5,   0.3,   0.4,   0.4,      0.3,     0.5],  # accompany 陪伴
    ],
    dtype=np.float64,
)


class PetEnvironment:
    """桌面宠物与用户交互的模拟环境（非真实数据）。

    遵循 OpenAI Gym 风格的最小接口：``reset()`` / ``step(action)``，
    便于日后迁移到 gym 环境或与真实识别模块对接。
    """

    # 状态/动作空间维度（类属性，供 DQN 与训练脚本直接引用）
    emotions: List[str] = EMOTIONS
    actions: List[str] = ACTIONS
    state_size: int = 12          # 7 (one-hot) + 3 (历史) + 2 (时间)
    action_size: int = 5          # 5 种反馈策略
    history_len: int = 3          # 历史交互窗口长度

    def __init__(
        self,
        max_steps: int = 50,
        seed: Optional[int] = None,
        p_worsen: float = 0.1,
        feedback_noise: float = 0.25,
    ) -> None:
        """初始化模拟环境。

        Args:
            max_steps:      单个 episode（一次用户会话）的最大交互步数。
            seed:           随机种子，保证环境模拟可复现。
            p_worsen:       情绪恶化的基础概率系数（见 ``_transition``）。
            feedback_noise: 模拟用户反馈中的高斯噪声标准差。
        """
        self.max_steps: int = int(max_steps)
        self.p_worsen: float = float(p_worsen)
        self.feedback_noise: float = float(feedback_noise)

        # 使用独立的 Generator 实例，避免污染 numpy 全局随机状态，且支持复现
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # 内部状态
        self.emotion_idx: int = 0                 # 当前情绪在 EMOTIONS 中的下标
        self.current_step: int = 0                # 当前 episode 已交互步数
        self.time_of_day: float = 0.0             # 当天交互时刻，归一化到 [0, 1]
        # 最近 history_len 次模拟用户反馈的滑动窗口
        self.history: Deque[float] = deque(
            [0.0] * self.history_len, maxlen=self.history_len
        )

        # 初始化当前状态向量
        self.state: np.ndarray = self._build_state(
            self.emotion_idx, self.time_of_day, self.current_step
        )

    # ------------------------------------------------------------------
    # Gym 风格接口
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """重置环境，开始一个新的交互会话，返回初始状态。

        每个 episode 模拟一次"用户会话"：
            - 随机采样一个初始情绪（模拟用户打开桌面宠物时的情绪状态）；
            - 随机采样当天交互时刻（模拟用户在不同时间段使用宠物）。

        Returns:
            初始状态向量，shape (12,)，dtype float32。
        """
        self.current_step = 0
        self.emotion_idx = int(self.rng.integers(0, len(EMOTIONS)))
        self.time_of_day = float(self.rng.uniform(0.0, 1.0))
        self.history = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.state = self._build_state(
            self.emotion_idx, self.time_of_day, self.current_step
        )
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步交互，返回 ``(next_state, reward, done, info)``。

        Args:
            action: 智能体选择的反馈策略下标 (0 ~ 4)。

        Returns:
            next_state: 转移后的状态向量。
            reward:     即时奖励（情绪改善度 + 模拟用户反馈）。
            done:       本 episode 是否结束。
            info:       辅助信息字典（当前情绪、情绪改善度、模拟反馈等）。
        """
        prev_emotion: str = EMOTIONS[self.emotion_idx]
        prev_valence: float = EMOTION_VALENCE[prev_emotion]

        # 1) 情绪转移（模拟）：根据动作有效性矩阵 + 随机数决定改善/恶化/维持
        next_emotion: str = self._transition(action)
        next_valence: float = EMOTION_VALENCE[next_emotion]

        # 2) 奖励设计（详见模块 docstring）
        #    主信号：情绪效价的改善程度，取值范围 [-2, 2]
        emotion_gain: float = next_valence - prev_valence

        #    辅助信号：模拟用户显式反馈。
        #    用"情绪改善度 + 高斯噪声"近似真实用户打分，噪声模拟用户主观性。
        user_feedback: float = float(
            np.clip(
                emotion_gain + self.rng.normal(0.0, self.feedback_noise),
                -1.0,
                1.0,
            )
        )

        #    综合奖励 = 1.0 * 主信号 + 0.5 * 辅助信号
        reward: float = 1.0 * emotion_gain + 0.5 * user_feedback

        # 3) 更新内部状态
        self.emotion_idx = EMOTIONS.index(next_emotion)
        self.current_step += 1
        self.history.append(user_feedback)  # 记录本次模拟反馈，供下一状态使用

        done: bool = self.current_step >= self.max_steps
        self.state = self._build_state(
            self.emotion_idx, self.time_of_day, self.current_step
        )

        info: Dict = {
            "emotion": next_emotion,
            "prev_emotion": prev_emotion,
            "emotion_gain": emotion_gain,
            "user_feedback": user_feedback,
            "valence": next_valence,
            "step": self.current_step,
        }
        return self.state, reward, done, info

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _transition(self, action: int) -> str:
        """根据动作有效性概率矩阵模拟情绪转移。

        转移规则：
            - 以 ``EFFECT[action][emotion]`` 的概率改善情绪（效价升高一档）；
            - 以 ``p_worsen * (1 - EFFECT[action][emotion])`` 的概率恶化情绪，
              即"不匹配的反馈更可能让用户反感"；
            - 其余情况维持当前情绪不变。

        Args:
            action: 动作下标。

        Returns:
            转移后的情绪标签。
        """
        p_improve: float = float(EFFECT[action, self.emotion_idx])
        p_worsen: float = self.p_worsen * (1.0 - p_improve)

        current_emotion: str = EMOTIONS[self.emotion_idx]
        rank: int = VALENCE_ORDER.index(current_emotion)

        roll: float = float(self.rng.random())
        if roll < p_improve:
            # 改善：向效价更高方向移动一档，已在最高档则维持
            rank = min(rank + 1, len(VALENCE_ORDER) - 1)
        elif roll < p_improve + p_worsen:
            # 恶化：向效价更低方向移动一档，已在最低档则维持
            rank = max(rank - 1, 0)
        # 否则：维持当前情绪不变

        return VALENCE_ORDER[rank]

    def _build_state(self, emotion_idx: int, time_of_day: float, step: int) -> np.ndarray:
        """拼接 12 维状态向量：情绪 one-hot + 历史交互 + 时间特征。

        Args:
            emotion_idx:  当前情绪下标。
            time_of_day:  当天交互时刻（归一化到 [0, 1]）。
            step:         当前会话步数。

        Returns:
            12 维 float32 状态向量。
        """
        # 1) 用户情绪 one-hot (7 维)
        one_hot: np.ndarray = np.zeros(len(EMOTIONS), dtype=np.float32)
        one_hot[emotion_idx] = 1.0

        # 2) 历史交互 (3 维)：最近几次模拟用户反馈，缺失值用 0 填充
        history: np.ndarray = np.asarray(self.history, dtype=np.float32)

        # 3) 时间特征 (2 维)：交互时刻 + 会话进度
        session_progress: float = min(step / self.max_steps, 1.0)
        time_features: np.ndarray = np.array(
            [time_of_day, session_progress], dtype=np.float32
        )

        return np.concatenate([one_hot, history, time_features]).astype(np.float32)
