"""BehaviorEngine —— 基于情绪标签与宠物状态的行为决策引擎。

对齐论文 4.1 节（桌面宠物架构）与 3.4 节（成长系统）的交汇点：
情感识别模块输出情绪标签后，本引擎结合当前 PetStats（精力/心情/好感），
从行为数据库与对话模板库中采样合适的 (动作, 对话) 二元组，构成
「感知 → 决策 → 反馈」的交互闭环。

决策规则（按优先级）：
    1. 精力低于阈值 → 无条件进入休息状态（不区分情绪）；
    2. 否则在对应情绪的候选动作 / 对话中随机采样。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .stats import PetStats

# 支持的 7 类情绪标签（对齐 config.yaml 的 emotion_labels）
SUPPORTED_EMOTIONS: Tuple[str, ...] = (
    "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral",
)

# ---------------------------------------------------------------------------
# 行为数据库：情绪 → 候选动作列表（论文 3.4/4.1 节动作空间）
# ---------------------------------------------------------------------------
BEHAVIOR_DATABASE: Dict[str, List[str]] = {
    "happy":    ["dance", "smile", "cheer", "bounce"],
    "sad":      ["comfort", "hug", "listen", "pat"],
    "angry":    ["calm", "breathe", "listen", "step_back"],
    "fear":     ["reassure", "comfort", "hide", "blink"],
    "surprise": ["jump", "blink", "tilt", "look"],
    "disgust":  ["grimace", "step_back", "shake", "ignore"],
    "neutral":  ["idle", "blink", "wander", "stretch"],
}

# ---------------------------------------------------------------------------
# 对话模板数据库：情绪 → 候选对话
# ---------------------------------------------------------------------------
DIALOGUE_TEMPLATES: Dict[str, List[str]] = {
    "happy":    ["今天心情不错，我也跟着开心！", "嘿嘿，看到你笑我就想跳舞～", "要不要一起玩个游戏？"],
    "sad":      ["别难过，我会一直在这里陪着你。", "想聊聊吗？我随时都在听。", "给你一个暖暖的抱抱。"],
    "angry":    ["深呼吸，先冷静一下好吗？", "我知道你很生气，慢慢说。", "我在呢，别着急。"],
    "fear":     ["别怕，有我在你身边。", "这里很安全，放松一点。", "我陪你一起面对。"],
    "surprise": ["哇！发生什么啦？", "你吓我一跳～", "这个真有意思！"],
    "disgust":  ["呃……这确实有点难顶。", "要不要换件事做？", "我懂，先离远一点。"],
    "neutral":  ["我在呢，想做什么？", "今天想聊点什么？", "随时听候差遣～"],
}

# ---------------------------------------------------------------------------
# 低精力覆盖策略（无论情绪，优先休息）
# ---------------------------------------------------------------------------
LOW_ENERGY_THRESHOLD: int = 25
LOW_ENERGY_BEHAVIOR: str = "rest"
LOW_ENERGY_DIALOGUES: List[str] = [
    "我有点累了，先眯一会儿……",
    "电量不足，休息一下吧。",
]


class BehaviorEngine:
    """状态 → (行为, 对话) 的决策引擎。

    随机采样支持传入 seed，保证实验可复现（对齐 config.yaml 的 seed=42）。
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 决策接口
    # ------------------------------------------------------------------
    def select_behavior(
        self, emotion: str, stats: Optional[PetStats] = None
    ) -> Tuple[str, str]:
        """根据情绪标签与宠物状态选择行为。

        Args:
            emotion: 情感识别输出的情绪标签（未知标签回退 neutral）。
            stats: 宠物当前状态，可为 None（此时忽略状态约束）。

        Returns:
            (behavior, dialogue) 二元组。
        """
        emotion = self._resolve_emotion(emotion)

        if stats is not None and stats.energy <= LOW_ENERGY_THRESHOLD:
            behavior = LOW_ENERGY_BEHAVIOR
            dialogue = self._rng.choice(LOW_ENERGY_DIALOGUES)
        else:
            behavior = self._rng.choice(BEHAVIOR_DATABASE[emotion])
            dialogue = self._rng.choice(DIALOGUE_TEMPLATES[emotion])

        return behavior, dialogue

    def select_dialogue(self, emotion: str) -> str:
        """仅采样对话模板（供用户主动对话交互场景复用）。"""
        emotion = self._resolve_emotion(emotion)
        return self._rng.choice(DIALOGUE_TEMPLATES[emotion])

    # ------------------------------------------------------------------
    # 只读查询（供训练/评测/扩展复用）
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_emotion(emotion: str) -> str:
        """将未知情绪回退到 neutral，保证系统鲁棒。"""
        return emotion if emotion in SUPPORTED_EMOTIONS else "neutral"

    @classmethod
    def behaviors_for(cls, emotion: str) -> List[str]:
        """返回指定情绪的行为候选列表（只读副本）。"""
        return list(BEHAVIOR_DATABASE.get(cls._resolve_emotion(emotion), []))

    @classmethod
    def dialogues_for(cls, emotion: str) -> List[str]:
        """返回指定情绪的对话候选列表（只读副本）。"""
        return list(DIALOGUE_TEMPLATES.get(cls._resolve_emotion(emotion), []))
