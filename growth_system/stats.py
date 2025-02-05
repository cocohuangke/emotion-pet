"""PetStats —— 宠物成长核心属性数据类。

宠物在长期交互过程中，通过心情(mood)、好感度(affinity)、精力(energy)、
经验(exp)等状态量的持续累积，驱动等级(level)提升、技能(skills)解锁与
外观(appearance)的阶段性演化。

本模块提供：
    - 带边界约束（clamp）的属性容器 PetStats
    - update(method, delta) 统一更新入口
    - to_dict / from_dict 序列化，供 SQLite 持久化与消息传递复用
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

# ---------------------------------------------------------------------------
# 状态空间约束（论文 3.4 节定义）
# ---------------------------------------------------------------------------
# 心情 / 好感 / 精力为 0-100 的百分制标度；经验无上限；等级从 1 起。
ATTR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "mood": (0.0, 100.0),
    "affinity": (0.0, 100.0),
    "energy": (0.0, 100.0),
    "exp": (0.0, math.inf),
    "level": (1.0, math.inf),
}

# 允许通过 update() 直接变更的属性（level 由 LevelSystem 统一管理）
UPDATABLE_METHODS: Tuple[str, ...] = ("mood", "affinity", "energy", "exp")

# 需要规范化为整型的属性（经验向下取整，其余四舍五入）
INTEGER_ATTRS: Tuple[str, ...] = ("mood", "affinity", "energy", "exp", "level")


@dataclass
class PetStats:
    """宠物成长属性的不可变语义快照。

    Attributes:
        mood: 心情值 0-100，数值越高越积极（论文 3.4 节核心状态量）。
        affinity: 对主人的好感度 0-100，长期互动累积。
        energy: 精力值 0-100，低于阈值时宠物进入休息状态。
        exp: 当前等级内累计经验，达到阈值后触发升级。
        level: 成长等级，从 1 开始（1 为幼崽期）。
        skills: 已解锁的交互技能标识列表（如 "comfort" 安慰）。
        appearance: 当前外观阶段（baby/child/teen/adult/elite）。
    """

    mood: int = 60
    affinity: int = 50
    energy: int = 80
    exp: int = 0
    level: int = 1
    skills: List[str] = field(default_factory=list)
    appearance: str = "baby"

    # ------------------------------------------------------------------
    # 更新与边界约束
    # ------------------------------------------------------------------
    def update(self, method: str, delta: float) -> "PetStats":
        """按给定方法更新对应属性，并自动 clamp 到合法区间。

        Args:
            method: 属性名（mood / affinity / energy / exp）。
            delta: 变化量，可正可负，可为浮点数。

        Returns:
            自身引用，支持链式调用（如 ``stats.update("mood", 5).gain_exp(10)``）。

        Raises:
            ValueError: 当 method 不在可更新属性集合中时抛出。
        """
        if method not in UPDATABLE_METHODS:
            raise ValueError(
                f"未知更新方法 {method!r}，可选值：{list(UPDATABLE_METHODS)}"
            )
        low, high = ATTR_BOUNDS[method]
        current = float(getattr(self, method))
        clamped = max(low, min(high, current + delta))
        setattr(self, method, self._cast(method, clamped))
        return self

    @staticmethod
    def _cast(method: str, value: float) -> int:
        """将数值规范化为整型：经验向下取整，其余四舍五入。"""
        if method == "exp":
            return int(math.floor(value))
        return int(round(value))

    def clamp_all(self) -> "PetStats":
        """将所有数值属性一次性 clamp 到合法区间（防御性校验）。"""
        for name in INTEGER_ATTRS:
            low, high = ATTR_BOUNDS[name]
            raw = float(getattr(self, name))
            setattr(self, name, self._cast(name, max(low, min(high, raw))))
        return self

    def gain_exp(self, amount: float) -> "PetStats":
        """经验获取的语义化封装（等价于 ``update("exp", amount)``）。"""
        return self.update("exp", amount)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典（供持久化与进程间传递）。"""
        return {
            "mood": self.mood,
            "affinity": self.affinity,
            "energy": self.energy,
            "exp": self.exp,
            "level": self.level,
            "skills": list(self.skills),
            "appearance": self.appearance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PetStats":
        """从字典构造实例。

        宽容策略：未知键忽略，缺失键回退默认值；构造后统一 clamp。
        """
        instance = cls()
        for key in ("mood", "affinity", "energy", "exp", "level", "appearance"):
            if key in data:
                setattr(instance, key, data[key])
        if "skills" in data and data["skills"] is not None:
            instance.skills = [str(skill) for skill in data["skills"]]
        return instance.clamp_all()

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        """调试友好的可读表示。"""
        return (
            f"PetStats(level={self.level}, mood={self.mood}, "
            f"affinity={self.affinity}, energy={self.energy}, "
            f"exp={self.exp}, skills={self.skills}, "
            f"appearance={self.appearance})"
        )

    def __repr__(self) -> str:
        return self.__str__()
