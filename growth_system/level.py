"""LevelSystem —— 等级成长与技能/外观演化引擎。

    - 经验曲线为线性递增：exp_needed = level * 100；
    - 经验达到阈值即触发升级（可连续升级）；
    - 每升至特定等级解锁一项交互技能（安慰/讲笑话/给建议/冥想指导/时间管理）；
    - 等级驱动外观阶段演化（幼崽→儿童→少年→成年→精英）。

技能与外观均写回 PetStats，保证可随 PetStats 一并持久化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .stats import PetStats

# ---------------------------------------------------------------------------
# 静态配置（论文 3.4 节定义）
# ---------------------------------------------------------------------------
# 技能解锁等级映射：等级 → 技能标识
SKILL_UNLOCK_LEVELS: Dict[int, str] = {
    2: "comfort",       # 安慰
    4: "joke",          # 讲笑话
    6: "advice",        # 给建议
    8: "meditation",    # 冥想指导
    10: "time_manage",  # 时间管理
}

# 技能标识 → 中文展示名（供 GUI 气泡/状态面板展示）
SKILL_LABELS: Dict[str, str] = {
    "comfort": "安慰",
    "joke": "讲笑话",
    "advice": "给建议",
    "meditation": "冥想指导",
    "time_manage": "时间管理",
}

# 外观阶段映射：等级达到阈值即演化为对应阶段（升序排列）
APPEARANCE_STAGES: Tuple[Tuple[int, str], ...] = (
    (1, "baby"),    # 幼崽期
    (3, "child"),   # 儿童期
    (5, "teen"),    # 少年期
    (8, "adult"),   # 成年期
    (12, "elite"),  # 精英期
)


@dataclass
class LevelUpResult:
    """一次经验结算后触发的成长结果（供上层展示/日志）。"""

    levels_gained: List[int] = field(default_factory=list)   # 本次升到的等级
    skills_unlocked: List[str] = field(default_factory=list)  # 本次新解锁技能
    appearance: str = ""                                      # 结算后的外观阶段


class LevelSystem:
    """基于 PetStats 的等级成长引擎。

    不持有额外可变状态，所有成长结果均写回传入的 PetStats 实例。
    """

    def __init__(self, stats: PetStats) -> None:
        self.stats = stats

    # ------------------------------------------------------------------
    # 经验曲线
    # ------------------------------------------------------------------
    @property
    def exp_needed(self) -> int:
        """升级所需经验：exp_needed = level * 100（论文 3.4 节线性曲线）。"""
        return self.stats.level * 100

    def progress_ratio(self) -> float:
        """当前等级内经验进度（0.0 ~ 1.0），用于进度条展示。"""
        return min(1.0, self.stats.exp / self.exp_needed)

    # ------------------------------------------------------------------
    # 升级结算
    # ------------------------------------------------------------------
    def check_level_up(self) -> LevelUpResult:
        """结算经验：循环升级直到经验不足以再升一级。

        每次升级连锁触发技能解锁与外观更新，并写回 PetStats。

        Returns:
            LevelUpResult 记录本次升到的等级、解锁技能与最新外观。
        """
        result = LevelUpResult()
        while self.stats.exp >= self.exp_needed:
            self.stats.exp -= self.exp_needed
            self.stats.level += 1
            result.levels_gained.append(self.stats.level)
            result.skills_unlocked.extend(self.unlock_skill(self.stats.level))
        result.appearance = self.update_appearance()
        return result

    def gain_exp(self, amount: int) -> LevelUpResult:
        """语义化入口：先累加经验，再统一结算升级。"""
        self.stats.update("exp", amount)
        return self.check_level_up()

    # ------------------------------------------------------------------
    # 技能解锁
    # ------------------------------------------------------------------
    def unlock_skill(self, level: int) -> List[str]:
        """解锁指定等级对应的技能（若尚未拥有）。

        Args:
            level: 目标等级。

        Returns:
            本次新解锁的技能标识列表（空列表表示无新技能）。
        """
        skill = SKILL_UNLOCK_LEVELS.get(level)
        if skill is None or skill in self.stats.skills:
            return []
        self.stats.skills.append(skill)
        return [skill]

    @property
    def unlocked_skill_labels(self) -> List[str]:
        """当前已解锁技能的中文展示名列表。"""
        return [SKILL_LABELS.get(skill, skill) for skill in self.stats.skills]

    # ------------------------------------------------------------------
    # 外观演化
    # ------------------------------------------------------------------
    def update_appearance(self) -> str:
        """根据当前等级确定外观阶段并写回 PetStats。"""
        stage = self.appearance_for_level(self.stats.level)
        self.stats.appearance = stage
        return stage

    @staticmethod
    def appearance_for_level(level: int) -> str:
        """纯函数：返回给定等级对应的外观阶段标识。"""
        stage = APPEARANCE_STAGES[0][1]
        for threshold, name in APPEARANCE_STAGES:
            if level >= threshold:
                stage = name
            else:
                break
        return stage
