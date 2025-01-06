"""宠物成长系统模块

子模块：
    stats     - 属性（心情/好感/精力/经验）
    level     - 等级与成长曲线
    behaviors - 状态→行为策略
"""
from .stats import PetStats
from .level import LevelSystem
from .behaviors import BehaviorEngine

__all__ = ["PetStats", "LevelSystem", "BehaviorEngine"]
