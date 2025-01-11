"""tests/test_growth_system.py - 成长系统单元测试。

覆盖：
    - PetStats 边界 clamp 与序列化往返；
    - LevelSystem 经验曲线 / 连续升级 / 技能解锁 / 外观演化；
    - BehaviorEngine 情绪->行为映射与低精力覆盖；
    - persistence SQLite 往返持久化。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from growth_system import BehaviorEngine, LevelSystem, PetStats
from growth_system.persistence import load_stats, save_stats


# ----------------------------------------------------------------------
# PetStats
# ----------------------------------------------------------------------
class TestPetStats:
    """PetStats 属性边界与序列化。"""

    def test_default_values(self) -> None:
        stats = PetStats()
        assert stats.mood == 60
        assert stats.affinity == 50
        assert stats.energy == 80
        assert stats.exp == 0
        assert stats.level == 1
        assert stats.skills == []
        assert stats.appearance == "baby"

    def test_update_clamp_high(self) -> None:
        stats = PetStats(mood=95)
        stats.update("mood", 50)  # 95+50=145 -> clamp 100
        assert stats.mood == 100

    def test_update_clamp_low(self) -> None:
        stats = PetStats(mood=10)
        stats.update("mood", -50)  # 10-50=-40 -> clamp 0
        assert stats.mood == 0

    def test_update_exp_no_upper_bound(self) -> None:
        stats = PetStats(exp=0)
        stats.update("exp", 9999)
        assert stats.exp == 9999

    def test_update_unknown_method_raises(self) -> None:
        stats = PetStats()
        with pytest.raises(ValueError, match="未知更新方法"):
            stats.update("unknown_attr", 10)

    def test_update_level_not_directly_updatable(self) -> None:
        """level 由 LevelSystem 管理，不允许通过 update 直接变更。"""
        stats = PetStats()
        with pytest.raises(ValueError):
            stats.update("level", 1)

    def test_gain_exp_chaining(self) -> None:
        stats = PetStats()
        result = stats.gain_exp(20)
        assert result is stats  # 链式返回自身
        assert stats.exp == 20

    def test_to_dict_roundtrip(self) -> None:
        stats = PetStats(mood=70, affinity=60, energy=40, exp=30, level=2,
                         skills=["comfort"], appearance="child")
        data = stats.to_dict()
        assert data["mood"] == 70
        assert data["skills"] == ["comfort"]

        restored = PetStats.from_dict(data)
        assert restored.mood == 70
        assert restored.skills == ["comfort"]
        assert restored.appearance == "child"

    def test_from_dict_tolerant(self) -> None:
        """未知键忽略，缺失键回退默认值。"""
        stats = PetStats.from_dict({"mood": 80, "unknown_key": 42})
        assert stats.mood == 80
        assert stats.affinity == 50  # 默认

    def test_clamp_all(self) -> None:
        stats = PetStats(mood=150, energy=-20)
        stats.clamp_all()
        assert stats.mood == 100
        assert stats.energy == 0


# ----------------------------------------------------------------------
# LevelSystem
# ----------------------------------------------------------------------
class TestLevelSystem:
    """等级成长曲线与连锁升级。"""

    def test_exp_needed_linear(self) -> None:
        stats = PetStats(level=1)
        ls = LevelSystem(stats)
        assert ls.exp_needed == 100

        stats.level = 3
        assert ls.exp_needed == 300

    def test_check_level_up_single(self) -> None:
        stats = PetStats(exp=100, level=1)
        ls = LevelSystem(stats)
        result = ls.check_level_up()
        assert result.levels_gained == [2]
        assert stats.level == 2
        assert stats.exp == 0

    def test_check_level_up_consecutive(self) -> None:
        """经验足够时连续升级。"""
        stats = PetStats(exp=250, level=1)
        # level 1 需要 100 -> 升级到 2，剩 150
        # level 2 需要 200 -> 不足，停
        ls = LevelSystem(stats)
        result = ls.check_level_up()
        assert result.levels_gained == [2]
        assert stats.level == 2
        assert stats.exp == 150

    def test_gain_exp_triggers_levelup(self) -> None:
        stats = PetStats(level=1)
        ls = LevelSystem(stats)
        result = ls.gain_exp(100)
        assert result.levels_gained == [2]
        assert stats.level == 2

    def test_unlock_skill_at_level_2(self) -> None:
        stats = PetStats(level=1)
        ls = LevelSystem(stats)
        unlocked = ls.unlock_skill(2)
        assert "comfort" in unlocked
        assert "comfort" in stats.skills

    def test_unlock_skill_idempotent(self) -> None:
        """重复调用同等级技能解锁只生效一次。"""
        stats = PetStats(level=1)
        ls = LevelSystem(stats)
        ls.unlock_skill(2)
        unlocked_again = ls.unlock_skill(2)
        assert unlocked_again == []
        assert stats.skills.count("comfort") == 1

    def test_appearance_evolution(self) -> None:
        stats = PetStats(level=1)
        ls = LevelSystem(stats)
        assert ls.update_appearance() == "baby"

        stats.level = 5
        assert ls.update_appearance() == "teen"

        stats.level = 12
        assert ls.update_appearance() == "elite"

    def test_full_levelup_chain(self) -> None:
        """一次大额经验同时触发升级 + 技能解锁 + 外观演化。"""
        stats = PetStats(level=1)
        ls = LevelSystem(stats)
        # level 1 -> 2 (需 100)，解锁 comfort
        # level 2 -> 3 (需 200)，无技能
        # level 3 -> 4 (需 300)，解锁 joke
        # 总需 100+200+300=600
        result = ls.gain_exp(600)
        assert stats.level == 4
        assert 4 in result.levels_gained
        assert "comfort" in result.skills_unlocked
        assert "joke" in result.skills_unlocked
        assert stats.appearance == "child"

    def test_progress_ratio(self) -> None:
        stats = PetStats(exp=50, level=1)
        ls = LevelSystem(stats)
        assert ls.progress_ratio() == 0.5


# ----------------------------------------------------------------------
# BehaviorEngine
# ----------------------------------------------------------------------
class TestBehaviorEngine:
    """情绪->行为决策。"""

    def test_select_behavior_returns_tuple(self) -> None:
        engine = BehaviorEngine(seed=42)
        behavior, dialogue = engine.select_behavior("happy", PetStats())
        assert isinstance(behavior, str)
        assert isinstance(dialogue, str)
        assert len(behavior) > 0
        assert len(dialogue) > 0

    def test_unknown_emotion_fallback_neutral(self) -> None:
        engine = BehaviorEngine(seed=42)
        behavior, dialogue = engine.select_behavior("unknown_emotion", PetStats())
        # 回退到 neutral 的候选池
        assert behavior in BehaviorEngine.behaviors_for("neutral")

    def test_low_energy_override(self) -> None:
        """精力低于阈值时强制休息，忽略情绪。"""
        engine = BehaviorEngine(seed=42)
        stats = PetStats(energy=10)  # 低于 LOW_ENERGY_THRESHOLD=25
        behavior, dialogue = engine.select_behavior("happy", stats)
        assert behavior == "rest"

    def test_reproducible_with_seed(self) -> None:
        """同 seed 两次运行结果一致。"""
        engine1 = BehaviorEngine(seed=42)
        engine2 = BehaviorEngine(seed=42)
        r1 = engine1.select_behavior("sad", PetStats(energy=80))
        r2 = engine2.select_behavior("sad", PetStats(energy=80))
        assert r1 == r2

    def test_select_dialogue(self) -> None:
        engine = BehaviorEngine(seed=42)
        d = engine.select_dialogue("angry")
        assert d in BehaviorEngine.dialogues_for("angry")


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
class TestPersistence:
    """SQLite 持久化往返。"""

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        db = str(tmp_path / "test_pet.db")
        stats = PetStats(
            mood=70, affinity=65, energy=40, exp=30, level=3,
            skills=["comfort", "joke"], appearance="teen",
        )
        save_stats(stats, db_path=db)

        loaded = load_stats(db_path=db)
        assert loaded is not None
        assert loaded.mood == 70
        assert loaded.affinity == 65
        assert loaded.energy == 40
        assert loaded.exp == 30
        assert loaded.level == 3
        assert loaded.skills == ["comfort", "joke"]
        assert loaded.appearance == "teen"

    def test_load_nonexistent_returns_none(self, tmp_path) -> None:
        db = str(tmp_path / "nonexistent.db")
        assert load_stats(db_path=db) is None

    def test_save_upsert_overwrites(self, tmp_path) -> None:
        """多次保存为 upsert 语义，不产生重复行。"""
        db = str(tmp_path / "test_upsert.db")
        save_stats(PetStats(mood=50), db_path=db)
        save_stats(PetStats(mood=80), db_path=db)

        loaded = load_stats(db_path=db)
        assert loaded is not None
        assert loaded.mood == 80
