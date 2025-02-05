"""persistence —— 基于 SQLAlchemy 2.0 的 SQLite 持久化。

将 PetStats 映射为关系表 ``pet_stats``，提供 ``save_stats`` / ``load_stats``
两个函数式接口，供桌面宠物退出时保存成长状态、启动时恢复，从而实现
「成长状态跨会话延续」。

采用 SQLAlchemy 2.0 声明式风格（DeclarativeBase + Mapped + mapped_column）。
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import JSON, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .stats import PetStats

# 单宠物系统的固定主键（仅存一行，upsert 语义）
_SINGLETON_ID = 1


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类。"""


class PetStatsModel(Base):
    """PetStats 的关系表映射。"""

    __tablename__ = "pet_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mood: Mapped[int] = mapped_column(Integer, nullable=False)
    affinity: Mapped[int] = mapped_column(Integer, nullable=False)
    energy: Mapped[int] = mapped_column(Integer, nullable=False)
    exp: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    # 技能列表以 JSON 存储（SQLite 下映射为 TEXT）
    skills: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    appearance: Mapped[str] = mapped_column(String(32), nullable=False)

    def to_stats(self) -> PetStats:
        """ORM 模型 → 领域对象。"""
        return PetStats(
            mood=self.mood,
            affinity=self.affinity,
            energy=self.energy,
            exp=self.exp,
            level=self.level,
            skills=list(self.skills or []),
            appearance=self.appearance,
        )

    @classmethod
    def from_stats(cls, stats: PetStats) -> "PetStatsModel":
        """领域对象 → ORM 模型。"""
        return cls(
            id=_SINGLETON_ID,
            mood=stats.mood,
            affinity=stats.affinity,
            energy=stats.energy,
            exp=stats.exp,
            level=stats.level,
            skills=list(stats.skills),
            appearance=stats.appearance,
        )


def _engine(db_path: str):
    """构造 SQLite 引擎（自动创建父目录）。"""
    directory = os.path.dirname(os.path.abspath(db_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


def save_stats(stats: PetStats, db_path: str = "data/pet.db") -> None:
    """保存（upsert）PetStats 到 SQLite。

    Args:
        stats: 待持久化的宠物状态。
        db_path: 数据库文件路径（默认 ``data/pet.db``）。
    """
    engine = _engine(db_path)
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            existing = session.get(PetStatsModel, _SINGLETON_ID)
            if existing is None:
                existing = PetStatsModel(id=_SINGLETON_ID)
                session.add(existing)
            _apply_fields(existing, stats)
            session.commit()
    finally:
        # 显式释放连接池，确保 Windows 下数据库文件句柄及时关闭
        engine.dispose()


def load_stats(db_path: str = "data/pet.db") -> Optional[PetStats]:
    """从 SQLite 读取 PetStats。

    Args:
        db_path: 数据库文件路径。

    Returns:
        若存在记录则返回 PetStats，否则返回 None。
    """
    if not os.path.exists(db_path):
        return None
    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            row = session.get(PetStatsModel, _SINGLETON_ID)
            return row.to_stats() if row is not None else None
    finally:
        engine.dispose()


def _apply_fields(model: PetStatsModel, stats: PetStats) -> None:
    """将领域对象字段同步到已有 ORM 模型（复用 upsert 逻辑）。"""
    model.mood = stats.mood
    model.affinity = stats.affinity
    model.energy = stats.energy
    model.exp = stats.exp
    model.level = stats.level
    model.skills = list(stats.skills)
    model.appearance = stats.appearance
