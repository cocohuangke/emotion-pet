"""模块配置加载器。

从项目根目录的 ``config.yaml`` 读取全局配置（情绪标签、设备、数据/模型/日志路径），
并为情感识别模块提供类型安全的访问接口。

设计要点
--------
* 使用 :class:`pathlib.Path` 解析路径，避免 ``os.path`` 字符串拼接。
* 默认配置文件位于 ``<project_root>/config.yaml``，可通过 ``ModuleConfig.from_yaml``
  显式指定其它路径（例如训练时 ``--config`` 传入实验专用配置）。
* 所有路径型配置以项目根目录为基准解析为绝对路径，保证任意工作目录下均可运行。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

# 项目根目录：本文件位于 <root>/emotion_recognition/config.py，向上两级即项目根。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# 默认配置文件路径（项目根目录下的 config.yaml）。
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

# 默认情绪标签（7 类，与论文设定一致），配置缺失时的兜底值。
DEFAULT_EMOTION_LABELS: List[str] = [
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]

# 默认音频 MFCC 维度、文本序列长度、设备。
DEFAULT_N_MFCC: int = 40
DEFAULT_MAX_TEXT_LENGTH: int = 64
DEFAULT_DEVICE: str = "cpu"


class ModuleConfig:
    """情感识别模块配置对象。

    对底层 YAML 字典做一次轻量封装，提供若干常用字段的强类型属性访问，
    同时保留 ``get`` 通用查询能力。

    Attributes
    ----------
    data : Dict[str, Any]
        原始配置字典（保留嵌套结构，便于扩展）。
    """

    def __init__(self, data: Mapping[str, Any], config_path: Optional[Path] = None) -> None:
        """用给定的映射初始化配置。

        Parameters
        ----------
        data : Mapping[str, Any]
            配置映射，通常来自 YAML 解析结果。
        config_path : Optional[Path]
            配置来源路径（仅用于日志与调试）。
        """
        self.data: Dict[str, Any] = dict(data)
        self.config_path: Path = config_path if config_path is not None else DEFAULT_CONFIG_PATH

    # ------------------------------------------------------------------
    # 构造方法
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "ModuleConfig":
        """从 YAML 文件加载配置。

        Parameters
        ----------
        config_path : Optional[Path]
            配置文件路径；缺省时使用项目根目录的 ``config.yaml``。
            文件不存在时回退到空配置（所有字段走默认值）。

        Returns
        -------
        ModuleConfig
            加载完成的配置对象。
        """
        path: Path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        if not path.exists():
            # 配置文件缺失时静默回退，保证 import 与 mock 训练不被阻断。
            return cls(data={}, config_path=path)
        with path.open("r", encoding="utf-8") as fp:
            raw: Any = yaml.safe_load(fp)
        data: Dict[str, Any] = raw if isinstance(raw, dict) else {}
        return cls(data=data, config_path=path)

    @classmethod
    def default(cls) -> "ModuleConfig":
        """构造一个使用默认值的配置对象（主要用于测试与 mock 运行）。"""
        return cls(data={})

    # ------------------------------------------------------------------
    # 通用访问
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """按点分路径读取配置项（如 ``"training.lr"``）。

        Parameters
        ----------
        key : str
            点分隔的键路径。
        default : Any
            键不存在时的默认值。
        """
        node: Any = self.data
        for part in key.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    # ------------------------------------------------------------------
    # 强类型属性
    # ------------------------------------------------------------------
    @property
    def emotion_labels(self) -> List[str]:
        """情绪标签列表（7 类，顺序即分类索引）。"""
        labels: Any = self.get("emotion_labels")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
        return list(DEFAULT_EMOTION_LABELS)

    @property
    def num_emotions(self) -> int:
        """情绪类别数。"""
        return len(self.emotion_labels)

    @property
    def label_to_index(self) -> Dict[str, int]:
        """标签名到分类索引的映射。"""
        return {label: i for i, label in enumerate(self.emotion_labels)}

    @property
    def device(self) -> str:
        """训练设备字符串（``cuda`` / ``cpu`` / ``mps``）。"""
        return str(self.get("device", DEFAULT_DEVICE))

    @property
    def seed(self) -> int:
        """全局随机种子。"""
        return int(self.get("seed", 42))

    @property
    def data_root(self) -> Path:
        """数据根目录（绝对路径）。"""
        raw: str = str(self.get("data_root", "./data"))
        return PROJECT_ROOT / raw if not Path(raw).is_absolute() else Path(raw)

    @property
    def checkpoint_root(self) -> Path:
        """模型 checkpoint 保存目录（绝对路径）。"""
        raw: str = str(self.get("checkpoint_root", "./checkpoints"))
        return PROJECT_ROOT / raw if not Path(raw).is_absolute() else Path(raw)

    @property
    def log_root(self) -> Path:
        """日志目录（绝对路径）。"""
        raw: str = str(self.get("log_root", "./logs"))
        return PROJECT_ROOT / raw if not Path(raw).is_absolute() else Path(raw)

    @property
    def n_mfcc(self) -> int:
        """音频 MFCC 特征维度（默认 40）。"""
        return int(self.get("audio.n_mfcc", DEFAULT_N_MFCC))

    @property
    def max_text_length(self) -> int:
        """文本最大序列长度。"""
        return int(self.get("text.max_length", DEFAULT_MAX_TEXT_LENGTH))

    def __repr__(self) -> str:
        """可读的配置摘要。"""
        return (
            f"ModuleConfig(path={self.config_path}, "
            f"num_emotions={self.num_emotions}, device={self.device})"
        )


def load_config(config_path: Optional[Path] = None) -> ModuleConfig:
    """加载模块配置的便捷函数。

    Parameters
    ----------
    config_path : Optional[Path]
        配置文件路径，缺省时加载项目根目录的 ``config.yaml``。

    Returns
    -------
    ModuleConfig
        配置对象。
    """
    return ModuleConfig.from_yaml(config_path)


if __name__ == "__main__":
    # 快速自检：打印关键配置，验证 YAML 解析与路径解析。
    cfg: ModuleConfig = load_config()
    print(f"[config] path        = {cfg.config_path}")
    print(f"[config] emotion     = {cfg.emotion_labels}")
    print(f"[config] num_emotions= {cfg.num_emotions}")
    print(f"[config] device      = {cfg.device}")
    print(f"[config] data_root   = {cfg.data_root}")
    print(f"[config] checkpoint  = {cfg.checkpoint_root}")
