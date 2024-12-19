"""多模态情感数据集加载。

数据集格式（CSV）约定如下三列：

    text       - 原始文本字符串
    audio_path - 音频文件相对路径（相对于 ``data_root``，如 ``raw/audio/xxx.wav``）
    label      - 情绪标签名（须在 ``emotion_labels`` 中）

设计要点
--------
* :class:`MultimodalEmotionDataset` 按行加载，``__getitem__`` 返回
  ``(text, audio_mfcc, label)`` 三元组，与融合模型前向输入顺序一致。
* 音频 MFCC 通过 librosa 惰性提取：librosa 未安装或音频文件缺失时优雅降级，
  返回全零 MFCC，保证数据链路在无音频环境下仍可跑通。
* 提供 ``split_dataframe`` 进行训练/验证/测试划分（分层抽样，保持类别均衡）。
* 提供 mock 数据集与统一 collate 函数，供离线冒烟测试使用。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# 默认 MFCC 维度与最大音频帧数。
DEFAULT_N_MFCC: int = 40
DEFAULT_MAX_AUDIO_FRAMES: int = 300

# 文本列 / 音频路径列 / 标签列名（与 data/README.md 约定的 labels.csv 对齐）。
TEXT_COLUMN: str = "text"
AUDIO_PATH_COLUMN: str = "audio_path"
LABEL_COLUMN: str = "label"


def extract_mfcc(
    audio_path: Path,
    n_mfcc: int = DEFAULT_N_MFCC,
    sr: int = 16000,
    max_frames: int = DEFAULT_MAX_AUDIO_FRAMES,
) -> np.ndarray:
    """从音频文件提取 MFCC 特征。

    若 librosa / soundfile 不可用，或文件不存在、解码失败，则返回全零矩阵，
    保证调用方拿到形状确定的 ``(frames, n_mfcc)`` 张量。

    Parameters
    ----------
    audio_path : Path
        音频文件路径。
    n_mfcc : int
        MFCC 系数个数。
    sr : int
        重采样目标采样率。
    max_frames : int
        截断的最大帧数（超长音频做截断，控制显存与计算量）。

    Returns
    -------
    np.ndarray
        形状 ``(frames, n_mfcc)`` 的 MFCC 特征。
    """
    if not audio_path.exists():
        # 音频缺失 -> 返回全零占位，维度仍保持 (max_frames, n_mfcc)。
        return np.zeros((max_frames, n_mfcc), dtype=np.float32)

    try:
        import librosa
    except ImportError:
        # librosa 未安装 -> 全零占位。
        return np.zeros((max_frames, n_mfcc), dtype=np.float32)

    try:
        # 加载并重采样到目标采样率。
        y, sr_actual = librosa.load(str(audio_path), sr=sr)
        # 提取 MFCC，输出形状 (n_mfcc, frames)。
        mfcc: np.ndarray = librosa.feature.mfcc(y=y, sr=sr_actual, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # (frames, n_mfcc)
    except Exception:
        # 解码失败等异常 -> 全零占位。
        return np.zeros((max_frames, n_mfcc), dtype=np.float32)

    if mfcc.shape[0] > max_frames:
        mfcc = mfcc[:max_frames, :]
    return mfcc.astype(np.float32)


class MultimodalEmotionDataset(Dataset):
    """多模态情绪数据集。

    Parameters
    ----------
    dataframe : pd.DataFrame
        至少包含 ``text`` / ``audio_path`` / ``label`` 三列。
    emotion_labels : Sequence[str]
        情绪标签列表，顺序即分类索引。
    data_root : Path
        数据根目录，音频相对路径以该目录为基准解析。
    n_mfcc : int
        MFCC 特征维度（默认 40）。
    max_audio_frames : int
        音频最大帧数。
    tokenize : Optional[Callable[[str], List[int]]]
        文本分词函数（返回词表索引列表）；为 ``None`` 时保留原始字符串，
        由下游 collate 决定如何处理文本。
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        emotion_labels: Sequence[str],
        data_root: Path,
        n_mfcc: int = DEFAULT_N_MFCC,
        max_audio_frames: int = DEFAULT_MAX_AUDIO_FRAMES,
        tokenize: Optional[Callable[[str], List[int]]] = None,
    ) -> None:
        self.emotion_labels: List[str] = list(emotion_labels)
        self.data_root: Path = data_root
        self.n_mfcc: int = n_mfcc
        self.max_audio_frames: int = max_audio_frames
        self.tokenize: Optional[Callable[[str], List[int]]] = tokenize

        self.label_to_index: Dict[str, int] = {
            label: i for i, label in enumerate(self.emotion_labels)
        }

        # 过滤标签缺失或非法的行，构造干净的内部表。
        df: pd.DataFrame = dataframe.copy()
        df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        df = df[df[LABEL_COLUMN].isin(self.emotion_labels)]
        self.df: pd.DataFrame = df.reset_index(drop=True)

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, int]:
        """返回单个样本 ``(text, audio_mfcc, label)``。

        Returns
        -------
        Tuple[Any, torch.Tensor, int]
            ``text`` 为原始字符串（或 tokenize 后的索引列表），
            ``audio_mfcc`` 为 ``(frames, 40)`` 的浮点张量，
            ``label`` 为情绪类别索引。
        """
        row: pd.Series = self.df.iloc[idx]

        text: Any = str(row[TEXT_COLUMN])
        if self.tokenize is not None:
            text = self.tokenize(text)

        audio_rel: str = str(row[AUDIO_PATH_COLUMN])
        audio_path: Path = self.data_root / audio_rel
        mfcc: np.ndarray = extract_mfcc(
            audio_path, n_mfcc=self.n_mfcc, max_frames=self.max_audio_frames
        )
        audio_tensor: torch.Tensor = torch.from_numpy(mfcc)

        label: int = self.label_to_index[str(row[LABEL_COLUMN])]
        return text, audio_tensor, label

    @property
    def num_classes(self) -> int:
        """情绪类别数。"""
        return len(self.emotion_labels)

    @property
    def class_counts(self) -> Dict[str, int]:
        """各类别样本计数（用于观察类别均衡性）。"""
        return dict(self.df[LABEL_COLUMN].value_counts())


def split_dataframe(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按比例划分训练/验证/测试集。

    Parameters
    ----------
    dataframe : pd.DataFrame
        待划分的数据表。
    train_ratio, val_ratio, test_ratio : float
        三部分比例（须归一化为 1）。
    seed : int
        随机种子，保证划分可复现。
    stratify : bool
        是否按标签分层抽样（保持类别分布一致）。

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        训练、验证、测试三个子表。
    """
    total: float = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    from sklearn.model_selection import train_test_split

    stratify_col = dataframe[LABEL_COLUMN] if stratify else None

    train_df, rest_df = train_test_split(
        dataframe,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=stratify_col,
    )
    if stratify:
        stratify_rest = rest_df[LABEL_COLUMN]
    else:
        stratify_rest = None
    # rest 中验证/测试比例按相对占比换算。
    val_df, test_df = train_test_split(
        rest_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=stratify_rest,
    )
    return train_df, val_df, test_df


def load_dataset(
    csv_path: Path,
    emotion_labels: Sequence[str],
    data_root: Path,
    n_mfcc: int = DEFAULT_N_MFCC,
    max_audio_frames: int = DEFAULT_MAX_AUDIO_FRAMES,
    tokenize: Optional[Callable[[str], List[int]]] = None,
) -> MultimodalEmotionDataset:
    """从 CSV 加载数据集。

    Parameters
    ----------
    csv_path : Path
        CSV 文件路径。
    emotion_labels : Sequence[str]
        情绪标签列表。
    data_root : Path
        数据根目录。
    tokenize : Optional[Callable[[str], List[int]]]
        文本分词函数。

    Returns
    -------
    MultimodalEmotionDataset
        数据集对象。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset csv not found: {csv_path}")
    df: pd.DataFrame = pd.read_csv(csv_path)
    return MultimodalEmotionDataset(
        dataframe=df,
        emotion_labels=emotion_labels,
        data_root=data_root,
        n_mfcc=n_mfcc,
        max_audio_frames=max_audio_frames,
        tokenize=tokenize,
    )


class MockMultimodalDataset(Dataset):
    """离线冒烟测试用的合成数据集。

    生成随机文本词表索引、随机 MFCC 与随机标签，供 ``train.py --mock`` 跑通
    完整训练循环，不依赖真实数据与预训练权重。

    Parameters
    ----------
    num_samples : int
        样本数。
    num_emotions : int
        类别数。
    seq_len : int
        文本序列长度。
    mfcc_frames : int
        MFCC 帧数。
    vocab_size : int
        词表大小（文本索引取值范围）。
    seed : int
        随机种子。
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_emotions: int = 7,
        seq_len: int = 32,
        mfcc_frames: int = 100,
        vocab_size: int = 30522,
        seed: int = 42,
    ) -> None:
        self.num_samples: int = num_samples
        self.num_emotions: int = num_emotions
        self.seq_len: int = seq_len
        self.mfcc_frames: int = mfcc_frames
        self.vocab_size: int = vocab_size

        rng: random.Random = random.Random(seed)
        # 预生成标签，保证类别分布尽量均匀。
        self.labels: List[int] = [
            i % num_emotions for i in range(num_samples)
        ]
        rng.shuffle(self.labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[List[int], torch.Tensor, int]:
        """返回合成样本 ``(token_ids, mfcc, label)``。"""
        # 文本：随机词表索引（跳过 padding id 0）。
        token_ids: List[int] = [
            random.randint(1, self.vocab_size - 1) for _ in range(self.seq_len)
        ]
        # 音频：标准正态随机 MFCC。
        mfcc: torch.Tensor = torch.randn(self.mfcc_frames, DEFAULT_N_MFCC)
        label: int = self.labels[idx]
        return token_ids, mfcc, label


def make_collate_fn(
    tokenizer: Optional[Callable[[Sequence[str]], Any]] = None,
    max_length: int = 64,
) -> Callable[[Sequence[Tuple[Any, torch.Tensor, int]]], Tuple[torch.Tensor, ...]]:
    """构造批处理 collate 函数。

    当 ``tokenizer`` 为 ``None`` 时，假定样本的 ``text`` 已经是词表索引列表
    （mock 场景），直接做 padding；否则对原始字符串做 tokenize + padding。

    Returns
    -------
    Callable
        接收样本列表，返回 ``(input_ids, attention_mask, audio, labels)``。
    """

    def collate(
        batch: Sequence[Tuple[Any, torch.Tensor, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        texts: List[Any] = [item[0] for item in batch]
        audios: List[torch.Tensor] = [item[1] for item in batch]
        labels: torch.Tensor = torch.tensor(
            [item[2] for item in batch], dtype=torch.long
        )

        # 音频 padding：按最长帧数对齐 -> (B, T_max, 40)。
        audio_batch: torch.Tensor = pad_sequence(audios, batch_first=True)

        if tokenizer is not None:
            # 真实场景：对原始字符串做 tokenize。
            encodings: Any = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids: torch.Tensor = encodings["input_ids"]
            attention_mask: torch.Tensor = encodings["attention_mask"]
        else:
            # mock 场景：text 已是指数列表，直接手工 padding。
            id_tensors: List[torch.Tensor] = [
                torch.tensor(t, dtype=torch.long) for t in texts
            ]
            input_ids = pad_sequence(id_tensors, batch_first=True, padding_value=0)
            attention_mask = (input_ids != 0).long()

        return input_ids, attention_mask, audio_batch, labels

    return collate


if __name__ == "__main__":
    # 快速自检：mock 数据集 + collate 跑通一次批处理。
    mock = MockMultimodalDataset(num_samples=16)
    collate = make_collate_fn()
    input_ids, mask, audio, labels = collate([mock[i] for i in range(8)])
    print(f"[dataset] input_ids shape  = {tuple(input_ids.shape)}")
    print(f"[dataset] attention shape  = {tuple(mask.shape)}")
    print(f"[dataset] audio shape      = {tuple(audio.shape)}")
    print(f"[dataset] labels shape     = {tuple(labels.shape)}")
    print(f"[dataset] label values     = {labels.tolist()}")
