"""多模态情绪识别模型训练入口。

用法示例（在项目根目录执行）：:

    # 使用合成 mock 数据冒烟测试（无需真实数据与 BERT 权重）
    python -m emotion_recognition.train --mock --epochs 1

    # 使用真实数据训练
    python -m emotion_recognition.train --epochs 20 --batch_size 32 --lr 1e-4

说明
----
* 采用 ``CrossEntropyLoss`` + ``Adam`` 优化器。
* 每个 epoch 结束打印平均 loss 与准确率；验证集准确率提升时保存最佳
  checkpoint 到 ``checkpoints/best_model.pt``。
* 训练结束额外保存 ``checkpoints/last_model.pt`` 供断点续训 / 评估使用。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import ModuleConfig, load_config
from .dataset import (
    MockMultimodalDataset,
    load_dataset,
    make_collate_fn,
    split_dataframe,
)
from .models.bert_text import BertTextEncoder
from .models.cnn_lstm_audio import CNNLSTMAudioEncoder
from .models.fusion import MultimodalFusionModel


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """解析训练设备，``cuda`` 不可用时回退到 ``cpu``。"""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[train] MPS requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算单批准确率。

    Parameters
    ----------
    logits : torch.Tensor
        形状 ``(B, C)`` 的模型输出。
    labels : torch.Tensor
        形状 ``(B,)`` 的真实标签。

    Returns
    -------
    float
        0~1 之间的准确率。
    """
    preds: torch.Tensor = logits.argmax(dim=1)
    correct: int = int((preds == labels).sum().item())
    return correct / labels.size(0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float]:
    """执行一个 epoch 的训练或验证。

    Parameters
    ----------
    optimizer : Optional[torch.optim.Optimizer]
        传入优化器时为训练模式（反向传播 + 更新）；为 ``None`` 时为验证模式。
    """
    is_train: bool = optimizer is not None
    model.train(is_train)

    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    for input_ids, attention_mask, audio, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        audio = audio.to(device)
        labels = labels.to(device)

        logits: torch.Tensor = model(
            input_ids, audio, attention_mask=attention_mask
        )
        loss: torch.Tensor = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds: torch.Tensor = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_samples += labels.size(0)

    avg_loss: float = total_loss / max(total_samples, 1)
    avg_acc: float = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def build_mock_loaders(
    batch_size: int,
    num_emotions: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """构造合成训练/验证数据加载器（离线冒烟测试）。"""
    collate = make_collate_fn(tokenizer=None)
    train_ds: Dataset = MockMultimodalDataset(
        num_samples=batch_size * 4, num_emotions=num_emotions, seed=seed
    )
    val_ds: Dataset = MockMultimodalDataset(
        num_samples=batch_size * 2, num_emotions=num_emotions, seed=seed + 1
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def build_real_loaders(
    cfg: ModuleConfig,
    batch_size: int,
    tokenizer: Optional[object] = None,
) -> Tuple[DataLoader, DataLoader]:
    """构造真实数据加载器（读取 CSV，按比例划分训练/验证）。"""
    data_root: Path = cfg.data_root
    csv_path: Path = data_root / "raw" / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"training csv not found: {csv_path}. "
            f"Run scripts/download_data.py or pass --mock for a smoke test."
        )

    import pandas as pd

    df = pd.read_csv(csv_path)
    train_df, val_df, _ = split_dataframe(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=cfg.seed)

    from .dataset import MultimodalEmotionDataset

    collate = make_collate_fn(tokenizer=tokenizer, max_length=cfg.max_text_length)
    train_ds = MultimodalEmotionDataset(train_df, cfg.emotion_labels, data_root)
    val_ds = MultimodalEmotionDataset(val_df, cfg.emotion_labels, data_root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def build_model(
    cfg: ModuleConfig,
    use_pretrained_bert: bool,
    freeze_bert: bool,
) -> MultimodalFusionModel:
    """按配置构造融合模型（离线场景关闭预训练 BERT 以跳过权重下载）。"""
    text_encoder = BertTextEncoder(
        use_pretrained=use_pretrained_bert, freeze=freeze_bert
    )
    audio_encoder = CNNLSTMAudioEncoder()
    model = MultimodalFusionModel(
        num_emotions=cfg.num_emotions,
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        use_visual=False,
    )
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    cfg: ModuleConfig,
    extra: Optional[dict] = None,
) -> None:
    """保存训练 checkpoint（模型权重 + 优化器状态 + 元信息）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "num_emotions": cfg.num_emotions,
        "emotion_labels": cfg.emotion_labels,
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[train] checkpoint saved -> {path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train multimodal emotion recognition model"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config yaml (default: <project_root>/config.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--mock", action="store_true",
        help="Run with synthetic mock data (no real data / BERT weights needed)",
    )
    parser.add_argument(
        "--use-pretrained-bert", action="store_true",
        help="Load pretrained bert-base-uncased weights (requires network/cache)",
    )
    parser.add_argument(
        "--freeze-bert", action="store_true",
        help="Freeze BERT backbone parameters during training",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Directory to save checkpoints (default: <config>.checkpoint_root)",
    )
    return parser.parse_args()


def main() -> None:
    """训练主流程。"""
    args: argparse.Namespace = parse_args()

    cfg: ModuleConfig = load_config(args.config)
    set_seed(cfg.seed)
    device: torch.device = resolve_device(cfg.device)
    print(f"[train] device = {device}, emotions = {cfg.num_emotions}")

    # mock 场景强制关闭预训练 BERT，避免触发权重下载。
    use_pretrained: bool = args.use_pretrained_bert and not args.mock

    model: MultimodalFusionModel = build_model(
        cfg, use_pretrained_bert=use_pretrained, freeze_bert=args.freeze_bert
    )
    model = model.to(device)
    print(f"[train] trainable params = {model.trainable_parameter_count()}")

    if args.mock:
        train_loader, val_loader = build_mock_loaders(
            args.batch_size, cfg.num_emotions, cfg.seed
        )
    else:
        train_loader, val_loader = build_real_loaders(cfg, args.batch_size)

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )

    checkpoint_dir: Path = (
        args.checkpoint_dir if args.checkpoint_dir is not None else cfg.checkpoint_root
    )
    best_acc: float = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"[train] Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # 验证集准确率提升时保存最佳模型。
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model, optimizer, epoch, best_acc, cfg,
                extra={"use_pretrained_bert": use_pretrained},
            )

    # 训练结束保存最后一个 checkpoint，便于断点续训。
    save_checkpoint(
        checkpoint_dir / "last_model.pt",
        model, optimizer, args.epochs, best_acc, cfg,
        extra={"use_pretrained_bert": use_pretrained},
    )
    print(f"[train] done. best val acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
