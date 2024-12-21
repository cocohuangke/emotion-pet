"""多模态情绪识别模型评估入口。

用法示例（在项目根目录执行）：:

    # 使用合成 mock 测试集评估（无需真实数据）
    python -m emotion_recognition.evaluate --checkpoint checkpoints/best_model.pt --mock

    # 使用真实测试集评估
    python -m emotion_recognition.evaluate \
        --checkpoint checkpoints/best_model.pt \
        --data ./data/raw/test.csv

说明
----
* 从 checkpoint 恢复模型结构与权重，在测试集上输出整体准确率与
  macro / weighted F1，并打印逐类别的 precision / recall / F1 报告。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import ModuleConfig, load_config
from .dataset import (
    MockMultimodalDataset,
    MultimodalEmotionDataset,
    make_collate_fn,
)
from .models.bert_text import BertTextEncoder
from .models.cnn_lstm_audio import CNNLSTMAudioEncoder
from .models.fusion import MultimodalFusionModel


def resolve_device(requested: str) -> torch.device:
    """解析评估设备，``cuda`` 不可用时回退到 ``cpu``。"""
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[MultimodalFusionModel, ModuleConfig, dict]:
    """从 checkpoint 恢复模型与配置元信息。

    Parameters
    ----------
    checkpoint_path : Path
        checkpoint 文件路径。
    device : torch.device
        目标设备。

    Returns
    -------
    Tuple[MultimodalFusionModel, ModuleConfig, dict]
        ``(模型, 配置, 元信息字典)``。
    """
    ckpt: dict = torch.load(checkpoint_path, map_location=device)

    num_emotions: int = int(ckpt.get("num_emotions", 7))
    emotion_labels: list = list(ckpt.get("emotion_labels", []))
    if not emotion_labels:
        emotion_labels = [str(i) for i in range(num_emotions)]

    cfg: ModuleConfig = ModuleConfig(
        data={"emotion_labels": emotion_labels, "num_emotions": num_emotions}
    )

    # 训练时是否用了预训练 BERT 决定重建路径（两种路径的 state_dict 键不同）。
    use_pretrained: bool = bool(ckpt.get("use_pretrained_bert", False))
    text_encoder = BertTextEncoder(use_pretrained=use_pretrained, freeze=False)
    audio_encoder = CNNLSTMAudioEncoder()
    model = MultimodalFusionModel(
        num_emotions=num_emotions,
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        use_visual=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, cfg, ckpt


def build_test_loader(
    cfg: ModuleConfig,
    data_path: Optional[Path],
    batch_size: int,
) -> DataLoader:
    """构造测试数据加载器（``data_path`` 为 CSV 或 mock 模式）。"""
    collate = make_collate_fn(tokenizer=None)
    if data_path is None:
        # mock 测试集。
        test_ds: Dataset = MockMultimodalDataset(
            num_samples=batch_size * 4, num_emotions=cfg.num_emotions, seed=123
        )
    else:
        csv_path: Path = data_path
        if csv_path.is_dir():
            csv_path = csv_path / "raw" / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"test csv not found: {csv_path}")
        import pandas as pd

        df = pd.read_csv(csv_path)
        test_ds = MultimodalEmotionDataset(
            df, cfg.emotion_labels, cfg.data_root
        )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    emotion_labels: list,
) -> dict:
    """在测试集上计算整体指标与逐类别报告。

    Returns
    -------
    dict
        包含 ``accuracy`` / ``macro_f1`` / ``weighted_f1`` 以及
        ``per_class``（逐类别 precision/recall/f1）的字典。
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )

    all_preds: list = []
    all_labels: list = []

    for input_ids, attention_mask, audio, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        audio = audio.to(device)

        logits: torch.Tensor = model(input_ids, audio, attention_mask=attention_mask)
        preds: torch.Tensor = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    y_true: np.ndarray = np.asarray(all_labels)
    y_pred: np.ndarray = np.asarray(all_preds)

    report: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    report["per_class"] = classification_report(
        y_true,
        y_pred,
        target_names=emotion_labels,
        zero_division=0,
        digits=4,
    )
    return report


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Evaluate multimodal emotion recognition model"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to checkpoint (.pt) file",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to test CSV (or data dir); omit to use mock test data",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic mock test data (no real data needed)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config yaml (used only for device preference)",
    )
    return parser.parse_args()


def main() -> None:
    """评估主流程。"""
    args: argparse.Namespace = parse_args()

    checkpoint_path: Path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    base_cfg: ModuleConfig = load_config(args.config)
    device: torch.device = resolve_device(base_cfg.device)

    model, cfg, meta = load_model_from_checkpoint(checkpoint_path, device)
    print(
        f"[evaluate] checkpoint epoch = {meta.get('epoch', '?')}, "
        f"best_acc = {meta.get('best_acc', '?')}"
    )
    print(f"[evaluate] device = {device}, emotions = {cfg.num_emotions}")

    loader: DataLoader = build_test_loader(
        cfg, None if args.mock else args.data, args.batch_size
    )
    report: dict = evaluate(model, loader, device, cfg.emotion_labels)

    print("\n================ Evaluation Report ================")
    print(f"  Accuracy      : {report['accuracy']:.4f}")
    print(f"  Macro F1      : {report['macro_f1']:.4f}")
    print(f"  Weighted F1   : {report['weighted_f1']:.4f}")
    print("--------------------------------------------------")
    print("  Per-class report:")
    print(report["per_class"])
    print("==================================================")


if __name__ == "__main__":
    main()
