"""多模态融合情绪分类模型。

对应论文 3.2 节的 ``EmotionRecognitionModel``。论文给出的示例代码存在
全角标点（中文逗号/括号/冒号）混入导致的语法错误，本实现保持其架构语义
不变并修复语法问题：

* 文本分支：BERT 输出 768 维 -> ``text_fc`` 投影到 ``num_emotions``；
* 语音分支：CNN+LSTM 输出 128 维 -> ``audio_fc`` 投影到 ``num_emotions``；
* 视觉分支：视觉特征 512 维 -> ``visual_fc`` 投影到 ``num_emotions``
  （视觉输入可缺省，此时以零 logits 占位，便于纯文本+语音场景运行）；
* 三路 ``num_emotions`` 维 logits 在特征维拼接 -> ``fusion`` 全连接
  （``num_emotions * 3 -> num_emotions``）输出最终分类 logits。

与论文的差异（有意为之）：
    * 论文 ``forward`` 签名顺序为 ``(visual, audio, text)``，本实现改为
      ``(input_ids, audio, attention_mask, visual)``，与数据集
      ``(text, audio, label)`` 的返回顺序一致，避免调用侧错位。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .bert_text import BertTextEncoder
from .cnn_lstm_audio import CNNLSTMAudioEncoder

# 各模态特征维度（与论文 3.2 节一致）。
TEXT_FEATURE_DIM: int = 768
AUDIO_FEATURE_DIM: int = 128
VISUAL_FEATURE_DIM: int = 512


class MultimodalFusionModel(nn.Module):
    """三模态（文本 + 语音 + 视觉）融合情绪分类模型。

    Parameters
    ----------
    num_emotions : int
        情绪类别数（本项目为 7）。
    text_encoder : BertTextEncoder
        文本编码器（输出 768 维）。
    audio_encoder : CNNLSTMAudioEncoder
        语音编码器（输出 128 维）。
    visual_dim : int
        视觉特征维度（默认 512）；仅用于构造 ``visual_fc``。
    use_visual : bool
        是否启用视觉分支（``True`` 时前向必须提供 ``visual_input``）。
    dropout : float
        各分支 logits 拼接前的 dropout 比例。
    """

    def __init__(
        self,
        num_emotions: int,
        text_encoder: Optional[BertTextEncoder] = None,
        audio_encoder: Optional[CNNLSTMAudioEncoder] = None,
        visual_dim: int = VISUAL_FEATURE_DIM,
        use_visual: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_emotions: int = num_emotions
        self.visual_dim: int = visual_dim
        self.use_visual: bool = use_visual

        # 默认构造两个子编码器（外部也可传入自定义实例，便于微调复用）。
        self.text_encoder: BertTextEncoder = (
            text_encoder if text_encoder is not None else BertTextEncoder()
        )
        self.audio_encoder: CNNLSTMAudioEncoder = (
            audio_encoder if audio_encoder is not None else CNNLSTMAudioEncoder()
        )

        # 各分支投影到情绪空间的 FC 层（对齐论文 3.2 节）。
        self.text_fc: nn.Module = nn.Linear(TEXT_FEATURE_DIM, num_emotions)
        self.audio_fc: nn.Module = nn.Linear(AUDIO_FEATURE_DIM, num_emotions)
        self.visual_fc: nn.Module = nn.Linear(visual_dim, num_emotions)

        self.dropout: nn.Module = nn.Dropout(dropout)

        # 融合层：拼接三路 logits 后输出最终 logits。
        self.fusion: nn.Module = nn.Linear(num_emotions * 3, num_emotions)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播，输出最终情绪分类 logits。

        Parameters
        ----------
        input_ids : torch.Tensor
            文本词表索引，形状 ``(batch_size, seq_len)``。
        audio_input : Optional[torch.Tensor]
            MFCC 序列，形状 ``(batch_size, time, 40)``；为 ``None`` 时
            语音分支以零 logits 占位（纯文本情感分析场景）。
        attention_mask : Optional[torch.Tensor]
            文本注意力掩码，形状与 ``input_ids`` 一致。
        visual_input : Optional[torch.Tensor]
            视觉特征，形状 ``(batch_size, visual_dim)``；为 ``None`` 时
            视觉分支以零 logits 占位。

        Returns
        -------
        torch.Tensor
            形状 ``(batch_size, num_emotions)`` 的最终 logits。
        """
        batch_size: int = input_ids.size(0)

        # 文本分支：768 -> num_emotions。
        text_feat: torch.Tensor = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_logit: torch.Tensor = self.text_fc(text_feat)

        # 语音分支：128 -> num_emotions；缺省时零占位以维持拼接维度
        # （纯文本场景：用户未开启麦克风，仅依赖文本情感分析）。
        if audio_input is not None:
            audio_feat: torch.Tensor = self.audio_encoder(audio_input)
            audio_logit: torch.Tensor = self.audio_fc(audio_feat)
        else:
            audio_feat = text_feat.new_zeros(batch_size, AUDIO_FEATURE_DIM)
            audio_logit = text_logit.new_zeros(batch_size, self.num_emotions)

        # 视觉分支：512 -> num_emotions；缺省时零占位以维持拼接维度。
        if visual_input is not None:
            visual_logit: torch.Tensor = self.visual_fc(visual_input)
        else:
            visual_logit = text_logit.new_zeros(batch_size, self.num_emotions)

        # 三路 logits 拼接（对齐论文顺序：visual, audio, text）。
        combined: torch.Tensor = torch.cat(
            (visual_logit, audio_logit, text_logit), dim=1
        )  # (B, 3*num_emotions)
        combined = self.dropout(combined)

        final_logits: torch.Tensor = self.fusion(combined)  # (B, num_emotions)
        return final_logits

    def forward_with_branch_features(
        self,
        input_ids: torch.Tensor,
        audio_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播，额外返回各分支的中间特征（用于消融实验与可视化）。

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            ``(final_logits, {"text": ..., "audio": ..., "visual": ...})``。
        """
        text_feat: torch.Tensor = self.text_encoder(input_ids, attention_mask=attention_mask)
        if audio_input is not None:
            audio_feat: torch.Tensor = self.audio_encoder(audio_input)
        else:
            audio_feat = text_feat.new_zeros(input_ids.size(0), AUDIO_FEATURE_DIM)

        text_logit: torch.Tensor = self.text_fc(text_feat)
        audio_logit: torch.Tensor = self.audio_fc(audio_feat)

        if visual_input is not None:
            visual_logit: torch.Tensor = self.visual_fc(visual_input)
        else:
            visual_logit = text_logit.new_zeros(input_ids.size(0), self.num_emotions)

        combined: torch.Tensor = torch.cat(
            (visual_logit, audio_logit, text_logit), dim=1
        )
        final_logits: torch.Tensor = self.fusion(self.dropout(combined))

        features: Dict[str, torch.Tensor] = {
            "text": text_feat,
            "audio": audio_feat,
            "visual": visual_logit,
        }
        return final_logits, features

    @property
    def branch_dims(self) -> List[int]:
        """返回三模态特征维度列表，顺序为 ``[text, audio, visual]``。"""
        return [TEXT_FEATURE_DIM, AUDIO_FEATURE_DIM, self.visual_dim]

    def trainable_parameter_count(self) -> int:
        """返回可训练参数总量（用于实验记录）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 快速自检：离线构造模型（不加载 BERT 权重），验证前向形状与消融接口。
    model = MultimodalFusionModel(
        num_emotions=7,
        text_encoder=BertTextEncoder(use_pretrained=False, freeze=False),
        audio_encoder=CNNLSTMAudioEncoder(),
        use_visual=False,
    )
    dummy_ids = torch.randint(1, 1000, (4, 32))
    dummy_mask = (dummy_ids != 0).long()
    dummy_audio = torch.randn(4, 100, 40)

    logits, feats = model.forward_with_branch_features(
        dummy_ids, dummy_audio, attention_mask=dummy_mask
    )
    print(f"[fusion] logits shape = {tuple(logits.shape)}")  # (4, 7)
    print(f"[fusion] branch feats = { {k: tuple(v.shape) for k, v in feats.items()} }")
    print(f"[fusion] trainable params = {model.trainable_parameter_count()}")
