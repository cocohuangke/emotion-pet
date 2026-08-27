"""多模态融合情绪分类模型（特征层融合）。

* 文本分支：BERT 输出 768 维文本特征；
* 语音分支：CNN+LSTM 输出 128 维语音特征；
* 视觉分支：视觉特征 512 维（可选，``use_visual=False`` 时不构造，
  避免产生死参数与零 logits 拼接浪费融合层容量）；
* 融合：文本 + 语音（+ 可选视觉）特征在特征维拼接 -> MLP（``256`` 隐藏单元）
  -> ``num_emotions`` 输出最终分类 logits。

相较早期「logits 级融合」（三路 ``num_emotions`` 维 logits 直接拼接再过一个
``3*num_emotions -> num_emotions`` 线性层），特征层融合保留了更丰富的跨模态
信息，让融合网络能学习模态间非线性交互，是更强、更常见的多模态融合范式。

单模态评估接口
---------------
``text_fc`` / ``audio_fc`` 仍保留为单模态分类头，供
``forward_branch_logits`` 返回各分支独立 logits（消融实验、``--modality`` 评估）。
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
# 融合 MLP 隐藏层维度。
FUSION_HIDDEN_DIM: int = 256


class MultimodalFusionModel(nn.Module):
    """三模态（文本 + 语音 + 视觉）特征层融合情绪分类模型。

    Parameters
    ----------
    num_emotions : int
        情绪类别数（本项目为 7）。
    text_encoder : BertTextEncoder
        文本编码器（输出 768 维）。
    audio_encoder : CNNLSTMAudioEncoder
        语音编码器（输出 128 维）。
    visual_dim : int
        视觉特征维度（默认 512）；仅 ``use_visual=True`` 时参与融合。
    use_visual : bool
        是否启用视觉分支（``True`` 时前向必须提供 ``visual_input``）。
    dropout : float
        融合 MLP 隐藏层的 dropout 比例。
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

        # 单模态分类头：供 forward_branch_logits 做消融/单模态评估。
        self.text_fc: nn.Module = nn.Linear(TEXT_FEATURE_DIM, num_emotions)
        self.audio_fc: nn.Module = nn.Linear(AUDIO_FEATURE_DIM, num_emotions)

        # 视觉分支仅在启用时构造，避免 use_visual=False 时产生死参数。
        self.visual_fc: Optional[nn.Module] = (
            nn.Linear(visual_dim, num_emotions) if use_visual else None
        )

        # 特征层融合 MLP：拼接各模态特征后投影到情绪空间。
        fusion_in: int = TEXT_FEATURE_DIM + AUDIO_FEATURE_DIM
        if use_visual:
            fusion_in += visual_dim
        self.fusion: nn.Module = nn.Sequential(
            nn.Linear(fusion_in, FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(FUSION_HIDDEN_DIM, num_emotions),
        )

    def _empty_text_mask(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """检测空文本行(attention_mask 全零或 input_ids 全为 PAD)。

        Returns
        -------
        torch.Tensor
            形状 ``(batch_size,)`` 的布尔张量;``True`` 表示该行为空文本
            (应跳过 BERT 前向,走纯音频路径)。
        """
        if attention_mask is not None:
            return attention_mask.sum(dim=1) == 0
        # 无 mask 时退化为检查 input_ids 是否全为 PAD(0)。
        return (input_ids != 0).sum(dim=1) == 0

    def _text_feature(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算文本特征；空文本行以零张量占位（避免空序列梯度污染）。"""
        batch_size: int = input_ids.size(0)
        empty_text: torch.Tensor = self._empty_text_mask(input_ids, attention_mask)
        if bool(empty_text.logical_not().any().item()):
            text_feat: torch.Tensor = self.text_encoder(
                input_ids, attention_mask=attention_mask
            )
            # 空文本行的特征显式置零，让融合网络学到「无文本」信号。
            if bool(empty_text.any().item()):
                text_feat = torch.where(
                    empty_text.unsqueeze(-1),
                    torch.zeros_like(text_feat),
                    text_feat,
                )
            return text_feat
        return input_ids.new_zeros(
            batch_size, self.text_encoder.output_dim, dtype=torch.float32
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播，输出最终情绪分类 logits。

        空文本路由
        -----------
        当某行的 ``attention_mask`` 全零(空文本,如音频-only 样本)时，其文本
        特征以零张量占位，融合网络据此学习「无文本」先验，不再需要绕过
        fusion 层的特殊分支。

        Parameters
        ----------
        input_ids : torch.Tensor
            文本词表索引，形状 ``(batch_size, seq_len)``。
        audio_input : Optional[torch.Tensor]
            MFCC 序列，形状 ``(batch_size, time, 80)``；为 ``None`` 时
            语音特征以零占位(纯文本情感分析场景)。
        attention_mask : Optional[torch.Tensor]
            文本注意力掩码，形状与 ``input_ids`` 一致。
        visual_input : Optional[torch.Tensor]
            视觉特征，形状 ``(batch_size, visual_dim)``；为 ``None`` 时
            视觉特征以零占位。

        Returns
        -------
        torch.Tensor
            形状 ``(batch_size, num_emotions)`` 的最终 logits。
        """
        batch_size: int = input_ids.size(0)
        text_feat: torch.Tensor = self._text_feature(input_ids, attention_mask)

        # 语音特征：128 维；缺省时零占位。
        if audio_input is not None:
            audio_feat: torch.Tensor = self.audio_encoder(audio_input)
        else:
            audio_feat = text_feat.new_zeros(batch_size, AUDIO_FEATURE_DIM)

        # 特征层融合：拼接各模态特征 -> MLP -> 最终 logits。
        feats: List[torch.Tensor] = [text_feat, audio_feat]
        if self.use_visual:
            if visual_input is not None:
                feats.append(visual_input)
            else:
                feats.append(text_feat.new_zeros(batch_size, self.visual_dim))
        combined: torch.Tensor = torch.cat(feats, dim=1)
        return self.fusion(combined)

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
            ``(final_logits, {"text": ..., "audio": ..., "visual": ...})``，
            其中 ``text``/``audio``/``visual`` 为各分支特征向量。
        """
        text_feat: torch.Tensor = self._text_feature(input_ids, attention_mask)
        if audio_input is not None:
            audio_feat: torch.Tensor = self.audio_encoder(audio_input)
        else:
            audio_feat = text_feat.new_zeros(input_ids.size(0), AUDIO_FEATURE_DIM)

        feats: List[torch.Tensor] = [text_feat, audio_feat]
        visual_feat: Optional[torch.Tensor] = visual_input
        if self.use_visual:
            visual_feat = (
                visual_input
                if visual_input is not None
                else text_feat.new_zeros(input_ids.size(0), self.visual_dim)
            )
            feats.append(visual_feat)

        combined: torch.Tensor = torch.cat(feats, dim=1)
        final_logits: torch.Tensor = self.fusion(combined)

        features: Dict[str, torch.Tensor] = {
            "text": text_feat,
            "audio": audio_feat,
            "visual": (
                visual_feat
                if visual_feat is not None
                else text_feat.new_zeros(input_ids.size(0), self.visual_dim)
            ),
        }
        return final_logits, features

    def forward_branch_logits(
        self,
        input_ids: torch.Tensor,
        audio_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(text_logit, audio_logit)``，跳过融合层，用于单模态评估。

        空文本行的 ``text_logit`` 置零(避免空序列 BERT 输出污染单模态评估)。

        Parameters
        ----------
        input_ids : torch.Tensor
            文本词表索引，形状 ``(batch_size, seq_len)``。
        audio_input : Optional[torch.Tensor]
            MFCC 序列，形状 ``(batch_size, time, 80)``；为 ``None`` 时
            语音分支以零 logits 占位（纯文本场景）。
        attention_mask : Optional[torch.Tensor]
            文本注意力掩码，形状与 ``input_ids`` 一致。

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(text_logit, audio_logit)``，形状均为 ``(batch_size, num_emotions)``。
        """
        batch_size: int = input_ids.size(0)
        text_feat: torch.Tensor = self._text_feature(input_ids, attention_mask)
        text_logit: torch.Tensor = self.text_fc(text_feat)

        # 语音分支：128 -> num_emotions；缺省时零占位。
        if audio_input is not None:
            audio_feat: torch.Tensor = self.audio_encoder(audio_input)
            audio_logit: torch.Tensor = self.audio_fc(audio_feat)
        else:
            audio_logit = text_logit.new_zeros(batch_size, self.num_emotions)

        return text_logit, audio_logit

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
    dummy_audio = torch.randn(4, 100, 80)

    logits, feats = model.forward_with_branch_features(
        dummy_ids, dummy_audio, attention_mask=dummy_mask
    )
    print(f"[fusion] logits shape = {tuple(logits.shape)}")  # (4, 7)
    print(f"[fusion] branch feats = { {k: tuple(v.shape) for k, v in feats.items()} }")
    print(f"[fusion] trainable params = {model.trainable_parameter_count()}")
