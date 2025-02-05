"""BERT 文本情感编码器。

使用 ``bert-base-uncased`` 作为文本语义提取骨干，取 ``pooler_output`` 作为 
768 维句子级特征，后续由融合模块的 ``text_fc`` 投影到情绪空间。

设计要点
--------
* 支持 ``freeze`` 冻结 BERT 参数（微调仅训练下游分类头，节省显存、加速收敛）。
* 提供 ``use_pretrained`` 离线兜底：当设为 ``False`` 时不再加载预训练权重，
  而是使用一个随机初始化的词嵌入 + 平均池化 + 线性投影，输出维度仍为 768，
  便于在没有网络 / 权重缓存的场景下跑通训练与评估流程。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# bert-base-uncased 的固定超参数（离线兜底路径复用，保证输出维度一致）。
BERT_HIDDEN_SIZE: int = 768
BERT_VOCAB_SIZE: int = 30522
BERT_PAD_ID: int = 0


class BertTextEncoder(nn.Module):
    """BERT 文本编码器，输出 768 维 pooled 句向量。

    Parameters
    ----------
    model_name : str
        HuggingFace 预训练模型名（默认 ``bert-base-uncased``）。
    freeze : bool
        是否冻结 BERT 骨干参数（``True`` 时 BERT 不参与梯度更新）。
    use_pretrained : bool
        是否加载预训练权重；``False`` 时使用随机嵌入兜底编码器。
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = True,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name: str = model_name
        self.freeze: bool = freeze
        self.use_pretrained: bool = use_pretrained
        self.hidden_size: int = BERT_HIDDEN_SIZE

        if use_pretrained:
            # 延迟导入：仅在真正需要预训练权重时才引入 transformers。
            from transformers import BertModel

            self.bert: nn.Module = BertModel.from_pretrained(model_name)
            self.hidden_size = int(self.bert.config.hidden_size)

            if freeze:
                # 冻结骨干，仅保留池化/分类层的可训练性。
                for param in self.bert.parameters():
                    param.requires_grad = False
        else:
            # 离线兜底：随机词嵌入 + 平均池化 + 投影，保持 768 维输出契约。
            self.bert: Optional[nn.Module] = None
            self.embedding: nn.Module = nn.Embedding(
                BERT_VOCAB_SIZE, BERT_HIDDEN_SIZE, padding_idx=BERT_PAD_ID
            )
            self.projection: nn.Module = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播，返回 768 维句子级文本特征。

        Parameters
        ----------
        input_ids : torch.Tensor
            词表索引张量，形状 ``(batch_size, seq_len)``。
        attention_mask : Optional[torch.Tensor]
            注意力掩码，形状与 ``input_ids`` 一致；仅预训练路径使用。

        Returns
        -------
        torch.Tensor
            形状 ``(batch_size, 768)`` 的 pooled 文本特征。
        """
        if self.use_pretrained:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # pooler_output: [CLS] 经过线性 + tanh 后的句子表征，维度 768。
            return outputs.pooler_output

        # 兜底路径：带掩码的平均池化（忽略 padding 位置）。
        emb: torch.Tensor = self.embedding(input_ids)  # (B, L, 768)
        mask: torch.Tensor = (input_ids != BERT_PAD_ID).unsqueeze(-1).float()  # (B, L, 1)
        pooled: torch.Tensor = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.tanh(self.projection(pooled))
        return pooled

    @property
    def output_dim(self) -> int:
        """文本特征输出维度（恒为 768）。"""
        return self.hidden_size

    def trainable_parameter_count(self) -> int:
        """返回可训练参数数量（便于日志记录与实验报告）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 快速自检：构造离线编码器，喂入随机词表索引验证形状。
    encoder = BertTextEncoder(use_pretrained=False, freeze=False)
    dummy_ids = torch.randint(1, 1000, (4, 32))  # (batch=4, seq=32)
    dummy_mask = (dummy_ids != 0).long()
    out: torch.Tensor = encoder(dummy_ids, attention_mask=dummy_mask)
    print(f"[bert_text] output shape = {tuple(out.shape)}")  # (4, 768)
    print(f"[bert_text] trainable params = {encoder.trainable_parameter_count()}")
