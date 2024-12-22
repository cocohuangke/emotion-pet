"""tests/test_emotion_recognition.py - 多模态情感识别模型单元测试。

覆盖：
    - BertTextEncoder 的 use_pretrained=False 回退路径（离线可跑）；
    - CNNLSTMAudioEncoder 前向输出形状；
    - MultimodalFusionModel 三分支融合前向。
"""

from __future__ import annotations

import pytest


# ----------------------------------------------------------------------
# BertTextEncoder (回退路径)
# ----------------------------------------------------------------------
class TestBertTextEncoder:
    """BERT 文本编码器（mock 模式，不下载预训练权重）。"""

    def test_forward_output_shape(self) -> None:
        import torch

        from emotion_recognition.models.bert_text import BertTextEncoder

        encoder = BertTextEncoder(use_pretrained=False, freeze=False)
        input_ids = torch.randint(0, 30522, (4, 32))
        mask = torch.ones(4, 32, dtype=torch.long)
        out = encoder(input_ids, attention_mask=mask)
        assert out.shape == (4, 768)

    def test_attention_mask_optional(self) -> None:
        import torch

        from emotion_recognition.models.bert_text import BertTextEncoder

        encoder = BertTextEncoder(use_pretrained=False)
        input_ids = torch.randint(0, 30522, (2, 16))
        out = encoder(input_ids)
        assert out.shape == (2, 768)


# ----------------------------------------------------------------------
# CNNLSTMAudioEncoder
# ----------------------------------------------------------------------
class TestCNNLSTMAudioEncoder:
    """CNN+LSTM 语音编码器。"""

    def test_forward_output_shape(self) -> None:
        import torch

        from emotion_recognition.models.cnn_lstm_audio import CNNLSTMAudioEncoder

        encoder = CNNLSTMAudioEncoder()
        # (batch, time, mfcc_dim=40)
        audio = torch.randn(4, 100, 40)
        out = encoder(audio)
        assert out.shape[0] == 4
        assert out.shape[1] == 128  # 输出维度


# ----------------------------------------------------------------------
# MultimodalFusionModel
# ----------------------------------------------------------------------
class TestMultimodalFusionModel:
    """三分支融合模型。"""

    def test_forward_text_only(self) -> None:
        """仅文本分支（audio/visual 为 None）时仍能前向。"""
        import torch

        from emotion_recognition import MultimodalFusionModel
        from emotion_recognition.models.bert_text import BertTextEncoder

        model = MultimodalFusionModel(
            num_emotions=7,
            text_encoder=BertTextEncoder(use_pretrained=False),
        )
        input_ids = torch.randint(0, 30522, (4, 32))
        mask = torch.ones(4, 32, dtype=torch.long)
        logits = model(input_ids, audio_input=None, attention_mask=mask)
        assert logits.shape == (4, 7)

    def test_forward_multimodal(self) -> None:
        """文本 + 语音双模态前向。"""
        import torch

        from emotion_recognition import MultimodalFusionModel
        from emotion_recognition.models.bert_text import BertTextEncoder

        model = MultimodalFusionModel(
            num_emotions=7,
            text_encoder=BertTextEncoder(use_pretrained=False),
        )
        input_ids = torch.randint(0, 30522, (4, 32))
        mask = torch.ones(4, 32, dtype=torch.long)
        audio = torch.randn(4, 100, 40)
        logits = model(input_ids, audio_input=audio, attention_mask=mask)
        assert logits.shape == (4, 7)

    def test_forward_batch1(self) -> None:
        import torch

        from emotion_recognition import MultimodalFusionModel
        from emotion_recognition.models.bert_text import BertTextEncoder

        model = MultimodalFusionModel(
            num_emotions=7,
            text_encoder=BertTextEncoder(use_pretrained=False),
        )
        input_ids = torch.randint(0, 30522, (1, 16))
        logits = model(input_ids, audio_input=None)
        assert logits.shape == (1, 7)

    def test_logits_finite(self) -> None:
        """输出为有限实数，无 NaN/Inf。"""
        import torch

        from emotion_recognition import MultimodalFusionModel
        from emotion_recognition.models.bert_text import BertTextEncoder

        model = MultimodalFusionModel(
            num_emotions=7,
            text_encoder=BertTextEncoder(use_pretrained=False),
        )
        input_ids = torch.randint(0, 30522, (2, 16))
        logits = model(input_ids, audio_input=None)
        assert torch.isfinite(logits).all()
