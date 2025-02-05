"""CNN + LSTM 语音情感编码器。

对 librosa 提取的 40 维 MFCC 序列 （形状 ``(batch, time, 40)``）
先用 1D 卷积提取局部频谱特征，再用双向 LSTM 建模时序上下文，最后取
末帧隐状态经全连接投影为 128 维语音情感特征向量。

设计要点
--------
* CNN 沿时间维做卷积 + 池化，逐层压缩时间分辨率、扩大感受野。
* BatchNorm + Dropout 提升训练稳定性并抑制过拟合。
* 双向 LSTM 的输出隐状态（前向 + 后向拼接）经 FC 压缩到 128 维，
  与融合模块约定的 ``audio_dim=128`` 对齐。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# 输入 MFCC 特征维度（librosa ``n_mfcc=40``）。
DEFAULT_MFCC_DIM: int = 40
# 输出语音特征维度（与论文 3.2 节 ``hidden_size=128`` 对齐）。
DEFAULT_OUTPUT_DIM: int = 128


class CNNLSTMAudioEncoder(nn.Module):
    """CNN + 双向 LSTM 语音编码器，输出 128 维音频特征。

    Parameters
    ----------
    input_dim : int
        输入 MFCC 特征维度（默认 40）。
    hidden_size : int
        LSTM 单向隐层维度（默认 128）。
    num_layers : int
        LSTM 层数（默认 2）。
    cnn_channels : int
        CNN 首层输出通道数（第二层翻倍）。
    dropout : float
        LSTM 层间与输出层的 dropout 比例。
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_MFCC_DIM,
        hidden_size: int = DEFAULT_OUTPUT_DIM,
        num_layers: int = 2,
        cnn_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.output_dim: int = hidden_size

        # 1D 卷积在时间维上提取局部频谱模式。
        # 输入 (B, 40, T) -> 卷积/池化后 (B, 2*cnn_channels, T/4)。
        self.cnn: nn.Module = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # 双向 LSTM：输入通道 = 2*cnn_channels，输出隐状态维度 = 2*hidden_size。
        self.lstm: nn.Module = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 双向隐状态拼接后投影到 128 维语音特征。
        self.fc: nn.Module = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """前向传播，返回 128 维语音情感特征。

        Parameters
        ----------
        mfcc : torch.Tensor
            MFCC 特征张量，形状 ``(batch_size, time, 40)``。

        Returns
        -------
        torch.Tensor
            形状 ``(batch_size, 128)`` 的语音特征向量。
        """
        # (B, T, 40) -> (B, 40, T) 以适配 Conv1d。
        x: torch.Tensor = mfcc.transpose(1, 2)
        x = self.cnn(x)  # (B, 2*cnn_channels, T')

        # (B, 2*cnn_channels, T') -> (B, T', 2*cnn_channels) 以适配 LSTM。
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)  # (B, T', 2*hidden_size)

        # 取末帧隐状态作为整段语音的时序汇总。
        last: torch.Tensor = lstm_out[:, -1, :]  # (B, 2*hidden_size)
        out: torch.Tensor = self.fc(self.dropout(last))  # (B, 128)
        return out

    @property
    def feature_dim(self) -> int:
        """语音特征输出维度（恒为 128）。"""
        return self.output_dim

    def trainable_parameter_count(self) -> int:
        """返回可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 快速自检：喂入随机 MFCC 序列，验证输出维度。
    encoder = CNNLSTMAudioEncoder()
    dummy_mfcc = torch.randn(4, 100, 40)  # (batch=4, time=100, 40)
    out: torch.Tensor = encoder(dummy_mfcc)
    print(f"[cnn_lstm_audio] output shape = {tuple(out.shape)}")  # (4, 128)
    print(f"[cnn_lstm_audio] trainable params = {encoder.trainable_parameter_count()}")
