"""情感识别模型子包"""
from .bert_text import BertTextEncoder
from .cnn_lstm_audio import CNNLSTMAudioEncoder
from .fusion import MultimodalFusionModel

__all__ = ["BertTextEncoder", "CNNLSTMAudioEncoder", "MultimodalFusionModel"]
