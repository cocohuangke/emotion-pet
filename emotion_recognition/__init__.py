"""多模态情感识别模块

子模块：
    models.bert_text       - BERT 文本情感分类分支
    models.cnn_lstm_audio  - CNN+LSTM 语音情感分支
    models.fusion          - 多模态融合分类头
    dataset                - 多模态数据集加载
    config                 - 模块配置加载
    train                  - 训练入口
    evaluate               - 评估入口
"""
from .models.bert_text import BertTextEncoder
from .models.cnn_lstm_audio import CNNLSTMAudioEncoder
from .models.fusion import MultimodalFusionModel
from .dataset import MultimodalEmotionDataset, MockMultimodalDataset, make_collate_fn
from .config import ModuleConfig, load_config

__all__ = [
    "BertTextEncoder",
    "CNNLSTMAudioEncoder",
    "MultimodalFusionModel",
    "MultimodalEmotionDataset",
    "MockMultimodalDataset",
    "make_collate_fn",
    "ModuleConfig",
    "load_config",
]
