"""PyQt5 桌面宠物 GUI 模块

子模块：
    pet_window      - 透明置顶桌面窗口
    pet_animation   - 宠物动画帧管理
    emotion_display - 情绪→表情映射
    main            - GUI 入口
"""
from .pet_window import PetWindow
from .pet_animation import PetAnimation
from .emotion_display import EmotionDisplay

__all__ = ["PetWindow", "PetAnimation", "EmotionDisplay"]
