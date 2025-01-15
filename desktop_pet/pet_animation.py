"""PetAnimation —— 宠物动画引擎（帧循环 + 位置/缩放动画）。

基于 QPropertyAnimation 驱动宠物在桌面上的两套动画：
    - idle bobbing：上下缓慢浮动（营造「活着」的观感）；
    - pulse：交互/情绪切换时的缩放脉冲（放大再回弹）。

帧管理：通过 QTimer 循环切换指定情绪对应的动画帧；帧贴图优先从
``assets/`` 目录加载，缺失时回退到 EmotionDisplay 生成的程序化占位图。
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from . import _qt_compat as qt

# PyQt5 缺失时的降级基类（保证 `from desktop_pet import PetAnimation` 不报错）
_AnimBase = qt.QObject if qt.HAS_PYQT5 else object

# ---------------------------------------------------------------------------
# 动画参数
# ---------------------------------------------------------------------------
IDLE_BOB_AMPLITUDE: int = 10          # 浮动幅度（像素）
IDLE_BOB_DURATION: int = 2200         # 单程时长（毫秒）
PULSE_SCALE_FACTOR: float = 1.15      # 缩放脉冲峰值倍率
PULSE_DURATION: int = 260             # 单程时长（毫秒）
FRAME_INTERVAL: int = 240             # 帧切换间隔（毫秒）

# assets 下每个情绪的子目录命名（占位图约定见 assets/README.md）
ASSET_DIR_NAME: str = "assets"


class PetAnimation(_AnimBase):
    """管理宠物 widget 的动画状态与帧循环。"""

    def __init__(
        self,
        target: Optional["qt.QWidget"] = None,
        on_frame: Optional[Callable[["qt.QPixmap"], None]] = None,
    ) -> None:
        if not qt.HAS_PYQT5:
            raise qt.missing_error("PetAnimation")
        super().__init__(target)

        self._target = target
        self._on_frame = on_frame

        # 帧管理：情绪 → 帧列表
        self._frames: Dict[str, List["qt.QPixmap"]] = {}
        self._current_emotion: str = "neutral"
        self._frame_index: int = 0

        # 帧循环定时器
        self._frame_timer = qt.QTimer(self)
        self._frame_timer.timeout.connect(self._advance_frame)

        # 动画对象（惰性创建，避免空 target 时无效）
        self._bob: Optional["qt.QPropertyAnimation"] = None
        self._pulse: Optional["qt.QPropertyAnimation"] = None

    # ------------------------------------------------------------------
    # 帧管理
    # ------------------------------------------------------------------
    def set_frames(self, emotion: str, frames: List["qt.QPixmap"]) -> None:
        """设置指定情绪的动画帧序列并立即显示首帧。

        Args:
            emotion: 情绪标签。
            frames: 帧列表；多于 1 帧时启动帧循环，否则仅静态显示。
        """
        self._current_emotion = emotion
        self._frames[emotion] = list(frames)
        self._frame_index = 0
        if frames:
            self._apply(frames[0])
        if len(frames) > 1:
            self._frame_timer.start(FRAME_INTERVAL)
        else:
            self._frame_timer.stop()

    def set_single_frame(self, pixmap: "qt.QPixmap") -> None:
        """显示单张静态贴图（停止帧循环）。"""
        self._frame_timer.stop()
        self._apply(pixmap)

    def _advance_frame(self) -> None:
        """切换到下一帧（循环）。"""
        frames = self._frames.get(self._current_emotion, [])
        if not frames:
            return
        self._frame_index = (self._frame_index + 1) % len(frames)
        self._apply(frames[self._frame_index])

    def _apply(self, pixmap: "qt.QPixmap") -> None:
        """将贴图应用到目标（优先走回调，其次调用 target.set_pet_pixmap）。"""
        if self._on_frame is not None:
            self._on_frame(pixmap)
        elif self._target is not None and hasattr(self._target, "set_pet_pixmap"):
            self._target.set_pet_pixmap(pixmap)

    # ------------------------------------------------------------------
    # idle bobbing（上下浮动）
    # ------------------------------------------------------------------
    def start_idle_bob(self, amplitude: int = IDLE_BOB_AMPLITUDE) -> None:
        """启动 idle 浮动动画（以当前窗口位置为基准无限循环）。"""
        if self._target is None:
            return
        base = self._target.pos()
        self._bob = qt.QPropertyAnimation(self._target, b"pos", self)
        self._bob.setDuration(IDLE_BOB_DURATION)
        self._bob.setStartValue(base)
        self._bob.setKeyValueAt(0.5, base + qt.QPoint(0, -amplitude))
        self._bob.setEndValue(base)
        self._bob.setLoopCount(-1)
        self._bob.setEasingCurve(qt.QEasingCurve.InOutSine)
        self._bob.start()

    def stop_idle_bob(self) -> None:
        """停止 idle 浮动动画。"""
        if self._bob is not None:
            self._bob.stop()
            self._bob = None

    # ------------------------------------------------------------------
    # 缩放脉冲
    # ------------------------------------------------------------------
    def pulse(
        self,
        factor: float = PULSE_SCALE_FACTOR,
        duration: int = PULSE_DURATION,
    ) -> None:
        """触发一次缩放脉冲（放大再回弹），结束后自动恢复浮动。"""
        if self._target is None:
            return
        self.stop_idle_bob()

        geometry = self._target.geometry()
        center = geometry.center()
        scaled = qt.QRect(
            0, 0,
            int(geometry.width() * factor),
            int(geometry.height() * factor),
        )
        scaled.moveCenter(center)

        self._pulse = qt.QPropertyAnimation(self._target, b"geometry", self)
        self._pulse.setDuration(duration * 2)
        self._pulse.setStartValue(geometry)
        self._pulse.setKeyValueAt(0.5, scaled)
        self._pulse.setEndValue(geometry)
        self._pulse.setEasingCurve(qt.QEasingCurve.OutInQuad)
        self._pulse.finished.connect(self.start_idle_bob)  # 结束后恢复浮动
        self._pulse.start()

    # ------------------------------------------------------------------
    # 资源加载（占位图约定）
    # ------------------------------------------------------------------
    def load_asset_frames(
        self, emotion: str, asset_root: str, n_frames: int = 4
    ) -> List["qt.QPixmap"]:
        """从 ``asset_root/emotion/`` 目录加载帧贴图。

        文件命名约定：``frame_{i:02d}.png``（见 assets/README.md）。
        缺失文件会被跳过；若一张都未加载则返回空列表，由调用方回退到
        程序化占位图。
        """
        frames: List["qt.QPixmap"] = []
        directory = os.path.join(asset_root, emotion)
        if not os.path.isdir(directory):
            return frames
        for index in range(n_frames):
            path = os.path.join(directory, f"frame_{index:02d}.png")
            if not os.path.isfile(path):
                continue
            pixmap = qt.QPixmap(path)
            if not pixmap.isNull():
                frames.append(pixmap)
        return frames

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """停止所有动画与帧循环。"""
        self._frame_timer.stop()
        self.stop_idle_bob()
        if self._pulse is not None:
            self._pulse.stop()
            self._pulse = None
