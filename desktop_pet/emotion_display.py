"""EmotionDisplay —— 情绪标签到宠物表情/配色的映射与渲染。

将情感识别模块输出的 7 类情绪标签映射为：
    - 主题色（用于占位贴图与样式表）；
    - 文字表情（占位表情，后续可替换为美术素材）。

并据此渲染宠物贴图（程序化占位图：渐变圆脸 + 表情文字 + 精力条 +
等级徽章）。美术素材替换方案见 ``assets/README.md``。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import _qt_compat as qt

# ---------------------------------------------------------------------------
# 情绪 → 主题色（HEX）
# ---------------------------------------------------------------------------
EMOTION_COLORS: Dict[str, str] = {
    "happy":    "#FFC93C",  # 暖黄
    "sad":      "#6C9BD1",  # 蓝
    "angry":    "#E25C5C",  # 红
    "fear":     "#9B7EDE",  # 紫
    "surprise": "#5CC8D7",  # 青
    "disgust":  "#7FB069",  # 绿
    "neutral":  "#B0B8C0",  # 灰
}

# ---------------------------------------------------------------------------
# 情绪 → 文字表情（占位表情符号）
# ---------------------------------------------------------------------------
EMOTION_EXPRESSIONS: Dict[str, str] = {
    "happy":    "^_^",
    "sad":      "T_T",
    "angry":    ">_<",
    "fear":     "O_O",
    "surprise": "●_●",
    "disgust":  ">_>",
    "neutral":  "-_-",
}

# ---------------------------------------------------------------------------
# 情绪 → 中文描述（供气泡/日志展示）
# ---------------------------------------------------------------------------
EMOTION_LABELS_ZH: Dict[str, str] = {
    "happy": "开心", "sad": "难过", "angry": "生气",
    "fear": "害怕", "surprise": "惊喜", "disgust": "厌恶", "neutral": "平静",
}


class EmotionDisplay:
    """情绪可视化渲染器（无状态，可复用）。"""

    def __init__(self, size: int = 200) -> None:
        self._size = size

    # ------------------------------------------------------------------
    # 映射查询
    # ------------------------------------------------------------------
    def color(self, emotion: str) -> str:
        """返回情绪对应的主题色（未知情绪回退 neutral）。"""
        return EMOTION_COLORS.get(self._resolve(emotion), EMOTION_COLORS["neutral"])

    def expression(self, emotion: str) -> str:
        """返回情绪对应的文字表情。"""
        return EMOTION_EXPRESSIONS.get(
            self._resolve(emotion), EMOTION_EXPRESSIONS["neutral"]
        )

    @staticmethod
    def label_zh(emotion: str) -> str:
        """返回情绪的中文描述。"""
        return EMOTION_LABELS_ZH.get(emotion, emotion)

    @staticmethod
    def _resolve(emotion: str) -> str:
        """未知情绪回退到 neutral。"""
        return emotion if emotion in EMOTION_EXPRESSIONS else "neutral"

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self, emotion: str, stats=None) -> "qt.QPixmap":
        """渲染单帧贴图。

        Args:
            emotion: 情绪标签。
            stats: 可选 PetStats，用于叠加精力条 / 等级徽章。

        Returns:
            程序化占位贴图 QPixmap。
        """
        if not qt.HAS_PYQT5:
            raise qt.missing_error("EmotionDisplay.render")
        return self._draw_face(emotion, stats, phase=0)

    def render_frames(
        self, emotion: str, stats=None, n_frames: int = 4
    ) -> List["qt.QPixmap"]:
        """渲染 ``n`` 帧带相位差异的贴图（用于 idle 动画循环）。"""
        if not qt.HAS_PYQT5:
            raise qt.missing_error("EmotionDisplay.render_frames")
        return [self._draw_face(emotion, stats, phase=i) for i in range(n_frames)]

    # ------------------------------------------------------------------
    # 内部绘制
    # ------------------------------------------------------------------
    def _draw_face(self, emotion: str, stats=None, phase: int = 0) -> "qt.QPixmap":
        """绘制单帧占位宠物脸。

        phase 引入轻微的上下位移，使帧序列产生动画差异。
        """
        size = self._size
        pixmap = qt.QPixmap(size, size)
        pixmap.fill(qt.Qt.transparent)

        painter = qt.QPainter(pixmap)
        painter.setRenderHint(qt.QPainter.Antialiasing)

        # phase 驱动的轻微纵向位移，形成动画帧差异
        dy = int(((phase % 4) - 1.5) * 2)  # 取值范围约 -3 ~ +3
        body_rect = qt.QRect(8, 8 + dy, size - 16, size - 16)

        # 渐变圆脸主体
        gradient = qt.QRadialGradient(body_rect.center(), size / 2)
        base_color = qt.QColor(self.color(emotion))
        gradient.setColorAt(0.0, base_color.lighter(120))
        gradient.setColorAt(1.0, base_color.darker(120))
        painter.setBrush(qt.QBrush(gradient))
        painter.setPen(qt.QPen(qt.QColor("#FFFFFF"), 2))
        painter.drawEllipse(body_rect)

        # 表情文字
        painter.setPen(qt.QPen(qt.QColor("#FFFFFF")))
        painter.setFont(qt.QFont("Segoe UI Emoji", size // 6))
        painter.drawText(body_rect, int(qt.Qt.AlignCenter), self.expression(emotion))

        # 状态叠加（精力条 + 等级徽章）
        if stats is not None:
            self._draw_status_overlay(painter, stats, size)

        painter.end()
        return pixmap

    def _draw_status_overlay(self, painter: "qt.QPainter", stats, size: int) -> None:
        """在贴图底部绘制精力条与等级文字。"""
        # 精力条底槽
        bar_rect = qt.QRect(int(size * 0.20), int(size * 0.84), int(size * 0.60), 6)
        painter.setBrush(qt.QBrush(qt.QColor("#3A3A3A")))
        painter.setPen(qt.Qt.NoPen)
        painter.drawRoundedRect(bar_rect, 3, 3)

        # 精力条填充（按 energy 百分比）
        energy_width = int(bar_rect.width() * (stats.energy / 100.0))
        painter.setBrush(qt.QBrush(qt.QColor("#7BD88F")))
        painter.drawRoundedRect(
            qt.QRect(bar_rect.x(), bar_rect.y(), energy_width, bar_rect.height()),
            3, 3,
        )

        # 等级文字
        painter.setPen(qt.QColor("#FFFFFF"))
        level_font = qt.QFont("Segoe UI", size // 18)
        level_font.setBold(True)
        painter.setFont(level_font)
        painter.drawText(
            qt.QRect(0, int(size * 0.90), size, int(size * 0.10)),
            int(qt.Qt.AlignCenter),
            f"Lv.{stats.level}",
        )

    # ------------------------------------------------------------------
    # 样式表输出（备选：供非贴图模式给 widget 上色）
    # ------------------------------------------------------------------
    def stylesheet(self, emotion: str) -> str:
        """返回与情绪对应的背景样式表。"""
        return f"background-color: {self.color(emotion)}; border-radius: 110px;"
