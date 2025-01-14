"""PetWindow —— 透明无边框置顶的桌面宠物主窗口。

提供：
    - 透明背景（WA_TranslucentBackground）+ 无边框（FramelessWindowHint）
      + 置顶（WindowStaysOnTopHint）+ 隐藏任务栏图标（Tool）；
    - 鼠标左键拖拽（mousePressEvent / mouseMoveEvent / mouseReleaseEvent）；
    - 右键上下文菜单（退出 / 设置 / 对话）；
    - 屏幕可用区域右下角自动定位；
    - 宠物贴图绘制入口（set_pet_pixmap + paintEvent）。

所有 Qt 依赖通过 ``_qt_compat`` 集中管理；PyQt5 缺失时，本类仍可被
导入（降级为 object 子类），仅在实例化时抛出清晰错误。
"""

from __future__ import annotations

from typing import Callable, Optional

from . import _qt_compat as qt

# PyQt5 缺失时的降级基类（保证 `from desktop_pet import PetWindow` 不报错）
_WidgetBase = qt.QWidget if qt.HAS_PYQT5 else object


class PetWindow(_WidgetBase):
    """桌面宠物窗口。"""

    # 窗口默认尺寸（像素，正方形贴图区域）
    DEFAULT_SIZE: tuple = (220, 220)
    # 距屏幕边缘的边距（像素）
    CORNER_MARGIN: int = 24

    def __init__(self, parent: Optional["qt.QWidget"] = None) -> None:
        if not qt.HAS_PYQT5:
            raise qt.missing_error("PetWindow")
        super().__init__(parent)

        self._drag_offset: Optional["qt.QPoint"] = None
        self._current_pixmap: Optional["qt.QPixmap"] = None
        self._menu: Optional["qt.QMenu"] = None

        # 交互回调（由 main.py 注入；均为可选，未注入时回退默认行为）
        self.on_chat_callback: Optional[Callable[[], None]] = None
        self.on_settings_callback: Optional[Callable[[], None]] = None
        self.on_quit_callback: Optional[Callable[[], None]] = None
        self.on_drag_start_callback: Optional[Callable[[], None]] = None
        self.on_drag_end_callback: Optional[Callable[[], None]] = None

        self._setup_window_flags()
        self._setup_context_menu()

    # ------------------------------------------------------------------
    # 窗口初始化
    # ------------------------------------------------------------------
    def _setup_window_flags(self) -> None:
        """配置透明无边框置顶窗口。"""
        flags = (
            qt.Qt.FramelessWindowHint
            | qt.Qt.WindowStaysOnTopHint
            | qt.Qt.Tool  # Tool 使窗口不占用任务栏，更符合桌面宠物定位
        )
        self.setWindowFlags(flags)
        self.setAttribute(qt.Qt.WA_TranslucentBackground, True)
        self.setWindowTitle("Emotion Pet")
        self.resize(*self.DEFAULT_SIZE)
        self.setCursor(qt.Qt.OpenHandCursor)

    def _setup_context_menu(self) -> None:
        """构建右键上下文菜单（退出 / 设置 / 对话）。"""
        self._menu = qt.QMenu(self)
        act_chat = self._menu.addAction("对话")
        act_settings = self._menu.addAction("设置")
        self._menu.addSeparator()
        act_quit = self._menu.addAction("退出")
        act_chat.triggered.connect(self._trigger_chat)
        act_settings.triggered.connect(self._trigger_settings)
        act_quit.triggered.connect(self._trigger_quit)

    # ------------------------------------------------------------------
    # 贴图与绘制
    # ------------------------------------------------------------------
    def set_pet_pixmap(self, pixmap: "qt.QPixmap") -> None:
        """更新当前宠物贴图并触发重绘。"""
        self._current_pixmap = pixmap
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        """绘制宠物贴图（透明区域不绘制）。"""
        if self._current_pixmap is None:
            return
        painter = qt.QPainter(self)
        painter.setRenderHint(qt.QPainter.SmoothPixmapTransform)
        painter.drawPixmap(self.rect(), self._current_pixmap)
        painter.end()

    # ------------------------------------------------------------------
    # 定位
    # ------------------------------------------------------------------
    def move_to_corner(self) -> None:
        """将窗口移动到主屏幕可用区域的右下角。"""
        screen = qt.QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.right() - self.width() - self.CORNER_MARGIN
        y = geo.bottom() - self.height() - self.CORNER_MARGIN
        self.move(x, y)

    # ------------------------------------------------------------------
    # 拖拽
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        """记录拖拽偏移，准备移动窗口。"""
        if event.button() == qt.Qt.LeftButton:
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            self.setCursor(qt.Qt.ClosedHandCursor)
            if self.on_drag_start_callback is not None:
                self.on_drag_start_callback()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """按住左键时跟随鼠标移动窗口。"""
        if self._drag_offset is not None and (event.buttons() & qt.Qt.LeftButton):
            self.move(event.globalPos() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """结束拖拽。"""
        self._drag_offset = None
        self.setCursor(qt.Qt.OpenHandCursor)
        if self.on_drag_end_callback is not None:
            self.on_drag_end_callback()

    # ------------------------------------------------------------------
    # 右键菜单
    # ------------------------------------------------------------------
    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """弹出右键菜单。"""
        if self._menu is not None:
            self._menu.exec_(event.globalPos())

    def _trigger_chat(self) -> None:
        if self.on_chat_callback is not None:
            self.on_chat_callback()

    def _trigger_settings(self) -> None:
        if self.on_settings_callback is not None:
            self.on_settings_callback()

    def _trigger_quit(self) -> None:
        if self.on_quit_callback is not None:
            self.on_quit_callback()
        else:
            qt.QApplication.instance().quit()
