"""Qt 兼容层 —— 集中处理 PyQt5 导入与优雅降级。

设计目标：PyQt5 属于可选 GUI 依赖（requirements.txt 中声明），
在无头/无 GUI 环境下导入 ``desktop_pet`` 包不应在导入期崩溃。

策略：
    - 尝试导入 PyQt5 各符号；
    - 成功则置 ``HAS_PYQT5 = True`` 并暴露所有 Qt 名称；
    - 失败则置 ``HAS_PYQT5 = False`` 并保留原始异常，由上层
      组件在**实例化时**通过 ``missing_error()`` 抛出清晰错误。

上层组件通过 ``_WidgetBase = qt.QWidget if qt.HAS_PYQT5 else object``
的写法，在 PyQt5 缺失时仍能完成类定义，从而保证
``from desktop_pet import PetWindow`` 等导入链清晰、不报错。
"""

from __future__ import annotations

from typing import Optional

try:  # 尝试导入 PyQt5（可选 GUI 依赖）
    from PyQt5.QtCore import (
        QEasingCurve,
        QObject,
        QPoint,
        QPropertyAnimation,
        QRect,
        Qt,
        QTimer,
    )
    from PyQt5.QtGui import (
        QBrush,
        QColor,
        QFont,
        QLinearGradient,
        QPainter,
        QPen,
        QPixmap,
        QRadialGradient,
    )
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QInputDialog,
        QMenu,
        QMessageBox,
        QWidget,
    )

    HAS_PYQT5: bool = True
    IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _exc:  # pragma: no cover - 环境缺失分支
    HAS_PYQT5 = False
    IMPORT_ERROR = _exc


def missing_error(class_name: str) -> ImportError:
    """构造清晰的缺失依赖错误（供上层组件实例化时抛出）。"""
    return ImportError(
        f"{class_name} 依赖 PyQt5，但当前环境未安装 PyQt5。\n"
        f"请执行 `pip install 'PyQt5>=5.15.0'` 后重试。\n"
        f"原始导入错误: {IMPORT_ERROR!r}"
    )
