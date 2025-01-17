"""main —— 桌面宠物 GUI 入口。

整合 growth_system 的成长逻辑与 desktop_pet 的 GUI 组件：
    - QApplication 启动；
    - PetStats + LevelSystem + BehaviorEngine 成长状态机；
    - EmotionDisplay + PetAnimation + PetWindow 渲染管线；
    - 定时器：每 5s 随机切换情绪，模拟情感识别输出的演示闭环；
    - 退出前将成长状态持久化到 SQLite（data/pet.db）。

运行方式（任选其一，均需已安装 PyQt5）：
    python -m desktop_pet.main
    python desktop_pet/main.py
"""

from __future__ import annotations

import os
import random
import sys
from typing import Optional

# 将项目根目录加入 sys.path，兼容 `python desktop_pet/main.py` 直接运行
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from desktop_pet import _qt_compat as qt  # noqa: E402
from desktop_pet.emotion_display import EmotionDisplay  # noqa: E402
from desktop_pet.pet_animation import PetAnimation  # noqa: E402
from desktop_pet.pet_window import PetWindow  # noqa: E402
from growth_system.behaviors import BehaviorEngine  # noqa: E402
from growth_system.level import LevelSystem  # noqa: E402
from growth_system.stats import PetStats  # noqa: E402

# 持久化数据库路径
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "pet.db")

# 演示情绪池（对齐 config.yaml 的 emotion_labels 7 类）
DEMO_EMOTIONS: tuple = (
    "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral",
)

# 定时器周期（毫秒）
EMOTION_INTERVAL_MS: int = 5000   # 每 5s 随机切换情绪（模拟情感识别输出）
STATS_TICK_MS: int = 30000        # 每 30s 结算一次成长（精力衰减 + 经验累积）


class PetController:
    """成长状态 + GUI 组件的粘合层（组合根）。"""

    def __init__(self) -> None:
        # ---- 成长系统状态机 ----
        self.stats = PetStats()
        self.level_system = LevelSystem(self.stats)
        self.behavior_engine = BehaviorEngine(seed=42)  # 可复现（对齐 config seed）

        # ---- GUI 渲染管线 ----
        self.display = EmotionDisplay(size=220)
        self.window = PetWindow()
        self.animation = PetAnimation(
            target=self.window, on_frame=self.window.set_pet_pixmap
        )

        self.current_emotion: str = "neutral"
        self._rng = random.Random(42)

        self._setup_callbacks()
        self._setup_timers()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _setup_callbacks(self) -> None:
        """注入菜单回调与拖拽时暂停/恢复浮动动画。"""
        self.window.on_chat_callback = self._on_chat
        self.window.on_settings_callback = self._on_settings
        self.window.on_quit_callback = self._on_quit
        self.window.on_drag_start_callback = self.animation.stop_idle_bob
        self.window.on_drag_end_callback = self.animation.start_idle_bob

    def _setup_timers(self) -> None:
        """注册情绪切换与成长结算定时器。"""
        self._emotion_timer = qt.QTimer(self.window)
        self._emotion_timer.timeout.connect(self._simulate_emotion)
        self._emotion_timer.start(EMOTION_INTERVAL_MS)

        self._stats_timer = qt.QTimer(self.window)
        self._stats_timer.timeout.connect(self._tick_stats)
        self._stats_timer.start(STATS_TICK_MS)

    def start(self) -> None:
        """显示窗口并进入 idle 状态。"""
        self.window.move_to_corner()
        self.window.show()
        self._apply_emotion("neutral")
        self.animation.start_idle_bob()

    # ------------------------------------------------------------------
    # 模拟情绪输入循环
    # ------------------------------------------------------------------
    def _simulate_emotion(self) -> None:
        """随机切换情绪，模拟情感识别模块的输出流。"""
        emotion = self._rng.choice(DEMO_EMOTIONS)
        self._apply_emotion(emotion)

    def _apply_emotion(self, emotion: str) -> None:
        """根据情绪更新行为决策与动画帧，构成完整反馈闭环。"""
        self.current_emotion = emotion
        behavior, dialogue = self.behavior_engine.select_behavior(emotion, self.stats)

        frames = self.display.render_frames(emotion, self.stats, n_frames=4)
        self.animation.set_frames(emotion, frames)
        self.animation.pulse()

        # 无语音合成依赖，对话以控制台日志 + 气泡形式输出
        print(
            f"[EmotionPet] 情绪={self.display.label_zh(emotion)} "
            f"行为={behavior} 对话='{dialogue}'"
        )
        self.window.setToolTip(dialogue)

    # ------------------------------------------------------------------
    # 成长结算
    # ------------------------------------------------------------------
    def _tick_stats(self) -> None:
        """周期性结算：精力自然衰减、心情向中性回归、累积少量经验。"""
        self.stats.update("energy", -2)
        self.stats.update("mood", -1)
        result = self.level_system.gain_exp(5)
        if result.levels_gained:
            print(
                f"[Growth] 升级! 新等级={self.stats.level} "
                f"解锁技能={result.skills_unlocked} 外观={result.appearance}"
            )
        # 刷新状态叠加（精力条 / 等级徽章）
        self._apply_emotion(self.current_emotion)

    # ------------------------------------------------------------------
    # 菜单回调
    # ------------------------------------------------------------------
    def _on_chat(self) -> None:
        """对话交互：读取用户输入并给出宠物回复。"""
        text, ok = qt.QInputDialog.getText(self.window, "对话", "和宠物说点什么：")
        if ok and text:
            reply = self.behavior_engine.select_dialogue(self.current_emotion)
            qt.QMessageBox.information(self.window, "宠物回复", reply)
            # 主动互动提升好感度
            self.stats.update("affinity", 2)

    def _on_settings(self) -> None:
        """设置/状态面板：展示成长属性。"""
        info = (
            f"等级 Lv.{self.stats.level}  "
            f"经验 {self.stats.exp}/{self.level_system.exp_needed}\n"
            f"心情 {self.stats.mood}  好感 {self.stats.affinity}  "
            f"精力 {self.stats.energy}\n"
            f"技能 {self.level_system.unlocked_skill_labels}\n"
            f"外观 {self.stats.appearance}"
        )
        qt.QMessageBox.information(self.window, "宠物状态", info)

    def _on_quit(self) -> None:
        """退出前持久化成长状态。"""
        self._save()
        qt.QApplication.instance().quit()

    def _save(self) -> None:
        """将成长状态写入 SQLite（持久化失败不阻断退出）。"""
        try:
            from growth_system.persistence import save_stats

            save_stats(self.stats, db_path=DB_PATH)
            print(f"[Persistence] 成长状态已保存到 {DB_PATH}")
        except Exception as exc:  # pragma: no cover - 磁盘/驱动异常兜底
            print(f"[Persistence] 保存失败: {exc}")


def main() -> int:
    """GUI 入口：构造 QApplication 并运行事件循环。"""
    if not qt.HAS_PYQT5:
        raise qt.missing_error("desktop_pet.main")

    app = qt.QApplication(sys.argv)
    app.setApplicationName("Emotion Pet")
    app.setQuitOnLastWindowClosed(False)  # 宠物窗口常驻，显式退出

    controller = PetController()
    controller.start()
    return app.exec_()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as exc:
        # PyQt5 缺失时的清晰错误提示
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)
