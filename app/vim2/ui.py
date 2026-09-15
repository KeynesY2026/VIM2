from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QSystemTrayIcon,
    QWidget,
)

from vim2.models import MODEL_SPECS, ModelId
from vim2.state import AppState

STATUS_COLORS: dict[AppState, tuple[int, int, int]] = {
    AppState.STARTING: (150, 150, 150),
    AppState.MODEL_LOADING: (150, 150, 150),
    AppState.READY: (60, 180, 75),
    AppState.RECORDING: (220, 50, 50),
    AppState.LIVE_TRANSCRIBING: (220, 50, 50),
    AppState.FINALIZING: (230, 180, 30),
    AppState.RETRY_PENDING: (230, 180, 30),
    AppState.MODEL_SWITCHING: (150, 150, 150),
    AppState.ERROR: (220, 50, 50),
    AppState.EXITING: (150, 150, 150),
}

STATUS_TOOLTIPS: dict[AppState, str] = {
    AppState.STARTING: "VIM2 - Loading",
    AppState.MODEL_LOADING: "VIM2 - Loading",
    AppState.READY: "VIM2 - Ready",
    AppState.RECORDING: "VIM2 - Recording",
    AppState.LIVE_TRANSCRIBING: "VIM2 - Recording",
    AppState.FINALIZING: "VIM2 - Processing",
    AppState.RETRY_PENDING: "VIM2 - Processing failed",
    AppState.MODEL_SWITCHING: "VIM2 - Loading",
    AppState.ERROR: "VIM2 - Error",
    AppState.EXITING: "VIM2 - Exiting",
}


@dataclass(frozen=True, slots=True)
class UiCapabilities:
    can_toggle_recording: bool
    can_switch_model: bool
    can_retry: bool
    can_discard: bool

    @classmethod
    def for_state(cls, state: AppState) -> UiCapabilities:
        return cls(
            can_toggle_recording=state
            in {
                AppState.READY,
                AppState.RECORDING,
                AppState.LIVE_TRANSCRIBING,
            },
            can_switch_model=state is AppState.READY,
            can_retry=state is AppState.RETRY_PENDING,
            can_discard=state is AppState.RETRY_PENDING,
        )


class TrayIconFactory:
    @staticmethod
    def create(state: AppState) -> QIcon:
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*STATUS_COLORS[state]))
        painter.drawEllipse(2, 2, 12, 12)
        painter.end()
        return QIcon(pixmap)


class _StatusDot(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(10, 10)
        self._color = QColor(230, 60, 60)
        self._opacity = 1.0

    def set_color(self, color: QColor) -> None:
        self._color = color
        self.update()

    def set_pulse_opacity(self, opacity: float) -> None:
        self._opacity = opacity
        self.update()

    def paintEvent(self, event) -> None:
        del event
        color = QColor(self._color)
        color.setAlphaF(self._opacity)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QRectF(0.5, 0.5, 9, 9))
        painter.end()


class VoiceOverlay(QWidget):
    _MINIMUM_WIDTH = 110
    _MAXIMUM_WIDTH = 560

    def __init__(self) -> None:
        window_flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        if not hasattr(ctypes, "WinDLL"):
            window_flags |= Qt.WindowType.WindowDoesNotAcceptFocus
        super().__init__(None, window_flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedSize(self._MINIMUM_WIDTH, 32)
        self.setStyleSheet(
            """
            QLabel {
                color: #F0F0F5;
                font-family: "Segoe UI";
                font-size: 12px;
                background: transparent;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 12, 0)
        layout.setSpacing(8)
        self._dot = _StatusDot()
        self._status = QLabel()
        self._preview = QLabel()
        self._preview.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction
        )
        self._preview.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self._dot)
        layout.addWidget(self._status)
        layout.addWidget(self._preview, 1)

        self.current_preview = ""
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(50)
        self._pulse_timer.timeout.connect(self._pulse)
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._fade = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade.setDuration(200)
        self._fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade.finished.connect(self._park_if_transparent)
        self._opacity.setOpacity(0.0)
        self._fade_target_visible = False
        self._target_window: object | None = None
        self.move(-9999, 0)
        self.show()

    @property
    def displayed_text(self) -> str:
        return self._preview.text()

    @property
    def status_text(self) -> str:
        return self._status.text()

    def set_recording(self, *, preview: str) -> None:
        self.current_preview = preview
        self._status.setText("正在录音")
        self._preview.setText(self._tail(preview))
        self._dot.set_color(QColor(230, 60, 60))
        self._resize_to_content()
        if not self._pulse_timer.isActive():
            self._pulse_timer.start()

    def set_message(self, title: str, detail: str = "") -> None:
        self.current_preview = detail
        self._status.setText(title)
        self._preview.setText(self._tail(detail))
        self._dot.set_color(QColor(230, 180, 30))
        self._pulse_timer.stop()
        self._dot.set_pulse_opacity(1.0)
        self._resize_to_content()

    @staticmethod
    def _tail(text: str, limit: int = 72) -> str:
        return text if len(text) <= limit else f"…{text[-limit:]}"

    def show_for_window(self, target_window: object | None = None) -> None:
        self._target_window = target_window
        self._move_on_screen()
        self._fade.stop()
        self._fade_target_visible = True
        self._fade.setStartValue(self._opacity.opacity())
        self._fade.setEndValue(1.0)
        if not hasattr(ctypes, "WinDLL"):
            self.show()
            self.raise_()
        self._ensure_topmost()
        self._fade.start()

    def _move_on_screen(self) -> None:
        screen = self._screen_for_window(self._target_window)
        area = screen.availableGeometry()
        self.move(
            area.x() + (area.width() - self.width()) // 2,
            area.bottom() - self.height() - 60 + 1,
        )

    def _resize_to_content(self) -> None:
        status_width = self._status.fontMetrics().horizontalAdvance(
            self._status.text()
        )
        preview_width = self._preview.fontMetrics().horizontalAdvance(
            self._preview.text()
        )
        content_width = 50 + status_width
        if self._preview.text():
            content_width += 8 + preview_width
        self.setFixedWidth(
            max(self._MINIMUM_WIDTH, min(self._MAXIMUM_WIDTH, content_width))
        )
        if self._fade_target_visible:
            self._move_on_screen()

    def fade_out(self) -> None:
        self._pulse_timer.stop()
        self._fade.stop()
        self._fade_target_visible = False
        self._fade.setStartValue(self._opacity.opacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _park_if_transparent(self) -> None:
        if not self._fade_target_visible:
            self.move(-9999, 0)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Source
        )
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setBrush(QColor(25, 25, 32, 224))
        pen = painter.pen()
        pen.setColor(QColor(80, 80, 100, 80))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(bounds, 14, 14)
        painter.end()
        super().paintEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(ctypes, "WinDLL"):
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            get_window_long = user32.GetWindowLongPtrW
            set_window_long = user32.SetWindowLongPtrW
            get_window_long.argtypes = (ctypes.c_void_p, ctypes.c_int)
            get_window_long.restype = ctypes.c_ssize_t
            set_window_long.argtypes = (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_ssize_t,
            )
            set_window_long.restype = ctypes.c_ssize_t
            handle = int(self.winId())
            extended_style = get_window_long(handle, -20)
            set_window_long(
                handle,
                -20,
                extended_style | 0x00000080 | 0x00000020 | 0x08000000,
            )

    def _ensure_topmost(self) -> None:
        if not hasattr(ctypes, "WinDLL"):
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SetWindowPos(
            int(self.winId()),
            ctypes.c_void_p(-1),
            0,
            0,
            0,
            0,
            0x0002 | 0x0001 | 0x0010 | 0x0040,
        )

    def _screen_for_window(self, target_window: object | None):
        app = QApplication.instance()
        if target_window and hasattr(ctypes, "WinDLL"):
            rect = wintypes_rect()
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            if user32.GetWindowRect(target_window, ctypes.byref(rect)):
                point = QPoint(
                    (rect.left + rect.right) // 2,
                    (rect.top + rect.bottom) // 2,
                )
                screen = app.screenAt(point)
                if screen:
                    return screen
        return app.primaryScreen()

    def _pulse(self) -> None:
        self._pulse_phase = (self._pulse_phase + 0.2) % (2 * math.pi)
        opacity = 0.65 + 0.35 * math.sin(self._pulse_phase)
        self._dot.set_pulse_opacity(max(0.3, min(1.0, opacity)))


class _Rect(ctypes.Structure):
    _fields_ = (
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    )


def wintypes_rect() -> _Rect:
    return _Rect()


class DesktopView:
    def __init__(
        self,
        supported_models: tuple[ModelId, ...] | None = None,
    ) -> None:
        self._supported_models = frozenset(
            tuple(ModelId) if supported_models is None else supported_models
        )
        self.overlay = VoiceOverlay()
        self.tray = QSystemTrayIcon()
        self.menu = QMenu()
        self.status_action = QAction("VIM2：启动中")
        self.status_action.setEnabled(False)
        self.recording_action = QAction("开始录音")
        self.model_menu = self.menu.addMenu("识别模型")
        self.cpu_model_action = QAction(
            "CPU：Qwen3-ASR 0.6B INT8", self.model_menu
        )
        self.fast_model_action = QAction(
            "快速：Qwen3-ASR 0.6B FP16", self.model_menu
        )
        self.accurate_model_action = QAction(
            "高精度：Qwen3-ASR 1.7B INT8", self.model_menu
        )
        self._model_actions = {
            ModelId.CPU: self.cpu_model_action,
            ModelId.FAST: self.fast_model_action,
            ModelId.ACCURATE: self.accurate_model_action,
        }
        for model_id, action in self._model_actions.items():
            action.setCheckable(True)
            action.setVisible(model_id in self._supported_models)
        self.about_action = QAction("关于")
        self.exit_action = QAction("退出")

        self.menu.addAction(self.status_action)
        self.menu.addSeparator()
        self.menu.addAction(self.recording_action)
        self.model_menu.addAction(self.cpu_model_action)
        self.model_menu.addSeparator()
        self.model_menu.addAction(self.fast_model_action)
        self.model_menu.addAction(self.accurate_model_action)
        self.menu.addSeparator()
        self.menu.addAction(self.about_action)
        self.menu.addAction(self.exit_action)
        self.tray.setContextMenu(self.menu)
        self.tray.setIcon(TrayIconFactory.create(AppState.STARTING))
        self.tray.setToolTip(STATUS_TOOLTIPS[AppState.STARTING])

        self._controller = None
        self._target_window: object | None = None
        self._latest_preview = ""
        self._preview_timer = QTimer()
        self._preview_timer.setInterval(1_000)
        self._preview_timer.timeout.connect(self._on_preview_tick)
        self._max_timer = QTimer()
        self._max_timer.setSingleShot(True)
        self._max_timer.timeout.connect(self._on_max_duration)
        self._error_hide_timer = QTimer()
        self._error_hide_timer.setSingleShot(True)
        self._error_hide_timer.setInterval(10_000)
        self._error_hide_timer.timeout.connect(self.overlay.fade_out)

    def bind(self, controller, quit_callback) -> None:
        self._controller = controller
        self.recording_action.triggered.connect(
            controller.toggle_recording
        )
        self.cpu_model_action.triggered.connect(
            lambda: controller.switch_model(ModelId.CPU)
        )
        self.fast_model_action.triggered.connect(
            lambda: controller.switch_model(ModelId.FAST)
        )
        self.accurate_model_action.triggered.connect(
            lambda: controller.switch_model(ModelId.ACCURATE)
        )
        self.about_action.triggered.connect(self._show_about)
        self.exit_action.triggered.connect(quit_callback)
        self.tray.activated.connect(self._on_tray_activated)

    def show(self) -> None:
        self.tray.show()

    def render_state(self, state: AppState, model_id: ModelId) -> None:
        capabilities = UiCapabilities.for_state(state)
        self.tray.setIcon(TrayIconFactory.create(state))
        self.tray.setToolTip(STATUS_TOOLTIPS[state])
        self.status_action.setText(f"VIM2：{self._state_label(state)}")
        self.recording_action.setEnabled(
            capabilities.can_toggle_recording
        )
        self.recording_action.setText(self._record_action_label(state))
        self.model_menu.setEnabled(capabilities.can_switch_model)
        for action_model_id, action in self._model_actions.items():
            action.setEnabled(
                capabilities.can_switch_model
                and action_model_id in self._supported_models
            )
        self.cpu_model_action.setChecked(model_id is ModelId.CPU)
        self.fast_model_action.setChecked(model_id is ModelId.FAST)
        self.accurate_model_action.setChecked(model_id is ModelId.ACCURATE)

        if state is AppState.MODEL_LOADING:
            self.overlay.set_message(
                "正在加载模型", MODEL_SPECS[model_id].display_name
            )
            self.overlay.show_for_window()
        elif state is AppState.MODEL_SWITCHING:
            self.overlay.set_message(
                "正在切换识别模型", MODEL_SPECS[model_id].display_name
            )
            self.overlay.show_for_window()
        elif state is AppState.FINALIZING:
            self.overlay.set_message("正在识别", self._latest_preview)
            self.overlay.show_for_window(self._target_window)
        elif state is AppState.READY:
            self.overlay.fade_out()

    def start_recording_timers(
        self,
        max_seconds: int,
        target_window: object,
        preview_interval_ms: int,
    ) -> None:
        self._target_window = target_window
        self._latest_preview = ""
        self._error_hide_timer.stop()
        self.overlay.set_recording(preview="")
        self.overlay.show_for_window(target_window)
        self._preview_timer.setInterval(preview_interval_ms)
        self._preview_timer.start()
        self._max_timer.start(max_seconds * 1000)

    def stop_recording_timers(self) -> None:
        self._preview_timer.stop()
        self._max_timer.stop()

    def show_preview(self, text: str) -> None:
        self._latest_preview = text
        self.overlay.set_recording(preview=text)

    def show_error(self, message: str) -> None:
        self.tray.showMessage(
            "VIM2",
            message,
            QSystemTrayIcon.MessageIcon.Critical,
            7000,
        )
        self.overlay.set_message("发生错误", message)
        self.overlay.show_for_window(self._target_window)
        self._error_hide_timer.start()

    def show_warning(self, message: str) -> None:
        self.tray.showMessage(
            "VIM2",
            message,
            QSystemTrayIcon.MessageIcon.Warning,
            5000,
        )

    def show_retry_error(self, message: str) -> None:
        if self._controller is None:
            return
        dialog = QMessageBox()
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("VIM2 识别失败")
        dialog.setText(f"识别失败：{message}")
        retry_button = dialog.addButton(
            "重试", QMessageBox.ButtonRole.AcceptRole
        )
        dialog.addButton("放弃", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        if dialog.clickedButton() is retry_button:
            self._controller.retry()
        else:
            self._controller.cancel()

    def hide_overlay(self) -> None:
        self._error_hide_timer.stop()
        self.overlay.fade_out()

    def _on_preview_tick(self) -> None:
        if self._controller is not None:
            self._controller.request_preview()

    def _on_max_duration(self) -> None:
        if self._controller is not None:
            self._controller.toggle_recording()

    def _on_tray_activated(
        self, reason: QSystemTrayIcon.ActivationReason
    ) -> None:
        if (
            reason is QSystemTrayIcon.ActivationReason.Trigger
            and self._controller is not None
        ):
            self._controller.toggle_recording()

    @staticmethod
    def _state_label(state: AppState) -> str:
        return {
            AppState.STARTING: "启动中",
            AppState.MODEL_LOADING: "模型加载中",
            AppState.READY: "就绪",
            AppState.RECORDING: "录音中",
            AppState.LIVE_TRANSCRIBING: "实时识别中",
            AppState.FINALIZING: "最终识别中",
            AppState.RETRY_PENDING: "等待重试",
            AppState.MODEL_SWITCHING: "模型切换中",
            AppState.ERROR: "错误",
            AppState.EXITING: "退出中",
        }[state]

    @staticmethod
    def _record_action_label(state: AppState) -> str:
        if state in {AppState.RECORDING, AppState.LIVE_TRANSCRIBING}:
            return "停止录音"
        if state is AppState.FINALIZING:
            return "识别中…"
        return "开始录音"

    def _show_about(self) -> None:
        QMessageBox.information(
            None,
            "关于 VIM2",
            "VIM2 本地语音输入\n"
            "Qwen3-ASR CPU / GPU 离线识别\n"
            "音频和识别文本始终保留在本机。",
        )
