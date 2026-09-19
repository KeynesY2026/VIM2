import ctypes
import os
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import vim2.ui as ui_module
from vim2.config import MacPasteShortcut
from vim2.models import ModelId
from vim2.state import AppState
from vim2.ui import (
    STATUS_COLORS,
    STATUS_TOOLTIPS,
    DesktopView,
    TrayIconFactory,
    UiCapabilities,
    VoiceOverlay,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_tray_icon_is_16px_transparent_canvas_with_12px_state_dot() -> None:
    _app()
    image = TrayIconFactory.create(AppState.READY).pixmap(16, 16).toImage()

    assert image.width() == 16
    assert image.height() == 16
    assert image.pixelColor(0, 0).alpha() == 0
    assert image.pixelColor(8, 8).getRgb()[:3] == STATUS_COLORS[
        AppState.READY
    ]


def test_status_tooltips_keep_v1_semantics() -> None:
    assert STATUS_TOOLTIPS[AppState.MODEL_LOADING] == "VIM2 - Loading"
    assert STATUS_TOOLTIPS[AppState.READY] == "VIM2 - Ready"
    assert STATUS_TOOLTIPS[AppState.RECORDING] == "VIM2 - Recording"
    assert STATUS_TOOLTIPS[AppState.FINALIZING] == "VIM2 - Processing"


def test_only_ready_state_allows_model_and_paste_shortcut_switching() -> None:
    ready = UiCapabilities.for_state(AppState.READY)
    assert ready.can_switch_model
    assert ready.can_switch_paste_shortcut
    for state in (
        AppState.MODEL_LOADING,
        AppState.RECORDING,
        AppState.LIVE_TRANSCRIBING,
        AppState.FINALIZING,
        AppState.RETRY_PENDING,
        AppState.MODEL_SWITCHING,
    ):
        capabilities = UiCapabilities.for_state(state)
        assert not capabilities.can_switch_model
        assert not capabilities.can_switch_paste_shortcut


def test_overlay_is_non_activating_click_through_compact_capsule() -> None:
    _app()
    overlay = VoiceOverlay()

    assert overlay.height() == 32
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert overlay.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )
    assert overlay.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert overlay.windowFlags() & Qt.WindowType.NoDropShadowWindowHint
    assert not overlay.windowFlags() & Qt.WindowType.WindowMinMaxButtonsHint


@pytest.mark.skipif(
    hasattr(ctypes, "WinDLL"),
    reason="the non-activating re-raise path is specific to non-Win32 hosts",
)
def test_show_for_window_reshows_and_raises_non_win32_overlay_without_focus() -> None:
    app = _app()
    overlay = VoiceOverlay()
    overlay.hide()
    app.processEvents()
    show = Mock(wraps=overlay.show)
    raise_window = Mock(wraps=overlay.raise_)
    activate = Mock(wraps=overlay.activateWindow)
    overlay.show = show
    overlay.raise_ = raise_window
    overlay.activateWindow = activate

    overlay.show_for_window()
    app.processEvents()

    show.assert_called_once_with()
    raise_window.assert_called_once_with()
    activate.assert_not_called()
    assert overlay.isVisible()
    assert overlay.windowFlags() & Qt.WindowType.Tool
    assert overlay.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert overlay.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert overlay.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert overlay.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )


def test_overlay_uses_cursor_monitor_instead_of_target_window(
    monkeypatch,
) -> None:
    _app()
    overlay = VoiceOverlay()

    class FakeScreen:
        def __init__(self, area: QRect) -> None:
            self._area = area

        def availableGeometry(self) -> QRect:
            return self._area

    left_screen = FakeScreen(QRect(0, 0, 1_000, 800))
    right_screen = FakeScreen(QRect(1_000, 0, 1_000, 800))
    cursor_position = [200, 300]

    class FakeApplication:
        @staticmethod
        def instance():
            return FakeApplication()

        def screenAt(self, point):
            return left_screen if point.x() < 1_000 else right_screen

        def primaryScreen(self):
            return right_screen

    class FakeCursor:
        @staticmethod
        def pos():
            return ui_module.QPoint(*cursor_position)

    class FakeFunction:
        def __init__(self, callback) -> None:
            self._callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self._callback(*args)

    class FakeUser32:
        def __init__(self) -> None:
            self.GetWindowRect = FakeFunction(self._get_window_rect)
            self.GetCursorPos = FakeFunction(self._get_cursor_pos)
            self.SetWindowPos = FakeFunction(lambda *args: 1)

        @staticmethod
        def _get_window_rect(window, rect_pointer):
            del window
            rect = rect_pointer._obj
            rect.left = 1_200
            rect.right = 1_800
            rect.top = 100
            rect.bottom = 700
            return 1

        @staticmethod
        def _get_cursor_pos(point_pointer):
            point = point_pointer._obj
            point.x = 1_500
            point.y = 300
            return 1

    monkeypatch.setattr(ui_module, "QApplication", FakeApplication)
    monkeypatch.setattr(ui_module, "QCursor", FakeCursor, raising=False)
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *args, **kwargs: FakeUser32(),
        raising=False,
    )

    overlay.show_for_cursor(123)

    assert overlay.x() == 445
    assert overlay.y() == 708

    cursor_position[:] = [1_500, 300]
    overlay.set_recording(preview="preview text " * 20)
    resized_x = overlay.x()
    overlay.set_message("正在识别", overlay.current_preview)
    overlay.show_on_current_screen()

    assert 0 <= resized_x < 1_000
    assert overlay.x() == resized_x


def test_overlay_paints_v1_translucent_background() -> None:
    app = _app()
    overlay = VoiceOverlay()
    overlay.set_recording(preview="")
    overlay.show_for_window()
    QTest.qWait(220)

    background = overlay.grab().toImage().pixelColor(
        overlay.width() - 20, 16
    )

    assert background.red() == 25
    assert background.green() == 25
    assert background.blue() == 32
    assert background.alpha() >= 220


def test_overlay_background_does_not_darken_when_children_repaint() -> None:
    _app()
    overlay = VoiceOverlay()
    overlay.set_message("正在识别", "实时识别文字")
    overlay._opacity.setOpacity(1.0)
    image = QImage(
        overlay.size(), QImage.Format.Format_ARGB32_Premultiplied
    )
    image.fill(Qt.GlobalColor.transparent)

    overlay.render(image)
    first_background = image.pixelColor(25, 16)
    overlay.render(image)
    second_background = image.pixelColor(25, 16)

    assert second_background == first_background


def test_recording_overlay_stays_compact_and_draws_red_dot() -> None:
    app = _app()
    overlay = VoiceOverlay()
    overlay.set_recording(preview="")
    overlay.show_for_window()
    QTest.qWait(220)

    dot = overlay._dot.grab().toImage().pixelColor(5, 5)

    assert overlay.width() < 300
    assert dot.red() >= 220
    assert dot.green() < 100
    assert dot.alpha() > 0


def test_overlay_retains_full_preview_while_showing_tail() -> None:
    _app()
    overlay = VoiceOverlay()
    full_text = "前缀" + ("very long mixed text " * 30) + "结尾"

    overlay.set_recording(preview=full_text)

    assert overlay.current_preview == full_text
    assert overlay.displayed_text.endswith("结尾")
    assert overlay.status_text == "正在录音"
    assert overlay._preview.alignment() & Qt.AlignmentFlag.AlignRight


def test_live_transcribing_updates_intermediate_preview() -> None:
    _app()
    view = DesktopView()
    view.start_recording_timers(
        max_seconds=90,
        target_window=123,
        preview_interval_ms=1_000,
    )
    view.render_state(AppState.LIVE_TRANSCRIBING, model_id=ModelId.FAST)

    view.show_preview("实时转录结果")

    assert view.overlay.current_preview == "实时转录结果"
    assert view.overlay.displayed_text == "实时转录结果"
    assert view.overlay.status_text == "正在录音"


def test_finalizing_shows_recognizing_state_with_latest_preview() -> None:
    _app()
    view = DesktopView()
    view.start_recording_timers(
        max_seconds=90,
        target_window=123,
        preview_interval_ms=1_000,
    )
    view.show_preview("这是最新识别出来的文字")

    view.render_state(AppState.FINALIZING, model_id=ModelId.FAST)

    dot = view.overlay._dot.grab().toImage().pixelColor(5, 5)

    assert view.overlay.current_preview == "这是最新识别出来的文字"
    assert view.overlay.displayed_text == "这是最新识别出来的文字"
    assert view.overlay.status_text == "正在识别"
    assert dot.red() >= 220
    assert 150 <= dot.green() <= 210
    assert dot.blue() < 80


def test_tray_menu_reflects_state_and_selected_model() -> None:
    _app()
    view = DesktopView()

    view.render_state(AppState.READY, model_id=ModelId.FAST)

    assert view.tray.toolTip() == "VIM2 - Ready"
    assert view.recording_action.text() == "开始录音"
    assert view.recording_action.isEnabled()
    assert view.fast_model_action.isChecked()
    assert view.fast_model_action.isEnabled()

    view.render_state(
        AppState.FINALIZING, model_id=ModelId.FAST
    )

    assert view.recording_action.text() == "识别中…"
    assert not view.recording_action.isEnabled()
    assert not view.fast_model_action.isEnabled()


def test_macos_tray_menu_shows_exclusive_paste_shortcuts_and_current_value() -> None:
    _app()
    view = DesktopView(
        macos_paste_shortcut=MacPasteShortcut.COMMAND_V,
    )
    controller = Mock()
    view.bind(controller, lambda: None)
    view.render_state(AppState.READY, model_id=ModelId.CPU)

    assert view.paste_shortcut_menu is not None
    assert view.paste_shortcut_menu.title() == "粘贴快捷键"
    assert view.command_v_action is not None
    assert view.control_v_action is not None
    assert view.command_v_action.text() == "macOS：⌘V"
    assert view.control_v_action.text() == "Windows / 远程桌面：Ctrl+V"
    assert view.command_v_action.isChecked()
    assert not view.control_v_action.isChecked()
    assert view.paste_shortcut_action_group is not None
    assert view.paste_shortcut_action_group.isExclusive()

    view.control_v_action.trigger()

    controller.switch_macos_paste_shortcut.assert_called_once_with(
        MacPasteShortcut.CONTROL_V
    )
    assert not view.command_v_action.isChecked()
    assert view.control_v_action.isChecked()

    view.render_macos_paste_shortcut(MacPasteShortcut.COMMAND_V)

    assert view.command_v_action.isChecked()
    assert not view.control_v_action.isChecked()


def test_macos_paste_shortcut_menu_is_enabled_only_while_ready() -> None:
    _app()
    view = DesktopView(
        macos_paste_shortcut=MacPasteShortcut.COMMAND_V,
    )

    for state in AppState:
        view.render_state(state, model_id=ModelId.CPU)
        assert view.paste_shortcut_menu is not None
        assert view.paste_shortcut_menu.isEnabled() is (
            state is AppState.READY
        )


def test_windows_tray_menu_does_not_show_macos_paste_shortcuts() -> None:
    _app()
    view = DesktopView()

    assert view.paste_shortcut_menu is None
    assert view.command_v_action is None
    assert view.control_v_action is None
    assert "粘贴快捷键" not in [action.text() for action in view.menu.actions()]


def test_tray_menu_supports_cpu_model() -> None:
    _app()
    view = DesktopView()

    view.render_state(AppState.READY, model_id=ModelId.CPU)

    assert view.cpu_model_action.text() == "CPU：Qwen3-ASR 0.6B INT8"
    assert view.cpu_model_action.isChecked()
    assert not view.fast_model_action.isChecked()
    assert not view.accurate_model_action.isChecked()


def test_tray_menu_exposes_hotword_file_and_ready_only_reload() -> None:
    _app()
    view = DesktopView()

    view.render_state(AppState.READY, model_id=ModelId.FAST)

    assert view.open_hotwords_action.text() == "打开热词文件"
    assert view.reload_hotwords_action.text() == "重新加载热词"
    assert view.open_hotwords_action.isEnabled()
    assert view.reload_hotwords_action.isEnabled()

    view.render_state(AppState.RECORDING, model_id=ModelId.FAST)

    assert view.open_hotwords_action.isEnabled()
    assert not view.reload_hotwords_action.isEnabled()


def test_live_preview_timer_uses_configured_interval() -> None:
    _app()
    view = DesktopView()

    view.start_recording_timers(
        max_seconds=90,
        target_window=123,
        preview_interval_ms=500,
    )

    assert view._preview_timer.interval() == 500


def test_error_overlay_stays_for_ten_seconds_then_fades() -> None:
    _app()
    view = DesktopView()

    view.show_error("Microphone unavailable")

    assert view.overlay.x() != -9999
    assert view._error_hide_timer.isActive()
    assert view._error_hide_timer.interval() == 10_000

    view._error_hide_timer.timeout.emit()
    QTest.qWait(250)

    assert view.overlay.x() == -9999


def test_ready_state_parks_loading_overlay_off_screen() -> None:
    _app()
    view = DesktopView()
    view.render_state(AppState.MODEL_LOADING, ModelId.FAST)
    assert view.overlay.isVisible()

    view.render_state(AppState.READY, ModelId.FAST)
    QTest.qWait(250)

    assert view.overlay.isVisible()
    assert view.overlay.x() == -9999
