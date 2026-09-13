import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

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


def test_only_ready_state_allows_model_switching() -> None:
    assert UiCapabilities.for_state(AppState.READY).can_switch_model
    for state in (
        AppState.MODEL_LOADING,
        AppState.RECORDING,
        AppState.LIVE_TRANSCRIBING,
        AppState.FINALIZING,
        AppState.RETRY_PENDING,
        AppState.MODEL_SWITCHING,
    ):
        assert not UiCapabilities.for_state(state).can_switch_model


def test_overlay_is_non_activating_click_through_compact_capsule() -> None:
    _app()
    overlay = VoiceOverlay()

    assert overlay.height() == 32
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert overlay.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )
    assert overlay.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert not overlay.windowFlags() & Qt.WindowType.WindowMinMaxButtonsHint


def test_overlay_paints_v1_translucent_background() -> None:
    app = _app()
    overlay = VoiceOverlay()
    overlay.set_recording(elapsed_seconds=1, preview="")
    overlay.show_for_window()
    QTest.qWait(220)

    background = overlay.grab().toImage().pixelColor(540, 16)

    assert background.red() == 25
    assert background.green() == 25
    assert background.blue() == 32
    assert background.alpha() >= 220


def test_overlay_retains_full_preview_while_showing_tail() -> None:
    _app()
    overlay = VoiceOverlay()
    full_text = "前缀" + ("very long mixed text " * 30) + "结尾"

    overlay.set_recording(elapsed_seconds=7, preview=full_text)

    assert overlay.current_preview == full_text
    assert overlay.displayed_text.endswith("结尾")
    assert "00:07" in overlay.status_text


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


def test_ready_state_parks_loading_overlay_off_screen() -> None:
    _app()
    view = DesktopView()
    view.render_state(AppState.MODEL_LOADING, ModelId.FAST)
    assert view.overlay.isVisible()

    view.render_state(AppState.READY, ModelId.FAST)
    QTest.qWait(250)

    assert view.overlay.isVisible()
    assert view.overlay.x() == -9999
