from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from vim2.audio import AudioRecorder
from vim2.config import (
    MacPasteShortcut,
    MacPasteShortcutSelection,
    Settings,
    SettingsRepository,
)
from vim2.controller import AppController
from vim2.hotkey import HotkeyDispatcher, parse_hotkey
from vim2.hotwords import HotwordRepository
from vim2.paths import AppPaths
from vim2.platform_services import PlatformServices, create_platform_services
from vim2.postprocessing import create_text_postprocessor
from vim2.recognizer import QwenRecognizer
from vim2.session import VoiceSession
from vim2.state import AppState, StateMachine
from vim2.ui import DesktopView


RESTART_EXIT_CODE = 75


class _WorkerSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()


class _Worker(QRunnable):
    def __init__(self, work: Callable[[], object]) -> None:
        super().__init__()
        self.work = work
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.work()
        except Exception as exc:
            logging.getLogger(__name__).exception("Background task failed")
            self.signals.failed.emit(exc)
        else:
            self.signals.succeeded.emit(result)
        finally:
            self.signals.finished.emit()


class QtTaskRunner:
    def __init__(self, pool: QThreadPool | None = None) -> None:
        self._pool = pool or QThreadPool.globalInstance()
        self._workers: set[_Worker] = set()

    def submit(
        self,
        work: Callable[[], object],
        on_success: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        worker = _Worker(work)
        self._workers.add(worker)
        worker.signals.succeeded.connect(on_success)
        worker.signals.failed.connect(on_error)
        worker.signals.finished.connect(
            lambda: self._workers.discard(worker)
        )
        self._pool.start(worker)

    def wait_for_done(self) -> None:
        self._pool.waitForDone()


class ProcessHandle(Protocol):
    def poll(self) -> int | None: ...


class HotwordEditorProcess(QObject):
    changed = Signal()

    def __init__(
        self,
        path: Path,
        *,
        launch: Callable[[list[str]], ProcessHandle] = subprocess.Popen,
    ) -> None:
        super().__init__()
        self._path = path.resolve()
        self._launch = launch
        self._process: ProcessHandle | None = None
        self._opened_modified_time_ns: int | None = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self.check_process)

    def open(self) -> bool:
        if self._process is not None:
            if self._process.poll() is None:
                return True
            self._finish_process()
        HotwordRepository(self._path).ensure_file()
        self._opened_modified_time_ns = self._file_modified_time_ns()
        try:
            self._process = self._launch(
                ["notepad.exe", str(self._path)]
            )
        except OSError:
            logging.getLogger(__name__).exception(
                "Failed to open hotword editor"
            )
            self._process = None
            return False
        self._poll_timer.start()
        return True

    def check_process(self) -> None:
        if self._process is None or self._process.poll() is None:
            return
        self._finish_process()

    def _finish_process(self) -> None:
        self._process = None
        self._poll_timer.stop()
        modified_time_ns = self._file_modified_time_ns()
        if (
            modified_time_ns is not None
            and modified_time_ns != self._opened_modified_time_ns
        ):
            self.changed.emit()
        self._opened_modified_time_ns = None

    def _file_modified_time_ns(self) -> int | None:
        try:
            return self._path.stat().st_mtime_ns
        except FileNotFoundError:
            return None


class _HotkeyBridge(QObject):
    toggle_requested = Signal()
    cancel_requested = Signal()


def _open_local_file(path: Path) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _open_hotwords_file(path: Path) -> bool:
    return _open_local_file(HotwordRepository(path).ensure_file())


def _create_hotword_editor(
    services: PlatformServices, path: Path
) -> HotwordEditorProcess | None:
    # The tracked Notepad process is Windows-specific. macOS uses its native
    # file association and retains the manual reload action in the tray.
    if services.profile.name != "windows":
        return None
    return HotwordEditorProcess(path)


def _restart_process(paths: AppPaths) -> None:
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "vim2",
            "--root",
            str(paths.root),
            "--windowed",
        ],
    )


def _create_macos_paste_shortcut_selection(
    services: PlatformServices,
    shortcut: MacPasteShortcut,
) -> MacPasteShortcutSelection | None:
    if services.profile.name != "macos":
        return None
    return MacPasteShortcutSelection(shortcut)


def _prepare_desktop_runtime(
    app: QApplication,
    view: DesktopView,
    recognizer: QwenRecognizer,
    controller: AppController,
) -> None:
    view.show()
    app.processEvents()
    recognizer.initialize_runtime(controller.selected_model)
    controller.start()


def run_qt_application(
    paths: AppPaths,
    settings: Settings,
    *,
    platform_services: PlatformServices | None = None,
) -> int:
    services = platform_services or create_platform_services(paths)
    services.enable_desktop_features()
    app = QApplication(sys.argv)
    app.setApplicationName("VIM2")
    app.setQuitOnLastWindowClosed(False)

    guard = services.create_single_instance_guard()
    if not guard.acquire():
        QMessageBox.information(None, "VIM2", "VIM2 已经在运行。")
        return 0
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, "VIM2", "系统托盘不可用，VIM2 无法启动。")
        guard.close()
        return 1

    machine = StateMachine()
    recognizer = QwenRecognizer(paths)
    recorder = AudioRecorder()
    bridge = _HotkeyBridge()
    dispatcher = HotkeyDispatcher(
        parse_hotkey(settings.hotkey),
        on_toggle=bridge.toggle_requested.emit,
        on_cancel=bridge.cancel_requested.emit,
        is_cancellable=lambda: machine.current
        in {
            AppState.RECORDING,
            AppState.LIVE_TRANSCRIBING,
            AppState.RETRY_PENDING,
        },
    )
    hotkey = services.create_hotkey_listener(dispatcher)
    macos_paste_shortcut_selection = (
        _create_macos_paste_shortcut_selection(
            services,
            settings.macos_paste_shortcut,
        )
    )
    if macos_paste_shortcut_selection is None:
        paster = services.create_clipboard_paster(hotkey)
    else:
        paster = services.create_clipboard_paster(
            hotkey,
            paste_shortcut_selection=macos_paste_shortcut_selection,
        )
    session = VoiceSession(
        machine,
        recorder,
        recognizer,
        paster,
        tail_overlap_seconds=settings.tail_overlap_seconds,
        preview_window_seconds=settings.preview_window_seconds,
        text_postprocessor=create_text_postprocessor(
            normalize_numbers=settings.normalize_numbers
        ),
    )
    view = DesktopView(
        supported_models=services.supported_models,
        macos_paste_shortcut=(
            macos_paste_shortcut_selection.shortcut
            if macos_paste_shortcut_selection is not None
            else None
        ),
    )
    runner = QtTaskRunner()
    hotword_editor = _create_hotword_editor(services, paths.hotwords_file)
    controller = AppController(
        machine=machine,
        settings=settings,
        settings_repository=SettingsRepository(paths.config_dir),
        recognizer=recognizer,
        session=session,
        view=view,
        task_runner=runner,
        foreground_window=services.capture_target,
        restart_application=lambda: app.exit(RESTART_EXIT_CODE),
        macos_paste_shortcut_selection=macos_paste_shortcut_selection,
        open_hotwords_file=(
            hotword_editor.open
            if hotword_editor is not None
            else lambda: _open_hotwords_file(paths.hotwords_file)
        ),
    )
    if hotword_editor is not None:
        hotword_editor.changed.connect(controller.hotwords_file_changed)
    bridge.toggle_requested.connect(controller.toggle_recording)
    bridge.cancel_requested.connect(controller.cancel)
    view.bind(controller, app.quit)

    try:
        hotkey.start()
    except (ImportError, OSError, RuntimeError) as exc:
        QMessageBox.critical(
            None, "VIM2", f"无法注册全局热键 {settings.hotkey}：{exc}"
        )
        guard.close()
        return 1

    cleaned_up = False

    def cleanup() -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        controller.shutdown()
        hotkey.stop()
        runner.wait_for_done()
        view.tray.hide()
        guard.close()

    app.aboutToQuit.connect(cleanup)
    _prepare_desktop_runtime(app, view, recognizer, controller)
    exit_code = app.exec()
    cleanup()
    if exit_code == RESTART_EXIT_CODE:
        logging.getLogger(__name__).info(
            "Restarting to isolate model backend"
        )
        _restart_process(paths)
    return exit_code
