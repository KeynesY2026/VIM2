from __future__ import annotations

import ctypes
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
    Signal,
    Slot,
)
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from vim2.application import SingleInstanceGuard
from vim2.audio import AudioRecorder
from vim2.clipboard import WindowsClipboardPaster
from vim2.config import Settings, SettingsRepository
from vim2.controller import AppController
from vim2.hotkey import (
    HotkeyDispatcher,
    WindowsHotkeyListener,
    parse_hotkey,
)
from vim2.hotwords import HotwordRepository
from vim2.paths import AppPaths
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


def _enable_per_monitor_dpi() -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))


def _foreground_window() -> int:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    return int(user32.GetForegroundWindow() or 0)


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


def run_qt_application(paths: AppPaths, settings: Settings) -> int:
    _enable_per_monitor_dpi()
    app = QApplication(sys.argv)
    app.setApplicationName("VIM2")
    app.setQuitOnLastWindowClosed(False)

    guard = SingleInstanceGuard()
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
    hotkey = WindowsHotkeyListener(dispatcher)
    paster = WindowsClipboardPaster(
        wait_until_hotkey_released=hotkey.wait_until_released
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
    view = DesktopView()
    runner = QtTaskRunner()
    hotword_editor = HotwordEditorProcess(paths.hotwords_file)
    controller = AppController(
        machine=machine,
        settings=settings,
        settings_repository=SettingsRepository(paths.config_dir),
        recognizer=recognizer,
        session=session,
        view=view,
        task_runner=runner,
        foreground_window=_foreground_window,
        restart_application=lambda: app.exit(RESTART_EXIT_CODE),
        open_hotwords_file=hotword_editor.open,
    )
    hotword_editor.changed.connect(controller.hotwords_file_changed)
    bridge.toggle_requested.connect(controller.toggle_recording)
    bridge.cancel_requested.connect(controller.cancel)
    view.bind(controller, app.quit)

    try:
        hotkey.start()
    except (OSError, RuntimeError) as exc:
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
