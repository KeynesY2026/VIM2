import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

import vim2.qt_runtime as qt_runtime
from vim2.models import ModelId
from vim2.qt_runtime import HotwordEditorProcess, QtTaskRunner


def _run_until_callback(submit) -> tuple[list[object], list[Exception]]:
    QApplication.instance() or QApplication([])
    loop = QEventLoop()
    results: list[object] = []
    errors: list[Exception] = []

    def success(result: object) -> None:
        results.append(result)
        loop.quit()

    def failure(error: Exception) -> None:
        errors.append(error)
        loop.quit()

    submit(success, failure)
    QTimer.singleShot(3000, loop.quit)
    loop.exec()
    return results, errors


def test_qt_task_runner_returns_result_on_ui_thread() -> None:
    runner = QtTaskRunner()

    results, errors = _run_until_callback(
        lambda success, failure: runner.submit(
            lambda: 42, success, failure
        )
    )

    assert results == [42]
    assert errors == []


def test_qt_task_runner_surfaces_worker_error() -> None:
    runner = QtTaskRunner()

    def fail() -> None:
        raise RuntimeError("worker failed")

    results, errors = _run_until_callback(
        lambda success, failure: runner.submit(fail, success, failure)
    )

    assert results == []
    assert len(errors) == 1
    assert str(errors[0]) == "worker failed"


def test_qt_task_runner_surfaces_unexpected_worker_exception() -> None:
    runner = QtTaskRunner()

    def fail() -> None:
        raise KeyError("unexpected worker failure")

    results, errors = _run_until_callback(
        lambda success, failure: runner.submit(fail, success, failure)
    )

    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], KeyError)


def test_model_runtime_initializes_on_ui_thread_after_tray_is_visible() -> None:
    events: list[str] = []

    class App:
        def processEvents(self) -> None:
            events.append("events processed")

    class View:
        def show(self) -> None:
            events.append("tray shown")

    class Recognizer:
        def initialize_runtime(self, model_id: ModelId) -> None:
            assert model_id is ModelId.CPU
            events.append("runtime initialized")

    class Controller:
        selected_model = ModelId.CPU

        def start(self) -> None:
            events.append("background model load started")

    qt_runtime._prepare_desktop_runtime(
        App(),
        View(),
        Recognizer(),
        Controller(),
    )

    assert events == [
        "tray shown",
        "events processed",
        "runtime initialized",
        "background model load started",
    ]


class FakeProcess:
    def __init__(self) -> None:
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code


def test_hotword_editor_checks_modified_time_only_after_process_closes(
    tmp_path: Path,
) -> None:
    _app = QApplication.instance() or QApplication([])
    path = tmp_path / "config" / "hotwords.txt"
    process = FakeProcess()
    commands: list[list[str]] = []
    editor = HotwordEditorProcess(
        path,
        launch=lambda command: commands.append(command) or process,
    )
    spy = QSignalSpy(editor.changed)

    assert editor.open()
    assert path.read_text(encoding="utf-8").startswith("#")
    assert commands == [["notepad.exe", str(path.resolve())]]

    path.write_text("VIM2\n", encoding="utf-8")
    editor.check_process()

    assert spy.count() == 0

    process.exit_code = 0
    editor.check_process()

    assert spy.count() == 1


def test_hotword_editor_does_not_reload_after_unchanged_process_closes(
    tmp_path: Path,
) -> None:
    _app = QApplication.instance() or QApplication([])
    path = tmp_path / "config" / "hotwords.txt"
    process = FakeProcess()
    editor = HotwordEditorProcess(path, launch=lambda command: process)
    spy = QSignalSpy(editor.changed)

    assert editor.open()
    process.exit_code = 0
    editor.check_process()

    assert spy.count() == 0
