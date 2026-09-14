import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

import vim2.qt_runtime as qt_runtime
from vim2.models import ModelId
from vim2.qt_runtime import QtTaskRunner


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


def test_open_local_file_uses_qt_file_url(
    tmp_path: Path, monkeypatch
) -> None:
    opened = []
    monkeypatch.setattr(
        qt_runtime.QDesktopServices,
        "openUrl",
        lambda url: opened.append(url) or True,
    )
    path = tmp_path / "配置 空格" / "hotwords.txt"

    result = qt_runtime._open_local_file(path)

    assert result is True
    assert opened[0].isLocalFile()
    assert Path(opened[0].toLocalFile()) == path
