import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

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
