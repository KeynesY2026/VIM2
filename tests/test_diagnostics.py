import logging
from pathlib import Path

import vim2.diagnostics as diagnostics
from vim2.paths import AppPaths


class RecordingFaultHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[object, bool]] = []

    def enable(self, *, file, all_threads: bool) -> None:
        self.calls.append((file, all_threads))


def test_configure_runtime_logging_writes_log_and_enables_fault_handler(
    tmp_path: Path,
) -> None:
    fault_handler = RecordingFaultHandler()
    paths = AppPaths.from_root(tmp_path)

    log_path = diagnostics.configure_runtime_logging(
        paths, fault_handler=fault_handler
    )
    logging.getLogger("vim2.test").warning("diagnostic marker")

    assert log_path == paths.runtime_dir / "vim2.log"
    assert "diagnostic marker" in log_path.read_text(encoding="utf-8")
    assert fault_handler.calls[0][1] is True
    assert fault_handler.calls[0][0].name == str(log_path)