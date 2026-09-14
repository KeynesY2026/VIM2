from __future__ import annotations

import faulthandler
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import ModuleType
from typing import TextIO

from vim2.paths import AppPaths


_fault_stream: TextIO | None = None


def configure_runtime_logging(
    paths: AppPaths,
    *,
    fault_handler: ModuleType = faulthandler,
) -> Path:
    global _fault_stream

    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = paths.runtime_dir / "vim2.log"
    logger = logging.getLogger("vim2")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_vim2_runtime_handler", False):
            logger.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        log_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler._vim2_runtime_handler = True
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(threadName)s "
            "%(name)s: %(message)s"
        )
    )
    logger.addHandler(handler)

    if _fault_stream is not None:
        _fault_stream.close()
    _fault_stream = log_path.open("a", encoding="utf-8", buffering=1)
    fault_handler.enable(file=_fault_stream, all_threads=True)

    def log_unhandled_exception(exc_type, exc_value, traceback) -> None:
        logger.critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, traceback),
        )

    def log_thread_exception(args: threading.ExceptHookArgs) -> None:
        logger.critical(
            "Unhandled exception in thread %s",
            args.thread.name if args.thread else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = log_unhandled_exception
    threading.excepthook = log_thread_exception
    logger.info("Runtime diagnostics initialized")
    return log_path