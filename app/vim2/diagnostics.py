from __future__ import annotations

import faulthandler
import io
import json
import logging
import sys
import tempfile
import threading
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, TextIO

from vim2.paths import AppPaths

if TYPE_CHECKING:
    from vim2.audio import AudioArtifact


_fault_stream: TextIO | None = None


def write_failure_case(
    destination: Path, audio: AudioArtifact, details: dict[str, object],
) -> None:
    import soundfile

    metadata = {
        **details,
        "audio": {
            "file": "audio.wav",
            "sample_rate": audio.sample_rate,
            "frames": audio.frame_count,
            "start_frame": audio.start_frame,
            "warnings": list(audio.warnings),
        },
        "environment": {
            "python": sys.version,
            "loaded_package_versions": {
                name: module.__version__
                for name in (
                    "numpy", "soundfile", "torch", "transformers",
                    "qwen_asr", "sherpa_onnx",
                )
                if (module := sys.modules.get(name)) is not None
                and isinstance(getattr(module, "__version__", None), str)
            },
        },
    }
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".vim2-failure-", suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with io.BytesIO() as wav:
            # Float WAV preserves the original samples without PCM quantization.
            soundfile.write(
                wav, audio.samples, audio.sample_rate, format="WAV", subtype="FLOAT"
            )
            with zipfile.ZipFile(temporary, "w") as archive:
                archive.writestr("audio.wav", wav.getvalue())
                archive.writestr(
                    "metadata.json",
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


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
    # Crash dumps go to their own file so that no handle is left open on
    # vim2.log. RotatingFileHandler renames vim2.log while rotating, and on
    # Windows that rename fails if anything else still has the file open.
    _fault_stream = (paths.runtime_dir / "vim2-fault.log").open(
        "a", encoding="utf-8", buffering=1
    )
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