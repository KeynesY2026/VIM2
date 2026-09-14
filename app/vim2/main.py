from __future__ import annotations

import argparse
import ctypes
import logging
import sys
from pathlib import Path
from typing import Sequence

from vim2.config import SettingsRepository
from vim2.diagnostics import configure_runtime_logging
from vim2.paths import AppPaths
from vim2.preflight import PreflightChecker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VIM2 portable voice input")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the portable runtime and selected model, then exit",
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _show_startup_message(message: str) -> None:
    ctypes.WinDLL("user32", use_last_error=True).MessageBoxW(
        None, message, "VIM2", 0x00000010
    )


def _report_startup_error(message: str, *, windowed: bool) -> None:
    if windowed:
        _show_startup_message(message)
    else:
        print(message, file=sys.stderr)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = AppPaths.from_root(args.root)
    configure_runtime_logging(paths)
    logger = logging.getLogger(__name__)
    logger.info("VIM2 starting with Python %s", sys.version.split()[0])
    try:
        settings = SettingsRepository(paths.config_dir).load()
    except (OSError, ValueError) as exc:
        _report_startup_error(
            f"VIM2 cannot start:\n- Invalid configuration: {exc}",
            windowed=args.windowed,
        )
        return 1

    result = PreflightChecker(paths).check(
        settings.selected_model,
        check_cuda=args.check and not args.skip_cuda_check,
        check_runtime=not args.skip_runtime_check,
    )
    if not result.ok:
        logger.error("Preflight failed: %s", "; ".join(result.errors))
        details = "\n".join(f"- {error}" for error in result.errors)
        _report_startup_error(
            f"VIM2 cannot start:\n{details}",
            windowed=args.windowed,
        )
        return 1

    if args.check:
        print("VIM2 preflight check passed.")
        return 0

    from vim2.application import run_desktop_application

    logger.info("Starting desktop application with model %s", settings.selected_model)
    return run_desktop_application(paths, settings)


if __name__ == "__main__":
    raise SystemExit(run())
