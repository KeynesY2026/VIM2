from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from vim2.config import SettingsRepository
from vim2.diagnostics import configure_runtime_logging
from vim2.paths import AppPaths
from vim2.platform_services import (
    create_platform_services,
    select_platform_profile,
    show_startup_error,
)
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
    show_startup_error(message)


def _report_startup_error(message: str, *, windowed: bool) -> None:
    if windowed:
        try:
            _show_startup_message(message)
            return
        except (ImportError, OSError, RuntimeError):
            logging.getLogger(__name__).exception(
                "Native startup error dialog is unavailable"
            )
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

    profile = select_platform_profile()
    result = PreflightChecker(paths, platform_profile=profile).check(
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

    try:
        platform_services = create_platform_services(paths, profile=profile)
        platform_errors = (
            platform_services.preflight_errors()
            if not args.skip_runtime_check
            else ()
        )
    except (ImportError, OSError, RuntimeError) as exc:
        platform_errors = (
            f"Cannot initialize the {profile.name} native adapter: {exc}",
        )
        platform_services = None
    if platform_errors:
        logger.error("Platform preflight failed: %s", "; ".join(platform_errors))
        details = "\n".join(f"- {error}" for error in platform_errors)
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
    return run_desktop_application(
        paths,
        settings,
        platform_services=platform_services,
    )


if __name__ == "__main__":
    raise SystemExit(run())
