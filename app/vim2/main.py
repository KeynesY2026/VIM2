from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from vim2.config import SettingsRepository
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
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = AppPaths.from_root(args.root)
    try:
        settings = SettingsRepository(paths.config_dir).load()
    except (OSError, ValueError) as exc:
        print(f"VIM2 cannot start:\n- Invalid configuration: {exc}", file=sys.stderr)
        return 1

    result = PreflightChecker(paths).check(
        settings.selected_model,
        check_cuda=not args.skip_cuda_check,
        check_runtime=not args.skip_runtime_check,
    )
    if not result.ok:
        details = "\n".join(f"- {error}" for error in result.errors)
        print(f"VIM2 cannot start:\n{details}", file=sys.stderr)
        return 1

    if args.check:
        print("VIM2 preflight check passed.")
        return 0

    from vim2.application import run_desktop_application

    return run_desktop_application(paths, settings)


if __name__ == "__main__":
    raise SystemExit(run())
