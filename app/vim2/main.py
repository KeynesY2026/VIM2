from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Sequence

from vim2.config import SettingsRepository
from vim2.diagnostics import configure_runtime_logging
from vim2.paths import AppPaths
from vim2.models import ModelId
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
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--data-root", type=Path, help=argparse.SUPPRESS)
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
    parser.add_argument(
        "--import-smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _show_startup_message(message: str) -> None:
    show_startup_error(message)


def _run_import_smoke(*, windowed: bool) -> int:
    """Import frozen native modules without platform services, GUI, or TCC prompts."""
    try:
        for module in (
            "sherpa_onnx",
            "sounddevice",
            "soundfile",
            "PySide6.QtWidgets",
        ):
            importlib.import_module(module)
    except (ImportError, OSError, RuntimeError) as exc:
        _report_startup_error(str(exc), windowed=windowed)
        return 1
    print("VIM2 native import smoke passed.")
    return 0


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


DISK_IMAGE_LAUNCH_MESSAGE = """请先把 VIM2.app 拖到“应用程序”（Applications），不要从磁盘镜像直接启动。
Drag VIM2.app into Applications. Do not launch it from the disk image.

1. 将 VIM2.app 拖到 Applications，并等待约 1.1GB 复制完成。
2. 弹出此磁盘镜像。
3. 从“应用程序”中的 VIM2 启动。未签名内测版请右键选择“打开”。

在镜像里启动会直接退出，并且不会在这个位置请求辅助功能或输入监控权限。
Launching from the image quits without requesting Accessibility or Input Monitoring for this copy.
"""


def launched_from_disk_image(executable: str | Path | None = None) -> bool:
    """Match mounted-image apps and the standard Gatekeeper translocation layout.

    Do not resolve symlinks: the launch path, not the original file behind a
    translocated copy, determines which app identity would request permissions.
    """
    raw = sys.executable if executable is None else executable
    parts = tuple(part.casefold() for part in Path(raw).parts)
    if not parts or parts[0] != "/":
        return False
    # Only a real bundle executable has the exact trailing bundle structure.
    if len(parts) < 7 or parts[-3:-1] != ("contents", "macos"):
        return False
    app = parts[-4]
    if len(app) <= 4 or not app.endswith(".app") or not parts[-1]:
        return False
    if parts[1] == "volumes":
        # The volume name must not masquerade as the .app bundle.
        return len(parts) == 7 and not parts[2].endswith(".app")
    # macOS uses /private/var/folders/<bucket>/<user>[/T]/AppTranslocation/
    # <token>/d/[nested dirs/]<app>.app/Contents/MacOS/<executable>.
    if parts[1:4] != ("private", "var", "folders"):
        return False
    for prefix in (("apptranslocation",), ("t", "apptranslocation")):
        start = 6
        if parts[start : start + len(prefix)] != prefix:
            continue
        tail = parts[start + len(prefix) :]
        if len(tail) >= 6 and tail[0] and tail[1] == "d":
            return True
    return False


def _blocked_disk_image_launch(args: argparse.Namespace) -> bool:
    if args.check or args.import_smoke or sys.platform != "darwin" or not getattr(sys, "frozen", False):
        return False
    return launched_from_disk_image(sys.executable)


def _should_request_gui_permissions(args: argparse.Namespace, profile: object) -> bool:
    """Frozen installed GUI launches may prompt; checks and source launches must not."""
    return (
        bool(getattr(sys, "frozen", False))
        and getattr(profile, "name", "") == "macos"
        and not args.check
        and not args.import_smoke
    )


def _macos_gui_permission_guidance(platform_services: object) -> str | None:
    from vim2.platform_macos import MacOSPlatformServices

    if not isinstance(platform_services, MacOSPlatformServices):
        return None
    return platform_services.prepare_gui_permissions()


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(sys, "frozen", False) and args.root is not None:
        _report_startup_error("Frozen builds do not accept --root", windowed=args.windowed)
        return 1
    if not getattr(sys, "frozen", False) and args.data_root is not None:
        _report_startup_error("--data-root is for frozen builds only", windowed=args.windowed)
        return 1
    if _blocked_disk_image_launch(args):
        _report_startup_error(DISK_IMAGE_LAUNCH_MESSAGE, windowed=True)
        return 1
    try:
        paths = (
            AppPaths.for_launch(data_root=args.data_root)
            if getattr(sys, "frozen", False)
            else AppPaths.from_root(args.root or Path(__file__).resolve().parents[2])
        )
        paths.initialize_user_data()
        configure_runtime_logging(paths)
    except OSError as exc:
        _report_startup_error(f"VIM2 cannot initialize user data: {exc}", windowed=args.windowed)
        return 1
    if args.import_smoke:
        return _run_import_smoke(windowed=args.windowed)
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
        if _should_request_gui_permissions(args, profile):
            guidance = _macos_gui_permission_guidance(platform_services)
            if guidance:
                logger.error("macOS GUI permissions are missing")
                _report_startup_error(guidance, windowed=True)
                return 1
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
        if not args.skip_runtime_check and settings.selected_model is ModelId.CPU:
            try:
                for module in ("sherpa_onnx", "sounddevice", "soundfile", "PySide6.QtWidgets"):
                    importlib.import_module(module)
            except (ImportError, OSError, RuntimeError) as exc:
                _report_startup_error(
                    f"VIM2 CPU runtime cannot be loaded: {exc}", windowed=args.windowed
                )
                return 1
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
