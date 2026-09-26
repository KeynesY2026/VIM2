from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Protocol

from vim2.config import MacPasteShortcutSelection
from vim2.models import ModelId
from vim2.paths import AppPaths


class SingleInstanceService(Protocol):
    def acquire(self) -> bool: ...

    def close(self) -> None: ...


class GlobalHotkeyService(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def wait_until_released(self, timeout: float | None = None) -> None: ...


class TextPaster(Protocol):
    def paste(self, text: str, target_context: object) -> None: ...


@dataclass(frozen=True, slots=True)
class PlatformProfile:
    name: str
    supported_models: tuple[ModelId, ...]
    required_modules: tuple[str, ...] = ()
    required_python: tuple[int, int] | None = None
    preflight_errors: tuple[str, ...] = ()


class PlatformServices(Protocol):
    profile: PlatformProfile

    @property
    def supported_models(self) -> tuple[ModelId, ...]: ...

    def preflight_errors(self) -> tuple[str, ...]: ...

    def enable_desktop_features(self) -> None: ...

    def create_single_instance_guard(self) -> SingleInstanceService: ...

    def capture_target(self) -> object: ...

    def create_hotkey_listener(
        self, dispatcher: object
    ) -> GlobalHotkeyService: ...

    def create_clipboard_paster(
        self,
        hotkey: GlobalHotkeyService,
        *,
        paste_shortcut_selection: MacPasteShortcutSelection | None = None,
    ) -> TextPaster: ...


_WINDOWS_PROFILE = PlatformProfile(
    name="windows",
    supported_models=tuple(ModelId),
)
_MACOS_REQUIRED_MODULES = (
    "AppKit",
    "Quartz",
    "ApplicationServices",
    "AVFoundation",
    "CoreFoundation",
)


def select_platform_profile(
    system_name: str | None = None,
    machine: str | None = None,
) -> PlatformProfile:
    """Return the public capability policy for the current desktop host.

    ``system_name`` and ``machine`` are injectable so capability selection can
    be tested without importing either native adapter.
    """

    selected_system = (system_name or sys.platform).lower()
    selected_machine = (machine or platform.machine()).lower()
    if selected_system.startswith("win"):
        if getattr(sys, "frozen", False):
            return PlatformProfile(name="windows", supported_models=(ModelId.CPU,))
        return _WINDOWS_PROFILE
    if selected_system == "darwin":
        errors: tuple[str, ...] = ()
        if selected_machine != "arm64":
            errors = (
                "The macOS CPU MVP requires a native arm64 Python process; "
                f"the current architecture is {selected_machine or 'unknown'}. "
                "Install Apple Silicon Python 3.11 and do not run VIM2 under Rosetta.",
            )
        return PlatformProfile(
            name="macos",
            supported_models=(ModelId.CPU,),
            required_modules=_MACOS_REQUIRED_MODULES,
            required_python=(3, 11),
            preflight_errors=errors,
        )
    return PlatformProfile(
        name="unsupported",
        supported_models=(),
        preflight_errors=(
            f"Unsupported desktop platform: {selected_system or 'unknown'}. "
            "VIM2 currently has Windows and macOS Apple Silicon adapters.",
        ),
    )


def create_platform_services(
    paths: AppPaths,
    *,
    profile: PlatformProfile | None = None,
) -> PlatformServices:
    selected = profile or select_platform_profile()
    if selected.name == "windows":
        from vim2.platform_windows import WindowsPlatformServices

        return WindowsPlatformServices(paths, selected)
    if selected.name == "macos":
        from vim2.platform_macos import MacOSPlatformServices

        return MacOSPlatformServices(paths, selected)
    raise RuntimeError("; ".join(selected.preflight_errors))


def show_startup_error(
    message: str,
    *,
    system_name: str | None = None,
) -> None:
    """Display a native startup error without importing the Qt runtime."""

    selected = (system_name or sys.platform).lower()
    if selected.startswith("win"):
        from vim2.platform_windows import show_startup_error as show
    elif selected == "darwin":
        from vim2.platform_macos import show_startup_error as show
    else:
        raise RuntimeError(message)
    show(message)
