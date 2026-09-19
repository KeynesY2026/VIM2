from __future__ import annotations

import ctypes

from vim2.application import SingleInstanceGuard
from vim2.clipboard import WindowsClipboardPaster
from vim2.config import MacPasteShortcutSelection
from vim2.hotkey import HotkeyDispatcher, WindowsHotkeyListener
from vim2.paths import AppPaths
from vim2.platform_services import (
    GlobalHotkeyService,
    PlatformProfile,
    TextPaster,
)


class WindowsPlatformServices:
    def __init__(self, paths: AppPaths, profile: PlatformProfile) -> None:
        self._paths = paths
        self.profile = profile

    @property
    def supported_models(self):
        return self.profile.supported_models

    def preflight_errors(self) -> tuple[str, ...]:
        return ()

    @staticmethod
    def enable_desktop_features() -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))

    @staticmethod
    def create_single_instance_guard() -> SingleInstanceGuard:
        return SingleInstanceGuard()

    @staticmethod
    def capture_target() -> int:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        return int(user32.GetForegroundWindow() or 0)

    @staticmethod
    def create_hotkey_listener(
        dispatcher: object,
    ) -> WindowsHotkeyListener:
        if not isinstance(dispatcher, HotkeyDispatcher):
            raise TypeError("dispatcher must be a HotkeyDispatcher")
        return WindowsHotkeyListener(dispatcher)

    @staticmethod
    def create_clipboard_paster(
        hotkey: GlobalHotkeyService,
        *,
        paste_shortcut_selection: MacPasteShortcutSelection | None = None,
    ) -> TextPaster:
        if paste_shortcut_selection is not None:
            raise ValueError(
                "Windows paste is fixed at Ctrl+V and has no macOS selection"
            )
        return WindowsClipboardPaster(
            wait_until_hotkey_released=hotkey.wait_until_released
        )


def show_startup_error(message: str) -> None:
    ctypes.WinDLL("user32", use_last_error=True).MessageBoxW(
        None, message, "VIM2", 0x00000010
    )
