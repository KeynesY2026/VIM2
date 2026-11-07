from __future__ import annotations

import ctypes
import logging
import ntpath
import os
from collections.abc import Callable
from ctypes import wintypes

EVENT_SYSTEM_FOREGROUND = 0x0003
WINEVENT_OUTOFCONTEXT = 0x0000
OBJID_WINDOW = 0
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_SHELL_PROCESS_NAMES = frozenset(
    {
        "shellexperiencehost.exe",
        "shellinfrastructurehost.exe",
        "startmenuexperiencehost.exe",
        "searchhost.exe",
        "searchapp.exe",
        "textinputhost.exe",
        "lockapp.exe",
        "widgets.exe",
    }
)

_SHELL_TRAY_CLASSES = frozenset(
    {
        "Shell_TrayWnd",
        "Shell_SecondaryTrayWnd",
        "NotifyIconOverflowWindow",
        "TopLevelWindowForOverflowXamlIsland",
    }
)


class ForegroundWindowHistory:
    def __init__(
        self,
        *,
        process_id: int,
        describe_window: Callable[[int], tuple[int, str, str] | None],
    ) -> None:
        self._process_id = process_id
        self._describe_window = describe_window
        self.last_window = 0

    def record(self, handle: int) -> None:
        if not handle:
            return
        details = self._describe_window(handle)
        if details is None:
            return
        process_id, class_name, executable_name = details
        if (
            process_id == self._process_id
            or class_name in _SHELL_TRAY_CLASSES
            or executable_name.casefold() in _SHELL_PROCESS_NAMES
        ):
            return
        self.last_window = handle


class ForegroundWindowTracker:
    def __init__(self, activate_window: Callable[[int], bool]) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_api()
        self._activate_window = activate_window
        self._history = ForegroundWindowHistory(
            process_id=os.getpid(), describe_window=self._describe_window
        )
        callback_type = ctypes.WINFUNCTYPE(
            None,
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.HWND,
            wintypes.LONG,
            wintypes.LONG,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        self._callback = callback_type(self._on_foreground_changed)
        self._hook = self._user32.SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND,
            EVENT_SYSTEM_FOREGROUND,
            None,
            self._callback,
            0,
            0,
            WINEVENT_OUTOFCONTEXT,
        )
        if not self._hook:
            logging.getLogger(__name__).warning(
                "Cannot monitor foreground windows: %s",
                ctypes.WinError(ctypes.get_last_error()),
            )
        self._history.record(self._foreground_window())

    def __call__(self) -> int:
        self._history.record(self._foreground_window())
        handle = self._history.last_window
        if handle and not self._user32.IsWindow(handle):
            self._history.last_window = 0
        return self._history.last_window

    def restore_last_active(self) -> None:
        handle = self()
        if handle and not self._activate_window(handle):
            logging.getLogger(__name__).warning(
                "Cannot restore previous foreground window: %s", handle
            )

    def close(self) -> None:
        if self._hook:
            self._user32.UnhookWinEvent(self._hook)
            self._hook = None

    def _on_foreground_changed(
        self,
        hook,
        event,
        handle,
        object_id,
        child_id,
        event_thread,
        event_time,
    ) -> None:
        del hook, event, child_id, event_thread, event_time
        if object_id == OBJID_WINDOW:
            self._history.record(int(handle or 0))

    def _describe_window(
        self, handle: int
    ) -> tuple[int, str, str] | None:
        if not self._user32.IsWindow(handle):
            return None
        process_id = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(
            handle, ctypes.byref(process_id)
        )
        class_name = ctypes.create_unicode_buffer(256)
        if not self._user32.GetClassNameW(
            handle, class_name, len(class_name)
        ):
            return None
        executable_name = self._process_name(process_id.value)
        return process_id.value, class_name.value, executable_name

    def _process_name(self, process_id: int) -> str:
        handle = self._kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, process_id
        )
        if not handle:
            return ""
        try:
            image_path = ctypes.create_unicode_buffer(260)
            length = wintypes.DWORD(len(image_path))
            if not self._kernel32.QueryFullProcessImageNameW(
                handle, 0, image_path, ctypes.byref(length)
            ):
                return ""
            return ntpath.basename(image_path.value)
        finally:
            self._kernel32.CloseHandle(handle)

    def _foreground_window(self) -> int:
        return int(self._user32.GetForegroundWindow() or 0)

    def _configure_api(self) -> None:
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.IsWindow.argtypes = (wintypes.HWND,)
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        )
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._user32.GetClassNameW.argtypes = (
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        )
        self._user32.GetClassNameW.restype = ctypes.c_int
        self._user32.SetWinEventHook.argtypes = (
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
            ctypes.WINFUNCTYPE(
                None,
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.HWND,
                wintypes.LONG,
                wintypes.LONG,
                wintypes.DWORD,
                wintypes.DWORD,
            ),
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        self._user32.SetWinEventHook.restype = wintypes.HANDLE
        self._user32.UnhookWinEvent.argtypes = (wintypes.HANDLE,)
        self._user32.UnhookWinEvent.restype = wintypes.BOOL
        self._kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        )
        self._kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL
