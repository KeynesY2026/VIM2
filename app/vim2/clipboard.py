from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Callable, Protocol

CF_UNICODETEXT = 13
GHND = 0x0042
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56


class ClipboardPasteError(RuntimeError):
    pass


class WindowsApi(Protocol):
    def set_clipboard_text(self, text: str) -> None: ...

    def set_foreground_window(self, handle: int) -> bool: ...

    def send_ctrl_v(self) -> bool: ...


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (("ki", _KeyboardInput),)


class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = (("type", wintypes.DWORD), ("value", _InputUnion))


class NativeWindowsApi:
    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
        self._kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        self._kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
        self._kernel32.GlobalLock.restype = wintypes.LPVOID
        self._kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
        self._kernel32.GlobalUnlock.restype = wintypes.BOOL
        self._kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
        self._kernel32.GlobalFree.restype = wintypes.HGLOBAL
        self._user32.SetClipboardData.argtypes = (
            wintypes.UINT,
            wintypes.HANDLE,
        )
        self._user32.SetClipboardData.restype = wintypes.HANDLE
        self._user32.SendInput.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(_Input),
            ctypes.c_int,
        )
        self._user32.SendInput.restype = wintypes.UINT

    def set_clipboard_text(self, text: str) -> None:
        opened = False
        for _ in range(10):
            if self._user32.OpenClipboard(None):
                opened = True
                break
            time.sleep(0.01)
        if not opened:
            raise OSError(
                ctypes.get_last_error(), "Cannot open the Windows clipboard"
            )

        memory = None
        try:
            if not self._user32.EmptyClipboard():
                raise OSError(
                    ctypes.get_last_error(), "Cannot clear the Windows clipboard"
                )
            encoded = text.encode("utf-16-le") + b"\0\0"
            memory = self._kernel32.GlobalAlloc(GHND, len(encoded))
            if not memory:
                raise MemoryError("Cannot allocate clipboard memory")
            pointer = self._kernel32.GlobalLock(memory)
            if not pointer:
                raise OSError(
                    ctypes.get_last_error(), "Cannot lock clipboard memory"
                )
            try:
                ctypes.memmove(pointer, encoded, len(encoded))
            finally:
                self._kernel32.GlobalUnlock(memory)
            if not self._user32.SetClipboardData(CF_UNICODETEXT, memory):
                raise OSError(
                    ctypes.get_last_error(), "Cannot write recognized text"
                )
            memory = None
        finally:
            self._user32.CloseClipboard()
            if memory:
                self._kernel32.GlobalFree(memory)

    def set_foreground_window(self, handle: int) -> bool:
        if self._user32.GetForegroundWindow() == handle:
            return True
        target_thread = self._user32.GetWindowThreadProcessId(handle, None)
        current_thread = self._kernel32.GetCurrentThreadId()
        attached = (
            target_thread
            and target_thread != current_thread
            and self._user32.AttachThreadInput(
                current_thread, target_thread, True
            )
        )
        try:
            self._user32.ShowWindow(handle, 9)
            self._user32.BringWindowToTop(handle)
            return bool(self._user32.SetForegroundWindow(handle))
        finally:
            if attached:
                self._user32.AttachThreadInput(
                    current_thread, target_thread, False
                )

    def send_ctrl_v(self) -> bool:
        events = (_Input * 4)(
            self._key_event(VK_CONTROL, 0),
            self._key_event(VK_V, 0),
            self._key_event(VK_V, KEYEVENTF_KEYUP),
            self._key_event(VK_CONTROL, KEYEVENTF_KEYUP),
        )
        return self._user32.SendInput(
            len(events), events, ctypes.sizeof(_Input)
        ) == len(events)

    @staticmethod
    def _key_event(key: int, flags: int) -> _Input:
        return _Input(
            type=INPUT_KEYBOARD,
            value=_InputUnion(
                ki=_KeyboardInput(
                    wVk=key,
                    wScan=0,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )


class WindowsClipboardPaster:
    def __init__(
        self,
        api: WindowsApi | None = None,
        *,
        wait_until_hotkey_released: Callable[[], None],
    ) -> None:
        self._api = api or NativeWindowsApi()
        self._wait_until_hotkey_released = wait_until_hotkey_released

    def paste(self, text: str, target_window: int) -> None:
        self._wait_until_hotkey_released()
        try:
            self._api.set_clipboard_text(text)
        except (OSError, MemoryError) as exc:
            raise ClipboardPasteError(
                f"Cannot copy recognized text to the clipboard: {exc}"
            ) from exc
        if not self._api.set_foreground_window(target_window):
            raise ClipboardPasteError(
                "Cannot restore the target window; recognized text remains "
                "in the clipboard."
            )
        time.sleep(0.05)
        if not self._api.send_ctrl_v():
            raise ClipboardPasteError(
                "Cannot send Ctrl+V; recognized text remains in the clipboard."
            )
