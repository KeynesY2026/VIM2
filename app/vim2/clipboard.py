from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Callable, Protocol

from vim2.hotkey import VIM2_INPUT_MARKER

CF_UNICODETEXT = 13
GHND = 0x0042
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56
VK_ESCAPE = 0x1B
GUI_INMENUMODE = 0x00000004
GUI_SYSTEMMENUMODE = 0x00000008
GUI_POPUPMENUMODE = 0x00000010
CLIPBOARD_SYNC_DELAY_SECONDS = 0.3


class ClipboardPasteError(RuntimeError):
    pass


class WindowsApi(Protocol):
    def set_clipboard_text(self, text: str) -> None: ...

    def set_foreground_window(self, handle: int) -> bool: ...

    def exit_menu_mode(self, handle: int) -> None: ...

    def wait_for_clipboard_sync(self) -> None: ...

    def send_ctrl_v(self) -> bool: ...


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class _HardwareInput(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (
        ("mi", _MouseInput),
        ("ki", _KeyboardInput),
        ("hi", _HardwareInput),
    )


class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = (("type", wintypes.DWORD), ("value", _InputUnion))


class _Rect(ctypes.Structure):
    _fields_ = (
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    )


class _GuiThreadInfo(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", _Rect),
    )


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
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.IsWindow.argtypes = (wintypes.HWND,)
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        self._user32.SetForegroundWindow.restype = wintypes.BOOL
        self._user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        )
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._user32.AttachThreadInput.argtypes = (
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.BOOL,
        )
        self._user32.AttachThreadInput.restype = wintypes.BOOL
        self._user32.GetGUIThreadInfo.argtypes = (
            wintypes.DWORD,
            ctypes.POINTER(_GuiThreadInfo),
        )
        self._user32.GetGUIThreadInfo.restype = wintypes.BOOL
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
        if not handle or not self._user32.IsWindow(handle):
            return False
        if self._user32.GetForegroundWindow() == handle:
            return True
        foreground = self._user32.GetForegroundWindow()
        target_thread = self._user32.GetWindowThreadProcessId(handle, None)
        foreground_thread = self._user32.GetWindowThreadProcessId(
            foreground, None
        )
        current_thread = self._kernel32.GetCurrentThreadId()
        attached_foreground = (
            foreground_thread
            and foreground_thread != current_thread
            and self._user32.AttachThreadInput(
                current_thread, foreground_thread, True
            )
        )
        attached_target = (
            target_thread
            and target_thread != current_thread
            and target_thread != foreground_thread
            and self._user32.AttachThreadInput(
                current_thread, target_thread, True
            )
        )
        try:
            self._user32.ShowWindow(handle, 5)
            self._user32.BringWindowToTop(handle)
            self._user32.SetForegroundWindow(handle)
        finally:
            if attached_target:
                self._user32.AttachThreadInput(
                    current_thread, target_thread, False
                )
            if attached_foreground:
                self._user32.AttachThreadInput(
                    current_thread, foreground_thread, False
                )
        time.sleep(0.1)
        return self._user32.GetForegroundWindow() == handle

    def exit_menu_mode(self, handle: int) -> None:
        target_thread = self._user32.GetWindowThreadProcessId(handle, None)
        if not target_thread:
            return
        info = _GuiThreadInfo(cbSize=ctypes.sizeof(_GuiThreadInfo))
        if not self._user32.GetGUIThreadInfo(
            target_thread, ctypes.byref(info)
        ):
            return
        menu_mask = (
            GUI_INMENUMODE | GUI_SYSTEMMENUMODE | GUI_POPUPMENUMODE
        )
        if not info.flags & menu_mask:
            return
        self._send_keys(
            (
                self._key_event(VK_ESCAPE, 0),
                self._key_event(VK_ESCAPE, KEYEVENTF_KEYUP),
            )
        )
        time.sleep(0.01)

    def send_ctrl_v(self) -> bool:
        return self._send_keys(
            (
                self._key_event(VK_CONTROL, 0),
                self._key_event(VK_V, 0),
                self._key_event(VK_V, KEYEVENTF_KEYUP),
                self._key_event(VK_CONTROL, KEYEVENTF_KEYUP),
            )
        )

    def wait_for_clipboard_sync(self) -> None:
        time.sleep(CLIPBOARD_SYNC_DELAY_SECONDS)

    def _send_keys(self, keys: tuple[_Input, ...]) -> bool:
        events = (_Input * len(keys))(*keys)
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
                    dwExtraInfo=VIM2_INPUT_MARKER,
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
        self._api.exit_menu_mode(target_window)
        self._api.wait_for_clipboard_sync()
        if not self._api.send_ctrl_v():
            raise ClipboardPasteError(
                "Cannot send Ctrl+V; recognized text remains in the clipboard."
            )
