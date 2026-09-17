from __future__ import annotations

import re
import ctypes
import threading
import time
from dataclasses import dataclass
from ctypes import wintypes
from typing import Callable

VIM2_INPUT_MARKER = 0x56494D32
_TOGGLE_DEBOUNCE_SECONDS = 0.25

_GENERIC_MODIFIERS = {
    "Ctrl": frozenset({"LeftCtrl", "RightCtrl"}),
    "Alt": frozenset({"LeftAlt", "RightAlt"}),
    "Shift": frozenset({"LeftShift", "RightShift"}),
    "Win": frozenset({"LeftWin", "RightWin"}),
}
_SPECIFIC_MODIFIERS = frozenset(
    side for sides in _GENERIC_MODIFIERS.values() for side in sides
)
_ALIASES = {
    "control": "Ctrl",
    "ctrl": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "win": "Win",
    "windows": "Win",
    **{name.lower(): name for name in _SPECIFIC_MODIFIERS},
}
_NAMED_KEYS = frozenset(
    {
        "Backspace",
        "Delete",
        "Down",
        "End",
        "Enter",
        "Esc",
        "Home",
        "Insert",
        "Left",
        "PageDown",
        "PageUp",
        "Right",
        "Space",
        "Tab",
        "Up",
    }
)


@dataclass(frozen=True, slots=True)
class HotkeyChord:
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KeyEvent:
    key: str
    is_down: bool


@dataclass(frozen=True, slots=True)
class HotkeyDecision:
    triggered: bool
    suppress: bool


def _canonical_key(value: str) -> str:
    stripped = value.strip()
    alias = _ALIASES.get(stripped.lower())
    if alias:
        return alias
    for named in _NAMED_KEYS:
        if stripped.lower() == named.lower():
            return named
    if re.fullmatch(r"[A-Za-z0-9]", stripped):
        return stripped.upper()
    if re.fullmatch(r"F(?:[1-9]|1[0-9]|2[0-4])", stripped, re.IGNORECASE):
        return stripped.upper()
    raise ValueError(f"Unsupported hotkey key: {value!r}")


def parse_hotkey(value: str) -> HotkeyChord:
    raw_keys = value.split("+")
    if not value.strip() or any(not key.strip() for key in raw_keys):
        raise ValueError("Hotkey must contain one or more keys")
    keys = tuple(dict.fromkeys(_canonical_key(key) for key in raw_keys))
    return HotkeyChord(keys)


def _matches(configured: str, pressed: set[str]) -> bool:
    if configured in _GENERIC_MODIFIERS:
        return bool(_GENERIC_MODIFIERS[configured] & pressed)
    return configured in pressed


def _event_matches(configured: str, event_key: str) -> bool:
    if configured in _GENERIC_MODIFIERS:
        return event_key in _GENERIC_MODIFIERS[configured]
    return configured == event_key


class HotkeyMatcher:
    def __init__(self, chord: HotkeyChord) -> None:
        self._chord = chord
        self._pressed: set[str] = set()
        self._armed = True

    def process(self, event: KeyEvent) -> HotkeyDecision:
        key = _canonical_key(event.key)
        if event.is_down:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)

        suppress = any(
            _event_matches(configured, key) for configured in self._chord.keys
        )
        complete = all(
            _matches(configured, self._pressed)
            for configured in self._chord.keys
        )
        triggered = event.is_down and complete and self._armed
        if triggered:
            self._armed = False
        elif not complete and not any(
            _matches(configured, self._pressed)
            for configured in self._chord.keys
        ):
            self._armed = True

        return HotkeyDecision(triggered=triggered, suppress=suppress)

    @property
    def is_fully_released(self) -> bool:
        return not any(
            _matches(configured, self._pressed)
            for configured in self._chord.keys
        )


_VK_NAMES = {
    0x08: "Backspace",
    0x09: "Tab",
    0x0D: "Enter",
    0x1B: "Esc",
    0x20: "Space",
    0x21: "PageUp",
    0x22: "PageDown",
    0x23: "End",
    0x24: "Home",
    0x25: "Left",
    0x26: "Up",
    0x27: "Right",
    0x28: "Down",
    0x2D: "Insert",
    0x2E: "Delete",
    0xA0: "LeftShift",
    0xA1: "RightShift",
    0xA2: "LeftCtrl",
    0xA3: "RightCtrl",
    0xA4: "LeftAlt",
    0xA5: "RightAlt",
    0x5B: "LeftWin",
    0x5C: "RightWin",
}


def key_name_from_virtual_key(virtual_key: int) -> str | None:
    if virtual_key in _VK_NAMES:
        return _VK_NAMES[virtual_key]
    if ord("0") <= virtual_key <= ord("9"):
        return chr(virtual_key)
    if ord("A") <= virtual_key <= ord("Z"):
        return chr(virtual_key)
    if 0x70 <= virtual_key <= 0x87:
        return f"F{virtual_key - 0x6F}"
    return None


class HotkeyDispatcher:
    def __init__(
        self,
        chord: HotkeyChord,
        *,
        on_toggle: Callable[[], None],
        on_cancel: Callable[[], None],
        is_cancellable: Callable[[], bool],
    ) -> None:
        self._matcher = HotkeyMatcher(chord)
        self._on_toggle = on_toggle
        self._on_cancel = on_cancel
        self._is_cancellable = is_cancellable
        self._released = threading.Event()
        self._released.set()
        self._last_toggle_at = float("-inf")

    def process(
        self,
        event: KeyEvent,
        *,
        lower_integrity_injected: bool,
        vim2_injected: bool,
    ) -> bool:
        if lower_integrity_injected or vim2_injected:
            return False
        if event.key == "Esc" and event.is_down and self._is_cancellable():
            self._on_cancel()
            return True
        decision = self._matcher.process(event)
        if self._matcher.is_fully_released:
            self._released.set()
        else:
            self._released.clear()
        if decision.triggered:
            now = time.monotonic()
            if now - self._last_toggle_at >= _TOGGLE_DEBOUNCE_SECONDS:
                self._last_toggle_at = now
                self._on_toggle()
        return decision.suppress

    def wait_until_released(self, timeout: float | None = None) -> None:
        self._released.wait(timeout)


class _LowLevelKeyboardInput(ctypes.Structure):
    _fields_ = (
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class WindowsHotkeyListener:
    _WH_KEYBOARD_LL = 13
    _WM_KEYDOWN = 0x0100
    _WM_KEYUP = 0x0101
    _WM_SYSKEYDOWN = 0x0104
    _WM_SYSKEYUP = 0x0105
    _WM_QUIT = 0x0012
    _LLKHF_LOWER_IL_INJECTED = 0x02

    def __init__(self, dispatcher: HotkeyDispatcher) -> None:
        self._dispatcher = dispatcher
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._hook: int | None = None
        self._callback: object | None = None
        self._ready = threading.Event()
        self._startup_error: OSError | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("The global hotkey listener is already running")
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._message_loop, name="VIM2 hotkey", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(3):
            raise RuntimeError("Timed out while installing the global hotkey")
        if self._startup_error:
            raise self._startup_error

    def stop(self) -> None:
        if self._thread_id is None:
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.PostThreadMessageW(self._thread_id, self._WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        self._thread_id = None

    def wait_until_released(self, timeout: float | None = None) -> None:
        self._dispatcher.wait_until_released(timeout)

    def _message_loop(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        self._thread_id = kernel32.GetCurrentThreadId()
        hook_type = ctypes.WINFUNCTYPE(
            wintypes.LPARAM, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
        )
        user32.SetWindowsHookExW.argtypes = (
            ctypes.c_int,
            hook_type,
            wintypes.HINSTANCE,
            wintypes.DWORD,
        )
        user32.SetWindowsHookExW.restype = wintypes.HHOOK
        user32.CallNextHookEx.argtypes = (
            wintypes.HHOOK,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        user32.CallNextHookEx.restype = wintypes.LPARAM

        def hook_callback(
            code: int, message: int, data_address: int
        ) -> int:
            if code >= 0:
                data = ctypes.cast(
                    data_address, ctypes.POINTER(_LowLevelKeyboardInput)
                ).contents
                key = key_name_from_virtual_key(data.vkCode)
                if key:
                    is_down = message in (
                        self._WM_KEYDOWN,
                        self._WM_SYSKEYDOWN,
                    )
                    lower_integrity_injected = bool(
                        data.flags & self._LLKHF_LOWER_IL_INJECTED
                    )
                    if self._dispatcher.process(
                        KeyEvent(key, is_down),
                        lower_integrity_injected=lower_integrity_injected,
                        vim2_injected=(
                            data.dwExtraInfo == VIM2_INPUT_MARKER
                        ),
                    ):
                        return 1
            return user32.CallNextHookEx(None, code, message, data_address)

        self._callback = hook_type(hook_callback)
        self._hook = user32.SetWindowsHookExW(
            self._WH_KEYBOARD_LL,
            self._callback,
            kernel32.GetModuleHandleW(None),
            0,
        )
        if not self._hook:
            self._startup_error = OSError(
                ctypes.get_last_error(),
                "Cannot install the Windows global keyboard hook",
            )
            self._ready.set()
            return
        self._ready.set()
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
        user32.UnhookWindowsHookEx(self._hook)
        self._hook = None
        self._callback = None
