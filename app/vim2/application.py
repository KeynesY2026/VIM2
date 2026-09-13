from __future__ import annotations

import ctypes
from typing import Protocol

from vim2.config import Settings
from vim2.paths import AppPaths

ERROR_ALREADY_EXISTS = 183
MUTEX_NAME = r"Local\VIM2.SingleInstance"


class MutexApi(Protocol):
    def CreateMutexW(
        self, security: object, initially_owned: bool, name: str
    ) -> int: ...

    def get_last_error(self) -> int: ...

    def CloseHandle(self, handle: int) -> None: ...


class _NativeMutexApi:
    def __init__(self) -> None:
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateMutexW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_wchar_p,
        )
        self._kernel32.CreateMutexW.restype = ctypes.c_void_p
        self._kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)

    def CreateMutexW(
        self, security: object, initially_owned: bool, name: str
    ) -> int:
        return int(
            self._kernel32.CreateMutexW(security, initially_owned, name) or 0
        )

    @staticmethod
    def get_last_error() -> int:
        return ctypes.get_last_error()

    def CloseHandle(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)


class SingleInstanceGuard:
    def __init__(
        self,
        *,
        api: MutexApi | None = None,
        name: str = MUTEX_NAME,
    ) -> None:
        self._api = api or _NativeMutexApi()
        self._name = name
        self._handle: int | None = None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        handle = self._api.CreateMutexW(None, False, self._name)
        if not handle:
            raise OSError(
                self._api.get_last_error(),
                "Cannot create the VIM2 single-instance mutex",
            )
        if self._api.get_last_error() == ERROR_ALREADY_EXISTS:
            self._api.CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def close(self) -> None:
        if self._handle is not None:
            self._api.CloseHandle(self._handle)
            self._handle = None


def run_desktop_application(paths: AppPaths, settings: Settings) -> int:
    from vim2.qt_runtime import run_qt_application

    return run_qt_application(paths, settings)
