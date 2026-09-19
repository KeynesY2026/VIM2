from __future__ import annotations

import importlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from vim2.clipboard import ClipboardPasteError
from vim2.config import MacPasteShortcut, MacPasteShortcutSelection
from vim2.hotkey import HotkeyDispatcher, KeyEvent, VIM2_INPUT_MARKER
from vim2.paths import AppPaths
from vim2.platform_services import PlatformProfile

CLIPBOARD_SYNC_DELAY_SECONDS = 0.3

# Device-dependent modifier masks from Darwin IOKit/hidsystem/IOLLEvent.h.
# PyObjC exposes the generic kCGEventFlagMask* values, but not these NX_* masks.
_NX_DEVICE_LEFT_CONTROL_MASK = 0x00000001
_NX_DEVICE_LEFT_SHIFT_MASK = 0x00000002
_NX_DEVICE_RIGHT_SHIFT_MASK = 0x00000004
_NX_DEVICE_LEFT_COMMAND_MASK = 0x00000008
_NX_DEVICE_RIGHT_COMMAND_MASK = 0x00000010
_NX_DEVICE_LEFT_ALTERNATE_MASK = 0x00000020
_NX_DEVICE_RIGHT_ALTERNATE_MASK = 0x00000040
_NX_DEVICE_RIGHT_CONTROL_MASK = 0x00002000

_MACOS_MODIFIER_DEVICE_MASKS = {
    54: _NX_DEVICE_RIGHT_COMMAND_MASK,
    55: _NX_DEVICE_LEFT_COMMAND_MASK,
    56: _NX_DEVICE_LEFT_SHIFT_MASK,
    58: _NX_DEVICE_LEFT_ALTERNATE_MASK,
    59: _NX_DEVICE_LEFT_CONTROL_MASK,
    60: _NX_DEVICE_RIGHT_SHIFT_MASK,
    61: _NX_DEVICE_RIGHT_ALTERNATE_MASK,
    62: _NX_DEVICE_RIGHT_CONTROL_MASK,
}


@dataclass(frozen=True, slots=True)
class MacTargetContext:
    process_id: int
    bundle_identifier: str


class MacNativeApi(Protocol):
    def permission_errors(self) -> tuple[str, ...]: ...

    def capture_target(self) -> MacTargetContext: ...

    def set_clipboard_text(self, text: str) -> None: ...

    def activate(self, target: MacTargetContext) -> bool: ...

    def wait_for_clipboard_sync(self) -> None: ...

    def is_frontmost(self, target: MacTargetContext) -> bool: ...

    def send_command_v(self) -> bool: ...

    def send_control_v(self) -> bool: ...


class MacFileLockApi(Protocol):
    def acquire(self, lock_file: object) -> bool: ...

    def release(self, lock_file: object) -> None: ...


class _FcntlFileLockApi:
    """Load fcntl only when the macOS lock is actually acquired."""

    def __init__(self) -> None:
        try:
            self._fcntl = importlib.import_module("fcntl")
        except ImportError as exc:
            raise RuntimeError(
                "The macOS single-instance lock requires the fcntl module"
            ) from exc

    def acquire(self, lock_file: object) -> bool:
        try:
            self._fcntl.flock(
                lock_file.fileno(),
                self._fcntl.LOCK_EX | self._fcntl.LOCK_NB,
            )
        except BlockingIOError:
            return False
        return True

    def release(self, lock_file: object) -> None:
        self._fcntl.flock(lock_file.fileno(), self._fcntl.LOCK_UN)


class MacSingleInstanceGuard:
    def __init__(
        self,
        lock_path: Path,
        *,
        lock_api: MacFileLockApi | None = None,
    ) -> None:
        self._lock_path = lock_path
        self._lock_api = lock_api
        self._file: object | None = None

    @property
    def _lock(self) -> MacFileLockApi:
        if self._lock_api is None:
            self._lock_api = _FcntlFileLockApi()
        return self._lock_api

    def acquire(self) -> bool:
        if self._file is not None:
            return True
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self._lock_path.open("a+", encoding="utf-8")
        if not self._lock.acquire(lock_file):
            lock_file.close()
            return False
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        self._file = lock_file
        return True

    def close(self) -> None:
        if self._file is None:
            return
        lock_file = self._file
        self._file = None
        try:
            self._lock.release(lock_file)
        finally:
            lock_file.close()


_MACOS_KEY_NAMES = {
    0: "A",
    1: "S",
    2: "D",
    3: "F",
    4: "H",
    5: "G",
    6: "Z",
    7: "X",
    8: "C",
    9: "V",
    11: "B",
    12: "Q",
    13: "W",
    14: "E",
    15: "R",
    16: "Y",
    17: "T",
    18: "1",
    19: "2",
    20: "3",
    21: "4",
    22: "6",
    23: "5",
    25: "9",
    26: "7",
    28: "8",
    29: "0",
    31: "O",
    32: "U",
    34: "I",
    35: "P",
    36: "Enter",
    37: "L",
    38: "J",
    40: "K",
    45: "N",
    46: "M",
    48: "Tab",
    49: "Space",
    51: "Backspace",
    53: "Esc",
    54: "RightWin",
    55: "LeftWin",
    56: "LeftShift",
    58: "LeftAlt",
    59: "LeftCtrl",
    60: "RightShift",
    61: "RightAlt",
    62: "RightCtrl",
    96: "F5",
    97: "F6",
    98: "F7",
    99: "F3",
    100: "F8",
    101: "F9",
    103: "F11",
    109: "F10",
    111: "F12",
    115: "Home",
    116: "PageUp",
    117: "Delete",
    118: "F4",
    119: "End",
    120: "F2",
    121: "PageDown",
    122: "F1",
    123: "Left",
    124: "Right",
    125: "Down",
    126: "Up",
}


def key_name_from_macos_keycode(keycode: int) -> str | None:
    """Map physical macOS key codes onto the existing cross-platform names."""

    return _MACOS_KEY_NAMES.get(keycode)


class MacHotkeyListener:
    """Quartz event-tap adapter for the existing HotkeyDispatcher contract."""

    def __init__(
        self,
        dispatcher: HotkeyDispatcher,
        *,
        module_loader: Callable[[str], object] = importlib.import_module,
        startup_timeout: float = 3.0,
        shutdown_timeout: float = 3.0,
    ) -> None:
        self._dispatcher = dispatcher
        self._module_loader = module_loader
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._startup_error: Exception | None = None
        self._runtime_error: RuntimeError | None = None
        self._core_foundation: object | None = None
        self._run_loop: object | None = None
        self._source: object | None = None
        self._tap: object | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                raise RuntimeError("The global hotkey listener is already running")
            self._clear_stopped_thread()
        self._ready.clear()
        self._stop_requested.clear()
        self._startup_error = None
        self._runtime_error = None
        thread = threading.Thread(
            target=self._event_loop,
            name="VIM2 macOS hotkey",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        if not self._ready.wait(self._startup_timeout):
            self._stop_requested.set()
            self._request_run_loop_stop()
            thread.join(timeout=self._shutdown_timeout)
            if thread.is_alive():
                raise RuntimeError(
                    "Timed out while installing the macOS global hotkey; "
                    "the listener thread is still running and its reference "
                    "has been retained for safe cleanup."
                )
            self._clear_stopped_thread()
            raise RuntimeError(
                "Timed out while installing the macOS global hotkey; "
                "native resources were cleaned up."
            )
        if self._startup_error is not None:
            startup_error = self._startup_error
            thread.join(timeout=self._shutdown_timeout)
            if thread.is_alive():
                raise RuntimeError(
                    "The macOS global hotkey failed to start and its listener "
                    "thread is still running."
                ) from startup_error
            self._clear_stopped_thread()
            raise startup_error

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_requested.set()
        stop_error = self._request_run_loop_stop()
        thread.join(timeout=self._shutdown_timeout)
        if thread.is_alive():
            raise RuntimeError(
                "Timed out while stopping the macOS global hotkey; the listener "
                "thread is still running and its reference has been retained."
            )
        runtime_error = self._runtime_error
        self._clear_stopped_thread()
        if stop_error is not None:
            raise stop_error
        if runtime_error is not None:
            raise runtime_error

    def wait_until_released(self, timeout: float | None = None) -> None:
        self._dispatcher.wait_until_released(timeout)

    def _request_run_loop_stop(self) -> RuntimeError | None:
        if self._core_foundation is None or self._run_loop is None:
            return None
        try:
            self._core_foundation.CFRunLoopStop(self._run_loop)
        except Exception as exc:
            return RuntimeError(
                f"Cannot stop the macOS hotkey run loop safely: {exc}"
            )
        return None

    def _clear_stopped_thread(self) -> None:
        self._thread = None
        self._core_foundation = None
        self._run_loop = None
        self._source = None
        self._tap = None

    def _event_loop(self) -> None:
        quartz: object | None = None
        core_foundation: object | None = None
        try:
            quartz = self._module_loader("Quartz")
            core_foundation = self._module_loader("CoreFoundation")
            self._core_foundation = core_foundation
            event_types = (
                quartz.kCGEventKeyDown,
                quartz.kCGEventKeyUp,
                quartz.kCGEventFlagsChanged,
            )
            event_mask = sum(1 << event_type for event_type in event_types)

            def callback(proxy, event_type, event, reference):
                del proxy, reference
                if event_type in {
                    quartz.kCGEventTapDisabledByTimeout,
                    quartz.kCGEventTapDisabledByUserInput,
                }:
                    quartz.CGEventTapEnable(self._tap, True)
                    return event
                keycode = int(
                    quartz.CGEventGetIntegerValueField(
                        event, quartz.kCGKeyboardEventKeycode
                    )
                )
                key = key_name_from_macos_keycode(keycode)
                if key is None:
                    return event
                if event_type == quartz.kCGEventFlagsChanged:
                    device_mask = _MACOS_MODIFIER_DEVICE_MASKS.get(keycode)
                    if device_mask is None:
                        return event
                    flags = int(quartz.CGEventGetFlags(event))
                    is_down = bool(flags & device_mask)
                else:
                    is_down = event_type == quartz.kCGEventKeyDown
                marker = quartz.CGEventGetIntegerValueField(
                    event, quartz.kCGEventSourceUserData
                )
                suppress = self._dispatcher.process(
                    KeyEvent(key, is_down),
                    lower_integrity_injected=False,
                    vim2_injected=marker == VIM2_INPUT_MARKER,
                )
                return None if suppress else event

            self._tap = quartz.CGEventTapCreate(
                quartz.kCGSessionEventTap,
                quartz.kCGHeadInsertEventTap,
                quartz.kCGEventTapOptionDefault,
                event_mask,
                callback,
                None,
            )
            if self._tap is None:
                raise RuntimeError(
                    "Cannot create the macOS global event tap. Grant VIM2 "
                    "Accessibility and Input Monitoring access in System "
                    "Settings > Privacy & Security, then restart VIM2."
                )
            self._source = quartz.CFMachPortCreateRunLoopSource(
                None, self._tap, 0
            )
            self._run_loop = core_foundation.CFRunLoopGetCurrent()
            core_foundation.CFRunLoopAddSource(
                self._run_loop,
                self._source,
                core_foundation.kCFRunLoopCommonModes,
            )
            quartz.CGEventTapEnable(self._tap, True)
            self._ready.set()
            if not self._stop_requested.is_set():
                core_foundation.CFRunLoopRun()
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            cleanup_error = self._cleanup_native_resources(
                quartz, core_foundation
            )
            if cleanup_error is not None:
                self._runtime_error = cleanup_error
            self._ready.set()

    def _cleanup_native_resources(
        self,
        quartz: object | None,
        core_foundation: object | None,
    ) -> RuntimeError | None:
        errors: list[str] = []
        if quartz is not None and self._tap is not None:
            try:
                quartz.CGEventTapEnable(self._tap, False)
            except Exception as exc:
                errors.append(f"cannot disable event tap: {exc}")
        if (
            core_foundation is not None
            and self._run_loop is not None
            and self._source is not None
        ):
            try:
                core_foundation.CFRunLoopRemoveSource(
                    self._run_loop,
                    self._source,
                    core_foundation.kCFRunLoopCommonModes,
                )
            except Exception as exc:
                errors.append(f"cannot remove run-loop source: {exc}")
        if core_foundation is not None and self._tap is not None:
            try:
                core_foundation.CFMachPortInvalidate(self._tap)
            except Exception as exc:
                errors.append(f"cannot invalidate event tap: {exc}")
        if errors:
            return RuntimeError(
                "Cannot clean up the macOS global hotkey safely: "
                + "; ".join(errors)
            )
        return None


class PyObjCMacNativeApi:
    """Public, injectable PyObjC seam used by the macOS platform services."""

    def __init__(
        self,
        *,
        module_loader: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self._appkit = self._load_module(module_loader, "AppKit")
        self._quartz = self._load_module(module_loader, "Quartz")
        self._application_services = self._load_module(
            module_loader, "ApplicationServices"
        )
        self._avfoundation = self._load_module(module_loader, "AVFoundation")

    @staticmethod
    def _load_module(
        module_loader: Callable[[str], object], name: str
    ) -> object:
        try:
            return module_loader(name)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load required macOS native module {name}: {exc}"
            ) from exc

    def permission_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            accessibility_check = getattr(
                self._application_services, "AXIsProcessTrusted"
            )
        except AttributeError as exc:
            raise RuntimeError(
                "Required macOS symbol "
                "ApplicationServices.AXIsProcessTrusted is unavailable"
            ) from exc
        try:
            accessibility_trusted = bool(accessibility_check())
        except Exception as exc:
            raise RuntimeError(
                f"Cannot query macOS Accessibility access: {exc}"
            ) from exc
        if not accessibility_trusted:
            errors.append(
                "macOS Accessibility access is required for the global hotkey "
                "and Command+V or Control+V paste injection. Enable VIM2 (or its "
                "Python launcher) in System "
                "Settings > Privacy & Security > Accessibility."
            )

        preflight_listen = getattr(
            self._quartz, "CGPreflightListenEventAccess", None
        )
        if preflight_listen is not None:
            try:
                input_access = bool(preflight_listen())
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot query macOS Input Monitoring access: {exc}"
                ) from exc
            if not input_access:
                errors.append(
                    "macOS Input Monitoring access is required for the global "
                    "hotkey. Enable VIM2 (or its Python launcher) in System "
                    "Settings > Privacy & Security > Input Monitoring."
                )
        try:
            status = self._avfoundation.AVCaptureDevice.authorizationStatusForMediaType_(
                self._avfoundation.AVMediaTypeAudio
            )
            denied_statuses = {
                self._avfoundation.AVAuthorizationStatusDenied,
                self._avfoundation.AVAuthorizationStatusRestricted,
            }
        except Exception as exc:
            raise RuntimeError(
                f"Cannot query macOS Microphone access: {exc}"
            ) from exc
        if status in denied_statuses:
            errors.append(
                "macOS Microphone access is denied. Enable VIM2 (or its Python "
                "launcher) in System Settings > Privacy & Security > Microphone."
            )
        return tuple(errors)

    def capture_target(self) -> MacTargetContext:
        try:
            application = (
                self._appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
            )
            if application is None:
                raise RuntimeError(
                    "Cannot determine the current macOS foreground app"
                )
            process_id = int(application.processIdentifier())
            bundle_identifier = application.bundleIdentifier()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Cannot inspect the current macOS foreground app: {exc}"
            ) from exc
        if process_id <= 0:
            raise RuntimeError(
                "The current macOS foreground app has no process id"
            )
        if not isinstance(bundle_identifier, str) or not bundle_identifier:
            raise RuntimeError(
                "The current macOS foreground app has no bundle identifier"
            )
        return MacTargetContext(process_id, bundle_identifier)

    def set_clipboard_text(self, text: str) -> None:
        try:
            pasteboard = self._appkit.NSPasteboard.generalPasteboard()
            pasteboard.clearContents()
            copied = pasteboard.setString_forType_(
                text, self._appkit.NSPasteboardTypeString
            )
        except Exception as exc:
            raise OSError(
                f"Cannot write recognized text to the macOS pasteboard: {exc}"
            ) from exc
        if not copied:
            raise OSError("Cannot write recognized text to the macOS pasteboard")

    def activate(self, target: MacTargetContext) -> bool:
        try:
            application = self._appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(
                target.process_id
            )
            if application is None:
                return False
            resolved_process_id = int(application.processIdentifier())
            resolved_bundle_identifier = application.bundleIdentifier()
            if (
                resolved_process_id != target.process_id
                or resolved_bundle_identifier != target.bundle_identifier
            ):
                return False
            options = (
                self._appkit.NSApplicationActivateAllWindows
                | self._appkit.NSApplicationActivateIgnoringOtherApps
            )
            return bool(application.activateWithOptions_(options))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot resolve or activate the target macOS application: {exc}"
            ) from exc

    def is_frontmost(self, target: MacTargetContext) -> bool:
        return self.capture_target() == target

    @staticmethod
    def wait_for_clipboard_sync() -> None:
        time.sleep(CLIPBOARD_SYNC_DELAY_SECONDS)

    def send_command_v(self) -> bool:
        try:
            command_flag = self._quartz.kCGEventFlagMaskCommand
        except Exception:
            return False
        return self._send_modified_v(
            modifier_keycode=55,
            modifier_flag=command_flag,
        )

    def send_control_v(self) -> bool:
        try:
            control_flag = self._quartz.kCGEventFlagMaskControl
        except Exception:
            return False
        return self._send_modified_v(
            modifier_keycode=59,
            modifier_flag=control_flag,
        )

    def _send_modified_v(
        self,
        *,
        modifier_keycode: int,
        modifier_flag: int,
    ) -> bool:
        try:
            modifier_down = self._quartz.CGEventCreateKeyboardEvent(
                None, modifier_keycode, True
            )
            v_down = self._quartz.CGEventCreateKeyboardEvent(None, 9, True)
            v_up = self._quartz.CGEventCreateKeyboardEvent(None, 9, False)
            modifier_up = self._quartz.CGEventCreateKeyboardEvent(
                None, modifier_keycode, False
            )
            events = (modifier_down, v_down, v_up, modifier_up)
            if any(event is None for event in events):
                return False
            for event in events:
                self._quartz.CGEventSetIntegerValueField(
                    event,
                    self._quartz.kCGEventSourceUserData,
                    VIM2_INPUT_MARKER,
                )
            for event in (modifier_down, v_down, v_up):
                self._quartz.CGEventSetFlags(event, modifier_flag)
            self._quartz.CGEventSetFlags(modifier_up, 0)
        except Exception:
            return False

        injection_succeeded = False
        try:
            for event in (modifier_down, v_down, v_up):
                self._quartz.CGEventPost(
                    self._quartz.kCGHIDEventTap, event
                )
            injection_succeeded = True
        except Exception:
            injection_succeeded = False
        finally:
            if not injection_succeeded:
                try:
                    self._quartz.CGEventPost(
                        self._quartz.kCGHIDEventTap, v_up
                    )
                except Exception:
                    pass
            try:
                self._quartz.CGEventPost(
                    self._quartz.kCGHIDEventTap, modifier_up
                )
            except Exception:
                injection_succeeded = False
        return injection_succeeded


class MacClipboardPaster:
    def __init__(
        self,
        api: MacNativeApi,
        *,
        wait_until_hotkey_released: Callable[[], None],
        paste_shortcut_selection: MacPasteShortcutSelection | None = None,
    ) -> None:
        self._api = api
        self._wait_until_hotkey_released = wait_until_hotkey_released
        self._paste_shortcut_selection = (
            paste_shortcut_selection or MacPasteShortcutSelection()
        )

    @property
    def paste_shortcut(self) -> MacPasteShortcut:
        return self._paste_shortcut_selection.shortcut

    def paste(self, text: str, target_context: object) -> None:
        if (
            not isinstance(target_context, MacTargetContext)
            or target_context.process_id <= 0
            or not target_context.bundle_identifier
        ):
            raise ClipboardPasteError("Invalid macOS target application context")
        self._wait_until_hotkey_released()
        try:
            self._api.set_clipboard_text(text)
        except Exception as exc:
            raise ClipboardPasteError(
                f"Cannot copy recognized text to the clipboard: {exc}"
            ) from exc
        try:
            activated = self._api.activate(target_context)
        except Exception as exc:
            raise ClipboardPasteError(
                "Cannot validate or restore the target application; recognized "
                f"text remains in the clipboard: {exc}"
            ) from exc
        if not activated:
            raise ClipboardPasteError(
                "Cannot validate or restore the target application; its PID or "
                "bundle identifier changed. Recognized text remains in the "
                "clipboard."
            )
        try:
            self._api.wait_for_clipboard_sync()
            target_is_frontmost = self._api.is_frontmost(target_context)
        except Exception as exc:
            raise ClipboardPasteError(
                "Cannot verify the frontmost macOS application; recognized text "
                f"remains in the clipboard: {exc}"
            ) from exc
        if not target_is_frontmost:
            raise ClipboardPasteError(
                "The target macOS application is no longer frontmost or its "
                "identity changed; recognized text remains in the clipboard."
            )
        shortcut = self._paste_shortcut_selection.shortcut
        shortcut_label = (
            "Command+V"
            if shortcut is MacPasteShortcut.COMMAND_V
            else "Control+V"
        )
        send_shortcut = (
            self._api.send_command_v
            if shortcut is MacPasteShortcut.COMMAND_V
            else self._api.send_control_v
        )
        try:
            injected = send_shortcut()
        except Exception as exc:
            raise ClipboardPasteError(
                f"Cannot send {shortcut_label}; recognized text remains in the "
                f"clipboard: {exc}"
            ) from exc
        if not injected:
            raise ClipboardPasteError(
                f"Cannot send {shortcut_label}; recognized text remains in the "
                "clipboard."
            )


class MacOSPlatformServices:
    def __init__(
        self,
        paths: AppPaths,
        profile: PlatformProfile,
        *,
        native_api: MacNativeApi | None = None,
    ) -> None:
        self._paths = paths
        self.profile = profile
        self._native_api = native_api

    @property
    def supported_models(self):
        return self.profile.supported_models

    @property
    def _native(self) -> MacNativeApi:
        if self._native_api is None:
            self._native_api = PyObjCMacNativeApi()
        return self._native_api

    def preflight_errors(self) -> tuple[str, ...]:
        try:
            return self._native.permission_errors()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Cannot complete the macOS native permission preflight: {exc}"
            ) from exc

    @staticmethod
    def enable_desktop_features() -> None:
        return None

    def create_single_instance_guard(self) -> MacSingleInstanceGuard:
        return MacSingleInstanceGuard(self._paths.runtime_dir / "vim2.lock")

    def capture_target(self) -> MacTargetContext:
        return self._native.capture_target()

    @staticmethod
    def create_hotkey_listener(dispatcher: object) -> MacHotkeyListener:
        if not isinstance(dispatcher, HotkeyDispatcher):
            raise TypeError("dispatcher must be a HotkeyDispatcher")
        return MacHotkeyListener(dispatcher)

    def create_clipboard_paster(
        self,
        hotkey,
        *,
        paste_shortcut_selection: MacPasteShortcutSelection | None = None,
    ) -> MacClipboardPaster:
        return MacClipboardPaster(
            self._native,
            wait_until_hotkey_released=hotkey.wait_until_released,
            paste_shortcut_selection=paste_shortcut_selection,
        )


def show_startup_error(message: str) -> None:
    try:
        appkit = importlib.import_module("AppKit")
        alert = appkit.NSAlert.alloc().init()
        alert.setMessageText_("VIM2 cannot start")
        alert.setInformativeText_(message)
        alert.setAlertStyle_(appkit.NSAlertStyleCritical)
        alert.runModal()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot display the macOS startup error dialog: {exc}"
        ) from exc
