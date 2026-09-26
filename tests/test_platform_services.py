from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from vim2.clipboard import ClipboardPasteError
from vim2.config import MacPasteShortcut, MacPasteShortcutSelection
from vim2.hotkey import HotkeyDispatcher, KeyEvent, parse_hotkey
from vim2.models import ModelId
from vim2.paths import AppPaths
from vim2.platform_services import (
    create_platform_services,
    select_platform_profile,
)
from vim2.platform_macos import (
    _MACOS_MODIFIER_DEVICE_MASKS,
    MacClipboardPaster,
    MacHotkeyListener,
    MacSingleInstanceGuard,
    MacTargetContext,
    PyObjCMacNativeApi,
    key_name_from_macos_keycode,
)


class PlatformSelectionTests(unittest.TestCase):
    def test_windows_profile_preserves_every_existing_model(self) -> None:
        profile = select_platform_profile("win32", "AMD64")

        self.assertEqual(profile.name, "windows")
        self.assertEqual(profile.supported_models, tuple(ModelId))
        self.assertEqual(profile.preflight_errors, ())

    def test_apple_silicon_profile_exposes_only_sherpa_cpu(self) -> None:
        profile = select_platform_profile("darwin", "arm64")

        self.assertEqual(profile.name, "macos")
        self.assertEqual(profile.supported_models, (ModelId.CPU,))
        self.assertEqual(profile.preflight_errors, ())
        self.assertEqual(
            profile.required_modules,
            (
                "AppKit",
                "Quartz",
                "ApplicationServices",
                "AVFoundation",
                "CoreFoundation",
            ),
        )
        self.assertEqual(profile.required_python, (3, 11))

    def test_rosetta_is_an_explicit_preflight_error(self) -> None:
        profile = select_platform_profile("darwin", "x86_64")

        self.assertEqual(profile.supported_models, (ModelId.CPU,))
        self.assertTrue(any("Rosetta" in error for error in profile.preflight_errors))

    def test_factory_selects_public_macos_service_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.from_root(Path(directory))
            profile = select_platform_profile("darwin", "arm64")

            services = create_platform_services(paths, profile=profile)

            self.assertEqual(services.profile, profile)
            self.assertEqual(services.supported_models, (ModelId.CPU,))

    def test_adapter_factories_do_not_import_the_opposite_platform(self) -> None:
        script = r'''
import sys
from pathlib import Path
from vim2.paths import AppPaths
from vim2.platform_services import create_platform_services, select_platform_profile

selected = sys.argv[1]
opposite = "vim2.platform_windows" if selected == "darwin" else "vim2.platform_macos"
profile = select_platform_profile(selected, "arm64" if selected == "darwin" else "AMD64")
services = create_platform_services(AppPaths.from_root(Path.cwd()), profile=profile)
assert services.profile == profile
assert opposite not in sys.modules, sorted(name for name in sys.modules if name.startswith("vim2.platform_"))
'''
        for selected in ("darwin", "win32"):
            completed = subprocess.run(
                [sys.executable, "-c", script, selected],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )

    def test_macos_adapter_import_does_not_require_fcntl(self) -> None:
        script = r'''
import importlib.abc
import sys

class RejectFcntl(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "fcntl":
            raise ImportError("fcntl is unavailable on this host")
        return None

sys.modules.pop("fcntl", None)
sys.meta_path.insert(0, RejectFcntl())
import vim2.platform_macos
assert "fcntl" not in sys.modules
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )


class _FakeLockApi:
    def __init__(self) -> None:
        self.locked = False
        self.released = False

    def acquire(self, lock_file: object) -> bool:
        del lock_file
        if self.locked:
            return False
        self.locked = True
        return True

    def release(self, lock_file: object) -> None:
        del lock_file
        self.locked = False
        self.released = True


class MacPlatformAdapterTests(unittest.TestCase):
    @unittest.skipIf(
        sys.platform.startswith("win"),
        "the real flock implementation is only available on POSIX hosts",
    )
    def test_single_instance_guard_uses_a_releasable_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "runtime" / "vim2.lock"
            first = MacSingleInstanceGuard(lock_path)
            second = MacSingleInstanceGuard(lock_path)

            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            first.close()
            self.assertTrue(second.acquire())
            second.close()

    def test_single_instance_guard_has_a_host_independent_lock_seam(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_api = _FakeLockApi()
            guard = MacSingleInstanceGuard(
                Path(directory) / "runtime" / "vim2.lock",
                lock_api=lock_api,
            )

            self.assertTrue(guard.acquire())
            guard.close()

            self.assertTrue(lock_api.released)

    def test_right_option_maps_to_existing_right_alt_semantics(self) -> None:
        self.assertEqual(key_name_from_macos_keycode(61), "RightAlt")
        self.assertEqual(key_name_from_macos_keycode(58), "LeftAlt")
        self.assertEqual(key_name_from_macos_keycode(53), "Esc")


class _FakePasteNative:
    def __init__(self, target: MacTargetContext) -> None:
        self.calls: list[tuple[str, object]] = []
        self.resolved_bundle_identifier = target.bundle_identifier
        self.frontmost = target
        self.activate_result = True
        self.activate_sets_frontmost = True
        self.change_target_while_waiting: MacTargetContext | None = None

    def set_clipboard_text(self, text: str) -> None:
        self.calls.append(("clipboard", text))

    def activate(self, target: MacTargetContext) -> bool:
        self.calls.append(("activate", target))
        if self.resolved_bundle_identifier != target.bundle_identifier:
            return False
        if self.activate_sets_frontmost:
            self.frontmost = target
        return self.activate_result

    def wait_for_clipboard_sync(self) -> None:
        self.calls.append(("sync", None))
        if self.change_target_while_waiting is not None:
            self.frontmost = self.change_target_while_waiting

    def is_frontmost(self, target: MacTargetContext) -> bool:
        self.calls.append(("frontmost", target))
        return self.frontmost == target

    def send_command_v(self) -> bool:
        self.calls.append(("command-v", None))
        return True

    def send_control_v(self) -> bool:
        self.calls.append(("control-v", None))
        return True


class MacClipboardSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = MacTargetContext(42, "com.example.Editor")
        self.other = MacTargetContext(77, "com.example.Other")

    def _paster(
        self,
        native: _FakePasteNative,
        selection: MacPasteShortcutSelection | None = None,
    ) -> MacClipboardPaster:
        return MacClipboardPaster(
            native,
            wait_until_hotkey_released=lambda: native.calls.append(
                ("released", None)
            ),
            paste_shortcut_selection=selection,
        )

    def test_paste_rechecks_pid_and_bundle_before_command_v(self) -> None:
        native = _FakePasteNative(self.target)

        self._paster(native).paste("Unicode 文本", self.target)

        self.assertEqual(
            native.calls,
            [
                ("released", None),
                ("clipboard", "Unicode 文本"),
                ("activate", self.target),
                ("sync", None),
                ("frontmost", self.target),
                ("command-v", None),
            ],
        )

    def test_each_paste_reads_the_shared_shortcut_selection(self) -> None:
        native = _FakePasteNative(self.target)
        selection = MacPasteShortcutSelection(MacPasteShortcut.COMMAND_V)
        paster = self._paster(native, selection)

        paster.paste("local text", self.target)
        selection.shortcut = MacPasteShortcut.CONTROL_V
        paster.paste("remote text", self.target)

        self.assertEqual(paster.paste_shortcut, MacPasteShortcut.CONTROL_V)
        self.assertEqual(
            [call for call in native.calls if call[0].endswith("-v")],
            [("command-v", None), ("control-v", None)],
        )
        self.assertFalse(hasattr(paster, "set_paste_shortcut"))

    def test_pid_reuse_fails_without_injecting_keys(self) -> None:
        native = _FakePasteNative(self.target)
        native.resolved_bundle_identifier = "com.example.ReusedProcess"

        with self.assertRaisesRegex(ClipboardPasteError, "remains in the clipboard"):
            self._paster(native).paste("recognized", self.target)

        self.assertIn(("clipboard", "recognized"), native.calls)
        self.assertNotIn(("command-v", None), native.calls)

    def test_activate_true_but_frontmost_identity_mismatch_never_injects(self) -> None:
        native = _FakePasteNative(self.target)
        native.activate_sets_frontmost = False
        native.frontmost = self.other

        with self.assertRaisesRegex(ClipboardPasteError, "frontmost"):
            self._paster(native).paste("recognized", self.target)

        self.assertIn(("clipboard", "recognized"), native.calls)
        self.assertNotIn(("command-v", None), native.calls)

    def test_target_change_during_clipboard_wait_never_injects(self) -> None:
        native = _FakePasteNative(self.target)
        native.change_target_while_waiting = self.other

        with self.assertRaisesRegex(ClipboardPasteError, "frontmost"):
            self._paster(native).paste("recognized", self.target)

        self.assertEqual(native.frontmost, self.other)
        self.assertNotIn(("command-v", None), native.calls)


class _FakeApplication:
    def __init__(self, process_id: int, bundle_identifier: str) -> None:
        self.process_id = process_id
        self.bundle_identifier = bundle_identifier
        self.activation_calls: list[int] = []

    def processIdentifier(self) -> int:
        return self.process_id

    def bundleIdentifier(self) -> str:
        return self.bundle_identifier

    def activateWithOptions_(self, options: int) -> bool:
        self.activation_calls.append(options)
        return True


class _FakeWorkspace:
    def __init__(self, frontmost: _FakeApplication | None) -> None:
        self.frontmost = frontmost

    def frontmostApplication(self) -> _FakeApplication | None:
        return self.frontmost


class _PostingQuartz:
    kCGEventSourceUserData = 88
    kCGHIDEventTap = 99
    kCGEventFlagMaskCommand = 1 << 20
    kCGEventFlagMaskControl = 1 << 18

    def __init__(self, fail_on: tuple[int, bool] | None = None) -> None:
        self.fail_on = fail_on
        self.posted: list[SimpleNamespace] = []

    @staticmethod
    def CGEventCreateKeyboardEvent(
        source: object, keycode: int, is_down: bool
    ) -> SimpleNamespace:
        del source
        return SimpleNamespace(
            keycode=keycode,
            is_down=is_down,
            flags=0,
            fields={},
        )

    @staticmethod
    def CGEventGetFlags(event: SimpleNamespace) -> int:
        return int(event.flags)

    @staticmethod
    def CGEventSetIntegerValueField(
        event: SimpleNamespace, field: int, marker: int
    ) -> None:
        event.fields[field] = marker

    @staticmethod
    def CGEventSetFlags(event: SimpleNamespace, flags: int) -> None:
        event.flags = int(flags)

    def CGEventPost(self, event_tap: int, event: SimpleNamespace) -> None:
        del event_tap
        self.posted.append(event)
        if (event.keycode, event.is_down) == self.fail_on:
            raise RuntimeError("partial injection failure")


def _native_modules(
    *,
    running: dict[int, _FakeApplication] | None = None,
    frontmost: _FakeApplication | None = None,
    quartz: object | None = None,
    application_services: object | None = None,
) -> dict[str, object]:
    workspace = _FakeWorkspace(frontmost)
    running = running or {}
    appkit = SimpleNamespace(
        NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: workspace),
        NSRunningApplication=SimpleNamespace(
            runningApplicationWithProcessIdentifier_=running.get
        ),
        NSApplicationActivateAllWindows=1,
        NSApplicationActivateIgnoringOtherApps=2,
    )
    avfoundation = SimpleNamespace(
        AVMediaTypeAudio="audio",
        AVAuthorizationStatusDenied=1,
        AVAuthorizationStatusRestricted=2,
        AVCaptureDevice=SimpleNamespace(
            authorizationStatusForMediaType_=lambda media_type: 0
        ),
    )
    return {
        "AppKit": appkit,
        "Quartz": quartz or SimpleNamespace(),
        "ApplicationServices": application_services
        or SimpleNamespace(AXIsProcessTrusted=lambda: True),
        "AVFoundation": avfoundation,
    }


def _native_api(modules: dict[str, object]) -> PyObjCMacNativeApi:
    return PyObjCMacNativeApi(module_loader=lambda name: modules[name])


class MacNativeApiTests(unittest.TestCase):
    def test_native_activation_rejects_reused_pid_with_another_bundle_id(self) -> None:
        reused = _FakeApplication(42, "com.example.Reused")
        api = _native_api(_native_modules(running={42: reused}))

        activated = api.activate(MacTargetContext(42, "com.example.Original"))

        self.assertFalse(activated)
        self.assertEqual(reused.activation_calls, [])

    def test_capture_requires_and_retains_pid_and_bundle_identifier(self) -> None:
        frontmost = _FakeApplication(42, "com.example.Editor")
        api = _native_api(_native_modules(frontmost=frontmost))

        self.assertEqual(
            api.capture_target(),
            MacTargetContext(42, "com.example.Editor"),
        )

    def test_frontmost_check_compares_both_pid_and_bundle_identifier(self) -> None:
        reused = _FakeApplication(42, "com.example.Reused")
        api = _native_api(_native_modules(frontmost=reused))

        self.assertFalse(
            api.is_frontmost(MacTargetContext(42, "com.example.Editor"))
        )
        self.assertFalse(
            api.is_frontmost(MacTargetContext(77, "com.example.Reused"))
        )

    def test_accessibility_trust_comes_from_application_services(self) -> None:
        api = _native_api(
            _native_modules(
                quartz=SimpleNamespace(AXIsProcessTrusted=lambda: True),
                application_services=SimpleNamespace(
                    AXIsProcessTrusted=lambda: False
                ),
            )
        )

        errors = api.permission_errors()

        self.assertTrue(any("Accessibility" in error for error in errors))

    def test_accessibility_module_import_failure_is_runtime_error(self) -> None:
        modules = _native_modules()
        del modules["ApplicationServices"]

        def load(name: str) -> object:
            try:
                return modules[name]
            except KeyError as exc:
                raise ImportError(name) from exc

        with self.assertRaisesRegex(RuntimeError, "ApplicationServices"):
            PyObjCMacNativeApi(module_loader=load)

    def test_missing_ax_symbol_is_runtime_error_not_attribute_error(self) -> None:
        api = _native_api(
            _native_modules(application_services=SimpleNamespace())
        )

        with self.assertRaisesRegex(
            RuntimeError, "ApplicationServices.AXIsProcessTrusted"
        ):
            api.permission_errors()

    def test_ax_query_exception_is_runtime_error(self) -> None:
        def fail() -> bool:
            raise ValueError("bridge failed")

        api = _native_api(
            _native_modules(
                application_services=SimpleNamespace(AXIsProcessTrusted=fail)
            )
        )

        with self.assertRaisesRegex(RuntimeError, "Accessibility"):
            api.permission_errors()

    def test_command_v_events_carry_chord_flags_marker_and_order(self) -> None:
        quartz = _PostingQuartz()
        api = _native_api(_native_modules(quartz=quartz))

        self.assertTrue(api.send_command_v())
        self.assertEqual(
            [(event.keycode, event.is_down) for event in quartz.posted],
            [(55, True), (9, True), (9, False), (55, False)],
        )
        command_down, v_down, v_up, command_up = quartz.posted
        self.assertEqual(v_down.flags, quartz.kCGEventFlagMaskCommand)
        self.assertEqual(v_up.flags, quartz.kCGEventFlagMaskCommand)
        self.assertEqual(
            command_down.flags,
            quartz.kCGEventFlagMaskCommand,
        )
        self.assertEqual(command_up.flags, 0)
        for event in quartz.posted:
            self.assertEqual(
                event.fields[quartz.kCGEventSourceUserData],
                0x56494D32,
            )

    def test_control_v_events_carry_chord_flags_marker_and_order(self) -> None:
        quartz = _PostingQuartz()
        api = _native_api(_native_modules(quartz=quartz))

        self.assertTrue(api.send_control_v())
        self.assertEqual(
            [(event.keycode, event.is_down) for event in quartz.posted],
            [(59, True), (9, True), (9, False), (59, False)],
        )
        control_down, v_down, v_up, control_up = quartz.posted
        self.assertEqual(control_down.flags, quartz.kCGEventFlagMaskControl)
        self.assertEqual(v_down.flags, quartz.kCGEventFlagMaskControl)
        self.assertEqual(v_up.flags, quartz.kCGEventFlagMaskControl)
        self.assertEqual(control_up.flags, 0)
        for event in quartz.posted:
            self.assertEqual(
                event.fields[quartz.kCGEventSourceUserData],
                0x56494D32,
            )

    def test_partial_command_v_failure_always_posts_key_up_cleanup(self) -> None:
        quartz = _PostingQuartz(fail_on=(9, True))
        api = _native_api(_native_modules(quartz=quartz))

        self.assertFalse(api.send_command_v())
        self.assertEqual(
            [(event.keycode, event.is_down) for event in quartz.posted],
            [(55, True), (9, True), (9, False), (55, False)],
        )
        self.assertEqual(
            quartz.posted[2].flags,
            quartz.kCGEventFlagMaskCommand,
        )
        self.assertEqual(quartz.posted[-1].flags, 0)
        for event in quartz.posted:
            self.assertEqual(
                event.fields[quartz.kCGEventSourceUserData],
                0x56494D32,
            )

    def test_partial_control_v_failure_always_posts_key_up_cleanup(self) -> None:
        quartz = _PostingQuartz(fail_on=(9, True))
        api = _native_api(_native_modules(quartz=quartz))

        self.assertFalse(api.send_control_v())
        self.assertEqual(
            [(event.keycode, event.is_down) for event in quartz.posted],
            [(59, True), (9, True), (9, False), (59, False)],
        )
        self.assertEqual(
            quartz.posted[2].flags,
            quartz.kCGEventFlagMaskControl,
        )
        self.assertEqual(quartz.posted[-1].flags, 0)
        for event in quartz.posted:
            self.assertEqual(
                event.fields[quartz.kCGEventSourceUserData],
                0x56494D32,
            )


class _EventTapQuartz:
    kCGEventKeyDown = 1
    kCGEventKeyUp = 2
    kCGEventFlagsChanged = 3
    kCGEventTapDisabledByTimeout = 4
    kCGEventTapDisabledByUserInput = 5
    kCGSessionEventTap = 6
    kCGHeadInsertEventTap = 7
    kCGEventTapOptionDefault = 8
    kCGKeyboardEventKeycode = 9
    kCGEventSourceStateCombinedSessionState = 10
    kCGEventSourceUserData = 11

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.tap = object()
        self.source = object()
        self.callback = None
        self.source_key_state_queries: list[int] = []
        self._source_key_state = False

    def CGEventTapCreate(self, *args) -> object:
        self.callback = args[4]
        self.calls.append("tap-created")
        return self.tap

    def CFMachPortCreateRunLoopSource(self, *args) -> object:
        del args
        self.calls.append("source-created")
        return self.source

    def CGEventTapEnable(self, tap: object, enabled: bool) -> None:
        self.calls.append("tap-enabled" if enabled else "tap-disabled")

    def CGEventGetIntegerValueField(self, event: object, field: int) -> int:
        if field == self.kCGKeyboardEventKeycode:
            return event.keycode
        if field == self.kCGEventSourceUserData:
            return event.marker
        raise AssertionError(f"unexpected event field: {field}")

    @staticmethod
    def CGEventGetFlags(event: object) -> int:
        return event.flags

    def CGEventSourceKeyState(self, state: int, keycode: int) -> bool:
        self.source_key_state_queries.append(keycode)
        return self._source_key_state

    def dispatch(
        self,
        event_type: int,
        keycode: int,
        *,
        flags: int = 0,
        source_key_state: bool = False,
    ) -> tuple[object, object]:
        if self.callback is None:
            raise AssertionError("event tap callback was not installed")
        self._source_key_state = source_key_state
        event = SimpleNamespace(keycode=keycode, flags=flags, marker=0)
        result = self.callback(None, event_type, event, None)
        return event, result

    def dispatch_flags_changed(
        self,
        keycode: int,
        flags: int,
        *,
        source_key_state: bool,
    ) -> object:
        return self.dispatch(
            self.kCGEventFlagsChanged,
            keycode,
            flags=flags,
            source_key_state=source_key_state,
        )[1]


class _EventTapCoreFoundation:
    kCFRunLoopCommonModes = object()

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.stopped = threading.Event()
        self.run_loop = object()

    def CFRunLoopGetCurrent(self) -> object:
        return self.run_loop

    def CFRunLoopAddSource(self, *args) -> None:
        del args
        self.calls.append("source-added")

    def CFRunLoopRun(self) -> None:
        self.calls.append("run")
        self.stopped.wait(2)

    def CFRunLoopStop(self, run_loop: object) -> None:
        del run_loop
        self.calls.append("stop")
        self.stopped.set()

    def CFRunLoopRemoveSource(self, *args) -> None:
        del args
        self.calls.append("source-removed")

    def CFMachPortInvalidate(self, tap: object) -> None:
        del tap
        self.calls.append("tap-invalidated")


class _RecordingDispatcher:
    def __init__(self, dispatcher: HotkeyDispatcher) -> None:
        self._dispatcher = dispatcher
        self.events: list[KeyEvent] = []

    def process(self, event: KeyEvent, **kwargs: bool) -> bool:
        self.events.append(event)
        return self._dispatcher.process(event, **kwargs)

    def wait_until_released(self, timeout: float | None = None) -> None:
        self._dispatcher.wait_until_released(timeout)


class MacHotkeyModifierStateTests(unittest.TestCase):
    @staticmethod
    def _listener(
        hotkey: str,
    ) -> tuple[
        MacHotkeyListener,
        _EventTapQuartz,
        _RecordingDispatcher,
        list[str],
    ]:
        calls: list[str] = []
        toggles: list[str] = []
        quartz = _EventTapQuartz(calls)
        core_foundation = _EventTapCoreFoundation(calls)
        dispatcher = _RecordingDispatcher(
            HotkeyDispatcher(
                parse_hotkey(hotkey),
                on_toggle=lambda: toggles.append("toggle"),
                on_cancel=lambda: None,
                is_cancellable=lambda: False,
            )
        )
        listener = MacHotkeyListener(
            dispatcher,
            module_loader=lambda name: {
                "Quartz": quartz,
                "CoreFoundation": core_foundation,
            }[name],
        )
        return listener, quartz, dispatcher, toggles

    def test_real_option_flags_drive_callback_dispatcher_and_matcher(self) -> None:
        listener, quartz, dispatcher, toggles = self._listener("RightAlt")

        listener.start()
        try:
            # Values captured by the M2 HITL probe. The source-state query was
            # inverted for Left Option and always UP for Right Option.
            quartz.dispatch_flags_changed(
                58, 0x80120, source_key_state=False
            )
            quartz.dispatch_flags_changed(58, 0x100, source_key_state=True)
            quartz.dispatch_flags_changed(
                61, 0x80140, source_key_state=False
            )
            quartz.dispatch_flags_changed(61, 0x100, source_key_state=False)
        finally:
            listener.stop()

        self.assertEqual(
            dispatcher.events,
            [
                KeyEvent("LeftAlt", True),
                KeyEvent("LeftAlt", False),
                KeyEvent("RightAlt", True),
                KeyEvent("RightAlt", False),
            ],
        )
        self.assertEqual(toggles, ["toggle"])
        self.assertEqual(quartz.source_key_state_queries, [])

    def test_device_modifier_masks_match_iollevent_values(self) -> None:
        self.assertEqual(
            _MACOS_MODIFIER_DEVICE_MASKS,
            {
                54: 0x10,
                55: 0x08,
                56: 0x02,
                58: 0x20,
                59: 0x01,
                60: 0x04,
                61: 0x40,
                62: 0x2000,
            },
        )

    def test_each_modifier_side_survives_other_side_press_and_release(self) -> None:
        cases = (
            ("Shift", 56, 60, 0x20000, 0x02, 0x04),
            ("Ctrl", 59, 62, 0x40000, 0x01, 0x2000),
            ("Alt", 58, 61, 0x80000, 0x20, 0x40),
            ("Win", 55, 54, 0x100000, 0x08, 0x10),
        )
        for hotkey, left_keycode, right_keycode, generic, left, right in cases:
            with self.subTest(hotkey=hotkey):
                listener, quartz, dispatcher, toggles = self._listener(hotkey)
                listener.start()
                try:
                    quartz.dispatch_flags_changed(
                        left_keycode,
                        generic | left,
                        source_key_state=False,
                    )
                    quartz.dispatch_flags_changed(
                        right_keycode,
                        generic | left | right,
                        source_key_state=False,
                    )
                    quartz.dispatch_flags_changed(
                        left_keycode,
                        generic | right,
                        source_key_state=True,
                    )
                    quartz.dispatch_flags_changed(
                        right_keycode,
                        0,
                        source_key_state=True,
                    )
                finally:
                    listener.stop()

                left_name = key_name_from_macos_keycode(left_keycode)
                right_name = key_name_from_macos_keycode(right_keycode)
                self.assertEqual(
                    dispatcher.events,
                    [
                        KeyEvent(left_name, True),
                        KeyEvent(right_name, True),
                        KeyEvent(left_name, False),
                        KeyEvent(right_name, False),
                    ],
                )
                self.assertEqual(toggles, ["toggle"])
                self.assertEqual(quartz.source_key_state_queries, [])

    def test_normal_key_events_and_ignored_caps_lock_are_unchanged(self) -> None:
        listener, quartz, dispatcher, toggles = self._listener("K")
        listener.start()
        try:
            quartz.dispatch(quartz.kCGEventKeyDown, 40)
            quartz.dispatch(quartz.kCGEventKeyUp, 40)
            caps_event, caps_result = quartz.dispatch(
                quartz.kCGEventFlagsChanged,
                57,
                flags=0x10100,
            )
        finally:
            listener.stop()

        self.assertEqual(
            dispatcher.events,
            [KeyEvent("K", True), KeyEvent("K", False)],
        )
        self.assertEqual(toggles, ["toggle"])
        self.assertIs(caps_result, caps_event)
        self.assertEqual(quartz.source_key_state_queries, [])


class MacHotkeyCleanupTests(unittest.TestCase):
    @staticmethod
    def _dispatcher() -> object:
        return SimpleNamespace(
            wait_until_released=lambda timeout=None: None,
            process=lambda *args, **kwargs: False,
        )

    def test_stop_removes_source_invalidates_tap_and_joins_thread(self) -> None:
        calls: list[str] = []
        quartz = _EventTapQuartz(calls)
        core_foundation = _EventTapCoreFoundation(calls)
        modules = {"Quartz": quartz, "CoreFoundation": core_foundation}
        listener = MacHotkeyListener(
            self._dispatcher(),
            module_loader=lambda name: modules[name],
        )

        listener.start()
        listener.stop()

        self.assertFalse(listener.is_running)
        self.assertIn("source-removed", calls)
        self.assertIn("tap-invalidated", calls)
        self.assertIn("tap-disabled", calls)

    def test_startup_timeout_retains_live_thread_reference_for_safe_retry(self) -> None:
        calls: list[str] = []
        gate = threading.Event()
        quartz = _EventTapQuartz(calls)
        core_foundation = _EventTapCoreFoundation(calls)
        modules = {"Quartz": quartz, "CoreFoundation": core_foundation}

        def blocked_loader(name: str) -> object:
            gate.wait(2)
            return modules[name]

        listener = MacHotkeyListener(
            self._dispatcher(),
            module_loader=blocked_loader,
            startup_timeout=0.01,
            shutdown_timeout=0.05,
        )

        with self.assertRaisesRegex(RuntimeError, "still running"):
            listener.start()
        self.assertTrue(listener.is_running)
        with self.assertRaisesRegex(RuntimeError, "already running"):
            listener.start()

        gate.set()
        listener.stop()
        self.assertFalse(listener.is_running)


# Darwin NX event types and left-modifier masks. The consumer below uses these
# as an external client's fixed rules; it does not replay VIM2's flag copying.
_NX_KEY_DOWN = 10
_NX_KEY_UP = 11
_NX_FLAGS_CHANGED = 12
_LEFT_COMMAND_KEYCODE = 55
_LEFT_CONTROL_KEYCODE = 59
_V_KEYCODE = 9
_GENERIC_COMMAND = 0x00100000
_GENERIC_CONTROL = 0x00040000
_LEFT_COMMAND_DEVICE = 0x00000008
_LEFT_CONTROL_DEVICE = 0x00000001
_CREATE_BASE_FLAGS = 0x20000000


def _consumed_event(
    event_type: int, keycode: int, flags: int
) -> SimpleNamespace:
    return SimpleNamespace(
        event_type=event_type,
        keycode=keycode,
        flags=flags,
    )


class _WindowsAppPasteConsumer:
    """Microsoft Windows App chord model, distinct from local TextEdit.

    TextEdit pastes when the V keyDown itself carries a generic modifier
    flag. Windows App does not: it arms paste only after a left-modifier
    flagsChanged event that contains both that side's device bit and the
    generic mask. Any other V keyDown is the character "v", even if V's own
    flags look like a Cocoa paste chord.
    """

    _COMPLETE_LEFT_MODIFIER = {
        _LEFT_COMMAND_KEYCODE: _GENERIC_COMMAND | _LEFT_COMMAND_DEVICE,
        _LEFT_CONTROL_KEYCODE: _GENERIC_CONTROL | _LEFT_CONTROL_DEVICE,
    }

    def interpret(self, events: object) -> str:
        armed = False
        outputs: list[str] = []
        for event in events:
            event_type = int(event.event_type)
            keycode = int(event.keycode)
            flags = int(event.flags)
            required = self._COMPLETE_LEFT_MODIFIER.get(keycode)
            if event_type == _NX_FLAGS_CHANGED and required is not None:
                armed = flags & required == required
            elif event_type == _NX_KEY_DOWN and keycode == _V_KEYCODE:
                outputs.append("PASTE" if armed else "v")
        return "".join(outputs)


class _HardwareBaselineQuartz:
    """CGEventCreateKeyboardEvent baselines confirmed by the no-post dry-run.

    Modifier creation is flagsChanged and already includes the side bit.
    This double does not implement VIM2's desired output flags.
    """

    kCGEventSourceUserData = 42
    kCGHIDEventTap = 0
    kCGEventFlagMaskCommand = _GENERIC_COMMAND
    kCGEventFlagMaskControl = _GENERIC_CONTROL

    def __init__(self) -> None:
        self.posted: list[SimpleNamespace] = []

    @staticmethod
    def CGEventCreateKeyboardEvent(
        source: object, keycode: int, is_down: bool
    ) -> SimpleNamespace:
        del source
        baselines = {
            (_LEFT_COMMAND_KEYCODE, True): (
                _NX_FLAGS_CHANGED,
                _CREATE_BASE_FLAGS | _GENERIC_COMMAND | _LEFT_COMMAND_DEVICE,
            ),
            (_LEFT_COMMAND_KEYCODE, False): (
                _NX_FLAGS_CHANGED,
                _CREATE_BASE_FLAGS,
            ),
            (_LEFT_CONTROL_KEYCODE, True): (
                _NX_FLAGS_CHANGED,
                _CREATE_BASE_FLAGS | _GENERIC_CONTROL | _LEFT_CONTROL_DEVICE,
            ),
            (_LEFT_CONTROL_KEYCODE, False): (
                _NX_FLAGS_CHANGED,
                _CREATE_BASE_FLAGS,
            ),
            (_V_KEYCODE, True): (_NX_KEY_DOWN, _CREATE_BASE_FLAGS),
            (_V_KEYCODE, False): (_NX_KEY_UP, _CREATE_BASE_FLAGS),
        }
        event_type, flags = baselines[(int(keycode), bool(is_down))]
        return SimpleNamespace(
            event_type=event_type,
            keycode=int(keycode),
            is_down=bool(is_down),
            flags=flags,
            fields={},
        )

    @staticmethod
    def CGEventGetFlags(event: SimpleNamespace) -> int:
        return int(event.flags)

    @staticmethod
    def CGEventSetFlags(event: SimpleNamespace, flags: int) -> None:
        event.flags = int(flags)

    @staticmethod
    def CGEventSetIntegerValueField(
        event: SimpleNamespace, field: int, value: int
    ) -> None:
        event.fields[int(field)] = value

    def CGEventPost(self, event_tap: int, event: SimpleNamespace) -> None:
        del event_tap
        self.posted.append(
            _consumed_event(
                int(event.event_type),
                int(event.keycode),
                int(event.flags),
            )
        )


class WindowsAppPasteConsumerTests(unittest.TestCase):
    def test_complete_left_flags_changed_pastes_stripped_chord_is_bare_v(
        self,
    ) -> None:
        consumer = _WindowsAppPasteConsumer()
        # V flags are intentionally empty: paste must come from flagsChanged.
        complete_command = (
            _consumed_event(_NX_FLAGS_CHANGED, _LEFT_COMMAND_KEYCODE, 0x20100008),
            _consumed_event(_NX_KEY_DOWN, _V_KEYCODE, 0),
            _consumed_event(_NX_KEY_UP, _V_KEYCODE, 0),
            _consumed_event(
                _NX_FLAGS_CHANGED,
                _LEFT_COMMAND_KEYCODE,
                _CREATE_BASE_FLAGS,
            ),
        )
        complete_control = (
            _consumed_event(_NX_FLAGS_CHANGED, _LEFT_CONTROL_KEYCODE, 0x20040001),
            _consumed_event(_NX_KEY_DOWN, _V_KEYCODE, 0),
            _consumed_event(_NX_KEY_UP, _V_KEYCODE, 0),
            _consumed_event(
                _NX_FLAGS_CHANGED,
                _LEFT_CONTROL_KEYCODE,
                _CREATE_BASE_FLAGS,
            ),
        )
        stripped_command = (
            _consumed_event(
                _NX_FLAGS_CHANGED, _LEFT_COMMAND_KEYCODE, _GENERIC_COMMAND
            ),
            _consumed_event(_NX_KEY_DOWN, _V_KEYCODE, _GENERIC_COMMAND),
            _consumed_event(_NX_KEY_UP, _V_KEYCODE, _GENERIC_COMMAND),
            _consumed_event(_NX_FLAGS_CHANGED, _LEFT_COMMAND_KEYCODE, 0),
        )
        textedit_style_v = (
            _consumed_event(_NX_KEY_DOWN, _V_KEYCODE, 0x20100008),
        )
        key_down_modifier = (
            _consumed_event(_NX_KEY_DOWN, _LEFT_COMMAND_KEYCODE, 0x20100008),
            _consumed_event(_NX_KEY_DOWN, _V_KEYCODE, 0x20100008),
        )
        device_bit_without_generic = (
            _consumed_event(
                _NX_FLAGS_CHANGED,
                _LEFT_COMMAND_KEYCODE,
                _LEFT_COMMAND_DEVICE,
            ),
            _consumed_event(_NX_KEY_DOWN, _V_KEYCODE, 0),
        )

        with self.subTest("golden left command flagsChanged"):
            self.assertEqual(consumer.interpret(complete_command), "PASTE")
        with self.subTest("golden left control flagsChanged"):
            self.assertEqual(consumer.interpret(complete_control), "PASTE")
        with self.subTest("generic-only flagsChanged is bare v"):
            self.assertEqual(consumer.interpret(stripped_command), "v")
        with self.subTest("V flags alone are not Windows App paste"):
            self.assertEqual(consumer.interpret(textedit_style_v), "v")
        with self.subTest("modifier keyDown is not flagsChanged"):
            self.assertEqual(consumer.interpret(key_down_modifier), "v")
        with self.subTest("device bit without generic mask is bare v"):
            self.assertEqual(consumer.interpret(device_bit_without_generic), "v")

        with self.subTest("send_command_v"):
            self.assertEqual(self._interpret_sent("send_command_v"), "PASTE")
        with self.subTest("send_control_v"):
            self.assertEqual(self._interpret_sent("send_control_v"), "PASTE")

    def _interpret_sent(self, method_name: str) -> str:
        quartz = _HardwareBaselineQuartz()
        api = _native_api(_native_modules(quartz=quartz))
        self.assertTrue(getattr(api, method_name)())
        return _WindowsAppPasteConsumer().interpret(quartz.posted)


class _NonPostingQuartz:
    """Forwards Quartz construction and readback, but never CGEventPost."""

    def __init__(self, quartz: object) -> None:
        self._quartz = quartz
        self.created: list[dict[str, int]] = []
        self.posts: list[tuple[int, dict[str, int]]] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self._quartz, name)

    def CGEventCreateKeyboardEvent(
        self, source: object, keycode: int, is_down: bool
    ) -> object:
        event = self._quartz.CGEventCreateKeyboardEvent(source, keycode, is_down)
        self.created.append(self._snapshot(event))
        return event

    def CGEventPost(self, event_tap: int, event: object) -> None:
        self.posts.append((int(event_tap), self._snapshot(event)))

    def _snapshot(self, event: object) -> dict[str, int]:
        quartz = self._quartz
        return {
            "type": int(quartz.CGEventGetType(event)),
            "keycode": int(
                quartz.CGEventGetIntegerValueField(
                    event, quartz.kCGKeyboardEventKeycode
                )
            ),
            "flags": int(quartz.CGEventGetFlags(event)),
            "marker": int(
                quartz.CGEventGetIntegerValueField(
                    event, quartz.kCGEventSourceUserData
                )
            ),
        }


class MacQuartzPasteDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            quartz = importlib.import_module("Quartz")
        except ImportError as exc:
            self.skipTest(f"PyObjC Quartz is unavailable: {exc}")
        required = (
            "CGEventCreateKeyboardEvent",
            "CGEventGetFlags",
            "CGEventGetIntegerValueField",
            "CGEventGetType",
            "CGEventPost",
            "CGEventSetFlags",
            "CGEventSetIntegerValueField",
            "CGEventSourceKeyState",
            "kCGEventFlagMaskCommand",
            "kCGEventFlagMaskControl",
            "kCGEventFlagsChanged",
            "kCGEventKeyDown",
            "kCGEventKeyUp",
            "kCGEventSourceStateHIDSystemState",
            "kCGEventSourceUserData",
            "kCGHIDEventTap",
            "kCGKeyboardEventKeycode",
        )
        missing = [name for name in required if not hasattr(quartz, name)]
        if missing:
            self.skipTest(
                "PyObjC Quartz is missing required symbols: "
                + ", ".join(missing)
            )
        self.quartz = quartz
        self._original_post = quartz.CGEventPost

        def forbid_real_post(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "native paste dry-run must not call Quartz.CGEventPost"
            )

        quartz.CGEventPost = forbid_real_post

    def tearDown(self) -> None:
        quartz = getattr(self, "quartz", None)
        original = getattr(self, "_original_post", None)
        if quartz is not None and original is not None:
            quartz.CGEventPost = original

    def test_command_v_preserves_device_flags_without_posting(self) -> None:
        self._assert_dry_run_chord(
            PyObjCMacNativeApi.send_command_v,
            _LEFT_COMMAND_KEYCODE,
            _LEFT_COMMAND_DEVICE,
            "kCGEventFlagMaskCommand",
        )

    def test_control_v_preserves_device_flags_without_posting(self) -> None:
        self._assert_dry_run_chord(
            PyObjCMacNativeApi.send_control_v,
            _LEFT_CONTROL_KEYCODE,
            _LEFT_CONTROL_DEVICE,
            "kCGEventFlagMaskControl",
        )

    def _assert_dry_run_chord(
        self,
        send: object,
        keycode: int,
        device_mask: int,
        generic_name: str,
    ) -> None:
        if bool(
            self.quartz.CGEventSourceKeyState(
                self.quartz.kCGEventSourceStateHIDSystemState,
                keycode,
            )
        ):
            self.skipTest(
                f"keycode {keycode} is physically down; dry-run needs it released"
            )
        proxy = _NonPostingQuartz(self.quartz)
        api = _native_api(_native_modules(quartz=proxy))
        self.assertTrue(send(api))
        self.assertEqual(len(proxy.created), 4)
        self.assertEqual(len(proxy.posts), 4)
        generic = int(getattr(self.quartz, generic_name))
        expected_types = (
            int(self.quartz.kCGEventFlagsChanged),
            int(self.quartz.kCGEventKeyDown),
            int(self.quartz.kCGEventKeyUp),
            int(self.quartz.kCGEventFlagsChanged),
        )
        expected_keycodes = (keycode, _V_KEYCODE, _V_KEYCODE, keycode)
        self.assertEqual(
            [event["type"] for event in proxy.created],
            list(expected_types),
        )
        self.assertEqual(
            [event["keycode"] for event in proxy.created],
            list(expected_keycodes),
        )
        down_initial = proxy.created[0]["flags"]
        release_baseline = proxy.created[3]["flags"]
        self.assertTrue(down_initial & device_mask)
        self.assertTrue(down_initial & generic)
        expected_down = down_initial | generic
        posted = []
        for index, (tap, event) in enumerate(proxy.posts):
            self.assertEqual(tap, int(self.quartz.kCGHIDEventTap))
            self.assertEqual(event["type"], expected_types[index])
            self.assertEqual(event["keycode"], expected_keycodes[index])
            self.assertEqual(event["marker"], 0x56494D32)
            posted.append(event)
        self.assertEqual(posted[0]["flags"], expected_down)
        self.assertEqual(posted[1]["flags"], expected_down)
        self.assertEqual(posted[2]["flags"], expected_down)
        self.assertTrue(posted[0]["flags"] & device_mask)
        self.assertTrue(posted[0]["flags"] & generic)
        self.assertEqual(posted[3]["flags"], release_baseline)
        self.assertFalse(posted[3]["flags"] & device_mask)
        self.assertFalse(posted[3]["flags"] & generic)


if __name__ == "__main__":
    unittest.main()
