"""First-launch contracts for the frozen macOS installer. No real TCC prompts."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import vim2.application as application_module
import vim2.main as main_module
from vim2.paths import AppPaths

ROOT = Path(__file__).resolve().parents[1]
VOLUME_EXECUTABLE = "/Volumes/VIM2 0.1.0/VIM2.app/Contents/MacOS/VIM2"
TRANSLOCATED_EXECUTABLE = (
    "/private/var/folders/xy/random/AppTranslocation/"
    "7139A238-9AB4-489B-BC29-04848A2E5608/d/VIM2.app/Contents/MacOS/VIM2"
)
TRANSLOCATED_WITH_T = TRANSLOCATED_EXECUTABLE.replace(
    "/random/AppTranslocation/", "/random/T/AppTranslocation/"
)
INSTALLED_EXECUTABLE = "/Applications/VIM2.app/Contents/MacOS/VIM2"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _entry():
    return _load_module("vim2_frozen_entry", ROOT / "packaging" / "entry.py")


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    config = bundle / "config"
    config.mkdir(parents=True)
    (config / "settings.json").write_text(
        '{"selected_model":"qwen3-asr-0.6b-int8-cpu"}\n',
        encoding="utf-8",
    )
    (config / "hotkey.conf").write_text("RightAlt\n", encoding="utf-8")
    (config / "hotwords.template.txt").write_text("# default\n", encoding="utf-8")
    return bundle


def _freeze(monkeypatch: pytest.MonkeyPatch, bundle: Path, executable: str) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "executable", executable)
    monkeypatch.setattr(sys, "platform", "darwin")


def _passing_checker() -> type:
    class Checker:
        def __init__(self, paths, *, platform_profile) -> None:
            del paths, platform_profile

        def check(self, selected_model, **kwargs):
            del selected_model, kwargs
            return SimpleNamespace(ok=True, errors=())

    return Checker


class _PermissionApi:
    def __init__(self, errors: tuple[str, ...] = ()) -> None:
        self.errors = errors
        self.prompts: list[str] = []
        self.microphone_requests: list[str] = []

    def permission_errors(self) -> tuple[str, ...]:
        return self.errors

    def prompt_accessibility(self) -> None:
        self.prompts.append("accessibility")

    def prompt_input_monitoring(self) -> bool:
        self.prompts.append("input")
        return True

    def request_microphone_access(self) -> None:
        self.microphone_requests.append("microphone")


def _services(tmp_path: Path, api: _PermissionApi):
    from vim2.platform_macos import MacOSPlatformServices
    from vim2.platform_services import select_platform_profile

    bundle = tmp_path / "service-bundle"
    bundle.mkdir()
    paths = AppPaths.for_launch(data_root=tmp_path / "service-data")
    return MacOSPlatformServices(
        paths,
        select_platform_profile("darwin", "arm64"),
        native_api=api,
    )


def test_finder_launch_injects_windowed_without_duplicating_or_hiding_cli() -> None:
    entry = _entry()

    assert entry.frozen_gui_argv([]) == ["--windowed"]
    assert entry.frozen_gui_argv(["--windowed"]) == ["--windowed"]
    assert entry.frozen_gui_argv(["--windowed", "--data-root", "/tmp/vim2"]) == [
        "--windowed",
        "--data-root",
        "/tmp/vim2",
    ]
    assert entry.frozen_gui_argv(["--check", "--data-root", "/tmp/vim2"]) == [
        "--check",
        "--data-root",
        "/tmp/vim2",
    ]
    assert entry.frozen_gui_argv(
        ["--import-smoke", "--data-root", "/tmp/vim2"]
    ) == ["--import-smoke", "--data-root", "/tmp/vim2"]
    assert "--windowed" not in entry.frozen_gui_argv(["--check"])
    assert "--windowed" not in entry.frozen_gui_argv(["--import-smoke"])


def test_entry_main_uses_public_argv_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv=None):
        seen["argv"] = list(argv) if argv is not None else None
        return 7

    monkeypatch.setattr(main_module, "run", fake_run)
    entry = _entry()
    entry.run = fake_run

    assert entry.main([]) == 7
    assert seen["argv"] == ["--windowed"]
    assert entry.main(["--check", "--skip-runtime-check"]) == 7
    assert seen["argv"] == ["--check", "--skip-runtime-check"]
    source = (ROOT / "app" / "vim2" / "__main__.py").read_text(encoding="utf-8")
    assert "frozen_gui_argv" not in source
    assert "run()" in source


def test_disk_image_executable_is_rejected_before_platform_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vim2.main import launched_from_disk_image

    assert launched_from_disk_image(VOLUME_EXECUTABLE)
    assert not launched_from_disk_image(INSTALLED_EXECUTABLE)
    assert not launched_from_disk_image("/Volumes/Data/usr/bin/python")
    assert not launched_from_disk_image("/Users/mandyw/.venvs/vim2-mac/bin/python")

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _freeze(monkeypatch, _bundle(tmp_path), VOLUME_EXECUTABLE)
    shown: list[str] = []
    monkeypatch.setattr(main_module, "_show_startup_message", shown.append)

    def forbidden(*args, **kwargs):
        raise AssertionError("disk-image launch must not load or request TCC")

    monkeypatch.setattr(main_module, "create_platform_services", forbidden)
    monkeypatch.setattr(main_module, "PreflightChecker", forbidden)
    monkeypatch.setattr(
        "vim2.platform_macos.PyObjCMacNativeApi",
        lambda *args, **kwargs: forbidden(),
    )

    assert main_module.run([]) == 1
    assert shown
    message = shown[0]
    assert "VIM2.app" in message
    assert "Applications" in message
    assert "弹出" in message
    assert not (home / "Library" / "Application Support" / "VIM2").exists()


@pytest.mark.parametrize(
    "executable", [TRANSLOCATED_EXECUTABLE, TRANSLOCATED_WITH_T, TRANSLOCATED_EXECUTABLE.upper()]
)
def test_translocated_gui_exits_before_user_data_preflight_or_native_permissions(
    executable: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _freeze(monkeypatch, _bundle(tmp_path), executable)
    shown: list[str] = []
    monkeypatch.setattr(main_module, "_show_startup_message", shown.append)

    def forbidden(*args, **kwargs):
        raise AssertionError("translocated app reached forbidden user-data/preflight/native seam")

    monkeypatch.setattr(main_module.AppPaths, "for_launch", forbidden)
    monkeypatch.setattr(main_module, "PreflightChecker", forbidden)
    monkeypatch.setattr(main_module, "create_platform_services", forbidden)
    monkeypatch.setattr("vim2.platform_macos.PyObjCMacNativeApi", forbidden)
    assert main_module.run([]) == 1
    assert len(shown) == 1
    assert "Applications" in shown[0] and "弹出" in shown[0]
    assert not (home / "Library" / "Application Support" / "VIM2").exists()


def test_translocated_gui_never_enters_preflight_or_native_permission_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _freeze(monkeypatch, _bundle(tmp_path), TRANSLOCATED_EXECUTABLE)
    shown: list[str] = []
    monkeypatch.setattr(main_module, "_show_startup_message", shown.append)

    def forbidden(*args, **kwargs):
        raise AssertionError("translocated GUI reached forbidden preflight/native seam")

    monkeypatch.setattr(main_module, "PreflightChecker", forbidden)
    monkeypatch.setattr(main_module, "create_platform_services", forbidden)
    monkeypatch.setattr("vim2.platform_macos.PyObjCMacNativeApi", forbidden)
    monkeypatch.setattr(_PermissionApi, "permission_errors", forbidden)
    assert main_module.run([]) == 1
    assert len(shown) == 1 and "Applications" in shown[0]
    assert not (home / "Library" / "Application Support" / "VIM2").exists()


def test_image_path_matcher_rejects_installed_similarly_named_and_source_paths() -> None:
    match = main_module.launched_from_disk_image
    assert match(VOLUME_EXECUTABLE)
    assert match(TRANSLOCATED_EXECUTABLE)
    assert match(TRANSLOCATED_EXECUTABLE.upper())
    assert match(TRANSLOCATED_WITH_T)
    assert match(TRANSLOCATED_EXECUTABLE.replace("/d/VIM2.app/", "/d/nested/VIM2.app/"))
    assert not match(INSTALLED_EXECUTABLE)
    assert not match("/Users/test/AppTranslocation/VIM2.app/Contents/MacOS/VIM2")
    assert not match("/Users/test/AppTranslocation/token/d/VIM2.app/Contents/MacOS/VIM2")
    assert not match("/private/var/folders/xy/random/AppTranslocation/token/not-d/VIM2.app/Contents/MacOS/VIM2")
    assert not match("/private/var/folders/xy/random/AppTranslocation/d/VIM2.app/Contents/MacOS/VIM2")
    assert not match("/private/var/folders/xy/random/other/AppTranslocation/token/d/VIM2.app/Contents/MacOS/VIM2")
    assert not match("/private/var/folders/xy/random/NotAppTranslocation/token/d/VIM2.app/Contents/MacOS/VIM2")
    assert not match("/Volumes/Installer/usr/bin/python")
    # The .app name alone is not evidence that an executable lives in a bundle.
    assert not match("/Volumes/Installer.app/usr/bin/python")
    assert not match("/Volumes/Installer.app/VIM2.app/Contents/MacOS/VIM2")
    assert not match("/Volumes/Installer/VIM2.app/usr/bin/python")
    assert not match("/Volumes/Installer/nested/VIM2.app/Contents/MacOS/VIM2")
    assert not match("/Volumes/Installer/VIM2.app/Contents/MacOS/VIM2/helper")
    assert not match(
        "/private/var/folders/xy/random/AppTranslocation/token/d/"
        "VIM2.app/usr/bin/python"
    )
    assert not match(
        "/private/var/folders/xy/random/AppTranslocation/token/d/"
        "VIM2.app/Contents/MacOS/VIM2/helper"
    )
    assert not match("/Users/test/.venvs/vim2-mac/bin/python")


@pytest.mark.parametrize("executable", [VOLUME_EXECUTABLE, TRANSLOCATED_EXECUTABLE, TRANSLOCATED_WITH_T])
def test_disk_image_check_and_import_smoke_remain_available(
    executable: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _bundle(tmp_path)
    data = tmp_path / "isolated"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _freeze(monkeypatch, bundle, executable)
    monkeypatch.setattr(main_module, "PreflightChecker", _passing_checker())
    api = _PermissionApi(("macOS Accessibility access is required",))
    monkeypatch.setattr(
        main_module,
        "create_platform_services",
        lambda *args, **kwargs: _services(tmp_path, api),
    )
    monkeypatch.setattr(
        main_module,
        "select_platform_profile",
        lambda: SimpleNamespace(name="macos", preflight_errors=(), supported_models=()),
    )
    shown: list[str] = []
    monkeypatch.setattr(main_module, "_show_startup_message", shown.append)

    assert (
        main_module.run(
            ["--check", "--skip-runtime-check", "--data-root", str(data)]
        )
        == 0
    )
    assert "VIM2 preflight check passed." in capsys.readouterr().out
    assert shown == []
    assert api.prompts == []
    assert (data / "config" / "settings.json").is_file()
    assert not (home / "Library" / "Application Support" / "VIM2").exists()

    imported: list[str] = []
    monkeypatch.setattr(
        main_module.importlib, "import_module", lambda name: imported.append(name)
    )
    assert main_module.run(["--import-smoke", "--data-root", str(data)]) == 0
    assert imported == [
        "sherpa_onnx",
        "sounddevice",
        "soundfile",
        "PySide6.QtWidgets",
    ]
    assert api.prompts == []


def test_installed_gui_requests_missing_permissions_then_shows_bilingual_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    _freeze(monkeypatch, bundle, INSTALLED_EXECUTABLE)
    monkeypatch.setattr(main_module, "PreflightChecker", _passing_checker())
    api = _PermissionApi(
        (
            "macOS Accessibility access is required for the global hotkey",
            "macOS Input Monitoring access is required for the global hotkey",
        )
    )
    monkeypatch.setattr(
        main_module,
        "create_platform_services",
        lambda *args, **kwargs: _services(tmp_path, api),
    )
    from vim2.platform_services import select_platform_profile

    monkeypatch.setattr(
        main_module,
        "select_platform_profile",
        lambda: select_platform_profile("darwin", "arm64"),
    )
    events: list[str] = []
    api.prompt_accessibility = lambda: events.append("prompt-accessibility")
    api.prompt_input_monitoring = lambda: events.append("prompt-input") or True

    def show(message: str) -> None:
        events.append("alert")
        events.append(message)

    monkeypatch.setattr(main_module, "_show_startup_message", show)
    monkeypatch.setattr(
        application_module,
        "run_desktop_application",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("missing permissions must not start the GUI")
        ),
    )
    monkeypatch.setattr(
        "vim2.platform_macos.PyObjCMacNativeApi",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("real PyObjC API must not be constructed")
        ),
    )

    assert main_module.run([]) == 1
    assert events[0] == "prompt-accessibility"
    assert events[1] == "prompt-input"
    assert events[2] == "alert"
    message = events[3]
    assert "辅助功能" in message and "Accessibility" in message
    assert "输入监控" in message and "Input Monitoring" in message
    assert "麦克风" in message and "Microphone" in message
    assert "系统设置" in message
    assert "重新启动" in message
    assert "不会" in message and "授予" in message
    assert api.microphone_requests == []


def test_installed_gui_does_not_prompt_when_permissions_are_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze(monkeypatch, _bundle(tmp_path), INSTALLED_EXECUTABLE)
    monkeypatch.setattr(main_module, "PreflightChecker", _passing_checker())
    api = _PermissionApi(())
    monkeypatch.setattr(
        main_module,
        "create_platform_services",
        lambda *args, **kwargs: _services(tmp_path, api),
    )
    from vim2.platform_services import select_platform_profile

    monkeypatch.setattr(
        main_module,
        "select_platform_profile",
        lambda: select_platform_profile("darwin", "arm64"),
    )
    started: list[bool] = []
    monkeypatch.setattr(
        application_module,
        "run_desktop_application",
        lambda *args, **kwargs: started.append(True) or 0,
    )
    shown: list[str] = []
    monkeypatch.setattr(main_module, "_show_startup_message", shown.append)

    assert main_module.run([]) == 0
    assert api.prompts == []
    assert api.microphone_requests == []
    assert shown == []
    assert started == [True]


def test_check_does_not_request_permissions_even_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _freeze(monkeypatch, _bundle(tmp_path), INSTALLED_EXECUTABLE)
    monkeypatch.setattr(main_module, "PreflightChecker", _passing_checker())
    api = _PermissionApi(("macOS Accessibility access is required",))
    monkeypatch.setattr(
        main_module,
        "create_platform_services",
        lambda *args, **kwargs: _services(tmp_path, api),
    )
    from vim2.platform_services import select_platform_profile

    monkeypatch.setattr(
        main_module,
        "select_platform_profile",
        lambda: select_platform_profile("darwin", "arm64"),
    )
    shown: list[str] = []
    monkeypatch.setattr(main_module, "_show_startup_message", shown.append)

    assert main_module.run(["--check", "--data-root", str(tmp_path / "data")]) == 1
    captured = capsys.readouterr()
    assert api.prompts == []
    assert api.microphone_requests == []
    assert shown == []
    assert "Accessibility" in captured.err
    assert "VIM2 preflight check passed." not in captured.out


def test_source_and_windows_do_not_request_macos_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    (root / "config").mkdir(parents=True)
    (root / "config" / "settings.json").write_text(
        '{"selected_model":"qwen3-asr-0.6b-int8-cpu"}\n',
        encoding="utf-8",
    )
    monkeypatch.delattr(sys, "frozen", raising=False)
    # A source interpreter can itself have an app-shaped launch path; only
    # frozen GUI launches may be blocked by the installer guard.
    monkeypatch.setattr(sys, "executable", TRANSLOCATED_EXECUTABLE)
    monkeypatch.setattr(main_module, "PreflightChecker", _passing_checker())
    api = _PermissionApi(("macOS Accessibility access is required",))
    monkeypatch.setattr(
        main_module,
        "create_platform_services",
        lambda *args, **kwargs: _services(tmp_path, api),
    )
    from vim2.platform_services import select_platform_profile

    monkeypatch.setattr(
        main_module,
        "select_platform_profile",
        lambda: select_platform_profile("darwin", "arm64"),
    )
    monkeypatch.setattr(
        application_module, "run_desktop_application", lambda *args, **kwargs: 0
    )

    assert main_module.run(["--root", str(root), "--skip-runtime-check"]) == 0
    assert api.prompts == []

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(_bundle(tmp_path)), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", r"C:\Users\vim2\AppData\Local\Programs\VIM2\VIM2.exe")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(
        main_module,
        "select_platform_profile",
        lambda: select_platform_profile("win32", "AMD64"),
    )
    windows_calls: list[str] = []

    def windows_services(*args, **kwargs):
        windows_calls.append("windows")
        return SimpleNamespace(preflight_errors=lambda: ())

    monkeypatch.setattr(main_module, "create_platform_services", windows_services)
    assert main_module.run(["--skip-runtime-check", "--data-root", str(tmp_path / "win")]) == 0
    assert windows_calls == ["windows"]
    assert api.prompts == []


def test_permission_queries_do_not_prompt_and_prompts_use_official_apis() -> None:
    from vim2.platform_macos import PyObjCMacNativeApi

    calls: list[object] = []

    def trusted() -> bool:
        calls.append("query-accessibility")
        return False

    def prompt(options: dict[str, object]) -> bool:
        calls.append(("prompt-accessibility", options))
        return False

    def preflight() -> bool:
        calls.append("query-input")
        return False

    def request() -> bool:
        calls.append("prompt-input")
        return False

    def request_microphone(*args, **kwargs):
        raise AssertionError("microphone must remain on the system flow")

    modules = {
        "AppKit": SimpleNamespace(),
        "Quartz": SimpleNamespace(
            CGPreflightListenEventAccess=preflight,
            CGRequestListenEventAccess=request,
        ),
        "ApplicationServices": SimpleNamespace(
            AXIsProcessTrusted=trusted,
            AXIsProcessTrustedWithOptions=prompt,
            kAXTrustedCheckOptionPrompt="AXTrustedCheckOptionPrompt",
        ),
        "AVFoundation": SimpleNamespace(
            AVMediaTypeAudio="audio",
            AVAuthorizationStatusDenied=1,
            AVAuthorizationStatusRestricted=2,
            AVCaptureDevice=SimpleNamespace(
                authorizationStatusForMediaType_=lambda media_type: 0,
                requestAccessForMediaType_=request_microphone,
            ),
        ),
    }
    api = PyObjCMacNativeApi(module_loader=lambda name: modules[name])

    errors = api.permission_errors()
    assert calls == ["query-accessibility", "query-input"]
    assert any("Accessibility" in error for error in errors)
    assert any("Input Monitoring" in error for error in errors)

    api.prompt_accessibility()
    assert api.prompt_input_monitoring() is True
    assert calls[-2] == (
        "prompt-accessibility",
        {"AXTrustedCheckOptionPrompt": True},
    )
    assert calls[-1] == "prompt-input"

    modules["Quartz"] = SimpleNamespace(CGPreflightListenEventAccess=preflight)
    silent = PyObjCMacNativeApi(module_loader=lambda name: modules[name])
    assert silent.prompt_input_monitoring() is False


def test_startup_alert_activates_application_before_modal(monkeypatch) -> None:
    from vim2 import platform_macos

    events: list[object] = []

    class Window:
        def setLevel_(self, level: int) -> None:
            events.append(("level", level))

        def orderFrontRegardless(self) -> None:
            events.append("orderFront")

    class Alert:
        def setMessageText_(self, text: str) -> None:
            events.append(("message", text))

        def setInformativeText_(self, text: str) -> None:
            events.append(("info", text))

        def setAlertStyle_(self, style: int) -> None:
            events.append(("style", style))

        def window(self) -> Window:
            events.append("window")
            return Window()

        def runModal(self) -> int:
            events.append("runModal")
            return 1

    class Alloc:
        def init(self) -> Alert:
            events.append("alert-init")
            return Alert()

    class Application:
        def setActivationPolicy_(self, policy: int) -> bool:
            events.append(("policy", policy))
            return True

        def activateIgnoringOtherApps_(self, flag: bool) -> None:
            events.append(("activate", flag))

    def shared_application() -> Application:
        events.append("sharedApplication")
        return Application()

    appkit = SimpleNamespace(
        NSApplication=SimpleNamespace(sharedApplication=shared_application),
        NSAlert=SimpleNamespace(alloc=lambda: Alloc()),
        NSAlertStyleCritical=2,
        NSApplicationActivationPolicyRegular=0,
        NSStatusWindowLevel=25,
    )
    monkeypatch.setattr(platform_macos.importlib, "import_module", lambda name: appkit)

    platform_macos.show_startup_error("第一行\n请把 VIM2 放到应用程序。")

    assert events.index("sharedApplication") < events.index(("policy", 0))
    assert events.index(("policy", 0)) < events.index(("activate", True))
    assert events.index(("activate", True)) < events.index("runModal")
    assert events.index(("level", 25)) < events.index("runModal")
    assert ("message", "第一行") in events
    assert ("info", "请把 VIM2 放到应用程序。") in events


def test_install_notes_are_staged_into_dmg_root_contract(tmp_path: Path) -> None:
    notes_path = ROOT / "packaging" / "安装说明.txt"
    notes = notes_path.read_text(encoding="utf-8")
    raw = notes_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    for phrase in (
        "VIM2.app",
        "Applications",
        "1.1GB",
        "弹出",
        "右键",
        "未签名",
        "辅助功能",
        "输入监控",
        "麦克风",
        "重新启动",
        "系统设置",
        "Gatekeeper",
        "xattr",
        "最后手段",
        "不会自动执行",
    ):
        assert phrase in notes

    stager = _load_module(
        "vim2_stage_dmg_notes", ROOT / "packaging" / "stage_dmg_notes.py"
    )
    destination = stager.stage_install_instructions(tmp_path)
    assert destination == tmp_path / "安装说明.txt"
    assert destination.read_bytes() == raw

    script = (ROOT / "tools" / "build-macos-installer.sh").read_text(encoding="utf-8")
    assert "packaging/stage_dmg_notes.py" in script
    assert script.index("stage_dmg_notes.py") < script.index("hdiutil create")
    assert "安装说明.txt" in script
    assert "xattr" not in script
    assert "tccutil" not in script
    attach = script.index("hdiutil attach")
    detach = script.index("hdiutil detach", attach)
    assert attach < script.index("--import-smoke", attach) < detach
    assert attach < script.index("安装说明.txt", attach) < detach
