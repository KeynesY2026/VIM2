from inspect import signature
from pathlib import Path
import threading

import numpy as np
import pytest

from vim2.audio import AudioArtifact
from vim2.config import (
    MacPasteShortcut,
    MacPasteShortcutSelection,
    Settings,
    SettingsRepository,
)
from vim2.controller import AppController
from vim2.models import ModelId
from vim2.platform_macos import MacClipboardPaster
from vim2.recognizer import TranscriptionCancelled
from vim2.session import VoiceSession
from vim2.state import AppState, StateMachine


class ImmediateRunner:
    def submit(self, work, on_success, on_error) -> None:
        try:
            result = work()
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
            on_error(exc)
        else:
            on_success(result)


class DeferredRunner:
    def __init__(self) -> None:
        self.tasks = []

    def submit(self, work, on_success, on_error) -> None:
        self.tasks.append((work, on_success, on_error))

    def complete_next(self) -> None:
        work, on_success, on_error = self.tasks.pop(0)
        try:
            result = work()
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
            on_error(exc)
        else:
            on_success(result)


class FakeLifecycleRecognizer:
    def __init__(self, responses: list[str | Exception] | None = None) -> None:
        self.responses = responses or []
        self.loaded: list[ModelId] = []
        self.switched: list[ModelId] = []
        self.unloaded = False
        self.transcribe_calls = 0

    def load(self, model_id: ModelId) -> None:
        self.loaded.append(model_id)

    def switch(self, model_id: ModelId) -> None:
        self.switched.append(model_id)

    def unload(self) -> None:
        self.unloaded = True

    def transcribe(
        self,
        artifact: AudioArtifact,
        model_id: ModelId,
        *,
        cancel_event: threading.Event | None = None,
    ) -> str:
        del artifact, model_id
        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelled()
        self.transcribe_calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeRecorder:
    def __init__(self, tmp_path: Path) -> None:
        del tmp_path
        self.started = False
        self.discarded: list[AudioArtifact] = []
        self.seal_calls = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> AudioArtifact:
        self.started = False
        return AudioArtifact(np.zeros(16_000, dtype=np.float32), 16_000)

    def seal(self) -> None:
        self.seal_calls += 1
        self.started = False

    def snapshot(self) -> AudioArtifact:
        return AudioArtifact(np.zeros(8_000, dtype=np.float32), 16_000)

    def cancel(self) -> None:
        self.started = False

    def discard(self, artifact: AudioArtifact) -> None:
        self.discarded.append(artifact)


class FakePaster:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def paste(self, text: str, target_window: int) -> None:
        self.calls.append((text, target_window))


class FakeSettingsRepository:
    def __init__(self) -> None:
        self.saved: list[Settings] = []
        self.shortcut_updates: list[MacPasteShortcut] = []

    def save(self, settings: Settings) -> None:
        self.saved.append(settings)

    def update_macos_paste_shortcut(
        self, shortcut: MacPasteShortcut
    ) -> None:
        self.shortcut_updates.append(shortcut)


class FakeView:
    def __init__(self) -> None:
        self.states: list[AppState] = []
        self.rendered_models: list[ModelId] = []
        self.previews: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.retry_errors: list[str] = []
        self.timers_started: list[int] = []
        self.preview_intervals: list[int] = []
        self.timers_stopped = 0
        self.macos_paste_shortcut = MacPasteShortcut.COMMAND_V
        self.command_v_checked = True
        self.control_v_checked = False
        self.paste_shortcuts: list[MacPasteShortcut] = []

    def render_state(self, state: AppState, model_id: ModelId) -> None:
        self.states.append(state)
        self.rendered_models.append(model_id)

    def start_recording_timers(
        self,
        max_seconds: int,
        target_window: int,
        preview_interval_ms: int,
    ) -> None:
        self.timers_started.append(max_seconds)
        self.preview_intervals.append(preview_interval_ms)

    def stop_recording_timers(self) -> None:
        self.timers_stopped += 1

    def show_preview(self, text: str) -> None:
        self.previews.append(text)

    def show_error(self, message: str) -> None:
        self.errors.append(message)

    def show_warning(self, message: str) -> None:
        self.warnings.append(message)

    def show_retry_error(self, message: str) -> None:
        self.retry_errors.append(message)

    def hide_overlay(self) -> None:
        pass

    def render_macos_paste_shortcut(
        self, shortcut: MacPasteShortcut
    ) -> None:
        self.macos_paste_shortcut = shortcut
        self.command_v_checked = shortcut is MacPasteShortcut.COMMAND_V
        self.control_v_checked = shortcut is MacPasteShortcut.CONTROL_V
        self.paste_shortcuts.append(shortcut)


def _controller(
    tmp_path: Path,
    responses: list[str | Exception],
    *,
    macos_paste_shortcut_selection: MacPasteShortcutSelection | None = None,
):
    machine = StateMachine()
    recognizer = FakeLifecycleRecognizer(responses)
    recorder = FakeRecorder(tmp_path)
    paster = FakePaster()
    session = VoiceSession(machine, recorder, recognizer, paster)
    view = FakeView()
    repository = FakeSettingsRepository()
    controller = AppController(
        machine=machine,
        settings=Settings(),
        settings_repository=repository,
        recognizer=recognizer,
        session=session,
        view=view,
        task_runner=ImmediateRunner(),
        foreground_window=lambda: 321,
        macos_paste_shortcut_selection=macos_paste_shortcut_selection,
    )
    return controller, recognizer, recorder, paster, view, repository


def test_cross_backend_model_switch_restarts_without_loading_both_runtimes(
    tmp_path: Path,
) -> None:
    machine = StateMachine()
    recognizer = FakeLifecycleRecognizer()
    recorder = FakeRecorder(tmp_path)
    paster = FakePaster()
    view = FakeView()
    repository = FakeSettingsRepository()
    restart_calls: list[bool] = []
    controller = AppController(
        machine=machine,
        settings=Settings(selected_model=ModelId.CPU),
        settings_repository=repository,
        recognizer=recognizer,
        session=VoiceSession(machine, recorder, recognizer, paster),
        view=view,
        task_runner=ImmediateRunner(),
        foreground_window=lambda: 321,
        restart_application=lambda: restart_calls.append(True),
    )
    controller.start()

    controller.switch_model(ModelId.FAST)

    assert recognizer.switched == []
    assert repository.saved[-1].selected_model is ModelId.FAST
    assert restart_calls == [True]


def test_startup_loads_selected_model_before_enabling_recording(
    tmp_path: Path,
) -> None:
    controller, recognizer, _, _, view, _ = _controller(tmp_path, [])

    controller.start()

    assert recognizer.loaded == [ModelId.FAST]
    assert controller.state is AppState.READY
    assert view.states[:2] == [AppState.MODEL_LOADING, AppState.READY]


def test_record_preview_and_final_result_flow(tmp_path: Path) -> None:
    controller, _, recorder, paster, view, _ = _controller(
        tmp_path, ["临时文本", "最终文本"]
    )
    controller.start()

    controller.toggle_recording()
    controller.request_preview()
    controller.toggle_recording()

    assert view.timers_started == [90]
    assert view.preview_intervals == [1_000]
    assert view.previews == ["临时文本"]
    assert paster.calls == [("最终文本", 321)]
    assert len(recorder.discarded) == 2
    assert controller.state is AppState.READY


def test_recognition_failure_offers_retry_and_preserves_model(
    tmp_path: Path,
) -> None:
    controller, _, _, paster, view, _ = _controller(
        tmp_path, [RuntimeError("CUDA OOM"), "recovered"]
    )
    controller.start()
    controller.toggle_recording()

    controller.toggle_recording()

    assert controller.state is AppState.RETRY_PENDING
    assert view.retry_errors == ["CUDA OOM"]

    controller.retry()

    assert paster.calls == [("recovered", 321)]
    assert controller.state is AppState.READY


def test_model_switch_persists_only_after_success(tmp_path: Path) -> None:
    controller, recognizer, _, _, view, repository = _controller(tmp_path, [])
    controller.start()

    controller.switch_model(ModelId.ACCURATE)

    assert recognizer.switched == [ModelId.ACCURATE]
    switching_index = view.states.index(AppState.MODEL_SWITCHING)
    assert view.rendered_models[switching_index] is ModelId.ACCURATE
    assert repository.saved[-1].selected_model is ModelId.ACCURATE
    assert controller.state is AppState.READY


def _shortcut_controller_with_real_repository(
    tmp_path: Path,
) -> tuple[
    AppController,
    SettingsRepository,
    MacPasteShortcutSelection,
    MacClipboardPaster,
    FakeView,
]:
    config_dir = tmp_path / "config"
    repository = SettingsRepository(config_dir)
    settings = Settings(
        hotkey="RightAlt",
        macos_paste_shortcut=MacPasteShortcut.COMMAND_V,
    )
    repository.save(settings)
    selection = MacPasteShortcutSelection(settings.macos_paste_shortcut)
    runtime_paster = MacClipboardPaster(
        object(),
        wait_until_hotkey_released=lambda: None,
        paste_shortcut_selection=selection,
    )
    machine = StateMachine()
    recognizer = FakeLifecycleRecognizer()
    recorder = FakeRecorder(tmp_path)
    view = FakeView()
    controller = AppController(
        machine=machine,
        settings=settings,
        settings_repository=repository,
        recognizer=recognizer,
        session=VoiceSession(machine, recorder, recognizer, FakePaster()),
        view=view,
        task_runner=ImmediateRunner(),
        foreground_window=lambda: 321,
        macos_paste_shortcut_selection=selection,
    )
    controller.start()
    return controller, repository, selection, runtime_paster, view


def test_macos_paste_shortcut_switch_atomically_updates_all_live_state(
    tmp_path: Path,
) -> None:
    controller, repository, selection, paster, view = (
        _shortcut_controller_with_real_repository(tmp_path)
    )
    hotkey_path = tmp_path / "config" / "hotkey.conf"
    original_hotkey = hotkey_path.read_bytes()

    controller.switch_macos_paste_shortcut(MacPasteShortcut.CONTROL_V)

    assert (
        repository.load().macos_paste_shortcut
        is MacPasteShortcut.CONTROL_V
    )
    assert hotkey_path.read_bytes() == original_hotkey
    assert selection.shortcut is MacPasteShortcut.CONTROL_V
    assert paster.paste_shortcut is MacPasteShortcut.CONTROL_V
    # This must inspect controller-owned Settings directly: the public shortcut
    # property aliases the shared selection and cannot prove Settings changed.
    assert (
        controller._settings.macos_paste_shortcut
        is MacPasteShortcut.CONTROL_V
    )
    assert controller.macos_paste_shortcut is MacPasteShortcut.CONTROL_V
    assert view.macos_paste_shortcut is MacPasteShortcut.CONTROL_V
    assert not view.command_v_checked
    assert view.control_v_checked
    assert view.paste_shortcuts == [MacPasteShortcut.CONTROL_V]
    assert view.errors == []

    # A later full settings save must not roll the persisted shortcut back.
    repository.save(controller._settings)

    assert (
        repository.load().macos_paste_shortcut
        is MacPasteShortcut.CONTROL_V
    )


def test_macos_paste_shortcut_switch_is_ignored_while_not_ready(
    tmp_path: Path,
) -> None:
    selection = MacPasteShortcutSelection(MacPasteShortcut.COMMAND_V)
    controller, _, _, _, view, repository = _controller(
        tmp_path,
        [],
        macos_paste_shortcut_selection=selection,
    )
    controller.start()
    controller.toggle_recording()

    controller.switch_macos_paste_shortcut(MacPasteShortcut.CONTROL_V)

    assert controller.macos_paste_shortcut is MacPasteShortcut.COMMAND_V
    assert selection.shortcut is MacPasteShortcut.COMMAND_V
    assert repository.shortcut_updates == []
    assert view.paste_shortcuts == []


@pytest.mark.parametrize("failure_phase", ["partial-write", "replace"])
def test_shortcut_persistence_failure_keeps_all_five_state_holders_old(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    controller, repository, selection, paster, view = (
        _shortcut_controller_with_real_repository(tmp_path)
    )
    config_dir = tmp_path / "config"
    settings_path = config_dir / "settings.json"
    temporary_path = config_dir / "settings.json.tmp"
    hotkey_path = config_dir / "hotkey.conf"
    original_settings = settings_path.read_bytes()
    original_hotkey = hotkey_path.read_bytes()
    original_hotkey_mtime_ns = hotkey_path.stat().st_mtime_ns
    original_write_atomic = SettingsRepository._write_atomic
    original_write_text = Path.write_text
    original_replace = Path.replace
    atomic_targets: list[Path] = []

    def track_write_atomic(path: Path, content: str) -> None:
        atomic_targets.append(path)
        original_write_atomic(path, content)

    def fail_partial_temporary_write(
        path: Path, content: str, *args, **kwargs
    ) -> int:
        if path == temporary_path:
            original_write_text(path, "partial", *args, **kwargs)
            raise OSError("simulated partial temporary write failure")
        return original_write_text(path, content, *args, **kwargs)

    def fail_settings_replace(path: Path, target: Path) -> Path:
        if path == temporary_path and target == settings_path:
            raise OSError("simulated atomic replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(
        SettingsRepository,
        "_write_atomic",
        staticmethod(track_write_atomic),
    )
    if failure_phase == "partial-write":
        monkeypatch.setattr(Path, "write_text", fail_partial_temporary_write)
        expected_error = "simulated partial temporary write failure"
    else:
        monkeypatch.setattr(Path, "replace", fail_settings_replace)
        expected_error = "simulated atomic replace failure"

    controller.switch_macos_paste_shortcut(MacPasteShortcut.CONTROL_V)

    assert settings_path.read_bytes() == original_settings
    assert not temporary_path.exists()
    assert repository.load().macos_paste_shortcut is MacPasteShortcut.COMMAND_V
    assert atomic_targets == [settings_path]
    assert hotkey_path.read_bytes() == original_hotkey
    assert hotkey_path.stat().st_mtime_ns == original_hotkey_mtime_ns
    assert not (config_dir / "hotkey.conf.tmp").exists()
    assert selection.shortcut is MacPasteShortcut.COMMAND_V
    assert paster.paste_shortcut is MacPasteShortcut.COMMAND_V
    assert (
        controller._settings.macos_paste_shortcut
        is MacPasteShortcut.COMMAND_V
    )
    assert controller.macos_paste_shortcut is MacPasteShortcut.COMMAND_V
    assert view.macos_paste_shortcut is MacPasteShortcut.COMMAND_V
    assert view.command_v_checked
    assert not view.control_v_checked
    assert view.paste_shortcuts == [MacPasteShortcut.COMMAND_V]
    assert view.errors == [
        f"无法切换 macOS 粘贴快捷键：{expected_error}"
    ]


def test_controller_has_no_runtime_rollback_callback() -> None:
    parameters = signature(AppController.__init__).parameters

    assert "set_macos_paste_shortcut" not in parameters
    assert "macos_paste_shortcut_selection" in parameters


def test_stop_requested_before_preview_worker_starts_is_serialized(
    tmp_path: Path,
) -> None:
    controller, recognizer, recorder, paster, view, repository = _controller(
        tmp_path, ["final"]
    )
    runner = DeferredRunner()
    machine = StateMachine()
    session = VoiceSession(machine, recorder, recognizer, paster)
    controller = AppController(
        machine=machine,
        settings=Settings(),
        settings_repository=repository,
        recognizer=recognizer,
        session=session,
        view=view,
        task_runner=runner,
        foreground_window=lambda: 321,
    )
    controller.start()
    runner.complete_next()
    controller.toggle_recording()
    controller.request_preview()
    controller.request_preview()

    controller.toggle_recording()

    assert view.states[-1] is AppState.FINALIZING
    assert view.timers_stopped == 1
    assert len(runner.tasks) == 1
    assert recorder.seal_calls == 1
    assert not recorder.started
    assert view.timers_stopped == 1
    runner.complete_next()
    assert len(runner.tasks) == 1
    assert recognizer.transcribe_calls == 0
    assert view.previews == []
    assert view.warnings == []
    runner.complete_next()
    assert paster.calls == [("final", 321)]
    assert controller.state is AppState.READY


def test_busy_preview_keeps_only_one_latest_follow_up(tmp_path: Path) -> None:
    controller, recognizer, recorder, paster, view, repository = _controller(
        tmp_path, ["first", "latest"]
    )
    runner = DeferredRunner()
    machine = StateMachine()
    session = VoiceSession(machine, recorder, recognizer, paster)
    controller = AppController(
        machine=machine,
        settings=Settings(),
        settings_repository=repository,
        recognizer=recognizer,
        session=session,
        view=view,
        task_runner=runner,
        foreground_window=lambda: 321,
    )
    controller.start()
    runner.complete_next()
    controller.toggle_recording()
    controller.request_preview()

    controller.request_preview()
    controller.request_preview()

    assert len(runner.tasks) == 1
    runner.complete_next()
    assert view.previews == ["first"]
    assert len(runner.tasks) == 1

    runner.complete_next()

    assert recognizer.transcribe_calls == 2
    assert view.previews == ["first", "latest"]
    assert runner.tasks == []
    assert controller.state is AppState.RECORDING


def test_nonfatal_capture_warning_is_reported_after_result(
    tmp_path: Path,
) -> None:
    controller, _, recorder, _, view, _ = _controller(
        tmp_path, ["recognized"]
    )
    def stop_with_warning() -> AudioArtifact:
        recorder.started = False
        return AudioArtifact(
            np.zeros(16_000, dtype=np.float32),
            sample_rate=16_000,
            warnings=("input overflow",),
        )

    recorder.stop = stop_with_warning
    controller.start()
    controller.toggle_recording()
    controller.toggle_recording()

    assert view.errors == []
    assert view.warnings == ["录音期间出现 input overflow，结果可能不完整。"]


def test_live_preview_failure_warns_without_hiding_recording(
    tmp_path: Path,
) -> None:
    controller, _, _, _, view, _ = _controller(
        tmp_path, [RuntimeError("preview failed")]
    )
    controller.start()
    controller.toggle_recording()

    controller.request_preview()

    assert view.errors == []
    assert view.warnings == [
        "实时转写失败，将在下一次刷新时重试：preview failed"
    ]
    assert controller.state is AppState.RECORDING


def test_configured_preview_interval_is_forwarded_to_view(
    tmp_path: Path,
) -> None:
    _, recognizer, recorder, paster, view, repository = _controller(
        tmp_path, []
    )
    machine = StateMachine()
    session = VoiceSession(machine, recorder, recognizer, paster)
    controller = AppController(
        machine=machine,
        settings=Settings(preview_interval_ms=500),
        settings_repository=repository,
        recognizer=recognizer,
        session=session,
        view=view,
        task_runner=ImmediateRunner(),
        foreground_window=lambda: 321,
    )
    controller.start()
    controller.toggle_recording()

    assert view.preview_intervals == [500]
