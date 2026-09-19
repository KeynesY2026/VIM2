from pathlib import Path
import threading

import numpy as np

from vim2.audio import AudioArtifact
from vim2.config import Settings
from vim2.controller import AppController
from vim2.hotwords import HotwordSnapshot
from vim2.models import ModelId
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
        self.loaded_model: ModelId | None = None
        self.unloaded = False
        self.transcribe_calls = 0
        self.reload_calls = 0
        self.reload_error: Exception | None = None
        self.switch_error: Exception | None = None

    def load(self, model_id: ModelId) -> None:
        self.loaded.append(model_id)
        self.loaded_model = model_id

    def switch(self, model_id: ModelId) -> None:
        self.switched.append(model_id)
        if self.switch_error is not None:
            self.loaded_model = None
            raise self.switch_error
        self.loaded_model = model_id

    def unload(self) -> None:
        self.unloaded = True
        self.loaded_model = None

    def reload_hotwords(self) -> HotwordSnapshot:
        self.reload_calls += 1
        if self.reload_error is not None:
            raise self.reload_error
        return HotwordSnapshot(("VIM2", "Qwen"))

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

    def snapshot(
        self, max_seconds: int | None = None
    ) -> AudioArtifact:
        del max_seconds
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
        self.saved_models: list[ModelId] = []

    def save(self, settings: Settings) -> None:
        self.saved.append(settings)

    def save_selected_model(self, model_id: ModelId) -> None:
        self.saved_models.append(model_id)


class FakeView:
    def __init__(self) -> None:
        self.states: list[AppState] = []
        self.rendered_models: list[ModelId] = []
        self.previews: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.retry_errors: list[str] = []
        self.timers_started: list[int] = []
        self.preview_intervals: list[int] = []
        self.timers_stopped = 0

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

    def show_info(self, message: str) -> None:
        self.infos.append(message)

    def show_retry_error(self, message: str) -> None:
        self.retry_errors.append(message)

    def hide_overlay(self) -> None:
        pass


def _controller(tmp_path: Path, responses: list[str | Exception]):
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
    assert repository.saved_models == [ModelId.FAST]
    assert repository.saved == []
    assert restart_calls == [True]


def test_startup_loads_selected_model_before_enabling_recording(
    tmp_path: Path,
) -> None:
    controller, recognizer, _, _, view, _ = _controller(tmp_path, [])

    controller.start()

    assert recognizer.loaded == [ModelId.FAST]
    assert controller.state is AppState.READY
    assert view.states[:2] == [AppState.MODEL_LOADING, AppState.READY]


def test_hotword_reload_blocks_recording_until_snapshot_is_ready(
    tmp_path: Path,
) -> None:
    controller, recognizer, recorder, paster, view, repository = _controller(
        tmp_path, []
    )
    runner = DeferredRunner()
    machine = StateMachine()
    controller = AppController(
        machine=machine,
        settings=Settings(),
        settings_repository=repository,
        recognizer=recognizer,
        session=VoiceSession(machine, recorder, recognizer, paster),
        view=view,
        task_runner=runner,
        foreground_window=lambda: 321,
    )
    controller.start()
    runner.complete_next()

    controller.reload_hotwords()
    controller.toggle_recording()

    assert not recorder.started
    runner.complete_next()
    assert recognizer.reload_calls == 1
    assert view.infos == ["已重新加载 2 个热词。"]
    controller.toggle_recording()
    assert recorder.started


def test_hotword_reload_is_ignored_during_recording(tmp_path: Path) -> None:
    controller, recognizer, _, _, _, _ = _controller(tmp_path, [])
    controller.start()
    controller.toggle_recording()

    controller.reload_hotwords()

    assert recognizer.reload_calls == 0


def test_failed_hotword_reload_reports_previous_snapshot_is_retained(
    tmp_path: Path,
) -> None:
    controller, recognizer, _, _, view, _ = _controller(tmp_path, [])
    controller.start()
    recognizer.reload_error = ValueError("line 2 contains an ASCII comma")

    controller.reload_hotwords()

    assert view.errors == [
        "重新加载热词失败，继续使用上次成功的词表："
        "line 2 contains an ASCII comma"
    ]


def test_open_hotword_file_reports_platform_failure(tmp_path: Path) -> None:
    controller, recognizer, recorder, paster, view, repository = _controller(
        tmp_path, []
    )
    machine = StateMachine()
    controller = AppController(
        machine=machine,
        settings=Settings(),
        settings_repository=repository,
        recognizer=recognizer,
        session=VoiceSession(machine, recorder, recognizer, paster),
        view=view,
        task_runner=ImmediateRunner(),
        foreground_window=lambda: 321,
        open_hotwords_file=lambda: False,
    )

    controller.open_hotwords_file()

    assert view.errors == ["无法打开热词文件。"]


def test_record_preview_and_final_result_flow(tmp_path: Path) -> None:
    controller, _, recorder, paster, view, _ = _controller(
        tmp_path, ["临时文本", "最终文本"]
    )
    controller.start()

    controller.toggle_recording()
    controller.request_preview()
    controller.toggle_recording()

    assert view.timers_started == [300]
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
    assert repository.saved_models == [ModelId.ACCURATE]
    assert repository.saved == []
    assert controller.state is AppState.READY


def test_failed_model_switch_without_resident_model_enters_error(
    tmp_path: Path,
) -> None:
    controller, recognizer, _, _, view, _ = _controller(tmp_path, [])
    controller.start()
    recognizer.switch_error = RuntimeError("switch and restore failed")

    controller.switch_model(ModelId.ACCURATE)

    assert controller.state is AppState.ERROR
    assert view.errors == [
        "模型切换失败：switch and restore failed"
    ]


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


def test_busy_preview_waits_for_next_timer_tick(tmp_path: Path) -> None:
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
    assert runner.tasks == []

    controller.request_preview()
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
