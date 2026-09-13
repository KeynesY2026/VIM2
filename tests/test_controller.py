from pathlib import Path

from vim2.audio import AudioArtifact
from vim2.config import Settings
from vim2.controller import AppController
from vim2.models import ModelId
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

    def load(self, model_id: ModelId) -> None:
        self.loaded.append(model_id)

    def switch(self, model_id: ModelId) -> None:
        self.switched.append(model_id)

    def unload(self) -> None:
        self.unloaded = True

    def transcribe(self, path: Path, model_id: ModelId) -> str:
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeRecorder:
    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "recording.wav"
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> AudioArtifact:
        self.path.write_bytes(b"audio")
        self.started = False
        return AudioArtifact(self.path, 16_000, 16_000)

    def snapshot(self) -> AudioArtifact:
        self.path.write_bytes(b"preview")
        return AudioArtifact(self.path, 8_000, 16_000)

    def cancel(self) -> None:
        self.started = False

    def discard(self, artifact: AudioArtifact) -> None:
        artifact.path.unlink(missing_ok=True)


class FakePaster:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def paste(self, text: str, target_window: int) -> None:
        self.calls.append((text, target_window))


class FakeSettingsRepository:
    def __init__(self) -> None:
        self.saved: list[Settings] = []

    def save(self, settings: Settings) -> None:
        self.saved.append(settings)


class FakeView:
    def __init__(self) -> None:
        self.states: list[AppState] = []
        self.rendered_models: list[ModelId] = []
        self.previews: list[str] = []
        self.errors: list[str] = []
        self.retry_errors: list[str] = []
        self.timers_started: list[int] = []
        self.timers_stopped = 0

    def render_state(self, state: AppState, model_id: ModelId) -> None:
        self.states.append(state)
        self.rendered_models.append(model_id)

    def start_recording_timers(
        self, max_seconds: int, target_window: int
    ) -> None:
        self.timers_started.append(max_seconds)

    def stop_recording_timers(self) -> None:
        self.timers_stopped += 1

    def show_preview(self, text: str) -> None:
        self.previews.append(text)

    def show_error(self, message: str) -> None:
        self.errors.append(message)

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
    assert view.previews == ["临时文本"]
    assert paster.calls == [("最终文本", 321)]
    assert not recorder.path.exists()
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


def test_stop_requested_before_preview_worker_starts_is_serialized(
    tmp_path: Path,
) -> None:
    controller, recognizer, recorder, paster, view, repository = _controller(
        tmp_path, ["preview", "final"]
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

    controller.toggle_recording()

    assert len(runner.tasks) == 1
    runner.complete_next()
    assert len(runner.tasks) == 1
    runner.complete_next()
    assert paster.calls == [("final", 321)]
    assert controller.state is AppState.READY
