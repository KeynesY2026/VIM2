from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol

from vim2.config import Settings
from vim2.models import ModelId
from vim2.session import FinalRecognitionError, VoiceSession
from vim2.state import AppState, StateMachine


class SettingsWriter(Protocol):
    def save(self, settings: Settings) -> None: ...


class ModelLifecycle(Protocol):
    def load(self, model_id: ModelId) -> None: ...

    def switch(self, model_id: ModelId) -> None: ...

    def unload(self) -> None: ...


class ControllerView(Protocol):
    def render_state(self, state: AppState, model_id: ModelId) -> None: ...

    def start_recording_timers(
        self, max_seconds: int, target_window: int
    ) -> None: ...

    def stop_recording_timers(self) -> None: ...

    def show_preview(self, text: str) -> None: ...

    def show_error(self, message: str) -> None: ...

    def show_retry_error(self, message: str) -> None: ...

    def hide_overlay(self) -> None: ...


class TaskRunner(Protocol):
    def submit(
        self,
        work: Callable[[], object],
        on_success: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None: ...


class AppController:
    def __init__(
        self,
        *,
        machine: StateMachine,
        settings: Settings,
        settings_repository: SettingsWriter,
        recognizer: ModelLifecycle,
        session: VoiceSession,
        view: ControllerView,
        task_runner: TaskRunner,
        foreground_window: Callable[[], int],
    ) -> None:
        self._machine = machine
        self._settings = settings
        self._settings_repository = settings_repository
        self._recognizer = recognizer
        self._session = session
        self._view = view
        self._runner = task_runner
        self._foreground_window = foreground_window
        self._preview_busy = False
        self._stop_requested = False
        self._cancel_requested = False
        self._operation_busy = False

    @property
    def state(self) -> AppState:
        return self._machine.current

    @property
    def selected_model(self) -> ModelId:
        return self._settings.selected_model

    def start(self) -> None:
        self._machine.transition_to(AppState.MODEL_LOADING)
        self._render()
        self._operation_busy = True
        self._runner.submit(
            lambda: self._recognizer.load(self.selected_model),
            self._on_model_loaded,
            self._on_startup_error,
        )

    def _on_model_loaded(self, result: object) -> None:
        del result
        self._operation_busy = False
        self._machine.transition_to(AppState.READY)
        self._render()

    def _on_startup_error(self, error: Exception) -> None:
        self._operation_busy = False
        self._machine.transition_to(AppState.ERROR)
        self._render()
        self._view.show_error(f"模型加载失败：{error}")

    def toggle_recording(self) -> None:
        if self._operation_busy:
            return
        if self.state is AppState.READY:
            self._start_recording()
        elif self.state is AppState.RECORDING:
            self._finalize()
        elif self.state is AppState.LIVE_TRANSCRIBING:
            self._stop_requested = True

    def _start_recording(self) -> None:
        target_window = self._foreground_window()
        try:
            self._session.start(
                target_window=target_window,
                model_id=self.selected_model,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self._view.show_error(f"无法开始录音：{exc}")
            return
        self._render()
        self._view.start_recording_timers(
            self._settings.max_recording_seconds,
            target_window,
        )

    def request_preview(self) -> None:
        if self.state is not AppState.RECORDING or self._preview_busy:
            return
        self._preview_busy = True
        self._view.render_state(
            AppState.LIVE_TRANSCRIBING, self.selected_model
        )
        self._runner.submit(
            self._session.preview,
            self._on_preview,
            self._on_preview_error,
        )

    def _on_preview(self, result: object) -> None:
        self._preview_busy = False
        self._view.show_preview(str(result))
        self._after_preview()

    def _on_preview_error(self, error: Exception) -> None:
        self._preview_busy = False
        self._view.show_error(f"实时转写失败，将在停止后重试：{error}")
        self._after_preview()

    def _after_preview(self) -> None:
        if self._cancel_requested:
            self._cancel_requested = False
            self.cancel()
        elif self._stop_requested:
            self._stop_requested = False
            self._finalize()
        else:
            self._render()

    def _finalize(self) -> None:
        self._view.stop_recording_timers()
        self._view.render_state(AppState.FINALIZING, self.selected_model)
        self._operation_busy = True
        self._runner.submit(
            self._session.stop,
            self._on_finalized,
            self._on_final_error,
        )

    def _on_finalized(self, result: object) -> None:
        del result
        self._operation_busy = False
        self._render()
        self._view.hide_overlay()

    def _on_final_error(self, error: Exception) -> None:
        self._operation_busy = False
        self._render()
        if isinstance(error, FinalRecognitionError):
            self._view.show_retry_error(str(error))
        else:
            self._view.show_error(str(error))

    def cancel(self) -> None:
        if self.state is AppState.LIVE_TRANSCRIBING:
            self._cancel_requested = True
            return
        if self.state not in {AppState.RECORDING, AppState.RETRY_PENDING}:
            return
        self._view.stop_recording_timers()
        self._session.cancel()
        self._render()
        self._view.hide_overlay()

    def retry(self) -> None:
        if self.state is not AppState.RETRY_PENDING or self._operation_busy:
            return
        self._view.render_state(AppState.FINALIZING, self.selected_model)
        self._operation_busy = True
        self._runner.submit(
            self._session.retry,
            self._on_finalized,
            self._on_final_error,
        )

    def switch_model(self, model_id: ModelId) -> None:
        if (
            self.state is not AppState.READY
            or model_id is self.selected_model
            or self._operation_busy
        ):
            return
        self._machine.transition_to(AppState.MODEL_SWITCHING)
        self._view.render_state(AppState.MODEL_SWITCHING, model_id)
        self._operation_busy = True
        self._runner.submit(
            lambda: self._recognizer.switch(model_id),
            lambda result: self._on_model_switched(model_id, result),
            self._on_model_switch_error,
        )

    def _on_model_switched(
        self, model_id: ModelId, result: object
    ) -> None:
        del result
        self._settings = replace(self._settings, selected_model=model_id)
        self._settings_repository.save(self._settings)
        self._operation_busy = False
        self._machine.transition_to(AppState.READY)
        self._render()

    def _on_model_switch_error(self, error: Exception) -> None:
        self._operation_busy = False
        self._machine.transition_to(AppState.READY)
        self._render()
        self._view.show_error(f"模型切换失败：{error}")

    def shutdown(self) -> None:
        self._session.shutdown()
        if self.state is not AppState.EXITING:
            self._machine.transition_to(AppState.EXITING)
        self._view.stop_recording_timers()
        self._recognizer.unload()

    def _render(self) -> None:
        self._view.render_state(self.state, self.selected_model)
