from __future__ import annotations

from dataclasses import replace
import threading
from typing import Callable, Protocol

from vim2.config import Settings
from vim2.hotwords import HotwordSnapshot
from vim2.models import MODEL_SPECS, ModelId
from vim2.recognizer import TranscriptionCancelled
from vim2.session import FinalRecognitionError, VoiceSession
from vim2.state import AppState, StateMachine


class SettingsWriter(Protocol):
    def save(self, settings: Settings) -> None: ...

    def save_selected_model(self, model_id: ModelId) -> None: ...


class ModelLifecycle(Protocol):
    @property
    def loaded_model(self) -> ModelId | None: ...

    def load(self, model_id: ModelId) -> None: ...

    def switch(self, model_id: ModelId) -> None: ...

    def unload(self) -> None: ...

    def reload_hotwords(self) -> HotwordSnapshot: ...


class ControllerView(Protocol):
    def render_state(self, state: AppState, model_id: ModelId) -> None: ...

    def start_recording_timers(
        self,
        max_seconds: int,
        target_window: int,
        preview_interval_ms: int,
    ) -> None: ...

    def stop_recording_timers(self) -> None: ...

    def show_preview(self, text: str) -> None: ...

    def show_error(self, message: str) -> None: ...

    def show_warning(self, message: str) -> None: ...

    def show_info(self, message: str) -> None: ...

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
        restart_application: Callable[[], None] = lambda: None,
        open_hotwords_file: Callable[[], bool] = lambda: False,
    ) -> None:
        self._machine = machine
        self._settings = settings
        self._settings_repository = settings_repository
        self._recognizer = recognizer
        self._session = session
        self._view = view
        self._runner = task_runner
        self._foreground_window = foreground_window
        self._restart_application = restart_application
        self._open_hotwords_file = open_hotwords_file
        self._preview_busy = False
        self._preview_pending = False
        self._preview_cancel_event: threading.Event | None = None
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
        if self._preview_busy:
            self._preview_pending = False
            if self._preview_cancel_event is not None:
                self._preview_cancel_event.set()
            if not self._stop_requested and self._seal_recording():
                self._stop_requested = True
                self._view.render_state(
                    AppState.FINALIZING, self.selected_model
                )
            return
        if self.state is AppState.READY:
            self._start_recording()
        elif self.state is AppState.RECORDING:
            self._finalize()
        elif self.state is AppState.LIVE_TRANSCRIBING:
            self._stop_requested = True

    def _start_recording(self) -> None:
        self._preview_pending = False
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
            self._settings.preview_interval_ms,
        )

    def request_preview(self) -> None:
        if self._preview_busy:
            if not self._stop_requested and not self._cancel_requested:
                self._preview_pending = True
            return
        if self.state is not AppState.RECORDING:
            return
        self._preview_busy = True
        cancel_event = threading.Event()
        self._preview_cancel_event = cancel_event
        self._view.render_state(
            AppState.LIVE_TRANSCRIBING, self.selected_model
        )
        self._runner.submit(
            lambda: self._session.preview(cancel_event=cancel_event),
            self._on_preview,
            self._on_preview_error,
        )

    def _on_preview(self, result: object) -> None:
        self._preview_busy = False
        self._preview_cancel_event = None
        self._view.show_preview(str(result))
        self._after_preview()

    def _on_preview_error(self, error: Exception) -> None:
        self._preview_busy = False
        self._preview_cancel_event = None
        if not isinstance(error, TranscriptionCancelled):
            self._view.show_warning(
                f"实时转写失败，将在下一次刷新时重试：{error}"
            )
        self._after_preview()

    def _after_preview(self) -> None:
        if self._cancel_requested:
            self._cancel_requested = False
            self.cancel()
        elif self._stop_requested:
            self._stop_requested = False
            self._finalize()
        else:
            self._preview_pending = False
            self._render()

    def _finalize(self) -> None:
        self._preview_pending = False
        if not self._seal_recording():
            return
        self._view.render_state(AppState.FINALIZING, self.selected_model)
        self._operation_busy = True
        self._runner.submit(
            self._session.stop,
            self._on_finalized,
            self._on_final_error,
        )

    def _seal_recording(self) -> bool:
        self._view.stop_recording_timers()
        try:
            self._session.seal()
        except (OSError, RuntimeError) as exc:
            self._stop_requested = False
            self._render()
            self._view.show_error(f"无法停止录音：{exc}")
            return False
        return True

    def _on_finalized(self, result: object) -> None:
        del result
        self._operation_busy = False
        self._render()
        self._view.hide_overlay()
        if self._session.warnings:
            warnings = "；".join(self._session.warnings)
            self._view.show_warning(
                f"录音期间出现 {warnings}，结果可能不完整。"
            )

    def _on_final_error(self, error: Exception) -> None:
        self._operation_busy = False
        self._render()
        if isinstance(error, FinalRecognitionError):
            self._view.show_retry_error(str(error))
        else:
            self._view.show_error(str(error))

    def cancel(self) -> None:
        self._preview_pending = False
        if self._preview_busy:
            if self._preview_cancel_event is not None:
                self._preview_cancel_event.set()
            self._cancel_requested = True
            self._stop_requested = False
            return
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

    def reload_hotwords(self) -> None:
        if self.state is not AppState.READY or self._operation_busy:
            return
        self._operation_busy = True
        self._runner.submit(
            self._recognizer.reload_hotwords,
            self._on_hotwords_reloaded,
            self._on_hotword_reload_error,
        )

    def _on_hotwords_reloaded(self, result: object) -> None:
        self._operation_busy = False
        if not isinstance(result, HotwordSnapshot):
            self._view.show_error("重新加载热词失败：返回了无效结果。")
            return
        self._view.show_info(f"已重新加载 {len(result.entries)} 个热词。")

    def _on_hotword_reload_error(self, error: Exception) -> None:
        self._operation_busy = False
        self._view.show_error(
            "重新加载热词失败，继续使用上次成功的词表："
            f"{error}"
        )

    def open_hotwords_file(self) -> None:
        try:
            opened = self._open_hotwords_file()
        except (OSError, RuntimeError):
            opened = False
        if not opened:
            self._view.show_error("无法打开热词文件。")

    def switch_model(self, model_id: ModelId) -> None:
        if (
            self.state is not AppState.READY
            or model_id is self.selected_model
            or self._operation_busy
        ):
            return
        if (
            MODEL_SPECS[model_id].backend
            is not MODEL_SPECS[self.selected_model].backend
        ):
            self._settings = replace(
                self._settings, selected_model=model_id
            )
            self._settings_repository.save_selected_model(model_id)
            self._restart_application()
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
        self._settings_repository.save_selected_model(model_id)
        self._operation_busy = False
        self._machine.transition_to(AppState.READY)
        self._render()

    def _on_model_switch_error(self, error: Exception) -> None:
        self._operation_busy = False
        next_state = (
            AppState.READY
            if self._recognizer.loaded_model is self.selected_model
            else AppState.ERROR
        )
        self._machine.transition_to(next_state)
        self._render()
        self._view.show_error(f"模型切换失败：{error}")

    def shutdown(self) -> None:
        self._preview_pending = False
        if self._preview_cancel_event is not None:
            self._preview_cancel_event.set()
        self._session.shutdown()
        if self.state is not AppState.EXITING:
            self._machine.transition_to(AppState.EXITING)
        self._view.stop_recording_timers()
        self._recognizer.unload()

    def _render(self) -> None:
        self._view.render_state(self.state, self.selected_model)
