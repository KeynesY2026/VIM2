from __future__ import annotations

import logging
import threading
from typing import Protocol

from vim2.audio import AudioArtifact
from vim2.models import ModelId
from vim2.postprocessing import (
    TextPostProcessor,
    TextPostProcessingPipeline,
)
from vim2.recognizer import TranscriptionCancelled
from vim2.state import AppState, StateMachine
from vim2.transcript import (
    StableCheckpoint,
    StablePrefixTracker,
    merge_stable_tail,
)

MAX_TAIL_RATIO = 0.70
MAX_PREVIEW_SECONDS = 8
RECOGNITION_ERRORS = (OSError, RuntimeError, ValueError, MemoryError)


class Recorder(Protocol):
    def start(self) -> None: ...

    def seal(self) -> None: ...

    def stop(self) -> AudioArtifact: ...

    def snapshot(
        self, max_seconds: int | None = None
    ) -> AudioArtifact: ...

    def slice_from(
        self, artifact: AudioArtifact, start_frame: int
    ) -> AudioArtifact: ...

    def cancel(self) -> None: ...

    def discard(self, artifact: AudioArtifact) -> None: ...


class Recognizer(Protocol):
    def transcribe(
        self,
        artifact: AudioArtifact,
        model_id: ModelId,
        *,
        cancel_event: threading.Event | None = None,
    ) -> str: ...


class Paster(Protocol):
    def paste(self, text: str, target_window: int) -> None: ...


class FinalRecognitionError(RuntimeError):
    pass


class VoiceSession:
    def __init__(
        self,
        state_machine: StateMachine,
        recorder: Recorder,
        recognizer: Recognizer,
        paster: Paster,
        *,
        tail_overlap_seconds: int = 5,
        preview_window_seconds: int | None = None,
        text_postprocessor: TextPostProcessor | None = None,
    ) -> None:
        self._machine = state_machine
        self._recorder = recorder
        self._recognizer = recognizer
        self._paster = paster
        self._target_window: int | None = None
        self._model_id: ModelId | None = None
        self._pending_audio: AudioArtifact | None = None
        self._warnings: tuple[str, ...] = ()
        self._shutting_down = False
        self._stable_prefix = StablePrefixTracker()
        self._capture_sealed = False
        self._tail_overlap_seconds = tail_overlap_seconds
        self._preview_window_seconds = (
            MAX_PREVIEW_SECONDS
            if preview_window_seconds is None
            else preview_window_seconds
        )
        self._text_postprocessor = (
            text_postprocessor or TextPostProcessingPipeline()
        )

    @property
    def state(self) -> AppState:
        return self._machine.current

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._warnings

    def start(self, *, target_window: int, model_id: ModelId) -> None:
        if self.state is not AppState.READY:
            raise RuntimeError(f"Cannot start recording while {self.state.value}")
        self._recorder.start()
        self._stable_prefix.reset()
        self._capture_sealed = False
        self._target_window = target_window
        self._model_id = model_id
        self._warnings = ()
        self._machine.transition_to(AppState.RECORDING)

    def seal(self) -> None:
        if self.state not in {
            AppState.RECORDING,
            AppState.LIVE_TRANSCRIBING,
        }:
            raise RuntimeError("No recording is in progress")
        if self._capture_sealed:
            return
        try:
            self._recorder.seal()
        except (OSError, RuntimeError):
            self._machine.transition_to(AppState.READY)
            raise
        self._capture_sealed = True

    def stop(self) -> str:
        if self.state is not AppState.RECORDING:
            raise RuntimeError("No recording is in progress")
        if not self._capture_sealed:
            self.seal()
        try:
            artifact = self._recorder.stop()
        except (OSError, RuntimeError):
            self._machine.transition_to(AppState.READY)
            raise
        self._capture_sealed = False
        self._pending_audio = artifact
        self._warnings = artifact.warnings
        self._machine.transition_to(AppState.FINALIZING)
        return self._recognize_pending(allow_tail=True)

    def retry(self) -> str:
        if self.state is not AppState.RETRY_PENDING:
            raise RuntimeError("There is no failed recognition to retry")
        self._machine.transition_to(AppState.FINALIZING)
        return self._recognize_pending(allow_tail=False)

    def preview(
        self, *, cancel_event: threading.Event | None = None
    ) -> str:
        if self.state is not AppState.RECORDING or self._model_id is None:
            raise RuntimeError("No recording is in progress")
        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelled()
        artifact = self._recorder.snapshot(
            max_seconds=self._preview_window_seconds
        )
        self._machine.transition_to(AppState.LIVE_TRANSCRIBING)
        try:
            text, is_complete = self._recognize_preview(
                artifact, self._model_id, cancel_event=cancel_event
            )
            if is_complete:
                self._stable_prefix.observe(text, artifact.end_frame)
            return text
        finally:
            self._recorder.discard(artifact)
            if (
                not self._shutting_down
                and self.state is AppState.LIVE_TRANSCRIBING
            ):
                self._machine.transition_to(AppState.RECORDING)

    def _recognize_preview(
        self,
        artifact: AudioArtifact,
        model_id: ModelId,
        *,
        cancel_event: threading.Event | None,
    ) -> tuple[str, bool]:
        max_frames = artifact.sample_rate * self._preview_window_seconds
        checkpoint = self._stable_prefix.checkpoint
        start_frame = max(0, artifact.frame_count - max_frames)
        if checkpoint is not None:
            overlap_frames = (
                artifact.sample_rate * self._tail_overlap_seconds
            )
            start_frame = max(
                start_frame,
                checkpoint.frame_count
                - overlap_frames
                - artifact.start_frame,
            )

        window = artifact
        sliced_window: AudioArtifact | None = None
        try:
            if start_frame > 0:
                sliced_window = self._recorder.slice_from(
                    artifact, start_frame
                )
                window = sliced_window
            self._log_preview_window(window, model_id)
            text = self._recognizer.transcribe(
                window, model_id, cancel_event=cancel_event
            ).strip()
            if checkpoint is not None:
                merged = merge_stable_tail(checkpoint, text)
                if merged is not None:
                    return merged, True
                return text, False
            is_complete = artifact.start_frame == 0 and start_frame == 0
            return text, is_complete
        finally:
            if sliced_window is not None:
                self._recorder.discard(sliced_window)

    @staticmethod
    def _log_preview_window(
        artifact: AudioArtifact, model_id: ModelId
    ) -> None:
        logging.getLogger(__name__).info(
            "Preview transcription input: model=%s duration=%.3fs frames=%d-%d",
            model_id,
            artifact.duration_seconds,
            artifact.start_frame,
            artifact.end_frame,
        )

    def _recognize_pending(self, *, allow_tail: bool) -> str:
        if (
            self._pending_audio is None
            or self._model_id is None
            or self._target_window is None
        ):
            raise RuntimeError("Recognition session data is incomplete")
        try:
            text = self._recognize_final(
                self._pending_audio,
                self._model_id,
                allow_tail=allow_tail,
            )
        except RECOGNITION_ERRORS as exc:
            self._machine.transition_to(AppState.RETRY_PENDING)
            raise FinalRecognitionError(str(exc)) from exc

        return self._complete(text)

    def _recognize_final(
        self,
        artifact: AudioArtifact,
        model_id: ModelId,
        *,
        allow_tail: bool,
    ) -> str:
        checkpoint = self._stable_prefix.checkpoint
        if allow_tail and checkpoint is not None:
            overlap_frames = (
                artifact.sample_rate * self._tail_overlap_seconds
            )
            start_frame = max(0, checkpoint.frame_count - overlap_frames)
            tail_frames = artifact.frame_count - start_frame
            if tail_frames <= artifact.frame_count * MAX_TAIL_RATIO:
                merged = self._recognize_tail(
                    artifact, model_id, checkpoint, start_frame
                )
                if merged is not None:
                    return merged
        return self._recognizer.transcribe(artifact, model_id).strip()

    def _recognize_tail(
        self,
        artifact: AudioArtifact,
        model_id: ModelId,
        checkpoint: StableCheckpoint,
        start_frame: int,
    ) -> str | None:
        tail: AudioArtifact | None = None
        try:
            tail = self._recorder.slice_from(artifact, start_frame)
            text = self._recognizer.transcribe(tail, model_id).strip()
            return merge_stable_tail(checkpoint, text)
        except RECOGNITION_ERRORS:
            return None
        finally:
            if tail is not None:
                self._recorder.discard(tail)

    def _complete(self, text: str) -> str:
        if self._pending_audio is None or self._target_window is None:
            raise RuntimeError("Recognition session data is incomplete")
        artifact = self._pending_audio
        try:
            formatted_text = self._text_postprocessor.process(text)
            if formatted_text:
                self._paster.paste(formatted_text, self._target_window)
            return formatted_text
        finally:
            self._recorder.discard(artifact)
            self._pending_audio = None
            self._capture_sealed = False
            self._machine.transition_to(AppState.READY)

    def cancel(self) -> None:
        if self.state is AppState.RECORDING:
            self._recorder.cancel()
            self._capture_sealed = False
        elif self.state is AppState.RETRY_PENDING and self._pending_audio:
            self._recorder.discard(self._pending_audio)
            self._pending_audio = None
        else:
            raise RuntimeError("There is no recording or failed audio to cancel")
        self._machine.transition_to(AppState.READY)

    def shutdown(self) -> None:
        self._shutting_down = True
        if self.state in {AppState.RECORDING, AppState.LIVE_TRANSCRIBING}:
            self._recorder.cancel()
            self._capture_sealed = False
        if self._pending_audio is not None:
            self._recorder.discard(self._pending_audio)
            self._pending_audio = None
