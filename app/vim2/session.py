from __future__ import annotations

from pathlib import Path
from typing import Protocol

from vim2.audio import AudioArtifact
from vim2.models import ModelId
from vim2.state import AppState, StateMachine
from vim2.transcript import (
    StableCheckpoint,
    StablePrefixTracker,
    merge_stable_tail,
)

TAIL_OVERLAP_SECONDS = 8
MAX_TAIL_RATIO = 0.70


class Recorder(Protocol):
    def start(self) -> None: ...

    def stop(self) -> AudioArtifact: ...

    def snapshot(self) -> AudioArtifact: ...

    def slice_from(
        self, artifact: AudioArtifact, start_frame: int
    ) -> AudioArtifact: ...

    def cancel(self) -> None: ...

    def discard(self, artifact: AudioArtifact) -> None: ...


class Recognizer(Protocol):
    def transcribe(self, path: Path, model_id: ModelId) -> str: ...


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
        self._target_window = target_window
        self._model_id = model_id
        self._warnings = ()
        self._machine.transition_to(AppState.RECORDING)

    def stop(self) -> str:
        if self.state is not AppState.RECORDING:
            raise RuntimeError("No recording is in progress")
        try:
            artifact = self._recorder.stop()
        except (OSError, RuntimeError):
            self._machine.transition_to(AppState.READY)
            raise
        self._pending_audio = artifact
        self._warnings = artifact.warnings
        self._machine.transition_to(AppState.FINALIZING)
        return self._recognize_pending(allow_tail=True)

    def retry(self) -> str:
        if self.state is not AppState.RETRY_PENDING:
            raise RuntimeError("There is no failed recognition to retry")
        self._machine.transition_to(AppState.FINALIZING)
        return self._recognize_pending(allow_tail=False)

    def preview(self) -> str:
        if self.state is not AppState.RECORDING or self._model_id is None:
            raise RuntimeError("No recording is in progress")
        artifact = self._recorder.snapshot()
        self._machine.transition_to(AppState.LIVE_TRANSCRIBING)
        try:
            text = self._recognizer.transcribe(
                artifact.path, self._model_id
            ).strip()
            self._stable_prefix.observe(text, artifact.frame_count)
            return text
        finally:
            self._recorder.discard(artifact)
            if not self._shutting_down:
                self._machine.transition_to(AppState.RECORDING)

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
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
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
            overlap_frames = artifact.sample_rate * TAIL_OVERLAP_SECONDS
            start_frame = max(0, checkpoint.frame_count - overlap_frames)
            tail_frames = artifact.frame_count - start_frame
            if tail_frames <= artifact.frame_count * MAX_TAIL_RATIO:
                merged = self._recognize_tail(
                    artifact, model_id, checkpoint, start_frame
                )
                if merged is not None:
                    return merged
        return self._recognizer.transcribe(artifact.path, model_id).strip()

    def _recognize_tail(
        self,
        artifact: AudioArtifact,
        model_id: ModelId,
        checkpoint: StableCheckpoint,
        start_frame: int,
    ) -> str | None:
        tail = self._recorder.slice_from(artifact, start_frame)
        try:
            text = self._recognizer.transcribe(tail.path, model_id).strip()
            return merge_stable_tail(checkpoint, text)
        finally:
            self._recorder.discard(tail)

    def _complete(self, text: str) -> str:
        if self._pending_audio is None or self._target_window is None:
            raise RuntimeError("Recognition session data is incomplete")
        artifact = self._pending_audio
        try:
            if text:
                self._paster.paste(text, self._target_window)
            return text
        finally:
            self._recorder.discard(artifact)
            self._pending_audio = None
            self._machine.transition_to(AppState.READY)

    def cancel(self) -> None:
        if self.state is AppState.RECORDING:
            self._recorder.cancel()
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
        if self._pending_audio is not None:
            self._recorder.discard(self._pending_audio)
            self._pending_audio = None
