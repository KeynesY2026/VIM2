from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class AudioArtifact:
    samples: np.ndarray
    sample_rate: int
    warnings: tuple[str, ...] = ()

    @property
    def frame_count(self) -> int:
        return len(self.samples)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


class AudioRecorder:
    def __init__(
        self,
        *,
        sounddevice: Any | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        if sounddevice is None:
            import sounddevice

            sounddevice = sounddevice
        self._sounddevice = sounddevice
        self._sample_rate = sample_rate
        self._stream: Any | None = None
        self._chunks: list[np.ndarray] = []
        self._status_errors: list[str] = []
        self._lock = threading.Lock()
        self._accepting_audio = False
        self._sealed = False

    def start(self) -> None:
        if self._stream is not None or self._sealed:
            raise RuntimeError("A recording is already in progress")
        self._chunks = []
        self._status_errors = []
        self._accepting_audio = True
        stream = None
        try:
            stream = self._sounddevice.InputStream(
                device=None,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=0,
                latency="high",
                callback=self._on_audio,
            )
            stream.start()
        except Exception as exc:
            self._accepting_audio = False
            if stream is not None:
                try:
                    stream.close()
                except Exception as close_exc:
                    raise RuntimeError(
                        "Cannot open the current default microphone and cannot "
                        f"close its partial stream: {close_exc}"
                    ) from exc
            raise RuntimeError(
                f"Cannot open the current default microphone: {exc}"
            ) from exc
        self._stream = stream

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del frames, time_info
        chunk = indata.copy()
        with self._lock:
            if not self._accepting_audio:
                return
            if status:
                self._status_errors.append(str(status))
            self._chunks.append(chunk)

    def seal(self) -> None:
        if self._stream is None:
            if self._sealed:
                return
            raise RuntimeError("No recording is in progress")
        stream = self._stream
        self._stream = None
        with self._lock:
            self._accepting_audio = False
        try:
            stream.stop()
        finally:
            stream.close()
        self._sealed = True

    def stop(self) -> AudioArtifact:
        if self._stream is not None:
            self.seal()
        elif not self._sealed:
            raise RuntimeError("No recording is in progress")

        with self._lock:
            chunks = list(self._chunks)
            warnings = tuple(self._status_errors)
            self._chunks = []
            self._status_errors = []
            self._sealed = False

        samples = self._join_chunks(chunks)
        return AudioArtifact(samples, self._sample_rate, warnings)

    def snapshot(self) -> AudioArtifact:
        if self._stream is None:
            raise RuntimeError("No recording is in progress")
        with self._lock:
            chunks = list(self._chunks)
        samples = self._join_chunks(chunks)
        return AudioArtifact(samples, self._sample_rate)

    @staticmethod
    def _join_chunks(chunks: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(chunks, axis=0).reshape(-1)
            if chunks
            else np.empty(0, dtype=np.float32)
        )

    def slice_from(
        self, artifact: AudioArtifact, start_frame: int
    ) -> AudioArtifact:
        if start_frame < 0 or start_frame >= artifact.frame_count:
            raise ValueError(
                "start_frame must reference an existing audio frame"
            )

        return AudioArtifact(
            samples=artifact.samples[start_frame:],
            sample_rate=artifact.sample_rate,
            warnings=artifact.warnings,
        )

    def cancel(self) -> None:
        with self._lock:
            self._accepting_audio = False
        if self._stream is not None:
            stream = self._stream
            self._stream = None
            stream.stop()
            stream.close()
        with self._lock:
            self._chunks = []
            self._status_errors = []
            self._sealed = False

    @staticmethod
    def discard(artifact: AudioArtifact) -> None:
        del artifact
