from __future__ import annotations

import threading
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    path: Path
    frame_count: int
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


class AudioRecorder:
    def __init__(
        self,
        temp_dir: Path,
        *,
        sounddevice: Any | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        if sounddevice is None:
            import sounddevice

            sounddevice = sounddevice
        self._sounddevice = sounddevice
        self._temp_dir = temp_dir
        self._sample_rate = sample_rate
        self._stream: Any | None = None
        self._chunks: list[np.ndarray] = []
        self._status_errors: list[str] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("A recording is already in progress")
        self._chunks = []
        self._status_errors = []
        portaudio_error = getattr(
            self._sounddevice, "PortAudioError", RuntimeError
        )
        try:
            stream = self._sounddevice.InputStream(
                device=None,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._on_audio,
            )
            stream.start()
        except (OSError, RuntimeError, portaudio_error) as exc:
            raise RuntimeError(
                f"Cannot open the current Windows default microphone: {exc}"
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
        with self._lock:
            if status:
                self._status_errors.append(str(status))
            self._chunks.append(indata.copy())

    def stop(self) -> AudioArtifact:
        if self._stream is None:
            raise RuntimeError("No recording is in progress")
        stream = self._stream
        self._stream = None
        stream.stop()
        stream.close()

        with self._lock:
            chunks = self._chunks
            errors = tuple(self._status_errors)
            self._chunks = []
            self._status_errors = []
        if errors:
            raise RuntimeError(f"Microphone capture failed: {'; '.join(errors)}")

        samples = self._join_chunks(chunks)
        return self._write_wav(samples, "recording")

    def snapshot(self) -> AudioArtifact:
        if self._stream is None:
            raise RuntimeError("No recording is in progress")
        with self._lock:
            samples = self._join_chunks(self._chunks)
        return self._write_wav(samples, "preview")

    @staticmethod
    def _join_chunks(chunks: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(chunks, axis=0).reshape(-1)
            if chunks
            else np.empty(0, dtype=np.float32)
        )

    def _write_wav(self, samples: np.ndarray, prefix: str) -> AudioArtifact:
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        path = self._temp_dir / f"{prefix}-{uuid.uuid4().hex}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(pcm.tobytes())
        return AudioArtifact(
            path=path,
            frame_count=len(samples),
            sample_rate=self._sample_rate,
        )

    def cancel(self) -> None:
        if self._stream is None:
            return
        stream = self._stream
        self._stream = None
        stream.stop()
        stream.close()
        with self._lock:
            self._chunks = []
            self._status_errors = []

    @staticmethod
    def discard(artifact: AudioArtifact) -> None:
        artifact.path.unlink(missing_ok=True)
