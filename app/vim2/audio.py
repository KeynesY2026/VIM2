from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InputDevice:
    index: int
    name: str
    host_api: str
    is_default: bool = False

    @property
    def key(self) -> tuple[str, str]:
        # PortAudio indices can change after re-enumeration.
        return self.host_api, self.name


@dataclass(frozen=True, slots=True, eq=False)
class AudioArtifact:
    samples: np.ndarray
    sample_rate: int
    warnings: tuple[str, ...] = ()
    start_frame: int = 0

    @property
    def frame_count(self) -> int:
        return len(self.samples)

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.frame_count

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


class AudioRecorder:
    def __init__(
        self,
        *,
        sounddevice: Any | None = None,
        sample_rate: int = 16_000,
        default_input_id: Callable[[], bytes] | None = None,
    ) -> None:
        if sounddevice is None:
            import sounddevice

            sounddevice = sounddevice
        self._sounddevice = sounddevice
        self._sample_rate = sample_rate
        self._stream: Any | None = None
        self._chunks: list[np.ndarray] = []
        self._captured_frames = 0
        self._status_errors: list[str] = []
        self._lock = threading.Lock()
        self._accepting_audio = False
        self._sealed = False
        self._default_input_id = default_input_id
        self._last_default_id = (
            default_input_id() if default_input_id is not None else None
        )
        self._selected_input: InputDevice | None = None
        self._current_input: InputDevice | None = None
        self._inputs: tuple[InputDevice, ...] = ()

    @property
    def current_input(self) -> InputDevice | None:
        return self._current_input

    @property
    def follows_default(self) -> bool:
        return self._selected_input is None

    @property
    def input_devices_cached(self) -> tuple[InputDevice, ...]:
        # WASAPI enumerates active Windows endpoints, without legacy API aliases.
        return tuple(
            device for device in self._inputs
            if device.host_api == "Windows WASAPI"
        )

    def _sync_default_input(self) -> bool:
        if self._default_input_id is None:
            return False
        current_id = self._default_input_id()
        if current_id == self._last_default_id:
            return False
        self._reload_devices()
        self._selected_input = None
        self._last_default_id = current_id
        return True

    def _read_inputs(self) -> None:
        apis = self._sounddevice.query_hostapis()
        self._inputs = tuple(
            InputDevice(
                index,
                device["name"],
                apis[device["hostapi"]]["name"],
                is_default=index == apis[device["hostapi"]]["default_input_device"],
            )
            for index, device in enumerate(self._sounddevice.query_devices())
            if device["max_input_channels"] > 0
        )

    def _resolve_input(self) -> InputDevice:
        self._read_inputs()
        if self._selected_input is None:
            device = self._sounddevice.query_devices(kind="input")
            apis = self._sounddevice.query_hostapis()
            key = (apis[device["hostapi"]]["name"], device["name"])
        else:
            key = self._selected_input.key
        matches = [device for device in self._inputs if device.key == key]
        if not matches or (self._selected_input is not None and len(matches) != 1):
            raise RuntimeError(
                "The selected microphone is unavailable or ambiguous; "
                "select an input device from the tray menu"
            )
        return matches[0]

    def input_devices(self) -> tuple[InputDevice, ...]:
        if self._stream is not None or self._sealed:
            return self.input_devices_cached
        portaudio_error = getattr(self._sounddevice, "PortAudioError", RuntimeError)
        try:
            if not self._sync_default_input():
                self._reload_devices()
            try:
                self._current_input = self._resolve_input()
            except (OSError, RuntimeError, portaudio_error) as exc:
                self._current_input = None
                logger.warning("Cannot resolve the microphone: %s", exc)
            return self.input_devices_cached
        except (OSError, RuntimeError, portaudio_error) as exc:
            self._current_input = None
            logger.exception("Cannot enumerate microphones")
            raise RuntimeError(f"Cannot enumerate microphones: {exc}") from exc

    def select_input(self, device: InputDevice | None) -> None:
        if self._stream is not None or self._sealed:
            raise RuntimeError("Cannot switch microphone during a recording")
        previous = self._selected_input
        portaudio_error = getattr(self._sounddevice, "PortAudioError", RuntimeError)
        try:
            self._sync_default_input()
            previous = self._selected_input
            self._selected_input = device
            current = self._resolve_input()
        except (OSError, RuntimeError, portaudio_error) as exc:
            self._selected_input = previous
            logger.exception("Cannot select microphone")
            raise RuntimeError(f"Cannot select microphone: {exc}") from exc
        self._current_input = current

    def start(self) -> None:
        if self._stream is not None or self._sealed:
            raise RuntimeError("A recording is already in progress")
        self._chunks = []
        self._captured_frames = 0
        self._status_errors = []
        self._accepting_audio = True
        portaudio_error = getattr(
            self._sounddevice, "PortAudioError", RuntimeError
        )
        try:
            self._sync_default_input()
            stream = self._open_input()
        except (OSError, RuntimeError, portaudio_error) as initial_error:
            # A microphone plugged in after VIM2 started stays invisible to
            # PortAudio, so re-enumerate devices once and retry before
            # giving up.
            logger.info(
                "Could not open the microphone, reloading devices: %s",
                initial_error,
            )
            try:
                self._reload_devices()
                self._sync_default_input()
                stream = self._open_input()
            except (OSError, RuntimeError, portaudio_error) as exc:
                self._accepting_audio = False
                self._current_input = None
                description = (
                    "the current Windows default microphone"
                    if self.follows_default
                    else "the selected microphone"
                )
                logger.exception(
                    "Cannot open %s: %s",
                    description,
                    exc,
                )
                raise RuntimeError(
                    f"Cannot open {description}: {exc}"
                ) from exc
        self._stream = stream

    def _open_input(self):
        """Resolve the current selection against the latest device indices."""
        device = self._resolve_input()
        extra_settings = (
            self._sounddevice.WasapiSettings(auto_convert=True)
            if device.host_api == "Windows WASAPI"
            else None
        )
        stream = self._sounddevice.InputStream(
            device=None if self.follows_default else device.index,
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=0,
            latency="high",
            extra_settings=extra_settings,
            callback=self._on_audio,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._current_input = device
        return stream

    def _reload_devices(self) -> None:
        """Re-enumerate devices so the current default input device is used.

        PortAudio resolves its device list and default input device only
        when it is initialized, so a microphone plugged in after VIM2
        started stays invisible until it is initialized again. Must only
        run while no stream is open, because terminating PortAudio closes
        every open stream.
        """
        terminate = getattr(self._sounddevice, "_terminate", None)
        initialize = getattr(self._sounddevice, "_initialize", None)
        if terminate is None or initialize is None:
            return
        terminate()
        initialize()

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
            self._captured_frames += len(chunk)

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
            self._captured_frames = 0
            self._status_errors = []
            self._sealed = False

        samples = self._join_chunks(chunks)
        return AudioArtifact(samples, self._sample_rate, warnings)

    def snapshot(self, max_seconds: int | None = None) -> AudioArtifact:
        if self._stream is None:
            raise RuntimeError("No recording is in progress")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        with self._lock:
            end_frame = self._captured_frames
            if max_seconds is None:
                chunks = list(self._chunks)
                requested_frames = end_frame
            else:
                requested_frames = min(
                    end_frame, max_seconds * self._sample_rate
                )
                chunks = []
                selected_frames = 0
                for chunk in reversed(self._chunks):
                    chunks.append(chunk)
                    selected_frames += len(chunk)
                    if selected_frames >= requested_frames:
                        break
                chunks.reverse()
        samples = self._join_chunks(chunks)
        if len(samples) > requested_frames:
            samples = samples[-requested_frames:]
        return AudioArtifact(
            samples,
            self._sample_rate,
            start_frame=end_frame - len(samples),
        )

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
            start_frame=artifact.start_frame + start_frame,
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
            self._captured_frames = 0
            self._status_errors = []
            self._sealed = False

    @staticmethod
    def discard(artifact: AudioArtifact) -> None:
        del artifact
