import wave
from pathlib import Path

import numpy as np
import pytest

from vim2.audio import AudioArtifact, AudioRecorder


class FakeInputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice:
    def __init__(self) -> None:
        self.stream: FakeInputStream | None = None

    def InputStream(self, **kwargs) -> FakeInputStream:
        self.stream = FakeInputStream(**kwargs)
        return self.stream


def test_each_recording_uses_current_default_input_device(tmp_path: Path) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(tmp_path, sounddevice=backend)

    recorder.start()

    assert backend.stream is not None
    assert backend.stream.kwargs["device"] is None
    assert backend.stream.kwargs["samplerate"] == 16_000
    assert backend.stream.kwargs["channels"] == 1
    assert backend.stream.kwargs["latency"] == "high"


def test_stop_writes_pcm_wav_and_cancel_removes_temporary_audio(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(tmp_path, sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.array([[0.25], [-0.25]], dtype=np.float32), 2, None, None)

    artifact = recorder.stop()

    assert artifact.path.is_file()
    assert artifact.frame_count == 2
    assert artifact.duration_seconds == 2 / 16_000

    recorder.discard(artifact)
    assert not artifact.path.exists()


def test_input_overflow_is_preserved_as_warning_without_losing_audio(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(tmp_path, sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.zeros((1, 1), dtype=np.float32), 1, None, "input overflow")

    artifact = recorder.stop()

    assert artifact.path.is_file()
    assert artifact.warnings == ("input overflow",)


def test_snapshot_writes_audio_without_stopping_active_stream(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(tmp_path, sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)

    snapshot = recorder.snapshot()

    assert snapshot.path.is_file()
    assert snapshot.frame_count == 2
    assert not backend.stream.stopped
    recorder.discard(snapshot)
    recorder.cancel()


def test_snapshot_does_not_hold_capture_lock_during_audio_concatenation(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(tmp_path, sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.array([[0.1]], dtype=np.float32), 1, None, None)
    original_join = recorder._join_chunks

    def inspect_lock(chunks):
        acquired = recorder._lock.acquire(blocking=False)
        if acquired:
            recorder._lock.release()
        assert acquired, "snapshot held the capture lock during concatenation"
        return original_join(chunks)

    recorder._join_chunks = inspect_lock
    snapshot = recorder.snapshot()

    recorder.discard(snapshot)
    recorder.cancel()


def test_slice_from_writes_requested_wav_suffix(tmp_path: Path) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(tmp_path, sounddevice=backend, sample_rate=4)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(
        np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
        4,
        None,
        None,
    )
    complete = recorder.stop()

    tail = recorder.slice_from(complete, start_frame=2)

    assert complete.path.exists()
    assert tail.frame_count == 2
    assert tail.sample_rate == 4
    with wave.open(str(tail.path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 4
        assert wav.getnframes() == 2
    recorder.discard(tail)
    recorder.discard(complete)


def test_slice_from_rejects_frame_outside_recording(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "recording.wav"
    with wave.open(str(artifact_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00")
    recorder = AudioRecorder(tmp_path, sounddevice=FakeSoundDevice())
    artifact = AudioArtifact(artifact_path, 1, 16_000)

    with pytest.raises(ValueError, match="existing audio frame"):
        recorder.slice_from(artifact, 1)
