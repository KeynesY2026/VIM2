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
        self.stop_error: Exception | None = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if self.stop_error:
            raise self.stop_error

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
    recorder = AudioRecorder(sounddevice=backend)

    recorder.start()

    assert backend.stream is not None
    assert backend.stream.kwargs["device"] is None
    assert backend.stream.kwargs["samplerate"] == 16_000
    assert backend.stream.kwargs["channels"] == 1
    assert backend.stream.kwargs["latency"] == "high"


def test_stop_returns_audio_in_memory_without_creating_wav(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.array([[0.25], [-0.25]], dtype=np.float32), 2, None, None)

    artifact = recorder.stop()

    np.testing.assert_array_equal(
        artifact.samples, np.array([0.25, -0.25], dtype=np.float32)
    )
    assert artifact.frame_count == 2
    assert artifact.duration_seconds == 2 / 16_000
    assert list(tmp_path.iterdir()) == []


def test_input_overflow_is_preserved_as_warning_without_losing_audio(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.zeros((1, 1), dtype=np.float32), 1, None, "input overflow")

    artifact = recorder.stop()

    np.testing.assert_array_equal(
        artifact.samples, np.zeros(1, dtype=np.float32)
    )
    assert artifact.warnings == ("input overflow",)


def test_seal_stops_stream_and_rejects_later_audio_callbacks(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.array([[0.1]], dtype=np.float32), 1, None, None)

    recorder.seal()
    callback(np.array([[0.9]], dtype=np.float32), 1, None, None)
    artifact = recorder.stop()

    assert backend.stream.stopped
    np.testing.assert_array_equal(
        artifact.samples, np.array([0.1], dtype=np.float32)
    )
    assert list(tmp_path.iterdir()) == []


def test_seal_closes_stream_when_stopping_fails() -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    backend.stream.stop_error = RuntimeError("stop failed")

    with pytest.raises(RuntimeError, match="stop failed"):
        recorder.seal()

    assert backend.stream.closed


def test_snapshot_keeps_audio_in_memory_without_stopping_stream(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)

    snapshot = recorder.snapshot()

    np.testing.assert_array_equal(
        snapshot.samples, np.array([0.1, 0.2], dtype=np.float32)
    )
    assert snapshot.frame_count == 2
    assert not backend.stream.stopped
    assert list(tmp_path.iterdir()) == []
    recorder.cancel()


def test_snapshot_does_not_hold_capture_lock_during_audio_concatenation(
    tmp_path: Path,
) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
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

    recorder.snapshot()

    recorder.cancel()


def test_slice_from_returns_requested_audio_suffix_in_memory(
    tmp_path: Path,
) -> None:
    recorder = AudioRecorder(sounddevice=FakeSoundDevice(), sample_rate=4)
    complete = AudioArtifact(
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        sample_rate=4,
        warnings=("input overflow",),
    )

    tail = recorder.slice_from(complete, start_frame=2)

    np.testing.assert_array_equal(
        tail.samples, np.array([0.3, 0.4], dtype=np.float32)
    )
    assert tail.frame_count == 2
    assert tail.sample_rate == 4
    assert tail.warnings == complete.warnings
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("start_frame", [-1, 1])
def test_slice_from_rejects_frame_outside_recording(
    tmp_path: Path, start_frame: int
) -> None:
    recorder = AudioRecorder(sounddevice=FakeSoundDevice())
    artifact = AudioArtifact(np.zeros(1, dtype=np.float32), 16_000)

    with pytest.raises(ValueError, match="existing audio frame"):
        recorder.slice_from(artifact, start_frame)
