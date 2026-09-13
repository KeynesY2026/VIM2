from pathlib import Path

import numpy as np

from vim2.audio import AudioRecorder


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


def test_audio_callback_error_is_reported_on_stop(tmp_path: Path) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(tmp_path, sounddevice=backend)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.zeros((1, 1), dtype=np.float32), 1, None, "input overflow")

    try:
        recorder.stop()
    except RuntimeError as exc:
        assert "input overflow" in str(exc)
    else:
        raise AssertionError("callback status should fail the recording")


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
