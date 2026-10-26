import logging
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
        self.start_error: Exception | None = None
        self.stop_error: Exception | None = None

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
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
        self.devices = [
            {"name": "Mic A", "hostapi": 0, "max_input_channels": 1},
            {"name": "Mic B", "hostapi": 0, "max_input_channels": 1},
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0},
        ]
        self.default_index = 0

    def query_devices(self, device=None, kind=None):
        if kind == "input":
            return self.devices[self.default_index]
        if device is not None:
            return self.devices[device]
        return self.devices

    def query_hostapis(self):
        return [
            {"name": "Windows WASAPI", "default_input_device": self.default_index}
        ]

    @staticmethod
    def WasapiSettings(**kwargs):
        return kwargs

    def InputStream(self, **kwargs) -> FakeInputStream:
        self.stream = FakeInputStream(**kwargs)
        return self.stream


class ReloadableSoundDevice(FakeSoundDevice):
    def __init__(
        self,
        *,
        fail_opens: int = 0,
        fail_starts: int = 0,
        fail_initialize: Exception | None = None,
    ) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.fail_opens = fail_opens
        self.fail_starts = fail_starts
        self.fail_initialize = fail_initialize
        self.streams: list[FakeInputStream] = []

    def _terminate(self) -> None:
        self.calls.append("terminate")

    def _initialize(self) -> None:
        self.calls.append("initialize")
        if self.fail_initialize is not None:
            raise self.fail_initialize

    def InputStream(self, **kwargs) -> FakeInputStream:
        self.calls.append("InputStream")
        if self.fail_opens > 0:
            self.fail_opens -= 1
            raise RuntimeError("Error opening InputStream")
        stream = super().InputStream(**kwargs)
        if self.fail_starts > 0:
            self.fail_starts -= 1
            stream.start_error = RuntimeError("Error starting InputStream")
        self.streams.append(stream)
        return stream


def test_each_recording_uses_current_default_input_device(tmp_path: Path) -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)

    recorder.start()

    assert backend.stream is not None
    assert backend.stream.kwargs["device"] is None
    assert backend.stream.kwargs["samplerate"] == 16_000
    assert backend.stream.kwargs["channels"] == 1
    assert backend.stream.kwargs["latency"] == "high"


def test_start_opens_stream_without_reloading_devices() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)

    recorder.start()

    assert backend.calls == ["InputStream"]
    recorder.cancel()


def test_choices_only_include_connected_wasapi_inputs(monkeypatch) -> None:
    backend = ReloadableSoundDevice()
    monkeypatch.setattr(
        backend,
        "query_hostapis",
        lambda: [
            {"name": "Windows WASAPI", "default_input_device": 0},
            {"name": "MME", "default_input_device": 3},
            {"name": "Windows DirectSound", "default_input_device": 5},
            {"name": "Windows WDM-KS", "default_input_device": 4},
        ],
    )
    backend.devices.extend([
        {"name": "Mic A", "hostapi": 1, "max_input_channels": 1},
        {"name": "Unplugged microphone", "hostapi": 3, "max_input_channels": 1},
        {"name": "Mic A", "hostapi": 2, "max_input_channels": 1},
    ])
    backend.default_index = 3
    recorder = AudioRecorder(sounddevice=backend)

    devices = recorder.input_devices()

    assert [(device.name, device.host_api) for device in devices] == [
        ("Mic A", "Windows WASAPI"),
        ("Mic B", "Windows WASAPI"),
    ]
    assert recorder.input_devices_cached == devices
    assert [device.name for device in devices if device.is_default] == ["Mic A"]
    recorder.start()
    assert recorder.current_input.name == "Mic A"
    assert recorder.current_input.host_api == "MME"
    assert backend.stream.kwargs["device"] is None
    assert recorder.input_devices() == devices
    recorder.stop()
    recorder.select_input(devices[1])
    recorder.start()
    assert backend.stream.kwargs["device"] == 1
    recorder.cancel()


def test_disconnected_wasapi_input_disappears_on_refresh() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.input_devices()
    backend.devices.pop(1)

    assert [device.name for device in recorder.input_devices()] == ["Mic A"]
    assert [device.name for device in recorder.input_devices_cached] == ["Mic A"]


def test_default_marker_refreshes_independently_of_manual_selection() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.select_input(recorder.input_devices()[1])

    assert [
        device.name for device in recorder.input_devices_cached if device.is_default
    ] == ["Mic A"]
    backend.default_index = 1
    devices = recorder.input_devices()
    assert [device.name for device in devices if device.is_default] == ["Mic B"]


def test_missing_system_default_does_not_mark_an_arbitrary_input() -> None:
    backend = ReloadableSoundDevice()
    backend.default_index = -1
    recorder = AudioRecorder(sounddevice=backend)

    assert not any(device.is_default for device in recorder.input_devices())


def test_default_change_refreshes_even_when_old_microphone_still_opens() -> None:
    backend = ReloadableSoundDevice()
    default_id = [b"a"]
    recorder = AudioRecorder(
        sounddevice=backend, default_input_id=lambda: default_id[0]
    )
    recorder.start()
    recorder.stop()
    default_id[0] = b"b"
    backend.default_index = 1

    recorder.start()

    assert backend.calls == [
        "InputStream", "terminate", "initialize", "InputStream"
    ]
    assert recorder.current_input.name == "Mic B"
    recorder.cancel()


def test_manual_selection_survives_recordings_but_not_system_default_change() -> None:
    backend = ReloadableSoundDevice()
    default_id = [b"a"]
    recorder = AudioRecorder(
        sounddevice=backend, default_input_id=lambda: default_id[0]
    )
    devices = recorder.input_devices()
    assert [device.name for device in devices] == ["Mic A", "Mic B"]
    recorder.select_input(devices[1])
    for _ in range(2):
        recorder.start()
        assert backend.stream.kwargs["device"] == 1
        assert recorder.current_input.name == "Mic B"
        recorder.stop()

    default_id[0] = b"new-default"
    recorder.start()
    assert backend.stream.kwargs["device"] is None
    assert recorder.follows_default
    assert recorder.current_input.name == "Mic A"
    recorder.cancel()


def test_manual_selection_resolves_new_indices_and_never_opens_other_device() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.select_input(recorder.input_devices()[1])
    backend.devices[0], backend.devices[1] = backend.devices[1], backend.devices[0]

    recorder.start()
    assert backend.stream.kwargs["device"] == 0
    recorder.stop()
    backend.devices = [backend.devices[1]]

    with pytest.raises(RuntimeError, match="selected microphone"):
        recorder.start()
    assert recorder.current_input is None


def test_unchanged_default_only_checks_identity_without_reinitializing() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend, default_input_id=lambda: b"a")

    for _ in range(2):
        recorder.start()
        recorder.stop()

    assert backend.calls == ["InputStream", "InputStream"]


def test_ambiguous_manual_device_is_rejected_instead_of_picking_another() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    device = recorder.input_devices()[1]
    backend.devices.append(dict(backend.devices[1]))

    with pytest.raises(RuntimeError, match="ambiguous"):
        recorder.select_input(device)
    assert recorder.follows_default


def test_selection_wraps_native_refresh_error() -> None:
    class NativeAudioError(Exception):
        pass

    backend = ReloadableSoundDevice()
    backend.PortAudioError = NativeAudioError
    default_id = [b"a"]
    recorder = AudioRecorder(
        sounddevice=backend, default_input_id=lambda: default_id[0]
    )
    device = recorder.input_devices()[1]
    default_id[0] = b"b"
    backend.fail_initialize = NativeAudioError("native refresh failed")

    with pytest.raises(RuntimeError, match="native refresh failed"):
        recorder.select_input(device)


def test_wasapi_uses_shared_mode_conversion_for_16khz_recording() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.select_input(recorder.input_devices()[1])

    recorder.start()

    assert backend.stream.kwargs["extra_settings"] == {"auto_convert": True}
    assert backend.stream.kwargs["samplerate"] == 16_000
    recorder.cancel()


def test_cannot_reload_or_select_microphone_during_recording() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    devices = recorder.input_devices()
    recorder.start()
    calls = list(backend.calls)

    assert recorder.input_devices() == devices
    assert backend.calls == calls
    with pytest.raises(RuntimeError, match="recording"):
        recorder.select_input(devices[1])
    recorder.cancel()


def test_follow_default_and_new_recorder_do_not_keep_manual_selection() -> None:
    backend = ReloadableSoundDevice()
    recorder = AudioRecorder(sounddevice=backend)
    recorder.select_input(recorder.input_devices()[1])
    recorder.select_input(None)
    recorder.start()
    assert backend.stream.kwargs["device"] is None
    recorder.stop()
    recorder.select_input(recorder.input_devices()[1])

    fresh_recorder = AudioRecorder(sounddevice=backend)
    fresh_recorder.start()
    assert backend.stream.kwargs["device"] is None
    fresh_recorder.cancel()


def test_start_reloads_devices_and_retries_when_open_fails() -> None:
    backend = ReloadableSoundDevice(fail_opens=1)
    recorder = AudioRecorder(sounddevice=backend)

    recorder.start()

    assert backend.calls == [
        "InputStream",
        "terminate",
        "initialize",
        "InputStream",
    ]
    assert backend.stream is not None
    assert backend.stream.started is True
    recorder.cancel()


def test_start_reports_error_when_retry_also_fails() -> None:
    backend = ReloadableSoundDevice(fail_opens=2)
    recorder = AudioRecorder(sounddevice=backend)

    with pytest.raises(
        RuntimeError,
        match="Cannot open the current Windows default microphone",
    ):
        recorder.start()

    assert backend.calls == [
        "InputStream",
        "terminate",
        "initialize",
        "InputStream",
    ]


def test_start_reports_device_reload_failure_as_microphone_error() -> None:
    backend = ReloadableSoundDevice(
        fail_opens=1, fail_initialize=RuntimeError("portaudio down")
    )
    recorder = AudioRecorder(sounddevice=backend)

    with pytest.raises(
        RuntimeError,
        match="Cannot open the current Windows default microphone",
    ):
        recorder.start()


def test_start_closes_stream_that_fails_to_start() -> None:
    backend = ReloadableSoundDevice(fail_starts=1)
    recorder = AudioRecorder(sounddevice=backend)

    recorder.start()

    assert backend.streams[0].closed is True
    assert backend.stream is backend.streams[1]
    recorder.cancel()


def test_start_logs_portaudio_error_that_triggered_the_retry(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logging.getLogger("vim2"), "propagate", True)
    backend = ReloadableSoundDevice(fail_opens=1)
    recorder = AudioRecorder(sounddevice=backend)

    with caplog.at_level(logging.INFO):
        recorder.start()

    assert "Error opening InputStream" in caplog.text
    recorder.cancel()


def test_start_logs_portaudio_error_when_retry_also_fails(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logging.getLogger("vim2"), "propagate", True)
    backend = ReloadableSoundDevice(fail_opens=2)
    recorder = AudioRecorder(sounddevice=backend)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        recorder.start()

    assert "Error opening InputStream" in caplog.text


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


def test_audio_artifact_tracks_absolute_frame_bounds() -> None:
    artifact = AudioArtifact(
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        sample_rate=4,
        start_frame=10,
    )

    assert artifact.frame_count == 3
    assert artifact.end_frame == 13


def test_bounded_snapshot_only_joins_requested_recent_chunks() -> None:
    backend = FakeSoundDevice()
    recorder = AudioRecorder(sounddevice=backend, sample_rate=2)
    recorder.start()
    assert backend.stream is not None
    callback = backend.stream.kwargs["callback"]
    callback(np.array([[1.0], [2.0]], dtype=np.float32), 2, None, None)
    callback(np.array([[3.0], [4.0]], dtype=np.float32), 2, None, None)
    callback(np.array([[5.0], [6.0]], dtype=np.float32), 2, None, None)
    joined_frame_counts = []
    original_join = recorder._join_chunks

    def record_joined_chunks(chunks):
        joined_frame_counts.append(sum(len(chunk) for chunk in chunks))
        return original_join(chunks)

    recorder._join_chunks = record_joined_chunks

    snapshot = recorder.snapshot(max_seconds=2)

    np.testing.assert_array_equal(
        snapshot.samples,
        np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    )
    assert snapshot.start_frame == 2
    assert snapshot.end_frame == 6
    assert joined_frame_counts == [4]
    recorder.cancel()


@pytest.mark.parametrize("max_seconds", [0, -1])
def test_snapshot_rejects_non_positive_bound(max_seconds: int) -> None:
    recorder = AudioRecorder(sounddevice=FakeSoundDevice())
    recorder.start()

    with pytest.raises(ValueError, match="max_seconds"):
        recorder.snapshot(max_seconds=max_seconds)

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
    assert tail.start_frame == 2
    assert tail.end_frame == 4
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
