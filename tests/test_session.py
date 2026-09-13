from pathlib import Path

import pytest

from vim2.audio import AudioArtifact
from vim2.models import ModelId
from vim2.session import FinalRecognitionError, VoiceSession
from vim2.state import AppState, StateMachine


class FakeRecorder:
    def __init__(self, artifact: AudioArtifact) -> None:
        self.artifact = artifact
        self.started = False
        self.discarded: list[AudioArtifact] = []
        self.stop_error: Exception | None = None
        self.snapshots: list[AudioArtifact] = []
        self.slices: list[tuple[AudioArtifact, int, AudioArtifact]] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> AudioArtifact:
        if self.stop_error:
            raise self.stop_error
        return self.artifact

    def snapshot(self) -> AudioArtifact:
        frame_count = (
            self.snapshots[-1].frame_count + 16_000
            if self.snapshots
            else 160_000
        )
        snapshot = AudioArtifact(
            path=self.artifact.path.with_name(
                f"preview-{len(self.snapshots)}.wav"
            ),
            frame_count=frame_count,
            sample_rate=self.artifact.sample_rate,
        )
        snapshot.path.write_bytes(b"preview")
        self.snapshots.append(snapshot)
        return snapshot

    def slice_from(
        self, artifact: AudioArtifact, start_frame: int
    ) -> AudioArtifact:
        tail = AudioArtifact(
            path=artifact.path.with_name("tail.wav"),
            frame_count=artifact.frame_count - start_frame,
            sample_rate=artifact.sample_rate,
            warnings=artifact.warnings,
        )
        tail.path.write_bytes(b"tail")
        self.slices.append((artifact, start_frame, tail))
        return tail

    def cancel(self) -> None:
        self.started = False

    def discard(self, artifact: AudioArtifact) -> None:
        artifact.path.unlink(missing_ok=True)
        self.discarded.append(artifact)


class FakeRecognizer:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[Path, ModelId]] = []

    def transcribe(self, path: Path, model_id: ModelId) -> str:
        self.calls.append((path, model_id))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakePaster:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def paste(self, text: str, target_window: int) -> None:
        self.calls.append((text, target_window))


def _ready_machine() -> StateMachine:
    machine = StateMachine()
    machine.transition_to(AppState.MODEL_LOADING)
    machine.transition_to(AppState.READY)
    return machine


def _artifact(tmp_path: Path) -> AudioArtifact:
    path = tmp_path / "recording.wav"
    path.write_bytes(b"audio")
    return AudioArtifact(path=path, frame_count=16_000, sample_rate=16_000)


def _long_artifact(tmp_path: Path) -> AudioArtifact:
    path = tmp_path / "recording.wav"
    path.write_bytes(b"audio")
    return AudioArtifact(path=path, frame_count=960_000, sample_rate=16_000)


def test_final_recognition_uses_complete_recording_then_pastes_and_deletes(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recognizer = FakeRecognizer(["最终文本"])
    paster = FakePaster()
    session = VoiceSession(_ready_machine(), recorder, recognizer, paster)

    session.start(target_window=42, model_id=ModelId.FAST)
    result = session.stop()

    assert result == "最终文本"
    assert recognizer.calls == [(artifact.path, ModelId.FAST)]
    assert paster.calls == [("最终文本", 42)]
    assert recorder.discarded == [artifact]
    assert session.state is AppState.READY


def test_empty_result_does_not_touch_clipboard(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    paster = FakePaster()
    session = VoiceSession(
        _ready_machine(),
        FakeRecorder(artifact),
        FakeRecognizer(["  "]),
        paster,
    )

    session.start(target_window=42, model_id=ModelId.FAST)

    assert session.stop() == ""
    assert paster.calls == []
    assert not artifact.path.exists()


def test_failed_recognition_retains_audio_and_retry_uses_original_model(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recognizer = FakeRecognizer([RuntimeError("CUDA OOM"), "retry result"])
    session = VoiceSession(
        _ready_machine(), recorder, recognizer, FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.ACCURATE)

    with pytest.raises(FinalRecognitionError, match="CUDA OOM"):
        session.stop()

    assert artifact.path.exists()
    assert session.state is AppState.RETRY_PENDING

    assert session.retry() == "retry result"
    assert recognizer.calls == [
        (artifact.path, ModelId.ACCURATE),
        (artifact.path, ModelId.ACCURATE),
    ]
    assert not artifact.path.exists()
    assert session.state is AppState.READY


def test_cancel_during_recording_never_recognizes_or_pastes(
    tmp_path: Path,
) -> None:
    recorder = FakeRecorder(_artifact(tmp_path))
    recognizer = FakeRecognizer(["unexpected"])
    paster = FakePaster()
    session = VoiceSession(_ready_machine(), recorder, recognizer, paster)
    session.start(target_window=7, model_id=ModelId.FAST)

    session.cancel()

    assert recognizer.calls == []
    assert paster.calls == []
    assert session.state is AppState.READY


def test_live_preview_uses_selected_model_and_keeps_recording(
    tmp_path: Path,
) -> None:
    recorder = FakeRecorder(_artifact(tmp_path))
    recognizer = FakeRecognizer(["临时文本"])
    session = VoiceSession(
        _ready_machine(), recorder, recognizer, FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.ACCURATE)

    preview = session.preview()

    assert preview == "临时文本"
    assert recognizer.calls[0][1] is ModelId.ACCURATE
    assert not recognizer.calls[0][0].exists()
    assert session.state is AppState.RECORDING


def test_stop_after_live_preview_transcribes_complete_recording(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    recognizer = FakeRecognizer(["前半句", "前半句和最后一段"])
    paster = FakePaster()
    session = VoiceSession(
        _ready_machine(), FakeRecorder(artifact), recognizer, paster
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    assert session.preview() == "前半句"

    result = session.stop()

    assert result == "前半句和最后一段"
    assert recognizer.calls[-1] == (artifact.path, ModelId.FAST)
    assert len(recognizer.calls) == 2
    assert paster.calls == [("前半句和最后一段", 7)]


def test_stable_prefix_uses_overlapped_tail_for_final_result(
    tmp_path: Path,
) -> None:
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(tmp_path / "seed.wav", 640_000, 16_000)
    ]
    recognizer = FakeRecognizer(
        ["第一句。未完成", "第一句。变化", "第一句。继续", "第一句。最后一段。"]
    )
    paster = FakePaster()
    session = VoiceSession(_ready_machine(), recorder, recognizer, paster)
    session.start(target_window=7, model_id=ModelId.FAST)

    assert session.preview() == "第一句。未完成"
    assert session.preview() == "第一句。变化"
    assert session.preview() == "第一句。继续"
    result = session.stop()

    assert result == "第一句。最后一段。"
    complete, start_frame, tail = recorder.slices[0]
    assert complete == artifact
    assert start_frame == 560_000
    assert recognizer.calls[-1] == (tail.path, ModelId.FAST)
    assert paster.calls == [("第一句。最后一段。", 7)]
    assert not tail.path.exists()
    assert not artifact.path.exists()


@pytest.mark.parametrize(
    "tail_response",
    ["没有锚点。", "第一句。中间。第一句。结尾。", "   "],
)
def test_unusable_tail_falls_back_to_complete_recording(
    tmp_path: Path, tail_response: str
) -> None:
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(tmp_path / "seed.wav", 640_000, 16_000)
    ]
    recognizer = FakeRecognizer(
        [
            "第一句。未完成",
            "第一句。变化",
            "第一句。继续",
            tail_response,
            "第一句。完整尾段。",
        ]
    )
    paster = FakePaster()
    session = VoiceSession(_ready_machine(), recorder, recognizer, paster)
    session.start(target_window=7, model_id=ModelId.FAST)
    for _ in range(3):
        session.preview()

    result = session.stop()

    tail = recorder.slices[0][2]
    assert result == "第一句。完整尾段。"
    assert recognizer.calls[-2:] == [
        (tail.path, ModelId.FAST),
        (artifact.path, ModelId.FAST),
    ]
    assert paster.calls == [("第一句。完整尾段。", 7)]
    assert not tail.path.exists()
    assert not artifact.path.exists()


def test_tail_over_ratio_limit_uses_complete_recording(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording.wav"
    path.write_bytes(b"audio")
    artifact = AudioArtifact(path, 320_000, 16_000)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(tmp_path / "seed.wav", 160_000, 16_000)
    ]
    recognizer = FakeRecognizer(
        ["第一句。未完成", "第一句。变化", "第一句。继续", "完整结果。"]
    )
    session = VoiceSession(
        _ready_machine(), recorder, recognizer, FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    for _ in range(3):
        session.preview()

    assert session.stop() == "完整结果。"
    assert recorder.slices == []
    assert recognizer.calls[-1] == (artifact.path, ModelId.FAST)


def test_stop_falls_back_to_full_audio_when_live_result_is_empty(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    recognizer = FakeRecognizer(["   ", "最终结果"])
    paster = FakePaster()
    session = VoiceSession(
        _ready_machine(), FakeRecorder(artifact), recognizer, paster
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    assert session.preview() == ""

    result = session.stop()

    assert result == "最终结果"
    assert len(recognizer.calls) == 2
    assert paster.calls == [("最终结果", 7)]


def test_live_preview_failure_does_not_stop_recording(tmp_path: Path) -> None:
    recorder = FakeRecorder(_artifact(tmp_path))
    session = VoiceSession(
        _ready_machine(),
        recorder,
        FakeRecognizer([RuntimeError("temporary failure")]),
        FakePaster(),
    )
    session.start(target_window=7, model_id=ModelId.FAST)

    with pytest.raises(RuntimeError, match="temporary failure"):
        session.preview()

    assert session.state is AppState.RECORDING


def test_microphone_stop_failure_returns_session_to_ready(
    tmp_path: Path,
) -> None:
    recorder = FakeRecorder(_artifact(tmp_path))
    recorder.stop_error = RuntimeError("input overflow")
    session = VoiceSession(
        _ready_machine(), recorder, FakeRecognizer([]), FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.FAST)

    with pytest.raises(RuntimeError, match="input overflow"):
        session.stop()

    assert session.state is AppState.READY


def test_shutdown_during_live_preview_stops_capture_without_state_race(
    tmp_path: Path,
) -> None:
    recorder = FakeRecorder(_artifact(tmp_path))
    session = VoiceSession(
        _ready_machine(), recorder, FakeRecognizer([]), FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    session._machine.transition_to(AppState.LIVE_TRANSCRIBING)

    session.shutdown()

    assert not recorder.started
    assert session.state is AppState.LIVE_TRANSCRIBING
