from pathlib import Path
import threading

import numpy as np
import pytest

from vim2.audio import AudioArtifact
from vim2.models import ModelId
from vim2.session import (
    MAX_PREVIEW_SECONDS,
    FinalRecognitionError,
    VoiceSession,
)
from vim2.state import AppState, StateMachine


class FakeRecorder:
    def __init__(self, artifact: AudioArtifact) -> None:
        self.artifact = artifact
        self.started = False
        self.discarded: list[AudioArtifact] = []
        self.stop_error: Exception | None = None
        self.seal_error: Exception | None = None
        self.slice_error: Exception | None = None
        self.snapshots: list[AudioArtifact] = []
        self.slices: list[tuple[AudioArtifact, int, AudioArtifact]] = []
        self.seal_calls = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> AudioArtifact:
        if self.stop_error:
            raise self.stop_error
        return self.artifact

    def seal(self) -> None:
        if self.seal_error:
            raise self.seal_error
        self.seal_calls += 1
        self.started = False

    def snapshot(self) -> AudioArtifact:
        frame_count = (
            self.snapshots[-1].frame_count + 16_000
            if self.snapshots
            else 160_000
        )
        snapshot = AudioArtifact(
            samples=np.zeros(frame_count, dtype=np.float32),
            sample_rate=self.artifact.sample_rate,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def slice_from(
        self, artifact: AudioArtifact, start_frame: int
    ) -> AudioArtifact:
        if self.slice_error:
            raise self.slice_error
        tail = AudioArtifact(
            samples=artifact.samples[start_frame:],
            sample_rate=artifact.sample_rate,
            warnings=artifact.warnings,
        )
        self.slices.append((artifact, start_frame, tail))
        return tail

    def cancel(self) -> None:
        self.started = False

    def discard(self, artifact: AudioArtifact) -> None:
        self.discarded.append(artifact)


class FakeRecognizer:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[AudioArtifact, ModelId]] = []

    def transcribe(
        self,
        artifact: AudioArtifact,
        model_id: ModelId,
        *,
        cancel_event: threading.Event | None = None,
    ) -> str:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        self.calls.append((artifact, model_id))
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


def _artifact(tmp_path) -> AudioArtifact:
    del tmp_path
    return AudioArtifact(np.zeros(16_000, dtype=np.float32), 16_000)


def _long_artifact(tmp_path) -> AudioArtifact:
    del tmp_path
    return AudioArtifact(np.zeros(960_000, dtype=np.float32), 16_000)


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
    assert recognizer.calls == [(artifact, ModelId.FAST)]
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
    assert session.state is AppState.READY


def test_failed_recognition_retains_audio_and_retry_uses_original_model(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recognizer = FakeRecognizer([RuntimeError("CUDA OOM"), "retry result"])
    session = VoiceSession(
        _ready_machine(),
        recorder,
        recognizer,
        FakePaster(),
        tail_overlap_seconds=5,
    )
    session.start(target_window=7, model_id=ModelId.ACCURATE)

    with pytest.raises(FinalRecognitionError, match="CUDA OOM"):
        session.stop()

    assert artifact not in recorder.discarded
    assert session.state is AppState.RETRY_PENDING

    assert session.retry() == "retry result"
    assert recognizer.calls == [
        (artifact, ModelId.ACCURATE),
        (artifact, ModelId.ACCURATE),
    ]
    assert artifact in recorder.discarded
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
        _ready_machine(),
        recorder,
        recognizer,
        FakePaster(),
        tail_overlap_seconds=5,
    )
    session.start(target_window=7, model_id=ModelId.ACCURATE)

    preview = session.preview()

    assert preview == "临时文本"
    assert recognizer.calls[0][1] is ModelId.ACCURATE
    assert recognizer.calls[0][0] in recorder.discarded
    assert session.state is AppState.RECORDING


def test_live_preview_uses_stable_checkpoint_tail_before_window_limit(
    tmp_path: Path,
) -> None:
    recorder = FakeRecorder(_long_artifact(tmp_path))
    recorder.snapshots = [
        AudioArtifact(np.zeros(144_000, dtype=np.float32), 16_000)
    ]
    recognizer = FakeRecognizer(
        [
            "第一句。第二句。第三句还没说完",
            "第一句。第二句。第三句。第四句还没说完",
        ]
    )
    session = VoiceSession(
        _ready_machine(), recorder, recognizer, FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.ACCURATE)

    assert session.preview() == "第一句。第二句。第三句还没说完"
    assert session.preview() == "第一句。第二句。第三句。第四句还没说完"

    complete, start_frame, tail = recorder.slices[-1]
    assert complete.frame_count == 176_000
    assert start_frame == 80_000
    assert tail.frame_count == 96_000
    assert recognizer.calls[-1] == (tail, ModelId.ACCURATE)


def test_long_live_preview_only_transcribes_bounded_recent_audio(
    tmp_path: Path,
) -> None:
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(640_000, dtype=np.float32), 16_000)
    ]
    recognizer = FakeRecognizer(["最近的文本", "完整文本"])
    session = VoiceSession(
        _ready_machine(), recorder, recognizer, FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.ACCURATE)

    assert session.preview() == "最近的文本"

    complete, start_frame, window = recorder.slices[0]
    assert complete.frame_count == 656_000
    assert start_frame == complete.frame_count - (
        complete.sample_rate * MAX_PREVIEW_SECONDS
    )
    assert window.duration_seconds == MAX_PREVIEW_SECONDS
    assert recognizer.calls == [(window, ModelId.ACCURATE)]
    assert window in recorder.discarded
    assert complete in recorder.discarded
    assert session.stop() == "完整文本"
    assert recognizer.calls[-1] == (artifact, ModelId.ACCURATE)


def test_bounded_preview_merges_an_exact_stable_sentence_anchor(
    tmp_path: Path,
) -> None:
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(144_000, dtype=np.float32), 16_000),
    ]
    recognizer = FakeRecognizer(
        [
            "第一句。未完成",
            "第一句。变化",
            "第一句。继续",
            "第一句。第二句。",
        ]
    )
    session = VoiceSession(
        _ready_machine(), recorder, recognizer, FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    for _ in range(3):
        session.preview()

    assert session.preview() == "第一句。第二句。"
    complete, start_frame, tail = recorder.slices[-1]
    assert complete.duration_seconds == 13
    assert start_frame == 112_000
    assert tail.duration_seconds == 6
    assert recognizer.calls[-1][0] is tail


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
    assert recognizer.calls[-1] == (artifact, ModelId.FAST)
    assert len(recognizer.calls) == 2
    assert paster.calls == [("前半句和最后一段", 7)]


def test_stable_prefix_uses_overlapped_tail_for_final_result(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("vim2.session.MAX_PREVIEW_SECONDS", 90)
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(640_000, dtype=np.float32), 16_000)
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
    complete, start_frame, tail = recorder.slices[-1]
    assert complete == artifact
    assert start_frame == 608_000
    assert recognizer.calls[-1] == (tail, ModelId.FAST)
    assert paster.calls == [("第一句。最后一段。", 7)]
    assert tail in recorder.discarded
    assert artifact in recorder.discarded


@pytest.mark.parametrize(
    "tail_response",
    ["没有锚点。", "第一句。中间。第一句。结尾。", "   "],
)
def test_unusable_tail_falls_back_to_complete_recording(
    tmp_path: Path, tail_response: str, monkeypatch
) -> None:
    monkeypatch.setattr("vim2.session.MAX_PREVIEW_SECONDS", 90)
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(640_000, dtype=np.float32), 16_000)
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

    tail = recorder.slices[-1][2]
    assert result == "第一句。完整尾段。"
    assert recognizer.calls[-2:] == [
        (tail, ModelId.FAST),
        (artifact, ModelId.FAST),
    ]
    assert paster.calls == [("第一句。完整尾段。", 7)]
    assert tail in recorder.discarded
    assert artifact in recorder.discarded


def test_tail_recognition_error_falls_back_to_complete_recording(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("vim2.session.MAX_PREVIEW_SECONDS", 90)
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(640_000, dtype=np.float32), 16_000)
    ]
    recognizer = FakeRecognizer(
        [
            "第一句。未完成",
            "第一句。变化",
            "第一句。继续",
            RuntimeError("tail failed"),
            "第一句。完整尾段。",
        ]
    )
    paster = FakePaster()
    session = VoiceSession(_ready_machine(), recorder, recognizer, paster)
    session.start(target_window=7, model_id=ModelId.FAST)
    session.preview()
    session.preview()
    session.preview()

    result = session.stop()

    tail = recorder.slices[-1][2]
    assert result == "第一句。完整尾段。"
    assert recognizer.calls[-2:] == [
        (tail, ModelId.FAST),
        (artifact, ModelId.FAST),
    ]
    assert paster.calls == [("第一句。完整尾段。", 7)]
    assert tail in recorder.discarded
    assert artifact in recorder.discarded


@pytest.mark.parametrize(
    "slice_error",
    [
        OSError("slice failed"),
        RuntimeError("slice failed"),
        ValueError("slice failed"),
        MemoryError("slice failed"),
    ],
)
def test_tail_slice_error_falls_back_to_complete_recording(
    tmp_path: Path, slice_error: Exception, monkeypatch
) -> None:
    monkeypatch.setattr("vim2.session.MAX_PREVIEW_SECONDS", 90)
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(640_000, dtype=np.float32), 16_000)
    ]
    recognizer = FakeRecognizer(
        [
            "第一句。未完成",
            "第一句。变化",
            "第一句。继续",
            "第一句。完整尾段。",
        ]
    )
    paster = FakePaster()
    session = VoiceSession(_ready_machine(), recorder, recognizer, paster)
    session.start(target_window=7, model_id=ModelId.FAST)
    for _ in range(3):
        session.preview()
    recorder.slice_error = slice_error

    result = session.stop()

    assert result == "第一句。完整尾段。"
    assert recognizer.calls[-1] == (artifact, ModelId.FAST)
    assert paster.calls == [("第一句。完整尾段。", 7)]
    assert artifact in recorder.discarded
    assert session.state is AppState.READY


def test_failed_tail_and_full_recognition_retries_complete_recording(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("vim2.session.MAX_PREVIEW_SECONDS", 90)
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(640_000, dtype=np.float32), 16_000)
    ]
    recognizer = FakeRecognizer(
        [
            "第一句。未完成",
            "第一句。变化",
            "第一句。继续",
            RuntimeError("tail failed"),
            RuntimeError("full failed"),
            "第一句。重试成功。",
        ]
    )
    paster = FakePaster()
    session = VoiceSession(_ready_machine(), recorder, recognizer, paster)
    session.start(target_window=7, model_id=ModelId.FAST)
    for _ in range(3):
        session.preview()

    with pytest.raises(FinalRecognitionError, match="full failed"):
        session.stop()

    tail = recorder.slices[-1][2]
    assert tail in recorder.discarded
    assert artifact not in recorder.discarded
    assert session.state is AppState.RETRY_PENDING

    assert session.retry() == "第一句。重试成功。"
    assert recognizer.calls[-1] == (artifact, ModelId.FAST)
    assert paster.calls == [("第一句。重试成功。", 7)]
    assert artifact in recorder.discarded
    assert session.state is AppState.READY


def test_cancel_after_optimized_finalization_failure_cleans_audio(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("vim2.session.MAX_PREVIEW_SECONDS", 90)
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(640_000, dtype=np.float32), 16_000)
    ]
    recognizer = FakeRecognizer(
        [
            "第一句。未完成",
            "第一句。变化",
            "第一句。继续",
            RuntimeError("tail failed"),
            RuntimeError("full failed"),
        ]
    )
    session = VoiceSession(
        _ready_machine(), recorder, recognizer, FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    for _ in range(3):
        session.preview()
    with pytest.raises(FinalRecognitionError, match="full failed"):
        session.stop()

    tail = recorder.slices[-1][2]
    session.cancel()

    assert tail in recorder.discarded
    assert artifact in recorder.discarded
    assert session.state is AppState.READY


def test_shutdown_after_optimized_finalization_failure_cleans_audio(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("vim2.session.MAX_PREVIEW_SECONDS", 90)
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(640_000, dtype=np.float32), 16_000)
    ]
    recognizer = FakeRecognizer(
        [
            "第一句。未完成",
            "第一句。变化",
            "第一句。继续",
            RuntimeError("tail failed"),
            RuntimeError("full failed"),
        ]
    )
    session = VoiceSession(
        _ready_machine(), recorder, recognizer, FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    for _ in range(3):
        session.preview()
    with pytest.raises(FinalRecognitionError, match="full failed"):
        session.stop()

    tail = recorder.slices[-1][2]
    session.shutdown()

    assert tail in recorder.discarded
    assert artifact in recorder.discarded


def test_tail_over_ratio_limit_uses_complete_recording(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("vim2.session.MAX_PREVIEW_SECONDS", 90)
    artifact = AudioArtifact(np.zeros(320_000, dtype=np.float32), 16_000)
    recorder = FakeRecorder(artifact)
    recorder.snapshots = [
        AudioArtifact(np.zeros(112_000, dtype=np.float32), 16_000)
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
    assert all(source is not artifact for source, _, _ in recorder.slices)
    assert recognizer.calls[-1] == (artifact, ModelId.FAST)


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


def test_microphone_seal_failure_during_preview_returns_session_to_ready(
    tmp_path: Path,
) -> None:
    recorder = FakeRecorder(_artifact(tmp_path))
    recorder.seal_error = RuntimeError("cannot stop input")
    session = VoiceSession(
        _ready_machine(), recorder, FakeRecognizer([]), FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    session._machine.transition_to(AppState.LIVE_TRANSCRIBING)

    with pytest.raises(RuntimeError, match="cannot stop input"):
        session.seal()

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
