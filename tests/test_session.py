import io
import json
from pathlib import Path
import threading
import zipfile

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
        self.snapshot_max_seconds: list[int | None] = []
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

    def snapshot(
        self, max_seconds: int | None = None
    ) -> AudioArtifact:
        self.snapshot_max_seconds.append(max_seconds)
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
            start_frame=artifact.start_frame + start_frame,
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

    def debug_context(self) -> dict[str, object]:
        return {"hotwords": ["VIM2"], "draft_tokens": [1, 2, 151645]}

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


class DraftRecognizer(FakeRecognizer):
    supports_live_draft = True

    def __init__(self, responses):
        super().__init__(responses)
        self.resets = 0
        self.preview_calls = 0
        self.final_calls = 0

    def reset_live_draft(self):
        self.resets += 1

    def transcribe_preview(self, artifact, model_id, *, cancel_event=None):
        self.preview_calls += 1
        return self.transcribe(artifact, model_id, cancel_event=cancel_event)

    def transcribe_final(self, artifact, model_id):
        self.final_calls += 1
        return self.transcribe(artifact, model_id)


def test_draft_session_uses_bounded_context_and_verifies_whole_audio_without_seams(tmp_path):
    artifact = _long_artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recognizer = DraftRecognizer(["old.", "corrected final."])
    session = VoiceSession(_ready_machine(), recorder, recognizer, FakePaster())
    session.start(target_window=42, model_id=ModelId.FAST)
    assert session.preview() == "old."
    assert recorder.snapshot_max_seconds == [60]
    assert session.stop() == "corrected final."
    assert recorder.slices == []
    assert recognizer.calls[-1][0] is artifact
    assert recognizer.preview_calls == recognizer.final_calls == 1
    assert recognizer.resets == 2


def test_draft_preview_rests_after_completion_and_stop_never_waits(tmp_path, monkeypatch):
    monkeypatch.setattr("webrtcvad.Vad.is_speech", lambda self, data, rate: any(data))
    now = [0.5]
    monkeypatch.setattr("vim2.session.monotonic", lambda: now[0], raising=False)
    recorder = FakeRecorder(_long_artifact(tmp_path))
    recorder.snapshot = lambda max_seconds=None: AudioArtifact(
        np.full(16_000, 0.1, dtype=np.float32), 16_000,
        start_frame=int(now[0] * 16_000),
    )
    recognizer = DraftRecognizer(["first", "second", "final"])
    session = VoiceSession(_ready_machine(), recorder, recognizer, FakePaster())
    session.start(target_window=42, model_id=ModelId.FAST)
    assert session.preview() == "first"
    now[0] = 5.49
    assert session.preview() is None
    now[0] = 5.5
    assert session.preview() == "second"
    assert session.stop() == "final"
    assert recognizer.preview_calls == 2
    assert recognizer.calls[-1][0] is recorder.artifact


def test_draft_preview_flushes_pause_once_then_skips_silence(tmp_path, monkeypatch):
    monkeypatch.setattr("webrtcvad.Vad.is_speech", lambda self, data, rate: any(data))
    now = [1.0]
    monkeypatch.setattr("vim2.session.monotonic", lambda: now[0], raising=False)
    recorder = FakeRecorder(_long_artifact(tmp_path))
    audio = np.concatenate((
        np.full(16_000, 0.1, dtype=np.float32),
        np.zeros(15 * 16_000, dtype=np.float32),
    ))
    recorder.snapshot = lambda max_seconds=None: AudioArtifact(
        audio[max(0, int(now[0] * 16_000) - 15 * 16_000):int(now[0] * 16_000)],
        16_000,
        start_frame=max(0, int(now[0] * 16_000) - 15 * 16_000),
    )
    recognizer = DraftRecognizer(["first", "sentence end"])
    session = VoiceSession(_ready_machine(), recorder, recognizer, FakePaster())
    session.start(target_window=42, model_id=ModelId.FAST)
    assert session.preview() == "first"
    now[0] = 6.0
    assert session.preview() == "sentence end"
    now[0] = 11.0
    assert session.preview() is None
    now[0] = 16.0
    assert session.preview() is None
    assert recognizer.preview_calls == 2
    assert session.state is AppState.RECORDING


def test_draft_preview_error_is_throttled_but_retry_keeps_full_audio(tmp_path, monkeypatch):
    now = [1.0]
    monkeypatch.setattr("vim2.session.monotonic", lambda: now[0], raising=False)
    recorder = FakeRecorder(_long_artifact(tmp_path))
    recognizer = DraftRecognizer([RuntimeError("preview failed"), "recovered"])
    session = VoiceSession(_ready_machine(), recorder, recognizer, FakePaster())
    session.start(target_window=42, model_id=ModelId.FAST)
    with pytest.raises(RuntimeError, match="preview failed"):
        session.preview()
    now[0] = 2.0
    assert session.preview() is None
    assert session.stop() == "recovered"
    assert recognizer.calls[-1][0] is recorder.artifact


def test_draft_preview_keeps_five_second_cadence_and_rest_after_completion(tmp_path, monkeypatch):
    now = [1.0]
    monkeypatch.setattr("vim2.session.monotonic", lambda: now[0], raising=False)
    monkeypatch.setattr("webrtcvad.Vad.is_speech", lambda self, data, rate: any(data))
    recorder = FakeRecorder(_long_artifact(tmp_path))
    recorder.snapshot = lambda max_seconds=None: AudioArtifact(
        np.full(16_000, 0.1, dtype=np.float32), 16_000,
        start_frame=int(now[0] * 16_000),
    )

    class SlowRecognizer(DraftRecognizer):
        def transcribe_preview(self, *args, **kwargs):
            now[0] += 2.0
            return super().transcribe_preview(*args, **kwargs)

    recognizer = SlowRecognizer(["first", "second"])
    session = VoiceSession(_ready_machine(), recorder, recognizer, FakePaster())
    session.start(target_window=42, model_id=ModelId.FAST)
    assert session.preview() == "first"
    now[0] = 5.99
    assert session.preview() is None
    now[0] = 6.0
    assert session.preview() == "second"


def test_draft_preview_resumes_after_long_pause_and_audits_missed_quiet_speech(
    tmp_path, monkeypatch
):
    now = [1.0]
    monkeypatch.setattr("vim2.session.monotonic", lambda: now[0], raising=False)
    monkeypatch.setattr("webrtcvad.Vad.is_speech", lambda self, data, rate: any(data))
    recorder = FakeRecorder(_long_artifact(tmp_path))
    recorder.snapshot = lambda max_seconds=None: AudioArtifact(
        np.full(16_000, 0.0001 if now[0] == 76.0 else 0.1, dtype=np.float32),
        16_000, start_frame=int(now[0] * 16_000),
    )
    recognizer = DraftRecognizer(["first", "resumed", "quiet audit"])
    session = VoiceSession(_ready_machine(), recorder, recognizer, FakePaster())
    session.start(target_window=42, model_id=ModelId.FAST)
    assert session.preview() == "first"
    now[0] = 61.0
    assert session.preview() == "resumed"
    monkeypatch.setattr("webrtcvad.Vad.is_speech", lambda self, data, rate: False)
    now[0] = 66.0
    assert session.preview() == "quiet audit"  # One sentence-end flush.
    recognizer.responses.append("quiet audit")
    now[0] = 76.0
    assert session.preview() is None
    now[0] = 81.0
    assert session.preview() == "quiet audit"


def test_missing_gpu_vad_does_not_leave_microphone_open(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    recorder = FakeRecorder(_long_artifact(tmp_path))
    session = VoiceSession(
        _ready_machine(), recorder, DraftRecognizer([]), FakePaster()
    )
    with pytest.raises(RuntimeError, match="webrtcvad-wheels"):
        session.start(target_window=42, model_id=ModelId.FAST)
    assert not recorder.started
    assert session.state is AppState.READY


def test_five_minute_pause_skips_inference_but_keeps_entire_final_audio(monkeypatch):
    now = [1.0]
    monkeypatch.setattr("vim2.session.monotonic", lambda: now[0])
    monkeypatch.setattr("webrtcvad.Vad.is_speech", lambda self, data, rate: any(data))
    source = np.concatenate((
        np.full(16_000, 0.01, dtype=np.float32),
        np.zeros(299 * 16_000, dtype=np.float32),
    ))
    full = AudioArtifact(source, 16_000)
    recorder = FakeRecorder(full)

    def snapshot(max_seconds=None):
        end = int(now[0] * 16_000)
        start = max(0, end - max_seconds * 16_000)
        return AudioArtifact(source[start:end], 16_000, start_frame=start)

    recorder.snapshot = snapshot
    recognizer = DraftRecognizer(["first", "sentence end", "final"])
    session = VoiceSession(_ready_machine(), recorder, recognizer, FakePaster())
    session.start(target_window=42, model_id=ModelId.FAST)
    assert session.preview() == "first"
    now[0] = 6.0
    assert session.preview() == "sentence end"
    for half_second in range(13, 601):
        now[0] = half_second / 2
        assert session.preview() is None
    assert recognizer.preview_calls == 2
    assert session.stop() == "final"
    assert recognizer.calls[-1][0] is full
    assert full.frame_count == 4_800_000


def test_draft_retry_clears_old_tokens_but_preserves_failed_recording(tmp_path):
    artifact = _artifact(tmp_path)
    recorder = FakeRecorder(artifact)
    recognizer = DraftRecognizer([RuntimeError("failed"), "retry"])
    session = VoiceSession(_ready_machine(), recorder, recognizer, FakePaster())
    session.start(target_window=42, model_id=ModelId.FAST)
    with pytest.raises(FinalRecognitionError):
        session.stop()
    assert recorder.discarded == []
    assert session.retry() == "retry"
    assert recognizer.resets == 3
    assert [call[0] for call in recognizer.calls] == [artifact, artifact]


def test_cancelling_draft_recording_clears_session_tokens(tmp_path):
    recognizer = DraftRecognizer([])
    session = VoiceSession(
        _ready_machine(), FakeRecorder(_artifact(tmp_path)), recognizer, FakePaster()
    )
    session.start(target_window=42, model_id=ModelId.FAST)
    session.cancel()
    assert recognizer.resets == 2


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


def test_final_output_adds_spaces_between_chinese_and_english(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    paster = FakePaster()
    session = VoiceSession(
        _ready_machine(),
        FakeRecorder(artifact),
        FakeRecognizer(["使用VS Code处理API请求，部署 Kubernetes服务"]),
        paster,
    )

    session.start(target_window=42, model_id=ModelId.FAST)
    result = session.stop()

    assert result == "使用 VS Code 处理 API 请求，部署 Kubernetes 服务"
    assert paster.calls == [
        ("使用 VS Code 处理 API 请求，部署 Kubernetes 服务", 42)
    ]


def test_final_output_uses_injected_text_postprocessor(
    tmp_path: Path,
) -> None:
    class PrefixPostProcessor:
        def process(self, text: str) -> str:
            return f"processed: {text}"

    artifact = _artifact(tmp_path)
    session = VoiceSession(
        _ready_machine(),
        FakeRecorder(artifact),
        FakeRecognizer(["one hundred files"]),
        FakePaster(),
        text_postprocessor=PrefixPostProcessor(),
    )

    session.start(target_window=42, model_id=ModelId.FAST)

    assert session.stop() == "processed: one hundred files"


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


def test_export_failed_recording_preserves_exact_audio_and_failure_context(tmp_path):
    import soundfile

    from vim2.draft import GpuGenerationLimitError

    samples = np.array([0.0, 0.12345678, -0.75, 1.125], dtype=np.float32)
    artifact = AudioArtifact(samples, 16_000, ("input overflow",))
    recorder = FakeRecorder(artifact)
    paster = FakePaster()
    session = VoiceSession(
        _ready_machine(), recorder,
        FakeRecognizer([GpuGenerationLimitError([3, 4]), "recovered"]), paster,
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    with pytest.raises(FinalRecognitionError):
        session.stop()
    assert list(tmp_path.iterdir()) == []

    destination = tmp_path / "failure.zip"
    session.export_failed_recording(destination)

    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {"audio.wav", "metadata.json"}
        wav = io.BytesIO(archive.read("audio.wav"))
        assert soundfile.info(wav).subtype == "FLOAT"
        wav.seek(0)
        restored, rate = soundfile.read(wav, dtype="float32")
        np.testing.assert_array_equal(restored, samples)
        assert rate == 16_000
        details = json.loads(archive.read("metadata.json"))
    assert details["schema_version"] == 1
    assert details["model_id"] == ModelId.FAST.value
    assert details["audio"]["frames"] == 4
    assert details["audio"]["warnings"] == ["input overflow"]
    assert details["recognizer"]["hotwords"] == ["VIM2"]
    assert details["recognizer"]["draft_tokens"] == [1, 2, 151645]
    assert details["error"]["type"] == "GpuGenerationLimitError"
    assert details["error"]["generated_tokens"] == [3, 4]
    assert "generation limit" in details["error"]["message"]
    assert "GpuGenerationLimitError" in details["error"]["traceback"]
    assert details["failed_at"]
    assert details["environment"]["python"]
    assert session.state is AppState.RETRY_PENDING
    assert recorder.discarded == []
    assert paster.calls == []
    assert session.retry() == "recovered"
    assert destination.is_file()
    with pytest.raises(RuntimeError, match="failed recording"):
        session.export_failed_recording(tmp_path / "not-failed.zip")


def test_export_write_failure_preserves_existing_file_and_pending_audio(
    tmp_path, monkeypatch,
):
    import soundfile

    artifact = _artifact(tmp_path)
    session = VoiceSession(
        _ready_machine(), FakeRecorder(artifact),
        FakeRecognizer([RuntimeError("decode failed"), "recovered"]), FakePaster(),
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    with pytest.raises(FinalRecognitionError):
        session.stop()
    destination = tmp_path / "failure.zip"
    destination.write_bytes(b"existing export")

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(soundfile, "write", fail_write)
    with pytest.raises(OSError, match="disk full"):
        session.export_failed_recording(destination)

    assert destination.read_bytes() == b"existing export"
    assert list(tmp_path.iterdir()) == [destination]
    assert session.state is AppState.RETRY_PENDING
    assert session.retry() == "recovered"


def test_discarded_failure_cannot_be_exported(tmp_path):
    session = VoiceSession(
        _ready_machine(), FakeRecorder(_artifact(tmp_path)),
        FakeRecognizer([RuntimeError("failed")]), FakePaster(),
    )
    session.start(target_window=7, model_id=ModelId.FAST)
    with pytest.raises(FinalRecognitionError):
        session.stop()
    session.cancel()

    with pytest.raises(RuntimeError, match="failed recording"):
        session.export_failed_recording(tmp_path / "discarded.zip")
    assert list(tmp_path.iterdir()) == []


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
        _ready_machine(),
        recorder,
        recognizer,
        FakePaster(),
        preview_window_seconds=12,
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
        _ready_machine(),
        recorder,
        recognizer,
        FakePaster(),
        preview_window_seconds=8,
    )
    session.start(target_window=7, model_id=ModelId.ACCURATE)

    assert session.preview() == "最近的文本"

    assert recorder.snapshot_max_seconds == [8]
    complete, start_frame, window = recorder.slices[0]
    assert complete.frame_count == 656_000
    assert start_frame == complete.frame_count - (complete.sample_rate * 8)
    assert window.duration_seconds == 8
    assert recognizer.calls == [(window, ModelId.ACCURATE)]
    assert window in recorder.discarded
    assert complete in recorder.discarded
    assert session.stop() == "完整文本"
    assert recognizer.calls[-1] == (artifact, ModelId.ACCURATE)


def test_stable_checkpoint_uses_bounded_snapshot_absolute_end_frame(
    tmp_path: Path,
) -> None:
    recorder = FakeRecorder(_long_artifact(tmp_path))
    snapshots = iter(
        [
            AudioArtifact(np.zeros(96_000, dtype=np.float32), 16_000),
            AudioArtifact(
                np.zeros(128_000, dtype=np.float32),
                16_000,
                start_frame=32_000,
            ),
        ]
    )
    recorder.snapshot = lambda max_seconds=None: next(snapshots)
    session = VoiceSession(
        _ready_machine(),
        recorder,
        FakeRecognizer(
            ["第一句。第二句。未完成", "第一句。第二句。第三句。"]
        ),
        FakePaster(),
    )
    session.start(target_window=7, model_id=ModelId.FAST)

    session.preview()
    session.preview()

    assert session._stable_prefix.checkpoint is not None
    assert session._stable_prefix.checkpoint.frame_count == 160_000


def test_bounded_snapshot_without_anchor_does_not_advance_checkpoint(
    tmp_path: Path,
) -> None:
    recorder = FakeRecorder(_long_artifact(tmp_path))
    snapshot = AudioArtifact(
        np.zeros(128_000, dtype=np.float32),
        16_000,
        start_frame=320_000,
    )
    recorder.snapshot = lambda max_seconds=None: snapshot
    session = VoiceSession(
        _ready_machine(), recorder, FakeRecognizer(["第一句。第二句。"]), FakePaster()
    )
    session.start(target_window=7, model_id=ModelId.FAST)

    session.preview()

    assert session._stable_prefix.checkpoint is None


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
        _ready_machine(),
        recorder,
        recognizer,
        FakePaster(),
        preview_window_seconds=12,
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
