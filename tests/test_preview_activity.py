import numpy as np
import pytest

from vim2.audio import AudioArtifact


def test_activity_does_not_reprocess_overlapping_snapshots(monkeypatch):
    from vim2.preview import PreviewActivity

    calls = []
    monkeypatch.setattr(
        "webrtcvad.Vad.is_speech",
        lambda self, data, rate: calls.append(data) or any(data),
    )
    activity = PreviewActivity()
    speech = np.full(1_000, 0.01, dtype=np.float32)
    activity.observe(AudioArtifact(speech, 16_000))
    assert activity.last_speech_frame == 960
    assert len(calls) == 2
    activity.observe(AudioArtifact(np.concatenate((speech, np.zeros(440))), 16_000))
    assert len(calls) == 3
    assert activity.last_speech_frame == 1440
    activity.observe(AudioArtifact(np.zeros(16_000), 16_000, start_frame=1440))
    assert activity.last_speech_frame == 1440


def test_activity_preserves_quiet_samples_instead_of_energy_gating(monkeypatch):
    from vim2.preview import PreviewActivity

    frames = []
    monkeypatch.setattr(
        "webrtcvad.Vad.is_speech",
        lambda self, data, rate: frames.append(data) or True,
    )
    activity = PreviewActivity()
    activity.observe(AudioArtifact(np.full(480, 0.0001), 16_000, start_frame=32_000))
    assert any(frames[0])
    assert activity.last_speech_frame == 32_480


def test_activity_real_vad_does_not_mark_digital_silence_as_speech():
    from vim2.preview import PreviewActivity

    activity = PreviewActivity()
    activity.observe(AudioArtifact(np.zeros(16_000, dtype=np.float32), 16_000))
    assert activity.last_speech_frame == 0


def test_activity_rejects_unsupported_sample_rate():
    from vim2.preview import PreviewActivity

    with pytest.raises(ValueError, match="16000"):
        PreviewActivity().observe(AudioArtifact(np.zeros(480), 48_000))


def test_first_preview_collects_one_second_before_inference():
    from vim2.preview import DraftPreviewPolicy

    policy = DraftPreviewPolicy()
    assert not policy.should_run(AudioArtifact(np.zeros(8_000), 16_000), 0.5)
    assert policy.should_run(AudioArtifact(np.zeros(16_000), 16_000), 1.0)


def test_slow_preview_gets_rest_proportional_to_inference_cost(monkeypatch):
    from vim2.preview import DraftPreviewPolicy

    monkeypatch.setattr("webrtcvad.Vad.is_speech", lambda self, data, rate: any(data))
    policy = DraftPreviewPolicy()
    first = AudioArtifact(np.full(16_000, 0.01), 16_000)
    assert policy.should_run(first, 1.0)
    policy.completed(first, 7.0, success=True)
    recent = AudioArtifact(np.full(16_000, 0.01), 16_000, start_frame=160_000)
    assert not policy.should_run(recent, 10.99)
    assert policy.should_run(recent, 11.0)


def test_silence_audit_does_not_redecode_old_speech_still_in_window(monkeypatch):
    from vim2.preview import DraftPreviewPolicy

    monkeypatch.setattr("webrtcvad.Vad.is_speech", lambda self, data, rate: any(data))
    policy = DraftPreviewPolicy()
    samples = np.concatenate((np.full(16_000, 0.01), np.zeros(59 * 16_000)))
    first = AudioArtifact(samples[:16_000], 16_000)
    assert policy.should_run(first, 1.0)
    policy.completed(first, 1.0, success=True)
    tail = AudioArtifact(samples[:6 * 16_000], 16_000)
    assert policy.should_run(tail, 6.0)
    policy.completed(tail, 6.0, success=True)
    assert not policy.should_run(AudioArtifact(samples[:21 * 16_000], 16_000), 21.0)
