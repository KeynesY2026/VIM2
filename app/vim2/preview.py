from __future__ import annotations

import numpy as np

from vim2.audio import AudioArtifact


class PreviewActivity:
    def __init__(self) -> None:
        try:
            import webrtcvad
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "GPU preview requires webrtcvad-wheels; install requirements-gpu-cu128.lock"
            ) from error

        self._vad = webrtcvad.Vad(0)
        self._next_frame = 0
        self.last_speech_frame = 0
        self.last_signal_frame = 0

    def observe(self, audio: AudioArtifact) -> None:
        if audio.sample_rate != 16_000:
            raise ValueError("GPU preview activity detection requires 16000 Hz audio")
        if audio.end_frame <= self._next_frame:
            return
        start = max(self._next_frame, audio.start_frame)
        frame_size = 480
        end = start + (audio.end_frame - start) // frame_size * frame_size
        samples = audio.samples[start - audio.start_frame:end - audio.start_frame]
        if not np.isfinite(samples).all():
            raise ValueError("Preview audio contains non-finite samples")
        if np.any(samples):
            self.last_signal_frame = end
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        for offset in range(0, len(pcm), frame_size):
            if self._vad.is_speech(pcm[offset:offset + frame_size].tobytes(), 16_000):
                self.last_speech_frame = start + offset + frame_size
        self._next_frame = end


class DraftPreviewPolicy:
    INTERVAL_SECONDS = 5.0
    MIN_REST_SECONDS = 2.0
    MAX_BUSY_FRACTION = 0.6
    PAUSE_SECONDS = 0.8
    AUDIT_SECONDS = 15.0

    def __init__(self) -> None:
        self._activity = PreviewActivity()
        self._completed_at: float | None = None
        self._started_at = 0.0
        self._next_at = 0.0
        self._preview_speech_frame = 0
        self._preview_end_frame = 0
        self._flush_pending = False
        self._failed = False

    def should_run(self, audio: AudioArtifact, now: float) -> bool:
        self._activity.observe(audio)
        if audio.frame_count == 0:
            return False
        if self._completed_at is None:
            if audio.frame_count < audio.sample_rate:
                return False
            self._started_at = now
            return True
        elapsed = now - self._completed_at
        if now < self._next_at:
            return False
        new_speech = self._activity.last_speech_frame > self._preview_speech_frame
        # A periodic audit prevents a VAD false negative from hiding quiet speech.
        audit = (
            elapsed >= self.AUDIT_SECONDS
            and self._activity.last_signal_frame > self._preview_end_frame
        )
        if new_speech or self._flush_pending or self._failed or audit:
            self._started_at = now
            return True
        return False

    def completed(self, audio: AudioArtifact, now: float, *, success: bool) -> None:
        self._completed_at = now
        duration = max(0.0, now - self._started_at)
        rest = max(
            self.MIN_REST_SECONDS,
            duration / self.MAX_BUSY_FRACTION - duration,
        )
        self._next_at = max(
            self._started_at + self.INTERVAL_SECONDS, now + rest
        )
        self._failed = not success
        if success:
            self._preview_end_frame = audio.end_frame
            self._preview_speech_frame = self._activity.last_speech_frame
            self._flush_pending = (
                self._preview_speech_frame > 0
                and audio.end_frame - self._preview_speech_frame
                < audio.sample_rate * self.PAUSE_SECONDS
            )
