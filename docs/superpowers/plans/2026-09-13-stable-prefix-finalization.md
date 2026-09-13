# Stable-prefix final recognition implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce stop-to-paste latency by recognizing only an overlapped audio tail when three previews establish a stable sentence prefix, while retaining full-audio fallback and retry.

**Architecture:** Add a pure transcript tracker that records a monotonic stable prefix and its audio-frame checkpoint. Extend the recorder to create a WAV suffix from the authoritative stopped recording. Let `VoiceSession` attempt exact-anchor tail merging only when it saves at least 30% of the audio, then fall back to the complete WAV whenever the optimized result is uncertain or fails.

**Tech Stack:** Python 3.10-3.13, NumPy, standard-library `wave`, pytest, Qwen3-ASR

**Spec:** `docs/superpowers/specs/2026-09-13-stable-prefix-finalization-design.md`

## Global constraints

- Keep the default and maximum recording duration at 90 seconds.
- Do not add Qwen Forced Aligner or any other model or dependency.
- Require three identical successful preview prefixes ending in `。！？.!?`.
- Keep 8 seconds of audio before the stable checkpoint.
- Attempt tail recognition only when the tail is at most 70% of the complete recording.
- Merge only when the last stable sentence occurs exactly once in the tail transcript.
- Never paste preview or stable-prefix text before finalization.
- Keep the complete stopped WAV until success, user cancellation, or shutdown.
- Retry always recognizes the complete WAV.
- Preserve current model selection, offline operation, warnings, clipboard behavior, and target-window behavior.
- Use test-driven development: run each new test before implementation and confirm the expected failure.
- Start implementation from commit `5eb6674c`.

---

### Task 1: Track stable sentence prefixes

**Files:**
- Create: `app/vim2/transcript.py`
- Create: `tests/test_transcript.py`

**Interfaces:**
- Consumes: preview `text: str` and snapshot `frame_count: int`
- Produces: `StableCheckpoint(prefix: str, anchor: str, frame_count: int)`
- Produces: `StablePrefixTracker.observe(text: str, frame_count: int) -> None`
- Produces: `StablePrefixTracker.checkpoint -> StableCheckpoint | None`
- Produces: `merge_stable_tail(checkpoint: StableCheckpoint, tail_text: str) -> str | None`

- [ ] **Step 1: Write failing extraction and stabilization tests**

Create `tests/test_transcript.py` with literal expected values:

```python
from vim2.transcript import StablePrefixTracker


def test_three_identical_complete_prefixes_create_checkpoint() -> None:
    tracker = StablePrefixTracker()

    tracker.observe("第一句。正在变化", 160_000)
    tracker.observe("第一句。仍在变化", 176_000)
    assert tracker.checkpoint is None

    tracker.observe("第一句。最后变化", 192_000)

    assert tracker.checkpoint is not None
    assert tracker.checkpoint.prefix == "第一句。"
    assert tracker.checkpoint.anchor == "第一句。"
    assert tracker.checkpoint.frame_count == 192_000


def test_text_without_sentence_ending_never_stabilizes() -> None:
    tracker = StablePrefixTracker()
    for frame_count in (16_000, 32_000, 48_000):
        tracker.observe("还没有结束", frame_count)

    assert tracker.checkpoint is None


def test_changed_candidate_restarts_three_preview_count() -> None:
    tracker = StablePrefixTracker()
    tracker.observe("旧句。", 16_000)
    tracker.observe("旧句。", 32_000)
    tracker.observe("新句。", 48_000)
    tracker.observe("新句。", 64_000)

    assert tracker.checkpoint is None

    tracker.observe("新句。", 80_000)
    assert tracker.checkpoint is not None
    assert tracker.checkpoint.prefix == "新句。"
```

- [ ] **Step 2: Run the new tests and verify the missing module failure**

Run:

```powershell
python -m pytest tests\test_transcript.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'vim2.transcript'`.

- [ ] **Step 3: Implement the immutable checkpoint and tracker**

Create `app/vim2/transcript.py` with these types and behavior:

```python
from __future__ import annotations

from dataclasses import dataclass

SENTENCE_ENDINGS = frozenset("。！？.!?")


@dataclass(frozen=True, slots=True)
class StableCheckpoint:
    prefix: str
    anchor: str
    frame_count: int


class StablePrefixTracker:
    def __init__(self, required_matches: int = 3) -> None:
        self._required_matches = required_matches
        self.reset()

    @property
    def checkpoint(self) -> StableCheckpoint | None:
        return self._checkpoint

    def reset(self) -> None:
        self._candidate = ""
        self._matches = 0
        self._checkpoint: StableCheckpoint | None = None

    def observe(self, text: str, frame_count: int) -> None:
        prefix = _complete_prefix(text.strip())
        if not prefix:
            self._candidate = ""
            self._matches = 0
            return
        if self._checkpoint and not prefix.startswith(self._checkpoint.prefix):
            self._candidate = ""
            self._matches = 0
            return
        if prefix == self._candidate:
            self._matches += 1
        else:
            self._candidate = prefix
            self._matches = 1
        if self._matches >= self._required_matches and (
            self._checkpoint is None
            or len(prefix) > len(self._checkpoint.prefix)
        ):
            self._checkpoint = StableCheckpoint(
                prefix=prefix,
                anchor=_last_sentence(prefix),
                frame_count=frame_count,
            )
```

Implement `_complete_prefix()` by finding the final character in `SENTENCE_ENDINGS`. Implement `_last_sentence()` by scanning backward from that final ending to the preceding ending and stripping only leading whitespace from the anchor.

- [ ] **Step 4: Add failing monotonic-growth tests**

Append:

```python
def test_stable_prefix_only_moves_forward() -> None:
    tracker = StablePrefixTracker()
    for frame_count in (16_000, 32_000, 48_000):
        tracker.observe("第一句。尾巴", frame_count)
    first = tracker.checkpoint

    tracker.observe("冲突句。", 64_000)
    tracker.observe("第一句。第二句。尾巴", 80_000)
    tracker.observe("第一句。第二句。尾巴", 96_000)
    tracker.observe("第一句。第二句。尾巴", 112_000)

    assert first is not None
    assert tracker.checkpoint is not None
    assert tracker.checkpoint.prefix == "第一句。第二句。"
    assert tracker.checkpoint.anchor == "第二句。"
    assert tracker.checkpoint.frame_count == 112_000
```

Run the test before adjusting the implementation. Expected: the first version fails to advance after the conflict because the candidate count was reset.

- [ ] **Step 5: Make monotonic advancement pass**

Ensure a candidate may replace the current stable checkpoint only when it starts with the stable prefix. A conflicting or shorter observation resets only the candidate count and leaves `_checkpoint` unchanged.

- [ ] **Step 6: Add exact-anchor merge tests**

Append:

```python
from vim2.transcript import StableCheckpoint, merge_stable_tail


def test_unique_anchor_merges_without_duplicate_text() -> None:
    checkpoint = StableCheckpoint(
        prefix="第一句。第二句。",
        anchor="第二句。",
        frame_count=320_000,
    )

    assert merge_stable_tail(
        checkpoint, "第二句。最后一段。"
    ) == "第一句。第二句。最后一段。"


def test_missing_or_repeated_anchor_requires_fallback() -> None:
    checkpoint = StableCheckpoint("第一句。", "第一句。", 160_000)

    assert merge_stable_tail(checkpoint, "不同内容。") is None
    assert merge_stable_tail(
        checkpoint, "第一句。中间。第一句。结尾。"
    ) is None
    assert merge_stable_tail(checkpoint, "   ") is None
```

- [ ] **Step 7: Run merge tests and verify they fail**

Run:

```powershell
python -m pytest tests\test_transcript.py -q
```

Expected: FAIL because `merge_stable_tail` is not defined.

- [ ] **Step 8: Implement exact-anchor merging**

Add:

```python
def merge_stable_tail(
    checkpoint: StableCheckpoint, tail_text: str
) -> str | None:
    tail = tail_text.strip()
    if not tail or tail.count(checkpoint.anchor) != 1:
        return None
    _, remainder = tail.split(checkpoint.anchor, maxsplit=1)
    return f"{checkpoint.prefix}{remainder}".strip()
```

- [ ] **Step 9: Run tracker tests**

Run:

```powershell
python -m pytest tests\test_transcript.py -q
```

Expected: all tests pass.

- [ ] **Step 10: Commit the tracker**

```powershell
git add app\vim2\transcript.py tests\test_transcript.py
git commit -m "feat: track stable transcript prefixes" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Slice authoritative WAV recordings

**Files:**
- Modify: `app/vim2/audio.py:21-151`
- Modify: `tests/test_audio.py:1-111`

**Interfaces:**
- Consumes: `AudioArtifact` created by `AudioRecorder.stop()`
- Produces: `AudioRecorder.slice_from(artifact: AudioArtifact, start_frame: int) -> AudioArtifact`
- Guarantees: source artifact remains unchanged; returned artifact contains frames `[start_frame:]`

- [ ] **Step 1: Write the failing WAV slice test**

Append to `tests/test_audio.py`:

```python
import wave


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
```

- [ ] **Step 2: Run the slice test and verify the missing method failure**

Run:

```powershell
python -m pytest tests\test_audio.py::test_slice_from_writes_requested_wav_suffix -q
```

Expected: FAIL with `AttributeError: 'AudioRecorder' object has no attribute 'slice_from'`.

- [ ] **Step 3: Implement WAV slicing**

Add `slice_from()` to `AudioRecorder`. Validate `0 <= start_frame < artifact.frame_count`, read the source WAV metadata, seek to `start_frame`, and write all remaining frames to a `tail-<uuid>.wav` file. Preserve channel count, sample width, frame rate, compression type, and compression name from the source WAV. Return an `AudioArtifact` with:

```python
AudioArtifact(
    path=path,
    frame_count=source_frame_count - start_frame,
    sample_rate=source_sample_rate,
    warnings=artifact.warnings,
)
```

Raise `ValueError("start_frame must reference an existing audio frame")` for negative values or values at or beyond the source frame count.

- [ ] **Step 4: Add and run invalid-boundary tests**

Append:

```python
import pytest


def test_slice_from_rejects_frame_outside_recording(tmp_path: Path) -> None:
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
```

Update the import to `from vim2.audio import AudioArtifact, AudioRecorder`.

Run:

```powershell
python -m pytest tests\test_audio.py -q
```

Expected: all audio tests pass.

- [ ] **Step 5: Commit WAV slicing**

```powershell
git add app\vim2\audio.py tests\test_audio.py
git commit -m "feat: slice recorded audio tails" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Use stable prefixes during final recognition

**Files:**
- Modify: `app/vim2/session.py:9-153`
- Modify: `tests/test_session.py:9-253`
- Modify: `tests/test_controller.py:173-254`

**Interfaces:**
- Consumes: `StablePrefixTracker`, `StableCheckpoint`, and `merge_stable_tail` from Task 1
- Consumes: `Recorder.slice_from(artifact: AudioArtifact, start_frame: int) -> AudioArtifact` from Task 2
- Produces: unchanged public `VoiceSession.preview() -> str`, `stop() -> str`, and `retry() -> str`
- Constants: `TAIL_OVERLAP_SECONDS = 8`, `MAX_TAIL_RATIO = 0.70`

- [ ] **Step 1: Extend the session test recorder**

In `tests/test_session.py`, extend `FakeRecorder` without changing production code:

```python
self.snapshots: list[AudioArtifact] = []
self.slices: list[tuple[AudioArtifact, int, AudioArtifact]] = []

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
```

Keep `discard()` recording every discarded artifact so cleanup remains observable.

- [ ] **Step 2: Write the failing optimized-finalization test**

Add a helper that creates a 60-second artifact:

```python
def _long_artifact(tmp_path: Path) -> AudioArtifact:
    path = tmp_path / "recording.wav"
    path.write_bytes(b"audio")
    return AudioArtifact(path=path, frame_count=960_000, sample_rate=16_000)
```

Add:

```python
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
```

Add a second test with the same preview setup, followed by tail response
`"没有锚点。"` and full response `"第一句。完整尾段。"`.
Assert that the final two recognizer calls use the tail and complete paths,
the pasted result is `"第一句。完整尾段。"`, and both artifacts are deleted.

Add a 20-second artifact test with a checkpoint at 15 seconds. The 8-second
overlap leaves a 13-second tail, which exceeds 70% of the complete recording.
Assert that `recorder.slices == []` and final recognition uses the complete
artifact.

The seeded snapshot makes the three generated preview frame counts `656_000`, `672_000`, and `688_000`. The expected start frame is `688_000 - 8 * 16_000 = 560_000`, producing a 400,000-frame tail, or 41.7% of the complete recording.

- [ ] **Step 3: Run the optimized test and verify failure**

Run:

```powershell
python -m pytest tests\test_session.py::test_stable_prefix_uses_overlapped_tail_for_final_result -q
```

Expected: FAIL because the session does not call `slice_from()` and consumes no tail response.

- [ ] **Step 4: Integrate preview tracking**

In `app/vim2/session.py`:

- Import `StablePrefixTracker` and `merge_stable_tail`
- Add `TAIL_OVERLAP_SECONDS = 8` and `MAX_TAIL_RATIO = 0.70`
- Add `slice_from()` to the `Recorder` protocol
- Create `self._stable_prefix = StablePrefixTracker()` in `__init__`
- Call `self._stable_prefix.reset()` after `recorder.start()` succeeds
- In `preview()`, call `observe(text, artifact.frame_count)` only after transcription succeeds

Do not change the value returned by `preview()`.

- [ ] **Step 5: Implement eligible tail recognition**

Change `stop()` to call `_recognize_pending(allow_tail=True)` and `retry()` to call `_recognize_pending(allow_tail=False)`.

Pass the already validated artifact and model into
`_recognize_final(artifact, model_id, allow_tail)`:

```python
def _recognize_final(
    self,
    artifact: AudioArtifact,
    model_id: ModelId,
    *,
    allow_tail: bool,
) -> str:
    checkpoint = self._stable_prefix.checkpoint
    if allow_tail and checkpoint is not None:
        overlap_frames = artifact.sample_rate * TAIL_OVERLAP_SECONDS
        start_frame = max(0, checkpoint.frame_count - overlap_frames)
        tail_frames = artifact.frame_count - start_frame
        if tail_frames <= artifact.frame_count * MAX_TAIL_RATIO:
            merged = self._recognize_tail(
                artifact, model_id, checkpoint, start_frame
            )
            if merged is not None:
                return merged
    return self._recognizer.transcribe(
        artifact.path, model_id
    ).strip()
```

Implement `_recognize_tail()` with signature
`(artifact: AudioArtifact, model_id: ModelId, checkpoint: StableCheckpoint, start_frame: int) -> str | None`.
It creates a tail artifact, transcribes it, returns
`merge_stable_tail(checkpoint, text)`, and discards the tail in `finally`.

- [ ] **Step 6: Run the optimized test and session suite**

Run:

```powershell
python -m pytest tests\test_session.py -q
```

Expected: all session tests pass. Update existing response lists only where the new three-preview setup intentionally activates tail optimization. Existing one-preview tests must continue to recognize the complete recording.

- [ ] **Step 7: Verify controller serialization**

Run:

```powershell
python -m pytest tests\test_controller.py -q
```

Expected: all controller tests pass. The deferred preview still completes before the finalization worker starts, and finalization still pastes exactly once.

- [ ] **Step 8: Commit optimized finalization**

```powershell
git add app\vim2\session.py tests\test_session.py tests\test_controller.py
git commit -m "feat: finalize stable transcript tails" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Add recognition-error fallback and lifecycle coverage

**Files:**
- Modify: `app/vim2/session.py:70-153`
- Modify: `tests/test_session.py`
- Modify: `REQUIREMENTS.md:30-70`
- Modify: `REQUIREMENTS.md:120-145`

**Interfaces:**
- Consumes: optimized path from Task 3
- Produces: full-audio fallback when tail recognition raises an expected recognition error
- Guarantees: retry bypasses tail optimization and all temporary artifacts are cleaned

- [ ] **Step 1: Write the failing tail-error fallback test**

Add this test to `tests/test_session.py`, using the 60-second artifact and
snapshot behavior from Task 3:

```python
def test_tail_recognition_error_falls_back_to_complete_recording(
    tmp_path: Path,
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

    tail = recorder.slices[0][2]
    assert result == "第一句。完整尾段。"
    assert recognizer.calls[-2:] == [
        (tail.path, ModelId.FAST),
        (artifact.path, ModelId.FAST),
    ]
    assert paster.calls == [("第一句。完整尾段。", 7)]
    assert not tail.path.exists()
    assert not artifact.path.exists()
```

- [ ] **Step 2: Run fallback tests and verify expected failures**

Run:

```powershell
python -m pytest tests\test_session.py -q
```

Expected: FAIL with `FinalRecognitionError: tail failed` because the tail
exception currently bypasses complete-audio fallback.

- [ ] **Step 3: Implement tail-to-full fallback**

Define the existing recognized exception tuple once in `session.py`:

```python
RECOGNITION_ERRORS = (OSError, RuntimeError, ValueError, MemoryError)
```

In `_recognize_tail()`, catch only `RECOGNITION_ERRORS` and return `None`, then always discard the tail in `finally`. A `None` return makes `_recognize_final()` transcribe the complete artifact. Keep the outer `_recognize_pending()` handler so a failed complete transcription enters `RETRY_PENDING` with `FinalRecognitionError`.

- [ ] **Step 4: Add failed-full-recognition and retry coverage**

Add a test with responses:

```python
[
    "第一句。未完成",
    "第一句。变化",
    "第一句。继续",
    RuntimeError("tail failed"),
    RuntimeError("full failed"),
    "第一句。重试成功。",
]
```

Assert:

- `stop()` raises `FinalRecognitionError("full failed")`
- the tail file is deleted
- the complete WAV still exists
- the state is `RETRY_PENDING`
- `retry()` calls the recognizer with the complete WAV
- retry pastes `第一句。重试成功。`
- successful retry deletes the complete WAV

- [ ] **Step 5: Add cancellation and shutdown cleanup coverage**

Add focused tests proving:

- cancelling from `RETRY_PENDING` after optimized finalization fails deletes the complete WAV
- shutdown after the same failure deletes the complete WAV
- no tail artifact remains after either path

Run:

```powershell
python -m pytest tests\test_session.py -q
```

Expected: all session tests pass.

- [ ] **Step 6: Update product requirements**

Update `REQUIREMENTS.md`:

- Replace section 2.2 step 5 with the stable-prefix, overlapped-tail, and full-fallback behavior
- Replace section 3.4 statements that permit direct preview submission
- State the three-preview rule, exact sentence endings, 8-second overlap, 30% minimum savings, unique exact anchor, and full-audio retry
- Keep section 3.2 at a 90-second maximum

- [ ] **Step 7: Run targeted and complete verification**

Run:

```powershell
python -m pytest tests\test_transcript.py tests\test_audio.py tests\test_session.py tests\test_controller.py -q
python -m pytest -q
git diff --check
```

Expected:

- all targeted tests pass
- all repository tests pass
- `git diff --check` produces no output

- [ ] **Step 8: Perform local model acceptance**

Start the application:

```powershell
.\Start.cmd
```

Record at least 30 seconds containing at least three distinct complete sentences followed by a final short sentence. Keep each complete sentence under 8 seconds. Confirm:

- the overlay previews the full transcript
- stopping pastes all sentences, including the final short sentence
- finalization completes faster than a comparable recording with no sentence-ending punctuation
- no `tail-*.wav`, `preview-*.wav`, or `recording-*.wav` remains in the application temp directory after success

Stop the application through the tray after acceptance.

- [ ] **Step 9: Commit fallback behavior and requirements**

```powershell
git add app\vim2\session.py tests\test_session.py REQUIREMENTS.md
git commit -m "fix: fall back from uncertain tail recognition" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
