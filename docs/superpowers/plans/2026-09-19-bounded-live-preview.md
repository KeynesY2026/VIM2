# Bounded Live Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound live-preview copying and inference to a configurable 8-second window while preventing overloaded preview inference and unchanged UI results from causing continuous work.

**Architecture:** Carry absolute frame offsets on bounded in-memory audio artifacts, let the recorder build only the requested recent snapshot, and make `VoiceSession` interpret stable checkpoints against absolute positions. Keep the periodic timer as the only preview-start clock, suppress immediate catch-up inference, and retain cooperative cancellation around synchronous sherpa decoding.

**Tech Stack:** Python 3.10+, NumPy, PySide6, sherpa-onnx, pytest

**Spec:** `docs/superpowers/specs/2026-09-19-bounded-live-preview-design.md`

## Global Constraints

- `preview_window_seconds` defaults to `8` and accepts integers from `1` through `30`.
- `tail_overlap_seconds` remains independent; the preview window is the hard resource boundary.
- No Python thread termination or CPU model process restart.
- Final recognition fallback and retry semantics must remain unchanged.
- Production changes follow red-green-refactor test cycles.

---

### Task 1: Configurable Preview Window

**Files:**
- Modify: `app/vim2/config.py`
- Modify: `app/vim2/qt_runtime.py`
- Modify: `config/settings.json`
- Test: `tests/test_config.py`
- Test: `tests/test_qt_runtime.py`

**Interfaces:**
- Produces: `Settings.preview_window_seconds: int`
- Produces: `VoiceSession(..., preview_window_seconds: int = 8)` construction input

- [ ] **Step 1: Write failing configuration tests**

Add assertions for `DEFAULT_PREVIEW_WINDOW_SECONDS == 8`, default loading, round-trip persistence, and invalid values `[0, 31, 8.0, "8"]`. Extend the runtime construction test so the configured value must reach `VoiceSession`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_config.py tests/test_qt_runtime.py -q`

Expected: failures because the constant, settings field, validation, and constructor forwarding do not exist.

- [ ] **Step 3: Implement the minimal configuration path**

Add:

```python
DEFAULT_PREVIEW_WINDOW_SECONDS = 8
MIN_PREVIEW_WINDOW_SECONDS = 1
MAX_PREVIEW_WINDOW_SECONDS = 30
```

Add `preview_window_seconds` to `Settings`, parse and validate it in `SettingsRepository.load()`, persist it through `asdict()`, add it to `config/settings.json`, and pass it from `run_qt_application()` into `VoiceSession`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_config.py tests/test_qt_runtime.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/vim2/config.py app/vim2/qt_runtime.py config/settings.json tests/test_config.py tests/test_qt_runtime.py
git commit -m "feat: configure live preview window"
```

### Task 2: Tail-Only Recorder Snapshots

**Files:**
- Modify: `app/vim2/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces: `AudioArtifact.start_frame: int = 0`
- Produces: `AudioArtifact.end_frame -> int`
- Produces: `AudioRecorder.snapshot(max_seconds: int | None = None) -> AudioArtifact`
- Preserves: `AudioRecorder.slice_from(artifact, start_frame)` uses artifact-relative `start_frame`

- [ ] **Step 1: Write failing artifact-position tests**

Add tests showing that an artifact with `start_frame=10` and three samples has `frame_count == 3` and `end_frame == 13`, and that `slice_from(..., 1)` returns `start_frame == 11`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_audio.py -q`

Expected: failures because `start_frame`, `end_frame`, and propagated offsets do not exist.

- [ ] **Step 3: Implement absolute artifact positions**

Add the field/property and propagate offsets through `slice_from()`. Keep `duration_seconds` based on local sample count.

- [ ] **Step 4: Run audio tests and verify GREEN**

Run: `python -m pytest tests/test_audio.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing bounded-snapshot tests**

Feed multiple known chunks, call `snapshot(max_seconds=2)` at a small sample rate, and assert only the newest samples are returned with correct `start_frame`/`end_frame`. Instrument `_join_chunks` to prove it receives no more chunks/samples than the requested tail. Add rejection tests for zero and negative bounds.

- [ ] **Step 6: Run bounded snapshot tests and verify RED**

Run: `python -m pytest tests/test_audio.py -q`

Expected: `snapshot()` rejects the new argument or returns the complete recording.

- [ ] **Step 7: Implement tail-only assembly**

Maintain `_captured_frames` as callbacks append chunks. Under the capture lock, select chunk references backward until `max_seconds * sample_rate` frames are covered and record the snapshot end position. Concatenate and trim outside the lock. Reset `_captured_frames` in start/stop/cancel. Reject non-positive integer bounds.

- [ ] **Step 8: Run audio tests and verify GREEN**

Run: `python -m pytest tests/test_audio.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add app/vim2/audio.py tests/test_audio.py
git commit -m "perf: snapshot only recent preview audio"
```

### Task 3: Absolute Stable Cursor and Configured Session Window

**Files:**
- Modify: `app/vim2/session.py`
- Modify: `tests/test_session.py`

**Interfaces:**
- Consumes: `Recorder.snapshot(max_seconds: int | None = None)`
- Consumes: `AudioArtifact.start_frame` and `AudioArtifact.end_frame`
- Produces: `VoiceSession(..., preview_window_seconds: int = 8)`

- [ ] **Step 1: Update fake recorder and write failing bounded-session tests**

Make `FakeRecorder.snapshot(max_seconds=None)` record requested bounds and return artifacts with absolute offsets. Assert `VoiceSession.preview()` requests the configured 8-second bound, passes no more than that window to the recognizer, and observes checkpoints at `artifact.end_frame` rather than local `frame_count`.

- [ ] **Step 2: Run focused session tests and verify RED**

Run: `python -m pytest tests/test_session.py -q`

Expected: failures because session calls unbounded `snapshot()` and treats local length as an absolute checkpoint.

- [ ] **Step 3: Implement configured bounded preview**

Store `preview_window_seconds`, call `snapshot(max_seconds=self._preview_window_seconds)`, replace `MAX_PREVIEW_SECONDS` branches with bounded-artifact logic, translate absolute checkpoint positions into artifact-relative slice indexes, and call `observe(text, artifact.end_frame)`.

- [ ] **Step 4: Add preview-window diagnostics**

Before each preview recognizer call, log model, duration, `start_frame`, and `end_frame`. Keep final recognition behavior unchanged.

- [ ] **Step 5: Run session tests and verify GREEN**

Run: `python -m pytest tests/test_session.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/vim2/session.py tests/test_session.py
git commit -m "feat: track bounded preview audio positions"
```

### Task 4: Non-Saturating Preview Scheduling

**Files:**
- Modify: `app/vim2/controller.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Preserves: `AppController.request_preview() -> None`
- Changes: busy requests are coalesced/dropped until the next timer tick; completion never recursively calls `request_preview()`

- [ ] **Step 1: Replace the immediate-follow-up test with the desired failing behavior**

Drive a deferred preview, issue repeated busy requests, complete it, and assert no task is queued. Invoke `request_preview()` once more to represent the next timer tick and assert exactly one fresh task is queued.

- [ ] **Step 2: Run the focused controller test and verify RED**

Run: `python -m pytest tests/test_controller.py::test_busy_preview_waits_for_next_timer_tick -q`

Expected: failure because `_after_preview()` immediately submits pending work.

- [ ] **Step 3: Implement timer-paced scheduling**

Do not recursively call `request_preview()` from `_after_preview()`. Clear `_preview_pending` after completion and render normally. Preserve stop/cancel precedence and ensure their paths clear pending state.

- [ ] **Step 4: Run controller tests and verify GREEN**

Run: `python -m pytest tests/test_controller.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/vim2/controller.py tests/test_controller.py
git commit -m "fix: pace preview inference by timer ticks"
```

### Task 5: Suppress Unchanged UI Results and Improve CPU Cancellation Boundaries

**Files:**
- Modify: `app/vim2/ui.py`
- Modify: `app/vim2/recognizer.py`
- Test: `tests/test_ui.py`
- Test: `tests/test_recognizer.py`

**Interfaces:**
- Preserves: `DesktopView.show_preview(text: str) -> None`
- Preserves: `QwenRecognizer.transcribe(..., cancel_event=...) -> str`

- [ ] **Step 1: Write a failing unchanged-preview UI test**

Patch or count `overlay.set_recording` calls, publish identical text twice, and assert the second publication does not repaint.

- [ ] **Step 2: Run the focused UI test and verify RED**

Run: `python -m pytest tests/test_ui.py::test_identical_preview_does_not_repaint_overlay -q`

Expected: two repaint calls instead of one.

- [ ] **Step 3: Implement unchanged-text suppression**

Return early in `DesktopView.show_preview()` when `text == self._latest_preview`; otherwise update state and overlay.

- [ ] **Step 4: Write failing CPU cancellation-boundary tests**

Use the fake sherpa stream/recognizer to set cancellation during `accept_waveform()` and assert `decode_stream()` is not entered. Add a log-capture test asserting successful CPU recognition logs completion.

- [ ] **Step 5: Run recognizer tests and verify RED**

Run: `python -m pytest tests/test_recognizer.py -q`

Expected: decode still runs after cancellation during waveform acceptance and CPU completion is not logged.

- [ ] **Step 6: Implement cooperative checks and completion logging**

Check `cancel_event` before creating the stream, after `accept_waveform()`, and after `decode_stream()`. Log successful CPU completion with the model ID in `transcribe()` without claiming native mid-decode interruption.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_ui.py tests/test_recognizer.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/vim2/ui.py app/vim2/recognizer.py tests/test_ui.py tests/test_recognizer.py
git commit -m "fix: reduce redundant preview work"
```

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `REQUIREMENTS.md`

**Interfaces:**
- Documents: `preview_window_seconds`, independent overlap semantics, timer-paced scheduling, and cooperative CPU cancellation

- [ ] **Step 1: Update user and requirement documentation**

Document the default 8-second bounded window, valid 1–30 range, 5-second overlap purpose, no immediate catch-up inference, unchanged-text UI suppression, and synchronous CPU cancellation limitation.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`

Expected: all tests pass with no warnings introduced by the change.

- [ ] **Step 3: Check formatting and repository diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and only intended files changed.

- [ ] **Step 4: Commit**

```bash
git add README.md REQUIREMENTS.md
git commit -m "docs: document bounded preview controls"
```

- [ ] **Step 5: Request code review**

Review all commits after `c286ff5` against `docs/superpowers/specs/2026-09-19-bounded-live-preview-design.md`, fix all critical/important findings with tests first, and rerun `python -m pytest -q`.
