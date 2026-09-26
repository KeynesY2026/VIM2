# Bounded Live Preview Design

## Goal

Keep long-running recording preview work bounded and responsive: preview only a configurable recent window (8 seconds by default), avoid copying the complete recording for every preview, prevent an overloaded recognizer from running continuously, avoid redundant UI repaints, and make CPU cancellation behavior explicit.

## Current problem

The current implementation has two different duration controls:

- `tail_overlap_seconds`, configurable and currently 5 seconds, preserves audio before a stable transcript checkpoint so text can be merged safely.
- `MAX_PREVIEW_SECONDS`, hard-coded to 12 seconds, caps model input when the stable checkpoint cannot provide a shorter usable tail.

Model input is bounded, but `AudioRecorder.snapshot()` still concatenates every captured chunk before `VoiceSession` slices the recent window. The scheduler also immediately starts its one pending preview when a slower preview completes. When inference takes longer than the 1-second request interval, this creates a continuous decode loop. Finally, the UI repaints for every successful result even when the text did not change.

The sherpa-onnx CPU API used here exposes a synchronous `decode_stream()` operation and no cancellation callback. A Python event cannot safely interrupt that native call.

## Configuration

Add `preview_window_seconds` to `Settings` and `config/settings.json`.

- Default: `8`
- Valid range: integer from `1` through `30`
- `tail_overlap_seconds` remains independently configurable.
- If the requested overlap reaches beyond the bounded snapshot, the snapshot start is the hard boundary. The preview window always wins as the resource limit.

The hard-coded `MAX_PREVIEW_SECONDS` constant is removed from session behavior. Documentation must distinguish the preview window from the overlap.

## Audio snapshot model

Extend `AudioArtifact` with an absolute `start_frame`, defaulting to zero. Add an `end_frame` property equal to `start_frame + frame_count`.

Change the recorder preview interface to:

```python
snapshot(max_seconds: int | None = None) -> AudioArtifact
```

For a bounded snapshot, `AudioRecorder` walks captured chunks backward until it has enough frames, concatenates only those chunks, trims the first selected chunk as needed, and sets `start_frame` to the absolute frame offset of the returned samples. It must not concatenate the complete recording first.

`stop()` continues returning the complete recording with `start_frame == 0`. `slice_from()` continues taking an artifact-relative frame index and propagates the resulting absolute `start_frame`.

This keeps preview copy and inference cost proportional to `preview_window_seconds`, not total recording duration.

## Stable-prefix audio cursor

`StableCheckpoint.frame_count` represents an absolute captured-audio position. Preview observations use `artifact.end_frame`, not the bounded artifact's local length.

For each preview:

1. Request at most `preview_window_seconds` from the recorder.
2. If a stable checkpoint exists inside the snapshot, begin at `checkpoint.frame_count - tail_overlap_seconds`, clipped to the snapshot boundary.
3. Otherwise recognize the complete bounded snapshot.
4. On a successful exact-anchor merge, advance the stable checkpoint using the snapshot's absolute end frame.
5. If no safe merge is possible, show only the bounded recognition result and do not advance the checkpoint.

This is intentionally conservative. The recognizers do not provide sentence timestamps, so the checkpoint is a safe approximation at a completed snapshot boundary rather than a claimed exact sentence-end timestamp. Already committed history is excluded except for the configured overlap needed to recover the merge anchor.

Final recognition retains the existing optimized-tail behavior and complete-audio fallback. Complete fallback is required for correctness when no reliable checkpoint or merge exists.

## Preview scheduling

The 1-second timer remains a request cadence, not a promise to execute or display once per second.

- At most one preview inference runs at a time.
- Timer requests received while inference is busy mark the result as stale but do not enqueue runnable work.
- When the current inference completes, no replacement starts immediately.
- The next timer tick starts a fresh snapshot containing the latest audio.
- Stop and cancel clear stale/pending state.

This intentionally introduces up to one timer interval of idle time after an overloaded inference. It prevents the current back-to-back decode loop while preserving fresh latest-state sampling.

## UI publication

The view stores the latest displayed preview. `show_preview()` returns without repainting when the new text equals the displayed text. Inference cadence and UI publication cadence are therefore independent: a timer may request work, and work may complete, without forcing a visible refresh.

## CPU cancellation

Keep cooperative cancellation around sherpa-onnx:

- Check cancellation before creating/feeding/decoding a stream.
- Check cancellation after `accept_waveform()` and after `decode_stream()`.
- Never start pending preview work after stop or cancel.
- Bound normal cancellation latency by the configurable preview window, default 8 seconds.

This design does not claim that `decode_stream()` itself is interruptible. Python threads must not be force-terminated. A process-isolated hard-cancel backend is out of scope because killing it would also discard the loaded CPU model and impose reload and IPC costs.

## Logging and diagnostics

`VoiceSession` logs the selected preview input duration and absolute start/end frames before recognition. The CPU recognizer adds a completion log matching the existing GPU completion log. These diagnostics make bounded preview behavior visible without changing the recognizer interface.

## Error handling

- Snapshot requests reject non-positive bounds.
- Existing recognition errors continue through the current warning/retry paths.
- A cancelled preview does not warn the user.
- Final recognition and retry semantics remain unchanged.
- Audio capture continues while snapshots are assembled; expensive concatenation remains outside the capture lock.

## Tests

Add or update tests proving:

1. Configuration defaults to 8 seconds, persists the value, and rejects non-integer/out-of-range values.
2. A long recorder snapshot copies only the requested tail and reports correct absolute frame bounds.
3. Repeated bounded snapshots remain bounded as total captured duration grows.
4. Session preview sends no more than the configured window to the recognizer.
5. Stable checkpoints use absolute frame positions with bounded snapshots.
6. Busy timer requests do not trigger an immediate follow-up inference on completion; a later timer tick does.
7. Stop/cancel discards stale preview demand.
8. Identical preview text does not repaint the overlay.
9. CPU cancellation is checked before and after synchronous decode without claiming mid-decode interruption.
10. Preview diagnostics contain duration and absolute frame bounds, and successful CPU recognition emits a completion log.
11. Existing final-tail merge, full fallback, retry, recording, and model tests continue to pass.
