from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_audio_recorder_without_numpy():
    module_name = "_vim2_audio_lifecycle_test_target"
    module_path = Path(__file__).resolve().parents[1] / "app" / "vim2" / "audio.py"
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_numpy.float32 = object()
    module = importlib.util.module_from_spec(module_spec)
    prior_numpy = sys.modules.get("numpy")
    sys.modules[module_name] = module
    sys.modules["numpy"] = fake_numpy
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if prior_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = prior_numpy
    return module.AudioRecorder


AudioRecorder = _load_audio_recorder_without_numpy()


class _FailingStartStream:
    def __init__(self) -> None:
        self.closed = False

    def start(self) -> None:
        raise ValueError("microphone start failed")

    def close(self) -> None:
        self.closed = True


class _SoundDevice:
    def __init__(self) -> None:
        self.stream = _FailingStartStream()

    def InputStream(self, **kwargs) -> _FailingStartStream:
        del kwargs
        return self.stream


class AudioRecorderLifecycleTests(unittest.TestCase):
    def test_stream_is_closed_when_start_fails_after_creation(self) -> None:
        backend = _SoundDevice()
        recorder = AudioRecorder(sounddevice=backend)

        with self.assertRaisesRegex(RuntimeError, "microphone start failed"):
            recorder.start()

        self.assertTrue(backend.stream.closed)


if __name__ == "__main__":
    unittest.main()
