import builtins
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vim2.audio import AudioArtifact
from vim2.models import MODEL_SPECS, ModelId
from vim2.paths import AppPaths
from vim2.recognizer import (
    ModelSwitchError,
    QwenRecognizer,
    enforce_offline_environment,
)


class FakeCuda:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.empty_cache_calls = 0
        self.ipc_collect_calls = 0

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def ipc_collect(self) -> None:
        self.ipc_collect_calls += 1


class FakeTorch:
    float16 = "float16"

    def __init__(self) -> None:
        self.cuda = FakeCuda()


class FakeModel:
    def __init__(self, text: str = "recognized") -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(text=self.text)]


class FakeLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.models: list[FakeModel] = []
        self.fail_for: str | None = None

    def from_pretrained(self, path: str, **kwargs) -> FakeModel:
        self.calls.append((path, kwargs))
        if path == self.fail_for:
            raise RuntimeError("load failed")
        model = FakeModel()
        self.models.append(model)
        return model


class FakeBitsConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _recognizer(tmp_path: Path):
    paths = AppPaths.from_root(tmp_path)
    torch = FakeTorch()
    loader = FakeLoader()
    recognizer = QwenRecognizer(
        paths,
        torch_module=torch,
        model_class=loader,
        bits_config_class=FakeBitsConfig,
    )
    return recognizer, loader, torch


def test_fast_model_loads_local_fp16_weights(tmp_path: Path) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)

    recognizer.load(ModelId.FAST)

    expected = (
        tmp_path.resolve()
        / ".models"
        / MODEL_SPECS[ModelId.FAST].directory_name
    )
    assert loader.calls == [
        (
            str(expected),
            {
                "device_map": "cuda:0",
                "dtype": "float16",
                "max_inference_batch_size": 1,
                "max_new_tokens": 512,
            },
        )
    ]
    assert recognizer.loaded_model is ModelId.FAST


def test_accurate_model_uses_int8_quantization(tmp_path: Path) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)

    recognizer.load(ModelId.ACCURATE)

    kwargs = loader.calls[0][1]
    quantization = kwargs["quantization_config"]
    assert isinstance(quantization, FakeBitsConfig)
    assert quantization.kwargs == {"load_in_8bit": True}


def test_transcribe_uses_loaded_model_and_auto_language(tmp_path: Path) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    samples = np.array([0.1, -0.2], dtype=np.float32)
    artifact = AudioArtifact(samples=samples, sample_rate=16_000)

    result = recognizer.transcribe(artifact, ModelId.FAST)

    assert result == "recognized"
    audio, sample_rate = loader.models[0].calls[0]["audio"]
    assert audio is samples
    assert sample_rate == 16_000
    assert loader.models[0].calls[0]["language"] is None


def test_transcribe_accepts_existing_audio_path_for_benchmarks(
    tmp_path: Path,
) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    audio_path = tmp_path / "sample.wav"

    result = recognizer.transcribe(audio_path, ModelId.FAST)

    assert result == "recognized"
    assert loader.models[0].calls == [
        {"audio": str(audio_path), "language": None}
    ]


def test_unload_releases_cuda_resources(tmp_path: Path) -> None:
    recognizer, _, torch = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)

    recognizer.unload()

    assert recognizer.loaded_model is None
    assert torch.cuda.empty_cache_calls == 1
    assert torch.cuda.ipc_collect_calls == 1


def test_failed_switch_restores_previous_model(tmp_path: Path) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    failed_path = (
        tmp_path.resolve()
        / ".models"
        / MODEL_SPECS[ModelId.ACCURATE].directory_name
    )
    loader.fail_for = str(failed_path)

    with pytest.raises(ModelSwitchError, match="restored"):
        recognizer.switch(ModelId.ACCURATE)

    assert recognizer.loaded_model is ModelId.FAST


def test_transcribe_rejects_model_other_than_resident_model(
    tmp_path: Path,
) -> None:
    recognizer, _, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)

    with pytest.raises(RuntimeError, match="not loaded"):
        recognizer.transcribe(
            AudioArtifact(np.zeros(1, dtype=np.float32), 16_000),
            ModelId.ACCURATE,
        )


def test_offline_environment_disables_implicit_model_downloads(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    enforce_offline_environment()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_constructor_does_not_import_model_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    original_import = builtins.__import__

    def reject_runtime_import(name, *args, **kwargs):
        if name in {"torch", "qwen_asr", "transformers"}:
            raise AssertionError(f"eager runtime import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_runtime_import)

    QwenRecognizer(AppPaths.from_root(tmp_path))


def test_load_reports_unavailable_cuda_before_loading_model(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    torch = FakeTorch()
    torch.cuda.available = False
    loader = FakeLoader()
    recognizer = QwenRecognizer(
        paths,
        torch_module=torch,
        model_class=loader,
        bits_config_class=FakeBitsConfig,
    )

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        recognizer.load(ModelId.FAST)

    assert loader.calls == []
