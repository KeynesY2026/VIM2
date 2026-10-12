import builtins
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from vim2.audio import AudioArtifact
from vim2.models import MODEL_SPECS, ModelId
from vim2.paths import AppPaths
from vim2.recognizer import (
    ModelSwitchError,
    QwenRecognizer,
    TranscriptionCancelled,
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
        self.model = FakeGenerationModel()
        self.audio_attentions: list[str] = []
        self.convolution_batches: list[int] = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        self.convolution_batches.append(self.model.thinker.audio_tower.conv_chunksize)
        self.audio_attentions.append(
            self.model.thinker.audio_tower.config._attn_implementation
        )
        self.model.generate()
        return [SimpleNamespace(text=self.text)]


class FakeGenerationModel:
    def __init__(self) -> None:
        self.cancel_on_generate: threading.Event | None = None
        self.cancel_observed = False
        self.thinker = SimpleNamespace(
            audio_tower=SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="sdpa"),
                conv_chunksize=500,
            ),
            model=SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="sdpa")
            ),
        )

    def generate(self, **kwargs) -> None:
        if self.cancel_on_generate is not None:
            self.cancel_on_generate.set()
        stopping_criteria = kwargs.get("stopping_criteria")
        if stopping_criteria is not None:
            self.cancel_observed = any(
                bool(criterion(None, None))
                for criterion in stopping_criteria
            )


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


class FakeAttentionInterface:
    @staticmethod
    def register(name, implementation):
        pass


class FakeSherpaStream:
    def __init__(self) -> None:
        self.waveforms: list[tuple[int, np.ndarray]] = []
        self.result = SimpleNamespace(text="cpu recognized")

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self.waveforms.append((sample_rate, samples))


class FakeSherpaRecognizer:
    def __init__(self) -> None:
        self.streams: list[FakeSherpaStream] = []
        self.decoded: list[FakeSherpaStream] = []

    def create_stream(self) -> FakeSherpaStream:
        stream = FakeSherpaStream()
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream: FakeSherpaStream) -> None:
        self.decoded.append(stream)


class FakeSherpaFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.recognizers: list[FakeSherpaRecognizer] = []
        self.fail_next = False

    def from_qwen3_asr(self, **kwargs) -> FakeSherpaRecognizer:
        self.calls.append(kwargs)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("rebuild failed")
        recognizer = FakeSherpaRecognizer()
        self.recognizers.append(recognizer)
        return recognizer


class FakeSherpaModule:
    def __init__(self) -> None:
        self.OfflineRecognizer = FakeSherpaFactory()


def _recognizer(tmp_path: Path):
    paths = AppPaths.from_root(tmp_path)
    torch = FakeTorch()
    loader = FakeLoader()
    recognizer = QwenRecognizer(
        paths,
        torch_module=torch,
        model_class=loader,
        bits_config_class=FakeBitsConfig,
        attention_interface=FakeAttentionInterface,
    )
    return recognizer, loader, torch


class FakeDraftDecoder:
    def __init__(self, model, torch_module, draft, **kwargs):
        self.model = model
        self.cancel_event = kwargs["cancel_event"]
        self.tokens = ()
        self.forward_calls = 1
        self.verified_tokens = len(draft)
        self.draft = tuple(draft)

    def generate(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TranscriptionCancelled()
        self.tokens = (1, 2, 151645)
        return None


def test_draft_preview_final_and_hotwords_reset_use_same_model(tmp_path, monkeypatch):
    monkeypatch.setattr("vim2.draft.GpuDraftDecoder", FakeDraftDecoder)
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    artifact = AudioArtifact(np.zeros(16_000, dtype=np.float32), 16_000)
    generation_model = loader.models[0].model
    original_generate = generation_model.generate
    assert recognizer.transcribe_preview(artifact, ModelId.FAST) == "recognized"
    assert recognizer._draft_tokens == (1, 2, 151645)
    assert generation_model.generate == original_generate
    assert recognizer.transcribe_final(artifact, ModelId.FAST) == "recognized"
    assert generation_model.generate == original_generate
    recognizer.reload_hotwords()
    assert recognizer._draft_tokens == ()
    recognizer.transcribe_preview(artifact, ModelId.FAST)
    recognizer.unload()
    assert recognizer._draft_tokens == ()


def test_failed_preview_keeps_last_draft_and_restores_generator(tmp_path, monkeypatch):
    monkeypatch.setattr("vim2.draft.GpuDraftDecoder", FakeDraftDecoder)
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    artifact = AudioArtifact(np.zeros(16_000, dtype=np.float32), 16_000)
    recognizer.transcribe_preview(artifact, ModelId.FAST)
    original_generate = loader.models[0].model.generate
    event = threading.Event()
    event.set()
    with pytest.raises(TranscriptionCancelled):
        recognizer.transcribe_preview(artifact, ModelId.FAST, cancel_event=event)
    assert recognizer._draft_tokens == (1, 2, 151645)
    assert loader.models[0].model.generate == original_generate
    loader.models[0].text = None
    with pytest.raises(RuntimeError, match="invalid text"):
        recognizer.transcribe_preview(artifact, ModelId.FAST)
    assert recognizer._draft_tokens == (1, 2, 151645)
    assert loader.models[0].model.generate == original_generate


def test_draft_capability_is_gated_to_validated_gpu_model(tmp_path):
    recognizer, _, _ = _recognizer(tmp_path)
    assert not recognizer.supports_live_draft
    recognizer.load(ModelId.ACCURATE)
    assert not recognizer.supports_live_draft
    with pytest.raises(RuntimeError, match="not enabled"):
        recognizer.transcribe_final(
            AudioArtifact(np.zeros(16_000, dtype=np.float32), 16_000),
            ModelId.ACCURATE,
        )


def test_cpu_does_not_enable_unverified_batched_draft_decoding(tmp_path):
    recognizer = QwenRecognizer(
        AppPaths.from_root(tmp_path), sherpa_module=FakeSherpaModule()
    )
    recognizer.load(ModelId.CPU)
    assert not recognizer.supports_live_draft
    with pytest.raises(RuntimeError, match="not enabled"):
        recognizer.transcribe_final(
            AudioArtifact(np.zeros(16_000, dtype=np.float32), 16_000),
            ModelId.CPU,
        )


@pytest.mark.parametrize("model_id", [ModelId.FAST, ModelId.ACCURATE])
@pytest.mark.parametrize("duration,original,expected", [(30, 500, 500), (60, 500, 16), (60, 8, 8)])
def test_long_gpu_convolution_batch_is_bounded_and_restored(
    tmp_path, model_id, duration, original, expected
):
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(model_id)
    model = loader.models[0]
    model.model.thinker.audio_tower.conv_chunksize = original
    artifact = AudioArtifact(np.zeros(duration * 16_000, dtype=np.float32), 16_000)
    recognizer.transcribe(artifact, model_id)
    assert model.convolution_batches == [
        expected if model_id is ModelId.FAST else original
    ]
    assert model.model.thinker.audio_tower.conv_chunksize == original
    assert model.model.thinker.model.config._attn_implementation == "sdpa"


def test_long_gpu_convolution_budget_restores_after_model_failure(tmp_path):
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    model = loader.models[0]

    def fail(**kwargs):
        assert model.model.thinker.audio_tower.conv_chunksize == 16
        raise RuntimeError("model failed")

    model.transcribe = fail
    with pytest.raises(RuntimeError, match="model failed"):
        recognizer.transcribe(
            AudioArtifact(np.zeros(60 * 16_000, dtype=np.float32), 16_000),
            ModelId.FAST,
        )
    assert model.model.thinker.audio_tower.conv_chunksize == 500


def test_gpu_text_attention_registers_causal_mask_for_cached_verification(tmp_path):
    from transformers import Qwen3Config
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from vim2.attention import TEXT_SDPA_IMPLEMENTATION
    import torch

    recognizer, _, _ = _recognizer(tmp_path)
    recognizer.initialize_runtime(ModelId.FAST)
    config = Qwen3Config(num_hidden_layers=1)
    config._attn_implementation = TEXT_SDPA_IMPLEMENTATION
    cache = DynamicCache(config=config)
    cache.update(torch.zeros(1, 2, 3, 8), torch.zeros(1, 2, 3, 8), 0)
    mask = create_causal_mask(
        config=config, input_embeds=torch.zeros(1, 2, 16),
        attention_mask=torch.ones(1, 5), cache_position=torch.arange(3, 5),
        past_key_values=cache, position_ids=torch.tensor([[3, 4]]),
    )
    assert mask is not None
    assert mask.shape[-2:] == (2, 5)
    assert mask[0, 0, 0, 3]
    assert not mask[0, 0, 0, 4]
    assert mask[0, 0, 1, 4]


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
                "attn_implementation": "sdpa",
                "max_inference_batch_size": 1,
                "max_new_tokens": 512,
            },
        )
    ]
    assert recognizer.loaded_model is ModelId.FAST


def test_accurate_model_uses_quantization(tmp_path: Path) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)

    recognizer.load(ModelId.ACCURATE)

    kwargs = loader.calls[0][1]
    quantization = kwargs["quantization_config"]
    assert isinstance(quantization, FakeBitsConfig)
    assert quantization.kwargs == {"load_in_8bit": True}


@pytest.mark.parametrize("model_id", [ModelId.FAST, ModelId.ACCURATE])
def test_gpu_audio_attention_preserves_window_boundaries(tmp_path, model_id):
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(model_id)

    thinker = loader.models[0].model.thinker
    recognizer.transcribe(
        AudioArtifact(np.zeros(31, dtype=np.float32), 1), model_id
    )
    assert loader.models[0].audio_attentions == ["vim2_audio_sdpa"]
    assert thinker.audio_tower.config._attn_implementation == "sdpa"
    assert thinker.model.config._attn_implementation == "sdpa"


def test_loaded_audio_attention_does_not_mix_distinct_windows(tmp_path):
    torch = pytest.importorskip("torch")
    backend = pytest.importorskip(
        "qwen_asr.core.transformers_backend.modeling_qwen3_asr"
    )
    transformers = pytest.importorskip("transformers")
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer._attention_interface = transformers.AttentionInterface
    recognizer.load(ModelId.FAST)
    config = loader.models[0].model.thinker.audio_tower.config
    config.d_model = 1
    config.encoder_attention_heads = 1
    config.attention_dropout = 0.0
    attention = backend.Qwen3ASRAudioAttention(config).eval()
    results = []
    with torch.no_grad():
        for projection in (
            attention.q_proj, attention.k_proj,
            attention.v_proj, attention.out_proj,
        ):
            projection.bias.zero_()
            projection.weight.zero_()
        attention.v_proj.weight.fill_(1)
        attention.out_proj.weight.fill_(1)
        hidden = torch.tensor([[1.0], [1.0], [10.0], [10.0]])
        boundaries = torch.tensor([0, 2, 4], dtype=torch.int32)
    def generate(**kwargs):
        with torch.no_grad():
            results.append(attention(hidden, cu_seqlens=boundaries))

    loader.models[0].model.generate = generate
    recognizer.transcribe(
        AudioArtifact(np.zeros(31, dtype=np.float32), 1), ModelId.FAST
    )
    assert len(results) == 1
    assert torch.equal(results[0], hidden)


@pytest.mark.parametrize("duration", [1, 12, 20, 30])
def test_short_gpu_audio_keeps_original_attention(tmp_path, duration):
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    recognizer.transcribe(
        AudioArtifact(np.zeros(duration, dtype=np.float32), 1), ModelId.FAST
    )
    assert loader.models[0].audio_attentions == ["sdpa"]


def test_long_gpu_attention_is_restored_after_failure(tmp_path):
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)

    def fail(**kwargs):
        raise RuntimeError("decode failure")

    loader.models[0].model.generate = fail
    with pytest.raises(RuntimeError, match="decode failure"):
        recognizer.transcribe(
            AudioArtifact(np.zeros(31, dtype=np.float32), 1), ModelId.FAST
        )
    assert loader.models[0].audio_attentions == ["vim2_audio_sdpa"]
    assert loader.models[0].model.thinker.audio_tower.config._attn_implementation == "sdpa"


def test_cpu_model_uses_sherpa_onnx_without_cuda(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    paths = AppPaths.from_root(tmp_path)
    torch = FakeTorch()
    torch.cuda.available = False
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(
        paths,
        torch_module=torch,
        sherpa_module=sherpa,
    )

    recognizer.load(ModelId.CPU)

    model_dir = (
        tmp_path.resolve()
        / ".models"
        / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    )
    assert sherpa.OfflineRecognizer.calls == [
        {
            "conv_frontend": str(model_dir / "conv_frontend.onnx"),
            "encoder": str(model_dir / "encoder.int8.onnx"),
            "decoder": str(model_dir / "decoder.int8.onnx"),
            "tokenizer": str(model_dir / "tokenizer"),
            "num_threads": 4,
            "provider": "cpu",
            "max_total_len": 4096,
            "max_new_tokens": 512,
            "hotwords": "",
        }
    ]
    assert recognizer.loaded_model is ModelId.CPU


@pytest.mark.parametrize("available,expected", [(1, 1), (2, 2), (4, 4), (8, 4), (64, 4)])
def test_cpu_inference_threads_are_bounded_by_processor_count(
    tmp_path, monkeypatch, available, expected
):
    monkeypatch.setattr(os, "cpu_count", lambda: available)
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(AppPaths.from_root(tmp_path), sherpa_module=sherpa)
    recognizer.load(ModelId.CPU)
    assert sherpa.OfflineRecognizer.calls[0]["num_threads"] == expected


def test_unknown_cpu_count_keeps_existing_threads_and_reports_it(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(AppPaths.from_root(tmp_path), sherpa_module=sherpa)
    recognizer.load(ModelId.CPU)
    assert sherpa.OfflineRecognizer.calls[0]["num_threads"] == 2
    assert "Cannot determine processor count" in caplog.text


def test_cpu_hotword_reload_preserves_bounded_inference_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(AppPaths.from_root(tmp_path), sherpa_module=sherpa)
    recognizer.load(ModelId.CPU)
    recognizer.reload_hotwords()
    assert [call["num_threads"] for call in sherpa.OfflineRecognizer.calls] == [4, 4]


def test_cpu_model_transcribes_in_memory_audio(tmp_path: Path) -> None:
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(
        AppPaths.from_root(tmp_path), sherpa_module=sherpa
    )
    recognizer.load(ModelId.CPU)
    samples = np.array([0.1, -0.2], dtype=np.float32)

    result = recognizer.transcribe(
        AudioArtifact(samples=samples, sample_rate=16_000), ModelId.CPU
    )

    backend = sherpa.OfflineRecognizer.recognizers[0]
    assert result == "cpu recognized"
    assert backend.decoded == backend.streams
    sample_rate, accepted = backend.streams[0].waveforms[0]
    assert sample_rate == 16_000
    assert accepted is samples


def test_cpu_cancellation_after_waveform_skips_decode(tmp_path: Path) -> None:
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(
        AppPaths.from_root(tmp_path), sherpa_module=sherpa
    )
    recognizer.load(ModelId.CPU)
    cancel_event = threading.Event()
    backend = sherpa.OfflineRecognizer.recognizers[0]
    stream = backend.create_stream()
    original_accept = stream.accept_waveform

    def cancel_after_accept(sample_rate, samples) -> None:
        original_accept(sample_rate, samples)
        cancel_event.set()

    stream.accept_waveform = cancel_after_accept
    backend.create_stream = lambda: stream

    with pytest.raises(TranscriptionCancelled):
        recognizer.transcribe(
            AudioArtifact(np.zeros(16_000, dtype=np.float32), 16_000),
            ModelId.CPU,
            cancel_event=cancel_event,
        )

    assert backend.decoded == []


def test_cpu_transcription_logs_completion(tmp_path: Path, caplog) -> None:
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(
        AppPaths.from_root(tmp_path), sherpa_module=sherpa
    )
    recognizer.load(ModelId.CPU)

    with caplog.at_level("INFO", logger="vim2.recognizer"):
        recognizer.transcribe(
            AudioArtifact(np.zeros(1, dtype=np.float32), 16_000),
            ModelId.CPU,
        )

    assert "Transcription completed with model" in caplog.text


def test_unload_cpu_model_does_not_touch_cuda(tmp_path: Path) -> None:
    torch = FakeTorch()
    recognizer = QwenRecognizer(
        AppPaths.from_root(tmp_path),
        torch_module=torch,
        sherpa_module=FakeSherpaModule(),
    )
    recognizer.load(ModelId.CPU)

    recognizer.unload()

    assert recognizer.loaded_model is None
    assert torch.cuda.empty_cache_calls == 0
    assert torch.cuda.ipc_collect_calls == 0


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
    assert loader.models[0].calls[0]["context"] == ""


def test_gpu_transcribe_uses_newline_hotword_context(tmp_path: Path) -> None:
    hotwords_path = tmp_path / "config" / "hotwords.txt"
    hotwords_path.parent.mkdir()
    hotwords_path.write_text("Visual Studio Code\nVIM2\n", encoding="utf-8")
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)

    recognizer.transcribe(
        AudioArtifact(np.zeros(1, dtype=np.float32), 16_000), ModelId.FAST
    )

    assert loader.models[0].calls[0]["context"] == "Visual Studio Code\nVIM2"


def test_cpu_model_uses_comma_separated_hotwords(tmp_path: Path) -> None:
    hotwords_path = tmp_path / "config" / "hotwords.txt"
    hotwords_path.parent.mkdir()
    hotwords_path.write_text("Visual Studio Code\nVIM2\n", encoding="utf-8")
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(
        AppPaths.from_root(tmp_path), sherpa_module=sherpa
    )

    recognizer.load(ModelId.CPU)

    assert sherpa.OfflineRecognizer.calls[0]["hotwords"] == (
        "Visual Studio Code,VIM2"
    )


def test_reload_updates_gpu_context_without_reloading_weights(
    tmp_path: Path,
) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    hotwords_path = tmp_path / "config" / "hotwords.txt"
    hotwords_path.write_text("Qwen\n", encoding="utf-8")

    snapshot = recognizer.reload_hotwords()
    recognizer.transcribe(
        AudioArtifact(np.zeros(1, dtype=np.float32), 16_000), ModelId.FAST
    )

    assert snapshot.entries == ("Qwen",)
    assert len(loader.models) == 1
    assert loader.models[0].calls[0]["context"] == "Qwen"


def test_reload_rebuilds_loaded_cpu_recognizer(tmp_path: Path) -> None:
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(
        AppPaths.from_root(tmp_path), sherpa_module=sherpa
    )
    recognizer.load(ModelId.CPU)
    hotwords_path = tmp_path / "config" / "hotwords.txt"
    hotwords_path.write_text("Qwen\nVIM2\n", encoding="utf-8")

    recognizer.reload_hotwords()

    assert len(sherpa.OfflineRecognizer.recognizers) == 2
    assert sherpa.OfflineRecognizer.calls[-1]["hotwords"] == "Qwen,VIM2"


def test_failed_cpu_reload_preserves_previous_model_and_snapshot(
    tmp_path: Path,
) -> None:
    hotwords_path = tmp_path / "config" / "hotwords.txt"
    hotwords_path.parent.mkdir()
    hotwords_path.write_text("previous\n", encoding="utf-8")
    sherpa = FakeSherpaModule()
    recognizer = QwenRecognizer(
        AppPaths.from_root(tmp_path), sherpa_module=sherpa
    )
    recognizer.load(ModelId.CPU)
    previous_model = recognizer._model
    hotwords_path.write_text("replacement\n", encoding="utf-8")
    sherpa.OfflineRecognizer.fail_next = True

    with pytest.raises(RuntimeError, match="rebuild failed"):
        recognizer.reload_hotwords()

    assert recognizer._model is previous_model
    assert recognizer.hotwords.entries == ("previous",)


def test_transcribe_skips_model_when_already_cancelled(tmp_path: Path) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(TranscriptionCancelled):
        recognizer.transcribe(
            AudioArtifact(np.zeros(1, dtype=np.float32), 16_000),
            ModelId.FAST,
            cancel_event=cancel_event,
        )

    assert loader.models[0].calls == []


def test_transcribe_cancels_during_model_generation(tmp_path: Path) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    cancel_event = threading.Event()
    generation_model = loader.models[0].model
    generation_model.cancel_on_generate = cancel_event

    with pytest.raises(TranscriptionCancelled):
        recognizer.transcribe(
            AudioArtifact(np.zeros(1, dtype=np.float32), 16_000),
            ModelId.FAST,
            cancel_event=cancel_event,
        )

    assert generation_model.cancel_observed


def test_transcribe_accepts_existing_audio_path_for_benchmarks(
    tmp_path: Path,
) -> None:
    recognizer, loader, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    audio_path = tmp_path / "sample.wav"
    import soundfile

    soundfile.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)

    result = recognizer.transcribe(audio_path, ModelId.FAST)

    assert result == "recognized"
    assert loader.models[0].calls == [
        {"audio": str(audio_path), "language": None, "context": ""}
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


def test_invalid_hotword_refresh_does_not_destroy_model_on_gpu_switch(
    tmp_path: Path,
) -> None:
    hotwords_path = tmp_path / "config" / "hotwords.txt"
    hotwords_path.parent.mkdir()
    hotwords_path.write_text("previous\n", encoding="utf-8")
    recognizer, _, _ = _recognizer(tmp_path)
    recognizer.load(ModelId.FAST)
    previous_model = recognizer._model
    hotwords_path.write_text("invalid,comma\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains an ASCII comma"):
        recognizer.reload_hotwords()
    with pytest.raises(ModelSwitchError, match="contains an ASCII comma"):
        recognizer.switch(ModelId.ACCURATE)

    assert recognizer.loaded_model is ModelId.FAST
    assert recognizer._model is previous_model
    assert recognizer.hotwords.entries == ("previous",)


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
        if name in {"torch", "qwen_asr", "sherpa_onnx", "transformers"}:
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
        attention_interface=FakeAttentionInterface,
    )

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        recognizer.load(ModelId.FAST)

    assert loader.calls == []
