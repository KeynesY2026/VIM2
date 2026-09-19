from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path
from typing import Any

from vim2.audio import AudioArtifact
from vim2.hotwords import HotwordRepository, HotwordSnapshot
from vim2.models import MODEL_SPECS, ModelBackend, ModelId, ModelSpec
from vim2.paths import AppPaths


logger = logging.getLogger(__name__)


class ModelSwitchError(RuntimeError):
    pass


class TranscriptionCancelled(RuntimeError):
    pass


class _CancellationStoppingCriteria:
    def __init__(self, cancel_event: threading.Event) -> None:
        self._cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs):
        del scores, kwargs
        cancelled = self._cancel_event.is_set()
        if input_ids is None:
            return cancelled
        return input_ids.new_full(
            (input_ids.shape[0],), cancelled
        ).bool()


def enforce_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


class QwenRecognizer:
    def __init__(
        self,
        paths: AppPaths,
        *,
        torch_module: Any | None = None,
        model_class: Any | None = None,
        bits_config_class: Any | None = None,
        sherpa_module: Any | None = None,
    ) -> None:
        enforce_offline_environment()
        self._paths = paths
        self._torch = torch_module
        self._model_class = model_class
        self._bits_config_class = bits_config_class
        self._sherpa = sherpa_module
        self._model: Any | None = None
        self._loaded_model: ModelId | None = None
        self._lock = threading.RLock()
        self._hotword_repository = HotwordRepository(paths.hotwords_file)
        self._hotwords = HotwordSnapshot(())

    @property
    def loaded_model(self) -> ModelId | None:
        return self._loaded_model

    @property
    def hotwords(self) -> HotwordSnapshot:
        return self._hotwords

    def load(self, model_id: ModelId) -> None:
        with self._lock:
            if self._loaded_model is model_id:
                return
            if self._model is not None:
                raise RuntimeError(
                    "Unload the resident model before loading another model"
                )
            hotwords = self._hotword_repository.load()
            self._load_model(model_id, hotwords)

    def _load_model(
        self, model_id: ModelId, hotwords: HotwordSnapshot
    ) -> None:
        spec = MODEL_SPECS[model_id]
        logger.info("Loading model %s via %s", model_id, spec.backend)
        self.initialize_runtime(model_id)
        if spec.backend is ModelBackend.SHERPA_ONNX_CPU:
            self._load_sherpa_model(spec, hotwords)
        else:
            self._load_qwen_model(spec)
        self._hotwords = hotwords
        self._loaded_model = model_id
        logger.info("Model loaded: %s", model_id)

    def _load_qwen_model(self, spec: ModelSpec) -> None:
        if not self._torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; install a compatible NVIDIA driver "
                "and confirm that the GPU is enabled."
            )
        kwargs: dict[str, object] = {
            "device_map": "cuda:0",
            "dtype": self._torch.float16,
            "attn_implementation": "sdpa",
            "max_inference_batch_size": 1,
            "max_new_tokens": 512,
        }
        if spec.load_in_8bit:
            kwargs["quantization_config"] = self._bits_config_class(
                load_in_8bit=True
            )
        model_path = self._paths.models_dir / spec.directory_name
        self._model = self._model_class.from_pretrained(
            str(model_path), **kwargs
        )

    def _load_sherpa_model(
        self, spec: ModelSpec, hotwords: HotwordSnapshot
    ) -> None:
        self._model = self._create_sherpa_model(spec, hotwords)

    def _create_sherpa_model(
        self, spec: ModelSpec, hotwords: HotwordSnapshot
    ) -> Any:
        model_path = self._paths.models_dir / spec.directory_name
        return self._sherpa.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(model_path / "conv_frontend.onnx"),
            encoder=str(model_path / "encoder.int8.onnx"),
            decoder=str(model_path / "decoder.int8.onnx"),
            tokenizer=str(model_path / "tokenizer"),
            num_threads=2,
            provider="cpu",
            max_total_len=1024,
            max_new_tokens=512,
            hotwords=hotwords.cpu_hotwords,
        )

    def reload_hotwords(self) -> HotwordSnapshot:
        with self._lock:
            hotwords = self._hotword_repository.load()
            if (
                self._loaded_model is not None
                and MODEL_SPECS[self._loaded_model].backend
                is ModelBackend.SHERPA_ONNX_CPU
            ):
                self._model = self._create_sherpa_model(
                    MODEL_SPECS[self._loaded_model], hotwords
                )
            self._hotwords = hotwords
            return hotwords

    def initialize_runtime(self, model_id: ModelId = ModelId.FAST) -> None:
        if MODEL_SPECS[model_id].backend is ModelBackend.SHERPA_ONNX_CPU:
            if self._sherpa is None:
                import sherpa_onnx

                self._sherpa = sherpa_onnx
            return
        if self._torch is None:
            import torch

            self._torch = torch
        if self._model_class is None:
            from qwen_asr import Qwen3ASRModel

            self._model_class = Qwen3ASRModel
        if self._bits_config_class is None:
            from transformers import BitsAndBytesConfig

            self._bits_config_class = BitsAndBytesConfig

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            loaded_model = self._loaded_model
            logger.info("Unloading model %s", loaded_model)
            self._model = None
            self._loaded_model = None
            gc.collect()
            if (
                loaded_model is not None
                and MODEL_SPECS[loaded_model].backend
                is ModelBackend.QWEN_CUDA
            ):
                self._torch.cuda.empty_cache()
                self._torch.cuda.ipc_collect()

    def switch(self, model_id: ModelId) -> None:
        with self._lock:
            previous = self._loaded_model
            if previous is model_id:
                return
            try:
                hotwords = self._hotword_repository.load()
            except (
                OSError,
                RuntimeError,
                ValueError,
                MemoryError,
            ) as switch_error:
                raise ModelSwitchError(
                    f"Failed to load {MODEL_SPECS[model_id].display_name}: "
                    f"{switch_error}"
                ) from switch_error
            previous_hotwords = self._hotwords
            self.unload()
            try:
                self._load_model(model_id, hotwords)
            except (OSError, RuntimeError, ValueError, MemoryError) as switch_error:
                if previous is None:
                    raise ModelSwitchError(
                        f"Failed to load {MODEL_SPECS[model_id].display_name}: "
                        f"{switch_error}"
                    ) from switch_error
                try:
                    self._load_model(previous, previous_hotwords)
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    MemoryError,
                ) as restore_error:
                    raise ModelSwitchError(
                        f"Failed to load {MODEL_SPECS[model_id].display_name}; "
                        f"restoring {MODEL_SPECS[previous].display_name} also "
                        f"failed: {restore_error}"
                    ) from switch_error
                raise ModelSwitchError(
                    f"Failed to load {MODEL_SPECS[model_id].display_name}; "
                    f"{MODEL_SPECS[previous].display_name} was restored."
                ) from switch_error

    def transcribe(
        self,
        audio: AudioArtifact | Path,
        model_id: ModelId,
        *,
        cancel_event: threading.Event | None = None,
    ) -> str:
        with self._lock:
            logger.info("Transcription started with model %s", model_id)
            if cancel_event is not None and cancel_event.is_set():
                raise TranscriptionCancelled()
            if self._model is None or self._loaded_model is not model_id:
                raise RuntimeError(
                    f"{MODEL_SPECS[model_id].display_name} is not loaded"
                )
            if (
                MODEL_SPECS[model_id].backend
                is ModelBackend.SHERPA_ONNX_CPU
            ):
                return self._transcribe_sherpa(audio, cancel_event)
            model_audio: tuple[Any, int] | str
            if isinstance(audio, AudioArtifact):
                model_audio = (audio.samples, audio.sample_rate)
            else:
                model_audio = str(audio)
            generation_model = getattr(self._model, "model", None)
            original_generate = None
            if cancel_event is not None and generation_model is not None:
                original_generate = generation_model.generate
                stopping_criterion = _CancellationStoppingCriteria(
                    cancel_event
                )

                def cancellable_generate(*args, **kwargs):
                    stopping_criteria = list(
                        kwargs.pop("stopping_criteria", None) or ()
                    )
                    stopping_criteria.append(stopping_criterion)
                    return original_generate(
                        *args,
                        stopping_criteria=stopping_criteria,
                        **kwargs,
                    )

                generation_model.generate = cancellable_generate
            try:
                results = self._model.transcribe(
                    audio=model_audio,
                    language=None,
                    context=self._hotwords.gpu_context,
                )
            finally:
                if original_generate is not None:
                    generation_model.generate = original_generate
            if cancel_event is not None and cancel_event.is_set():
                raise TranscriptionCancelled()
            if not results:
                raise RuntimeError("Qwen3-ASR returned no recognition result")
            text = results[0].text
            if not isinstance(text, str):
                raise RuntimeError("Qwen3-ASR returned an invalid text result")
            logger.info("Transcription completed with model %s", model_id)
            return text

    def _transcribe_sherpa(
        self,
        audio: AudioArtifact | Path,
        cancel_event: threading.Event | None,
    ) -> str:
        if isinstance(audio, AudioArtifact):
            samples = audio.samples
            sample_rate = audio.sample_rate
        else:
            import soundfile

            samples, sample_rate = soundfile.read(
                str(audio), dtype="float32", always_2d=True
            )
            samples = samples[:, 0]
        stream = self._model.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._model.decode_stream(stream)
        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelled()
        text = stream.result.text
        if not isinstance(text, str):
            raise RuntimeError("sherpa-onnx returned an invalid text result")
        return text
