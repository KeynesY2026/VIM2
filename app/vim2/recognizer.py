from __future__ import annotations

import gc
import os
import threading
from pathlib import Path
from typing import Any

from vim2.audio import AudioArtifact
from vim2.models import MODEL_SPECS, ModelId
from vim2.paths import AppPaths


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
    ) -> None:
        enforce_offline_environment()
        self._paths = paths
        self._torch = torch_module
        self._model_class = model_class
        self._bits_config_class = bits_config_class
        self._model: Any | None = None
        self._loaded_model: ModelId | None = None
        self._lock = threading.RLock()

    @property
    def loaded_model(self) -> ModelId | None:
        return self._loaded_model

    def load(self, model_id: ModelId) -> None:
        with self._lock:
            if self._loaded_model is model_id:
                return
            if self._model is not None:
                raise RuntimeError(
                    "Unload the resident model before loading another model"
                )
            self.initialize_runtime()
            if not self._torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable; install a compatible NVIDIA driver "
                    "and confirm that the GPU is enabled."
                )
            spec = MODEL_SPECS[model_id]
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
            self._loaded_model = model_id

    def initialize_runtime(self) -> None:
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
            self._model = None
            self._loaded_model = None
            gc.collect()
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()

    def switch(self, model_id: ModelId) -> None:
        with self._lock:
            previous = self._loaded_model
            if previous is model_id:
                return
            self.unload()
            try:
                self.load(model_id)
            except (OSError, RuntimeError, ValueError, MemoryError) as switch_error:
                if previous is None:
                    raise ModelSwitchError(
                        f"Failed to load {MODEL_SPECS[model_id].display_name}: "
                        f"{switch_error}"
                    ) from switch_error
                try:
                    self.load(previous)
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
            if cancel_event is not None and cancel_event.is_set():
                raise TranscriptionCancelled()
            if self._model is None or self._loaded_model is not model_id:
                raise RuntimeError(
                    f"{MODEL_SPECS[model_id].display_name} is not loaded"
                )
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
                    audio=model_audio, language=None
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
            return text
