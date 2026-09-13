from __future__ import annotations

import gc
import os
import threading
from pathlib import Path
from typing import Any

from vim2.models import MODEL_SPECS, ModelId
from vim2.paths import AppPaths


class ModelSwitchError(RuntimeError):
    pass


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
        if torch_module is None:
            import torch

            torch_module = torch
        if model_class is None:
            from qwen_asr import Qwen3ASRModel

            model_class = Qwen3ASRModel
        if bits_config_class is None:
            from transformers import BitsAndBytesConfig

            bits_config_class = BitsAndBytesConfig

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
            spec = MODEL_SPECS[model_id]
            kwargs: dict[str, object] = {
                "device_map": "cuda:0",
                "dtype": self._torch.float16,
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

    def transcribe(self, path: Path, model_id: ModelId) -> str:
        with self._lock:
            if self._model is None or self._loaded_model is not model_id:
                raise RuntimeError(
                    f"{MODEL_SPECS[model_id].display_name} is not loaded"
                )
            results = self._model.transcribe(
                audio=str(path),
                language=None,
            )
            if not results:
                raise RuntimeError("Qwen3-ASR returned no recognition result")
            text = results[0].text
            if not isinstance(text, str):
                raise RuntimeError("Qwen3-ASR returned an invalid text result")
            return text
