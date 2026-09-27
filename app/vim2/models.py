from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ModelId(StrEnum):
    CPU = "qwen3-asr-0.6b-int8-cpu"
    FAST = "qwen3-asr-0.6b-fp16"
    ACCURATE = "qwen3-asr-1.7b-int8"


class ModelBackend(StrEnum):
    QWEN_CUDA = "qwen-cuda"
    SHERPA_ONNX_CPU = "sherpa-onnx-cpu"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: ModelId
    display_name: str
    directory_name: str
    upstream_id: str
    revision: str
    torch_dtype: str
    load_in_8bit: bool
    backend: ModelBackend = ModelBackend.QWEN_CUDA


MODEL_SPECS = {
    ModelId.CPU: ModelSpec(
        model_id=ModelId.CPU,
        display_name="Qwen3-ASR 0.6B INT8 (CPU)",
        directory_name="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        upstream_id=(
            "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
        ),
        revision="68818b2313fe77bd06f6a7c5068ff3ef59d02b8a",
        torch_dtype="",
        load_in_8bit=False,
        backend=ModelBackend.SHERPA_ONNX_CPU,
    ),
    ModelId.FAST: ModelSpec(
        model_id=ModelId.FAST,
        display_name="Qwen3-ASR 0.6B FP16",
        directory_name="Qwen3-ASR-0.6B",
        upstream_id="Qwen/Qwen3-ASR-0.6B",
        revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        torch_dtype="float16",
        load_in_8bit=False,
    ),
    ModelId.ACCURATE: ModelSpec(
        model_id=ModelId.ACCURATE,
        display_name="Qwen3-ASR 1.7B INT8",
        directory_name="Qwen3-ASR-1.7B-INT8",
        upstream_id="Qwen/Qwen3-ASR-1.7B",
        revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        torch_dtype="float16",
        load_in_8bit=True,
    ),
}

_REQUIRED_METADATA = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)

_REQUIRED_SHERPA_FILES = (
    "conv_frontend.onnx",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
)


def validate_model_directory(
    model_dir: Path,
    backend: ModelBackend = ModelBackend.QWEN_CUDA,
) -> list[str]:
    if backend is ModelBackend.SHERPA_ONNX_CPU:
        return [
            f"missing model file: {name}"
            for name in _REQUIRED_SHERPA_FILES
            if not (model_dir / name).is_file()
        ]

    errors = [
        f"missing model file: {name}"
        for name in _REQUIRED_METADATA
        if not (model_dir / name).is_file()
    ]
    index_path = model_dir / "model.safetensors.index.json"
    single_weight = model_dir / "model.safetensors"

    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files = sorted(set(index["weight_map"].values()))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"invalid model index: {exc}")
        else:
            errors.extend(
                f"missing model file: {name}"
                for name in weight_files
                if not (model_dir / name).is_file()
            )
    elif not single_weight.is_file():
        errors.append("missing model weights")

    return errors
