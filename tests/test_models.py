from pathlib import Path

from vim2.models import MODEL_SPECS, ModelId, validate_model_directory


def test_model_catalog_matches_release_requirements() -> None:
    fast = MODEL_SPECS[ModelId.FAST]
    accurate = MODEL_SPECS[ModelId.ACCURATE]

    assert fast.directory_name == "Qwen3-ASR-0.6B"
    assert fast.torch_dtype == "float16"
    assert fast.revision == "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
    assert accurate.directory_name == "Qwen3-ASR-1.7B-INT8"
    assert accurate.load_in_8bit is True
    assert accurate.revision == "7278e1e70fe206f11671096ffdd38061171dd6e5"


def test_sharded_model_requires_every_referenced_weight(tmp_path: Path) -> None:
    for name in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )

    errors = validate_model_directory(tmp_path)

    assert errors == ["missing model file: model-00002-of-00002.safetensors"]


def test_single_file_model_is_accepted_when_metadata_exists(tmp_path: Path) -> None:
    for name in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    assert validate_model_directory(tmp_path) == []
