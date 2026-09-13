from pathlib import Path

from vim2.models import MODEL_SPECS, ModelId
from vim2.paths import AppPaths
from vim2.preflight import PreflightChecker


def _create_minimal_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ):
        (model_dir / name).write_text("{}", encoding="utf-8")


def test_preflight_reports_missing_selected_model(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    checker = PreflightChecker(paths, python_version=(3, 13, 0))

    result = checker.check(
        ModelId.FAST, check_cuda=False, check_runtime=False
    )

    assert not result.ok
    assert result.errors == (
        f"Model is incomplete: {paths.models_dir / MODEL_SPECS[ModelId.FAST].directory_name}",
        "missing model file: config.json",
        "missing model file: preprocessor_config.json",
        "missing model file: tokenizer_config.json",
        "missing model weights",
    )


def test_preflight_accepts_complete_portable_model_without_cuda_probe(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    _create_minimal_model(
        paths.models_dir / MODEL_SPECS[ModelId.FAST].directory_name
    )

    result = PreflightChecker(paths, python_version=(3, 13, 0)).check(
        ModelId.FAST, check_cuda=False, check_runtime=False
    )

    assert result.ok
    assert result.errors == ()


def test_preflight_rejects_unsupported_python_version(tmp_path: Path) -> None:
    paths = AppPaths.from_root(tmp_path)
    checker = PreflightChecker(paths, python_version=(3, 14, 0))

    result = checker.check(
        ModelId.FAST, check_cuda=False, check_runtime=False
    )

    assert result.errors[0] == (
        "Python 3.14 is unsupported; install Python 3.10 through 3.13."
    )


def test_preflight_reports_missing_portable_dependencies(
    tmp_path: Path,
) -> None:
    paths = AppPaths.from_root(tmp_path)
    _create_minimal_model(
        paths.models_dir / MODEL_SPECS[ModelId.FAST].directory_name
    )
    checker = PreflightChecker(
        paths,
        python_version=(3, 13, 0),
        dependency_finder=lambda name: object() if name == "torch" else None,
    )

    result = checker.check(ModelId.FAST, check_cuda=False)

    assert result.errors == (
        "Portable runtime dependency is missing: PySide6",
        "Portable runtime dependency is missing: qwen_asr",
        "Portable runtime dependency is missing: sounddevice",
        "Portable runtime dependency is missing: bitsandbytes",
        "Portable runtime dependency is missing: numpy",
        "Portable runtime dependency is missing: soundfile",
        "Portable runtime dependency is missing: transformers",
    )
