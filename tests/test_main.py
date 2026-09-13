from pathlib import Path

import vim2.main as main_module
from vim2.main import run
from vim2.models import MODEL_SPECS, ModelId


def _create_minimal_model(root: Path) -> None:
    model_dir = root / ".models" / MODEL_SPECS[ModelId.FAST].directory_name
    model_dir.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ):
        (model_dir / name).write_text("{}", encoding="utf-8")


def test_check_mode_succeeds_for_complete_portable_layout(
    tmp_path: Path, capsys
) -> None:
    _create_minimal_model(tmp_path)

    exit_code = run(
        [
            "--root",
            str(tmp_path),
            "--check",
            "--skip-cuda-check",
            "--skip-runtime-check",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "VIM2 preflight check passed.\n"


def test_check_mode_prints_actionable_errors(tmp_path: Path, capsys) -> None:
    exit_code = run(
        [
            "--root",
            str(tmp_path),
            "--check",
            "--skip-cuda-check",
            "--skip-runtime-check",
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().err
    assert "VIM2 cannot start:" in output
    assert "Model is incomplete:" in output


def test_windowed_startup_reports_errors_in_message_box(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        main_module, "_show_startup_message", messages.append
    )

    exit_code = run(
        [
            "--root",
            str(tmp_path),
            "--windowed",
            "--skip-cuda-check",
            "--skip-runtime-check",
        ]
    )

    assert exit_code == 1
    assert messages and "Model is incomplete:" in messages[0]
    assert capsys.readouterr().err == ""
