from pathlib import Path
from types import SimpleNamespace

import pytest

import vim2.application as application_module
import vim2.main as main_module
from vim2.main import run
from vim2.models import MODEL_SPECS, ModelId


@pytest.fixture(autouse=True)
def isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))


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


def test_desktop_startup_defers_cuda_probe_until_after_tray_creation(
    tmp_path: Path, monkeypatch
) -> None:
    checks: list[tuple[ModelId, bool, bool]] = []

    class RecordingChecker:
        def __init__(self, paths) -> None:
            del paths

        def check(
            self,
            selected_model: ModelId,
            *,
            check_cuda: bool,
            check_runtime: bool,
        ):
            checks.append((selected_model, check_cuda, check_runtime))
            return SimpleNamespace(ok=True, errors=())

    monkeypatch.setattr(main_module, "PreflightChecker", RecordingChecker)
    monkeypatch.setattr(
        application_module,
        "run_desktop_application",
        lambda paths, settings: 0,
    )

    assert run(["--root", str(tmp_path), "--skip-runtime-check"]) == 0
    assert checks == [(ModelId.FAST, False, False)]
