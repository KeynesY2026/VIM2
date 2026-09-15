import json
from pathlib import Path

import pytest

from vim2.config import (
    DEFAULT_HOTKEY,
    DEFAULT_PREVIEW_INTERVAL_MS,
    DEFAULT_TAIL_OVERLAP_SECONDS,
    MacPasteShortcut,
    Settings,
    SettingsRepository,
)
from vim2.models import ModelId


def _install_shortcut_atomic_io_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Path], list[Path], list[tuple[Path, Path]]]:
    atomic_targets: list[Path] = []
    write_targets: list[Path] = []
    replace_targets: list[tuple[Path, Path]] = []
    original_write_atomic = SettingsRepository._write_atomic
    original_write_text = Path.write_text
    original_replace = Path.replace

    def spy_write_atomic(path: Path, content: str) -> None:
        atomic_targets.append(path)
        original_write_atomic(path, content)

    def spy_write_text(path: Path, content: str, *args, **kwargs) -> int:
        write_targets.append(path)
        return original_write_text(path, content, *args, **kwargs)

    def spy_replace(path: Path, target: Path) -> Path:
        replace_targets.append((path, target))
        return original_replace(path, target)

    monkeypatch.setattr(
        SettingsRepository,
        "_write_atomic",
        staticmethod(spy_write_atomic),
    )
    monkeypatch.setattr(Path, "write_text", spy_write_text)
    monkeypatch.setattr(Path, "replace", spy_replace)
    return atomic_targets, write_targets, replace_targets


def _assert_only_settings_json_was_atomically_replaced(
    *,
    settings_path: Path,
    atomic_targets: list[Path],
    write_targets: list[Path],
    replace_targets: list[tuple[Path, Path]],
) -> None:
    settings_temporary_path = settings_path.with_suffix(
        settings_path.suffix + ".tmp"
    )
    assert atomic_targets == [settings_path], (
        f"shortcut atomic targets must be settings.json only: {atomic_targets}"
    )
    assert write_targets == [settings_temporary_path], (
        f"shortcut temp writes must be settings.json.tmp only: {write_targets}"
    )
    assert replace_targets == [(settings_temporary_path, settings_path)], (
        "shortcut replace must target settings.json only: "
        f"{replace_targets}"
    )


def test_missing_configuration_uses_documented_defaults(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")

    settings = repository.load()

    assert settings == Settings(
        selected_model=ModelId.FAST,
        hotkey=DEFAULT_HOTKEY,
        max_recording_seconds=90,
        preview_interval_ms=DEFAULT_PREVIEW_INTERVAL_MS,
        tail_overlap_seconds=DEFAULT_TAIL_OVERLAP_SECONDS,
        macos_paste_shortcut=MacPasteShortcut.COMMAND_V,
    )


def test_settings_round_trip_in_portable_config_directory(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")
    settings = Settings(
        selected_model=ModelId.ACCURATE,
        hotkey="LeftCtrl+RightAlt",
        max_recording_seconds=45,
        preview_interval_ms=500,
        tail_overlap_seconds=7,
        macos_paste_shortcut=MacPasteShortcut.CONTROL_V,
    )

    repository.save(settings)

    assert repository.load() == settings
    assert (tmp_path / "config" / "hotkey.conf").read_text(
        encoding="utf-8"
    ) == "LeftCtrl+RightAlt\n"
    persisted = json.loads(
        (tmp_path / "config" / "settings.json").read_text(encoding="utf-8")
    )
    assert persisted == {
        "macos_paste_shortcut": "control-v",
        "max_recording_seconds": 45,
        "preview_interval_ms": 500,
        "selected_model": "qwen3-asr-1.7b-int8",
        "tail_overlap_seconds": 7,
    }


def test_cpu_model_can_be_selected(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")
    settings = Settings(selected_model=ModelId.CPU)

    repository.save(settings)

    assert repository.load().selected_model is ModelId.CPU


def test_shortcut_update_preserves_settings_and_only_replaces_settings_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    hotkey_path = config_dir / "hotkey.conf"
    original_values = {
        "selected_model": ModelId.CPU.value,
        "max_recording_seconds": 45,
        "preview_interval_ms": 500,
        "tail_overlap_seconds": 7,
        "macos_paste_shortcut": MacPasteShortcut.COMMAND_V.value,
        "future_option": {"preserve": True},
    }
    settings_path.write_text(
        json.dumps(original_values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    hotkey_path.write_text("RightAlt\n", encoding="utf-8")
    atomic_targets, write_targets, replace_targets = (
        _install_shortcut_atomic_io_spy(monkeypatch)
    )

    SettingsRepository(config_dir).update_macos_paste_shortcut(
        MacPasteShortcut.CONTROL_V
    )

    expected_values = dict(original_values)
    expected_values["macos_paste_shortcut"] = "control-v"
    assert json.loads(settings_path.read_text(encoding="utf-8")) == expected_values
    assert hotkey_path.read_text(encoding="utf-8") == "RightAlt\n"
    _assert_only_settings_json_was_atomically_replaced(
        settings_path=settings_path,
        atomic_targets=atomic_targets,
        write_targets=write_targets,
        replace_targets=replace_targets,
    )
    assert hotkey_path.with_suffix(".conf.tmp") not in write_targets
    assert all(target != hotkey_path for _, target in replace_targets)


def test_shortcut_atomic_io_spy_kills_same_content_hotkey_write_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    hotkey_path = config_dir / "hotkey.conf"
    settings_path.write_text(
        '{"macos_paste_shortcut":"command-v"}\n',
        encoding="utf-8",
    )
    hotkey_path.write_text("RightAlt\n", encoding="utf-8")
    atomic_targets, write_targets, replace_targets = (
        _install_shortcut_atomic_io_spy(monkeypatch)
    )
    original_update = SettingsRepository.update_macos_paste_shortcut

    def update_with_erroneous_same_content_hotkey_write(
        repository: SettingsRepository,
        shortcut: MacPasteShortcut,
    ) -> None:
        original_update(repository, shortcut)
        repository._write_atomic(
            hotkey_path,
            hotkey_path.read_text(encoding="utf-8"),
        )

    monkeypatch.setattr(
        SettingsRepository,
        "update_macos_paste_shortcut",
        update_with_erroneous_same_content_hotkey_write,
    )

    SettingsRepository(config_dir).update_macos_paste_shortcut(
        MacPasteShortcut.CONTROL_V
    )

    assert atomic_targets == [settings_path, hotkey_path]
    assert hotkey_path.with_suffix(".conf.tmp") in write_targets
    assert (hotkey_path.with_suffix(".conf.tmp"), hotkey_path) in replace_targets
    with pytest.raises(AssertionError, match="settings.json only"):
        _assert_only_settings_json_was_atomically_replaced(
            settings_path=settings_path,
            atomic_targets=atomic_targets,
            write_targets=write_targets,
            replace_targets=replace_targets,
        )


@pytest.mark.parametrize("failure_phase", ["write", "replace"])
def test_shortcut_update_failure_keeps_original_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    temporary_path = config_dir / "settings.json.tmp"
    original_content = (
        '{"macos_paste_shortcut":"command-v","unknown":"keep"}\n'
    )
    settings_path.write_text(original_content, encoding="utf-8")
    original_write_text = Path.write_text
    original_replace = Path.replace

    def fail_temporary_write(
        path: Path, content: str, *, encoding: str
    ) -> int:
        if path == temporary_path:
            original_write_text(path, "partial", encoding=encoding)
            raise OSError("simulated temporary write failure")
        return original_write_text(path, content, encoding=encoding)

    def fail_replace(path: Path, target: Path) -> Path:
        if path == temporary_path and target == settings_path:
            raise OSError("simulated atomic replace failure")
        return original_replace(path, target)

    if failure_phase == "write":
        monkeypatch.setattr(Path, "write_text", fail_temporary_write)
    else:
        monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match=f"simulated .*{failure_phase} failure"):
        SettingsRepository(config_dir).update_macos_paste_shortcut(
            MacPasteShortcut.CONTROL_V
        )

    assert settings_path.read_text(encoding="utf-8") == original_content
    assert not temporary_path.exists()


@pytest.mark.parametrize(
    "shortcut",
    ["command-or-control", True, ["command-v"]],
)
def test_invalid_macos_paste_shortcut_is_reported(
    tmp_path: Path, shortcut: object
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"macos_paste_shortcut": shortcut}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid macos_paste_shortcut.*command-v.*control-v",
    ):
        SettingsRepository(config_dir).load()


def test_invalid_model_configuration_is_reported(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        '{"selected_model": "unknown", "max_recording_seconds": 90}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selected_model"):
        SettingsRepository(config_dir).load()


@pytest.mark.parametrize("interval", [249, 1001, 500.0, "500"])
def test_invalid_preview_interval_is_reported(
    tmp_path: Path, interval: object
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"preview_interval_ms": interval}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preview_interval_ms"):
        SettingsRepository(config_dir).load()


@pytest.mark.parametrize("seconds", [0, 16, 5.0, "5"])
def test_invalid_tail_overlap_is_reported(
    tmp_path: Path, seconds: object
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"tail_overlap_seconds": seconds}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tail_overlap_seconds"):
        SettingsRepository(config_dir).load()
