import json
from pathlib import Path

import pytest

from vim2.config import (
    DEFAULT_HOTKEY,
    DEFAULT_PREVIEW_INTERVAL_MS,
    DEFAULT_PREVIEW_WINDOW_SECONDS,
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


@pytest.fixture(autouse=True)
def isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))


def test_missing_configuration_uses_documented_defaults(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")

    settings = repository.load()

    assert settings == Settings(
        selected_model=ModelId.FAST,
        hotkey=DEFAULT_HOTKEY,
        max_recording_seconds=300,
        preview_interval_ms=DEFAULT_PREVIEW_INTERVAL_MS,
        preview_window_seconds=DEFAULT_PREVIEW_WINDOW_SECONDS,
        tail_overlap_seconds=DEFAULT_TAIL_OVERLAP_SECONDS,
        macos_paste_shortcut=MacPasteShortcut.COMMAND_V,
        normalize_numbers=True,
    )


def test_settings_round_trip_in_portable_config_directory(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")
    settings = Settings(
        selected_model=ModelId.ACCURATE,
        hotkey="LeftCtrl+RightAlt",
        max_recording_seconds=45,
        preview_interval_ms=500,
        preview_window_seconds=9,
        tail_overlap_seconds=7,
        macos_paste_shortcut=MacPasteShortcut.CONTROL_V,
        normalize_numbers=False,
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
        "preview_window_seconds": 9,
        "selected_model": "qwen3-asr-1.7b-int8",
        "tail_overlap_seconds": 7,
        "normalize_numbers": False,
    }


def test_shortcut_and_local_model_override_survive_both_updates(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    repository = SettingsRepository(config_dir)
    repository.save(Settings(selected_model=ModelId.CPU))
    local_path = tmp_path / "home" / ".vim2" / "settings.local.json"
    local_path.parent.mkdir()
    local_path.write_text(
        json.dumps({"selected_model": ModelId.FAST.value}), encoding="utf-8"
    )

    repository.update_macos_paste_shortcut(MacPasteShortcut.CONTROL_V)
    repository.save_selected_model(ModelId.ACCURATE)

    loaded = repository.load()
    assert loaded.macos_paste_shortcut is MacPasteShortcut.CONTROL_V
    assert loaded.selected_model is ModelId.ACCURATE
    assert json.loads((config_dir / "settings.json").read_text(
        encoding="utf-8"
    ))["macos_paste_shortcut"] == "control-v"
    assert json.loads(local_path.read_text(encoding="utf-8")) == {
        "selected_model": ModelId.ACCURATE.value,
    }


def test_local_shortcut_override_rejects_portable_only_change(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    repository = SettingsRepository(config_dir)
    repository.save(Settings(selected_model=ModelId.CPU))
    settings_path = config_dir / "settings.json"
    original = settings_path.read_bytes()
    local_path = tmp_path / "home" / ".vim2" / "settings.local.json"
    local_path.parent.mkdir()
    local_path.write_text(
        json.dumps({"macos_paste_shortcut": "command-v"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Remove macos_paste_shortcut"):
        repository.update_macos_paste_shortcut(MacPasteShortcut.CONTROL_V)

    assert settings_path.read_bytes() == original
    assert repository.load().macos_paste_shortcut is MacPasteShortcut.COMMAND_V
    assert not settings_path.with_suffix(".json.tmp").exists()


def test_save_selected_model_does_not_add_default_settings(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"normalize_numbers": False}),
        encoding="utf-8",
    )

    SettingsRepository(config_dir).save_selected_model(ModelId.ACCURATE)

    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "normalize_numbers": False,
        "selected_model": ModelId.ACCURATE.value,
    }


def test_save_selected_model_leaves_portable_settings_unchanged_when_local_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    original_content = '{"selected_model": "qwen3-asr-0.6b-fp16"}\n'
    settings_path.write_text(original_content, encoding="utf-8")
    home_dir = tmp_path / "home"
    local_config_dir = home_dir / ".vim2"
    local_config_dir.mkdir(parents=True)
    local_settings_path = local_config_dir / "settings.local.json"
    local_settings_path.write_text(
        json.dumps(
            {
                "normalize_numbers": False,
                "selected_model": ModelId.CPU.value,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))

    repository = SettingsRepository(config_dir)
    repository.save_selected_model(ModelId.ACCURATE)

    assert settings_path.read_text(encoding="utf-8") == original_content
    assert json.loads(local_settings_path.read_text(encoding="utf-8")) == {
        "normalize_numbers": False,
        "selected_model": ModelId.ACCURATE.value,
    }
    assert repository.load().selected_model is ModelId.ACCURATE


def test_local_settings_in_home_directory_override_portable_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "max_recording_seconds": 45,
                "normalize_numbers": True,
                "selected_model": ModelId.FAST.value,
            }
        ),
        encoding="utf-8",
    )
    home_dir = tmp_path / "home"
    local_config_dir = home_dir / ".vim2"
    local_config_dir.mkdir(parents=True)
    (local_config_dir / "settings.local.json").write_text(
        json.dumps(
            {
                "normalize_numbers": False,
                "selected_model": ModelId.ACCURATE.value,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))

    settings = SettingsRepository(config_dir).load()

    assert settings.max_recording_seconds == 45
    assert settings.normalize_numbers is False
    assert settings.selected_model is ModelId.ACCURATE


def test_invalid_number_normalization_setting_is_reported(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"normalize_numbers": "yes"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="normalize_numbers"):
        SettingsRepository(config_dir).load()


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


@pytest.mark.parametrize("seconds", [0, 31, 8.0, "8"])
def test_invalid_preview_window_is_reported(
    tmp_path: Path, seconds: object
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"preview_window_seconds": seconds}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preview_window_seconds"):
        SettingsRepository(config_dir).load()


def test_maximum_recording_duration_is_300_seconds(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"max_recording_seconds": 300}),
        encoding="utf-8",
    )

    assert SettingsRepository(config_dir).load().max_recording_seconds == 300


def test_recording_duration_above_300_seconds_is_reported(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"max_recording_seconds": 301}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_recording_seconds"):
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
