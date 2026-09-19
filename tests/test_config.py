import json
from pathlib import Path

import pytest

from vim2.config import (
    DEFAULT_HOTKEY,
    DEFAULT_PREVIEW_INTERVAL_MS,
    DEFAULT_PREVIEW_WINDOW_SECONDS,
    DEFAULT_TAIL_OVERLAP_SECONDS,
    Settings,
    SettingsRepository,
)
from vim2.models import ModelId


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
        "max_recording_seconds": 45,
        "preview_interval_ms": 500,
        "preview_window_seconds": 9,
        "selected_model": "qwen3-asr-1.7b-int8",
        "tail_overlap_seconds": 7,
        "normalize_numbers": False,
    }


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
