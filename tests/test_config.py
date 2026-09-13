import json
from pathlib import Path

import pytest

from vim2.config import DEFAULT_HOTKEY, Settings, SettingsRepository
from vim2.models import ModelId


def test_missing_configuration_uses_documented_defaults(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")

    settings = repository.load()

    assert settings == Settings(
        selected_model=ModelId.FAST,
        hotkey=DEFAULT_HOTKEY,
        max_recording_seconds=90,
    )


def test_settings_round_trip_in_portable_config_directory(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")
    settings = Settings(
        selected_model=ModelId.ACCURATE,
        hotkey="LeftCtrl+RightAlt",
        max_recording_seconds=45,
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
        "selected_model": "qwen3-asr-1.7b-int8",
    }


def test_invalid_model_configuration_is_reported(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        '{"selected_model": "unknown", "max_recording_seconds": 90}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selected_model"):
        SettingsRepository(config_dir).load()
