import json
from pathlib import Path

import pytest

from vim2.config import (
    DEFAULT_HOTKEY,
    DEFAULT_PREVIEW_INTERVAL_MS,
    DEFAULT_TAIL_OVERLAP_SECONDS,
    Settings,
    SettingsRepository,
)
from vim2.models import ModelId


def test_missing_configuration_uses_documented_defaults(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")

    settings = repository.load()

    assert settings == Settings(
        selected_model=ModelId.FAST,
        hotkey=DEFAULT_HOTKEY,
        max_recording_seconds=300,
        preview_interval_ms=DEFAULT_PREVIEW_INTERVAL_MS,
        tail_overlap_seconds=DEFAULT_TAIL_OVERLAP_SECONDS,
    )


def test_settings_round_trip_in_portable_config_directory(tmp_path: Path) -> None:
    repository = SettingsRepository(tmp_path / "config")
    settings = Settings(
        selected_model=ModelId.ACCURATE,
        hotkey="LeftCtrl+RightAlt",
        max_recording_seconds=45,
        preview_interval_ms=500,
        tail_overlap_seconds=7,
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
        "selected_model": "qwen3-asr-1.7b-int8",
        "tail_overlap_seconds": 7,
    }


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
