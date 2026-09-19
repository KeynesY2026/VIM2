from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vim2.models import ModelId

DEFAULT_HOTKEY = "RightAlt"
DEFAULT_PREVIEW_INTERVAL_MS = 1_000
MIN_PREVIEW_INTERVAL_MS = 250
MAX_PREVIEW_INTERVAL_MS = 1_000
DEFAULT_PREVIEW_WINDOW_SECONDS = 8
MIN_PREVIEW_WINDOW_SECONDS = 1
MAX_PREVIEW_WINDOW_SECONDS = 30
DEFAULT_TAIL_OVERLAP_SECONDS = 5
MIN_TAIL_OVERLAP_SECONDS = 1
MAX_TAIL_OVERLAP_SECONDS = 15


@dataclass(frozen=True, slots=True)
class Settings:
    selected_model: ModelId = ModelId.FAST
    hotkey: str = DEFAULT_HOTKEY
    max_recording_seconds: int = 300
    preview_interval_ms: int = DEFAULT_PREVIEW_INTERVAL_MS
    preview_window_seconds: int = DEFAULT_PREVIEW_WINDOW_SECONDS
    tail_overlap_seconds: int = DEFAULT_TAIL_OVERLAP_SECONDS
    normalize_numbers: bool = True


class SettingsRepository:
    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._settings_path = config_dir / "settings.json"
        self._local_settings_path = Path.home() / ".vim2" / "settings.local.json"
        self._hotkey_path = config_dir / "hotkey.conf"

    def load(self) -> Settings:
        values: dict[str, object] = {}
        if self._settings_path.is_file():
            values = json.loads(self._settings_path.read_text(encoding="utf-8"))
        if self._local_settings_path.is_file():
            values.update(
                json.loads(self._local_settings_path.read_text(encoding="utf-8"))
            )

        try:
            model = ModelId(values.get("selected_model", ModelId.FAST))
        except ValueError as exc:
            raise ValueError(
                f"Invalid selected_model: {values.get('selected_model')!r}"
            ) from exc

        max_seconds = values.get("max_recording_seconds", 300)
        if not isinstance(max_seconds, int) or not 1 <= max_seconds <= 300:
            raise ValueError(
                "max_recording_seconds must be an integer from 1 through 300"
            )

        preview_interval_ms = values.get(
            "preview_interval_ms", DEFAULT_PREVIEW_INTERVAL_MS
        )
        if (
            not isinstance(preview_interval_ms, int)
            or not MIN_PREVIEW_INTERVAL_MS
            <= preview_interval_ms
            <= MAX_PREVIEW_INTERVAL_MS
        ):
            raise ValueError(
                "preview_interval_ms must be an integer from 250 through 1000"
            )

        preview_window_seconds = values.get(
            "preview_window_seconds", DEFAULT_PREVIEW_WINDOW_SECONDS
        )
        if (
            not isinstance(preview_window_seconds, int)
            or not MIN_PREVIEW_WINDOW_SECONDS
            <= preview_window_seconds
            <= MAX_PREVIEW_WINDOW_SECONDS
        ):
            raise ValueError(
                "preview_window_seconds must be an integer from 1 through 30"
            )

        tail_overlap_seconds = values.get(
            "tail_overlap_seconds", DEFAULT_TAIL_OVERLAP_SECONDS
        )
        if (
            not isinstance(tail_overlap_seconds, int)
            or not MIN_TAIL_OVERLAP_SECONDS
            <= tail_overlap_seconds
            <= MAX_TAIL_OVERLAP_SECONDS
        ):
            raise ValueError(
                "tail_overlap_seconds must be an integer from 1 through 15"
            )

        normalize_numbers = values.get("normalize_numbers", True)
        if not isinstance(normalize_numbers, bool):
            raise ValueError("normalize_numbers must be a boolean")

        hotkey = DEFAULT_HOTKEY
        if self._hotkey_path.is_file():
            hotkey = self._hotkey_path.read_text(encoding="utf-8").strip()
            if not hotkey:
                raise ValueError("hotkey.conf must not be empty")

        return Settings(
            selected_model=model,
            hotkey=hotkey,
            max_recording_seconds=max_seconds,
            preview_interval_ms=preview_interval_ms,
            preview_window_seconds=preview_window_seconds,
            tail_overlap_seconds=tail_overlap_seconds,
            normalize_numbers=normalize_numbers,
        )

    def save(self, settings: Settings) -> None:
        self._config_dir.mkdir(parents=True, exist_ok=True)
        settings_values = asdict(settings)
        settings_values.pop("hotkey")
        settings_values["selected_model"] = settings.selected_model.value
        self._write_atomic(
            self._settings_path,
            json.dumps(
                settings_values,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        self._write_atomic(self._hotkey_path, f"{settings.hotkey}\n")

    def save_selected_model(self, model_id: ModelId) -> None:
        settings_path = (
            self._local_settings_path
            if self._local_settings_path.is_file()
            else self._settings_path
        )
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_values: dict[str, object] = {}
        if settings_path.is_file():
            settings_values = json.loads(
                settings_path.read_text(encoding="utf-8")
            )
        settings_values["selected_model"] = model_id.value
        self._write_atomic(
            settings_path,
            json.dumps(
                settings_values,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    @staticmethod
    def _write_atomic(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
