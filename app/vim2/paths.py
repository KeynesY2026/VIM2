from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path
    config_dir: Path
    runtime_dir: Path
    models_dir: Path
    temp_dir: Path

    @property
    def hotwords_file(self) -> Path:
        return self.config_dir / "hotwords.txt"

    @classmethod
    def from_root(cls, root: Path) -> AppPaths:
        resolved = root.resolve()
        return cls(
            root=resolved,
            config_dir=resolved / "config",
            runtime_dir=resolved / "runtime",
            models_dir=resolved / ".models",
            temp_dir=resolved / "temp",
        )
