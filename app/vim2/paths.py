from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import secrets
import shutil
import sys


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
    def for_launch(
        cls, *, data_root: Path | None = None, platform: str | None = None
    ) -> AppPaths:
        """Use immutable PyInstaller data for models; all mutable paths live per user."""
        if not getattr(sys, "frozen", False):
            return cls.from_root(Path(__file__).resolve().parents[2])
        bundle = Path(sys._MEIPASS).resolve()
        if data_root is None:
            selected = platform or sys.platform
            if selected == "darwin":
                data_root = Path.home() / "Library" / "Application Support" / "VIM2"
            elif selected == "win32":
                local = os.environ.get("LOCALAPPDATA")
                if not local:
                    raise OSError("LOCALAPPDATA is required for the Windows installer")
                data_root = Path(local) / "VIM2"
            else:
                raise OSError(f"Unsupported frozen platform: {selected}")
        user = data_root.resolve()
        return cls(
            root=bundle,
            config_dir=user / "config",
            runtime_dir=user / "runtime",
            models_dir=bundle / ".models",
            temp_dir=user / "temp",
        )

    def initialize_user_data(self) -> None:
        """Install defaults without ever replacing an existing user file."""
        if not getattr(sys, "frozen", False):
            return
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        for name in ("settings.json", "hotkey.conf", "hotwords.txt"):
            source_name = "hotwords.template.txt" if name == "hotwords.txt" else name
            _install_missing_file(self.root / "config" / source_name, self.config_dir / name)

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


def _install_missing_file(source: Path, destination: Path) -> None:
    """Create one missing user file atomically; a failed attempt leaves no partial file."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if destination.exists():
            return
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
