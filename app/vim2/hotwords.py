from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAX_HOTWORD_ENTRIES = 100
MAX_HOTWORD_LENGTH = 100
HOTWORDS_TEMPLATE = (
    "# One hotword or phrase per line. Lines beginning with # are ignored.\n"
)


@dataclass(frozen=True)
class HotwordSnapshot:
    entries: tuple[str, ...]

    @property
    def gpu_context(self) -> str:
        if not self.entries:
            return ""
        return (
            "Transcribe only the speech in the audio. The following terms are "
            "spelling references, not text to transcribe. Use a term only if it "
            "is spoken; do not output this list: " + ", ".join(self.entries)
        )

    def echoes_gpu_context(self, text: str) -> bool:
        words = [
            "".join(char for char in entry.casefold() if char.isalnum())
            for entry in self.entries
        ]
        words = [word for word in words if word]
        if len(words) < 4:
            return False
        normalized = "".join(char for char in text.casefold() if char.isalnum())
        return normalized.startswith("".join(words))

    @property
    def cpu_hotwords(self) -> str:
        return ",".join(self.entries)


def parse_hotwords(text: str) -> HotwordSnapshot:
    entries: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        text.removeprefix("\ufeff").splitlines(), start=1
    ):
        entry = line.strip()
        if not entry or entry.startswith("#") or entry in seen:
            continue
        if "," in entry:
            raise ValueError(
                f"Hotword on line {line_number} contains an ASCII comma"
            )
        if len(entry) > MAX_HOTWORD_LENGTH:
            raise ValueError(
                f"Hotword on line {line_number} exceeds "
                f"{MAX_HOTWORD_LENGTH} characters"
            )
        entries.append(entry)
        seen.add(entry)
        if len(entries) > MAX_HOTWORD_ENTRIES:
            raise ValueError(
                f"Hotword file exceeds {MAX_HOTWORD_ENTRIES} entries"
            )
    return HotwordSnapshot(tuple(entries))


class HotwordRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def ensure_file(self) -> Path:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(HOTWORDS_TEMPLATE, encoding="utf-8")
        return self.path

    def load(self) -> HotwordSnapshot:
        path = self.ensure_file()
        return parse_hotwords(path.read_text(encoding="utf-8-sig"))