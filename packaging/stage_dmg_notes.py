"""Copy UTF-8 DMG-root install notes. Does not mount, sign, or change quarantine."""
from __future__ import annotations

import sys
from pathlib import Path

NOTES_NAME = "安装说明.txt"


def stage_install_instructions(stage: Path, source: Path | None = None) -> Path:
    """Place the install notes at the disk-image root, outside VIM2.app."""
    destination_dir = Path(stage)
    if not destination_dir.is_dir():
        raise FileNotFoundError(destination_dir)
    notes = (
        Path(source)
        if source is not None
        else Path(__file__).resolve().with_name(NOTES_NAME)
    )
    data = notes.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("install notes must be UTF-8 without a BOM")
    data.decode("utf-8")
    destination = destination_dir / NOTES_NAME
    destination.write_bytes(data)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: stage_dmg_notes.py STAGE_DIR", file=sys.stderr)
        return 2
    stage_install_instructions(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
