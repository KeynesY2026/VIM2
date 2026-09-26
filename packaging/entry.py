"""PyInstaller entry point: no project files are written next to the executable."""
from __future__ import annotations

import sys
from collections.abc import Sequence

from vim2.main import run

# Finder launches a windowed binary with no console. These explicit checks must
# keep their stdout/stderr and exit codes for installer verification.
_OBSERVABLE_CLI = ("--check", "--import-smoke")


def frozen_gui_argv(argv: Sequence[str] | None = None) -> list[str]:
    """Inject ``--windowed`` for an ordinary frozen launch without duplicating it.

    ``--check`` and ``--import-smoke`` are returned unchanged so a disk image
    or installed app can be probed without a modal dialog.
    """
    selected = list(sys.argv[1:] if argv is None else argv)
    if any(flag in selected for flag in _OBSERVABLE_CLI):
        return selected
    if "--windowed" in selected:
        return selected
    return ["--windowed", *selected]


def main(argv: Sequence[str] | None = None) -> int:
    return run(frozen_gui_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
