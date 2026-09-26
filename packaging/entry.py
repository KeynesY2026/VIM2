"""PyInstaller entry point: no project files are written next to the executable."""
from vim2.main import run


if __name__ == "__main__":
    raise SystemExit(run())
