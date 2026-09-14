from pathlib import Path

from vim2.paths import AppPaths


def test_all_portable_paths_are_resolved_from_application_root(tmp_path: Path) -> None:
    root = tmp_path / "带 空格" / "VIM2"

    paths = AppPaths.from_root(root)

    assert paths.root == root.resolve()
    assert paths.config_dir == root.resolve() / "config"
    assert paths.hotwords_file == root.resolve() / "config" / "hotwords.txt"
    assert paths.runtime_dir == root.resolve() / "runtime"
    assert paths.models_dir == root.resolve() / ".models"
    assert paths.temp_dir == root.resolve() / "temp"
